"""Build the offline artifact used by the 06_AI_Application Streamlit demo.

Run from the repository root:
    python 06_AI_Application/prepare_demo.py

This script trains models and creates the demo artifact. The Streamlit app
only loads that artifact; it never trains a model during startup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from energy_forecast.data_analysis.data_loader import load_energy_data
from energy_forecast.machine_learning.baseline import predict_last_measurement
from energy_forecast.machine_learning.evaluation import regression_metrics
from energy_forecast.machine_learning.features import (
    build_supervised_dataset,
    chronological_split,
    get_feature_columns,
)
from energy_forecast.machine_learning.linear_regression import train_linear_regression
from energy_forecast.machine_learning.random_forest import train_random_forest


TARGET_COLUMN = "target_appliances_wh"
BASELINE_NAME = "Son ölçüm (Baseline)"
LINEAR_NAME = "Linear Regression"
FOREST_NAME = "Random Forest"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "energy_forecast_demo.joblib"


def build_demo_artifact() -> dict:
    """Train/select a model and package it together with untouched test rows."""
    raw_data = load_energy_data()
    supervised = build_supervised_dataset(raw_data)
    train, validation, test = chronological_split(supervised)
    feature_columns = get_feature_columns(supervised)

    X_train = train[feature_columns]
    y_train = train[TARGET_COLUMN]
    X_validation = validation[feature_columns]
    y_validation = validation[TARGET_COLUMN]

    # Learned estimators are fitted on the training partition only.
    fitted_models = {
        BASELINE_NAME: None,
        LINEAR_NAME: train_linear_regression(X_train, y_train),
        FOREST_NAME: train_random_forest(X_train, y_train),
    }

    validation_predictions = {
        BASELINE_NAME: predict_last_measurement(validation),
        LINEAR_NAME: fitted_models[LINEAR_NAME].predict(X_validation),
        FOREST_NAME: fitted_models[FOREST_NAME].predict(X_validation),
    }
    validation_results = []
    for model_name, predictions in validation_predictions.items():
        validation_results.append(
            {
                "model": model_name,
                **regression_metrics(y_validation, predictions),
            }
        )

    # Validation MAE chooses the model. The test partition is not consulted.
    best_result = min(validation_results, key=lambda result: result["mae_wh"])
    selected_model_name = best_result["model"]

    record_columns = [
        "origin_time",
        "target_time",
        TARGET_COLUMN,
        *feature_columns,
    ]
    test_records = test.loc[:, record_columns].reset_index(drop=True)

    return {
        "artifact_version": 1,
        "model_name": selected_model_name,
        "model": fitted_models[selected_model_name],
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "test_records": test_records,
        "validation_mae_wh": float(best_result["mae_wh"]),
        "validation_results": validation_results,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }


def main() -> None:
    """Create/update the local artifact; never called by the Streamlit app."""
    try:
        artifact = build_demo_artifact()
    except FileNotFoundError as error:
        raise SystemExit(
            "\nEnerji veri seti bulunamadı. Önce UCI Appliances Energy "
            "Prediction veri setinin CSV dosyasını indirin ve şu konuma "
            "yerleştirin:\n"
            f"  {PROJECT_ROOT / 'data' / 'raw' / 'energydata_complete.csv'}\n"
            "Ardından bu komutu yeniden çalıştırın."
        ) from error

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ARTIFACT_PATH.with_name(ARTIFACT_PATH.name + ".tmp")
    joblib.dump(artifact, temporary_path, compress=3)
    temporary_path.replace(ARTIFACT_PATH)

    print("06_AI_Application demo dosyası hazır.")
    print(f"Seçilen model: {artifact['model_name']}")
    print(f"Doğrulama MAE: {artifact['validation_mae_wh']:.3f} Wh")
    print(f"Test kaydı: {artifact['test_rows']}")
    print(f"Dosya: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
