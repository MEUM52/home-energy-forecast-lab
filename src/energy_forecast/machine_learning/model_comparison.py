"""Leakage-aware comparison of baseline, Linear Regression and Random Forest.

Run from the repository root:
    python -m energy_forecast.machine_learning.model_comparison
"""

import pandas as pd

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


def run_model_comparison(raw_data=None):
    """Train on train only, select on validation MAE, evaluate once on test.

    All three methods are evaluated on exactly the same rows within each
    partition. The test partition is not used for model selection.
    """
    if raw_data is None:
        raw_data = load_energy_data()

    supervised = build_supervised_dataset(raw_data)
    train, validation, test = chronological_split(supervised)
    feature_columns = get_feature_columns(supervised)

    X_train, y_train = train[feature_columns], train[TARGET_COLUMN]
    X_validation, y_validation = validation[feature_columns], validation[TARGET_COLUMN]
    X_test, y_test = test[feature_columns], test[TARGET_COLUMN]

    # Fit learned models exclusively on training data.
    linear_model = train_linear_regression(X_train, y_train)
    forest_model = train_random_forest(X_train, y_train)

    models = {
        "Son ölçüm (Baseline)": None,
        "Linear Regression": linear_model,
        "Random Forest": forest_model,
    }

    validation_predictions = {
        "Son ölçüm (Baseline)": predict_last_measurement(validation),
        "Linear Regression": linear_model.predict(X_validation),
        "Random Forest": forest_model.predict(X_validation),
    }
    validation_results = []
    for name, predictions in validation_predictions.items():
        validation_results.append({
            "model": name,
            **regression_metrics(y_validation, predictions),
        })
    validation_table = pd.DataFrame(validation_results).sort_values(
        "mae_wh", ascending=True
    ).reset_index(drop=True)
    selected_model = str(validation_table.iloc[0]["model"])

    # Test is used only after validation-based selection is complete.
    test_predictions = {
        "Son ölçüm (Baseline)": predict_last_measurement(test),
        "Linear Regression": linear_model.predict(X_test),
        "Random Forest": forest_model.predict(X_test),
    }
    test_results = []
    for name, predictions in test_predictions.items():
        test_results.append({
            "model": name,
            **regression_metrics(y_test, predictions),
        })
    test_table = pd.DataFrame(test_results).sort_values(
        "mae_wh", ascending=True
    ).reset_index(drop=True)

    return {
        "validation": validation_table,
        "test": test_table,
        "selected_model": selected_model,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "feature_count": len(feature_columns),
        "validation_target_times": validation["target_time"].copy(),
        "test_target_times": test["target_time"].copy(),
    }


def main():
    results = run_model_comparison()
    print("\nVERİ BÖLÜMLERİ")
    print(
        f"Eğitim: {results['train_rows']} | "
        f"Doğrulama: {results['validation_rows']} | "
        f"Test: {results['test_rows']}"
    )
    print(f"Özellik sayısı: {results['feature_count']}")

    print("\nDOĞRULAMA SONUÇLARI (model seçimi yalnızca burada yapılır)")
    print(results["validation"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nDoğrulama MAE değerine göre seçilen model: {results['selected_model']}")

    print("\nTEST SONUÇLARI (son değerlendirme; MAE/RMSE Wh)")
    print(results["test"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    baseline = results["test"].set_index("model").loc["Son ölçüm (Baseline)", "mae_wh"]
    selected = results["test"].set_index("model").loc[results["selected_model"], "mae_wh"]
    if results["selected_model"] == "Son ölçüm (Baseline)":
        print("\nSONUÇ: Basit son ölçüm yöntemi doğrulamada en iyi sonucu verdi.")
    elif baseline < selected:
        print(
            "\nDİKKAT: Doğrulamada seçilen model testte baseline'dan daha kötü. "
            "Test sonucuna bakarak model seçimi değiştirilmedi."
        )
    else:
        print("\nSONUÇ: Doğrulamada seçilen model testte baseline ile karşılaştırıldı.")
    print("\nNot: Test metrikleri raporlama içindir; model seçimini geriye dönük değiştirmez.")


if __name__ == "__main__":
    main()
