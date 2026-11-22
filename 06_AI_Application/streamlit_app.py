"""Tek ekranlı, geçmiş test kayıtlarıyla çalışan enerji tahmin demosu."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "energy_forecast_demo.joblib"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

BASELINE_NAME = "Son ölçüm (Baseline)"
EXPECTED_ARTIFACT_VERSION = 1


st.set_page_config(
    page_title="Ev Enerjisi | 10 Dakikalık Tahmin",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource(show_spinner="Hazırlanmış tahmin dosyası yükleniyor...")
def load_artifact(artifact_path: str) -> dict:
    """Load an artifact created offline by prepare_demo.py; never fit a model."""
    return joblib.load(artifact_path)


def show_missing_artifact_help() -> None:
    """Explain how to create the local artifact if it does not exist."""
    st.error("Tahmin dosyası bulunamadı. Uygulamayı kullanmadan önce demo dosyasını hazırlayın.")
    st.markdown(
        f"""
1. UCI **Appliances Energy Prediction** veri setini indirin:
   [UCI veri seti sayfası](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
2. CSV dosyasını şu adla proje içindeki data/raw/ klasörüne koyun:
   energydata_complete.csv
3. Proje ana klasöründe terminal açıp aşağıdaki hazırlama komutunu bir kez çalıştırın.
4. Hazırlama tamamlandıktan sonra Streamlit uygulamasını başlatın.
"""
    )
    st.code(
        "python 06_AI_Application/prepare_demo.py\n"
        "streamlit run 06_AI_Application/streamlit_app.py",
        language="bash",
    )
    st.caption(
        "İlk komut modeli eğitir ve models/energy_forecast_demo.joblib dosyasını "
        "oluşturur. İkinci komut yalnızca bu dosyayı yükler; modeli yeniden eğitmez."
    )


def validate_artifact(artifact: dict) -> tuple[pd.DataFrame, list[str], str]:
    """Check artifact structure before using it in the interface."""
    if not isinstance(artifact, dict):
        raise ValueError("Model dosyasının içeriği beklenen biçimde değil.")
    if artifact.get("artifact_version") != EXPECTED_ARTIFACT_VERSION:
        raise ValueError(
            "Model dosyasının sürümü bu uygulamayla uyumlu değil. "
            "prepare_demo.py dosyasını yeniden çalıştırın."
        )

    records = artifact.get("test_records")
    feature_columns = artifact.get("feature_columns")
    model_name = artifact.get("model_name")

    if not isinstance(records, pd.DataFrame) or records.empty:
        raise ValueError("Model dosyasında kullanılabilir test kayıtları bulunamadı.")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Model dosyasında özellik listesi bulunamadı.")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Model dosyasında seçilen model bilgisi bulunamadı.")

    required_columns = {
        "origin_time",
        "target_time",
        artifact.get("target_column", "target_appliances_wh"),
        *feature_columns,
    }
    missing_columns = sorted(required_columns.difference(records.columns))
    if missing_columns:
        raise ValueError(
            "Test kayıtlarında gerekli sütunlar eksik: "
            + ", ".join(missing_columns)
        )
    if model_name != BASELINE_NAME and artifact.get("model") is None:
        raise ValueError(
            "Seçilen model dosyada bulunamadı. prepare_demo.py dosyasını yeniden çalıştırın."
        )

    return records, feature_columns, model_name


def predict_record(record: pd.Series, artifact: dict, feature_columns: list[str]) -> float:
    """Predict one held-out row using the already-trained model."""
    if artifact["model_name"] == BASELINE_NAME:
        return float(record["appliances_lag_0"])

    model_input = pd.DataFrame(
        [[record[column] for column in feature_columns]],
        columns=feature_columns,
    )
    prediction = artifact["model"].predict(model_input)
    return float(prediction[0])


def format_timestamp(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("Seçilen kaydın tarih/saat bilgisi okunamadı.")
    return timestamp.strftime("%d.%m.%Y %H:%M")


st.title("⚡ Ev Enerjisi — 10 Dakikalık Tüketim Tahmini")
st.caption("06_AI_Application | Geçmiş test kayıtlarıyla tahmin demosu")

st.info(
    "Bu uygulama **canlı izleme yapmaz** ve sensörlere bağlı değildir. "
    "Ekrandaki tahminler, daha önce ayrılmış test bölümündeki geçmiş kayıtlar "
    "üzerinden gösterilir. Model uygulama açılırken yeniden eğitilmez."
)

if not ARTIFACT_PATH.is_file():
    show_missing_artifact_help()
    st.stop()

try:
    artifact = load_artifact(str(ARTIFACT_PATH))
    test_records, feature_columns, selected_model_name = validate_artifact(artifact)
except Exception as error:
    st.error("Hazırlanmış tahmin dosyası okunamadı veya beklenen yapıda değil.")
    st.write(
        "Bağımlılık sürümlerini kontrol edin ve proje ana klasöründe "
        "python 06_AI_Application/prepare_demo.py komutuyla demo dosyasını yeniden oluşturun."
    )
    with st.expander("Teknik ayrıntı"):
        st.code(str(error))
    st.stop()

st.markdown(f"**Kullanılan model:** {selected_model_name}")
st.caption(
    f"Doğrulama bölümündeki MAE: "
    f"{float(artifact.get('validation_mae_wh', float('nan'))):.2f} Wh"
)


def record_label(index: int) -> str:
    row = test_records.iloc[index]
    return (
        f"Kayıt {index + 1} — "
        f"{format_timestamp(row['origin_time'])} → "
        f"{format_timestamp(row['target_time'])}"
    )


selected_index = st.selectbox(
    "Test bölümünden bir kayıt seçin",
    options=list(range(len(test_records))),
    format_func=record_label,
    key="selected_test_record",
)
if st.session_state.get("_last_record_index") != selected_index:
    st.session_state["_last_record_index"] = selected_index
    st.session_state["show_actual_value"] = False

selected_record = test_records.iloc[int(selected_index)]

try:
    predicted_wh = predict_record(selected_record, artifact, feature_columns)
    if not pd.notna(predicted_wh):
        raise ValueError("Model geçerli bir tahmin üretmedi.")
    origin_time = format_timestamp(selected_record["origin_time"])
    target_time = format_timestamp(selected_record["target_time"])
except Exception as error:
    st.error(f"Seçilen kayıt için tahmin üretilemedi: {error}")
    st.stop()

st.divider()
st.subheader("Tahmin sonucu")

forecast_col, target_col, prediction_col = st.columns(3)
forecast_col.metric("Tahmin anı", origin_time)
target_col.metric("Hedef zamanı", target_time)
prediction_col.metric("Tahmin edilen tüketim", f"{predicted_wh:.1f} Wh")
st.caption("Tahmin edilen değer, hedef zamana karşılık gelen 10 dakikalık enerji tüketimidir.")

threshold_col, _ = st.columns([1, 2])
with threshold_col:
    threshold_wh = st.number_input(
        "Uyarı eşiği (Wh / 10 dakika)",
        min_value=0.0,
        value=200.0,
        step=10.0,
        help="Tahmin bu değeri aşarsa yüksek tüketim uyarısı gösterilir.",
    )

if predicted_wh > threshold_wh:
    st.warning("⚠️ Yüksek tüketim bekleniyor")
else:
    st.success("Tahmin seçilen tüketim eşiğini aşmıyor.")

st.divider()
button_label = (
    "Gerçekleşen değeri gizle"
    if st.session_state.get("show_actual_value", False)
    else "Gerçekleşen değeri göster"
)
if st.button(button_label, type="secondary"):
    st.session_state["show_actual_value"] = not st.session_state.get(
        "show_actual_value", False
    )

if st.session_state.get("show_actual_value", False):
    target_column = artifact.get("target_column", "target_appliances_wh")
    actual_wh = float(selected_record[target_column])
    absolute_error_wh = abs(actual_wh - predicted_wh)

    actual_col, error_col = st.columns(2)
    actual_col.metric("Gerçekleşen tüketim", f"{actual_wh:.1f} Wh")
    error_col.metric("Mutlak hata", f"{absolute_error_wh:.1f} Wh")
    st.caption("Mutlak hata = |gerçekleşen tüketim − tahmin|.")
