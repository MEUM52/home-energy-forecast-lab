"""UCI Appliances Energy Prediction CSV loading and validation.

Öğrenci notu:
- CSV dosyası proje kökündeki data/raw/ klasöründe tutulur.
- Tarihler kronolojik sıraya getirilir.
- Yinelenen tarihler ve tam 10 dakikalık olmayan aralıklar hata üretir.
- Appliances sütunu 10 dakikalık ölçüm aralığındaki enerji tüketimini Wh cinsinden verir.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


EXPECTED_COLUMNS = [
    "date",
    "Appliances",
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T4",
    "RH_4",
    "T5",
    "RH_5",
    "T6",
    "RH_6",
    "T7",
    "RH_7",
    "T8",
    "RH_8",
    "T9",
    "RH_9",
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
    "rv1",
    "rv2",
]

NUMERIC_COLUMNS = [column for column in EXPECTED_COLUMNS if column != "date"]
EXPECTED_INTERVAL = pd.Timedelta(minutes=10)


def load_energy_data(file_path=None):
    """Load and validate the UCI energy CSV.

    Args:
        file_path: CSV path. If omitted, uses
            <project_root>/data/raw/energydata_complete.csv.

    Returns:
        A pandas DataFrame sorted by date in ascending order.

    Raises:
        FileNotFoundError: CSV file does not exist.
        ValueError: CSV is empty, malformed, missing required columns,
            contains invalid values or dates, has duplicate timestamps,
            or does not follow the 10-minute sampling interval.
    """
    if file_path is None:
        project_root = Path(__file__).resolve().parents[3]
        file_path = (
            project_root / "data" / "raw" / "energydata_complete.csv"
        )

    csv_path = Path(file_path).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(
            "Enerji veri seti bulunamadı: "
            f"{csv_path}\n"
            "energydata_complete.csv dosyasını proje içindeki "
            "data/raw/ klasörüne yerleştirin."
        )

    try:
        data = pd.read_csv(csv_path)
    except (EmptyDataError, ParserError, UnicodeDecodeError) as error:
        raise ValueError(
            f"CSV dosyası okunamadı veya biçimi hatalı: {csv_path}. "
            "Dosyanın UCI CSV sürümü olduğundan emin olun."
        ) from error

    if data.empty:
        raise ValueError(f"CSV dosyası boş: {csv_path}")

    data.columns = data.columns.str.strip()

    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Veri setinde gerekli sütunlar eksik: "
            + ", ".join(missing_columns)
        )

    # Parse timestamps; invalid date strings become NaT and are reported.
    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    invalid_date_count = int(parsed_dates.isna().sum())
    if invalid_date_count:
        raise ValueError(
            f"'date' sütununda {invalid_date_count} geçersiz veya boş tarih var."
        )
    data["date"] = parsed_dates

    # The official dataset defines all columns except date as numeric.
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_count = int(converted.isna().sum())
        if invalid_count:
            raise ValueError(
                f"'{column}' sütununda {invalid_count} sayısal olmayan "
                "veya boş değer var."
            )

        non_finite_count = int((~np.isfinite(converted)).sum())
        if non_finite_count:
            raise ValueError(
                f"'{column}' sütununda {non_finite_count} sonlu olmayan "
                "değer (inf veya -inf) var."
            )
        data[column] = converted

    if data[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError(
            "Gerekli sütunlarda eksik değer bulundu. "
            "Veri dosyasını kontrol edin."
        )

    # Sort before checking the sequence of timestamps.
    data = data.sort_values("date", ascending=True).reset_index(drop=True)

    duplicate_count = int(data["date"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"'date' sütununda {duplicate_count} yinelenen zaman damgası var. "
            "Her 10 dakikalık zaman aralığı için tek kayıt bulunmalıdır."
        )

    if len(data) > 1:
        intervals = data["date"].diff().dropna()
        irregular_intervals = intervals[intervals != EXPECTED_INTERVAL]

        if not irregular_intervals.empty:
            first_bad_index = irregular_intervals.index[0]
            previous_date = data.loc[first_bad_index - 1, "date"]
            current_date = data.loc[first_bad_index, "date"]
            actual_interval = current_date - previous_date

            raise ValueError(
                "Veri setinde tam 10 dakikalık olmayan tarih aralığı bulundu. "
                f"Örnek: {previous_date} ile {current_date} arası "
                f"{actual_interval}. Eksik veya düzensiz kayıtları kontrol edin."
            )

    return data
