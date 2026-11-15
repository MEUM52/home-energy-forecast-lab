"""Small OOP example for loading and summarizing appliance energy data."""
from pathlib import Path
import pandas as pd

from energy_forecast.data_analysis.data_loader import load_energy_data


class ApplianceEnergyAnalyzer:
    """Load validated energy data and calculate daily totals."""

    def __init__(self, file_path=None):
        self.file_path = Path(file_path) if file_path is not None else None
        self.data = None

    def load(self):
        """Reuse the shared CSV loader and keep the result on this object."""
        self.data = load_energy_data(self.file_path)
        return self.data

    def daily_summary(self):
        """Return one row per day with total energy in Wh and kWh."""
        if self.data is None:
            raise RuntimeError("Önce load() metodunu çağırarak veriyi yükleyin.")

        summary = (
            self.data.assign(day=self.data["date"].dt.date)
            .groupby("day", as_index=False)["Appliances"]
            .sum()
            .rename(columns={"Appliances": "total_wh"})
        )
        summary["total_kwh"] = summary["total_wh"] / 1000
        return summary
