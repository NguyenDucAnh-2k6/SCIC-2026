"""
In-Gauge Environmental Sensing Module:
Streams real indoor climate telemetry (CO2, temperature, noise) across 16 classrooms from:
data/in_gauge_engage/longitudinal/{room_id}.csv
"""

import random
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd

AVAILABLE_ROOMS = [
    "19", "20", "27", "28", "29", "30", "31", "40", "41", "43",
    "KB1", "KB2", "KB3", "KB4", "KB5", "KB6"
]

class EnvironmentalSensorStreamer:
    """Manages longitudinal indoor microclimate streaming."""
    def __init__(self, data_root: Path, environmental_dir: Optional[Union[str, Path]] = None):
        self.data_root = data_root
        self.environmental_dir = Path(environmental_dir) if environmental_dir else (self.data_root / "in_gauge_engage" / "longitudinal")
        self._env_streams: Dict[str, pd.DataFrame] = {}
        self._active_room_id: str = "19"

    def set_active_classroom_room(self, room_id: Optional[str] = None) -> str:
        """Switches active classroom sensor stream."""
        if room_id in AVAILABLE_ROOMS:
            self._active_room_id = room_id
        else:
            self._active_room_id = random.choice(AVAILABLE_ROOMS)
        return self._active_room_id

    def get_environmental_context(self, step_idx: int = 0) -> Dict[str, float]:
        """Streams real classroom indoor temperature, CO2, and noise."""
        if self._active_room_id not in self._env_streams:
            fpath = self.environmental_dir / f"{self._active_room_id}.csv"
            if not fpath.exists():
                fpath = self.environmental_dir / "19.csv"

            if fpath.exists():
                try:
                    cols = ["IndoorTemperature", "IndoorCO2", "IndoorNoise", "Occupied"]
                    df = pd.read_csv(fpath, usecols=cols)
                    df_occ = df[df["Occupied"] == 1].reset_index(drop=True)
                    self._env_streams[self._active_room_id] = df_occ if len(df_occ) > 0 else df
                except Exception:
                    self._env_streams[self._active_room_id] = pd.DataFrame()
            else:
                self._env_streams[self._active_room_id] = pd.DataFrame()

        df_room = self._env_streams.get(self._active_room_id, pd.DataFrame())
        if len(df_room) > 0:
            row_idx = step_idx % len(df_room)
            row = df_room.iloc[row_idx]
            return {
                "co2_ppm": float(row.get("IndoorCO2", 680.0)),
                "temp_c": float(row.get("IndoorTemperature", 24.2)),
                "noise_db": float(row.get("IndoorNoise", 48.0)),
                "room_id": self._active_room_id
            }

        return {"co2_ppm": 680.0, "temp_c": 24.2, "noise_db": 48.0, "room_id": "sim_default"}

