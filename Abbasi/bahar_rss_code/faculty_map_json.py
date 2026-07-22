from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MAP_JSON = ROOT / "data" / "map.json"


def decode_map_text(text: str) -> str:
    """Decode Persian labels if the database export contains mojibake text."""
    try:
        return str(text).encode("latin1").decode("utf-8")
    except UnicodeError:
        return str(text)


def to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert latitude/longitude to local meter coordinates."""
    meters_per_lat = 111_320.0
    meters_per_lon = 111_320.0 * math.cos(math.radians(lat0))
    x = (lon - lon0) * meters_per_lon
    y = (lat - lat0) * meters_per_lat
    return x, y


def table_from_map_json(table_name: str) -> pd.DataFrame:
    """Load one phpMyAdmin-exported table from data/map.json."""
    data = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    for item in data:
        if isinstance(item, dict) and item.get("type") == "table" and item.get("name") == table_name:
            return pd.DataFrame(item.get("data", []))
    raise KeyError(f"{table_name} was not found in {MAP_JSON}")


def read_floor_map(floor_id: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return map spaces and their polygon coordinates for one floor."""
    spaces = table_from_map_json("map_space").copy()
    coords = table_from_map_json("map_space_coordinate").copy()

    for col in ["id", "floor_id", "type_spaces_id"]:
        spaces[col] = spaces[col].astype(int)
    for col in ["lat", "lon"]:
        spaces[col] = spaces[col].astype(float)
    spaces["nam"] = spaces["nam"].map(decode_map_text)
    spaces["ref"] = spaces["ref"].map(decode_map_text)

    coords["space_id"] = coords["space_id"].astype(int)
    coords["rank"] = coords["rank"].astype(int)
    coords["lat"] = coords["lat"].astype(float)
    coords["lon"] = coords["lon"].astype(float)

    spaces = spaces[spaces["floor_id"] == floor_id].copy()
    coords = coords[coords["space_id"].isin(spaces["id"])].copy()
    return spaces, coords


def draw_floor_map(
    ax: plt.Axes,
    lat0: float,
    lon0: float,
    floor_id: int = 4,
) -> pd.DataFrame:
    """Draw the real map outlines from map_space_coordinate and return spaces."""
    spaces, coords = read_floor_map(floor_id=floor_id)

    for _, group in coords.sort_values("rank").groupby("space_id"):
        if len(group) < 2:
            continue
        x, y = to_local_xy(group["lat"].to_numpy(), group["lon"].to_numpy(), lat0, lon0)
        closed_x = np.r_[x, x[0]]
        closed_y = np.r_[y, y[0]]
        if len(group) >= 3:
            ax.fill(closed_x, closed_y, color="#eef1f5", alpha=0.45, zorder=0)
        ax.plot(closed_x, closed_y, color="#aeb6c2", linewidth=0.8, alpha=0.95, zorder=1)

    spaces["x_m"], spaces["y_m"] = to_local_xy(
        spaces["lat"].to_numpy(), spaces["lon"].to_numpy(), lat0, lon0
    )
    ax.scatter(
        spaces["x_m"],
        spaces["y_m"],
        s=9,
        c="#9da6b2",
        edgecolors="none",
        label="faculty map",
        zorder=2,
    )
    return spaces
