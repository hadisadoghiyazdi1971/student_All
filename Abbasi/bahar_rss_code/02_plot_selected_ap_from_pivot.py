import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from faculty_map_json import draw_floor_map


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs_rss_ap"
PIVOT_DATA = DATA_DIR / "pivot_signal_table.csv"
SELECTION_FILE = OUT_DIR / "selected_access_point.json"


def red_green_rss_cmap() -> ListedColormap:
    """Use only red shades for weak RSS and green shades for strong RSS."""
    weak_reds = plt.cm.Reds(np.linspace(0.95, 0.45, 128))
    strong_greens = plt.cm.Greens(np.linspace(0.45, 0.95, 128))
    return ListedColormap(np.vstack([weak_reds, strong_greens]), name="rss_red_green")


def to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert latitude/longitude to local meter coordinates."""
    meters_per_lat = 111_320.0
    meters_per_lon = 111_320.0 * math.cos(math.radians(lat0))
    return (lon - lon0) * meters_per_lon, (lat - lat0) * meters_per_lat


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # Read the selected AP from step 1 and plot its raw pivot-table RSS values.
    selection = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    ground_truth = selection["ground_truth_ap"]
    ap_column = selection["selected_ap"]["ap_column_raw"]
    bssid = selection["selected_ap"]["bssid"]

    pivot = pd.read_csv(PIVOT_DATA)
    if ap_column not in pivot.columns:
        raise KeyError(f"{ap_column} was not found in {PIVOT_DATA}")

    pivot[ap_column] = pivot[ap_column].astype(float)
    samples = pivot[pivot[ap_column] > -100.0].copy()
    samples["rss_dbm"] = samples[ap_column]

    lat0 = float(pivot["lat"].mean())
    lon0 = float(pivot["lon"].mean())
    samples["x_m"], samples["y_m"] = to_local_xy(
        samples["lat"].to_numpy(), samples["lon"].to_numpy(), lat0, lon0
    )
    target_x, target_y = to_local_xy(
        np.array([ground_truth["lat"]]), np.array([ground_truth["lon"]]), lat0, lon0
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_floor_map(ax, lat0, lon0, floor_id=4)

    points = ax.scatter(
        samples["x_m"],
        samples["y_m"],
        c=samples["rss_dbm"],
        cmap=red_green_rss_cmap(),
        s=36,
        edgecolors="#202020",
        linewidths=0.35,
        label="RSS samples",
    )
    ax.scatter(
        target_x,
        target_y,
        marker="x",
        s=180,
        c="#e0182d",
        linewidths=2.8,
        label="ground-truth AP location",
        zorder=4,
    )

    ax.set_xlabel("east-west distance from origin (m)")
    ax.set_ylabel("north-south distance from origin (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(target_x[0]) - 45.0, float(target_x[0]) + 45.0)
    ax.set_ylim(float(target_y[0]) - 80.0, float(target_y[0]) + 80.0)
    ax.legend(loc="best")
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("RSS (dBm)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_pivot_rss_map.png", dpi=220)
    plt.close(fig)

    print("AP:", bssid)
    print("Pivot samples:", len(samples))
    print("Saved:", OUT_DIR / "02_pivot_rss_map.png")


if __name__ == "__main__":
    main()
