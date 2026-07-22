import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.optimize import least_squares

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


def to_lat_lon(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Convert local meter coordinates back to latitude/longitude."""
    lat = lat0 + y / 111_320.0
    lon = lon0 + x / (111_320.0 * math.cos(math.radians(lat0)))
    return float(lat), float(lon)


def log_distance_model(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """RSS model: rss = rss_1m - 10 * n * log10(distance_m)."""
    ap_x, ap_y, rss_1m, path_loss_n = params
    distance = np.maximum(np.hypot(x - ap_x, y - ap_y), 1.0)
    return rss_1m - 10.0 * path_loss_n * np.log10(distance)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # Fit directly from the pivot table. No train/test or intermediate CSV is used.
    selection = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    ground_truth = selection["ground_truth_ap"]
    bssid = selection["selected_ap"]["bssid"]
    ap_column = selection["selected_ap"]["ap_column_raw"]

    pivot = pd.read_csv(PIVOT_DATA)
    pivot[ap_column] = pivot[ap_column].astype(float)
    samples = pivot[pivot[ap_column] > -100.0].copy()
    samples["rss_dbm"] = samples[ap_column]

    lat0 = float(samples["lat"].mean())
    lon0 = float(samples["lon"].mean())
    x, y = to_local_xy(samples["lat"].to_numpy(), samples["lon"].to_numpy(), lat0, lon0)
    rss = samples["rss_dbm"].to_numpy(dtype=float)
    gt_x, gt_y = to_local_xy(
        np.array([ground_truth["lat"]]), np.array([ground_truth["lon"]]), lat0, lon0
    )

    # The ground-truth AP coordinate is not used by the optimizer. The initial
    # AP position is the location where this AP has its strongest RSS sample.
    max_rss_idx = int(np.argmax(rss))
    initial = np.array([float(x[max_rss_idx]), float(y[max_rss_idx]), float(np.percentile(rss, 95)), 2.0])
    lower = [float(x.min() - 30.0), float(y.min() - 30.0), -95.0, 0.5]
    upper = [float(x.max() + 30.0), float(y.max() + 30.0), -20.0, 6.0]

    def residual(params: np.ndarray) -> np.ndarray:
        return log_distance_model(params, x, y) - rss

    result = least_squares(residual, x0=initial, bounds=(lower, upper), loss="soft_l1")
    predicted = log_distance_model(result.x, x, y)
    est_lat, est_lon = to_lat_lon(float(result.x[0]), float(result.x[1]), lat0, lon0)
    target_error_m = float(
        math.hypot(float(result.x[0]) - float(gt_x[0]), float(result.x[1]) - float(gt_y[0]))
    )

    summary = {
        "bssid": bssid,
        "estimated_lat": est_lat,
        "estimated_lon": est_lon,
        "ground_truth_lat": ground_truth["lat"],
        "ground_truth_lon": ground_truth["lon"],
        "distance_from_ground_truth_m": target_error_m,
        "initial_lat": float(samples.iloc[max_rss_idx]["lat"]),
        "initial_lon": float(samples.iloc[max_rss_idx]["lon"]),
        "initial_rule": "maximum_rss_sample",
        "rss_1m_dbm": float(result.x[2]),
        "path_loss_n": float(result.x[3]),
        "rmse_db": float(np.sqrt(np.mean((predicted - rss) ** 2))),
        "mae_db": float(np.mean(np.abs(predicted - rss))),
        "success": bool(result.success),
        "message": str(result.message),
    }
    (OUT_DIR / "03_fit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_floor_map(ax, lat0, lon0, floor_id=4)
    points = ax.scatter(
        x,
        y,
        c=rss,
        cmap=red_green_rss_cmap(),
        s=34,
        edgecolors="#202020",
        linewidths=0.3,
        label="RSS samples",
    )
    ax.scatter(
        gt_x,
        gt_y,
        marker="x",
        s=180,
        c="#e0182d",
        linewidths=2.8,
        label="ground-truth AP location",
        zorder=4,
    )
    ax.scatter(
        [result.x[0]],
        [result.x[1]],
        marker="*",
        s=330,
        c="#111111",
        edgecolors="#ffffff",
        linewidths=1.1,
        label="estimated AP location",
        zorder=5,
    )
    ax.set_title(f"Estimated AP location from RSS fitting ({bssid})")
    ax.set_xlabel("east-west distance from origin (m)")
    ax.set_ylabel("north-south distance from origin (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(gt_x[0]) - 45.0, float(gt_x[0]) + 45.0)
    ax.set_ylim(float(gt_y[0]) - 80.0, float(gt_y[0]) + 80.0)
    ax.legend(loc="best")
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("RSS (dBm)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_estimated_ap_map.png", dpi=220)
    plt.close(fig)

    print("Estimated AP lat/lon:", est_lat, est_lon)
    print("Distance from ground truth (m):", round(target_error_m, 3))
    print("Path-loss n:", round(summary["path_loss_n"], 3))
    print("RMSE dB:", round(summary["rmse_db"], 3))
    print("Saved:", OUT_DIR / "03_fit_summary.json")


if __name__ == "__main__":
    main()
