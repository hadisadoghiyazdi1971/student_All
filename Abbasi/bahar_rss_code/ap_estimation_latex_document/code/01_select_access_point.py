import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from faculty_map_json import read_floor_map


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs_rss_ap"
PIVOT_DATA = DATA_DIR / "pivot_signal_table.csv"

# In the experiment dataset, this is the AP under study.
SELECTED_AP_BSSID = "b28ba99a8d2a"

# This coordinate is used only as ground truth for final error reporting.
GROUND_TRUTH_AP_LAT = 36 + 18 / 60 + 46.1 / 3600
GROUND_TRUTH_AP_LON = 59 + 31 / 60 + 34.9 / 3600


def to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert latitude/longitude to local meter coordinates."""
    meters_per_lat = 111_320.0
    meters_per_lon = 111_320.0 * math.cos(math.radians(lat0))
    return (lon - lon0) * meters_per_lon, (lat - lat0) * meters_per_lat


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # This workflow intentionally uses only the pivot RSS table.
    df = pd.read_csv(PIVOT_DATA)
    spaces, _ = read_floor_map(floor_id=4)

    ap_column = f"signal_{SELECTED_AP_BSSID}"
    if ap_column not in df.columns:
        raise KeyError(f"{ap_column} was not found in {PIVOT_DATA}")

    valid = df[df[ap_column].astype(float) > -100.0].copy()
    if valid.empty:
        raise RuntimeError(f"No valid RSS samples were found for {SELECTED_AP_BSSID}.")

    lat0 = float(valid["lat"].mean())
    lon0 = float(valid["lon"].mean())
    x, y = to_local_xy(valid["lat"].to_numpy(), valid["lon"].to_numpy(), lat0, lon0)
    rss = valid[ap_column].to_numpy(dtype=float)

    # This centroid is descriptive only. It is not used as ground truth.
    weights = np.power(10.0, (rss - rss.max()) / 10.0)
    centroid_x = float(np.average(x, weights=weights))
    centroid_y = float(np.average(y, weights=weights))

    gt_x, gt_y = to_local_xy(
        np.array([GROUND_TRUTH_AP_LAT]), np.array([GROUND_TRUTH_AP_LON]), lat0, lon0
    )
    centroid_error_m = float(math.hypot(centroid_x - float(gt_x[0]), centroid_y - float(gt_y[0])))

    max_idx = int(np.argmax(rss))
    max_rss_lat = float(valid.iloc[max_idx]["lat"])
    max_rss_lon = float(valid.iloc[max_idx]["lon"])

    nearby_spaces = spaces[
        (spaces["lat"].sub(GROUND_TRUTH_AP_LAT).abs() < 0.00035)
        & (spaces["lon"].sub(GROUND_TRUTH_AP_LON).abs() < 0.00035)
    ][["id", "nam", "ref", "lat", "lon"]]

    result = {
        "ground_truth_ap": {
            "lat": GROUND_TRUTH_AP_LAT,
            "lon": GROUND_TRUTH_AP_LON,
            "dms": "36 deg 18'46.1\"N, 59 deg 31'34.9\"E",
            "usage": "evaluation_only",
        },
        "selected_ap": {
            "ap_column_raw": ap_column,
            "bssid": SELECTED_AP_BSSID,
            "num_measurements": int(len(valid)),
            "max_rss": float(rss.max()),
            "mean_rss": float(rss.mean()),
            "min_rss": float(rss.min()),
            "rss_weighted_centroid_x_m": centroid_x,
            "rss_weighted_centroid_y_m": centroid_y,
            "centroid_error_to_ground_truth_m": centroid_error_m,
            "max_rss_sample_lat": max_rss_lat,
            "max_rss_sample_lon": max_rss_lon,
        },
        "origin": {"lat": lat0, "lon": lon0},
        "nearby_map_spaces": nearby_spaces.to_dict(orient="records"),
    }

    (OUT_DIR / "selected_access_point.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Selected AP:", SELECTED_AP_BSSID)
    print("Valid samples:", len(valid))
    print("Max RSS sample:", max_rss_lat, max_rss_lon)
    print("Centroid error to ground truth (m):", round(centroid_error_m, 3))
    print("Saved:", OUT_DIR / "selected_access_point.json")


if __name__ == "__main__":
    main()
