"""Find and extract RSS measurements belonging to an AP at a known location.

Expected project structure
--------------------------
project/
├── extract_target_ap_data.py
└── data/
    └── pivot_signal_table.csv

The input table is expected to contain:
- measurement coordinates: lat, lon
- floor identifiers: floor1_id and/or floor2_id
- RSS columns whose names start with signal_
- -100 as the non-detection/floor RSS value

Outputs are written to data/target_ap_output/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "pivot_signal_table.csv"
OUTPUT_DIR = DATA_DIR / "target_ap_output"

# 36°18'46.1"N, 59°31'34.9"E converted to decimal degrees.
TARGET_LAT = 36 + 18 / 60 + 46.1 / 3600
TARGET_LON = 59 + 31 / 60 + 34.9 / 3600

# The uploaded data show that the closest fingerprints belong to floor 5.
# Set this to None to search all floors.
TARGET_FLOOR: int | None = 5

MISSING_RSS_DBM = -100
NEAR_RADIUS_M = 10.0
TOP_FRACTION_FOR_CENTROID = 0.05
MIN_DETECTIONS = 20
STRONG_PEAK_THRESHOLD_DBM = -55.0
CLOSE_CENTROID_THRESHOLD_M = 5.0


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def haversine_m(
    lat: np.ndarray | pd.Series | float,
    lon: np.ndarray | pd.Series | float,
    ref_lat: float,
    ref_lon: float,
) -> np.ndarray:
    """Great-circle distance from points to a reference coordinate in metres."""
    earth_radius_m = 6_371_000.0
    lat_arr = np.radians(np.asarray(lat, dtype=float))
    lon_arr = np.radians(np.asarray(lon, dtype=float))
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    dlat = lat_arr - ref_lat_rad
    dlon = lon_arr - ref_lon_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(ref_lat_rad)
        * np.cos(lat_arr)
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * np.arcsin(np.sqrt(a))


def weighted_peak_centroid(group: pd.DataFrame, signal_col: str) -> tuple[float, float]:
    """Estimate the signal peak location from the strongest RSS observations."""
    top_count = max(5, int(np.ceil(len(group) * TOP_FRACTION_FOR_CENTROID)))
    strongest = group.nlargest(top_count, signal_col).copy()

    # Convert relative dBm values to positive linear-power weights.
    strongest_rss = strongest[signal_col].to_numpy(dtype=float)
    weights = 10.0 ** ((strongest_rss - strongest_rss.max()) / 10.0)

    centroid_lat = float(
        np.average(strongest["lat"].to_numpy(dtype=float), weights=weights)
    )
    centroid_lon = float(
        np.average(strongest["lon"].to_numpy(dtype=float), weights=weights)
    )
    return centroid_lat, centroid_lon


# -----------------------------------------------------------------------------
# AP search
# -----------------------------------------------------------------------------
def load_and_prepare_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {path.resolve()}\n"
            "Place pivot_signal_table.csv inside the data folder."
        )

    df = pd.read_csv(path)
    required_columns = {"lat", "lon"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["lat", "lon"]).copy()
    df["distance_to_known_ap_m"] = haversine_m(
        df["lat"], df["lon"], TARGET_LAT, TARGET_LON
    )
    return df


def filter_target_floor(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_FLOOR is None:
        return df.copy()

    floor_columns = [
        col for col in ("floor1_id", "floor2_id") if col in df.columns
    ]
    if not floor_columns:
        print("Warning: no floor column was found; all floors will be searched.")
        return df.copy()

    floor_mask = np.zeros(len(df), dtype=bool)
    for col in floor_columns:
        floor_mask |= df[col].eq(TARGET_FLOOR).to_numpy()

    floor_df = df.loc[floor_mask].copy()
    if floor_df.empty:
        raise ValueError(f"No samples were found for floor {TARGET_FLOOR}.")
    return floor_df


def rank_ap_candidates(df: pd.DataFrame) -> pd.DataFrame:
    signal_columns = [col for col in df.columns if col.startswith("signal_")]
    if not signal_columns:
        raise ValueError("No columns beginning with 'signal_' were found.")

    records: list[dict[str, float | int | str | bool]] = []

    for signal_col in signal_columns:
        detected = df.loc[
            df[signal_col].notna() & df[signal_col].gt(MISSING_RSS_DBM),
            ["lat", "lon", "distance_to_known_ap_m", signal_col],
        ].copy()

        if len(detected) < MIN_DETECTIONS:
            continue

        centroid_lat, centroid_lon = weighted_peak_centroid(detected, signal_col)
        centroid_distance_m = float(
            haversine_m(centroid_lat, centroid_lon, TARGET_LAT, TARGET_LON)
        )

        strongest_index = detected[signal_col].idxmax()
        peak_distance_m = float(
            detected.loc[strongest_index, "distance_to_known_ap_m"]
        )
        near = detected[detected["distance_to_known_ap_m"] <= NEAR_RADIUS_M]

        max_rss = float(detected[signal_col].max())
        near_median = float(near[signal_col].median()) if not near.empty else np.nan
        near_mean = float(near[signal_col].mean()) if not near.empty else np.nan

        is_close_strong_candidate = bool(
            centroid_distance_m <= CLOSE_CENTROID_THRESHOLD_M
            and max_rss >= STRONG_PEAK_THRESHOLD_DBM
        )

        records.append(
            {
                "signal_column": signal_col,
                "bssid_compact": signal_col.removeprefix("signal_"),
                "detections": int(len(detected)),
                "detections_within_10m": int(len(near)),
                "max_rss_dbm": max_rss,
                "mean_rss_within_10m_dbm": near_mean,
                "median_rss_within_10m_dbm": near_median,
                "strongest_sample_distance_m": peak_distance_m,
                "peak_centroid_lat": centroid_lat,
                "peak_centroid_lon": centroid_lon,
                "peak_centroid_distance_m": centroid_distance_m,
                "close_strong_candidate": is_close_strong_candidate,
            }
        )

    candidates = pd.DataFrame(records)
    if candidates.empty:
        raise ValueError("No AP had enough valid detections to be evaluated.")

    # First isolate radios whose estimated peak is very close to the known AP.
    # Within that group, prefer the radio with more usable RSS measurements.
    candidates = candidates.sort_values(
        by=[
            "close_strong_candidate",
            "detections",
            "median_rss_within_10m_dbm",
            "peak_centroid_distance_m",
        ],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates


def extract_selected_ap(
    full_df: pd.DataFrame,
    selected_signal_col: str,
) -> pd.DataFrame:
    selected = full_df.loc[
        full_df[selected_signal_col].notna()
        & full_df[selected_signal_col].gt(MISSING_RSS_DBM)
    ].copy()

    if TARGET_FLOOR is not None:
        floor_columns = [
            col for col in ("floor1_id", "floor2_id") if col in selected.columns
        ]
        if floor_columns:
            floor_mask = np.zeros(len(selected), dtype=bool)
            for col in floor_columns:
                floor_mask |= selected[col].eq(TARGET_FLOOR).to_numpy()
            selected = selected.loc[floor_mask].copy()

    metadata_columns = [
        "id",
        "building_id",
        "floor1_id",
        "floor2_id",
        "space_id",
        "lat",
        "lon",
        "timestamp_epoch",
        "phone",
        "gps_lat",
        "gps_lon",
        "gps_accuracy",
        "ori_x",
        "ori_y",
        "ori_z",
        "distance_to_known_ap_m",
    ]
    metadata_columns = [col for col in metadata_columns if col in selected.columns]

    selected = selected[metadata_columns + [selected_signal_col]].rename(
        columns={selected_signal_col: "rss_dbm"}
    )
    selected.insert(0, "selected_bssid_compact", selected_signal_col.removeprefix("signal_"))
    selected = selected.sort_values(
        ["distance_to_known_ap_m", "timestamp_epoch"]
        if "timestamp_epoch" in selected.columns
        else ["distance_to_known_ap_m"]
    ).reset_index(drop=True)
    return selected


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_df = load_and_prepare_data(INPUT_FILE)
    search_df = filter_target_floor(full_df)
    candidates = rank_ap_candidates(search_df)

    close_candidates = candidates[candidates["close_strong_candidate"]]
    if close_candidates.empty:
        raise RuntimeError(
            "No strong AP peak was found within the configured distance threshold. "
            "Inspect target_ap_candidates.csv and adjust the thresholds if necessary."
        )

    selected_signal_col = str(close_candidates.iloc[0]["signal_column"])
    selected_data = extract_selected_ap(full_df, selected_signal_col)

    candidates_path = OUTPUT_DIR / "target_ap_candidates.csv"
    selected_path = OUTPUT_DIR / f"target_ap_{selected_signal_col.removeprefix('signal_')}.csv"
    metadata_path = OUTPUT_DIR / "target_ap_selection.json"

    candidates.to_csv(candidates_path, index=False)
    selected_data.to_csv(selected_path, index=False)

    selection_metadata = {
        "known_ap_coordinate": {
            "latitude": TARGET_LAT,
            "longitude": TARGET_LON,
            "original_dms": '36°18\'46.1"N, 59°31\'34.9"E',
        },
        "target_floor": TARGET_FLOOR,
        "selected_signal_column": selected_signal_col,
        "selected_bssid_compact": selected_signal_col.removeprefix("signal_"),
        "number_of_saved_measurements": int(len(selected_data)),
        "candidate_metrics": close_candidates.iloc[0].to_dict(),
        "note": (
            "A physical access point may broadcast more than one BSSID. "
            "Review target_ap_candidates.csv before combining radios."
        ),
    }
    metadata_path.write_text(
        json.dumps(selection_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTarget AP search completed.")
    print(f"Known coordinate: {TARGET_LAT:.9f}, {TARGET_LON:.9f}")
    print(f"Search floor: {TARGET_FLOOR}")
    print(f"Selected signal: {selected_signal_col}")
    print(f"Saved measurements: {len(selected_data)}")
    print(f"Candidate report: {candidates_path}")
    print(f"Selected AP data: {selected_path}")
    print(f"Selection metadata: {metadata_path}\n")

    print("Top close candidates:")
    print(
        close_candidates[
            [
                "rank",
                "signal_column",
                "detections",
                "max_rss_dbm",
                "median_rss_within_10m_dbm",
                "peak_centroid_distance_m",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
