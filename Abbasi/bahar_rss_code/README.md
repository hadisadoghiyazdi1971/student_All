# RSS Access Point Localization

This project estimates the location of one Wi-Fi access point from RSS samples.
The current workflow uses only the pivot RSS table:

```text
data/pivot_signal_table.csv
```

The train/test CSV files are not used by the current scripts.

## Files

```text
01_select_access_point.py
```

Stores the AP under study, computes RSS statistics, and keeps the measured AP
coordinate only as ground truth for evaluation.

```text
02_plot_selected_ap_from_pivot.py
```

Reads the selected AP from `outputs_rss_ap/selected_access_point.json`, reads its
RSS samples from the pivot table, and plots them on the faculty map.

```text
03_fit_selected_ap_location.py
```

Fits a log-distance RSS model and estimates the AP location. The fitting starts
from the measurement point with the maximum RSS, not from the ground-truth AP
coordinate.

```text
faculty_map_json.py
```

Loads `data/map.json` and draws real room outlines from the
`map_space_coordinate` table.

```text
run_split_rss_ap_pipeline.ps1
```

Runs the full workflow. The first two steps use conda `base`, and the fitting
step uses conda `gaussianfit`.

## Run

```powershell
PowerShell -ExecutionPolicy Bypass -File .\run_split_rss_ap_pipeline.ps1
```

## Outputs

The cleaned workflow keeps only the main output files:

```text
outputs_rss_ap/selected_access_point.json
outputs_rss_ap/02_pivot_rss_map.png
outputs_rss_ap/03_fit_summary.json
outputs_rss_ap/03_estimated_ap_map.png
```

The RSS color map uses only red and green shades:

- red: weaker RSS
- green: stronger RSS

## Ground-Truth AP Coordinate

```text
36 deg 18'46.1"N, 59 deg 31'34.9"E
```

This coordinate is used only to report localization error.
