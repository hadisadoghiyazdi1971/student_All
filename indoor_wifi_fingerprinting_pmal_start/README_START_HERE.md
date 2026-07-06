# Indoor WiFi Fingerprinting + PMAL/KME Starter Pack

This folder is a clean, portable subset of the larger Ghafarian project. It was copied, not moved, so the original completed project remains untouched.

## Why These Files Were Selected

The target idea is to reuse the thesis work on active learning on distributions for indoor positioning with WiFi fingerprinting.

In WiFi fingerprinting, each reference point or location can be treated as a distributional object:

- A location/reference point is one example `P_i`.
- Repeated RSSI scans at that location are the bag of samples `B_i = {x_ij}`.
- Each scan vector `x_ij` contains RSSI values from access points.
- The label can be a room, zone, floor, building area, or coordinate cell.
- Active learning asks: which location/fingerprint should be measured or labeled next?

This matches the thesis setting: active learning when each candidate is not a single vector, but a distribution estimated from finite samples.

## Folder Map

### `01_thesis_chapters_3_4_source`

Core thesis source for Chapters 3 and 4:

- `Ghafarian_Ch3_4_English_complete.tex`
- `Chapter_3_4_English_patched_body.tex`
- `figures/`
- `sinkhorn_pmal_distribution_active_learning.tex`

Use this for the mathematical base: PMAL, active learning on distributions, KME/Bayesian KME, and experiments.

### `02_transport_kme_pmal_paper_core`

Main paper-level source and code for the newer transport-calibrated KME-PMAL direction:

- `Transport_Calibrated_KME_PMAL_Forward_v3.tex`
- `Transport_Calibrated_KME_PMAL_Revised.tex`
- `Transport_Calibrated_KME_PMAL_IEEE_CYB_Submission.tex`
- `code/`
- `figures/`
- `results_journal_upgrade/`

This is the best base for a new IEEE Sensors Journal manuscript, because WiFi fingerprinting is naturally sensor-distribution data.

### `03_heart_risk_transport_extension`

Optional newer extension that adds risk/uncertainty language:

- `HeartRisk_OT_KME_PMAL_v5.tex`
- `code/heart_risk_transport_core.py`
- `results/heart_risk_metrics.csv`

Use this if the new paper emphasizes robust sensor uncertainty, risk-aware querying, distribution shift, or unreliable RSSI measurements.

### `04_indoor_wifi_adaptation_notes`

Reserved for the new WiFi-specific work. Suggested first files to add here:

- dataset description from the IEEE Sensors Journal paper
- RSSI preprocessing notes
- proposed method sketch
- experiment protocol
- table mapping old datasets to WiFi datasets

## Practical Adaptation Plan

1. Start from `02_transport_kme_pmal_paper_core/code/tc_kme_common.py`.
2. Create a WiFi dataset loader similar to `tc_kme_datasets.py`.
3. Convert each location/fingerprint into a bag of RSSI vectors.
4. Keep KME/Bayesian KME as the representation.
5. Replace GAUSSDIST/USPS/20NG experiments with WiFi indoor positioning datasets.
6. Use localization metrics:
   - mean positioning error
   - median positioning error
   - CDF of positioning error
   - floor/room/zone accuracy, if labels are categorical
7. Write the contribution as:
   - active fingerprint acquisition under finite RSSI-sample uncertainty
   - transport-calibrated Bayesian KME-PMAL for WiFi fingerprint distributions
   - query strategy for reducing site-survey labeling cost

## Files Intentionally Not Included

Temporary LaTeX/build artifacts were not copied:

- `.aux`
- `.log`
- `.out`
- `.synctex.gz`
- rendered page screenshots
- `__pycache__`
- `.pyc`

Those can be regenerated and would only make the starter pack noisy.
