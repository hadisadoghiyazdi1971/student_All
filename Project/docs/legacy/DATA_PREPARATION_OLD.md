# DataPreparation

This folder contains the data preparation stage of the health digital-twin pipeline.
It converts Samsung Health exports and optional posture/voice signals into daily features, then creates the merged dataset used by the modeling notebook in `../project`.

## Current Outputs

Files currently present in this folder:

- `daily_steps_summary.csv`
- `daily_heart_rate_summary.csv`
- `daily_oxygen_saturation_summary.csv`
- `posture_analysis.csv`
- `voice_analysis.csv`
- `merged_health_data.csv`

## Current Notebooks

### `daily_steps.ipynb`

- Creates daily step summaries from Samsung step export files.
- Saves `daily_steps_summary.csv`.

### `daily_heart_rate.ipynb`

- Builds daily heart-rate summary features.
- Saves `daily_heart_rate_summary.csv`.

### `daily_oxygen_saturation.ipynb`

- Builds daily oxygen saturation summary features.
- Saves `daily_oxygen_saturation_summary.csv`.

### `posture.ipynb`

- Extracts posture metrics from pose keypoints.
- Uses YOLO pose weights.
- Saves `posture_analysis.csv`.

### `voice.ipynb`

- Extracts voice biomarkers (including MFCC-related features).
- Saves `voice_analysis.csv`.

### `data.ipynb`

- Merges daily heart, oxygen, steps, posture, and voice signals.
- Saves `merged_health_data.csv`.

### `rate.ipynb`

- Alternative integration flow focused on core daily metrics.
- Can be used to produce merged data and optional text feedback workflow.

## Data Assets

- `New Data/`: latest Samsung Health export batch.
- `com.samsung.*.csv` in this folder: previous export snapshots.
- `yolov8s-pose.pt`, `yolov8n-pose.pt`: pose estimation weights for posture analysis.

## Recommended Run Order

1. `daily_steps.ipynb`
2. `daily_heart_rate.ipynb`
3. `daily_oxygen_saturation.ipynb`
4. `posture.ipynb`
5. `voice.ipynb`
6. `data.ipynb` (or `rate.ipynb` for a lighter merge flow)

Then use `merged_health_data.csv` in `../project/health_digital_twin.ipynb`.

## Dependencies

Install dependencies from:

- `../project/requirements.txt`

Core libraries used in this stage include:

- `pandas`
- `numpy`
- `ultralytics`
- `parselmouth`
- `librosa`

## Notes

- Samsung export columns can change between versions; notebooks include fallback logic for common schema variations.
- If multiple files match the same input pattern in `New Data/`, the first match may be selected.
- Keep API keys in environment variables or `.env`, not in notebook cells.
