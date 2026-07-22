# Personal Health Digital Twin

This project creates a **closed-loop personal health digital twin** from multimodal daily observations. It synchronizes Samsung Health wearable data, posture features extracted with YOLOv8, and acoustic voice biomarkers; represents the person with a five-factor health state; forecasts the next day's state; and produces prioritized behavioral guidance.

![Project architecture](docs/architecture/digital_twin_architecture.pdf)

The editable TikZ source is [docs/architecture/digital_twin_architecture.tex](docs/architecture/digital_twin_architecture.tex). See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the repository map and file-placement rules.

## System in one glance

1. **Sense:** heart rate, SpO2, steps, posture images, and voice samples.
2. **Synchronize:** clean and aggregate signals into daily features aligned by date.
3. **Represent:** compute five sub-scores and the composite health state:

   `HI_v2 = 0.35 activity + 0.25 cardio + 0.20 respiration + 0.10 posture + 0.10 vocal`

4. **Predict:** forecast `HI_v2` for the next day using Linear Regression and a time-series-validated Random Forest.
5. **Act:** select a prioritized recommendation, report it, observe the person's response, and feed new measurements back into the twin.

The last feedback step is the central upgrade direction: it turns the current batch predictive workflow into an adaptive digital twin rather than a one-way health dashboard.

## Repository map

| Path | Purpose |
|---|---|
| `data/raw/` | Immutable Samsung Health exports |
| `data/processed/` | Daily summaries and `merged_health_data.csv` |
| `notebooks/data_preparation/` | Wearable, posture, voice, and merge pipelines |
| `notebooks/modeling/` | Five-factor index, forecasting, recommendations, reports |
| `models/pose/` | YOLOv8 pose-estimation weights |
| `reports/` | Persian/English LaTeX reports and generated results |
| `docs/architecture/` | Standalone TikZ architecture diagram |

## Reproduce the workflow

Create an environment and install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Run notebooks from their containing directories in this order:

1. `notebooks/data_preparation/daily_steps.ipynb`
2. `notebooks/data_preparation/daily_heart_rate.ipynb`
3. `notebooks/data_preparation/daily_oxygen_saturation.ipynb`
4. `notebooks/data_preparation/posture.ipynb`
5. `notebooks/data_preparation/voice.ipynb`
6. `notebooks/data_preparation/data.ipynb`
7. `notebooks/modeling/health_digital_twin.ipynb`

Prepared input for modeling is `data/processed/merged_health_data.csv`. Generated plots and narratives are written to `reports/generated/`.

## Current limits and upgrade priorities

- The available time series is small, so performance estimates are not yet evidence of clinical generalization.
- Recommendation thresholds are static and should evolve into personal baselines calibrated from outcomes.
- Posture and voice collection are sparse compared with wearable signals; explicit missingness and confidence handling are needed.
- The next version should add ingestion validation, uncertainty estimates, online recalibration, intervention/outcome logging, and drift monitoring.
- This is a research prototype, not a medical diagnostic device.

## Security

Store API credentials only in environment variables or an untracked `.env` file. A previously embedded Groq key was removed from the integration notebook during repository cleanup; that credential should be revoked and replaced because repository history or backups may still contain it.
