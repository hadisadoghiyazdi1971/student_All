# Project structure

```text
Project/
|-- data/
|   |-- raw/samsung_health/
|   |   |-- export_20260516/   # complete timestamped Samsung export
|   |   `-- legacy/            # earlier three-signal snapshots
|   `-- processed/             # daily features and merged model table
|-- notebooks/
|   |-- data_preparation/      # sensor feature extraction and integration
|   `-- modeling/              # health index, forecasting, recommendations
|-- models/pose/               # YOLOv8 pose weights
|-- reports/
|   |-- persian/               # Persian LaTeX source and compiled report
|   |-- english/               # English LaTeX source and compiled report
|   `-- generated/             # plots, JSON feedback, text/PDF outputs
|-- docs/architecture/         # standalone TikZ system diagram
|-- requirements.txt
`-- README.md
```

## Ownership rule

- Immutable source exports belong in `data/raw`.
- Reproducible tables derived from source data belong in `data/processed`.
- Executable experiments and workflows belong in `notebooks`.
- Downloaded model weights belong in `models`.
- Human-facing and machine-generated results belong in `reports`.
- Explanations of the system itself belong in `docs`.

Run each notebook with its own directory as the working directory. Its relative paths point back to the repository-level `data`, `models`, and `reports` folders.
