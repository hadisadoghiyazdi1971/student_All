# Research Direction: Transport-Calibrated Bayesian KME-PMAL for WiFi Fingerprinting

## Working Title

Active Fingerprint Acquisition for Indoor WiFi Positioning via Transport-Calibrated Bayesian Kernel Mean Embeddings

## Core Scientific Move

The previous work treats each candidate as a distribution observed through a finite bag of samples. In WiFi fingerprinting, this is a natural model:

- Reference point/location: distribution `P_i`
- RSSI scans collected at that location: sample bag `B_i`
- Unknown or expensive label: location/zone/coordinate annotation
- Query: which fingerprint/location should be measured, labeled, or refined next?

The new contribution should not simply apply old active learning to another dataset. The stronger angle is:

> WiFi fingerprints are uncertain distributional sensor objects; Bayesian KME estimates this uncertainty, and transport calibration gives a geometry-aware interpretation of RSSI perturbations and environmental shift.

## Method Adaptation

Represent every fingerprint location as a bag:

```text
B_i = [
  rssi_scan_1,
  rssi_scan_2,
  ...
  rssi_scan_n
]
```

Each `rssi_scan_j` is a vector over access points:

```text
x_ij = [RSSI_AP1, RSSI_AP2, ..., RSSI_APd]
```

Missing AP readings can be encoded by a fixed floor value such as `-100 dBm`, plus optional missingness indicators.

## Candidate New Contributions

1. Distributional formulation of WiFi fingerprinting:
   each location is modeled as a finite-sample distribution rather than a single averaged RSSI vector.

2. Active site-survey reduction:
   PMAL selects the next fingerprint location or RSSI bag to label/measure.

3. Bayesian KME uncertainty:
   finite scans at a location produce uncertainty, so low-sample or unstable locations are handled explicitly.

4. Transport calibration:
   RSSI perturbations caused by device orientation, temporal drift, obstacles, or AP instability are interpreted through transport ambiguity.

5. Sensor-journal relevance:
   the method targets lower calibration cost, robustness to sensor noise, and better sample efficiency for indoor positioning.

## Experiments To Build

Suggested experimental comparisons:

- random fingerprint acquisition
- uncertainty sampling
- core-set or diversity sampling
- standard KME-PMAL
- transport-calibrated Bayesian KME-PMAL
- optional heart-of-risk variant

Suggested metrics:

- mean localization error in meters
- median localization error
- 75th/90th percentile error
- CDF curve of localization error
- room/zone accuracy
- number of labeled fingerprints or scans needed to reach a target error

## Code Starting Points

Use these files first:

- `../02_transport_kme_pmal_paper_core/code/tc_kme_common.py`
- `../02_transport_kme_pmal_paper_core/code/tc_kme_datasets.py`
- `../02_transport_kme_pmal_paper_core/code/run_full_suite.py`
- `../03_heart_risk_transport_extension/code/heart_risk_transport_core.py`

Create a new loader:

```text
load_wifi_fingerprints(...)
```

Expected output should mirror the existing distributional datasets:

```text
bags: list of arrays, one array per location/reference point
labels: room/zone/floor/cell labels or coordinate bins
metadata: coordinates, building, floor, AP names, scan timestamps
```

## First Manuscript Pivot

In the manuscript, replace generic examples such as images/documents/patients with:

- RSSI fingerprint distributions
- repeated scans under temporal and device variability
- finite site-survey budget
- sensor drift and environmental uncertainty
- indoor positioning accuracy under reduced calibration cost
