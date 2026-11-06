# Flowracle: Physics-Informed Probabilistic Seasonal Streamflow Forecasting Across California's Snow-fed Basins

*Disclaimer: This is not an officially supported Google product.*

## Overview

This repository contains the data acquisition pipelines, causal spatio-temporal
feature engineering, three-submodel quantile gradient-boosted decision tree
(GBDT) ensemble architecture, expanding-window cross-validation workflow, and
core probabilistic evaluation metrics for **Flowracle**.

Flowracle issues 100-member probabilistic forecasts of monthly unimpaired **Full
Natural Flow (FNF)** across 25 California Data Exchange Center (CDEC) snow-fed
watersheds (spanning the Sierra Nevada, Southern Cascades, and Klamath–Trinity
mountains) at 1- to 6-month lead times ($l \in \{1, \dots, 6\}$) on the 1st of
each calendar month. Across the 23 operational California Department of Water
Resources (DWR) Bulletin 120 snow-fed watersheds, monthly predictions aggregate
directly over April–July to provide seasonal water-supply outlooks alongside
full monthly hydrograph trajectories.

## Repository Structure

```text
flowracle/
├── README.md                       # Documentation and workflow guide
├── pyproject.toml                  # Package metadata (`pip install -e .`)
├── requirements.txt                # Core PyPI dependencies
├── run.sh                          # Self-contained CPU unit-test runner (<10 s)
├── flowracle/                      # Core Python package
│   ├── __init__.py                 # Public package exports
│   ├── data_pipeline.py            # Zarr ingestion, causal lag/lead Terrax DataLoader & CV slices
│   ├── features.py                 # 55-feature regional & 3D-IDW watershed representation
│   ├── model.py                    # Anchor / Core / Anomaly 13-quantile XGBoost ensemble & inference
│   ├── metrics.py                  # Pinball quantile loss, CV CRPS, B120 guardrail & timing metrics
│   └── model_test.py               # CPU unit tests for features, monotonicity, training & metrics
└── notebooks/
    ├── train_flowracle.ipynb       # Pre-2021 3-fold expanding-window CV & checkpoint training
    ├── evaluate_flowracle.ipynb    # Standalone 2021-2025 out-of-sample evaluation & B120 benchmark
    └── data_preprocessing/         # Public data download & Zarr consolidation Colabs
        ├── 01_download_cdec_fnf.ipynb
        ├── 02_download_b120_forecasts.ipynb
        ├── 03_download_cdec_snow_sensors.ipynb
        ├── 04_download_ua_swe_monthly.ipynb
        ├── 05_download_ua_swe_daily.ipynb
        ├── 06_download_mjo_rmm_index.ipynb
        ├── 07_download_prism_monthly.ipynb
        ├── 08_download_gridmet_pdsi.ipynb
        └── 09_download_snotel_daily.ipynb
```

## Model & Physical Regularization Summary

1. **Causal Multi-Source Alignment (`flowracle/data_pipeline.py`)**:
   - Forecasts issued on the 1st of month $m$ ingest real-time monthly and daily
     snow/climate telemetry through month $m - 1$, while restricting finalized
     CDEC monthly FNF gauge observations to lags $\le -1$ month relative to
     month $m - 1$ (i.e., month $m - 2$ and earlier) to prevent reporting-lag
     leakage.
   - Upstream preprocessing notebooks explicitly mask late-month observations
     whose operational latency exceeds 1st-of-month issuance (omitting the final
     calendar day in UA 4 km SWE and BoM MJO RMM, and applying a day-25 cutoff
     for NLDAS-2-backed gridMET PDSI pentads).

2. **Physics-Informed Feature Representation (`flowracle/features.py`)**:
   - **Latitudinal mountain bands**: Log-precipitation accumulations, positive
     degree-day thermal forcing, UA 4 km SWE, snowpack ripeness, SWE velocity,
     and time-since-peak-SWE across four California latitudinal bands
     (`32.5-42.0°N`).
   - **3D Inverse-Distance Weighting (IDW)**: Top-$K$ station interpolation in
     $(\text{lat}, \text{lon}, \text{elev} / 720\,\text{m})$ coordinates over
     USDA NRCS SNOTEL sites and California Cooperative Snow Surveys pillows
     (with CDEC pillows conditionally activated south of $37.0^\circ\text{N}$
     where SNOTEL density drops in the southern Sierra).
   - **Soil priming & kinematic surge damping**: Sigmoidal antecedent soil
     moisture priming combining regional PDSI and 6-month PRISM precipitation,
     paired with March 1st storm-surge damping (`SURGE_SENSITIVITY = 0.75`) and
     cross-basin storm synchrony metrics.

3. **Three-Submodel Quantile GBDT Architecture (`flowracle/model.py`)**:
   - **Anchor (`models_elev`, 44 features, $d_{\max}=5$, 4,000 trees)**:
     Shallow regularized baseline omitting ripeness and peak-timing counters.
   - **Core (`models_melt`, 51 features, $d_{\max}=9$, 7,000 trees,
     $\lambda=1100$)**: Deep nonlinear snowmelt learner resolving
     elevation-dependent melt timing.
   - **Anomaly (`models_resid`, 47 features, $d_{\max}=6$, 6,000 trees)**:
     Predicts log-flow residuals relative to each station and target month's
     historical median climatology from centered predictor anomalies.
   - Enforces `+1` monotonicity constraints across all precipitation, SWE, soil
     priming, and antecedent runoff predictors, combines sub-models via
     4-fold out-of-fold monthly inverse-RMSE weights (`ENSEMBLE_POWER = 4.2`,
     `SHRINKAGE_FACTOR = 0.18`), applies log-space Jensen bias correction
     (`0.65 * sigma2_resid`) before `expm1` transformation, and interpolates 13
     sorted quantiles ($\tau \in [0.01, 0.99]$) to $M = 100$ ensemble members.

## Quickstart & Unit Tests

Run the self-contained CPU unit test suite (executes in `< 10` seconds without
requiring external downloads or a GPU):

```bash
chmod +x run.sh
./run.sh
```

Or install the package in editable mode inside an existing Python $\ge 3.10$
environment:

```bash
pip install -e ".[dataloader,preprocessing]"
python -m unittest discover -s flowracle -p "*_test.py" -v
```

## Reproducing the Dataset, Training, and Out-of-Sample Evaluation

### Step 1: Download & Consolidate Public Hydroclimatic Datasets

Set `FLOWRACLE_DATA_DIR` to a local storage directory (recommended $\ge 15$ GB
free disk space) and run the notebooks in `notebooks/data_preprocessing/`:

| Notebook | Public Upstream Source | Generated Zarr Store (`$FLOWRACLE_DATA_DIR/...`) | Role in Flowracle |
| :--- | :--- | :--- | :--- |
| `01_download_cdec_fnf.ipynb` | CA DWR CDEC (Sensor 65) | `ca-monthly-fnf-extended/monthly_fnf_consolidated.zarr` | Target FNF & lagged runoff predictor (25 basins) |
| `02_download_b120_forecasts.ipynb` | CA DWR CDEC (Sensors 260–262) | `b120-forecasts/b120_consolidated.zarr` | Operational Bulletin 120 benchmark (23 basins) |
| `03_download_cdec_snow_sensors.ipynb` | CA DWR CDEC (Sensors 3, 18, 82) | `daily-ca-snow-sensors/daily_snow_sensors.zarr` & `end-of-month-ca-snow-sensors/endofmonth_snow_sensors.zarr` | Southern Sierra snow pillow SWE telemetry |
| `04_download_ua_swe_monthly.ipynb` | Univ. of Arizona 4 km SWE | `ua-swe-ca-region-operational-monthly-averages/ua_swe_CA_region.zarr` | Gridded causal partial-monthly SWE (days $1..N-1$) |
| `05_download_ua_swe_daily.ipynb` | Univ. of Arizona 4 km SWE | `ua-swe-daily-ca-region/ua_swe_daily_CA_all_months.zarr` | Optional daily 4 km SWE archive for diagnostics |
| `06_download_mjo_rmm_index.ipynb` | Australian BoM RMM Index | `mjo-rmm-index/mjo_rmm_index.zarr` | Subseasonal MJO phase & amplitude (2-day lag mask) |
| `07_download_prism_monthly.ipynb` | OSU PRISM Climate Group (4 km) | `prism-monthly-ca-region/prism_monthly.zarr` | Monthly 4 km precipitation & temperature |
| `08_download_gridmet_pdsi.ipynb` | UC Merced / Univ. of Idaho THREDDS | `pdsi-gridmet-california/pdsi_gridmet_monthly_CA.zarr` | Antecedent drought index (day-25 latency cutoff) |
| `09_download_snotel_daily.ipynb` | USDA NRCS AWDB / SNOTEL | `snotel-zarr/snotel.zarr` | Daily SNOTEL SWE & precipitation telemetry |

### Step 2: Run Pre-2021 Cross-Validation & Train Checkpoint

Open `notebooks/train_flowracle.ipynb` (a CUDA-capable GPU such as an NVIDIA T4,
L4, or A100 is recommended for full 4,000–7,000 tree training; `flowracle.model`
automatically falls back to CPU `hist` when no GPU is present):

```bash
export FLOWRACLE_DATA_DIR=/path/to/flowracle_data
export FLOWRACLE_OUTPUT_DIR=/path/to/model_output
jupyter notebook notebooks/train_flowracle.ipynb
```

Executing all cells runs the three pre-2021 expanding-window validation folds
(`2011-2014`, `2015-2017`, `2019-2020`, with 6 prepended predictor-history
months per slice) and writes the frozen production checkpoint
(`${FLOWRACLE_OUTPUT_DIR}/xgb_1634_v135_ensemble.pkl`) without accessing any
post-2020 observations.

### Step 3: Run Standalone Out-of-Sample Evaluation (`2021-2025`)

Open `notebooks/evaluate_flowracle.ipynb` (runs on CPU without requiring a GPU
once the preprocessed Zarr stores and trained checkpoint are stored locally):

```bash
export FLOWRACLE_DATA_DIR=/path/to/flowracle_data
export FLOWRACLE_OUTPUT_DIR=/path/to/model_output
jupyter notebook notebooks/evaluate_flowracle.ipynb
```

Executing all cells loads `xgb_1634_v135_ensemble.pkl`, strips ground-truth
targets prior to `predict()`, renders continuous probabilistic fan-chart
hydrographs across the elevation transect (`WWR`, `TLG`, `AMF`), benchmarks
held-out `2021-2025` forecasts against operational California DWR Bulletin 120
(pinball quantile loss per issuance month, head-to-head win rates, volumetric
bias, and discretized PIT calibration), and evaluates year-round pinball loss
per issuance month alongside water-year hydrograph timing diagnostics (`CT`,
`SDoT`, and late-summer runoff fraction).

## License

Copyright 2026 The Google Research Authors. All code in this directory is
licensed under the Apache License, Version 2.0 (see the root `LICENSE` file of
the `google-research` repository).
