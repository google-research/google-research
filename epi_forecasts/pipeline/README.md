# Forecast Pipeline Utilities

This directory contains the utility scripts used around the individual forecast
models: preparing the input datasets, and combining the individual model
forecasts into the `Google_SAI` ensemble submissions.

The steps are listed in the order in which they are run to produce a forecast:

Step | Directory / File                     | Description
:--- | :----------------------------------- | :----------
1    | `data_prep/download_nssp_parquet.sh` | Downloads the latest NSSP data from the CDC RSV Forecast Hub (RSV only).
1    | `data_prep/preprocess_input.py`      | Builds the influenza or COVID-19 input dataset from NHSN data.
1    | `data_prep/preprocess_input_rsv.py`  | Builds the RSV input dataset from NHSN and NSSP data.
2    | `../*_hub/model_py/`                 | The individual forecast models (not part of this directory). Their submissions are stored in `../*_hub/model_output/`.
3    | `create_ensemble.py`                 | Builds the ensemble forecast for one forecast date from the individual model submissions.
4    | `merge_forecasts.py`                 | Optional: merges all weekly submissions of the ensemble members into one wide-format CSV.

`infection_configs.json` is the configuration shared by the scripts. For each
pathogen it defines the target column, the CDC target name, the ensemble name,
the model output directory and the ensemble members.

## Setup

```sh
pip install -r pipeline/requirements.txt
```

All commands below are run from the `epi_forecasts` directory.

## Step 1: Preparing input data

The scripts use the following public inputs:

*   **NHSN**: the "Weekly Hospital Respiratory Data (HRD) Metrics by
    Jurisdiction, National Healthcare Safety Network (NHSN) (Preliminary)" CSV
    export from [data.cdc.gov](https://data.cdc.gov).
*   **Locations**: the `auxiliary-data/locations.csv` file of the CDC forecast
    hub repositories.
*   **NSSP** (RSV only): downloaded with `data_prep/download_nssp_parquet.sh`.

```sh
# Influenza or COVID-19 dataset. Writes <output_path>/dataset_<suffix>.csv,
# where <suffix> is the part of the NHSN file name after the last underscore.
python pipeline/data_prep/preprocess_input.py \
    --infection=flu \
    --nhsn_input=data/Weekly_Hospital_Respiratory_Data_Preliminary_20250924.csv \
    --locations_path=data/locations.csv \
    --output_path=data/flu/

# RSV dataset (NHSN + NSSP).
bash pipeline/data_prep/download_nssp_parquet.sh data
python pipeline/data_prep/preprocess_input_rsv.py \
    --nhsn_input_path=data/Weekly_Hospital_Respiratory_Data_Preliminary_20260107.csv \
    --nssp_input_path=data/latest_20260107.parquet \
    --locations_path=data/locations.csv \
    --output_csv_path=data/rsv/dataset_with_nssp_20260107.csv
```

## Step 2: Running the individual models

The individual models are in the `model_py/` directory of each pathogen
directory (`flu_hub/`, `covid_hub/`, `rsv_hub/`). Their forecasts are stored as
`<hub>/model_output/<model_name>/<forecast_date>-<model_name>.csv`, which is
where the next step reads them from.

Model names are the external model IDs (e.g. `Google_SAI-Adapted_3`). The
`README.md` of each pathogen directory maps them to the names used in the paper.

## Step 3: Creating the ensemble

`create_ensemble.py` reads the submissions of the ensemble members for one
forecast date, averages each quantile across models, enforces non-negative and
monotonically increasing quantiles with isotonic regression, and writes a CSV in
the CDC hub submission format.

Submissions are read from
`<model_output_dir>/<model_name>/<forecast_date>-<model_name>.csv`. Members
without a submission for the requested date are skipped with a warning.

```sh
python pipeline/create_ensemble.py --infection=flu --forecast_date=2026-01-10
```

By default:

*   the ensemble members are the `model_names_for_ensemble` of the pathogen in
    `infection_configs.json` (override with `--model_names=A,B,C`);
*   submissions are read from the pathogen's `*_hub/model_output` directory
    (override with `--model_output_dir`);
*   the ensemble is written to
    `<model_output_dir>/<ensemble_name>/<forecast_date>-<ensemble_name>.csv`
    (override the directory with `--output_dir`).

## Step 4 (optional): Merging forecasts

`merge_forecasts.py` collects all weekly submissions of the ensemble members
into a single wide-format CSV with one column per quantile. It is not needed to
create a submission.

```sh
python pipeline/merge_forecasts.py \
    --infection=flu \
    --output_path=/tmp/flu_merged_forecasts.csv
```
