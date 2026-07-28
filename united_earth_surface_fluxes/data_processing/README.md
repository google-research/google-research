# Data Processing


## Prerequisites

Dependencies can be installed via standard Python package managers. Ensure you have the following installed:
- `absl-py`
- `numpy`
- `pandas`
- `scipy`
- `xarray`
- `zarr`
- `tensorflow`

Since data streams natively from the `gs://gcp-public-data-arco-era5` bucket, no preliminary data downloading is required.

## Parameter Generation Notebook
This directory contains a Jupyter Notebook (`Vegetation_and_Soil_Parameters.ipynb`) describing the explicit logical pipeline mathematically interpolating ERA5 surface-level tabular bounds into physical modeling variables (such as ground heat, spatial capacities, and thermal boundaries).

Executing the full notebook dynamically pulls the raw data distributions, normalizes matrices, handles imputation for missing geographical entities, and exports `veg_params.csv`, `soil_params.csv`, and `ground_heat_params.csv` into the neighboring `../data` folder. These parameterized structures are required payloads in the overarching pipeline.

## Spatial Coordinates File
To target specific terrestrial coordinates along the map for the data pipeline to extract surface fluxes from, you must provide a JSON configuration file mapping latitude keys to lists of longitude intervals `[start, end]`. The pipeline expands these intervals at a 0.25-degree resolution to generate the target coordinates.

By default, the script looks for this relative to the root package at
`../data/spatial_coordinates.json`.

**Example format:**
```json
{
  "-54.0": [[291.75, 292.0]],
  "-53.75": [[291.0, 291.5]]
}
```

In our paper submission experiments, all spatial points defined in the provided `spatial_coordinates.json` were successfully utilized along with a `--period_list` spanning the entire year of 2024 (`"2024-01-01:2024-12-31"`), executing data processing and model training efficiently on a single NVIDIA A100 GPU.


## Running the Data Pipeline

Execute the data processing routines from within this directory (`data_processing/`). Doing so natively respects the relative standard mapping back to the root `data/` folder.

```bash
python main.py \
  --output_dir=../data \
  --spatial_coordinates_path=../data/spatial_coordinates.json \
  --era5_zarr_path="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3" \
  --period_list="2024-01-01:2024-12-31" \
  --pad_size=4 \
  --num_workers=30
```

### Arguments

* `--era5_zarr_path`: Path to the ARCO-ERA5 Zarr store. By default, it points to Google's public cloud bucket (`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`).
* `--period_list`: String defining exactly which days to process out of ERA5. Supports two distinct syntax patterns:
  1. **Colon Range Expansion (`start:end`)**: Automatically generates a continuous daily range between two dates (e.g., `"2024-01-01:2024-12-31"` expands to all 366 days in 2024).
  2. **Comma-Separated List (`day1,day2`)**: Processes discrete, non-sequential dates (e.g., `"2024-01-01,2024-06-03"`).
* `--pad_size`: Integer declaring the number of padding hours to buffer out the daily period boundaries (default is 4).
* `--num_workers`: Concurrent threads to allocate during dataset compilation (default is 30).
* `--output_dir`: Target directory mapped out to save the finalized `.tfrecord` shards (default is `../data`).
* `--spatial_coordinates_path`: Path targeting your JSON (default is `../data/spatial_coordinates.json`).
