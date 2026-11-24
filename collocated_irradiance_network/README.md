# Code accompanying the Collocated Irradiance Network paper

This package contains code accompanying the paper titled
"Estimates Of Broadband Upwelling Irradiance From GOES-16 ABI"

## CERES point spread function implementation

Point spread function weights can be calculated for arbitrary locations
quickly in this vectorized numpy implementation. See the unittest for usage
examples.

## Reading training/validation tfrecords

The training/validation data can be experimented with using other model
architectures. An example of reading the tfrecord format is provided in
the colab notebook.

## Loading irradiances

Model outputs can be used for researching climate phenomena which benefit from
the spatio-temporal resolution that GOES-16 ABI offers
(2km nominal, 10-15 minute refresh rate).
An example of reading the model outputs (irradiances) is provided
in the colab notebook.

## Running single-scantime COIN inference from public GOES East ABI L1b data

This package includes an end-to-end inference pipeline
(`run_single_scantime.py`) that reads the 16 spectral bands of raw GOES East ABI
L1b Full Disk radiances (`ABI-L1b-RadF`) from the public Google Cloud Storage
mirrors (`gs://gcp-public-data-goes-16` or `gs://gcp-public-data-goes-19`),
downscales 0.5 km and 1 km channels onto the 2 km GOES East fixed grid
(`5424 x 5424`), computes solar zenith and solar azimuth angles, and runs the
5-layer COIN MLPs (`525 -> 178 -> 60 -> 20 -> 1`) in TensorFlow to predict
top-of-atmosphere (TOA) Outgoing Longwave Radiation (`predicted_olr`, W m-2) and
Reflected Shortwave Radiation (`predicted_rsr`, W m-2).

Inference runs `SPLIT` model bundles (`olr/` with 15 inputs across longwave
channels 8--16; `rsr/` with 13 inputs across shortwave channels 1--7), using
periodic solar azimuth features (`goes_sin_solar_azimuths` and
`goes_cos_solar_azimuths`), and writes two CF-1.7 compliant files or Zarr stores
per scantime: `<stem>_olr.{nc|zarr}` and `<stem>_rsr.{nc|zarr}`.

### COIN v1.1 release notes

The original Collocated Irradiance Network (COIN) manuscript used a `DUAL` mode
(predicting both OLR and RSR from a shared set of model weights). Here we
provide `SPLIT` mode (separate OLR and RSR models) and periodic `(sin, cos)`
preprocessing of solar azimuth angle inputs. These changes fix two artifacts in
the original COIN outputs: `SPLIT` mode prevents shortwave (solar) channels from
influencing the OLR estimate, and `(sin, cos)` solar azimuth preprocessing fixes
a small but noticeable step-change artifact along the solar-noon and
solar-midnight semi-meridians (where solar azimuth wraps between `0` and
`2*pi`). Training used mean squared error (MSE) loss instead of the mean
absolute error (MAE) loss used in the manuscript to more reliably minimize
global average bias against CERES labels, and was restricted to GOES-16 ABI
10-minute cadence (`Mode 6A`) observations (`2019-04-02` through `2020-12-31`)
matching the current and future GOES-East operational scan cadence.

The portable `coin_v1.1` `SPLIT` bundle layout is:

```text
coin_v1.1/
  MODEL_MANIFEST.json
  olr/
    MODEL_MANIFEST.json
    coin_weights.npz
    train_config.json
  rsr/
    MODEL_MANIFEST.json
    coin_weights.npz
    train_config.json
```

| Subsystem | `label_mode` | `model_version` | Input / Layer Shapes | `weights_sha256` (`coin_weights.npz`) | `train_config_sha256` |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `olr/` | `OLR` | `coin_v1.1_olr` | `15 -> 525 -> 178 -> 60 -> 20 -> 1` | `cea16bfa0c494125b5aef08befe2f4e568da4fffdaa99024484fc9a7d6bca2fb` | `615f581b5fdb9a4cb4800d6ee0af2cabc181bbbfb99e69d7214ec6806cf6dc2c` |
| `rsr/` | `RSR` | `coin_v1.1_rsr` | `13 -> 525 -> 178 -> 60 -> 20 -> 1` | `0d8eb521129e792ec396371a8de833d9d4459bbe37229ef6958a36321f93a8fa` | `98c999fa8ce2809ccdedd4344b31cf24188312979263185f87c65bc1f21c9805` |

Evaluated across the 10-minute cadence CERES validation split (`2019-04-02`
through `2020-12-31`, `N = 8,085,440` collocated footprints), `coin_v1.1`
achieves:

- **CERES Level 2 (`SSF`) footprint accuracy**: `OLR` `Bias = +0.020 W/m^2`,
  `RMSE = 6.629 W/m^2`, `R^2 = 0.9766`; daytime `RSR` `Bias = -0.080 W/m^2`,
  `RMSE = 24.389 W/m^2`, `R^2 = 0.9737` (post-April 23, 2019 Band 2 calibration
  update: `Bias = -0.338 W/m^2`, `RMSE = 24.311 W/m^2`, `R^2 = 0.9740`).
- **CERES Level 3 (`SYN1deg`) gridded accuracy**: `OLR` `RMSE = 4.745 W/m^2`
  (`Bias = +0.106 W/m^2`); daytime `RSR` `RMSE = 22.149 W/m^2`
  (`Bias = -0.347 W/m^2`).

### Quickstart (Checkout, Tests, and Single-Scantime Inference)

```bash
# 1. Sparse-checkout only collocated_irradiance_network from google-research.
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-research/google-research.git
cd google-research
git sparse-checkout set collocated_irradiance_network

# 2. Create coin_venv, install requirements.txt, and run unit tests.
./collocated_irradiance_network/run.sh

# 3. Download the portable coin_v1.1 model bundle and run full-disk inference on
#    s20191521700447 (2019-06-01 17:00:44.7 UTC, GOES-16 Mode 6 Full Disk).
mkdir -p ./models/coin_v1.1
gcloud storage cp -r \
  "gs://upwelling_irradiance/models/coin_v1.1/*" \
  ./models/coin_v1.1/

source ./coin_venv/bin/activate
python -m collocated_irradiance_network.run_single_scantime \
  --scan_timekey=s20191521700447 \
  --model_dir=./models/coin_v1.1 \
  --output_path=/tmp/coin_v1.1_s20191521700447.nc \
  --batch_size=262144 \
  --work_dir=/tmp/coin_abi_staging_s20191521700447 \
  --input_source=gcs \
  --gcs_bucket=gcp-public-data-goes-16 \
  --max_decode_workers=8
```

Pass `--gcs_bucket=gcp-public-data-goes-19` for GOES-19 scans,
`--scan_timekey=s20180211630395` (2018-01-21 16:30 UTC, Mode 3) to target a
different scan, or `--output_path=/tmp/coin_v1.1_s20191521700447.zarr` to write
Zarr v2 (`.zarr`) instead of NetCDF4 (`.nc`).

### Full-disk runtime benchmarks

Benchmarked on GOES-16 Mode 6 Full Disk scan `s20191521700447` (`5424 x 5424`
grid, `21,199,940` valid in-range pixels, `batch_size=262144`), reporting both
first-scan cold-grid (`69.47 s`) and warm cached-grid (`~54.0 s`) end-to-end
wall times:

| Configuration | Device | MLP + Postprocess + Scatter | End-to-End Wall Time (Cold Grid) | End-to-End Wall Time (Cached Grid) | Peak RSS (GiB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `SPLIT` (TensorFlow) | CPU (16 decode workers) | 17.56 s | 69.47 s | ~54.0 s | 8.75 |
