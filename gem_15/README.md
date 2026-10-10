# GEM-15: Global Elevation Model (15m) Validation & Benchmarking Suite

This directory contains the validation, benchmarking, and figure generation
code for **GEM-15** (15m resolution):

1. **Satellite, Aerial and Drone LiDAR DSM & DTM Benchmarks**
   (`validate_gem15_against_lidar.py`): Evaluates GEM-15 DSM and DTM against 12
   high-resolution (1m) airborne and drone LiDAR reference datasets across 6
   continents, alongside global 30m satellite DEM baselines (Copernicus GLO-30,
   ALOS AW3D30, NASADEM, and FABDEM), including multi-year temporal evaluation
   across US NEON (2013–2025).
2. **Global GNSS / CORS Geodetic Station Benchmarks**
   (`validate_gem15_against_cors.py`): Evaluates GEM-15 against 1,192
   international CORS geodetic stations (2025) worldwide.
3. **Percentile Benchmarks** (`validate_lidar_percentiles.py`):
   Evaluates GEM-15 $P_{95}$, DSM, $P_{5}$, and DTM layers against 15m
   percentile aggregations of 1m airborne and drone LiDAR.
4. **Figures & Visualizations** (`plots/`, `main_figure_3d_generation.py`,
   `generate_hydrology_figure.py`, `generate_forestry_figure.py`,
   `generate_urban_figure.py`):
   Generates all quantitative benchmark charts, stratified slope/canopy
   analyses, error distribution plots, qualitative multi-dataset comparisons, 3D
   digital twin visualizations, and environmental application figures.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Authenticate with Google Earth Engine:

```bash
earthengine authenticate
```

### Configuring Earth Engine Assets (`ee_assets_config.py`)

All Earth Engine asset paths used across the validation and plotting scripts are
centralized in `ee_assets_config.py`:

* **Public Earth Engine Datasets**: Global satellite DEMs (`GEM-15`,
  `Copernicus GLO-30`, `ALOS AW3D30`, `NASADEM`, `FABDEM`, `SRTM`), spaceborne
  `GEDI L2A`, land-cover maps (`ESA WorldCover`, `MODIS MCD12Q1`), and the
  UK Environment Agency (`UK/EA/ENGLAND_1M_TERRAIN/2022`) and US NEON
  (`projects/neon-prod-earthengine/assets/DEM/001`) aerial LiDAR collections
  are already available directly in the public Earth Engine catalog and work out
  of the box.
* **Validation Aerial & Drone LiDAR Collections**: To run the full aerial/drone
  LiDAR benchmarking across the remaining regions, you must first ingest the
  validation LiDAR datasets into your own Earth Engine project and update the
  corresponding placeholder paths (`projects/YOUR_PROJECT/assets/lidar/...`) in
  `ee_assets_config.py`. Most of these LiDAR collections are publicly available
  from open data repositories (including Canada HRDEM, Spain PNOA, New Zealand
  LINZ, US USGS 3DEP, Brazil Sustainable Landscapes, Indonesia, Mozambique, and
  3D Trees Drone), with the exception of the proprietary Tech Mahindra Drone
  datasets.
* **Geoid Offset Grids**: Similarly, vertical datum transformations require
  geoid height grids (`EGM96`, `EGM2008`, `OSGM15`, `CGVD2013`, `REDNAP`, and
  `NZVD2016`), which can be obtained from public geodetic data sources (such as
  PROJ / Agisoft geoid grids or national mapping agencies), uploaded as Earth
  Engine image assets, and configured in `ee_assets_config.py`
  (`projects/YOUR_PROJECT/assets/geoid_offsets/...`).

## Running the Full End-to-End Validation & Plotting Pipeline

To run all validation benchmarks and generate all quantitative and qualitative
figures in one command, launch `run_full_validation_and_benchmarking.sh` with
your Earth Engine Google Cloud project ID:

```bash
./run_full_validation_and_benchmarking.sh \
  --ee_project=your-cloud-project-id \
  --output_root=./validation_outputs
```

Useful options for `run_full_validation_and_benchmarking.sh`:

* `--ee_project=<project_id>`: Google Cloud project ID for Earth Engine
  (required unless `--only_plots` is set).
* `--ee_service_account=<email>`: Optional service account principal to
  impersonate.
* `--output_root=<dir>`: Directory where all CSV metrics and PNG/JPG plots are
  saved (default: `./validation_outputs`).
* `--only_plots`: Skip Earth Engine validation extraction and regenerate all
  plots from existing CSVs in `--output_root`.
* `--wipe`: Clear existing outputs in `--output_root` before running.
* `--gedi_samples=<N>`: Maximum number of GEDI regions to evaluate (default:
  `100000`).
* `--aerial_samples=<N>`: Maximum number of Aerial/Drone LiDAR regions to
  evaluate per dataset (default: `1000`).

## Running Unit Tests

From the `google_research` repository root:

```bash
python3 -m unittest gem_15/metrics_test.py
```
