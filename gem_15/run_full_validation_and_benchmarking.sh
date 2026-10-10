#!/bin/bash
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



set -euo pipefail

# ==============================================================================
# Configuration & Defaults
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-./validation_outputs}"
EE_PROJECT="${EE_PROJECT:-}"
EE_SERVICE_ACCOUNT="${EE_SERVICE_ACCOUNT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GEDI_SAMPLES="${GEDI_SAMPLES:-100000}"
AERIAL_SAMPLES="${AERIAL_SAMPLES:-1000}"
WIPE_FIRST="${WIPE_FIRST:-false}"
ONLY_PLOTS="${ONLY_PLOTS:-false}"

PKG="gem_15"

VALIDATE_MOD="${PKG}.validate_gem15_against_lidar"
VALIDATE_CORS_MOD="${PKG}.validate_gem15_against_cors"
VALIDATE_PERCENTILES_MOD="${PKG}.validate_lidar_percentiles"
PLOT_PRIMARY_MOD="${PKG}.plots.plot_primary_benchmarks"
PLOT_STRATIFIED_MOD="${PKG}.plots.plot_ground_and_above_ground_benchmarks"
PLOT_SLOPE_MOD="${PKG}.plots.plot_slope_stratified_benchmarks"
PLOT_TEMPORAL_MOD="${PKG}.plots.plot_temporal_improvement_aerial"
PLOT_PERCENTILES_MOD="${PKG}.plots.plot_percentiles_benchmarks"
PLOT_NEON_MOD="${PKG}.plots.plot_temporal_neon"
PLOT_CORS_MOD="${PKG}.plots.plot_cors_benchmark"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
  echo -e "${GREEN}[INFO] $(date '+%Y-%m-%d %H:%M:%S')${NC} $1"
}

log_step() {
  echo -e "\n${BLUE}======================================================================${NC}"
  echo -e "${BLUE}[STEP] $1${NC}"
  echo -e "${BLUE}======================================================================${NC}"
}

log_warn() {
  echo -e "${YELLOW}[WARN] $1${NC}"
}

log_error() {
  echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S')${NC} $1"
}

# ==============================================================================
# Parse Command Line Options
# ==============================================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --wipe)
      WIPE_FIRST=true
      shift
      ;;
    --only_plots)
      ONLY_PLOTS=true
      shift
      ;;
    --output_root=*)
      OUTPUT_ROOT="${1#*=}"
      shift
      ;;
    --ee_project=*)
      EE_PROJECT="${1#*=}"
      shift
      ;;
    --ee_service_account=*)
      EE_SERVICE_ACCOUNT="${1#*=}"
      shift
      ;;
    --gedi_samples=*)
      GEDI_SAMPLES="${1#*=}"
      shift
      ;;
    --aerial_samples=*)
      AERIAL_SAMPLES="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --ee_project=PROJECT [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --ee_project=PROJECT        Earth Engine Cloud project (required unless --only_plots)"
      echo "  --ee_service_account=EMAIL  Optional service account for EE authentication"
      echo "  --output_root=PATH          Base output directory (default: $OUTPUT_ROOT)"
      echo "  --wipe                      Wipe existing data in OUTPUT_ROOT before starting"
      echo "  --only_plots                Skip Step 1 validation and run Step 3 plotting only"
      echo "  --gedi_samples=N            Max GEDI regions to evaluate (default: $GEDI_SAMPLES)"
      echo "  --aerial_samples=N          Max Aerial/Drone regions to evaluate (default: $AERIAL_SAMPLES)"
      echo "  -h, --help                  Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "${ONLY_PLOTS}" != "true" && -z "${EE_PROJECT}" ]]; then
  log_error "--ee_project=PROJECT is required when running validation benchmarks."
  exit 1
fi

LOG_DIR="/tmp/gem15_benchmarks_$(date +%s)"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
cd "${REPO_ROOT}"

log_info "Using Output Destination: ${OUTPUT_ROOT}"
log_info "Using Python Binary     : ${PYTHON_BIN} (package: ${PKG})"
log_info "Using EE Cloud Project  : ${EE_PROJECT}"
log_info "Using Logs Directory    : ${LOG_DIR}"

EE_ARGS=()
if [[ -n "${EE_PROJECT}" ]]; then
  EE_ARGS+=("--ee_project=${EE_PROJECT}")
fi
if [[ -n "${EE_SERVICE_ACCOUNT}" ]]; then
  EE_ARGS+=("--ee_service_account=${EE_SERVICE_ACCOUNT}")
fi

# ==============================================================================
# Step 0: Pre-cleanup (Optional)
# ==============================================================================
if [[ "${WIPE_FIRST}" == "true" ]]; then
  if [[ -z "${OUTPUT_ROOT}" || "${OUTPUT_ROOT}" == "/" || "${OUTPUT_ROOT}" == "${REPO_ROOT}" || "${OUTPUT_ROOT}" == "${SCRIPT_DIR}" ]]; then
    log_error "Refusing to wipe unsafe OUTPUT_ROOT: '${OUTPUT_ROOT}'"
    exit 1
  fi
  log_step "Step 0: Clearing existing data from ${OUTPUT_ROOT}"
  log_warn "Deleting previous files under ${OUTPUT_ROOT}..."
  rm -rf "${OUTPUT_ROOT:?}"
  mkdir -p "${OUTPUT_ROOT}"
  log_info "Cleanup complete. Directory is clean and empty."
fi

# ==============================================================================
# Step 1: Launch all Validation Targets in Parallel Threads
# ==============================================================================
if [[ "${ONLY_PLOTS}" != "true" ]]; then
  log_step "Launching Validation Benchmarks in Parallel Threads"

  declare -A PIDS=()
  declare -A TASK_NAMES=()

  run_task() {
    local task_key="$1"
    local task_name="$2"
    shift 2
    local log_file="${LOG_DIR}/${task_key}.log"

    local out_dir=""
    for arg in "$@"; do
      if [[ "${arg}" == --output_dir=* ]]; then
        out_dir="${arg#*=}"
      fi
    done
    if [[ -n "${out_dir}" && -s "${out_dir}/results.csv" ]]; then
      log_info "Skipping [${task_name}]: results.csv already exists in ${out_dir}"
      return 0
    fi

    log_info "Starting [${task_name}] in background thread..."
    "${PYTHON_BIN}" -m "${VALIDATE_MOD}" "${EE_ARGS[@]}" "$@" > "${log_file}" 2>&1 &
    local pid=$!
    PIDS["${pid}"]="${task_key}"
    TASK_NAMES["${pid}"]="${task_name}"
  }

  run_binary_task() {
    local task_key="$1"
    local task_name="$2"
    local mod_target="$3"
    shift 3
    local log_file="${LOG_DIR}/${task_key}.log"

    local out_dir=""
    for arg in "$@"; do
      if [[ "${arg}" == --output_dir=* ]]; then
        out_dir="${arg#*=}"
      fi
    done
    if [[ -n "${out_dir}" ]]; then
      if [[ -s "${out_dir}/results.csv" || -s "${out_dir}/lidar_percentiles_results.csv" ]]; then
        log_info "Skipping [${task_name}]: outputs already exist in ${out_dir}"
        return 0
      fi
    fi

    log_info "Starting [${task_name}] in background thread..."
    "${PYTHON_BIN}" -m "${mod_target}" "${EE_ARGS[@]}" "$@" > "${log_file}" 2>&1 &
    local pid=$!
    PIDS["${pid}"]="${task_key}"
    TASK_NAMES["${pid}"]="${task_name}"
  }

  # 1. Spaceborne GEDI
  run_task "gedi" "GEDI Spaceborne" \
    --mode=gedi \
    --max_regions_limit="${GEDI_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/gedi_global_$((GEDI_SAMPLES / 1000))k" \
    --num_export_chunks=32

  # 2. LiDAR Percentiles & Sub-pixel Vertical Envelope (All 12 Drone & Aerial LiDAR Datasets)
  PERCENTILE_DATASETS=(
    "global_tech_mahindra:Global Drone"
    "india_tech_mahindra:India Drone"
    "3dtrees:3D Trees Drone"
    "uk_ea:UK England Aerial"
    "new_zealand:New Zealand Aerial"
    "canada:Canada Aerial"
    "spain:Spain Aerial"
    "brazil:Brazil Aerial"
    "indonesia:Indonesia Aerial"
    "mozambique:Mozambique Aerial"
    "us_tropical:US Tropical Aerial"
    "neon_2022:US NEON Aerial (2022)"
    "neon_2023:US NEON Aerial (2023)"
    "neon_2024:US NEON Aerial (2024)"
    "neon_2025:US NEON Aerial (2025)"
  )

  for entry in "${PERCENTILE_DATASETS[@]}"; do
    ds_key="${entry%%:*}"
    ds_name="${entry#*:}"
    run_binary_task "percentiles_${ds_key}" "LiDAR Percentiles (${ds_name})" "${VALIDATE_PERCENTILES_MOD}" \
      --lidar_dataset="${ds_key}" \
      --output_dir="${OUTPUT_ROOT}/lidar_percentiles_${ds_key}" \
      --max_regions_limit="${AERIAL_SAMPLES}" \
      --random_seed=42
  done

  # 3. International GNSS CORS Geodetic Survey Checkpoints (2025, 6 Continents)
  run_binary_task "cors" "International GNSS CORS Checkpoints (2025)" "${VALIDATE_CORS_MOD}" \
    --output_dir="${OUTPUT_ROOT}/cors_checkpoints" \
    --year=2025 \
    --num_workers=12 \
    --resample_bilinear=True

  # 4. Drone LiDAR (3 datasets)
  run_task "global_drone" "Global Drone LiDAR" \
    --mode=lidar \
    --lidar_dataset=global_tech_mahindra \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/global_tech_mahindra_drone_lidar_gem15"

  run_task "india_drone" "India Drone LiDAR" \
    --mode=lidar \
    --lidar_dataset=india_tech_mahindra \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/india_tech_mahindra_drone_lidar_gem15"

  run_task "3dtrees_drone" "3D Trees Drone LiDAR" \
    --mode=lidar \
    --lidar_dataset=3dtrees \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/3dtrees_drone_lidar_gem15"

  # 5. National & Regional Aerial LiDAR (8 datasets)
  run_task "new_zealand" "New Zealand Aerial LiDAR" \
    --mode=lidar \
    --lidar_dataset=new_zealand \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/new_zealand_lidar_gem15"

  run_task "canada" "Canada Aerial LiDAR" \
    --mode=lidar \
    --lidar_dataset=canada \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/canada_lidar_gem15"

  run_task "spain" "Spain Aerial LiDAR" \
    --mode=lidar \
    --lidar_dataset=spain \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/spain_lidar_gem15"

  run_task "uk_ea" "UK England Aerial LiDAR" \
    --mode=lidar \
    --lidar_dataset=uk_ea \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/uk_ea_lidar_gem15"

  run_task "brazil" "Brazil Aerial LiDAR" \
    --mode=lidar \
    --lidar_dataset=brazil \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/brazil_lidar_gem15"

  run_task "indonesia" "Indonesia Aerial LiDAR" \
    --mode=lidar \
    --lidar_dataset=indonesia \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/indonesia_lidar_gem15"

  run_task "mozambique" "Mozambique Aerial LiDAR" \
    --mode=lidar \
    --lidar_dataset=mozambique \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/mozambique_lidar_gem15"

  run_task "us_tropical" "US Tropical Aerial LiDAR" \
    --mode=lidar \
    --lidar_dataset=us_tropical \
    --max_regions_limit="${AERIAL_SAMPLES}" \
    --output_dir="${OUTPUT_ROOT}/us_tropical_lidar_gem15"

  # 6. US NEON Annual Aerial LiDAR Surveys (2013-2025)
  # Range must match --min_year/--max_year passed to plot_temporal_neon below.
  for year in $(seq 2013 2025); do
    run_task "neon_${year}" "US NEON Aerial LiDAR (${year})" \
      --mode=lidar \
      --lidar_dataset="neon_${year}" \
      --max_regions_limit="${AERIAL_SAMPLES}" \
      --output_dir="${OUTPUT_ROOT}/neon_${year}_lidar_gem15"
  done

  # 7. US NEON Constant Cohort Series:
  # Series 1: 14 Continental US Sites (2016-2018)
  for year in 2016 2017 2018; do
    run_task "neon_series_1_${year}" "US NEON Series 1 14-Sites (${year})" \
      --mode=lidar \
      --lidar_dataset="neon_series_1_${year}" \
      --max_regions_limit="${AERIAL_SAMPLES}" \
      --output_dir="${OUTPUT_ROOT}/neon_series_1_${year}_lidar_gem15"
  done

  # Series 2: 12 Continental US Sites (2019, 2021, 2023, 2025)
  for year in 2019 2021 2023 2025; do
    run_task "neon_series_2_${year}" "US NEON Series 2 12-Sites (${year})" \
      --mode=lidar \
      --lidar_dataset="neon_series_2_${year}" \
      --max_regions_limit="${AERIAL_SAMPLES}" \
      --output_dir="${OUTPUT_ROOT}/neon_series_2_${year}_lidar_gem15"
  done

  # ==============================================================================
  # Step 2: Monitor and Wait for All Parallel Threads
  # ==============================================================================
  FAILED_TASKS=0
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    log_info "All ${#PIDS[@]} validation threads launched concurrently."
    for pid in "${!PIDS[@]}"; do
      task_key="${PIDS[${pid}]}"
      task_name="${TASK_NAMES[${pid}]}"
      if wait "${pid}"; then
        log_info "SUCCESS: [${task_name}] completed successfully."
      else
        log_error "FAILED: [${task_name}] (PID ${pid}) exited with error. Check log: ${LOG_DIR}/${task_key}.log"
        FAILED_TASKS=$((FAILED_TASKS + 1))
      fi
    done
  else
    log_info "All 44 validation datasets already present in ${OUTPUT_ROOT}; proceeding directly to Step 3 plotting."
  fi

  if [[ ${FAILED_TASKS} -gt 0 ]]; then
    log_error "${FAILED_TASKS} validation task(s) failed. Review logs in: ${LOG_DIR}"
    exit 1
  fi
fi

log_info "All validation benchmark datasets ready!"

# ==============================================================================
# Step 3: Compile Summary Metrics & Render Benchmark Publication Plots
# ==============================================================================
log_step "Compiling benchmark_summary.csv and Generating Publication Plots"

# 3.1 Primary Benchmark Plots (Debiased RMSE, Direct RMSE, Mean Bias)
log_info "Generating Primary Benchmark Charts..."
"${PYTHON_BIN}" -m "${PLOT_PRIMARY_MOD}" \
  --data_dir="${OUTPUT_ROOT}" \
  --output_dir="${OUTPUT_ROOT}/plots"

# 3.2 Stratified Ground vs Above Ground Benchmark Plots (16 charts)
log_info "Generating Ground and Above Ground Benchmark Charts..."
"${PYTHON_BIN}" -m "${PLOT_STRATIFIED_MOD}" \
  --data_dir="${OUTPUT_ROOT}" \
  --output_dir="${OUTPUT_ROOT}/plots"

# 3.3 Topographic Slope Stratified Benchmark Plots (Flat < 10 deg vs Steep >= 10 deg, 16 charts)
log_info "Generating Topographic Slope Stratified Benchmark Charts (Flat < 10° vs Steep >= 10°)..."
"${PYTHON_BIN}" -m "${PLOT_SLOPE_MOD}" \
  --data_dir="${OUTPUT_ROOT}" \
  --output_dir="${OUTPUT_ROOT}/plots"

# 3.4 Primary Sloped Terrain (Slope > 5 deg) DTM Benchmark Plots (3 charts)
log_info "Generating Primary Sloped Terrain (Slope > 5 deg) DTM Benchmark Charts..."
"${PYTHON_BIN}" -m "${PLOT_PRIMARY_MOD}" \
  --data_dir="${OUTPUT_ROOT}" \
  --summary_csv="${OUTPUT_ROOT}/benchmark_summary_slope_gt_5.csv" \
  --min_slope_deg=5.0 \
  --dtm_only=true \
  --title_suffix=" (Slope > 5°)" \
  --output_suffix="_slope_gt_5" \
  --output_dir="${OUTPUT_ROOT}/plots"

# 3.5 Stratified Sloped Terrain (Slope > 5 deg) DTM Benchmark Plots (6 charts)
log_info "Generating Stratified Sloped Terrain (Slope > 5 deg) DTM Benchmark Charts..."
"${PYTHON_BIN}" -m "${PLOT_STRATIFIED_MOD}" \
  --data_dir="${OUTPUT_ROOT}" \
  --summary_csv="${OUTPUT_ROOT}/benchmark_summary_slope_gt_5.csv" \
  --min_slope_deg=5.0 \
  --dtm_only=true \
  --title_suffix=" (Slope > 5°)" \
  --output_suffix="_slope_gt_5" \
  --output_dir="${OUTPUT_ROOT}/plots"

# 3.6 Temporal Improvement Over Copernicus Aerial Plot
log_info "Generating Temporal Improvement Chart..."
"${PYTHON_BIN}" -m "${PLOT_TEMPORAL_MOD}" \
  --data_dir="${OUTPUT_ROOT}" \
  --output_dir="${OUTPUT_ROOT}/plots"

# 3.7 LiDAR Percentiles Multi-Dataset Benchmark Plots
log_info "Generating LiDAR Percentiles Multi-Dataset Benchmark Charts..."
"${PYTHON_BIN}" -m "${PLOT_PERCENTILES_MOD}" \
  --data_dir="${OUTPUT_ROOT}" \
  --summary_csv="${OUTPUT_ROOT}/plots/percentiles_benchmark_summary.csv" \
  --output_dir="${OUTPUT_ROOT}/plots"

# 3.8 US NEON Annual Timeseries (2013-2025) Plots
log_info "Generating US NEON Annual Timeseries (2013-2025) Charts..."
"${PYTHON_BIN}" -m "${PLOT_NEON_MOD}" \
  --data_dir="${OUTPUT_ROOT}" \
  --min_year=2013 \
  --max_year=2025 \
  --output_dir="${OUTPUT_ROOT}/plots"

# 3.9 International GNSS CORS Continent-Stratified Benchmark Plots (2025)
log_info "Generating International GNSS CORS Continent-Stratified Benchmark Charts..."
"${PYTHON_BIN}" -m "${PLOT_CORS_MOD}" \
  --data_dir="${OUTPUT_ROOT}" \
  --output_dir="${OUTPUT_ROOT}/plots" \
  --year=2025

log_step "ALL VALIDATION DATASETS & BENCHMARK PLOTS SUCCESSFULLY GENERATED!"
log_info "Output Release Directory : ${OUTPUT_ROOT}"
log_info "Output Plots Directory   : ${OUTPUT_ROOT}/plots"
log_info "Compiled Summary CSV     : ${OUTPUT_ROOT}/benchmark_summary.csv"
log_info "Sloped Summary CSV       : ${OUTPUT_ROOT}/benchmark_summary_slope_gt_5.csv"
log_info "Percentiles CSV          : ${OUTPUT_ROOT}/plots/percentiles_benchmark_summary.csv"
log_info "Thread Execution Logs    : ${LOG_DIR}"
