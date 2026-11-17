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


#
# Downloads the latest NSSP raw Parquet data from the CDC RSV Forecast Hub.
#
# Usage:
#   bash pipeline/data_prep/download_nssp_parquet.sh [OUTPUT_DIR]
#
# The file is saved as OUTPUT_DIR/latest_YYYYMMDD.parquet. OUTPUT_DIR defaults
# to ./data.

set -e

# The GitHub URL for the *raw* Parquet file.
GITHUB_RAW_URL="https://raw.githubusercontent.com/CDCgov/rsv-forecast-hub/main/auxiliary-data/nssp-raw-data/latest.parquet"
OUTPUT_DIR="${1:-./data}"

echo "Starting download of NSSP raw Parquet data..."

# Generate a timestamp for the output.
TIMESTAMP=$(date +%Y%m%d)
OUTPUT_FILENAME="latest_${TIMESTAMP}.parquet"
OUTPUT_FILE_PATH="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

# Create a temporary file.
TMP_FILE=$(mktemp)
echo "Downloading raw file to temporary location: ${TMP_FILE}"

# Download the Parquet file using curl.
if ! curl -sL "${GITHUB_RAW_URL}" -o "${TMP_FILE}"; then
  echo "Error: Failed to download file from ${GITHUB_RAW_URL}"
  rm -f "${TMP_FILE}" # Clean up temp file
  exit 1
fi

if [[ ! -s "${TMP_FILE}" ]]; then
  echo "Error: Downloaded file is empty. The URL might be incorrect or the file does not exist."
  rm -f "${TMP_FILE}" # Clean up temp file
  exit 1
fi

echo "Successfully downloaded Parquet data to ${TMP_FILE}"

echo "Ensuring output directory exists: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Copying ${TMP_FILE} to: ${OUTPUT_FILE_PATH}"
if ! cp -f "${TMP_FILE}" "${OUTPUT_FILE_PATH}"; then
  echo "Error: Failed to copy file to ${OUTPUT_FILE_PATH}."
  rm -f "${TMP_FILE}" # Clean up temp file
  exit 1
fi

rm -f "${TMP_FILE}"

echo "Successfully saved NSSP data to: ${OUTPUT_FILE_PATH}"
