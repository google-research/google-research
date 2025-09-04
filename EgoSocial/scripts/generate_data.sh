#!/bin/bash
# Copyright 2025 The Google Research Authors.
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



# This script runs the data.py script to extract frames and audio from Ego4D clips.
# --- Configuration ---
# Replace these paths with your actual paths
ANNOTATION_FILE="../annotations/egosocial_annotations.json"
EGO4D_CLIPS_DIR="/path/to/ego4d/clips/"
OUTPUT_DIR="/path/to/output/dir"

# Path to the data extraction script
SCRIPT_PATH="$(dirname "$0")/data.py"

# --- Execution ---
echo "Starting data extraction..."

python3 "${SCRIPT_PATH}" \
    --annotation_file "${ANNOTATION_FILE}" \
    --ego4d_clips_dir "${EGO4D_CLIPS_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "Data extraction finished."
