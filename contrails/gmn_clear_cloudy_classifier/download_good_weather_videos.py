# coding=utf-8
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

import os
import subprocess
import re
import sys
import json
import datetime


# Configuration
SSH_KEY = "TODO_INSERT_PATH_TO_YOUR_SSH_KEY"
SFTP_USER = "TODO_INSERT_YOUR_SFTP_USER"
REMOTE_BASE_DIR = "/home/contrails/CA"
CAWEC_IDS = ["CAWEC2", "CAWEC3", "CAWEC4", "CAWEC5", "CAWEC6"]
TARGET_MP4_COUNT = 200
LOCAL_DOWNLOAD_DIR = "raw_data_daytime"
GOOD_DATES_FILE = "good_weather_dates.json"
BATCH_SIZE = 10

def get_daytime_dirs(cawec_id):
  """Gets a list of remote directory names that appear to be daytime clips."""
  remote_path = f"{REMOTE_BASE_DIR}/{cawec_id}/"
  cmd = f"sftp -o IdentitiesOnly=yes -i {SSH_KEY} {SFTP_USER} <<< 'ls {remote_path}'"
  print(f"Fetching directory list for {cawec_id}...")
  try:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=45)
    if result.returncode != 0:
      return []

    all_dirs = [os.path.basename(line.strip()) for line in result.stdout.splitlines() if line.strip() and not line.startswith('sftp>')]

    daytime_dirs = []
    for dirname in all_dirs:
      match = re.match(rf"{cawec_id}_(\d{{8}})_(\d{{6}})_to_.*", dirname)
      if match and 8 <= int(match.group(2)[:2]) < 18:
        daytime_dirs.append(dirname)
    return daytime_dirs
  except:
    return []

def convert_dir_to_file_base(dirname):
    return re.sub(r"(\d{8})_(\d{6})", r"\1-\2", dirname)

def get_target_dates(good_dates_list):
    target_dates = set()
    for d_str in good_dates_list:
        try:
            d = datetime.datetime.strptime(d_str, "%Y%m%d")
            target_dates.add(d.strftime("%Y%m%d"))
            target_dates.add((d + datetime.timedelta(days=1)).strftime("%Y%m%d"))
        except:
            pass
    return target_dates

def main():
  if not os.path.exists(GOOD_DATES_FILE):
    print("Good dates file not found!")
    return

  with open(GOOD_DATES_FILE, 'r') as f:
    good_dates_data = json.load(f)

  all_download_paths = []

  # Process each camera
  all_cameras_dirs = {}

  for cawec_id in CAWEC_IDS:
    good_dates = good_dates_data.get(cawec_id, [])
    target_dates_set = get_target_dates(good_dates)

    raw_dirs = get_daytime_dirs(cawec_id)

    filtered_dirs = []
    for dirname in raw_dirs:
      match = re.match(rf"{cawec_id}_(\d{{8}})_.*", dirname)
      if match and match.group(1) in target_dates_set:
        filtered_dirs.append(dirname)

    filtered_dirs.sort(reverse=True)
    all_cameras_dirs[cawec_id] = filtered_dirs
    print(f"  {cawec_id}: Found {len(filtered_dirs)} videos matching good weather dates.")

  # Round-robin selection
  camera_idx = 0
  while len(all_download_paths) < TARGET_MP4_COUNT and any(all_cameras_dirs.values()):
    cawec_id = CAWEC_IDS[camera_idx % len(CAWEC_IDS)]

    if cawec_id not in all_cameras_dirs or not all_cameras_dirs[cawec_id]:
      if cawec_id in all_cameras_dirs:
        del all_cameras_dirs[cawec_id]
      camera_idx += 1
      continue

    dirname = all_cameras_dirs[cawec_id].pop(0)
    file_base = convert_dir_to_file_base(dirname)
    expected_mp4_filename = f"{file_base}_frames_timelapse.mp4"
    full_remote_path = f"{REMOTE_BASE_DIR}/{cawec_id}/{dirname}/{expected_mp4_filename}"

    all_download_paths.append(full_remote_path)
    camera_idx += 1

  if not all_download_paths:
    print("No videos found.")
    return

  # Generate Batch Scripts
  num_batches = (len(all_download_paths) + BATCH_SIZE - 1) // BATCH_SIZE
  print(f"\nGenerating {num_batches} batch scripts for {len(all_download_paths)} files...")

  for i in range(num_batches):
    batch_paths = all_download_paths[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    script_name = f"download_batch_{i+1}.sh"

    script_lines = [
        "#!/bin/bash",
        "set -e",
        (
            f'echo "--- Batch {i+1}/{num_batches}: Downloading'
            f' {len(batch_paths)} files ---"'
        ),
        f"mkdir -p {LOCAL_DOWNLOAD_DIR}",
    ]

    for path in batch_paths:
      script_lines.append(f'echo "Downloading {os.path.basename(path)}..."')
      script_lines.append(f'sftp -o IdentitiesOnly=yes -i {SSH_KEY} {SFTP_USER}:"{path}" ./{LOCAL_DOWNLOAD_DIR}/ || echo "Warning: Failed {os.path.basename(path)}"')

    script_lines.append(f'echo "--- Batch {i+1} complete ---"')

    with open(script_name, 'w') as f:
      f.write('\n'.join(script_lines))
    os.chmod(script_name, 0o755)
    print(f"Created {script_name}")

if __name__ == "__main__":
    main()
