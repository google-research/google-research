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

import json
import os
import re
import shutil
import subprocess

# Configuration
SSH_KEY = "TODO_INSERT_PATH_TO_YOUR_SSH_KEY"
SFTP_USER = "TODO_INSERT_YOUR_SFTP_USER"
REMOTE_BASE_DIR = "/home/extracted_data/CA"
CAWEC_IDS = ["CAWEC2", "CAWEC3", "CAWEC4", "CAWEC5", "CAWEC6"]
TEMP_FLUX_DIR = "temp_flux_files"

def get_detected_dirs(cawec_id, limit=60):
  """Gets a list of recent _detected directories."""
  remote_path = f"{REMOTE_BASE_DIR}/{cawec_id}/"
  cmd = (
      f"sftp -o IdentitiesOnly=yes -i {SSH_KEY} {SFTP_USER} <<< 'ls"
      f" {remote_path}'"
  )
  print(f"Fetching directory list for {cawec_id}...")
  try:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
      return []

    all_dirs = [
        os.path.basename(line.strip())
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("sftp>")
    ]

    detected_dirs = []
    for dirname in all_dirs:
      if "_detected" in dirname:
        detected_dirs.append(dirname)

    detected_dirs.sort(reverse=True)
    return detected_dirs[:limit]
  except Exception as e:
    print(f"Error listing dirs: {e}")
    return []

def download_flux_files(cawec_id, detected_dirs):
  """Downloads flux_time_intervals.json for the given directories."""
  download_list = []

  local_cawec_dir = os.path.join(TEMP_FLUX_DIR, cawec_id)
  os.makedirs(local_cawec_dir, exist_ok=True)

  batch_file = f"sftp_batch_{cawec_id}.txt"
  with open(batch_file, "w") as f:
    for dirname in detected_dirs:
      remote_file = (
          f"{REMOTE_BASE_DIR}/{cawec_id}/{dirname}/flux_time_intervals.json"
      )
      local_file = os.path.join(local_cawec_dir, f"{dirname}_flux.json")
      f.write(f"get {remote_file} {local_file}\n")
      download_list.append((dirname, local_file))

  cmd = f"sftp -o IdentitiesOnly=yes -i {SSH_KEY} -b {batch_file} {SFTP_USER}"
  try:
    subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
  except Exception as e:
    print(f"Error during sftp batch: {e}")

  if os.path.exists(batch_file):
    os.remove(batch_file)

  return download_list

def is_flux_good(json_path):
    """Checks if the JSON file contains meaningful time intervals."""
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # We look for "time_intervals" list being non-empty
            if isinstance(data, dict):
                intervals = data.get("time_intervals", [])
                if isinstance(intervals, list) and len(intervals) > 0:
                    return True
    except:
        pass
    return False

def main():
  if os.path.exists(TEMP_FLUX_DIR):
    shutil.rmtree(TEMP_FLUX_DIR)
  os.makedirs(TEMP_FLUX_DIR)

  good_dates = {}

  for cawec_id in CAWEC_IDS:
    dirs = get_detected_dirs(cawec_id, limit=60)
    if not dirs:
      continue

    print(
        f"Downloading clearsky interval (aka flux) files for {cawec_id}"
        f" ({len(dirs)} dirs)..."
    )
    downloaded = download_flux_files(cawec_id, dirs)

    good_dates[cawec_id] = []
    for dirname, local_path in downloaded:
      if is_flux_good(local_path):
        # Extract date
        match = re.match(rf"{cawec_id}_(\d{{8}})_.*", dirname)
        if match:
          date_str = match.group(1)
          if date_str not in good_dates[cawec_id]:
            good_dates[cawec_id].append(date_str)

    print(f"  -> Found {len(good_dates[cawec_id])} good dates for {cawec_id}.")

  with open("good_weather_dates.json", "w") as f:
    json.dump(good_dates, f, indent=2)

  print("\nSaved good dates to good_weather_dates.json")

if __name__ == "__main__":
    main()
