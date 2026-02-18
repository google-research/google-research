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

import datetime
import json
import os
import re
import shutil
import subprocess

# Configuration
SSH_KEY = "TODO_INSERT_PATH_TO_YOUR_SSH_KEY"
SFTP_USER = "TODO_INSERT_YOUR_SFTP_USER"
REMOTE_CONTRAILS_DIR = "/home/contrails/CA"
REMOTE_EXTRACTED_DIR = "/home/extracted_data/CA"
CAWEC_IDS = ["CAWEC2", "CAWEC3", "CAWEC4", "CAWEC5", "CAWEC6"]

LOCAL_RAW_DIR = "raw_data"
LOCAL_FLUX_DIR = "flux_time_intervals"
# No limit, we want all matching dates from the list
DOWNLOAD_LIMIT_PER_CAM = 999


def get_target_dates():
  """Parses original image lists to find relevant dates.

  Returns a set of date strings (YYYYMMDD) that represent likely video start
  dates.
  """
  target_dates = set()
  files = ["original_clear_image_names.txt", "original_cloudy_image_names.txt"]

  print("Reading target dates from original image lists...")
  for fname in files:
    if not os.path.exists(fname):
      print(f"Warning: {fname} not found.")
      continue

    with open(fname, "r") as f:
      for line in f:
        # Format: 20251106_234624_983.png,clear
        # Extract YYYYMMDD and HHMMSS
        match = re.match(r"(\d{8})_(\d{6})", line)
        if match:
          date_str = match.group(1)
          time_str = match.group(2)
          hour = int(time_str[:2])

          # Determine the "Night Date" (start of the night)
          # If hour is 00-11 (morning), it belongs to the previous day's night
          # If hour is 12-23 (evening), it belongs to this date
          if hour < 12:
            try:
              d = datetime.datetime.strptime(date_str, "%Y%m%d")
              prev_day = d - datetime.timedelta(days=1)
              target_dates.add(prev_day.strftime("%Y%m%d"))
            except ValueError:
              pass
          else:
            target_dates.add(date_str)

  print(f"Identified {len(target_dates)} potential target video dates.")
  return target_dates


def sftp_ls(remote_path):
  cmd = (
      f"sftp -o IdentitiesOnly=yes -i {SSH_KEY} {SFTP_USER} <<< 'ls"
      f" {remote_path}'"
  )
  try:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
      return []
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("sftp>")
    ]
    return [os.path.basename(p) for p in lines]
  except Exception as e:
    print(f"Error listing {remote_path}: {e}")
    return []


def find_night_matches(cawec_id, target_dates):
  print(f"Scanning {cawec_id}...")

  # 1. Get Video Directories
  video_dirs = sftp_ls(f"{REMOTE_CONTRAILS_DIR}/{cawec_id}/")
  night_video_map = {}  # Date -> Dirname

  for d in video_dirs:
    # Match CAWECx_YYYYMMDD_HHMMSS_to_...
    # We want starts in evening (e.g. >= 18) or early morning (<= 06)
    match = re.match(rf"{cawec_id}_(\d{{8}})_(\d{{6}})_to_.*", d)
    if match:
      date_str = match.group(1)
      hour = int(match.group(2)[:2])

      # Filter by Target Date immediately
      if date_str not in target_dates:
        continue

      if hour >= 18 or hour <= 6:
        night_video_map[date_str] = d

  if not night_video_map:
    return []

  # 2. Get Clear Sky (aka Flux) Directories
  flux_dirs = sftp_ls(f"{REMOTE_EXTRACTED_DIR}/{cawec_id}/")
  night_flux_map = {}  # Date -> Dirname

  for d in flux_dirs:
    # Match CAWECx_YYYYMMDD_HHMMSS_..._detected
    match = re.match(rf"{cawec_id}_(\d{{8}})_.*_detected", d)
    if match:
      date_str = match.group(1)
      # Filter by Target Date
      if date_str not in target_dates:
        continue
      night_flux_map[date_str] = d

  # 3. Find Intersection
  matches = []
  sorted_dates = sorted(night_video_map.keys(), reverse=True)

  for date_str in sorted_dates:
    if date_str in night_flux_map:
      video_dir = night_video_map[date_str]
      flux_dir = night_flux_map[date_str]
      matches.append((date_str, video_dir, flux_dir))

  return matches


def generate_download_script(matches):
  script_lines = [
      "#!/bin/bash",
      "set -e",
      f"mkdir -p {LOCAL_RAW_DIR}",
      f"mkdir -p {LOCAL_FLUX_DIR}",
      "echo 'Starting Download...'",
  ]

  total_files = 0
  for cawec_id, cam_matches in matches.items():
    for date_str, video_dir, flux_dir in cam_matches:
      # Helper to convert Dir format to File format
      def dir_to_file_base(dname):
        new_name = re.sub(r"(\d{8})_(\d{6})", r"\1-\2", dname)
        return new_name

      file_base = dir_to_file_base(video_dir)
      video_file = f"{file_base}_frames_timelapse.mp4"
      json_file = f"{file_base}_frametimes.json"

      remote_video_dir = f"{REMOTE_CONTRAILS_DIR}/{cawec_id}/{video_dir}"

      local_video = f"{LOCAL_RAW_DIR}/{video_file}"
      local_json = f"{LOCAL_RAW_DIR}/{json_file}"

      # Use 'cd' then 'get' for reliability
      # Note: We must quote the paths
      sftp_cmd_video = (
          f'cd "{remote_video_dir}"\nget "{video_file}" "{local_video}"\nget'
          f' "{json_file}" "{local_json}"'
      )

      script_lines.append(f"echo 'Downloading Video: {video_file}'")
      # We use a single sftp invocation with a here-doc for the video files
      script_lines.append(
          f"sftp -o IdentitiesOnly=yes -i {SSH_KEY} {SFTP_USER} <<<"
          f" '{sftp_cmd_video}' || echo 'Failed {video_file}'"
      )

      # Clear sky time interval (aka Flux) File
      remote_flux = f"{REMOTE_EXTRACTED_DIR}/{cawec_id}/{flux_dir}/flux_time_intervals.json"
      local_flux = f"{LOCAL_FLUX_DIR}/{video_dir}_flux_time_intervals.json"

      script_lines.append(f"echo 'Downloading Clearsky Intervals: {flux_dir}'")
      script_lines.append(
          f"sftp -o IdentitiesOnly=yes -i {SSH_KEY} {SFTP_USER} <<< 'get"
          f' "{remote_flux}" "{local_flux}"\' || echo \'Failed flux for'
          f" {date_str}'"
      )

      total_files += 1

  script_name = "download_night_data.sh"
  with open(script_name, "w") as f:
    f.write("\n".join(script_lines))
  os.chmod(script_name, 0o755)
  print(f"\nGenerated {script_name} with {total_files} datasets.")


def main():
  target_dates = get_target_dates()
  if not target_dates:
    print("No target dates found. Exiting.")
    return

  all_matches = {}

  for cawec_id in CAWEC_IDS:
    matches = find_night_matches(cawec_id, target_dates)
    print(f"  Found {len(matches)} matching nights.")
    if matches:
      all_matches[cawec_id] = matches[:DOWNLOAD_LIMIT_PER_CAM]

  if all_matches:
    generate_download_script(all_matches)
  else:
    print("No matches found.")


if __name__ == "__main__":
  main()
