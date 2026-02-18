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
import json
import shutil


# Define directories
SOURCE_DIR = "./flux_time_intervals"
DEST_DIR = "./flux_time_intervals/all_cloudy_time_intervals"

def main():
  # Create destination directory if it doesn't exist
  if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)
    print(f"Created directory: {DEST_DIR}")

  # Iterate through files in the source directory
  if not os.path.exists(SOURCE_DIR):
    print(f"Source directory {SOURCE_DIR} does not exist.")
    return

  files_moved = 0

  for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".json"):
      file_path = os.path.join(SOURCE_DIR, filename)

      try:
        with open(file_path, 'r') as f:
          data = json.load(f)

        # Check if 'time_intervals' exists and is an empty list
        if 'time_intervals' in data and isinstance(data['time_intervals'], list) and len(data['time_intervals']) == 0:
          dest_path = os.path.join(DEST_DIR, filename)
          shutil.move(file_path, dest_path)
          print(f"Moved: {filename}")
          files_moved += 1

      except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {filename}")
      except Exception as e:
        print(f"Error processing file {filename}: {e}")

  print(f"Processing complete. Moved {files_moved} files to {DEST_DIR}.")

if __name__ == "__main__":
    main()
