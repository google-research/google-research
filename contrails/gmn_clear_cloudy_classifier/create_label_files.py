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

import csv
import os


INPUT_CSV = 'inference_results_precise_clearsky.csv'
IMAGE_DIR = 'extracted_day_test'

def main():
  if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    return

  if not os.path.exists(IMAGE_DIR):
    print(f"Error: {IMAGE_DIR} not found.")
    return

  print(f"Reading {INPUT_CSV} and creating label files in {IMAGE_DIR}...")

  count = 0
  with open(INPUT_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      filename = row['filename']
      prediction = row['prediction']

      # Construct the base name (remove .png extension)
      base_name = os.path.splitext(filename)[0]

      # Construct the new label filename
      label_filename = f"{base_name}.{prediction}.txt"
      label_filepath = os.path.join(IMAGE_DIR, label_filename)

      # Touch the file (create empty file)
      try:
        with open(label_filepath, 'w') as lf:
          pass
        count += 1
      except Exception as e:
        print(f"Failed to create {label_filepath}: {e}")

  print(f"Successfully created {count} label files.")

if __name__ == "__main__":
    main()
