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

import glob
import json
import os
import pandas as pd


def load_clear_intervals_from_flux(flux_files):
  """Loads all clear sky time intervals from a list of flux_time_intervals.json files.

  Args:
      flux_files (list): A list of paths to the flux JSON files; each file
        contains a list of clear sky time intervals for a given date.

  Returns:
      list: A list of tuples, where each tuple is a (start_time, end_time) pair
            of timezone-aware pandas Timestamps.
  """
  all_intervals = []
  for file_path in flux_files:
    try:
      with open(file_path, "r") as f:
        data = json.load(f)

      for interval in data.get("time_intervals", []):
        start_ts = pd.Timestamp(interval[0], tz="UTC")
        end_ts = pd.Timestamp(interval[1], tz="UTC")
        all_intervals.append((start_ts, end_ts))
    except (json.JSONDecodeError, IndexError, TypeError) as e:
      print(
          f"Warning: Could not parse or understand file {file_path}. Error: {e}"
      )
  return all_intervals


def get_image_timestamp_from_filename(image_filename):
  """Parses the UTC timestamp directly from the image filename."""
  try:
    ts_str = image_filename.replace(".png", "")
    return pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S_%f").tz_localize("UTC")
  except ValueError:
    return None


def main():
  # --- Configuration ---
  FLUX_DIR = "flux_time_intervals"
  NIGHT_IMAGES_DIR = "extracted_night"

  # --- Main Logic ---
  # Find all flux time interval files in the flux directory
  # We use glob to find files directly in the directory, effectively ignoring subdirectories
  flux_files = glob.glob(os.path.join(FLUX_DIR, "*_flux_time_intervals.json"))
  if not flux_files:
    print(
        f"Error: No '*_flux_time_intervals.json' files found in '{FLUX_DIR}'."
        " Exiting."
    )
    return

  print(
      f"Found {len(flux_files)} flux time interval files. Loading clear sky"
      " periods..."
  )
  clear_intervals = load_clear_intervals_from_flux(flux_files)

  if not clear_intervals:
    print("No valid clear time intervals could be loaded. Exiting.")
    return

  print(f"Successfully loaded {len(clear_intervals)} clear sky intervals.")

  labeled_data = []

  image_files = sorted(
      [f for f in os.listdir(NIGHT_IMAGES_DIR) if f.endswith(".png")]
  )
  if not image_files:
    print(f"No images found in '{NIGHT_IMAGES_DIR}'.")
    return

  print(f"\n--- Labeling {len(image_files)} Night Images ---")
  print(f"{ 'Image Filename':<30} | { 'Label':<7}")
  print("-" * 40)

  for image_file in image_files:
    img_timestamp = get_image_timestamp_from_filename(image_file)
    if not img_timestamp:
      print(f"{image_file:<30} | {'ERROR: Bad Filename'}")
      continue

    label = "cloudy"
    # Check if the image timestamp falls within any of the clear intervals
    for start_time, end_time in clear_intervals:
      if start_time <= img_timestamp <= end_time:
        label = "clear"
        break

    labeled_data.append({"image_filename": image_file, "label": label})

  # Create a DataFrame and save it to a CSV file
  df = pd.DataFrame(labeled_data)
  output_path = "extracted_night_labels.csv"
  df.to_csv(output_path, index=False)

  print(f"\n--- Labeling complete. ---")
  print(f"Labeled {len(df)} images. Results saved to '{output_path}'.")
  print("\nLabel Distribution:")
  print(df["label"].value_counts())


if __name__ == "__main__":
  main()
