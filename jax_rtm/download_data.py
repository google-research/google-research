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

"""Script to download the JAX-RTM data assets from Zenodo."""

import os
import urllib.error
import urllib.request

# ==============================================================================
# Zenodo Data Hosting Configuration
# ==============================================================================
ZENODO_RECORD_ID = "21228209"
ICE_DATA_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/ping_yang_multi_habit.npz?download=1"
WEATHER_85_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/weather_85x85.npz?download=1"
WEATHER_339_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/weather_339x339.npz?download=1"

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ASSETS = {
    "ping_yang_multi_habit.npz": ICE_DATA_URL,
    "weather_85x85.npz": WEATHER_85_URL,
    "weather_339x339.npz": WEATHER_339_URL,
}


def download_file(url, target_path):
  """Downloads a file from a URL to a target path, handling errors."""
  print(f"Downloading {url}...")
  try:
    urllib.request.urlretrieve(url, target_path)
    print(f"Successfully downloaded to {target_path}")
  except (urllib.error.URLError, OSError) as e:
    print(f"Error downloading {target_path}: {e}")
    print(
        "Please ensure your Zenodo record is published and public, and that"
        f" the file exists in record ID: {ZENODO_RECORD_ID}."
    )


def main():
  if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print(f"Created target directory: {TARGET_DIR}")

  print(
      "================================================================================"
  )
  print(
      "Downloading JAX-RTM public data assets from Zenodo (Record:"
      f" {ZENODO_RECORD_ID})"
  )
  print(
      "================================================================================"
  )

  for filename, url in ASSETS.items():
    target_path = os.path.join(TARGET_DIR, filename)
    if os.path.exists(target_path):
      print(f"Asset {filename} already exists at {target_path}. Skipping.")
    else:
      download_file(url, target_path)

  print("Done!")


if __name__ == "__main__":
  main()
