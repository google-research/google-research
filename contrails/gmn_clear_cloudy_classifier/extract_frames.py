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
import re
import subprocess


def get_target_images():
  """Parses original image lists to find relevant image filenames."""
  target_images = set()
  files = ["original_clear_image_names.txt", "original_cloudy_image_names.txt"]

  print("Reading target image names...")
  for fname in files:
    if not os.path.exists(fname):
      print(f"Warning: {fname} not found.")
      continue

    with open(fname, "r") as f:
      for line in f:
        parts = line.strip().split(",")
        if parts:
          target_images.add(parts[0])

  print(f"Identified {len(target_images)} target images.")
  return target_images


def extract_frames_ffmpeg_batch(video_file, index_to_name, output_dir):
  """Extracts multiple specific frames from a video file using one ffmpeg command."""
  if not index_to_name:
    return True

  # We use -vf "select='eq(n,idx1)+eq(n,idx2)+..." -vsync 0
  select_expr = "+".join(
      [f"eq(n\,{idx})" for idx in sorted(index_to_name.keys())]
  )

  # ffmpeg -i input.mp4 -vf "select='eq(n,0)+eq(n,100)'" -vsync 0 out_%d.png
  # But we want specific names.
  # To keep it simple and reliable with specific names, we can use a complex filter or just multiple outputs.
  # However, many outputs in one command might be tricky with many frames.
  # Another way: extract all to a temp dir with frame indices, then rename.

  temp_pattern = os.path.join(output_dir, "temp_frame_%d.png")

  cmd = [
      "ffmpeg",
      "-y",
      "-i",
      video_file,
      "-vf",
      f"select='{select_expr}'",
      "-vsync",
      "0",
      temp_pattern,
  ]

  try:
    subprocess.run(cmd, capture_output=True, check=True)
    # Now rename the files. ffmpeg -vsync 0 with select filter will output frames as 1, 2, 3...
    # in the order they were selected.
    sorted_indices = sorted(index_to_name.keys())
    for i, idx in enumerate(sorted_indices):
      temp_file = os.path.join(output_dir, f"temp_frame_{i+1}.png")
      target_file = os.path.join(output_dir, index_to_name[idx])
      if os.path.exists(temp_file):
        os.rename(temp_file, target_file)
    return True
  except subprocess.CalledProcessError as e:
    print(f"ffmpeg error for {video_file}: {e.stderr.decode()}")
    return False


def main():
  RAW_DATA_DIR = "raw_data"
  OUTPUT_DIR = "extracted_night"
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  target_images = get_target_images()
  video_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.mp4"))

  for video_file in sorted(video_files):
    json_file = video_file.replace("_frames_timelapse.mp4", "_frametimes.json")
    if not os.path.exists(json_file):
      continue

    with open(json_file, "r") as f:
      frametime_data = json.load(f)

    timestamp_to_idx = {}
    for idx_str, ts_val in frametime_data.items():
      clean_ts = ts_val.rsplit("_", 1)[0] + ".png"
      timestamp_to_idx[clean_ts] = int(idx_str)

    matches = target_images.intersection(timestamp_to_idx.keys())

    # Filter out already extracted images
    index_to_name = {}
    for img_name in matches:
      if not os.path.exists(os.path.join(OUTPUT_DIR, img_name)):
        index_to_name[timestamp_to_idx[img_name]] = img_name

    if index_to_name:
      print(
          f"Processing {os.path.basename(video_file)}: extracting"
          f" {len(index_to_name)} frames..."
      )
      extract_frames_ffmpeg_batch(video_file, index_to_name, OUTPUT_DIR)

  print(
      f"\nExtraction complete. Total images in {OUTPUT_DIR}:"
      f" {len(os.listdir(OUTPUT_DIR))}"
  )


if __name__ == '__main__':
    main()
