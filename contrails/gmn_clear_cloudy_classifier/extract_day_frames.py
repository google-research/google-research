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
import glob
import random
import cv2


def extract_frames_simple(video_file, output_dir, num_frames_to_extract):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

  cap = cv2.VideoCapture(video_file)
  if not cap.isOpened():
    print(f"Error: Could not open {video_file}")
    return

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames <= 0:
    print(f"Warning: Could not determine frame count for {video_file}")
    cap.release()
    return

  # Randomly select frame indices
  if total_frames < num_frames_to_extract:
    indices = list(range(total_frames))
  else:
    indices = sorted(random.sample(range(total_frames), num_frames_to_extract))

  video_basename = os.path.basename(video_file).replace('.mp4', '')
  # Simplify name: CAWECx_Date
  parts = video_basename.split('_')
  # Some files might be CAWEC2_20260101-120545_to...
  if len(parts) >= 2:
    short_name = f"{parts[0]}_{parts[1]}"
  else:
    short_name = video_basename

  print(f"Extracting {len(indices)} frames from {short_name} (Total: {total_frames})...")

  for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
      out_name = f"daytime_{short_name}_frame_{idx:06d}.png"
      cv2.imwrite(os.path.join(output_dir, out_name), frame)
    else:
      print(f"Failed to read frame {idx}")

  cap.release()

def main():
  random.seed(42)
  INPUT_DIR = "raw_data_daytime"
  OUTPUT_DIR = "extracted_day_test"
  TOTAL_TARGET = 1000 # Updated target

  video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
  if not video_files:
    print("No videos found.")
    return

  print(f"Found {len(video_files)} videos.")

  # We want approx 1000 frames total.
  # If we have 52 videos, we need approx 20 frames per video.
  per_video = max(1, TOTAL_TARGET // len(video_files))
  remainder = TOTAL_TARGET % len(video_files)

  print(f"Extracting approximately {per_video} frames per video to reach target of {TOTAL_TARGET}...")

  for i, vid in enumerate(video_files):
    # Distribute the remainder to the first few videos
    count = per_video + (1 if i < remainder else 0)
    extract_frames_simple(vid, OUTPUT_DIR, count)

  print("Done.")

if __name__ == "__main__":
    main()
