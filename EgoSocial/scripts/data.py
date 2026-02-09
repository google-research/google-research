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

"""Processes Ego4D clips to extract frames and audio for EgoSocial dataset.

This script reads an annotation file, and for each entry, it extracts
10 frames at 1 FPS and a 10-second audio clip from the corresponding
Ego4D video file. The extracted data is stored in 'frames' and 'audio'
directories within the specified output directory.

Prerequisites:
  - ffmpeg must be installed and accessible in the system PATH.
"""

import argparse
import json
import os
import shutil
import subprocess


def create_dirs(output_dir):
  """Creates the necessary output directories."""
  os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
  os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)


def extract_frames(video_path, start_sec, end_sec, output_frame_dir):
  """Extracts 10 frames at 1 FPS from the video clip."""
  os.makedirs(output_frame_dir, exist_ok=True)
  duration = end_sec - start_sec
  print(
      f"Extracting frames for {video_path}: start={start_sec}, end={end_sec},"
      f" duration={duration}"
  )
  if duration <= 0:
    print(
        f"Warning: Duration is non-positive for {video_path} ({start_sec} to"
        f" {end_sec})"
    )
    return

  # Extract frames at 1 FPS
  temp_frame_pattern = os.path.join(output_frame_dir, "temp_%04d.jpg")
  ffmpeg_cmd = [
      "ffmpeg",
      "-i",
      video_path,
      "-ss",
      str(start_sec),
      "-to",
      str(end_sec),
      "-vf",
      "fps=1",
      "-q:v",
      "2",
      temp_frame_pattern,
  ]
  try:
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
  except subprocess.CalledProcessError as e:
    print(f"Error extracting frames for {video_path}: {e}")
    print(f"FFMPEG stderr: {e.stderr.decode()}")
    return

  # Select up to 10 frames
  extracted_frames = sorted(
      [f for f in os.listdir(output_frame_dir) if f.startswith("temp_")]
  )
  num_frames = len(extracted_frames)

  if num_frames == 0:
    print(f"Warning: No frames extracted for {video_path}")
    return

  step = max(1, num_frames // 10)
  selected_indices = range(0, num_frames, step)[:10]

  for i, frame_index in enumerate(selected_indices):
    old_path = os.path.join(output_frame_dir, extracted_frames[frame_index])
    new_path = os.path.join(output_frame_dir, f"{i}.jpg")
    shutil.move(old_path, new_path)

  # Clean up temporary frames
  for f in os.listdir(output_frame_dir):
    if f.startswith("temp_"):
      os.remove(os.path.join(output_frame_dir, f))


def extract_audio(video_path, start_sec, end_sec, output_audio_path):
  """Extracts a 10-second audio clip from the video clip."""
  duration = end_sec - start_sec
  print(
      f"Extracting audio for {video_path}: start={start_sec}, end={end_sec},"
      f" duration={duration}"
  )
  if duration <= 0:
    print(
        f"Warning: Duration is non-positive for {video_path} ({start_sec} to"
        f" {end_sec})"
    )
    return

  actual_duration = min(duration, 10.0)
  end_sec = start_sec + actual_duration

  ffmpeg_cmd = [
      "ffmpeg",
      "-i",
      video_path,
      "-ss",
      str(start_sec),
      "-to",
      str(end_sec),
      "-vn",  # Disable video recording
      "-c:a",
      "pcm_s16le",  # Use WAV format
      "-ar",
      "16000",  # Sample rate
      "-ac",
      "1",  # Mono channel
      output_audio_path,
  ]
  try:
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
  except subprocess.CalledProcessError as e:
    print(f"Error extracting audio for {video_path}: {e}")
    print(f"FFMPEG stderr: {e.stderr.decode()}")


def main():
  """Main function to parse arguments and process clips."""
  parser = argparse.ArgumentParser(
      description="Extract frames and audio from Ego4D clips."
  )
  parser.add_argument(
      "--annotation_file",
      type=str,
      required=True,
      help="Path to the annotation JSON file.",
  )
  parser.add_argument(
      "--ego4d_clips_dir",
      type=str,
      required=True,
      help="Directory containing Ego4D video clips (.mp4 files).",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Directory to store the extracted frames and audio.",
  )
  args = parser.parse_args()

  create_dirs(args.output_dir)

  with open(args.annotation_file, "r") as f:
    annotations = json.load(f)

  for clip_key, value in annotations.items():
    context = value.get("context")
    # print(context)
    if not context:
      print(f"Warning: Skipping clip {clip_key} due to missing 'context'.")
      continue

    clip_uid = context.get("segment_parent_key")
    start_sec = context.get("image/start_time_seconds")
    end_sec = context.get("image/end_time_seconds")

    if not clip_uid or start_sec is None or end_sec is None:
      print(
          f"Warning: Skipping clip {clip_key} due to missing information in"
          " 'context'."
      )
      continue

    video_path = os.path.join(args.ego4d_clips_dir, f"{clip_uid}.mp4")
    if clip_uid == "0511dce9-c126-4a11-92ba-8082a4ec89bd":
      print(f"Found clip {clip_key} with video path {video_path}.")
    if not os.path.exists(video_path):
      # print(
      #     f"Warning: Video file not found {video_path}, skipping clip"
      #     f" {clip_key}."
      # )
      continue

    print(f"Processing clip: {clip_key}")

    # Prepare output paths
    output_frame_dir = os.path.join(args.output_dir, "frames", clip_key)
    output_audio_path = os.path.join(
        args.output_dir, "audio", f"{clip_key}.wav"
    )

    # Extract frames
    extract_frames(video_path, start_sec, end_sec, output_frame_dir)

    # Extract audio
    extract_audio(video_path, start_sec, end_sec, output_audio_path)

  print("Data extraction complete.")


if __name__ == "__main__":
  main()
