# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

r"""Converts raw data from RICO and annotations into tfrecord format.

Sample command:
1. cd to `google_research`.
2. Run: python -m rico_semantics.convert_raw_data_to_tfrecords \
--data_path=raw_data/ --task_name=icon_semantics \
--annotations_dir=annotations/ --output_dir=/tmp/tfrecords/
"""

import argparse
import collections
import json
import os
from typing import Any, Sequence

import tensorflow as tf

NUM_TFRECORD_FILES = 10


def _add_bytes_feature(
    tf_example,
    feature_name,
    feature_values,
):
  feature = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=feature_values)
  )
  tf_example.features.feature[feature_name].CopyFrom(feature)
  return tf_example


def _add_float_feature(
    tf_example,
    feature_name,
    feature_values,
):
  feature = tf.train.Feature(
      float_list=tf.train.FloatList(value=feature_values)
  )
  tf_example.features.feature[feature_name].CopyFrom(feature)
  return tf_example


def _read_annotations_data(file_path):
  with open(file_path, "r") as f:
    annotations_data = json.load(f)
  return annotations_data


def create_tf_example(
    annotation_data, raw_data_dir
):
  """Creates a Tf Example from annotation and raw image data.

  Args:
    annotation_data: Annotation data for the current screen.
    raw_data_dir: Directory containing images.

  Returns:
    Tf Example with data containing the image data and labels.
  """
  tf_example = tf.train.Example()

  current_key = annotation_data.get("screen_id", "-1")
  image_file_path = os.path.join(f"{raw_data_dir}/{current_key}.png")
  with open(image_file_path, "rb") as f:
    image_bytes = f.read()
  tf_example = _add_bytes_feature(tf_example, "image/encoded", [image_bytes])
  labels_data = collections.defaultdict(list)
  for curr_label in annotation_data.get("screen_elements", []):
    for key, value in curr_label.items():
      labels_data[key].append(value)

  # Add bounding box features.
  for bbox_field in ["xmin", "xmax", "ymin", "ymax"]:
    bbox_feature = f"image/object/bbox/{bbox_field}"
    tf_example = _add_float_feature(
        tf_example,
        feature_name=bbox_feature,
        feature_values=labels_data[bbox_field],
    )
  # Add label feature.
  label_values = labels_data.get("label", [])
  label_bytes = [l.encode("utf-8") for l in label_values]
  tf_example = _add_bytes_feature(
      tf_example,
      feature_name="image/object/class/text",
      feature_values=label_bytes,
  )
  # Add key for debugging.
  key_bytes = [current_key.encode("utf-8")]
  tf_example = _add_bytes_feature(
      tf_example, feature_name="debug/key", feature_values=key_bytes
  )
  return tf_example


def main(parsed_args):
  raw_data_dir = parsed_args.raw_data_dir
  task_name = parsed_args.task_name
  output_dir = parsed_args.output_dir
  annotations_dir = parsed_args.annotations_dir

  data_splits = ["train", "val", "test"]
  tf_examples = collections.defaultdict(list)

  for curr_split in data_splits:
    full_file_name = os.path.join(
        annotations_dir, f"{task_name}_{curr_split}.json"
    )
    if not os.path.exists(full_file_name):
      raise ValueError(
          f"Annotations file for split: {curr_split} "
          f"not found at: {full_file_name}."
      )
    annotation_data = _read_annotations_data(full_file_name)
    for curr_label in annotation_data:
      tf_examples[curr_split].append(
          create_tf_example(curr_label, raw_data_dir)
      )

  # Write all examples as tf record.
  for curr_split, split_tf_examples in tf_examples.items():
    num_examples_per_file = len(split_tf_examples) // NUM_TFRECORD_FILES
    if len(split_tf_examples) % NUM_TFRECORD_FILES:
      num_examples_per_file += 1

    for tfrec_num in range(NUM_TFRECORD_FILES):
      samples = split_tf_examples[
          (tfrec_num * num_examples_per_file) : (
              (tfrec_num + 1) * num_examples_per_file
          )
      ]
      curr_output_file = os.path.join(
          output_dir, f"{curr_split}_{tfrec_num:03}.tfrecord"
      )
      with tf.io.TFRecordWriter(curr_output_file) as writer:
        for tf_example in samples:
          writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--raw_data_dir",
      type=str,
      default=None,
      help=(
          "Path to directory with images and view hierarchies downloaded from"
          " the RICO website."
      ),
      required=True,
  )
  parser.add_argument(
      "--task_name",
      type=str,
      default=None,
      help="Name of the task to process data and create Tf Records.",
      required=True,
  )
  parser.add_argument(
      "--annotations_dir",
      type=str,
      default=None,
      help=(
          "Directory containing annotations data downloaded from Rico Semantics"
          " website."
      ),
      required=True,
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default=None,
      help="Directory to write output Tf Record files.",
      required=True,
  )
  args = parser.parse_args()
  main(args)
