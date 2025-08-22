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

"""Example script to parse a TFRecord file."""

from collections.abc import Sequence

from absl import app
import tensorflow as tf


def parse_tfrecord_file(filename):
  """Parses a TFRecord file and prints the contents."""

  raw_dataset = tf.data.TFRecordDataset(filename)
  for raw_record in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    feat_map = example.features.feature

    # Original filename which can be mapped to images in pick-a-pic dataset.
    filename = feat_map['filename'].bytes_list.value[0].decode()

    # 4 fine-grained scores.
    aesthetics_score = feat_map['aesthetics_score'].float_list.value[0]
    artifact_score = feat_map['artifact_score'].float_list.value[0]
    misalignment_score = feat_map['misalignment_score'].float_list.value[0]
    overall_score = feat_map['overall_score'].float_list.value[0]

    # Artifact and misalignment heatmaps.
    artifact_map = feat_map['artifact_map'].bytes_list.value[0]
    artifact_map = tf.image.decode_image(artifact_map, channels=1).numpy()

    misalignment_map = feat_map['misalignment_map'].bytes_list.value[0]
    misalignment_map = tf.image.decode_image(
        misalignment_map, channels=1
    ).numpy()

    # Mislignment label, which can be mapped to tokens in original prompt using
    # match_label_to_token.py.
    token_label = feat_map['prompt_misalignment_label'].bytes_list.value[0]
    token_label = token_label.decode()

    print('Filename:', filename)
    print('Aesthetics score:', aesthetics_score)
    print('Artifact score:', artifact_score)
    print('Misalignment score:', misalignment_score)
    print('Overall score:', overall_score)

    # Both heatmaps are 512x512 with values between [0, 255], indicating the
    # heatmap intensity.
    print('Artifact heatmap shape:', artifact_map.shape)
    print('Misalignment heatmap shape:', misalignment_map.shape)

    # 0: misaligned token, 1: aligned token.
    print('Misalignment token label:', token_label)

    break


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Replace the path to the downloaded TFRecord file.
  parse_tfrecord_file('train.tfrecord')


if __name__ == '__main__':
  app.run(main)
