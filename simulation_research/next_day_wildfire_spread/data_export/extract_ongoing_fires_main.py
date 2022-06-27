# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Job for extracting ongoing fires from the dataset exported using export_ee_data.py.

Ongoing fires are fires for which there is at least one positive fire label in
the PrevFireMask and in the FireMask. Samples of ongoing fires are written to
new TFRecords.
"""

from typing import List, Text, Sequence

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from simulation_research.next_day_wildfire_spread import constants
from simulation_research.next_day_wildfire_spread import dataset

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'file_pattern', None,
    'Unix glob pattern for the files containing the input data.')

flags.DEFINE_string('out_file_prefix', None,
                    'File prefix to save the TFRecords.')

flags.DEFINE_integer(
    'data_size', 64,
    'Size of the tiles in pixels (square) as read from input files.')

flags.DEFINE_integer('num_samples_per_file', 1000,
                     'Number of samples to write per TFRecord.')

flags.DEFINE_string(
    'compression_type', 'GZIP',
    'Compression used for the input. Must be one of "GZIP", "ZLIB", or "" (no '
    'compression).')


def _parse_fn(
    example_proto,
    data_size,
    feature_names,
):
  """Reads a serialized example.

  Args:
    example_proto: A TensorFlow example protobuf.
    data_size: Size of tiles in pixels (square) as read from input files.
    feature_names: Names of all the features.

  Returns:
    (input_img, output_img) tuple of inputs and outputs to the ML model.
  """
  features_dict = dataset.get_features_dict(data_size, feature_names)
  features = tf.io.parse_single_example(example_proto, features_dict)
  feature_list = []
  for key in feature_names:
    if 'FireMask' in key:
      feature_list.append(dataset.map_fire_labels(features.get(key)))
    else:
      feature_list.append(features.get(key))
  return feature_list


def get_dataset(
    file_pattern,
    data_size,
    compression_type,
    feature_names,
):
  """Gets the dataset from the file pattern.

  Args:
    file_pattern: Input file pattern.
    data_size: Size of tiles in pixels (square) as read from input files.
    compression_type: Type of compression used for the input files. Must be one
      of "GZIP", "ZLIB", or "" (no compression).
    feature_names: Names of the all the features.

  Returns:
    A TensorFlow dataset loaded from the input file pattern, with features
    described in the constants, and with the shapes determined from the input
    parameters to this function.
  """
  tf_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  tf_dataset = tf_dataset.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  tf_dataset = tf_dataset.map(
      lambda x: _parse_fn(x, data_size, feature_names),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return tf_dataset


def write_to_tfrecord(tf_writer,
                      feature_names,
                      feature_list):
  """Writes the features to TFRecord files.

  Args:
    tf_writer: TFRecord writer.
    feature_names: Names of all the features.
    feature_list: Values of all the features.
  """
  feature_dict = {}
  for i, feature_name in enumerate(feature_names):
    feature_dict[feature_name] = tf.train.Feature(
        float_list=tf.train.FloatList(
            value=feature_list[i].numpy().reshape(-1)))
  tf_example = tf.train.Example(
      features=tf.train.Features(feature=feature_dict))
  tf_writer.write(tf_example.SerializeToString())


def write_ongoing_dataset(tf_dataset,
                          feature_names, file_prefix,
                          num_samples_per_file, compression_type):
  """Writes dataset of ongoing fires extracted from input tf_dataset.

  Args:
    tf_dataset: Input dataset.
    feature_names: Names of all the features.
    file_prefix: File prefix to use for writing the ouput TFRecords.
    num_samples_per_file: Number of samples to write per TFRecord.
    compression_type: Compression type to use for the output TFRecords. Must be
      one of "GZIP", "ZLIB", or "" (no compression).
  """
  ongoing_count = 0
  ongoing_tfrecord_count = 0
  prev_fire_index = feature_names.index('PrevFireMask')
  compression_type = tf.io.TFRecordOptions(compression_type=compression_type)
  for feature_list in tf_dataset:
    if ongoing_count % num_samples_per_file == 0:
      out_file = (f'{file_prefix}_ongoing_{ongoing_tfrecord_count:03d}'
                  '.tfrecord.gz')
      ongoing_tfrecord_count += 1
      ongoing_writer = tf.io.TFRecordWriter(out_file, options=compression_type)
    # Only keep samples with at least one positive fire label in the previous
    # day.
    if np.amax(feature_list[prev_fire_index].numpy()) == 1:
      write_to_tfrecord(ongoing_writer, feature_names, feature_list)
      ongoing_count += 1


def main(_):
  feature_names = constants.INPUT_FEATURES + constants.OUTPUT_FEATURES

  tf_dataset = get_dataset(
      FLAGS.file_pattern,
      data_size=FLAGS.data_size,
      feature_names=feature_names,
      compression_type=FLAGS.compression_type)
  write_ongoing_dataset(tf_dataset, feature_names, FLAGS.out_file_prefix,
                        FLAGS.num_samples_per_file, FLAGS.compression_type)


if __name__ == '__main__':
  flags.mark_flag_as_required('file_pattern')
  flags.mark_flag_as_required('out_file_prefix')
  app.run(main)
