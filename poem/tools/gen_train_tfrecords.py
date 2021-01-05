# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Generates training TFRecords from 2D keypoint CSV and 3D keypoint H5 files.

The CSV file is expected to have:

1. The first row as header as follows:

image/object/subject,image/subaction,image/object/camera,image/frame_index,image/width,image/height,image/object/part/NOSE_TIP/center/x,image/object/part/NOSE_TIP/center/y,image/object/part/NOSE_TIP/score,image/object/part/LEFT_SHOULDER/center/x,image/object/part/LEFT_SHOULDER/center/y,image/object/part/LEFT_SHOULDER/score,image/object/part/RIGHT_SHOULDER/center/x,image/object/part/RIGHT_SHOULDER/center/y,image/object/part/RIGHT_SHOULDER/score,image/object/part/LEFT_ELBOW/center/x,image/object/part/LEFT_ELBOW/center/y,image/object/part/LEFT_ELBOW/score,image/object/part/RIGHT_ELBOW/center/x,image/object/part/RIGHT_ELBOW/center/y,image/object/part/RIGHT_ELBOW/score,image/object/part/LEFT_WRIST/center/x,image/object/part/LEFT_WRIST/center/y,image/object/part/LEFT_WRIST/score,image/object/part/RIGHT_WRIST/center/x,image/object/part/RIGHT_WRIST/center/y,image/object/part/RIGHT_WRIST/score,image/object/part/LEFT_HIP/center/x,image/object/part/LEFT_HIP/center/y,image/object/part/LEFT_HIP/score,image/object/part/RIGHT_HIP/center/x,image/object/part/RIGHT_HIP/center/y,image/object/part/RIGHT_HIP/score,image/object/part/LEFT_KNEE/center/x,image/object/part/LEFT_KNEE/center/y,image/object/part/LEFT_KNEE/score,image/object/part/RIGHT_KNEE/center/x,image/object/part/RIGHT_KNEE/center/y,image/object/part/RIGHT_KNEE/score,image/object/part/LEFT_ANKLE/center/x,image/object/part/LEFT_ANKLE/center/y,image/object/part/LEFT_ANKLE/score,image/object/part/RIGHT_ANKLE/center/x,image/object/part/RIGHT_ANKLE/center/y,image/object/part/RIGHT_ANKLE/score,image/object/subject,image/subaction,image/object/camera,image/frame_index,image/width,image/height,image/object/part/NOSE_TIP/center/x,image/object/part/NOSE_TIP/center/y,image/object/part/NOSE_TIP/score,image/object/part/LEFT_SHOULDER/center/x,image/object/part/LEFT_SHOULDER/center/y,image/object/part/LEFT_SHOULDER/score,image/object/part/RIGHT_SHOULDER/center/x,image/object/part/RIGHT_SHOULDER/center/y,image/object/part/RIGHT_SHOULDER/score,image/object/part/LEFT_ELBOW/center/x,image/object/part/LEFT_ELBOW/center/y,image/object/part/LEFT_ELBOW/score,image/object/part/RIGHT_ELBOW/center/x,image/object/part/RIGHT_ELBOW/center/y,image/object/part/RIGHT_ELBOW/score,image/object/part/LEFT_WRIST/center/x,image/object/part/LEFT_WRIST/center/y,image/object/part/LEFT_WRIST/score,image/object/part/RIGHT_WRIST/center/x,image/object/part/RIGHT_WRIST/center/y,image/object/part/RIGHT_WRIST/score,image/object/part/LEFT_HIP/center/x,image/object/part/LEFT_HIP/center/y,image/object/part/LEFT_HIP/score,image/object/part/RIGHT_HIP/center/x,image/object/part/RIGHT_HIP/center/y,image/object/part/RIGHT_HIP/score,image/object/part/LEFT_KNEE/center/x,image/object/part/LEFT_KNEE/center/y,image/object/part/LEFT_KNEE/score,image/object/part/RIGHT_KNEE/center/x,image/object/part/RIGHT_KNEE/center/y,image/object/part/RIGHT_KNEE/score,image/object/part/LEFT_ANKLE/center/x,image/object/part/LEFT_ANKLE/center/y,image/object/part/LEFT_ANKLE/score,image/object/part/RIGHT_ANKLE/center/x,image/object/part/RIGHT_ANKLE/center/y,image/object/part/RIGHT_ANKLE/score

2. The following rows are CSV values according to the header, a pair of samples
   per row for the same 3D pose from two different camera views.

Below is a dummy sample CSV row:

S1,Eating,60457274,001346,1000,1002,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,0.1234,0.4321,1.0,S1,Eating,58860488,001346,1000,1000,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0,0.2468,0.2155,1.0

Notes:
a. The "image/frame_index" values are expected to start at 0.
b. The 2D keypoint coordinate values are required to be normalized by image
   sizes to within [0, 1].
c. The 3D keypoint coordinate values are assumed to be unnormalized.
d. In practice, we include all possible pairs in our training tables. For
   example, for one pose in the Human3.6M dataset (which has 4 views C{0..3}),
   we include all 12 pairs in the CSV, i.e., (C0, C1), (C0, C2), (C0, C3), (C1,
   C0), (C1, C2), (C1, C3), (C2, C0), (C2, C1), (C2, C3), (C3, C0), (C3, C1),
   and (C3, C2).

The H5 files contain 3D poses from Human3.6M and CSV files contain 2D poses.
Produces TFRecords file for input into training pipelines. The instructions
for downloading the H5 files are located in the Github page:
https://github.com/una-dinosauria/3d-pose-baseline.

"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os

import h5py
import numpy as np
import tensorflow.compat.v1 as tf

from poem.core import keypoint_profiles

flags = tf.app.flags
FLAGS = flags.FLAGS

# Indices of Human3.6M keypoints to be read from the H5 file. The
# keypoint ordering follows the H5 file.
KEYPOINT_3D_INDICES_H5 = [
    0, 15, 14, 13, 17, 25, 18, 26, 19, 27, 12, 6, 1, 7, 2, 8, 3
]

flags.DEFINE_enum('input_data_type', 'H36M_H5', ['H36M_H5'],
                  'Input type of Human3.6M dataset.')

flags.DEFINE_string(
    'input_root_dir', '', 'Input root directory that contains the Human3.6M 3D '
    'keypoint files.')

flags.DEFINE_string('input_csv_file', '',
                    'Input CSV file containing 2D keypoints to parse.')

flags.DEFINE_bool(
    'input_paired_entries', True,
    'Whether each row in the CSV file stores entry pairs. Use False to disable'
    'reading rows as pairs.')

flags.DEFINE_string(
    'subjects_to_read', '', 'A string for either "train", "val", "all", or a'
    'comma separated list of subjects in Human3.6M for which we want to read'
    'the 3D keypoints. For example: S1,S5,S7.')

flags.DEFINE_string('output_tfrecords', '',
                    'Output TFRecords from parsing the 2d and 3d keypoints.')

flags.DEFINE_integer(
    'output_num_shards', 100,
    'Output number of shards for the TFRecords. Use < 1 to disable sharding.')


def load_h36m_3d_keypoints_h5_files(input_root_dir, subjects):
  """Loads a dictionary of 3D keypoints given input directory to H5 files.

  Args:
    input_root_dir: A string for the root directory path containing H5 files
      with 3D poses keypoints from Human3.6M dataset.
    subjects: A list of strings for subjects to read from the directory.

  Returns:
    keypoint_dict: A dictionary for loaded 3D keypoints. Keys are (subject,
      action) and values are of shape [sequence_length, num_keypoints, 3].
  """

  keypoint_dict = {}
  for subject in subjects:
    h5_path = os.path.join(input_root_dir, subject, 'MyPoses/3D_positions/*.h5')
    for fname in tf.io.gfile.glob(h5_path):
      # The H5 file is located in path with subject and action:
      # h36m/subject/MyPoses/3D_positions/action.h5.
      action = fname.split('/')[-1][:-3]
      with h5py.File(fname, 'r') as hf:
        keypoint_dict[(subject, action)] = (
            hf['3D_positions'].value.reshape(32, 3, -1).transpose(2, 0, 1))
  return keypoint_dict


def extend_example_with_2d_3d_keypoints(example, input_list, headers):
  """Extends an existing tf.example with 2D and 3D keypoint information.

  Args:
    example: A tf.example to add keypoint information.
    input_list: A list of values with 2D keypoints, 3D keypoints and metadata
      (subject name, action, name, camera, timestamp and image size).
    headers: A list of string headers with 2D keypoint names, 3D keypoint names,
      and metadata to write to the tf.example.

  Returns:
    example: A tf.example with appended information from input_list.
  """
  # Write subject name.
  example.features.feature[headers[0]].bytes_list.value.extend(
      [input_list[0].encode()])
  # Write action name.
  example.features.feature[headers[1]].bytes_list.value.extend(
      [input_list[1].encode()])
  # Write camera name.
  example.features.feature[headers[2]].int64_list.value.extend(
      [int(input_list[2])])
  # Write timestamp.
  example.features.feature[headers[3]].int64_list.value.extend(
      [int(input_list[3])])
  # Write image width and height.
  example.features.feature[headers[4]].int64_list.value.extend(
      [int(input_list[4])])
  example.features.feature[headers[5]].int64_list.value.extend(
      [int(input_list[5])])

  # Write the 2D and 3D keypoints as floats.
  for i in range(6, len(headers)):
    example.features.feature[headers[i]].float_list.value.extend(
        [float(input_list[i])])
  return example


def create_serialized_example_with_2d_3d_keypoints(input_list, headers,
                                                   write_pairs):
  """Creates serialized example with 2D and 3D keypoints to write to TFRecord.

  Args:
    input_list: A list of values with 2D keypoints, 3D keypoints and metadata
      (subject name, action, name, camera, timestamp and image size).
    headers: A list of string headers with 2D keypoint names, 3D keypoint names,
      and metadata to write to the tf.example.
    write_pairs: A boolean for whether we want to write paired entries to output
      TFRecord. If False, we write a single entry.

  Returns:
    serialized_example: A string for serialized tf.example to write to TFRecord.
  """
  example = tf.train.Example()

  example = extend_example_with_2d_3d_keypoints(example,
                                                input_list[:len(headers)],
                                                headers)

  if write_pairs:
    example = extend_example_with_2d_3d_keypoints(example,
                                                  input_list[len(headers):],
                                                  headers)

  return example.SerializeToString()


def load_2d_keypoints_and_write_tfrecord_with_3d_keypoints(
    input_csv_file, keypoint_dict, output_tfrecord_file, read_csv_pairs,
    num_shards):
  """Loads 2D keypoints from a CSV file and write TFRecord with 3D poses.

  The TFRecord written contains the 2D keypoints with corresponding 3D keypoints
  stored in keypoint_dict.

  Args:
    input_csv_file: A string of the CSV file name containing 2D keypoints to
      load with subjects, actions and timestamps to be matched to 3D keypoints
      from keypoint_dict.
    keypoint_dict: A dictionary for loaded 3D keypoints. Keys are (subject,
      action) and values are of shape [sequence_length, num_keypoints, 3].
    output_tfrecord_file: A string of output filename for the TFRecord
      containing 2D and 3D keypoints.
    read_csv_pairs: A boolean that is True when each row of the CSV file stores
      paired entried and is False when the row contains a single entry.
    num_shards: An integer for the number of shards in the output TFRecord file.
  """

  # Read the first row of the file as the header.
  read_header = True

  keypoint_profile_h36m17 = (
      keypoint_profiles.create_keypoint_profile_or_die('LEGACY_3DH36M17'))

  tfrecord_writers = []
  if num_shards > 1:
    for i in range(num_shards):
      output_tfrecord_file_sharded = (
          output_tfrecord_file + '-{:05d}-of-{:05d}'.format(i, num_shards))
      writer = tf.python_io.TFRecordWriter(output_tfrecord_file_sharded)
      tfrecord_writers.append(writer)
  else:
    writer = tf.python_io.TFRecordWriter(output_tfrecord_file)
    tfrecord_writers.append(writer)

  with tf.io.gfile.GFile(input_csv_file, 'r') as csv_rows:
    for shard_counter, row in enumerate(csv_rows):
      writer = tfrecord_writers[shard_counter % num_shards]
      row = row.split(',')

      feature_size = len(row)
      if read_csv_pairs:
        feature_size = len(row) // 2
        if len(row) != feature_size * 2 or len(row) % 2 != 0:
          raise ValueError('CSV row has length {} but it should have an even'
                           'number of elements.'.format(len(row)))
      if read_header:
        read_header = False
        # Keep the first half of the row as header if the csv file contains
        # pairs. Otherwise, keep the full row.
        headers = row[:feature_size]
        # Add 3D pose headers using the keypoint names to the header list.
        prefix = 'image/object/part_3d/'
        suffix = '/center/'
        for name in keypoint_profile_h36m17.keypoint_names:
          headers.append(prefix + name + suffix + 'x')
          headers.append(prefix + name + suffix + 'y')
          headers.append(prefix + name + suffix + 'z')
        continue

      anchor_subject = row[0]
      anchor_action = row[1]
      anchor_frame_index = int(row[3])
      # Replace names to be consistent with H5 file names.
      anchor_action = anchor_action.replace('TakingPhoto', 'Photo').replace(
          'WalkingDog', 'WalkDog')

      # Obtain matching 3D keypoints from keypoint_dict.
      anchor_keypoint_3d = keypoint_dict[(
          anchor_subject,
          anchor_action)][anchor_frame_index,
                          KEYPOINT_3D_INDICES_H5, :].reshape([-1])

      # If we need to read csv pairs, the second element in the pair in the row
      # is the positive match.
      if read_csv_pairs:
        positive_subject = row[feature_size]
        positive_action = row[feature_size + 1]
        positive_frame_index = int(row[feature_size + 3])
        positive_action = positive_action.replace('TakingPhoto',
                                                  'Photo').replace(
                                                      'WalkingDog', 'WalkDog')
        positive_keypoint_3d = keypoint_dict[(
            positive_subject,
            positive_action)][positive_frame_index,
                              KEYPOINT_3D_INDICES_H5, :].reshape([-1])

        # Concatenate 3D keypoints into current row with 2D keypoints.
        row_with_3d_keypoints = np.concatenate(
            (row[:feature_size], anchor_keypoint_3d, row[feature_size:],
             positive_keypoint_3d))
      else:
        row_with_3d_keypoints = np.concatenate((row, anchor_keypoint_3d))

      serialized_example = create_serialized_example_with_2d_3d_keypoints(
          row_with_3d_keypoints, headers, write_pairs=read_csv_pairs)
      writer.write(serialized_example)


def main(_):

  if FLAGS.subjects_to_read.lower() == 'all':
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
  elif FLAGS.subjects_to_read.lower() == 'train':
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
  elif FLAGS.subjects_to_read.lower() == 'val':
    subjects = ['S9', 'S11']
  elif not FLAGS.subjects_to_read:
    raise ValueError('Must add flag specifying subjects to read from H5 files.')
  else:
    subjects = FLAGS.subjects_to_read.split()

  if FLAGS.input_data_type == 'H36M_H5':
    tf.logging.info('Loading 3D keypoints from H5 files.')
    keypoint_dict = load_h36m_3d_keypoints_h5_files(FLAGS.input_root_dir,
                                                    subjects)
  else:
    raise NotImplementedError

  tf.logging.info('3D keypoints loaded.')

  load_2d_keypoints_and_write_tfrecord_with_3d_keypoints(
      FLAGS.input_csv_file,
      keypoint_dict,
      FLAGS.output_tfrecords,
      read_csv_pairs=FLAGS.input_paired_entries,
      num_shards=FLAGS.output_num_shards)


if __name__ == '__main__':
  tf.app.run(main)
