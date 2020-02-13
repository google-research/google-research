# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Off-policy input data pipelines for training RL algorithms.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import numpy as np
import tensorflow.compat.v1 as tf


SARSTransition = collections.namedtuple('SARSTransition',
                                        ['state', 'action', 'reward',
                                         'state_p1', 'done', 'aux'])


@gin.configurable
def parse_tfexample_v0(example_proto,
                       img_height=48,
                       img_width=48,
                       action_size=None):
  """Parse TFExamples saved by episode_to_transitions.

  Args:
    example_proto: tf.String tensor representing a serialized protobuf.
    img_height: Height of parsed image tensors.
    img_width: Width of parsed image tensors.
    action_size: Size of continuous actions. If None, actions are assumed to be
      integer-encoded discrete actions.
  Returns:
    NamedTuple of type SARSTransition containing unbatched Tensors.
  """
  if action_size is None:
    # Is discrete.
    action_feature_spec = tf.FixedLenFeature((), tf.int64)
  else:
    # Vector-encoded float feature.
    action_feature_spec = tf.FixedLenFeature((action_size,), tf.float32)

  features = {'S/img': tf.FixedLenFeature((), tf.string),
              'A': action_feature_spec,
              'R': tf.FixedLenFeature((), tf.float32),
              'S_p1/img': tf.FixedLenFeature((), tf.string),
              'done': tf.FixedLenFeature((), tf.int64),
              't': tf.FixedLenFeature((), tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  # Decode the jpeg-encoded images into numeric tensors.
  states = []
  for key in ['S/img', 'S_p1/img']:
    img = tf.image.decode_jpeg(parsed_features[key], channels=3)
    img.set_shape([img_height, img_width, 3])
    img = tf.cast(img, tf.float32)
    states.append(img)

  action = parsed_features['A']
  reward = parsed_features['R']
  done = tf.cast(parsed_features['done'], tf.float32)
  step = tf.cast(parsed_features['t'], tf.int32)
  aux = {'step': step}
  return SARSTransition(
      (states[0], step), action, reward, (states[1], step + 1), done, aux)


@gin.configurable
def parse_tfexample_pm_v1(example_proto, action_size=None):
  """Parse TFExamples saved by episode_to_transitions_pm with all metrics.

  Args:
    example_proto: tf.String tensor representing a serialized protobuf.
    action_size: Size of continuous actions. If None, actions are assumed to be
      integer-encoded discrete actions.

  Returns:
    NamedTuple of type SARSTransition containing unbatched Tensors.
  """
  if action_size is None:
    # Is discrete.
    action_feature_spec = tf.FixedLenFeature((), tf.int64)
  else:
    # Vector-encoded float feature.
    action_feature_spec = tf.FixedLenFeature((action_size,), tf.float32)

  features = {
      'A': action_feature_spec,
      'R': tf.FixedLenFeature((), tf.float32),
      'done': tf.FixedLenFeature((), tf.int64),
      't': tf.FixedLenFeature((), tf.int64)
  }

  for key in ['x', 'v', 'vtg', 'vtg_gt', 'prev_a', 'ae_steps', 'as_steps']:
    features[key] = tf.FixedLenFeature((), tf.float32)
    features[key + '_tp1'] = tf.FixedLenFeature((), tf.float32)

  parsed_features = tf.parse_single_example(example_proto, features)
  state_t = [parsed_features['x'], parsed_features['v']]
  state_tp1 = [parsed_features['x_tp1'], parsed_features['v_tp1']]
  vtg_t = parsed_features['vtg']
  vtg_tp1 = parsed_features['vtg_tp1']
  vtg_gt_t = parsed_features['vtg_gt']
  vtg_gt_tp1 = parsed_features['vtg_gt_tp1']
  prev_a_t = parsed_features['prev_a']
  prev_a_tp1 = parsed_features['prev_a_tp1']
  ae_steps_t = parsed_features['ae_steps']
  ae_steps_tp1 = parsed_features['ae_steps_tp1']
  as_steps_t = parsed_features['as_steps']
  as_steps_tp1 = parsed_features['as_steps_tp1']
  action = parsed_features['A']
  reward = parsed_features['R']
  done = tf.cast(parsed_features['done'], tf.float32)
  step = tf.cast(parsed_features['t'], tf.int32)
  aux = {'step': step}
  return SARSTransition(
      (state_t, step, vtg_t, vtg_gt_t, prev_a_t, ae_steps_t, as_steps_t),
      action, reward, (state_tp1, step + 1, vtg_tp1, vtg_gt_tp1, prev_a_tp1,
                       ae_steps_tp1, as_steps_tp1), done, aux)


@gin.configurable
def parse_tfexample_sequence(example_proto,
                             img_height=48,
                             img_width=48,
                             action_size=None,
                             episode_length=16):
  """Parse TFExamples saved by episode_to_transitions.

  Args:
    example_proto: tf.String tensor representing a serialized protobuf.
    img_height: Height of parsed image tensors.
    img_width: Width of parsed image tensors.
    action_size: Size of continuous actions. If None, actions are assumed to be
      integer-encoded discrete actions.
    episode_length: Intended length of each episode.
  Returns:
    NamedTuple of type SARSTransition containing unbatched Tensors.
  """
  if action_size is None:
    # Is discrete.
    action_feature_spec = tf.FixedLenFeature((episode_length,), tf.int64)
  else:
    # Vector-encoded float feature.
    action_feature_spec = tf.FixedLenFeature((episode_length, action_size),
                                             tf.float32)

  features = {'S/img': tf.FixedLenFeature((episode_length,), tf.string),
              'A': action_feature_spec,
              'R': tf.FixedLenFeature((episode_length,), tf.float32),
              'S_p1/img': tf.FixedLenFeature((episode_length,), tf.string),
              'done': tf.FixedLenFeature((episode_length,), tf.int64),
              't': tf.FixedLenFeature((episode_length,), tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  # Decode the jpeg-encoded images into numeric tensors.
  states = []
  for key in 'S/img', 'S_p1/img':
    state = tf.stack(
        [tf.image.decode_jpeg(img, channels=3)
         for img in tf.unstack(parsed_features[key], num=episode_length)])
    state.set_shape([episode_length, img_height, img_width, 3])
    states.append(tf.cast(state, tf.float32))

  action = parsed_features['A']
  reward = parsed_features['R']
  done = tf.cast(parsed_features['done'], tf.float32)
  step = tf.cast(parsed_features['t'], tf.int32)
  aux = {'step': step}

  return SARSTransition(
      (states[0], step), action, reward, (states[1], step + 1), done, aux)


@gin.configurable
def get_data(file_patterns=None,
             parse_fn=parse_tfexample_v0,
             shuffle_filenames=True,
             batch_size=32,
             shuffle_buffer_size=10000,
             data_format='tfrecord'):
  """Randomly samples and batches transitions from RecordIO files.

  Args:
    file_patterns: Comma-separated list of file patterns.
    parse_fn: Python function that is used to parse tensors from Proto strings.
    shuffle_filenames: If True, shuffle filenames. Otherwise, data is fed into
      the shuffle queue in reverse alphabetical order.
    batch_size: Batch dimension of returned tensors for SGD training.
    shuffle_buffer_size: How many serialized protos to shuffle before passing it
      on to the deserializing/minibatching steps. The larger shuffle_buffer_size
      is, the more de-correlated sampled transitions will be.
    data_format: One of 'recordio', 'sstable', or 'tfrecord'.

  Returns:
    TensorFlow Dataset representing the off-policy transition tensors.

  Raises:
    ValueError: If no file_patterns or no files match any patterns in
      file_patterns.
    RuntimeError: If data_format is not one of 'recordio', 'sstable',
      or 'tfrecords'.
  """
  if not file_patterns:
    raise ValueError('Invalid file patterns %s.' % file_patterns)
  filenames = []
  for pattern in file_patterns.split(','):
    try:
      filenames += tf.gfile.Glob(pattern)
    except tf.errors.NotFoundError:
      print('Pattern %s does not match any files. Skipping...' % pattern)
  if not filenames:
    raise ValueError('No files found!')
  if shuffle_filenames:
    np.random.shuffle(filenames)
  else:
    filenames = list(reversed(filenames))

  allowed_formats = ['tfrecord']

  if data_format not in allowed_formats:
    raise RuntimeError('data_format must be one of %s' % allowed_formats)

  if data_format == 'tfrecord':
    dataset = tf.data.TFRecordDataset(filenames)

  dataset = dataset.repeat()
  dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
  # Parse just enough into tensors to keep prefetch queue full.
  dataset = dataset.map(parse_fn,
                        num_parallel_calls=2)
  dataset = dataset.prefetch(2 * batch_size)
  dataset = dataset.batch(batch_size)
  return dataset
