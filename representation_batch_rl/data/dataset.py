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

"""Create tensorflow datasets."""

from typing import List

from absl import logging
import tensorflow as tf

from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import example_encoding
from tf_agents.utils import example_encoding_dataset


def filter_episodes(traj):
  """Map TFRecord windows (of adjacent TimeSteps) to single episodes.

  Outputs the last episode within a sample window. It does this by using
  the step_type tensor to break up sequences into single episode sections.
  For example, if step_type is: [FIRST, MID, LAST, FIRST, MID, MID], we
  will return a sample, whos tensor indices are sampled as:
  [3, 3, 3, 3, 4, 5]. So that the 3rd index frame is replicated 3 times to
  across the beginning of the tensor.

  Args:
    traj: Trajectory.

  Returns:
    Trajectory containing filtered sample with only one episode.
  """
  step_types = traj.step_type
  seq_len = tf.cast(tf.shape(step_types)[0], tf.int32)

  # Find the last start frame in the window. e.g. if we have step types
  # [FIRST, MID, LAST, FIRST, MID, MID], we want index 3.
  first_frames = tf.where(step_types == StepType.FIRST)

  if tf.shape(first_frames)[0] == 0:
    # No first frame, return sequence as is.
    inds = tf.range(0, seq_len)
  else:
    ind_start = tf.cast(first_frames[-1, 0], tf.int32)
    if ind_start == 0:
      # Last episode starts on the first frame, return as is.
      inds = tf.range(0, seq_len)
    else:
      # Otherwise, resample so that the last episode's first frame is
      # replicated to the beginning of the sample. In the example above we want:
      # [3, 3, 3, 3, 4, 5].
      inds_start = tf.tile(ind_start[None], ind_start[None])
      inds_end = tf.range(ind_start, seq_len)
      inds = tf.concat([inds_start, inds_end], axis=0)

  def _resample(arr):
    if isinstance(arr, tf.Tensor):
      return tf.gather(arr, inds)
    else:
      return arr  # empty or None

  observation = tf.nest.map_structure(_resample, traj.observation)

  return Trajectory(
      step_type=_resample(traj.step_type),
      action=_resample(traj.action),
      policy_info=_resample(traj.policy_info),
      next_step_type=_resample(traj.next_step_type),
      reward=_resample(traj.reward),
      discount=_resample(traj.discount),
      observation=observation)


def filter_episodes_rnn(traj):
  """Map TFRecord windows (of adjacent TimeSteps) so only first episode valid.

  Args:
    traj: Trajectory.

  Returns:
    Trajectory containing filtered sample with only one episode.
  """
  step_types = traj.step_type
  seq_len = tf.cast(tf.shape(step_types)[0], tf.int32)

  # Find the first "LAST" frame.  Set everything after it to be invalid.
  last_frames = tf.where(step_types == StepType.LAST)
  if tf.shape(last_frames)[0] == 0:
    discount = tf.ones_like(traj.discount)
  else:
    first_last = tf.cast(last_frames[0], tf.int32) + 1
    if first_last == seq_len:
      discount = tf.ones_like(traj.discount)
    else:
      valid = tf.ones(first_last)
      not_valid = tf.zeros(seq_len - first_last)
      discount = tf.concat([valid, not_valid], axis=0)

  return traj._replace(discount=discount)


def load_tfrecord_dataset_sequence(
    path_to_shards,
    buffer_size_per_shard = 100,
    seq_len = 1,
    deterministic = False,
    compress_image = True,
    for_rnn = False,
    check_all_specs = False):
  """A version of load_tfrecord_dataset that returns fixed length sequences.

  Note that we pad on the first frame to output seq_len. So a sequence of
  [0, 1, 2], with seq_len = 2 will produce samples of [0, 1], [1, 2], [0, 0],
  [0, 1], [1, 2], etc

  Args:
    path_to_shards: Path to TFRecord shards.
    buffer_size_per_shard: per-shard TFRecordReader buffer size.
    seq_len: fixed length output sequence.
    deterministic: If True, maintain deterministic sampling of shards (typically
      for testing).
    compress_image: Whether to decompress image. It is assumed that any uint8
      tensor of rank 3 with shape (w,h,c) is an image.
      If the tensor was compressed in the encoder, it needs to be decompressed.
    for_rnn: if True, see filter_episodes_rnn()
    check_all_specs: if True, check every spec.

  Returns:
    tf.data.Dataset object.
  """
  specs = []
  check_shards = path_to_shards if check_all_specs else path_to_shards[:1]
  for dataset_file in check_shards:
    spec_path = dataset_file + example_encoding_dataset._SPEC_FILE_EXTENSION  # pylint: disable=protected-access
    dataset_spec = example_encoding_dataset.parse_encoded_spec_from_file(
        spec_path)
    specs.append(dataset_spec)
    if not all([dataset_spec == spec for spec in specs]):
      raise ValueError('One or more of the encoding specs do not match.')
  decoder = example_encoding.get_example_decoder(specs[0], batched=True,
                                                 compress_image=compress_image)

  # Note: window cannot be called on TFRecordDataset(shards) directly as it
  # interleaves samples across the shards. Instead, we'll sample windows on
  # shards independently using interleave.
  def interleave_func(shard):
    dataset = tf.data.TFRecordDataset(
        shard, buffer_size=buffer_size_per_shard).cache().repeat()
    dataset = dataset.window(seq_len, shift=1, stride=1, drop_remainder=True)
    return dataset.flat_map(
        lambda window: window.batch(seq_len, drop_remainder=True))

  dataset = tf.data.Dataset.from_tensor_slices(path_to_shards).repeat()
  num_parallel_calls = None if deterministic else len(path_to_shards)
  dataset = dataset.interleave(interleave_func,
                               deterministic=deterministic,
                               cycle_length=len(path_to_shards),
                               block_length=1,
                               num_parallel_calls=num_parallel_calls)

  # flat_map doesn't work with Dict[str, tf.Tensor], so for now decode after
  # the window sample (this causes unnecessary decode of protos).
  # TODO(tompson): It would be more efficient to decode before window.
  dataset = dataset.map(decoder, num_parallel_calls=num_parallel_calls)

  # We now have decoded sequences, each sample containing adjacent frames
  # within a single shard. However, the window may span multiple episodes, so
  # we need to filter these.

  if for_rnn:
    return dataset.map(
        filter_episodes_rnn, num_parallel_calls=num_parallel_calls)
  else:
    dataset = dataset.map(
        filter_episodes, num_parallel_calls=num_parallel_calls)

    # Set observation shape.
    def set_shape_obs(traj):

      def set_elem_shape(obs):
        obs_shape = obs.get_shape()
        return tf.ensure_shape(obs, [seq_len] + obs_shape[1:])

      observation = tf.nest.map_structure(set_elem_shape, traj.observation)
      return traj._replace(observation=observation)

    dataset = dataset.map(set_shape_obs, num_parallel_calls=num_parallel_calls)
    return dataset


def get_shards(dataset_path, max_data_shards, separator=','):
  """Globs a dataset or aggregates records from a set of datasets."""
  if separator in dataset_path:
    # Data is a ','-separated list of training paths. Glob them all and then
    # aggregate into one dataset.
    dataset_paths = dataset_path.split(separator)
    shards = []
    for d in dataset_paths:
      # Glob task data.
      task_data = tf.io.gfile.glob(d)
      # Optionally limit each task to max shards.
      if max_data_shards != -1:
        task_data = task_data[:max_data_shards]
        logging.info('limited to %d shards', max_data_shards)
      shards.extend(task_data)
  else:
    shards = tf.io.gfile.glob(dataset_path)
    if max_data_shards != -1:
      shards = shards[:max_data_shards]
      logging.info('limited to %d shards', max_data_shards)
  return shards


def create_sequence_datasets(dataset_path,
                             sequence_length,
                             replay_capacity,
                             batch_size,
                             for_rnn,
                             eval_fraction,
                             max_data_shards=-1):
  """Make train and eval datasets."""
  path_to_shards = get_shards(dataset_path, max_data_shards)
  if not path_to_shards:
    raise ValueError('No data found at %s' % dataset_path)

  num_eval_shards = int(len(path_to_shards) * eval_fraction)
  num_train_shards = len(path_to_shards) - num_eval_shards
  train_shards = path_to_shards[0:num_train_shards]
  if num_eval_shards > 0:
    eval_shards = path_to_shards[num_train_shards:]

  def _make_dataset(path_to_shards):
    sequence_dataset = load_tfrecord_dataset_sequence(
        path_to_shards, seq_len=sequence_length, for_rnn=for_rnn)
    sequence_dataset = sequence_dataset.repeat().shuffle(replay_capacity).batch(
        batch_size, drop_remainder=True)
    return sequence_dataset

  train_dataset = _make_dataset(train_shards)
  if num_eval_shards > 0:
    eval_dataset = _make_dataset(eval_shards)
  else:
    eval_dataset = None

  return train_dataset, eval_dataset
