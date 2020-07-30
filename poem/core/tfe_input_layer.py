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

"""Tensorflow Example input layer."""

import tensorflow as tf

from poem.core import common


def add_decoder_image_sizes(instance_shape, common_module=common):
  """Adds decoders for image sizes.

  Args:
    instance_shape: A list of integers for the shape (layout) of instances for
      each record.
    common_module: A Python module that defines common constants.

  Returns:
    A dictionary for decoders.
  """
  return {
      common_module.TFE_KEY_IMAGE_HEIGHT:
          tf.io.FixedLenFeature(instance_shape, dtype=tf.int64),
      common_module.TFE_KEY_IMAGE_WIDTH:
          tf.io.FixedLenFeature(instance_shape, dtype=tf.int64),
  }


def add_decoder_keypoints_2d(keypoint_names_2d,
                             include_keypoint_scores_2d,
                             instance_shape,
                             common_module=common):
  """Adds decoders for 2D keypoints.

  Args:
    keypoint_names_2d: A list of strings for 2D keypoint names.
    include_keypoint_scores_2d: A boolean for whether to include 2D keypoint
      scores.
    instance_shape: A list of integers for the shape (layout) of instances for
      each record.
    common_module: A Python module that defines common constants.

  Returns:
    decoders: A dictionary for decoders.
  """
  decoders = {}

  for name in keypoint_names_2d:
    for key_suffix in common_module.TFE_KEY_SUFFIX_KEYPOINT_2D:
      key = common_module.TFE_KEY_PREFIX_KEYPOINT_2D + name + key_suffix
      decoders[key] = tf.io.FixedLenFeature(instance_shape, dtype=tf.float32)
    if include_keypoint_scores_2d:
      key = (
          common_module.TFE_KEY_PREFIX_KEYPOINT_2D + name +
          common_module.TFE_KEY_SUFFIX_KEYPOINT_SCORE)
      decoders[key] = tf.io.FixedLenFeature(instance_shape, dtype=tf.float32)

  return decoders


def add_decoder_keypoints_3d(keypoint_names_3d,
                             include_keypoint_scores_3d,
                             instance_shape,
                             common_module=common):
  """Adds decoders for 3D keypoints.

  Args:
    keypoint_names_3d: A list of strings for 3D keypoint names.
    include_keypoint_scores_3d: A boolean for whether to include 3D keypoint
      scores.
    instance_shape: A list of integers for the shape (layout) of instances for
      each record.
    common_module: A Python module that defines common constants.

  Returns:
    decoders: A dictionary for decoders.
  """
  decoders = {}

  for name in keypoint_names_3d:
    for key_suffix in common_module.TFE_KEY_SUFFIX_KEYPOINT_3D:
      key = common_module.TFE_KEY_PREFIX_KEYPOINT_3D + name + key_suffix
      decoders[key] = tf.io.FixedLenFeature(instance_shape, dtype=tf.float32)
    if include_keypoint_scores_3d:
      key = (
          common_module.TFE_KEY_PREFIX_KEYPOINT_3D + name +
          common_module.TFE_KEY_SUFFIX_KEYPOINT_SCORE)
      decoders[key] = tf.io.FixedLenFeature(instance_shape, dtype=tf.float32)

  return decoders


def process_decoded_image_sizes(decoded_tensors, common_module=common):
  """Processes decoded image sizes.

  Args:
    decoded_tensors: A dictionary for decoded tensors.
    common_module: A Python module that defines common constants.

  Returns:
    A dictionary for processed 2D keypoint tensors.
  """
  image_heights = decoded_tensors[common_module.TFE_KEY_IMAGE_HEIGHT]
  image_widths = decoded_tensors[common_module.TFE_KEY_IMAGE_WIDTH]
  return {
      common_module.KEY_IMAGE_SIZES:
          tf.stack([image_heights, image_widths], axis=-1)
  }


def process_decoded_keypoints_2d(decoded_tensors,
                                 keypoint_names_2d,
                                 include_keypoint_scores_2d,
                                 common_module=common):
  """Processes decoded 2D keypoint tensors.

  Args:
    decoded_tensors: A dictionary for decoded tensors.
    keypoint_names_2d: A list of strings for 2D keypoint names.
    include_keypoint_scores_2d: A boolean for whether to include 2D keypoint
      scores.
    common_module: A Python module that defines common constants.

  Returns:
    outputs: A dictionary for processed 2D keypoint tensors.
  """
  outputs = {}

  keypoints_2d = []
  for name in keypoint_names_2d:
    sub_keypoints_2d = []
    for key_suffix in common_module.TFE_KEY_SUFFIX_KEYPOINT_2D:
      key = common_module.TFE_KEY_PREFIX_KEYPOINT_2D + name + key_suffix
      sub_keypoints_2d.append(decoded_tensors[key])
    keypoints_2d.append(tf.stack(sub_keypoints_2d, axis=-1))
  outputs[common_module.KEY_KEYPOINTS_2D] = tf.stack(keypoints_2d, axis=-2)

  if include_keypoint_scores_2d:
    keypoint_scores_2d = []
    for name in keypoint_names_2d:
      key = (
          common_module.TFE_KEY_PREFIX_KEYPOINT_2D + name +
          common_module.TFE_KEY_SUFFIX_KEYPOINT_SCORE)
      keypoint_scores_2d.append(decoded_tensors[key])
    outputs[common_module.KEY_KEYPOINT_SCORES_2D] = tf.stack(
        keypoint_scores_2d, axis=-1)

  return outputs


def process_decoded_keypoints_3d(decoded_tensors,
                                 keypoint_names_3d,
                                 include_keypoint_scores_3d,
                                 common_module=common):
  """Processes decoded 3D keypoint tensors.

  Args:
    decoded_tensors: A dictionary for decoded tensors.
    keypoint_names_3d: A list of strings for 3D keypoint names.
    include_keypoint_scores_3d: A boolean for whether to include 2D keypoint
      scores.
    common_module: A Python module that defines common constants.

  Returns:
    outputs: A dictionary for processed 2D keypoint tensors.
  """
  outputs = {}

  keypoints_3d = []
  for name in keypoint_names_3d:
    sub_keypoints_3d = []
    for key_suffix in common_module.TFE_KEY_SUFFIX_KEYPOINT_3D:
      key = common_module.TFE_KEY_PREFIX_KEYPOINT_3D + name + key_suffix
      sub_keypoints_3d.append(decoded_tensors[key])
    keypoints_3d.append(tf.stack(sub_keypoints_3d, axis=-1))
  outputs[common_module.KEY_KEYPOINTS_3D] = tf.stack(keypoints_3d, axis=-2)

  if include_keypoint_scores_3d:
    keypoint_scores_3d = []
    for name in keypoint_names_3d:
      key = (
          common_module.TFE_KEY_PREFIX_KEYPOINT_3D + name +
          common_module.TFE_KEY_SUFFIX_KEYPOINT_SCORE)
      keypoint_scores_3d.append(decoded_tensors[key])
    outputs[common_module.KEY_KEYPOINT_SCORES_3D] = tf.stack(
        keypoint_scores_3d, axis=-1)

  return outputs


def get_tfe_parser_fn(decoders, post_process_fn):
  """Creates a tf.Example parser function.

  Args:
    decoders: A dictionary for keyed tf.Example field decoders.
    post_process_fn: A function handle for post processing decoded tensors.

  Returns:
    parser_fn: A function handle for the parser function.
  """

  def parser_fn(*inputs):
    """Decoder function."""
    # Here `inputs` can be either just a serialized example or a (key,
    # serialized example) tuple (in which we ignore the key), and we would like
    # to handle both cases.
    serialized_example = inputs[-1]
    decoded_tensors = tf.io.parse_single_example(serialized_example, decoders)
    return post_process_fn(decoded_tensors)

  return parser_fn


def create_tfe_parser(keypoint_names_2d=None,
                      keypoint_names_3d=None,
                      include_keypoint_scores_2d=True,
                      include_keypoint_scores_3d=False,
                      num_objects=1,
                      common_module=common):
  """Creates default tf.Example parser function.

  Args:
    keypoint_names_2d: A list of strings for 2D keypoint names. Use None to skip
      reading 2D keypoints.
    keypoint_names_3d: A list of strings for 3D keypoint names. Use None to skip
      reading 3D keypoints.
    include_keypoint_scores_2d: A boolean for whether to read 2D keypoint
      scores. Only used if `keypoint_names_2d` is specified.
    include_keypoint_scores_3d: A boolean for whether to read 3D keypoint
      scores. Only used if `keypoint_names_3d` is specified.
    num_objects: An integer for the number of objects each example has.
    common_module: A Python module that defines common constants.

  Returns:
    parser_fn: A function handle for the parser.
  """
  decoders = add_decoder_image_sizes(
      instance_shape=[num_objects], common_module=common_module)

  if keypoint_names_2d:
    decoders.update(
        add_decoder_keypoints_2d(
            keypoint_names_2d,
            include_keypoint_scores_2d=include_keypoint_scores_2d,
            instance_shape=[num_objects],
            common_module=common_module))

  if keypoint_names_3d:
    decoders.update(
        add_decoder_keypoints_3d(
            keypoint_names_3d,
            include_keypoint_scores_3d=include_keypoint_scores_3d,
            instance_shape=[num_objects],
            common_module=common_module))

  def post_process_decoded_tensors(decoded_tensors):
    """Post processes decoded tensors."""
    outputs = process_decoded_image_sizes(decoded_tensors, common_module)

    if keypoint_names_2d:
      outputs.update(
          process_decoded_keypoints_2d(
              decoded_tensors,
              keypoint_names_2d=keypoint_names_2d,
              include_keypoint_scores_2d=include_keypoint_scores_2d,
              common_module=common_module))

    if keypoint_names_3d:
      outputs.update(
          process_decoded_keypoints_3d(
              decoded_tensors,
              keypoint_names_3d=keypoint_names_3d,
              include_keypoint_scores_3d=include_keypoint_scores_3d,
              common_module=common_module))

    return outputs

  return get_tfe_parser_fn(decoders, post_process_decoded_tensors)


def read_from_table(table_pattern,
                    shuffle=True,
                    num_epochs=None,
                    shuffle_buffer_size=4096,
                    num_shards=1,
                    shard_index=None,
                    dataset_class=tf.data.TFRecordDataset,
                    parser_fn=None,
                    seed=None):
  """Reads tf.Examples from input table.

  Args:
    table_pattern: Path or pattern to input tables.
    shuffle: A boolean for whether to shuffle the common queue when reading.
    num_epochs: An integer for the number of epochs to read. Use None to read
      indefinitely.
    shuffle_buffer_size: An integer for the buffer size used for shuffling. A
      large buffer size benefits shuffling quality.
    num_shards: An integer for the number of shards to divide the dataset. This
      is useful to distributed training. See `tf.data.Dataset.shard` for
      details.
    shard_index: An integer for the shard index to use. This is useful to
      distributed training, and should usually be set to the id of a
      synchronized worker. See `tf.data.Dataset.shard` for details. Note this
      must be specified if `num_shards` is greater than 1.
    dataset_class: A dataset class to use. Must match input table type.
    parser_fn: A function handle for parser function.
    seed: An integer for random seed.

  Returns:
    A dictionary of parsed input tensors.

  Raises:
    ValueError: If `num_shards` is greater than 1 but `shard_index` is not
      specified.
  """
  dataset = tf.data.Dataset.list_files(
      table_pattern, shuffle=shuffle, seed=seed)
  dataset = dataset.interleave(
      dataset_class, cycle_length=tf.data.experimental.AUTOTUNE)
  dataset = dataset.repeat(num_epochs)

  if num_shards > 1:
    if shard_index is None:
      raise ValueError('Shard index is not specified: `%s.`' % str(shard_index))
    dataset = dataset.shard(num_shards, index=shard_index)

  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

  dataset = dataset.map(
      parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset


def read_batch_from_tables(table_patterns,
                           batch_sizes,
                           drop_remainder,
                           seed=None,
                           **reader_kwargs):
  """Reads and batches inputs from tf.Example tables.

  Args:
    table_patterns: A list of strings for the paths or pattern to input tables.
    batch_sizes: A list of integers for the batch sizes to read from each table.
    drop_remainder: A boolean for whether to drop remaining elements that cannot
      make up a full batch at the end of an epoch. Usually set to True for
      evaluation.
    seed: An integer for random seed.
    **reader_kwargs: A dictionary of additional arguments passed to
      `read_from_table`.

  Returns:
    A dictionary of parsed input tensors.

  Raises:
    ValueError: If the size of `table_patterns` is different than that of
      `nums_samples`.
    ValueError: If the size of `table_patterns` is different than that of
      `batch_sizes`.
  """
  if not table_patterns:
    raise ValueError('No table pattern is provided.')

  if len(table_patterns) != len(batch_sizes):
    raise ValueError(
        'Number of table patterns is different than that of batch sizes: %d vs.'
        ' %d.' % (len(table_patterns), len(batch_sizes)))

  if len(table_patterns) == 1:
    dataset = read_from_table(table_patterns[0], seed=seed, **reader_kwargs)
  else:
    datasets = [
        read_from_table(table_pattern, seed=seed, **reader_kwargs)
        for table_pattern in table_patterns
    ]
    dataset = tf.data.experimental.sample_from_datasets(
        datasets, weights=[float(x) for x in batch_sizes], seed=seed)
  return dataset.batch(sum(batch_sizes), drop_remainder=drop_remainder)
