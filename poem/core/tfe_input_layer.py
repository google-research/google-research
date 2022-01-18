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


def add_decoder_features(feature_dim, instance_shape, common_module=common):
  """Adds decoder for pre-computed features.

  Args:
    feature_dim: An integer for size of feature vectors.
    instance_shape: A list of integers for the shape (layout) of instances for
      each record.
    common_module: A Python module that defines common constants.

  Returns:
    A dictionary for decoder.
  """
  feature_shape = list(instance_shape) + [feature_dim]
  return {
      common_module.TFE_KEY_FEATURE:
          tf.io.FixedLenFeature(feature_shape, dtype=tf.float32)
  }


def add_decoder_class_labels(common_module=common):
  """Adds decoders for class label ids and confidences.

  IMPORTANT: Note that we assume there is one copy of label ids and confidences
  in each record and they apply to all the objects in the record.

  Args:
    common_module: A Python module that defines common constants.

  Returns:
    A dictionary for decoders.
  """
  return {
      common_module.TFE_KEY_CLASS_LABEL_ID:
          tf.io.VarLenFeature(dtype=tf.int64),
      common_module.TFE_KEY_CLASS_LABEL_CONFIDENCE:
          tf.io.VarLenFeature(dtype=tf.float32)
  }


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


def process_decoded_features(decoded_tensors, common_module=common):
  """Processes decoded features.

  Args:
    decoded_tensors: A dictionary for decoded tensors.
    common_module: A Python module that defines common constants.

  Returns:
    A dictionary for processed 2D keypoint tensors.
  """
  return {
      common_module.KEY_FEATURES: decoded_tensors[common_module.TFE_KEY_FEATURE]
  }


def generate_class_targets(label_ids,
                           label_confidences,
                           num_classes,
                           positive_label_confidence_threshold=0.5):
  """Generates class targets and weights from label ids and confidences.

  Note that we use `class_targets` to represent if a label is positive or
  negative, and use `class_weights` to represent if a label exists in the input
  or not.

  Example usage:
    num_classes = 5
    label_ids = [0, 1, 3]
    label_confidences = [0.9, 0.3, 0.7]
    positive_label_confidence_threshold = 0.5
    -->
    class_targets = [1, 0, 0, 1, 0]
    class_weights = [1.0, 1.0, 0.0, 1.0, 0.0]

  Args:
    label_ids: A tensor for label ids. Shape = [num_classes].
    label_confidences: A tensor for label confidences. Shape = [num_classes].
    num_classes: An integer for total number of classes.
    positive_label_confidence_threshold: A float for the threshold to determine
      class target for label ids. If the confidence of a label id is greater
      than this value, it has positive class target (1), otherwise negative
      target (0).

  Returns:
    class_targets: A int64 tensor for class targets. Shape = [num_classes].
    class_weights: A float32 tensor for class weights. Shape = [num_classes].

  Raises:
    ValueError: If `label_ids` or `label_confidences` is not 1D tensor.
  """
  if len(label_ids.shape.as_list()) != 1:
    raise ValueError('Label id tensor must be 1D: %d.' %
                     len(label_ids.shape.as_list()))
  if len(label_confidences.shape.as_list()) != 1:
    raise ValueError('Label confidence tensor must be 1D: %d.' %
                     len(label_confidences.shape.as_list()))

  if isinstance(label_ids, tf.SparseTensor):
    label_ids = tf.sparse.to_dense(label_ids)
  if isinstance(label_confidences, tf.SparseTensor):
    label_confidences = tf.sparse.to_dense(label_confidences)
  positive_label_id_masks = tf.math.greater(
      label_confidences, positive_label_confidence_threshold)
  positive_label_ids = tf.boolean_mask(label_ids, mask=positive_label_id_masks)
  class_targets = tf.math.reduce_sum(
      tf.one_hot(positive_label_ids, num_classes, dtype=tf.int64), axis=0)
  class_weights = tf.math.reduce_sum(
      tf.one_hot(label_ids, num_classes, dtype=tf.float32), axis=0)
  return class_targets, class_weights


def process_class_labels(decoded_tensors,
                         num_classes,
                         num_objects,
                         common_module=common):
  """Processes decoded class labels and confidences into targets and weights.

  IMPORTANT: Note that we assume there is one copy of label ids and confidences
  in each record and they apply to all the objects in the record.

  Args:
    decoded_tensors: A dictionary for decoded tensors.
    num_classes: An integer for total number of classification label classes to
      read labels for.
    num_objects: An integer for the number of objects each example has.
    common_module: A Python module that defines common constants.

  Returns:
    outputs: A dictionary for processed 2D keypoint tensors.
  """
  class_targets, class_weights = (
      generate_class_targets(
          decoded_tensors[common_module.TFE_KEY_CLASS_LABEL_ID],
          decoded_tensors[common_module.TFE_KEY_CLASS_LABEL_CONFIDENCE],
          num_classes=num_classes))

  # Stack the same class targets and weights for multiple objects.
  class_targets = tf.stack([class_targets for i in range(num_objects)], axis=0)
  class_weights = tf.stack([class_weights for i in range(num_objects)], axis=0)

  outputs = {
      common_module.KEY_CLASS_TARGETS: class_targets,
      common_module.KEY_CLASS_WEIGHTS: class_weights,
  }
  return outputs


def get_tfe_parser_fn(decoders, post_process_fn):
  """Creates a tf.Example parser function.

  Args:
    decoders: A dictionary for keyed tf.Example field decoders.
    post_process_fn: A function handle for postprocessing decoded tensors.

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
                      feature_dim=None,
                      num_classes=None,
                      num_objects=1,
                      sequence_length=None,
                      common_module=common):
  """Creates tf.Example parser function.

  IMPORTANT: Currently only supports all objects having the same class label
  information, and the class label related fields in tf.Examples are expected to
  only have values for one object.

  Args:
    keypoint_names_2d: A list of strings for 2D keypoint names. Use None to skip
      reading 2D keypoints.
    keypoint_names_3d: A list of strings for 3D keypoint names. Use None to skip
      reading 3D keypoints.
    include_keypoint_scores_2d: A boolean for whether to read 2D keypoint
      scores. Only used if `keypoint_names_2d` is specified.
    include_keypoint_scores_3d: A boolean for whether to read 3D keypoint
      scores. Only used if `keypoint_names_3d` is specified.
    feature_dim: An integer for size of pre-computed feature vectors. Only reads
      features if specified.
    num_classes: An integer for the number of classification label classes to
      read labels for. Only reads labels if specified.
    num_objects: An integer for the number of objects each example has.
    sequence_length: An integer for the length of sequence per object each
      example has. Skips adding the sequence dimension if None.
    common_module: A Python module that defines common constants.

  Returns:
    parser_fn: A function handle for the parser.
  """
  instance_shape = ([num_objects] if sequence_length is None else
                    [num_objects, sequence_length])

  decoders = add_decoder_image_sizes(
      instance_shape=instance_shape, common_module=common_module)

  if keypoint_names_2d:
    decoders.update(
        add_decoder_keypoints_2d(
            keypoint_names_2d,
            include_keypoint_scores_2d=include_keypoint_scores_2d,
            instance_shape=instance_shape,
            common_module=common_module))

  if keypoint_names_3d:
    decoders.update(
        add_decoder_keypoints_3d(
            keypoint_names_3d,
            include_keypoint_scores_3d=include_keypoint_scores_3d,
            instance_shape=instance_shape,
            common_module=common_module))

  if feature_dim:
    decoders.update(
        add_decoder_features(
            feature_dim=feature_dim,
            instance_shape=instance_shape,
            common_module=common_module))

  if num_classes:
    decoders.update(add_decoder_class_labels(common_module=common_module))

  def post_process_decoded_tensors(decoded_tensors):
    """Postprocesses decoded tensors."""
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

    if feature_dim:
      outputs.update(
          process_decoded_features(
              decoded_tensors, common_module=common_module))

    if num_classes:
      outputs.update(
          process_class_labels(
              decoded_tensors,
              num_classes=num_classes,
              num_objects=num_objects,
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
