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

"""Pipeline utility functions."""

import json
import os
import sys

from absl import logging
import tensorflow.compat.v1 as tf
import tf_slim

from poem.core import common
from poem.core import keypoint_utils
from poem.core import tfe_input_layer


def read_batch_from_dataset_tables(input_table_patterns,
                                   batch_sizes,
                                   num_instances_per_record,
                                   shuffle,
                                   num_epochs,
                                   keypoint_names_3d=None,
                                   keypoint_names_2d=None,
                                   min_keypoint_score_2d=-1.0,
                                   shuffle_buffer_size=4096,
                                   num_shards=1,
                                   shard_index=None,
                                   common_module=common,
                                   dataset_class=tf.data.TFRecordDataset,
                                   input_example_parser_creator=None,
                                   seed=None):
  """Reads data from dataset table.

  IMPORTANT: We assume that 2D keypoints from the input have been normalized by
  image size. This function will reads image sizes from the input and
  denormalize the 2D keypoints with them. No normalization is expected and no
  denormalization will be performed for 3D keypoints.

  Output tensors may include:
    keypoints: A tensor for standardized 2D keypoints. Shape = [batch_size,
      num_instances_per_record, num_keypoints_2d, 2].
    keypoint_scores: A tensor for 2D keypoint scores. Shape = [batch_size,
      num_instances_per_record, num_keypoints_2d].
    keypoints_3d: A tensor for standardized 3D keypoints. Shape = [batch_size,
      num_instances_per_record, num_keypoints_3d, 3].

  Args:
    input_table_patterns: A list of strings for the paths or pattern to input
      tables.
    batch_sizes: A list of integers for the batch sizes to read from each table.
    num_instances_per_record: An integer for the number of instances per
      tf.Example record.
    shuffle: A boolean for whether to shuffle batch.
    num_epochs: An integer for the number of epochs to read. Use None to read
      indefinitely, in which case remainder batch will be dropped.
    keypoint_names_3d: A list of strings for 3D keypoint names to read
      (coordinates). Use None to skip reading 2D keypoints.
    keypoint_names_2d: A list of strings for 2D keypoint names to read
      (coordinates and scores). Use None to skip reading 2D keypoints.
    min_keypoint_score_2d: A float for the minimum score to consider a 2D
      keypoint as invalid.
    shuffle_buffer_size: An integer for the buffer size used for shuffling. A
      large buffer size benefits shuffling quality.
    num_shards: An integer for the number of shards to divide the dataset. This
      is useful to distributed training. See `tf.data.Dataset.shard` for
      details.
    shard_index: An integer for the shard index to use. This is useful to
      distributed training, and should usually be set to the id of a
      synchronized worker. See `tf.data.Dataset.shard` for details. Note this
      must be specified if `num_shards` is greater than 1.
    common_module: A Python module that defines common constants.
    dataset_class: A dataset class to use. Must match input table type.
    input_example_parser_creator: A function handle for creating parser
      function. If None, uses the default parser creator.
    seed: An integer for random seed.

  Returns:
    outputs: A dictionary for output tensor inputs.
  """
  parser_kwargs = {
      'num_objects': num_instances_per_record,
  }

  if keypoint_names_3d:
    parser_kwargs.update({
        'keypoint_names_3d': keypoint_names_3d,
        'include_keypoint_scores_3d': False,
    })

  if keypoint_names_2d:
    parser_kwargs.update({
        'keypoint_names_2d': keypoint_names_2d,
        'include_keypoint_scores_2d': True,
    })

  if input_example_parser_creator is None:
    input_example_parser_creator = tfe_input_layer.create_tfe_parser
  parser_fn = input_example_parser_creator(
      common_module=common_module, **parser_kwargs)

  # TODO(lzyuan): consider to refactor read_batch_from_batches into other file.
  outputs = tfe_input_layer.read_batch_from_tables(
      input_table_patterns,
      batch_sizes=batch_sizes,
      drop_remainder=num_epochs is None,
      num_epochs=num_epochs,
      num_shards=num_shards,
      shard_index=shard_index,
      shuffle=shuffle,
      shuffle_buffer_size=shuffle_buffer_size,
      dataset_class=dataset_class,
      parser_fn=parser_fn,
      seed=seed)
  outputs = tf.data.make_one_shot_iterator(outputs).get_next()

  if keypoint_names_2d:
    # Since we assume 2D keypoints from the input have been normalized by image
    # size, so we need to denormalize them to restore correctly aspect ratio.
    keypoints_2d = keypoint_utils.denormalize_points_by_image_size(
        outputs[common_module.KEY_KEYPOINTS_2D],
        image_sizes=outputs[common_module.KEY_IMAGE_SIZES])

    keypoint_scores_2d = outputs[common_module.KEY_KEYPOINT_SCORES_2D]
    if min_keypoint_score_2d < 0.0:
      keypoint_masks_2d = tf.ones_like(keypoint_scores_2d, dtype=tf.float32)
    else:
      keypoint_masks_2d = tf.cast(
          tf.math.greater_equal(keypoint_scores_2d, min_keypoint_score_2d),
          dtype=tf.float32)

    outputs.update({
        common_module.KEY_KEYPOINTS_2D: keypoints_2d,
        common_module.KEY_KEYPOINT_MASKS_2D: keypoint_masks_2d
    })

  return outputs


def get_learning_rate(schedule_type,
                      init_learning_rate,
                      global_step=None,
                      **kwargs):
  """Creates learning rate with schedules.

  Currently supported schedules include:
    'EXP_DECAY'

  Args:
    schedule_type: A string for the type of learning rate schedule to choose.
    init_learning_rate: A float or tensor for the learning rate.
    global_step: A tensor for the global step. If None, uses default value.
    **kwargs: A dictionary of assorted arguments used by learning rate
      schedulers, keyed in the format of '${schedule_type}_${arg}'.

  Returns:
    learning_rate: A learning rate tensor.

  Raises:
    ValueError: If the schedule type is not supported.
  """
  if schedule_type == 'EXP_DECAY':
    if global_step is None:
      global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        init_learning_rate,
        global_step=global_step,
        decay_steps=kwargs.get('EXP_DECAY_decay_steps'),
        decay_rate=kwargs.get('EXP_DECAY_decay_rate'),
        staircase=kwargs.get('EXP_DECAY_staircase', False))
  else:
    raise ValueError('Unsupported optimizer type: `%s`.' % str(schedule_type))

  return learning_rate


def get_optimizer(optimizer_type, learning_rate, **kwargs):
  """Creates optimizer with learning rate.

  Currently supported optimizers include:
    'ADAGRAD'

  Args:
    optimizer_type: A string for the type of optimizer to choose.
    learning_rate: A float or tensor for the learning rate.
    **kwargs: A dictionary of assorted arguments used by optimizers, keyed in
      the format of '${optimizer_type}_${arg}'.

  Returns:
    optimizer: An optimizer class object.

  Raises:
    ValueError: If the optimizer type is not supported.
  """
  if optimizer_type == 'ADAGRAD':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=kwargs.get(
            'ADAGRAD_initial_accumulator_value', 0.1))
  elif optimizer_type == 'ADAM':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=kwargs.get('ADAM_beta1', 0.9),
        beta2=kwargs.get('ADAM_beta2', 0.999),
        epsilon=kwargs.get('ADAM_epsilon', 1e-8))
  elif optimizer_type == 'RMSPROP':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate,
        decay=kwargs.get('RMSPROP_decay', 0.9),
        momentum=kwargs.get('RMSPROP_momentum', 0.9),
        epsilon=kwargs.get('RMSPROP_epsilon', 1e-10))
  else:
    raise ValueError('Unsupported optimizer type: `%s`.' % str(optimizer_type))

  return optimizer


def add_moving_average(decay):
  """Sets up exponential moving averages for training.

  Args:
    decay: A float as the moving average decay factor.

  Returns:
    train_op: An update training op object.
  """
  variables_to_average = tf.trainable_variables()
  variable_averages = tf.train.ExponentialMovingAverage(
      decay, num_updates=tf.train.get_or_create_global_step())

  train_op = variable_averages.apply(variables_to_average)
  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_op)

  return train_op


def get_moving_average_variables_to_restore(global_step=None):
  """Gets variables to restore.

  Args:
    global_step: A tensor of global step to include. If None, do not restore
      global step variable, which is for exporting inference graph. For only
      evaluation, specifying a global step is needed.

  Returns:
    variables_to_restore: A dictionary of variables to restore.
  """
  variable_averages = tf.train.ExponentialMovingAverage(0.0, global_step)
  variables_to_restore = variable_averages.variables_to_restore()

  if global_step is not None:
    variables_to_restore[global_step.op.name] = global_step

  return variables_to_restore


def get_init_fn(train_dir=None,
                model_checkpoint=None,
                exclude_list=None,
                include_list=None,
                reset_global_step_if_necessary=True,
                ignore_missing_vars=True):
  """Gets model initializer function.

  The initialization logic is as follows:
    1. If a checkpoint is found in `train_dir`, initialize from it.
    2. Otherwise, if `model_checkpoint` is valid, initialize from it, and reset
       global step if necessary.
    3. Otherwise, do not initialize from any checkpoint.

  Args:
    train_dir: A string as the path to an existing training directory to resume.
      Use None to skip.
    model_checkpoint: A string as the path to an existing model checkpoint to
      initialize from. Use None to skip.
    exclude_list: A list of strings for the names of variables not to load.
    include_list: A list of strings for the names of variables to load. Use
      None to load all variables.
    reset_global_step_if_necessary: A boolean for whether to reset global step.
      Only used in the case of initializing from an existing checkpoint
      `model_checkpoint` rather than resuming training from `train_dir`.
    ignore_missing_vars: A boolean for whether to ignore missing variables. If
      False, errors will be raised if there is a missing variable.

  Returns:
    An model initializer function if an existing checkpoint is found. None
      otherwise.
  """
  # Make sure the exclude list is a list.
  if not exclude_list:
    exclude_list = []

  if train_dir:
    train_checkpoint = tf.train.latest_checkpoint(train_dir)
    if train_checkpoint:
      model_checkpoint = train_checkpoint
      logging.info('Resume latest training checkpoint in: %s.', train_dir)
    elif model_checkpoint:
      logging.info('Use initial checkpoint: %s.', model_checkpoint)
      if reset_global_step_if_necessary:
        exclude_list.append('global_step')
        logging.info('Reset global step.')
  elif model_checkpoint:
    logging.info('Use initial checkpoint: %s.', model_checkpoint)
    if reset_global_step_if_necessary:
      exclude_list.append('global_step')
      logging.info('Reset global step.')

  if not model_checkpoint:
    logging.info('Do not initialize from a checkpoint.')
    return None

  variables_to_restore = tf_slim.get_variables_to_restore(
      include=include_list, exclude=exclude_list)

  return tf_slim.assign_from_checkpoint_fn(
      model_checkpoint,
      variables_to_restore,
      ignore_missing_vars=ignore_missing_vars)


def add_summary(scalars_to_summarize=None,
                histograms_to_summarize=None,
                images_to_summarize=None):
  """Adds summaries to the default summary collection.

  Args:
    scalars_to_summarize: A dictionary of (name, scalar tensor) tuples to
      summarize.
    histograms_to_summarize: A dictionary of (name, histogram tensor) tuples to
      summarize.
    images_to_summarize: A dictionary of (name, image tensor) tuples to
      summarize.
  """
  if scalars_to_summarize:
    for key, value in scalars_to_summarize.items():
      tf.summary.scalar(key, value)

  if histograms_to_summarize:
    for key, value in histograms_to_summarize.items():
      tf.summary.histogram(key, value)

  if images_to_summarize:
    for key, value in images_to_summarize.items():
      tf.summary.image(key, value)


def profile(graph=None, variables=None):
  """Profiles model sizes and computation.

  Args:
    graph: A Tensorflow Graph to profile. If None, use the default graph.
    variables: A list of model variables to profile. If None, use the default
      model variable list.
  """
  if graph is None:
    graph = tf.get_default_graph()
  tf_slim.model_analyzer.analyze_ops(graph, print_info=True)

  if variables is None:
    variables = tf.model_variables()
  tf_slim.model_analyzer.analyze_vars(variables, print_info=True)


def create_dir_and_save_flags(flags_module, log_dir, json_filename):
  """Creates log directory and saves flags to a JSON file.

  Args:
    flags_module: An absl.flags module.
    log_dir: A string for log directory.
    json_filename: A string for output JSON file name.
  """
  # Create log directory if necessary.
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)

  # Save all key flags.
  key_flag_dict = {
      flag.name: flag.value
      for flag in flags_module.FLAGS.get_key_flags_for_module(sys.argv[0])
  }
  json_path = os.path.join(log_dir, json_filename)
  with tf.io.gfile.GFile(json_path, 'w') as f:
    json.dump(key_flag_dict, f, indent=2, sort_keys=True)


def get_embedding_keys(distance_type,
                       replace_samples_with_means=False,
                       common_module=common):
  """Gets model embedding output keys based on distance type.

  Args:
    distance_type: An enum string for distance type.
    replace_samples_with_means: A boolean for whether to replace embedding
      sample keys with embedding mean keys.
    common_module: A Python module that defines common flags and constants.

  Returns:
    A list for enum strings for model embedding output keys.
  """
  if distance_type == common_module.DISTANCE_TYPE_CENTER:
    return [common_module.KEY_EMBEDDING_MEANS]
  if distance_type == common_module.DISTANCE_TYPE_SAMPLE:
    return [common_module.KEY_EMBEDDING_SAMPLES]
  # distance_type == common_module.DISTANCE_TYPE_CENTER_AND_SAMPLE.
  return [
      common_module.KEY_EMBEDDING_MEANS, common_module.KEY_EMBEDDING_STDDEVS,
      (common_module.KEY_EMBEDDING_MEANS
       if replace_samples_with_means else common_module.KEY_EMBEDDING_SAMPLES)
  ]


def stack_embeddings(model_outputs, embedding_keys, common_module=common):
  """Selects and stacks embeddings by key.

  Args:
    model_outputs: A dictionary for model output tensors.
    embedding_keys: A list for enum strings for tensor keys to select.
    common_module: A Python module that defines common flags and constants.

  Returns:
    A tensor for stacked embeddings. Shape = [..., num_embeddings_per_instance,
      embedding_dim].
  """
  embeddings_to_stack = []
  for key in embedding_keys:
    if key in [
        common_module.KEY_EMBEDDING_MEANS, common_module.KEY_EMBEDDING_STDDEVS
    ]:
      embeddings_to_stack.append(tf.expand_dims(model_outputs[key], axis=-2))
    elif key == common_module.KEY_EMBEDDING_SAMPLES:
      embeddings_to_stack.append(model_outputs[key])
    else:
      raise ValueError('Unsupported embedding key: `%s`.' % str(key))
  return tf.concat(embeddings_to_stack, axis=-2)


def get_sigmoid_parameters(name,
                           raw_a_initial_value=0.0,
                           b_initial_value=0.0,
                           a_range=(None, None),
                           b_range=(None, None),
                           reuse=tf.AUTO_REUSE):
  """Gets sigmoid parameter variables.

  Args:
    name: A string for the variable scope name.
    raw_a_initial_value: A float for initial value of the raw `a` parameter.
    b_initial_value: A float for initial value of the `b` parameter.
    a_range: A tuple of (min, max) range of `a` parameter. Uses None or
      non-positive value to indicate unspecified boundaries.
    b_range: A tuple of (min, max) range of `b` parameter. Uses None to indicate
      unspecified boundaries. Does NOT use non-positive value to indicate
      unspecified boundaries.
    reuse: Type of variable reuse.

  Returns:
    raw_a: A variable for `raw_a` parameter.
    a: A tensor for `a` parameter.
    b: A tensor for `b` parameter.

  Raises:
    ValueError: If `a_range` or `b_range` is invalid.
  """

  def maybe_clamp(x, x_range, ignored_if_non_positive):
    """Clamps `x` to `x_range`."""
    x_min, x_max = x_range
    if x_min is not None and x_max is not None and x_min > x_max:
      raise ValueError('Invalid range: %s.' % str(x_range))
    if (x_min is not None) and (not ignored_if_non_positive or x_min > 0.0):
      x = tf.math.maximum(x_min, x)
    if (x_max is not None) and (not ignored_if_non_positive or x_max > 0.0):
      x = tf.math.minimum(x_max, x)
    return x

  with tf.variable_scope(name, reuse=reuse):
    # TODO(liuti): Currently the variable for `raw_a` is named `a` in
    # checkpoints for historic reasons. Consolidate the naming.
    raw_a = tf.get_variable(
        'a',
        shape=[],
        dtype=tf.float32,
        initializer=tf.initializers.constant(raw_a_initial_value))
    a = tf.nn.elu(raw_a) + 1.0
    a = maybe_clamp(a, a_range, ignored_if_non_positive=True)

    b = tf.get_variable(
        'b',
        shape=[],
        dtype=tf.float32,
        initializer=tf.initializers.constant(b_initial_value))
    b = maybe_clamp(b, b_range, ignored_if_non_positive=False)

  return raw_a, a, b
