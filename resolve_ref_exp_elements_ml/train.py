# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

r"""Training script for the VIS model.

See model.py for more details and usage.
"""
import os
import common  # pylint: disable=unused-import
from deeplab import preprocess_utils
from deeplab import train_utils
import model
import model_input
import six
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import app

ZERO_DIV_OFFSET = 1e-20

flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('save_interval_secs', 60,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summary_steps', 100, '')

# Settings for training strategry.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step', 'cosine'],
                  'Learning rate policy for training.')

flags.DEFINE_float('base_learning_rate', 3e-5,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer(
    'learning_rate_decay_step', 8000,
    'Decay the base learning rate at a fixed step.'
    'Not used if learning_policy == "poly"')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 1000000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

flags.DEFINE_integer('batch_size', 8,
                     'The number of images in each batch during training.')

flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_integer('resize_factor', None,
                     'Resized dimensions are multiple of factor plus one.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', '',
                    'The initial checkpoint in tensorflow format.')

flags.DEFINE_boolean('initialize_last_layer', False,
                     'Initialize the last layer.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_string('split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_bool('debug', False, 'Whether to use tf dbg.')

flags.DEFINE_boolean('profile', False, '')

flags.DEFINE_boolean('use_sigmoid', True,
                     'Use the custom sigmoid cross entropy function')

flags.DEFINE_float('sigmoid_recall_weight', 5,
                   'If <1 value precision, if >1 recall')

flags.DEFINE_enum(
    'distance_metric', 'euclidean_iter',
    ['mse', 'euclidean', 'euclidean_sqrt', 'euclidean_iter'],
    'the cost metric for the Click Regression'
    '"mse" for mean squared error'
    '"euclidean" for euclidean distance'
    '"euclidean_sqrt" for square root of euclidean distance')

flags.DEFINE_bool('ratio_box_distance', False,
                  'normalize the distance loss by the size of the box')

flags.DEFINE_integer('euclidean_step', 300000,
                     'decrease exponent of distance loss every euclidean_step')


def logits_summary(logits):
  if model_input.dataset_descriptors[FLAGS.dataset].num_classes == 2:
    logits_for_sum = tf.concat([logits, tf.zeros_like(logits[:, :, :, 0:1])], 3)
  else:
    logits_for_sum = logits

  tf.summary.image('logits', logits_for_sum, 4)
  resized = tf.image.resize_bilinear(
      logits_for_sum, [FLAGS.image_size, FLAGS.image_size], align_corners=True)
  tf.summary.image('resized_logits', resized, 4)


def label_summary(labels):
  labels = tf.clip_by_value(labels, 0, 3) * int(255 / 3)
  tf.summary.image('label', tf.cast(labels, tf.uint8), 4)


def add_cross_entropy_loss(labels, logits, add_loss):
  """Adds accuracy summary. Adds the loss if add_loss is true."""
  if add_loss:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss, name='selected_loss')
    tf.losses.add_loss(loss)

  pred = tf.argmax(logits, 1)
  correct = tf.equal(pred, labels)
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  tf.summary.scalar('selected_accuracy', accuracy)


def add_sigmoid_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  scope=None):
  """Adds sigmoid cross entropy loss for logits of each scale.

  Implemented based on deeplab's add_softmax_cross_entropy_loss_for_each_scale
  in deeplab/utils/train_utils.py.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    ignore_label: Integer, label to ignore.
    loss_weight: Float, loss weight.
    upsample_logits: Boolean, upsample logits or not.
    scope: String, the scope for the loss.

  Raises:
    ValueError: Label or logits is None.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize_bilinear(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True)

    logits = logits[:, :, :, 1]
    scaled_labels = tf.to_float(scaled_labels)
    scaled_labels = tf.squeeze(scaled_labels)
    not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                               ignore_label)) * loss_weight
    losses = tf.nn.weighted_cross_entropy_with_logits(
        scaled_labels, logits, FLAGS.sigmoid_recall_weight)

    # Loss added later in model_fn by tf.losses.get_total_loss()
    tf.losses.compute_weighted_loss(
        losses, weights=not_ignore_mask, scope=loss_scope)


def add_distance_loss_to_center(labels, logits, groundtruth_coords):
  """Add distance loss function for ClickRegression."""
  weights = tf.to_int32(
      tf.not_equal(labels,
                   model_input.dataset_descriptors[FLAGS.dataset].ignore_label))
  labels *= weights

  # Use GT box to get center if it exists. Less computation required.
  # Otherwise, calculate from label mask.
  if FLAGS.use_groundtruth_box:
    center_x = (groundtruth_coords['xmin'] + groundtruth_coords['xmax']) / 2.0
    center_y = (groundtruth_coords['ymin'] + groundtruth_coords['ymax']) / 2.0
    center = tf.stack([center_y, center_x], axis=1)
  else:
    # Make array of coordinates (each row contains three coordinates)
    ii, jj = tf.meshgrid(
        tf.range(FLAGS.image_size), tf.range(FLAGS.image_size), indexing='ij')
    coords = tf.stack([tf.reshape(ii, (-1,)), tf.reshape(jj, (-1,))], axis=-1)
    coords = tf.cast(coords, tf.int32)

    # Rearrange input into one vector per volume
    volumes_flat = tf.reshape(labels,
                              [-1, FLAGS.image_size * FLAGS.image_size * 1, 1])
    # Compute total mass for each volume. Add 0.00001 to prevent division by 0
    total_mass = tf.cast(tf.reduce_sum(volumes_flat, axis=1),
                         tf.float32) + ZERO_DIV_OFFSET
    # Compute centre of mass
    center = tf.cast(tf.reduce_sum(volumes_flat * coords, axis=1),
                     tf.float32) / total_mass
    center = center / FLAGS.image_size

  # Normalize coordinates by size of image
  logits = logits / FLAGS.image_size

  # Calculate loss based on the distance metric specified
  # Loss added later in model_fn by tf.losses.get_total_loss()
  if FLAGS.distance_metric == 'mse':
    tf.losses.mean_squared_error(center, logits)
  elif FLAGS.distance_metric in [
      'euclidean', 'euclidean_sqrt', 'euclidean_iter'
  ]:
    distance_to_center = tf.sqrt(
        tf.reduce_sum(tf.square(logits - center), axis=-1) + ZERO_DIV_OFFSET)
    if FLAGS.ratio_box_distance:
      distance_to_box = calc_distance_to_edge(groundtruth_coords, logits)
      box_distance_to_center = (
          tf.to_float(distance_to_center) - distance_to_box)
      loss = distance_to_center / (box_distance_to_center + ZERO_DIV_OFFSET)
    else:
      loss = distance_to_center

    if FLAGS.distance_metric == 'euclidean_sqrt':
      loss = tf.sqrt(loss)
    if FLAGS.distance_metric == 'euclidean_iter':
      iter_num = tf.to_float(tf.train.get_or_create_global_step())
      step = (iter_num // FLAGS.euclidean_step) + 1.0
      loss = tf.pow(loss, tf.to_float(1.0 / step))
    tf.losses.compute_weighted_loss(loss)


def calc_distance_to_edge(groundtruth_coords, logits):
  """Calculate distance between predicted point to box of ground truth."""

  # Returns 0 if predicted point is inside the groundtruth box
  dx = tf.maximum(
      tf.maximum(groundtruth_coords['xmin'] - logits[:, 1],
                 logits[:, 1] - groundtruth_coords['xmax']), 0)
  dy = tf.maximum(
      tf.maximum(groundtruth_coords['ymin'] - logits[:, 0],
                 logits[:, 0] - groundtruth_coords['ymax']), 0)

  distance = tf.sqrt(tf.square(dx) + tf.square(dy))
  return distance


def add_distance_loss_to_edge(groundtruth_coords, logits):
  distance = calc_distance_to_edge(groundtruth_coords, logits)
  tf.losses.compute_weighted_loss(distance)


def _build_deeplab(samples, outputs_to_num_classes, ignore_label):
  """Builds a clone of DeepLab.

  Args:
    samples: Feature map from input pipeline.
    outputs_to_num_classes: A map from output type to the number of classes.
      For example, for the task of semantic segmentation with 21 semantic
      classes, we would have outputs_to_num_classes['semantic'] = 21.
    ignore_label: Ignore label.

  Returns:
    A map of maps from output_type (e.g., semantic prediction) to a
      dictionary of multi-scale logits names to logits. For each output_type,
      the dictionary has keys which correspond to the scales and values which
      correspond to the logits. For example, if `scales` equals [1.0, 1.5],
      then the keys would include 'merged_logits', 'logits_1.00' and
      'logits_1.50'.
  """

  tf.summary.image('image', samples['image'], 4)
  if 'label' in samples:
    label_summary(samples['label'])
  if FLAGS.use_ref_exp:
    tf.summary.text('ref', samples[model_input.REF_EXP_ID])

  outputs_to_scales_to_logits = model.multi_scale_logits(
      samples['image'],
      samples,
      FLAGS,
      outputs_to_num_classes=outputs_to_num_classes,
      image_pyramid=FLAGS.image_pyramid,
      merge_method=FLAGS.merge_method,
      atrous_rates=FLAGS.atrous_rates,
      add_image_level_feature=FLAGS.add_image_level_feature,
      aspp_with_batch_norm=FLAGS.aspp_with_batch_norm,
      aspp_with_separable_conv=FLAGS.aspp_with_separable_conv,
      multi_grid=FLAGS.multi_grid,
      depth_multiplier=FLAGS.depth_multiplier,
      output_stride=FLAGS.output_stride,
      decoder_output_stride=FLAGS.decoder_output_stride,
      decoder_use_separable_conv=FLAGS.decoder_use_separable_conv,
      logits_kernel_size=FLAGS.logits_kernel_size,
      crop_size=[FLAGS.image_size, FLAGS.image_size],
      model_variant=FLAGS.model_variant,
      weight_decay=FLAGS.weight_decay,
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)

  for output, num_classes in outputs_to_num_classes.iteritems():
    if output == 'segment':
      logits_summary(outputs_to_scales_to_logits[output]['merged_logits'])
      if FLAGS.use_sigmoid:
        add_sigmoid_cross_entropy_loss_for_each_scale(
            outputs_to_scales_to_logits[output], samples['label'], ignore_label,
            1.0, FLAGS.upsample_logits, output)
      else:
        train_utils.add_softmax_cross_entropy_loss_for_each_scale(
            outputs_to_scales_to_logits[output],
            samples['label'],
            num_classes,
            ignore_label,
            loss_weight=1.0,
            upsample_logits=FLAGS.upsample_logits,
            scope=output)

    elif output == 'regression':
      for _, logits in outputs_to_scales_to_logits[output].iteritems():
        groundtruth_box = {
            'xmin': samples[model_input.GROUNDTRUTH_XMIN_ID],
            'xmax': samples[model_input.GROUNDTRUTH_XMAX_ID],
            'ymin': samples[model_input.GROUNDTRUTH_YMIN_ID],
            'ymax': samples[model_input.GROUNDTRUTH_YMAX_ID]
        }
        add_distance_loss_to_center(samples['label'], logits, groundtruth_box)
  return outputs_to_scales_to_logits


def model_fn(features, labels, mode, params):
  """Defines the model compatible with tf.estimator."""
  del labels, params
  if mode == tf.estimator.ModeKeys.TRAIN:
    _build_deeplab(features, model.get_output_to_num_classes(FLAGS),
                   model_input.dataset_descriptors[FLAGS.dataset].ignore_label)

    #  Print out the objective loss and regularization loss independently to
    #  track NaN loss issue
    objective_losses = tf.losses.get_losses()
    objective_losses = tf.Print(
        objective_losses, [objective_losses],
        message='Objective Losses: ',
        summarize=100)
    objective_loss = tf.reduce_sum(objective_losses)
    tf.summary.scalar('objective_loss', objective_loss)

    reg_losses = tf.losses.get_regularization_losses()
    reg_losses = tf.Print(
        reg_losses, [reg_losses], message='Reg Losses: ', summarize=100)
    reg_loss = tf.reduce_sum(reg_losses)
    tf.summary.scalar('regularization_loss', reg_loss)

    loss = objective_loss + reg_loss

    learning_rate = train_utils.get_model_learning_rate(
        FLAGS.learning_policy, FLAGS.base_learning_rate,
        FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
        FLAGS.training_number_of_steps, FLAGS.learning_power,
        FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    grads_and_vars = optimizer.compute_gradients(loss)
    grad_updates = optimizer.apply_gradients(grads_and_vars,
                                             tf.train.get_global_step())
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_op = tf.identity(loss, name='train_op')

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
    )


def tf_dbg_sess_wrapper(sess):
  if FLAGS.debug:
    print 'DEBUG'
    sess = tf_debug.LocalCLIDebugWrapperSession(
        sess,
        thread_name_filter='MainThread$',
        dump_root=os.path.join(FLAGS.train_logdir, 'tfdbg2'))
    sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
  return sess


def main(unused_argv):
  config = tf.estimator.RunConfig(
      model_dir=FLAGS.train_logdir,
      save_summary_steps=FLAGS.save_summary_steps,
      save_checkpoints_secs=FLAGS.save_interval_secs,
  )

  ws = None
  if FLAGS.tf_initial_checkpoint:
    checkpoint_vars = tf.train.list_variables(FLAGS.tf_initial_checkpoint)
    # Add a ':' so we will only match the specific variable and not others.
    checkpoint_vars = [var[0] + ':' for var in checkpoint_vars]
    checkpoint_vars.remove('global_step:')

    ws = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=FLAGS.tf_initial_checkpoint,
        vars_to_warm_start=checkpoint_vars)

  estimator = tf.estimator.Estimator(
      model_fn, FLAGS.train_logdir, config, warm_start_from=ws)

  with tf.contrib.tfprof.ProfileContext(
      FLAGS.train_logdir, enabled=FLAGS.profile):
    estimator.train(
        model_input.get_input_fn(FLAGS),
        max_steps=FLAGS.training_number_of_steps)


if __name__ == '__main__':
  flags.mark_flag_as_required('train_logdir')
  app.run()
