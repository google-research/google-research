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
from deeplab import train_utils
import model
import model_input
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import app

flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('save_interval_secs', 600,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summary_steps', 200, '')

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

flags.DEFINE_float(
    'last_layer_gradient_multiplier', 1.0,
    'The gradient multiplier for last layers, which is used to '
    'boost the gradient of last layers if the value > 1.')

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
    logits_summary(outputs_to_scales_to_logits[output]['merged_logits'])
    train_utils.add_softmax_cross_entropy_loss_for_each_scale(
        outputs_to_scales_to_logits[output],
        samples['label'],
        num_classes,
        ignore_label,
        loss_weight=1.0,
        upsample_logits=FLAGS.upsample_logits,
        scope=output)

  return outputs_to_scales_to_logits


def model_fn(features, labels, mode, params):
  """Defines the model compatible with tf.estimator."""
  del labels, params
  if mode == tf.estimator.ModeKeys.TRAIN:
    _build_deeplab(features, {
        'semantic': model_input.dataset_descriptors[FLAGS.dataset].num_classes
    }, model_input.dataset_descriptors[FLAGS.dataset].ignore_label)

    loss = tf.losses.get_total_loss()

    learning_rate = train_utils.get_model_learning_rate(
        FLAGS.learning_policy, FLAGS.base_learning_rate,
        FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
        FLAGS.training_number_of_steps, FLAGS.learning_power,
        FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    grads_and_vars = optimizer.compute_gradients(loss)
    last_layers = model.get_extra_layer_scopes()
    grad_mult = train_utils.get_model_gradient_multipliers(
        last_layers, FLAGS.last_layer_gradient_multiplier)
    if grad_mult:
      grads_and_vars = tf.contrib.slim.learning.multiply_gradients(
          grads_and_vars, grad_mult)

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
