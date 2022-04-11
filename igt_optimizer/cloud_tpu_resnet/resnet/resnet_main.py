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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Train a ResNet-50 model on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from igt_optimizer import exp_igt_optimizer
from igt_optimizer.cloud_tpu_resnet.hyperparameters import common_hparams_flags
from igt_optimizer.cloud_tpu_resnet.hyperparameters import common_tpu_flags
from igt_optimizer.cloud_tpu_resnet.hyperparameters import hyperparameters
from igt_optimizer.cloud_tpu_resnet.resnet import imagenet_input
from igt_optimizer.cloud_tpu_resnet.resnet import lars_util
from igt_optimizer.cloud_tpu_resnet.resnet import resnet_model
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import summary
# copybara:strip_begin
from tensorflow.contrib.compiler import xla
# copybara:strip_end
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.training.python.training import evaluation
# pylint:disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator
from tensorflow.python.lib.io import file_io
# pylint:enable=g-direct-tensorflow-import

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()

FLAGS = flags.FLAGS

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

flags.DEFINE_string(
    'hparams_file',
    default=None,
    help=('Set of model parameters to override the default mparams.'))

flags.DEFINE_multi_string(
    'hparams',
    default=None,
    help=('This is used to override only the model hyperparameters. It should '
          'not be used to override the other parameters like the tpu specific '
          'flags etc. For example, if experimenting with larger numbers of '
          'train_steps, a possible value is '
          '--hparams=train_steps=28152.'))

flags.DEFINE_string(
    'default_hparams_file',
    default=os.path.join(os.path.dirname(__file__), './configs/default.yaml'),
    help=('Default set of model parameters to use with this model. Look the at '
          'configs/default.yaml for this.'))

flags.DEFINE_integer(
    'resnet_depth',
    default=None,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))

flags.DEFINE_integer(
    'num_train_images', default=None, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=None, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'num_label_classes', default=None, help='Number of classes, at least 2')

flags.DEFINE_string(
    'data_format',
    default=None,
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))

flags.DEFINE_bool(
    'transpose_input',
    default=None,
    help='Use TPU double transpose optimization')

flags.DEFINE_bool(
    'use_cache', default=None, help=('Enable cache for training input.'))

flags.DEFINE_integer('image_size', None, 'The input image size.')

flags.DEFINE_string(
    'dropblock_groups',
    None,
    help=('A string containing comma separated integers indicating ResNet '
          'block groups to apply DropBlock. `3,4` means to apply DropBlock to '
          'block groups 3 and 4. Use an empty string to not apply DropBlock to '
          'any block group.'))
flags.DEFINE_float(
    'dropblock_keep_prob',
    default=None,
    help=('keep_prob parameter of DropBlock. Will not be used if '
          'dropblock_groups is empty.'))
flags.DEFINE_integer(
    'dropblock_size',
    default=None,
    help=('size parameter of DropBlock. Will not be used if dropblock_groups '
          'is empty.'))

# copybara:strip_begin
flags.DEFINE_boolean(
    'xla_compile',
    default=False,
    help=('Compile computation with XLA, this flag has no effect when running '
          'on TPU.'))
flags.DEFINE_string(
    'tpu_job_name', None,
    'Name of TPU worker binary. Only necessary if job name is changed from'
    ' default tpu_worker.')
# copybara:strip_end

flags.DEFINE_integer(
    'profile_every_n_steps',
    default=0,
    help=('Number of steps between collecting profiles if larger than 0'))

flags.DEFINE_string(
    'mode',
    default='train_and_eval',
    help='One of {"train_and_eval", "train", "eval", "eval_igt"}.')

flags.DEFINE_string(
    'igt_eval_mode', default='shift', help='One of {"shift", "true"}.')

flags.DEFINE_string(
    'igt_eval_set', default='train', help='One of {"train", "eval"}.')

flags.DEFINE_integer(
    'steps_per_eval',
    default=1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_integer(
    'num_parallel_calls',
    default=None,
    help=('Number of parallel threads in CPU for the input pipeline.'
          ' Recommended value is the number of cores per CPU host.'))

flags.DEFINE_integer(
    'num_cores',
    default=None,
    help=('Number of TPU cores in total. For a single TPU device, this is 8'
          ' because each TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string(
    'bigtable_project', None,
    'The Cloud Bigtable project.  If None, --gcp_project will be used.')
flags.DEFINE_string('bigtable_instance', None,
                    'The Cloud Bigtable instance to load data from.')
flags.DEFINE_string('bigtable_table', 'imagenet',
                    'The Cloud Bigtable table to load data from.')
flags.DEFINE_string('bigtable_train_prefix', 'train_',
                    'The prefix identifying training rows.')
flags.DEFINE_string('bigtable_eval_prefix', 'validation_',
                    'The prefix identifying evaluation rows.')
flags.DEFINE_string('bigtable_column_family', 'tfexample',
                    'The column family storing TFExamples.')
flags.DEFINE_string('bigtable_column_qualifier', 'example',
                    'The column name storing TFExamples.')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_bool(
    'export_to_tpu',
    default=False,
    help=('Whether to export additional metagraph with "serve, tpu" tags'
          ' in addition to "serve" only metagraph.'))

flags.DEFINE_enum(
    'optimizer',
    default='momentum',
    enum_values=['momentum', 'adam', 'eigt'],
    help=('The optimizer to use.'))

flags.DEFINE_enum(
    'igt_optimizer',
    default='gd',
    enum_values=['gd', 'mom', 'adam'],
    help=('The optimizer to use to apply the IGT update.'))

flags.DEFINE_float(
    'tail_fraction',
    default=2.,
    help=('Fraction of the data to use (e.g. 2. for t/2.).'))

flags.DEFINE_float(
    'base_learning_rate',
    default=None,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float('lr_decay', default=0.1, help='The learning rate decay.')

flags.DEFINE_float(
    'lr_decay_step_fraction',
    default=0.9,
    help='The fraction of steps at which full decay is reached.')

flags.DEFINE_float(
    'momentum',
    default=None,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'weight_decay',
    default=None,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing',
    default=None,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_bool(
    'enable_lars',
    default=None,
    help=('Enable LARS optimizer for large batch training.'))

flags.DEFINE_float(
    'poly_rate', default=None, help=('Set LARS/Poly learning rate.'))

flags.DEFINE_bool(
    'use_async_checkpointing', default=None, help=('Enable async checkpoint'))

flags.DEFINE_integer(
    'log_step_count_steps', 64, 'The number of steps at '
    'which the global step information is logged.')

# Inference configuration.
flags.DEFINE_bool(
    'add_warmup_requests', False,
    'Whether to add warmup requests into the export saved model dir,'
    'especially for TPU inference.')
flags.DEFINE_string('model_name', 'resnet',
                    'Serving model name used for the model server.')
flags.DEFINE_multi_integer(
    'inference_batch_sizes', [8],
    'Known inference batch sizes used to warm up for each core.')

# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def get_model_dir(params):
  """Returns the model directory as a function of the params dict."""
  suffix = 'opt={}'.format(FLAGS.optimizer)
  if FLAGS.optimizer == 'momentum':
    suffix += '_mom={}'.format(params['momentum'])
  if FLAGS.optimizer == 'eigt':
    suffix += '_{}'.format(FLAGS.igt_optimizer)
    suffix += '_tail={}'.format(FLAGS.tail_fraction)

  suffix += '-lr={}'.format(params['base_learning_rate'])
  suffix += '-decay={}-decaystep={}'.format(FLAGS.lr_decay,
                                            FLAGS.lr_decay_step_fraction)

  model_dir = os.path.join(FLAGS.model_dir, suffix)
  print('model_dir: {}'.format(model_dir))
  return model_dir


def get_lr_schedule(train_steps, num_train_images, train_batch_size):
  """learning rate schedule."""
  steps_per_epoch = np.floor(num_train_images / train_batch_size)
  train_epochs = train_steps / steps_per_epoch
  return [  # (multiplier, epoch to start) tuples
      (1.0, np.floor(5 / 90 * train_epochs)),
      (0.1, np.floor(30 / 90 * train_epochs)),
      (0.01, np.floor(60 / 90 * train_epochs)),
      (0.001, np.floor(80 / 90 * train_epochs))
  ]


def learning_rate_schedule(params, current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.

  Args:
    params: Python dict containing parameters for this run.
    current_epoch: `Tensor` for current epoch.

  Returns:
    A scaled `Tensor` for current learning rate.
  """
  scaled_lr = params['base_learning_rate'] * (
      params['train_batch_size'] / 256.0)

  lr_schedule = get_lr_schedule(
      train_steps=params['train_steps'],
      num_train_images=params['num_train_images'],
      train_batch_size=params['train_batch_size'])
  decay_rate = (
      scaled_lr * lr_schedule[0][0] * current_epoch / lr_schedule[0][1])
  for mult, start_epoch in lr_schedule:
    decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
                          scaled_lr * mult)
  return decay_rate


def linear_learning_rate_schedule(params, global_step):
  tf.logging.info(
      'Using a linear LR schedule. step fraction {} decay {}'.format(
          FLAGS.lr_decay_step_fraction, FLAGS.lr_decay))
  initial_lr = params['base_learning_rate'] * (
      params['train_batch_size'] / 256.0)
  fully_decayed_step = params['train_steps'] * FLAGS.lr_decay_step_fraction

  return tf.train.polynomial_decay(
      learning_rate=initial_lr,
      global_step=global_step,
      decay_steps=fully_decayed_step,
      end_learning_rate=initial_lr * FLAGS.lr_decay)


def resnet_model_fn(features, labels, mode, params):
  """The model_fn for ResNet to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images. If transpose_input is enabled, it is
      transposed to device layout and reshaped to 1D tensor.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
      `params['batch_size']` is always provided and should be used as the
      effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    features = features['feature']

  # In most cases, the default data format NCHW instead of NHWC should be
  # used for a significant performance boost on GPU/TPU. NHWC should be used
  # only if the network needs to be run on CPU since the pooling operations
  # are only supported on NHWC.
  if params['data_format'] == 'channels_first':
    assert not params['transpose_input']  # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])

  if params['transpose_input'] and mode != tf_estimator.ModeKeys.PREDICT:
    image_size = tf.sqrt(tf.shape(features)[0] / (3 * tf.shape(labels)[0]))
    features = tf.reshape(features, [image_size, image_size, 3, -1])
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  # Normalize the image to zero mean and unit variance.
  features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
  features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

  # DropBlock keep_prob for the 4 block groups of ResNet architecture.
  # None means applying no DropBlock at the corresponding block group.
  dropblock_keep_probs = [None] * 4
  if params['dropblock_groups']:
    # Scheduled keep_prob for DropBlock.
    train_steps = tf.cast(params['train_steps'], tf.float32)
    current_step = tf.cast(tf.train.get_global_step(), tf.float32)
    current_ratio = current_step / train_steps
    dropblock_keep_prob = (1 - current_ratio *
                           (1 - params['dropblock_keep_prob']))

    # Computes DropBlock keep_prob for different block groups of ResNet.
    dropblock_groups = [int(x) for x in params['dropblock_groups'].split(',')]
    for block_group in dropblock_groups:
      if block_group < 1 or block_group > 4:
        raise ValueError(
            'dropblock_groups should be a comma separated list of integers '
            'between 1 and 4 (dropblcok_groups: {}).'.format(
                params['dropblock_groups']))
      dropblock_keep_probs[block_group - 1] = 1 - (
          (1 - dropblock_keep_prob) / 4.0**(4 - block_group))

  # This nested function allows us to avoid duplicating the logic which
  # builds the network, for different values of --precision.
  def build_network():
    network = resnet_model.resnet_v1(
        resnet_depth=params['resnet_depth'],
        num_classes=params['num_label_classes'],
        dropblock_size=params['dropblock_size'],
        dropblock_keep_probs=dropblock_keep_probs,
        data_format=params['data_format'])
    return network(
        inputs=features, is_training=(mode == tf_estimator.ModeKeys.TRAIN))

  if params['precision'] == 'bfloat16':
    with contrib_tpu.bfloat16_scope():
      logits = build_network()
    logits = tf.cast(logits, tf.float32)
  elif params['precision'] == 'float32':
    logits = build_network()

  if mode == tf_estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf_estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf_estimator.export.PredictOutput(predictions)
        })

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  batch_size = params['batch_size']  # pylint: disable=unused-variable

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, params['num_label_classes'])
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=params['label_smoothing'])

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = cross_entropy + params['weight_decay'] * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.trainable_variables()
      if 'batch_normalization' not in v.name
  ])

  host_call = None
  if mode == tf_estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_global_step()
    steps_per_epoch = params['num_train_images'] / params['train_batch_size']
    current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)
    # LARS is a large batch optimizer. LARS enables higher accuracy at batch 16K
    # and larger batch sizes.
    if params['enable_lars']:
      learning_rate = 0.0
      optimizer = lars_util.init_lars_optimizer(current_epoch, params)
      raise ValueError('LARS unexpected in the context of IGT experiments.')
    else:
      learning_rate = linear_learning_rate_schedule(params, global_step)

      if FLAGS.optimizer == 'momentum':
        tf.logging.info('Using MomentumOptimizer ({}).'.format(
            params['momentum']))
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=params['momentum'],
            use_nesterov=False)

      elif FLAGS.optimizer == 'adam':
        tf.logging.info('Using AdamOptimizer')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

      elif FLAGS.optimizer == 'eigt':
        tf.logging.info('Using ExpIgtOptimizer {} tail: {}'.format(
            FLAGS.igt_optimizer, FLAGS.tail_fraction))
        optimizer = exp_igt_optimizer.ExpIgtOptimizer(
            learning_rate,
            tail_fraction=FLAGS.tail_fraction,
            optimizer=FLAGS.igt_optimizer)

      else:
        raise ValueError('{} is not a supported optimizer'.format(
            FLAGS.optimizer))

    if params['use_tpu']:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = contrib_tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    if not params['skip_host_call']:

      def host_call_fn(gs, loss, lr, ce):
        """Training host call.

        Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed params['iterations_per_loop'] times after
        # one TPU loop is finished, setting max_queue value to the same as
        # number of iterations will make the summary writer only flush the data
        # to storage once per loop.
        with summary.create_file_writer(
            get_model_dir(params),
            max_queue=params['iterations_per_loop']).as_default():
          with summary.always_record_summaries():
            summary.scalar('loss', loss[0], step=gs)
            summary.scalar('learning_rate', lr[0], step=gs)
            summary.scalar('current_epoch', ce[0], step=gs)

            return summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(loss, [1])
      lr_t = tf.reshape(learning_rate, [1])
      ce_t = tf.reshape(current_epoch, [1])

      host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

  else:
    train_op = None

  eval_metrics = None
  scaffold_fn = None
  if mode == tf_estimator.ModeKeys.EVAL:

    def metric_fn(labels, logits):
      """Evaluation metric function.

      Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      return {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

    eval_metrics = (metric_fn, [labels, logits])

    if FLAGS.mode == 'eval_igt' and FLAGS.igt_eval_mode == 'true':
      tf.logging.info('Using true param loading saver.')

      def scaffold_fn_true_params():
        """Returns a scaffold that loads the true values into vars."""
        var_mapping = {}
        trainable_vars = set(tf.trainable_variables())
        for var in tf.global_variables():
          if var in trainable_vars:
            var_mapping[var.op.name + '/true_param'] = var
          else:
            var_mapping[var.op.name] = var

        tf.logging.info('Mapping: {}'.format(var_mapping))
        saver = tf.train.Saver(var_list=var_mapping, sharded=True)
        return tf.train.Scaffold(saver=saver)

      scaffold_fn = scaffold_fn_true_params

  return contrib_tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)


def _verify_non_empty_string(value, field_name):
  """Ensures that a given proposed field value is a non-empty string.

  Args:
    value:  proposed value for the field.
    field_name:  string name of the field, e.g. `project`.

  Returns:
    The given value, provided that it passed the checks.

  Raises:
    ValueError:  the value is not a string, or is a blank string.
  """
  if not isinstance(value, str):
    raise ValueError('Bigtable parameter "%s" must be a string.' % field_name)
  if not value:
    raise ValueError('Bigtable parameter "%s" must be non-empty.' % field_name)
  return value


def _select_tables_from_flags():
  """Construct training and evaluation Bigtable selections from flags.

  Returns:
    [training_selection, evaluation_selection]
  """
  project = _verify_non_empty_string(
      FLAGS.bigtable_project or FLAGS.gcp_project, 'project')
  instance = _verify_non_empty_string(FLAGS.bigtable_instance, 'instance')
  table = _verify_non_empty_string(FLAGS.bigtable_table, 'table')
  train_prefix = _verify_non_empty_string(FLAGS.bigtable_train_prefix,
                                          'train_prefix')
  eval_prefix = _verify_non_empty_string(FLAGS.bigtable_eval_prefix,
                                         'eval_prefix')
  column_family = _verify_non_empty_string(FLAGS.bigtable_column_family,
                                           'column_family')
  column_qualifier = _verify_non_empty_string(FLAGS.bigtable_column_qualifier,
                                              'column_qualifier')
  return [
      imagenet_input.BigtableSelection(
          project=project,
          instance=instance,
          table=table,
          prefix=p,
          column_family=column_family,
          column_qualifier=column_qualifier)
      for p in (train_prefix, eval_prefix)
  ]


def main(unused_argv):
  params = hyperparameters.get_hyperparameters(FLAGS.default_hparams_file,
                                               FLAGS.hparams_file, FLAGS,
                                               FLAGS.hparams)
  tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
      FLAGS.tpu if (FLAGS.tpu or params['use_tpu']) else '',
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  if params['use_async_checkpointing']:
    save_checkpoints_steps = None
  else:
    save_checkpoints_steps = max(2500, params['iterations_per_loop'])
  config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=get_model_dir(params),
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=None,  # Keep all checkpoints.
      log_step_count_steps=FLAGS.log_step_count_steps,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))),
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=params['iterations_per_loop'],
          num_shards=params['num_cores'],
          # copybara:strip_begin
          tpu_job_name=FLAGS.tpu_job_name,
          # copybara:strip_end
          per_host_input_for_training=contrib_tpu.InputPipelineConfig
          .PER_HOST_V2))  # pylint: disable=line-too-long

  resnet_classifier = contrib_tpu.TPUEstimator(
      use_tpu=params['use_tpu'],
      model_fn=resnet_model_fn,
      config=config,
      params=params,
      train_batch_size=params['train_batch_size'],
      eval_batch_size=params['eval_batch_size'],
      export_to_tpu=FLAGS.export_to_tpu)

  # copybara:strip_begin
  if FLAGS.xla_compile:
    resnet_classifier = contrib_tpu.TPUEstimator(
        use_tpu=params['use_tpu'],
        model_fn=xla.estimator_model_fn(resnet_model_fn),
        config=config,
        params=params,
        train_batch_size=params['train_batch_size'],
        eval_batch_size=params['eval_batch_size'],
        export_to_tpu=FLAGS.export_to_tpu)
  # copybara:strip_end
  assert (params['precision'] == 'bfloat16' or
          params['precision'] == 'float32'), (
              'Invalid value for precision parameter; '
              'must be bfloat16 or float32.')
  tf.logging.info('Precision: %s', params['precision'])
  use_bfloat16 = params['precision'] == 'bfloat16'

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  if FLAGS.bigtable_instance:
    tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
    select_train, select_eval = _select_tables_from_flags()
    imagenet_train = imagenet_input.ImageNetBigtableInput(
        is_training=True,
        use_bfloat16=use_bfloat16,
        transpose_input=params['transpose_input'],
        selection=select_train)
    imagenet_eval = imagenet_input.ImageNetBigtableInput(
        is_training=False,
        use_bfloat16=use_bfloat16,
        transpose_input=params['transpose_input'],
        selection=select_eval)
  else:
    if FLAGS.data_dir == FAKE_DATA_DIR:
      tf.logging.info('Using fake dataset.')
    else:
      tf.logging.info('Using dataset: %s', FLAGS.data_dir)
    imagenet_train, imagenet_eval = [
        imagenet_input.ImageNetInput(
            is_training=is_training,
            data_dir=FLAGS.data_dir,
            transpose_input=params['transpose_input'],
            cache=params['use_cache'] and is_training,
            image_size=params['image_size'],
            num_parallel_calls=params['num_parallel_calls'],
            use_bfloat16=use_bfloat16) for is_training in [True, False]
    ]

  steps_per_epoch = params['num_train_images'] // params['train_batch_size']
  eval_steps = params['num_eval_images'] // params['eval_batch_size']

  if FLAGS.mode == 'eval':

    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(
        get_model_dir(params), timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d', eval_results,
                        elapsed_time)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= params['train_steps']:
          tf.logging.info('Evaluation finished after training step %d',
                          current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint',
                        ckpt)

  elif FLAGS.mode == 'eval_igt':
    # IGT evaluation mode. Evaluate metrics for the desired parameters
    # (true or shifted) on the desired dataset (train or eval). Note that
    # train is still with data augmentation.

    # Get checkpoint file names.
    index_files = tf.gfile.Glob(
        os.path.join(get_model_dir(params), 'model.ckpt-*.index'))
    checkpoints = [fn[:-len('.index')] for fn in index_files]
    # Need to sort them to get proper tensorboard plotting (increasing event
    # timestamps correspond to increasing steps).
    checkpoint_steps = []
    for ckpt in checkpoints:
      tf.logging.info(ckpt)
      step_match = re.match(r'.*model.ckpt-([0-9]*)', ckpt)
      checkpoint_steps.append(int(step_match.group(1)))
    checkpoints = [
        ckpt for _, ckpt in sorted(zip(checkpoint_steps, checkpoints))
    ]
    tf.logging.info('There are {} checkpoints'.format(len(checkpoints)))
    tf.logging.info(', '.join(checkpoints))

    # Keep track of the last processed checkpoint (fault tolerance).
    analysis_state_path = os.path.join(
        get_model_dir(params),
        'analysis_state_' + FLAGS.igt_eval_set + '_' + FLAGS.igt_eval_mode)
    next_analysis_index = 0
    if tf.gfile.Exists(analysis_state_path):
      with tf.gfile.Open(analysis_state_path) as fd:
        next_analysis_index = int(fd.read())

    # Process each checkpoint.
    while next_analysis_index < len(checkpoints):
      tf.logging.info('Next analysis index: {}'.format(next_analysis_index))
      ckpt_path = checkpoints[next_analysis_index]
      tf.logging.info('Starting to evaluate: {}.'.format(ckpt_path))
      start_timestamp = time.time()  # This time will include compilation time

      if FLAGS.igt_eval_set == 'train':
        the_input_fn = imagenet_train.input_fn
        the_steps = steps_per_epoch
      elif FLAGS.igt_eval_set == 'eval':
        the_input_fn = imagenet_eval.input_fn
        the_steps = eval_steps
      else:
        raise ValueError('Unsupported igt_eval_set')

      eval_results = resnet_classifier.evaluate(
          input_fn=the_input_fn,
          steps=the_steps,
          checkpoint_path=ckpt_path,
          name=FLAGS.igt_eval_set + '_' + FLAGS.igt_eval_mode)
      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Eval results: %s. Elapsed seconds: %d', eval_results,
                      elapsed_time)

      next_analysis_index += 1
      file_io.atomic_write_string_to_file(analysis_state_path,
                                          str(next_analysis_index))

  else:  # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    current_step = estimator._load_global_step_from_checkpoint_dir(get_model_dir(params))  # pylint:disable=protected-access,g-line-too-long
    steps_per_epoch = params['num_train_images'] // params['train_batch_size']
    tf.logging.info(
        'Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', params['train_steps'],
        params['train_steps'] / steps_per_epoch, current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      hooks = []
      if params['use_async_checkpointing']:
        hooks.append(
            async_checkpoint.AsyncCheckpointSaverHook(
                checkpoint_dir=get_model_dir(params),
                save_steps=max(2500, params['iterations_per_loop'])))
      resnet_classifier.train(
          input_fn=imagenet_train.input_fn,
          max_steps=params['train_steps'],
          hooks=hooks)

    else:
      assert FLAGS.mode == 'train_and_eval'
      while current_step < params['train_steps']:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              params['train_steps'])
        resnet_classifier.train(
            input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=params['num_eval_images'] // params['eval_batch_size'])
        tf.logging.info('Eval results at step %d: %s', next_checkpoint,
                        eval_results)

      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                      params['train_steps'], elapsed_time)

    if FLAGS.export_dir is not None:
      # The guide to serve a exported TensorFlow model is at:
      #    https://www.tensorflow.org/serving/serving_basic
      tf.logging.info('Starting to export model.')
      unused_export_path = resnet_classifier.export_saved_model(
          export_dir_base=FLAGS.export_dir,
          serving_input_receiver_fn=imagenet_input.image_serving_input_fn)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
