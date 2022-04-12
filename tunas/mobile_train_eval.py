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

"""Standalone model training/evaluation binary for mobile search space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
from typing import Any, Dict, Optional, Text, Tuple, Union

from absl import flags
from six.moves import zip
import tensorflow.compat.v1 as tf  # tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v2 as tf2  # pylint:disable=g-bad-import-order
from tunas import custom_layers
from tunas import fast_imagenet_input
from tunas import mobile_classifier_factory
from tunas import mobile_search_space_v3
from tunas import schema_io
from tunas import search_space_utils



flags.DEFINE_string(
    'tpu', '',
    'The Cloud TPU to connect to. This should either be the name that was '
    'used when creating the TPU or else grpc://tpu.ip.address:8470.')
flags.DEFINE_string(
    'gcp_project', None,
    'GCP project name to use when running on Cloud TPU. If not specified, '
    'we will attempt to infer it automatically.')
flags.DEFINE_string(
    'tpu_zone', None,
    'GCE zone where the Cloud TPU is located. If not specified, we will '
    'attempt to infer it automatically.')
flags.DEFINE_enum(
    'mode', 'train_and_eval',
    ['train', 'eval', 'train_and_eval'],
    'Mode to run the job in.')
flags.DEFINE_string(
    'checkpoint_dir', None,
    'Path of the directory to read/write training checkpoints to.')
flags.DEFINE_string(
    'dataset_dir', None,
    'Path of the TFDS data directory to load the ImageNet dataset from.')
flags.DEFINE_float(
    'base_learning_rate', 0.165,
    'Learning rate to use for SGD weight updates. This number assumes a batch '
    'size of 256. If you use a different batch size, the true learning rate '
    'will be scaled accordingly.')
flags.DEFINE_float(
    'momentum', 0.9,
    'Optimizer momentum to use for SGD weight updates.')
flags.DEFINE_float(
    'weight_decay', None,
    'L2 regularization for the model parameters. If not set, we will select '
    'a default value automatically.')
flags.DEFINE_integer(
    'epochs', None,
    'Number of epochs to train for. If not set, we will select a default '
    'value automatically.')
flags.DEFINE_float(
    'warmup_steps_fraction', 0.025,
    'Fraction of max_global_step that will be used to linearly warmup the '
    'learning rate at the very beginning of training.')
flags.DEFINE_integer(
    'train_batch_size', 4096,
    'Batch size to use for model training.')
flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size to use for model evaluation.')
flags.DEFINE_integer(
    'tpu_iterations_per_loop', 100,
    'Number of training iterations to run on TPU before returning to the host.')
flags.DEFINE_string(
    'indices', '',
    'Colon-separated list of integers controlling the values to use for all '
    'the OneOf nodes in the model.')
flags.DEFINE_boolean(
    'use_held_out_test_set', False,
    'If false, we will train and evaluate on the l2l_train and l2l_valid '
    'portions of the training dataset. If true, we will train of the full '
    'training dataset and evaluate on the held-out validation set.')
flags.DEFINE_boolean(
    'use_bfloat16', False,
    'Enable mixed-precision training on TPUs using bfloat16.')
flags.DEFINE_float(
    'filters_multiplier', 1.0,
    'Set to a value greater than 1 (resp. less than 1) to increase '
    '(resp. decrease) the number of filters in each layer of the network. '
    'Scaling is monotonic but may not be perfectly linear.',
    lower_bound=0)
flags.DEFINE_float(
    'dropout_rate', None,
    'The fraction of elements to drop immediately before the final 1x1 '
    'convolution. If not set, we will select a default value automatically.',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_float(
    'path_dropout_rate', 0,
    'The probability of zeroing out a skippable layer of the network during '
    'training. Will be applied separately to each layer',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_enum(
    'ssd', mobile_search_space_v3.PROXYLESSNAS_MOBILE,
    mobile_classifier_factory.ALL_SSDS,
    'search space definition.')

FLAGS = flags.FLAGS


def model_fn(features,
             labels, mode,
             params):
  """Construct a TPUEstimatorSpec for a model."""
  training = (mode == tf_estimator.ModeKeys.TRAIN)

  if mode == tf_estimator.ModeKeys.EVAL:
    # At evaluation time, the function argument `features` is really a 2-element
    # tuple containing:
    # * A tensor of features w/ shape [batch_size, image_height, image_width, 3]
    # * A tensor of masks w/ shape [batch_size]. Each element of the tensor is
    #   1 (if the element is a normal image) or 0 (if it's a dummy input that
    #   should be ignored). We use this tensor to simulate dynamic batch sizes
    #   during model evaluation. It allows us to handle cases where the
    # validation set size is not a multiple of the eval batch size.
    features, mask = features

  # Data was transposed from NHWC to HWCN on the host side. Transpose it back.
  # This transposition will be optimized away by the XLA compiler. It serves
  # as a hint to the compiler that it should expect the input data to come
  # in HWCN format rather than NHWC.
  features = tf.transpose(features, [3, 0, 1, 2])

  model_spec = mobile_classifier_factory.get_model_spec(
      ssd=params['ssd'],
      indices=params['indices'],
      filters_multipliers=params['filters_multiplier'],
      path_dropout_rate=params['path_dropout_rate'],
      training=training)

  tf.io.gfile.makedirs(params['checkpoint_dir'])
  model_spec_filename = os.path.join(
      params['checkpoint_dir'], 'model_spec.json')
  with tf.io.gfile.GFile(model_spec_filename, 'w') as handle:
    handle.write(schema_io.serialize(model_spec))

  # We divide the weight_decay by 2 for backwards compatibility with the
  # tf.contrib version of the kernel regularizer, which was used in the
  # experiments from our published paper.
  kernel_regularizer = tf.keras.regularizers.l2(params['weight_decay'] / 2)
  model = mobile_classifier_factory.get_standalone_model(
      model_spec=model_spec,
      kernel_regularizer=kernel_regularizer,
      dropout_rate=params['dropout_rate'])

  model.build(features.shape)
  logits, _ = model.apply(
      inputs=features,
      training=training)
  regularization_loss = model.regularization_loss()
  # Cast back to float32 (effectively only when using use_bfloat16 is true).
  logits = tf.cast(logits, tf.float32)

  if mode == tf_estimator.ModeKeys.PREDICT:
    return tf_estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        },
        export_outputs={
            'logits': tf_estimator.export.PredictOutput({'logits': logits}),
        })

  empirical_loss = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=labels,
      label_smoothing=0.1)
  loss = empirical_loss + regularization_loss

  # Optionally define an op for model training.
  global_step = tf.train.get_global_step()
  if mode == tf_estimator.ModeKeys.TRAIN:
    # linearly scale up the learning rate before switching to cosine decay
    learning_rate = custom_layers.cosine_decay_with_linear_warmup(
        peak_learning_rate=params['learning_rate'],
        global_step=global_step,
        max_global_step=params['max_global_step'],
        warmup_steps=params['warmup_steps'])

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=0.9,
        momentum=params['momentum'],
        epsilon=1.0)

    scaffold_fn = None

    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    with tf.control_dependencies(model.updates()):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None
    scaffold_fn = None

  # Optionally define evaluation metrics.
  if mode == tf_estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits, mask):
      label_values = tf.argmax(labels, axis=1)
      predictions = tf.argmax(logits, axis=1)
      accuracy = tf.metrics.accuracy(label_values, predictions, weights=mask)
      return {'accuracy': accuracy}
    eval_metrics = (metric_fn, [labels, logits, mask])
  else:
    eval_metrics = None

  # NOTE: host_call only works on rank-1 tensors. There's also a fairly
  # large performance penalty if we try to pass too many distinct tensors
  # from the TPU to the host at once. We avoid these problems by (i) calling
  # tf.stack to merge all of the float32 scalar values into a single rank-1
  # tensor that can be sent to the host relatively cheaply and (ii) reshaping
  # the remaining values from scalars to rank-1 tensors.
  if mode == tf_estimator.ModeKeys.TRAIN:
    tensorboard_scalars = collections.OrderedDict()
    tensorboard_scalars['model/loss'] = loss
    tensorboard_scalars['model/empirical_loss'] = empirical_loss
    tensorboard_scalars['model/regularization_loss'] = regularization_loss
    tensorboard_scalars['model/learning_rate'] = learning_rate

    def host_call_fn(step, scalar_values):
      values = tf.unstack(scalar_values)
      with tf2.summary.create_file_writer(
          params['checkpoint_dir']).as_default():
        with tf2.summary.record_if(
            tf.equal(step[0] % params['tpu_iterations_per_loop'], 0)):
          for key, value in zip(list(tensorboard_scalars.keys()), values):
            tf2.summary.scalar(key, value, step=step[0])
          return tf.summary.all_v2_summary_ops()

    host_call_values = tf.stack(list(tensorboard_scalars.values()))
    host_call = (host_call_fn, [tf.reshape(global_step, [1]), host_call_values])
  else:
    host_call = None

  # Construct the estimator specification.
  return tf_estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn,
      host_call=host_call)


def _make_estimator(
    params, is_training
):
  """Returns a TPUEstimator for use in training or evaluation."""
  if is_training:
    input_mode = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V2
  else:
    # Enable BROADCAST mode for model evaluation. In BROADCAST mode, input
    # preprocessing happens on a single host. We then broadcast each batch of
    # examples to all of the TPU cores. This is less efficient than PER_HOST_V2.
    # However, it's needed to make TPUEstimator work with multi-host setups
    # right now, since TF doesn't support PER_HOST_V2 input pipelines with more
    # than one host.
    input_mode = tf_estimator.tpu.InputPipelineConfig.BROADCAST

  config = tf_estimator.tpu.RunConfig(
      cluster=tf.distribute.cluster_resolver.TPUClusterResolver(
          params['tpu'],
          zone=params['tpu_zone'],
          project=params['gcp_project']),
      tpu_config=tf_estimator.tpu.TPUConfig(
          iterations_per_loop=params['tpu_iterations_per_loop'],
          per_host_input_for_training=input_mode))
  return tf_estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      model_dir=params['checkpoint_dir'],
      config=config,
      train_batch_size=params['train_batch_size'],
      eval_batch_size=params['eval_batch_size'],
      params=params)


def run_model(params, mode):
  """Train and/or evaluate a model."""
  assert mode in ['train', 'eval', 'train_and_eval'], mode
  dataset_fn_fn = fast_imagenet_input.make_dataset

  if mode in ['train', 'train_and_eval']:
    dataset_mode = 'train' if params['use_held_out_test_set'] else 'l2l_train'
    dataset_fn, _ = dataset_fn_fn(
        params['dataset_dir'],
        dataset_mode,
        training=True,
        use_bfloat16=params['use_bfloat16'])
    estimator = _make_estimator(params, is_training=True)
    estimator.train(dataset_fn, max_steps=params['max_global_step'])
  if mode in ['eval', 'train_and_eval']:
    dataset_mode = 'test' if params['use_held_out_test_set'] else 'l2l_valid'
    dataset_fn, dataset_size = dataset_fn_fn(
        params['dataset_dir'],
        dataset_mode,
        training=False,
        use_bfloat16=params['use_bfloat16'],
        final_batch_mode=fast_imagenet_input.FinalBatchMode.PAD)

    estimator = _make_estimator(params, is_training=False)
    for checkpoint_path in tf.train.checkpoints_iterator(
        params['checkpoint_dir']):
      eval_metrics = estimator.evaluate(
          dataset_fn,
          steps=dataset_size // params['eval_batch_size'],
          checkpoint_path=checkpoint_path)
      tf.logging.info('eval metrics = %s', eval_metrics)
      if eval_metrics['global_step'] >= params['max_global_step']:
        return eval_metrics


def _write_params_to_checkpoint_dir(params):
  tf.io.gfile.makedirs(params['checkpoint_dir'])
  params_file = os.path.join(params['checkpoint_dir'], 'params.json')
  with tf.io.gfile.GFile(params_file, 'w') as handle:
    json.dump(params, handle, indent=2, sort_keys=True)


def main(argv):
  del argv  # Unused.

  if FLAGS.use_held_out_test_set:
    default_epochs = 360
    train_dataset_size = fast_imagenet_input.dataset_size_for_mode('train')
  else:
    default_epochs = 90
    train_dataset_size = fast_imagenet_input.dataset_size_for_mode('l2l_train')

  epochs = FLAGS.epochs
  if epochs is None:
    epochs = default_epochs

  weight_decay = FLAGS.weight_decay
  if weight_decay  is None:
    weight_decay = 3e-5 if FLAGS.use_bfloat16 else 4e-5

  dropout_rate = FLAGS.dropout_rate
  if dropout_rate is None:
    # Select a dropout rate automatically. For MnasNet-sized models, dropout
    # substantially improves accuracy when training for 360 epochs, but we
    # haven't investigated whether it helps when training for 90 epochs. We
    # currently enable dropout only for long training runs.
    if epochs < 150:
      # Disable dropout when training for less than 150 epochs.
      dropout_rate = 0
    elif FLAGS.ssd in mobile_search_space_v3.MOBILENET_V3_LIKE_SSDS:
      # MobileNetV3-based search space, training for at least 150 epochs.
      dropout_rate = 0.25
    else:
      # MobileNetV2-based search space, training for at least 150 epochs.
      dropout_rate = 0.15

  max_global_step = train_dataset_size * epochs // FLAGS.train_batch_size

  params = {
      'checkpoint_dir': FLAGS.checkpoint_dir,
      'dataset_dir': FLAGS.dataset_dir,
      'learning_rate': FLAGS.base_learning_rate * FLAGS.train_batch_size / 256,
      'tpu': FLAGS.tpu,
      'tpu_zone': FLAGS.tpu_zone,
      'gcp_project': FLAGS.gcp_project,
      'momentum': FLAGS.momentum,
      'weight_decay': weight_decay,
      'max_global_step': max_global_step,
      'warmup_steps': int(FLAGS.warmup_steps_fraction * max_global_step),
      'tpu_iterations_per_loop': FLAGS.tpu_iterations_per_loop,
      'train_batch_size': FLAGS.train_batch_size,
      'eval_batch_size': FLAGS.eval_batch_size,
      'indices': search_space_utils.parse_list(FLAGS.indices, int),
      'use_held_out_test_set': FLAGS.use_held_out_test_set,
      'use_bfloat16': FLAGS.use_bfloat16,
      'filters_multiplier': FLAGS.filters_multiplier,
      'dropout_rate': dropout_rate,
      'path_dropout_rate': FLAGS.path_dropout_rate,
      'ssd': FLAGS.ssd,
  }

  if FLAGS.mode in ['train', 'train_and_eval']:
    _write_params_to_checkpoint_dir(params)

  run_model(params, FLAGS.mode)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  tf.disable_v2_behavior()
  tf.app.run(main)
