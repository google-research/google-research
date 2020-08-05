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

# Lint as: python2, python3
"""Binary for TuNAS-based architecture search jobs on TPU."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import json
import os
from typing import Any, Dict, Iterable, Text, Tuple

from absl import flags
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf  # tf
import tensorflow.compat.v2 as tf2  # pylint:disable=g-bad-import-order

from tunas import controller
from tunas import cost_model_lib
from tunas import custom_layers
from tunas import fast_imagenet_input
from tunas import mobile_classifier_factory
from tunas import mobile_cost_model
from tunas import mobile_search_space_v3
from tunas import schema_io
from tunas import search_space_utils
from tunas import tpu_optimizer_ops
from tunas.rematlib import layers



# Command-line flags
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
flags.DEFINE_string(
    'checkpoint_dir', None,
    'Path of the directory to read/write training checkpoints to.')
flags.DEFINE_string(
    'dataset_dir', None,
    'Path of the TFDS data directory to load the ImageNet dataset from.')
flags.DEFINE_float(
    'model_learning_rate', 0.165,
    'Learning rate to use for SGD weight updates. The learning rate will '
    'be scaled to account for changes in train_batch_size.')
flags.DEFINE_float(
    'model_warmup_steps_fraction', 0.025,
    'Fraction of epochs that will be used to linearly warmup the '
    'learning rate at the very beginning of training.',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_float(
    'model_momentum', 0.9,
    'Optimizer momentum to use for SGD weight updates.')
flags.DEFINE_float(
    'model_weight_decay', None,
    'L2 regularization for the model parameters. If not set, we will select '
    'a default value automatically.')
flags.DEFINE_integer(
    'train_batch_size', 4096,
    'Batch size to use when training the shared model weights.')
flags.DEFINE_float(
    'rl_learning_rate', 0.0003,
    'Learning rate to use for the RL controller. The learning rate will '
    'be scaled to account for changes in train_batch_size.')
flags.DEFINE_boolean(
    'use_exponential_rl_learning_rate_schedule', True,
    'If true, we will gradually increase the RL learning rate over the course '
    'of a search.')
flags.DEFINE_float(
    'rl_entropy_regularization', 0.0,
    'Entropy regularization to use for the RL controller.')
flags.DEFINE_float(
    'rl_baseline_momentum', 0.95,
    'Momentum to apply to the exponential moving average accumulator for the '
    'RL baseline.')
flags.DEFINE_float(
    'rl_batch_size_multiplier', 4.0,
    'Ratio of RL validation batch size to training batch size. If 1, we will '
    'use the same batch size for training the shared model weights as for '
    'updating the RL controller. If greater than 1, we will use a larger '
    'batch size when updating the RL controller.')
flags.DEFINE_float(
    'rl_delay_steps_fraction', 0.25,
    'Set the RL learning rate to 0 for this fraction of steps at the start of '
    'training.',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_float(
    'increase_ops_warmup_steps_fraction', 0.25,
    'If greater than 0, we will enable all possible operations at the '
    'start of training and gradually increase the rate of path dropout '
    'over this fraction of training steps.',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_float(
    'increase_filters_warmup_steps_fraction', 0.25,
    'When searching over filter sizes, set to a number greater than 0 to '
    'enable all filters at the beginning of training and gradually increase '
    'the rate of filter dropout over this fraction of training steps.',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_enum(
    'rl_reward_function',
    search_space_utils.RL_REWARD_ABS, search_space_utils.RL_REWARDS,
    'RL reward function to use during the search.')
flags.DEFINE_boolean(
    'enable_cost_model', True,
    'If true, we will perform a constrained search that optimizes model '
    'accuracy subject to latency constraints. If false, we will perform '
    'an unconstrained search that only optimizes for accuracy.')
flags.DEFINE_float(
    'rl_cost_model_target', None,
    'Cost model target value to use for the search. Costs below this value '
    'will be rewarded, while costs above this value will be penalized. If '
    'not set, we will select a value automatically.')
flags.DEFINE_float(
    'rl_cost_model_exponent', -0.10,
    'Cost model exponent to use for the search. The more negative this '
    'exponent is, the more heavily the RL reward will penalize expensive '
    'architectures.')
flags.DEFINE_integer(
    'epochs', 90,
    'Number of epochs to run the search for.')
flags.DEFINE_integer(
    'tpu_iterations_per_loop', 100,
    'Number of training iterations to run on TPU before returning to the host.')
flags.DEFINE_enum(
    'use_bfloat16', 'auto', ['true', 'false', 'ontpu', 'auto'],
    'Enable mixed-precision training on TPUs using bfloat16. If set to "ontpu" '
    'we will use float32 in the input pipeline but cast to bfloat16 on the TPU '
    'chip to work around a tf.data bug. If set to "auto", we will '
    'automatically select a value.')
flags.DEFINE_boolean(
    'use_gradient_sync_barrier', True,
    'Performance optimization that may improve model training throughput in '
    'some cases. If enabled, we will force each TPU core to finish computing '
    'gradients for the current training step before applying any AllReduce '
    'ops. We will also force it to finish computing gradients for the shared '
    'model weights before performing a forward pass for the RL controller.')
flags.DEFINE_enum(
    'ssd', mobile_search_space_v3.PROXYLESSNAS_SEARCH,
    mobile_classifier_factory.ALL_SSDS,
    'Search space definition.')


FLAGS = flags.FLAGS


def _grads_and_vars_barrier(
    grads_and_vars
):
  """Barrier that forces all grads to be computed before any are used."""
  current_grads, current_vars = list(zip(*grads_and_vars))
  current_grads = layers.with_data_dependencies(current_grads, current_grads)
  return list(zip(current_grads, current_vars))


def model_fn(
    features, labels,
    mode, params
):
  """Construct a TPUEstimatorSpec for a model."""
  if mode != tf.estimator.ModeKeys.TRAIN:
    raise NotImplementedError(
        'Expected that mode == TRAIN, but got {:!r}'.format(mode))

  # Data was transposed from NHWC to HWCN on the host side. Transpose it back.
  # This transposition will be optimized away by the XLA compiler. It serves
  # as a hint to the compiler that it should expect the input data to come
  # in HWCN format rather than NHWC.
  train_features = tf.transpose(features['train'], [3, 0, 1, 2])
  validation_features = tf.transpose(features['validation'], [3, 0, 1, 2])

  if params['use_bfloat16'] == 'ontpu':
    train_features = tf.cast(train_features, tf.bfloat16)
    validation_features = tf.cast(validation_features, tf.bfloat16)

  global_step = tf.train.get_global_step()

  # Randomly sample a network architecture.
  with tf.variable_scope('rl_controller') as rl_scope:
    pass

  model_spec = mobile_classifier_factory.get_model_spec(params['ssd'])

  tf.io.gfile.makedirs(params['checkpoint_dir'])
  model_spec_filename = os.path.join(
      params['checkpoint_dir'], 'model_spec.json')
  with tf.io.gfile.GFile(model_spec_filename, 'w') as handle:
    handle.write(schema_io.serialize(model_spec))

  increase_ops_prob = custom_layers.linear_decay(
      global_step, params['increase_ops_warmup_steps'])
  increase_filters_prob = custom_layers.linear_decay(
      global_step, params['increase_filters_warmup_steps'])
  model_spec, dist_info = controller.independent_sample(
      model_spec,
      increase_ops_probability=increase_ops_prob,
      increase_filters_probability=increase_filters_prob,
      name=rl_scope)

  if params['enable_cost_model']:
    cost_model_features = mobile_cost_model.coupled_tf_features(model_spec)
    estimated_cost = cost_model_lib.estimate_cost(
        cost_model_features, params['ssd'])

  # We divide the regularization strength by 2 for backwards compatibility with
  # the deprecated tf.contrib.layers.l2_regularizer() function, which was used
  # in our published experiments.
  kernel_regularizer = tf.keras.regularizers.l2(
      params['model_weight_decay'] / 2)

  # Set up the basic TensorFlow training/inference graph.
  model = mobile_classifier_factory.get_model_for_search(
      model_spec,
      kernel_regularizer=kernel_regularizer)
  model.build(train_features.shape)

  with tf.name_scope('training'):
    model_logits, _ = model.apply(train_features, training=True)
    # Cast back to float32 (effectively only when using use_bfloat16 is true).
    model_logits = tf.cast(model_logits, tf.float32)

    model_empirical_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels['train'],
        logits=model_logits,
        label_smoothing=0.1)
    model_regularization_loss = model.regularization_loss()
    model_loss = model_empirical_loss + model_regularization_loss

    # Set up the model weight training logic.
    model_learning_rate = custom_layers.cosine_decay_with_linear_warmup(
        peak_learning_rate=params['model_learning_rate'],
        global_step=global_step,
        max_global_step=params['max_global_step'],
        warmup_steps=params['model_warmup_steps'])

    model_optimizer = tf.tpu.CrossShardOptimizer(
        tf.train.RMSPropOptimizer(
            model_learning_rate,
            decay=0.9,
            momentum=params['model_momentum'],
            epsilon=1.0))

    model_vars = model.trainable_variables()
    model_update_ops = model.updates()
    with tf.control_dependencies(model_update_ops):
      grads_and_vars = model_optimizer.compute_gradients(
          model_loss, var_list=model_vars)
      if params['use_gradient_sync_barrier']:
        # Force all gradients to be computed before any are applied.
        grads_and_vars = _grads_and_vars_barrier(grads_and_vars)

      # NOTE: We do not pass `global_step` to apply_gradients(), so the global
      # step is not incremented by `model_optimizer`. The global_step will be
      # incremented later on, when we update the RL controller weights. If we
      # incremented it here too, we'd end up incrementing the global_step twice
      # at each training step.
      model_op = model_optimizer.apply_gradients(grads_and_vars)
      if params['use_gradient_sync_barrier']:
        # Finish computing gradients for the shared model weights before we
        # start on the RL update step.
        #
        # NOTE: The barrier above forces TensorFlow to finish computing grads
        # for all of the trainable variables before any of the grads can be
        # consumed. So while the call to with_data_dependencies() here only
        # explicitly depends on grads_and_vars[0][0], the call implicitly forces
        # TensorFlow to finish computing the gradients for *all* trainable
        # variables before computing the validation features.
        validation_features = layers.with_data_dependencies(
            [grads_and_vars[0][0]], [validation_features])[0]

  with tf.name_scope('validation'):
    # Estimate the model accuracy on a batch of examples from the validation
    # set. Force this logic to run after the model optimization step.
    with tf.control_dependencies([model_op]):
      validation_logits, _ = model.apply(
          validation_features, training=False)

    # NOTE(b/130311965): An earlier version of this code cast validation_logits
    # from bfloat16 to float32 before applying an argmax when the --use_bfloat16
    # flag was true. As of cl/240923609, this caused XLA to compute incorrect
    # model accuracies. Please avoid casting from bfloat16 to bfloat32 before
    # taking the argmax.
    is_prediction_correct = tf.equal(
        tf.argmax(validation_logits, axis=1),
        tf.argmax(labels['validation'], axis=1))
    validation_accuracy = tf.reduce_mean(
        tf.cast(is_prediction_correct, tf.float32))

  # Estimate the reward for the current network architecture and update the
  # reward to incorporate the cost of the network architecture.
  if params['enable_cost_model']:
    rl_stats = search_space_utils.reward_for_single_cost_model(
        validation_accuracy,
        rl_reward_function=params['rl_reward_function'],
        estimated_cost=estimated_cost,
        rl_cost_model_target=params['rl_cost_model_target'],
        rl_cost_model_exponent=params['rl_cost_model_exponent'])
    rl_cost_ratio = rl_stats['rl_cost_ratio']
    rl_reward = rl_stats['rl_reward']
    rl_cost_adjustment = rl_stats['rl_cost_adjustment']
  else:
    rl_reward = validation_accuracy

  # Compute a baseline. We first take a cross-replica sum of the rewards
  # for all the TPU shards, then incorporate the result into an exponential
  # moving average. Within a single batch, each TPU shard will select a
  # different set of op masks from the RL controller. Each shard will basically
  # evaluate a different candidate architecture in our search space.

  # Count the number of TPU shards (cores) used for training.
  num_tpu_shards = tf.tpu.cross_replica_sum(
      tf.ones(shape=(), dtype=rl_reward.dtype))
  rl_step_baseline = tf.tpu.cross_replica_sum(rl_reward)
  rl_step_baseline = rl_step_baseline / num_tpu_shards

  rl_baseline = custom_layers.update_exponential_moving_average(
      rl_step_baseline,
      momentum=params['rl_baseline_momentum'])

  # Apply a REINFORCE update to the RL controller.
  log_prob = dist_info['sample_log_prob']
  rl_advantage = rl_reward - rl_baseline
  rl_empirical_loss = -tf.stop_gradient(rl_advantage) * log_prob

  # We set rl_entropy_loss proportional to (-entropy) so that minimizing the
  # loss will lead to an entropy that is as large as possible.
  rl_entropy = dist_info['entropy']
  rl_entropy_loss = -params['rl_entropy_regularization'] * rl_entropy

  # We use an RL learning rate of 0 for the first N epochs of training. See
  # Appendix A of FBNet. (https://arxiv.org/pdf/1812.03443.pdf). Although they
  # don't mention it explicitly, there are some indications that ProxylessNAS
  # (https://openreview.net/forum?id=HylVB3AqYm) might also be doing this.
  enable_rl_optimizer = tf.cast(
      tf.greater_equal(global_step, params['rl_delay_steps']),
      tf.float32)
  rl_learning_rate = params['rl_learning_rate'] * enable_rl_optimizer

  if params['use_exponential_rl_learning_rate_schedule']:
    #  rl_learning_rate_progress will be 0 when the RL controller starts
    #  learning and 1 when the search ends.
    rl_learning_rate_progress = tf.nn.relu(tf.div(
        tf.cast(global_step - params['rl_delay_steps'], tf.float32),
        max(1, params['max_global_step'] - params['rl_delay_steps'])))
    # exponentially increase the RL learning rate over time.
    rl_learning_rate_multiplier = tf.pow(10.0, rl_learning_rate_progress)
    rl_learning_rate = rl_learning_rate * rl_learning_rate_multiplier

  rl_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, rl_scope.name)
  with tf.control_dependencies(rl_update_ops):
    # In order to evaluate train_op, we must first evaluate validation_accuracy.
    # And to evaluate validation_accuracy, we must first evaluate model_op. So
    # running this op will perform a step of model training followed by
    # a step of RL controller training.
    if params['use_gradient_sync_barrier']:
      transform_grads_fn = _grads_and_vars_barrier
    else:
      transform_grads_fn = None

    train_op = tpu_optimizer_ops.apply_adam(
        rl_empirical_loss,
        regularization_loss=rl_entropy_loss,
        global_step=global_step,
        var_list=tf.trainable_variables(rl_scope.name),
        learning_rate=rl_learning_rate,
        beta1=0.0,
        beta2=0.999,
        epsilon=1e-8,
        transform_grads_fn=transform_grads_fn)

  # TensorBoard logging
  tensorboard_scalars = collections.OrderedDict([
      ('model/loss', model_loss),
      ('model/empirical_loss', model_empirical_loss),
      ('model/regularization_loss', model_regularization_loss),
      ('model/learning_rate', model_learning_rate),
      ('rlcontroller/empirical_loss', rl_empirical_loss),
      ('rlcontroller/entropy_loss', rl_entropy_loss),
      ('rlcontroller/validation_accuracy', validation_accuracy),
      ('rlcontroller/reward', rl_reward),
      ('rlcontroller/step_baseline', rl_step_baseline),
      ('rlcontroller/baseline', rl_baseline),
      ('rlcontroller/advantage', rl_advantage),
      ('rlcontroller/log_prob', log_prob),
  ])

  if params['enable_cost_model']:
    tensorboard_scalars['rlcontroller/estimated_cost'] = estimated_cost
    tensorboard_scalars['rlcontroller/cost_ratio'] = rl_cost_ratio
    tensorboard_scalars['rlcontroller/cost_adjustment'] = rl_cost_adjustment
    tensorboard_scalars['rlcontroller/learning_rate'] = rl_learning_rate

  tensorboard_scalars['rlcontroller/increase_ops_prob'] = increase_ops_prob
  tensorboard_scalars['rlcontroller/increase_filters_prob'] = (
      increase_filters_prob)

  # Log the values of all the choices made by the RL controller.
  for name_i, logits_i in dist_info['logits_by_path'].items():
    assert len(logits_i.shape) == 1, logits_i
    for j in range(int(logits_i.shape[0])):
      key = 'rlpathlogits/{:s}/{:d}'.format(name_i, j)
      tensorboard_scalars[key] = logits_i[j]

  for name_i, logits_i in dist_info['logits_by_tag'].items():
    assert len(logits_i.shape) == 1, logits_i
    for j in range(int(logits_i.shape[0])):
      key = 'rltaglogits/{:s}/{:d}'.format(name_i, j)
      tensorboard_scalars[key] = logits_i[j]

  # NOTE: host_call only works on rank-1 tensors. There's also a fairly
  # large performance penalty if we try to pass too many distinct tensors
  # from the TPU to the host at once. We avoid these problems by (i) calling
  # tf.stack to merge all of the float32 scalar values into a single rank-1
  # tensor that can be sent to the host relatively cheaply and (ii) reshaping
  # the remaining values from scalars to rank-1 tensors.
  def host_call_fn(step, scalar_values):
    values = tf.unstack(scalar_values)
    with tf2.summary.create_file_writer(
        params['checkpoint_dir']).as_default():
      with tf2.summary.record_if(
          tf.math.equal(step[0] % params['tpu_iterations_per_loop'], 0)):
        for key, value in zip(list(tensorboard_scalars.keys()), values):
          tf2.summary.scalar(key, value, step=step[0])
        return tf.summary.all_v2_summary_ops()

  host_call_values = tf.stack(list(tensorboard_scalars.values()))
  host_call = (
      host_call_fn,
      [tf.reshape(global_step, [1]), host_call_values])

  # Construct the estimator specification.
  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=model_loss,
      train_op=train_op,
      host_call=host_call)


def _merge_datasets(
    train_batch,
    valid_batch
):
  """Merge training features/labels with validation features/labels.

  Args:
    train_batch: A 2-element tuple containing training set features and labels.
    valid_batch: A 2-element tuple containing validation features and labels.

  Returns:
    A 2-element tuple `(features, labels)`, where `features` and `labels` are
        dictionaries of training set or validation set features/labels.
  """
  train_features, train_labels = train_batch
  valid_features, valid_labels = valid_batch
  features = {'train': train_features, 'validation': valid_features}
  labels = {'train': train_labels, 'validation': valid_labels}
  return (features, labels)


def _make_input_dataset(params):
  """Construct a dataset containing both training and validation examples."""
  dataset_fn_fn = fast_imagenet_input.make_dataset
  train_fn, _ = dataset_fn_fn(params['dataset_dir'],
                              'l2l_train',
                              training=True,
                              shuffle_and_repeat=True,
                              use_bfloat16=params['use_bfloat16'] == 'true')
  valid_fn, _ = dataset_fn_fn(params['dataset_dir'],
                              'l2l_valid',
                              training=False,
                              shuffle_and_repeat=True,
                              use_bfloat16=params['use_bfloat16'] == 'true')

  train_dataset = train_fn(params)

  valid_params = dict(params)
  valid_params['batch_size'] = int(round(
      params['batch_size'] * params['rl_batch_size_multiplier']))
  valid_dataset = valid_fn(valid_params)

  dataset = tf.data.Dataset.zip((train_dataset, valid_dataset))
  dataset = dataset.map(_merge_datasets)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


def run_model(params):
  """Train and/or evaluate a model."""
  if params['increase_ops_warmup_steps'] > params['rl_delay_steps']:
    raise ValueError(
        'increase_ops_warmup_steps ({:d}) cannot be greater than '
        'rl_delay_steps ({:d})'
        .format(params['increase_ops_warmup_steps'], params['rl_delay_steps']))

  if params['increase_filters_warmup_steps'] > params['rl_delay_steps']:
    raise NotImplementedError(
        'We currently disable the case where increase_filters_warmup_steps > '
        'rl_delay_steps. If you want to support this case, you should '
        'probably update the code to sample a different architecture for '
        'shared weight training than for RL controller training. And use '
        'increased filter sizes only for the shared weight training.')

  per_host_input_v2 = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  config = tf.estimator.tpu.RunConfig(
      cluster=tf.distribute.cluster_resolver.TPUClusterResolver(
          params['tpu'],
          zone=params['tpu_zone'],
          project=params['gcp_project']),
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=params['tpu_iterations_per_loop'],
          per_host_input_for_training=per_host_input_v2))

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      model_dir=params['checkpoint_dir'],
      config=config,
      train_batch_size=params['train_batch_size'],
      params=params)

  estimator.train(
      _make_input_dataset,
      max_steps=params['max_global_step'])


def _write_params_to_checkpoint_dir(params):
  tf.io.gfile.makedirs(params['checkpoint_dir'])
  params_file = os.path.join(params['checkpoint_dir'], 'params.json')
  with tf.io.gfile.GFile(params_file, 'w') as handle:
    json.dump(params, handle, indent=2, sort_keys=True)


def main(argv):
  del argv  # Unused.

  train_dataset_size = fast_imagenet_input.dataset_size_for_mode('l2l_train')
  max_global_step = train_dataset_size * FLAGS.epochs // FLAGS.train_batch_size

  rl_delay_steps = int(FLAGS.rl_delay_steps_fraction * max_global_step)
  increase_ops_warmup_steps = int(
      FLAGS.increase_ops_warmup_steps_fraction * max_global_step)
  increase_filters_warmup_steps = int(
      FLAGS.increase_filters_warmup_steps_fraction * max_global_step)

  rl_cost_model_target = FLAGS.rl_cost_model_target
  if rl_cost_model_target is None:
    if FLAGS.ssd in mobile_search_space_v3.MOBILENET_V3_LIKE_SSDS:
      # MobileNet V3-based search space. The default value below tries to match
      # the simulated latency of MobileNet V3 on a Pixel 1 phone.
      rl_cost_model_target = 57.0
    else:
      # The default value below tries to match the simulated latency of
      # MnasNet / ProxylessNAS-Mobile on a Pixel 1 phone.
      rl_cost_model_target = 84.0

  use_bfloat16 = FLAGS.use_bfloat16
  if use_bfloat16 == 'auto':
    if FLAGS.ssd == mobile_search_space_v3.MOBILENET_V3_LIKE_SEARCH:
      # Enable bfloat16 by default for this search space. This search space
      # requires more memory than many others. If use_bfloat16 is false,
      # the job will run out of HBM when run on TPU v2 chips but will work
      # fine on TPU v3 chips, which have twice as much HBM. If use_bfloat16
      # is set to 'ontpu', the job will run on both TPU v2 and TPU v3 chips.
      use_bfloat16 = 'ontpu'
    else:
      use_bfloat16 = 'false'

  if use_bfloat16 == 'true':
    tf.logging.warning(
        'WARNING: Because --use_bfloat16=true, the job may hang due to a bug '
        'related to the tf.data library. If this happens, you should be able '
        'to resolve the issue by setting use_bfloat16=ontpu instead.')

  model_weight_decay = FLAGS.model_weight_decay
  if model_weight_decay is None:
    if use_bfloat16 in ['true', 'ontpu']:
      model_weight_decay = 3e-5
    else:
      model_weight_decay = 4e-5

  params = {
      'checkpoint_dir':
          FLAGS.checkpoint_dir,
      'dataset_dir':
          FLAGS.dataset_dir,
      'model_learning_rate':
          FLAGS.model_learning_rate * FLAGS.train_batch_size / 256,
      'model_warmup_steps':
          int(FLAGS.model_warmup_steps_fraction * max_global_step),
      'model_momentum':
          FLAGS.model_momentum,
      'model_weight_decay':
          model_weight_decay,
      'rl_learning_rate':
          FLAGS.rl_learning_rate * FLAGS.train_batch_size / 256,
      'use_exponential_rl_learning_rate_schedule':
          FLAGS.use_exponential_rl_learning_rate_schedule,
      'rl_entropy_regularization':
          FLAGS.rl_entropy_regularization,
      'rl_baseline_momentum':
          FLAGS.rl_baseline_momentum,
      'rl_batch_size_multiplier':
          FLAGS.rl_batch_size_multiplier,
      'rl_delay_steps':
          rl_delay_steps,
      'increase_ops_warmup_steps':
          increase_ops_warmup_steps,
      'increase_filters_warmup_steps':
          increase_filters_warmup_steps,
      'rl_reward_function':
          FLAGS.rl_reward_function,
      'enable_cost_model':
          FLAGS.enable_cost_model,
      'rl_cost_model_exponent':
          FLAGS.rl_cost_model_exponent,
      'rl_cost_model_target':
          rl_cost_model_target,
      'tpu':
          FLAGS.tpu,
      'tpu_zone':
          FLAGS.tpu_zone,
      'gcp_project':
          FLAGS.gcp_project,
      'max_global_step':
          max_global_step,
      'tpu_iterations_per_loop':
          FLAGS.tpu_iterations_per_loop,
      'train_batch_size':
          FLAGS.train_batch_size,
      'use_bfloat16':
          use_bfloat16,
      'use_gradient_sync_barrier':
          FLAGS.use_gradient_sync_barrier,
      'ssd':
          FLAGS.ssd,
  }
  _write_params_to_checkpoint_dir(params)
  run_model(params)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  tf.disable_v2_behavior()
  tf.app.run(main)
