# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Train a joint model from a source dataset and a target set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from absl import app
from absl import flags
from inputs import data_input
import model
from models import resnet_params
import tensorflow as tf
from tensorflow.contrib import summary
from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow_probability as tfp
from utils import model_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('pre_train_steps', 100, help=('pretrain steps'))
flags.DEFINE_integer('finetune_steps', 100, help=('finetune steps'))
flags.DEFINE_integer('ctrl_steps', 100, help=('control steps'))
flags.DEFINE_string(
    'param_file',
    None,
    help=(
        'Base set of model parameters to use with this model. To see '
        'documentation on the parameters, see the docstring in resnet_params.'))
flags.DEFINE_multi_string(
    'param_overrides',
    None,
    help=('Model parameter overrides for this model. For example, if '
          'experimenting with larger numbers of train_steps, a possible value '
          'is --param_overrides=train_steps=28152. If you have a collection of '
          'parameters that make sense to use together repeatedly, consider '
          'extending resnet_params.param_sets_table.'))
flags.DEFINE_string(
    'data_dir',
    '',
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))
flags.DEFINE_string(
    'model_dir',
    None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))
flags.DEFINE_integer(
    'profile_every_n_steps',
    0,
    help=('Number of steps between collecting profiles if larger than 0'))
flags.DEFINE_string(
    'mode',
    'train_and_eval',
    help='One of {"train_and_eval", "train", "eval"}.')
flags.DEFINE_integer(
    'steps_per_eval',
    1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))
flags.DEFINE_integer(
    'eval_timeout',
    None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_integer(
    'log_step_count_steps', 64, 'The number of steps at '
    'which the global step information is logged.')
flags.DEFINE_string('model_name', 'resnet',
                    'Serving model name used for the model server.')
flags.DEFINE_multi_integer(
    'inference_batch_sizes', [8],
    'Known inference batch sizes used to warm up for each core.')
flags.DEFINE_integer('use_cosine_lr', 0,
                     'Whether to use cosine learning rate scheduling.')
flags.DEFINE_string('model_type', 'resnet', 'The type of convolutional model')
flags.DEFINE_string(
    'warm_start_ckpt_path', None, 'The path to the checkpoint '
    'that will be used before training.')
flags.DEFINE_integer('train_steps', 120000, 'Number of total training steps.')
flags.DEFINE_float('label_smoothing', 0.0,
                   'Label smoothing for the target dataset.')
flags.DEFINE_float('src_label_smoothing', 0.0,
                   'Label smoothing for the source dataset')
flags.DEFINE_integer('num_choices', 10,
                     'Number of actions for the scaling variable.')
flags.DEFINE_float('base_learning_rate_scale', 0.001,
                   'The value of the learning rate')
flags.DEFINE_float('dst_weight_decay', 0.0,
                   'Weight decay for the target dataset.')
flags.DEFINE_integer('save_checkpoints_steps', 100,
                     'Number of steps for each checkpoint saving.')
flags.DEFINE_bool('reset_dense_layer', False,
                  'Whether to re-initialize the final dense layer.')
flags.DEFINE_integer('soft_weight', 0,
                     'The normalizetion strategy for the coefficient.')
flags.DEFINE_bool('dst_add_dropout', False,
                  'Whether to add the dropout to the target dataset.')
flags.DEFINE_float('rl_learning_rate', 0.1, 'Learning rate for RL updates.')
flags.DEFINE_string('source_dataset', None, 'Name of the source dataset.')
flags.DEFINE_string('target_dataset', None, 'Name of the target dataset.')
flags.DEFINE_integer('train_batch_size', 1024,
                     'The batch size during training.')
flags.DEFINE_string('model_optimizer', 'rms', 'Optimizer to update the model.')
flags.DEFINE_integer('decay_steps', 40000,
                     'The number of steps for a learning rate decay.')
flags.DEFINE_integer('loss_type', 0, 'The choice of different losses.')
flags.DEFINE_integer('use_smooth_update', 0, 'Whether to use smooth update.')
flags.DEFINE_integer(
    'train_batch_size_multiplier', 0,
    'The multiplier will be used to increase the batch size '
    'to sample more examples.')
flags.DEFINE_integer('load_pretrain_dense', 1,
                     'Whether to load the pretrained dense layer.')
flags.DEFINE_string('extra', None, 'Extra string about model configuration.')
flags.DEFINE_float('cosine_alpha', 0.01,
                   'Alpha value in the cosine learning rate scheduling.')
flags.DEFINE_integer('multi_cls_branch', 0,
                     'The use of aux classification branch in the target.')
flags.DEFINE_integer('src_multi_branch', 0,
                     'The use of aux classification branch in the source.')
flags.DEFINE_float('loss_weight_scale', 1000.0, 'Scaling of the loss weight.')
flags.DEFINE_integer('first_pretrain_steps', 0,
                     'Number of steps for pretraining.')
flags.DEFINE_integer('target_batch_multiplier', 4,
                     'Multiplier for the target evaluation batch size.')
flags.DEFINE_integer('target_train_batch_multiplier', 1,
                     'Multiplier for the target evaluation train batch size.')
flags.DEFINE_integer('uniform_weight', 0,
                     'Use of uniform weight in the ablation studies.')


def get_global_step(name):
  """Returns the global step variable."""
  global_step = tf.get_variable(
      name,
      shape=[],
      dtype=tf.int64,
      initializer=tf.initializers.zeros(),
      trainable=False,
      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES])
  return global_step


def get_src_num_classes():
  """Returns the number of classes give the dataset name."""
  # For Imagenet:
  num_classes = 1001
  return num_classes


def src_lr_schedule(params, current_step):  # pylint: disable=unused-argument
  """Returns the scheduled learning rate for source."""
  scaled_lr = 0.1 * (FLAGS.train_batch_size / 256.0)
  return model_utils.multi_stage_lr(
      FLAGS.decay_steps,
      scaled_lr * FLAGS.base_learning_rate_scale,
      current_step,
  )


def get_src_train_op(loss, num_all_iters, params):  # pylint: disable=unused-argument
  """Returns the source training op."""
  global_step = tf.train.get_global_step()
  src_learning_rate = src_lr_schedule(params, global_step)
  if FLAGS.model_optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=src_learning_rate,
        momentum=params['momentum'],
        use_nesterov=True)
  elif FLAGS.model_optimizer == 'rms':
    optimizer = tf.train.RMSPropOptimizer(
        src_learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
  if params['use_tpu']:
    src_optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
  with tf.variable_scope('src'):
    return src_optimizer.minimize(loss, global_step), src_learning_rate


def meta_train_op(acc, rl_entropy, log_prob, rl_scope, params):  # pylint: disable=unused-argument
  """Returns the target training op.

  Update the control variables using policy gradient.
  Args:
    acc: reward on validation set. In our case, the reward is the top-1 acc;
    rl_entropy: entropy of action logits;
    log_prob: log prob of the action;
    rl_scope: variable scope;
    params: other params;

  Returns:
    target_train_op: train op;
    rl_learning_rate: lr;
    out_metric: metric dict;
  """
  target_global_step = get_global_step('train_rl_global_step')
  rl_reward = acc
  rl_step_baseline = tf.contrib.tpu.cross_replica_sum(rl_reward)
  rl_baseline_momentum = 0.95
  rl_entropy_regularization = 0.

  def update_rl_baseline():
    return model_utils.update_exponential_moving_average(
        rl_step_baseline, momentum=rl_baseline_momentum)

  rl_baseline = update_rl_baseline()

  rl_advantage = rl_reward - rl_baseline
  rl_empirical_loss = -tf.stop_gradient(rl_advantage) * log_prob

  rl_entropy_loss = -rl_entropy_regularization * rl_entropy

  enable_rl_optimizer = tf.cast(
      tf.greater_equal(target_global_step, FLAGS.first_pretrain_steps),
      tf.float32)
  rl_learning_rate = FLAGS.rl_learning_rate * enable_rl_optimizer

  # TO BE MODIFIED:
  target_train_op = tf.train.AdamOptimizer()

  out_metric = {
      'rl_empirical_loss': rl_empirical_loss,
      'rl_entropy_loss': rl_entropy_loss,
      'rl_reward': rl_reward,
      'rl_step_baseline': rl_step_baseline,
      'rl_baseline': rl_baseline,
      'rl_advantage': rl_advantage,
      'log_prob': log_prob,
  }
  return target_train_op, rl_learning_rate, out_metric


def get_logits(feature, params, mode):
  """Returns the network logits."""
  end_points = None
  avg_pool = model.resnet_v1_model(feature, mode, params)
  return avg_pool, end_points


def get_model_logits(src_features, finetune_features, params, mode, num_classes,
                     target_num_classes):
  """Gets the logits from different models."""
  src_avg_pool, _ = get_logits(src_features, params, mode)
  dst_avg_pool, _ = get_logits(finetune_features, params, mode)

  src_logits, _ = do_cls(
      src_avg_pool, num_classes, name='dense', add_dropout=False)
  dst_logits, _ = do_cls(
      dst_avg_pool,
      target_num_classes,
      name='final_dense_dst',
      add_dropout=False)
  src_logits, dst_logits = do_cast(src_logits), do_cast(dst_logits)
  src_aux_logits, dst_aux_logits = None, None
  return src_logits, src_aux_logits, dst_logits, dst_aux_logits


def get_loss(logits, inst_weights, one_hot_labels):
  """Returns the loss function."""
  label_smoothing = 0.
  aux_loss = 0.

  loss = tf.losses.softmax_cross_entropy(
      logits=logits,
      weights=inst_weights,
      onehot_labels=one_hot_labels,
      label_smoothing=label_smoothing)
  loss = loss + aux_loss
  return loss


def get_final_loss(src_logits, src_one_hot_labels, dst_logits,
                   finetune_one_hot_labels, global_step, loss_weights,
                   inst_weights):
  """Gets the final loss for ."""
  if FLAGS.uniform_weight:
    inst_weights = 1.0
  src_loss = get_loss(src_logits, inst_weights, src_one_hot_labels)
  dst_loss = get_loss(dst_logits, 1., finetune_one_hot_labels)
  l2_loss = []
  for v in tf.trainable_variables():
    if 'batch_normalization' not in v.name and 'rl_controller' not in v.name:
      l2_loss.append(tf.nn.l2_loss(v))
  l2_loss = FLAGS.dst_weight_decay * tf.add_n(l2_loss)

  decay_rate = tf.train.cosine_decay_restarts(
      1.,
      global_step,
      FLAGS.decay_steps,
      t_mul=2.0,
      m_mul=1.0,
      alpha=FLAGS.cosine_alpha,
  )

  enable_pretrain = tf.cast(
      tf.greater_equal(global_step, FLAGS.first_pretrain_steps), tf.float32)

  if FLAGS.loss_type == 3:
    loss = decay_rate * src_loss + (1 - decay_rate) * dst_loss + l2_loss
  elif FLAGS.loss_type == 2:
    loss = decay_rate + dst_loss + l2_loss
  elif FLAGS.loss_type == 1:
    loss = src_loss + dst_loss + l2_loss
  elif FLAGS.loss_type == 5:
    loss = 0.1 * src_loss + dst_loss + l2_loss
  elif FLAGS.loss_type == 6:
    loss = 0.0 * src_loss + dst_loss + l2_loss
  elif FLAGS.loss_type == 4:
    loss = src_loss * tf.stop_gradient(loss_weights) * enable_pretrain
    loss += dst_loss + l2_loss
  else:
    loss = decay_rate * src_loss + dst_loss + l2_loss

  return tf.identity(loss), src_loss, dst_loss


def do_cast(logits):
  """Casts data type."""
  logits = tf.cast(logits, tf.float32)
  return logits


def do_cls(avg_pool, num_classes, name='dense', add_dropout=False):
  """Applies classification."""
  if add_dropout:
    avg_pool = tf.nn.dropout(avg_pool, 0.5)
  with tf.variable_scope('target_CLS', reuse=tf.AUTO_REUSE):
    logits = tf.layers.dense(
        inputs=avg_pool,
        units=num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=.01),
        name=name)
    return logits, None


def dense(avg_pool, _in, num_classes, name='dense', stddev=0.01):  # pylint: disable=invalid-name
  """Applies dense layers."""
  _out = num_classes  # pylint: disable=invalid-name
  kernel = tf.get_variable(
      name='{}/kernel'.format(name),
      initializer=tf.random_normal_initializer(stddev=stddev),
      shape=[_in, _out],
      dtype=tf.float32)
  bias = tf.get_variable(
      name='{}/bias'.format(name),
      initializer=tf.initializers.zeros(),
      shape=[_out],
      dtype=tf.float32)
  outputs = tf.matmul(avg_pool, kernel)
  outputs = tf.nn.bias_add(outputs, bias)

  kernel_assign_op = kernel.assign(tf.random_normal([_in, _out], stddev=stddev))
  bias_assign_op = bias.assign(tf.zeros([_out]))
  return outputs, [kernel_assign_op, bias_assign_op]


def do_cls_v2(avg_pool, end_points, num_classes, name='dense'):
  logits, assign_op_dense = dense(
      avg_pool, 2048, num_classes, name, stddev=0.01)
  aux_pool = end_points['AuxLogits_Pool']
  aux_logits, assign_op_aux = dense(
      aux_pool, 768, num_classes, 'Aux{}'.format(name), stddev=0.001)
  return logits, aux_logits, assign_op_dense + assign_op_aux


def train_model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
  """Defines the model function."""
  target_num_classes = data_input.num_classes_map[FLAGS.target_dataset]
  global_step = tf.train.get_global_step()
  num_all_iters = FLAGS.pre_train_steps + FLAGS.finetune_steps + FLAGS.ctrl_steps

  src_features, src_labels = features['src'], labels['src']
  finetune_features = features['finetune']
  target_features = features['target']

  num_classes = get_src_num_classes()

  finetune_one_hot_labels = tf.one_hot(
      tf.cast(labels['finetune'], tf.int64), target_num_classes)
  target_one_hot_labels = tf.one_hot(
      tf.cast(labels['target'], tf.int64), target_num_classes)

  with tf.variable_scope('rl_controller') as rl_scope:
    # It creates a `rl_scope` which will be used for ops.
    pass
  _, rl_entropy, label_weights, log_prob = rl_label_weights(rl_scope)
  if FLAGS.loss_type == 4:
    if FLAGS.use_smooth_update:
      _, loss_entropy, loss_weights, loss_log_prob = get_loss_weights_smooth(
          rl_scope)
    else:
      _, loss_entropy, loss_weights, loss_log_prob = get_loss_weights(rl_scope)
  else:
    loss_entropy, loss_weights, loss_log_prob = 0., 1., 0.

  branch_entropy, _ = 0., 1.
  branch_log_prob, _ = 0., 1.

  src_branch_entropy, _, src_branch_log_prob, _ = 0., 1., 0., 1.

  def gather_init_weights():
    inst_weights = tf.stop_gradient(tf.gather(label_weights, src_labels))
    return inst_weights

  if FLAGS.soft_weight == 1:
    inst_weights = tf.nn.softmax(label_weights) * num_classes
    inst_weights = gather_init_weights()
  else:
    inst_weights = gather_init_weights()
    inst_weights, indices = tf.nn.top_k(
        inst_weights,
        k=params['batch_size'],
        sorted=True,
    )
    hw = 224
    src_features = tf.reshape(
        src_features,
        [hw, hw, 3, params['batch_size'] * FLAGS.train_batch_size_multiplier])
    src_features = tf.gather(src_features, indices, axis=-1)
    src_features = tf.transpose(src_features, [3, 0, 1, 2])
    src_features = tf.stop_gradient(src_features)

    src_labels = tf.gather(src_labels, indices)

    if FLAGS.soft_weight == 2:
      inst_weights = tf.nn.softmax(inst_weights)
    bs = params['batch_size']
    inst_weights = bs * inst_weights / tf.reduce_sum(inst_weights)

  src_one_hot_labels = tf.one_hot(tf.cast(src_labels, tf.int64), num_classes)

  with tf.contrib.tpu.bfloat16_scope():
    ret = get_model_logits(src_features, finetune_features, params, mode,
                           num_classes, target_num_classes)
    src_logits, _ = ret[0:2]
    dst_logits, _ = ret[2:4]

  loss, src_loss, dst_loss = get_final_loss(src_logits, src_one_hot_labels,
                                            dst_logits, finetune_one_hot_labels,
                                            global_step, loss_weights,
                                            inst_weights)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  with tf.control_dependencies(update_ops):
    src_train_op, src_learning_rate = get_src_train_op(loss, num_all_iters,
                                                       params)
    with tf.control_dependencies([src_train_op]):
      # CHECK THE SCOPE USE
      # scope.reuse_variables()
      target_avg_pool, _ = get_logits(target_features, params, mode)
      target_logits, _ = do_cls(
          target_avg_pool,
          target_num_classes,
          name='final_dense_dst',
          add_dropout=False)
      is_prediction_correct = tf.equal(
          tf.argmax(tf.identity(target_logits), axis=1),
          tf.argmax(target_one_hot_labels, axis=1))
      acc = tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))

      entropy = src_branch_entropy + branch_entropy + loss_entropy + rl_entropy
      log_prob = src_branch_log_prob + branch_log_prob + loss_log_prob + log_prob
      train_op, rl_learning_rate, rl_metric = meta_train_op(
          acc, entropy, log_prob, rl_scope, params)

  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    if not FLAGS.skip_host_call:
      tensorboard_scalars = collections.OrderedDict([
          ('classifier/src_loss', src_loss),
          ('classifier/finetune_loss', dst_loss),
          ('classifier/src_learning_rate', src_learning_rate),
          ('rlcontroller/rl_learning_rate', rl_learning_rate),
          ('rlcontroller/empirical_loss', rl_metric['rl_empirical_loss']),
          ('rlcontroller/entropy_loss', rl_metric['rl_entropy_loss']),
          ('rlcontroller/reward', rl_metric['rl_reward']),
          ('rlcontroller/step_baseline', rl_metric['rl_step_baseline']),
          ('rlcontroller/baseline', rl_metric['rl_baseline']),
          ('rlcontroller/advantage', rl_metric['rl_advantage']),
          ('rlcontroller/log_prob', rl_metric['log_prob']),
      ])

      def host_call_fn(gs, scalar_values):
        """Returns summary."""
        gs = gs[0]
        values = tf.unstack(scalar_values)

        with summary.create_file_writer(
            FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
          with summary.always_record_summaries():
            for key, value in zip(tensorboard_scalars.keys(), values):
              tf.contrib.summary.scalar(key, value, step=gs)

            return summary.all_summary_ops()

      gs_t = tf.reshape(global_step, [1])

      host_call_values = tf.stack(tensorboard_scalars.values())
      host_call = (host_call_fn, [gs_t, host_call_values])

  else:
    train_op = None

  eval_metrics = None

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics)


def rl_label_weights(name=None):
  """Returns the weight for importance."""
  with tf.variable_scope(name, 'rl_op_selection'):
    num_classes = get_src_num_classes()
    num_choices = FLAGS.num_choices

    logits = tf.get_variable(
        name='logits_rl_w',
        initializer=tf.initializers.zeros(),
        shape=[num_classes, num_choices],
        dtype=tf.float32)
    dist_logits_list = logits.value()
    dist = tfp.distributions.Categorical(logits=logits)
    dist_entropy = tf.reduce_sum(dist.entropy())

    sample = dist.sample()
    sample_masks = 1. * tf.cast(sample, tf.float32) / num_choices
    sample_log_prob = tf.reduce_mean(dist.log_prob(sample))

  return (dist_logits_list, dist_entropy, sample_masks, sample_log_prob)


def get_loss_weights(name=None):
  """Returns the weight for loss."""
  with tf.variable_scope(name, 'rl_op_selection'):
    logits = tf.get_variable(
        name='loss_logits_rl_w',
        initializer=tf.initializers.zeros(),
        shape=[
            100,
        ],
        dtype=tf.float32)
    dist_logits_list = logits.value()
    dist = tfp.distributions.Categorical(logits=logits)
    dist_entropy = tf.reduce_sum(dist.entropy())

    sample = dist.sample()
    sample_masks = 1. * tf.cast(sample, tf.float32) / FLAGS.loss_weight_scale
    sample_log_prob = tf.reduce_mean(dist.log_prob(sample))

  return (dist_logits_list, dist_entropy, sample_masks, sample_log_prob)


def get_loss_weights_smooth(name=None):
  """Returns the smooth weight for loss."""
  with tf.variable_scope(name, 'rl_op_selection'):
    init_loss = tf.get_variable(
        name='init_weight',
        trainable=False,
        initializer=tf.initializers.ones(),
        shape=[1],
        dtype=tf.float32)
    logits = tf.get_variable(
        name='loss_logits_rl_w',
        initializer=tf.initializers.zeros(),
        shape=[
            10,
        ],
        dtype=tf.float32)
    dist_logits_list = logits.value()
    dist = tfp.distributions.Categorical(logits=logits)
    dist_entropy = tf.reduce_sum(dist.entropy())

    sample = dist.sample()
    sample_masks = 1. * tf.cast(sample, tf.float32) / 100.
    sample_masks -= 0.05
    new_loss = init_loss + tf.reshape(sample_masks, [1])
    new_loss = tf.minimum(new_loss, [1.])
    new_loss = tf.maximum(new_loss, [0.])
    init_loss = tf.assign(init_loss, new_loss)
    sample_log_prob = tf.reduce_mean(dist.log_prob(sample))

  return (dist_logits_list, dist_entropy, init_loss, sample_log_prob)


def main(unused_argv):
  params = resnet_params.from_file(FLAGS.param_file)
  params = resnet_params.override(params, FLAGS.param_overrides)
  resnet_params.log_hparams_to_model_dir(params, FLAGS.model_dir)
  tf.logging.info('Model params: {}'.format(params))

  if params['use_async_checkpointing']:
    save_checkpoints_steps = None
  else:
    save_checkpoints_steps = FLAGS.save_checkpoints_steps

  # TO BE MODIFIED
  config = tf.contrib.tpu.RunConfig(
      cluster='',
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=50,
      log_step_count_steps=FLAGS.log_step_count_steps,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))))

  if FLAGS.warm_start_ckpt_path:
    var_names = []
    checkpoint_path = FLAGS.warm_start_ckpt_path
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    for key in reader.get_variable_to_shape_map():
      extra_str = ''
      keep_str = 'Momentum|global_step|finetune_global_step'
      if not re.findall('({}{})'.format(keep_str, extra_str), key):
        var_names.append(key)

    tf.logging.info('Warm-starting tensors: %s', sorted(var_names))

    vars_to_warm_start = var_names
    warm_start_settings = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=checkpoint_path,
        vars_to_warm_start=vars_to_warm_start)
  else:
    warm_start_settings = None

  # TO BE MODIFIED
  resnet_classifier = tf.estimator.Estimator(
      model_fn=train_model_fn,
      config=config,
      params=params,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=1024,
      warm_start_from=warm_start_settings)

  use_bfloat16 = params['precision'] == 'bfloat16'

  num_classes = get_src_num_classes()

  def make_input_dataset(params):
    """return input dataset."""

    def _merge_datasets(train_batch, finetune_batch, target_batch):
      """merge different splits."""
      train_features, train_labels = train_batch
      finetune_features, finetune_labels = finetune_batch
      target_features, target_labels = target_batch
      features = {
          'src': train_features,
          'finetune': finetune_features,
          'target': target_features
      }
      labels = {
          'src': train_labels,
          'finetune': finetune_labels,
          'target': target_labels
      }
      return (features, labels)

    # TO BE MODIFIED
    data_dir = ''

    num_parallel_calls = 8
    src_train = data_input.ImageNetInput(
        dataset_name=FLAGS.source_dataset,
        is_training=True,
        data_dir=data_dir,
        transpose_input=params['transpose_input'],
        cache=False,
        image_size=params['image_size'],
        num_parallel_calls=num_parallel_calls,
        use_bfloat16=use_bfloat16,
        num_classes=num_classes)
    finetune_dataset = data_input.ImageNetInput(
        dataset_name=FLAGS.target_dataset,
        task_id=1,
        is_training=True,
        data_dir=data_dir,
        dataset_split='l2l_train',
        transpose_input=params['transpose_input'],
        cache=False,
        image_size=params['image_size'],
        num_parallel_calls=num_parallel_calls,
        use_bfloat16=use_bfloat16)
    target_dataset = data_input.ImageNetInput(
        dataset_name=FLAGS.target_dataset,
        task_id=2,
        is_training=True,
        data_dir=data_dir,
        dataset_split='l2l_valid',
        transpose_input=params['transpose_input'],
        cache=False,
        image_size=params['image_size'],
        num_parallel_calls=num_parallel_calls,
        use_bfloat16=use_bfloat16)

    train_params = dict(params)
    train_params['batch_size'] = int(
        round(params['batch_size'] * FLAGS.train_batch_size_multiplier))

    train_data = src_train.input_fn(train_params)

    target_train_params = dict(params)
    target_train_params['batch_size'] = int(
        round(params['batch_size'] * FLAGS.target_train_batch_multiplier))
    finetune_data = finetune_dataset.input_fn(target_train_params)

    target_params = dict(params)
    target_params['batch_size'] = int(
        round(params['batch_size'] * FLAGS.target_batch_multiplier))

    target_data = target_dataset.input_fn(target_params)
    dataset = tf.data.Dataset.zip((train_data, finetune_data, target_data))
    dataset = dataset.map(_merge_datasets)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

  if FLAGS.mode == 'train':
    max_train_steps = FLAGS.train_steps

    resnet_classifier.train(make_input_dataset, max_steps=max_train_steps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
