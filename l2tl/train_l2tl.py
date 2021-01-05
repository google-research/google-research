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

"""Trains an L2TL model jointly on the source and target datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl import app
from absl import flags
import model
import model_utils
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir',
    None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))
flags.DEFINE_integer(
    'log_step_count_steps', 200, 'The number of steps at '
    'which the global step information is logged.')
flags.DEFINE_string(
    'warm_start_ckpt_path', None, 'The path to the checkpoint '
    'that will be used before training.')
flags.DEFINE_integer('train_steps', 120000, 'Number of total training steps.')
flags.DEFINE_integer('num_choices', 100,
                     'Number of actions for the scaling variable.')
flags.DEFINE_float('base_learning_rate_scale', 0.001,
                   'The value of the learning rate')
flags.DEFINE_float('dst_weight_decay', 0.0005,
                   'Weight decay for the target dataset.')
flags.DEFINE_integer('save_checkpoints_steps', 100,
                     'Number of steps for each checkpoint saving.')
flags.DEFINE_float('rl_learning_rate', 0.001, 'Learning rate for RL updates.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for l2tl.')
flags.DEFINE_integer('target_num_classes', 10,
                     'The number of classes in the target dataset.')
flags.DEFINE_integer('train_batch_size', 128, 'The batch size during training.')
flags.DEFINE_integer(
    'source_train_batch_multiplier', 5,
    'The multiplier will be used to increase the batch size '
    'to sample more examples.')
flags.DEFINE_float('loss_weight_scale', 1000.0, 'Scaling of the loss weight.')
flags.DEFINE_integer('first_pretrain_steps', 0,
                     'Number of steps for pretraining.')
flags.DEFINE_integer('target_val_batch_multiplier', 4,
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
      collections=[tf.GraphKeys.GLOBAL_VARIABLES])
  return global_step


def get_src_train_op(loss):  # pylint: disable=unused-argument
  """Returns the source training op."""
  global_step = tf.train.get_global_step()
  src_learning_rate = FLAGS.learning_rate
  src_learning_rate = tf.train.piecewise_constant(global_step, [
      800,
  ], [FLAGS.learning_rate, FLAGS.learning_rate * 0.1])
  optimizer = tf.train.MomentumOptimizer(
      learning_rate=src_learning_rate, momentum=0.9, use_nesterov=True)
  with tf.variable_scope('src'):
    return optimizer.minimize(loss, global_step), src_learning_rate


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
  rl_step_baseline = rl_reward
  rl_baseline_momentum = 0.9
  rl_entropy_regularization = 0.001

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
  rl_learning_rate = tf.train.piecewise_constant(target_global_step, [
      800,
  ], [rl_learning_rate, rl_learning_rate * 0.1])

  optimizer = tf.train.AdamOptimizer(rl_learning_rate)
  target_train_op = optimizer.minimize(
      rl_empirical_loss,
      target_global_step,
      var_list=tf.trainable_variables(rl_scope.name))

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


def get_logits(feature, mode, dataset_name, reuse=None):
  """Returns the network logits."""
  avg_pool = model.conv_model(
      feature,
      mode,
      target_dataset=FLAGS.target_dataset,
      src_hw=FLAGS.src_hw,
      target_hw=FLAGS.target_hw,
      dataset_name=dataset_name,
      reuse=reuse)
  return avg_pool


def do_cls(avg_pool, num_classes, name='dense'):
  """Applies classification."""
  with tf.variable_scope('target_CLS', reuse=tf.AUTO_REUSE):
    logits = tf.layers.dense(
        inputs=avg_pool,
        units=num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=.05),
        name=name)
    return logits


def get_model_logits(src_features, finetune_features, mode, num_classes,
                     target_num_classes):
  """Gets the logits from different models."""
  src_avg_pool = get_logits(
      src_features, mode, FLAGS.source_dataset, reuse=None)
  dst_avg_pool = get_logits(
      finetune_features, mode, FLAGS.target_dataset, reuse=True)

  src_logits = do_cls(src_avg_pool, num_classes, name='final_dense_dst')
  dst_logits = do_cls(
      dst_avg_pool, target_num_classes, name='final_target_dense')
  return src_logits, dst_logits


def get_final_loss(src_logits, src_one_hot_labels, dst_logits,
                   finetune_one_hot_labels, global_step, loss_weights,
                   inst_weights):
  """Gets the final loss for l2tl."""
  if FLAGS.uniform_weight:
    inst_weights = 1.0

  def get_loss(logits, inst_weights, one_hot_labels):
    """Returns the loss function."""
    loss = tf.losses.softmax_cross_entropy(
        logits=logits, weights=inst_weights, onehot_labels=one_hot_labels)
    return loss

  src_loss = get_loss(src_logits, inst_weights, src_one_hot_labels)
  dst_loss = get_loss(dst_logits, 1., finetune_one_hot_labels)
  l2_loss = []
  for v in tf.trainable_variables():
    if 'batch_normalization' not in v.name and 'rl_controller' not in v.name:
      l2_loss.append(tf.nn.l2_loss(v))
  l2_loss = FLAGS.dst_weight_decay * tf.add_n(l2_loss)

  enable_pretrain = tf.cast(
      tf.greater_equal(global_step, FLAGS.first_pretrain_steps), tf.float32)

  loss = src_loss * tf.stop_gradient(loss_weights) * enable_pretrain
  loss += dst_loss + l2_loss

  return tf.identity(loss), src_loss, dst_loss


def train_model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
  """Defines the model function."""
  target_num_classes = FLAGS.target_num_classes
  global_step = tf.train.get_global_step()

  src_features, src_labels = features['src'], tf.cast(labels['src'], tf.int64)
  finetune_features = features['finetune']
  target_features = features['target']

  num_classes = FLAGS.src_num_classes

  finetune_one_hot_labels = tf.one_hot(
      tf.cast(labels['finetune'], tf.int64), target_num_classes)
  target_one_hot_labels = tf.one_hot(
      tf.cast(labels['target'], tf.int64), target_num_classes)

  with tf.variable_scope('rl_controller') as rl_scope:
    # It creates a `rl_scope` which will be used for ops.
    pass
  rl_entropy, label_weights, log_prob = rl_label_weights(rl_scope)
  loss_entropy, loss_weights, loss_log_prob = get_loss_weights(rl_scope)

  def gather_init_weights():
    inst_weights = tf.stop_gradient(tf.gather(label_weights, src_labels))
    return inst_weights

  inst_weights = gather_init_weights()
  bs = FLAGS.train_batch_size
  hw = FLAGS.src_hw
  inst_weights, indices = tf.nn.top_k(
      inst_weights,
      k=bs,
      sorted=True,
  )

  src_features = tf.reshape(src_features, [
      bs * FLAGS.source_train_batch_multiplier,
      hw,
      hw,
      1,
  ])
  src_features = tf.gather(src_features, indices, axis=0)
  src_features = tf.stop_gradient(src_features)

  src_labels = tf.gather(src_labels, indices)

  inst_weights = bs * inst_weights / tf.reduce_sum(inst_weights)

  src_one_hot_labels = tf.one_hot(tf.cast(src_labels, tf.int64), num_classes)

  src_logits, dst_logits = get_model_logits(src_features, finetune_features,
                                            mode, num_classes,
                                            target_num_classes)

  loss, _, _ = get_final_loss(src_logits, src_one_hot_labels, dst_logits,
                              finetune_one_hot_labels, global_step,
                              loss_weights, inst_weights)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  with tf.control_dependencies(update_ops):
    src_train_op, _ = get_src_train_op(loss)
    with tf.control_dependencies([src_train_op]):
      target_avg_pool = get_logits(
          target_features, mode, FLAGS.target_dataset, reuse=True)
      target_logits = do_cls(
          target_avg_pool, target_num_classes, name='final_target_dense')
      is_prediction_correct = tf.equal(
          tf.argmax(tf.identity(target_logits), axis=1),
          tf.argmax(target_one_hot_labels, axis=1))
      acc = tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))

      entropy = loss_entropy + rl_entropy
      log_prob = loss_log_prob + log_prob
      train_op, _, _ = meta_train_op(acc, entropy, log_prob, rl_scope, params)

  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def rl_label_weights(name=None):
  """Returns the weight for importance."""
  with tf.variable_scope(name, 'rl_op_selection'):
    num_classes = FLAGS.src_num_classes
    num_choices = FLAGS.num_choices

    logits = tf.get_variable(
        name='logits_rl_w',
        initializer=tf.initializers.zeros(),
        shape=[num_classes, num_choices],
        dtype=tf.float32)
    dist = tfp.distributions.Categorical(logits=logits)
    dist_entropy = tf.reduce_sum(dist.entropy())

    sample = dist.sample()
    sample_masks = 1. * tf.cast(sample, tf.float32) / num_choices
    sample_log_prob = tf.reduce_mean(dist.log_prob(sample))

  return (dist_entropy, sample_masks, sample_log_prob)


def get_loss_weights(name=None):
  """Returns the weight for loss."""
  with tf.variable_scope(name, 'rl_op_selection'):

    logits = tf.get_variable(
        name='loss_logits_rl_w',
        initializer=tf.initializers.zeros(),
        shape=[
            FLAGS.num_choices,
        ],
        dtype=tf.float32)
    dist = tfp.distributions.Categorical(logits=logits)
    dist_entropy = tf.reduce_sum(dist.entropy())

    sample = dist.sample()
    sample_masks = 1. * tf.cast(sample, tf.float32) / FLAGS.loss_weight_scale
    sample_log_prob = tf.reduce_mean(dist.log_prob(sample))

  return (dist_entropy, sample_masks, sample_log_prob)


def main(unused_argv):
  tf.set_random_seed(FLAGS.random_seed)

  run_config_args = {
      'model_dir': FLAGS.model_dir,
      'save_checkpoints_steps': FLAGS.save_checkpoints_steps,
      'log_step_count_steps': FLAGS.log_step_count_steps,
      'keep_checkpoint_max': 100,
  }
  config = tf.contrib.tpu.RunConfig(**run_config_args)

  if FLAGS.warm_start_ckpt_path:
    var_names = []
    checkpoint_path = FLAGS.warm_start_ckpt_path
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    for key in reader.get_variable_to_shape_map():
      keep_str = 'Momentum|global_step|finetune_global_step'
      if not re.findall('({})'.format(keep_str,), key):
        var_names.append(key)

    tf.logging.info('Warm-starting tensors: %s', sorted(var_names))

    vars_to_warm_start = var_names
    warm_start_settings = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=checkpoint_path,
        vars_to_warm_start=vars_to_warm_start)
  else:
    warm_start_settings = None

  l2tl_classifier = tf.estimator.Estimator(
      train_model_fn, config=config, warm_start_from=warm_start_settings)

  def make_input_dataset():
    """Return input dataset."""

    def _merge_datasets(train_batch, finetune_batch, target_batch):
      """Merge different splits."""
      train_features, train_labels = train_batch['image'], train_batch['label']
      finetune_features, finetune_labels = finetune_batch[
          'image'], finetune_batch['label']
      target_features, target_labels = target_batch['image'], target_batch[
          'label']
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

    source_train_batch_size = int(
        round(FLAGS.train_batch_size * FLAGS.source_train_batch_multiplier))

    train_data = tfds.load(name=FLAGS.source_dataset, split='train')
    train_data = train_data.shuffle(512).repeat().batch(source_train_batch_size)

    target_train_batch_size = int(
        round(FLAGS.train_batch_size * FLAGS.target_train_batch_multiplier))
    finetune_data = tfds.load(name=FLAGS.target_dataset, split='train')
    finetune_data = finetune_data.shuffle(512).repeat().batch(
        target_train_batch_size)

    target_val_batch_size = int(
        round(FLAGS.train_batch_size * FLAGS.target_val_batch_multiplier))

    target_data = tfds.load(name=FLAGS.target_dataset, split='validation')
    target_data = target_data.shuffle(512).repeat().batch(target_val_batch_size)

    dataset = tf.data.Dataset.zip((train_data, finetune_data, target_data))
    dataset = dataset.map(_merge_datasets)
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

  max_train_steps = FLAGS.train_steps
  l2tl_classifier.train(make_input_dataset, max_steps=max_train_steps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
