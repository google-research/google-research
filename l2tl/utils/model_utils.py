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

"""Model utils."""

from absl import flags
import tensorflow as tf  # tf
import tensorflow_probability as tfp

FLAGS = flags.FLAGS
flags.DEFINE_float('ramp_prop', 0.0, 'Proportion for the ramp steps.')
flags.DEFINE_float('constant_prop', 0.0, 'Proportion for the constant steps.')
flags.DEFINE_float('epsilon_prop', 0.0, 'Proportion for the epsilon steps.')
flags.DEFINE_string('metric', 'acc', 'The evaluation metric used.')


def get_branch_v2(name=None, num_branches=0, branch_name='branch_logits_rl_w'):
  """Gets multile branches, not used."""
  with tf.variable_scope(name, 'rl_op_selection'):
    logits = tf.get_variable(
        name=branch_name,
        initializer=tf.initializers.zeros(),
        shape=[2, num_branches],
        dtype=tf.float32)
    output_index = tf.argmax(logits, axis=1)
    dist_logits_list = logits.value()
    dist = tfp.distributions.Categorical(logits=logits)
    dist_entropy = tf.reduce_sum(dist.entropy())

    sample = dist.sample()
    sample_masks = sample
    sample_log_prob = tf.reduce_mean(dist.log_prob(sample))

  return (dist_logits_list, dist_entropy, tf.stop_gradient(sample_masks),
          sample_log_prob, tf.stop_gradient(output_index))


def build_vars(name, input_dim, stddev=0.01, num_branches=5, num_classes=None):
  """Builds dense layer."""
  ws, bs = [], []
  for weight_idx in range(num_branches):
    with tf.variable_scope(name):
      w = tf.get_variable(
          'kernel{}'.format(weight_idx), [input_dim, num_classes],
          initializer=tf.random_normal_initializer(stddev=stddev))
      b = tf.get_variable(
          'bias{}'.format(weight_idx), [num_classes],
          initializer=tf.constant_initializer())
      ws.append(w)
      bs.append(b)
  ws = tf.stack(ws)
  bs = tf.stack(bs)
  return ws, bs


def multi_stage_lr(total_train_steps, target_learning_rate, current_step):
  """Learning rate scheduling."""
  ramp_prop = FLAGS.ramp_prop
  constant_prop = FLAGS.constant_prop
  epsilon_prop = FLAGS.epsilon_prop
  epsilon = 1e-6

  ramp_steps = int(ramp_prop * total_train_steps)
  ramp_lr = target_learning_rate
  if ramp_steps > 0:
    ramp_lr = target_learning_rate * tf.to_float(current_step) / ramp_steps

  constant_steps = int(constant_prop * total_train_steps)
  constant_lr = target_learning_rate

  epsilon_steps = int(epsilon_prop * total_train_steps)
  cosine_steps = (
      total_train_steps - ramp_steps - constant_steps - epsilon_steps)
  assert cosine_steps > 0
  cosine_decay_lr = tf.train.cosine_decay(
      target_learning_rate - epsilon,
      current_step - ramp_steps - constant_steps, cosine_steps) + epsilon

  return tf.piecewise_constant(current_step, [
      ramp_steps, ramp_steps + constant_steps,
      ramp_steps + constant_steps + cosine_steps
  ], [ramp_lr, constant_lr, cosine_decay_lr, epsilon])


def metric_fn(labels, logits):
  """Metric function for evaluation."""
  predictions = tf.argmax(logits, axis=1)
  top_1_accuracy = tf.metrics.accuracy(labels, predictions)
  in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
  top_5_accuracy = tf.metrics.mean(in_top_5)

  return {
      'top_1_accuracy': top_1_accuracy,
      'top_5_accuracy': top_5_accuracy,
  }


def get_label(labels, params, num_classes, batch_size=-1):  # pylint: disable=unused-argument
  """Returns the label."""
  if FLAGS.target_dataset == 'chest' or FLAGS.target_dataset == 'chexpert':
    labels = tf.reshape(labels, [-1, batch_size])
    labels = tf.transpose(labels, [1, 0])
    olabels = tf.to_float(labels)
    if False:  # pylint: disable=using-constant-test
      label_sum = tf.reduce_sum(olabels, axis=1)
      olabels /= tf.reshape(label_sum, (-1, 1))
    one_hot_labels = olabels
  else:
    one_hot_labels = tf.one_hot(tf.cast(labels, tf.int64), num_classes)
  return one_hot_labels


def update_exponential_moving_average(tensor, momentum, name=None):
  """Returns an exponential moving average of `tensor`.

  We will update the moving average every time the returned `tensor` is
  evaluated. A zero-debias will be applied, so we will return unbiased
  estimates during the first few training steps.

  Args:
    tensor: A floating point tensor.
    momentum: A scalar floating point Tensor with the same dtype as `tensor`.
    name: Optional string, the name of the operation in the TensorFlow graph.

  Returns:
    A Tensor with the same shape and dtype as `tensor`.
  """
  with tf.variable_scope(name, 'update_exponential_moving_average',
                         [tensor, momentum]):
    numerator = tf.get_variable(
        'numerator', initializer=0.0, trainable=False, use_resource=True)
    denominator = tf.get_variable(
        'denominator', initializer=0.0, trainable=False, use_resource=True)
    update_ops = [
        numerator.assign(momentum * numerator + (1 - momentum) * tensor),
        denominator.assign(momentum * denominator + (1 - momentum)),
    ]
    with tf.control_dependencies(update_ops):
      return numerator.read_value() / denominator.read_value()
