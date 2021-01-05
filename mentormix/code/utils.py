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

# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for training the MentorNet models."""

import collections

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp


MentorNetNetHParams = collections.namedtuple(
    'MentorNetNetHParams', 'label_embedding_size, epoch_embedding_size, '
    'num_label_embedding, num_fc_nodes')


def summarize_data_utilization(v, tf_global_step, batch_size, epsilon=0.001):
  """Summarizes the samples of non-zero weights during training.

  Args:
    v: a tensor [batch_size, 1] represents the sample weights.
      0: loss, 1: loss difference to the moving average, 2: label and 3: epoch,
      where epoch is an integer between 0 and 99 (the first and the last epoch).
    tf_global_step: the tensor of the current global step.
    batch_size: an integer batch_size
    epsilon: the rounding error. If the weight is smaller than epsilon then set
      it to zero.
  Returns:
    data_util: a tensor of data utilization.
  """
  nonzero_v = tf.get_variable('data_util/nonzero_v', [],
                              initializer=tf.zeros_initializer(),
                              trainable=False,
                              dtype=tf.float32)

  rounded_v = tf.maximum(v - epsilon, tf.to_float(0))

  # Log data utilization
  nonzero_v = tf.assign_add(nonzero_v, tf.count_nonzero(
      rounded_v, dtype=tf.float32))

  data_util = (nonzero_v) / tf.to_float(batch_size) / (
      tf.to_float(tf_global_step) + 2)
  data_util = tf.minimum(data_util, 1)
  tf.stop_gradient(data_util)

  slim.summaries.add_scalar_summary(data_util, 'data_util/data_util')
  slim.summaries.add_scalar_summary(tf.reduce_sum(v), 'data_util/batch_sum_v')
  return data_util


def parse_dropout_rate_list(str_list):
  """Parse a comma-separated string to a list.

  The format follows [dropout rate, epoch_num]+ and the result is a list of 100
  dropout rate.

  Args:
    str_list: the input string.
  Returns:
    result: the converted list
  """
  str_list = np.array(str_list)
  values = str_list[np.arange(0, len(str_list), 2)]
  indexes = str_list[np.arange(1, len(str_list), 2)]

  values = [float(t) for t in values]
  indexes = [int(t) for t in indexes]

  assert len(values) == len(indexes) and np.sum(indexes) == 100
  for t in values:
    assert t >= 0.0 and t <= 1.0

  result = []
  for t in range(len(str_list) // 2):
    result.extend([values[t]] * indexes[t])
  return result


def mentornet_nn(input_features,
                 label_embedding_size=2,
                 epoch_embedding_size=5,
                 num_label_embedding=2,
                 num_fc_nodes=20):
  """The neural network form of the MentorNet.

  An implementation of the mentornet. The details are in:
  Jiang, Lu, et al. "MentorNet: Learning Data-Driven Curriculum for Very Deep
  Neural Networks on Corrupted Labels." ICML. 2018.

  Args:
    input_features: a [batch_size, 4] tensor. Each dimension corresponds to
      0: loss, 1: loss difference to the moving average, 2: label and 3: epoch,
      where epoch is an integer between 0 and 99 (the first and the last epoch).
    label_embedding_size: the embedding size for the label feature.
    epoch_embedding_size: the embedding size for the epoch feature.
    num_label_embedding: the number of different labels.
    num_fc_nodes: number of hidden nodes in the fc layer.
  Returns:
    v: [batch_size, 1] weight vector.
  """
  batch_size = int(input_features.get_shape()[0])

  losses = tf.reshape(input_features[:, 0], [-1, 1])
  loss_diffs = tf.reshape(input_features[:, 1], [-1, 1])
  labels = tf.to_int32(tf.reshape(input_features[:, 2], [-1, 1]))
  epochs = tf.to_int32(tf.reshape(input_features[:, 3], [-1, 1]))
  epochs = tf.minimum(epochs, tf.ones([batch_size, 1], dtype=tf.int32) * 99)

  if len(losses.get_shape()) <= 1:
    num_steps = 1
  else:
    num_steps = int(losses.get_shape()[1])

  with tf.variable_scope('mentornet', reuse=tf.AUTO_REUSE):
    label_embedding = tf.get_variable(
        'label_embedding', [num_label_embedding, label_embedding_size])
    epoch_embedding = tf.get_variable(
        'epoch_embedding', [100, epoch_embedding_size])

    lstm_inputs = tf.stack([losses, loss_diffs], axis=1)
    lstm_inputs = tf.squeeze(lstm_inputs)
    lstm_inputs = [lstm_inputs]

    forward_cell = tf.contrib.rnn.BasicLSTMCell(1, forget_bias=0.0)
    backward_cell = tf.contrib.rnn.BasicLSTMCell(1, forget_bias=0.0)

    _, out_state_fw, out_state_bw = tf.contrib.rnn.static_bidirectional_rnn(
        forward_cell,
        backward_cell,
        inputs=lstm_inputs,
        dtype=tf.float32,
        sequence_length=np.ones(batch_size) * num_steps)

    label_inputs = tf.squeeze(tf.nn.embedding_lookup(label_embedding, labels))
    epoch_inputs = tf.squeeze(tf.nn.embedding_lookup(epoch_embedding, epochs))

    h = tf.concat([out_state_fw[0], out_state_bw[0]], 1)
    feat = tf.concat([label_inputs, epoch_inputs, h], 1)
    feat_dim = int(feat.get_shape()[1])

    fc_1 = tf.add(
        tf.matmul(
            feat,
            tf.get_variable(
                'Variable',
                initializer=tf.random_normal([feat_dim, num_fc_nodes]))),
        tf.get_variable(
            'Variable_1', initializer=tf.random_normal([num_fc_nodes])))
    fc_1 = tf.nn.tanh(fc_1)
    # Output layer with linear activation
    out_layer = tf.matmul(
        fc_1,
        tf.get_variable(
            'Variable_2', initializer=tf.random_normal([num_fc_nodes, 1])) +
        tf.get_variable('Variable_3', initializer=tf.random_normal([1])))
    return out_layer


def loss_thresholding_function(loss, para_lambda=0.75):
  """The simplest MentorNet is a loss thresholding function.

  Args:
    loss: [batch_size, 1] the loss vector.
    para_lambda: the age parameter, in [0,1], indicates the percentile in a
      mini-batch, where 1 indicates selecting all samples and 0 is selecting
      zero samples.
  Returns:
    v: [batch_size, 1] weight vector.
  """
  assert para_lambda >= 0 and para_lambda <= 1
  with tf.variable_scope('mentornet/thresholding'):

    one_weights = tf.ones(tf.shape(loss), tf.float32)
    zero_weights = tf.zeros(tf.shape(loss), tf.float32)

    percentile_loss = tfp.stats.percentile(
        loss, int(para_lambda * 100))
    # Replace this with loss moving average to get better results.
    percentile_loss = tf.reshape(percentile_loss, [1])

    weights = tf.where(loss >= percentile_loss, zero_weights, one_weights)
    v = tf.reshape(weights, [-1, 1], name='v')
  return v


def mentornet(epoch,
              loss,
              labels,
              loss_p_percentile,
              example_dropout_rates,
              burn_in_epoch=18,
              fixed_epoch_after_burn_in=False,
              loss_moving_average_decay=0.5,
              mentornet_net_hparams=None,
              avg_name='cumulative',
              debug=False):
  """The MentorNet to train with the StudentNet.

     The details are in:
    Jiang, Lu, et al. "MentorNet: Learning Data-Driven Curriculum for Very Deep
    Neural Networks on Corrupted Labels." ICML. 2018.

  Args:
    epoch: a tensor [batch_size, 1] representing the training percentage. Each
      epoch is an integer between 0 and 99.
    loss: a tensor [batch_size, 1] representing the sample loss.
    labels: a tensor [batch_size, 1] representing the label. Every label is set
      to 0 in the current version.
    loss_p_percentile: a 1-d tensor of size 100, where each element is the
      p-percentile at that epoch to compute the moving average.
    example_dropout_rates: a 1-d tensor of size 100, where each element is the
      dropout rate at that epoch. Dropping out means the probability of setting
      sample weights to zeros proposed in Liang, Junwei, et al. "Learning to
      Detect Concepts from Webly-Labeled Video Data." IJCAI. 2016.
    burn_in_epoch: the number of burn_in_epoch. In the first burn_in_epoch, all
      samples have 1.0 weights.
    fixed_epoch_after_burn_in: whether to fix the epoch after the burn-in.
    loss_moving_average_decay: the decay factor to compute the moving average.
    mentornet_net_hparams: mentornet hyperparameters.
    avg_name: name of the loss moving average variable.
    debug: whether to print the weight information for debugging purposes.

  Returns:
    v: [batch_size, 1] weight vector.
  """
  with tf.variable_scope('mentor_inputs'):
    loss_moving_avg = tf.get_variable(
        avg_name, [], initializer=tf.zeros_initializer(), trainable=False)

    if not fixed_epoch_after_burn_in:
      cur_epoch = epoch
    else:
      cur_epoch = tf.to_int32(tf.minimum(epoch, burn_in_epoch))

    v_ones = tf.ones(tf.shape(loss), tf.float32)
    v_zeros = tf.zeros(tf.shape(loss), tf.float32)
    upper_bound = tf.cond(cur_epoch < (burn_in_epoch - 1), lambda: v_ones,
                          lambda: v_zeros)

    this_dropout_rate = tf.squeeze(
        tf.nn.embedding_lookup(example_dropout_rates, cur_epoch))
    this_percentile = tf.squeeze(
        tf.nn.embedding_lookup(loss_p_percentile, cur_epoch))

    percentile_loss = tf.contrib.distributions.percentile(
        loss, this_percentile * 100)
    percentile_loss = tf.convert_to_tensor(percentile_loss)

    loss_moving_avg = loss_moving_avg.assign(
        loss_moving_avg * loss_moving_average_decay +
        (1 - loss_moving_average_decay) * percentile_loss)

    slim.summaries.add_scalar_summary(percentile_loss,
                                      '{}/percentile_loss'.format(avg_name))
    slim.summaries.add_scalar_summary(cur_epoch,
                                      '{}/percentile_loss'.format(avg_name))
    slim.summaries.add_scalar_summary(loss_moving_avg,
                                      '{}/percentile_loss'.format(avg_name))

    ones = tf.ones([tf.shape(loss)[0], 1], tf.float32)

    epoch_vec = tf.scalar_mul(tf.to_float(cur_epoch), ones)
    lossdiff = loss - tf.scalar_mul(loss_moving_avg, ones)

  input_data = tf.squeeze(tf.stack([loss, lossdiff, labels, epoch_vec], 1))
  hparams = mentornet_net_hparams
  if hparams:
    v = tf.nn.sigmoid(
        mentornet_nn(
            input_data,
            label_embedding_size=hparams.label_embedding_size,
            epoch_embedding_size=hparams.epoch_embedding_size,
            num_label_embedding=hparams.num_label_embedding,
            num_fc_nodes=hparams.num_fc_nodes),
        name='v')
  else:
    v = tf.nn.sigmoid(mentornet_nn(input_data), name='v')
  # Force select all samples in the first burn_in_epochs
  v = tf.maximum(v, upper_bound, 'v_bound')

  v_dropout = tf.py_func(probabilistic_sample,
                         [v, this_dropout_rate, 'random'], tf.float32)
  v_dropout = tf.reshape(v_dropout, [-1, 1], name='v_dropout')

  # Print information in the debug mode.
  if debug:
    v_dropout = tf.Print(
        v_dropout,
        data=[cur_epoch, loss_moving_avg, percentile_loss],
        summarize=64,
        message='epoch, loss_moving_avg, percentile_loss')
    v_dropout = tf.Print(
        v_dropout, data=[lossdiff], summarize=64, message='loss_diff')
    v_dropout = tf.Print(v_dropout, data=[v], summarize=64, message='v')
    v_dropout = tf.Print(
        v_dropout, data=[v_dropout], summarize=64, message='v_dropout')
  return v_dropout


def mentor_mix_up(x, l, v, beta_param):
  """MentorMix method.

  Args:
    x: the input image batch [batch_size, H, W, C]
    l: the label batch  [batch_size, num_of_class]
    v: mentornet weights
    beta_param: the parameter to sample the weight average.
  Returns:
    result: The mixed images and label batches.
  """
  if beta_param <= 0:
    return x, l

  v_flat = tf.reshape(v, [-1])
  dist = tfp.distributions.Categorical(probs=tf.nn.softmax(v_flat))
  idx = dist.sample(tf.shape(x)[0])

  x2 = tf.gather(x, idx)
  l2 = tf.gather(l, idx)

  mix = tf.distributions.Beta(beta_param,
                              beta_param).sample([tf.shape(x)[0], 1, 1, 1])

  mix = tf.maximum(mix, 1 - mix)
  mix = tf.where(v_flat >= 0.5, mix, 1 - mix)

  xmix = x * mix + x2 * (1 - mix)
  lmix = l * mix[:, :, 0, 0] + l2 * (1 - mix[:, :, 0, 0])
  v = tf.stop_gradient(v)
  v_flat = tf.stop_gradient(v_flat)
  xmix = tf.stop_gradient(xmix)
  lmix = tf.stop_gradient(lmix)
  return xmix, lmix


def probabilistic_sample(v, rate=0.5, mode='binary'):
  """Implement the sampling techniques.

  Args:
    v: [batch_size, 1] the weight column vector.
    rate: in [0,1]. 0 indicates using all samples and 1 indicates
      using zero samples.
    mode: a string. One of the following 1) actual returns the actual sampling;
      2) binary returns binary weights; 3) random performs random sampling.
  Returns:
    v: [batch_size, 1] weight vector.
  """
  assert rate >= 0 and rate <= 1
  epsilon = 1e-5
  with tf.variable_scope('mentornet/prob_sampling'):
    p = np.copy(v)
    p = np.reshape(p, -1)
    if mode == 'random':
      ids = np.random.choice(
          p.shape[0], int(p.shape[0] * (1 - rate)), replace=False)
    else:
      # Avoid 1) all zero loss and 2) zero loss are never selected.
      p += epsilon
      p /= np.sum(p)
      ids = np.random.choice(
          p.shape[0], int(p.shape[0] * (1 - rate)), p=p, replace=False)
    result = np.zeros(v.shape, dtype=np.float32)
    if mode == 'binary':
      result[ids, 0] = 1
    else:
      result[ids, 0] = v[ids, 0]
    return result


def get_mentornet_network_hyperparameter(checkpoint_path):
  """Get MentorNet network configuration from the checkpoint file.

  Args:
    checkpoint_path: the file path to restore MentorNet.

  Returns:
    a named tuple MentorNetNetHParams.
  """
  if checkpoint_path and tf.gfile.IsDirectory(checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

  reader = tf.train.load_checkpoint(checkpoint_path)
  var_to_shape_map = reader.get_variable_to_shape_map()

  label_embedding_size = 2
  epoch_embedding_size = 5
  num_label_embedding = 2
  num_fc_nodes = 20

  key = 'mentornet/epoch_embedding'
  if key in var_to_shape_map:
    epoch_embedding_size = reader.get_tensor(key).shape[1]

  key = 'mentornet/label_embedding'
  if key in var_to_shape_map:
    num_label_embedding = reader.get_tensor(key).shape[0]
    label_embedding_size = reader.get_tensor(key).shape[1]

  # FC layer.
  key = 'mentornet/Variable'
  if key in var_to_shape_map:
    num_fc_nodes = reader.get_tensor(key).shape[1]

  hparams = MentorNetNetHParams(
      label_embedding_size=label_embedding_size,
      epoch_embedding_size=epoch_embedding_size,
      num_label_embedding=num_label_embedding,
      num_fc_nodes=num_fc_nodes)

  return hparams
