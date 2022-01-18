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

"""Wrapper for generative models used to derive intrinsic rewards.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from dopamine.discrete_domains import atari_lib
import gin
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim


PSEUDO_COUNT_QUANTIZATION_FACTOR = 8
PSEUDO_COUNT_OBSERVATION_SHAPE = (42, 42)
NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE


@slim.add_arg_scope
def masked_conv2d(inputs, num_outputs, kernel_size,
                  activation_fn=tf.nn.relu,
                  weights_initializer=tf.initializers.glorot_normal(),
                  biases_initializer=tf.initializers.zeros(),
                  stride=(1, 1),
                  scope=None,
                  mask_type='A',
                  collection=None,
                  output_multiplier=1):
  """Creates masked convolutions used in PixelCNN.

  There are two types of masked convolutions, type A and B, see Figure 1 in
  https://arxiv.org/abs/1606.05328 for more details.

  Args:
    inputs: input image.
    num_outputs: int, number of filters used in the convolution.
    kernel_size: int, size of convolution kernel.
    activation_fn: activation function used after the convolution.
    weights_initializer: distribution used to initialize the kernel.
    biases_initializer: distribution used to initialize biases.
    stride: convolution stride.
    scope: name of the tensorflow scope.
    mask_type: type of masked convolution, must be A or B.
    collection: tf variables collection.
    output_multiplier: number of convolutional network stacks.

  Returns:
    frame post convolution.
  """
  assert mask_type in ('A', 'B') and num_outputs % output_multiplier == 0
  num_inputs = int(inputs.get_shape()[-1])
  kernel_shape = tuple(kernel_size) + (num_inputs, num_outputs)
  strides = (1,) + tuple(stride) + (1,)
  biases_shape = [num_outputs]

  mask_list = [np.zeros(
      tuple(kernel_size) + (num_inputs, num_outputs // output_multiplier),
      dtype=np.float32) for _ in range(output_multiplier)]
  for i in range(output_multiplier):
    # Mask type A
    if kernel_shape[0] > 1:
      mask_list[i][:kernel_shape[0]//2] = 1.0
    if kernel_shape[1] > 1:
      mask_list[i][kernel_shape[0]//2, :kernel_shape[1]//2] = 1.0
    # Mask type B
    if mask_type == 'B':
      mask_list[i][kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
  mask_values = np.concatenate(mask_list, axis=3)

  with tf.variable_scope(scope):
    w = tf.get_variable('W', kernel_shape, trainable=True,
                        initializer=weights_initializer)
    b = tf.get_variable('biases', biases_shape, trainable=True,
                        initializer=biases_initializer)
    if collection is not None:
      tf.add_to_collection(collection, w)
      tf.add_to_collection(collection, b)

    mask = tf.constant(mask_values, dtype=tf.float32)
    mask.set_shape(kernel_shape)

    convolution = tf.nn.conv2d(inputs, mask * w, strides, padding='SAME')
    convolution_bias = tf.nn.bias_add(convolution, b)

    if activation_fn is not None:
      convolution_bias = activation_fn(convolution_bias)
  return convolution_bias


def gating_layer(x, embedding, hidden_units, scope_name=''):
  """Create the gating layer used in the PixelCNN architecture."""
  with tf.variable_scope(scope_name):
    out = masked_conv2d(x, 2*hidden_units, [3, 3],
                        mask_type='B',
                        activation_fn=None,
                        output_multiplier=2,
                        scope='masked_conv')
    out += slim.conv2d(embedding, 2*hidden_units, [1, 1],
                       activation_fn=None)
    out = tf.reshape(out, [-1, 2])
    out = tf.tanh(out[:, 0]) + tf.sigmoid(out[:, 1])
  return tf.reshape(out, x.get_shape())


@gin.configurable
class CTSIntrinsicReward(object):
  """Class used to instantiate a CTS density model used for exploration."""

  def __init__(self,
               reward_scale,
               convolutional=False,
               observation_shape=PSEUDO_COUNT_OBSERVATION_SHAPE,
               quantization_factor=PSEUDO_COUNT_QUANTIZATION_FACTOR):
    """Constructor.

    Args:
      reward_scale: float, scale factor applied to the raw rewards.
      convolutional: bool, whether to use convolutional CTS.
      observation_shape: tuple, 2D dimensions of the observation predicted
        by the model. Needs to be square.
      quantization_factor: int, number of bits for the predicted image
    Raises:
      ValueError: when the `observation_shape` is not square.
    """
    self._reward_scale = reward_scale
    if  (len(observation_shape) != 2
         or observation_shape[0] != observation_shape[1]):
      raise ValueError('Observation shape needs to be square')
    self._observation_shape = observation_shape
    self.density_model = shannon.CTSTensorModel(
        observation_shape, convolutional)
    self._quantization_factor = quantization_factor

  def update(self, observation):
    """Updates the density model with the given observation.

    Args:
      observation: Input frame.

    Returns:
      Update log-probability.
    """
    input_tensor = self._preprocess(observation)
    return self.density_model.Update(input_tensor)

  def compute_intrinsic_reward(self, observation, training_steps, eval_mode):
    """Updates the model, returns the intrinsic reward.

    Args:
      observation: Input frame. For compatibility with other models, this
        may have a batch-size of 1 as its first dimension.
      training_steps: int, number of training steps.
      eval_mode: bool, whether or not running eval mode.

    Returns:
      The corresponding intrinsic reward.
    """
    del training_steps
    input_tensor = self._preprocess(observation)
    if not eval_mode:
      log_rho_t = self.density_model.Update(input_tensor)
      log_rho_tp1 = self.density_model.LogProb(input_tensor)
      ipd = log_rho_tp1 - log_rho_t
    else:
      # Do not update the density model in evaluation mode
      ipd = self.density_model.IPD(input_tensor)

    # Compute the pseudo count
    ipd_clipped = min(ipd, 25)
    inv_pseudo_count = max(0, math.expm1(ipd_clipped))
    reward = float(self._reward_scale) * math.sqrt(inv_pseudo_count)
    return reward

  def _preprocess(self, observation):
    """Converts the given observation into something the model can use.

    Args:
      observation: Input frame.

    Returns:
      Processed frame.

    Raises:
      ValueError: If observation provided is not 2D.
    """
    if observation.ndim != 2:
      raise ValueError('Observation needs to be 2D.')
    input_tensor = cv2.resize(observation,
                              self._observation_shape,
                              interpolation=cv2.INTER_AREA)
    input_tensor //= (256 // self._quantization_factor)
    # Convert to signed int (this may be unpleasantly inefficient).
    input_tensor = input_tensor.astype('i', copy=False)
    return input_tensor


@gin.configurable
class PixelCNNIntrinsicReward(object):
  """PixelCNN class to instantiate a bonus using a PixelCNN density model."""

  def __init__(self,
               sess,
               reward_scale,
               ipd_scale,
               observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
               resize_shape=PSEUDO_COUNT_OBSERVATION_SHAPE,
               quantization_factor=PSEUDO_COUNT_QUANTIZATION_FACTOR,
               tf_device='/cpu:*',
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.0001,
                   momentum=0.9,
                   epsilon=0.0001)):
    self._sess = sess
    self.reward_scale = reward_scale
    self.ipd_scale = ipd_scale
    self.observation_shape = observation_shape
    self.resize_shape = resize_shape
    self.quantization_factor = quantization_factor
    self.optimizer = optimizer

    with tf.device(tf_device), tf.name_scope('intrinsic_pixelcnn'):
      observation_shape = (1,) + observation_shape + (1,)
      self.obs_ph = tf.placeholder(tf.uint8, shape=observation_shape,
                                   name='obs_ph')
      self.preproccessed_obs = self._preprocess(self.obs_ph, resize_shape)
      self.iter_ph = tf.placeholder(tf.uint32, shape=[], name='iter_num')
      self.eval_ph = tf.placeholder(tf.bool, shape=[], name='eval_mode')
      self.network = tf.make_template('PixelCNN', self._network_template)
      self.ipd = tf.cond(tf.logical_not(self.eval_ph),
                         self.update,
                         self.virtual_update)
      self.reward = self.ipd_to_reward(self.ipd, self.iter_ph)

  def compute_intrinsic_reward(self, observation, training_steps, eval_mode):
    """Updates the model (during training), returns the intrinsic reward.

    Args:
      observation: Input frame. For compatibility with other models, this
        may have a batch-size of 1 as its first dimension.
      training_steps: Number of training steps, int.
      eval_mode: bool, whether or not running eval mode.

    Returns:
      The corresponding intrinsic reward.
    """
    observation = observation[np.newaxis, :, :, np.newaxis]
    return self._sess.run(self.reward, {self.obs_ph: observation,
                                        self.iter_ph: training_steps,
                                        self.eval_ph: eval_mode})

  def _preprocess(self, obs, obs_shape):
    """Preprocess the input."""
    obs = tf.cast(obs, tf.float32)
    obs = tf.image.resize_bilinear(obs, obs_shape)
    denom = tf.constant(256 // self.quantization_factor, dtype=tf.float32)
    return tf.floordiv(obs, denom)

  @gin.configurable
  def _network_template(self, obs, num_layers, hidden_units):
    """PixelCNN network architecture."""
    with slim.arg_scope([slim.conv2d, masked_conv2d],
                        weights_initializer=tf.variance_scaling_initializer(
                            distribution='uniform'),
                        biases_initializer=tf.constant_initializer(0.0)):
      net = masked_conv2d(obs, hidden_units, [7, 7], mask_type='A',
                          activation_fn=None, scope='masked_conv_1')

      embedding = slim.model_variable(
          'embedding',
          shape=(1,) + self.resize_shape + (4,),
          initializer=tf.variance_scaling_initializer(
              distribution='uniform'))
      for i in range(1, num_layers + 1):
        net2 = gating_layer(net, embedding, hidden_units,
                            'gating_{}'.format(i))
        net += masked_conv2d(net2, hidden_units, [1, 1],
                             mask_type='B',
                             activation_fn=None,
                             scope='masked_conv_{}'.format(i+1))

      net += slim.conv2d(embedding, hidden_units, [1, 1],
                         activation_fn=None)
      net = tf.nn.relu(net)
      net = masked_conv2d(net, 64, [1, 1], scope='1x1_conv_out',
                          mask_type='B',
                          activation_fn=tf.nn.relu)
      logits = masked_conv2d(net, self.quantization_factor, [1, 1],
                             scope='logits', mask_type='B',
                             activation_fn=None)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.cast(obs, tf.int32),
        logits=logits,
        reduction=tf.losses.Reduction.MEAN)
    return collections.namedtuple('PixelCNN_network', ['logits', 'loss'])(
        logits, loss)

  def update(self):
    """Computes the log likehood difference and update the density model."""
    with tf.name_scope('update'):
      with tf.name_scope('pre_update'):
        loss = self.network(self.preproccessed_obs).loss

      train_op = self.optimizer.minimize(loss)

      with tf.name_scope('post_update'), tf.control_dependencies([train_op]):
        loss_post_training = self.network(self.preproccessed_obs).loss
        ipd = (loss - loss_post_training) * (
            self.resize_shape[0] * self.resize_shape[1])
    return ipd

  def virtual_update(self):
    """Computes the log likelihood difference without updating the network."""
    with tf.name_scope('virtual_update'):
      with tf.name_scope('pre_update'):
        loss = self.network(self.preproccessed_obs).loss

      grads_and_vars = self.optimizer.compute_gradients(loss)
      model_vars = [gv[1] for gv in grads_and_vars]
      saved_vars = [tf.Variable(v.initialized_value()) for v in model_vars]
      backup_op = tf.group(*[t.assign(s)
                             for t, s in zip(saved_vars, model_vars)])
      with tf.control_dependencies([backup_op]):
        train_op = self.optimizer.apply_gradients(grads_and_vars)
      with tf.control_dependencies([train_op]), tf.name_scope('post_update'):
        loss_post_training = self.network(self.preproccessed_obs).loss
      with tf.control_dependencies([loss_post_training]):
        restore_op = tf.group(*[d.assign(s)
                                for d, s in zip(model_vars, saved_vars)])
      with tf.control_dependencies([restore_op]):
        ipd = (loss - loss_post_training) * \
              self.resize_shape[0] * self.resize_shape[1]
      return ipd

  def ipd_to_reward(self, ipd, steps):
    """Computes the intrinsic reward from IPD."""
    # Prediction gain decay
    ipd = self.ipd_scale * ipd / tf.sqrt(tf.to_float(steps))
    inv_pseudo_count = tf.maximum(tf.expm1(ipd), 0.0)
    return self.reward_scale * tf.sqrt(inv_pseudo_count)


@gin.configurable
class RNDIntrinsicReward(object):
  """Class used to instantiate a bonus using random network distillation."""

  def __init__(self,
               sess,
               embedding_size=512,
               observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
               tf_device='/gpu:0',
               reward_scale=1.0,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.0001,
                   epsilon=0.00001),
               summary_writer=None):
    self.embedding_size = embedding_size
    self.reward_scale = reward_scale
    self.optimizer = optimizer
    self._sess = sess
    self.summary_writer = summary_writer

    with tf.device(tf_device), tf.name_scope('intrinsic_rnd'):
      obs_shape = (1,) + observation_shape + (1,)
      self.iter_ph = tf.placeholder(tf.uint64, shape=[], name='iter_num')
      self.iter = tf.cast(self.iter_ph, tf.float32)
      self.obs_ph = tf.placeholder(tf.uint8, shape=obs_shape,
                                   name='obs_ph')
      self.eval_ph = tf.placeholder(tf.bool, shape=[], name='eval_mode')
      self.obs = tf.cast(self.obs_ph, tf.float32)
      # Placeholder for running mean and std of observations and rewards
      self.obs_mean = tf.Variable(tf.zeros(shape=obs_shape),
                                  trainable=False,
                                  name='obs_mean',
                                  dtype=tf.float32)
      self.obs_std = tf.Variable(tf.ones(shape=obs_shape),
                                 trainable=False,
                                 name='obs_std',
                                 dtype=tf.float32)
      self.reward_mean = tf.Variable(tf.zeros(shape=[]),
                                     trainable=False,
                                     name='reward_mean',
                                     dtype=tf.float32)
      self.reward_std = tf.Variable(tf.ones(shape=[]),
                                    trainable=False,
                                    name='reward_std',
                                    dtype=tf.float32)
      self.obs = self._preprocess(self.obs)
      self.target_embedding = self._target_network(self.obs)
      self.prediction_embedding = self._prediction_network(self.obs)
      self._train_op = self._build_train_op()

  def _preprocess(self, obs):
    return tf.clip_by_value((obs - self.obs_mean) / self.obs_std, -5.0, 5.0)

  def compute_intrinsic_reward(self, obs, training_step, eval_mode=False):
    """Computes the RND intrinsic reward."""
    obs = obs[np.newaxis, :, :, np.newaxis]
    to_evaluate = [self.intrinsic_reward]
    if not eval_mode:
      # Also update the prediction network
      to_evaluate.append(self._train_op)
    reward = self._sess.run(to_evaluate,
                            {self.obs_ph: obs,
                             self.iter_ph: training_step,
                             self.eval_ph: eval_mode})[0]
    return self.reward_scale * float(reward)

  def _target_network(self, obs):
    """Implements the random target network used by RND."""
    with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable=False,
                        weights_initializer=tf.orthogonal_initializer(
                            gain=np.sqrt(2)),
                        biases_initializer=tf.zeros_initializer()):
      net = slim.conv2d(obs, 32, [8, 8], stride=4,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.conv2d(net, 64, [4, 4], stride=2,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.conv2d(net, 64, [3, 3], stride=1,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.flatten(net)
      embedding = slim.fully_connected(net, self.embedding_size,
                                       activation_fn=None)
    return embedding

  def _prediction_network(self, obs):
    """Prediction network used by RND to predict to target network output."""
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.orthogonal_initializer(
                            gain=np.sqrt(2)),
                        biases_initializer=tf.zeros_initializer()):
      net = slim.conv2d(obs, 32, [8, 8], stride=4,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.conv2d(net, 64, [4, 4], stride=2,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.conv2d(net, 64, [3, 3], stride=1,
                        activation_fn=tf.nn.leaky_relu)
      net = slim.flatten(net)
      net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
      net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
      embedding = slim.fully_connected(net, self.embedding_size,
                                       activation_fn=None)
    return embedding

  def _update_moments(self):
    """Update the moments estimates, assumes a batch size of 1."""
    def update():
      """Update moment function passed later to a tf.cond."""
      moments = [
          (self.obs, self.obs_mean, self.obs_std),
          (self.loss, self.reward_mean, self.reward_std)
      ]
      ops = []
      for value, mean, std in moments:
        delta = value - mean
        assign_mean = mean.assign_add(delta / self.iter)
        std_ = std * self.iter + (delta ** 2) * self.iter / (self.iter + 1)
        assign_std = std.assign(std_ / (self.iter + 1))
        ops.extend([assign_mean, assign_std])
      return ops

    return tf.cond(
        tf.logical_not(self.eval_ph),
        update,
        # false_fn must have the same number and type of outputs.
        lambda: 4 * [tf.constant(0., tf.float32)])

  def _build_train_op(self):
    """Returns train op to update the prediction network."""
    prediction = self.prediction_embedding
    target = tf.stop_gradient(self.target_embedding)
    self.loss = tf.losses.mean_squared_error(
        target, prediction, reduction=tf.losses.Reduction.MEAN)
    with tf.control_dependencies(self._update_moments()):
      self.intrinsic_reward = (self.loss - self.reward_mean) / self.reward_std
    return self.optimizer.minimize(self.loss)
