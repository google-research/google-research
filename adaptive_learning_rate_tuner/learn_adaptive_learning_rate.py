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

"""Adaptive learning rate tuner."""

from absl import app
from absl import flags

import numpy as np
import resnet_model_fast as resnet_model
from tensorflow import keras
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

# Controller params.
flags.DEFINE_float('actor_learning_rate', 0.001, 'Actor learning rate.')
flags.DEFINE_float('critic_learning_rate', 0.01, 'Critic learning rate.')
flags.DEFINE_float('gradient_clipping_l2', 5.0, 'Gradient clipping l2 norm.')
flags.DEFINE_integer('same_batch_update_steps', 10, 'same_batch_update_steps.')
flags.DEFINE_string('actor_model', 'mlp', 'Support mlp and lstm.')

# Trainee model params.
flags.DEFINE_string('trainee_model', 'mlp', 'mlp, cnn, resnet.')
flags.DEFINE_integer('train_steps', 5, 'train_steps.')
flags.DEFINE_integer('num_episode', 5, 'num_episode.')
flags.DEFINE_float('default_learning_rate', 0.01, 'default_learning_rate')

# Reward function related flags.
flags.DEFINE_string('reward_discount', 'Backward',
                    'None, Forward, Backward, Final, Mixed')
flags.DEFINE_string('reward_baseline', 'Fixed',
                    'Fixed, Average, Exponential, Ratio')
flags.DEFINE_float('fixed_baseline_value', 2.0, 'fixed baseline value.')
flags.DEFINE_float('reward_numerator', 2.0, 'baseline_numerator.')
flags.DEFINE_integer('keep_lr_interval', 1, 'keep lr the same in the interval.')

# Adjust the controller output.
flags.DEFINE_float('clip_lower_bound', -0.2, 'clip lower bound.')
flags.DEFINE_float('clip_upper_bound', 0.5, 'clip upper bound.')
flags.DEFINE_float('lr_shift', 0.15, 'learning rate shift, minus this value.')
flags.DEFINE_float('lr_scale', 0.1, 'learning rate scale, multiply this value.')
flags.DEFINE_float('action_variance', 0.1, 'action_variance.')


GAMMA = 0.9  # Reward discount in TD error.
EPSILON = 0.2  # Hyperparameter for surrogate objective.
LOG_BASE = 1e-10  # For numeric stability.


def compute_accuracy(prediction, label):
  """Compute accuracy given predictions and labels."""
  final_prediction = tf.argmax(prediction, 1)
  final_prediction = tf.cast(final_prediction, tf.int32)
  label = tf.cast(label, tf.int32)
  label = tf.squeeze(label)
  correct_prediction = tf.equal(final_prediction, label)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy


def build_cnn_model(features, reuse):
  """Model function for LeNet CNN."""
  input_shape = [-1, 28, 28, 1]
  final_input_dim = 7 * 7 * 64
  input_layer = tf.reshape(features, input_shape)

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      reuse=reuse,
      name='conv1')
  pool1 = tf.layers.max_pooling2d(
      inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      reuse=reuse,
      name='conv2')
  pool2 = tf.layers.max_pooling2d(
      inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')

  pool2_flat = tf.reshape(pool2, [-1, final_input_dim])
  dense = tf.layers.dense(
      inputs=pool2_flat,
      units=512,
      activation=tf.nn.relu,
      reuse=reuse,
      name='w1')
  logits = tf.layers.dense(inputs=dense, units=10, reuse=reuse, name='w2')
  return logits


def build_mlp_model(features, reuse):
  """MLP model."""
  dense = tf.layers.dense(
      features, 128, activation=tf.nn.relu, name='w1', reuse=reuse)
  logits = tf.layers.dense(
      dense, 10, activation=None, name='w2', reuse=reuse)
  return logits


class AdaptiveTuner(object):
  """Adaptive learning rate tuner class."""

  def __init__(
      self,
      sess,
      n_actions,
      n_features,
      actor_lr=0.0001,
      critic_lr=0.0002,
      proposed_learning_rate=0.01,
      learn_scale=False,
  ):
    self.sess = sess
    self.n_actions = n_actions
    self.n_features = n_features
    self.actor_lr = actor_lr
    self.critic_lr = critic_lr
    self.proposed_learning_rate = proposed_learning_rate
    self.max_nn_output = 0
    self.min_nn_output = 10
    self.best_validation_loss = 10
    self.obs = tf.placeholder(tf.float32, [None, n_features], 'observation')
    self.action = tf.placeholder(tf.float32, None, 'action')
    self.actor_adv = tf.placeholder(tf.float32, [None, 1], 'actor_advantage')
    self.learn_scale = learn_scale

    with tf.variable_scope('critic_scope'):
      layer1 = tf.layers.dense(self.obs, 32, tf.nn.relu)
      self.value = tf.layers.dense(layer1, 1)
      self.discounted_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
      self.critic_adv = self.discounted_r - self.value
      self.critic_loss = tf.reduce_mean(tf.square(self.critic_adv))
      self.critic_train_op = tf.train.AdamOptimizer(critic_lr).minimize(
          self.critic_loss)

    dist, dist_params = self.build_action_model('dist', trainable=True)
    dist_old, dist_old_params = self.build_action_model(
        'dist_old', trainable=False)
    self.sample_action = tf.squeeze(dist.sample(1), axis=0)

    with tf.variable_scope('update_distribution'):
      self.update_dist_op = [
          old_prob.assign(new_prob)
          for new_prob, old_prob in zip(dist_params, dist_old_params)
      ]

    with tf.variable_scope('actor_loss'):
      new_action_prob = dist.prob(self.action) + LOG_BASE
      old_action_prob = dist_old.prob(self.action) + LOG_BASE
      prob_ratio = tf.exp(tf.log(new_action_prob) - tf.log(old_action_prob))
      surrogate_loss = prob_ratio * self.actor_adv
      self.actor_loss = -tf.reduce_mean(
          tf.minimum(
              surrogate_loss,
              tf.clip_by_value(prob_ratio, 1.0 - EPSILON, 1.0 + EPSILON) *
              self.actor_adv))

    with tf.variable_scope('actor_train'):
      actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr)
      actor_gvs = actor_optimizer.compute_gradients(self.actor_loss)
      actor_capped_gvs = [(tf.clip_by_norm(grad, FLAGS.gradient_clipping_l2)
                           if grad is not None else grad, var)
                          for grad, var in actor_gvs]
      self.actor_train_op = actor_optimizer.apply_gradients(actor_capped_gvs)

    self.sess.run(tf.global_variables_initializer())

  def build_action_model(self, name, trainable):
    """Build action model to propose new learning rate distribution."""
    with tf.variable_scope(name):
      if FLAGS.actor_model == 'mlp':
        layer1 = tf.layers.dense(
            inputs=self.obs,
            units=32,
            activation=tf.nn.relu,
            trainable=trainable,
            name='layer1')
      elif FLAGS.actor_model == 'lstm':
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(32)
        output, _ = tf.nn.static_rnn(rnn_cell, [self.obs], dtype=tf.float32)
        layer1 = output[-1]
      else:
        raise ValueError('Wrong actor_model flag value.')

      # Mean
      nn_mu = tf.layers.dense(
          inputs=layer1,
          units=self.n_actions,
          activation=None,
          trainable=trainable,
          name='mean')

      # Variance
      scale = FLAGS.action_variance
      if self.learn_scale:
        sigma_layer1 = tf.layers.dense(
            inputs=self.obs,
            units=32,
            activation=tf.nn.relu,
            trainable=trainable,
            name='sigma_layer1')
        scale = tf.layers.dense(
            inputs=sigma_layer1,
            units=self.n_actions,
            activation=tf.nn.softplus,
            trainable=trainable,
            name='variance')
      norm_dist = tf.distributions.Normal(loc=nn_mu, scale=scale)
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return norm_dist, params

  def update(self, observation, action, reward):
    """Update both actor and critic networks."""
    self.sess.run(self.update_dist_op)
    advantage = self.sess.run(self.critic_adv, {
        self.obs: observation,
        self.discounted_r: reward
    })
    for _ in range(FLAGS.same_batch_update_steps):
      self.sess.run(self.actor_train_op, {
          self.obs: observation,
          self.action: action,
          self.actor_adv: advantage})
      self.sess.run(self.critic_train_op, {
          self.obs: observation,
          self.discounted_r: reward
      })

  def choose_action(self, observation):
    """Choose new learning rate based on current state."""
    act_lr = self.sess.run(self.sample_action,
                           {self.obs: observation[np.newaxis, :]})[0]
    # Choose the final learning rate
    lr_multiplier = act_lr[0]
    lr_multiplier = np.clip(lr_multiplier, FLAGS.clip_lower_bound,
                            FLAGS.clip_upper_bound)
    lr_multiplier = (lr_multiplier - FLAGS.lr_shift) * FLAGS.lr_scale + 1
    self.proposed_learning_rate *= lr_multiplier
    return self.proposed_learning_rate, lr_multiplier

  def get_value(self, observation):
    """Return value given current observations."""
    if observation.ndim < 2: observation = observation[np.newaxis, :]
    return self.sess.run(self.value, {self.obs: observation})


def main(unused_argv):

  train_steps = FLAGS.train_steps
  num_episode = FLAGS.num_episode
  trainee_model = FLAGS.trainee_model

  best_validation_loss = 10

  batch_size = 1000
  train_size = 50000
  valid_size = 10000
  test_batch_num = 10
  if trainee_model == 'resnet':
    test_size = 1000
  else:
    test_size = 10000

  fashion_mnist = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images,
                                 test_labels) = fashion_mnist.load_data()
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  # Different input tensor shape for different trainee models.
  if trainee_model == 'mlp':
    num_pixels = train_images.shape[1] * train_images.shape[2]
    x_train = train_images.reshape(train_images.shape[0],
                                   num_pixels).astype('float32')
    train_set = x_train[0:train_size, :]
    train_label = train_labels[0:train_size].astype('int64')
    valid_set = x_train[train_size:, :]
    valid_label = train_labels[train_size:].astype('int64')

    test_set = test_images.reshape(test_images.shape[0],
                                   num_pixels).astype('float32')
    test_label = test_labels.astype('int64')
  elif trainee_model == 'cnn' or trainee_model == 'resnet':
    x_train = train_images.reshape(train_images.shape[0],
                                   train_images.shape[1],
                                   train_images.shape[2], 1).astype('float32')
    train_set = x_train[0:train_size, :, :, :]
    train_label = train_labels[0:train_size].astype('int64')
    valid_set = x_train[train_size:, :, :, :]
    valid_label = train_labels[train_size:].astype('int64')

    test_set = test_images.reshape(test_images.shape[0],
                                   test_images.shape[1],
                                   test_images.shape[2], 1).astype('float32')
    test_label = test_labels.astype('int64')

  init_observation = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  observation_dim = init_observation.shape[0]

  adaptive_tuner_sess = tf.Session()
  adaptive_tuner = AdaptiveTuner(
      sess=adaptive_tuner_sess,
      n_features=observation_dim,
      n_actions=1,
      actor_lr=FLAGS.actor_learning_rate,
      critic_lr=FLAGS.critic_learning_rate,
      proposed_learning_rate=FLAGS.default_learning_rate)

  # Log the training process.
  running_reward = 0
  valid_loss_matrix = np.zeros((num_episode, train_steps))
  train_loss_matrix = np.zeros((num_episode, train_steps))
  valid_accuracy_matrix = np.zeros((num_episode, train_steps))
  test_loss_matrix = np.zeros((num_episode, train_steps))
  test_accuracy_matrix = np.zeros((num_episode, train_steps))
  reward_matrix = np.zeros((num_episode, train_steps))
  observation_matrix = np.zeros((num_episode, train_steps, observation_dim))
  learning_rate_matrix = np.zeros((num_episode, train_steps))
  running_reward_array = np.zeros(num_episode)

  gnn = tf.Graph()
  with gnn.as_default():

    # Prepare train/valid/test split for Fashion Mnist
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_set, train_label)).repeat().batch(batch_size)
    train_iter = tf.data.make_one_shot_iterator(dataset)
    feature, label = train_iter.get_next()
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_set, valid_label)).repeat().batch(valid_size)
    valid_iter = tf.data.make_one_shot_iterator(valid_dataset)
    valid_feature, valid_label = valid_iter.get_next()
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_set, test_label)).repeat().batch(test_size)
    test_iter = tf.data.make_one_shot_iterator(test_dataset)
    test_feature, test_label = test_iter.get_next()

    learned_learning_rate = tf.placeholder(tf.float32, shape=[])

    # Build trainee model.
    if trainee_model == 'mlp':
      train_logits = build_mlp_model(feature, reuse=False)
      valid_logits = build_mlp_model(valid_feature, reuse=True)
      test_logits = build_mlp_model(test_feature, reuse=True)
    elif trainee_model == 'cnn':
      train_logits = build_cnn_model(feature, reuse=False)
      valid_logits = build_cnn_model(valid_feature, reuse=True)
      test_logits = build_cnn_model(test_feature, reuse=True)
    elif trainee_model == 'resnet':
      resnet_size = 18
      resnet_18 = resnet_model.FastCifar10Model(
          resnet_size=resnet_size, data_format='channels_first')
      train_logits = resnet_18(feature, True)
      train_logits = tf.cast(train_logits, tf.float32)
      valid_logits = resnet_18(valid_feature, False)
      valid_logits = tf.cast(valid_logits, tf.float32)
      test_logits = resnet_18(test_feature, False)
      test_logits = tf.cast(test_logits, tf.float32)
    else:
      raise ValueError('Wrong trainee_model flag value.')

    prediction = tf.nn.softmax(train_logits)
    valid_prediction = tf.nn.softmax(valid_logits)
    test_prediction = tf.nn.softmax(test_logits)

    if trainee_model == 'resnet':
      w1 = tf.get_default_graph().get_tensor_by_name(
          'resnet_model/dense/kernel:0')
      w2 = tf.get_default_graph().get_tensor_by_name(
          'resnet_model/dense/bias:0')
    else:
      w1 = tf.get_default_graph().get_tensor_by_name('w1/kernel:0')
      w2 = tf.get_default_graph().get_tensor_by_name('w2/kernel:0')

    mean_w1, var_w1 = tf.nn.moments(w1, axes=[0, 1])
    if trainee_model == 'resnet':
      mean_w2, var_w2 = tf.nn.moments(w2, axes=[0])
    else:
      mean_w2, var_w2 = tf.nn.moments(w2, axes=[0, 1])

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=label, logits=train_logits)

    train_op = tf.train.AdamOptimizer(
        learning_rate=learned_learning_rate).minimize(loss)

    if trainee_model == 'resnet':
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = tf.group([train_op, update_ops])

    valid_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=valid_label, logits=valid_logits)
    valid_accuracy = compute_accuracy(valid_prediction, valid_label)
    test_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=test_label, logits=test_logits)
    test_accuracy = compute_accuracy(test_prediction, test_label)

  for i_episode in range(num_episode):

    obs_t, action_t, reward_t, obs_t_1 = [], [], [], []
    best_target_valid_loss = 10.0
    with tf.Session(config=tf.ConfigProto(), graph=gnn) as sess:
      sess.run(tf.global_variables_initializer())
      observation = init_observation
      prev_prediction = np.random.rand(batch_size, 10)
      prediction_log_var_ema = 0
      prediction_change_log_var_ema = 0
      reward = 0
      num_reward = 0
      average_reward = 0
      reward_sum = 0
      keep_lr_interval = 1
      exp_decay = 0.8
      action = 0.0
      track_reward = []
      adaptive_tuner.proposed_learning_rate = FLAGS.default_learning_rate
      for i in range(train_steps):
        if keep_lr_interval == 1 or i > train_steps - 10:
          action, action_multiplier = adaptive_tuner.choose_action(observation)
          keep_lr_interval = FLAGS.keep_lr_interval
          learning_rate_matrix[i_episode, i] = action

          _, loss_value, prediction_value, valid_loss_value, valid_accuracy_value, mean_w1_value, var_w1_value, mean_w2_value, var_w2_value = (
              sess.run([
                  train_op, loss, prediction, valid_loss, valid_accuracy,
                  mean_w1, var_w1, mean_w2, var_w2
              ],
                       feed_dict={learned_learning_rate: action}))

          if valid_loss_value < best_target_valid_loss:
            best_target_valid_loss = valid_loss_value
            if trainee_model == 'resnet':
              total_test_loss = 0.0
              total_test_accuracy = 0.0
              for _ in range(test_batch_num):
                batch_test_loss_value, batch_test_accuracy_value = sess.run(
                    [test_loss, test_accuracy])
                total_test_loss += batch_test_loss_value
                total_test_accuracy += batch_test_accuracy_value
              test_loss_value = total_test_loss / test_batch_num
              test_accuracy_value = total_test_accuracy / test_batch_num
            else:
              test_loss_value, test_accuracy_value = sess.run(
                  [test_loss, test_accuracy])

            test_loss_matrix[i_episode, i] = test_loss_value
            test_accuracy_matrix[i_episode, i] = test_accuracy_value

          train_loss_matrix[i_episode, i] = loss_value
          valid_loss_matrix[i_episode, i] = valid_loss_value
          valid_accuracy_matrix[i_episode, i] = valid_accuracy_value
          diff_prediction = prediction_value - prev_prediction

          if prediction_log_var_ema == 0:
            prediction_log_var_ema = np.log(np.var(prediction_value))
          else:
            prediction_log_var_ema = exp_decay * prediction_log_var_ema + (
                1 - exp_decay) * np.log(np.var(prediction_value) + LOG_BASE)

          if prediction_change_log_var_ema == 0:
            prediction_change_log_var_ema = np.log(np.var(diff_prediction))
          else:
            prediction_change_log_var_ema = exp_decay * prediction_change_log_var_ema + (
                1 - exp_decay) * np.log(np.var(diff_prediction) + LOG_BASE)

          # Collect all state observations.
          new_observation = np.array([
              valid_loss_value, prediction_log_var_ema,
              prediction_change_log_var_ema, loss_value, mean_w1_value,
              var_w1_value, mean_w2_value, var_w2_value, action
          ])
          observation_matrix[i_episode, i, :] = observation
          # Different reward functions.
          if FLAGS.reward_baseline == 'Fixed':
            reward = FLAGS.fixed_baseline_value - valid_loss_value
          elif FLAGS.reward_baseline == 'Average':
            reward_sum += valid_loss_value
            num_reward += 1
            average_reward = float(reward_sum) / num_reward
            reward = average_reward - valid_loss_value
          elif FLAGS.reward_baseline == 'Exponential':
            if average_reward == 0:
              average_reward = valid_loss_value
            else:
              average_reward = average_reward * exp_decay + (
                  valid_loss_value * (1 - exp_decay))
            reward = average_reward - valid_loss_value
          elif FLAGS.reward_baseline == 'Ratio':
            reward = FLAGS.reward_numerator / valid_loss_value - FLAGS.fixed_baseline_value
          else:
            raise ValueError('Wrong reward_baseline flag value.')

          reward_matrix[i_episode, i] = reward

          track_reward.append(reward)
          obs_t.append(observation)
          action_t.append(np.squeeze(action_multiplier))
          reward_t.append(reward)
          obs_t_1.append(new_observation)
          observation = new_observation
          prev_prediction = prediction_value
        else:
          keep_lr_interval -= 1
          sess.run([train_op], feed_dict={learned_learning_rate: action})

        if i == train_steps - 1:
          current_best_validation_loss = valid_loss_matrix[i_episode, i]
          if current_best_validation_loss < best_validation_loss:
            best_validation_loss = current_best_validation_loss

          obs_t_ts, action_t_ts, reward_t_ts, obs_t_1_ts = np.vstack(
              obs_t), np.vstack(action_t), np.vstack(reward_t), np.vstack(
                  obs_t_1)
          value_t = adaptive_tuner.get_value(obs_t_1_ts)
          td_reward = reward_t_ts + GAMMA * value_t
          adaptive_tuner.update(obs_t_ts, action_t_ts, td_reward)
          obs_t, action_t, reward_t, obs_t_1 = [], [], [], []
          episode_reward = sum(track_reward)
          if running_reward == 0:
            running_reward = episode_reward
          else:
            running_reward = running_reward * 0.99 + episode_reward * 0.01
          running_reward_array[i_episode] = running_reward


if __name__ == '__main__':
  tf.disable_v2_behavior()  # Disable eager mode when running with TF2.
  app.run(main)
