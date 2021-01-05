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

"""CAQL network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from caql import dual_ibp_method
from caql import dual_method

FLAGS = flags.FLAGS


class CaqlNet(object):
  """CAQL network class."""

  def __init__(self,
               session,
               state_spec,
               action_spec,
               hidden_layers,
               learning_rate,
               learning_rate_action,
               learning_rate_ga,
               batch_size,
               action_maximization_iterations,
               name,
               l2_loss_flag=False,
               simple_lambda_flag=True,
               solver=None,
               sufficient_ascent_flag=False,
               initial_lambda=10.0,
               lambda_max=5e3):
    """Creates CAQL networks.

    Args:
      session: TF session.
      state_spec: tf_agents.specs.array_spec.ArraySpec. Specification for state.
      action_spec: tf_agents.specs.array_spec.ArraySpec. Specification for
        action.
      hidden_layers: list of integers. Number of hidden units for each hidden
        layer.
      learning_rate: float on Q function learning rate.
      learning_rate_action: float on action function learning rate.
      learning_rate_ga: float. Learning rate for gradient ascent optimizer.
      batch_size: int on batch size for training.
      action_maximization_iterations: int on CEM/gradient ascent iterations.
      name: string on name of network.
      l2_loss_flag: bool on using l2 loss.
      simple_lambda_flag: bool on using lambda hinge loss.
      solver: string on inner max optimizer. Supported optimizers are
        "gradient_ascent", "cross_entropy", "ails", "mip".
      sufficient_ascent_flag: bool on using sufficient ascent.
      initial_lambda: float on initial lambda (only for simple_lambda_flag).
      lambda_max: float on lambda upper-bound.
    """
    self._session = session
    self.state_spec = state_spec
    self.action_spec = action_spec
    self.state_dim = state_spec.shape[0]
    self.action_dim = action_spec.shape[0]
    self.action_max = action_spec.maximum
    self.action_min = action_spec.minimum
    self.hidden_layers = hidden_layers
    self.learning_rate = learning_rate
    self.learning_rate_action = learning_rate_action
    self.learning_rate_ga = learning_rate_ga
    self.batch_size = batch_size
    self.action_maximization_iterations = action_maximization_iterations

    self.name = name
    self.lambda_max = lambda_max
    if solver == "ails" or solver == "mip":
      raise ValueError("AILS and MIP solvers are not supported yet.")

    # define placeholders
    self._state_tensor = tf.placeholder(
        dtype=tf.float32, name="state_tensor", shape=(None, self.state_dim))
    self._state_deviation_tensor = tf.placeholder(
        dtype=tf.float32,
        name="state_deviation_tensor",
        shape=(None, self.state_dim))
    self._action_tensor = tf.placeholder(
        dtype=tf.float32, name="action_tensor", shape=(None, self.action_dim))
    self._next_state_tensor = tf.placeholder(
        dtype=tf.float32,
        name="next_state_tensor",
        shape=(None, self.state_dim))
    self._reward_tensor = tf.placeholder(
        dtype=tf.float32, name="reward_tensor", shape=(None, 1))
    self._done_tensor = tf.placeholder(
        dtype=tf.bool, name="done_tensor", shape=(None, 1))
    self._discount_factor = tf.placeholder(
        dtype=tf.float32, name="discounting_factor", shape=())
    self._maxq_label = tf.placeholder(
        dtype=tf.float32, shape=(None, 1), name="maxq_label")

    self._backup_tensor = self._reward_tensor + (1.0 - tf.to_float(
        self._done_tensor)) * self._discount_factor * self._maxq_label

    self._true_label = tf.placeholder(
        dtype=tf.float32, shape=(None, 1), name="true_label")

    self.q_function_network = self._build_q_function_net(
        self._state_tensor, self._action_tensor)
    self.state_perturbed_q_function_network = self.q_function_network \
        + tf.expand_dims(tf.einsum("ij,ij->i",
                                   tf.gradients(self.q_function_network,
                                                self._state_tensor)[0],
                                   self._state_deviation_tensor),
                         axis=-1)

    self._td_rmse = tf.sqrt(
        tf.losses.mean_squared_error(
            self._reward_tensor + (1.0 - tf.to_float(self._done_tensor)) *
            self._discount_factor * self._maxq_label, self.q_function_network))

    if simple_lambda_flag:
      with tf.variable_scope("{}_{}".format(self.name, "lambda_function")):
        lambda_var = tf.Variable(
            initial_value=initial_lambda, trainable=True, name="lambda_var")
        self.lambda_function_network = tf.tile(
            tf.reshape(
                tf.minimum(
                    lambda_max, tf.maximum(0.0, lambda_var),
                    name="lambda_proj"), (-1, 1)), (self.batch_size, 1))
    else:
      self.lambda_function_network = self._build_lambda_function_net(
          self._state_tensor, self._action_tensor)

    # define loss
    if l2_loss_flag:
      self._q_function_loss = tf.losses.mean_squared_error(
          self._true_label, self.q_function_network)
    else:
      self._q_function_loss = tf.reduce_mean(
          self.q_function_network + self.lambda_function_network *
          tf.maximum(0.0, self._true_label - self.q_function_network))

    self._lambda_function_loss = tf.reduce_mean(
        -self.lambda_function_network *
        (self._true_label - self.q_function_network))

    # Action network to learn argmax of Q
    self._best_q_label = tf.placeholder(
        dtype=tf.float32, shape=(None, 1), name="best_q_label")

    # create network placeholders
    self._create_network_var_ph()

    self.action_function_network = self._build_action_function_net(
        self._state_tensor)
    self.dummy_q_function_network = self._build_q_function_net(
        self._state_tensor, self.action_function_network)

    self._action_function_loss = tf.losses.mean_squared_error(
        self._best_q_label, self.dummy_q_function_network)

    # optimizer
    # NOTE: Increment this by one by inlcuding it only in main_q trainer.
    global_step = tf.Variable(
        0, name="{}_global_step".format(self.name), trainable=False)
    with tf.variable_scope("{}_{}".format(self.name, "optimizer")):
      self._action_function_optimizer = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate).minimize(
              self._action_function_loss,
              var_list=tf.trainable_variables("{}_{}".format(
                  self.name, "action_function")))
      self._q_function_optimizer = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate).minimize(
              self._q_function_loss,
              global_step=global_step,
              var_list=tf.trainable_variables("{}_{}".format(
                  self.name, "q_function")))
      self._lambda_function_optimizer = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate).minimize(
              self._lambda_function_loss,
              var_list=tf.trainable_variables("{}_{}".format(
                  self.name, "lambda_function")))

    # Tensors for dual solvers
    self._create_dual_maxq_label_tensor()
    self._create_dual_active_constraint_condition_tensor()

    self.solver = solver
    self.sufficient_ascent_flag = sufficient_ascent_flag

  def _create_network_var_ph(self):
    """Create network variable placeholders."""
    self._dummy_network_var_ph = {}
    self._vars_tf = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope="{}_{}".format(self.name, "q_function"))
    for _, var in enumerate(self._vars_tf):
      # define placeholder for weights
      self._dummy_network_var_ph["{}_ph".format(var.name)] = tf.placeholder(
          dtype=tf.float32, shape=var.shape)

  def _create_cross_entropy_action_tensors(self,
                                           num_samples=200,
                                           top_k_portion=0.5):
    """Create tensorflow operations for cross_entropy max_actions."""
    top_k_num = int(top_k_portion * num_samples)

    self._dynamic_batch_size = tf.placeholder(
        dtype=tf.int32, name="dynamic_batch_size")
    self._action_init_tensor = tf.placeholder(
        dtype=tf.float32,
        name="action_init_tensor",
        shape=(None, self.action_dim))
    self._tolerance_tensor = tf.placeholder(
        dtype=tf.float32, name="tolerance_tensor", shape=())

    sample_mean_init = self._action_init_tensor
    sample_covariance_diag_init = tf.ones_like(self._action_init_tensor)
    top_k_value_init = tf.constant(
        [np.inf]) * tf.ones(shape=(self._dynamic_batch_size, 1))
    top_k_action_samples_init = tf.tile(
        tf.expand_dims(tf.zeros_like(self._action_init_tensor), axis=1),
        [1, top_k_num, 1])
    random_sampler = tfp.distributions.MultivariateNormalDiag(
        loc=np.zeros(self.action_dim), scale_diag=np.ones(self.action_dim))

    def cond_cross_entropy(itr, cond_terminate, sample_mean,
                           sample_covariance_diag, top_k_value,
                           top_k_action_samples):
      del sample_mean, sample_covariance_diag, top_k_value, top_k_action_samples
      cond_1 = tf.math.less(itr, self.action_maximization_iterations)
      return tf.math.logical_and(cond_1, tf.logical_not(cond_terminate))

    def body_cross_entropy(itr, cond_terminate, sample_mean,
                           sample_covariance_diag, top_k_value,
                           top_k_action_samples):
      """Function for cross entropy search of actions."""
      del top_k_action_samples
      top_k_value_prev = top_k_value
      batch_sample_mean = tf.reshape(
          tf.tile(sample_mean, [1, num_samples]),
          [self._dynamic_batch_size * num_samples, self.action_dim])
      batch_sample_covariance_diag = tf.reshape(
          tf.tile(sample_covariance_diag, [1, num_samples]),
          [self._dynamic_batch_size * num_samples, self.action_dim])

      action_samples = self._action_projection(
          batch_sample_mean + batch_sample_covariance_diag * tf.cast(
              random_sampler.sample(
                  sample_shape=[self._dynamic_batch_size * num_samples]),
              dtype=tf.float32))

      state_samples = tf.reshape(
          tf.tile(self._state_tensor, [1, num_samples]),
          [self._dynamic_batch_size * num_samples, self.state_dim])
      action_samples = tf.reshape(
          action_samples,
          [self._dynamic_batch_size * num_samples, self.action_dim])
      values = tf.reshape(
          self._build_q_function_net(state_samples, action_samples),
          [self._dynamic_batch_size, num_samples])

      # everything is in batch mode
      top_k_index = tf.argsort(
          values, axis=1, direction="DESCENDING")[:, 0:top_k_num]
      top_k_index_1d = tf.reshape(top_k_index,
                                  [self._dynamic_batch_size * top_k_num, 1])
      counter_tensor_1d = tf.reshape(
          tf.tile(
              tf.reshape(
                  tf.range(self._dynamic_batch_size),
                  [self._dynamic_batch_size, 1]), [1, top_k_num]),
          [self._dynamic_batch_size * top_k_num, 1])

      top_k_index_2d = tf.concat([counter_tensor_1d, top_k_index_1d], axis=1)

      action_samples = tf.reshape(
          action_samples,
          [self._dynamic_batch_size, num_samples, self.action_dim])
      top_k_action_samples = tf.gather_nd(action_samples, top_k_index_2d)
      top_k_action_samples = tf.reshape(
          top_k_action_samples,
          [self._dynamic_batch_size, top_k_num, self.action_dim])

      top_k_values = tf.gather_nd(values, top_k_index_2d)
      top_k_values = tf.reshape(top_k_values,
                                [self._dynamic_batch_size, top_k_num])

      # it's a batch_size x 1 tensor
      top_k_value = tf.reshape(
          tf.reduce_mean(top_k_values, axis=1), [self._dynamic_batch_size, 1])

      sample_mean = tf.reduce_mean(top_k_action_samples, axis=1)
      sample_covariance_diag = tf.math.reduce_variance(
          top_k_action_samples, axis=1)

      itr = itr + 1
      cond_terminate = tf.less_equal(
          tf.reduce_mean(tf.math.abs(top_k_value - top_k_value_prev)),
          self._tolerance_tensor)
      return itr, cond_terminate, sample_mean, sample_covariance_diag, \
          top_k_value, top_k_action_samples

    self.cost_optimizer = tf.while_loop(
        cond_cross_entropy, body_cross_entropy, [
            tf.constant(0),
            tf.constant(False), sample_mean_init, sample_covariance_diag_init,
            top_k_value_init, top_k_action_samples_init
        ])

  def _create_gradient_ascent_action_tensors(self, eps=1e-6):
    """Create tensorflow operations for gradient ascent max_actions."""
    self._action_init_tensor = tf.placeholder(
        dtype=tf.float32,
        name="action_init_tensor",
        shape=(None, self.action_dim))
    self._tolerance_tensor = tf.placeholder(
        dtype=tf.float32, name="tolerance_tensor", shape=())

    with tf.variable_scope("{}_{}".format(self.name, "action_variable")):
      self._action_variable_tensor = tf.Variable(
          initial_value=self._action_init_tensor,
          trainable=True,
          name="action_var")

      # gradient ascentd
      self.cost_now = -tf.reduce_mean(
          self._build_q_function_net(self._state_tensor,
                                     self._action_variable_tensor))
      self.action_gradient = tf.gradients(self.cost_now,
                                          self._action_variable_tensor)[0]
      # normalize the gradient
      self.normalized_action_gradient = self.action_gradient / (
          eps + tf.linalg.norm(self.action_gradient))

      if self.sufficient_ascent_flag:

        def cond_sufficient_descent(learning_rate_action,
                                    cond_sufficient_descent, cost_perturbed):
          del cost_perturbed
          cond_1 = tf.math.greater(learning_rate_action,
                                   self.learning_rate_action)
          return tf.math.logical_and(cond_1,
                                     tf.logical_not(cond_sufficient_descent))

        def body_sufficient_descent(learning_rate_action,
                                    cond_sufficient_descent,
                                    cost_perturbed,
                                    c_armijo=0.01,
                                    c_goldstein=0.25,
                                    lr_decay=0.1):
          """Function for sufficient descent."""
          del cond_sufficient_descent, cost_perturbed
          action_variable_perturbed_tensor = self._action_variable_tensor - \
            learning_rate_action * self.normalized_action_gradient

          cost_perturbed = -tf.reduce_mean(
              self._build_q_function_net(self._state_tensor,
                                         action_variable_perturbed_tensor))

          # Here the negative gradient corresponds to maximization of Q fun.
          sufficient_descent = tf.reduce_sum(self.action_gradient *
                                             -self.normalized_action_gradient)

          goldstein_condition = tf.greater_equal(
              cost_perturbed, self.cost_now +
              c_goldstein * learning_rate_action * sufficient_descent)
          armijo_condition = tf.less_equal(
              cost_perturbed, self.cost_now +
              c_armijo * learning_rate_action * sufficient_descent)
          cond_sufficient_descent = tf.logical_and(goldstein_condition,
                                                   armijo_condition)

          with tf.control_dependencies([cond_sufficient_descent]):
            learning_rate_action = learning_rate_action * lr_decay

          return learning_rate_action, cond_sufficient_descent, cost_perturbed

      # Construct the while loop.
      def cond_gradient_ascent(itr, cond_terminate):
        cond_1 = tf.math.less(itr, self.action_maximization_iterations)
        return tf.math.logical_and(cond_1, tf.logical_not(cond_terminate))

      def body_gradient_ascent(itr, cond_terminate, lr_init=100.0):
        """Function for gradient descent."""
        del cond_terminate
        if self.sufficient_ascent_flag:
          # first calculate sufficeint descent
          result_sufficient_descent = tf.while_loop(
              cond_sufficient_descent, body_sufficient_descent,
              [tf.constant(lr_init),
               tf.constant(False),
               tf.constant(np.inf)])
          lr_action = result_sufficient_descent[0]
          cost_perturbed = result_sufficient_descent[2]

          cond_terminate = tf.less_equal(
              tf.math.abs(cost_perturbed - self.cost_now),
              self._tolerance_tensor)
        else:
          # no sufficient descent step
          lr_action = self.learning_rate_ga
          action_variable_perturbed_tensor = self._action_variable_tensor - \
            lr_action * self.normalized_action_gradient

          cost_perturbed = -tf.reduce_mean(
              self._build_q_function_net(self._state_tensor,
                                         action_variable_perturbed_tensor))
          cond_terminate = tf.less_equal(
              tf.math.abs(cost_perturbed - self.cost_now),
              self._tolerance_tensor)

        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=lr_action).apply_gradients(
                grads_and_vars=[(self.normalized_action_gradient,
                                 self._action_variable_tensor)])
        # Ensure that the update is applied before continuing.
        with tf.control_dependencies([train_op]):
          itr = itr + 1

          return itr, cond_terminate

      self.cost_optimizer = tf.while_loop(
          cond_gradient_ascent, body_gradient_ascent,
          [tf.constant(0), tf.constant(False)])

    self.action_init_op = tf.initializers.variables(
        tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope="{}_{}".format(self.name, "action_variable")))

  def _create_dual_maxq_label_tensor(self, method="duality_based"):
    """Approximate the maxq label with dual."""
    w_transpose_list = []
    b_transpose_list = []
    num_layers = 1

    for itr, var in enumerate(self._vars_tf):
      if itr % 2 == 0:
        # even itr, multiplicative weights
        if itr == 0:
          wx_transpose = self._dummy_network_var_ph["{}_ph".format(
              var.name)][:self.state_dim, :]
          w_transpose_list.append(self._dummy_network_var_ph["{}_ph".format(
              var.name)][self.state_dim:, :])

        else:
          w_transpose_list.append(self._dummy_network_var_ph["{}_ph".format(
              var.name)])
        num_layers += 1

      else:
        # odd itr, additive weights
        if itr == 1:
          b_transpose_list.append(
              tf.tile(
                  tf.expand_dims(
                      self._dummy_network_var_ph["{}_ph".format(var.name)],
                      axis=0), [self.batch_size, 1]) +
              tf.matmul(self._next_state_tensor, wx_transpose))
        else:
          b_transpose_list.append(
              tf.tile(
                  tf.expand_dims(
                      self._dummy_network_var_ph["{}_ph".format(var.name)],
                      axis=0), [self.batch_size, 1]))

    action_tensor_center = tf.zeros(shape=[self.batch_size, self.action_dim])

    l_infty_norm_bound = np.max(self.action_max)
    if method == "duality_based":
      self.dual_maxq_tensor = dual_method.create_dual_approx(
          num_layers, self.batch_size, l_infty_norm_bound, w_transpose_list,
          b_transpose_list, action_tensor_center)
    elif method == "ibp":
      # ibp dual solver
      self.dual_maxq_tensor = dual_ibp_method.create_dual_ibp_approx(
          num_layers, self.batch_size, l_infty_norm_bound, w_transpose_list,
          b_transpose_list, action_tensor_center)
    else:
      # mix method
      dual_maxq_tensor = dual_method.create_dual_approx(
          num_layers, self.batch_size, l_infty_norm_bound, w_transpose_list,
          b_transpose_list, action_tensor_center)
      dual_ibp_maxq_tensor = dual_ibp_method.create_dual_ibp_approx(
          num_layers, self.batch_size, l_infty_norm_bound, w_transpose_list,
          b_transpose_list, action_tensor_center)
      # minimum of the upper-bound
      self.dual_maxq_tensor = tf.minimum(dual_maxq_tensor, dual_ibp_maxq_tensor)

  def _create_dual_active_constraint_condition_tensor(self):
    """Create active constraint condition."""

    # It's a 1D boolean tensor with length=batch_size
    self.dual_active_constraint_condition_tensor = tf.reshape(
        tf.math.greater(self._backup_tensor, self.q_function_network), [-1])

  def _action_projection(self, action):
    """Action projection."""
    if isinstance(action, np.ndarray):
      return np.minimum(self.action_spec.maximum,
                        np.maximum(self.action_spec.minimum, action))
    else:
      # tf version
      return tf.minimum(
          self.action_spec.maximum,
          tf.maximum(self.action_spec.minimum, tf.cast(action, tf.float32)))

  def _build_action_function_net(self, state):
    """Build action network."""
    # define network
    with tf.variable_scope(
        "{}_{}".format(self.name, "action_function"),
        reuse=tf.compat.v1.AUTO_REUSE):
      net = tf.layers.flatten(state, name="flatten_0")
      for i, hidden_units in enumerate(self.hidden_layers):
        net = tf.layers.dense(net, hidden_units, name="dense_%d" % i)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
      net = tf.layers.dense(net, self.action_dim, name="action_output")
      # make sure actions are bounded
      net = self._action_projection(net)
      return net

  def _build_q_function_net(self, state, action):
    """Build q_function network."""
    # define network
    with tf.variable_scope(
        "{}_{}".format(self.name, "q_function"), reuse=tf.compat.v1.AUTO_REUSE):
      net = tf.layers.flatten(state, name="q_flatten_0")
      net = tf.concat([net, action], axis=-1)
      for i, hidden_units in enumerate(self.hidden_layers):
        net = tf.layers.dense(
            net, hidden_units, activation=tf.nn.relu, name="q_dense_%d" % i)
      net = tf.layers.dense(net, 1, name="q_output")
      return net

  def _build_lambda_function_net(self, state, action):
    """Build lambda_function network."""
    # define network
    with tf.variable_scope(
        "{}_{}".format(self.name, "lambda_function"),
        reuse=tf.compat.v1.AUTO_REUSE):
      net = tf.layers.flatten(state, name="lambda_flatten_0")
      net = tf.concat([net, action], axis=-1)
      for i, hidden_units in enumerate(self.hidden_layers):
        net = tf.layers.dense(
            net,
            hidden_units,
            activation=tf.nn.relu,
            name="lambda_dense_%d" % i)
      net = tf.layers.dense(net, 1, name="lambda_output")
      net = tf.minimum(
          self.lambda_max,
          tf.maximum(0.0, tf.cast(net, tf.float32)),
          name="lambda_proj")
      return net

  def predict_action_function(self, state):
    """Predict action function.

    Predict the best action for the given state using action function.

    Args:
      state: np.ndarray for state.

    Returns:
      Tensor for the predicted best action for the given `state`.
    """
    state_tensor = np.reshape(state, [-1, self.state_dim])
    return self._session.run(
        self.action_function_network,
        feed_dict={
            self._state_tensor: state_tensor,
        })

  def predict_q_function(self, state, action):
    """Predict Q function.

    Args:
      state: np.ndarray for state.
      action: np.ndarray for action.

    Returns:
      Tensorfor the predicted Q value for the given `state` and `action` pair.
    """
    state_tensor = np.reshape(state, [-1, self.state_dim])
    action_tensor = np.reshape(action, [-1, self.action_dim])
    return self._session.run(
        self.q_function_network,
        feed_dict={
            self._state_tensor: state_tensor,
            self._action_tensor: action_tensor
        })

  def predict_state_perturbed_q_function(self, centroid_states,
                                         centroid_actions, state_deviation):
    """Predict state perturbed Q function.

    Args:
      centroid_states: np.ndarray for centroid states.
      centroid_actions: np.ndarray for the actions of the centroid states.
      state_deviation: np.ndarray for the vector distance between non-centroid
        states and their centroids.

    Returns:
      Tensor for the predicted Q values for the non-centroid states.
    """
    centroid_states = np.reshape(centroid_states, [-1, self.state_dim])
    centroid_actions = np.reshape(centroid_actions, [-1, self.action_dim])
    state_deviation = np.reshape(state_deviation, [-1, self.state_dim])
    return self._session.run(
        self.state_perturbed_q_function_network,
        feed_dict={
            self._state_tensor: centroid_states,
            self._action_tensor: centroid_actions,
            self._state_deviation_tensor: state_deviation
        })

  def predict_lambda_function(self, state, action):
    """Predict lambda function.

    Args:
      state: np.ndarray for state.
      action: np.ndarray for action.

    Returns:
      Tensor for the predicted lambda for the given `state` and `action` pair.
    """
    state_tensor = np.reshape(state, [-1, self.state_dim])
    action_tensor = np.reshape(action, [-1, self.action_dim])

    return self._session.run(
        self.lambda_function_network,
        feed_dict={
            self._state_tensor: state_tensor,
            self._action_tensor: action_tensor
        })

  def compute_backup(self, maxq_labels, rewards, dones, discount_factor):
    """Compute Bellman backup.

    Args:
      maxq_labels: np.ndarray for max-Q labels.
      rewards: np.ndarray for immediate rewards.
      dones: np.ndarray for done flags. True if a state is a terminating state,
        False otherwise.
      discount_factor: float. Discount factor gamma.

    Returns:
      Tensor for TD targets.
    """
    maxq_label = np.reshape(maxq_labels, [-1, 1])
    reward_tensor = np.reshape(rewards, [-1, 1])
    done_tensor = np.reshape(dones, [-1, 1])

    feed = {
        self._maxq_label: maxq_label,
        self._reward_tensor: reward_tensor,
        self._done_tensor: done_tensor,
        self._discount_factor: discount_factor
    }
    return self._session.run(self._backup_tensor, feed_dict=feed)

  def compute_td_rmse(self, states, actions, maxq_labels, rewards, dones,
                      discount_factor):
    """Compute TD rmse.

    Args:
      states: np.ndarray for states.
      actions: np.ndarray for actions.
      maxq_labels: np.ndarray for max-Q labels.
      rewards: np.ndarray for immediate rewards.
      dones: np.ndarray for done flags. True if a state is a terminating state,
        False otherwise.
      discount_factor: float. Discount factor gamma.

    Returns:
      Tensor for TD RMSE.
    """
    state_tensor = np.reshape(states, [-1, self.state_spec.shape[0]])
    action_tensor = np.reshape(actions, [-1, self.action_spec.shape[0]])
    maxq_label = np.reshape(maxq_labels, [-1, 1])
    reward_tensor = np.reshape(rewards, [-1, 1])
    done_tensor = np.reshape(dones, [-1, 1])

    feed = {
        self._state_tensor: state_tensor,
        self._action_tensor: action_tensor,
        self._maxq_label: maxq_label,
        self._reward_tensor: reward_tensor,
        self._done_tensor: done_tensor,
        self._discount_factor: discount_factor
    }
    return self._session.run(self._td_rmse, feed_dict=feed)

  def compute_dual_active_constraint_condition(self, states, actions,
                                               dual_maxq_labels, rewards, dones,
                                               discount_factor):
    """Compute dual active constraint condition.

    Args:
      states: np.ndarray for states.
      actions: np.ndarray for actions.
      dual_maxq_labels: np.ndarray for max-Q labels computed by dual method.
      rewards: np.ndarray for immediate rewards.
      dones: np.ndarray for done flags. True if a state is a terminating state,
        False otherwise.
      discount_factor: float. Discount factor gamma.

    Returns:
      Tensor for bool flags. True if a TD target is larger than a predicted
      Q value for a pair of state and action.
    """
    state_tensor = np.reshape(states, [-1, self.state_dim])
    action_tensor = np.reshape(actions, [-1, self.action_dim])
    dual_maxq_label = np.reshape(dual_maxq_labels, [-1, 1])
    reward_tensor = np.reshape(rewards, [-1, 1])
    done_tensor = np.reshape(dones, [-1, 1])

    feed = {
        self._state_tensor: state_tensor,
        self._action_tensor: action_tensor,
        self._maxq_label: dual_maxq_label,
        self._reward_tensor: reward_tensor,
        self._done_tensor: done_tensor,
        self._discount_factor: discount_factor
    }
    return self._session.run(
        self.dual_active_constraint_condition_tensor, feed_dict=feed)

  def compute_best_actions(self, states, tolerance, warmstart=True,
                           tf_summary_vals=None):
    """Compute best actions.

    Args:
      states: np.ndarray for states.
      tolerance: float. Optimizer tolerance. This is used as a stopping
        condition for the optimizer.
      warmstart: bool on warmstarting flag.
      tf_summary_vals: list to store tf.Summary.Value objects.

    Returns:
      Tensor for the best actions for the given `states`.
    """
    state_tensor = np.reshape(states, [-1, self.state_dim])
    assert len(state_tensor) > 0
    if tf_summary_vals is not None:
      tf_summary_vals.append(
          tf.Summary.Value(tag="tolerance", simple_value=tolerance))

    # profiling the batch action maximization.
    ts_begin = time.time()

    if self.solver == "gradient_ascent":
      if not hasattr(self, "_action_init_tensor"):
        print("Create action variables for gradient ascent.")
        self._create_gradient_ascent_action_tensors()
      best_actions = self.gradient_ascent_best_actions(state_tensor, tolerance,
                                                       warmstart,
                                                       tf_summary_vals)
    elif self.solver == "cross_entropy":
      if not hasattr(self, "_action_init_tensor"):
        print("Create action variables for cross entropy.")
        self._create_cross_entropy_action_tensors()
      best_actions = self.cross_entropy_best_actions(state_tensor, tolerance,
                                                     warmstart, tf_summary_vals)
    elif self.solver == "ails" or self.solver == "mip":
      raise ValueError("AILS and MIP solvers are not supported yet.")
    else:
      raise ValueError("Solver is not implemented!")

    elapsed_in_msecs = int((time.time() - ts_begin) * 1000)
    if tf_summary_vals is not None:
      tf_summary_vals.append(
          tf.Summary.Value(
              tag="batch_maxq/elapsed_msec", simple_value=elapsed_in_msecs))
    return best_actions

  def cross_entropy_best_actions(self, state_tensor, tolerance_tensor,
                                 warmstart, tf_summary_vals=None):
    """Get best action with cross entropy for train network."""
    dynamic_batch_size = len(state_tensor)
    if warmstart:
      action_init_tensor = self.predict_action_function(state_tensor)
    else:
      # randomly sample actions
      action_init_tensor = self.action_min + np.random.rand(
          dynamic_batch_size, self.action_dim) * (
              self.action_max - self.action_min)
    feed = {
        self._state_tensor: state_tensor,
        self._tolerance_tensor: tolerance_tensor,
        self._action_init_tensor: action_init_tensor,
        self._dynamic_batch_size: dynamic_batch_size
    }
    vars_vals = self._session.run(self._vars_tf)
    for var, val in zip(self._vars_tf, vars_vals):
      feed[self._dummy_network_var_ph["{}_ph".format(var.name)]] = val

    # 1) maximize actions through cross entropy
    result = self._session.run(self.cost_optimizer, feed_dict=feed)
    if tf_summary_vals is not None:
      tf_summary_vals.append(
          tf.Summary.Value(tag="batch_maxq/iterations", simple_value=result[0]))

    # itr, cond_terminate, sample_mean, sample_covariance_diag,
    # top_k_value, top_k_actions
    top_k_actions = result[-1]
    return top_k_actions[:, 0, :]

  def gradient_ascent_best_actions(self, state_tensor, tolerance_tensor,
                                   warmstart, tf_summary_vals=None):
    """Get best action with gradient ascent for train network."""
    dynamic_batch_size = len(state_tensor)
    if warmstart:
      action_init_tensor = self.predict_action_function(state_tensor)
    else:
      # randomly sample actions
      action_init_tensor = self.action_min + np.random.rand(
          dynamic_batch_size, self.action_dim) * (
              self.action_max - self.action_min)

    # 1) initialize tensors in feed_dict
    feed = {
        self._state_tensor: state_tensor,
        self._tolerance_tensor: tolerance_tensor,
        self._action_init_tensor: action_init_tensor
    }

    vars_vals = self._session.run(self._vars_tf)
    for var, val in zip(self._vars_tf, vars_vals):
      feed[self._dummy_network_var_ph["{}_ph".format(var.name)]] = val

    # 2) initialize action variable in dummy q_network
    self._session.run(self.action_init_op, feed_dict=feed)

    # 3) maximize actions through gradient ascent
    result = self._session.run(self.cost_optimizer, feed_dict=feed)
    if tf_summary_vals is not None:
      tf_summary_vals.append(
          tf.Summary.Value(tag="batch_maxq/iterations", simple_value=result[0]))

    # 4) get max action solutions
    return self._action_projection(
        self._session.run(self._action_variable_tensor))

  def compute_dual_maxq_label(self, next_states):
    """Compute max Q label via the dual method.

    Args:
      next_states: np.ndarray for states.

    Returns:
      Tensor for the best action for the given `next_states` computed by the
      duality.
    """
    feed = {self._next_state_tensor: next_states}
    vars_vals = self._session.run(self._vars_tf)
    for var, val in zip(self._vars_tf, vars_vals):
      feed[self._dummy_network_var_ph["{}_ph".format(var.name)]] = val

    return self._session.run(self.dual_maxq_tensor, feed_dict=feed)

  def batch_train_action_function(self, state_tensor_stack, best_q_stack):
    """Train action function.

    Args:
      state_tensor_stack: np.ndarray for states.
      best_q_stack: np.ndarray for the max-Q labels.

    Returns:
      TF op for the action function loss.
    """
    feed = {
        self._state_tensor: state_tensor_stack,
        self._best_q_label: best_q_stack,
    }

    vars_vals = self._session.run(self._vars_tf)
    for var, val in zip(self._vars_tf, vars_vals):
      feed[self._dummy_network_var_ph["{}_ph".format(var.name)]] = val

    action_function_loss, _ = self._session.run(
        [self._action_function_loss, self._action_function_optimizer],
        feed_dict=feed)
    return action_function_loss

  def batch_train_q_function(self, state_tensor_stack, action_tensor_stack,
                             true_label_stack):
    """Train Q function function.

    Args:
      state_tensor_stack: np.ndarray for states.
      action_tensor_stack: np.ndarray for actions.
      true_label_stack: np.ndarray for the TD targets.

    Returns:
      TF op for the Q function loss.
    """
    feed = {
        self._state_tensor: state_tensor_stack,
        self._action_tensor: action_tensor_stack,
        self._true_label: true_label_stack,
    }

    q_function_loss, _ = self._session.run(
        [self._q_function_loss, self._q_function_optimizer], feed_dict=feed)
    return q_function_loss

  def batch_train_lambda_function(self, state_tensor_stack, action_tensor_stack,
                                  true_label_stack):
    """Train lambda function.

    Args:
      state_tensor_stack: np.ndarray for states.
      action_tensor_stack: np.ndarray for actions.
      true_label_stack: np.ndarray for the TD targets.

    Returns:
      TF op for the lambda function loss.
    """
    feed = {
        self._state_tensor: state_tensor_stack,
        self._action_tensor: action_tensor_stack,
        self._true_label: true_label_stack,
    }

    lambda_function_loss, _ = self._session.run(
        [self._lambda_function_loss, self._lambda_function_optimizer],
        feed_dict=feed)
    return lambda_function_loss
