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

"""Tensorflow Model for CAQL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow.compat.v1 as tf

from caql import caql_network


class CaqlAgent(object):
  """CAQL Agent."""

  def __init__(self, session, state_spec, action_spec, discount_factor,
               hidden_layers, learning_rate, learning_rate_action,
               learning_rate_ga, action_maximization_iterations, tau_copy,
               clipped_target_flag, hard_update_steps, batch_size, l2_loss_flag,
               simple_lambda_flag, dual_filter_clustering_flag, solver,
               dual_q_label, initial_lambda, tolerance_min_max):
    """Creates CAQL agent.

    Args:
      session: TF session.
      state_spec: tf_agents.specs.array_spec.ArraySpec. Specification for state.
      action_spec: tf_agents.specs.array_spec.ArraySpec. Specification for
        action.
      discount_factor: float on discounting factor.
      hidden_layers: list of integers. Number of hidden units for each hidden
        layer.
      learning_rate: float on Q function learning rate.
      learning_rate_action: float on action function learning rate.
      learning_rate_ga: float. Learning rate for gradient ascent optimizer.
      action_maximization_iterations: int on CEM/gradient ascent iterations.
      tau_copy: float on portion to copy train net to target net.
      clipped_target_flag: bool. Enable clipped double DQN when True.
      hard_update_steps: Number of gradient steps for hard-updating a target
        network.
      batch_size: int on batch size for training.
      l2_loss_flag: bool on using l2 loss.
      simple_lambda_flag: bool on using lambda hinge loss.
      dual_filter_clustering_flag: bool on using dual filter and clustering.
      solver: string on inner max optimizer. Supported optimizers are
        "gradient_ascent", "cross_entropy", "ails", "mip".
      dual_q_label: bool on using dual max-Q label for action function training.
        If False, use primal max-Q label.
      initial_lambda: float on initial lambda.
      tolerance_min_max: list of float. First is the minimum value and the
        second is the maximum value for the tolerance of a maxQ solver.
    """
    assert len(state_spec.shape) == 1
    assert len(action_spec.shape) == 1
    assert len(tolerance_min_max) == 2

    self._session = session
    self.state_spec = state_spec
    self.action_spec = action_spec
    self.discount_factor = discount_factor

    self.learning_rate = learning_rate
    self.learning_rate_action = learning_rate_action
    self.action_maximization_iterations = action_maximization_iterations

    self._clipped_target_flag = clipped_target_flag
    self._hard_update_steps = hard_update_steps

    self.batch_size = batch_size
    self.l2_loss_flag = l2_loss_flag
    self.simple_lambda_flag = simple_lambda_flag
    self.dual_filter_clustering_flag = dual_filter_clustering_flag
    self.solver = solver
    self._dual_q_label = dual_q_label
    self.initial_lambda = initial_lambda
    self._tolerance_min = tolerance_min_max[0]
    self._tolerance_max = tolerance_min_max[1]

    self.target_network = caql_network.CaqlNet(
        self._session,
        state_spec=self.state_spec,
        action_spec=self.action_spec,
        hidden_layers=hidden_layers,
        learning_rate=self.learning_rate,
        learning_rate_action=self.learning_rate_action,
        learning_rate_ga=learning_rate_ga,
        batch_size=self.batch_size,
        action_maximization_iterations=self.action_maximization_iterations,
        name="target",
        l2_loss_flag=self.l2_loss_flag,
        simple_lambda_flag=self.simple_lambda_flag,
        initial_lambda=self.initial_lambda)

    if self._clipped_target_flag:
      self.target_network2 = caql_network.CaqlNet(
          self._session,
          state_spec=self.state_spec,
          action_spec=self.action_spec,
          hidden_layers=hidden_layers,
          learning_rate=self.learning_rate,
          learning_rate_action=self.learning_rate_action,
          learning_rate_ga=learning_rate_ga,
          batch_size=self.batch_size,
          action_maximization_iterations=self.action_maximization_iterations,
          name="target2",
          l2_loss_flag=self.l2_loss_flag,
          simple_lambda_flag=self.simple_lambda_flag,
          initial_lambda=self.initial_lambda)

    self.train_network = caql_network.CaqlNet(
        self._session,
        state_spec=self.state_spec,
        action_spec=self.action_spec,
        hidden_layers=hidden_layers,
        learning_rate=self.learning_rate,
        learning_rate_action=self.learning_rate_action,
        learning_rate_ga=learning_rate_ga,
        batch_size=self.batch_size,
        action_maximization_iterations=self.action_maximization_iterations,
        name="train",
        l2_loss_flag=self.l2_loss_flag,
        simple_lambda_flag=self.simple_lambda_flag,
        solver=self.solver,
        initial_lambda=self.initial_lambda)

    self._copy_var_ops = self._get_copy_var_ops(
        tau_copy, dest_scope_name="target", src_scope_name="train")
    if self._clipped_target_flag:
      self._hard_copy_var_ops = self._get_hard_copy_var_ops(
          dest_scope_name="target2", src_scope_name="train")

  def initialize(self, saver, checkpoint_dir=None):
    """Initialize network or load from checkpoint.

    Args:
      saver: TF saver.
      checkpoint_dir: string. Directory path where checkpoint files are saved.

    Returns:
      integer. The initial global step value.
    """
    init_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="train") + tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="target")

    if (saver and checkpoint_dir and tf.gfile.Exists(checkpoint_dir)):
      if tf.gfile.IsDirectory(checkpoint_dir):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
      else:
        checkpoint_file = checkpoint_dir

      if checkpoint_file:
        print("Loading model weights from checkpoint %s", checkpoint_file)
        saver.restore(self._session, checkpoint_file)
      else:
        self._session.run(tf.initializers.variables(init_variables))
    else:
      self._session.run(tf.initializers.variables(init_variables))

    init_step_tensor = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="train_global_step")[0]
    init_step = tf.train.global_step(self._session, init_step_tensor)

    self.update_target_network()
    if self._clipped_target_flag:
      self.update_target_network2()

    return init_step

  def update_target_network(self):
    self._session.run(self._copy_var_ops)

  def update_target_network2(self):
    assert self._clipped_target_flag
    self._session.run(self._hard_copy_var_ops)

  def _get_copy_var_ops(self, tau_copy, dest_scope_name, src_scope_name,
                        names_to_copy=None):
    """Creates TF ops that copy weights from `src_scope` to `dest_scope`."""
    # Copy variables src_scope to dest_scope
    op_holder = []

    if names_to_copy is None:
      # copy all variables
      src_vars = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
      dest_vars = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    else:
      src_vars = []
      dest_vars = []
      for name in names_to_copy:
        src_scope_name_now = "_".join([src_scope_name, name])
        dest_scope_name_now = "_".join([dest_scope_name, name])
        src_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name_now)
        dest_vars += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name_now)

    if tau_copy < 1.0:
      for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(
            dest_var.assign((1 - tau_copy) * dest_var.value() +
                            tau_copy * src_var.value()))
    else:
      for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

  def _get_hard_copy_var_ops(self, dest_scope_name, src_scope_name):
    """Creates TF ops that copy weights from `src_scope` to `dest_scope`."""
    assert self._clipped_target_flag
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
      op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

  @staticmethod
  def compute_cluster_masks(states, mask, eps_approx_ratio=0.01):
    """Farthest-first (FF) traversal clustering algorithm.

    Args:
      states: np.ndarray of floats. State input.
      mask: 1-D np.ndarray of bools. A mask indicating the states that are
        candidates for centroids. Centroids are selected among them. The length
        of this array should be same as the length of the first dimension of
        `states` (i.e., batch size).
      eps_approx_ratio: float. The "radius" of a cluster.

    Returns:
      (np.ndarray, np.ndarray, list). The first array is the boolean mask w.r.t.
      states that are cluster centers. Second array is the boolean mask w.r.t.
      states that are non-clusters. Third element is a list of dictionary of
      information (element index and centroid) for non-clusters.
    """
    max_num_clusters = min([sum(mask), max([1, int(sum(mask))])])
    eps_approx = eps_approx_ratio * np.mean(np.linalg.norm(states, axis=1))

    cluster_mask = np.array([False] * len(states))
    noncluster_mask = np.array([False] * len(states))

    if sum(mask) > 0:
      noncluster_states = [
          (ind, ele) for ind, ele in enumerate(states) if mask[ind]
      ]

      cluster_states = []
      # now compute the states for clusters using FF-traversal algorithm
      # FF-Traversal Algorithm:
      # 1. Pick C = {x}, for an arbitrary point x
      init_state = random.choice(noncluster_states)
      cluster_states.append(init_state)
      # 2. Repeat until C has k centers:
      noncluster_states.remove(init_state)

      def distance_fn(inp, c_list):
        distances = [np.linalg.norm(inp[1] - element[1]) for element in c_list]
        min_index = np.argmin(distances)
        return {"centroid": c_list[min_index], "distance": distances[min_index]}

      while len(cluster_states) < max_num_clusters and noncluster_states:
        # Let y maximize d(y, C), where d(y, C) = min_(x in C) d(x, y)
        # C = C U {y}
        distance = [
            distance_fn(state, cluster_states)["distance"]
            for state in noncluster_states
        ]
        if max(distance) < eps_approx:
          # for all the remaining nonclusters, there's an eps-close cluster
          break
        # update cluster set and remaining set
        state_with_max_distance = noncluster_states[np.argmax(distance)]
        cluster_states.append(state_with_max_distance)
        noncluster_states.remove(state_with_max_distance)

      # return 1) Flags that are cluster/non-cluster,
      cluster_mask[np.array([ele[0] for ele in cluster_states])] = True
      if noncluster_states:
        noncluster_mask[np.array([ele[0] for ele in noncluster_states])] = True

      # 2) cluster_info, a list of dict of nearest centroid and distance
      cluster_info = [{
          "non_cluster_index": ele[0],
          "centroid": distance_fn(ele, cluster_states)["centroid"]
      } for ele in noncluster_states]
    else:
      cluster_info = None
    return cluster_mask, noncluster_mask, cluster_info

  def _compute_extrapolated_target(self, states, actions, cluster_info):
    """Extrapolate target next value by first-oder Taylor series."""
    state_deviation = []
    centroid_actions = []
    centroid_states = []
    for ele in cluster_info:
      state_deviation.append(states[ele["non_cluster_index"]] -
                             ele["centroid"][1])
      centroid_actions.append(actions[ele["centroid"][0]])
      centroid_states.append(ele["centroid"][1])
    return self.target_network.predict_state_perturbed_q_function(
        centroid_states, centroid_actions, state_deviation)

  def _compute_tolerance(self, states, actions, next_states, rewards, dones,
                         tolerance_init=None, tolerance_decay=None):
    """Compute the (dynamic) tolerance for a max-Q solver."""
    if tolerance_init is None:
      tolerance_init = self._tolerance_min

    if tolerance_decay is not None:
      target_next_values_tolerance = self.target_network.predict_q_function(
          next_states, self.train_network.predict_action_function(next_states))
      td_rmse_tolerance = self.train_network.compute_td_rmse(
          states, actions, target_next_values_tolerance, rewards, dones,
          self.discount_factor)
      tolerance = tolerance_init * td_rmse_tolerance * tolerance_decay
      return min([max([tolerance, self._tolerance_min]), self._tolerance_max])
    else:
      return tolerance_init

  def train_q_function_network(self, batch, tolerance_init, tolerance_decay,
                               warmstart=True, tf_summary_vals=None):
    """Train Q function network.

    Args:
      batch: list of states, actions, rewards, next_states, dones, unused_infos.
      tolerance_init: float on initial tolerance.
      tolerance_decay: float on current tolerance decay.
      warmstart: bool on warmstarting flag.
      tf_summary_vals: list to store tf.Summary.Value objects.

    Returns:
      (float, float, float, dict, float, float)
      The first element is the loss of q_function, second element is TD target,
      third element is loss of lambda update (only active for hinge loss),
      fourth element is a dict containing the batch of states and Q labels for
      training action function, fifth element is the portion of active data
      after dual filter, and sixth element is the portion of active data after
      dual filter and clustering.
    """
    [states, actions, rewards, next_states, dones, unused_infos] = zip(*batch)

    if self.solver == "dual":
      # dual methods for approximating target_next_values, but that isn't DDQN!
      target_next_values = np.reshape(
          self.target_network.compute_dual_maxq_label(next_states),
          (len(states),))
      # portion_active_data: data portion for maxq after dual filter
      portion_active_data = 0.0
      # portion_active_data_and_cluster: data portion for maxq after dual filter
      # and cluster
      portion_active_data_and_cluster = 0.0

    else:
      assert self.solver in ["gradient_ascent", "cross_entropy", "ails", "mip"]

      # Trick 1: Find dual objective and use it to check for active constraints
      if self.dual_filter_clustering_flag:
        target_next_values = self.target_network.compute_dual_maxq_label(
            next_states)
        dual_mask = self.train_network.compute_dual_active_constraint_condition(
            states, actions, target_next_values, rewards, dones,
            self.discount_factor)
        portion_active_data = np.mean(dual_mask)
        # Trick 2: Construct a buffer (of size k) of state and max-q values
        # This is still w.r.t. same Q function!
        # So, this is still an offline k-center problem
        # 1) "Extrapolate" the max-q value when states are close to a center
        # 2) Compute the maxq value otherwise
        cluster_mask, noncluster_mask, noncluster_info = (
            self.compute_cluster_masks(next_states, dual_mask))

        if np.all(np.logical_not(cluster_mask)):
          # if there's no centroid, do not need to compute next-Q values.
          target_next_values = np.zeros_like(dual_mask).reshape([-1, 1])
          portion_active_data_and_cluster = 0.0
        else:
          portion_active_data_and_cluster = np.mean(cluster_mask)

          # calculate the target values w.r.t. clusters
          masked_states = np.reshape(
              states, [-1, self.state_spec.shape[0]])[cluster_mask]
          masked_actions = np.reshape(
              actions, [-1, self.action_spec.shape[0]])[cluster_mask]
          masked_rewards = np.reshape(rewards, [-1, 1])[cluster_mask]
          masked_dones = np.reshape(dones, [-1, 1])[cluster_mask]
          # Compute the max-Q values for what's remaining
          masked_next_states = np.reshape(
              next_states, [-1, self.state_spec.shape[0]])[cluster_mask]

          masked_next_actions = self.train_network.compute_best_actions(
              masked_next_states,
              tolerance=self._compute_tolerance(
                  masked_states,
                  masked_actions,
                  masked_next_states,
                  masked_rewards,
                  masked_dones,
                  tolerance_init=tolerance_init,
                  tolerance_decay=tolerance_decay),
              warmstart=warmstart)
          # Update the target next values w.r.t. the remaining masked_states
          target_next_values[cluster_mask] = (
              self.target_network.predict_q_function(
                  masked_next_states, masked_next_actions))

          if np.any(noncluster_mask):
            # if there exists non-clusters
            # extrapolate the target values w.r.t. non-clusters
            next_actions = np.zeros_like(actions)
            next_actions[cluster_mask] = masked_next_actions

            # A numerical trick to reduce over-estimation error
            # Choose the minimum one between the dual approx
            # and the extrapolation estimators
            target_next_values[noncluster_mask] = np.minimum(
                self._compute_extrapolated_target(next_states, next_actions,
                                                  noncluster_info),
                target_next_values[noncluster_mask])
      else:
        portion_active_data = 1.0
        portion_active_data_and_cluster = 1.0
        next_actions = self.train_network.compute_best_actions(
            next_states,
            tolerance=self._compute_tolerance(
                states,
                actions,
                next_states,
                rewards,
                dones,
                tolerance_init=tolerance_init,
                tolerance_decay=tolerance_decay),
            warmstart=warmstart,
            tf_summary_vals=tf_summary_vals)

        target_next_values = self.target_network.predict_q_function(
            next_states, next_actions)
        if self._clipped_target_flag:
          target_next_values2 = self.target_network2.predict_q_function(
              next_states, next_actions)
          target_next_values = np.minimum(target_next_values,
                                          target_next_values2)

    target_reward_value = self.train_network.compute_backup(
        target_next_values, rewards, dones, self.discount_factor)

    q_function_loss = self.train_network.batch_train_q_function(
        states, actions, target_reward_value)
    lambda_function_loss = self.train_network.batch_train_lambda_function(
        states, actions, target_reward_value)

    # `best_train_label_batch` is returned to be used for action function
    # training.
    if self._dual_q_label:
      best_q_stack = self.train_network.compute_dual_maxq_label(next_states)
    else:
      best_q_stack = target_next_values
    best_train_label_batch = {
        "state_tensor_stack": next_states,
        "best_q_stack": best_q_stack
    }

    return (q_function_loss, target_reward_value, lambda_function_loss,
            best_train_label_batch, portion_active_data,
            portion_active_data_and_cluster)

  def train_action_function_network(self, best_train_label_batch):
    """Train action function.

    Args:
      best_train_label_batch: dict. "state_tensor_stack" maps to state batch
        tensor. "best_q_stack" maps to a Q label tensor.

    Returns:
      TF op for the action function loss.
    """
    action_function_loss = self.train_network.batch_train_action_function(
        best_train_label_batch["state_tensor_stack"],
        best_train_label_batch["best_q_stack"])
    return action_function_loss

  def best_action(self, state, use_action_function=True):
    """Computes the best action for the given state.

    Args:
      state: Tensor for state.
      use_action_function: Use action function predicting the best action for
        the given `state` if True, otherwise, compute directly argmax_a Q(x, a)
        for the best action.

    Returns:
      (Tensor, None, None, True). The first is the best action tensor for the
        given `state`, and the following three are dummy.
    """
    if use_action_function:
      max_action = self.train_network.predict_action_function(state)
    else:
      max_action = self.train_network.compute_best_actions(
          state, self._tolerance_min)
    # just to fit into the API
    return max_action, None, None, True
