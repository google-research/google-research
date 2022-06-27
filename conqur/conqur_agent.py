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

"""ConQUR agent to fine-tune the second-last layer of a Q-network."""
import collections

from dopamine.agents.dqn import dqn_agent
import gin
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim


@gin.configurable
class ConqurAgent(dqn_agent.DQNAgent):
  """DQN agent with last layer training.

  This is a ConQUR Agent that actually does all the heavily lifting
  of the training process and neural network specification.
  """

  def __init__(self, session, num_actions, random_state):
    """Initializes the agent and constructs the components of its graph.

    Args:
      session: tf.Session, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      random_state: np.random.RandomState, random generator state.
    """
    self.eval_mode = True
    self.random_state = random_state
    super(ConqurAgent, self).__init__(session, num_actions)

  def reload_checkpoint(self, checkpoint_path):
    """Reload variables from a fully specified checkpoint.

    Args:
      checkpoint_path: string, full path to a checkpoint to reload.
    """
    assert checkpoint_path
    variables_to_restore = tf_slim.get_variables_to_restore()
    reloader = tf.train.Saver(var_list=variables_to_restore)
    reloader.restore(self._sess, checkpoint_path)

    var = [
        v for v in variables_to_restore
        if v.name == 'Online/fully_connected_1/weights:0'
    ][0]
    wts = self._sess.run(var)
    var = [
        v for v in variables_to_restore
        if v.name == 'Online/fully_connected_1/biases:0'
    ][0]
    biases = self._sess.run(var)
    num_wts = wts.size + biases.size

    target_var = [
        v for v in variables_to_restore
        if v.name == 'Target/fully_connected_1/weights:0'
    ][0]
    target_wts = self._sess.run(target_var)
    target_var = [
        v for v in variables_to_restore
        if v.name == 'Target/fully_connected_1/biases:0'
    ][0]
    target_biases = self._sess.run(target_var)
    self.target_wts = target_wts
    self.target_biases = target_biases

    self.last_layer_weights = wts
    self.last_layer_biases = biases
    self.last_layer_wts = np.append(wts, np.expand_dims(biases, axis=0), axis=0)
    self.last_layer_wts = self.last_layer_wts.reshape((num_wts,), order='F')

  def _get_network_type(self):
    """Return the type of the outputs of a Q value network.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('DQN_network', ['q_values'])

  def _network_template(self, state):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      state: tf.Placeholder, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    net = tf.cast(state, tf.float32)
    net = tf.math.truediv(net, 255.)
    net = tf_slim.conv2d(net, 32, [8, 8], stride=4, trainable=False)
    net = tf_slim.conv2d(net, 64, [4, 4], stride=2, trainable=False)
    net = tf_slim.conv2d(net, 64, [3, 3], stride=1, trainable=False)
    net = tf_slim.flatten(net)
    linear_features = tf_slim.fully_connected(net, 512, trainable=True)
    q_values = tf_slim.fully_connected(
        linear_features, self.num_actions, activation_fn=None)
    return self._get_network_type()(q_values), linear_features

  def _create_network(self, name):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.

    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    network = self.network(self.num_actions, name=name)
    return network

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
      self.linear_features: The linear features from second last layer
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)
    self._net_outputs, self.linear_features = self.online_convnet(self.state_ph)
    self._next_target_net_outputs_q, self.target_linear_features = self.target_convnet(
        self.state_ph)
    self.next_qt_max = tf.reduce_max(self._next_target_net_outputs_q)
    self.ddqn_replay_next_target_net_outputs, _ = self.online_convnet(
        self._replay.next_states)
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

    self._replay_net_outputs, _ = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs, _ = self.target_convnet(
        self._replay.next_states)

  def update_observation(self, observation, reward, is_terminal):
    self._last_observation = self._observation
    self._record_observation(observation)

  def _select_action(self):
    if self.random_state.uniform() <= self.eps_action:
      # Choose a random action with probability epsilon.
      return self.random_state.randint(0, self.num_actions - 1)
    else:
      # Choose the action with highest Q-value at the current state.
      return self._sess.run(self._q_argmax, {self.state_ph: self.state})

  def step(self):
    return self._select_action()

  def observation_to_linear_features(self):
    # This is essentially the observation for the DNN
    return self._sess.run(self.linear_features, {self.state_ph: self.state})

  def get_target_q_op(self, reward, is_terminal):
    return self.next_qt_max, self._next_target_net_outputs_q

  def get_target_q_label(self, reward, is_terminal):
    next_qt_max, q_all_actions = self._sess.run(
        self.get_target_q_op(reward, is_terminal), {self.state_ph: self.state})
    is_terminal = is_terminal * 1.0
    return reward + self.cumulative_gamma * next_qt_max * (
        1. - is_terminal), q_all_actions

  def reset_state(self):
    self._reset_state()

  def get_target_q_label_single_target_layer(self, reward, is_terminal, fc):
    target_linear_feature = self._sess.run(self.target_linear_features,
                                           {self.state_ph: self.state})
    # The state here is the next state
    q_all_actions = fc(target_linear_feature)
    # Raw no reward actions
    q_all_actions_no_ward = q_all_actions
    q_target_max = tf.reduce_max(q_all_actions)
    q_target = reward + self.cumulative_gamma * q_target_max * (1. -
                                                                is_terminal)
    q_all_actions = reward + self.cumulative_gamma * q_all_actions * (
        1. - is_terminal)
    return q_target, q_all_actions, q_all_actions_no_ward

  def get_target_q_label_multiple_target_layers(self, reward, is_terminal,
                                                fc_list, number_actions):
    target_linear_feature = self._sess.run(self.target_linear_features,
                                           {self.state_ph: self.state})
    # the state here is the next state
    q_all_actions_list = []
    q_all_actions_no_ward_list = []
    q_target_list = []
    for fc in fc_list:
      q_all_actions = fc(target_linear_feature)
      # Raw no reward actions
      q_all_actions_no_ward = q_all_actions
      q_target_max = tf.reduce_max(q_all_actions)
      q_target = reward + self.cumulative_gamma * q_target_max * (1. -
                                                                  is_terminal)
      q_all_actions = reward + self.cumulative_gamma * q_all_actions * (
          1. - is_terminal)

      q_all_actions_list.append(q_all_actions)
      q_all_actions_no_ward_list.append(q_all_actions_no_ward)
      q_target_list.append(q_target)
    return q_target_list, q_all_actions_list, q_all_actions_no_ward_list
