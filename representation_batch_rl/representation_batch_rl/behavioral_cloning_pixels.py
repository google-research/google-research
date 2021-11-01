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

"""Behavioral Clonning training."""
import typing

from dm_env import specs as dm_env_specs
import numpy as np
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl.encoders import ConvStack
from representation_batch_rl.batch_rl.encoders import ImageEncoder
from representation_batch_rl.batch_rl.encoders import make_impala_cnn_network
from representation_batch_rl.representation_batch_rl import policies_pixels as policies
from representation_batch_rl.representation_batch_rl import tf_utils

tmp_map_procgen = {
    'bigfish': [[a, 4] for a in range(9, 15)],  # 9,10,11,12,13,14 -> no-op (4)
    'bossfight': [[a, 4] for a in range(10, 15)
                 ],  # 10,11,12,13,14 -> no-op since only 1 special move
    'caveflyer': [[a, 4] for a in range(10, 15)],
    'chaser': [[a, 4] for a in range(9, 15)],
    'climber': [[0, 1], [3, 4], [6, 7]] +
               [[a, 4] for a in range(9, 15)
               ],  # clip vel_y to >= 0 and no special move
    'coinrun': [[a, 4] for a in range(9, 15)],
    'dodgeball': [[a, 4] for a in range(10, 15)],
    'fruitbot': [[0, 1], [3, 4], [6, 7], [2, 1], [5, 4], [8, 7]] +
                [[a, 4] for a in range(10, 15)],
    'heist': [[a, 4] for a in range(9, 15)],
    'jumper': [[0, 1], [3, 4], [6, 7]] + [[a, 4] for a in range(9, 15)],
    'leaper': [[a, 4] for a in range(9, 15)],
    'maze': [[0, 1], [6, 7], [2, 1], [8, 7]] + [[a, 4] for a in range(9, 15)],
    'miner': [[0, 1], [6, 7], [2, 1], [8, 7]] + [[a, 4] for a in range(9, 15)],
    'ninja': [[0, 1], [3, 4], [6, 7], [13, 4], [14, 4]],
    'plunder': [[0, 1], [3, 4], [6, 7], [2, 1], [5, 4], [8, 7]] +
               [[a, 4] for a in range(10, 15)],
    'starpilot': []
}
PROCGEN_ACTION_MAT = {}
for _env_name, vec in tmp_map_procgen.items():
  mat = np.eye(15)
  for (x, y) in vec:
    mat[x] = 0.
    mat[x, y] = 1.
  PROCGEN_ACTION_MAT[_env_name] = mat


class BehavioralCloning(object):
  """Training class for behavioral clonning."""

  def __init__(self,
               observation_spec,
               action_spec,
               mixture = False,
               encoder=None,
               num_augmentations = 1,
               env_name = '',
               rep_learn_keywords = 'outer',
               batch_size = 256):
    if observation_spec.shape == (64, 64, 3):
      state_dim = 256
    else:
      state_dim = 50

    self.batch_size = batch_size
    self.num_augmentations = num_augmentations
    self.rep_learn_keywords = rep_learn_keywords.split('__')

    self.discrete_actions = False if len(action_spec.shape) else True

    self.action_spec = action_spec

    if encoder is None:
      if observation_spec.shape == (64, 64, 3):
        # IMPALA for Procgen
        def conv_stack():
          return make_impala_cnn_network(
              depths=[16, 32, 32], use_batch_norm=False, dropout_rate=0.)

        state_dim = 256
      else:
        # Reduced architecture for DMC
        def conv_stack():
          return ConvStack(observation_spec.shape)
        state_dim = 50

      conv_stack_bc = conv_stack()

      if observation_spec.shape == (64, 64, 3):
        conv_stack_bc.output_size = state_dim
      # Combine and stop_grad some of the above conv stacks
      encoder = ImageEncoder(
          conv_stack_bc, feature_dim=state_dim, bprop_conv_stack=True)

      if self.num_augmentations == 0:
        dummy_state = tf.constant(
            np.zeros(shape=[1] + list(observation_spec.shape)))
      else:  # account for padding of +4 everywhere and then cropping out 68
        dummy_state = tf.constant(np.zeros(shape=[1, 68, 68, 3]))

      @tf.function
      def init_models():
        encoder(dummy_state)

      init_models()

    if self.discrete_actions:
      if 'linear_Q' in self.rep_learn_keywords:
        hidden_dims = ()
      else:
        hidden_dims = (256, 256)
      self.policy = policies.CategoricalPolicy(
          state_dim, action_spec, hidden_dims=hidden_dims, encoder=encoder)
      action_dim = action_spec.maximum.item() + 1
    else:
      action_dim = action_spec.shape[0]
      if mixture:
        self.policy = policies.MixtureGuassianPolicy(
            state_dim, action_spec, encoder=encoder)
      else:
        self.policy = policies.DiagGuassianPolicy(
            state_dim, action_spec, encoder=encoder)

    self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-4)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.alpha_optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-4)

    self.target_entropy = -action_dim

    if env_name and env_name.startswith('procgen'):
      self.procgen_action_mat = PROCGEN_ACTION_MAT[env_name.split('-')[1]]

    self.bc = None

    self.model_dict = {
        'policy': self.policy,
        'optimizer': self.optimizer
    }

  @property
  def alpha(self):
    return tf.exp(self.log_alpha)

  @tf.function
  def update_step(self, replay_buffer_iter):
    """Performs a single training step for critic and embedding.

    Args:
      replay_buffer_iter: A tensorflow graph iteratable object.

    Returns:
      Dictionary with losses to track.
    """
    transition = next(replay_buffer_iter)
    numpy_dataset = isinstance(replay_buffer_iter, np.ndarray)
    # observation: n_batch x n_timesteps x 1 x H*W*3*n_frames x 1 ->
    # n_batch x H x W x 3*n_frames
    if not numpy_dataset:
      states = transition.observation[:, 0]
      next_states = transition.observation[:, 1]
      actions = transition.action[:, 0]

      if transition.observation.dtype == tf.uint8:
        states = tf.cast(states, tf.float32) / 255.
        next_states = tf.cast(next_states, tf.float32) / 255.
    else:
      states, actions, _, next_states, _ = transition

    if self.num_augmentations > 0:
      states, next_states = tf_utils.image_aug(
          states,
          next_states,
          img_pad=4,
          num_augmentations=self.num_augmentations,
          obs_dim=64,
          channels=3,
          cropped_shape=[self.batch_size, 68, 68, 3])
      states = states[0]
      next_states = next_states[0]

    # actions = tf.gather(self.PROCGEN_ACTION_MAP.astype(np.int32).argmax(1),
    #                     actions,axis=0)

    variables = self.policy.trainable_variables

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(variables)

      log_probs, entropy = self.policy.log_probs(
          states, actions, with_entropy=True)

      loss = -tf.reduce_mean(log_probs)  # self.alpha * entropy +

    grads = tape.gradient(loss, variables)

    self.optimizer.apply_gradients(zip(grads, variables))

#     with tf.GradientTape(watch_accessed_variables=False) as tape:
#       tape.watch([self.log_alpha])
#       alpha_loss = tf.reduce_mean(self.alpha * (entropy - self.target_entropy)

#     alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
#     self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
    alpha_loss = tf.constant(0.)

    return {
        'bc_actor_loss': loss,
        'bc_alpha': self.alpha,
        'bc_alpha_loss': alpha_loss,
        'bc_log_probs': tf.reduce_mean(log_probs),
        'bc_entropy': tf.reduce_mean(entropy)
    }

  @tf.function
  def act(self, states, data_aug=False):
    """Act with batch of states.

    Args:
      states: tf.tensor n_batch x 64 x 64 x 3
      data_aug: bool, whether to use stochastic data aug (else deterministic)

    Returns:
      action: tf.tensor
    """
    if data_aug and self.num_augmentations > 0:
      states = states[0]
    if self.num_augmentations > 0:
      # use pad of 2 to bump 64 to 68 with 2 + 64 + 2 on each side
      img_pad = 2
      paddings = tf.constant(
          [[0, 0], [img_pad, img_pad], [img_pad, img_pad], [0, 0]],
          dtype=tf.int32)
      states = tf.cast(
          tf.pad(tf.cast(states * 255., tf.int32), paddings, 'SYMMETRIC'),
          tf.float32) / 255.
    return self.policy(states, sample=False)
