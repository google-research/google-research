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

"""Behavior cloning learning."""
import tensorflow.compat.v2 as tf
from tensorflow_addons import optimizers as tfa_optimizers
from tf_agents.specs import tensor_spec
import policy_eval.actor as actor_lib


class BehaviorCloning(object):
  """Behavior cloning."""

  def __init__(self,
               state_dim,
               action_spec,
               learning_rate,
               weight_decay):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      learning_rate: Learning rate.
      weight_decay: Weight decay.
    """
    self.actor = actor_lib.Actor(state_dim, action_spec)
    self.action_spec = action_spec
    self.optimizer = tfa_optimizers.AdamW(learning_rate=learning_rate,
                                          weight_decay=weight_decay)

  def __call__(self, states, actions):
    dist, _ = self.actor.get_dist_and_mode(states)
    actions = tf.clip_by_value(actions, 1e-4 + self.action_spec.minimum,
                               -1e-4 + self.action_spec.maximum)
    log_probs = dist.log_prob(actions)
    return dist, log_probs

  @tf.function
  def update(self,
             states,
             actions,
             weights):
    """Updates actor parameters.

    Args:
      states: A batch of states.
      actions: A batch of actions.
      weights: A batch of weights.
    Returns:
      Actor loss.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.actor.trainable_variables)
      actions = tf.clip_by_value(actions, 1e-4 + self.action_spec.minimum,
                                 -1e-4 + self.action_spec.maximum)
      log_prob = self.actor.get_log_prob(states, actions)
      actor_loss = (
          tf.reduce_sum(-log_prob * weights) /
          tf.reduce_sum(weights))
    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.optimizer.apply_gradients(
        zip(actor_grads, self.actor.trainable_variables))
    tf.summary.scalar('train/actor loss', actor_loss,
                      step=self.optimizer.iterations)
