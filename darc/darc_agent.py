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

"""Module for implementing the DARC agent."""
import collections
import gin
import tensorflow as tf

from tf_agents.agents.sac import sac_agent
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

DarcLossInfo = collections.namedtuple(
    "DarcLossInfo",
    (
        "critic_loss",
        "actor_loss",
        "alpha_loss",
        "sa_classifier_loss",
        "sas_classifier_loss",
    ),
)


@gin.configurable
class DarcAgent(sac_agent.SacAgent):
  """An agent that implements the DARC algorithm."""

  def __init__(self, *args,
               classifier=None,
               classifier_optimizer=None,
               classifier_loss_weight=1.0,
               use_importance_weights=False,
               unnormalized_delta_r=False,
               **kwargs):
    self._classifier = classifier
    self._classifier_optimizer = classifier_optimizer
    self._classifier_loss_weight = classifier_loss_weight
    self._use_importance_weights = use_importance_weights
    self._unnormalized_delta_r = unnormalized_delta_r
    super(DarcAgent, self).__init__(*args, **kwargs)

  def _train(self, experience, weights, real_experience=None):
    assert real_experience is not None
    if self._use_importance_weights:
      assert weights is None
      sas_input = tf.concat(
          [
              experience.observation[:, 0],
              experience.action[:, 0],
              experience.observation[:, 1],
          ],
          axis=-1,
      )
      # Set training=False so no input noise is added.
      sa_probs, sas_probs = self._classifier(sas_input, training=False)
      weights = (
          sas_probs[:, 1] * sa_probs[:, 0] / (sas_probs[:, 0] * sa_probs[:, 1]))
    loss_info = super(DarcAgent, self)._train(experience, weights)
    trainable_classifier_variables = self._classifier.trainable_variables

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert (trainable_classifier_variables
             ), "No trainable classifier variables to optimize."
      tape.watch(trainable_classifier_variables)
      (
          classifier_loss,
          sa_classifier_loss,
          sas_classifier_loss,
      ) = self.classifier_loss(experience, real_experience)
      classifier_loss = self._classifier_loss_weight * classifier_loss

    tf.debugging.check_numerics(classifier_loss,
                                "classifier loss is inf or nan.")
    tf.debugging.check_numerics(sa_classifier_loss,
                                "sa classifier loss is inf or nan.")
    tf.debugging.check_numerics(sas_classifier_loss,
                                "sas classifier loss is inf or nan.")
    critic_grads = tape.gradient(classifier_loss,
                                 trainable_classifier_variables)
    self._apply_gradients(critic_grads, trainable_classifier_variables,
                          self._classifier_optimizer)
    darc_loss_info = DarcLossInfo(
        critic_loss=loss_info.extra.critic_loss,
        actor_loss=loss_info.extra.actor_loss,
        alpha_loss=loss_info.extra.alpha_loss,
        sa_classifier_loss=sa_classifier_loss,
        sas_classifier_loss=sas_classifier_loss,
    )
    loss_info = loss_info._replace(extra=darc_loss_info)
    return loss_info

  def _experience_to_sas(self, experience):
    squeeze_time_dim = not self._critic_network_1.state_spec
    (
        time_steps,
        policy_steps,
        next_time_steps,
    ) = trajectory.experience_to_transitions(experience, squeeze_time_dim)
    actions = policy_steps.action
    return tf.concat(
        [time_steps.observation, actions, next_time_steps.observation], axis=-1)

  def classifier_loss(self, experience, real_experience):
    with tf.name_scope("classifier_loss"):
      sim_sas_input = self._experience_to_sas(experience)
      real_sas_input = self._experience_to_sas(real_experience)
      sas_input = tf.concat([sim_sas_input, real_sas_input], axis=0)
      batch_size = tf.shape(real_sas_input)[0]
      y_true = tf.concat(
          [
              tf.zeros(batch_size, dtype=tf.int32),
              tf.ones(batch_size, dtype=tf.int32),
          ],
          axis=0,
      )
      tf.debugging.assert_equal(
          tf.reduce_mean(tf.cast(y_true, tf.float32)),
          0.5,
          "Classifier labels should be 50% ones.",
      )

      # Must enable training=True to use input noise.
      sa_probs, sas_probs = self._classifier(sas_input, training=True)
      sa_classifier_loss = tf.keras.losses.sparse_categorical_crossentropy(
          y_true, sa_probs)
      sas_classifier_loss = tf.keras.losses.sparse_categorical_crossentropy(
          y_true, sas_probs)
      classifier_loss = sa_classifier_loss + sas_classifier_loss

      sa_correct = tf.argmax(sa_probs, axis=1, output_type=tf.int32) == y_true
      sa_accuracy = tf.reduce_mean(tf.cast(sa_correct, tf.float32))
      sas_correct = tf.argmax(sas_probs, axis=1, output_type=tf.int32) == y_true
      sas_accuracy = tf.reduce_mean(tf.cast(sas_correct, tf.float32))
      tf.compat.v2.summary.scalar(
          name="classifier_loss",
          data=tf.reduce_mean(classifier_loss),
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name="sa_classifier_loss",
          data=tf.reduce_mean(sa_classifier_loss),
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name="sas_classifier_loss",
          data=tf.reduce_mean(sas_classifier_loss),
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name="sa_classifier_accuracy",
          data=sa_accuracy,
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name="sas_classifier_accuracy",
          data=sas_accuracy,
          step=self.train_step_counter,
      )
      return classifier_loss, sa_classifier_loss, sas_classifier_loss

  @gin.configurable
  def critic_loss(
      self,
      time_steps,
      actions,
      next_time_steps,
      td_errors_loss_fn,
      gamma=1.0,
      reward_scale_factor=1.0,
      weights=None,
      training=False,
      delta_r_scale=1.0,
      delta_r_warmup=0,
  ):
    sas_input = tf.concat(
        [time_steps.observation, actions, next_time_steps.observation], axis=-1)
    # Set training=False so no input noise is added.
    sa_probs, sas_probs = self._classifier(sas_input, training=False)
    sas_log_probs = tf.math.log(sas_probs)
    sa_log_probs = tf.math.log(sa_probs)
    if self._unnormalized_delta_r:  # Option for ablation experiment.
      delta_r = sas_log_probs[:, 1] - sas_log_probs[:, 0]
    else:  # Default option (i.e., the correct version).
      delta_r = (
          sas_log_probs[:, 1] - sas_log_probs[:, 0] - sa_log_probs[:, 1] +
          sa_log_probs[:, 0])
    common.generate_tensor_summaries("delta_r", delta_r,
                                     self.train_step_counter)
    is_warmup = tf.cast(self.train_step_counter < delta_r_warmup, tf.float32)
    tf.compat.v2.summary.scalar(
        name="is_warmup", data=is_warmup, step=self.train_step_counter)
    next_time_steps = next_time_steps._replace(reward=next_time_steps.reward +
                                               delta_r_scale *
                                               (1 - is_warmup) * delta_r)
    return super(DarcAgent, self).critic_loss(
        time_steps,
        actions,
        next_time_steps,
        td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        weights=weights,
        training=training,
    )

  def _check_train_argspec(self, kwargs):
    """Overwrite to avoid checking that real_experience has right dtype."""
    del kwargs
    return
