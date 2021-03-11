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

# Lint as: python3
"""Helper function for model creation for Distracting DM control."""

import os
import time

from absl import logging

import gin
import tensorflow.compat.v2 as tf
from tf_agents.policies import actor_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import learner
from tf_agents.utils import common

from pse.dm_control.utils import networks

POLICY_SAVED_MODEL_DIR = learner.POLICY_SAVED_MODEL_DIR
TRAIN_DIR = learner.TRAIN_DIR


@gin.configurable
class Actor(networks.Actor):
  """Actor with additional function to calculate representations."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               image_encoder=None,
               fc_layers=(1024, 1024),
               image_encoder_representation=False):
    super().__init__(
        input_tensor_spec,
        output_tensor_spec,
        image_encoder,
        fc_layers=fc_layers)
    if not image_encoder_representation:
      self._representation = tf.keras.layers.Dense(256, activation='relu')
    self._image_encoder_representation = image_encoder_representation

  def representation(self,
                     observations,
                     step_type=(),
                     network_state=(),
                     training=False):
    if self._image_encoder:
      encoded, network_state = self._image_encoder(
          observations, training=training)
      encoded = self._fc_encoder(encoded)
    else:
      # dm_control state observations need to be flattened as they are
      # structured as a dict(position, velocity)
      encoded = tf.keras.layers.concatenate(
          [observations['position'], observations['velocity']])

    if not self._image_encoder_representation:
      encoded = self._representation(self._dense_layers(encoded))

    return encoded

  @property
  def encoder_variables(self):
    encoder_variables = self._image_encoder.trainable_variables
    encoder_variables += self._fc_encoder.variables
    if not self._image_encoder_representation:
      encoder_variables += self._dense_layers.trainable_variables
      encoder_variables += self._representation.trainable_variables
    return encoder_variables

  @property
  def latent_dim(self):
    if self._image_encoder_representation:
      # pylint: disable=protected-access
      return self._fc_encoder.layers[-1]._saved_model_inputs_spec.shape[-1]
      # pylint: enable=protected-access
    else:
      return self._representation.units


@gin.configurable
class Learner(learner.Learner):
  """Learned adapted for using auxiliary CME loss during training."""

  def __init__(self,
               root_dir,
               train_step,
               agent,
               alpha=1.0,
               experience_dataset_fn=None,
               triggers=None,
               checkpoint_interval=1000,
               summary_interval=1000,
               max_checkpoints_to_keep=3,
               use_kwargs_in_agent_train=False,
               run_optimizer_variable_init=True,
               load_episode_data=True,
               strategy=None):

    super().__init__(
        root_dir=root_dir,
        train_step=train_step,
        agent=agent,
        experience_dataset_fn=experience_dataset_fn,
        triggers=triggers,
        checkpoint_interval=checkpoint_interval,
        summary_interval=summary_interval,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        use_kwargs_in_agent_train=use_kwargs_in_agent_train,
        run_optimizer_variable_init=run_optimizer_variable_init,
        strategy=strategy)
    self.alpha = alpha
    self._load_episode_data = load_episode_data

  def single_train_step(self, iterator):
    if self._load_episode_data:
      (experience, _), episode_data = next(iterator)
    else:
      (experience, _) = next(iterator)
      episode_data = None
    if self.use_kwargs_in_agent_train:
      experience.update({'episode_data': episode_data})
      loss_info = self.strategy.run(self._agent.train, kwargs=experience)
    else:
      loss_info = self.strategy.run(
          self._agent.train,
          args=(experience,),
          kwargs={
              'episode_data': episode_data,
              'alpha': self.alpha
          })
    return loss_info


class ActorPolicy(actor_policy.ActorPolicy):
  """Actor Policy."""

  def update_partial(self, policy, tau=1.0):
    """Update the current policy with another policy.

    This would include copying the variables from the other policy.

    Args:
      policy: Another policy it can update from.
      tau: A float scalar in [0, 1]. When tau is 1.0 (the default), we do a hard
        update. This is used for trainable variables.

    Returns:
      An TF op to do the update.
    """
    if self.variables():
      policy_vars = policy.variables()
      return common.soft_variables_update(
          policy_vars,
          self.variables()[:len(policy_vars)],
          tau=tau,
          tau_non_trainable=None,
          sort_variables_by_name=True)
    else:
      return tf.no_op()


def load_pretrained_policy(saved_model_dir, max_train_step):
  saved_model_dir = os.path.join(saved_model_dir, 'policies/greedy_policy')
  logging.info('Loading pretrained policies from %s', saved_model_dir)
  return load_policy(saved_model_dir, max_train_step)


@gin.configurable
def load_policy(saved_model_dir,
                max_train_step,
                seconds_between_checkpoint_polls=5,
                num_retries=10):
  """Loads the latest checkpoint in a directory.

  Checkpoints for the saved model to evaluate are assumed to be at the same
  directory level as the saved_model dir. ie:

  * saved_model_dir: root_dir/policies/greedy_policy
  * checkpoints_dir: root_dir/checkpoints

  Args:
    saved_model_dir: String path to the saved model directory.
    max_train_step: Int, Maximum number of train step.
    seconds_between_checkpoint_polls: The amount of time in seconds to wait
      between polls to see if new checkpoints appear in the continuous setting.
    num_retries: Number of retries for reading checkpoints.

  Returns:
    Policy loaded from the latest checkpoint in saved_model_dir.

  Raises:
    IOError: on repeated failures to read checkpoints after all the retries.
  """
  split = os.path.split(saved_model_dir)
  # Remove trailing slash if we have one.
  if not split[-1]:
    saved_model_dir = split[0]
  for _ in range(num_retries):
    try:
      policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
          saved_model_dir, load_specs_from_pbtxt=True)
      break
    except (tf.errors.OpError, tf.errors.DataLossError, IndexError,
            FileNotFoundError):
      logging.warning(
          'Encountered an error while loading a policy. This can '
          'happen when reading a checkpoint before it is fully written. '
          'Retrying...')
      time.sleep(seconds_between_checkpoint_polls)
  else:
    logging.error('Failed to load a checkpoint after retrying: %s',
                  saved_model_dir)

  checkpoint_list = _get_checkpoints_to_evaluate(set(), saved_model_dir)
  checkpoint_numbers = [int(ckpt.split('_')[-1]) for ckpt in checkpoint_list]
  checkpoint_list = [
      ckpt for ckpt, num in zip(checkpoint_list, checkpoint_numbers)
      if num <= max_train_step
  ]
  latest_checkpoint = checkpoint_list.pop()
  assert int(
      latest_checkpoint.split('_')[-1]) <= max_train_step, 'Get a valid ckpt'

  for _ in range(num_retries):
    try:
      policy.update_from_checkpoint(latest_checkpoint)
      break
    except (tf.errors.OpError, IndexError):
      logging.warning(
          'Encountered an error while loading a checkpoint. This can '
          'happen when reading a checkpoint before it is fully written. '
          'Retrying...')
      time.sleep(seconds_between_checkpoint_polls)

  logging.info('Loading:\n\tStep:%d\tcheckpoint: %s',
               policy.get_train_step(), latest_checkpoint)
  return policy


def _get_checkpoints_to_evaluate(evaluated_checkpoints, saved_model_dir):
  """Get an ordered list of checkpoint directories that have not been evaluated.

  Note that the checkpoints are in reversed order here, because we are popping
  the checkpoints later.

  Args:
    evaluated_checkpoints: a set of checkpoint directories that have already
      been evaluated.
    saved_model_dir: directory where checkpoints are saved. Often
      root_dir/policies/greedy_policy.

  Returns:
    A sorted list of checkpoint directories to be evaluated.
  """
  checkpoints_dir = os.path.join(
      os.path.dirname(saved_model_dir), 'checkpoints', '*')
  checkpoints = tf.io.gfile.glob(checkpoints_dir)
  # Sort checkpoints, such that .pop() will return the most recent one.
  return sorted(list(set(checkpoints) - evaluated_checkpoints))
