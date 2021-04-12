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

"""Utlity functions for multigrid environments."""
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.metrics import py_metrics
from tf_agents.specs import tensor_spec
from tf_agents.train import ppo_learner
from tf_agents.utils import common
from tf_agents.utils import nest_utils

from social_rl.multiagent_tfagents import multiagent_metrics


class MultiagentPPOLearner(ppo_learner.PPOLearner):
  """A modified PPOLearner for multiagent training.

  Currently disables normalizer updates, since we don't have normalizers
  implemented and our policy info is formatted differently.
  """

  @common.function(autograph=True)
  def _update_normalizers(self, iterator):
    """Update the normalizers and count the total number of frames."""

    reward_spec = tensor_spec.TensorSpec(shape=[None], dtype=tf.float32)
    def _update(traj):
      if traj.reward.shape:

        outer_shape = nest_utils.get_outer_shape(traj.reward, reward_spec)
        batch_size = outer_shape[0]
        if len(outer_shape) > 1:
          batch_size *= outer_shape[1]
      else:
        batch_size = 1
      return batch_size

    num_frames = 0
    traj, _ = next(iterator)
    num_frames += _update(traj)

    for _ in tf.range(1, self._num_batches):
      traj, _ = next(iterator)
      num_frames += _update(traj)

    return num_frames


def collect_metrics(buffer_size, n_agents):
  """Utility to create metrics often used during data collection."""
  metrics = [
      py_metrics.NumberOfEpisodes(),
      py_metrics.EnvironmentSteps(),
      multiagent_metrics.AverageReturnPyMetric(n_agents=n_agents,
                                               buffer_size=buffer_size),
      py_metrics.AverageEpisodeLengthMetric(buffer_size=buffer_size),
  ]
  return metrics
