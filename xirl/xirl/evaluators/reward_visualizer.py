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

"""Reward visualizer."""

from typing import List

from .base import Evaluator
from .base import EvaluatorOutput
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from xirl.models import SelfSupervisedOutput


class RewardVisualizer(Evaluator):
  """Distance to goal state visualizer."""

  def __init__(self, distance, num_plots):
    """Constructor.

    Args:
      distance: The distance metric to use when calculating nearest-neighbours.
      num_plots: The number of reward plots to display.

    Raises:
      ValueError: If the distance metric is invalid.
    """
    super().__init__(inter_class=False)

    if distance not in ["sqeuclidean", "cosine"]:
      raise ValueError(
          "{} is not a supported distance metric.".format(distance))

    # For plotting, we don't want to display squared euclidean distances so we'll
    # override to `euclidean` if it was selected.
    if distance == "sqeuclidean":
      distance = "euclidean"

    self.distance = distance
    self.num_plots = num_plots

  def _gen_reward_plot(self, rewards):
    """Create a pyplot plot and save to buffer."""
    fig, axes = plt.subplots(1, len(rewards), figsize=(6.4 * len(rewards), 4.8))
    if len(rewards) == 1:
      axes = [axes]
    for i, rew in enumerate(rewards):
      axes[i].plot(rew)
    fig.text(0.5, 0.04, "Timestep", ha="center")
    fig.text(0.04, 0.5, "Reward", va="center", rotation="vertical")
    fig.canvas.draw()
    img_arr = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close()
    return img_arr

  def _compute_goal_emb(self, embs):
    """Compute the mean of all last frame embeddings."""
    goal_emb = [emb[-1, :] for emb in embs]
    goal_emb = np.stack(goal_emb, axis=0)
    goal_emb = np.mean(goal_emb, axis=0, keepdims=True)
    return goal_emb

  def evaluate(self, outs):
    embs = [o.embs for o in outs]
    goal_emb = self._compute_goal_emb(embs)

    # Make sure we sample only as many as are available.
    num_plots = min(len(embs), self.num_plots)
    rand_idxs = np.random.choice(
        np.arange(len(embs)), size=num_plots, replace=False)

    # Compute rewards as distances to the goal embedding.
    rewards = []
    for idx in rand_idxs:
      emb = embs[idx]
      dists = cdist(emb, goal_emb, self.distance)
      rewards.append(-dists)

    image = self._gen_reward_plot(rewards)
    return EvaluatorOutput(image=image)
