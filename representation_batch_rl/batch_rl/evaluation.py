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

"""Policy evaluation."""
from typing import Optional

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def _annotate_frame(frame, is_last, episode_return):
  """Render returns and is_last string to an image frame."""
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)
  font = ImageFont.load_default()
  text = 'is_last=%s, episode_return=%.1f' % (str(is_last), episode_return)
  draw.text((0, 0), text, (255, 255, 255), font=font)
  return np.asarray(img)


def evaluate(env,
             policy,
             num_episodes = 10,
             video_filename = None,
             max_episodes_per_video = 5,
             return_distributions=False,
             return_level_ids=False):
  """Evaluates the policy.

  Args:
    env: Environment to evaluate the policy on.
    policy: Policy to evaluate.
    num_episodes: A number of episodes to average the policy on.
    video_filename: If not None, save num_episodes_per_video to a video file.
    max_episodes_per_video: When saving a video, how many episodes to render.
    return_distributions: Whether to return per-step rewards and episode return
                           distributions instead of mean
    return_level_ids: Whether to return level ids to agent in ProcGen.
  Returns:
    Averaged reward and a total number of steps.
  """
  del video_filename  # placeholder
  del max_episodes_per_video

  total_timesteps = 0.
  total_returns = 0.0
  total_log_probs = 0.0

  return_acc = []
  reward_acc = []
  for _ in range(num_episodes):
    episode_return = 0.
    episode_log_prob = 0.
    episode_timesteps = 0.
    timestep = env.reset()

    while not timestep.is_last():
      if type(policy).__name__ == 'TfAgentsPolicy':
        action, log_probs = policy.act(timestep.observation)
        episode_log_prob += log_probs.numpy().item()
      else:
        if return_level_ids:
          action = policy.act(timestep.observation, env._infos[0]['level_seed'])  # pylint: disable=protected-access
        else:
          action = policy.act(timestep.observation)
      if hasattr(action, 'numpy'):
        action = action.numpy()
      timestep = env.step(action)

      total_returns += timestep.reward[0]
      episode_return += timestep.reward[0]
      total_timesteps += 1.0
      episode_timesteps += 1.0
      reward_acc.append(timestep.reward[0])

    episode_log_prob /= episode_timesteps
    total_log_probs += episode_log_prob

    return_acc.append(episode_return)
  if return_distributions:
    return (reward_acc, return_acc,
            total_timesteps / num_episodes, total_log_probs / num_episodes)
  if type(policy).__name__ == 'tfAgentsPolicy':
    return (total_returns / num_episodes,
            total_timesteps / num_episodes, total_log_probs / num_episodes)
  else:
    return total_returns / num_episodes, total_timesteps / num_episodes
