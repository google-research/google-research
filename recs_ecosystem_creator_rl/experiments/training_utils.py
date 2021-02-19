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

"""Utility functions for trainig scripts."""

import numpy as np
import tensorflow as tf


def save_summary(summary_writer,
                 user_value_model,
                 creator_value_model,
                 experience_replay,
                 num_users,
                 num_creators,
                 num_documents,
                 policy_probs,
                 selected_probs,
                 overall_scaled_accumulated_reward=None,
                 epoch=None):
  """Save summaries for tensorboard."""
  # Write train loss summary for tensorboard.
  with summary_writer.as_default():
    # Training losses.
    tf.summary.scalar(
        'user_loss', user_value_model.train_loss.result(), step=epoch)
    tf.summary.scalar(
        'creator_loss', creator_value_model.train_loss.result(), step=epoch)
    tf.summary.scalar(
        'user_relative_loss',
        user_value_model.train_relative_loss.result(),
        step=epoch)
    tf.summary.scalar(
        'creator_relative_loss',
        creator_value_model.train_relative_loss.result(),
        step=epoch)

    if overall_scaled_accumulated_reward:
      # Overall utilities.
      tf.summary.scalar(
          'overall_scaled_accumulated_reward',
          overall_scaled_accumulated_reward,
          step=epoch)

    # User utilities.
    user_utilities = list(experience_replay.user_utilities.values())
    user_masks = list(experience_replay.user_masks.values())
    tf.summary.scalar(
        'average_user_utility',
        np.sum(user_utilities) / np.sum(user_masks),
        step=epoch)
    tf.summary.scalar(
        'average_user_accumulated_reward',
        np.mean(list(experience_replay.user_accumulated_reward.values())),
        step=epoch)

    # Creator utilities.
    creator_utilities = list(experience_replay.creator_utilities.values())
    creator_masks = list(experience_replay.creator_masks.values())
    creator_num_recs = list(experience_replay.creator_num_recs.values())
    creator_num_clicks = list(experience_replay.creator_num_clicks.values())
    creator_user_rewards = list(experience_replay.creator_user_rewards.values())
    creator_accumulated_reward = np.array(
        list(experience_replay.creator_accumulated_reward.values()))

    tf.summary.scalar(
        'average_creator_utility',
        np.sum(creator_utilities) / np.sum(creator_masks),
        step=epoch)
    tf.summary.scalar(
        'average_creator_num_recs',
        np.sum(creator_num_recs) / np.sum(creator_masks),
        step=epoch)
    tf.summary.scalar(
        'average_creator_num_clicks',
        np.sum(creator_num_clicks) / np.sum(creator_masks),
        step=epoch)
    tf.summary.scalar(
        'average_creator_user_rewards',
        np.sum(creator_user_rewards) / np.sum(creator_masks),
        step=epoch)
    tf.summary.scalar(
        'average_creator_accumulated_reward',
        np.mean(creator_accumulated_reward),
        step=epoch)

    # Number of users, creators, documents during the simulations.
    tf.summary.scalar('user_num', np.mean(num_users), step=epoch)
    tf.summary.scalar('creator_num', np.mean(num_creators), step=epoch)
    tf.summary.scalar('document_num', np.mean(num_documents), step=epoch)

    # Agent policy and selected probabilities during the simulations.
    tf.summary.histogram('policy_probs', policy_probs, step=epoch)
    tf.summary.scalar(
        'average_selected_prob', np.mean(selected_probs), step=epoch)
