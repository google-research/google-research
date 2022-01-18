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

"""Tests for agent."""

import os

from absl import flags
from absl.testing import absltest
import numpy as np

from proxy_rewards import agent
from proxy_rewards import generate_data
from proxy_rewards import movie_lens_simulator

FLAGS = flags.FLAGS


class AgentTest(absltest.TestCase):

  def setUp(self):
    super(AgentTest, self).setUp()
    self.data_dir = os.path.join(FLAGS.test_srcdir,
                                 os.path.split(os.path.abspath(__file__))[0],
                                 'test_data')

    self.seeds = movie_lens_simulator.Seeds(
        user_model=0, user_sampler=0, train_eval_test=0)

    self.user_config = movie_lens_simulator.UserConfig(
        accept_prob=0.5, diversity_prob=0.5)

    self.env_config = movie_lens_simulator.EnvConfig(
        data_dir=self.data_dir,
        embeddings_path=f'{self.data_dir}/embeddings.json',
        genre_history_path=f'{self.data_dir}/genre_history.json',
        seeds=self.seeds,
        slate_size=5,
        user_config=self.user_config)

    self.df = generate_data.generate_data(
        self.env_config, slate_type='test', n_samples=50, seed=0,
        intercept=-5., rating_coef=1., div_seek_coef=3., diversity_coef=-0.5)

  def test_agent_runs(self):
    data = {
        'recommendation': self.df['rec'].to_numpy(),
        'reward': self.df['ltr'].to_numpy(),
        'user_id': self.df['user'].to_numpy(),
    }

    agent_model = agent.FittedAgent(
        movies=np.unique(self.df.rec),
        embed_path=self.env_config.embeddings_path,
        genre_history_path=self.env_config.genre_history_path)

    agent_model.fit(data)


if __name__ == '__main__':
  absltest.main()
