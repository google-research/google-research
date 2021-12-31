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

"""Makes sure that batch_rl/train_eval_offline.py runs without error."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v2 as tf
from rl_repr.batch_rl import train_eval_offline


class TrainEvalOfflineTest(tf.test.TestCase):

  def test_train_eval_offline_bc(self):
    flags.FLAGS.algo_name = 'bc'
    flags.FLAGS.embed_learner = 'acl'
    flags.FLAGS.state_embed_dim = 64
    flags.FLAGS.embed_pretraining_steps = 10
    flags.FLAGS.eval_interval = 10
    flags.FLAGS.num_updates = 30
    train_eval_offline.main(None)

  def test_train_eval_offline_brac(self):
    flags.FLAGS.algo_name = 'brac'
    flags.FLAGS.embed_learner = 'acl'
    flags.FLAGS.state_embed_dim = 64
    flags.FLAGS.embed_pretraining_steps = 10
    flags.FLAGS.eval_interval = 10
    flags.FLAGS.num_updates = 30
    train_eval_offline.main(None)

  def test_train_eval_offline_sac(self):
    flags.FLAGS.algo_name = 'sac'
    flags.FLAGS.embed_learner = 'acl'
    flags.FLAGS.state_embed_dim = 64
    flags.FLAGS.embed_pretraining_steps = 10
    flags.FLAGS.eval_interval = 10
    flags.FLAGS.num_updates = 30
    train_eval_offline.main(None)

  def test_train_eval_offline_trail(self):
    flags.FLAGS.algo_name = 'latent_bc'
    flags.FLAGS.embed_learner = 'action_fourier'
    flags.FLAGS.state_action_embed_dim = 64
    flags.FLAGS.embed_pretraining_steps = 10
    flags.FLAGS.eval_interval = 10
    flags.FLAGS.num_updates = 30
    train_eval_offline.main(None)

  def test_train_eval_offline_opal(self):
    flags.FLAGS.algo_name = 'latent_bc'
    flags.FLAGS.embed_learner = 'action_opal'
    flags.FLAGS.state_action_embed_dim = 64
    flags.FLAGS.embed_pretraining_steps = 10
    flags.FLAGS.eval_interval = 10
    flags.FLAGS.num_updates = 30
    train_eval_offline.main(None)


if __name__ == '__main__':
  tf.test.main()
