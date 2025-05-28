# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Makes sure that scripts/train_eval.py runs without error."""

from absl import app
from absl import flags
import tensorflow.compat.v2 as tf
from procedure_cloning.scripts import train_eval


class TrainEvalTest(tf.test.TestCase):

  def test_train_eval(self):
    flags.FLAGS.train_seeds = 5
    flags.FLAGS.test_seeds = 1
    flags.FLAGS.num_trajectory = 4
    flags.FLAGS.load_dir = './tests/testdata/'
    flags.FLAGS.algo_name = 'pc'
    flags.FLAGS.num_steps = 10
    flags.FLAGS.eval_interval = 10
    flags.FLAGS.batch_size = 1
    flags.FLAGS.num_eval_episodes = 1
    flags.FLAGS.max_eval_episode_length = 5
    with self.assertRaises(SystemExit):
      app.run(train_eval.main)


if __name__ == '__main__':
  tf.test.main()
