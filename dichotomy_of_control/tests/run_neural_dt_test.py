# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Makes sure that scripts/run_neural_dt.py runs without error."""

from absl import app
from absl import flags
import tensorflow.compat.v2 as tf
from dichotomy_of_control.scripts import run_neural_dt


class RunNeuralDTTest(tf.test.TestCase):

  def test_run_neural_dt(self):
    flags.FLAGS.load_dir = './tests/testdata/'
    flags.FLAGS.num_steps_per_iter = 1
    flags.FLAGS.max_iters = 1
    with self.assertRaises(SystemExit):
      app.run(run_neural_dt.main)


if __name__ == '__main__':
  tf.test.main()
