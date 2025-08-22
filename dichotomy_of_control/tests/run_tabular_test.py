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

"""Makes sure that scripts/run_tabular.py runs without error."""

from absl import app
from absl import flags
import tensorflow.compat.v2 as tf
from dichotomy_of_control.scripts import run_tabular


class RunTabularTest(tf.test.TestCase):

  def test_run_tabular(self):
    flags.FLAGS.load_dir = './tests/testdata/'
    flags.FLAGS.algo_name = 'sdt'
    flags.FLAGS.num_steps = 10
    flags.FLAGS.eval_interval = 10
    flags.FLAGS.batch_size = 1
    with self.assertRaises(SystemExit):
      app.run(run_tabular.main)


if __name__ == '__main__':
  tf.test.main()
