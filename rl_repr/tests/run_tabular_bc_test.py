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

"""Makes sure that contrastive_fourier/run_tabular_bc.py runs without error."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v2 as tf
from rl_repr.contrastive_fourier import run_tabular_bc


class RunTabularBCTest(tf.test.TestCase):

  def test_run_tabular_bc(self):
    flags.FLAGS.embed_learner = 'energy'
    flags.FLAGS.embed_dim = 4
    flags.FLAGS.embed_pretraining_steps = 10
    flags.FLAGS.num_steps = 10
    flags.FLAGS.eval_interval = 10
    run_tabular_bc.main(None)


if __name__ == '__main__':
  tf.test.main()
