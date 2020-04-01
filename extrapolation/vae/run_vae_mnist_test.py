# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import absltest
import tensorflow.compat.v1 as tf
from extrapolation.utils import utils
from extrapolation.utils.running_average_loss import RunningAverageLoss as RALoss


FLAGS = flags.FLAGS


class RunVaeMnistTest(absltest.TestCase):

  def test_update_losses(self):

    l1 = RALoss('l1', 5)
    l2 = RALoss('l2', 5)
    losses = [(l1, 10), (l2, 20)]
    utils.update_losses(losses)
    utils.update_losses(losses)
    self.assertEqual(l1.get_history(), [10, 10])
    self.assertEqual(l2.get_history(), [20, 20])


if __name__ == '__main__':
  absltest.main()
