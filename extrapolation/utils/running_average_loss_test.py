# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

from absl.testing import absltest
from extrapolation.utils.running_average_loss import RunningAverageLoss as RALoss


class RunningAverageLossTest(absltest.TestCase):

  def test_get_value(self):
    l = RALoss('name', 3)

    dat = [0, 1, 2, 3, 4, 5, 6, 7]
    for x in dat:
      l.update(x)
    self.assertEqual(dat, l.get_history())
    self.assertEqual(6.0, l.get_value())
    self.assertEqual(3.0, l.get_value(i=4))

if __name__ == '__main__':
  absltest.main()
