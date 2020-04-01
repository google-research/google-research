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

# Lint as: python2, python3
import json

from absl.testing import absltest
from eeg_modelling.eeg_viewer import utils


class UtilsTest(absltest.TestCase):

  def testGetSampleRange(self):
    self.assertEqual((10, 20), utils.GetSampleRange(1, 10, 10))

  def testInitDataTableInputsWithTimeAxis(self):
    data, _ = utils.InitDataTableInputsWithTimeAxis(1, 3, 3, 8)
    self.assertEqual([{'seconds': 3.0}, {'seconds': 4.0}, {'seconds': 5.0}],
                     data)

  def testConvertToDataTableJSon(self):
    data, _ = utils.InitDataTableInputsWithTimeAxis(freq=1,
                                                    chunk_duration_sec=4,
                                                    chunk_start=4,
                                                    max_samples=10)

    json_data = json.loads(utils.ConvertToDataTableJSon(data, ['seconds']))
    self.assertEqual([{'id': 'seconds', 'label': 'seconds', 'type': 'number'}],
                     json_data['cols'])
    self.assertEqual([{'c': [{'v': 4}]}, {'c': [{'v': 5}]}, {'c': [{'v': 6}]},
                      {'c': [{'v': 7}]}], json_data['rows'])

  def testCreateEmptyTable(self):
    return_value = utils.CreateEmptyTable(5, 10)
    json_data = json.loads(return_value)

    self.assertEqual([{'id': 'seconds', 'label': 'seconds', 'type': 'number'}],
                     json_data['cols'])
    self.assertEqual([{'c': [{'v': 0}]}, {'c': [{'v': 1}]}, {'c': [{'v': 2}]},
                      {'c': [{'v': 3}]}, {'c': [{'v': 4}]}, {'c': [{'v': 5}]}],
                     json_data['rows'])


if __name__ == '__main__':
  absltest.main()
