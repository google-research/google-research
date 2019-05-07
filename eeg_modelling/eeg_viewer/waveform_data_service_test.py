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

# Lint as: python2, python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from absl.testing import absltest
import mock
from six.moves import range
import tensorflow as tf

from eeg_modelling.eeg_viewer import data_source
from eeg_modelling.eeg_viewer import utils
from eeg_modelling.eeg_viewer import waveform_data_service
from eeg_modelling.pyprotos import data_pb2


class WaveformDataServiceTest(absltest.TestCase):

  def setUp(self):
    super(WaveformDataServiceTest, self).setUp()
    tf_ex = tf.train.Example()
    feature = tf_ex.features.feature
    feature['eeg_channel/num_samples'].int64_list.value.append(20)
    feature['eeg_channel/sampling_frequency_hz'].float_list.value.append(1.0)
    for i in range(20):
      feature['test_feature'].float_list.value.append(i*2)
    for i in range(20):
      feature[('eeg_channel/EEG '
               'test_sub-REF/samples')].float_list.value.append(i+1)
    for i in range(20):
      feature[('eeg_channel/EEG '
               'test_min-REF/samples')].float_list.value.append(1)
    self.waveform_data_source = data_source.TfExampleEegDataSource(tf_ex,
                                                                   'test_key')

  def testAddDataTableSeries(self):
    data, _ = utils.InitDataTableInputsWithTimeAxis(freq=1,
                                                    chunk_duration_sec=4,
                                                    chunk_start=0,
                                                    max_samples=10)

    waveform_data_service._AddDataTableSeries({'test row': [1, 4, 9, 16]}, data)
    self.assertEqual([{'seconds': 0, 'test row': 1},
                      {'seconds': 1, 'test row': 4},
                      {'seconds': 2, 'test row': 9},
                      {'seconds': 3, 'test row': 16}], data)

  def testCreateDataTable(self):
    request = data_pb2.DataRequest()
    request.chunk_duration_secs = 4
    request.chunk_start = 0
    channel_data_id = request.channel_data_ids.add()
    channel_data_id.bipolar_channel.index = 0
    channel_data_id.bipolar_channel.referential_index = 1
    request.low_cut = 0
    request.high_cut = 0
    request.notch = 0

    datatable, freq = waveform_data_service._CreateChunkDataTableJSon(
        self.waveform_data_source, request, 10)
    json_data = json.loads(datatable)

    self.assertEqual([
        {'id': 'seconds', 'label': 'seconds', 'type': 'number'},
        {'id': 'TEST_MIN-TEST_SUB', 'label': 'TEST_MIN-TEST_SUB',
         'type': 'number'}], json_data['cols'])
    self.assertEqual([
        {'c': [{'v': 0}, {'v': 0}]},
        {'c': [{'v': 1}, {'v': 1}]},
        {'c': [{'v': 2}, {'v': 2}]},
        {'c': [{'v': 3}, {'v': 3}]}], json_data['rows'])
    self.assertEqual(1.0, freq)

  @mock.patch.object(waveform_data_service, '_CreateChunkDataTableJSon')
  def testGetChunk(self, mock_create):
    mock_create.return_value = ('test data', 1)

    request = data_pb2.DataRequest()
    request.chunk_duration_secs = 10
    request.chunk_start = 0
    channel_data_id = request.channel_data_ids.add()
    channel_data_id.single_channel.index = 0
    request.low_cut = 1.0
    request.high_cut = 70.0
    request.notch = 60.0

    response = waveform_data_service.GetChunk(self.waveform_data_source,
                                              request, 10)
    mock_create.assert_called_with(self.waveform_data_source, request, 10)
    self.assertEqual('test data', response.waveform_datatable)
    self.assertEqual(0, response.channel_data_ids[0].single_channel.index)

  @mock.patch.object(waveform_data_service, '_CreateChunkDataTableJSon')
  def testGetChunkReturns_IndexRaisesValueError(self, mock_create):
    with self.assertRaises(ValueError):
      request = data_pb2.DataRequest()
      request.chunk_duration_secs = 10
      request.chunk_start = 21
      channel_data_id = request.channel_data_ids.add()
      channel_data_id.single_channel.index = 0
      request.low_cut = 1.0
      request.high_cut = 70.0
      request.notch = 60.0

      waveform_data_service.GetChunk(self.waveform_data_source, request, 10)
    mock_create.assert_not_called()


if __name__ == '__main__':
  absltest.main()
