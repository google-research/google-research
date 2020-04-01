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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl.testing import absltest
import flask
import mock
import numpy as np

from six.moves import range
import tensorflow.compat.v1 as tf
from google.protobuf import timestamp_pb2

from eeg_modelling.eeg_viewer import server
from eeg_modelling.eeg_viewer import signal_helper
from eeg_modelling.pyprotos import data_pb2
from eeg_modelling.pyprotos import event_pb2
from eeg_modelling.pyprotos import prediction_output_pb2

TEST_TFEX_PATH = '/test/path.txt'
TEST_PRED_PATH = '/test/path2.txt'
TEST_KEY = '/test/key0.txt'

TEST_ANNOTATION = event_pb2.Event()
TEST_ANNOTATION.label = 'test annotation'
TEST_ANNOTATION.start_time_sec = 1

TEST_TIME = timestamp_pb2.Timestamp()
TEST_TIME.seconds = 10
TEST_TIME.nanos = 800000000

TEST_TF_EX = tf.train.Example()
feature = TEST_TF_EX.features.feature
feature['eeg_channel/num_samples'].int64_list.value.append(4)
feature['eeg_channel/sampling_frequency_hz'].float_list.value.append(1.0)
feature['test_feature'].float_list.value.extend([i * 2 for i in range(20)])
feature['eeg_channel/EEG test_min-REF/samples'].float_list.value.extend(
    [1, 2, 3, 4])
feature['eeg_channel/EEG test_sub-REF/samples'].float_list.value.extend(
    [1, 1, 1, 1])
feature['raw_label_events'].bytes_list.value.append(
    TEST_ANNOTATION.SerializeToString())
feature['start_time'].bytes_list.value.append(TEST_TIME.SerializeToString())
feature['segment/patient_id'].bytes_list.value.append('test id')

TEST_PRED = prediction_output_pb2.PredictionOutputs()
pred_output = TEST_PRED.prediction_output.add()
pred_output.chunk_info.chunk_id = 'test chunk'
pred_output.chunk_info.chunk_start_time.seconds = 10
pred_output.chunk_info.chunk_start_time.nanos = 800000000
pred_output.chunk_info.chunk_size_sec = 2
label = pred_output.label.add()
label.name = 'test label'
label.predicted_value.score = 3
label.actual_value.score = 1
label.attribution_map.attribution.extend(list(range(1, 9)))
label.attribution_map.width = 4
label.attribution_map.height = 2
label.attribution_map.feature_names.extend([
    'eeg_channel/EEG test_sub-REF/samples#eeg_channel/EEG test_min-REF/samples',
    'eeg_channel/EEG test_min-REF/samples#eeg_channel/EEG test_sub-REF/samples',
])


class ServerTest(absltest.TestCase):

  def setUp(self):
    super(ServerTest, self).setUp()
    server.flask_app.testing = True
    server.flask_app.request_class = server.Request

    server.RegisterErrorHandlers(server.flask_app)
    server.RegisterRoutes(server.flask_app)

  @mock.patch.object(flask, 'render_template')
  def testIndexPage(self, mock_render):
    server.IndexPage()
    mock_render.assert_called_once()
    mock_render.assert_called_with('index.html', file_type='EEG')

  @mock.patch.object(signal_helper, 'downsample_attribution')
  @mock.patch.object(signal_helper, 'threshold_attribution')
  @mock.patch.object(server, 'FetchPredictionsFromFile')
  @mock.patch.object(server, 'FetchTfExFromFile')
  def testRequestData(self, mock_tf_ex, mock_pred, mock_threshold,
                      mock_downsample):
    mock_tf_ex.return_value = TEST_TF_EX
    mock_pred.return_value = TEST_PRED
    mock_threshold.side_effect = lambda x: x
    mock_downsample.side_effect = lambda x, ratio=1: np.array(x)

    request = data_pb2.DataRequest()
    request.tf_ex_file_path = TEST_TFEX_PATH
    request.prediction_file_path = TEST_PRED_PATH
    request.chunk_duration_secs = 1
    request.chunk_start = 0
    channel_data_id = request.channel_data_ids.add()
    channel_data_id.bipolar_channel.index = 0
    channel_data_id.bipolar_channel.referential_index = 1
    request.low_cut = 0
    request.high_cut = 0
    request.notch = 0
    request = request.SerializeToString()
    request_path = '/waveform_data/chunk'

    with server.flask_app.test_request_context(request_path, method='POST',
                                               data=request):
      rv = server.flask_app.dispatch_request()

    response = data_pb2.DataResponse.FromString(rv.data)

    self.assertEqual({
        'cols': [
            {'id': 'seconds',
             'type': 'number',
             'label': 'seconds'},
            {'id': 'TEST_MIN-TEST_SUB',
             'type': 'number',
             'label': 'TEST_MIN-TEST_SUB'},
        ],
        'rows': [
            {'c': [{'v': 0.0}, {'v': 0.0}]},
        ]}, json.loads(response.waveform_chunk.waveform_datatable))
    channel_id = response.waveform_chunk.channel_data_ids[0]
    self.assertEqual(0, channel_id.bipolar_channel.index)
    self.assertEqual(1, channel_id.bipolar_channel.referential_index)
    self.assertEqual(10.8, response.waveform_metadata.abs_start)
    self.assertEqual(1, response.waveform_metadata.labels[0].start_time)
    self.assertEqual('test annotation',
                     response.waveform_metadata.labels[0].label_text)
    self.assertEqual('TEST_MIN', response.waveform_metadata.channel_dict['0'])
    self.assertEqual('TEST_SUB', response.waveform_metadata.channel_dict['1'])
    self.assertEqual('EEG', response.waveform_metadata.file_type)
    self.assertEqual(4, response.waveform_metadata.num_secs)
    self.assertEqual('test id', response.waveform_metadata.patient_id)

    self.assertIn('test label', response.prediction_chunk.attribution_data)
    attribution_data = response.prediction_chunk.attribution_data['test label']
    self.assertIn('0-1', attribution_data.attribution_map)
    attribution = attribution_data.attribution_map['0-1'].attribution
    self.assertEqual([0.625, 0.75, 0.875, 1.0], attribution)
    chunk_score_data = response.prediction_metadata.chunk_scores[0]
    self.assertEqual(2.0, chunk_score_data.duration)
    self.assertEqual(0.0, chunk_score_data.start_time)
    self.assertIn('test label', chunk_score_data.score_data)
    score_data = chunk_score_data.score_data['test label']
    self.assertEqual(3.0, score_data.predicted_value)
    self.assertEqual(1.0, score_data.actual_value)


if __name__ == '__main__':
  absltest.main()
