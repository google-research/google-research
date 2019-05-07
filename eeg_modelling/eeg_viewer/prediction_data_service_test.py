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

from absl.testing import absltest
import mock
import numpy as np
from six.moves import range
import tensorflow as tf

from google.protobuf import timestamp_pb2
from eeg_modelling.eeg_viewer import data_source
from eeg_modelling.eeg_viewer import prediction_data_service
from eeg_modelling.eeg_viewer import signal_helper
from eeg_modelling.pyprotos import data_pb2
from eeg_modelling.pyprotos import prediction_output_pb2


class PredictionDataServiceTest(absltest.TestCase):

  def setUp(self):
    super(PredictionDataServiceTest, self).setUp()
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
    time = timestamp_pb2.Timestamp()
    time.seconds = 10
    time.nanos = 800000000
    feature['start_time'].bytes_list.value.append(time.SerializeToString())
    feature['segment/patient_id'].bytes_list.value.append(b'test patient')
    waveform_data_source = data_source.TfExampleEegDataSource(tf_ex, 'test key')
    pred_outputs = prediction_output_pb2.PredictionOutputs()
    pred_output = pred_outputs.prediction_output.add()
    pred_output.chunk_info.chunk_id = 'test chunk'
    pred_output.chunk_info.chunk_start_time.seconds = 10
    pred_output.chunk_info.chunk_start_time.nanos = 800000000
    pred_output.chunk_info.chunk_size_sec = 2
    label = pred_output.label.add()
    label.name = 'test label'
    pred_output_2 = pred_outputs.prediction_output.add()
    pred_output_2.chunk_info.chunk_id = 'test chunk 2'
    pred_output_2.chunk_info.chunk_start_time.seconds = 12
    pred_output_2.chunk_info.chunk_start_time.nanos = 800000000
    pred_output_2.chunk_info.chunk_size_sec = 2
    label = pred_output_2.label.add()
    label.name = 'test label'
    self._pred = prediction_data_service.PredictionDataService(
        pred_outputs, waveform_data_source, 100)

  @mock.patch.object(signal_helper, 'downsample_attribution')
  @mock.patch.object(signal_helper, 'threshold_attribution')
  def testGetChunk(self, mock_threshold, mock_downsample):
    label = self._pred._prediction_outputs.prediction_output[0].label[0]
    label.attribution_map.attribution.extend([0, 2, 5, 6, 6, 3, 10, 9])
    label.attribution_map.width = 4
    label.attribution_map.height = 2
    label.attribution_map.feature_names.extend([
        ('eeg_channel/EEG test_sub-REF/samples#'
         'eeg_channel/EEG test_min-REF/samples'),
        ('eeg_channel/EEG test_min-REF/samples#'
         'eeg_channel/EEG test_sub-REF/samples'),
    ])

    label = self._pred._prediction_outputs.prediction_output[1].label[0]
    label.attribution_map.attribution.extend([9, 10, 5, 6, 5, 8, 9, 2])
    label.attribution_map.width = 4
    label.attribution_map.height = 2
    label.attribution_map.feature_names.extend([
        ('eeg_channel/EEG test_sub-REF/samples#'
         'eeg_channel/EEG test_min-REF/samples'),
        ('eeg_channel/EEG test_min-REF/samples#'
         'eeg_channel/EEG test_sub-REF/samples'),
    ])

    mock_threshold.side_effect = lambda x: x
    mock_downsample.side_effect = lambda x, ratio=1: np.array(x)

    request = data_pb2.DataRequest()
    request.chunk_start = 2
    request.chunk_duration_secs = 2
    channel_id = request.channel_data_ids.add()
    channel_id.bipolar_channel.index = 0
    channel_id.bipolar_channel.referential_index = 1
    channel_id = request.channel_data_ids.add()
    channel_id.single_channel.index = 0

    response = self._pred.GetChunk(request)

    self.assertIn('test label', response.attribution_data)
    attribution_map = response.attribution_data['test label'].attribution_map
    self.assertIn('0-1', attribution_map)
    self.assertIn('0', attribution_map)
    attribution_values = attribution_map['0-1'].attribution
    self.assertAlmostEqual(0.5, attribution_values[0])
    self.assertAlmostEqual(0.8, attribution_values[1])
    self.assertAlmostEqual(0.9, attribution_values[2])
    self.assertAlmostEqual(0.2, attribution_values[3])
    self.assertLen(attribution_values, 4)
    self.assertEqual([0, 0, 0, 0], attribution_map['0'].attribution)
    self.assertEqual(2, response.chunk_start)
    self.assertEqual(2, response.chunk_duration)

  def testGetMetadata(self):
    label = self._pred._prediction_outputs.prediction_output[0].label[0]
    label.predicted_value.score = 3
    label.actual_value.score = 1

    label = self._pred._prediction_outputs.prediction_output[1].label[0]
    label.predicted_value.score = 5
    label.actual_value.score = 2

    response = self._pred.GetMetadata()

    scores = response.chunk_scores
    self.assertLen(scores, 2)
    self.assertAlmostEqual(0, scores[0].start_time)
    self.assertAlmostEqual(2, scores[0].duration)
    self.assertEqual(3, scores[0].score_data['test label'].predicted_value)
    self.assertEqual(1, scores[0].score_data['test label'].actual_value)
    self.assertAlmostEqual(2, scores[1].start_time)
    self.assertAlmostEqual(2, scores[1].duration)
    self.assertEqual(5, scores[1].score_data['test label'].predicted_value)
    self.assertEqual(2, scores[1].score_data['test label'].actual_value)


if __name__ == '__main__':
  absltest.main()
