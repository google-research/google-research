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

"""Tests for kws_streaming.models.utils."""

from absl import flags
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
from kws_streaming.models import dnn
from kws_streaming.models import utils

FLAGS = flags.FLAGS


class Flags(object):  # dummy class for creating tmp flags structure
  pass


class UtilsTest(tf.test.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    tf1.reset_default_graph()
    self.sess = tf1.Session()
    tf1.keras.backend.set_session(self.sess)

    self.flags = Flags()
    self.flags.desired_samples = 16000
    self.flags.window_size_ms = 30.0
    self.flags.window_stride_ms = 20.0
    self.flags.sample_rate = 16000.0
    self.flags.window_stride_samples = 320
    self.flags.window_size_samples = 480
    self.flags.label_count = 3
    self.flags.preemph = 0.0
    self.flags.window_type = 'hann'
    self.flags.feature_type = 'mfcc_tf'
    self.flags.mel_num_bins = 40
    self.flags.mel_lower_edge_hertz = 20
    self.flags.mel_upper_edge_hertz = 4000
    self.flags.fft_magnitude_squared = False
    self.flags.dct_num_features = 10
    self.flags.use_tf_fft = False
    self.flags.units1 = '32'
    self.flags.act1 = "'relu'"
    self.flags.pool_size = 2
    self.flags.strides = 2
    self.flags.dropout1 = 0.1
    self.flags.units2 = '256,256'
    self.flags.act2 = "'relu','relu'"
    self.flags.train_dir = FLAGS.test_tmpdir
    self.flags.mel_non_zero_only = 1
    self.flags.batch_size = 1

    self.model = dnn.model(self.flags)
    self.model.summary()

  def test_to_streaming_inference(self):
    """Validate that model can be converted to any streaming mode with TF."""
    model_non_streaming = utils.to_streaming_inference(
        self.model, self.flags, Modes.NON_STREAM_INFERENCE)
    self.assertTrue(model_non_streaming)
    model_streaming_ext_state = utils.to_streaming_inference(
        self.model, self.flags, Modes.STREAM_EXTERNAL_STATE_INFERENCE)
    self.assertTrue(model_streaming_ext_state)
    model_streaming_int_state = utils.to_streaming_inference(
        self.model, self.flags, Modes.STREAM_INTERNAL_STATE_INFERENCE)
    self.assertTrue(model_streaming_int_state)

  def test_model_to_tflite(self):
    """TFLite supports stateless graphs."""
    self.assertTrue(utils.model_to_tflite(self.sess, self.model, self.flags))

  def test_model_to_saved(self):
    """SavedModel supports both stateless and stateful graphs."""
    utils.model_to_saved(self.model, self.flags, FLAGS.test_tmpdir)

  def testNextPowerOfTwo(self):
    self.assertEqual(utils.next_power_of_two(11), 16)


if __name__ == '__main__':
  tf.test.main()
