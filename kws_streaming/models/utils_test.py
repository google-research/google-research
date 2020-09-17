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
from absl.testing import parameterized
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
from kws_streaming.models import model_params
from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.train import model_flags
tf1.disable_eager_execution()

FLAGS = flags.FLAGS


# two models are tested with all cobinations of speech frontend
# and all models are tested with one frontend
class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def _testTFLite(self,
                  preprocess='raw',
                  feature_type='mfcc_op',
                  model_name='svdf'):
    params = model_params.HOTWORD_MODEL_PARAMS[model_name]
    params.clip_duration_ms = 100  # make it shorter for testing

    # set parameters to test
    params.preprocess = preprocess
    params.feature_type = feature_type
    params = model_flags.update_flags(params)

    # create model
    model = models.MODELS[params.model_name](params)

    # convert TF non streaming model to TFLite non streaming inference
    self.assertTrue(
        utils.model_to_tflite(self.sess, model, params,
                              Modes.NON_STREAM_INFERENCE))

  def setUp(self):
    super(UtilsTest, self).setUp()
    tf1.reset_default_graph()
    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(self.sess)

  @parameterized.named_parameters([
      {
          'testcase_name': 'raw with mfcc_tf',
          'preprocess': 'raw',
          'feature_type': 'mfcc_tf'
      },
      {
          'testcase_name': 'raw with mfcc_op',
          'preprocess': 'raw',
          'feature_type': 'mfcc_op'
      },
      {
          'testcase_name': 'mfcc',
          'preprocess': 'mfcc',
          'feature_type': 'mfcc_op'
      },  # feature_type will be ignored
      {
          'testcase_name': 'micro',
          'preprocess': 'micro',
          'feature_type': 'mfcc_op'
      },  # feature_type will be ignored
  ])
  def testPreprocessNonStreamInferenceTFandTFLite(self,
                                                  preprocess,
                                                  feature_type,
                                                  model_name='svdf'):
    # Validate that model with different preprocessing
    # can be converted to non stream inference mode.
    self._testTFLite(preprocess, feature_type, model_name)

  @parameterized.named_parameters([
      {
          'testcase_name': 'raw with mfcc_tf',
          'preprocess': 'raw',
          'feature_type': 'mfcc_tf'
      },
      {
          'testcase_name': 'raw with mfcc_op',
          'preprocess': 'raw',
          'feature_type': 'mfcc_op'
      },
      {
          'testcase_name': 'mfcc',
          'preprocess': 'mfcc',
          'feature_type': 'mfcc_op'
      },  # feature_type will be ignored
      {
          'testcase_name': 'micro',
          'preprocess': 'micro',
          'feature_type': 'mfcc_op'
      },  # feature_type will be ignored
  ])
  def testPreprocessStreamInferenceModeTFandTFLite(self,
                                                   preprocess,
                                                   feature_type,
                                                   model_name='gru'):
    # Validate that model with different preprocessing
    # can be converted to stream inference mode with TF and TFLite.
    params = model_params.HOTWORD_MODEL_PARAMS[model_name]
    # set parameters to test
    params.preprocess = preprocess
    params.feature_type = feature_type
    params = model_flags.update_flags(params)

    # create model
    model = models.MODELS[params.model_name](params)

    # convert TF non streaming model to TFLite streaming inference
    # with external states
    self.assertTrue(utils.model_to_tflite(
        self.sess, model, params, Modes.STREAM_EXTERNAL_STATE_INFERENCE))

    # convert TF non streaming model to TF streaming with external states
    self.assertTrue(utils.to_streaming_inference(
        model, params, Modes.STREAM_EXTERNAL_STATE_INFERENCE))

    # convert TF non streaming model to TF streaming with internal states
    self.assertTrue(utils.to_streaming_inference(
        model, params, Modes.STREAM_INTERNAL_STATE_INFERENCE))

  def test_model_to_saved(self, model_name='dnn'):
    """SavedModel supports both stateless and stateful graphs."""
    params = model_params.HOTWORD_MODEL_PARAMS[model_name]
    params = model_flags.update_flags(params)

    # create model
    model = models.MODELS[params.model_name](params)
    utils.model_to_saved(model, params, FLAGS.test_tmpdir)

  def testNextPowerOfTwo(self):
    self.assertEqual(utils.next_power_of_two(11), 16)

  @parameterized.parameters('att_mh_rnn', 'att_rnn', 'dnn', 'ds_cnn', 'cnn',
                            'tc_resnet', 'crnn', 'gru', 'lstm', 'svdf',
                            'mobilenet', 'mobilenet_v2', 'xception',
                            'inception', 'inception_resnet', 'ds_tc_resnet')
  def testNonStreamInferenceTFandTFLite(self, model_name):
    # Validate that all models with selected preprocessing
    # can be converted to non stream inference mode.
    self._testTFLite(model_name=model_name)


if __name__ == '__main__':
  tf.test.main()
