# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for kws_streaming.layers.average_pooling2d."""

import numpy as np
from kws_streaming.layers import average_pooling2d
from kws_streaming.layers import modes
from kws_streaming.layers import stream
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.models import utils
from kws_streaming.train import test
tf1.disable_eager_execution()


class AveragePooling2DTest(tf.test.TestCase):

  def setUp(self):
    super(AveragePooling2DTest, self).setUp()
    test_utils.set_seed(123)

  def test_average_pooling_stream(self):

    # prepare input data
    params = test_utils.Params([1])
    params.desired_samples = 5

    batch_size = 1
    time1 = params.desired_samples  # it is time dim (will not be averaged out)
    time2 = 3  # this dim will be averaged out and become 1
    feature = 16  # it is a feature dim

    # override data shape for streaming mode testing
    params.preprocess = 'custom'
    params.data_shape = (1, time2, feature)

    inp_audio = np.random.rand(batch_size, time1, time2, feature)
    inputs = tf.keras.layers.Input(
        shape=(time1, time2, feature), batch_size=batch_size)

    net = stream.Stream(
        cell=average_pooling2d.AveragePooling2D(
            kernel_size=(time1, time2),
            padding='valid'),
        use_one_step=False,
        pad_time_dim='causal')(inputs)

    model = tf.keras.Model(inputs, net)
    model.summary()

    # prepare streaming model
    model_stream = utils.to_streaming_inference(
        model, params, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    model_stream.summary()

    # run inference and compare streaming vs non streaming
    non_stream_out = model.predict(inp_audio)
    stream_out = test.run_stream_inference(params, model_stream, inp_audio)
    self.assertAllClose(stream_out, non_stream_out)

    net = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    model_global = tf.keras.Model(inputs, net)
    model_global.summary()

    global_out = model_global.predict(inp_audio)
    # last result in streaming output has to be the same with global average
    self.assertAllClose(stream_out[0, -1, 0, :], global_out[0, :])


if __name__ == '__main__':
  tf.test.main()
