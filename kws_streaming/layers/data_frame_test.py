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

"""Tests for kws_streaming.layers.data_frame."""

import numpy as np
from kws_streaming.layers import data_frame
from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
import kws_streaming.layers.test_utils as tu
from kws_streaming.models import utils
tf1.disable_eager_execution()


class DataFrameTest(tu.FrameTestBase):

  def test_tf_non_streaming_vs_streaming_internal_state(self):
    # prepare streaming frame extraction model with internal state
    data_frame_stream = data_frame.DataFrame(
        mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        inference_batch_size=self.inference_batch_size,
        frame_size=self.frame_size,
        frame_step=self.frame_step)
    # it received input data incrementally with step: frame_step
    input2 = tf.keras.layers.Input(
        shape=(self.frame_step,),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    output2 = data_frame_stream(input2)
    model_stream = tf.keras.models.Model(input2, output2)

    # initialize internal state of data framer
    pre_state = self.signal[:, 0:data_frame_stream.frame_size -
                            data_frame_stream.frame_step]
    state_init = np.concatenate((np.zeros(
        shape=(1, data_frame_stream.frame_step), dtype=np.float32), pre_state),
                                axis=1)
    data_frame_stream.set_weights([state_init])

    start = self.frame_size - self.frame_step
    end = self.frame_size
    streamed_frames = []

    # run streaming frames extraction
    while end <= self.data_size:

      # next data update
      stream_update = self.signal[:, start:end]

      # get new frame from stream of data
      output_frame = model_stream.predict(stream_update)
      streamed_frames.append(output_frame)

      start = end
      end = start + self.frame_step

    # compare streaming vs non streaming frames extraction
    for i in range(0, len(self.output_frames_tf[0])):
      self.assertAllEqual(streamed_frames[i][0][0], self.output_frames_tf[0][i])

  def test_tf_non_streaming_vs_streaming_external_state(self):
    mode = modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
    input_tensors = [
        tf.keras.layers.Input(
            shape=(self.frame_step,), batch_size=1, name="inp1")
    ]

    # convert non streaming trainable model to a streaming one
    model_stream = utils.convert_to_inference_model(self.model_tf,
                                                    input_tensors, mode)

    # initialize input state of streaming data framer
    pre_state = self.signal[:, 0:self.frame_size - self.frame_step]
    states = np.concatenate(
        (np.zeros(shape=(1, self.frame_step), dtype=np.float32), pre_state),
        axis=1)

    start = self.frame_size - self.frame_step
    end = self.frame_size
    streamed_frames = []

    # run streaming frames extraction
    while end <= self.data_size:
      # next data update
      stream_update = self.signal[:, start:end]

      # get new frame from stream of data
      output_frame, new_states = model_stream.predict([stream_update, states])
      # update frame states and feed it as input in the next iteration
      states = new_states

      streamed_frames.append(output_frame)

      start = end
      end = start + self.frame_step

    # compare streaming vs non streaming frames extraction
    for i in range(0, len(self.output_frames_tf[0])):
      self.assertAllEqual(streamed_frames[i][0][0], self.output_frames_tf[0][i])


if __name__ == "__main__":
  tf.test.main()
