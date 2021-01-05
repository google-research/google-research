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

"""A layer which applies preemphasis on input speech data."""
from kws_streaming.layers.compat import tf


class Preemphasis(tf.keras.layers.Layer):
  """Apply high pass filter on input data.

  This is useful for amplifying high frequency bands
  and decreasing amplitudes of lower bands.
  There are several versions of preemphasis. Here we support:

  Method1 y = [x[0] * (1 - preemph),  x[i] - preemph * x[i-1]],
          where i = 1..N-1
  """

  def __init__(self, preemph=0.97, **kwargs):
    super(Preemphasis, self).__init__(**kwargs)
    self.preemph = preemph

  def call(self, inputs, training=None):
    # last dim is frame with features
    frame_axis = inputs.shape.rank - 1

    # Makes general slice tuples. This would be equivalent to the [...]
    # slicing sugar, if we knew which axis we wanted.
    def make_framed_slice(start, stop):
      s = [slice(None)] * inputs.shape.rank
      s[frame_axis] = slice(start, stop)
      return tuple(s)

    # Slice containing the first frame element.
    slice_0 = make_framed_slice(0, 1)
    # Slice containing the rightmost frame_size-1 elements.
    slice_right = make_framed_slice(1, None)
    # Slice containing the leftmost frame_size-1 elements.
    slice_left = make_framed_slice(0, -1)

    preemphasized = tf.concat(
        (inputs[slice_0] * (1 - self.preemph),
         inputs[slice_right] - self.preemph * inputs[slice_left]),
        axis=frame_axis)
    return preemphasized

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'preemph': self.preemph}
    base_config = super(Preemphasis, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
