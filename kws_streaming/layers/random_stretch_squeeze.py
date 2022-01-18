# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Augment audio data with random stretchs and squeeze."""
from kws_streaming.layers.compat import tf
from tensorflow.python.keras.utils import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import array_ops  # pylint: disable=g-direct-tensorflow-import


@tf.function
def random_stretch_squeeze(inputs,
                           resample_offset,
                           seed=None):
  """Stretches and squeezes audio data in time dim.

  It can be useful for augmenting training data
  with random stretchs squeezes in time dim
  for making model more robust to input audio sampling frequency
  and human speech frequency.

  Args:
    inputs: input tensor [batch_size, time]
    resample_offset: defines stretch squeeze range:
      1-resample_offset...1+resample_offset
    seed: random seed
  Returns:
    masked image
  Raises:
    ValueError: if inputs.shape.rank != 2
  """
  if inputs.shape.rank != 2:
    raise ValueError('inputs.shape.rank:%d must be 2' % inputs.shape.rank)

  inputs_shape = inputs.shape.as_list()
  batch_size = inputs_shape[0]
  sequence_length = inputs_shape[1]

  image = tf.expand_dims(inputs, 2)  # feature
  image = tf.expand_dims(image, 3)  # channels

  resample = 1.0  # when it is equal to 1 - no stretching or squeezing
  time_stretch_squeeze = tf.random.uniform(
      shape=[batch_size],
      minval=resample - resample_offset,
      maxval=resample + resample_offset,
      dtype=tf.float32,
      seed=seed)
  tf.print(time_stretch_squeeze)
  print(time_stretch_squeeze)
  shape = tf.shape(inputs)
  outputs = tf.TensorArray(inputs.dtype, 0, dynamic_size=True)
  for i in tf.range(batch_size):
    image_resized = tf.image.resize(
        images=image[i],
        size=(tf.cast((tf.cast(shape[1], tf.float32) * time_stretch_squeeze[i]),
                      tf.int32), 1),
        preserve_aspect_ratio=False)
    image_resized_cropped = tf.image.resize_with_crop_or_pad(
        image_resized,
        target_height=sequence_length,
        target_width=1,
    )

    outputs = outputs.write(i, image_resized_cropped)

  outputs = tf.squeeze(outputs.stack(), axis=[2, 3])
  outputs.set_shape(inputs_shape)
  return outputs


class RandomStretchSqueeze(tf.keras.layers.Layer):
  """Randomly stretches and squeezes audio data in time dim.

  It can be useful for augmenting training data
  with random stretchs squeezes in time dim
  for making model more robust to input audio sampling frequency
  and human speech frequency.

  Attributes:
    resample_offset: defines stretch squeeze range:
      1-resample_offset...1+resample_offset - it can be considered as
      audio frequency multipler, so that it audio will sound
      with higher or lower pitch.
    seed: random seed
    **kwargs: additional layer arguments
  """

  def __init__(self,
               resample_offset=0.0,
               seed=None,
               **kwargs):
    super().__init__(**kwargs)
    self.resample_offset = resample_offset
    self.seed = seed

  def call(self, inputs, training=None):

    if inputs.shape.rank != 2:  # [batch, time]
      raise ValueError('inputs.shape.rank:%d must be 2' % inputs.shape.rank)

    if self.resample_offset == 0.0:
      return inputs

    if training is None:
      training = tf.keras.backend.learning_phase()

    # pylint: disable=g-long-lambda
    return control_flow_util.smart_cond(
        training, lambda: random_stretch_squeeze(
            inputs,
            self.resample_offset,
            seed=self.seed), lambda: array_ops.identity(inputs))
    # pylint: enable=g-long-lambda

  def get_config(self):
    config = {
        'resample_offset': self.resample_offset,
        'seed': self.seed,
    }
    base_config = super(RandomStretchSqueeze, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
