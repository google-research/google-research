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

"""Spectrogram Cutout augmentation for model regularization."""
from kws_streaming.layers.compat import tf
from tensorflow.python.keras.utils import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import array_ops  # pylint: disable=g-direct-tensorflow-import


def random_cutout(
    inputs,
    mask_size,
    mask_value=0,
    seed=None,
    data_format='channels_last',
):
  """Applies cutout (https://arxiv.org/abs/1708.04552) to inputs.

  It is based on addons/tensorflow_addons/image/cutout_ops.py
  kept here here for backward compatibility

  Args:
    inputs: input tensor [batch_size, time, feature, channels]
    mask_size: mask size (time feature)
    mask_value: mask will be filled with this value
    seed: random seed
    data_format: dimesnions order
  Returns:
    masked image
  Raises:
    ValueError: if inputs.shape.rank != 4
  """

  if inputs.shape.rank != 4:
    raise ValueError('inputs.shape.rank:%d must be 4' % inputs.shape.rank)

  mask_size = tf.convert_to_tensor(mask_size)
  if tf.rank(mask_size) == 0:
    mask_size = tf.stack([mask_size, mask_size])

  if data_format == 'channels_last':
    time_size, feature_size = tf.shape(inputs)[1], tf.shape(inputs)[2]
  else:
    time_size, feature_size = tf.shape(inputs)[2], tf.shape(inputs)[3]

  batch_size = tf.shape(inputs)[0]

  cutout_center_time = tf.random.uniform(
      shape=[batch_size], minval=0, maxval=time_size, dtype=tf.int32, seed=seed
  )
  cutout_center_feature = tf.random.uniform(
      shape=[batch_size],
      minval=0,
      maxval=feature_size,
      dtype=tf.int32,
      seed=seed)
  offset = tf.transpose([cutout_center_time, cutout_center_feature], [1, 0])
  origin_shape = inputs.shape
  offset = tf.convert_to_tensor(offset)
  mask_size = mask_size // 2
  cutout_center_time = offset[:, 0]
  cutout_center_feature = offset[:, 1]

  lower_pads = tf.maximum(0, cutout_center_time - mask_size[0])
  upper_pads = tf.maximum(0, time_size - cutout_center_time - mask_size[0])
  left_pads = tf.maximum(0, cutout_center_feature - mask_size[1])
  right_pads = tf.maximum(0,
                          feature_size - cutout_center_feature - mask_size[1])

  cutout_shape = tf.transpose(
      [
          time_size - (lower_pads + upper_pads),
          feature_size - (left_pads + right_pads),
      ],
      [1, 0],
  )
  masks = tf.TensorArray(inputs.dtype, 0, dynamic_size=True)
  for i in tf.range(tf.shape(cutout_shape)[0]):
    padding_dims = [
        [lower_pads[i], upper_pads[i]],
        [left_pads[i], right_pads[i]],
    ]
    mask = tf.pad(
        tf.zeros(cutout_shape[i], dtype=inputs.dtype),
        padding_dims,
        constant_values=1,
    )
    masks = masks.write(i, mask)

  if data_format == 'channels_last':
    mask = tf.expand_dims(masks.stack(), -1)
    mask = tf.tile(mask, [1, 1, 1, tf.shape(inputs)[-1]])
  else:
    mask = tf.expand_dims(masks.stack(), 1)
    mask = tf.tile(mask, [1, tf.shape(inputs)[1], 1, 1])

  inputs = tf.where(
      tf.equal(mask, 0),
      tf.ones_like(inputs, dtype=inputs.dtype) * mask_value,
      inputs,
  )
  inputs.set_shape(origin_shape)
  return inputs


class SpecCutout(tf.keras.layers.Layer):
  """Cutout data augmentation.

  Applies Cutout on speech spectrogram:
  Improved Regularization of Convolutional Neural Networks with Cutout
  https://arxiv.org/abs/1708.04552

  Attributes:
    masks_number: number of masks
    time_mask_size: mask size in time dim
    frequency_mask_size: mask size in frequency dim
    seed: seed to create a reproducible sequence of tensors on multiple calls
    **kwargs: additional layer arguments
  """

  def __init__(self,
               masks_number=2,
               time_mask_size=5,
               frequency_mask_size=2,
               seed=None,
               **kwargs):
    super(SpecCutout, self).__init__(**kwargs)
    self.masks_number = masks_number
    self.time_mask_size = time_mask_size
    self.frequency_mask_size = frequency_mask_size
    self.seed = seed

  def call(self, inputs, training=None):

    if inputs.shape.rank != 3:  # [batch, time, feature]
      raise ValueError('inputs.shape.rank:%d must be 3' % inputs.shape.rank)

    if training is None:
      training = tf.keras.backend.learning_phase()

    def masked_inputs():
      net = tf.keras.backend.expand_dims(inputs, axis=-1)
      for i in range(self.masks_number):
        net = random_cutout(
            net, (self.time_mask_size, self.frequency_mask_size),
            seed=self.seed + i if self.seed else self.seed)
      net = tf.keras.backend.squeeze(net, axis=-1)
      return net

    outputs = control_flow_util.smart_cond(training, masked_inputs,
                                           lambda: array_ops.identity(inputs))
    return outputs

  def get_config(self):
    config = {
        'masks_number': self.masks_number,
        'time_mask_size': self.time_mask_size,
        'frequency_mask_size': self.frequency_mask_size,
        'seed': self.seed,
    }
    base_config = super(SpecCutout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
