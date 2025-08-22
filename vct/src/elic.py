# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Building blocks from ELIC and (Cheng 2020).

Blocks include the residual block and simplified attention from (Cheng 2020):

Learned Image Compression with Discretized Gaussian Mixture Likelihoods and
Attention Modules
https://arxiv.org/abs/2001.01568

Reference implementation by the authors:
https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention

And the analysis and synthesis transforms from ELIC (He 2022):

ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel
Contextual Adaptive Coding
https://arxiv.org/abs/2203.10886
"""
import functools
from typing import Optional, Sequence, Union, Callable
import tensorflow as tf

_Act = Union[str, Callable[Ellipsis, tf.Tensor], None]


class ResidualBlock(tf.keras.layers.Layer):
  """Residual block from (Cheng 2020) and ELIC.

  (Cheng 2020) = https://arxiv.org/abs/2001.01568

  Reference PyTorch code from CompressAI:
  https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/layers.py#L208

  Reference TF code from Cheng:
  https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py#L41
  """

  def __init__(self, activation = "relu", **kwargs):
    super().__init__(**kwargs)
    self._activation = activation

  def build(self, input_shape):
    c = input_shape[-1]
    # Conv layers = [1x1 @ N/2, 3x3 @ N/2, 1x1 @ N].
    self._block = tf.keras.Sequential([
        build_conv(output_channels=c // 2, kernel_size=1, act=self._activation),
        build_conv(output_channels=c // 2, kernel_size=3, act=self._activation),
        build_conv(output_channels=c, kernel_size=1, act=None),
    ])

  def call(self, x):
    x += self._block(x)
    return x


class SimpleAttention(tf.keras.layers.Layer):
  """Simplified attention block from (Cheng 2020).

  (Cheng 2020) = https://arxiv.org/abs/2001.01568

  Reference PyTorch code from CompressAI:
  https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/layers.py#L193

  Reference TF code from Cheng:
  https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py#L67

  Devil is in the Details (Zhou 2022) = https://arxiv.org/abs/2203.08450
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._trunk = tf.keras.Sequential([ResidualBlock("relu") for _ in range(3)],
                                      name="trunk")
    self._branch_layers = [ResidualBlock("relu") for _ in range(3)]

  def build(self, input_shape):
    conv = build_conv(
        output_channels=input_shape[-1], kernel_size=1, act=tf.nn.sigmoid)
    self._attention_branch = tf.keras.Sequential(
        self._branch_layers + [conv], name="attention_branch")

  def call(self, x):
    trunk = self._trunk(x)
    attention = self._attention_branch(x)
    return x + tf.multiply(trunk, attention, name="gating")


class ElicAnalysis(tf.keras.layers.Layer):
  """Analysis transform from ELIC.

  Can be configured to match the analysis transform from the "Devil's in the
  Details" paper or a combination of ELIC + Devil.

  ELIC = https://arxiv.org/abs/2203.10886
  Devil = https://arxiv.org/abs/2203.08450

  Note that the paper uses channels = [192, 192, 192, 320].
  """

  def __init__(self,
               num_residual_blocks = 3,
               channels = (128, 160, 192, 192),
               output_channels = None,
               name = "ElicAnalysis",
               **kwargs):
    super().__init__(name=name, **kwargs)
    if len(channels) != 4:
      raise ValueError(f"ELIC uses 4 conv layers (not {channels}).")
    if output_channels is not None and output_channels != channels[-1]:
      raise ValueError("output_channels specified but does not match channels: "
                       f"{output_channels} vs. {channels}")

    self._output_depth = channels[-1]

    # Keep activation separate from conv layer for clarity and symmetry.
    conv = functools.partial(
        build_conv, kernel_size=5, strides=2, act=None, up_or_down="down")
    convs = [conv(output_channels=c) for c in channels]

    rb = functools.partial(ResidualBlock, activation="relu")

    def build_act():
      return [rb() for _ in range(num_residual_blocks)]

    blocks = [
        convs[0],
        *build_act(),
        convs[1],
        *build_act(),
        SimpleAttention(),
        convs[2],
        *build_act(),
        convs[3],
        SimpleAttention(),
    ]
    blocks = list(filter(None, blocks))  # remove None elements
    self._transform = tf.keras.Sequential(blocks)

  def call(self, x, training = None):
    del training
    return self._transform(x)

  @property
  def output_depth(self):
    return self._output_depth

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    h, w = shape[-3], shape[-2]
    shape[-3:] = [h // 16, w // 16, self.output_depth]
    return tf.TensorShape(shape)


class ElicSynthesis(tf.keras.layers.Layer):
  """Synthesis transform from ELIC.

  ELIC = https://arxiv.org/abs/2203.10886
  """

  def __init__(self,
               num_residual_blocks = 3,
               channels = (192, 160, 128, 3),
               output_channels = None,
               name = "ElicSynthesis",
               **kwargs):
    super().__init__(name=name, **kwargs)
    if len(channels) != 4:
      raise ValueError(f"ELIC uses 4 conv layers (not {channels}).")
    if output_channels is not None and output_channels != channels[-1]:
      raise ValueError("output_channels specified but does not match channels: "
                       f"{output_channels} vs. {channels}")

    self._output_depth = channels[-1]

    # Keep activation separate from conv layer for clarity and because
    # second conv is followed by attention, not an activation.
    conv = functools.partial(
        build_conv, kernel_size=5, strides=2, act=None, up_or_down="up")
    convs = [conv(output_channels=c) for c in channels]

    rb = functools.partial(ResidualBlock, activation="relu")

    def build_act():
      return [rb() for _ in range(num_residual_blocks)]

    blocks = [
        SimpleAttention(),
        convs[0],
        *build_act(),
        convs[1],
        SimpleAttention(),
        *build_act(),
        convs[2],
        *build_act(),
        convs[3],
    ]
    blocks = list(filter(None, blocks))  # remove None elements
    self._transform = tf.keras.Sequential(blocks)

  def call(self, x, training = None):
    del training  # Unused.
    return self._transform(x)

  @property
  def output_depth(self):
    return self._output_depth

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    h, w = shape[-3], shape[-2]
    shape[-3:] = [h * 16, w * 16, self.output_depth]
    return tf.TensorShape(shape)


def build_conv(output_channels = 0,
               kernel_size = 3,
               strides = 1,
               act = "relu",
               up_or_down = "down",
               name = None):
  """Builds either an upsampling or downsampling conv layer."""
  layer_cls = dict(
      up=tf.keras.layers.Conv2DTranspose,
      down=tf.keras.layers.Conv2D)[up_or_down]
  return layer_cls(
      filters=output_channels,
      kernel_size=kernel_size,
      strides=strides,
      activation=act,
      use_bias=True,
      padding="same",
      name=name)
