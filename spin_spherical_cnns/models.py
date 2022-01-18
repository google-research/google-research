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

"""Spin-weighted spherical CNN models.

Implements CNNs based on spin-weighted spherical convolutions, and corresponding
baselines, reproducing models from [1]. There are still missing features so
perfect reproduction is not yet achieved.


[1] Spin-Weighted Spherical CNNs, NeurIPS'20.
"""
import functools
import operator
from typing import Any, Optional, Sequence, Union
from flax import linen as nn
import jax.numpy as jnp
import numpy as np

from spin_spherical_cnns import layers
from spin_spherical_cnns import sphere_utils
from spin_spherical_cnns import spin_spherical_harmonics


Array = Union[np.ndarray, jnp.ndarray]

# All classifiers used for spherical MNIST in the original SWSCNN paper [1] have
# the same resolution per layer.
_SMALL_CLASSIFIER_RESOLUTIONS = (64, 64, 64, 32, 32, 16, 16)


class SpinSphericalBlock(nn.Module):
  """Spin-weighted block with pooling, convolution, batch norm and nonlinearity.

  Attributes:
    num_channels: Number of convolution output channels.
    spins_in: A sequence of input spin weights.
    spins_out: A sequence of output spin weights.
    downsampling_factor: How much to downsample before applying the
      convolution. Will not downsample when downsampling_factor==1.
    axis_name: Identifier for the mapped axis in parallel training.
    transformer: SpinSphericalFourierTransformer instance.
    num_filter_params: Number of filter parameters in the convolutional layer.
  """
  num_channels: int
  spins_in: Sequence[int]
  spins_out: Sequence[int]
  downsampling_factor: int
  axis_name: Any
  transformer: spin_spherical_harmonics.SpinSphericalFourierTransformer
  num_filter_params: Optional[int] = None

  @nn.compact
  def __call__(self, inputs, train):
    """Apply block to `inputs`.

    Args:
      inputs: (batch_size, resolution, resolution, n_spins_in, n_channels_in)
        array of spin-weighted spherical functions with equiangular sampling.
      train: whether to run in training or inference mode.
    Returns:
      A (batch_size, resolution // downsampling_factor, resolution //
        downsampling_factor, n_spins_out, num_channels) complex64 array.
    """
    feature_maps = inputs
    if self.downsampling_factor != 1:
      feature_maps = layers.SphericalPooling(
          stride=self.downsampling_factor, name='spherical_pool')(feature_maps)

    feature_maps = layers.SpinSphericalConvolution(
        features=self.num_channels,
        spins_in=self.spins_in,
        spins_out=self.spins_out,
        num_filter_params=self.num_filter_params,
        transformer=self.transformer,
        name='spherical_conv')(feature_maps)

    return layers.SpinSphericalBatchNormalizationNonlinearity(
        spins=self.spins_out,
        use_running_stats=not train,
        axis_name=self.axis_name,
        name='batch_norm_nonlin')(feature_maps)


class SpinSphericalClassifier(nn.Module):
  """Construct a spin-weighted spherical CNN for classification.

  Attributes:
    num_classes: Number of nodes in the final layer.
    resolutions: (n_layers,) list of resolutions at each layer. For consecutive
      resolutions a, b, we must have either a == b or a == 2*b. The latter
      triggers inclusion of a pooling layer.
    spins: A (n_layers,) list of (n_spins,) lists of spin weights per layer.
    widths: (n_layers,) list of width per layer (number of channels).
    axis_name: Identifier for the mapped axis in parallel training.
    num_filter_params: (n_layers,) the number of filter parameters per layer.
    input_transformer: None, or SpinSphericalFourierTransformer
      instance. Will be computed automatically if None.
  """
  num_classes: int
  resolutions: Sequence[int]
  spins: Sequence[Sequence[int]]
  widths: Sequence[int]
  axis_name: Any
  num_filter_params: Optional[Sequence[int]] = None
  input_transformer: Optional[
      spin_spherical_harmonics.SpinSphericalFourierTransformer] = None

  def setup(self):
    if self.input_transformer is None:
      # Flatten spins.
      all_spins = functools.reduce(operator.concat, self.spins)
      self.transformer = spin_spherical_harmonics.SpinSphericalFourierTransformer(
          resolutions=np.unique(self.resolutions),
          spins=np.unique(all_spins))
    else:
      self.transformer = self.input_transformer

    num_layers = len(self.resolutions)
    if len(self.spins) != num_layers or len(self.widths) != num_layers:
      raise ValueError('resolutions, spins, and widths must be the same size!')
    model_layers = []
    for layer_id in range(num_layers - 1):
      resolution_in = self.resolutions[layer_id]
      resolution_out = self.resolutions[layer_id + 1]
      spins_in = self.spins[layer_id]
      spins_out = self.spins[layer_id + 1]
      if self.num_filter_params is None:
        num_filter_params = None
      else:
        num_filter_params = self.num_filter_params[layer_id + 1]

      num_channels = self.widths[layer_id + 1]

      # We pool before conv to avoid expensive increase of number of channels at
      # higher resolution.
      if resolution_out == resolution_in // 2:
        downsampling_factor = 2
      elif resolution_out != resolution_in:
        raise ValueError('Consecutive resolutions must be equal or halved.')
      else:
        downsampling_factor = 1

      model_layers.append(
          SpinSphericalBlock(num_channels=num_channels,
                             spins_in=spins_in,
                             spins_out=spins_out,
                             downsampling_factor=downsampling_factor,
                             num_filter_params=num_filter_params,
                             axis_name=self.axis_name,
                             transformer=self.transformer,
                             name=f'spin_block_{layer_id}'))

    self.layers = model_layers

    self.final_dense = nn.Dense(self.num_classes, name='final_dense')

  def __call__(self, inputs, train):
    """Apply the network to `inputs`.

    Args:
      inputs: (batch_size, resolution, resolution, n_spins, n_channels) array of
        spin-weighted spherical functions (SWSF) with equiangular sampling.
      train: whether to run in training or inference mode.
    Returns:
      A (batch_size, num_classes) float32 array with per-class scores (logits).
    """
    resolution, num_spins, num_channels = inputs.shape[2:]
    if (resolution != self.resolutions[0] or
        num_spins != len(self.spins[0]) or
        num_channels != self.widths[0]):
      raise ValueError('Incorrect input dimensions!')

    feature_maps = inputs
    for layer in self.layers:
      feature_maps = layer(feature_maps, train=train)

    # Current feature maps are still spin spherical. Do final processing.
    # Global pooling is not equivariant for spin != 0, so me must take the
    # absolute values before.
    mean_abs = sphere_utils.spin_spherical_mean(jnp.abs(feature_maps))
    mean = sphere_utils.spin_spherical_mean(feature_maps).real
    spins = jnp.expand_dims(jnp.array(self.spins[-1]), [0, 2])
    feature_maps = jnp.where(spins == 0, mean, mean_abs)
    # Shape is now (batch, spins, channel).
    feature_maps = feature_maps.reshape((feature_maps.shape[0], -1))

    return self.final_dense(feature_maps)


class CNNClassifier(nn.Module):
  """Construct a conventional CNN for classification.

  This serves as a baseline. It takes the same inputs as the spin models and
  uses the same format for number of layers, resolutions and channels per layer.

  Attributes:
    num_classes: Number of nodes in the final layer.
    resolutions: (num_layers,) list of resolutions at each layer. For
      consecutive resolutions a, b, we must have either a == b or a == 2*b. The
      latter triggers inclusion of a pooling layer.
    widths: (num_layers,) list of widths per layer (number of channels).
    axis_name: Identifier for the mapped axis in parallel training.
  """
  num_classes: int
  resolutions: Sequence[int]
  widths: Sequence[int]
  axis_name: Any

  @nn.compact
  def __call__(self, inputs, train):
    """Applies the network to inputs.

    Args:
      inputs: (batch_size, resolution, resolution, n_spins, n_channels) array.
      train: whether to run in training or inference mode.
    Returns:
      A (batch_size, num_classes) float32 array with per-class scores (logits).
    Raises:
      ValueError: If resolutions cannot be enforced with 2x2 pooling.
    """
    num_layers = len(self.resolutions)
    # Merge spin and channel dimensions.
    features = inputs.reshape((*inputs.shape[:3], -1))
    for layer_id in range(num_layers - 1):
      resolution_in = self.resolutions[layer_id]
      resolution_out = self.resolutions[layer_id + 1]
      n_channels = self.widths[layer_id + 1]

      if resolution_out == resolution_in // 2:
        features = nn.avg_pool(features,
                               window_shape=(2, 2),
                               strides=(2, 2),
                               padding='SAME')
      elif resolution_out != resolution_in:
        raise ValueError('Consecutive resolutions must be equal or halved.')

      features = nn.Conv(features=n_channels,
                         kernel_size=(3, 3),
                         strides=(1, 1))(features)

      features = nn.BatchNorm(use_running_average=not train,
                              axis_name=self.axis_name)(features)
      features = nn.relu(features)

    features = jnp.mean(features, axis=(1, 2))
    features = nn.Dense(self.num_classes)(features)

    return features


def tiny_classifier(num_classes, axis_name=None, input_transformer=None):
  """Wrapper around SpinSphericalClassifier; builds tiny model for testing."""
  return SpinSphericalClassifier(num_classes,
                                 resolutions=(8, 4),
                                 spins=((0,), (0, 1)),
                                 widths=(1, 3),
                                 axis_name=axis_name,
                                 input_transformer=input_transformer)


# The hyperparameters for small (six layers) classifiers used for spherical
# MNIST follow the original SWSCNN paper [1].
def spin_classifier_6_layers(num_classes, axis_name):
  """Returns the SpinSphericalClassifier used for spherical MNIST."""
  # Input layer has only spin zero. All others have spins zero and one.
  num_layers = len(_SMALL_CLASSIFIER_RESOLUTIONS)
  spins = tuple([(0,)] + [(0, 1)] * (num_layers - 1))
  widths = (1, 16, 16, 20, 24, 28, 32)
  num_filter_params_per_layer = (1, 6, 6, 4, 4, 3, 3)
  return SpinSphericalClassifier(num_classes,
                                 resolutions=_SMALL_CLASSIFIER_RESOLUTIONS,
                                 spins=spins,
                                 widths=widths,
                                 num_filter_params=num_filter_params_per_layer,
                                 axis_name=axis_name)


def spherical_classifier_6_layers(num_classes, axis_name):
  """Returns the Spherical CNN baseline used for spherical MNIST."""
  num_layers = len(_SMALL_CLASSIFIER_RESOLUTIONS)
  widths = (1, 16, 16, 32, 32, 58, 58)
  num_filter_params_per_layer = tuple([8] * num_layers)
  # The difference between spherical and spin-weighted models is that spins are
  # zero in every layer for the spherical.
  spins = tuple([(0,)] * num_layers)
  return SpinSphericalClassifier(num_classes,
                                 resolutions=_SMALL_CLASSIFIER_RESOLUTIONS,
                                 spins=spins,
                                 widths=widths,
                                 num_filter_params=num_filter_params_per_layer,
                                 axis_name=axis_name)


def cnn_classifier_6_layers(num_classes, axis_name):
  """Returns the conventional CNN baseline used for spherical MNIST."""
  widths = (1, 16, 16, 32, 32, 54, 54)
  return CNNClassifier(num_classes,
                       resolutions=_SMALL_CLASSIFIER_RESOLUTIONS,
                       widths=widths,
                       axis_name=axis_name)
