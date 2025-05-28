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

"""Pretext Models.

All models defined here are used for pretext tasks.

When called, they should output a dictionary with a key 'pretext'
which can contain any of the pretext outputs.
"""
from typing import Optional

import flax.linen as nn
import jax
from jax import numpy as jnp

from q_match.models import models


def l2_normalize(
    x,
    axis = -1,
    epsilon = 1e-12,
):
  """L2 normalize a tensor on an axis with numerical stability."""
  norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
  return x/jnp.maximum(norm, epsilon)


class Template(nn.Module):

  @nn.compact
  def __call__(self,):
    return {'pretext': None}


class VimeMaskModel(nn.Module):
  """Predicts the mask using a sigmoid layer at the end of an MLP."""

  training: bool
  input_flat_shape: int

  def setup(self):
    self.mask_model = models.MLP(layer_sizes=[512, self.input_flat_shape],
                                 training=self.training)

  def __call__(self, inputs):
    return jax.nn.sigmoid(self.mask_model(inputs))


class Pretext(nn.Module):
  """Standard pretext model that outputs at least 4 values.

    1. An encoded version of the corruptd input (expect to be flat.), 'encoded'.
       This is to be used for downstream representations.
    3. A projection of the encoded output, useful for pretext training, 'proj'.
    2. A prediction for the corrupted input, 'reconstruction'.
    3. A prediction for the mask, 'mask_guess'.

    If the algorithm is SimSiam, there are additional prediction and projection
    heads built and outputed.

    If the algorithm is Dino, there is an additional 'proto' projection head.

    The architecture follows the iMix paper.
  """

  training: bool
  algorithm: Optional[str] = None

  @nn.compact
  def __call__(self, inputs):
    """Pretext outputs.

    Args:
      inputs: input features (batch_size, dim)

    Returns:
      Encoded data, reconstruction of the uncorrupted data, the prediction of
      the mask.
    """
    encoder = models.ImixMLP(training=self.training)
    reconstruction_model = models.MLP(layer_sizes=[512, inputs.shape[-1]],
                                      training=self.training)
    mask_model = VimeMaskModel(training=self.training,
                               input_flat_shape=inputs.shape[-1])
    mlp_projection = models.MLP(layer_sizes=[512, 128], training=self.training)

    if self.algorithm == 'dino_pretext+supervised_training':
      dino_projection = DinoPretextProjectionHead(training=self.training)

    if self.algorithm == 'simsiam_pretext+supervised_training':
      simsiam_prediction = SimSiamPretextPredictionHead(training=self.training)
      simsiam_projection = SimSiamPretextProjectionHead(training=self.training)

    z = encoder(inputs)
    proj = mlp_projection(z)

    additional_outputs = []
    if self.algorithm == 'dino_pretext+supervised_training':
      protos = dino_projection(z)
      additional_outputs.append({'protos': protos})

    if self.algorithm == 'simsiam_pretext+supervised_training':
      siam_proj = simsiam_projection(z)
      siam_pred = simsiam_prediction(siam_proj)
      additional_outputs.append({'siam_proj': siam_proj})
      additional_outputs.append({'siam_pred': siam_pred})

    reconstruction = reconstruction_model(z)
    mask_guess = mask_model(z)

    outputs = {
        'pretext': {
            'encoded': z,
            'proj': proj,
            'reconstruction': reconstruction,
            'mask_guess': mask_guess,
        }
    }
    for output in additional_outputs:
      outputs['pretext'].update(output)

    return outputs


class SimSiamPretextProjectionHead(nn.Module):
  """Projection Head for the pretext model according to SimSiam.

  We follow the implementation where 2048 dimensions is used for the predictor
  size and the projection size.

  Reference this link
  https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

  Attributes:
    training: Boolean for whether to update the centering params.
    output_dim: Dimension of hidden and output layers

  """
  training: bool
  output_dim: int = 2048

  @nn.compact
  def __call__(self, x):
    """Projection ouputs.

    Args:
      x: input features (batch_size, dim)

    Returns:
      Projection.
    """
    x = nn.Dense(self.output_dim, use_bias=False)(x)
    x = nn.BatchNorm(use_running_average=(not self.training))(x)
    x = nn.relu(x)
    x = nn.Dense(self.output_dim, use_bias=False)(x)
    x = nn.BatchNorm(use_running_average=(not self.training))(x)
    x = nn.relu(x)
    x = nn.Dense(self.output_dim, use_bias=False)(x)
    x = nn.BatchNorm(use_scale=False, use_bias=False,
                     use_running_average=(not self.training))(x)

    return x


class SimSiamPretextPredictionHead(nn.Module):
  """Prediction Head for the pretext model according to SimSiam.

  We follow the implementation where 2048 dimensions is used for the predictor
  size and the projection size.

  Reference this link.
  https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

  Attributes:
    training: Boolean for whether to update the centering params.
    hidden_dim: Dimension of the hidden layer in the projection mlp.
  """
  training: bool
  hidden_dim: int = 512

  @nn.compact
  def __call__(self, x):
    """Protos pretext outputs.

    Args:
      x: input features (batch_size, dim)

    Returns:
      Predictor
    """
    output_dim = x.shape[-1]

    x = nn.Dense(self.hidden_dim, use_bias=False)(x)
    x = nn.BatchNorm(use_running_average=(not self.training))(x)
    x = nn.relu(x)

    x = nn.Dense(output_dim)(x)

    return x


class WeightNormDense(nn.Dense):
  """Linear layer with weight normalized kernel."""

  def param(self, name, *args, **kwargs):
    param = super().param(name, *args, **kwargs)
    if name == 'kernel':
      param /= (jnp.linalg.norm(param, axis=0, keepdims=True) + 1e-10)
    return param


class DinoPretextProjectionHead(nn.Module):
  """Projection Head for the pretext model according to DINO.

  Attributes:
    training: Boolean for whether to update the centering params.
    hidden_dim: Dimension of the hidden layer in the projection mlp.
    bottleneck_dim: Dimension of the bottleneck.
    output_dim: Dimension of the output ("number of prototypes").
    normalize_last_layer: Normalize the last layer of prototypes.
  """
  training: bool
  hidden_dim: int = 2048
  bottleneck_dim: int = 256
  output_dim: int = 4096
  normalize_last_layer: bool = True

  @nn.compact
  def __call__(self, x):
    """Protos pretext outputs.

    Args:
      x: input features (batch_size, dim)

    Returns:
      Protos
    """
    x = nn.Dense(self.hidden_dim)(x)
    x = nn.gelu(x)
    x = nn.Dense(self.hidden_dim)(x)
    x = nn.gelu(x)
    x = nn.Dense(self.bottleneck_dim)(x)

    # L2 Normalize.
    x = l2_normalize(x)

    # Last layer.
    if self.normalize_last_layer:
      x = WeightNormDense(self.output_dim, use_bias=False, name='prototypes')(x)
    else:
      x = nn.Dense(self.output_dim, use_bias=False, name='prototypes')(x)

    return x


class ResnetPretext(nn.Module):
  """A model using the resnet arch that outputs 3 values.

    1. an encoded version of the corruptd input (expect to be flat.)
    2. a prediction for the corrupted input
    3. a prediction for the mask

    The corrupted values are inputted according to the paper.
  """

  training: bool
  algorithm: Optional[str] = None

  @nn.compact
  def __call__(self, inputs):
    """Vime pretext outputs.

    Args:
      inputs: input features (batch_size, dim)

    Returns:
      Encoded data, reconstruction of the uncorrupted data, the prediction of
      the mask.
    """

    encoder = models.Resnet(
        training=self.training)
    reconstruction_model = models.MLP(
        layer_sizes=[512, inputs.shape[-1]], training=self.training)
    mask_model = VimeMaskModel(
        training=self.training, input_flat_shape=inputs.shape[-1])
    mlp_projection = models.MLP(layer_sizes=[512, 128], training=self.training)

    if self.algorithm == 'dino_pretext+supervised_training':
      dino_projection = DinoPretextProjectionHead(training=self.training)

    if self.algorithm == 'simsiam_pretext+supervised_training':
      simsiam_prediction = SimSiamPretextPredictionHead(training=self.training)
      simsiam_projection = SimSiamPretextProjectionHead(training=self.training)

    z = encoder(inputs)
    proj = mlp_projection(z)
    reconstruction = reconstruction_model(z)
    mask_guess = mask_model(z)

    additional_outputs = []
    if self.algorithm == 'dino_pretext+supervised_training':
      protos = dino_projection(z)
      additional_outputs.append({'protos': protos})

    if self.algorithm == 'simsiam_pretext+supervised_training':
      siam_proj = simsiam_projection(z)
      siam_pred = simsiam_prediction(siam_proj)
      additional_outputs.append({'siam_proj': siam_proj})
      additional_outputs.append({'siam_pred': siam_pred})

    outputs = {
        'pretext': {
            'encoded': z,
            'proj': proj,
            'reconstruction': reconstruction,
            'mask_guess': mask_guess,
        }
    }
    for output in additional_outputs:
      outputs['pretext'].update(output)

    return outputs
