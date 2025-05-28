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

"""Topological Vision Transformer (TopViT).

Vision Transformer with topological attention based on 2-level block Toeplitz
masking mechanism introduced in https://arxiv.org/abs/2107.07999. The regular
attention is modulated by the learnable function of the L1-distance between the
patches in the original 2d-grid rather than its flattened 1d-representation.

TopViT is built on the regular ViT Transformer from the scenic Library:
https://github.com/google-research/scenic. It changes its attention mechanism.
They key addition is the aforementioned attention modulation. Furthermore, in
the current TopViT implementation we replace softmax-kernel in attention with
the ReLU-kernel. That choice makes the attention mechanism numerically
equivalent (not just approximately) to its implicit (sub-quadratic) attention
variation obtained via
(a) the Performer method (https://arxiv.org/abs/2009.14794) and
(b) implicit masking methods from https://arxiv.org/abs/2107.07999.
The currently implemented variant is the explicit one.
"""

from typing import Any, Callable, Optional, Sequence
from absl import logging

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
import scipy

from topological_transformer.images import attention as ta

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: Positional embedding initializer.

  Returns:
    Output in shape `[bs, timesteps, in_dim]`.
  """
  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  @nn.compact
  def __call__(self, inputs):
    # Inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape,
                    inputs.dtype)
    return inputs + pe


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.
    nb_x_patches: number of patches in a fixed column,
    nb_y_patches: number of patches in a fixed row,

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  nb_x_patches: int = 0
  nb_y_patches: int = 0

  @nn.compact
  def __call__(self, inputs, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = ta.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        nb_x_patches=self.nb_x_patches,
        nb_y_patches=self.nb_y_patches)(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    inputs_positions: Input subsequence positions for packed examples.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows timm
      library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Dtype of activations.
    nb_x_patches: number of patches in a fixed column,
    nb_y_patches: number of patches in a fixed row,
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32
  nb_x_patches: int = 0
  nb_y_patches: int = 0

  @nn.compact
  def __call__(self, inputs, *, train = False):
    """Applies Transformer model on the inputs."""

    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    x = AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder.
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
          self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype,
          nb_x_patches=self.nb_x_patches,
          nb_y_patches=self.nb_y_patches)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class TopologicalViT(nn.Module):
  """Topological Vision Transformer model.

    Attributes:
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    dtype: JAX data type for activations.
  """

  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  classifier: str = 'gap'
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, *, train, debug = False):

    fh, fw = self.patches.size
    # Extracting patches and then embedding is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(
            x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        nb_x_patches=h,
        nb_y_patches=w,
        name='Transformer')(
            x, train=train)

    if self.classifier in ('token', '0'):
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=1)

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)
    return x


def _merge_params(params, restored_params, model_cfg, restored_model_cfg):
  """Merges `restored_params` into `params`."""
  # Start moving parameters, one-by-one and apply changes if needed.
  for m_key, m_params in restored_params.items():
    if m_key == 'output_projection':
      # For the classifier head, we use a the randomly initialized params and
      #   ignore the one from pretrained model.
      pass

    elif m_key == 'pre_logits':
      if model_cfg.model.representation_size is None:
        # We don't have representation_size in the new model, so let's ignore
        #   it from the pretained model, in case it has it.
        # Note, removing the key from the dictionary is necessary to prevent
        #   obscure errors from the Flax optimizer.
        params.pop(m_key, None)
      else:
        assert restored_model_cfg.model.representation_size
        params[m_key] = m_params

    elif m_key == 'Transformer':
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input':  # Might need resolution change.
          posemb = params[m_key]['posembed_input']['pos_embedding']
          restored_posemb = m_params['posembed_input']['pos_embedding']

          if restored_posemb.shape != posemb.shape:
            # Rescale the grid of pos, embeddings: param shape is (1, N, d).
            logging.info('Resized variant: %s to %s', restored_posemb.shape,
                         posemb.shape)
            ntok = posemb.shape[1]
            if restored_model_cfg.model.classifier == 'token':
              # The first token is the CLS token.
              restored_posemb_grid = restored_posemb[0, 1:]
              if model_cfg.model.classifier == 'token':
                # CLS token in restored model and in target.
                cls_tok = restored_posemb[:, :1]
                ntok -= 1
              else:
                # CLS token in restored model, but not target.
                cls_tok = restored_posemb[:, :0]
            else:
              restored_posemb_grid = restored_posemb[0]
              if model_cfg.model.classifier == 'token':
                # CLS token in target, but not restored model.
                cls_tok = posemb[:, :1]
                ntok -= 1
              else:
                # CLS token not in target or restored model.
                cls_tok = restored_posemb[:, :0]

            restored_gs = int(np.sqrt(len(restored_posemb_grid)))
            gs = int(np.sqrt(ntok))
            if restored_gs != gs:  # We need resolution change.
              logging.info('Grid-size from %s to %s.', restored_gs, gs)
              restored_posemb_grid = restored_posemb_grid.reshape(
                  restored_gs, restored_gs, -1)
              zoom = (gs / restored_gs, gs / restored_gs, 1)
              restored_posemb_grid = scipy.ndimage.zoom(
                  restored_posemb_grid, zoom, order=1)
            # Attach the CLS token again.
            restored_posemb_grid = restored_posemb_grid.reshape(1, gs * gs, -1)
            restored_posemb = jnp.array(
                np.concatenate([cls_tok, restored_posemb_grid], axis=1))

          params[m_key][tm_key]['pos_embedding'] = restored_posemb
        # Other parameters of the Transformer encoder if they are in the target.
        elif tm_key in params[m_key]:
          params[m_key][tm_key] = tm_params
        else:
          logging.info(
              'Ignoring %s. In restored model\'s Transformer,'
              'but not in target', m_key)

    elif m_key in params:
      # Use the rest if they are in the pretrained model.
      params[m_key] = m_params

    else:
      logging.info('Ignoring %s. In restored model, but not in target', m_key)


def init_topvit_from_train_state(
    train_state, restored_train_state,
    model_cfg,
    restored_model_cfg):
  """Updates the train_state with data from restored_train_state.

  This function is written to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    model_cfg: Configuration of the model. Usually used for some asserts.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.

  Returns:
    Updated train_state.
  """
  params = flax.core.unfreeze(train_state.optimizer.target)
  restored_params = flax.core.unfreeze(restored_train_state.optimizer.target)

  _merge_params(params, restored_params, model_cfg, restored_model_cfg)

  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))

