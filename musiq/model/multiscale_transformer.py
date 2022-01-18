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

"""Multiscale image quality transformer. https://arxiv.org/abs/2108.05997."""

from flax import nn
import jax.numpy as jnp
import numpy as np

import musiq.model.multiscale_transformer_utils as utils
import musiq.model.resnet as resnet

RESNET_TOKEN_DIM = 64


class Model(nn.Module):
  """Multiscale patch transformer."""

  def apply(self,
            x,
            num_classes=1,
            train=False,
            hidden_size=None,
            transformer=None,
            resnet_emb=None,
            representation_size=None):
    """Apply model on inputs.

    Args:
      x: the processed input patches and position annotations.
      num_classes: the number of output classes. 1 for single model.
      train: train or eval.
      hidden_size: the hidden dimension for patch embedding tokens.
      transformer: the model config for Transformer backbone.
      resnet_emb: the config for patch embedding w/ small resnet.
      representation_size: size of the last FC before prediction.

    Returns:
      Model prediction output.
    """
    assert transformer is not None
    # Either 3: (batch size, seq len, channel) or
    # 4: (batch size, crops, seq len, channel)
    assert len(x.shape) in [3, 4]

    multi_crops_input = False
    if len(x.shape) == 4:
      multi_crops_input = True
      batch_size, num_crops, l, channel = x.shape
      x = jnp.reshape(x, [batch_size * num_crops, l, channel])

    # We concat (x, spatial_positions, scale_posiitons, input_masks)
    # when preprocessing.
    inputs_spatial_positions = x[:, :, -3]
    inputs_spatial_positions = inputs_spatial_positions.astype(jnp.int32)
    inputs_scale_positions = x[:, :, -2]
    inputs_scale_positions = inputs_scale_positions.astype(jnp.int32)
    inputs_masks = x[:, :, -1]
    inputs_masks = inputs_masks.astype(jnp.bool_)
    x = x[:, :, :-3]
    n, l, channel = x.shape
    if hidden_size:
      if resnet_emb:
        # channel = patch_size * patch_size * 3
        patch_size = int(np.sqrt(channel // 3))
        x = jnp.reshape(x, [-1, patch_size, patch_size, 3])
        x = resnet.StdConv(
            x, RESNET_TOKEN_DIM, (7, 7), (2, 2), bias=False, name="conv_root")
        x = nn.GroupNorm(x, name="gn_root")
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        if resnet_emb.num_layers > 0:
          blocks, bottleneck = resnet.get_block_desc(resnet_emb.num_layers)
          if blocks:
            x = resnet.ResNetStage(
                x,
                blocks[0],
                RESNET_TOKEN_DIM,
                first_stride=(1, 1),
                bottleneck=bottleneck,
                name="block1")
            for i, block_size in enumerate(blocks[1:], 1):
              x = resnet.ResNetStage(
                  x,
                  block_size,
                  RESNET_TOKEN_DIM * 2**i,
                  first_stride=(2, 2),
                  bottleneck=bottleneck,
                  name=f"block{i + 1}")
        x = jnp.reshape(x, [n, l, -1])

      x = nn.Dense(x, hidden_size, name="embedding")

    # Here, x is a list of embeddings.
    x = utils.Encoder(
        x,
        inputs_spatial_positions,
        inputs_scale_positions,
        inputs_masks,
        train=train,
        name="Transformer",
        **transformer)

    x = x[:, 0]

    if representation_size:
      x = nn.Dense(x, representation_size, name="pre_logits")
      x = nn.tanh(x)
    else:
      x = resnet.IdentityLayer(x, name="pre_logits")

    x = nn.Dense(x, num_classes, name="head", kernel_init=nn.initializers.zeros)
    if multi_crops_input:
      _, channel = x.shape
      x = jnp.reshape(x, [batch_size, num_crops, channel])
    return x
