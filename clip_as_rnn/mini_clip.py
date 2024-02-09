# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""This files has the minimal implementation of a CLIP model."""

import collections
from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
  """Subclass torch's LayerNorm to handle fp16."""

  def forward(self, x):
    original_type = x.dtype
    return_tensor = super().forward(x.type(torch.float32))
    return return_tensor.type(original_type)


class QuickGELU(nn.Module):
  """A quick implementation of GELU activation function."""

  def forward(self, x):
    return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
  """The implementation of a residual attention block.

  Attributes:
    attn: A attention module.
    layer_norm_1: The first layer normalization layer.
    mlp: A MLP layer.
    layer_norm_2: The second layer normalization layer.
    attn_mask: The mask for attention.
  """

  def __init__(
      self,
      model_dim,
      num_heads,
      attention_mask = None,
  ):
    """Initialize ResidualAttentionBlock.

    Args:
      model_dim: An integer indicating the model dimension.
      num_heads: An integer indicating the number of heads.
      attention_mask: A torch.Tensor indicating the mask for attention.
    """
    super().__init__()

    self.attn = nn.MultiheadAttention(model_dim, num_heads)
    self.layer_norm_1 = LayerNorm(model_dim)
    self.mlp = nn.Sequential(
        collections.OrderedDict([
            ("c_fc", nn.Linear(model_dim, model_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(model_dim * 4, model_dim)),
        ])
    )
    self.layer_norm_2 = LayerNorm(model_dim)
    self.attn_mask = attention_mask


class Transformer(nn.Module):
  """The implementation of Transformer.

  Attributes:
    model_dim: number of channels for each attention block.
    num_layers: number of layers for the residual blocks.
    residual_blocks: the trunk blocks of the transformer.
  """

  def __init__(
      self,
      model_dim,
      num_layers,
      num_heads,
      attention_mask = None,
  ):
    """Initialize Transformer.

    Args:
      model_dim: An integer indicating the dimension of the attention.
      num_layers: An integer indicating the number of layers.
      num_heads: An integer indicating the number of heads.
      attention_mask: A torch.Tensor indicating the attention mask.
    """
    super().__init__()
    self.model_dim = model_dim
    self.num_layers = num_layers
    assert num_heads > 0
    self.residual_blocks = nn.Sequential(
        *[
            ResidualAttentionBlock(model_dim, num_heads, attention_mask)
            for _ in range(num_layers)
        ]
    )


class VisionTransformer(nn.Module):
  """The implementation of Vision Transformer.

  Attributes:
    input_resolution: the resolution of the image input.
    output_dim: the output dimension.
    conv1: the first convolution layer that converts images to patches.
    class_embedding: the class embedding for classification.
    positional_embedding: the positional embedding representing the positional
      information.
    pre_activation_layer_norm: the pre-activation layer normalization layer.
    transformer: the trunk transformer for modelling.
    post_activation_layer_norm: the last post-activation layer normalization
      layer.
    proj: the final projection.
    patch_size: the patch size for the network.
  """

  def __init__(
      self,
      input_resolution,
      patch_size,
      dim,
      num_layers,
      num_heads,
      output_dim,
      seed = None
  ):
    """Initialize Vision Transformer.

    Args:
      input_resolution: An integer indicating the default training resolution.
      patch_size: An integer indicating the default patch size.
      dim: An integer indicating the attention output dimension.
      num_layers: An integer indicating the number of layers
      num_heads: An integer indicating the number of heads.
      output_dim: An integer indicating the output dimension after
        projection.
      seed: An integer representing the random seed.
    """
    super().__init__()
    self.input_resolution = input_resolution
    self.output_dim = output_dim
    self.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=dim,
        kernel_size=patch_size,
        stride=patch_size,
        bias=False,
    )

    scale = dim**-0.5
    if seed is not None:
      torch.manual_seed(seed)
    self.class_embedding = nn.Parameter(scale * torch.randn(dim))
    self.positional_embedding = nn.Parameter(
        scale
        * torch.randn((input_resolution // patch_size) ** 2 + 1, dim)
    )
    self.pre_activation_layer_norm = LayerNorm(dim)

    self.transformer = Transformer(dim, num_layers, num_heads)

    self.post_activation_layer_norm = LayerNorm(dim)
    if seed is not None:
      torch.manual_seed(seed)
    self.proj = nn.Parameter(scale * torch.randn(dim, output_dim))
    self.patch_size = patch_size


class CLIP(nn.Module):
  """The implementation of CLIP.

  Attributes:
    context_length: the length of the text encoder.
    visual: the visual encoder.
    transformer: the text encoder.
    vocab_size: the size of the entire vocabulary.
    token_embedding: the text token embedding.
    positional_embedding: the positional embedding for the visual encoder.
    final_layer_norm: the last layer normalization layer.
    text_projection: the last text projection layer.
    logit_scale: the scale of logit.
  """

  def __init__(
      self,
      embed_dim,
      # vision
      image_resolution,
      vision_layers,
      vision_dimension,
      vision_patch_size,
      vision_heads,
      # text
      context_length,
      vocabulary_size,
      transformer_dimension,
      transformer_heads,
      transformer_layers,
  ):
    """A minimal implementation of CLIP without forward functions.

    Args:
      embed_dim: An integer indicating the dimension of the final
        embedding.
      image_resolution: An integer indicating the image resolution.
      vision_layers: An integer number of layers for vision transformer.
      vision_dimension: An integer indicating the dimension of vision
        transformer.
      vision_patch_size: An integer indicating the patch size for
        VisionTransformer.
      vision_heads: An integer indicating the number of heads of
        VisionTransformer.
      context_length: An integer indicating the length of the context.
      vocabulary_size: An integer indicating the size of vocabulary.
      transformer_dimension: An integer indicating the width of text
        transformer.
      transformer_heads: An integer indicating the number of heads for text
        transformers.
      transformer_layers: An integer indicating the number of layers for text
        transformers.
    """
    super().__init__()

    self.context_length = context_length

    self.visual = VisionTransformer(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        dim=vision_dimension,
        num_layers=vision_layers,
        num_heads=vision_heads,
        output_dim=embed_dim,
    )

    self.transformer = Transformer(
        model_dim=transformer_dimension,
        num_layers=transformer_layers,
        num_heads=transformer_heads,
        attention_mask=self.build_attention_mask(),
    )

    self.vocabulary_size = vocabulary_size
    self.token_embedding = nn.Embedding(vocabulary_size, transformer_dimension)
    self.positional_embedding = nn.Parameter(
        torch.empty(self.context_length, transformer_dimension)
    )
    self.final_layer_norm = LayerNorm(transformer_dimension)

    self.text_projection = nn.Parameter(
        torch.empty(transformer_dimension, embed_dim)
    )
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

  def build_attention_mask(self):
    """Build a default empty attention mask.

    Returns:
      A torch.Tensor indicating the mask for attention.
    """
    # Lazily create causal attention mask, with full attention between
    # the vision tokens. PyTorch uses additive attention mask.
    mask = torch.empty(self.context_length, self.context_length)
    # Fill the mask with -inf and these positions will become 0 after softmax.
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask
