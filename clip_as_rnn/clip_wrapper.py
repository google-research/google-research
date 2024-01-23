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

"""A wrapper for CLIP model to make it CAM-friendly."""


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _upsample_position_embedding(embed, new_size):
  """Upsample the pretrained embedding to a higher resolution.

  Args:
      embed (torch.Tensor): the pretrained embedding.
      new_size (Tuple[int, int]): the new size of the embedding.

  Returns:
      torch.Tensor: the upsampled embedding.
  """
  # emb size NxD
  first = embed[:1, :]
  embed = embed[1:, :]
  n, dim = embed.size(0), embed.size(1)
  size = int(np.sqrt(n))
  if size * size != n:
    raise ValueError(
        f'The size of embed {n} is not a perfect square number.')
  embed = embed.permute(1, 0)
  embed = embed.view(1, dim, size, size).contiguous()
  embed = F.upsample(
      embed,
      size=new_size,
      mode='bilinear',
  )
  embed = embed.view(dim, -1).contiguous()
  embed = embed.permute(1, 0)
  embed = torch.cat([first, embed], 0)
  embed = nn.parameter.Parameter(embed.half())
  return embed


class _CustomBlock(nn.Module):
  """A customized Block to support CAM calculation."""

  def __init__(self, block):
    """Initialize the wrapper.

    Args:
        block: A nn.Module representing the block to be wrapped.
    """
    super().__init__()
    for k, v in vars(block).items():
      setattr(self, k, v)
    if hasattr(block, 'attn'):
      self.attention_layer = self.attn

  def attention(self, x):
    self.attn_mask = (
        self.attn_mask.to(dtype=x.dtype, device=x.device)
        if self.attn_mask is not None
        else None
    )

    self.attention_layer = self.attention_layer.to(
        dtype=x.dtype, device=x.device
    )
    # Set `need_weights` to True so attention weights are returned.
    return self.attention_layer(
        x, x, x, need_weights=True, attn_mask=self.attn_mask
    )

  def forward(self, x):
    """The forward pass for an attention block.

    Args:
      x : A torch.Tensor representing the input tensor with shape
        [num_tokens, heads, dim].

    Returns:
      x : A torch.Tensor representing the output tensor with shape
        [num_tokens, heads, dim].
      attention_weight (torch.Tensor): the output tensor with shape
        [heads, num_tokens, num_tokens].
    """
    attention_output, attention_weight = self.attention(self.layer_norm_1(x))
    x = x + attention_output
    x = x + self.mlp(self.layer_norm_2(x))
    return x, attention_weight


class _CustomTransformer(nn.Module):
  """A customized Transformer to support CAM calculation."""

  def __init__(self, transformer):
    """Initialize the wrapper.

    Args:
        transformer: A nn.Module representing the Transformer to be wrapped.
    """
    super().__init__()
    for k, v in vars(transformer).items():
      setattr(self, k, v)

    if hasattr(transformer, 'resblocks'):
      self.residual_blocks = transformer.resblocks
    if hasattr(transformer, 'layers'):
      self.num_layers = transformer.layers
    self.residual_blocks = nn.Sequential(
        *[_CustomBlock(block) for block in self.residual_blocks]
    )

  def forward(self, x, do_full_forward = True):
    """The forward pass for the transformer.

    Args:
      x: The input tensor of the transformer.
      do_full_forward: A boolean flag determining if to do a full forward of
        the transformer or just extract the feature of the second last layer.

    Returns:
      x: A torch.Tensor representing the output tensor with shape
        [num_tokens, heads, dim].
      attention_weight (torch.Tensor): the output tensor with shape
        [heads, num_tokens, num_tokens].
    """
    attn_weights = []
    with torch.no_grad():
      layers = self.num_layers if do_full_forward else self.num_layers - 1
      for i in range(layers):
        x, attn_weight = self.residual_blocks[i](x)
        attn_weights.append(attn_weight)
    return x, attn_weights


class _CustomVisionTransformer(nn.Module):
  """A customized VisionTransformer to support CAM calculation."""

  def __init__(self, model):
    """Initialize the wrapper.

    Args:
        model: A nn.Module representing the VisionTransformer to be wrapped.
    """
    super().__init__()
    for k, v in vars(model).items():
      setattr(self, k, v)
    self.patch_size = self.conv1.kernel_size[0]
    self.transformer = _CustomTransformer(self.transformer)
    if hasattr(model, 'ln_post'):
      self.post_activation_layer_norm = model.ln_post
    if hasattr(model, 'ln_pre'):
      self.pre_activation_layer_norm = model.ln_pre
    if hasattr(model, 'proj'):
      self.projection = model.proj

  def forward(self, x, h, w):
    """The forward pass for VisionTransformer.

    Args:
      x: A torch.Tensor representing the input tensor.
      h: An integer representing the input height.
      w: An integer representing the input width.

    Returns:
      x: A torch.Tensor representing the output tensor with shape
        [num_tokens, num_heads, dim].
      attention_weight: A torch.Tensor representing the output tensor with shape
        [num_heads, num_tokens, num_tokens].
    """
    self.positional_embedding_new = _upsample_position_embedding(
        self.positional_embedding, (h // self.patch_size, w // self.patch_size)
    )
    # x shape: [*, dim, h // self.patch_size, w // self.patch_size]
    x = self.conv1(x)
    # x shape: [*, dim, num_tokens-1]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    # x shape: [*, num_tokens-1, dim]
    x = x.permute(0, 2, 1)
    zeros = torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    # shape: [*, num_tokens, dim]
    x = torch.cat([self.class_embedding.to(x.dtype) + zeros, x], dim=1)
    x = x + self.positional_embedding_new.to(x.dtype)
    x = self.pre_activation_layer_norm(x)
    x = x.permute(1, 0, 2)
    x, attn_weight = self.transformer(x, do_full_forward=False)
    return x, attn_weight


class CLIPWrapper(nn.Module):
  """A wrapper for CLIP to support forward with a list of text inputs."""

  def __init__(self, clip_model):
    """Initialize the wrapper.

    Args:
        clip_model: A nn.Module representing the CLIP model to be wrapped.
    """
    super().__init__()
    # Copy all attributes from clip_model to self.
    for k, v in vars(clip_model).items():
      setattr(self, k, v)
    self.visual = _CustomVisionTransformer(self.visual)
    self.transformer = _CustomTransformer(self.transformer)
    if hasattr(clip_model, 'ln_pre'):
      self.pre_activation_layer_norm = clip_model.ln_pre
    if hasattr(clip_model, 'ln_final'):
      self.final_layer_norm = clip_model.ln_final

  @property
  def dtype(self):
    return self.visual.conv1.weight.dtype

  def encode_image(self, image, h, w):
    """Encode the image with the CLIP image encoder.

    Args:
      image: A torch.Tensor representing the image input.
      h: image height.
      w: image width.

    Returns:
      x: a torch.Tensor representing the image feature representation.
    """
    return self.visual(image.type(self.dtype), h, w)

  def encode_text(self, text):
    # x shape: [batch_size, context_length, d_model]
    x = self.token_embedding(text).type(self.dtype)

    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)
    x, _ = self.transformer(x)
    x = x.permute(1, 0, 2)
    x = self.final_layer_norm(x).type(self.dtype)

    # x shape: [batch_size, context_length, transformer.width]
    # Take features from the eot embedding.
    # (the last element is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    return x

  def pool_visual(
      self, x, use_cls_token = False
  ):
    """The pooling function for the visual tokens of CLIP.

    Args:
      x: A torch.Tensor representing the feature to be pooled.
      use_cls_token: A boolean flag to determine if we need to use the class
        token of Transformer.

    Returns:
      A torch.Tensor after pooling.

    """
    if use_cls_token:
      return x[:, 0]
    else:
      return torch.mean(x[:, 1:, :], dim=1)

  def forward_last_layer(
      self, image_features, text_features, use_cls_token=False, repeat_last=True
  ):
    """Forward the last layer of CLIP.

    Args:
        image_features: A torch.Tensor representing the image features.
        text_features: A torch.Tensor representing the text features.
        use_cls_token: A boolean representing whether to use the CLS token.
          Default to False.
        repeat_last: A boolean representing whether to repeat the last layer.
          Default to True.

    Returns:
        torch.Tensor: the logits.
        torch.Tensor: the attention weights.
    """
    if repeat_last:
      x, attention_weight = self.visual.transformer.residual_blocks[
          self.visual.transformer.num_layers - 1
      ](image_features)
    else:
      x = image_features
      attention_weight = None
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = self.visual.post_activation_layer_norm(x)
    x = self.pool_visual(x, use_cls_token=use_cls_token)

    if self.visual.projection is not None:
      x = x @ self.visual.projection

    image_features = x

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # cosine similarity as logits
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()

    # shape: [global_batch_size, global_batch_size]
    logits_per_image = F.softmax(logits_per_image.float(), dim=-1)

    return logits_per_image, attention_weight

  def forward(self, image, text, h=224, w=224):
    """The forward function of CLIP to get classification scores for each image.

    Args:
      image: A torch.Tensor representing the image input.
      text: A torch.Tensor representing the tokenized text embeddings.
      h: An integer representing the height of the image input.
      w: An integer representing the width of the image input.

    Returns:
      logits_per_image: A torch.Tensor representing the output logits for each
        image after softmax.
    """
    with torch.no_grad():
      text_features = self.encode_text(text)
      feature_map, _ = self.visual(image.type(self.dtype), h, w)

      logits_per_image, _ = self.forward_last_layer(
          feature_map, text_features, use_cls_token=True, repeat_last=False
      )
    return logits_per_image
