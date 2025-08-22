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

# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLIP text encoder.

We use CLIP text encoder to compute the text embeddings.
CLIP paper: https://arxiv.org/abs/2103.00020

Adapted from Scenic CLIP baseline:
https://github.com/google-research/scenic/tree/main/scenic/projects/baselines/clip
"""

import functools
import hashlib
import os
import tempfile
from typing import Any, Mapping, Optional
import urllib

from absl import logging
import clip

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile
import tensorflow_text as tf_text
import torch
import tqdm


# Match PyTorch default LayerNorm epsilon of 1e-5 (FLAX defaults to 1e-6).
LayerNorm = functools.partial(nn.LayerNorm, epsilon=1e-5)

DEFAULT_DOWNLOAD_DIR = './data/'

CHECKPOINTS_TORCH = {
    'resnet_50': 'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt',
    'resnet_50x4': 'https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt',
    'resnet_50x16': 'https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt',
}

_CLIP_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def hash_file(path):
  return hashlib.sha256(gfile.GFile(path, 'rb').read()).hexdigest()


def download(
    url,
    root = DEFAULT_DOWNLOAD_DIR,
    expected_sha256 = None
):
  """Download a file if it does not exist, with a progress bar.

  Based on https://github.com/openai/CLIP/blob/main/clip/clip.py#L4

  Args:
    url (str): URL of file to download.
    root (str): Directory to place the downloaded file.
    expected_sha256: Optional sha256 sum. If provided, checks downloaded file.
  Raises:
    RuntimeError: Downloaded file existed as a directory, or sha256 of dowload
                  does not match expected_sha256.
  Returns:
    download_target (str): path to downloaded file
  """
  gfile.makedirs(root)
  filename = os.path.basename(url)
  if '?' in filename:
    # strip trailing HTTP GET arguments
    filename = filename[:filename.rindex('?')]

  download_target = os.path.join(root, filename)

  if gfile.exists(download_target):
    if gfile.isdir(download_target):
      raise RuntimeError(f'{download_target} exists and is not a regular file')
    elif expected_sha256:
      if hash_file(download_target) == expected_sha256:
        return download_target
      logging.warning('%s exists, but the SHA256 checksum does not match;'
                      're-downloading the file', download_target)

  temp_file = tempfile.NamedTemporaryFile(delete=False).name
  with gfile.GFile(temp_file, 'wb') as output:
    with urllib.request.urlopen(url) as source:
      loop = tqdm.tqdm(total=int(source.info().get('Content-Length')),
                       ncols=80, unit='iB', unit_scale=True, unit_divisor=1024)
      while True:
        buffer = source.read(8192)
        if not buffer:
          break

        output.write(buffer)
        loop.update(len(buffer))

  if expected_sha256 and hash_file(temp_file) != expected_sha256:
    raise RuntimeError(
        'Model has been downloaded but the SHA256 checksum does not not match')

  # Use copy+remove instead of rename in case source and target are on different
  # file systems:
  gfile.copy(temp_file, download_target, overwrite=True)
  gfile.remove(temp_file)

  return download_target


CONFIGS = {
    'resnet_50': dict(embed_dim=1024,
                      vocab_size=49408,
                      text_features=512,
                      text_num_heads=8,
                      text_num_layers=12),
    'resnet_50x4': dict(embed_dim=640,
                        vocab_size=49408,
                        text_features=640,
                        text_num_heads=10,
                        text_num_layers=12),
    'resnet_50x16': dict(embed_dim=768,
                         vocab_size=49408,
                         text_features=768,
                         text_num_heads=12,
                         text_num_layers=12),
}


def load_model_vars(
    model_name,
    checkpoint_path = None,
    download_dir = DEFAULT_DOWNLOAD_DIR,
):
  """Load model variables from a checkpoint, downloading if necessary."""
  checkpoint_path = os.path.join(download_dir, model_name + '.npy')
  if not gfile.exists(checkpoint_path):
    # Download PyTorch checkpoint
    url = CHECKPOINTS_TORCH.get(model_name)
    logging.info('Downloading checkpoint from %s to %s', url, download_dir)
    checkpoint_path_torch = download(
        url, download_dir, expected_sha256=url.split('/')[-2])

    # Load and convert checkpoint to numpy
    logging.info('Converting checkpoint %s to numpy', checkpoint_path_torch)
    params = torch.jit.load(
        checkpoint_path_torch, map_location='cpu').state_dict()
    params = jax.tree_util.tree_map(lambda p: p.cpu().numpy(), params)

    # Save converted checkpoint
    logging.info('Saving checkpoint %s', checkpoint_path)
    with gfile.GFile(checkpoint_path, 'wb') as f:
      np.save(f, params)
    del params
    gfile.remove(checkpoint_path_torch)

  with gfile.GFile(checkpoint_path, 'rb') as f:
    np_params = np.load(f, allow_pickle=True).tolist()
  return _convert_vars(np_params)


def _convert_attn_layers(params,
                         dim_head = 64):
  """Convert attention parameters."""
  new_params = {}
  processed_attn_layers = []
  for k, v in params.items():
    if 'attn.' in k:
      base = k[:k.rindex('attn.')+5]
      if base in processed_attn_layers:
        continue
      processed_attn_layers.append(base)
      dim = params[base + 'out_proj.bias'].shape[-1]
      heads = dim // dim_head
      new_params[base + 'out.weight'] = params[
          base + 'out_proj.weight'].T.reshape(heads, dim_head, dim)
      new_params[base + 'out.bias'] = params[base + 'out_proj.bias']
      qkv_bias = params[base + 'in_proj_bias'].reshape(3, heads, dim_head)
      qkv_kernel = np.transpose(params[base + 'in_proj_weight'].reshape(
          3, heads, dim_head, dim), (0, 3, 1, 2))
      for i, kk in enumerate(('query', 'key', 'value')):
        new_params[base + f'{kk}.bias'] = qkv_bias[i]
        new_params[base + f'{kk}.weight'] = qkv_kernel[i]
    else:
      new_params[k] = v
  return new_params


def _convert_vars(torch_vars,
                  dim_head = 64):
  """Convert torch parameters to flax parameters."""
  # Expand QKV dense input projection to separate Q, K, V projections
  # and fix shape/transposing of attention layers.
  torch_vars = _convert_attn_layers(torch_vars, dim_head)
  flax_vars = {}
  torch_vars.pop('context_length', None)
  torch_vars.pop('input_resolution', None)
  torch_vars.pop('vocab_size', None)
  for torch_key, v in torch_vars.items():
    if 'num_batches_tracked' in torch_key:
      continue

    if 'conv' in torch_key or 'downsample.0.weight' in torch_key:
      v = v.transpose(2, 3, 1, 0)
    elif 'weight' in torch_key and v.ndim == 2 and 'embedding' not in torch_key:
      # Fully connected layers are transposed, embeddings are not
      v = v.T

    jax_key = torch_key.replace('visual.proj', 'visual.proj.kernel')
    jax_key = jax_key.replace('text_projection', 'text_projection.kernel')
    if 'bn' in jax_key or 'ln' in jax_key or 'downsample.1' in jax_key:
      jax_key = jax_key.replace('.weight', '.scale')
    else:
      jax_key = jax_key.replace('.weight', '.kernel')
    if (jax_key.startswith('transformer') or
        jax_key.startswith('text_projection') or
        jax_key.startswith('ln_final') or
        jax_key.startswith('positional_embedding')):
      jax_key = 'text.' + jax_key

    jax_key = jax_key.replace(
        'token_embedding.kernel', 'text.token_embedding.embedding')

    jax_key = jax_key.replace('attnpool.k_proj', 'attnpool.attn.key')
    jax_key = jax_key.replace('attnpool.q_proj', 'attnpool.attn.query')
    jax_key = jax_key.replace('attnpool.v_proj', 'attnpool.attn.value')
    jax_key = jax_key.replace('attnpool.c_proj', 'attnpool.attn.out')
    if 'attnpool.attn.out' in jax_key:
      if jax_key.endswith('kernel'):
        v = v.reshape(-1, dim_head, v.shape[-1])
    elif 'attnpool.attn' in jax_key:
      if jax_key.endswith('bias'):
        v = v.reshape(-1, dim_head)
      else:
        v = v.reshape(v.shape[0], -1, dim_head)

    if jax_key.endswith('running_mean'):
      jax_key = 'batch_stats.' + jax_key.replace('.running_mean', '.mean')
    elif jax_key.endswith('running_var'):
      jax_key = 'batch_stats.' + jax_key.replace('.running_var', '.var')
    else:
      jax_key = 'params.' + jax_key

    jax_key = jax_key.replace('.', '/')
    jax_key = jax_key.replace('resblocks/', 'resblocks.')
    jax_key = jax_key.replace('resblocks/', 'resblocks.')

    flax_vars[tuple(jax_key.split('/'))] = jnp.asarray(v)

  # Transform the flattened param dict to the original nested structure.
  new_vars = flax.core.freeze(flax.traverse_util.unflatten_dict(flax_vars))
  return new_vars


def quick_gelu(x):
  return x * jax.nn.sigmoid(1.702 * x)


class MLP(nn.Module):
  """Simple MLP for Transformer."""

  @nn.compact
  def __call__(self, x):
    ch = x.shape[-1]
    x = nn.Dense(4 * ch, name='c_fc')(x)
    x = quick_gelu(x)
    x = nn.Dense(ch, name='c_proj')(x)
    return x


class ResidualAttentionBlock(nn.Module):
  """Self-attention block of Transformer.

  Attributes:
    num_heads: Number of heads.
  """
  num_heads: int

  @nn.compact
  def __call__(self, x, attn_mask=None):
    xn = LayerNorm(name='ln_1')(x)
    x = x + nn.SelfAttention(
        self.num_heads, name='attn', deterministic=True)(xn, attn_mask)
    xn = LayerNorm(name='ln_2')(x)
    x = x + MLP(name='mlp')(xn)
    return x


class Transformer(nn.Module):
  """Transformer module.

  Attributes:
    features: Number of features.
    num_layers: Number of layers for each block.
    num_heads: Number of heads.
  """
  features: int
  num_layers: int
  num_heads: int

  @nn.compact
  def __call__(self,
               x,
               attn_mask = None):
    for i in range(self.num_layers):
      x = ResidualAttentionBlock(
          num_heads=self.num_heads, name=f'resblocks.{i}')(x, attn_mask)
    return x


class TextEncoder(nn.Module):
  """Text Transformer.

  Attributes:
    vocab_size: Size of the vocabulary.
    text_features: Number of features.
    text_num_layers: Number of transformer blocks (self-attn + MLP).
    text_num_heads: Number of attention heads.
    embed_dim: Size of the final text embedding.
  """
  vocab_size: int
  text_features: int
  text_num_layers: int
  text_num_heads: int
  embed_dim: int

  @nn.compact
  def __call__(self, text):
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.zeros,
                                      (text.shape[1], self.text_features))
    mask = nn.combine_masks(
        nn.make_attention_mask(text > 0, text > 0), nn.make_causal_mask(text))
    x = nn.Embed(self.vocab_size, self.text_features,
                 name='token_embedding')(text)
    x = x + positional_embedding[None]
    x = Transformer(
        self.text_features, self.text_num_layers,
        self.text_num_heads, name='transformer')(x, attn_mask=mask)
    x = LayerNorm(name='ln_final')(x)
    x = x[jnp.arange(x.shape[0]), text.argmax(-1)]
    x = nn.Dense(self.embed_dim, use_bias=False, name='text_projection')(x)
    x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x


class TextEncoderWrapper(nn.Module):
  """Wrapper over the text transformer.

  Attributes:
    text: Text encoder model
  """
  text: Any

  def __call__(self, text):
    return self.text(text)


def get_clip_text_fn(model_name):
  """Get CLIP text model function."""
  model = TextEncoderWrapper(text=TextEncoder(**CONFIGS[model_name]))
  clip_vars = load_model_vars(model_name)
  model_bound = model.bind(clip_vars)
  model_fn = jax.pmap(model_bound.__call__)
  def clip_fn(cls_name):
    cls_feat = []
    for t in _CLIP_TEMPLATES:
      cls_str = t.format(cls_name)
      tokens = clip.tokenize(cls_str)
      cls_temp_feat = model_fn(tokens.detach().numpy()[np.newaxis, :])
      cls_feat.append(cls_temp_feat[0])  # Remove extra pmap dimension.
    mean_cls_feat = np.mean(cls_feat, axis=0)
    return mean_cls_feat
  return clip_fn


def get_tokenizer(model_path, vocab_size=64000):
  """Load a saved tokenizer."""
  with open(model_path, 'rb') as f:
    model = f.read()
  tokenizer = tf_text.SentencepieceTokenizer(model=model)
  tokenizer.add_bos = True
  tokenizer.add_eos = False
  tokenizer.vocab_size = vocab_size
  return tokenizer


def tokenize_pad_fn(tokenizer, text_model, cls_name, max_text_len=64):
  """Get an input tokenize function."""
  cls_feat = []
  for t in _CLIP_TEMPLATES:
    cls_str = t.format(cls_name)
    cls_str = cls_str.lower()
    ids = tokenizer.tokenize([cls_str])
    ids, _ = tf_text.pad_model_inputs(ids, max_seq_length=max_text_len)
    output = text_model({'text': ids})
    cls_feat.append(output[0])
  mean_cls_feat = np.mean(cls_feat, axis=0, keepdims=True)
  return mean_cls_feat
