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

"""UViT JAX implementation."""

from typing import Optional, Sequence
from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from moe_mtl.modeling.modules import TAPSDense


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == "learn":
    return self.param(name, nn.initializers.normal(stddev=1 / np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")


def window_partition(seq, feat_h, feat_w, window_size=2):
  """Window-partitions a sequence."""
  assert feat_h % window_size == 0
  assert feat_w % window_size == 0

  b, c = seq.shape[0], seq.shape[-1]
  feat = jnp.reshape(seq, [-1, feat_h, feat_w, c])
  h_new, w_new = feat_h // window_size, feat_w // window_size

  # Split feat along axis 0 and 1.
  # TODO(xianzhi): Explore using the gather op for a more efficient and
  # flexible implementation.
  tmp = [
      jnp.split(f, window_size, axis=2)
      for f in jnp.split(feat, window_size, axis=1)
  ]
  window_feats = []
  for t in tmp:
    window_feats += t
  # Concate window splits at the batch dimension.
  window_feats = jnp.concatenate(
      [jnp.reshape(x, [b, h_new * w_new, c]) for x in window_feats], axis=0)
  return window_feats


def window_merge(b_model, seqs, feat_h, feat_w, window_size=2):
  """Merges a list of sequences to 2D features."""
  b, c = seqs.shape[0], seqs.shape[-1]

  # Return if no window partition.
  if b == b_model:
    return seqs

  n_windows = b // b_model
  h_new, w_new = feat_h // window_size, feat_w // window_size
  seqs = jnp.split(seqs, n_windows, axis=0)
  window_feats = [jnp.reshape(seq, [-1, h_new, w_new, c]) for seq in seqs]

  column_feats = []
  for i in range(window_size):
    column_feats.append(
        jnp.concatenate(
            window_feats[i * window_size:(i + 1) * window_size], axis=2))
  merged_feats = jnp.concatenate(column_feats, axis=1)
  return jnp.reshape(merged_feats, [b_model, feat_h * feat_w, c])


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    n, l, d = x.shape  # pylint: disable=unused-variable
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, **inits)(x)
    return x


class TAPSMlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, first, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    n, l, d = x.shape  # pylint: disable=unused-variable
    x, mask_1 = TAPSDense(self.mlp_dim or 4 * d, **inits)(x, first)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x, mask_2 = TAPSDense(d, **inits)(x, first)
    return x, jnp.array([mask_1, mask_2])


def _threshold(x, threshold = 0.1):
  return (x > threshold).astype("float")


def straight_through_threshold(x, threshold = 0.1):
  # Create an exactly-zero expression with Sterbenz lemma that has
  # an exactly-one gradient.
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(_threshold(x, threshold))


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  layer_drop_p: float = 0.0
  use_taps: Optional[bool] = False

  def get_drop_pattern(self, x, deterministic):
    """Randomly drop residual along the first dimension."""
    if not deterministic and self.layer_drop_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng("dropout"), self.layer_drop_p, shape).astype("float32")
    else:
      return 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    y = nn.LayerNorm(name="LayerNorm_0")(x)
    y = out["sa"] = nn.SelfAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        name="MultiHeadDotProductAttention_1",
    )(
        y)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = y * (1.0 - self.get_drop_pattern(y, deterministic)) + x

    y = nn.LayerNorm(name="LayerNorm_2")(x)
    if not self.use_taps:
      y = out["mlp"] = MlpBlock(
          mlp_dim=self.mlp_dim,
          dropout=self.dropout,
          name="MlpBlock_3",
      )(y, deterministic)
    else:
      y, masks = MlpBlock(
          mlp_dim=self.mlp_dim,
          dropout=self.dropout,
          name="MlpBlock_3",
      )(y, deterministic)
      out["mask"] = masks
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = y * (1.0 - self.get_drop_pattern(y, deterministic)) + x
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  stoch_depth: float = 0.0
  use_taps: Optional[bool] = False

  @nn.compact
  def __call__(
      self,
      x,
      window_size=None,
      feat_h=None,
      feat_w=None,
      deterministic=True):
    out = {}

    # Window-partition the input then concat at the batch dimension.
    if window_size:
      x = window_partition(x, feat_h, feat_w, window_size)

    # Input Encoder
    for lyr in range(self.depth):
      block = Encoder1DBlock(
          name=f"encoderblock_{lyr}",
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
          use_taps=self.use_taps,
          layer_drop_p=(lyr / max(self.depth - 1, 1)) * self.stoch_depth)
      x = block(x, deterministic)[0]

    # Merge patches from window splits back to one sequence.
    logging.info(x.shape)
    x = out["pre_ln"] = window_merge(x, feat_h, feat_w, window_size)

    return nn.LayerNorm(name="encoder_norm")(x), out


class Model(nn.Module):
  """ViT backbone with windowed attention."""

  patch_size: Sequence[int] = (8, 8)
  width: int = 384
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 6
  window_size: Optional[int] = None
  patch_overlap: int = 0
  dropout: float = 0.0
  stoch_depth: float = 0.0
  use_taps: Optional[bool] = False
  pool_type: str = "gap"  # Can also be "map" or "tok"

  @nn.compact
  def __call__(self, x, *, train=False):
    out = {}
    assert self.patch_size[0] == self.patch_size[1]

    # Patch extraction
    x = out["stem"] = nn.Conv(
        self.width, [p + self.patch_overlap for p in self.patch_size],
        strides=self.patch_size,
        padding="SAME",
        name="embedding")(x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    if self.pool_type == "tok":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    # Add posemb
    n, l, c = x.shape
    x = out["with_posemb"] = x + get_posemb(self, "learn", l, c,
                                            "pos_embedding", x.dtype)
    x = nn.Dropout(rate=self.dropout)(x, not train)

    if self.window_size and self.pool_type == "tok":
      raise ValueError(
          "Pool type `token` is not supported for window partition.")

    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        stoch_depth=self.stoch_depth,
        use_taps=self.use_taps,
        name="Transformer")(
            x,
            window_size=self.window_size,
            feat_h=h,
            feat_w=w,
            deterministic=not train)

    # Reshape to 2D features at the output level and assert patch size is a
    # power of 2.
    level = np.log2(self.patch_size[0])
    assert np.ceil(level) == np.floor(level)

    x = out[str(round(level))] = jnp.reshape(x, [n, h, w, -1])
    return x, out

