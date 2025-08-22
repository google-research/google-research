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

"""Module with gating layers and multitask learning."""
from typing import Any, Callable, Iterable, Mapping, Optional, Type, Union, Sequence
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from vmoe import utils

from moe_mtl.modeling.modules import TAPSDense
from moe_mtl.router import router


Array = jnp.ndarray
PRNGKey = jnp.ndarray
DType = type(jnp.float32)
KwArgs = Mapping[str, Any]
Metrics = Mapping[str, Array]
Shape = Iterable[int]

InitializerFn = Callable[[PRNGKey, Shape, DType], Array]


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


def sparse_moe_spmd_mtl(
    target,
    split_rngs,
    has_aux = False,
    methods=None):
  """Distribute multiple inputs."""
  assert not has_aux

  def wrapper(expert_fn):

    def transformed(scopes, dispatcher_det, inputs_det, dispatcher_cls,
                    inputs_cls):
      # Prepare inputs to be processed by each expert.
      inputs_det = jax.tree.map(dispatcher_det.dispatch, inputs_det)
      inputs_cls = jax.tree.map(dispatcher_cls.dispatch, inputs_cls)
      outputs_det, outputs_cls = flax.core.lift.vmap(
          expert_fn,
          in_axes=0,
          out_axes=0,
          variable_axes={
              "params": 0,
              "intermediates": 0
          },
          split_rngs=split_rngs)(scopes, inputs_det, inputs_cls)
      # Combine outputs.
      if has_aux:
        outputs_det, _ = outputs_det
        outputs_cls, _ = outputs_cls
      outputs_det = jax.tree.map(dispatcher_det.combine, outputs_det)
      outputs_cls = jax.tree.map(dispatcher_cls.combine, outputs_cls)
      return outputs_det, outputs_cls

    return transformed

  return flax.linen.transforms.lift_transform(wrapper, target, methods=methods)


class MTLMlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  out_dim = None
  dropout_rate: float = 0.1
  kernel_init = None
  bias_init = None
  dropout_rate: float = 0.0
  dtype: Optional[DType] = None
  deterministic: bool = False
  use_taps: bool = False

  @nn.nowrap
  def _make_dense1(self):
    return nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name="dense1")

  @nn.nowrap
  def _make_taps_dense1(self):
    return TAPSDense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name="dense1")

  @nn.nowrap
  def _make_dense_dim(self, dim):
    return nn.Dense(
        features=dim,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name="dense2")

  @nn.nowrap
  def _make_taps_dense_dim(self, dim):
    return TAPSDense(
        features=dim,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name="dense2")

  @nn.compact
  def __call__(self, inputs_det, inputs_cls, *, deterministic=None):
    """Applies Transformer MlpBlock module."""
    if deterministic is None:
      deterministic = self.deterministic
    if self.out_dim:
      actual_out_dim = self.out_dim
    else:
      actual_out_dim = inputs_det.shape[-1]

    if not self.use_taps:
      dense1 = self._make_dense1()
      out_det = dense1(inputs_det)
      out_cls = dense1(inputs_cls)
    else:
      dense1 = self._make_taps_dense1()
      out_det, mask_det_1 = dense1(inputs_det, first=True)  # pylint: disable=wrong-keyword-args
      out_cls, mask_cls_1 = dense1(inputs_cls, first=False)  # pylint: disable=wrong-keyword-args
    out_det = nn.gelu(out_det)
    out_cls = nn.gelu(out_cls)
    dropout = nn.Dropout(rate=self.dropout_rate)
    out_det = dropout(out_det, deterministic=deterministic)
    out_cls = dropout(out_cls, deterministic=deterministic)

    if not self.use_taps:
      output = self._make_dense_dim(actual_out_dim)
      out_det = output(out_det)
      out_cls = output(out_cls)
    else:
      output = self._make_taps_dense_dim(actual_out_dim)
      out_det, mask_det_2 = output(out_det, first=True)  # pylint: disable=wrong-keyword-args
      out_cls, mask_cls_2 = output(out_cls, first=False)  # pylint: disable=wrong-keyword-args
    dropout2 = nn.Dropout(rate=self.dropout_rate)
    out_det = dropout2(out_det, deterministic=deterministic)
    out_cls = dropout2(out_cls, deterministic=deterministic)

    # out_det = inputs_det
    if not self.use_taps:
      return out_det, out_cls
    else:
      return out_det, out_cls, jnp.array(
          [mask_det_1, mask_cls_1, mask_det_2, mask_cls_2])


class MTLMlpMoeBlock(nn.Module):
  """Sparse MoE layer of MLPs.

  Attributes:
    mlp_dim: Size of the bottleneck in the MLP.
    num_experts: Number of experts in the MoE.
    group_size: Group size to use. All tokens (from all sequences) are split
      into groups of this size and routed independently.
    dropout_rate: Dropout rate used in the MLP.
    deterministic: If True, runs this layer in deterministic mode. Notice that
      the router can override this, by passing `deterministic=False` to the
      router-specific arguments.
    router: Specific parameters for the router (e.g. num_selected_experts,
      noise_std, ...).
    dtype: DType used in this layer.
    split_rngs: If True, initializes the parameters of each expert with a
      different random seed. Otherwise, it will use the same PRNG for all.
    router_cls: Router class used by the MLP MoE layer.
  """
  mlp_dim: int
  num_experts: int
  group_size_det: int
  group_size_cls: int
  dropout_rate: float = 0.0
  router: Optional[KwArgs] = None
  dtype: Optional[DType] = None
  split_rngs: Union[bool, Iterable[str]] = False
  deterministic: Optional[bool] = True
  use_single_router: bool = False
  use_taps: bool = False
  @nn.nowrap
  def create_router(self, deterministic, name):
    router_kwargs = dict(num_experts=self.num_experts, **(self.router or {}))
    # By default, the router will be deterministic during inference. But we
    # allow to override it.
    router_kwargs["deterministic"] = router_kwargs.get("deterministic",
                                                       deterministic)
    # Create instance of the router class.
    router_cls = router_kwargs.pop("name", "CustomNoisyTopExpertsPerItemRouter")
    router_cls = getattr(router, router_cls)
    if "cls" in name:
      rng_name = "gating_cls"
      top_k = router_kwargs.pop("num_selected_experts_cls", 1)
      router_kwargs.pop("num_selected_experts_det")
    else:
      rng_name = "gating_det"
      top_k = router_kwargs.pop("num_selected_experts_det", 1)
      router_kwargs.pop("num_selected_experts_cls")

    if self.use_single_router:
      name = "Router"  # override name to use the same weight
    return router_cls(
        dtype=self.dtype,
        name=name,
        rng_name=rng_name,
        num_selected_experts=top_k,
        **router_kwargs)

  @nn.nowrap
  def create_split_rngs(self):
    if isinstance(self.split_rngs, bool):
      return {
          "params": self.split_rngs,
          "dropout_cls": self.split_rngs,
          "dropout_det": self.split_rngs
      }
    else:
      split_rngs = set(self.split_rngs)
      return {
          "params": "params" in split_rngs,
          "dropout_cls": "dropout_cls" in split_rngs,
          "dropout_det": "dropout_det" in split_rngs,
      }

  @nn.compact
  def __call__(self, inputs_det, inputs_cls, deterministic=True):
    assert inputs_det.ndim == 3, f"Expected ndim = 3, but got shape {inputs_det.shape}"
    assert inputs_cls.ndim == 3, f"Expected ndim = 3, but got shape {inputs_cls.shape}"
    # Reshape inputs from (num_seqs, seq_length, hidden_size) to
    # (num_groups, groups_size, hidden_size).
    inputs_shape_cls = inputs_cls.shape
    inputs_shape_det = inputs_det.shape
    # logging.info(f"Moe Input Shape: {inputs_shape}")
    inputs_cls = inputs_cls.reshape(-1, self.group_size_cls,
                                    inputs_cls.shape[-1])
    inputs_det = inputs_det.reshape(-1, self.group_size_det,
                                    inputs_det.shape[-1])
    # logging.info(f"Moe Reshape Input Shape: {inputs_shape}")
    if not self.use_single_router:
      dispatcher_cls, metrics_cls = self.create_router(
          deterministic, name="Router_cls")(inputs_cls)
      dispatcher_det, metrics_det = self.create_router(
          deterministic, name="Router_det")(inputs_det)
    else:
      router_single = self.create_router(deterministic, name="Router")
      dispatcher_cls, metrics_cls = router_single(inputs_cls)
      dispatcher_det, metrics_det = router_single(inputs_det)
    # Use the dispatcher to apply a MoE of MlpBlocks.
    mlp_moe_layer = sparse_moe_spmd_mtl(
        MTLMlpBlock, has_aux=False, split_rngs=self.create_split_rngs())(
            deterministic=deterministic,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            name="Mlp")
    outputs_det, outputs_cls = mlp_moe_layer(dispatcher_det, inputs_det,
                                             dispatcher_cls, inputs_cls)
    # logging.info(f"Moe Output Shape: {outputs.shape}")
    # Reshape outputs from (num_groups, group_size, output_dim) to
    # (num_seqs, seqs_length, output_dim).
    outputs_cls = outputs_cls.reshape(*inputs_shape_cls[:-1],
                                      outputs_cls.shape[-1])
    outputs_det = outputs_det.reshape(*inputs_shape_det[:-1],
                                      outputs_det.shape[-1])

    metrics = {}
    for key in metrics_det:
      metrics[key + "_det"] = metrics_det[key]
      metrics[key + "_cls"] = metrics_cls[key]
    return outputs_det, outputs_cls, metrics


class EncoderBlock(nn.Module):
  """Encoder block with a Sparse MoE of MLPs."""
  mlp_block: Type[nn.Module]
  num_heads: int
  dtype: Optional[DType] = None
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  layer_drop_p: float = 0.0
  use_taps: bool = False

  def get_drop_pattern(self, x, deterministic, rng_name):
    """Randomly drop residual along the first dimension."""
    if not deterministic and self.layer_drop_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng(f"dropout_{rng_name}"), self.layer_drop_p,
          shape).astype("float32")
    else:
      return 0.0

  @nn.compact
  def __call__(self, inputs_det, inputs_cls, deterministic=True):
    # Attention Block.
    out = {}
    ln1 = nn.LayerNorm(dtype=self.dtype)
    x_det = ln1(inputs_det)
    x_cls = ln1(inputs_cls)
    sa = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        name="SelfAttention")
    x_cls = out["sa_cls"] = sa(inputs_q=x_cls, inputs_kv=x_cls)
    x_det = out["sa_det"] = sa(inputs_q=x_det, inputs_kv=x_det)
    d1 = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)
    x_cls = d1(x_cls)
    x_det = d1(x_det)
    x_cls = out["+sa_cls"] = x_cls * (
        1.0 - self.get_drop_pattern(x_cls, deterministic, "cls")) + inputs_cls
    x_det = out["+sa_det"] = x_det * (
        1.0 - self.get_drop_pattern(x_det, deterministic, "det")) + inputs_det
    # MLP-MoE block.
    ln2 = nn.LayerNorm(dtype=self.dtype)
    y_cls = ln2(x_cls)
    y_det = ln2(x_det)
    y_out = self.mlp_block(dtype=self.dtype)(
        y_det, y_cls, deterministic=deterministic)
    if len(y_out) == 3:
      y_det, y_cls, metrics = y_out
    else:
      if len(y_out) == 3:
        y_det, y_cls, masks = y_out
        metrics = {"taps_masks": masks}
      else:
        y_det, y_cls = y_out
        metrics = {}
    for key in metrics:
      out[key] = metrics[key]
    out["mlp_cls"] = y_cls
    out["mlp_det"] = y_det
    x_cls = out["+mlp_cls"] = x_cls + y_cls * (
        1.0 - self.get_drop_pattern(y_cls, deterministic, "cls"))
    x_det = out["+mlp_det"] = x_det + y_det * (
        1.0 - self.get_drop_pattern(y_det, deterministic, "det"))
    return x_det, x_cls, out


class EncoderMoe(nn.Module):
  """Transformer encoder with optional blocks of Sparse MoE of MLPs.

  To add MoEs to a given block of the encoder, pass a sequence of integers
  (block IDs) named 'layers' to the `moe` parameters dictionary.

  For example, to replace the MLPs in the last two layers with MoEs, use:
  ```
    encoder = EncoderMoe(
      # ... rest of arguments
      num_layers: 8,
      moe={
        'layers': (6, 7),
        # ... other MoE options
      })
  ```

  Attributes:
    num_layers: Number of encoder blocks.
    mlp_dim: Size of the bottleneck in the MLP.
    num_heads: Number of attention heads.
    dropout_rate: Dropout rate to use after the attention layer and in the MLPs.
    attention_dropout_rate: Dropout rate to use in the attention layers.
    moe: Specific parameters for the blocks with MoE layers (e.g. num_experts,
      group_size, router-specific options, etc).
    deterministic: If True, run the encoder in deterministic mode. Notice that
      the routers in the MoE layers can override this.
    dtype: DType used in this layer.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  moe: Optional[KwArgs] = None
  dtype: Optional[DType] = None
  stoch_depth: float = 0.0
  use_single_router: bool = False
  use_taps: bool = False
  @nn.compact
  def __call__(
      self,
      inputs_det,
      inputs_cls,
      window_size_det=None,
      window_size_cls=None,
      feat_hc=None,
      feat_wc=None,
      feat_hd=None,
      feat_wd=None,
      deterministic=True):
    b_model_cls = inputs_cls.shape[0]
    b_model_det = inputs_det.shape[0]
    if window_size_cls > 1:
      inputs_cls = window_partition(
          inputs_cls, feat_hc, feat_wc, window_size_cls)
    if window_size_det > 1:
      inputs_det = window_partition(
          inputs_det, feat_hd, feat_wd, window_size_det)
    # logging.info(f"Inputs shape: {inputs.shape}")
    dense_mlp_params = dict(
        mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate,
        use_taps=self.use_taps)
    moe_mlp_params = {**dense_mlp_params, **(self.moe or {})}
    moe_mlp_layers = moe_mlp_params.pop("layers", ())
    moe_mlp_params["use_single_router"] = self.use_single_router
    dense_mlp_cls = utils.partialclass(
        MTLMlpBlock, **dense_mlp_params, name="Mlp")
    moe_mlp_cls = utils.partialclass(
        MTLMlpMoeBlock, **moe_mlp_params, name="Moe")
    block_cls = utils.partialclass(
        EncoderBlock,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype)

    out = {}

    aux_loss_det = 0
    aux_loss_cls = 0
    logits_det = {}
    logits_cls = {}
    for block_idx in range(self.num_layers):
      block_name = f"encoderblock_{block_idx}"
      if block_idx in moe_mlp_layers:
        inputs_det, inputs_cls, out[block_name] = block_cls(
            name=block_name,
            mlp_block=moe_mlp_cls,
            layer_drop_p=(block_idx / max(self.num_layers - 1, 1)) *
            self.stoch_depth)(inputs_det, inputs_cls, deterministic)
        # logging.info(out[f"encoderblock_{block}"]["auxiliary_loss"])
        aux_loss_cls += out[block_name]["auxiliary_loss_cls"]
        aux_loss_det += out[block_name]["auxiliary_loss_det"]
        logits_det[block_name] = out[block_name]["logits_det"]
        logits_cls[block_name] = out[block_name]["logits_cls"]

      else:
        inputs_det, inputs_cls, out[block_name] = block_cls(
            name=block_name,
            mlp_block=dense_mlp_cls,
            layer_drop_p=(block_idx / max(self.num_layers - 1, 1)) *
            self.stoch_depth)(inputs_det, inputs_cls, deterministic)

        if self.use_taps:
          aux_loss_det += jnp.sum(jnp.abs(
              out[block_name]["taps_masks"]) * 0.1) / self.num_layers / 4

    x_cls = out["pre_ln_cls"] = window_merge(b_model_cls, inputs_cls, feat_hc,
                                             feat_wc, window_size_cls)
    x_det = out["pre_ln_det"] = window_merge(b_model_det, inputs_det, feat_hd,
                                             feat_wd, window_size_det)

    out["auxiliary_loss_det"] = aux_loss_det
    out["auxiliary_loss_cls"] = aux_loss_cls
    out["logits_det"] = logits_det
    out["logits_cls"] = logits_cls
    ln = nn.LayerNorm(name="encoder_norm")
    encoded_cls = ln(x_cls)
    encoded_det = ln(x_det)
    return encoded_det, encoded_cls, out


class VisionTransformerMoeMTL(nn.Module):
  """Vision Transformer with Sparse MoE layers and Multitask .

  This is the model used in the paper https://arxiv.org/abs/2106.05974.
  """

  encoder: KwArgs
  width: int = 384
  depth: int = 12
  patch_size_det: Sequence[int] = (8, 8)
  patch_size_cls: Sequence[int] = (8, 8)
  encoder_cls: Type[nn.Module] = EncoderMoe
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 6
  window_size_det: Optional[int] = None
  window_size_cls: Optional[int] = None
  patch_overlap: int = 0
  dropout: float = 0.0
  stoch_depth: float = 0.0
  pool_type_cls: str = "gap"
  pool_type_det: str = "gap"
  use_taps: bool = False
  use_shared_embed_conv: bool = False
  use_single_router: bool = False

  @nn.compact
  def __call__(self, inputs_det, inputs_cls, train=False, second_stage=False):
    out = {}
    if self.use_shared_embed_conv:
      conv = nn.Conv(
          self.width, [p + self.patch_overlap for p in self.patch_size_det],
          strides=self.patch_size_det,
          padding="SAME",
          name="embedding")
      x_cls = out["stem_cls"] = conv(inputs_cls)
      x_det = out["stem_cls"] = conv(inputs_det)
    else:
      conv_det = nn.Conv(
          self.width, [p + self.patch_overlap for p in self.patch_size_det],
          strides=self.patch_size_det,
          padding="SAME",
          name="embedding_det")
      conv_cls = nn.Conv(
          self.width, [p + self.patch_overlap for p in self.patch_size_cls],
          strides=self.patch_size_cls,
          padding="SAME",
          name="embedding_cls")
      x_cls = out["stem_cls"] = conv_cls(inputs_cls)
      x_det = out["stem_cls"] = conv_det(inputs_det)
    nc, hc, wc, cc = x_cls.shape
    x_cls = jnp.reshape(x_cls, [nc, hc * wc, cc])
    nd, hd, wd, cd = x_det.shape
    x_det = jnp.reshape(x_det, [nd, hd * wd, cd])
    if self.pool_type_cls == "tok":
      cls_cls = self.param("cls_cls", nn.initializers.zeros, (1, 1, cc),
                           x_cls.dtype)
      x_cls = jnp.concatenate([jnp.tile(cls_cls, [nc, 1, 1]), x_cls], axis=1)
    if self.pool_type_det == "tok":
      cls_det = self.param("cls_det", nn.initializers.zeros, (1, 1, cd),
                           x_det.dtype)
      x_det = jnp.concatenate([jnp.tile(cls_det, [nd, 1, 1]), x_det], axis=1)
    # Encode tokens unsing the MoE encoder.

    nc, lc, cc = x_cls.shape
    nd, ld, cd = x_det.shape
    x_cls = out["with_posemb_cls"] = x_cls + get_posemb(
        self, "learn", lc, cc, "pos_embedding_cls", x_cls.dtype)
    x_det = out["with_posemb_det"] = x_det + get_posemb(
        self, "learn", ld, cd, "pos_embedding_det", x_det.dtype)
    dropout = nn.Dropout(rate=self.dropout)
    x_cls = dropout(x_cls, not train)
    x_det = dropout(x_det, not train)
    if self.window_size_det > 1 and self.pool_type_det == "tok":
      raise ValueError(
          "Pool type `token` is not supported for window partition.")
    if self.window_size_cls > 1 and self.pool_type_cls == "tok":
      raise ValueError(
          "Pool type `token` is not supported for window partition.")
    # logging.info(f"Before Encoder CLS: {x.shape}")
    x_det, x_cls, metrics = self.encoder_cls(
        name="Encoder",
        num_layers=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout_rate=self.dropout,
        use_single_router=self.use_single_router,
        stoch_depth=self.stoch_depth,
        use_taps=self.use_taps,
        **self.encoder)(
            x_det,
            x_cls,
            window_size_det=self.window_size_det,
            window_size_cls=self.window_size_cls,
            feat_hc=hc,
            feat_wc=wc,
            feat_hd=hd,
            feat_wd=wd,
            deterministic=not train)

    level = np.log2(self.patch_size_det[0])
    assert np.ceil(level) == np.floor(level)

    if self.pool_type_det == "gap":
      x_det = out[str(round(level)) + "_det"] = jnp.reshape(
          x_det, [nd, hd, wd, -1])
    elif self.pool_type_det != "tok":
      raise NotImplementedError

    if self.pool_type_cls == "gap":
      x_cls = out[str(round(level)) + "_cls"] = jnp.reshape(
          x_cls, [nc, hc, wc, -1])
    elif self.pool_type_cls != "tok":
      raise NotImplementedError

    out["metrics"] = metrics
    return x_det, x_cls, out
