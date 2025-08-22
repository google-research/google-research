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

"""Final layer heads used by various implementations."""

import dataclasses
import functools
from typing import Sequence

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp

from imp.max.core import constants
from imp.max.core import utils
from imp.max.modeling import attention
from imp.max.modeling import linear
from imp.max.modeling import normalization
from imp.max.modeling import transformers
from imp.max.utils import sharding
from imp.max.utils import typing

EPSILON = 1e-6
AggregationType = constants.AggregationType
_NON_PARAMETRIC_AGGREGATION_TYPES = frozenset({
    AggregationType.SPECIAL_TOKEN,
    AggregationType.GLOBAL_AVERAGE_POOL,
    AggregationType.GLOBAL_MAX_POOL,
    AggregationType.GLOBAL_SUM_POOL,
})


class MAPHead(nn.Module):
  """Multihead Attention Pooling used by BigVision-ViT.

  Attributes:
    num_heads: int. number of self-attention heads to use
    mlp_dim: int. optional size of the intermediate layer; defaults to 4x the
      input dim
    buggy: bool. whether to use the `buggy MAPHead` mode; needed for certain
      checkpoints
    use_bias: bool. whether to add a learned bias to the MLP layer
    kernel_init: initializer for the attention kernel
    bias_init: initializer for the attention bias
    dtype: data type to use
    dot_general: the function that performs dot product.
  """

  num_heads: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  use_bias: bool = True
  dropout_rate: float = 0.1
  kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
  bias_init: nn.initializers.Initializer = nn.initializers.zeros
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32
  mha_qkv_kernel_shardings: typing.ShardingAxes = ()
  mha_out_kernel_shardings: typing.ShardingAxes = ()
  mha_activation_shardings: typing.ShardingAxes = ()
  mha_layernorm_shardings: typing.ShardingAxes = ()
  ffn_inner_kernel_shardings: typing.ShardingAxes = ()
  ffn_outer_kernel_shardings: typing.ShardingAxes = ()
  ffn_intermediate_shardings: typing.ShardingAxes = ()
  probe_kernel_shardings: typing.ShardingAxes = ()
  probe_activation_shardings: typing.ShardingAxes = ()
  dot_general: typing.DotGeneral = lax.dot_general
  precision: typing.Precision = None
  lora_rank: int = 4
  lora_scale: float = 0.

  @nn.compact
  def __call__(self,
               x,
               deterministic = True,
               token_mask = None):
    """Constructs a learnable probe and prepends to the sequence for aggregation."""
    b, n, t, d = x.shape  # pylint: disable=unused-variable

    probe_init = sharding.modulate_param_init(
        self.kernel_init, self.probe_kernel_shardings)
    probe = self.param(
        name='probe',
        init_fn=probe_init,
        shape=(1, d),
        dtype=self.param_dtype,
        unbox=True,
    )
    probe = jnp.asarray(probe, self.dtype)
    probe = jnp.tile(probe[jnp.newaxis, jnp.newaxis, :, :], [b, n, 1, 1])
    probe = sharding.shard_array(probe, self.probe_activation_shardings)

    d_head = d // self.num_heads
    mlp_dim = self.mlp_dim or d * 4

    if token_mask is not None:
      probe_mask = jnp.ones((b, n, 1), dtype=token_mask.dtype)
      attention_mask = utils.create_attention_mask(
          query_token_mask=probe_mask,
          key_token_mask=token_mask,
          dtype=token_mask.dtype)
    else:
      attention_mask = None

    x = attention.MultiHeadAttention(
        d_head=d_head,
        num_heads=self.num_heads,
        d_model=d,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        qkv_kernel_shardings=self.mha_qkv_kernel_shardings,
        out_kernel_shardings=self.mha_out_kernel_shardings,
        activation_shardings=self.mha_activation_shardings,
        layernorm_shardings=self.mha_layernorm_shardings,
        qkv_dot_general=self.dot_general,
        out_dot_general=self.dot_general,
        einsum_dot_general=self.dot_general,
        precision=self.precision,
        name='cross_attention',
    )(query=probe, key=x, value=x,
      deterministic=deterministic,
      attention_mask=attention_mask)

    y = normalization.LayerNorm(
        dtype=self.dtype,
        name='layer_norm',
    )(x)

    x = x + transformers.FeedForward(
        d_ff=mlp_dim,
        d_model=d,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        approximate_gelu=True,
        dot_general=self.dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        inner_kernel_shardings=self.ffn_inner_kernel_shardings,
        outer_kernel_shardings=self.ffn_outer_kernel_shardings,
        intermediate_shardings=self.ffn_intermediate_shardings,
        name='feed_forward',
    )(y, deterministic=deterministic)

    return utils.take_along_axis(x, 0, axis=2, precision='bfloat16')


@dataclasses.dataclass
class NonParametricAggregatorHead:
  """Feature aggregation head for transformer encoders w/o learnable params.

  Attributes:
    aggregation_type: the aggregation type to use. See
      `AggregationType` for supported aggregation types.
  """

  aggregation_type: str | None = None

  def __call__(
      self,
      inputs,
      token_mask = None,
  ):
    """Applies the aggregation to the inputs.

    Args:
      inputs: The input features to aggregate. The expected shape is
        (B, N, T, D) - if there is no special token, OR (B, N, T + 1, D) - if
        there is a special token in the sequence.
      token_mask: The corresponding optional token mask. The expected shape is
        (B, N, T) - if there is no special token, OR (B, N, T + 1) - if
        there is a special token in the sequence.

    Returns:
      A dict with aggregated features and (possibly reshaped) original features.
    """

    if token_mask is not None:
      token_mask = jnp.expand_dims(token_mask, axis=-1)
      token_mask = jnp.asarray(token_mask, dtype=inputs.dtype)
      inputs = inputs * token_mask

    features = inputs
    if self.aggregation_type == AggregationType.SPECIAL_TOKEN:
      # fetch features [..., 0, :] and [..., 1:, :] efficiently
      features_agg = utils.take_along_axis(
          features, 0, axis=-2, precision='bfloat16')
      indices = 1 + jax.lax.iota(jnp.int32, features.shape[-2] - 1)
      features = utils.take_along_axis(
          features, indices, axis=-2, precision='bfloat16')

    elif self.aggregation_type == AggregationType.GLOBAL_AVERAGE_POOL:
      if token_mask is not None:
        epsilon = jnp.asarray(EPSILON, dtype=inputs.dtype)
        features_agg = features.sum(axis=-2) / (
            token_mask.sum(axis=-2) + epsilon)
      else:
        features_agg = jnp.mean(features, axis=-2)

    elif self.aggregation_type == AggregationType.GLOBAL_MAX_POOL:
      if token_mask is not None:
        raise NotImplementedError(
            f'{AggregationType.GLOBAL_MAX_POOL} does not support token_mask.')
      features_agg = jnp.max(features, axis=-2)

    elif self.aggregation_type == AggregationType.GLOBAL_SUM_POOL:
      features_agg = jnp.sum(features, axis=-2)

    else:
      raise NotImplementedError('The requested pooling mechanism not '
                                f'supported: {self.aggregation_type}')

    return {
        constants.DataFeatureName.FEATURES: features,  # (B, N, T, D)
        constants.DataFeatureName.FEATURES_AGG: features_agg,  # (B, N, D)
    }


class VitPostEncoderHead(nn.Module):
  """Projects the features to a final space, classifies and fetches all."""

  aggregation_type: str
  d_post_proj: int | None
  post_proj_position: str | None
  num_classes: int | None
  head_bias_init: float
  dtype: jax.typing.DTypeLike
  # For aggregation_type == AggregationType.MULTI_HEAD_ATTENTION_POOL
  num_heads: int | None = None
  d_ff: int | None = None
  dropout_rate: float | None = None
  use_bias: bool | None = None
  post_proj_dot_general: typing.DotGeneral = lax.dot_general
  logits_dot_general: typing.DotGeneral = lax.dot_general
  pre_logits_kernel_shardings: typing.ShardingAxes = ()
  logits_kernel_shardings: typing.ShardingAxes = ()
  precision: typing.Precision = None
  lora_rank: int = 4
  lora_scale: float = 0.

  def _verify_post_projection(self):
    supported_positions = ('pre_aggregation', 'post_aggregation')
    if (self.d_post_proj is not None
        and self.post_proj_position not in supported_positions):
      raise ValueError('Please provide a valid post projection position. '
                       f'Expected {supported_positions}, instead received '
                       f'{self.post_proj_position!r}.')

  @nn.compact
  def __call__(self,
               inputs,
               patched_shape,
               deterministic):

    outputs = {}
    self._verify_post_projection()

    if (self.post_proj_position == 'pre_aggregation'
        and self.d_post_proj is not None):
      inputs = linear.Dense(
          features=self.d_post_proj,
          dtype=self.dtype,
          kernel_shardings=self.pre_logits_kernel_shardings,
          dot_general=self.post_proj_dot_general,
          precision=self.precision,
          lora_rank=self.lora_rank,
          lora_scale=self.lora_scale,
          name='pre_logits',
      )(inputs)
      inputs = nn.tanh(inputs)

    # this is only used for LILM
    outputs['features_all'] = inputs

    if self.aggregation_type in _NON_PARAMETRIC_AGGREGATION_TYPES:
      aggregated_outputs = NonParametricAggregatorHead(self.aggregation_type)(
          inputs=inputs)
      outputs.update(aggregated_outputs)

    elif self.aggregation_type == AggregationType.MULTI_HEAD_ATTENTION_POOL:
      outputs[constants.DataFeatureName.FEATURES] = inputs
      outputs[constants.DataFeatureName.FEATURES_AGG] = MAPHead(
          num_heads=self.num_heads,
          mlp_dim=self.d_ff,
          dropout_rate=self.dropout_rate,
          use_bias=self.use_bias,
          lora_rank=self.lora_rank,
          lora_scale=self.lora_scale,
      )(inputs, deterministic=deterministic)

    else:
      raise NotImplementedError('The requested pooling mechanism not '
                                f'supported: {self.aggregation_type}')

    # If the original embeddings shape specified, reshape to the feature maps
    if patched_shape is not None:
      features = outputs[constants.DataFeatureName.FEATURES]
      feature_map_shape = patched_shape[:-1] + (features.shape[-1],)
      feature_maps = jnp.reshape(features, feature_map_shape)
      outputs[constants.DataFeatureName.FEATURE_MAPS] = feature_maps

    if (self.post_proj_position == 'post_aggregation'
        and self.d_post_proj is not None):
      # in this case we only apply projection on the aggregated features
      features_agg = linear.Dense(
          features=self.d_post_proj,
          dtype=self.dtype,
          kernel_shardings=self.pre_logits_kernel_shardings,
          dot_general=self.post_proj_dot_general,
          precision=self.precision,
          lora_rank=self.lora_rank,
          lora_scale=self.lora_scale,
          name='pre_logits',
      )(outputs[constants.DataFeatureName.FEATURES_AGG])
      outputs[constants.DataFeatureName.FEATURES_AGG] = nn.tanh(features_agg)

    if self.num_classes is not None:
      outputs[constants.DataFeatureName.LOGITS] = linear.Dense(
          features=self.num_classes,
          dtype=self.dtype,
          kernel_init=nn.initializers.zeros,
          bias_init=nn.initializers.constant(self.head_bias_init),
          kernel_shardings=self.logits_kernel_shardings,
          dot_general=self.logits_dot_general,
          precision=self.precision,
          name='logits',
      )(outputs[constants.DataFeatureName.FEATURES_AGG])

    return outputs


class MLP(nn.Module):
  """A two-layer mlp module for common space projection."""

  d_common: int
  d_hidden: int | None = None
  dtype: jax.typing.DTypeLike = jnp.float32
  dot_general: typing.DotGeneral = lax.dot_general
  precision: typing.Precision = None
  lora_rank: int = 4
  lora_scale: float = 0.
  inner_kernel_shardings: typing.ShardingAxes = ()
  outer_kernel_shardings: typing.ShardingAxes = ()
  layernorm_shardings: typing.ShardingAxes = ()
  intermediate_shardings: typing.ShardingAxes = ()

  def setup(self):
    if self.d_hidden is not None:
      self.wi = linear.Dense(features=self.d_hidden,
                             dtype=self.dtype,
                             kernel_shardings=self.inner_kernel_shardings,
                             dot_general=self.dot_general,
                             precision=self.precision,
                             lora_rank=self.lora_rank,
                             lora_scale=self.lora_scale,
                             name='wi')
    self.wo = linear.Dense(features=self.d_common,
                           dtype=self.dtype,
                           kernel_shardings=self.outer_kernel_shardings,
                           dot_general=self.dot_general,
                           precision=self.precision,
                           lora_rank=self.lora_rank,
                           lora_scale=self.lora_scale,
                           name='wo')
    self.layer_norm = normalization.LayerNorm(
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='layer_norm',
    )

  def __call__(self,
               inputs,
               deterministic = None):
    del deterministic

    if self.d_hidden is not None:
      inputs = self.wi(inputs)
      inputs = nn.gelu(inputs)
      inputs = sharding.shard_array(inputs, self.intermediate_shardings)
    inputs = self.wo(inputs)
    outputs = self.layer_norm(inputs)

    return outputs


class Classifier(nn.Module):
  """A generic classification head.

  This module can be configured as below:
  ```
  classes_1 = (('k400', 400), ('jft', 10))
  classes_2 = 100,
  classes_3 = (('c4', 43),)
  cls_layer_1 = Classifier(classes=classes_1)
  cls_layer_2 = Classifier(classes=classes_2)
  cls_layer_3 = Classifier(classes=classes_3)
  ```

  and the user can expect the following when calling it wrt each 'classes':
  ```
  Parameters:
  layer_1 -> classifier_k400/kernel -> shape: (16, 400)
  layer_1 -> classifier_k400/bias -> shape: (400,)
  layer_1 -> classifier_jft/kernel -> shape: (16, 10)
  layer_1 -> classifier_jft/bias -> shape: (10,)
  layer_2 -> classifier/kernel -> shape: (16, 100)
  layer_2 -> classifier/bias -> shape: (100,)
  layer_3 -> classifier_c4/kernel -> shape: (16, 43)
  layer_3 -> classifier_c4/bias -> shape: (43,)

  Inputs:
  layer_1 -> shape: (1, 3, 16)
  layer_2 -> shape: (1, 3, 16)
  layer_3 -> shape: (1, 3, 16)

  Outputs:
  layer_1 -> logits_k400 (1, 3, 400)
  layer_1 -> logits_jft (1, 3, 10)
  layer_2 -> logits (1, 3, 100)
  layer_3 -> logits_c4 (1, 3, 43)
  ```

  Attributes:
    classes: An integer or a sequence of (str, int) pairs.
    predictions_key: The predictions (logits) prefix key. If integer is used
      to configure the classes, the `predictions_key` will be solely used to
      refer to the logits. Otherwise, it will be used as a prefix followed by
      the name of the classes. For example, if classes=('dataset_1', 23), then
      all logits for this very classification will have the name
      f'{predictions_key}_dataset_1'.
    kernel_shardings: Sharding annotations for dense kernel sharding.
    dot_general: The function that performs dot product.
    precision: The precision with which the dot product is performed.
    dtype: The activation dtype of the entire module (across all heads).
  """

  classes: typing.ClassificationHead = None
  predictions_key: str = constants.DataFeatureName.LOGITS
  kernel_shardings: typing.ShardingAxes = ()
  dot_general: typing.DotGeneral = lax.dot_general
  precision: typing.Precision = None
  dtype: jax.typing.DTypeLike = jnp.float32

  def _fetch_classification_config(
      self,
      classes
  ):
    """Verifies and returns the cls config as a tuple of (str, int)."""
    if isinstance(classes, tuple):
      for classes_name, num_classes in classes:
        if (not classes_name
            or not isinstance(num_classes, int)
            or not utils.is_sub_np_dtype(num_classes, int)):
          raise ValueError(
              'The `classes` configuration should contain valid (str, int) '
              f'pairs. Instead, received ({classes_name}, {num_classes}).')
      return classes

    elif isinstance(classes, int) or utils.is_sub_np_dtype(classes, int):
      return (('', int(classes)),)

    else:
      raise ValueError(
          'The `classes` config should be either an integer or a tuple of '
          f'(str, int) pairs. Instead, received {classes=}')

  def _setup_classifiers(
      self,
      cls_cfg,
      ):
    """Constructs the multi-head classification layers.

    Args:
      cls_cfg: A sequence of (str, int) pairs that define the classification
        heads.

    Returns:
      A nested dictionary that contains projections heads. This dictionary has
      the following structure:
          projections: {name_1: flax_module_1,
                        name_2: flax_module_2,
                        ...
                        name_n: flax_module_n}
    """
    projections = {}
    dense_class = functools.partial(
        linear.Dense,
        use_bias=True,
        dtype=self.dtype,
        dot_general=self.dot_general,
        precision=self.precision,
        kernel_shardings=self.kernel_shardings)
    for classes_name, num_classes in cls_cfg:
      layer_name = 'classifier'
      if classes_name:
        layer_name += f'_{classes_name}'
      projections[classes_name] = dense_class(
          features=num_classes, name=layer_name)

    return projections

  def setup(self):
    projections = {}
    if self.classes is not None:
      cls_cfg = self._fetch_classification_config(self.classes)
      projections.update(
          self._setup_classifiers(cls_cfg)
      )
    self.projections = projections

  def __call__(self, inputs):
    outputs = {}
    for cls_name, cls_layer in self.projections.items():
      output_name = self.predictions_key
      if cls_name:
        output_name += f'_{cls_name}'
      outputs[output_name] = cls_layer(inputs)
    return outputs


class DisjointCommonSpace(nn.Module):
  """Pair-wise cross-modal common space mapping."""

  d_common: int
  d_hidden: int
  target_modalities: Sequence[str]
  dtype: jax.typing.DTypeLike = jnp.float32
  dot_general: typing.DotGeneral = lax.dot_general
  precision: typing.Precision = None
  lora_rank: int = 4
  lora_scale: float = 0.
  inner_kernel_shardings: typing.ShardingAxes = ()
  outer_kernel_shardings: typing.ShardingAxes = ()
  layernorm_shardings: typing.ShardingAxes = ()
  intermediate_shardings: typing.ShardingAxes = ()

  def setup(self):
    projections = {}
    for target_modality in self.target_modalities:
      projections[target_modality] = MLP(
          d_hidden=self.d_hidden,
          d_common=self.d_common,
          dtype=self.dtype,
          dot_general=self.dot_general,
          precision=self.precision,
          lora_rank=self.lora_rank,
          lora_scale=self.lora_scale,
          inner_kernel_shardings=self.inner_kernel_shardings,
          outer_kernel_shardings=self.outer_kernel_shardings,
          layernorm_shardings=self.layernorm_shardings,
          intermediate_shardings=self.intermediate_shardings,
          name=f'to_{target_modality}',
      )
    self.projections = projections

  def __call__(self,
               inputs,
               target_modalities,
               deterministic = None):
    outputs = {}
    for target_modality in target_modalities:
      outputs[target_modality] = (
          self.projections[target_modality](inputs=inputs)
      )
    return outputs


class JointCommonSpace(nn.Module):
  """Joint common space mapping."""

  d_common: int
  d_hidden: int | None = None
  dtype: jax.typing.DTypeLike = jnp.float32
  dot_general: typing.DotGeneral = lax.dot_general
  precision: typing.Precision = None
  lora_rank: int = 4
  lora_scale: float = 0.
  inner_kernel_shardings: typing.ShardingAxes = ()
  outer_kernel_shardings: typing.ShardingAxes = ()
  layernorm_shardings: typing.ShardingAxes = ()
  intermediate_shardings: typing.ShardingAxes = ()

  def setup(self):
    self.projections = MLP(
        d_hidden=self.d_hidden,
        d_common=self.d_common,
        dtype=self.dtype,
        dot_general=self.dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        inner_kernel_shardings=self.inner_kernel_shardings,
        outer_kernel_shardings=self.outer_kernel_shardings,
        layernorm_shardings=self.layernorm_shardings,
        intermediate_shardings=self.intermediate_shardings,
        name='to_common',
    )

  def __call__(self,
               inputs,
               target_modalities,
               deterministic = None):

    common_outputs = self.projections(inputs=inputs)
    outputs = {}
    for target_modality in target_modalities:
      outputs[target_modality] = common_outputs

    return outputs
