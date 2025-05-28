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

"""Common standard model configuration."""

import dataclasses
from typing import Any, Type, TypeVar

import jax
from jax import numpy as jnp

from imp.max.config import base
from imp.max.utils import typing

ModelT = Type[TypeVar('_ModelT', bound='Model')]


@dataclasses.dataclass
class Model(base.Config):
  """Base configuration for any model.

  Attributes:
    name: the name of the config.
    dtype: the base dtype of the model activations.
  """

  name: str | None = None
  # TODO(b/234949870): deprecate this and merge with checkpointing pipeline
  init_override: str | None = None
  dtype: jax.typing.DTypeLike = jnp.float32


@dataclasses.dataclass
class RematScan(base.Config):
  """Configuration of remat+scan.

  Attributes:
    remat: Level of which remat is performed to re-compute certain ops/states
      during backward call. Acceptable levels: ('zero', 'minimal', 'full'). See
      https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.remat.html
    scanned_layers: Whether or not scanning repetitive layers should be
      performed. This allows fast compilation and enables partitioned layers,
      in which each layer is placed on certain devices. See
      https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.scan.html
    scan_axis: The axis index along which the layers would be rolled.
    scan_sharding_axis: The sharding axis name (as configured in the global
      mesh). It could hold None values for replication. Set this to non-None
      values for pipelining.
  """

  remat: str = 'zero'
  scanned_layers: bool = False
  scan_axis: int = 0
  scan_sharding_axis: str | None = None


@dataclasses.dataclass
class BaseMixtureOfExperts(base.Config):
  """Base configuration of Mixture-of-Experts transformation module.

  Attributes:
    num_experts: Number of available experts (feed-forward modules) in this
      layer.
    num_moe_layers: int, a positive integer indicating the total number of
      sparse layers.
    moe_layers_distribution: str, either 'last' or 'uniform'.
      If 'last' is used, then sparse layers are placed at the end. If 'uniform'
      is used, then sparse layers are distributed uniformly among all layers.
    ignore_padding_tokens: Whether to ignore padding tokens during routing. Note
      that some routers (e.g. TokensChooseMaskedRouter) will completely ignore
      padding tokens, while others (e.g. TokensChooseScatterRouter and
      ExpertsChooseMaskedRouter) will simply down-weight the probability of
      selecting padding tokens.
    jitter_noise: Amplitude of jitter noise applied to router logits.
    comm_dtype: The numeric type (default: bfloat16). This determines dtype
      of the entire tensor communication when dispatching tokens to experts.
      We recommend a truncated float type (e.g. bfloat16) to reduce all-to-all
      communication overhead.
    split_params: Whether or not to initialize each expert's parameters
      independently.
    optimize_parallel_comms: EXPERIMENTAL flag. If using
      model-parallelism for experts (experts spanning multiple devices), this
      flag is used to ensure that we do not perform duplicate all-to-all
      communication for experts spread across multiple devices, by partitioning
      the model/hidden dimension along the model-parallel axis for the experts.
      This same partition is used in Mesh Tensorflow:
      https://github.com/tensorflow/mesh/blob/6b31c0fc/mesh_tensorflow/transformer/moe.py#L487-L496
      This flag is experimental because, depending on model size and hardware
      topology, the reduction in all-to-all communication costs may be
      outweighed by the increased costs from the new reshape and all-gather.
      Current recommendation, roughly following Mesh Tensorflow, is only to use
      this flag for large models.
    router_kwargs: Optional tuple of pairs with further router parameters.
    router_kernel_shardings: Sharding annotations for router weights' kernels.
    tokens_shardings: Sharding annotations of the input array. This should
      represent (('expert', 'data'), 'instance', 'length', 'dim'), where
      ('expert', 'data') represent the 'batch' super-axis. Please note that
      'instance', 'length', and 'dim' could all be None in the global mesh.
    model_axis_size: Size of the model parallel submesh. Model parallelism
      is used if model_axis_size > 1.
    model_axis_name: The name of model axis in the global mesh.
  """

  num_experts: int = 8
  num_moe_layers: int = 6
  moe_layers_distribution: str = 'last'
  ignore_padding_tokens: bool = False
  jitter_noise: float = 0.0
  comm_dtype: jax.typing.DTypeLike = jnp.bfloat16
  split_params: bool = True
  optimize_parallel_comms: bool = False
  router_kwargs: tuple[tuple[str, Any], Ellipsis] = ()
  router_kernel_shardings: typing.ShardingAxes = ()
  tokens_shardings: typing.ShardingAxes = (('expert', 'data'), None, None, None)
  model_axis_size: int = 1
  model_axis_name: str = 'model'


@dataclasses.dataclass
class SparseMixtureOfExperts(BaseMixtureOfExperts):
  """Configuration of Sparse Mixture-of-Experts transformation module.

  Attributes:
    max_group_size: The total number of tokens (across the global batch) is
      subdivided into groups of this size, on each device. Router computations
      are then performed on a per-group basis. A larger group size will result
      in slower but more accurate top-k and sorting computations, whereas a
      smaller group size will result in faster but more approximate (and
      potentially less stable) routing choices. Note that actual group size may
      be smaller than max_group_size for consistency with the number of experts
      and tokens; see also `strict_group_size` attribute. In practice, we find
      that imperfect routing choices are tolerable and recommend choosing a
      group size on the order of 4096 tokens, although this number will vary
      based on model configuration and size.
    capacity_factor: Scaling factor to increase the expert token capacity
      during training. This factor plays an analogous, but slightly different,
      role depending on the routing assignment algorithm:
      - For 'tokens choose' routing, the capacity factor only affects the
        maximum number of tokens that an expert will process. It does not affect
        how many experts a given token is routed to; see the
        num_selected_experts attributes of 'tokens choose' routers.
      - For 'experts choose' routing, because experts always fill their buffer,
        increasing the capacity factor will increase the number of tokens that
        an expert will process AND will indirectly increase the number of
        experts that a given token is routed to.
    min_expert_capacity: Minimum token processing capacity for each expert.
    router_type: Type of the routing mechanism. Currently, 'TokensChooseScatter'
      'TokensChooseMasked', and 'ExpertsChooseMasked' are supported. Please
      refer to Flaxformer's implementation of these routing mechanisms at:
      modeling/routing.py for more details.
    router_bias: Whether to use bias in routing projection.
    strict_group_size: If True, fail if unable to set the token group size equal
      to max_group_size. If False (default), the actual group size may be
      smaller than max_group_size for consistency with the number of experts
      and tokens.
    num_selected_experts: Maximum number of experts to which each token is
      routed. Tokens may be routed to fewer experts if particular experts are
      oversubscribed / reach capacity. ONLY used in routing types
        'TokensChooseScatter' and 'TokensChooseMasked'.
    batch_prioritized_routing: Whether or not to use Batch Prioritized Routing
      (BPR), originally introduced in V-MoE (https://arxiv.org/abs/2106.05974).
        With BPR, we prioritize routing those top-k tokens with the highest
        router probability, rather than simply using each tokens left-to-right
        ordering in the batch. This prioritization is important because the
        expert's have limited capacity. ONLY used in routing types
        'TokensChooseScatter' and 'TokensChooseMasked'.
  """

  max_group_size: int = 8 * 256  # experts * patch**2
  capacity_factor: float = 1.0
  min_expert_capacity: int = 0
  router_type: str = 'ExpertsChooseMasked'
  router_bias: bool = False
  strict_group_size: bool = False
  num_selected_experts: int = 1
  batch_prioritized_routing: bool = False


@dataclasses.dataclass
class SoftMixtureOfExperts(BaseMixtureOfExperts):
  """Configuration of Soft Mixture-of-Experts transformation module.

  Attributes:
    expert_capacity: Minimum token processing capacity for each expert.
  """

  expert_capacity: int = 32

