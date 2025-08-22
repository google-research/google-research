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

"""Base transform module for Mixture-of-Experts modeling.

   This code is borrowed from Flaxformer's official MoE branch:
   https://github.com/google/flaxformer/tree/main/flaxformer/architectures/moe
   We make some changes to adapt it to the modeling modules under MAX.
"""

import functools
from typing import Any

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp

from imp.max.core import constants
from imp.max.core import probing
from imp.max.core import utils
from imp.max.modeling import routing
from imp.max.utils import sharding
from imp.max.utils import typing

PARAMS = constants.FlaxCollection.PARAMS
Router = routing.Router


class BaseMoE(nn.Module):
  """Base SPMD-friendly MoE layer with per-token routing.

  Attributes:
    num_experts: Number of available experts (feed-forward modules) in this
      layer.
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
    model_axis_name: Axis name used in global mesh for the model axis.
  """
  num_experts: int
  ignore_padding_tokens: bool
  jitter_noise: float
  comm_dtype: jax.typing.DTypeLike
  split_params: bool
  optimize_parallel_comms: bool
  router_kwargs: tuple[tuple[str, Any], Ellipsis]
  router_kernel_shardings: typing.ShardingAxes
  tokens_shardings: typing.ShardingAxes
  model_axis_size: int
  model_axis_name: str

  def assign_expert(self, expert, rng_keys):
    """Assigns expert and rng_keys from outer Setup call."""

    self.expert = expert
    self.rng_keys = rng_keys

  def _assert_configuration(self):
    """Verifies that the MoeLayer is correctly configured."""
    if self.optimize_parallel_comms and self.model_axis_size <= 1:
      raise ValueError(
          'optimize_parallel_comms=True with '
          f'model_axis_size={self.model_axis_size} has no effect; '
          f'please set optimize_parallel_comms=False.')

    if not hasattr(self, 'expert') or not hasattr(self, 'rng_keys'):
      raise ValueError('`expert` and/or `rng_keys` are not assigned. '
                       'Please use assign_expert method under an outer'
                       'setup method, or directly use BaseMoEwithExpert.')

    if self.ignore_padding_tokens:
      raise NotImplementedError

    if not self.tokens_shardings:
      raise ValueError('`tokens_shardings` should be assigned.')

    elif len(self.tokens_shardings) != 4:
      raise ValueError(
          '`tokens_shardings` should be 4 representing the following: '
          '(batch, instance, length, dim). Instead, configured '
          f'{self.tokens_shardings=}.')

    if len(self.tokens_shardings[0]) != 2:
      raise ValueError(
          'The `batch` dimension of `tokens_shardings` should be a 2D '
          "super-axis such as ('expert', 'data'). Instead, received "
          f'{self.tokens_shardings=}.')

  def _get_num_expert_replicas(self):
    """Infer the number of expert replicas.

    This computation assumes that we are using ('expert', 'data', 'model') as
    the partitioning mesh. In particular, we assume that experts are replicated
    along the 'data' axis, whose dimension is inversely proportional to the
    number of experts and number of model parallel dimensions. See also
    https://github.com/google-research/t5x/blob/bdd3928/t5x/contrib/moe/partitioning.py.

    Returns:
      Number of replicas per expert.
    """
    return max(
        1, jax.device_count() // (self.num_experts * self.model_axis_size))

  def _mask_and_dispatch_to_experts(
      self,
      router,
      token_inputs,
      apply_jitter,
      expert_capacity,
      metadata,
      **kwargs,
  ):
    """Wraps expert masked routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute dispatch mask and combine array using router.
    (2) Dispatch inputs to experts based on dispatch mask.
    (3) Recombine individual expert outputs using combine array.

    Args:
      router: Token dispatch router. The router determines which tokens are
        dispatched to which expert, and how the expert outputs are combined.
      token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to
        send to experts.
      apply_jitter: If True, apply jitter noise during routing.
      expert_capacity: Each group will send this many tokens to each expert.
      metadata: Dict, contains batch metadata (such as modality).
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      <float>[num_groups, tokens_per_group, hidden_dim] outputs from experts.
    """
    num_groups, tokens_per_group, _ = token_inputs.shape
    num_tokens = num_groups * tokens_per_group

    router_mask = router(
        token_inputs,
        self.num_experts,
        expert_capacity,
        apply_jitter=apply_jitter,
        metadata=metadata)

    # Shape: [num_groups, num_experts, expert_capacity, hidden_dim].
    expert_inputs = jnp.einsum(
        '...th,...tec->...ech',
        token_inputs,
        router_mask.dispatch_mask,
        precision=jax.lax.Precision.DEFAULT)

    expert_outputs = self._call_experts(expert_inputs, **kwargs)

    # Shape: [num_groups, tokens_per_group, hidden_dim]
    combined_outputs = jnp.einsum(
        '...ech,...tec->...th',
        expert_outputs,
        router_mask.combine_array,
        precision=jax.lax.Precision.DEFAULT)

    # Gather and sow expert metrics.
    # Number of tokens that were dispatched to at least one expert.
    num_tokens_dispatched_somewhere = jnp.max(
        router_mask.dispatch_mask > 0, axis=(-1, -2)).sum(dtype=jnp.float32)
    fraction_tokens_left_behind = 1.0 - num_tokens_dispatched_somewhere / float(
        num_tokens)
    # Total number of tokens that were dispatched (one token could be
    # dispatched to multiple experts).
    num_tokens_dispatched = router_mask.dispatch_mask.sum()
    # Of the tokens dispatched, how confident was the router in its routing?
    router_confidence = router_mask.combine_array.sum() / num_tokens_dispatched

    if isinstance(router, routing.ExpertsChooseMaskedRouter):
      expert_usage = 1.  # Experts fully utilized when 'expert choose tokens'
    else:
      total_expert_capacity = self.num_experts * expert_capacity * num_groups
      expert_usage = num_tokens_dispatched / total_expert_capacity

    self._sow_expert_metrics(
        auxiliary_loss=router_mask.auxiliary_loss,
        router_z_loss=router_mask.router_z_loss,
        routing_probs=router_mask.router_probs,
        fraction_tokens_left_behind=fraction_tokens_left_behind,
        router_confidence=router_confidence,
        expert_usage=expert_usage,
        metadata=metadata,
        probe_routing_distributions=False,
    )

    return combined_outputs, router_mask.router_probs

  def _call_experts(self, inputs, **kwargs):
    """Sends and receives inputs to experts using pjit induced all_to_all calls.

    Assumes training is distributed using jax.experimental.pjit and induces
    all_to_all calls using reshapes and sharding constraints. We use Flax's
    lifted vmap to apply the expert transformation.

    The entire computations is performed using self.comm_dtype. We recommend a
    truncated float type (e.g. bfloat16) to reduce all-to-all communication
    overhead.

    Args:
      inputs: <float>[num_groups, num_experts, expert_capacity, hidden_dim]
        inputs to be dispatched to experts. Each slice across the first
        dimension is passed to a different expert.
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      <float>[num_groups, num_experts, expert_capacity, hidden_dim] outputs from
      expert computation.
    """
    num_groups, num_experts, capacity, hidden_dim = inputs.shape
    inputs_dtype = inputs.dtype
    inputs = jax.lax.convert_element_type(inputs, self.comm_dtype)

    # Send examples to their target devices.
    # Note that the ordering of the logical mesh axes in these sharding
    # constraints should map consistently to the same underlying mesh axes; i.e.
    # 'batch' --> ('expert', 'data') and
    # ('expert', 'expert_replica') --> ('expert', 'data').

    # Re-assure sharding with: ('batch', None, 'length', 'dim')
    inputs = sharding.shard_array(
        inputs, (self.tokens_shardings[0],
                 None,
                 self.tokens_shardings[2],
                 self.tokens_shardings[3]))

    # Split the batch dim to number of experts and their residue
    inputs = inputs.reshape(
        num_experts, num_groups // num_experts,
        num_experts, capacity, hidden_dim)

    # Re-assure sharding with: ('expert', 'data', None, 'length', 'dim')
    inputs = sharding.shard_array(
        inputs, (self.tokens_shardings[0][0],
                 self.tokens_shardings[0][1],
                 None,
                 self.tokens_shardings[2],
                 self.tokens_shardings[3]))

    if self.optimize_parallel_comms:
      # Partition inputs along model parallel submesh axis to reduce duplicate
      # all-to-all communications in model parallelism cases.
      # Shape: [num_experts, num_groups // num_experts, num_experts, capacity,
      # hidden_dim, self.model_axis_size]
      inputs = self._swapaxes_with_sharding_constraint(inputs, 0, 2, capacity,
                                                       hidden_dim)
    else:
      # Shape: [num_experts, num_groups // num_experts, num_experts, capacity,
      # hidden_dim]
      inputs = jnp.swapaxes(inputs, 0, 2)

    inputs = inputs.reshape(num_experts, num_groups * capacity, hidden_dim)

    # Re-assure sharding with: ('expert', 'data', 'dim')
    inputs = sharding.shard_array(
        inputs, (self.tokens_shardings[0][0],
                 self.tokens_shardings[0][1],
                 self.tokens_shardings[3])
    )

    # Apply expert transformation.

    # Vectorize over the 'expert' axis of `inputs`. We use Flax's Lifted vmap
    # to introduce parameters along the mapped `expert` axis.
    expert_axis_name = self.tokens_shardings[0][0]
    @functools.partial(
        nn.vmap,
        in_axes=(0,),
        variable_axes={PARAMS: 0},  # Each expert has its own parameters
        # Any mapped sharding constraints should be applied along 'expert' axis.
        spmd_axis_name=expert_axis_name,
        split_rngs={k: True for k in self.rng_keys + (PARAMS,)},
        metadata_params={nn.meta.PARTITION_NAME: expert_axis_name},
    )
    def layer_fn(mapped_expert, inputs):
      outputs = mapped_expert(inputs, **kwargs)
      return jax.lax.convert_element_type(outputs, self.comm_dtype)

    outputs = layer_fn(self.expert, inputs)
    hidden_dim = outputs.shape[-1]

    # Send examples back to their original devices.
    outputs = outputs.reshape(num_experts, num_groups, capacity, hidden_dim)
    # Re-assure sharding with: ('expert', 'data', 'length', 'dim')
    outputs = sharding.shard_array(
        outputs, (self.tokens_shardings[0][0],
                  self.tokens_shardings[0][1],
                  self.tokens_shardings[2],
                  self.tokens_shardings[3]))
    # Prepare outputs for swapping axes
    outputs = outputs.reshape(num_experts, num_groups // num_experts,
                              num_experts, capacity, hidden_dim)

    if self.optimize_parallel_comms:
      # Shape: [num_experts, num_groups // num_experts, num_experts, capacity,
      # hidden_dim, self.model_axis_size]
      outputs = self._swapaxes_with_sharding_constraint(outputs, 0, 2, capacity,
                                                        hidden_dim)
    else:
      # Shape: [num_experts, num_groups // num_experts, num_experts, capacity,
      # hidden_dim]
      outputs = jnp.swapaxes(outputs, 0, 2)

    # Reshape outputs after swap to the original inputs shape
    outputs = outputs.reshape(num_groups, num_experts, capacity, hidden_dim)

    # Re-assure sharding with: ('batch', None, 'length', 'dim')
    outputs = sharding.shard_array(
        outputs, (self.tokens_shardings[0],
                  None,
                  self.tokens_shardings[2],
                  self.tokens_shardings[3]))

    return jax.lax.convert_element_type(outputs, inputs_dtype)

  def _sow_expert_metrics(self,
                          auxiliary_loss,
                          router_z_loss,
                          routing_probs,
                          fraction_tokens_left_behind,
                          router_confidence,
                          expert_usage,
                          metadata,
                          probe_routing_distributions = False):
    """Sows metrics to analyze expert routing.

    Args:
      auxiliary_loss: Auxiliary load balancing loss.
      router_z_loss: Router z-loss. Encourages router logits to remain small in
        an effort to improve stability.
      routing_probs: array, contains the routing probabilities.
      fraction_tokens_left_behind: Fraction of tokens NOT processed by any
        expert.
      router_confidence: How confident the router is about the tokens that it
        has routed.
      expert_usage: Fraction of total capacity, across all experts, used to
        process tokens. Larger expert capacities or non-uniform token routing
        will result in smaller expert usage values.
      metadata: Dict, contains batch metadata (such as modality).
      probe_routing_distributions: bool, default to False. If True, for each
        expert and modality, it adds a histogram with the distribution
        of routing probabilities.
    """

    if probe_routing_distributions:
      # Add distribution of routing probabilities for each (modality, expert).
      num_experts = routing_probs.shape[-1]
      routing_probs = jnp.reshape(routing_probs, (-1, num_experts))
      modality = (metadata or {}).get('modality', None)
      if modality is not None:
        for expert_idx in range(num_experts):
          routing_to_expert_idx = utils.take_along_axis(
              routing_probs, expert_idx, axis=1, precision='bfloat16'
          )
          probing.add_histogram_probe(
              data=routing_to_expert_idx,
              name=f'moe_routing_probs/{modality}/expert_{expert_idx}')

    probing.add_scalar_probe(data=auxiliary_loss,
                             name='moe_auxiliary_loss')
    probing.add_scalar_probe(data=router_z_loss,
                             name='moe_router_z_loss')
    probing.add_scalar_probe(data=router_confidence,
                             name='moe_router_confidence')
    probing.add_scalar_probe(data=expert_usage,
                             name='moe_expert_usage')
    probing.add_scalar_probe(data=fraction_tokens_left_behind,
                             name='moe_fraction_tokens_left_behind')
    probing.add_aux_loss(data=auxiliary_loss + router_z_loss,
                         name='moe_loss')

  def _swapaxes_with_sharding_constraint(
      self,
      array,
      axis1,
      axis2,
      expert_capacity,
      hidden_dim,
  ):
    """Interchanges two array axes under model-parallel sharding constraints.

    For model-parallel experts (self.model_axis_size > 1), to ensure that
    multiple devices are not performing all-to-all duplicate communications, we
    partition the model/hidden dimension along the expert model-parallel axis
    ('expert_embed') before performing the all-to-all transfer. See also:
    https://github.com/tensorflow/mesh/blob/6b31c0fc/mesh_tensorflow/transformer/moe.py#L487-L496


    Args:
      array: Input array.
      axis1: First axis.
      axis2: Second axis.
      expert_capacity: Token capacity per expert.
      hidden_dim: Model/hidden dimension of inputs.

    Returns:
      View or copy of input array with axes swapped.

    Raises:
      ValueError if model_axis_size is less than or equal to 1.
    """
    if self.model_axis_size <= 1:
      raise ValueError('Expected model_axis_size to be > 1 but got: '
                       f'{self.model_axis_size}')
    array = array.reshape(self.num_experts, -1, self.num_experts,
                          expert_capacity,
                          hidden_dim // self.model_axis_size,
                          self.model_axis_size)

    # Sharding will represent ('expert', 'data', None, 'length', 'dim', 'model')
    sharding_axes = (
        self.tokens_shardings[0][0],
        self.tokens_shardings[0][1],
        None,
        self.tokens_shardings[2],
        self.tokens_shardings[3],
        self.model_axis_name,
    )
    array = sharding.shard_array(array, sharding_axes)
    array = jnp.swapaxes(array, axis1, axis2)
    array = sharding.shard_array(array, sharding_axes)
    return array


class BaseSparseMoE(BaseMoE):
  """Sparse variant of the MoE layer.

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
  max_group_size: int
  capacity_factor: float
  min_expert_capacity: int
  router_type: str    # TODO(b/267474477): Merge it into `router_kwargs` below.
  router_bias: bool
  strict_group_size: bool
  num_selected_experts: int
  batch_prioritized_routing: bool

  @nn.compact
  def __call__(self,
               inputs,
               deterministic_routing = True,
               **kwargs):
    """Applies MoeLayer module.

    If the 'intermediates' collection is marked as mutable, this method will sow
    diversity metrics.

    Args:
      inputs: Batch of input embeddings of shape <float>[batch_size, seq_length,
        hidden_dim].
      deterministic_routing: Disables jitter augmentation in routing projection
        if set to True.
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      Output of the transformed expert module:
        <float>[batch_size, seq_length, transformed_hidden_dim].

    Raises:
      ValueError if an unrecognized dispatch algorithm is given.
    """
    self._assert_configuration()
    deterministic_routing = kwargs.get('deterministic', deterministic_routing)

    # Re-assure sharding for inputs
    inputs = sharding.shard_array(inputs, self.tokens_shardings)

    batch_size, instance, seq_length, hidden_dim = inputs.shape
    num_tokens = batch_size * instance * seq_length
    metadata = kwargs.pop('metadata', {})
    num_expert_replicas = self._get_num_expert_replicas()
    num_groups = _num_groups(num_tokens, self.max_group_size, self.num_experts,
                             num_expert_replicas, self.strict_group_size)
    tokens_per_group = num_tokens // num_groups

    # Each group will send expert_capacity tokens to each expert.
    expert_capacity = int(
        round(self.capacity_factor * tokens_per_group / self.num_experts))
    expert_capacity = max(expert_capacity, self.min_expert_capacity)

    # Reshape batch and sequence/token dimensions for expert routing.
    inputs = jnp.reshape(inputs, (num_groups, tokens_per_group, hidden_dim))

    if self.router_type == 'TokensChooseScatter':
      router_module = functools.partial(
          routing.TokensChooseScatterRouter,
          num_selected_experts=self.num_selected_experts,
          batch_prioritized_routing=self.batch_prioritized_routing)
      routing_call = self._scatter_to_experts

    elif self.router_type == 'TokensChooseMasked':
      router_module = functools.partial(
          routing.TokensChooseMaskedRouter,
          num_selected_experts=self.num_selected_experts,
          batch_prioritized_routing=self.batch_prioritized_routing)
      routing_call = self._mask_and_dispatch_to_experts

    elif self.router_type == 'ExpertsChooseMasked':
      router_module = routing.ExpertsChooseMaskedRouter
      routing_call = self._mask_and_dispatch_to_experts

    else:
      raise ValueError(f'Unrecognized router type: {self.router_type}')

    # Router inputs shardings should represent (batch, length, dim)
    router_inputs_shardings = (self.tokens_shardings[0],
                               self.tokens_shardings[2],
                               self.tokens_shardings[3])
    # Router inputs shardings should represent (batch, length, None)
    router_logits_shardings = (self.tokens_shardings[0],
                               self.tokens_shardings[2],
                               None)
    router = router_module(
        router_weights=routing.RouterWeights(
            use_bias=self.router_bias,
            dtype=jnp.float32,  # router dtype is always float32 for accuracy
            kernel_shardings=self.router_kernel_shardings,
            name='router',
        ),
        ignore_padding_tokens=self.ignore_padding_tokens,
        jitter_noise=self.jitter_noise,
        dtype=self.comm_dtype,
        inputs_shardings=router_inputs_shardings,
        logits_shardings=router_logits_shardings,
        router_kwargs=dict(self.router_kwargs))
    outputs, routing_probs = routing_call(
        router=router,
        token_inputs=inputs,
        apply_jitter=not deterministic_routing,
        expert_capacity=expert_capacity,
        metadata=metadata,
        **kwargs)
    # TODO(b/267155837): compute and store metrics here with `routing_probs`.
    del routing_probs
    # Return to original input shape.
    hidden_dim = outputs.shape[-1]
    outputs = outputs.reshape((batch_size, instance, seq_length, hidden_dim))
    outputs = sharding.shard_array(outputs, self.tokens_shardings)
    return outputs

  def _scatter_to_experts(self,
                          router,
                          token_inputs,
                          apply_jitter,
                          expert_capacity,
                          metadata,
                          **kwargs):
    """Wraps expert scatter routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute expert dispatch indices and combine weights using router.
    (2) Scatter inputs to experts based on dispatch indices.
    (3) Recombine individual expert outputs using combine weights.

    Args:
      router: Token dispatch router. The router determines which tokens are
        dispatched to which expert, and how the expert outputs are combined.
      token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to
        send to experts.
      apply_jitter: If True, apply jitter noise during routing.
      expert_capacity: Each group will send this many tokens to each expert.
      metadata: Dict, contains batch metadata (such as modality).
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      <float>[num_groups, tokens_per_group, hidden_dim] outputs from experts.
    """
    num_groups, tokens_per_group, hidden_dim = token_inputs.shape
    num_tokens = num_groups * tokens_per_group

    router_indices = router(
        token_inputs,
        self.num_experts,
        expert_capacity,
        apply_jitter=apply_jitter,
        metadata=metadata)
    num_selected_experts = router.num_selected_experts

    # We need num_selected_experts copies of inputs for dispatching. This is a
    # no-op if num_selected_experts = 1.
    token_inputs = jnp.repeat(token_inputs, num_selected_experts, axis=1)

    # Mask out inputs that should not be routed.
    # Shape: [num_groups, tokens_per_group, num_selected_experts].
    successfully_routed = jnp.logical_and(
        router_indices.dispatch_indices[Ellipsis, 0] < self.num_experts,
        router_indices.dispatch_indices[Ellipsis, 1] < expert_capacity)
    successfully_routed = successfully_routed.reshape((num_groups, -1))
    # Shape: [num_groups, tokens_per_group * num_selected_experts, hidden_dim].
    masked_inputs = jnp.einsum(
        '...th,...t->...th',
        token_inputs,
        successfully_routed,
        precision=jax.lax.Precision.DEFAULT)

    # Combine tokens_per_group and num_selected_experts axes.
    flattened_dispatch_indices = router_indices.dispatch_indices.reshape(
        num_groups, -1, 2)

    # Scatter masked inputs.
    shape = (self.num_experts, expert_capacity, hidden_dim)
    # Note: scatter_nd can be slow under pjit on TPUs, presumably due to
    # suboptimal SPMD compilations. On TPUs, the recommendation is to use
    # MaskedRouter(s) instead.
    # Shape: [num_groups, num_experts, expert_capacity, hidden_dim].
    expert_inputs = jax.vmap(
        lambda i, x: utils.scatter_nd(i, x, shape)
    )(flattened_dispatch_indices, masked_inputs)

    expert_outputs = self._call_experts(expert_inputs, **kwargs)
    hidden_dim = expert_outputs.shape[-1]

    # Gather outputs.
    # Shape: [num_groups, tokens_per_group * num_selected_experts, hidden_dim].
    expert_outputs = jax.vmap(lambda i, x: x[i[:, 0], i[:, 1]])(
        flattened_dispatch_indices, expert_outputs)
    # Separate out num_selected_experts dimension.
    # Shape: [num_groups, tokens_per_group, num_selected_experts, hidden_dim].
    expert_outputs = expert_outputs.reshape(
        (num_groups, tokens_per_group, num_selected_experts, hidden_dim))

    # Shape: [num_groups, tokens_per_group, num_selected_experts, hidden_dim].
    # Weighted sum of the outputs from the different experts.
    combined_outputs = jnp.einsum(
        '...tkh,...tk->...th',
        expert_outputs,
        router_indices.combine_weights,
        precision=jax.lax.Precision.DEFAULT)

    # Gather and sow expert metrics.
    successfully_routed = successfully_routed.reshape(
        (num_groups, tokens_per_group, num_selected_experts))
    # Number of tokens that were dispatched to at least one expert.
    num_tokens_dispatched_somewhere = jnp.max(
        successfully_routed, axis=-1).sum()
    fraction_tokens_left_behind = 1.0 - num_tokens_dispatched_somewhere / float(
        num_tokens)
    # Total number of tokens that were dispatched (one token could be dispatched
    # to multiple experts).
    num_tokens_dispatched = successfully_routed.sum()
    total_expert_capacity = self.num_experts * expert_capacity * num_groups
    expert_usage = num_tokens_dispatched / total_expert_capacity
    # Of the tokens dispatched, how confident was the router in its routing.
    router_confidence = (
        router_indices.combine_weights.sum() / num_tokens_dispatched)

    self._sow_expert_metrics(
        auxiliary_loss=router_indices.auxiliary_loss,
        router_z_loss=router_indices.router_z_loss,
        routing_probs=router_indices.router_probs,
        fraction_tokens_left_behind=fraction_tokens_left_behind,
        router_confidence=router_confidence,
        expert_usage=expert_usage,
        metadata=metadata,
        probe_routing_distributions=False,
    )

    return combined_outputs, router_indices.router_probs


class BaseSoftMoE(BaseMoE):
  """Soft variant of the MoE layer.

  Attributes:
    expert_capacity: Minimum token processing capacity for each expert.
  """
  expert_capacity: int

  @nn.compact
  def __call__(self,
               inputs,
               deterministic_routing = True,
               **kwargs):
    """Applies MoeLayer module.

    If the 'intermediates' collection is marked as mutable, this method will sow
    diversity metrics.

    Args:
      inputs: Batch of input embeddings of shape <float>[batch_size, seq_length,
        hidden_dim].
      deterministic_routing: Disables jitter augmentation in routing projection
        if set to True.
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      Output of the transformed expert module:
        <float>[batch_size, seq_length, transformed_hidden_dim].

    Raises:
      ValueError if an unrecognized dispatch algorithm is given.
    """
    self._assert_configuration()
    deterministic_routing = kwargs.get('deterministic', deterministic_routing)

    # Re-assure sharding for inputs
    inputs = sharding.shard_array(inputs, self.tokens_shardings)

    batch_size, instance, seq_length, hidden_dim = inputs.shape
    metadata = kwargs.pop('metadata', {})

    # Reshape inputs to have shape [num_groups, tokens_per_group, hidden_dim]
    # with num_groups = batch_size * instance, tokens_per_group = seq_length.
    inputs = jnp.reshape(inputs, (-1, seq_length, hidden_dim))

    # Router inputs shardings should represent (batch, length, dim)
    router_inputs_shardings = (self.tokens_shardings[0],
                               self.tokens_shardings[2],
                               self.tokens_shardings[3])
    # Router inputs shardings should represent (batch, length, None)
    router_logits_shardings = (self.tokens_shardings[0],
                               self.tokens_shardings[2],
                               None)

    router = routing.SoftRouter(
        router_weights=routing.NormalizedRouterWeights(
            dtype=jnp.float32,
            mu_shardings=self.router_kernel_shardings,
            name='router',
        ),
        ignore_padding_tokens=self.ignore_padding_tokens,
        jitter_noise=self.jitter_noise,
        dtype=self.comm_dtype,
        inputs_shardings=router_inputs_shardings,
        logits_shardings=router_logits_shardings,
        router_kwargs=dict(self.router_kwargs))
    outputs, _ = self._mask_and_dispatch_to_experts(
        router=router,
        token_inputs=inputs,
        apply_jitter=not deterministic_routing,
        expert_capacity=self.expert_capacity,
        metadata=metadata,
        **kwargs)
    # Return to original input shape.
    hidden_dim = outputs.shape[-1]
    outputs = outputs.reshape((batch_size, instance, seq_length, hidden_dim))
    outputs = sharding.shard_array(outputs, self.tokens_shardings)
    return outputs


class SparseMoEwithExpert(BaseSparseMoE):
  """SparseMoE with expert as init args."""

  expert: nn.Module
  rng_keys: tuple[str, Ellipsis]


class SoftMoEwithExpert(BaseSoftMoE):
  """SoftMoE with expert as init args."""

  expert: nn.Module
  rng_keys: tuple[str, Ellipsis]


def _num_groups(num_tokens,
                max_group_size,
                num_experts,
                num_expert_replicas,
                strict_group_size = False):
  """Returns the number of token routing groups.

  Note: For pjit-based training, all quantities are global.

  We select the smallest num_groups such that:
  - num_groups >= num_tokens / max_group_size (ensuring the group size is no
    larger than max_group_size),
  - num_tokens % num_groups = 0 (ensuring that the group size evenly divides
    into the num_tokens),
  - num_groups % (num_expert_replicas * num_experts) = 0 (ensuring that number
    of groups can be split across the total number of experts).

  Args:
    num_tokens: Number of tokens from input batch.
    max_group_size: Maximum size of each token routing group. Actual group size
      may end up being smaller.
    num_experts: Total number of unique experts.
    num_expert_replicas: Number of copies of each expert.
    strict_group_size: If True, fail if unable to set the token group size equal
      to max_group_size.

  Returns:
    Number of token routing groups.

  Raises:
    ValueError if we cannot find a group_size satisfying the above requirements.
  """
  # For pjit-based partitioning, we manipulated arrays globally. The number of
  # experts must evenly divide the number of (global) groups.
  min_num_groups = num_tokens // max_group_size
  min_num_groups = max(min_num_groups, num_expert_replicas * num_experts)

  def viable(n):
    """Returns true iff n is a viable number of groups."""
    return num_tokens % n == 0 and n % (num_expert_replicas * num_experts) == 0

  # Increase the number of groups (and decrease the group size) until we have
  # a viable number of groups.
  num_groups = min_num_groups
  while num_groups < num_tokens and not viable(num_groups):
    num_groups += 1

  if num_tokens % num_groups > 0:
    raise ValueError(
        'Group size and the number of experts must divide evenly into the '
        f'global number of tokens, but num_tokens={num_tokens}, while '
        f'num_groups={num_groups} for max_group_size={max_group_size} '
        f'and num_experts={num_experts}, each with {num_expert_replicas} '
        'replicas')

  group_size = num_tokens // num_groups
  logging.info(
      'Selected group_size=%d and num_groups=%d for input num_tokens=%d, '
      'max_group_size=%d, num_experts=%d and num_expert_replicas=%d',
      group_size, num_groups, num_tokens, max_group_size, num_experts,
      num_expert_replicas)

  if strict_group_size and group_size != max_group_size:
    raise ValueError(
        f'Selected group_size={group_size} is less than the '
        f'max_group_size={max_group_size}. Exiting because strict mode is '
        'active (strict_group_size=True)')

  return num_groups
