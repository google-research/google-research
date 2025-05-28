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

"""Mixture of Experts routing mechanisms."""

import typing as pytyping
from typing import Any

from absl import logging
import flax
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp

from imp.max.core import constants
from imp.max.modeling import linear
from imp.max.utils import sharding
from imp.max.utils import typing

RouterOutput = Any
Modality = constants.Modality


def _favor_one_hot_slices():
  """Returns true iff running on TPUs."""
  return jax.default_backend() == 'tpu' or jax.devices()[0].platform == 'tpu'


def _take_along_axis(array,
                     indices,
                     axis):
  """Takes values from the input array by matching 1D index and data slices.

  This function serves the same purpose as jax.numpy.take_along_axis, except
  that it uses one-hot matrix multiplications under the hood on TPUs:
  (1) On TPUs, we use one-hot matrix multiplications to select elements from the
      array; this is particularly helpful for avoiding erroneous all-gather ops
      when running under pjit.
  (2) Otherwise, we fall back to jax.numpy.take_along_axis.

  Notes:
    - To simplify matters in case (1), we only support slices along the second
      or last dimensions.
    - We may wish to revisit (1) for very large arrays.

  Args:
    array: Source array.
    indices: Indices to take along each 1D slice of array.
    axis: Axis along which to take 1D slices.

  Returns:
    The indexed result.
  """
  if array.ndim != indices.ndim:
    raise ValueError(
        'indices and array must have the same number of dimensions; '
        f'{indices.ndim} vs. {array.ndim}.')

  if (axis != -1 and axis != array.ndim - 1 and  # Not last dimension
      axis != 1 and axis != -array.ndim + 1):  # Not second dimension
    raise ValueError(
        'Only slices along the second or last dimension are supported; '
        f'array.ndim = {array.ndim}, while axis = {axis}.')

  if _favor_one_hot_slices():
    one_hot_length = array.shape[axis]
    one_hot_indices = jax.nn.one_hot(indices, one_hot_length, axis=axis)

    if axis == -1 or array.ndim == 1:
      # Take i elements from last dimension (s).
      # We must use HIGHEST precision to accurately reproduce indexing
      # operations with matrix multiplications.
      result = jnp.einsum(
          '...s,...is->...i',
          array,
          one_hot_indices,
          precision=jax.lax.Precision.HIGHEST)
    else:
      # Take i elements from second dimension (s). We assume here that we always
      # want to slice along the second dimension.
      # We must use HIGHEST precision to accurately reproduce indexing
      # operations with matrix multiplications.
      result = jnp.einsum(
          'ns...,nis...->ni...',
          array,
          one_hot_indices,
          precision=jax.lax.Precision.HIGHEST)
    return jax.lax.convert_element_type(result, array.dtype)
  else:
    return jnp.take_along_axis(array, indices, axis=axis)


# TODO(hassanak): Seems like this is resolved. Fall back to jax.lax.top_k
def _top_k(array, k):
  """Returns top k values and their indices along the last axis of the array.

  This function serves the same purpose as jax.lax.top_k, but in a more XLA
  friendly manner for TPUs:
  (1) On TPUs, we use one-hot matrix multiplications to select the top k values.
      This convoluted way of obtaining the top k values is generally faster on
      TPUs, and, for pjit in particular, avoids adding extra all-gather ops
      during backpropagation.
  (2) Otherwise, we fall back to jax.lax.top_k (and its underlying scatter op).

  Args:
    array: Source array.
    k: Number of top values to select.

  Returns:
    - Top k values
    - Associated top k indices.
  """
  if _favor_one_hot_slices():
    top_k_indices = jax.lax.top_k(array, k)[-1]
    top_k_values = _take_along_axis(array, top_k_indices, axis=-1)
    return top_k_values, top_k_indices
  else:
    return jax.lax.top_k(array, k)


def _load_balancing_loss(router_probs,
                         expert_indices):
  """Computes auxiliary load balancing loss as in Switch Transformer.

  See Switch Transformer (https://arxiv.org/abs/2101.03961). This function
  implements the loss function presented in equations (4) - (6). It aims to
  penalize those cases where the routing between experts is unbalanced.

  Args:
    router_probs: Probability assigned to each expert per token. Shape:
      <float32>[num_groups, tokens_per_group, num_experts].
    expert_indices: <int>[num_groups, tokens_per_group, num_selected_experts]
      indices identifying the top num_selected_experts for a given token.

  Returns:
    The auxiliary loss.
  """
  num_experts = router_probs.shape[-1]

  # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
  expert_mask = jax.nn.one_hot(expert_indices, num_experts, dtype=jnp.int32)
  # For a given token, determine if it was routed to a given expert.
  # Shape: [num_groups, tokens_per_group, num_experts]
  expert_mask = jnp.max(expert_mask, axis=-2)

  tokens_per_group_and_expert = jnp.mean(
      expert_mask, dtype=jnp.float32, axis=-2)
  router_prob_per_group_and_expert = jnp.mean(
      router_probs, dtype=jnp.float32, axis=-2)
  return (
      jnp.mean(
          tokens_per_group_and_expert * router_prob_per_group_and_expert,
          dtype=jnp.float32,
      )
      * num_experts**2
  )


def _router_z_loss(router_logits):
  """Compute router z-loss.

   The router z-loss was introduced in Designing Effective Sparse Expert Models
   (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
   small in an effort to improve stability.

  Args:
    router_logits: <float>[num_groups, tokens_per_group, num_experts] router
      logits.

  Returns:
    Scalar router z-loss.
  """
  num_groups, tokens_per_group, _ = router_logits.shape
  log_z = jax.nn.logsumexp(router_logits, axis=-1)
  z_loss = log_z**2
  return jnp.sum(z_loss, dtype=jnp.float32) / (num_groups * tokens_per_group)

# Switch Transformer (https://arxiv.org/abs/2101.03961) suggests using
# nn.initializers.variance_scaling(0.1, "fan_in", "truncated_normal")
# scaling throughout MoE models, but we find slightly better results adopting
# typical normally-distributed scaling for the router specifically.
default_kernel_init = nn.initializers.normal(stddev=2e-2)
default_bias_init = nn.initializers.zeros


@flax.struct.dataclass
class RouterIndices:
  """Dispatch indices and combine weights for scatter/gather-based routing.

  Attributes:
    dispatch_indices: <int32>[num_groups, tokens_per_group,
      num_selected_experts, 2] dispatch indices indicating, for each token, its
      preferred expert and its priority in that expert's buffer.
    combine_weights: <float>[num_groups, tokens_per_group, num_selected_experts]
      combine weights used for scaling expert outputs with the router's dispatch
      probability/confidence.
    auxiliary_loss: Load balancing loss for router.
    router_probs: <float>[num_groups, tokens_per_group, num_experts]
      contains the probabilities output by the router for each token.
    router_z_loss: Router z-loss. Encourages router logits to remain small in an
      effort to improve stability.
  """
  dispatch_indices: jax.Array
  combine_weights: jax.Array
  auxiliary_loss: jax.Array
  router_probs: jax.Array | None
  router_z_loss: jax.Array


@flax.struct.dataclass
class RouterMask:
  """Dispatch and combine arrays for expert routing with masked matmuls.

  Attributes:
    dispatch_mask: <bool>[num_groups, tokens_per_group, num_experts,
      expert_capacity] dispatch array that is 1 if the token gets routed to the
      corresponding expert, and 0 otherwise.
    combine_array: <float>[num_groups, tokens_per_group, num_experts,
      expert_capacity] combine array used for combining expert outputs and
      scaling with router probability.
    auxiliary_loss: Load balancing loss for router.
    router_probs: <float>[num_groups, tokens_per_group, num_experts]
      contains the probabilities output by the router for each token.
    router_z_loss: Router z-loss. Encourages router logits to remain small in an
      effort to improve stability.
  """
  dispatch_mask: jax.Array
  combine_array: jax.Array
  auxiliary_loss: jax.Array
  router_probs: jax.Array | None
  router_z_loss: jax.Array


class RouterWeights(nn.Module):
  """Router module converting token inputs to router logits.

  Attributes:
    use_bias: Whether or not to use the bias term in computing the logits.
    dtype: Numerical float type for router logit computation.
    kernel_init: Initialization scheme for kernel.
    bias_init: Initialization scheme for bias.
    precision: XLA precision for array computations.
    axis: Axes along which to apply the dense router weights transformation.
      Defaults to final axis (typically the "hidden dimension").
    kernel_shardings: Sharding annotations to use for kernel sharding.
    dense_dot_general: The function that performs dot product between the
      weights and the inputs.
  """
  use_bias: bool = True
  dtype: jax.typing.DTypeLike = jnp.bfloat16
  kernel_init: nn.initializers.Initializer = default_kernel_init
  bias_init: nn.initializers.Initializer = default_bias_init
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
  axis: typing.Axes = -1
  kernel_shardings: tuple[str, Ellipsis] = ()
  dense_dot_general: typing.DotGeneral = lax.dot_general

  @nn.compact
  def __call__(self, token_inputs, num_experts):
    """Applies RouterWeights module.

    Args:
      token_inputs: Flattened batch of tokens with shape <float>[num_groups,
        group_size, hidden_dim].
      num_experts: Number of experts.

    Returns:
      Router logits with shape <float>[num_groups, group_size, num_experts].
    """
    return linear.DenseGeneral(
        features=num_experts,
        axis=self.axis,
        use_bias=self.use_bias,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
        kernel_shardings=self.kernel_shardings,
        dot_general=self.dense_dot_general,
        name='w')(token_inputs)


class NormalizedRouterWeights(nn.Module):
  """Router module converting token inputs to router logits, with unit norm.

  With the standard RouterWeights, the magnitude of the logits typically grows
  with sqrt(hidden_size). This becomes problematic when scaling the backbone
  models and hidden_size becomes too big, since the softmax applied on the
  logits collapses to all 0s except one value, and gradients vanish.

  Instead, we normalize the token inputs to have unit norm, and normalize the
  router parameters (called mu) to have unit norm as well. In addition, we add
  a scale parameter to control how spiky the output of the softmax is (i.e. we
  train the temperature of the softmax). The scale is a single scalar, so the
  logits' magnitude does not depend on the hidden_size in any way.

  Attributes:
    dtype: Numerical float type for router logit computation.
    mu_init: Initialization scheme for mu param.
    scale_init: Initialization scheme for the scale param.
    mu_shardings: Sharding annotations to use for the mu sharding.
    precision: XLA precision for array computations.
    axis: Axis along which to apply the dense router weights transformation.
      Defaults to final axis (typically the "hidden dimension").
    dense_dot_general: The function that performs dot product between the
      weights and the inputs.
  """
  dtype: jax.typing.DTypeLike = jnp.bfloat16
  mu_init: nn.initializers.Initializer = default_kernel_init
  scale_init: nn.initializers.Initializer = jax.nn.initializers.ones
  mu_shardings: typing.ShardingAxes = ()
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
  axis: int = -1
  dense_dot_general: typing.DotGeneral = lax.dot_general

  @nn.compact
  def __call__(self, token_inputs, num_slots):
    # Normalize negative axis. E.g. axis=-1 -> axis=ndim - 1.
    (axis,) = linear._normalize_axes((self.axis,), token_inputs.ndim)
    # Parameters.
    mu_shape = (token_inputs.shape[axis], num_slots)
    mu_init = sharding.modulate_param_init(self.mu_init, self.mu_shardings)
    mu = pytyping.cast(jax.Array, self.param(
        name='mu',
        init_fn=mu_init,
        shape=mu_shape,
        dtype=jnp.float32,
        unbox=True,
    ))
    scale_init = sharding.modulate_param_init(self.scale_init, ())
    scale = pytyping.cast(jax.Array, self.param(
        name='scale',
        init_fn=scale_init,
        shape=(),
        dtype=jnp.float32,
        unbox=True,
    ))
    # Normalize the token inputs and mu to have unit norm. The normalization is
    # done along the hidden_size/feature_dim axis, so that the norm of these
    # vectors does not depend on the feature dimension.
    token_inputs = _normalize(token_inputs, axis=axis)
    mu = _normalize(mu, axis=0)
    # We control the magnitude of the logits with a single scale parameter.
    # We simply pre-multiply the smallest array.
    if token_inputs.size < mu.size:
      token_inputs *= scale
    else:
      mu *= scale

    mu = mu.astype(self.dtype)
    token_inputs = token_inputs.astype(self.dtype)
    logits = self.dense_dot_general(
        token_inputs, mu, (((axis,), (0,)), ((), ())), precision=self.precision
    )
    return logits


def _normalize(
    x, axis = -1, eps = 1e-6):
  """Normalizes an array to have unit norm along the given axis."""
  # Note: this is similar to jnp.linalg.norm, but adds eps to avoid NaNs when
  # the input array is 0.
  m = jax.lax.rsqrt(jnp.square(x).sum(axis=axis, keepdims=True) + eps)
  return x * m


class Router(nn.Module):
  """Abstract base router class, defining router API and inner workings.

  Attributes:
    router_weights: Configurable module used to compute router logits from token
      inputs.
    jitter_noise: Amplitude of jitter noise applied to router logits.
    dtype: Numeric float type for returned combine array. All actual
      computations are performed in float32 of the input for stability.
    ignore_padding_tokens: Whether to ignore padding tokens during routing. Note
      that some routers (e.g. TokensChooseMaskedRouter) will completely ignore
      padding tokens, while others (e.g. TokensChooseScatterRouter and
      ExpertsChooseMaskedRouter) will simply down-weight the probability of
      selecting padding tokens.
    router_kwargs: Dictionary with further router parameters if needed.
  """
  router_weights: RouterWeights | NormalizedRouterWeights
  jitter_noise: float
  dtype: jnp.dtype
  inputs_shardings: typing.ShardingAxes
  logits_shardings: typing.ShardingAxes
  ignore_padding_tokens: bool
  router_kwargs: dict[str, Any]

  def __call__(self,
               token_inputs,
               num_experts,
               expert_capacity,
               metadata = None,
               apply_jitter = True):
    """Computes dispatch and combine arrays for routing to experts.

    Args:
      token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to
        send to experts.
      num_experts: Number of experts.
      expert_capacity: Each group will send this many tokens to each expert.
      metadata: Dict, contains batch metadata (such as modality).
      apply_jitter: If true, apply jitter noise during routing.

    Returns:
      Router indices or mask arrays (depending on router type).
    """
    token_inputs = sharding.shard_array(token_inputs, self.inputs_shardings)
    router_probs, router_logits = self._compute_router_probabilities(
        token_inputs, num_experts, apply_jitter, metadata=metadata)
    router_probs = sharding.shard_array(router_probs, self.logits_shardings)
    router_logits = sharding.shard_array(router_logits, self.logits_shardings)

    if self.ignore_padding_tokens:
      # To identify non-padding tokens, we rely on the fact that padding tokens
      # in the inputs have already been masked in the default T5 architecture.
      # See
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L315
      # and
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L603.
      padding_mask = jnp.array((jnp.sum(jnp.abs(token_inputs), axis=-1) > 0),
                               dtype=token_inputs.dtype)
      router_logits *= jnp.expand_dims(padding_mask, axis=-1)
    else:
      padding_mask = None

    instructions = self._compute_routing_instructions(router_probs,
                                                      padding_mask,
                                                      expert_capacity)

    return instructions.replace(router_z_loss=_router_z_loss(router_logits))

  def _compute_router_probabilities(
      self,
      token_inputs,
      num_experts,
      apply_jitter,
      metadata,
  ):
    """Computes router probabilities from input tokens.

    Args:
      token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] from which
        router probabilities are computed.
      num_experts: Number of experts.
      apply_jitter: If true, apply jitter noise.
      metadata: Dict, contains batch metadata (such as modality).

    Returns:
      - <float32>[num_groups, tokens_per_group, num_experts] probabilities for
        each token and expert. Used for routing tokens to experts.
      - <float>[num_groups, tokens_per_group, num_experts] raw router logits.
        Used for computing router z-loss.
    """
    # For remainder of routing computation we use float32 to ensure stability.
    # See the discussion of "selective precision" in
    # https://arxiv.org/abs/2101.03961.
    token_inputs = jax.lax.convert_element_type(token_inputs, jnp.float32)

    if apply_jitter and self.jitter_noise > 0:
      token_inputs *= jax.random.uniform(
          self.make_rng('jitter'),
          token_inputs.shape,
          token_inputs.dtype,
          minval=1.0 - self.jitter_noise,
          maxval=1.0 + self.jitter_noise)

    # TODO(b/267156503): Add a routing algorithm based on `modality`.
    modality = (metadata or {}).get('modality', None)
    logging.info('Routing: modality for the batch = %s.', modality)
    logging.info('Routing: routing params for batch = %s.', self.router_kwargs)

    # Shape: [num_groups, tokens_per_group, num_experts]
    router_logits = self.router_weights(token_inputs, num_experts)
    router_probabilities = jax.nn.softmax(router_logits, axis=-1)

    return router_probabilities, router_logits

  def _compute_routing_instructions(self, router_probs,
                                    padding_mask,
                                    expert_capacity):
    """Computes instructions for routing inputs to experts."""
    raise NotImplementedError(
        'Router is an abstract class that should be subclassed.')


class ScatterRouter(Router):
  """Abstract base router class for scatter dispatch routers.

  ScatterRouter(s) return RouterIndices containing dispatch indices and combine
  weights for sending token inputs (via scatter) and receiving outputs (via
  gather) to and from experts.

  Scatter-based routing is generally faster than masked matmul routing on CPUs
  and GPUs.
  """

  def _compute_routing_instructions(self, router_probs,
                                    padding_mask,
                                    expert_capacity):
    """Computes instructions for routing inputs to experts.

    Args:
      router_probs: <float32>[num_groups, tokens_per_group, num_experts]
        probabilities used to determine the routing of tokens to the experts.
      padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
        used to identify padding tokens that should be ignored by the router.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Router indices containing dispatch indices and combine weights.
    """
    raise NotImplementedError(
        'ScatterRouter is an abstract class that should be subclassed.')


class MaskedRouter(Router):
  """Abstract base router class for masked matmul dispatch routers.

  MaskedRouter(s) return RouterMask(s) containing a dispatch mask and combine
  array for sending and receiving (via masked matmuls) inputs and outputs to and
  from experts.

  Routing using masked matmuls is generally faster than scatter-based routing on
  TPUs.
  """

  def _compute_routing_instructions(self, router_probs,
                                    padding_mask,
                                    expert_capacity):
    """Computes masks for the top-k experts per token.

    Args:
      router_probs: <float32>[num_groups, tokens_per_group, num_experts]
        probabilities used to determine the routing of tokens to the experts.
      padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
        used to identify padding tokens that should be ignored by the router.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Router mask arrays.
    """
    raise NotImplementedError(
        'MaskedRouter is an abstract class that should be subclassed.')


class TokensChooseScatterRouter(ScatterRouter):
  """Scatter router using tokens choose top-k experts assignment.

  This router uses the same mechanism as in Switch Transformer
  (https://arxiv.org/abs/2101.03961) and V-MoE
  (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are
  sorted by router_probs and then routed to their choice of expert until the
  expert's expert_capacity is reached. There is no guarantee that each token is
  processed by an expert, or that each expert receives at least one token.

  Attributes:
    num_selected_experts: Maximum number of experts to which each token is
      routed. Tokens may be routed to fewer experts if particular experts are
      oversubscribed / reach capacity.
    batch_prioritized_routing: Whether or not to use Batch Prioritized Routing
      (BPR), originally introduced in V-MoE (https://arxiv.org/abs/2106.05974).
        With BPR, we prioritize routing those top-k tokens with the highest
        router probability, rather than simply using each tokens left-to-right
        ordering in the batch. This prioritization is important because the
        expert's have limited capacity.
  """
  num_selected_experts: int
  batch_prioritized_routing: bool

  def _compute_routing_instructions(self, router_probs,
                                    padding_mask,
                                    expert_capacity):
    """Computes dispatch indices and combine weights for the top-k experts.

    Args:
      router_probs: <float32>[num_groups, tokens_per_group, num_experts]
        probabilities used to determine the routing of tokens to the experts.
      padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
        used to identify padding tokens that should be down-weighted by the
        router.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
        Dispatch indices and combine weights for scatter/gather-based routing.
    """
    num_groups, tokens_per_group, num_experts = router_probs.shape

    if padding_mask is not None:
      # Because `expert_indices` are directly used for scatter-based routing, we
      # mask probabilities corresponding to tokens before the top-k operation.
      # Note that, unlike for mask-based tokens-choose routing, the
      # (down-weighted) padding tokens may still be selected.
      router_probs *= jnp.expand_dims(padding_mask, axis=-1)

    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [num_groups, tokens_per_group, num_selected_experts].
    combine_weights, expert_indices = _top_k(
        router_probs, k=self.num_selected_experts)

    auxiliary_loss = _load_balancing_loss(router_probs, expert_indices)

    if self.batch_prioritized_routing:
      # Sort tokens according to their routing probability per token group, so
      # that the highest probability tokens are routed first.
      token_ordering = jnp.argsort(-combine_weights[Ellipsis, 0], axis=-1)
      expert_indices = _take_along_axis(
          expert_indices, jnp.expand_dims(token_ordering, axis=-1), axis=-2)

    # Identify each token's preferred expert.
    # Make num_selected_experts the leading axis to ensure that top-1 choices
    # have priority over top-2 choices, which have priority over top-3
    # choices...
    preferred_experts = jnp.swapaxes(expert_indices, 1, 2)
    # Shape: [num_groups, num_selected_experts * tokens_per_group]
    preferred_experts = preferred_experts.reshape(num_groups, -1)

    # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
    expert_mask = jax.nn.one_hot(
        preferred_experts, num_experts, dtype=jnp.int32)

    # Experts have a fixed capacity that we cannot exceed. A token's priority
    # within the expert's buffer is given by the masked, cumulative capacity of
    # its target expert.
    # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
    token_priority = jnp.cumsum(expert_mask, axis=1) * expert_mask - 1.0
    # Shape: [num_groups, num_selected_experts, tokens_per_group, num_experts].
    token_priority = token_priority.reshape(
        (num_groups, self.num_selected_experts, -1, num_experts))
    # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
    token_priority = jnp.swapaxes(token_priority, 1, 2)
    # For each token, across all experts, select the only non-negative
    # (unmasked) priority. Shape: [num_groups, tokens_per_group,
    # num_selected_experts].
    token_priority = jnp.max(token_priority, axis=-1)

    # Return to original index shape.
    preferred_experts = preferred_experts.reshape(num_groups,
                                                  self.num_selected_experts,
                                                  tokens_per_group)
    # Shape: [num_groups, tokens_per_group, num_selected_experts]
    preferred_experts = jnp.swapaxes(preferred_experts, 1, 2)

    if self.batch_prioritized_routing:
      # Place tokens in their original ordering.
      inverse_token_ordering = jnp.argsort(token_ordering, axis=-1)  # pylint: disable=undefined-variable
      preferred_experts = _take_along_axis(
          preferred_experts,
          jnp.expand_dims(inverse_token_ordering, axis=-1),
          axis=-2)
      token_priority = _take_along_axis(
          token_priority,
          jnp.expand_dims(inverse_token_ordering, axis=-1),
          axis=-2)

    # Mask out tokens that overflow the maximum expert capacities.
    # Shape: [num_groups, tokens_per_group, num_selected_experts].
    combine_weights *= token_priority < expert_capacity

    # Expert index and priority within the expert capacity buffer.
    # Shape: [num_groups, tokens_per_group, num_selected_experts, 2].
    dispatch_indices = jnp.stack([preferred_experts, token_priority], axis=-1)

    # Return to default dtype now that router computation is complete.
    combine_weights = jax.lax.convert_element_type(combine_weights, self.dtype)
    dispatch_indices = jax.lax.convert_element_type(dispatch_indices, jnp.int32)

    # Initializing router z-loss
    router_z_loss = jnp.zeros((), dtype=jnp.float32)

    return RouterIndices(dispatch_indices=dispatch_indices,
                         combine_weights=combine_weights,
                         auxiliary_loss=auxiliary_loss,
                         router_probs=router_probs,
                         router_z_loss=router_z_loss)


class TokensChooseMaskedRouter(MaskedRouter):
  """Masked matmul router using tokens choose top-k experts assignment.

  This router uses the same mechanism as in Switch Transformer
  (https://arxiv.org/abs/2101.03961) and V-MoE
  (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are
  sorted by router_probs and then routed to their choice of expert until the
  expert's expert_capacity is reached. There is no guarantee that each token is
  processed by an expert, or that each expert receives at least one token.

  Attributes:
    num_selected_experts: Maximum number of experts to which each token is
      routed. Tokens may be routed to fewer experts if particular experts are
      oversubscribed / reach capacity.
    batch_prioritized_routing: Whether or not to use Batch Prioritized Routing
      (BPR), originally introduced in V-MoE (https://arxiv.org/abs/2106.05974).
        With BPR, we prioritize routing those top-k tokens with the highest
        router probability, rather than simply using each tokens left-to-right
        ordering in the batch. This prioritization is important because the
        experts have limited capacity.
    einsum_dot_general: The function that performs dot product in the einsum.
  """
  num_selected_experts: int
  batch_prioritized_routing: bool
  einsum_dot_general: typing.DotGeneral = lax.dot_general

  def _compute_routing_instructions(self, router_probs,
                                    padding_mask,
                                    expert_capacity):
    """Computes masks for the top-k experts per token.

    Args:
      router_probs: <float32>[num_groups, tokens_per_group, num_experts]
        probabilities used to determine the routing of tokens to the experts.
      padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
        used to identify padding tokens that should be ignored by the router.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
        Dispatch and combine arrays for routing with masked matmuls.
    """
    num_groups, _, num_experts = router_probs.shape

    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [num_groups, tokens_per_group, num_selected_experts].
    expert_gate, expert_index = _top_k(
        router_probs, k=self.num_selected_experts)

    if padding_mask is not None:
      # Mask applied to gate. Exclude choices corresponding to padding tokens.
      gate_mask = jnp.expand_dims(padding_mask, axis=-1)
      expert_gate *= gate_mask

      # Set `expert_index` elements corresponding to padding to negative
      # numbers. Negative `expert_index` elements will ultimately be dropped in
      # the one_hot conversion to the `expert_mask`.
      # First convert nonzero padding elements to negative values.
      expert_index *= 2 * gate_mask - 1.
      # Handle zero padding elements by negatively shifting all padding.
      expert_index += jnp.repeat(
          gate_mask - 1., self.num_selected_experts, axis=-1)

      # To correctly compute load balancing loss, we also mask out probs.
      router_probs *= gate_mask

    auxiliary_loss = _load_balancing_loss(router_probs, expert_index)

    if self.batch_prioritized_routing:
      # Sort tokens according to their routing probability per group, so that
      # the highest probability tokens are routed first.
      permutation = jnp.argsort(-expert_gate[Ellipsis, 0], axis=-1)
      # Shape: [num_groups, tokens_per_group, num_selected_experts]
      expert_index = _take_along_axis(
          expert_index, jnp.expand_dims(permutation, axis=-1), axis=-2)

    # Make num_selected_experts the leading axis to ensure that top-1 choices
    # have priority over top-2 choices, which have priority over top-3 choices,
    # etc.
    expert_index = jnp.swapaxes(expert_index, 1, 2)
    # Shape: [num_groups, num_selected_experts * tokens_per_group]
    expert_index = expert_index.reshape(num_groups, -1)

    # Create mask out of indices.
    # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
    expert_mask = jax.nn.one_hot(expert_index, num_experts, dtype=jnp.int32)

    # Experts have a fixed capacity that we cannot exceed. A token's priority
    # within the expert's buffer is given by the masked, cumulative capacity of
    # its target expert.
    # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
    token_priority = jnp.cumsum(expert_mask, axis=1) * expert_mask - 1.0
    # Shape: [num_groups, num_selected_experts, tokens_per_group, num_experts].
    token_priority = token_priority.reshape(
        (num_groups, self.num_selected_experts, -1, num_experts))
    # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
    token_priority = jnp.swapaxes(token_priority, 1, 2)
    # For each token, across all selected experts, select the only non-negative
    # (unmasked) priority. Now, for group G routing to expert E, token T has
    # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
    # is its targeted expert.
    # Shape: [num_groups, tokens_per_group, num_experts].
    token_priority = jnp.max(token_priority, axis=2)

    if self.batch_prioritized_routing:
      # Place token priorities in original ordering of tokens.
      inv_permutation = jnp.argsort(permutation, axis=-1)  # pylint: disable=undefined-variable
      token_priority = _take_along_axis(
          token_priority, jnp.expand_dims(inv_permutation, axis=-1), axis=-2)

    # Token T can only be routed to expert E if its priority is positive and
    # less than the expert capacity. One-hot matrix will ignore indices outside
    # the range [0, expert_capacity).
    # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity].
    dispatch_mask = jax.nn.one_hot(
        token_priority, expert_capacity, dtype=jnp.bool_)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
    # expert_capacity].
    combine_array = jnp.einsum(
        '...te,...tec->...tec',
        router_probs,
        dispatch_mask,
        precision=jax.lax.Precision.DEFAULT,
        _dot_general=self.einsum_dot_general)

    # Return to default dtype now that router computation is complete.
    combine_array = jax.lax.convert_element_type(combine_array, self.dtype)

    # Initializing router z-loss
    router_z_loss = jnp.zeros((), dtype=jnp.float32)

    return RouterMask(dispatch_mask=dispatch_mask,
                      combine_array=combine_array,
                      auxiliary_loss=auxiliary_loss,
                      router_probs=router_probs,
                      router_z_loss=router_z_loss)


class ExpertsChooseMaskedRouter(MaskedRouter):
  """Masked matmul router using experts choose tokens assignment.

  This router uses the same mechanism as in Mixture-of-Experts with Expert
  Choice (https://arxiv.org/abs/2202.09368): each expert selects its top
  expert_capacity tokens. An individual token may be processed by multiple
  experts or none at all.

  Note: "experts choose routing" should not be used in decoder blocks because it
  breaks the autoregressive behavior -- the model will learn to cheat by using
  future token information to improve current token predictions.

  Attributes:
    einsum_dot_general: The function that performs dot product in the einsum.
  """

  einsum_dot_general: typing.DotGeneral = lax.dot_general

  def _compute_routing_instructions(self, router_probs,
                                    padding_mask,
                                    expert_capacity):
    """Computes masks for the highest probability token per expert.

    Args:
      router_probs: <float32>[num_groups, tokens_per_group, num_experts]
        probabilities used to determine the routing of tokens to the experts.
      padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
        used to identify padding tokens that should be down-weighted by the
        router.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
        Dispatch and combine arrays for routing with masked matmuls.
    """
    tokens_per_group = router_probs.shape[1]

    if padding_mask is not None:
      # Because experts choose tokens, we mask probabilities corresponding to
      # tokens before the top-k operation. Note that, unlike for masked-based
      # tokens-choose routing, the experts here may still choose to select the
      # (down-weighted) padding tokens.
      router_probs *= jnp.expand_dims(padding_mask, axis=-1)

    # vmap over group dimension.
    router_probs_t = jax.vmap(lambda m: m.transpose())(router_probs)

    # Top expert_capacity router probability and corresponding token indices for
    # each expert. Shapes: [num_groups, num_experts, expert_capacity].
    expert_gate, expert_index = _top_k(router_probs_t, k=expert_capacity)

    # Convert to one-hot mask of expert indices for each token in each group.
    # Shape: [num_groups, num_experts, expert_capacity, tokens_per_group].
    dispatch_mask = jax.nn.one_hot(
        expert_index, tokens_per_group, dtype=jnp.int32)

    # Move axes to conform with shape expected by MoeLayer API.
    # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
    dispatch_mask = jnp.moveaxis(dispatch_mask, 3, 1)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [num_groups, num_experts, tokens_per_group,
    # expert_capacity].
    combine_array = jnp.einsum(
        '...ec,...tec->...tec',
        expert_gate,
        dispatch_mask,
        precision=jax.lax.Precision.DEFAULT,
        _dot_general=self.einsum_dot_general)

    # Return to default dtype now that router computation is complete.
    combine_array = jax.lax.convert_element_type(combine_array, self.dtype)

    # Each expert is choosing tokens until it reaches full capacity, so we don't
    # need an auxiliary loading balancing loss for expert choice routing.
    auxiliary_loss = jnp.zeros((), dtype=jnp.float32)

    # Initializing router z-loss
    router_z_loss = jnp.zeros((), dtype=jnp.float32)

    return RouterMask(dispatch_mask=dispatch_mask,
                      combine_array=combine_array,
                      auxiliary_loss=auxiliary_loss,
                      router_probs=router_probs,
                      router_z_loss=router_z_loss)


class SoftRouter(Router):
  """Router used in SoftMoE layers."""

  @nn.compact
  def __call__(self,
               token_inputs,
               num_experts,
               expert_capacity,
               metadata = None,
               apply_jitter = True):
    token_inputs = sharding.shard_array(token_inputs, self.inputs_shardings)
    # Note: we have expert_capacity vector parameters per expert, instead of a
    # single vector parameter per expert.
    _, router_logits = self._compute_router_probabilities(
        token_inputs, num_experts * expert_capacity, apply_jitter,
        metadata=metadata)
    router_logits = sharding.shard_array(router_logits, self.logits_shardings)

    if self.ignore_padding_tokens:
      # To identify non-padding tokens, we rely on the fact that padding tokens
      # in the inputs have already been masked in the default T5 architecture.
      # See
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L315
      # and
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L603.
      padding_mask = jnp.sum(jnp.abs(token_inputs), axis=-1, keepdims=True) > 0
      padding_bias = jax.lax.select(padding_mask,
                                    jnp.zeros_like(router_logits),
                                    jnp.full_like(router_logits, -1e10))
    else:
      padding_mask, padding_bias = 1, 0

    # Shape is [num_groups, group_size, num_experts * expert_capacity].
    dispatch_weights = jax.nn.softmax(router_logits + padding_bias, axis=1)
    combine_weights = jax.nn.softmax(router_logits + padding_bias, axis=-1)
    combine_weights *= padding_mask
    # Reshape to the expected shape by RouterMask:
    # [num_groups, group_size, num_experts, expert_capacity].
    new_shape = router_logits.shape[:-1] + (num_experts, expert_capacity)
    dispatch_weights = dispatch_weights.reshape(new_shape)
    combine_weights = combine_weights.reshape(new_shape)

    return RouterMask(dispatch_mask=dispatch_weights,
                      combine_array=combine_weights,
                      auxiliary_loss=jnp.zeros((), dtype=jnp.float32),
                      router_probs=None,
                      router_z_loss=jnp.zeros((), dtype=jnp.float32))
