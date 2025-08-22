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

from typing import Any, Callable, Sequence, Tuple

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

# Type Stubs
PRNGKey = Any
RouterOutput = Any
Shape = Sequence[int]

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
    dispatch_indices: <int32>[NUM_GROUPS, TOKENS_PER_GROUP,
      NUM_SELECTED_EXPERTS, 2] dispatch indices indicating, for each token, its
      preferred expert and its priority in that expert's buffer.
    combine_weights: <float>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS]
      combine weights used for scaling expert outputs with the router's dispatch
      probability/confidence.
    auxiliary_loss: Load balancing loss for router.
    router_z_loss: Router z-loss. Encourages router logits to remain small in an
      effort to improve stability.
  """
  dispatch_indices: jnp.ndarray
  combine_weights: jnp.ndarray
  auxiliary_loss: float
  router_z_loss: float = 0.


@flax.struct.dataclass
class RouterMask:
  """Dispatch and combine arrays for expert routing with masked matmuls.

  Attributes:
    dispatch_mask: <bool>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS,
      EXPERT_CAPACITY] dispatch array that is 1 if the token gets routed to the
      corresponding expert, and 0 otherwise.
    combine_array: <float>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS,
      EXPERT_CAPACITY] combine array used for combining expert outputs and
      scaling with router probability.
    auxiliary_loss: Load balancing loss for router.
    router_z_loss: Router z-loss. Encourages router logits to remain small in an
      effort to improve stability.
  """
  dispatch_mask: jnp.ndarray
  combine_array: jnp.ndarray
  auxiliary_loss: float
  router_z_loss: float = 0.


def _favor_one_hot_slices():
  """Returns true iff running on TPUs."""
  return jax.default_backend() == "tpu" or jax.devices()[0].platform == "tpu"


def _take_along_axis(array, indices,
                     axis):
  """Takes values from the input array by matching 1D index and data slices.

  This function serves the same purpose as jax.numpy.take_along_axis, except
  that it uses one-hot matrix multiplications under the hood on TPUs:
  (1) On TPUs, we use one-hot matrix multiplications to select elements from the
      array.
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
        "indices and array must have the same number of dimensions; "
        f"{indices.ndim} vs. {array.ndim}.")

  if (axis != -1 and axis != array.ndim - 1 and  # Not last dimension
      axis != 1 and axis != -array.ndim + 1):  # Not second dimension
    raise ValueError(
        "Only slices along the second or last dimension are supported; "
        f"array.ndim = {array.ndim}, while axis = {axis}.")

  if _favor_one_hot_slices():
    one_hot_length = array.shape[axis]
    one_hot_indices = jax.nn.one_hot(indices, one_hot_length, axis=axis)

    if axis == -1 or array.ndim == 1:
      # Take i elements from last dimension (s).
      # We must use HIGHEST precision to accurately reproduce indexing
      # operations with matrix multiplications.
      result = jnp.einsum(
          "...s,...is->...i",
          array,
          one_hot_indices,
          precision=jax.lax.Precision.HIGHEST)
    else:
      # Take i elements from second dimension (s). We assume here that we always
      # want to slice along the second dimension.
      # We must use HIGHEST precision to accurately reproduce indexing
      # operations with matrix multiplications.
      result = jnp.einsum(
          "ns...,nis...->ni...",
          array,
          one_hot_indices,
          precision=jax.lax.Precision.HIGHEST)
    return jax.lax.convert_element_type(result, array.dtype)
  else:
    return jnp.take_along_axis(array, indices, axis=axis)


def _top_k(array, k):
  """Returns top k values and their indices along the last axis of the array.

  This function serves the same purpose as jax.lax.top_k, but in a more XLA
  friendly manner for TPUs:
  (1) On TPUs, we use one-hot matrix multiplications to select the top k values.
      This convoluted way of obtaining the top k values is generally faster on
      TPUs.
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


class RouterWeights(nn.Module):
  """Router module converting token inputs to router logits.

  Attributes:
    use_bias: Whether or not to use the bias term in computing the logits.
    dtype: Numerical float type for router logit computation.
    kernel_init: Initialization scheme for kernel.
    bias_init: Initialization scheme for bias.
    precision: XLA precision for array computations.
  """
  use_bias: bool = True
  dtype: jnp.dtype = jnp.bfloat16
  kernel_init: Callable[[PRNGKey, Shape, jnp.dtype],
                        jnp.ndarray] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, jnp.dtype],
                      jnp.ndarray] = default_bias_init
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self, token_inputs,
               num_experts):
    """Applies RouterWeights module.

    Args:
      token_inputs: Flattened batch of tokens with shape <float>[NUM_GROUPS,
        TOKENS_PER_GROUP, HIDDEN_DIM].
      num_experts: Number of experts.

    Returns:
      Router logits with shape <float>[NUM_GROUPS, TOKENS_PER_GROUP,
      NUM_EXPERTS].
    """
    return nn.DenseGeneral(
        num_experts,
        use_bias=self.use_bias,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            token_inputs)


class Router(nn.Module):
  """Abstract base router class, defining router API and inner workings.

  Attributes:
    router_weights: Configurable module used to compute router logits from token
      inputs.
    jitter_noise: Amplitude of jitter noise applied to router logits.
    dtype: Numeric float type for returned combine array. All actual
      computations are performed in float32 of the input for stability.
  """
  router_weights: RouterWeights
  jitter_noise: float
  dtype: jnp.dtype

  def __call__(self,
               token_inputs,
               num_experts,
               expert_capacity,
               apply_jitter = True):
    """Computes dispatch and combine arrays for routing to experts.

    Args:
      token_inputs: <float>[NUM_GROUPS, TOKENS_PER_GROUP, HIDDEN_DIM] inputs to
        send to experts.
      num_experts: Number of experts.
      expert_capacity: Each group will send this many tokens to each expert.
      apply_jitter: If true, apply jitter noise during routing.

    Returns:
      Router indices or mask arrays (depending on router type).
    """
    router_probs, router_logits = self._compute_router_probabilities(
        token_inputs, num_experts, apply_jitter)
    instructions = self._compute_routing_instructions(router_probs,
                                                      expert_capacity)
    return instructions.replace(router_z_loss=_router_z_loss(router_logits))

  def _compute_router_probabilities(
      self, token_inputs, num_experts,
      apply_jitter):
    """Computes router probabilities from input tokens.

    Args:
      token_inputs: <float>[NUM_GROUPS, TOKENS_PER_GROUP, HIDDEN_DIM] from which
        router probabilities are computed.
      num_experts: Number of experts.
      apply_jitter: If true, apply jitter noise.

    Returns:
      - <float32>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS] probabilities for
        each token and expert. Used for routing tokens to experts.
      - <float>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS] raw router logits.
        Used for computing router z-loss.
    """
    # For remainder of routing computation we use float32 to ensure stability.
    # See the discussion of "selective precision" in
    # https://arxiv.org/abs/2101.03961.
    token_inputs = jax.lax.convert_element_type(token_inputs, jnp.float32)

    if apply_jitter and self.jitter_noise > 0:
      token_inputs *= jax.random.uniform(
          self.make_rng("jitter"),
          token_inputs.shape,
          token_inputs.dtype,
          minval=1.0 - self.jitter_noise,
          maxval=1.0 + self.jitter_noise)

    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
    router_logits = self.router_weights(token_inputs, num_experts)

    router_probabilities = jax.nn.softmax(router_logits, axis=-1)

    return router_probabilities, router_logits

  def _compute_routing_instructions(self, router_probs,
                                    expert_capacity):
    """Computes instructions for routing inputs to experts."""
    raise NotImplementedError(
        "Router is an abstract class that should be subclassed.")


class ScatterRouter(Router):
  """Abstract base router class for scatter dispatch routers.

  ScatterRouter(s) return RouterIndices containing dispatch indices and combine
  weights for sending token inputs (via scatter) and receiving outputs (via
  gather) to and from experts.

  Scatter-based routing is generally faster than masked matmul routing on CPUs
  and GPUs.
  """

  def _compute_routing_instructions(self, router_probs,
                                    expert_capacity):
    """Computes instructions for routing inputs to experts.

    Args:
      router_probs: <float32>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
        probabilities used to determine the routing of tokens to the experts.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Router indices containing dispatch indices and combine weights.
    """
    raise NotImplementedError(
        "ScatterRouter is an abstract class that should be subclassed.")


class MaskedRouter(Router):
  """Abstract base router class for masked matmul dispatch routers.

  MaskedRouter(s) return RouterMask(s) containing a dispatch mask and combine
  array for sending and receiving (via masked matmuls) inputs and outputs to and
  from experts.

  Routing using masked matmuls is generally faster than scatter-based routing on
  CPUs and GPUs.
  """

  def _compute_routing_instructions(self, router_probs,
                                    expert_capacity):
    """Computes masks for the top-k experts per token.

    Args:
      router_probs: <float32>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
        probabilities used to determine the routing of tokens to the experts.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Router mask arrays.
    """
    raise NotImplementedError(
        "MaskedRouter is an abstract class that should be subclassed.")


class TokensChooseScatterRouter(ScatterRouter):
  """Scatter router using tokens choose top-k experts assignment.

  This router uses the same mechanism as in Switch Transformer
  (https://arxiv.org/abs/2101.03961): tokens choose their top experts. Items are
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
  """
  num_selected_experts: int
  batch_prioritized_routing: bool

  def _compute_routing_instructions(self, router_probs,
                                    expert_capacity):
    """Computes dispatch indices and combine weights for the top-k experts.

    Args:
      router_probs: <float32>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
        probabilities used to determine the routing of tokens to the experts.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Dispatch indices and combine weights for scatter/gather-based routing.
    """
    num_groups, tokens_per_group, num_experts = router_probs.shape

    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS].
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
    # Make NUM_SELECTED_EXPERTS the leading axis to ensure that top-1 choices
    # have priority over top-2 choices, which have priority over top-3
    # choices...
    preferred_experts = jnp.swapaxes(expert_indices, 1, 2)
    # Shape: [NUM_GROUPS, NUM_SELECTED_EXPERTS * TOKENS_PER_GROUP]
    preferred_experts = preferred_experts.reshape(num_groups, -1)

    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP * NUM_SELECTED_EXPERTS, NUM_EXPERTS].
    expert_mask = jax.nn.one_hot(
        preferred_experts, num_experts, dtype=jnp.int32)

    # Experts have a fixed capacity that we cannot exceed. A token's priority
    # within the expert's buffer is given by the masked, cumulative capacity of
    # its target expert.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP * NUM_SELECTED_EXPERTS, NUM_EXPERTS].
    token_priority = jnp.cumsum(expert_mask, axis=1) * expert_mask - 1.0
    # Shape: [NUM_GROUPS, NUM_SELECTED_EXPERTS, TOKENS_PER_GROUP, NUM_EXPERTS].
    token_priority = token_priority.reshape(
        (num_groups, self.num_selected_experts, -1, num_experts))
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS, NUM_EXPERTS].
    token_priority = jnp.swapaxes(token_priority, 1, 2)
    # For each token, across all experts, select the only non-negative
    # (unmasked) priority.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS].
    token_priority = jnp.max(token_priority, axis=-1)

    # Return to original index shape.
    preferred_experts = preferred_experts.reshape(num_groups,
                                                  self.num_selected_experts,
                                                  tokens_per_group)
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS]
    preferred_experts = jnp.swapaxes(preferred_experts, 1, 2)

    if self.batch_prioritized_routing:
      # Place tokens in their original ordering.
      inverse_token_ordering = jnp.argsort(token_ordering, axis=-1)
      preferred_experts = _take_along_axis(
          preferred_experts,
          jnp.expand_dims(inverse_token_ordering, axis=-1),
          axis=-2)
      token_priority = _take_along_axis(
          token_priority,
          jnp.expand_dims(inverse_token_ordering, axis=-1),
          axis=-2)

    # Mask out tokens that overflow the maximum expert capacities.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS].
    combine_weights *= token_priority < expert_capacity

    # Expert index and priority within the expert capacity buffer.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS, 2].
    dispatch_indices = jnp.stack([preferred_experts, token_priority], axis=-1)

    # Return to default dtype now that router computation is complete.
    combine_weights = jax.lax.convert_element_type(combine_weights, self.dtype)
    dispatch_indices = jax.lax.convert_element_type(dispatch_indices, jnp.int32)

    return RouterIndices(dispatch_indices, combine_weights, auxiliary_loss)


class TokensChooseMaskedRouter(MaskedRouter):
  """Masked matmul router using tokens choose top-k experts assignment.

  This router uses the same mechanism as in Switch Transformer
  (https://arxiv.org/abs/2101.03961): tokens choose their top experts. Items are
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
                                    expert_capacity):
    """Computes masks for the top-k experts per token.

    Args:
      router_probs: <float32>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
        probabilities used to determine the routing of tokens to the experts.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Dispatch and combine arrays for routing with masked matmuls.
    """
    num_groups, _, num_experts = router_probs.shape

    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS].
    expert_gate, expert_index = _top_k(
        router_probs, k=self.num_selected_experts)

    auxiliary_loss = _load_balancing_loss(router_probs, expert_index)

    if self.batch_prioritized_routing:
      # Sort tokens according to their routing probability per group, so that
      # the highest probability tokens are routed first.
      permutation = jnp.argsort(-expert_gate[Ellipsis, 0], axis=-1)
      # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS]
      expert_index = _take_along_axis(
          expert_index, jnp.expand_dims(permutation, axis=-1), axis=-2)

    # Make NUM_SELECTED_EXPERTS the leading axis to ensure that top-1 choices
    # have priority over top-2 choices, which have priority over top-3 choices,
    # etc.
    expert_index = jnp.swapaxes(expert_index, 1, 2)
    # Shape: [NUM_GROUPS, NUM_SELECTED_EXPERTS * TOKENS_PER_GROUP]
    expert_index = expert_index.reshape(num_groups, -1)

    # Create mask out of indices.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP * NUM_SELECTED_EXPERTS, NUM_EXPERTS].
    expert_mask = jax.nn.one_hot(expert_index, num_experts, dtype=jnp.int32)

    # Experts have a fixed capacity that we cannot exceed. A token's priority
    # within the expert's buffer is given by the masked, cumulative capacity of
    # its target expert.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP * NUM_SELECTED_EXPERTS, NUM_EXPERTS].
    token_priority = jnp.cumsum(expert_mask, axis=1) * expert_mask - 1.0
    # Shape: [NUM_GROUPS, NUM_SELECTED_EXPERTS, TOKENS_PER_GROUP, NUM_EXPERTS].
    token_priority = token_priority.reshape(
        (num_groups, self.num_selected_experts, -1, num_experts))
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS, NUM_EXPERTS].
    token_priority = jnp.swapaxes(token_priority, 1, 2)
    # For each token, across all selected experts, select the only non-negative
    # (unmasked) priority. Now, for group G routing to expert E, token T has
    # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
    # is its targeted expert.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS].
    token_priority = jnp.max(token_priority, axis=2)

    if self.batch_prioritized_routing:
      # Place token priorities in original ordering of tokens.
      inv_permutation = jnp.argsort(permutation, axis=-1)
      token_priority = _take_along_axis(
          token_priority, jnp.expand_dims(inv_permutation, axis=-1), axis=-2)

    # Token T can only be routed to expert E if its priority is positive and
    # less than the expert capacity. One-hot matrix will ignore indices outside
    # the range [0, EXPERT_CAPACITY).
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS, EXPERT_CAPACITY].
    dispatch_mask = jax.nn.one_hot(
        token_priority, expert_capacity, dtype=jnp.bool_)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS,
    # EXPERT_CAPACITY].
    combine_array = jnp.einsum(
        "...te,...tec->...tec",
        router_probs,
        dispatch_mask,
        precision=jax.lax.Precision.DEFAULT)

    # Return to default dtype now that router computation is complete.
    combine_array = jax.lax.convert_element_type(combine_array, self.dtype)

    return RouterMask(dispatch_mask, combine_array, auxiliary_loss)


class ExpertsChooseMaskedRouter(MaskedRouter):
  """Masked matmul router using experts choose tokens assignment.

  This router uses the same mechanism as in Mixture-of-Experts with Expert
  Choice (https://arxiv.org/abs/2202.09368): each expert selects its top
  expert_capacity tokens. An individual token may be processed by multiple
  experts or none at all.
  """

  def _compute_routing_instructions(self, router_probs,
                                    expert_capacity):
    """Computes masks for the highest probability token per expert.

    Args:
      router_probs: <float32>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
        probabilities used to determine the routing of tokens to the experts.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      Dispatch and combine arrays for routing with masked matmuls.
    """
    tokens_per_group = router_probs.shape[1]

    # vmap over group dimension.
    router_probs_t = jax.vmap(lambda m: m.transpose())(router_probs)

    # Top expert_capacity router probability and corresponding token indices for
    # each expert. Shapes: [NUM_GROUPS, NUM_EXPERTS, EXPERT_CAPACITY].
    expert_gate, expert_index = _top_k(router_probs_t, k=expert_capacity)

    # Convert to one-hot mask of expert indices for each token in each group.
    # Shape: [NUM_GROUPS, NUM_EXPERTS, EXPERT_CAPACITY, TOKENS_PER_GROUP].
    dispatch_mask = jax.nn.one_hot(
        expert_index, tokens_per_group, dtype=jnp.int32)

    # Move axes to conform with shape expected by MoeLayer API.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS, EXPERT_CAPACITY]
    dispatch_mask = jnp.moveaxis(dispatch_mask, 3, 1)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [NUM_GROUPS, NUM_EXPERTS, TOKENS_PER_GROUP,
    # EXPERT_CAPACITY].
    combine_array = jnp.einsum(
        "...ec,...tec->...tec",
        expert_gate,
        dispatch_mask,
        precision=jax.lax.Precision.DEFAULT)

    # Return to default dtype now that router computation is complete.
    combine_array = jax.lax.convert_element_type(combine_array, self.dtype)

    # Each expert is choosing tokens until it reaches full capacity, so we don't
    # need an auxiliary loading balancing loss for expert choice routing.
    auxiliary_loss = 0.0

    return RouterMask(dispatch_mask, combine_array, auxiliary_loss)


def _load_balancing_loss(router_probs,
                         expert_indices):
  """Computes auxiliary load balancing loss as in Switch Transformer.

  See Switch Transformer (https://arxiv.org/abs/2101.03961). This function
  implements the loss function presented in equations (4) - (6). It aims to
  penalize those cases where the routing between experts is unbalanced.

  Args:
    router_probs: Probability assigned to each expert per token. Shape:
      <float32>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS].
    expert_indices: <int>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS]
      indices identifying the top NUM_SELECTED_EXPERTS for a given token.

  Returns:
    The auxiliary loss.
  """
  num_experts = router_probs.shape[-1]

  # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS, NUM_EXPERTS].
  expert_mask = jax.nn.one_hot(expert_indices, num_experts, dtype=jnp.int32)
  # For a given token, determine if it was routed to a given expert.
  # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
  expert_mask = jnp.max(expert_mask, axis=-2)

  tokens_per_group_and_expert = jnp.mean(
      expert_mask, dtype=jnp.float32, axis=-2)
  router_prob_per_group_and_expert = jnp.mean(
      router_probs, dtype=jnp.float32, axis=-2)
  return jnp.mean(  # pytype: disable=bad-return-type  # jnp-type
      tokens_per_group_and_expert * router_prob_per_group_and_expert,
      dtype=jnp.float32) * num_experts**2


def _router_z_loss(router_logits):
  """Compute router z-loss.

   The router z-loss was introduced in Designing Effective Sparse Expert Models
   (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
   small in an effort to improve stability.

  Args:
    router_logits: <float>[NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS] router
      logits.

  Returns:
    Scalar router z-loss.
  """
  num_groups, tokens_per_group, _ = router_logits.shape
  log_z = jax.nn.logsumexp(router_logits, axis=-1)
  z_loss = log_z**2
  return jnp.sum(z_loss, dtype=jnp.float32) / (num_groups * tokens_per_group)  # pytype: disable=bad-return-type  # jnp-type
