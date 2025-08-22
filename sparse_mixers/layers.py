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

"""Model layers."""

import functools
import math
from typing import Any, Callable, Optional, Sequence

from absl import logging
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scipy import linalg

from sparse_mixers import core_utils
from sparse_mixers import routing

# Type Stubs
PRNGKey = Any
Shape = Sequence[int]

default_kernel_init = nn.initializers.normal(stddev=2e-2)
default_bias_init = nn.initializers.zeros

LAYER_NORM_EPSILON = 1e-12


@flax.struct.dataclass
class DiversityMetrics:
  """Metrics for analyzing diversity among experts in mixture of experts models.

  Attributes:
    auxiliary_loss: Auxiliary load balancing loss.
    router_z_loss: Router z-loss. Encourages router logits to remain small in an
      effort to improve stability.
    fraction_tokens_left_behind: Fraction of tokens NOT processed by any expert.
    expert_usage: Fraction of total capacity, across all experts, used to
      process tokens. Larger expert capacities or non-uniform token routing will
      result in smaller expert usage values.
    router_confidence: How confident the router is about the tokens that it has
      routed.
  """
  auxiliary_loss: float
  router_z_loss: float

  fraction_tokens_left_behind: float
  expert_usage: float
  router_confidence: float

  def __add__(self, other):
    return DiversityMetrics(
        self.auxiliary_loss + other.auxiliary_loss,
        self.router_z_loss + other.router_z_loss,
        self.fraction_tokens_left_behind + other.fraction_tokens_left_behind,
        self.expert_usage + other.expert_usage,
        self.router_confidence + other.router_confidence,
    )


class GeneralizedFeedForwardLayer(nn.Module):
  """Base class for feed-forward layers.

  This class cannot be used directly.
  """

  @nn.compact
  def __call__(self,
               input_emb,
               deterministic = False):
    """Applies GeneralizedFeedForwardLayer module.

    Args:
      input_emb: Batch of input embeddings, typically of shape
        <float>[BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM].
      deterministic: Whether or not to apply dropout and jitter noise to input.

    Returns:
      MLP-transformed inputs along last dimension of the input, typically with
      the same shape and type as inputs.
    """
    raise NotImplementedError


class MixingLayer(nn.Module):
  """Base class for mixing layers.

  This class cannot be used directly.
  """

  @nn.compact
  def __call__(self,
               input_emb,
               deterministic = False):
    """Applies MixingLayer module.

    Args:
      input_emb: Batch of input embeddings, typically of shape
        <float>[BATCH_SIZE, INPUT_SEQ_LENGTH, HIDDEN_DIM].
      deterministic: Whether to apply dropout to input.

    Returns:
      Transformed inputs, typically with the same shape and type as the input
        data.
    """
    raise NotImplementedError


class FeedForwardLayer(GeneralizedFeedForwardLayer):
  """Position independent, MLP layer.

  Attributes:
    d_ff: Dimension of feed-forward layer.
    dropout_rate: The dropout probability.
    dtype: The numeric type of the computation (default: float32).
    intermediate_activation: (Nonlinear) transform applied in layer.
    kernel_init: Initialization scheme for kernel.
    bias_init: Initialization scheme for bias.
    precision: XLA precision for array computations.
  """
  d_ff: int
  dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32
  intermediate_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  kernel_init: Callable[[PRNGKey, Shape, jnp.dtype],
                        jnp.ndarray] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, jnp.dtype],
                      jnp.ndarray] = default_bias_init
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self,
               input_emb,
               deterministic = False):
    """Applies FeedForwardLayer module.

    Args:
      input_emb: Batch of input embeddings, typically of shape
        <float>[BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM].
      deterministic: Whether to apply dropout to input.

    Returns:
      MLP-transformed inputs with same shape as original inputs:
      <float>[BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM].
    """
    d_model = input_emb.shape[-1]
    x = nn.DenseGeneral(
        self.d_ff,
        use_bias=True,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
        name="intermediate")(
            input_emb)
    x = self.intermediate_activation(x)

    x = nn.DenseGeneral(
        d_model,
        use_bias=True,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name="output",
        precision=self.precision)(
            x)
    return nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)


def _favor_one_hot_slices():
  """Returns true iff running on TPUs."""
  return jax.default_backend() == "tpu" or jax.devices()[0].platform == "tpu"


def truncated_dtype():
  """Returns platform specific truncated float type."""
  platform = jax.local_devices()[0].platform
  if platform == "tpu":
    return jnp.bfloat16
  elif platform == "gpu":
    return jnp.float16
  return jnp.float32


class MoeLayer(GeneralizedFeedForwardLayer):
  """Sparse MoE SPMD layer with per-token routing.

  Under pmap, we locally manipulate arrays and handle the all-to-all device
  communications manually. The axis_name attribute, should match the axis name
  used to shard data in pmap.

  Attributes:
    num_experts: Number of available experts (feed-forward modules) in this
      layer.
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
    train_capacity_factor: Scaling factor to increase the expert token capacity
      during training. This factor plays an analogous, but slightly different,
      role depending on the routing assignment algorithm:
      - For "tokens choose" routing, the capacity factor only affects the
        maximum number of tokens that an expert will process. It does not affect
        how many experts a given token is routed to; see the
        num_selected_experts attributes of "tokens choose" routers.
      - For "experts choose" routing, because experts always fill their buffer,
        increasing the capacity factor will increase the number of tokens that
        an expert will process AND will indirectly increase the number of
        experts that a given token is routed to.
    eval_capacity_factor: As above, but used during evaluation.
    expert: The actual expert, currently constrained to be a
     GeneralizedFeedForwardLayer.
    router: Token dispatch router. The router determines which tokens are
      dispatched to which expert, and how the expert outputs are combined.
    min_expert_capacity: Minimum token processing capacity for each expert.
    dropout_rate: Dropout rate for each expert.
    dtype: The numeric type (default: bfloat16). We recommend a truncated float
      type (e.g. bfloat16) to reduce all-to-all communication overhead. This
      numeric type is used for all computations, except the router, which always
      uses float32 for stability.
    axis_name: Axis name used by JAX for SPMD under pmap.  Should match the axis
      name used in jax.pmap.
    split_params: Whether to initialize each expert's parameters independently.
    precision: XLA precision for array computations.
  """
  num_experts: int
  max_group_size: int
  train_capacity_factor: float
  eval_capacity_factor: float
  expert: GeneralizedFeedForwardLayer
  router: routing.Router
  min_expert_capacity: int = 4
  dropout_rate: float = 0.1
  dtype: jnp.dtype = jnp.float32
  axis_name: str = "batch"  # Data and expert parallelism
  split_params: bool = True
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  @property
  def _is_parallel_computation(self):
    """Returns true if layer is executed in parallel (within a pmap).

    AllToAll device communication can only be used to distribute data among
    devices during SPMD mode. In particular, the Flax nn.Module initialization
    does NOT run in parallel, but we still want the single device initialization
    to trace the rest of the layer.
    """
    try:
      _ = jax.lax.axis_index(self.axis_name)
      return True
    except NameError:
      return False

  @nn.compact
  def __call__(self,
               input_emb,
               deterministic = False):
    """Applies MoeLayer module.

    If the "intermediates" collection is marked as mutable, this method will sow
    diversity metrics.

    Args:
      input_emb: Batch of input embeddings of shape <float>[BATCH_SIZE,
        SEQ_LENGTH, HIDDEN_DIM].
      deterministic: Whether or not to apply dropout to input and jitter noise
        to router.

    Returns:
      Transformed inputs with same shape as inputs:
      <float>[BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM].

    Raises:
      ValueError if an unrecognized dispatch algorithm is given.
    """
    batch_size, seq_length, hidden_dim = input_emb.shape
    num_tokens = batch_size * seq_length

    num_groups = self._num_groups(num_tokens, self.max_group_size)
    tokens_per_group = num_tokens // num_groups

    if deterministic:
      capacity_factor = self.eval_capacity_factor
    else:
      capacity_factor = self.train_capacity_factor
    # Each group will send expert_capacity tokens to each expert.
    expert_capacity = int(
        round(capacity_factor * tokens_per_group / self.num_experts))
    expert_capacity = max(expert_capacity, self.min_expert_capacity)

    # Reshape batch and sequence/token dimensions for expert routing.
    token_inputs = jnp.reshape(input_emb,
                               (num_groups, tokens_per_group, hidden_dim))

    if isinstance(self.router, routing.ScatterRouter):
      outputs = self._scatter_to_experts(token_inputs, deterministic,
                                         expert_capacity)
    elif isinstance(self.router, routing.MaskedRouter):
      outputs = self._mask_and_dispatch_to_experts(token_inputs, deterministic,
                                                   expert_capacity)
    else:
      raise ValueError("Unrecognized router type: %s" % self.router)

    # Return to original input shape.
    result = outputs.reshape((batch_size, seq_length, hidden_dim))
    return result

  @nn.nowrap
  def _num_groups(self, num_tokens, max_group_size):
    """Returns the number of token routing groups.

    Note: For pmap-based training, all quantities are local to the device.

    We select the smallest num_groups such that:
    - num_groups >= num_tokens / max_group_size (ensuring the group size is no
      larger than max_group_size),
    - num_tokens % num_groups = 0 (ensuring that the group size evenly divides
      into the num_tokens).

    Args:
      num_tokens: Number of tokens from input batch.
      max_group_size: Maximum size of each token routing group. Actual group
        size may end up being smaller.

    Returns:
      Number of token routing groups.

    Raises:
      ValueError if we cannot find a group_size satisfying the above
        requirements.
    """
    # All-to-all communications will be fine as long as num_groups >= 1.
    min_num_groups = num_tokens // max_group_size
    min_num_groups = max(min_num_groups, 1)

    def viable(n):
      """Returns true iff n is a viable number of groups."""
      return num_tokens % n == 0

    # Increase the number of groups (and decrease the group size) until we have
    # a viable number of groups.
    num_groups = min_num_groups
    while num_groups < num_tokens and not viable(num_groups):
      num_groups += 1

    if num_tokens % num_groups > 0:
      raise ValueError(
          "Local (per-device) group size must divide evenly into the local "
          f"number of tokens, but local num_tokens={num_tokens}, while local "
          f"num_groups={num_groups} for max_group_size={max_group_size}.")

    group_size = num_tokens // num_groups
    logging.info(
        "Selected group_size=%d and num_groups=%d for input num_tokens=%d, "
        "max_group_size=%d, and num_experts=%d.", group_size, num_groups,
        num_tokens, max_group_size, self.num_experts)

    return num_groups

  def _scatter_to_experts(self, token_inputs, deterministic,
                          expert_capacity):
    """Wraps expert scatter routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute expert dispatch indices and combine weights using self.router.
    (2) Scatter inputs to experts based on dispatch indices.
    (3) Recombine individual expert outputs using combine weights.

    Args:
      token_inputs: <float>[NUM_GROUPS, TOKENS_PER_GROUP, HIDDEN_DIM] inputs to
        send to experts.
      deterministic: If false, apply jitter noise during routing and dropout
        during expert computation.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      <float>[NUM_GROUPS, TOKENS_PER_GROUP, HIDDEN_DIM] outputs from experts.
    """
    num_groups, tokens_per_group, hidden_dim = token_inputs.shape
    num_tokens = num_groups * tokens_per_group

    router_indices = self.router(
        token_inputs,
        self.num_experts,
        expert_capacity,
        apply_jitter=not deterministic)
    num_selected_experts = self.router.num_selected_experts

    # We need NUM_SELECTED_EXPERT copies of inputs for dispatching. This is a
    # no-op if num_selected_experts = 1.
    token_inputs = jnp.repeat(token_inputs, num_selected_experts, axis=1)

    # Mask out inputs that should not be routed.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS].
    successfully_routed = jnp.logical_and(
        router_indices.dispatch_indices[Ellipsis, 0] < self.num_experts,
        router_indices.dispatch_indices[Ellipsis, 1] < expert_capacity)
    successfully_routed = successfully_routed.reshape((num_groups, -1))
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP * NUM_SELECTED_EXPERTS, HIDDEN_DIM].
    masked_inputs = jnp.einsum(
        "...th,...t->...th",
        token_inputs,
        successfully_routed,
        precision=self.precision)

    # Combine TOKENS_PER_GROUP and NUM_SELECTED_EXPERTS axes.
    flattened_dispatch_indices = router_indices.dispatch_indices.reshape(
        num_groups, -1, 2)

    # Scatter masked inputs.
    shape = (self.num_experts, expert_capacity, hidden_dim)
    # Shape: [NUM_GROUPS, NUM_EXPERTS, EXPERT_CAPACITY, HIDDEN_DIM].
    expert_inputs = jax.vmap(lambda i, x: core_utils.scatter_nd(i, x, shape))(
        flattened_dispatch_indices, masked_inputs)

    expert_outputs = self._call_experts(expert_inputs, deterministic)

    # Gather outputs.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP * NUM_SELECTED_EXPERTS, HIDDEN_DIM].
    expert_outputs = jax.vmap(lambda i, x: x[i[:, 0], i[:, 1]])(
        flattened_dispatch_indices, expert_outputs)
    # Separate out NUM_SELECTED_EXPERTS dimension.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS, HIDDEN_DIM].
    expert_outputs = expert_outputs.reshape(
        (num_groups, tokens_per_group, num_selected_experts, hidden_dim))

    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS, HIDDEN_DIM].
    # Weighted sum of the outputs from the different experts.
    combined_outputs = jnp.einsum(
        "...tkh,...tk->...th",
        expert_outputs,
        router_indices.combine_weights,
        precision=self.precision)

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

    self._sow_expert_metrics(router_indices.auxiliary_loss,  # pytype: disable=wrong-arg-types  # jax-types
                             router_indices.router_z_loss,
                             fraction_tokens_left_behind, router_confidence,
                             expert_usage)

    return combined_outputs

  def _mask_and_dispatch_to_experts(self, token_inputs,
                                    deterministic,
                                    expert_capacity):
    """Wraps expert masked routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute dispatch mask and combine array using self.router.
    (2) Dispatch inputs to experts based on dispatch mask.
    (3) Recombine individual expert outputs using combine array.

    Args:
      token_inputs: <float>[NUM_GROUPS, TOKENS_PER_GROUP, HIDDEN_DIM] inputs to
        send to experts.
      deterministic: If false, apply jitter noise during routing and dropout
        during expert computation.
      expert_capacity: Each group will send this many tokens to each expert.

    Returns:
      <float>[NUM_GROUPS, NUM_TOKENS_PER_GROUP, HIDDEN_DIM] outputs from
      experts.
    """
    num_groups, tokens_per_group, _ = token_inputs.shape
    num_tokens = num_groups * tokens_per_group

    router_mask = self.router(
        token_inputs,
        self.num_experts,
        expert_capacity,
        apply_jitter=not deterministic)

    # Shape: [NUM_GROUPS, NUM_EXPERTS, EXPERT_CAPACITY, HIDDEN_DIM].
    expert_inputs = jnp.einsum(
        "...th,...tec->...ech",
        token_inputs,
        router_mask.dispatch_mask,
        precision=self.precision)

    expert_outputs = self._call_experts(expert_inputs, deterministic)

    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, HIDDEN_DIM]
    combined_outputs = jnp.einsum(
        "...ech,...tec->...th",
        expert_outputs,
        router_mask.combine_array,
        precision=self.precision)

    # Gather and sow expert metrics.
    # Number of tokens that were dispatched to at least one expert.
    num_tokens_dispatched_somewhere = jnp.max(
        router_mask.dispatch_mask, axis=(-1, -2)).sum()
    fraction_tokens_left_behind = 1.0 - num_tokens_dispatched_somewhere / float(
        num_tokens)
    # Total number of tokens that were dispatched (one token could be
    # dispatched to multiple experts).
    num_tokens_dispatched = router_mask.dispatch_mask.sum()
    # Of the tokens dispatched, how confident was the router in its routing?
    router_confidence = router_mask.combine_array.sum() / num_tokens_dispatched

    if isinstance(self.router, routing.ExpertsChooseMaskedRouter):
      expert_usage = 1.  # Experts fully utilized when "expert choose tokens"
    else:
      total_expert_capacity = self.num_experts * expert_capacity * num_groups
      expert_usage = num_tokens_dispatched / total_expert_capacity

    self._sow_expert_metrics(router_mask.auxiliary_loss,  # pytype: disable=wrong-arg-types  # jnp-type
                             router_mask.router_z_loss,
                             fraction_tokens_left_behind, router_confidence,
                             expert_usage)

    return combined_outputs

  def _call_experts(self, inputs,
                    deterministic):
    """Sends and receives inputs to experts using manual all_to_all calls.

    The entire computation is performed using truncated_dtype() to reduce
    all_to_all communication overhead.

    Args:
      inputs: <float>[NUM_GROUPS, NUM_EXPERTS, EXPERT_CAPACITY, HIDDEN_DIM]
        inputs to be dispatched to experts. Each slice across the NUM_EXPERTS
        dimension is passed to a different expert.
      deterministic: Whether or not experts should apply dropout.

    Returns:
      <float>[NUM_GROUPS, NUM_EXPERTS, EXPERT_CAPACITY, HIDDEN_DIM] outputs
      from expert computation.
    """
    # Reshape inputs so they can be mapped to the number of devices.
    num_groups, num_experts, expert_capacity, hidden_dim = inputs.shape
    experts_per_device = num_experts // jax.device_count()
    # Shape: [NUM_DEVICES, EXPERTS_PER_DEVICE, NUM_GROUPS * EXPERT_CAPACITY,
    # HIDDEN_DIM].
    inputs = inputs.reshape(
        (jax.device_count(), experts_per_device, -1, hidden_dim))
    inputs_dtype = inputs.dtype
    inputs = jax.lax.convert_element_type(inputs, truncated_dtype())

    # Send examples to their target devices.
    if self._is_parallel_computation:
      inputs = jax.lax.all_to_all(
          inputs, axis_name=self.axis_name, split_axis=0, concat_axis=0)

    # Apply expert transformation.
    def layer_fn(mapped_expert, expert_inputs):
      return mapped_expert(expert_inputs, deterministic=deterministic)

    # Within each device we have EXPERTS_PER_DEVICE local experts that we vmap
    # over.
    outputs = nn.vmap(
        layer_fn,
        in_axes=(1,),  # 1 is the EXPERTS_PER_DEVICE dimension of `inputs`
        out_axes=1,
        variable_axes={"params": 1},  # Each local expert has its own parameters
        split_rngs={
            # Whether to initialize each expert's params independently.
            "params": self.split_params,
            "dropout": True  # Always use different dropout key for each expert
        })(self.expert, inputs)

    # Send examples back to their original devices.
    if self._is_parallel_computation:
      outputs = jax.lax.all_to_all(
          outputs, axis_name=self.axis_name, split_axis=0, concat_axis=0)

    outputs = jax.lax.convert_element_type(outputs, inputs_dtype)
    # Return to original input shape.
    return outputs.reshape(
        (num_groups, num_experts, expert_capacity, hidden_dim))

  def _sow_expert_metrics(self, auxiliary_loss, router_z_loss,
                          fraction_tokens_left_behind,
                          router_confidence, expert_usage):
    """Sows metrics to analyze expert routing."""
    self.sow(
        "intermediates",
        "diversity_metrics",
        DiversityMetrics(auxiliary_loss, router_z_loss,
                         fraction_tokens_left_behind, expert_usage,
                         router_confidence),
        init_fn=lambda: DiversityMetrics(0., 0., 0., 0., 0.),
        # DiversityMetrics are summed if repeated calls are made.
        reduce_fn=lambda a, b: a + b)


class AttentionLayer(nn.Module):
  """Attention layer, wrapping Flax's MultiHeadDotProductAttention module.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    d_model: dimension of the key, query, and value.
    dropout_rate: The dropout probability.
    dtype: Numerical type of the computation (default: float32).
    precision: XLA precision for matrix multiplication computations.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    pad_id: Token ID representing padding. A mask is used to distinguish padding
      from actual inputs.
  """
  num_heads: int
  d_model: int
  dropout_rate: float = 0.
  dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
  kernel_init: Callable[[PRNGKey, Shape, jnp.dtype],
                        jnp.ndarray] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, jnp.dtype],
                      jnp.ndarray] = default_bias_init
  pad_id: int = 0

  @nn.compact
  def __call__(self,
               input_emb,
               input_ids,
               deterministic = False):
    """Applies AttentionLayer module.

    Args:
      input_emb: Batch of input embeddings of shape <float>[BATCH_SIZE,
        INPUT_SEQ_LENGTH, HIDDEN_DIM]. Input embeddings are used for all mixing
        modes.
      input_ids: Tokenized inputs of shape [BATCH_SIZE, INPUT_SEQ_LENGTH]. Only
        used to construct attention mask.
      deterministic: Whether to apply dropout to input.

    Returns:
      Attention-weighted output with shape <float>[BATCH_SIZE, INPUT_SEQ_LENGTH,
      HIDDEN_DIM].
    """
    attention_mask = nn.make_attention_mask(
        input_ids != self.pad_id, input_ids != self.pad_id, dtype=self.dtype)

    return nn.SelfAttention(
        num_heads=self.num_heads,
        use_bias=True,
        qkv_features=self.d_model,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        kernel_init=default_kernel_init,
        bias_init=default_bias_init,
        precision=self.precision,
        name="self")(
            inputs_q=input_emb,
            mask=attention_mask,
            deterministic=deterministic)


def _init_fourier_transform(
    use_fft, input_seq_length, d_model,
    precision):
  """Initializes 2D discrete Fourier Transform (DFT).

  Recommendations on computing the DFT using the FFT or matrix multiplications:
    - On GPUs/CPUs: FFT implementation is optimal for all sequence lengths.
    - On TPUs: For relatively shorter sequences, it is faster to use matrix
        multiplications. For longer sequences, the FFT is faster, provided the
        sequence lengths are a power of 2.

  Args:
    use_fft: Whether to compute DFT using the FFT or matrix multiplications.
    input_seq_length: The maximum input sequence length after tokenization.
    d_model: Hidden dimension of model.
    precision: XLA precision for matrix multiplication computation.

  Returns:
    2D discrete Fourier Transform function.

  Raises:
    - ValueError if attempting to compute DFT for very long sequences without
      FFT.
  """
  if use_fft:
    if input_seq_length > 4096 and not math.log2(input_seq_length).is_integer():
      raise ValueError(
          "For large input sequence lengths (>4096), the maximum input "
          "sequence length must be a power of 2 to take advantage of FFT "
          "optimizations. We encourage the same for the model hidden "
          "dimension. input_seq_length: %d. d_model: $d" % input_seq_length,
          d_model)

    return jnp.fft.fftn

  dft_mat_hidden = jnp.asarray(linalg.dft(d_model))
  dft_mat_seq = jnp.asarray(linalg.dft(input_seq_length))
  return functools.partial(
      jnp.einsum,
      "ni,jk,ij->nk",
      dft_mat_seq,
      dft_mat_hidden,
      optimize=True,
      precision=precision)


class FourierTransform(MixingLayer):
  """Fourier Transform layer.

  Applies 2D Fourier Transform over final two dimensions of inputs - typically
  the sequence and hidden dimensions.

  Attributes:
    use_fft: If true, use FFT algorithm to compute the discrete Fourier
      Transform (DFT). If false, caches the DFT matrix and computes transform
      using matrix multiplications.
    precision: XLA precision for matrix multiplication computations.
  """
  use_fft: bool = False
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self,
               input_emb,
               deterministic = False):
    """Applies FourierTransform module.

    Args:
      input_emb: Batch of input embeddings of shape <float>[BATCH_SIZE,
        INPUT_SEQ_LENGTH, HIDDEN_DIM]. Input embeddings are used for all mixing
        modes.
      deterministic: Ignored. Whether to apply dropout to input.

    Returns:
      Real part of discrete Fourier Transform of input data. Shape:
      <float>[BATCH_SIZE, INPUT_SEQ_LENGTH, HIDDEN_DIM].
    """
    del deterministic  # Fourier sublayer is always deterministic.

    d_model = input_emb.shape[-1]
    input_seq_length = input_emb.shape[-2]
    fourier_transform = _init_fourier_transform(self.use_fft, input_seq_length,
                                                d_model, self.precision)

    return jax.vmap(fourier_transform)(input_emb).real


class HartleyTransform(MixingLayer):
  """Hartley Transform layer.

  Applies 2D Hartley transform over final two dimensions of inputs - typically
  the sequence and hidden dimensions.

  Attributes:
    use_fft: If true, use FFT algorithm to compute the discrete Fourier
      Transform (DFT). If false, caches the DFT matrix and computes transform
      using matrix multiplications.
    precision: XLA precision for matrix multiplication computations.
  """
  use_fft: bool = False
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self,
               input_emb,
               deterministic = False):
    """Applies HartleyTransform module.

    Args:
      input_emb: Batch of input embeddings of shape <float>[BATCH_SIZE,
        INPUT_SEQ_LENGTH, HIDDEN_DIM]. Input embeddings are used for all mixing
        modes.
      deterministic: Ignored. Whether to apply dropout to input.

    Returns:
      Discrete Hartley Transform of input data. Shape: <float>[BATCH_SIZE,
      INPUT_SEQ_LENGTH, HIDDEN_DIM].
    """
    del deterministic  # Hartley Transform is always deterministic.

    d_model = input_emb.shape[-1]
    input_seq_length = input_emb.shape[-2]
    # The Hartley Transform is computed using the Fourier Transform.
    fourier_transform = _init_fourier_transform(self.use_fft, input_seq_length,
                                                d_model, self.precision)

    frequencies = jax.vmap(fourier_transform)(input_emb)
    return frequencies.real - frequencies.imag


class LinearTransform(MixingLayer):
  """Dense, linear transformation layer.

  Applies matrix multiplications over sequence and hidden dimensions.

  Attributes:
    kernel_init: Initializer scheme for (matrix) kernel parameters.
    precision: XLA precision for matrix multiplication computations.
  """
  kernel_init: Callable[[PRNGKey, Shape, jnp.dtype],
                        jnp.ndarray] = default_kernel_init
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self,
               input_emb,
               deterministic = False):
    """Applies LinearTransform module.

    Args:
      input_emb: Batch of input embeddings of shape <float>[BATCH_SIZE,
        INPUT_SEQ_LENGTH, HIDDEN_DIM]. Input embeddings are used for all mixing
        modes.
      deterministic: Ignored. Whether to apply dropout to input.

    Returns:
      Inputs linearly mixed with learnable weights. Shape: <float>[BATCH_SIZE,
      INPUT_SEQ_LENGTH, HIDDEN_DIM].
    """
    del deterministic  # LinearTransform is always deterministic.

    mat_hidden = self.param("hidden_kernel", self.kernel_init,
                            (input_emb.shape[-1], input_emb.shape[-1]))
    mat_seq = self.param("input_kernel", self.kernel_init,
                         (input_emb.shape[-2], input_emb.shape[-2]))

    return jnp.einsum(  # pytype: disable=wrong-arg-types  # jnp-type
        "bij,jk,ni->bnk",
        input_emb,
        mat_hidden,
        mat_seq,
        optimize=True,
        precision=self.precision)


def _circulant_matrix(inputs):
  """Constructs a circulant matrix from the input vector.

  We construct the circulant matrix as a special case of a toeplitz matrix.

  Args:
    inputs: Vector of shape [N].

  Returns:
    Circulant matrix of shape [N, N].

  Raises:
    ValueError: If inputs dimension is not 1.
  """
  if inputs.ndim != 1:
    raise ValueError("Input vector must be 1D. Given input has dimension: %d" %
                     inputs.ndim)

  flipped_inputs = jnp.flip(inputs)
  circ = jax.lax.concatenate([flipped_inputs, flipped_inputs[:-1]], dimension=0)
  return _toeplitz_matrix(circ)


def _apply_2d_fft_circulant(inputs,
                            circ_vector_dim_zero,
                            circ_vector_dim_one):
  """Performs 2D circulant matrix multiplication on inputs, using the FFT.

  Args:
    inputs: [N0, N1] inputs to be multiplied by circulant matrices.
    circ_vector_dim_zero: [N0] vector representation of circulant matrix.
      Applied to zeroth dimension of inputs.
    circ_vector_dim_one: [N1] vector representation of circulant matrix. Applied
      to first dimension (zero-based) of inputs.

  Returns:
    2D circularly convolved inputs of shape [N0, N1].

  Raises:
    ValueError: If inputs is not two-dimensional, or circular vectors are not
      one dimensional.
  """
  if inputs.ndim != 2:
    raise ValueError("Expected inputs dimension is 2. Received: %d" %
                     inputs.ndim)

  if circ_vector_dim_zero.ndim != 1 or circ_vector_dim_one.ndim != 1:
    raise ValueError(
        "Circular vectors should have dimension 1. Received: %d and %d" %
        (circ_vector_dim_zero.ndim, circ_vector_dim_one.ndim))

  x = jnp.fft.ifft(
      jnp.transpose(
          jnp.fft.fft(circ_vector_dim_zero) *
          jnp.transpose(jnp.fft.fft(inputs, axis=0))),
      axis=0)
  return jnp.fft.ifft(
      jnp.fft.fft(circ_vector_dim_one) * jnp.fft.fft(x, axis=1), axis=1).real


class CirculantTransform(MixingLayer):
  """Circulant matrix transformation layer.

  Applies parameterized circulant matrix multiplications over sequence and
  hidden dimensions.

  Attributes:
    use_fft: If true, use FFT algorithm to compute the circulant matrix matrix
      multiplication; otherwise, performs regular matrix multiplication.
    kernel_init: Initializer scheme for (matrix) kernel parameters.
    precision: XLA precision for matrix multiplication computations.
  """
  use_fft: bool = False
  kernel_init: Callable[[PRNGKey, Shape, jnp.dtype],
                        jnp.ndarray] = default_kernel_init
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self,
               input_emb,
               deterministic = False):
    """Applies CirculantTransform module.

    Args:
      input_emb: Batch of input embeddings of shape <float>[BATCH_SIZE,
        INPUT_SEQ_LENGTH, HIDDEN_DIM]. Input embeddings are used for all mixing
        modes.
      deterministic: Ignored. Whether to apply dropout to input.

    Returns:
      Inputs mixed with learned weights, constrained to circulant matrix
      structure. Shape: <float>[BATCH_SIZE, INPUT_SEQ_LENGTH, HIDDEN_DIM].
    """
    del deterministic  # CirculantTransform is always deterministic.

    hidden_circ = self.param("hidden_kernel", self.kernel_init,
                             (input_emb.shape[-1],))

    if self.use_fft:
      seq_circ = self.param("input_kernel", self.kernel_init,
                            (input_emb.shape[-2],))
      circulant_matmul = functools.partial(
          _apply_2d_fft_circulant,
          circ_vector_dim_zero=seq_circ,
          circ_vector_dim_one=hidden_circ)
      return jax.vmap(circulant_matmul)(input_emb)

    mat_hidden = _circulant_matrix(hidden_circ)
    seq_circ = self.param("input_kernel", self.kernel_init,
                          (input_emb.shape[-2],))
    mat_seq = _circulant_matrix(seq_circ)

    circulant_matmul = functools.partial(
        jnp.einsum,
        "ni,kj,ij->nk",
        mat_seq,
        mat_hidden,
        optimize=True,
        precision=self.precision)

    return jax.vmap(circulant_matmul)(input_emb)


def _toeplitz_matrix(inputs):
  """Constructs a toeplitz matrix from the input vector.

  We first construct a "skewed" matrix by tiling and squashing the input array
  into a rectangular matrix, before extracting the square toeplitz sub-matrix.

  Args:
    inputs: Vector of shape [2 * N - 1].

  Returns:
    Toeplitz matrix of shape [N, N].

  Raises:
    ValueError: If inputs dimension is not 1.
  """
  if inputs.ndim != 1:
    raise ValueError("Input vector must be 1D. Given input has dimension: %d" %
                     inputs.ndim)

  toe_dim = inputs.shape[0] // 2 + 1
  reshape_num_cols = max(inputs.shape[0] - 1, 1)

  rolled_inputs = jnp.roll(inputs, toe_dim)
  repeated_arr = jnp.tile(rolled_inputs, toe_dim)[:toe_dim * reshape_num_cols]

  skewed_arr = jnp.reshape(repeated_arr, (-1, reshape_num_cols))
  return skewed_arr[:, :toe_dim]


def _embed_in_circulant(inputs):
  """Embeds [N] input vector in [N+1] circular vector."""
  n = inputs.size
  return jnp.concatenate([
      jnp.flip(inputs[:n // 2 + 1]), inputs[n // 2, None],
      jnp.flip(inputs[n // 2 + 1:])
  ])


def _apply_2d_fft_toeplitz(inputs,
                           toe_vector_dim_zero,
                           toe_vector_dim_one):
  """Performs 2D toeplitz matrix multiplication on inputs, using the FFT.

  To multiply the toeplitz matrices (represented by the 1D toe_vector_dim_*
  arrays) by the inputs, we:
  (1) embed the toeplitz vectors in larger circulant vector,
  (2) apply the _apply_2d_fft_circulant() algorithm, and then
  (3) extract the relevant slice of the result.

  Args:
    inputs: [N0, N1] inputs to be multiplied by toeplitz matrices.
    toe_vector_dim_zero: [2*N0-1] vector representation of [N0,N0] toeplitz
      matrix. Applied to zeroth dimension of inputs.
    toe_vector_dim_one: [2*N1-1] vector representation of [N1,N1] toeplitz
      matrix. Applied to first dimension (zero-based) of inputs.

  Returns:
    2D toeplitz convoled inputs of shape [N0, N1].

  Raises:
    ValueError: If inputs is not two-dimensional, or toeplitz vectors are not
      one dimensional.
  """
  if inputs.ndim != 2:
    raise ValueError("Expected inputs dimension is 2. Received: %d" %
                     inputs.ndim)
  n0, n1 = inputs.shape

  if toe_vector_dim_zero.ndim != 1 or toe_vector_dim_one.ndim != 1:
    raise ValueError(
        "Toeplitz vectors should have dimension 1. Received: %d and %d" %
        (toe_vector_dim_zero.ndim, toe_vector_dim_one.ndim))

  # Shape [2*N0]
  circ_vector_dim_zero = _embed_in_circulant(toe_vector_dim_zero)
  # Shape [2*N1]
  circ_vector_dim_one = _embed_in_circulant(toe_vector_dim_one)
  # Shape [2*N0, 2*N1]
  x = jnp.pad(inputs, [(0, n0), (0, n1)])

  z = _apply_2d_fft_circulant(x, circ_vector_dim_zero, circ_vector_dim_one)

  # Extract the [n0,n1] slice of the result
  return z[:n0, :n1]


class ToeplitzTransform(MixingLayer):
  """Toeplitz matrix transformation layer.

  Applies parameterized toeplitz matrix multiplications over sequence and
  hidden dimensions.

  Attributes:
    use_fft: If true, use FFT algorithm to compute the toeplitz matrix matrix
      multiplication; otherwise, performs regular matrix multiplication.
    kernel_init: Initializer scheme for (matrix) kernel parameters.
    precision: XLA precision for matrix multiplication computations.
  """
  use_fft: bool = False
  kernel_init: Callable[[PRNGKey, Shape, jnp.dtype],
                        jnp.ndarray] = default_kernel_init
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self,
               input_emb,
               deterministic = False):
    """Applies ToeplitzTransform module.

    Args:
      input_emb: Batch of input embeddings of shape <float>[BATCH_SIZE,
        INPUT_SEQ_LENGTH, HIDDEN_DIM]. Input embeddings are used for all mixing
        modes.
      deterministic: Ignored. Whether to apply dropout to input.

    Returns:
      Inputs mixed with learned weights, constrained to toeplitz matrix
      structure. Shape: float>[BATCH_SIZE, INPUT_SEQ_LENGTH, HIDDEN_DIM].
    """
    del deterministic  # ToeplitzTransform is always deterministic.

    hidden_toe = self.param("hidden_kernel", self.kernel_init,
                            (2 * input_emb.shape[-1] - 1,))
    seq_toe = self.param("input_kernel", self.kernel_init,
                         (2 * input_emb.shape[-2] - 1,))

    if self.use_fft:
      toeplitz_matmul = functools.partial(
          _apply_2d_fft_toeplitz,
          toe_vector_dim_zero=seq_toe,
          toe_vector_dim_one=hidden_toe)
      return jax.vmap(toeplitz_matmul)(input_emb)

    mat_hidden = _toeplitz_matrix(hidden_toe)
    mat_seq = _toeplitz_matrix(seq_toe)

    toeplitz_matmul = functools.partial(
        jnp.einsum,
        "ni,kj,ij->nk",
        mat_seq,
        mat_hidden,
        optimize=True,
        precision=self.precision)

    return jax.vmap(toeplitz_matmul)(input_emb)


class EncoderBlock(nn.Module):
  """Post layer norm encoder model block.

  An EncoderBlock consists of applying the following submodules:
    (1) mixing_sublayer
    (2) Residual connection
    (3) Layer norm
    (4) feed_forward_sublayer
    (5) Residual connection
    (6) Layer norm

  Attributes:
    feed_forward_sublayer: Generalized feed-forward module.
    mixing_sublayer: Mixing module.
    attention_sublayer: Attention module. Either mixing_sublayer or
      attention_sublayer must be specified, but not both.
    dtype: The numeric type of the computation (default: float32).
  """
  mixing_sublayer: Optional[MixingLayer]
  attention_sublayer: Optional[AttentionLayer]
  feed_forward_sublayer: GeneralizedFeedForwardLayer
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    if (self.mixing_sublayer is None) == (self.attention_sublayer is None):
      raise ValueError(
          "One, and only one, of {self.mixing_sublayer, "
          "self.attention_sublayer} must be nonempty. Received: %s and %s" %
          (self.mixing_sublayer, self.attention_sublayer))

    self.mixing_layer_norm = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON, dtype=self.dtype, name="mixing_layer_norm")
    self.output_layer_norm = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON, dtype=self.dtype, name="output_layer_norm")

  @nn.compact
  def __call__(self,
               input_emb,
               input_ids,
               deterministic = False):
    """Applies EncoderBlock module.

    Args:
      input_emb: Batch of input embeddings of shape <float>[BATCH_SIZE,
        SEQ_LENGTH, HIDDEN_DIM].
      input_ids: Tokenized inputs of shape <int>[BATCH_SIZE, SEQ_LENGTH]. Only
        used for constructing the attention mask in attention layers.
      deterministic: Whether to apply dropout.

    Returns:
      Encoded outputs of shape: <float>[BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM].
    """
    if self.mixing_sublayer:
      mixing_output = self.mixing_sublayer(
          input_emb, deterministic=deterministic)
    else:
      mixing_output = self.attention_sublayer(
          input_emb, input_ids=input_ids, deterministic=deterministic)
    x = self.mixing_layer_norm(input_emb + mixing_output)

    feed_forward_output = self.feed_forward_sublayer(
        x, deterministic=deterministic)
    return self.output_layer_norm(x + feed_forward_output)


class OutputProjection(nn.Module):
  """A dense projection layer for computing output logits.

  Attributes:
    kernel: Pre-computed kernel parameters of shape <float>[n_out, HIDDEN_DIM].
    n_out: Number of output dimensions. Required if kernel is None.
    bias: Whether or not to apply a bias term.
    kernel_init: Initializer scheme for kernel parameters.
    bias_init: Initializer scheme for bias parameters.
    precision: XLA precision for matrix multiplication computation.
  """
  kernel: Optional[jnp.ndarray] = None
  n_out: Optional[int] = None  # Required if kernel is None.
  bias: bool = True
  kernel_init: Callable[[PRNGKey, Shape, jnp.dtype],
                        jnp.ndarray] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, jnp.dtype],
                      jnp.ndarray] = default_bias_init
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self, input_emb):
    """Applies OutputProjection module.

    Args:
      input_emb: Batch of input embeddings of shape <float>[BATCH_SIZE, ...,
        HIDDEN_DIM].

    Returns:
      Output projected logits of shape <float>[BATCH_SIZE, ..., n_out]

    Raises:
      ValueError: If self.kernel and self.n_out are both None.
    """
    if self.kernel is None:
      if self.n_out is None:
        raise ValueError(
            "OutputProjection must be initialized with n_out attribute when "
            "not re-using an existing kernel, such as an embedding matrix.")
      kernel = self.param("output_kernel", self.kernel_init,
                          (self.n_out, input_emb.shape[-1]))
    else:
      kernel = self.kernel
    y = jnp.matmul(
        input_emb, jnp.transpose(kernel, (1, 0)), precision=self.precision)
    if self.bias:
      bias = self.param("output_bias", self.bias_init, (y.shape[-1],))
      y = y + bias
    return y


class EmbeddingLayer(nn.Module):
  """Sums word, position and type embeddings.

  Attributes:
    config: Model configuration.
  """
  config: ml_collections.FrozenConfigDict

  @nn.compact
  def __call__(self,
               input_ids,
               type_ids,
               deterministic = False):
    """Applies EmbeddingLayer module.

    Args:
      input_ids: Batch of tokenized inputs of shape <int>[BATCH_SIZE,
        SEQ_LENGTH].
      type_ids: Ids partitioning input into different types.
      deterministic: Whether to apply dropout to output embeddings.

    Returns:
      Embedded tokens of shape <float>[BATCH_SIZE, SEQ_LENGTH, EMB_DIM].
    """
    word_embeddings = nn.Embed(
        num_embeddings=self.config.vocab_size,
        features=self.config.d_emb,
        dtype=self.config.dtype,
        embedding_init=default_kernel_init,
        name="word")(
            input_ids)
    position_embeddings = PositionalEncoding(  # pytype: disable=wrong-arg-types  # jax-types
        seq_length=self.config.max_seq_length,
        posemb_init=default_kernel_init,
        name="position")(
            word_embeddings)
    type_embeddings = nn.Embed(
        num_embeddings=self.config.type_vocab_size,
        features=self.config.d_emb,
        dtype=self.config.dtype,
        embedding_init=default_kernel_init,
        name="type")(
            type_ids)

    embeddings = word_embeddings + position_embeddings + type_embeddings
    embeddings = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON, dtype=self.config.dtype, name="layer_norm")(
            embeddings)
    embeddings = nn.DenseGeneral(
        self.config.d_model,
        use_bias=True,
        dtype=self.config.dtype,
        name="hidden_mapping_in")(
            embeddings)
    return nn.Dropout(rate=self.config.dropout_rate)(
        embeddings, deterministic=deterministic)


class PositionalEncoding(nn.Module):
  """Learned positional embeddings.

  Attributes:
    seq_length: Maximum sequence length.
    posemb_init: Initializer scheme for positional embedding parameters.
  """
  seq_length: int
  posemb_init: Callable[[PRNGKey, Shape, jnp.dtype],
                        jnp.ndarray] = default_kernel_init

  @nn.compact
  def __call__(self, word_embeddings):
    """Applies PositionalEncoding module.

    Args:
      word_embeddings: Embeddings of input tokens of shape <float>[BATCH_SIZE,
        SEQ_LENGTH, EMB_DIM].

    Returns:
      Positional embeddings <float>[1, SEQ_LENGTH, EMB_DIM] associated with
      input word embeddings.

    Raises:
      ValueError: If word_embeddings dimension is not 3.
    """
    if word_embeddings.ndim != 3:
      raise ValueError(
          "Input word_embeddings dimension should be 3, but it is: %d" %
          word_embeddings.ndim)

    length = word_embeddings.shape[1]
    pos_emb_shape = (1, self.seq_length, word_embeddings.shape[-1])
    pos_embedding = self.param("embedding", self.posemb_init, pos_emb_shape)
    return pos_embedding[:, :length, :]


def gather(sequence, indices):
  """Gathers sequence at the specified indices.

  Args:
    sequence: Sequence of shape <float>[BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM].
    indices: <int>[BATCH_SIZE, MAX_PREDICTIONS_PER_SEQ] indices of tokens in
      sequence to gather.  We favor one-hot matrix multiplication over slices
      when running on TPUs. This may need to be revisited for very large
      MAX_PREDICTIONS_PER_SEQs or HIDDEN_DIMs.

  Returns:
    <float>[BATCH_SIZE * MAX_PREDICTIONS_PER_SEQ, HIDDEN_DIM] elements of input
    sequence at specified indices.

  Raises:
    ValueError: If input sequence and indices have different batch sizes or
    MAX_PREDICTIONS_PER_SEQ > SEQ_LENGTH.
  """
  if sequence.shape[0] != indices.shape[0]:
    raise ValueError(
        "Input sequence and indices must have the same batch size: "
        "sequence.shape[0] = %d whereas indices.shape[0] = %d." %
        (sequence.shape[0], indices.shape[0]))

  if indices.shape[1] > sequence.shape[1]:
    raise ValueError(
        "The maximum number of predictions per sequence cannot be greater "
        "than the maximum sequence length. indices.shape[1] = %d and "
        "sequence.shape[1] = %d." % (indices.shape[1], sequence.shape[1]))

  _, seq_length, hidden_dim = sequence.shape

  if _favor_one_hot_slices():
    indices_oh = jax.nn.one_hot(indices, seq_length)
    out = jnp.einsum(
        "bsh,bis->bih",
        sequence,
        indices_oh,
        precision=jax.lax.Precision.DEFAULT)
    return out.reshape((-1, hidden_dim))
  else:
    return jax.vmap(lambda s, i: jnp.take(s, i, axis=0))(
        sequence, indices).reshape([-1, sequence.shape[-1]])
