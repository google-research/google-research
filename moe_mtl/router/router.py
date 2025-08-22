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

"""Router Implementation."""
import functools
from typing import Any, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from vmoe import moe
from vmoe.nn.routing import Array
from vmoe.nn.routing import BaseDispatcher
from vmoe.nn.routing import DType
from vmoe.nn.routing import KwArgs
from vmoe.nn.routing import Metrics


class CustomNoisyTopExpertsPerItemRouter(nn.Module):
  """Noisy TopExpertsPerItem router used in https://arxiv.org/abs/2106.05974.

  First, a dense (i.e. the gating) layer computes logits for each pair of
  (item, expert). Noise is added to these logits. The logits are normalized
  using a softmax over the expert dimension. This score will be used to
  determine which items are dispatched to which experts and how the outputs of
  the experts are combined.

  Because the routing algorithm is non-differentiable, the only way to train the
  parameters of the dense (a.k.a. gating layer) is through the weights used
  to combine the output of the experts, and through two auxiliary losses that
  depend on the output of the gating.
  """
  num_experts: int
  num_selected_experts: int = 1
  noise_std: float = 1.0
  importance_loss_weight: float = 1.0
  load_loss_weight: float = 1.0
  dispatcher: Optional[KwArgs] = None
  deterministic: bool = False
  dtype: Optional[DType] = None
  return_gates: Optional[bool] = False
  rng_name: Optional[str] = None
  @nn.compact
  def __call__(
      self, inputs
  ):
    gates_softmax, metrics = self._compute_gates_softmax_and_metrics(
        inputs, self.num_experts)
    dispatcher = self._create_dispatcher(gates_softmax)
    if self.return_gates:
      return dispatcher, metrics, gates_softmax
    else:
      return dispatcher, metrics

  @nn.nowrap
  def _compute_gates_softmax_and_metrics(
      self, inputs, num_experts):
    if inputs.ndim != 3:
      raise ValueError(f"inputs.ndim must be 3, but it is {inputs.ndim}")
    if not num_experts >= self.num_selected_experts >= 1:
      raise ValueError(f"num_experts >= num_selected_experts >= 1, but got "
                       f"num_experts = {num_experts} and "
                       f"num_selected_experts = {self.num_selected_experts}.")
    dtype = self.dtype or inputs.dtype
    # Compute the gating logits for each pair of (item, expert).
    gates_logits = nn.Dense(features=num_experts, use_bias=False,
                            dtype=dtype, name="dense")(inputs)
    # Compute the auxiliary losses defined in Appendix A.2, from
    # https://arxiv.org/abs/2106.05974. Notice that the "Load Loss" can only be
    # computed if the router is stochastic (i.e. deterministic = False).
    # Notice that the auxiliary losses are computed on each group independently
    # (i.e. through the vmaps surrounding the calls).
    gates_softmax = jax.nn.softmax(gates_logits)
    # gates_softmax = jnp.ones_like(gates_softmax, dtype=jnp.bfloat16)
    importance_loss = jax.vmap(self._importance_auxiliary_loss)(gates_softmax)
    if self.deterministic or self.noise_std == 0.0:
      metrics = {
          "auxiliary_loss": self.importance_loss_weight * importance_loss,
          "importance_loss": importance_loss,
          "logits": gates_logits,
      }
      return gates_softmax, metrics
    else:
      noise_std = (1.0 / num_experts) * self.noise_std
      if self.rng_name:
        logits_noise = noise_std * jax.random.normal(
            key=self.make_rng(self.rng_name), shape=gates_logits.shape)
      else:
        logits_noise = noise_std * jax.random.normal(
            key=self.make_rng("gating"), shape=gates_logits.shape)
      gates_logits_noisy = gates_logits + logits_noise
      gates_softmax_noisy = jax.nn.softmax(gates_logits_noisy)
      load_loss = jax.vmap(
          functools.partial(
              self._load_auxiliary_loss,
              num_selected_experts=self.num_selected_experts,
              noise_std=noise_std))(gates_logits, gates_logits_noisy)
      metrics = {
          "auxiliary_loss": (self.importance_loss_weight * importance_loss +
                             self.load_loss_weight * load_loss),
          "importance_loss": importance_loss,
          "load_loss": load_loss,
          "logits": gates_logits,
      }
      return gates_softmax_noisy, metrics

  @nn.nowrap
  def _create_dispatcher(self, gates_softmax):
    # Creates a dispatcher implementing the TopExpertsPerItem routing algorithm,
    # that uses at most `num_selected_experts` per item. Notice that each
    # group is dispatched independently.
    dispatcher_kwargs = dict(**(self.dispatcher or {}))
    use_bfloat16 = dispatcher_kwargs.pop("bfloat16", False)
    get_top_experts_per_item_dispatcher_vmapped = jax.vmap(
        functools.partial(
            moe.get_top_experts_per_item_dispatcher,
            num_selected_experts=self.num_selected_experts,
            **dispatcher_kwargs))
    dispatcher = get_top_experts_per_item_dispatcher_vmapped(
        gates_softmax)
    if use_bfloat16:
      dispatcher = moe.Bfloat16Dispatcher(dispatcher)
    return dispatcher

  @classmethod
  def _importance_auxiliary_loss(cls, gates):
    axis = tuple(range(gates.ndim - 1))  # All except last.
    importance_per_expert = jnp.sum(gates, axis=axis)
    std_importance_per_expert = jnp.std(importance_per_expert)
    mean_importance_per_expert = jnp.mean(importance_per_expert)
    # Compute coefficient of variation (i.e. std/mean) squared.
    return (std_importance_per_expert / mean_importance_per_expert)**2

  @classmethod
  def _load_auxiliary_loss(
      cls, logits, logits_noisy,
      noise_std,
      num_selected_experts):
    # For each example, compute the weight required for an expert to be selected
    # among the top-k.
    # NOTE: DO NOT TRY TO SIMPLIFY THIS. This convoluted way of obtaining the
    # threshold_per_item avoids adding all-gather ops during backpropagation.
    num_experts = logits_noisy.shape[-1]
    threshold_per_item_index = jax.lax.top_k(
        logits_noisy, num_selected_experts)[-1][Ellipsis, -1]
    threshold_per_item = jnp.sum(
        jax.nn.one_hot(threshold_per_item_index, num_experts) * logits_noisy,
        axis=-1)
    # For each example and expert, find how far they were from the threshold and
    # normalize this value by the noise_std to use the standard Gaussian CDF.
    noise_required_to_win = threshold_per_item[Ellipsis, None] - logits
    noise_required_to_win /= noise_std
    # p is the probability of being above the threshold for each (item, expert)
    # if the random noise (with its std) was re-sampled again.
    p = 1. - jax.scipy.stats.norm.cdf(noise_required_to_win)
    # We compute the average such probability for each expert over examples.
    p_mean = jnp.mean(p, axis=0)
    # Compute p_mean's coefficient of variation squared.
    return (jnp.std(p_mean) / jnp.mean(p_mean))**2
