# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Differentiable top-k using perturbed input for gradient computation.

See https://q-berthet.github.io/papers/BerBloTeb20.pdf.
"""

import enum
import functools

import jax
import jax.numpy as jnp


class SortBy(enum.Enum):
  NONE = None
  POSITION = "position"
  VALUES = "values"


class Noise(enum.Enum):
  NORMAL = "normal"


def sorted_topk_indicators(x, k, sort_by = SortBy.POSITION):
  """Finds the (sorted) positions of the topk values in x.

  Args:
    x: The input scores of dimension (d,).
    k: The number of top elements to find.
    sort_by: Strategy to order the extracted values. This is useful when this
      function is applied to many perturbed input and average. As topk's output
      does not have a fixed order, the indicator vectors could be swaped and the
      average of the indicators would not be spiky.

  Returns:
    Indicator vectors in a tensor of shape (k, d)
  """
  n = x.shape[-1]
  values, ranks = jax.lax.top_k(x, k)

  if sort_by == SortBy.NONE:
    sorted_ranks = ranks
  if sort_by == SortBy.VALUES:
    sorted_ranks = ranks[jnp.argsort(values)]
  if sort_by == SortBy.POSITION:
    sorted_ranks = jnp.sort(ranks)

  one_hot_fn = jax.vmap(functools.partial(jax.nn.one_hot, num_classes=n))
  indicators = one_hot_fn(sorted_ranks)
  return indicators


def perturbed(func,
              num_samples = 1000,
              noise = Noise.NORMAL):
  """Creates a function that applies func on multiple perturbed input.

  Args:
    func: The function to make a perturbed version of.
    num_samples: The number of perturbed input to generate, pass through func
      and average to obtain the perturbed output.
    noise: Type of the noise.

  Returns:
    A function with the same signature as `func` but that will compute an
    expectation of the perturbed output.
  """
  noise = Noise(noise)
  assert noise == Noise.NORMAL, "Only normal noise is supported for now."

  @jax.custom_vjp
  def foo(input_tensor, sigma, rng_key):
    return forward(input_tensor, sigma, rng_key)[0]

  def forward(input_tensor, sigma, rng_key):
    noise_shape = (num_samples,) + input_tensor.shape
    noise = jax.random.normal(rng_key, shape=noise_shape)
    noise_gradient = noise
    noisy_input_tensor = input_tensor + noise * sigma
    perturbed_outputs = jax.vmap(func)(noisy_input_tensor)
    forward_outputs = perturbed_outputs.mean(axis=0)
    keep_for_bwd = (perturbed_outputs, noise_gradient, sigma)
    return forward_outputs, keep_for_bwd

  def backward(keep_for_bwd, output_grad):
    perturbed_outputs, noise_gradient, sigma = keep_for_bwd
    expected_gradient = jnp.mean(
        perturbed_outputs * noise_gradient[:, None, :] / sigma, axis=0)
    return ((output_grad * expected_gradient).sum(axis=0), None, None)

  foo.defvjp(forward, backward)
  return foo


def perturbed_sorted_topk_indicators(x, rng, k,
                                     sigma,
                                     num_samples = 1000,
                                     noise = "normal"):
  return perturbed(
      functools.partial(sorted_topk_indicators, k=k, sort_by=SortBy.POSITION),
      num_samples, noise)(x, sigma, rng)
