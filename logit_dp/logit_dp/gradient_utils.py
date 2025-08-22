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

"""Utilities for computing and clipping gradients."""

import functools
import jax
import jax.numpy as jnp
import optax
# ==============================================================================
# Gradient accumulation utils.
# ==============================================================================


@jax.jit
def compute_partial_contrastive_loss_gradient(
    scaled_cosine_values,
    scaled_clipped_cosine_gradients,
):
  """Computes the gradient of the constrastive loss using cosine scalars."""
  exponent_cosine_values = jnp.exp(scaled_cosine_values)
  denominator = jnp.einsum("ij->i", exponent_cosine_values)

  def loss_from_dot_prods_fn(clipped_gradient_segment):
    numerator_segment = -jnp.einsum(
        "ij...,ij->i...", clipped_gradient_segment, exponent_cosine_values
    )
    return numerator_segment

  numerator_all = jax.tree.map(
      loss_from_dot_prods_fn, scaled_clipped_cosine_gradients
  )[0]
  return numerator_all, denominator


def compute_contrastive_loss_gradient_sequentially(
    batch_input_pairs,
    forward_fn,
    params,
    key,
    sequential_computation_steps=1,
    temperature=1.0,
    l2_norm_clip=None,
):
  """Compute contrastive loss gradient sequentially by minibatch processing.

  Args:
    batch_input_pairs: batch of anchors and positives samples.
    forward_fn: model forward function.
    params: model parameters.
    key: jax random number generator.
    sequential_computation_steps: number of accumulation steps. Must divide
      batch size.
    temperature: temperature for gradients.
    l2_norm_clip: l2_norm_clip.

  Returns:
    Contrastive loss gradient.
  Raises:
    ValueError: raises a value error if batch size is not a multiple of the
    number of sequential computation steps.
  """
  batch_size = batch_input_pairs.shape[0]
  if batch_size % sequential_computation_steps != 0:
    raise ValueError(
        "Number of sequential computation steps should divide batch size."
    )
  mini_batch_size = batch_size // sequential_computation_steps
  final_grad = jax.tree.map(jnp.zeros_like, params)
  for i in range(sequential_computation_steps):
    a_i = jnp.zeros_like(mini_batch_size)
    g_i = jax.tree.map(jnp.zeros_like, params)
    anchors = batch_input_pairs[
        i * mini_batch_size : (i + 1) * mini_batch_size, 0
    ]
    for j in range(sequential_computation_steps):
      # We process the diagonal at the end.
      if i == j:
        continue
      positives = batch_input_pairs[
          j * mini_batch_size : (j + 1) * mini_batch_size, 1
      ]
      step_batch = jnp.stack([anchors, positives], axis=1)
      cosine_values, clipped_cosine_grads = (
          compute_batch_cosine_values_and_clipped_gradients(
              step_batch, forward_fn, params, key, l2_norm_clip=l2_norm_clip
          )
      )
      scaled_cosine_values = cosine_values / temperature
      scaled_clipped_cosine_grads = jax.tree.map(
          lambda x: x / temperature, clipped_cosine_grads
      )
      numerator, denominator = compute_partial_contrastive_loss_gradient(
          scaled_cosine_values, scaled_clipped_cosine_grads
      )
      a_i += denominator
      g_i = jax.tree.map(lambda a, b: a + b, numerator, g_i)

    positives = batch_input_pairs[
        i * mini_batch_size : (i + 1) * mini_batch_size, 1
    ]
    step_batch = jnp.stack([anchors, positives], axis=1)

    cosine_values, clipped_cosine_grads = (
        compute_batch_cosine_values_and_clipped_gradients(
            step_batch, forward_fn, params, key, l2_norm_clip=l2_norm_clip
        )
    )
    scaled_cosine_values = cosine_values / temperature
    scaled_clipped_cosine_grads = jax.tree.map(
        lambda x: x / temperature, clipped_cosine_grads
    )
    numerator, denominator = compute_partial_contrastive_loss_gradient(
        scaled_cosine_values, scaled_clipped_cosine_grads
    )

    diagonal = jax.tree.map(
        lambda grad: jnp.einsum("ii...->i...", grad),
        scaled_clipped_cosine_grads,
    )[0]

    a_i += denominator
    g_i = jax.tree.map(lambda a, b: a + b, numerator, g_i)

    second_term = jax.tree.map(
        lambda num_leave: jnp.einsum("i...,i->i...", num_leave, 1 / a_i),  # pylint: disable=cell-var-from-loop
        g_i,
    )

    final_g_i = jax.tree.map(
        lambda a, b: jnp.einsum("i...->...", a + b), diagonal, second_term
    )

    final_grad = jax.tree.map(lambda a, b: a + b, final_g_i, final_grad)

  return jax.tree.map(lambda x: -x / batch_size, final_grad)


def compute_contrastive_loss_gradient_sensitivity(
    l2_norm_clip, batch_size, similarity_bound
):
  numerator_term = jnp.exp(2 * similarity_bound)
  first_bound = 2 * numerator_term / (batch_size + numerator_term - 1)
  sensitivity = l2_norm_clip * (1 / batch_size + min([first_bound, 2]))
  return sensitivity


def compute_batch_cosine_values_and_clipped_gradients(
    batch_input_pairs,
    forward_fn,
    params,
    key,
    l2_norm_clip=None,
):
  """Computes the cosine similarities and their clipped gradients.

  This function computes the cosine similarity for all pair combinations in the
  batch of input pairs, and the corresponding gradient respect to params. If
  l2_norm_clip is not `None` it clips the gradient l2 norm to the specified
  value.

  Args:
    batch_input_pairs: array where the first dimension is the batch size and the
      second one differentiates anchors from positive samples.
    forward_fn: forward function to differentiate.
    params: trainable variables.
    key: random number generator.
    l2_norm_clip: if specified, float indicating the maximum l2_norm_clip.

  Returns:
    Tuple with an array of cosine similarities and a jax tree structure with
    jacobian vector products.
  """

  def compute_cosine_value_and_clipped_gradient(anchor, positive, key):
    # NOTE: Implement this with the following return call:
    #   return cosine_values, cosine_gradients
    diff_fn_at_x = functools.partial(forward_fn.apply, rng=key, x=anchor)
    diff_fn_at_y = functools.partial(forward_fn.apply, rng=key, x=positive)

    u, vjp_fn_at_x = jax.vjp(diff_fn_at_x, params)
    v, vjp_fn_at_y = jax.vjp(diff_fn_at_y, params)
    sum_of_vec_jacobian_products = jax.tree.map(
        lambda a, b: a + b, vjp_fn_at_x(v), vjp_fn_at_y(u)
    )
    # This step is necessary for Resnet that includes an extra dimension for
    # each sample.
    u = jnp.ravel(u)
    v = jnp.ravel(v)

    if l2_norm_clip:
      sum_of_vec_jacobian_products = compute_clipped_gradients(
          sum_of_vec_jacobian_products,
          l2_norm_clip,
      )
    return jnp.matmul(v.T, u), sum_of_vec_jacobian_products

  vectorized_products_over_v = jax.vmap(
      compute_cosine_value_and_clipped_gradient, in_axes=(None, 0, None)
  )

  # The output will be (batch_size, batch_size, ...) where entry (i,j,...)
  # corresponds to pairwise gradient of batch[i,0] and batch[j,1]
  vectorized_products_over_u_and_v = jax.jit(
      jax.vmap(vectorized_products_over_v, in_axes=(0, None, None))
  )

  _, new_key = jax.random.split(key)
  return vectorized_products_over_u_and_v(
      batch_input_pairs[:, 0], batch_input_pairs[:, 1], new_key
  )


def compute_clipped_gradients(gradients, cosine_norm_clip):
  """Computes the clipped gradients with respect to a specific clip value."""
  grads_norm = jnp.maximum(
      optax.global_norm(gradients), cosine_norm_clip
  )
  clipped_grads = jax.tree.map(
      lambda t: (t / grads_norm) * cosine_norm_clip, gradients
  )
  return clipped_grads


@jax.jit
def compute_contrastive_loss_gradient(
    cosine_values,
    clipped_cosine_gradients,
    temperature,
):
  """Computes the gradient of the constrastive loss using cosine scalars."""

  def loss_from_dot_prods_fn(clipped_gradient):
    exponent_cosine_values = jnp.exp(cosine_values / temperature)
    clipped_gradient /= temperature
    numerator = -jnp.einsum(
        "ij...,ij->i...", clipped_gradient, exponent_cosine_values
    )
    denominator = 1 / jnp.einsum("ij->i", exponent_cosine_values)
    second_term = jnp.einsum("i...,i->i...", numerator, denominator)
    diagonal = jnp.einsum("ii...->i...", clipped_gradient)
    gradient = jnp.einsum("i...->...", diagonal + second_term)
    return -gradient / cosine_values.shape[0]

  return jax.tree.map(loss_from_dot_prods_fn, clipped_cosine_gradients)[0]


@jax.jit
def compute_noised_gradients(
    gradients,
    l2_sensitivity,
    noise_multiplier,
    key,
):
  """Adds Gaussian noise to a set of averaged loss gradients."""
  stddev = l2_sensitivity * noise_multiplier
  _, treedef = jax.tree.flatten(gradients)
  grads_flat = treedef.flatten_up_to(gradients)
  noisy_grads_flat = [
      gd + stddev * jax.random.normal(key, gd.shape) for gd in grads_flat
  ]
  return jax.tree.unflatten(treedef, noisy_grads_flat)


def compute_dp_gradients(
    params,
    input_pairs,
    forward_fn,
    l2_norm_clip,
    noise_multiplier,
    temperature,
    key,
    sequential_computation_steps,
):
  """Wraps the above functions to generate differentially private gradients."""
  key, new_key = jax.random.split(key)
  batch_size = input_pairs.shape[0]
  similarity_bound = 1 / temperature
  if sequential_computation_steps > 1:
    cl_grads = compute_contrastive_loss_gradient_sequentially(
        batch_input_pairs=input_pairs,
        forward_fn=forward_fn,
        params=params,
        key=new_key,
        sequential_computation_steps=sequential_computation_steps,
        temperature=temperature,
        l2_norm_clip=l2_norm_clip,
    )
  else:
    cosine_sim_vals, clipped_cos_grads = (
        compute_batch_cosine_values_and_clipped_gradients(
            input_pairs,
            forward_fn,
            params,
            new_key,
            l2_norm_clip,
        )
    )
    cl_grads = compute_contrastive_loss_gradient(
        cosine_sim_vals,
        clipped_cos_grads,
        temperature,
    )
  if l2_norm_clip is None:
    return cl_grads

  sensitivity = compute_contrastive_loss_gradient_sensitivity(
      l2_norm_clip, batch_size, similarity_bound
  )
  _, new_key = jax.random.split(key)
  noisy_grads = compute_noised_gradients(
      cl_grads, sensitivity, noise_multiplier, new_key
  )
  return noisy_grads
