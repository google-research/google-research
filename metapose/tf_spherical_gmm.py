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

"""GPU-accelerated weighted EM for fitting spherical GMMs to heatmaps.

This implementation turns a heatmap into a flat array of 2D positions and
weights (probabilities) and applies weighted EM algorithm to it as described
in "Gaussian Mixture Estimation from Weighted Samples" by Frisch & Hanebeck.

To avoid re-tracing `fast_fit_gmm()` for different values of `limits`, to
achieve best performance, and parallelize computations, compile a specialized
fitting function via `get_specialized_parallel_gmm_fit_fn()` just once.

Example - fitting 17 heatmaps in parallel 88x64 each:

```
  fit_fn = get_specialized_parallel_fit_gmm((17, 88, 64), n_comp, n_steps)
  for ht in heatmaps:
    gmm = fit_fn(ht, [[0.0, 0.0], [1.0, 1.0]])
```

"""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def generate_weighted_samples(heatmap, limits):
  """Transforms a 2D heatmap into 2D samples weighted by probabilities."""
  res = tf.shape(heatmap)
  unnorm_weights = tf.reshape(heatmap, (-1,))

  mesh = tf.meshgrid(tf.linspace(0, 1, res[1]), tf.linspace(0, 1, res[0]))
  samples_grid = tf.reshape(tf.stack(mesh, axis=-1), (-1, 2))[:, ::-1]

  scale = tf.convert_to_tensor([limits[0][1] - limits[0][0],
                                limits[1][1] - limits[1][0]])
  shift = tf.convert_to_tensor([limits[0][0], limits[1][0]])
  samples_grid, res, scale, shift = [
      tf.cast(x, tf.float32) for x in [samples_grid, res, scale, shift]]
  samples = samples_grid * scale + shift
  weights = unnorm_weights / tf.reduce_sum(unnorm_weights)
  samples, weights = [tf.cast(x, tf.float32) for x in [samples, weights]]
  return samples, weights


def get_gaussian_pdf_equal_split_points(k):
  tf.assert_greater(k, 2)
  lp = tf.cast(tf.linspace(0, 1, k+1), dtype=tf.float32)
  pk = tfd.Normal(0.0, 1.0).quantile(lp[1:-1])
  qk_pre = (pk[:-1] + pk[1:])/2
  qk_first = 2*pk[0] - qk_pre[0]
  qk_last = 2*pk[-1] - qk_pre[-1]
  qk = tf.concat([[qk_first], qk_pre, [qk_last]], axis=0)
  return qk


def init_means_with_pca(samples, weights, k_comp, init_stddev):
  """Produces initial means for GMM component - along the first eigenvector."""
  del init_stddev
  data_mean = tf.reduce_sum(samples * weights[:, None], axis=0)
  cent_data = samples - data_mean
  # same as diag(weights) @ data
  diag_at_cent_data = weights[:, None] * cent_data
  eigw, eigv = tf.linalg.eigh(tf.transpose(cent_data) @ diag_at_cent_data)
  prim_vec = eigv[:, -1] * tf.math.sqrt(eigw[-1])

  if k_comp > 2:
    linspace = get_gaussian_pdf_equal_split_points(k_comp)
  elif k_comp == 2:
    linspace = tf.constant([-0.3, 0.3], dtype=prim_vec.dtype)
  else:
    linspace = tf.constant([0.0], dtype=prim_vec.dtype)

  init_centers = data_mean[None, :] + linspace[:, None] * prim_vec[None, :]
  return init_centers


def compute_assignment(samples, mus, sigmas, pis):
  """Computes the point-component assignment (E-step)."""
  dist = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(probs=pis),
      components_distribution=tfd.MultivariateNormalDiag(
          loc=mus, scale_diag=tf.repeat(sigmas[:, None], 2, axis=1)))

  x = dist._pad_sample_dims(samples)  # pylint: disable=protected-access
  log_prob_x = dist.components_distribution.log_prob(x)  # [N, 1, k]
  log_mix_prob = tf.math.log_softmax(
      dist.mixture_distribution.logits_parameter(), axis=-1)  # [N, k]
  logp = log_prob_x + log_mix_prob
  nnorm_probs = tf.math.exp(logp)
  probs = nnorm_probs / (
      tf.reduce_sum(nnorm_probs, axis=1, keepdims=True) + 1e-15)
  return probs


def estiamte_params(samples, sample_weights, assign_probs):
  """Estimate updated mixture parameters from weighted samples (M-step)."""
  total_sample_comp_weights = sample_weights[:, None] * assign_probs  # [N, k]

  new_nn_pis = sample_weights[:, None] * assign_probs  # [N, k]
  new_pis = tf.reduce_sum(new_nn_pis, axis=0) / tf.reduce_sum(new_nn_pis)

  # [N, k, d]
  weighted_samples = samples[:, None, :] * total_sample_comp_weights[:, :, None]
  new_means = (
      tf.reduce_sum(weighted_samples, axis=0)
      / (tf.reduce_sum(total_sample_comp_weights, axis=0)[:, None] + 1e-15))
  new_diffs = samples[:, None, :] - new_means[None, :, :]  # [N, k, d]
  new_dist_sq = tf.reduce_mean(tf.math.square(new_diffs), axis=2)  # [N, k]

  new_sigmas_sq = (
      tf.reduce_sum(new_dist_sq * total_sample_comp_weights, axis=0)
      / (tf.reduce_sum(total_sample_comp_weights, axis=0) + 1e-15)
  )
  new_sigmas = tf.math.sqrt(new_sigmas_sq)
  new_pis, new_means, new_sigmas = [
      tf.cast(x, tf.float32) for x in [new_pis, new_means, new_sigmas]]
  return new_pis, new_means, new_sigmas


def compensate_heatmap_sigma(sigma, res, limits):
  scale = tf.convert_to_tensor([limits[0][1] - limits[0][0],
                                limits[1][1] - limits[1][0]])
  res = tf.cast(res, tf.float32)
  pixel_noise_multiplier = 0.5
  noise_level = tf.reduce_max(scale / res) * pixel_noise_multiplier
  return tf.math.sqrt(sigma**2 + noise_level**2)


def fast_fit_gmm(heatmap, limits, k_comp, steps=5, init_stddev=2.0):
  """Performs weighted EM clustering over the heatmap."""
  samples, weights = generate_weighted_samples(heatmap, limits)
  mus = init_means_with_pca(samples, weights, k_comp, init_stddev)
  sigmas = tf.ones(k_comp)
  pis = tf.ones(k_comp, dtype=tf.float32) / tf.cast(k_comp, tf.float32)

  samples, weights, mus, sigmas, pis = [
      tf.cast(x, tf.float32) for x in [samples, weights, mus, sigmas, pis]]

  for _ in range(steps):
    assign_probs = compute_assignment(samples, mus, sigmas, pis)
    pis, mus, sigmas = estiamte_params(samples, weights, assign_probs)
    sigmas = compensate_heatmap_sigma(sigmas, heatmap.shape, limits)

  return pis, mus, sigmas


def get_specialized_parallel_gmm_fit_fn(heatmap_shape, k_comp, steps):
  """Produces a vectorized specialization for a particular heatmap shape."""
  def _func(heatmaps, limits):
    return tf.vectorized_map(
        lambda x: fast_fit_gmm(x, limits, k_comp, steps),
        heatmaps)

  input_signature = (
      tf.TensorSpec(shape=list(heatmap_shape), dtype=tf.float32),
      tf.TensorSpec(shape=[2, 2], dtype=tf.float32))

  fit_gmm_fn = tf.function(input_signature=input_signature)(_func)
  return fit_gmm_fn
