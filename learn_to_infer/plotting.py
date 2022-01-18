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

"""Plotting code."""

from functools import partial
import io

from . import ring_dist

import jax
from jax import vmap
import jax.numpy as jnp
import jax.scipy as jscipy
import matplotlib.pyplot as plt
import numpy as onp
from PIL import Image
from sklearn import datasets
from sklearn.mixture import GaussianMixture


def plot_to_numpy_image(mpl_plt):
  image_buf = io.BytesIO()
  mpl_plt.savefig(image_buf, format="png")
  image_buf.seek(0)
  img = onp.array(Image.open(image_buf))
  mpl_plt.close()
  return img[:, :, :-1]


def categorical_logpmf(x, logits):
  """Calculates the log pmf of a categorical distribution.

  Args:
    x: Integer in [0, k-1], the value of x to calculate the probability of.
    logits: A vector of size [K], the unnormalized log probabilities.

  Returns:
    A scalar float, the log probability of x.
  """
  log_probs = jnp.log(jax.nn.softmax(logits))
  return log_probs[x]


def log_joint(xs, cs, mus, covs, ws):
  """Evaluates the log joint probability of a GMM.

  The GMM is defined by the following sampling process:

  c_i ~ Categorical(w) for i=1,...,N
  x_i ~ Normal(mus[c_i], scale^2) for i=1,...,N

  Args:
    xs: A set of [N, D] values to compute the log probability of.
    cs: A shape [N] integer vector, the cluster assignments for each x.
    mus: A set of [K, D] mixture component means.
    covs: A set of [K, D, D] mixture componet covariance matrices.
    ws: A vector of shape [K], the mixture weights of the GMM. Need not be
      normalized.

  Returns:
    A [N] float vector, the log probabilities of each X.
  """
  log_p_c = vmap(categorical_logpmf, in_axes=(0, None))(cs, ws)
  log_p_x = vmap(jscipy.stats.multivariate_normal.logpdf)(xs, mus[cs], covs[cs])
  return log_p_c + log_p_x


@partial(jnp.vectorize, excluded=(1, 2, 3), signature=("(d)->()"))
def log_marginal(x, mus, covs, ws):
  """Computes the marginal probability of x under a GMM.

  Args:
    x: A shape [D] vector, the data point to compute p(x) for.
    mus: A [K,D] matrix, the K D-dimensional mixture component means.
    covs: A [K,D,D] matrix, the covariance matrices of the mixture components.
    ws: A shape [K] vector, the mixture weights.

  Returns:
    p(x), a float scalar.
  """
  K = mus.shape[0]
  cs = jnp.arange(K)[:, jnp.newaxis]
  x = x[jnp.newaxis, :]
  log_ps = vmap(
      log_joint, in_axes=(None, 0, None, None, None))(x, cs, mus, covs, ws)
  return jscipy.special.logsumexp(log_ps)


def plot_gmms(xs, num_modes, true_cs, true_params, pred_cs, pred_params, em_cs,
              em_params):
  true_means, true_covs, true_weights = true_params
  pred_means, pred_covs, pred_weights = pred_params
  em_means, em_covs, em_weights = em_params
  fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(14, 4))
  plot_gmm_on_ax(ax[0], xs, num_modes, true_cs, true_means, true_covs,
                 true_weights)
  ax[0].set_title("True Clustering")
  plot_gmm_on_ax(ax[1], xs, num_modes, pred_cs, pred_means, pred_covs,
                 pred_weights)
  ax[1].set_title("Predicted Clustering")
  plot_gmm_on_ax(ax[2], xs, num_modes, em_cs, em_means, em_covs, em_weights)
  ax[2].set_title("EM Clustering")
  return fig


def plot_rings(xs, num_modes, true_cs, true_params, pred_cs, pred_params):
  true_r_means, true_r_scales, true_centers, true_log_weights = true_params
  pred_r_means, pred_r_scales, pred_centers, pred_log_weights = pred_params
  fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 4))

  @partial(jnp.vectorize, signature="(2)->()")
  def true_log_p(x):
    return ring_dist.ring_mixture_log_p(x, true_r_means, true_r_scales,
                                        true_centers, true_log_weights)

  @partial(jnp.vectorize, signature="(2)->()")
  def pred_log_p(x):
    return ring_dist.ring_mixture_log_p(x, pred_r_means, pred_r_scales,
                                        pred_centers, pred_log_weights)

  plot_density_and_points_on_ax(ax[0], xs, num_modes, true_cs,
                                true_centers, true_log_p)
  ax[0].set_title("True Clustering")
  plot_density_and_points_on_ax(ax[1], xs, num_modes, pred_cs,
                                pred_centers, pred_log_p)
  ax[1].set_title("Predicted Clustering")
  return fig


def plot_points_on_ax(ax, xs, num_modes, cs, means):

  for i in range(num_modes):
    i_xs = xs[cs == i]
    if len(i_xs > 0):
      ax.plot(i_xs[:, 0], i_xs[:, 1], "o")
    if means is not None:
      ax.plot(means[i, 0], means[i, 1], "r*")


def plot_density_and_points_on_ax(ax, xs, num_modes, cs, means, log_p_fn):
  plot_points_on_ax(ax, xs, num_modes, cs, means)
  y_bounds = ax.get_ylim()
  x_bounds = ax.get_xlim()
  X, Y, P = evaluate_density(log_p_fn, (x_bounds, y_bounds), 100)
  ax.contour(X, Y, P, alpha=0.6)


def plot_gmm_on_ax(ax, xs, num_modes, cs, means, covs, weights):
  means = means[:num_modes]
  covs = covs[:num_modes]
  weights = weights[:num_modes]
  log_p_fn = lambda x: log_marginal(x, means, covs, weights)
  return plot_density_and_points_on_ax(ax, xs, num_modes, cs, means, log_p_fn)


def evaluate_density(log_p_fn, bounds, num_points):
  xs = jnp.linspace(bounds[0][0], bounds[0][1], num=num_points)
  ys = jnp.linspace(bounds[1][0], bounds[1][1], num=num_points)
  X, Y = jnp.meshgrid(xs, ys)
  xs = jnp.stack([X, Y], axis=-1)
  log_ps = log_p_fn(xs)
  return X, Y, jnp.exp(log_ps)


def fit_em(xs, k):
  gmm = GaussianMixture(n_components=k, covariance_type="full")
  labels = jnp.array(gmm.fit_predict(xs))
  mus = jnp.array(gmm.means_)
  covs = jnp.array(gmm.covariances_)
  weights = jnp.log(gmm.weights_)
  return labels, (mus, covs, weights)


def make_comparison_gmm_datasets():
  onp.random.seed(0)

  n_samples = 1500
  _ = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
  _ = datasets.make_moons(n_samples=n_samples, noise=.05)
  blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
  no_structure = onp.random.rand(n_samples, 2), None

  # Anisotropicly distributed data
  random_state = 170
  X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
  transformation = [[0.6, -0.6], [-0.4, 0.8]]
  X_aniso = onp.dot(X, transformation)
  aniso = (X_aniso, y)

  # blobs with varied variances
  varied = datasets.make_blobs(n_samples=n_samples,
                               cluster_std=[1.0, 2.5, 0.5],
                               random_state=random_state)
  return varied, aniso, blobs, no_structure


def make_comparison_bananas():
  onp.random.seed(0)

  n_samples = 1500
  _ = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
  noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
  return noisy_moons


def make_comparison_rings():
  onp.random.seed(0)

  n_samples = 1500
  noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                        noise=.05)
  return noisy_circles


def plot_comparison_gmm(xs, true_cs, pred_cs, pred_params):
  pred_means, pred_covs, pred_weights = pred_params
  fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 4))
  plot_points_on_ax(ax[0], xs, 3, true_cs, None)
  ax[0].set_title("True Clustering")
  plot_gmm_on_ax(ax[1], xs, 3, pred_cs, pred_means, pred_covs, pred_weights)
  ax[1].set_title("Predicted Clustering")
  return fig


def plot_comparison_rings(xs, true_cs, pred_cs, pred_params):
  r_means, r_scales, centers, log_weights = pred_params
  fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 4))
  plot_points_on_ax(ax[0], xs, 3, true_cs, None)
  ax[0].set_title("True Clustering")

  @partial(jnp.vectorize, signature="(2)->()")
  def pred_log_p(x):
    return ring_dist.ring_mixture_log_p(
        x, r_means, r_scales, centers, log_weights)

  plot_density_and_points_on_ax(ax[1], xs, 2, pred_cs, centers, pred_log_p)
  ax[1].set_title("Predicted Clustering")
  return fig
