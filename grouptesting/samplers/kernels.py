# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""SMC Invariant Kernels / Proposals.

Follows more or less https://arxiv.org/pdf/1101.6037.pdf in addition to Gibbs
sampler.
"""
import collections
from typing import Dict

import gin
import jax
import jax.nn
import jax.numpy as np
import jax.scipy.special

from grouptesting import bayes


@jax.jit
def gibbs_kernel(rng,
                 particles,
                 rho,
                 log_posterior_params,
                 cycles = 2,
                 liu_modification = True):
  """Applies a (Liu modified) Gibbs kernel (with MH) update.

  Implements vanilla (sequential, looping over coordinates) Gibbs sampling.
  When
  The Liu variant comes from Jun Liu's remarks in
  https://academic.oup.com/biomet/article-abstract/83/3/681/241540?redirectedFrom=fulltext

  which essentially changes the acceptance of a flip from
    p(flip) / [ p(no flip) + p(flip) ]
  to
    min(1, p(flip) / p(no flip) )

  In other words, Liu's modification increases the probability to flip.

  Args:
   rng: np.ndarray<int> random key.
   particles: np.ndarray [n_particles,n_patients] plausible infections states.
   rho: float, scaling for posterior.
   log_posterior_params: Dict of parameters to compute log-posterior.
   cycles: the number of times we want of do Gibbs sampling.
   liu_modification : use or not Liu's modification.

  Returns:
   A np.array representing the new particles.
  """

  def gibbs_loop(i, rng_particles_log_posteriors):
    rng, particles, log_posteriors = rng_particles_log_posteriors
    i = i % num_patients
    # flip values at index i
    particles_flipped = jax.ops.index_update(particles, jax.ops.index[:, i],
                                             np.logical_not(particles[:, i]))
    # compute log_posterior of flipped particles
    log_posteriors_flipped_at_i = rho * bayes.log_posterior(
        particles_flipped, **log_posterior_params)
    # compute acceptance probability, depending on whether we use Liu mod.
    if liu_modification:
      log_proposal_ratio = log_posteriors_flipped_at_i - log_posteriors
    else:
      log_proposal_ratio = log_posteriors_flipped_at_i - np.logaddexp(
          log_posteriors_flipped_at_i, log_posteriors)
    # here the MH thresholding is implicitly done.
    rng, rng_unif = jax.random.split(rng, 2)
    random_values = jax.random.uniform(rng_unif, particles.shape[:1])
    flipped_at_i = np.log(random_values) < log_proposal_ratio
    selected_at_i = np.logical_xor(flipped_at_i, particles[:, i])
    particles = jax.ops.index_update(
        particles, jax.ops.index[:, i], selected_at_i)
    log_posteriors = np.where(
        flipped_at_i, log_posteriors_flipped_at_i, log_posteriors)
    return [rng, particles, log_posteriors]

  num_patients = particles.shape[1]
  log_posteriors = bayes.log_posterior(particles, **log_posterior_params)
  rng_particles = jax.lax.fori_loop(0, cycles * num_patients,
                                    gibbs_loop,
                                    [rng, particles, log_posteriors])
  # TODO(cuturi) : might be relevant to forward log_posterior_particles
  return rng_particles[1]


@gin.configurable
class Gibbs:
  """A Gibbs sampler."""

  def __init__(self,
               cycles = 2,
               liu_modification = False):
    self.cycles = cycles
    self.liu_modification = liu_modification
    self.model = None

  # TODO(oliviert): use jit here with `@partial(jit, static_argnums=(0,))`
  def __call__(self, rng, particles, rho, log_posterior_params):
    return gibbs_kernel(rng, particles, rho, log_posterior_params, self.cycles)

  def fit_model(self, particle_weights, particles):
    """Because Gibbs sampler do not use any model, we return nothing."""
    return


@gin.configurable
class Chopin:
  """SMC kernel as described in https://arxiv.org/pdf/1101.6037.pdf ."""

  def __init__(self, sparse_model_lr=True):
    self.sparse_model_lr = sparse_model_lr

  def __call__(self,
               rng,
               particles,
               rho,
               log_posterior_params):
    """Call carries out procedures 4 in https://arxiv.org/pdf/1101.6037.pdf.

    One expects that fit_model has been called right before to store the model
    in self.model

    Args:

     rng: np.ndarray<int> random key
     particles: np.ndarray [n_particles,n_patients] plausible infections states
     rho: float, scaling for posterior.
     log_posterior_params: Dict of parameters to compute log-posterior.

    Returns:
     A np.ndarray representing the new particles.
    """
    rngs = jax.random.split(rng, 2)
    n_samples = particles.shape[0]

    proposed, logprop_proposed, logprop_particles = self.sample_from_model(
        rngs[0], particles)
    llparticles = rho * bayes.log_posterior(particles, **log_posterior_params)
    llproposed = rho * bayes.log_posterior(proposed, **log_posterior_params)
    logratio = llproposed - llparticles + logprop_particles - logprop_proposed
    p_replacement = np.minimum(np.exp(logratio), 1)
    replacement = (
        jax.random.uniform(rngs[1], shape=(n_samples,)) < p_replacement)
    not_replacement = np.logical_not(replacement)
    return (replacement[:, np.newaxis] * proposed +
            not_replacement[:, np.newaxis] * particles)

  def sample_from_model(self, rng, particles):
    """proposes new particles using logistic model as in Chopin paper."""
    n_samples, n_patients = particles.shape
    params = self.model
    # store samples proposals
    proposed_particles = np.zeros(particles.shape, dtype=bool)
    # and log posterior
    lpproposed_particles = np.zeros((n_samples,))
    for i in range(n_patients):
      rng, rng_uni = jax.random.split(rng, 2)
      q = jax.nn.log_sigmoid(params[i, i] + np.sum(
          params[i, :i][np.newaxis, :] * proposed_particles[:, :i], axis=-1))
      qbar = np.log(1 - np.exp(q))
      sample = np.log(jax.random.uniform(rng_uni, shape=(n_samples,))) < q
      proposed_particles = jax.ops.index_update(proposed_particles,
                                                jax.ops.index[:, i], sample)
      lpproposed_particles += proposed_particles[:, i] * (q - qbar) + qbar

    # Computes all log posteriors for particles in one go.
    q = np.transpose(
        jax.nn.log_sigmoid(
            np.matmul(params, np.transpose(particles)) +
            np.diag(params)[:, np.newaxis]))
    qbar = np.log(1 - np.exp(q))
    lpparticles = np.sum(particles * (q - qbar) + qbar, axis=-1)
    return proposed_particles, lpproposed_particles, lpparticles

  def fit_model(self, particle_weights,
                particles):
    """Fits a binary model using weighted particles.

    The model will be a sparse lower triangular logistic regression as in
    Procedure 5 from
    https://arxiv.org/pdf/1101.6037.pdf

    Args:
      particle_weights: a np.array<float> of simplicial weights
      particles: np.array<bool>[groups, n_patients]

    Returns:
     A np.array<float>[n_patients, n_patients] model.
    """
    n_groups, n_patients = particles.shape
    model = np.zeros((n_patients, n_patients))
    eps = 1e-5
    # keep track of basic stats
    xbar = (1 - eps) * np.sum(
        particle_weights[:, np.newaxis] * particles, axis=0) + eps * 0.5
    xcov = np.matmul(
        np.transpose(particles), particle_weights[:, np.newaxis] * particles)
    xb1mxb = xbar * (1.0 - xbar)
    cov_matrix = (xcov - xbar[:, np.newaxis] * xbar[np.newaxis, :]) / np.sqrt(
        xb1mxb[:, np.newaxis] * xb1mxb[np.newaxis, :])

    # TODO(oliviert): turn this into parameters.
    eps = 0.01
    delta = 0.05
    indices_model = np.logical_and(xbar > eps, xbar < 1 - eps)
    indices_single = np.logical_or(xbar <= eps, xbar >= 1 - eps)
    # no regression for first variable
    indices_single = jax.ops.index_update(indices_single, 0, True)
    indices_model = jax.ops.index_update(indices_model, 0, False)

    # look for sparse blocks of variables to regress on others
    if self.sparse_model_lr:
      regressed, regressor = np.where(np.abs(cov_matrix) > delta)
      dic_regressors = collections.defaultdict(list)
      for i, j in zip(regressed, regressor):
        if j < i:
          dic_regressors[i].append(j)

    # Where there exists cross-correlation we estimate a model
    # TODO(cuturi) : switch to predefined number of regressors (i.e. top k
    # corellated variables. From kth patient we can then jit this regression.
    for i in np.where(indices_model)[0]:
      if self.sparse_model_lr:
        indices_i = dic_regressors[i]
      else:
        indices_i = list(range(i))

      regressors = np.concatenate(
          (particles[:, indices_i], np.ones((n_groups, 1))), axis=-1)
      y = particles[:, i]

      # initialize loop
      # TODO(oliviert): turn those hard coded constants into parameters
      b = np.zeros((regressors.shape[1],))
      diff = 1e10
      iterations = 0
      reg = .05

      while diff > 1e-2 and iterations < 30:
        iterations += 1
        regressorsb = np.dot(regressors, b)
        p = jax.scipy.special.expit(regressorsb)
        q = p * (1 - p)
        cov = np.matmul(
            particle_weights[np.newaxis, :] * q[np.newaxis, :] *
            np.transpose(regressors), regressors)
        cov = cov + reg * np.eye(len(indices_i) + 1)
        c = np.dot(
            np.transpose(regressors) * particle_weights[np.newaxis, :],
            q * regressorsb + y - p)
        bnew = np.linalg.solve(cov, c)
        diff = np.sum((bnew - b)**2)
        b = bnew
      # add constant, to list of indices, to be stored in [i,i]
      indices_i.append(i)
      # update line i of model
      model = jax.ops.index_update(model,
                                   jax.ops.index[i, np.asarray(indices_i)],
                                   bnew)

    # Where there are no cross-correlations, or posterior is very peaked,
    # we flip randomly and indvidually
    v = np.zeros((n_patients,))
    v = jax.ops.index_update(v, jax.ops.index[indices_single],
                             jax.scipy.special.logit(xbar[indices_single]))
    model = model + np.diag(v)
    self.model = model


@gin.configurable
class ChopinGibbs(Chopin):
  """Defines a kernel that does first Chopin kernel moves then Gibbs."""

  def __call__(self, rng, particles, rho, log_posterior_params):
    rngs = jax.random.split(rng, 2)
    new_particles = super().__call__(rngs[0], particles, rho,
                                     log_posterior_params)
    new_particles = gibbs_kernel(rngs[1], new_particles, rho,
                                 log_posterior_params)
    return new_particles
