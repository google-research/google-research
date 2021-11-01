# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Shared utilities for experiments in the paper."""

import dataclasses
import functools
import sys
import time
import types
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from gumbel_max_causal_gadgets import coupling_util

NDArray = PRNGKey = Any


@dataclasses.dataclass(eq=False)
class CouplingExperimentConfig:
  """Configuration and helper object for coupling experiments.

  Attributes:
    name: Name of the experiment
    model: Model definition (either gadget 1 or 2)
    logit_pair_distribution_fn: Function that produces random pairs of logits
      from the distribution D, given a key.
    coupling_loss_matrix_fn: Function that produces a matrix of penalties for
      each counterfactual coupling pair, given the two logit vectors.
    inner_num_samples: How many (relaxed) samples from the coupling to draw for
      each pair of logits.
    batch_size: How many pairs of logits to compute losses for at a time.
    use_transpose: Whether to pass a `transpose` argument to one of the sampled
      pairs. Should be True if and only if the model is Gadget 1.
    tx: optax optimizer definition to use.
    num_steps: How many training steps to use.
    print_every: Minimum frequency at which to print training progress. Always
      prints at powers of 2 regardless of this value.
    metadata: Arbitrary metadata to associate with this experiment.
  """
  name: str
  model: Any
  logit_pair_distribution_fn: Callable[[PRNGKey], Tuple[NDArray, NDArray]]
  coupling_loss_matrix_fn: Callable[[NDArray, NDArray], NDArray]
  inner_num_samples: int
  batch_size: int
  use_transpose: bool
  tx: Any
  num_steps: int
  print_every: int = 100
  metadata: Any = None

  def loss_and_metrics_one_pair(self, params, rng):
    """Samples a pair of logits, and computes loss and metrics."""
    key_pq, key_samples = jax.random.split(rng)
    p_logits, q_logits = self.logit_pair_distribution_fn(key_pq)

    def sample_loss(key_sample):
      """Computes loss for a single sample of a relaxed pair of outcomes."""
      q_kwargs = dict(transpose=True) if self.use_transpose else {}
      soft_p = self.model.apply(
          params, p_logits, key_sample, method=self.model.sample_relaxed)
      soft_q = self.model.apply(
          params,
          q_logits,
          key_sample,
          method=self.model.sample_relaxed,
          **q_kwargs)
      coupling_loss_matrix = self.coupling_loss_matrix_fn(p_logits, q_logits)
      coupling_loss = jnp.sum(soft_p[:, None] * soft_q[None, :] *
                              coupling_loss_matrix)

      return coupling_loss

    loss_samples = jax.vmap(sample_loss)(
        jax.random.split(key_samples, self.inner_num_samples))
    loss = jnp.mean(loss_samples)
    return loss, {"loss": loss}

  @functools.partial(jax.jit, static_argnums=0)
  def opt_step(self, opt_state, params, rng):
    """Performs one training step."""

    def batch_loss(params, rng):
      stuff = jax.vmap(lambda rng: self.loss_and_metrics_one_pair(params, rng))(
          jax.random.split(rng, self.batch_size))
      return jax.tree_map(jnp.mean, stuff)

    grads, metrics = jax.grad(batch_loss, has_aux=True)(params, rng)
    updates, new_opt_state = self.tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    any_was_nan = jax.tree_util.tree_reduce(
        jnp.logical_or, jax.tree_map(lambda v: jnp.any(jnp.isnan(v)), grads))
    new_opt_state, new_params = jax.tree_multimap(
        lambda a, b: jnp.where(any_was_nan, a, b), (opt_state, params),
        (new_opt_state, new_params))
    return new_opt_state, new_params, metrics, grads, any_was_nan

  def train(self, rng):
    """Training loop entry point.

    Calling this method runs the experiment described by this config, and
    returns various results collected during training.

    Args:
      rng: PRNGKey to use to initialize model and draw training examples.

    Returns:
      types.SimpleNamespace containing various results. Of importance:
        finished_reason: The reason we stopped training.
        params: The parameters we learned.
    """
    # pylint: disable=possibly-unused-variable
    rng, init_key = jax.random.split(rng)
    params = self.model.init(init_key, jnp.zeros([self.model.S_dim]))
    opt_state = self.tx.init(params)
    start_time = time.time()

    count_since_reset = 0
    all_metrics = []
    try:
      i = 0
      while i < self.num_steps:
        rng, key = jax.random.split(rng)
        # Pass the inputs in and take a gradient step.
        opt_state, params, metrics, grads, bad = self.opt_step(
            opt_state, params, key)
        all_metrics.append(jax.tree_map(np.array, metrics))
        if bad:
          finished_reason = "nan"
          return types.SimpleNamespace(**locals())

        count_since_reset += 1
        if i % self.print_every == 0 or np.remainder(np.log2(i), 1) == 0:
          now = time.time()
          rate = count_since_reset / (now - start_time)
          start_time = now
          count_since_reset = 0
          print(f"{i} [{rate}/s]:", jax.tree_map(float, metrics))
          sys.stdout.flush()
          time.sleep(0.02)
        i += 1

    except KeyboardInterrupt:
      finished_reason = "interrupt"
      return types.SimpleNamespace(**locals())

    finished_reason = "done"
    (opt_state, params) = jax.tree_map(np.array, (opt_state, params))
    return types.SimpleNamespace(**locals())
    # pylint: enable=possibly-unused-variable

  def build_sampler(self, params):
    """Helper to build a joint sampler function for the model."""

    def sampler(logits_1, logits_2, key):
      q_kwargs = dict(transpose=True) if self.use_transpose else {}
      x = self.model.bind(params).sample(logits_1, key)
      y = self.model.bind(params).sample(logits_2, key, **q_kwargs)
      return jnp.zeros([10, 10]).at[x, y].set(1.)

    return sampler


def get_coupling_estimates(experiments,
                           results,
                           seed,
                           logits_1=None,
                           logits_2=None,
                           num_joint_samples=10_000_000,
                           logit_kwargs=None):
  """Computes couplings for a collection of experiments.

  All experiments should have the same logit_pair_distribution_fn.

  Args:
    experiments: List of experiments to evaluate.
    results: List of results, produced by calling `train` on each experiment.
    seed: Seed to use when estimating the coupling.
    logits_1: Optional logits. If not provided, uses logit_pair_distribution_fn.
    logits_2: Optional logits. If not provided, uses logit_pair_distribution_fn.
    num_joint_samples: How many samples to draw from the coupling when
      estimating it.
    logit_kwargs: Any keyword arguments that should be passed to the logit pair
      generator.

  Returns:
    (logits_1, logits_2), couplings
    where `couplings` is a dictionary whose keys include each of the experiments
    along with baselines, and the values are coupling matrices.
  """
  logits_key, vis_key = jax.random.split(jax.random.PRNGKey(seed))

  if logits_1 is None and logits_2 is None:
    logits_1, logits_2 = experiments[0].logit_pair_distribution_fn(
        logits_key, **(logit_kwargs or {}))

  logits_1 -= jax.scipy.special.logsumexp(logits_1)
  logits_2 -= jax.scipy.special.logsumexp(logits_2)

  probs_1 = jnp.exp(logits_1)
  probs_2 = jnp.exp(logits_2)

  independent_coupling = probs_1[:, None] * probs_2[None, :]
  gumbel_max_estimate = coupling_util.joint_from_samples(
      coupling_util.gumbel_max_sampler,
      logits_1,
      logits_2,
      vis_key,
      num_joint_samples,
      loop_size=500)
  icdf = coupling_util.inverse_cdf_coupling(logits_1, logits_2)
  icdf_perm = coupling_util.permuted_inverse_cdf_coupling(logits_1, logits_2)

  couplings = {
      "Independent": independent_coupling,
      "ICDF": icdf,
      "ICDF (permuted)": icdf_perm,
      "Gumbel-max": gumbel_max_estimate,
  }
  for experiment, result in zip(experiments, results):
    couplings[experiment.name] = coupling_util.joint_from_samples(
        experiment.build_sampler(result.params),
        logits_1,
        logits_2,
        vis_key,
        num_joint_samples,
        loop_size=500)

  return (logits_1, logits_2), couplings


def compute_coupling_losses(experiments, logits_1, logits_2,
                            estimated_couplings):
  """Estimate losses for each experiment.

  All experiments should have the same coupling_loss_matrix_fn.

  Args:
    experiments: List of experiments to evaluate.
    logits_1: First set of logits.
    logits_2: Second set of logits,
    estimated_couplings: A dictionary whose values are coupling matrices.

  Returns:
    A dictionary with the same keys as estimated_couplings whose values are
    estimates of the loss for this coupling.
  """
  test_losses = {}
  for name, coupling in estimated_couplings.items():
    loss_value = jnp.sum(
        coupling * experiments[0].coupling_loss_matrix_fn(logits_1, logits_2))
    test_losses[name] = loss_value
  return test_losses


def visualize_coupling_experiments(loss_values, logits_1, logits_2, couplings):
  """Visualizes the couplings using matplotlib.

  Args:
    loss_values: Loss values from `compute_coupling_losses`
    logits_1: First set of logits.
    logits_2: Second set of logits,
    couplings: A dictionary whose values are coupling matrices.
  """
  ncols = 2 + len(couplings)
  _, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))
  axs[0].imshow(jnp.exp(logits_1)[:, None], vmin=0)
  axs[1].imshow(jnp.exp(logits_2)[None, :], vmin=0)
  for j, (name, coupling) in enumerate(couplings.items()):
    axs[j + 2].imshow(coupling, vmin=0)
    axs[j + 2].set_title(f"{name}:\n{loss_values[name]}")


def compute_and_visualize_coupling_experiments(experiments, results, seed,
                                               **kwargs):
  """Helper function to both compute couplings and visualize them."""
  (logits_1,
   logits_2), couplings = get_coupling_estimates(experiments, results, seed,
                                                 **kwargs)
  test_losses = compute_coupling_losses(experiments, logits_1, logits_2,
                                        couplings)
  visualize_coupling_experiments(test_losses, logits_1, logits_2, couplings)


def evaluate_joint(joint_sampler,
                   experiment,
                   seed,
                   num_pairs,
                   joint_correction_num_samples=None):
  """Evaluates a particular coupling sampler for a particular task.

  Args:
    joint_sampler: Function from (p_logits, q_logits, key) to an approximate
      coupling.
    experiment: Experiment that determines the logit pair distribution.
    seed: PRNGKey to use.
    num_pairs: How many pairs of logits to evaluate on.
    joint_correction_num_samples: Correction term for number of samples, used
      when estimating variance across samples for a single model. Not important
      for results in the paper.

  Returns:
    A summary string, along with average loss, standard error of average loss
    across logit pairs, and estimate of the variance of the loss viewed as a
    random variable over sampled counterfactual pairs. Note that only the
    average loss is used to compute the results in the paper; the others were
    used for early experiments.
  """
  rng = jax.random.PRNGKey(seed)

  def run_pair(key):
    k1, k2 = jax.random.split(key, 2)
    p_logits, q_logits = experiment.logit_pair_distribution_fn(k1)
    joint_estimate = joint_sampler(p_logits, q_logits, k2)
    coupling_loss_matrix = experiment.coupling_loss_matrix_fn(
        p_logits, q_logits)
    loss_average = jnp.sum(joint_estimate * coupling_loss_matrix)
    loss_inner_variance = jnp.sum(
        joint_estimate * jnp.square(coupling_loss_matrix - loss_average))
    if joint_correction_num_samples:
      n = joint_correction_num_samples
      loss_inner_variance = loss_inner_variance * n / (n - 1)
    return loss_average, loss_inner_variance

  pair_averages, pair_variances = jax.lax.map(run_pair,
                                              jax.random.split(rng, num_pairs))

  overall_average = jnp.mean(pair_averages)
  overall_average_stderr = jnp.std(pair_averages) / jnp.sqrt(num_pairs)
  overall_pair_std = jnp.sqrt(jnp.mean(pair_variances))
  # overall_pair_variance_stderr = jnp.std(pair_variances) / jnp.sqrt(num_pairs)

  summary = (f"average: {overall_average:.4f}, "
             f"inner st.dev.: +/- {overall_pair_std:.4}, "
             f"errorbars: +/- {overall_average_stderr:.4f}")

  return summary, overall_average, overall_average_stderr, overall_pair_std


def evaluate_experiment(experiment,
                        result,
                        seed,
                        num_pairs,
                        samples_per_pair,
                        loop_size=None):
  """Helper function to evaluate a single experiment."""
  joint_sampler = functools.partial(
      coupling_util.joint_from_samples,
      experiment.build_sampler(result.params),
      num_samples=samples_per_pair,
      loop_size=loop_size)
  return {
      experiment.name:
          evaluate_joint(
              joint_sampler,
              experiment,
              seed,
              num_pairs,
              joint_correction_num_samples=samples_per_pair)
  }


def evaluate_baselines(experiment,
                       seed,
                       num_pairs,
                       samples_per_pair,
                       loop_size=None):
  """Helper function to evaluate the set of baselines."""
  gumbel_max_joint_fn = functools.partial(
      coupling_util.joint_from_samples,
      coupling_util.gumbel_max_sampler,
      num_samples=samples_per_pair,
      loop_size=loop_size)
  return {
      "Independent":
          evaluate_joint(
              lambda p, q, _: coupling_util.independent_coupling(p, q),
              experiment, seed, num_pairs),
      "ICDF":
          evaluate_joint(
              lambda p, q, _: coupling_util.inverse_cdf_coupling(p, q),
              experiment, seed, num_pairs),
      "ICDF (permuted)":
          evaluate_joint(
              lambda p, q, _: coupling_util.permuted_inverse_cdf_coupling(p, q),
              experiment, seed, num_pairs),
      "Gumbel-max":
          evaluate_joint(
              gumbel_max_joint_fn,
              experiment,
              seed,
              num_pairs,
              joint_correction_num_samples=samples_per_pair),
  }


def evaluate_all(experiments,
                 results,
                 seed,
                 num_pairs,
                 samples_per_pair,
                 loop_size=None):
  """Helper function to evaluate all experiments and baselines."""
  eval_results = evaluate_baselines(
      experiments[0], seed, num_pairs, samples_per_pair, loop_size=None)
  for ex, res in zip(experiments, results):
    eval_results.update(
        evaluate_experiment(ex, res, seed, num_pairs, samples_per_pair,
                            loop_size))
  return eval_results
