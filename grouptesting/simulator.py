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
"""Simulates an environment to try Group testing Strategies."""

import time
from typing import Iterable, Union

from absl import logging
import gin
import jax
import jax.numpy as np
import numpy as onp
import pandas as pd

from grouptesting import metrics
from grouptesting import policy as group_policy
from grouptesting import state
from grouptesting import wet_lab
from grouptesting.samplers import loopy_belief_propagation as lbp
from grouptesting.samplers import sequential_monte_carlo as smc


@gin.configurable
class Simulator(object):
  """Runs Simulator to assess performance of group testing strategy.

  A simulator takes a wetlab that returns can test patients, a policy to select
  groups, a sampler to estimate the posterior probability plus some parameters,
  assesses the quality of a policy w.r.t. the conditions of the simulation.

  Attributes:
   metrics: the current metrics of the simulation.
   state: the current state.State of what have been tested in the past.
   sampler: the main sampler of the simulator, since it can have two samplers,
    one only used to compute marginals.
   marginal: the current best guess about the marginal probabilities of each
    patient.
   is_over: (bool) True if the simulation is over, else otherwise.
  """

  def __init__(self,
               workdir = None,
               wetlab=wet_lab.WetLab(),
               policy=group_policy.Dorfman(),
               sampler=smc.SmcSampler(),
               cheap_sampler=lbp.LbpSampler(),
               num_simulations = 1,
               num_tests_per_cycle = 10,
               max_test_cycles = 5,
               max_group_size = 8,
               prior_specificity = 0.97,
               prior_sensitivity = 0.85,
               prior_infection_rate = 0.05,
               metrics_cls=metrics.Metrics,
               export_metrics_every = 5):
    """Initializes simulation.

    Args:
      workdir: where results will be stored.
      wetlab: WetLab objet tasked with producing test results given groups.
      policy: group testing policy, a sequence of algorithms tasked with
        choosing groups to test. Can be adaptive to test environment (spec/sens)
        of tests. Can be adaptive to previously tested groups. Can leverage
        samplers to build information on what are the most likely disease status
        among patients.
      sampler: sampler that produces a posterior approximation. Instantiated by
        default to SmcSampler that resamples at each iteration to fix cases
        where the LBP sampler does not quite work the way it should.
      cheap_sampler: LBP object to yield cheap approximation of marginal.
      num_simulations: number of simulations run consecutively. Here randomness
        can come from the group testing policy (if it uses randomness), as well
        as diseased label, if WetLab.freeze_diseased is False.
      num_tests_per_cycle: number of tests a testing machine can carry out in
        the next testing cycle.
      max_test_cycles: number of cycles in total that should be considered.
      max_group_size: maximal size of groups, how many individuals can be pooled
      prior_specificity: best guess one has prior to simulation of the testing
        device's specificity per group size (or for all sizes, if singleton).
      prior_sensitivity: best guess one has prior to simulation of the testing
        device's sensitivity per group size (or for all sizes, if singleton).
      prior_infection_rate: best guess of prior probability for patient to be
        infected (same for all patients, if singleton)
      metrics_cls: class of metrics object used to store results.
      export_metrics_every: frequency of exports to file when carrying our
        num_simulations results.
    """

    self._wetlab = wetlab
    self._policy = policy
    self._samplers = [cheap_sampler, sampler]

    self._max_test_cycles = max_test_cycles
    self._num_simulations = num_simulations
    self.state = state.State(self._wetlab.num_patients,
                             num_tests_per_cycle,
                             max_group_size,
                             prior_infection_rate,
                             prior_specificity,
                             prior_sensitivity)
    self.metrics = metrics_cls(
        workdir,
        self._num_simulations,
        self._max_test_cycles,
        self._wetlab.num_patients,
        self.state.num_tests_per_cycle)
    self._export_every = export_metrics_every

  def reset(self, rng):
    """Resets all objects used in experiment."""
    rngs = jax.random.split(rng, 2)
    self.state.reset()
    self._wetlab.reset(rngs[0])
    self._policy.reset()
    for sampler in self._samplers:
      sampler.reset()
    self._resample(rngs[1])

  @property
  def sampler(self):
    """Returns the main sampler (index 0 is for marginals only)."""
    return self._samplers[-1]

  def run(self, rngkey=None):
    """Runs sequentially all simulations."""
    if self._num_simulations > 1 and self._wetlab.freeze_diseased:
      logging.warning("Running several simulations with the exact same group of"
                      " patients. You might want to consider using the "
                      "freeze_diseased=False parameter to the WetLab.")
    rngkey = int(time.time()) if rngkey is None else rngkey
    rng = jax.random.PRNGKey(rngkey)
    for sim_idx in range(self._num_simulations):
      logging.debug("Starting experiment %s", sim_idx)
      rng, sim_rng = jax.random.split(rng)
      self.run_simulation(sim_rng, sim_idx)

      for i in range(self.state.curr_cycle, self._max_test_cycles):
        self._on_iteration_end(sim_idx, i)

      if (sim_idx % self._export_every == 0 or
          sim_idx == self._num_simulations - 1):
        self.metrics.export()

  @property
  def is_over(self):
    """Returns True if the simulation is over, False otherwise."""
    return (self.state.curr_cycle >= self._max_test_cycles or
            self.state.all_cleared)

  def run_simulation(self, rng, sim_idx):
    """Carries out up to max_test_cycles, or less if policy terminates."""
    rng, reset_rng = jax.random.split(rng)
    self.reset(reset_rng)
    while not self.is_over:
      rng, cycle_rng = jax.random.split(rng)
      self.run_one_cycle(cycle_rng, sim_idx)

  def run_one_cycle(self, rng, sim_idx):
    """The policy generates groups, wetlab tests them and the state is updated.

    At each testing cycle, policy is called to produce up to num_tests_per_cycle
    groups. They are then tested by wetlab, which produces (possibly noisy)
    test results. These tests are then used to update posterior estimates
    (both particle posterior approximation and marginal), that might
    be used by the policy upon the next cycle. Results (notably the marginal
    approximation, which helps keep track of who is likely to be infected) are
    logged on file.

    Args:
     rng: the random key.
     sim_idx: (int) the current index of the simulation.
    """
    rngs = jax.random.split(rng, 3)
    new_groups = self.get_new_groups(rngs[0])
    new_tests = self._wetlab.group_tests_outputs(rngs[1], new_groups)
    self.process_tests_results(rngs[2], new_tests)
    self._on_iteration_end(sim_idx, self.state.curr_cycle, new_groups)
    self.state.curr_cycle += 1

  @property
  def marginal(self):
    """Returns our best guess on the current probability of each patient."""
    if self.state.marginals:
      return self.state.marginals[0]
    else:
      return self.sampler.marginal

  def process_tests_results(self, rng, new_tests):
    """What to do when receiving the tests results."""
    self.state.add_test_results(new_tests)
    self._resample(rng)

  def get_new_groups(self, rng):
    if self.state.extra_tests_needed > 0:
      self.state = self._policy.act(rng, self.state)
    return self.state.next_groups_to_test()

  def _resample(self, rng):
    """(Re)Samples posterior/marginals given past test results.

    Args:
      rng: random key

    Produces and examines first the marginal produced by LBP.
    If that marginal is not valid because LBP did not converge,
    or that posterior samples are needed in the next iteration of the
    simulator by a group selector, we compute both marginals
    and posterior  using the more expensive sampler.
    """
    # reset marginals
    self.state.marginals = []
    # compute marginal using a cheap LBP sampler.
    lbp_sampler = self._samplers[0]
    lbp_sampler.produce_sample(rng, self.state)
    # if marginal is valid (i.e. LBP has converged), append it to state.
    if not np.any(np.isnan(lbp_sampler.marginal)):
      self.state.marginals.append(lbp_sampler.marginal)
      self.state.update_particles(lbp_sampler)
    # if marginal has not converged, or expensive sampler is needed, use it.
    if (np.any(np.isnan(lbp_sampler.marginal)) or
        (self._policy.next_selector.NEEDS_POSTERIOR and
         self.state.extra_tests_needed > 0 and
         (self.state.curr_cycle == 0 or
          self.state.curr_cycle < self._max_test_cycles - 1))):
      sampler = self._samplers[1]
      sampler.produce_sample(rng, self.state)
      self.state.marginals.append(sampler.marginal)
      self.state.update_particles(sampler)

  def _on_iteration_end(self,
                        sim_idx,
                        cycle_idx,
                        new_groups=None,
                        new_tests=None):
    """Save metrics and ROC data at the end of one iteration."""

    lbp_conv = self._samplers[0].convergence_metric
    self._samplers[0].reset_convergence_metric()
    smc_conv = self._samplers[1].convergence_metric
    self._samplers[1].reset_convergence_metric()
    self.metrics.update(sim_idx,
                        cycle_idx,
                        self.state.marginals[0],
                        self._wetlab.diseased,
                        new_groups,
                        new_tests,
                        lbp_conv,
                        smc_conv)

  def interactive_loop(self, rngkey = None, show_fn=None):
    """Runs an interactive loop producing groups and asking for results.

    Args:
     rngkey: a random seed for this simulation.
     show_fn: a function that takes 3 arguments and returns None:
       - the group to be displayed as np.ndarray<bool>[num_groups, num_patients]
       - the current marginal as np.ndarray<bool>[num_patients]
       - a state.State of the simulator.
      If show_fn is None, then the 3 inputs are printed.
    """
    rngkey = int(time.time()) if rngkey is None else rngkey
    rng = jax.random.PRNGKey(rngkey)
    if show_fn is None:
      pd.options.display.float_format = "{:.2f}".format

    groups = None
    self.reset(rng)
    while not self.is_over:
      new_tests = None

      if groups is not None:
        positive_indices_str = input(
            f"Cycle {self.state.curr_cycle}: Enter positive groups, as csv: ")

        new_tests = onp.zeros((groups.shape[0],))
        if positive_indices_str:
          try:
            indices = [int(x) - 1 for x in positive_indices_str.split(",")]
            new_tests[indices] = True
          except (ValueError, IndexError):
            logging.warning("Wrong format. Expecting comma-separated values."
                            "e.g. 3,2,1,9")
            continue

      rng, *rngs = jax.random.split(rng, 3)
      if new_tests is not None:
        self.process_tests_results(rngs[0], new_tests)
      groups = self.get_new_groups(rngs[1])
      self.state.curr_cycle += 1

      if show_fn is not None:
        show_fn(groups, self.marginal, self.state)
      else:
        print("{0} Cycle {1} {0}".format("-" * 10, self.state.curr_cycle))
        df = pd.DataFrame(groups.astype(int),
                          index=1 + onp.arange(groups.shape[0]))
        df.columns = 1 + onp.arange(groups.shape[1])
        print("Groups\n", df)
        print(self.marginal.shape)
        print("Marginal\n", pd.DataFrame(self.marginal[onp.newaxis, :]))
