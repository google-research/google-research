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

"""Computes bounds for DP-SGD with various batch samplers.

The source in this file is based on the following papers:

Title: How Private are DP-SGD implementations?
Authors: Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi,
         Amer Sinha, Chiyuan Zhang
Link: https://arxiv.org/abs/2403.17673

Title: Scalable DP-SGD: Shuffling vs. Poisson Subsampling
Authors: Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi,
         Amer Sinha, Chiyuan Zhang
Link: https://arxiv.org/abs/2411.04205

For deterministic batch samplers, the analysis uses the Gaussian mechanism, and
the implementation uses Gaussian CDFs (provided in scipy library). This analysis
is efficient and provides nearly exact bounds.

For Poisson batch samplers, the analysis uses the Poisson subsampled Gaussian
mechanism, and the implementation uses the dp_accounting library. The PLD
based analysis provides upper and lower bounds on the privacy parameters. The
RDP based analysis, while more efficient, only provides an upper bound.

For Shuffle batch samplers, the analysis uses a "cube" set to establish a lower
bound on the hockey stick divergence.
"""

import collections
import functools
import math
from typing import Callable, Sequence

import dp_accounting
from dp_accounting import pld
from dp_accounting import rdp
import numpy as np
from scipy import stats


def inverse_decreasing_function(
    function, value):
  """Returns the smallest x at which function(x) <= value."""
  # A heuristic initial guess of 10 is chosen. This only costs in efficiency.
  search_params = pld.common.BinarySearchParameters(0, np.inf, 10)
  value = pld.common.inverse_monotone_function(function, value, search_params)
  if value is None:
    raise ValueError(f'No input x found for {value=}.')
  return value


class DeterministicAccountant:
  """Privacy accountant for ABLQ with deterministic batch sampler."""

  def __init__(self):
    pass

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_deltas(self,
                 sigma,
                 epsilons,
                 num_epochs = 1):
    """Returns delta values for which (epsilon, delta)-DP is satisfied.

    Args:
      sigma: The scale of the Gaussian noise.
      epsilons: A list or numpy array of epsilon values for which to compute the
        hockey stick divergence.
      num_epochs: The number of epochs.

    Returns:
      A list of hockey stick divergence values corresponding to each epsilon.
    """
    sigma = sigma / np.sqrt(num_epochs)
    epsilons = np.atleast_1d(epsilons)
    upper_cdfs = stats.norm.cdf(0.5 / sigma - sigma * epsilons)
    lower_log_cdfs = stats.norm.logcdf(-0.5 / sigma - sigma * epsilons)
    return list(upper_cdfs - np.exp(epsilons + lower_log_cdfs))

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_epsilons(self,
                   sigma,
                   deltas,
                   num_epochs = 1):
    """Returns epsilon values for which (epsilon, delta)-DP is satisfied.

    Args:
      sigma: The scale of the Gaussian noise.
      deltas: The privacy parameters delta.
      num_epochs: The number of epochs.
    """
    delta_from_epsilon = lambda epsilon: self.get_deltas(
        sigma, (epsilon,), num_epochs)[0]
    return [
        inverse_decreasing_function(delta_from_epsilon, delta)
        for delta in deltas
    ]

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_sigma(self,
                epsilon,
                delta,
                num_epochs = 1):
    """Returns the scale of the Gaussian noise for (epsilon, delta)-DP."""
    delta_from_sigma = lambda sigma: self.get_deltas(
        sigma, (epsilon,), num_epochs)[0]
    return inverse_decreasing_function(delta_from_sigma, delta)


class PoissonPLDAccountant:
  """Privacy accountant for ABLQ with Poisson batch sampler using PLD analysis.

  Attributes:
    discretization: The discretization interval to be used in computing the
      privacy loss distribution.
    pessimistic_estimate: When True, an upper bound on the hockey stick
      divergence is returned. When False, a lower bound on the hockey stick
      divergence is returned.
  """

  def __init__(self,
               pessimistic_estimate = True):
    self.pessimistic_estimate = pessimistic_estimate

  def _get_pld(
      self,
      sigma,
      num_compositions,
      sampling_prob,
      discretization = 1e-3,
  ):
    """Returns the composed PLD for ABLQ_P.

    Args:
      sigma: The scale of the Gaussian noise.
      num_compositions: The number of compositions.
      sampling_prob: The sampling probability in each step.
      discretization: The discretization interval to be used in computing the
        privacy loss distribution.
    """
    pl_dist = pld.privacy_loss_distribution.from_gaussian_mechanism(
        sigma,
        pessimistic_estimate=self.pessimistic_estimate,
        value_discretization_interval=discretization,
        sampling_prob=sampling_prob,
        use_connect_dots=self.pessimistic_estimate,
    )
    return pl_dist.self_compose(num_compositions)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_deltas(self,
                 sigma,
                 epsilons,
                 num_steps_per_epoch,
                 num_epochs = 1,
                 discretization = 1e-3):
    """Returns delta values for which (epsilon, delta)-DP is satisfied.

    Args:
      sigma: The scale of the Gaussian noise.
      epsilons: A list or numpy array of epsilon values.
      num_steps_per_epoch: The number of steps per epoch. The subsampling
        probability is set to be 1 / num_steps_per_epoch.
      num_epochs: The number of epochs.
      discretization: The discretization interval to be used in computing the
        privacy loss distribution.

    Returns:
      A list of hockey stick divergence estimates corresponding to each epsilon.
    """
    pl_dist = self._get_pld(
        sigma=sigma,
        num_compositions=num_steps_per_epoch * num_epochs,
        sampling_prob=1.0/num_steps_per_epoch,
        discretization=discretization)
    return list(pl_dist.get_delta_for_epsilon(epsilons))

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_epsilons(self,
                   sigma,
                   deltas,
                   num_steps_per_epoch,
                   num_epochs = 1,
                   discretization = 1e-3):
    """Returns epsilon values for which (epsilon, delta)-DP is satisfied.

    Args:
      sigma: The scale of the Gaussian noise.
      deltas: The privacy parameters delta.
      num_steps_per_epoch: The number of steps per epoch. The subsampling
        probability is set to be 1 / num_steps_per_epoch.
      num_epochs: The number of epochs.
      discretization: The discretization interval to be used in computing the
        privacy loss distribution.
    """
    pl_dist = self._get_pld(
        sigma=sigma,
        num_compositions=num_steps_per_epoch * num_epochs,
        sampling_prob=1.0/num_steps_per_epoch,
        discretization=discretization)
    return [pl_dist.get_epsilon_for_delta(delta) for delta in deltas]

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_sigma(self,
                epsilon,
                delta,
                num_steps_per_epoch,
                num_epochs = 1,
                discretization = 1e-3):
    """Returns the scale of the Gaussian noise for (epsilon, delta)-DP."""
    delta_from_sigma = lambda sigma: self.get_deltas(
        sigma, (epsilon,), num_steps_per_epoch, num_epochs, discretization)[0]
    return inverse_decreasing_function(delta_from_sigma, delta)


class PoissonRDPAccountant:
  """Privacy accountant for ABLQ with Poisson batch sampler using RDP analysis.
  """

  def __init__(self):
    pass

  def _get_rdp_accountant(
      self,
      sigma,
      num_compositions,
      sampling_prob
  ):
    """Returns RDP accountant after composition of Poisson subsampled Gaussian.

    Args:
      sigma: The scale of the Gaussian noise.
      num_compositions: The number of compositions.
      sampling_prob: The sampling probability in each step.
    """
    accountant = rdp.RdpAccountant()
    event = dp_accounting.dp_event.PoissonSampledDpEvent(
        sampling_prob, dp_accounting.dp_event.GaussianDpEvent(sigma)
    )
    accountant.compose(event, num_compositions)
    return accountant

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_deltas(self,
                 sigma,
                 epsilons,
                 num_steps_per_epoch,
                 num_epochs = 1):
    """Returns delta values for which (epsilon, delta)-DP is satisfied.

    Args:
      sigma: The scale of the Gaussian noise.
      epsilons: A list or numpy array of epsilon values.
      num_steps_per_epoch: The number of steps per epoch. The subsampling
        probability is set to be 1 / num_steps_per_epoch.
      num_epochs: The number of epochs.

    Returns:
      A list of hockey stick divergence estimates corresponding to each epsilon.
    """
    accountant = self._get_rdp_accountant(
        sigma=sigma,
        num_compositions=num_steps_per_epoch * num_epochs,
        sampling_prob=1.0/num_steps_per_epoch)
    return [accountant.get_delta(epsilon) for epsilon in epsilons]

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_epsilons(self,
                   sigma,
                   deltas,
                   num_steps_per_epoch,
                   num_epochs = 1):
    """Returns epsilon values for which (epsilon, delta)-DP is satisfied.

    Args:
      sigma: The scale of the Gaussian noise.
      deltas: The privacy parameters delta.
      num_steps_per_epoch: The number of steps per epoch. The subsampling
        probability is set to be 1 / num_steps_per_epoch.
      num_epochs: The number of epochs.
    """
    accountant = self._get_rdp_accountant(
        sigma=sigma,
        num_compositions=num_steps_per_epoch * num_epochs,
        sampling_prob=1.0/num_steps_per_epoch)
    return [accountant.get_epsilon(delta) for delta in deltas]

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_sigma(self,
                epsilon,
                delta,
                num_steps_per_epoch,
                num_epochs = 1):
    """Returns the scale of the Gaussian noise for (epsilon, delta)-DP."""
    delta_from_sigma = lambda sigma: self.get_deltas(
        sigma, (epsilon,), num_steps_per_epoch, num_epochs)[0]
    return inverse_decreasing_function(delta_from_sigma, delta)


class ShuffleAccountant:
  """Accountant for ABLQ with shuffling; only provides lower bounds.

  This class only provides a lower bound on the privacy parameters. The hockey
  stick divergence is computed for the pair (P, Q), where P and Q are mixtures
  of Gaussian distributions P = MoG(mean_upper) and Q = MoG(mean_lower), where
  MoG(mean) := sum_{t=1}^T (1/T) * N(mean * e_t, sigma^2 * I_t),
  where T refers to the number of steps in a single epoch.

  Multiple epochs are analyzed as compositions of the single epoch mechanism.
  """

  def __init__(self,
               mean_upper = 2.0,
               mean_lower = 1.0):
    self.mean_upper = mean_upper
    self.mean_lower = mean_lower

  def _log_in_cube_mass(self,
                        sigma,
                        num_steps,
                        caps,
                        mean):
    """Returns log probability that max_t x_t <= cap, under MoG(mean).

    Args:
      sigma: The scale of the Gaussian noise.
      num_steps: The number of steps of composition.
      caps: The parameter defining the cube E_C given as {x : max_t x_t <= C}.
      mean: The mean parameter of MoG distribution.
    """
    return (
        stats.norm.logcdf((caps - mean) / sigma) +
        (num_steps - 1) * stats.norm.logcdf(caps / sigma)
    )

  def _out_cube_mass(self,
                     sigma,
                     num_steps,
                     caps,
                     mean):
    """Returns mass assigned by mixture of Gaussians to complement of a cube.

    Args:
      sigma: The scale of the Gaussian noise.
      num_steps: The number of steps of composition, denoted as T above.
      caps: The parameter defining the cube E_C given as {x : max_t x_t <= C}.
      mean: The mean parameter of MoG distribution.
    """
    return - np.expm1(self._log_in_cube_mass(sigma, num_steps, caps, mean))

  def _log_slice_mass(self,
                      sigma,
                      num_steps,
                      caps_1,
                      caps_2,
                      mean):
    """Returns log probability of cap_1 <= max_t x_t <= cap_2 under MoG(mean).

    For all (cap_1, cap_2) in zip(caps_1, caps_2). Requires that all coordinates
    of caps_1 are smaller than corresponding coordinates of caps_2.

    Args:
      sigma: The scale of the Gaussian noise.
      num_steps: The number of steps of composition.
      caps_1: The lower bounds of the slice.
      caps_2: The upper bounds of the slice.
      mean: The mean parameter of MoG distribution.

    Raises:
      ValueError if caps_1 is not coordinate-wise less than caps_2.
    """
    if np.any(caps_1 >= caps_2):
      raise ValueError('caps_1 must be coordinate-wise less than caps_2. '
                       f'Found {caps_1=} and {caps_2=}.')
    log_cube_mass_1 = self._log_in_cube_mass(sigma, num_steps, caps_1, mean)
    log_cube_mass_2 = self._log_in_cube_mass(sigma, num_steps, caps_2, mean)
    return log_cube_mass_1 + np.log(np.expm1(log_cube_mass_2 - log_cube_mass_1))

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_deltas_single_epoch(
      self,
      sigma,
      epsilons,
      num_steps,
      caps = None,
      verbose = True,
  ):
    """Returns lower bounds on delta values for corresponding epsilons.

    Args:
      sigma: The scale of the Gaussian noise.
      epsilons: A list or numpy array of epsilon values.
      num_steps: The number of steps of the mechanism.
      caps: The parameter defining the cube E_C given as {x : max_t x_t <= C}.
      verbose: When True, prints the optimal C value for each epsilon.
    """
    caps = np.arange(0, 100, 0.01) if caps is None else np.asarray(caps)
    upper_masses = self._out_cube_mass(
        sigma, num_steps, caps, self.mean_upper)
    lower_masses = self._out_cube_mass(
        sigma, num_steps, caps, self.mean_lower)
    epsilons = np.atleast_1d(epsilons)
    if verbose:
      print('Shuffle hockey stick divergence logs:')
      ans = []
      for epsilon in epsilons:
        hsd = upper_masses - np.exp(epsilon) * lower_masses
        i = np.argmax(hsd)
        print(f'* optimal C for {epsilon=} is: {caps[i]}')
        ans.append(hsd[i])
    else:
      ans = [
          np.max(upper_masses - np.exp(epsilon) * lower_masses)
          for epsilon in epsilons
      ]
    return ans

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_epsilons_single_epoch(
      self,
      sigma,
      deltas,
      num_steps,
      caps = None,
  ):
    """Returns epsilon values for which (epsilon, delta)-DP is satisfied.

    Args:
      sigma: The scale of the Gaussian noise.
      deltas: The privacy parameters delta.
      num_steps: The number of steps of the mechanism.
      caps: The parameter defining the cube E_C given as {x : max_t x_t <= C}.

    Returns:
      The epsilon value for which ABLQ_S satisfies (epsilon, delta)-DP.
    """
    delta_from_epsilon = lambda epsilon: self.get_deltas_single_epoch(
        sigma, (epsilon,), num_steps, caps, verbose=False)[0]
    return [
        inverse_decreasing_function(delta_from_epsilon, delta)
        for delta in deltas
    ]

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_sigma_single_epoch(self,
                             epsilon,
                             delta,
                             num_steps,
                             caps = None):
    """Returns the scale of the Gaussian noise for (epsilon, delta)-DP."""
    delta_from_sigma = lambda sigma: self.get_deltas_single_epoch(
        sigma, (epsilon,), num_steps, caps, verbose=False)[0]
    return inverse_decreasing_function(delta_from_sigma, delta)

  def _get_single_epoch_pld(
      self,
      sigma,
      num_steps,
      discretization = 1e-3,
      pessimistic_estimate = False,
      log_truncation_mass = -40,
      verbose = False,
  ):
    """Returns Privacy Loss Distribution for a single epoch.

    Args:
      sigma: The scale of the Gaussian noise.
      num_steps: The number of steps of the mechanism in a single epoch.
      discretization: The discretization interval to be used in computing the
        privacy loss distribution.
      pessimistic_estimate: When True, an upper bound on the hockey stick
        divergence is returned. When False, a lower bound on the hockey stick
        divergence is returned.
      log_truncation_mass: The log of the truncation mass.
      verbose: When True, intermediate computations of the single-epoch PLD
        construction are printed.
    """
    log_truncation_mass -= math.log(2)
    truncation_mass = math.exp(log_truncation_mass)
    lower_cap = sigma * stats.norm.ppf(
        math.exp(log_truncation_mass / num_steps)
    )
    upper_cap = self.mean_upper + sigma * stats.norm.isf(
        -1.0 * math.expm1(math.log1p(-truncation_mass) / num_steps)
    )

    pmf = collections.defaultdict(lambda: 0.0)
    sigma_square = sigma ** 2
    # We use a heuristic value of gap that ensures that max_t x_t / sigma^2
    # changes by `discretization` between two consecutive slices.
    gap = discretization * sigma_square

    caps_1 = np.arange(lower_cap, upper_cap, gap)
    caps_2 = caps_1 + gap
    upper_cap = caps_2[-1]

    if verbose:
      print(
          f'truncating to {lower_cap=}, {upper_cap=}\n'
          'lower log mass truncated: '
          f'{self._log_in_cube_mass(sigma, num_steps, lower_cap, 2)}\n'
          'upper log mass truncated: '
          f'{math.log(self._out_cube_mass(sigma, num_steps, upper_cap, 2))}\n'
          f'num intervals = {len(caps_1)}'
      )

    if pessimistic_estimate:
      log_upper_probs = self._log_slice_mass(
          sigma, num_steps, caps_1, caps_2, self.mean_upper)
      # The following is an upper bound on the privacy loss anywhere in the
      # slice between cap_1 and cap_2.
      rounded_privacy_losses = np.ceil(
          (self.mean_upper - self.mean_lower)
          * (caps_2 - (self.mean_upper + self.mean_lower) / 2)
          / (sigma_square * discretization)
      ).astype(int)

      # max_t x_t >= upper_cap
      infinity_mass = self._out_cube_mass(
          sigma, num_steps, upper_cap, self.mean_upper)

      # max_t x_t <= lower_cap
      rounded_privacy_loss = math.ceil(
          (self.mean_upper - self.mean_lower)
          * (lower_cap - (self.mean_upper + self.mean_lower) / 2)
          / (sigma_square * discretization)
      )
      pmf[rounded_privacy_loss] += math.exp(
          self._log_in_cube_mass(sigma, num_steps, lower_cap, self.mean_upper))
    else:
      log_upper_probs = self._log_slice_mass(
          sigma, num_steps, caps_1, caps_2, self.mean_upper)
      log_lower_probs = self._log_slice_mass(
          sigma, num_steps, caps_1, caps_2, self.mean_lower)
      rounded_privacy_losses = np.floor(
          (log_upper_probs - log_lower_probs) / discretization
      ).astype(int)

      # max_t x_t >= upper_C
      infinity_mass = 0
      upper_out_cube_mass = self._out_cube_mass(
          sigma, num_steps, upper_cap, self.mean_upper)
      lower_out_cube_mass = self._out_cube_mass(
          sigma, num_steps, upper_cap, self.mean_lower)
      rounded_privacy_loss = math.floor(
          (math.log(upper_out_cube_mass) - math.log(lower_out_cube_mass))
          / discretization
      )
      pmf[rounded_privacy_loss] += upper_out_cube_mass

      # max_t x_t <= lower_C
      upper_log_in_cube_mass = self._log_in_cube_mass(
          sigma, num_steps, lower_cap, self.mean_upper)
      lower_log_in_cube_mass = self._log_in_cube_mass(
          sigma, num_steps, lower_cap, self.mean_lower)
      rounded_privacy_loss = math.floor(
          (upper_log_in_cube_mass - lower_log_in_cube_mass)
          / discretization
      )
      pmf[rounded_privacy_loss] += math.exp(upper_log_in_cube_mass)

    upper_probs = np.exp(log_upper_probs)
    for rounded_priv_loss, prob in zip(rounded_privacy_losses, upper_probs):
      pmf[rounded_priv_loss] += prob

    return pld.privacy_loss_distribution.PrivacyLossDistribution(
        pld.pld_pmf.SparsePLDPmf(
            pmf, discretization, infinity_mass, pessimistic_estimate
        )
    )

  def _get_multi_epoch_pld(
      self,
      sigma,
      num_steps_per_epoch,
      num_epochs = 1,
      discretization = 1e-3,
      pessimistic_estimate = False,
      log_truncation_mass = -40,
      verbose = False,
  ):
    """Returns Privacy Loss Distribution for multiple epochs.

    Args:
      sigma: The scale of the Gaussian noise.
      num_steps_per_epoch: The number of steps of the mechanism in a single
        epoch.
      num_epochs: The number of epochs.
      discretization: The discretization interval to be used in computing the
        privacy loss distribution.
      pessimistic_estimate: When True, an upper bound on the hockey stick
        divergence is returned. When False, a lower bound on the hockey stick
        divergence is returned.
      log_truncation_mass: The log of the truncation mass.
      verbose: When True, intermediate computations of the single-epoch PLD
        construction are printed.
    """
    pl_dist = self._get_single_epoch_pld(
        sigma, num_steps_per_epoch, discretization, pessimistic_estimate,
        log_truncation_mass - math.log(num_epochs), verbose)
    return pl_dist.self_compose(num_epochs)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_deltas(
      self,
      sigma,
      epsilons,
      num_steps_per_epoch,
      num_epochs = 1,
      reshuffle = True,
      discretization = 1e-3,
      pessimistic_estimate = False,
      log_truncation_mass = -40,
      verbose = False,
  ):
    """Returns lower bounds on delta values for corresponding epsilons.

    Args:
      sigma: The scale of the Gaussian noise.
      epsilons: A list or numpy array of epsilon values.
      num_steps_per_epoch: The number of steps of the mechanism per epoch.
      num_epochs: The number of epochs.
      reshuffle: When True, the shuffle mechanism is assumed to employ
        reshuffle between different epochs. When False, it is assumed that
        the same shuffled order is used in all epochs.
      discretization: The discretization interval to be used in computing the
        privacy loss distribution.
      pessimistic_estimate: When False, a lower bound on the hockey stick
        divergence is returned. When True, an upper bound estimate is returned;
        but since we do not know for sure that the pair we consider is
        dominating, this may not be an upper bound on the hockey stick
        divergence.
      log_truncation_mass: The log of the truncation mass.
      verbose: When True, intermediate computations of the single-epoch PLD
        construction are printed.
    """
    if not reshuffle or num_epochs == 1:
      return self.get_deltas_single_epoch(
          sigma / math.sqrt(num_epochs), epsilons, num_steps_per_epoch,
          verbose=verbose
      )
    pl_dist = self._get_multi_epoch_pld(
        sigma, num_steps_per_epoch, num_epochs, discretization,
        pessimistic_estimate, log_truncation_mass, verbose)
    return list(pl_dist.get_delta_for_epsilon(epsilons))

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_epsilons(
      self,
      sigma,
      deltas,
      num_steps_per_epoch,
      num_epochs = 1,
      reshuffle = True,
      discretization = 1e-3,
      pessimistic_estimate = False,
      log_truncation_mass = -40,
      verbose = False,
  ):
    """Returns epsilon values for which (epsilon, delta)-DP is satisfied.

    Args:
      sigma: The scale of the Gaussian noise.
      deltas: The privacy parameters delta.
      num_steps_per_epoch: The number of steps of the mechanism per epoch.
      num_epochs: The number of epochs.
      reshuffle: When True, the shuffle mechanism is assumed to employ
        reshuffle between different epochs. When False, it is assumed that
        the same shuffled order is used in all epochs.
      discretization: The discretization interval to be used in computing the
        privacy loss distribution.
      pessimistic_estimate: When False, a lower bound on the hockey stick
        divergence is returned. When True, an upper bound estimate is returned;
        but since we do not know for sure that the pair we consider is
        dominating, this may not be an upper bound on the hockey stick
        divergence.
      log_truncation_mass: The log of the truncation mass.
      verbose: When True, intermediate computations of the single-epoch PLD
        construction are printed.
    """
    if not reshuffle or num_epochs == 1:
      return self.get_epsilons_single_epoch(
          sigma / math.sqrt(num_epochs), deltas, num_steps_per_epoch,
      )
    pl_dist = self._get_multi_epoch_pld(
        sigma, num_steps_per_epoch, num_epochs, discretization,
        pessimistic_estimate, log_truncation_mass, verbose)
    return [pl_dist.get_epsilon_for_delta(delta) for delta in deltas]

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_sigma(self,
                epsilon,
                delta,
                num_steps_per_epoch,
                num_epochs = 1,
                reshuffle = True,
                discretization = 1e-3,
                pessimistic_estimate = False,
                log_truncation_mass = -40):
    """Returns the scale of the Gaussian noise for (epsilon, delta)-DP."""
    delta_from_sigma = lambda sigma: self.get_deltas(
        sigma, (epsilon,), num_steps_per_epoch, num_epochs, reshuffle,
        discretization, pessimistic_estimate, log_truncation_mass,
        verbose=False)[0]
    return inverse_decreasing_function(delta_from_sigma, delta)
