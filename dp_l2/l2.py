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

"""Functions for the l_2 mechanism."""

import functools
import numpy as np
from scipy import special

from dp_l2 import utils


def get_radius_upper_bound(d, sigma, beta, tolerance=1e-4):
  """Returns 1-beta probability upper bound on l_2 norm of sample from M_sigma.

  Args:
    d: Integer dimension.
    sigma: Float noise scale of l_2 mechanism.
    beta: Float failure parameter beta. With probability beta, a sample from
      M_sigma has l_2 norm that exceeds the value computed by this function.
    tolerance: Float accuracy for computed bound. Note that this errs on the
      side of being conservative.
  """
  gammaincc_partial = lambda d, sigma, radius: special.gammaincc(
      d, radius / sigma
  )
  # Note that we search over gammaincc (Gamma(s, x), the upper incomplete gamma
  # function) instead of gammainc (gamma(s, x), the lower incomplete gamma
  # function) because Gammma(s, x) is decreasing in x, and
  # utils.binary_search takes a decreasing function.
  binary_search_function = functools.partial(gammaincc_partial, d, sigma)
  return utils.binary_search(binary_search_function, beta, tolerance)


def get_cap_fraction_from_h(d, h, r):
  """Returns fraction of radius-r sphere occupied by cap of height h in [0, 2r].

  See Lemma 3.9 in the paper for details.

  Args:
    d: Integer dimension.
    h: Float height of cap.
    r: Float radius of sphere.

  Raises:
    RuntimeError: Requires h in [0, 2r], but h = [h] and r = [r].
  """
  if h < 0 or h > 2 * r:
    raise RuntimeError(
        "Requires h in [0, 2r], but h = " + str(h) + " and r = " + str(r) + "."
    )
  if h <= r:
    return 0.5 * special.betainc(
        (d - 1) / 2, 1 / 2, (2 * r * h - h**2) / (r**2)
    )
  h_prime = 2 * r - h
  return 1 - (
      0.5
      * special.betainc(
          (d - 1) / 2, 1 / 2, (2 * r * h_prime - h_prime**2) / (r**2)
      )
  )


def get_cap_height(eps, r, sigma):
  """Returns high privacy loss spherical cap height.

  See Lemma 3.6 in the paper for details.

  Args:
    eps: Float privacy parameter eps.
    r: Float sphere radius.
    sigma: Float noise scale of l_2 mechanism.
  """
  return min(
      2 * r, max(r * (1 - (eps * sigma)) + (1 - (eps * sigma) ** 2) / 2, 0)
  )


def get_cap_fraction_for_loss(d, eps, r, sigma):
  """Returns cap fraction at radius-r sphere so that privacy loss is <= eps.

  In more detail, the spherical cap in question is centered on the x-axis and
  occupies a portion of the radius-r sphere, and any point on the sphere
  outside of the spherical cap realizes privacy loss at most eps. This function
  returns 0 when no point on the sphere realizes privacy loss more than eps and
  returns 1 when every point on the sphere realizes privacy loss more than eps.

  Args:
    d: Integer dimension.
    eps: Float desired upper bound on privacy loss.
    r: Float radius of sphere.
    sigma: Float noise scale of l_2 mechanism.
  """
  cap_height = get_cap_height(eps, r, sigma)
  return get_cap_fraction_from_h(d, cap_height, r)


def compute_shell_mass(d, sigma, r1, r2):
  """Returns the probability of M_sigma drawing x with ||x||_2 in [r1, r2).

  See Lemma 3.11 in the paper for details.

  Args:
    d: Integer dimension.
    sigma: Float noise scale of l_2 mechanism.
    r1: Float lower bound on radius.
    r2: Float upper bound on radius.
  """
  return special.gammainc(d, r2 / sigma) - special.gammainc(d, r1 / sigma)


def get_plrv_1_upper_bound(d, sigma, num_rs, last_r, eps):
  """Returns upper bound on P[L_{M,X,X'} >= eps].

  Note that L_{M,X,X'} is a privacy loss random variable (PLRV). See Algorithm 1
  in the paper for details.

  Args:
    d: Integer dimension.
    sigma: Float noise scale of l_2 mechanism.
    num_rs: Integer grid size for radii.
    last_r: Float largest radius.
    eps: Float desired upper bound on privacy loss.
  """
  if sigma >= 1 / eps:
    return 0
  if d == 1:
    return 1 - 0.5 * np.exp(0.5 * (eps - 1 / sigma))
  first_r = (1 - eps * sigma) / 2
  rjs = np.linspace(first_r, last_r, num_rs)
  upper_bound = special.gammainc(d, rjs[0] / sigma)
  for r_idx in range(len(rjs)-1):
    r1 = rjs[r_idx]
    r2 = rjs[r_idx+1]
    cap_fraction = get_cap_fraction_for_loss(d, eps, r1, sigma)
    shell_mass = compute_shell_mass(d, sigma, r1, r2)
    upper_bound += cap_fraction * shell_mass
  last_cap_fraction = get_cap_fraction_for_loss(d, eps, last_r, sigma)
  last_mass = special.gammaincc(d, last_r / sigma)
  upper_bound += last_cap_fraction * last_mass
  return upper_bound


def get_1_centered_cap_fraction_for_loss(d, eps, sigma, e1_r):
  """Returns high privacy loss cap fraction at e_1-centered radius-e1_r sphere.

  This is analogous to get_cap_fraction_for_loss, but for a different sphere.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter epsilon.
    sigma: Float noise scale of underlying l_2 mechanism.
    e1_r: Float radius of e_1-centered sphere.
  """
  tau = eps * sigma
  # See Lemma 3.17 in the paper for this branch.
  if e1_r < (1 + tau) / 2:
    return 0
  else:
    # See Lemma 3.18 in the paper for this branch.
    intersection_e1 = 0.5 * (1 + tau ** 2 - (2 * tau * e1_r))
    cap_height = e1_r - 1 + intersection_e1
    return get_cap_fraction_from_h(d, cap_height, e1_r)


def get_plrv_2_lower_bound(d, eps, sigma, e1_rs):
  """Returns lower bound on P[L_{M,X',X} <= -eps].

  Note that L_{M,X',X} is a privacy loss random variable (PLRV). See Algorithm 1
  in the paper for details.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter epsilon.
    sigma: Float noise scale of underlying Euclidean mechanism.
    e1_rs: Float array of radii for spheres around e_1.
  """
  if sigma >= 1 / eps:
    return 0
  if d == 1:
    return 0.5 * np.exp(0.5 * (-eps - 1 / sigma))
  lower_bound = 0
  for i in range(len(e1_rs) - 1):
    radius_1 = e1_rs[i]
    radius_2 = e1_rs[i + 1]
    shell_mass = compute_shell_mass(d, sigma, radius_1, radius_2)
    high_loss_cap_fraction = get_1_centered_cap_fraction_for_loss(
        d, eps, sigma, radius_1
    )
    lower_bound += shell_mass * high_loss_cap_fraction
  last_cap_fraction = get_1_centered_cap_fraction_for_loss(
      d, eps, sigma, e1_rs[-1]
  )
  last_mass = special.gammaincc(d, e1_rs[-1] / sigma)
  lower_bound += last_cap_fraction * last_mass
  return lower_bound


def get_plrv_difference(d, eps, delta, num_rs, num_e1_rs, sigma):
  """Returns PLRV difference used in (eps, delta)-DP condition.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter epsilon.
    delta: Float privacy parameter delta.
    num_rs: Integer grid size for radii for spheres around 0.
    num_e1_rs: Integer grid size for radii for spheres around e_1.
    sigma: Float noise scale of underlying l_2 mechanism.
  """
  last_r = get_radius_upper_bound(d, sigma, 0.01 * delta)
  plrv_1_upper_bound = get_plrv_1_upper_bound(d, sigma, num_rs, last_r, eps)
  # See Section 3.1.3 in the paper for details.
  e1_rs = np.linspace((1 + eps * sigma) / 2, last_r, num_e1_rs)
  plrv_2_lower_bound = get_plrv_2_lower_bound(d, eps, sigma, e1_rs)
  return plrv_1_upper_bound - np.exp(eps) * plrv_2_lower_bound


def get_l2_sigma(d, eps, delta, num_rs, num_e1_rs, tolerance=1e-3):
  """Returns minimum sigma to achieve (eps, delta)-DP M_sigma mechanism.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter eps.
    delta: Float privacy parameter delta.
    num_rs: Integer grid size for radii for spheres around 0.
    num_e1_rs: Integer grid size for radii for spheres around e_1.
    tolerance: Float tolerance for binary search in subroutines. Note that this
      errs on the side of being conservative.
  """
  binary_search_function = functools.partial(
      get_plrv_difference, d, eps, delta, num_rs, num_e1_rs
  )
  return utils.binary_search(
      function=binary_search_function, threshold=delta, tolerance=tolerance
  )


def sample_l2_ball(d, num_samples):
  """Returns samples of shape (num_samples, d) from the d-dim unit l_2 ball.

  This is a folklore sampling algorithm.

  Args:
    d: Integer dimension.
    num_samples: Integer number of samples.
  """
  # The Gaussian distribution is spherically symmetric, so samples divided by
  # their norms are uniformly distributed on the unit sphere.
  samples = np.random.randn(num_samples, d)
  norms = np.linalg.norm(samples, axis=1, keepdims=True)
  samples /= norms
  # The volume of a radius-r ball is proportional to r^d, so we sample radii
  # <= 1 such that the probability of drawing radius <= r is proportional to
  # r^d.
  radii = np.random.rand(num_samples, 1) ** (1/d)
  samples *= radii
  return samples


def get_l2_samples(d, sigma, num_samples):
  """Returns samples of shape (num_samples, d) from the l_2 mechanism.

  See Lemma 2.4 in the paper for details.

  Args:
    d: Integer dimension.
    sigma: Float noise scale.
    num_samples: Integer number of samples to generate.
  """
  radii = np.random.gamma(shape=d + 1, scale=sigma, size=num_samples)
  ball_samples = sample_l2_ball(d, num_samples)
  return radii[:, np.newaxis] * ball_samples


def get_l2_mean_squared_l2_error(d, sigma):
  """Returns the mean squared l_2 error of the specified l_2 mechanism.

  See Corollary 4.3 in the paper for details.

  Args:
    d: Integer dimension.
    sigma: Float noise scale.
  """
  return d * (d+1) * sigma ** 2

