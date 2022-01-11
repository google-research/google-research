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

"""Analyze omega-trajectories on SL(2)x7.

Usage:
  python3 -i -m dim4.papers.bfis2021.analyze_sl2x7

Given an omega-deformation trajectory for the gauge group SO(8) on
(SL(2)/U(1))**7 for which at least one coordinate runs off to the
boundary of a Poincare disk and the scalar potential diverges, this
code determines the boundary-gauging and shows the trajectory.


The 48 critical points of de Wit-Nicolai supergravity that can be
found on this submanifold are
(cf. Table 2 of: http://dx.doi.org/10.1007/jhep01(2020)099):


S0600000 S0668740 S0698771 S0719157 S0779422 S0800000
S0847213 S0869596 S0880733 S0983994 S1006758 S1039230
S1046017 S1067475 S1068971 S1075828 S1165685 S1200000
S1301601 S1362365 S1366864 S1384096 S1384135 S1400000
S1424025 S1470986 S1473607 S1477609 S1497038 S1571627
S1600000 S1637792 S1637802 S1650772 S1652212 S1805269
S1810641 S2096313 S2099077 S2140848 S2389433 S2443607
S2457396 S2519962 S2547536 S2702580 S2707528 S3305153

"""

# Linter-violations in this file all have a good reason.
# pylint:disable=g-short-docstring-punctuation
# pylint:disable=unused-import

import itertools
import os
import pdb  # For interactive debugging.
from dim4.so8.src import analysis
from dim4.theta.src import gaugings
from m_theory_lib import algebra
from m_theory_lib import m_util as mu
from matplotlib import pyplot
import numpy
import scipy.interpolate
import scipy.linalg


def _interp1d(*args, **kwargs):
  """scipy.interpolate.interp1d that returns a number, not ndarray."""
  f = scipy.interpolate.interp1d(*args, **kwargs)
  return lambda *f_args, **f_kwargs: f(*f_args, **f_kwargs).item()


def v14_from_7z(zs):
  """Maps SL(2)**7 coordinates to the corresponding E7 70-vector."""
  # This needs an ad-hoc trick/hack: Should we accidentally
  # have left the circle, we 'tuck in the coordinate a bit.'
  zzs = [z if abs(z) < 1 else z / (abs(z) * (1 + 1e-5)) for z in zs]
  cs = numpy.array([0.25 * mu.undiskify(z) for z in zzs])
  return numpy.concatenate([cs.real, cs.imag], axis=0)


def get_7z_from_bfp_z123(z123):
  """Implements Eq. (7.7) from: https://arxiv.org/pdf/1909.10969.pdf"""
  z1, z2, z3 = z123
  return numpy.array([-z2, -z3, z1, z1, z1, -z3, -z3])


def v70_from_7z(zs, e7=None):
  """Maps SL2x7 coordinates to the corresponding E7 70-vector."""
  if e7 is None:
    e7 = algebra.g.e7
  v14 = v14_from_7z(zs)
  return e7.sl2x7[:2, :, :70].reshape(-1, 70).T.dot(v14)


def get_7z_from_v70(v70, e7=None, check_residuals=False):
  """Obtains SL2x7 Poincare disc coordinates from E7 70-vector."""
  if e7 is None:
    e7 = algebra.g.e7
  v14, residuals, *_ = scipy.linalg.lstsq(
      e7.sl2x7[:2, :, :70].reshape(-1, 70).T,
      v70)
  residual = sum(abs(residuals.ravel()))
  if check_residuals and residual > 1e-3:
    # We allow a fairly large residual here, since close to a boundary,
    # we may actually be quite a bit off. This is just for catching
    # some 'something is really wrong here' cases.
    raise ValueError(f'Cannot express v70 as 7z (residual: {residual}.')
  return numpy.array(
      [mu.diskify(4 * v)
       for v in v14[:7] + 1j * v14[7:]])


def get_pot_stat_zs_js_by_omega_from_trajectory_data(tdata, num_js=4):
  """Computes pot_stat_zs_js_by_omega from dyonic.collect_trajectory_logs()."""
  pot_stat_zs_js_by_omega = {}
  for row in tdata:
    pot, stat = row[:2]
    v70 = row[3: 3 + 70]
    omega = row[3 + 70]
    js = row[-num_js:]
    pot_stat_zs_js_by_omega[omega] = (pot, stat, get_7z_from_v70(v70), js)
  return pot_stat_zs_js_by_omega


def get_boosted_theta_so8(omega, v70, e7=None):
  """Returns a boosted Theta-tensor for dyonic SO(8) gauging."""
  if e7 is None:
    e7 = algebra.g.e7
  theta = numpy.zeros([56, 133])
  cw, sw = numpy.cos(omega), numpy.sin(omega)
  theta[:28, 3*35:] = +cw * numpy.eye(28) * 0.25
  theta[28:, 3*35:] = sw * numpy.eye(28) * 0.25
  g56 = mu.expm(mu.nsum('a,_aM^N->^N_M', v70, e7.t56r[:70]))
  g133 = mu.expm(mu.nsum('a,_ab^C->^C_b', -v70, e7.f_abC[:70]))
  return mu.nsum('_M^a,^M_N,^b_a->_N^b', theta, g56, g133)


# Extrapolating a trajectory towards the boundary: We currently do
# this (as described in the paper) by extracting a geometric sequence
# along which the omega-values are spaced geometrically as
# omega[n] - omega_target = a * b**n. This then is used, in a 2nd step,
# as input to a convergence accelerator. There very likely are much
# better ways to do this step, so all this code may go away at some point.
def get_geometrically_spaced_trajectory_samples(xs, ys, x_target,
                                                num_points=20,
                                                interpolation_kind=7,
                                                endpoint_avoiding_eps=0.01
                                                ):
  """Produces geometrically spaced positions and interpolated values."""
  # In general, the user is expected to pass in a subsequence of the
  # recorded trajectory that focuses on the behavior as we are
  # approaching the diverging behavior.
  num_steps = num_points - 1
  x_dist = lambda x: abs(x - x_target)
  x_closest = min(xs, key=x_dist)
  x_farthest = max(xs, key=x_dist)
  # We do shy away a tiny bit from the starting and end point,
  # in order to avoid accidentally asking interpolation to extrapolate due to
  # numerical rounding errors.
  d_x_max = x_target - x_farthest
  d_x_min = x_target - x_closest
  assert d_x_max * d_x_min > 0, '"farthest" and "closest" not on same side.'
  x_shrink_step_factor = (
      (d_x_min / d_x_max)**(1 / (endpoint_avoiding_eps + num_steps)))
  xs_geom = numpy.array([
      x_target -
      d_x_max * x_shrink_step_factor**(n + 0.5 * endpoint_avoiding_eps)
      for n in range(num_steps + 1)])
  fz = _interp1d(xs, ys, kind=interpolation_kind)
  ys_geom = numpy.stack([fz(x) for x in xs_geom], axis=-1)
  if not numpy.isfinite(xs_geom).all():
    raise RuntimeError('xs_geom is not finite.')
  return xs_geom, ys_geom


def get_limit_zs(omegas, zss, omega_target,
                 accel_order=5,
                 **geom_spacing_kwargs):
  """Computes limit-coordinates via concergence acceleration.

  Args:
    omegas: The trajectory's omega-values.
    zss: [7, num_omegas]-numpy.ndarray. The z-coordinate-vectors,
      indexed (on final index) in parallel to `omegas`.
    omega_target: The target omega-value.
    accel_order: Acceleration order for Aitken acceleration.
    **geom_spacing_kwargs: Keyword arguments to use for
      get_geometrically_spaced_trajectory_samples().

  Returns:
    A tuple `(omegas_geom_ext, zss_geom_ext, fzs_ext)`, where
    `omegas_geom_ext` are geometrically spaced omegas,
    ending with the limit-omega, zss_geom_ext are the associated complex
    z-coordinate vectors (indexed in parallel to `omegas_geom_ext`, as for zss),
    and `fzs_ext` is an omega -> zs interpolation-function.
  """
  # zss is indexed as zss[num_zs, omega__sequence_index].
  g_spaced = get_geometrically_spaced_trajectory_samples
  omega_zs_s_geom = [
      g_spaced(omegas, zs, omega_target, **geom_spacing_kwargs)
      for zs in zss]
  omegas_geom = omega_zs_s_geom[0][0]
  zss_geom = numpy.stack([zs for omegas, zs in omega_zs_s_geom], axis=0)
  zs_limit = numpy.array(
      [list(mu.aitken_accelerated(zs, order=accel_order))[-1]
       for zs in zss_geom])
  if sum(abs(zss_geom[:, -1] - zs_limit)) > 0.1:
    raise ValueError('Limits look broken.')
  # Extend the geometrically spaced omegas with the endpoint,
  # and then interpolate towards it.
  omegas_geom_ext = numpy.concatenate(
      [omegas_geom, numpy.array([omega_target])], axis=0)
  zss_geom_ext = numpy.concatenate([zss_geom, zs_limit[:, numpy.newaxis]],
                                   axis=-1)
  # Interpolation-functions that allow us to go very close to the border.
  fzs_ext = scipy.interpolate.interp1d(omegas_geom_ext, zss_geom_ext, kind=3)
  return omegas_geom_ext, zss_geom_ext, fzs_ext


def refine_omega_zs(omega, zs, verbosity='SF',
                    sugra=None, e7=None, debug=False):
  """Refines z-coordinate vectors to low stationarity-violation."""
  if e7 is None:
    e7 = algebra.g.e7
  if sugra is None:
    sugra = analysis.SO8_SUGRA(e7=e7)
  cs = numpy.array(
      # Factor 0.25 is due to normalization of SL(2)-generators.
      [0.25 * mu.undiskify(z / max(1, abs(z) + 1e-5)) for z in zs])
  current_opt_stat = 1.0
  current_opt_pos = numpy.concatenate([cs.real, cs.imag], axis=0)
  # It is very important that we do get a good-quality equilibrium here.
  # If we do not, this would seriously mess up convergence acceleration
  # as we use it to distill out the boundary-tensor.
  while current_opt_stat > 1e-12:
    opt = sugra.find_equilibrium(
        current_opt_pos,
        verbosity=verbosity,
        submanifold_embedding=e7.sl2x7[:2, :, :70].reshape(14, 70),
        t_omega=mu.tff64(omega),
        minimize_kwargs=dict(default_gtol=1e-14,
                             default_maxiter=10**5)
    )
    current_opt_pot, current_opt_stat, current_opt_pos = opt
  if debug:
    print(
        f'Debug: refined opt_pot={current_opt_pot}, '
        f'opt_stat={current_opt_stat}')
  refined_zs = numpy.array(
      [mu.diskify(4 * z)
       for z in current_opt_pos[:7] + 1j * current_opt_pos[7:]])
  return omega, current_opt_pot, current_opt_stat, refined_zs


def accelerated_tensor_limit(tensors, digits=6, order=2):
  """Uses sequence-acceleration to estimate a limit-tensor."""
  merged = numpy.stack(tensors, axis=-1)
  threshold = 10**(-digits)
  *shape, num_tensors = merged.shape
  del num_tensors  # Unused, named for documentation only.
  result = numpy.zeros_like(merged[..., 0])
  limit_by_fingerprint = {}
  for indices in itertools.product(*map(range, shape)):
    seq = merged[indices]
    if max(abs(seq)) < threshold:
      continue  # All entries in the sequence are roughly-zero.
    fingerprint = ','.join(map(repr, seq.round(digits)))
    lim = limit_by_fingerprint.get(fingerprint)
    if lim is not None:
      result[indices] = lim
    else:
      result[indices] = limit_by_fingerprint[fingerprint] = (
          mu.final(mu.aitken_accelerated(seq, order=order)))
  return result


def get_trajectory_fn_zs(
    sugra,
    zs_by_omega,
    omega_min,
    omega_max):
  """Returns an interpolating omega -> zs trajectory-function."""
  local_zs_by_omega = dict(sorted(zs_by_omega.items())[1:-1])
  omegas0 = sorted(local_zs_by_omega)
  zss0 = numpy.stack([local_zs_by_omega[omega] for omega in omegas0], axis=-1)
  if omega_min is not None:
    omegas1, zss1, _ = get_limit_zs(omegas0, zss0, omega_min)
    for omega, zs in zip(omegas1, zss1.T):
      local_zs_by_omega[omega] = zs
  if omega_max is not None:
    omegas2, zss2, _ = get_limit_zs(omegas0, zss0, omega_max)
    for omega, zs in zip(omegas2, zss2.T):
      local_zs_by_omega[omega] = zs
  all_omegas = sorted(local_zs_by_omega)
  zss = numpy.stack([local_zs_by_omega[omega] for omega in all_omegas],
                    axis=-1)
  f_zs = scipy.interpolate.interp1d(
      all_omegas, zss, kind='cubic',
      bounds_error=False, fill_value='extrapolate')
  omega_cache = {}
  def fn_result(omega, refined=False, cache=True, max_refinable_abs_z=0.98):
    if not omega_min <= omega <= omega_max:
      raise ValueError('Out-of-range omega.')
    cache_key = (omega, bool(refined))
    cached = omega_cache.get(cache_key)
    if cached is not None:
      return cached
    zs = f_zs(omega)
    if not refined or (abs(zs) > max_refinable_abs_z).any():
      # If any z is 'near-boundary', or perhaps even on-boundary,
      # refining may produce undesirable jumps.
      if cache:
        omega_cache[cache_key] = zs
      return zs
    # We have to refine this interpolated solution.
    v70 = v70_from_7z(zs, e7=sugra.e7)
    wsspo = sugra.wsspo_from_v70_and_omega(v70, omega)
    pot_refined, stat_refined, wsspo_refined = sugra.refined_pot_stat_wsspo(
        wsspo)
    del pot_refined, stat_refined  # Unused, named for documentation only.
    assert abs(wsspo - wsspo_refined).sum() < 1e-6, (
        'Refined/unrefined misalignment.')
    v71 = sugra.v71_from_wsspo.dot(wsspo_refined)
    result = get_7z_from_v70(v71[:-1], e7=sugra.e7)
    if cache:
      omega_cache[cache_key] = result
    return result
  return fn_result


def get_boundary_gauging(
    sugra,
    omega_limit,
    pot_stat_zs_by_omega,
    e7=None,
    threshold_stationarity=1e-15,
    # These parameters work well for e.g. SO(3) N=1 S1384096.
    scan_delta_omegas=tuple(0.01 * 1.1**n for n in reversed(range(21))),
    diverging_zs_mask=None,
    verbose=True,
    report=print):
  """Analyzes the endpoint-divergence of an omega-deformation trajectory.

  This function determines the boundary-gauging that is reached in the
  limit when proceeding along the given trajectory in increasing index
  order. Users may want to turn around a trajectory (such as: a complete
  omega-trajectory) when determining the limit at the other end.

  Args:
    sugra: the `Supergravity` model instance.
    omega_limit: float, the limit-omega to extrapolate the trajectory towards.
    pot_stat_zs_by_omega: A mapping of omega-values to (pot, stat, zs).
    e7: Optional definition of E7 to use. Default of `None` uses `algebra.g.e7`.
    threshold_stationarity: Maximal acceptable stationarity-violation for
      which extrapolation will accept an already-present entry in
      pot_stat_zs_by_omega as not requiring recomputation.
    scan_delta_omegas: Tuple[float], the (positive) omega-differences from
      `omega_limit` to use for determining boundary-behavior.
    diverging_zs_mask: If `None`, the mask of diverging zs will be determined
      by looking at which z-values approach the boundary of the Poincare disc.
      Otherwise, the mask in the form of an ArrayLike with seven entries
      1.0 or 0.0, the 1.0-values in the places of the diverging z-s.
    verbose: Whether to verbosely report progress.
    report: Reporting function that prints/logs its single argument.

  Returns:
    A pair `(boundary_gauging, extended_pot_stat_zs_by_omega)`,
    where `boundary_gauging` is a dim4.theta.src.gaugings.Gauging instance
    that describes the limit-gauging, and `extended_pot_stat_zs_by_omega`
    is a mapping with the same data as pot_stat_zs_by_omega, plus extra
    entries for omegas that had to be computed to determine
    the boundary-gauging.
  """
  # The trajectories are expensive-to-compute, so generally sparse enough
  # that it makes sense to represent them as mappings. This simplifies
  # handing out extended trajectories.
  # Indeed, we do return an expanded copy.
  re_pot_stat_zs_by_omega = dict(pot_stat_zs_by_omega)
  e7 = sugra.e7
  omegas = numpy.array(sorted(pot_stat_zs_by_omega))
  zss = numpy.stack([pot_stat_zs_by_omega[omega][-1]
                     for omega in omegas], axis=-1)
  omega_lim_is_largest = omega_limit > omegas[-1]
  if not (omegas[1:] > omegas[:-1]).all():
    raise ValueError('omegas must be in increasing order.')
  zss = numpy.asarray(zss)
  # First estimate for the z-trajectories as we are approaching
  # the limiting omega.
  fzs_est1 = get_trajectory_fn_zs(
      sugra,
      {omega: psz[2] for omega, psz in pot_stat_zs_by_omega.items()},
      min(omega_limit, min(pot_stat_zs_by_omega)),
      max(omega_limit, max(pot_stat_zs_by_omega)))
  zs_near_lim = fzs_est1(omega_limit)
  if verbose:
    report(f'zs_near_lim={zs_near_lim.tolist()}')
  a_diverging_zs_mask = (
      numpy.asarray(diverging_zs_mask) if diverging_zs_mask is not None
      else numpy.array([1.0 if abs(z) > 0.99 else 0.0 for z in zs_near_lim]))
  if verbose:
    report(f'a_diverging_zs_mask={a_diverging_zs_mask.tolist()}')
  direction_sign = +1 if omega_lim_is_largest else -1
  sampling_omegas = [omega_limit - direction_sign * delta
                     for delta in scan_delta_omegas]
  # In general, we have to make sure that we have geometrically spaced
  # good-quality trajectory-points available.
  refined_samples = []
  for omega in sampling_omegas:
    available_data = re_pot_stat_zs_by_omega.get(omega)
    if available_data and available_data[1] <= threshold_stationarity:
      refined_samples.append((omega,) + available_data)
    else:
      # No good-quality data available, we have to refine an estimate.
      if verbose:
        report(
            f'resampling - available_data: {omega, available_data}, '
            f'fzs_est={fzs_est1(omega).round(7).tolist()!r}')
      omega_pot_stat_zs = refine_omega_zs(
          omega, fzs_est1(omega), sugra=sugra, e7=e7)
      refined_samples.append(omega_pot_stat_zs)
      re_pot_stat_zs_by_omega[omega] = omega_pot_stat_zs[1:]
  get_v70_diverging = lambda zs: v70_from_7z(zs * a_diverging_zs_mask, e7=e7)
  get_v70_finite = lambda zs: v70_from_7z(zs * (1 - a_diverging_zs_mask), e7=e7)
  def get_scaled_boosted_theta_so8(omega, v70, potential):
    return get_boosted_theta_so8(omega, v70, e7=e7) * (-6 / potential)**.5
  thetas_diverging = [
      get_scaled_boosted_theta_so8(omega, get_v70_diverging(zs), potential)
      for omega, potential, _, zs in refined_samples]
  v70s_finite = [get_v70_finite(zs)
                 for omega, potential, _, zs in refined_samples]
  v70s_finite_lim = accelerated_tensor_limit(v70s_finite)
  theta_lim = accelerated_tensor_limit(thetas_diverging)
  theta_gaugeability_violations = (
      gaugings.get_gaugeability_condition_violations(theta_lim, atol=0.1))
  if theta_gaugeability_violations:
    raise ValueError(
        f'theta_lim is not gaugeable: {theta_gaugeability_violations!r}')
  return (gaugings.get_gauging_from_theta(theta_lim),
          v70s_finite_lim,
          re_pot_stat_zs_by_omega)


def plot_trajectory(
    sugra,
    trajectory_fn_zs,
    # Typically, numpy.linspace(omega_min, omega_max, 200) or some such thing.
    plot_omegas,
    z_selectors,
    *,
    title=None,
    # Per z-coordinate, sequence of: (omega_value, label).
    per_z_special_omegas=(),
    num_nongoldstone_jacobian_singular_values=14,
    refined_points=False,
    z_styles=('#ff0000', '#ff8800', '#88ff00', '#44ff44',
              '#00cccc', '#0000ff', '#cc00cc'),
    fontsize=8,
    margin=5,
    z_index_offset=1,
    show=False,
    filename=None):
  """Plots a SL2x7 trajectory."""
  # TODO(tfish): Properly flesh out the docstring.
  graph_omegas = numpy.array(sorted(
      set(plot_omegas) | set(
          itertools.chain.from_iterable(
              [[omega for omega, *_ in special_omegas]
               for special_omegas in per_z_special_omegas]))))
  graph_zs = numpy.stack([trajectory_fn_zs(omega, refined=refined_points)
                          for omega in graph_omegas],
                         axis=0)
  figs = [pyplot.figure() for _ in (1, 2, 3)]
  ax1, ax2, ax3 = (fig.gca() for fig in figs)
  ax1.grid()
  ax1.set_aspect('equal')
  ys = numpy.linspace(0, 2* numpy.pi, 1001)
  ax1.plot(numpy.cos(ys), numpy.sin(ys), '-k')
  for k, (z_index, sign) in enumerate(z_selectors):
    special_omegas = per_z_special_omegas[k]
    zk = sign * graph_zs[:, z_index]
    ax1.plot(zk.real, zk.imag, color=z_styles[k],
             label=rf'$z_{k + z_index_offset}(\omega)$')
    special_zks = numpy.array(
        [trajectory_fn_zs(omega, refined=refined_points)[z_index] * sign
         for omega, *_ in special_omegas])
    ax1.plot(special_zks.real, special_zks.imag, 'ok', markersize=2)
    for (_, wtext, woffset), z in zip(special_omegas, special_zks):
      z_shifted = z + 0.045 - 0.02j + woffset
      ax1.annotate(wtext, (z_shifted.real, z_shifted.imag), fontsize=fontsize)
  ax1.legend(loc='best')
  #
  if title is not None:
    ax1.set_title(title)
  #
  narrowed_graph_omegas = graph_omegas[margin:-margin]
  narrowed_graph_zs = graph_zs[margin:-margin]
  ax2.grid()
  ax2.set_xlabel(r'$\omega/\pi$')
  def pot_stat_7z(zs, omega):
    return sugra.potential_and_stationarity(
        v70_from_7z(zs, e7=sugra.e7), t_omega=mu.tff64(omega))
  graph_pot_stat = [
      pot_stat_7z(zs, omega)
      for zs, omega in zip(narrowed_graph_zs,
                           narrowed_graph_omegas)]
  ax2.plot(
      narrowed_graph_omegas / numpy.pi,
      [numpy.arcsinh(pot) for pot, stat in graph_pot_stat],
      '-b', label=r'$\operatorname{asinh}(\mathrm{Potential})$')
  ax2.plot(
      narrowed_graph_omegas / numpy.pi,
      [10 + numpy.log10(max(stat, 1e-100)) / 2 for pot, stat in graph_pot_stat],
      '-r', label=r'$10 + \operatorname{log}_{10}(|\nabla P|)$')
  ax2.legend(loc='best')
  #
  ax3.grid()
  ax3.set_xlabel(r'$\omega/\pi$')
  ax3.set_ylabel(r'$\operatorname{asinh}(\mathrm{JacobianSingularValue})$')
  singular_value_data = []
  for omega, zs in zip(narrowed_graph_omegas, narrowed_graph_zs):
    v70 = v70_from_7z(zs)
    num_goldstone, svd_s = sugra.jac_stationarity_singular_values(
        v70, omega)
    singular_value_data.append((num_goldstone, svd_s))
  min_num_goldstone = min((n for n, _ in singular_value_data))
  plot_index_start = (
      -min_num_goldstone - num_nongoldstone_jacobian_singular_values - 1)
  plot_index_end = -min_num_goldstone - 1
  singular_values_to_plot = numpy.array(
      [v for _, v in singular_value_data])[:, plot_index_start:plot_index_end]
  for num_singular_value in range(num_nongoldstone_jacobian_singular_values):
    ax3.plot(narrowed_graph_omegas / numpy.pi,
             numpy.arcsinh(singular_values_to_plot[:, num_singular_value]),
             '-k')
  #
  if show:
    for fig in figs:
      fig.show()
  if filename:
    base, ext = os.path.splitext(filename)
    for num_fig, fig in enumerate(figs):
      fig.savefig(f'{base}_{num_fig}{ext}')
  return figs, singular_values_to_plot
