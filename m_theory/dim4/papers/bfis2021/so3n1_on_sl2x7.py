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

"""Obtaining the trajectory of the SO(3) N=1 vacuum on SL(2)x7.

Producing all the artefacts:
  python3 -i -m dim4.papers.bfis2021.so3n1_on_sl2x7 all

"""

import os
# For interactive debugging only.
import pdb  # pylint:disable=unused-import
import pprint

from dim4.papers.bfis2021 import analyze_sl2x7
from dim4.papers.bfis2021 import u4xr12_boundary_equilibrium
from dim4.so8.src import dyonic
from dim4.theta.src import gaugings
from m_theory_lib import algebra
from m_theory_lib import m_util as mu

from matplotlib import pyplot
import numpy


def get_theta_u4xr12(c=1.0):
  """Returns the Dyonic-U(4)|xR12 Theta-tensor."""
  spin8 = algebra.g.spin8
  su8 = algebra.g.su8
  theta = numpy.zeros([56, 133])
  d6 = numpy.diag([0.0] * 6 + [1.0] * 2)
  cd6 = numpy.eye(8) - d6
  theta[:28, 105:] += (-1 / 64.0) * mu.nsum(
      'Iij,Kkl,ijab,klcd,ac,bd->IK',
      su8.m_28_8_8, su8.m_28_8_8,
      spin8.gamma_vvss, spin8.gamma_vvss,
      cd6, numpy.eye(8))
  theta[28:, 105:] += (-1 / 64.0) * mu.nsum(
      'Iij,Kkl,ijab,klcd,ac,bd->IK',
      su8.m_28_8_8, su8.m_28_8_8,
      spin8.gamma_vvss, spin8.gamma_vvss,
      c * d6, numpy.eye(8))
  theta[:28, :35] += -(1 / 16.0) * mu.nsum(
      'Iij,Aab,ijcd,ac,bd->IA',
      su8.m_28_8_8,
      su8.m_35_8_8,
      spin8.gamma_vvss,
      numpy.eye(8), d6)
  theta[28:, :35] += -(1 / 16.0) * mu.nsum(
      'Iij,Aab,ijcd,ac,bd->IA',
      su8.m_28_8_8,
      su8.m_35_8_8,
      spin8.gamma_vvss,
      numpy.eye(8), c * d6)
  return theta


if __name__ == '__main__':
  target_dir = mu.home_relative('tmp/traj_so3n1')
  trajectory_npy_filename = os.path.join(
      target_dir, 'trajectory_so3n1.npy')
  sl2x7 = algebra.g.e7.sl2x7[:2, :, :].reshape(14, 133)
  subspace_an = sl2x7[:, :70].T
  sugra = dyonic.SO8c_SUGRA(subspace_an=subspace_an)
  ds_step = 0.003
  scan_boundary_gauging_num_samples = 50
  scan_file = os.path.join(target_dir, 'u4xr12_equilibria.csv')
  analyzed_file = os.path.join(target_dir, 'u4xr12_equilibria_analyzed.pytxt')
  os.makedirs(target_dir, exist_ok=True)


if mu.arg_enabled(__name__, 'compute_trajectory'):
  print('# Computing SO(3) N=1 trajectory on SL2x7...')
  v14 = analyze_sl2x7.v14_from_7z(analyze_sl2x7.get_7z_from_bfp_z123(
      # Numbers match Eq. (4.31) in BFP, https://arxiv.org/abs/1909.10969
      (0.1696360+0.1415740j, 0.4833214+0.3864058j, -0.3162021-0.5162839j)))
  v70_so3n1 = subspace_an.dot(v14)
  # Check that we do have the correct equilibrium.
  pot, stat = sugra.potential_and_stationarity(v70_so3n1,
                                               t_omega=mu.tff64(0.0))
  assert abs(-13.84096 - pot) < 1e-4 and stat < 1e-8
  dyonic.analyze_omega_deformation(
      mu.home_relative(target_dir),
      v70_so3n1,
      ds=ds_step)
  glob_pos, glob_neg = (
      os.path.join(target_dir, f'S1384096/omega_0.0000_{tag}_*.log')
      for tag in ('pos', 'neg'))
  tdata = dyonic.collect_trajectory_logs(glob_pos, glob_neg)
  numpy.save(trajectory_npy_filename, tdata)


if mu.arg_enabled(__name__, 'extrapolate_and_plot'):
  print('# Extrapolating trajectory and plotting...')
  tdata = numpy.load(trajectory_npy_filename)
  omega_min, omega_max = (-0.25 * numpy.pi), (0.5 * numpy.pi)
  pot_stat_zs_js_by_omega = (
      analyze_sl2x7.get_pot_stat_zs_js_by_omega_from_trajectory_data(tdata))
  trajectory_fn_zs = analyze_sl2x7.get_trajectory_fn_zs(
      sugra,
      {omega: psz[2] for omega, psz in pot_stat_zs_js_by_omega.items()},
      omega_min, omega_max)
  figs, singular_values = analyze_sl2x7.plot_trajectory(
      sugra,
      trajectory_fn_zs,
      numpy.linspace(omega_min, omega_max, 200),
      [(0, +1), (2, -1), (6, +1)],  # z_selectors,
      z_styles=('#00cccc', '#0000ff', '#ff0000'),
      per_z_special_omegas=(
          [(0, '0', 0), (numpy.pi/8, r'$\pi/8$', 0)],
          [(0, '0', -0.07+0.04j), (numpy.pi/8, r'$\pi/8$', 0)],
          [(0, '0', 0),
           (numpy.pi/8, r'$\pi/8$', 0),
           (numpy.pi/2, r'$\pi/2$', -0.16-0.04j),
           (-numpy.pi/4, r'$-\pi/4$', -0.18-0.07j),
           ],
      ),
      refined_points=True,
      filename=os.path.join(target_dir, 'traj_so3n1_sl2z7.pdf'))
  for fig in figs:
    pyplot.close(fig)


if mu.arg_enabled(__name__, 'get_boundary_gauging'):
  print('# Obtaining boundary gauging...')
  tdata = numpy.load(trajectory_npy_filename)
  pot_stat_zs_js_by_omega = (
      analyze_sl2x7.get_pot_stat_zs_js_by_omega_from_trajectory_data(tdata))
  boundary_gauging, v70_finite_lim, ext_pot_stat_zs_by_omega = (
      analyze_sl2x7.get_boundary_gauging(
          sugra,
          numpy.pi / 2,
          # Here, we have to trim off the already-added extrapolated points
          # again.
          {omega: pszj[:-1]
           for omega, pszj in sorted(pot_stat_zs_js_by_omega.items())[1:-1]}))
  boundary_gauging.save(os.path.join(target_dir, 'boundary_gauging.npz'))
  numpy.save(os.path.join(target_dir, 'v70_finite_lim.npy'), v70_finite_lim)


if mu.arg_enabled(__name__, 'scan_u4xr12'):
  print('# Scan for equilibria in the (for now conjectured) boundary-gauging.')
  # Here, we only do a small scan. The specific solution used further down
  # was found with index #1715 in a deeper scan.
  mu.rm(scan_file)
  theta_sugra = gaugings.Dim4SUGRA(
      get_theta_u4xr12(c=1.0),
      gaugeability_atol=1e-10)
  sols = []
  for nn, (pot, stat, params) in zip(
      range(scan_boundary_gauging_num_samples),
      theta_sugra.scan(
          x0s=theta_sugra.get_generator_x0s(seed=2, scale=0.25),
          minimize_kwargs=dict(default_maxiter=2000),
          verbosity='SF')):
    sol = nn, pot, stat, *params.tolist()
    sols.append(sol)
    print(f'### nn={nn} P={pot:.8f} S={stat:.6g}\n')
    with open(scan_file, 'at') as h_out:
      print(','.join(map(repr, sol)), file=h_out)


if mu.arg_enabled(__name__, 'analyze_u4xr12_scan'):
  # A deeper scan would reveal 24 different critical points.
  # There might be more.
  stationarity_limit = 1e-14
  theta_sugra = gaugings.Dim4SUGRA(
      get_theta_u4xr12(c=1.0),
      gaugeability_atol=1e-10)
  scanned = list(mu.csv_numdata(scan_file))
  analyzed = {}
  for row in scanned:
    num_row = int(row[0])
    if num_row % 10 == 0:
      print(f'Row {num_row}...')
    if row[2] > stationarity_limit:
      continue  # Skip bad data.
    m_grav = theta_sugra.gravitino_masses_from_position(row[-70:])
    key = f'{row[1]:.6f}'
    analyzed.setdefault(key, []).append(
        (num_row, ' '.join(f'{m:+.4f}' for m in m_grav)))
  with open(analyzed_file, 'wt') as h_out:
    h_out.write(pprint.pformat(analyzed, width=120))


if mu.arg_enabled(__name__, 'align_u4xr12_equilibrium'):
  v70 = numpy.array(u4xr12_boundary_equilibrium.v70)
  theta_sugra = gaugings.Dim4SUGRA(
      get_theta_u4xr12(c=1.0),
      gaugeability_atol=1e-10,
      # Target mass spectrum (from anlytic-continuation boundary gauging):
      # (Details are discussed in u4xr12_boundary_equilibrium.py)
      stationarity_tweak=('M2G', [41/9] * 3 + [4.0] * 4 + [1.0]))
  eq_info = theta_sugra.find_equilibrium(
      v70, verbosity='S',
      minimize_kwargs=dict(default_gtol=1e-30, default_maxiter=10**4))
  phys = theta_sugra.get_physics(eq_info[-1], {})
  print('=== The U4xR12 Equilibrium ===')
  print(theta_sugra.show_physics_text(phys))
  with open(os.path.join(target_dir, 'physics_u4xr12.tex'), 'wt') as h_tex:
    h_tex.write(theta_sugra.show_physics_tex(phys)[0])
    h_tex.write('\n\n%%%\n\n')
    h_tex.write(theta_sugra.show_physics_tex(phys)[1])
  #
  gauging_bg = gaugings.Gauging.load(
      os.path.join(target_dir, 'boundary_gauging.npz'))
  theta_bg = gauging_bg.theta
  v70_finite_lim = numpy.load(
      os.path.join(target_dir, 'v70_finite_lim.npy'))
  sugra_bg = gaugings.Dim4SUGRA(theta_bg,
                                check_gaugeability=False)
  opt_pot, opt_stat, opt_v70 = sugra_bg.find_equilibrium(
      # The way the boundary gauging was constructed,
      # we have to take the *negative* finite-part-limit-v70.
      -v70_finite_lim, verbosity='S')
  opt_phys = sugra_bg.get_physics(opt_v70, metadata={})
  print('=== The Boundary Gauging Equilibrium ===')
  print(sugra_bg.show_physics_text(opt_phys))
  with open(os.path.join(target_dir, 'physics_bg.tex'), 'wt') as h_tex:
    h_tex.write(theta_sugra.show_physics_tex(opt_phys)[0])
    h_tex.write('\n\n%%%\n\n')
    h_tex.write(theta_sugra.show_physics_tex(opt_phys)[1])


if mu.arg_enabled(__name__, 'plot_trajectory70'):
  print('# Plotting "70-parameters" trajectory...')
  tdata = numpy.load(trajectory_npy_filename)
  dyonic.plot_trajectory(
      tdata,
      filename=os.path.join(target_dir, 'traj_so3n1_70.pdf'))


if mu.arg_enabled(__name__, 'get_trajectory_story'):
  print('# Getting trajectory story...')
  tdata = numpy.load(trajectory_npy_filename)
  story_dir = os.path.join(target_dir, 'traj_story')
  os.makedirs(story_dir, exist_ok=True)
  story = dyonic.trajectory_get_story(sugra, tdata, story_dir)
  with open(os.path.join(story_dir, 'story.pytxt'), 'wt') as h_out:
    h_out.write(repr(story))
