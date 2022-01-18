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

"""Finds and processes extrema."""

import ast  # For ast.literal_eval() only.
import glob
import os
import pdb  # To simplify interactive debugging.
import pprint

from dim4.so8_supergravity_extrema.code import distillation
from dim4.so8_supergravity_extrema.code import scalar_sector_mpmath
from dim4.so8_supergravity_extrema.code import scalar_sector_tensorflow
from dim4.so8_supergravity_extrema.code import symmetries

import mpmath
import numpy


def read_v70(fileglob):
  """Reads a location.pytxt sparse 70-vector that describes an extremum."""
  files = glob.glob(fileglob)
  if len(files) != 1:
    raise ValueError('Got %d matches for glob %r: %r' % (
        len(files), fileglob, files))
  model = distillation.read_distillate_model(files[0])
  return numpy.dot(model.v70_from_params, model.params)


def read_physics(solution_tag, directory='../extrema'):
  distillate_file = os.path.join(directory, solution_tag, 'location.pytxt')
  physics_file = os.path.join(directory, solution_tag, 'physics.pytxt')
  v70 = read_v70(distillate_file)
  with open(physics_file, 'r') as h:
    physics_info = ast.literal_eval(h.read())
  return v70, physics_info


def analyze_physics(solution_tag, directory='../extrema'):
  v70, physics_info = read_physics(solution_tag, directory=directory)
  v70f = v70.astype(float)
  symm = symmetries.get_residual_gauge_symmetry(v70f)
  semisimple, u1s = symmetries.decompose_reductive_lie_algebra(symm)
  csym = symmetries.canonicalize_residual_spin3u1_symmetry(symm)
  vsc_ad_branching, spectra = symmetries.spin3u1_physics(
      csym,
      mass_tagged_eigenspaces_gravitinos=physics_info.get('gravitinos', ()),
      mass_tagged_eigenspaces_fermions=physics_info.get('fermions', ()),
      mass_tagged_eigenspaces_scalars=physics_info.get('scalars', ()))
  return csym, vsc_ad_branching, spectra


def generate_symmetry_info(solution_tags,
                           directory='../extrema',
                           re_raise_exceptions=True):
  def generator_list(gs):
    return [list(g) for g in gs.T]
  for solution_tag in solution_tags:
    print('S:', solution_tag)
    try:
      csym, vsc_ad_branching, spectra = analyze_physics(solution_tag,
                                                        directory=directory)
      solution_dir = os.path.join(directory, solution_tag)
      with open(os.path.join(solution_dir, 'symmetry.pytxt'), 'w') as h:
        h.write(pprint.pformat(dict(
            u1s=generator_list(csym.u1s),
            semisimple_part=generator_list(csym.semisimple_part),
            # Only for reference. We did not try to rotate these
            # Cartan-generators into some nice embedding, so these
            # are not overly informative.
            spin3_cartan_gens=generator_list(csym.spin3_cartan_gens),
            branching=vsc_ad_branching,
            spectra=spectra)))
    except Exception as exn:
      print('FAILED: %r' % repr(exn))
      if re_raise_exceptions:
        raise


def process_raw_v70(raw_v70, work_dir,
                    newton_steps=4,
                    skip_gradient_descent=False):
  """Processes a raw 70-vector."""
  model, _ = distillation.distill(raw_v70,
                                  target_digits_position=25,
                                  newton_steps=newton_steps,
                                  skip_gradient_descent=skip_gradient_descent)
  v70 = distillation.v70_from_model(model)
  sinfo = scalar_sector_mpmath.mpmath_scalar_manifold_evaluator(v70)
  tag = scalar_sector_tensorflow.S_id(sinfo.potential)
  data_dir = os.path.join(work_dir, tag)
  os.makedirs(data_dir, exist_ok=True)
  location_filename = os.path.join(data_dir, 'location.pytxt')
  with open(location_filename, 'wt') as out_handle:
    distillation.write_model(out_handle, model, dict(
      potential=str(sinfo.potential),
      stationarity=str(sinfo.stationarity)))
  physics = distillation.explain_physics(location_filename)
  with open(os.path.join(data_dir, 'physics.pytxt'),
            'wt') as out_handle:
    # The output file provides angular momenta and charges for the
    # mass eigenstates.
    out_handle.write(pprint.pformat(physics))
  generate_symmetry_info([tag], directory=work_dir)
  print('\n=== Done. Data is in %s/*.pytxt ===\n' % data_dir)


def scan_for_solutions(seed=1, scale=0.1,
                       stationarity_threshold=1e-6,
                       output_filename=None,
                       rpow=None,
                       susy_regulator=None):
  return scalar_sector_tensorflow.scan(
      output_filename,
      rpow=rpow,
      susy_regulator=susy_regulator,
      stationarity_threshold=stationarity_threshold,
      seed=seed, scale=scale)


def demo(work_dir='EXAMPLE_SOLUTIONS',
         num_runs=3, seed=1, scale=0.1,
         stationarity_threshold=1e-6,
         rpow=None, susy_regulator=None):
  """Demonstrates basic usage of extrema-scanning code."""
  os.makedirs(work_dir, exist_ok=True)
  output_filename = os.path.join(work_dir, 'scan_log.txt')
  scan_iter = scan_for_solutions(
      output_filename=output_filename,
      rpow=rpow,
      susy_regulator=susy_regulator,
      stationarity_threshold=stationarity_threshold,
      seed=seed, scale=scale)
  solutions = []
  for n in range(num_runs):
    sol = next(scan_iter)
    print('Found: %.5f' % sol[0])
    solutions.append(sol)
  # Pick the solution with lowest cosmological constant. (Highest == -6.0.)
  pot, stat, raw_v70 = min(solutions, key=lambda p_s_v70: p_s_v70[0])
  print(
    '\n=== Distilling P=%.8f/S=%12.6g (this will take a while)... ===\n' % (
      pot, stat))
  process_raw_v70(raw_v70, work_dir)


if __name__ == '__main__':
  print('=== Running Demo ===')
  # Setting up `mpmath` global default precision at initialization time.
  # Uses value from the environment variable `MPMATH_DPS`, or a default if unset.
  mpmath.mp.dps = int(os.getenv('MPMATH_DPS', '60'))
  demo()
