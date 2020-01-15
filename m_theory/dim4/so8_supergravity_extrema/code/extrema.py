# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Code to process extrema."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast  # For ast.literal_eval() only.
import glob
import os
import pdb  # To simplify interactive debugging only.
import pprint
import time

from dim4.so8_supergravity_extrema.code import distillation
from dim4.so8_supergravity_extrema.code import scalar_sector_mpmath
from dim4.so8_supergravity_extrema.code import scalar_sector_tensorflow
from dim4.so8_supergravity_extrema.code import symmetries

import mpmath
import numpy

# Setting up `mpmath` global default precision at initialization time.
# Uses value from the environment variable `MPMATH_DPS`, or 100 if unset.
mpmath.mp.dps = int(os.getenv('MPMATH_DPS', '100'))


def scan_for_solutions(seed, scale, num_iterations, output_basename):
  """Scans for critical points (with TensorFlow)."""
  scanner = scalar_sector_tensorflow.get_scanner(output_basename)
  return scanner(seed, scale, num_iterations)


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


def demo(work_dir='EXAMPLE_SOLUTIONS'):
  """Demonstrates basic usage of extrema-scanning code."""
  def makedirs(path):
    try:
      # Python2 does not have the exist_ok keyword arg for os.makedirs(),
      # and it would be silly to break Python2/Python3 compatibility simply
      # for that.
      os.makedirs(path, mode=0o755)
    except OSError:
      pass  # Benign, directory already existed.
  def generate_some_solutions():
    # Let us first get some critical points.
    print('\n=== Scanning for solutions ===\n')
    # We pause for a second after each such "stage" message to give the user
    # an opportunity to see what is currently going on, amongst all the
    # messages flying by that come from optimizers whose output can not
    # be suppressed.
    time.sleep(1)
    # This scans for solutions and writes the result to the file
    # given as last arg.
    makedirs(work_dir)
    # A scale of 0.2 for the initial vector only probes a small region
    # "near the origin". We do not expect to find difficult-to-analyze solutions
    # there. A more appropriate starting value for a deeper search would be
    # e.g. 2.0.
    extrema = scan_for_solutions(
        1, 0.2, 10, os.path.join(work_dir, 'SCANS.pytxt'))
    tags = sorted(extrema)
    print('\n=== Found: %s ===\n' % tags)
    time.sleep(1)
    # For this demo, we only process the very first one.
    # First, canonicalize.
    v70_last_extremum = numpy.array(extrema[tags[-1]][0][-1])
    print('\n=== Distilling (this will take a while)... ===\n')
    time.sleep(1)
    distilled, _ = distillation.distill(v70_last_extremum,
                                        target_digits_position=40)
    out_dir = os.path.join(work_dir, tags[-1])
    makedirs(out_dir)
    v70 = distillation.v70_from_model(distilled)
    sinfo = scalar_sector_mpmath.mpmath_scalar_manifold_evaluator(v70)
    with open(os.path.join(out_dir, 'location.pytxt'), 'w') as out_handle:
      distillation.write_model(out_handle, distilled, dict(
          potential=str(sinfo.potential),
          stationarity=str(sinfo.stationarity)))
  #
  location_glob = os.path.join(work_dir, 'S*/location.pytxt')
  location_files = sorted(glob.glob(location_glob))
  if location_files:
    print('Skipping generation of example solution - using earlier result.')
  else:
    generate_some_solutions()
    location_files = sorted(glob.glob(location_glob))
  location_file = location_files[0]
  data_dir = os.path.dirname(location_files[0])
  solution_tag = os.path.basename(data_dir)
  # Load and process the critical point.
  physics = distillation.explain_physics(location_file)
  with open(os.path.join(data_dir, 'physics.pytxt'),
            'w') as out_handle:
    # The output file provides angular momenta and charges for the
    # mass eigenstates.
    out_handle.write(pprint.pformat(physics))
  generate_symmetry_info([solution_tag], directory=work_dir)
  print('\n=== Done. Data is in %s/*.pytxt ===\n' % data_dir)


if __name__ == '__main__':
  print('=== Running Demo ===')
  demo()
