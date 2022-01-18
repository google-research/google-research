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

"""Analyzing the potentials of wrapped-branes models.

This is mostly a pedagogical example.
TensorFlow-based analysis really shines if the number of true scalars
(i.e. with degeneracies removed) is >= ca. 20.

The scaffolding included here makes the code quite easy to adopt to
other reasonably straightforward cases. One only needs to introduce a
function that computes the scalar potential like dim7_potential()
below (remembering that one is operating on TensorFlow objects rather
than numerical quantities), and then call:

  scan({{number_of_scalars}}, {{potential_function}}).

"""

import pdb  # For interactive debugging only.

import collections
import dataclasses
import numpy
import scipy.optimize
import sys
import tensorflow as tf

from m_theory_lib import m_util

# The actual problem definitions.
from wrapped_branes import potentials


@dataclasses.dataclass(frozen=True)
class Solution(object):
  potential: float
  stationarity: float
  pos: numpy.ndarray


def scan_for_critical_points(
    problem,
    starting_points,
    stationarity_threshold=1e-4,
    mdnewton=True,
    debug=True,
    *problem_extra_args,
    **problem_extra_kwargs):
  """Scans for critical points of a scalar function.

  Args:
    problem: The potential-function specifying the problem.
    starting_points: iterable with starting points to start the search from.
    stationarity_threshold: Upper bound on permissible post-optimization
      stationarity for a solution to be considered good.
    debug: Whether to print newly found solutions right when they
     are discovered.
    problem_extra_args: Extra positional arguments for the problem-function.
    problem_extra_kwargs: Extra keyword arguments for the problem-function.

  Yields:
    A `Solution` numerical solution.
  """
  def f_problem(pos):
    return problem(pos, *problem_extra_args, **problem_extra_kwargs)
  tf_stat_func = m_util.tf_stationarity(f_problem)
  tf_grad_stat_func = m_util.tf_grad(tf_stat_func)
  tf_grad_pot_func = None
  tf_jacobian_pot_func = None
  if mdnewton:
    tf_grad_pot_func = m_util.tf_grad(f_problem)
    tf_jacobian_pot_func = m_util.tf_jacobian(tf_grad_pot_func)
  for x0 in starting_points:
    val_opt, xs_opt = m_util.tf_minimize(tf_stat_func, x0,
                                         tf_grad_func=tf_grad_stat_func,
                                         precise=False)
    if val_opt > stationarity_threshold:
      continue  # with next starting point.
    # We found a point that apparently is close to a critical point.
    t_xs_opt = tf.constant(xs_opt, dtype=tf.float64)
    if not mdnewton:
      yield Solution(potential=f_problem(t_xs_opt).numpy(),
                     stationarity=tf_stat_func(t_xs_opt).numpy(),
                     pos=xs_opt)
      continue  # with next solution.
    # We could use MDNewton to force each gradient-component
    # of the stationarity condition to zero. It is however
    # more straightforward to instead do this directly
    # for the gradient of the potential.
    *_, xs_opt_mdnewton = m_util.tf_mdnewton(
      f_problem,
      t_xs_opt,
      maxsteps=4,
      debug_func=None,
      tf_grad_func=tf_grad_pot_func,
      tf_jacobian_func=tf_jacobian_pot_func)
    t_xs_opt_mdnewton = tf.constant(xs_opt_mdnewton, dtype=tf.float64)
    yield Solution(potential=f_problem(t_xs_opt_mdnewton).numpy(),
                   stationarity=tf_stat_func(t_xs_opt_mdnewton).numpy(),
                   pos=xs_opt_mdnewton)


if __name__ == '__main__':
  # Set numpy's default array-formatting width to large width.
  numpy.set_printoptions(linewidth=200)
  if len(sys.argv) != 2 or sys.argv[-1] not in potentials.PROBLEMS:
    sys.exit('\n\nUsage: python3 -i -m wrapped_branes.wrapped_branes {problem_name}.\n'
             'Known problem names are: %s' % ', '.join(
                 sorted(potentials.PROBLEMS)))
  problem = potentials.PROBLEMS[sys.argv[-1]]
  rng = numpy.random.RandomState(seed=0)
  def gen_x0s():
    while True:
      yield rng.normal(scale=0.15, size=problem.num_scalars)
  solutions_iter = scan_for_critical_points(
    problem.tf_potential,
    gen_x0s(),
    mdnewton=True,
    **problem.tf_potential_kwargs)
  for n, solution in zip(range(100), solutions_iter):
    print('P=%+12.8f S=%8.3g at: %s' % (solution.potential,
                                        solution.stationarity,
                                        numpy.round(solution.pos, 4)))
