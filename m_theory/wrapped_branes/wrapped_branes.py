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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import dataclasses
import numpy
import scipy.optimize
import sys
import tensorflow.compat.v1 as tf

# The actual problem definitions.
from wrapped_branes import potentials

@dataclasses.dataclass(frozen=True)
class Solution(object):
  potential: float
  stationarity: float
  scalars: numpy.ndarray


def call_with_evaluator(
    num_scalars,
    problem,
    f,
    punish_scalars_beyond=2.0,
    problem_args=(), problem_kwargs={},
    f_args=(), f_kwargs={}):
  """Calls `f` with potential-stationarity-gradient-evaluator in TF context.

  Sets up the TensorFlow graph that implements a wrapped-branes potential,
  such as (3.20) of arXiv:1906.08900.

  Args:
    num_scalars: The number of scalars.
    problem: The function that specifies the scalar potential computation
      for the problem under study. This must have the following siguature:
      problem(scalars) -> potential.
    f: The function to call with an evaluator (and optionally extra arguments).
    punish_scalars_beyond: Threshold numerical magnitude of scalar parameters
      beyond which a regularizing term drives optimization back to a physically
      plausible region.
    problem_args: Extra positional arguments for `problem`.
    problem_kwargs: Extra keyword arguments for `problem`.
    f_args: Extra positional arguments for `f`
    f_kwargs: Extra keyword arguments for `f`.

  Returns:
    The result of f(evaluator, *f_args, **f_kwargs) as evaluated in a TensorFlow
    session context set up as required by the evaluator.
  """
  graph = tf.Graph()
  with graph.as_default():
    t_input = tf.Variable(numpy.zeros(num_scalars), dtype=tf.float64)
    t_potential = problem(t_input, *problem_args, **problem_kwargs)
    t_grad_potential = tf.gradients(t_potential, [t_input])[0]
    t_stationarity = tf.reduce_sum(tf.square(t_grad_potential))
    # Punish large scalars.
    # This drives the search away from 'unphysical' regions.
    t_eff_stationarity = t_stationarity + tf.reduce_sum(
        tf.sinh(  # Make stationarity-violation grow rapidly for far-out scalars.
            tf.nn.relu(tf.abs(t_input) - punish_scalars_beyond)))
    t_grad_stationarity = tf.gradients(t_eff_stationarity, [t_input])[0]
    with tf.compat.v1.Session() as session:
      session.run([tf.compat.v1.global_variables_initializer()])
      def evaluator(scalars):
        return session.run(
            (t_potential, t_stationarity, t_grad_stationarity),
            feed_dict={t_input: scalars})
      return f(evaluator, *f_args, **f_kwargs)


def scan(
    num_scalars,
    problem,
    num_solutions=40,  # We normally would want to run many more scan trials.
    seed=1,
    scale=0.15,
    stationarity_threshold=1e-4,
    debug=True,
    *problem_extra_args,
    **problem_extra_kwargs):
  """Scans for critical points in the scalar potential.

  Args:
    num_scalars: The number of scalars.
    problem: The function specifying the problem,
      such as `call_with_cgr_psg_evaluator`.
    num_solutions: Number of critical points to collect before returning.
    seed: Random number generator seed for generating starting points.
    scale: Scale for normal-distributed search starting point coordinates.
    stationarity_threshold: Upper bound on permissible post-optimization
      stationarity for a solution to be considered good.
    debug: Whether to print newly found solutions right when they
     are discovered.
    problem_extra_args: Extra positional arguments for the problem-function.
    problem_extra_kwargs: Extra keyword arguments for the problem-function.

  Returns:
    A list of `Solution` with numerical solutions.
  """
  # Use a seeded random number generator for better reproducibility
  # (but note that scipy's optimizers may themselves use independent
  # and not-easily-controllable random state).
  rng = numpy.random.RandomState(seed=seed)
  def get_x0():
    return rng.normal(scale=scale, size=num_scalars)
  def do_scans(evaluator):
    def f_opt(scalars):
      unused_potential, stationarity, unused_grad = evaluator(scalars)
      return stationarity
    def fprime_opt(scalars):
      unused_potential, unused_stationarity, grad = evaluator(scalars)
      return grad
    ret = []
    while len(ret) < num_solutions:
      opt = scipy.optimize.fmin_bfgs(f_opt, get_x0(), fprime=fprime_opt,
                                     maxiter=10**5)
      opt_pot, opt_stat, opt_grad = evaluator(opt)
      if numpy.isnan(opt_pot) or not opt_stat < 1e-4:
        continue  # Optimization ran into a bad solution.
      solution = Solution(potential=opt_pot,
                          stationarity=opt_stat,
                          scalars=opt)
      ret.append(solution)
      if debug:
        print(solution)
    return ret
  return call_with_evaluator(num_scalars, problem, do_scans,
                             problem_args=problem_extra_args,
                             problem_kwargs=problem_extra_kwargs)


if __name__ == '__main__':
  # Set numpy's default array-formatting width to 160 characters.
  numpy.set_printoptions(linewidth=160)
  if len(sys.argv) != 2 or sys.argv[-1] not in potentials.PROBLEMS:
    sys.exit('\n\nUsage: python3 -i -m dim5.cgr.cgr_theory {problem_name}.\n'
             'Known problem names are: %s' % ', '.join(
                 sorted(potentials.PROBLEMS)))
  problem = potentials.PROBLEMS[sys.argv[-1]]
  solutions = scan(problem.num_scalars,
                   problem.tf_potential,
                   **problem.tf_potential_kwargs)
  for sol in sorted(solutions, key=lambda s: s.potential):
    print('P=%+8.3f S=%8.3f at: %s' % (sol.potential, sol.stationarity,
                                       numpy.round(sol.scalars, 4)))
  # At this point, a summary of `solutions` has been printed, and accurate data
  # is available in `solutions`. If the process was started with
  # `python -i -m ...`, this can now be inspected from the Python prompt.
