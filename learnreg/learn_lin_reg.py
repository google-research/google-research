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

"""Code to learn coefficients of an optimal linear regularizer.

This implements the LearnLinReg algorithm described in the paper,
"Learning Optimal Regularizers".
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
from scipy import optimize
import six
from six.moves import xrange
from typing import Iterable, List, Mapping, Sequence, Text, Tuple


# Types used in annotations.
#
# A mapping from variable name to coefficient.
VariableValues = Mapping[Text, float]
# A constraint in a linear program.
Constraint = Tuple[VariableValues, float]


DataPoint = collections.namedtuple(
    'DataPoint',
    (
        'test_loss',       # type: float
        'training_loss',   # type: float
        'feature_vector',  # type: List[float]
    )
)


def learn_linear_regularizer(
    data_points,  # type: Sequence[DataPoint],
    verbose=False
):
  # type: (...) -> Tuple[float, List[float]]
  """Returns coefficients for best linear regularizer.

  Here "best" means the linear regularizer whose argmin yields the best
  possible test loss over the observed data points.  See the
  LearnLinReg algorithm and its analysis in section 3 of the paper.

  Args:
    data_points: a sequence of DataPoints.  Each data point corresponds
      to a run using different regularization hyperparameters.
    verbose: whether to print details of algorithm progress

  Returns:
    a tuple (alpha, coefficients), with the property that for each data
    point p:

      p.training_loss + coefficients * p.feature_vector >= p.test_loss
  """

  num_features = len(data_points[0].feature_vector)
  if any(len(p.feature_vector) != num_features for p in data_points):
    raise ValueError('Not all feature vectors have same length.')

  # The LP solver is happier if training and test loss have roughly the
  # same scale, so rescale training loss to make this the case.
  sum_test_loss = sum(p.test_loss for p in data_points)
  sum_training_loss = sum(p.training_loss for p in data_points)
  if sum_training_loss < 0:
    raise ValueError('Training losses should be non-negative.')
  if sum_training_loss == 0.:
    scale_factor = 1.
  else:
    scale_factor = sum_test_loss / sum_training_loss
  rescaled_data_points = [
      rescale_training_loss(point, scale_factor)
      for point in data_points
  ]

  # Sort points in ascending order of test loss.
  sorted_points = sorted(rescaled_data_points, key=lambda p: p.test_loss)

  for j_hat in xrange(len(sorted_points)):
    if verbose:
      print(' j_hat =', j_hat, 'test loss =',
            sorted_points[j_hat].test_loss)

    # Let L_i be the ith test loss value, and let q_i be the ith feature
    # vector.  Want to solve the following linear program:
    #
    # Minimize: sum_i D_i
    #
    # Subject to:
    # alpha > 0
    # D_i >= 0 for all i
    # f_i = alpha*L_i + D_i  for all i
    # f_i = ~L_i + Lambda*q_i  for all i
    # f_{j_hat} <= f_i for all i

    n = len(sorted_points)
    cost = {'D_%d' % i: 1 for i in xrange(n)}

    def get_equality_constraints(i, point):
      # type: (int, DataPoint) -> Constraint
      """Returns list of equality constraints for given DataPoint."""

      # Constraint: f_i = ~L_i + Lambda*q_i
      d0 = {'f_%d' % i: -1}
      for j, v in enumerate(point.feature_vector):
        d0['Lambda_%d' % j] = v
      c0 = (d0, -point.training_loss)

      # Constraint: f_i = alpha*L_i + D_i
      d1 = {'f_%d' % i: 1, 'alpha': -point.test_loss, 'D_%d' % i: -1}
      c1 = (d1, 0)

      return [c0, c1]

    equality_constraints = [
        constraint
        for i, point in enumerate(sorted_points)
        for constraint in get_equality_constraints(i, point)
    ]

    j = argmin([test_loss for test_loss, _, _ in sorted_points])
    assert j == 0
    upper_bounds = (
        # alpha > 0
        [({'alpha': -1}, -0.01)] +
        # D_i >= 0 for all i, or equivalently:
        # -D_i <= 0 for all i.
        [({'D_%d' % i: -1}, 0) for i in xrange(n)] +
        # f_{j_hat} - f_i <= 0 for all i.
        [({'f_%d' % j_hat: 1, 'f_%d' % i: -1}, 0)
         for i in xrange(n) if i != j_hat]
    )

    variables = solve_lp(cost, equality_constraints, upper_bounds)
    if variables is not None:
      rescaled_coefficients = [
          variables['Lambda_%d' % i]
          for i in xrange(num_features)
      ]
      rescaled_alpha = variables['alpha']

      alpha = rescaled_alpha / scale_factor
      coefficients = [c / scale_factor for c in rescaled_coefficients]
      return alpha, coefficients

  raise ValueError('Could not upper bound test loss.')


def rescale_training_loss(point, scale_factor):
  """Returns DataPoint with rescaled training loss."""
  return DataPoint(point.test_loss,
                   point.training_loss * scale_factor,
                   point.feature_vector)


def solve_lp(
    cost,                  # type: VariableValues
    equality_constraints,  # type: Iterable[Constraint]
    upper_bounds           # type: Iterable[Constraint]

):
  # type: (...) -> Optional[VariableValues]
  """Solves a linear program.

  WARNING: this constrains all variables to be non-negative.

  Args:
    cost: a linear cost function, represented as a mapping from variable to
      coefficient (absent coefficients are implicily 0)
    equality_constraints: an iterable of pairs (d, x), where d is a
      variable-to-coefficient mapping implicitly representing a vector,
      and the dot product between this vector and the variable vector is
      constrained to be x.
    upper_bounds: similar to equality_constraints, but x is an upper bound

  Returns:
    mapping from variable name to optimal value, or None if the LP is not
    feasible
  """

  all_vars = set(six.iterkeys(cost))
  for d, _ in itertools.chain(equality_constraints, upper_bounds):
    all_vars.update(six.iterkeys(d))
  varnames = sorted(all_vars)

  def to_vector(d):
    # type: (Mapping[Text, float]) -> List[float]
    return [d.get(v, 0.) for v in varnames]

  c = to_vector(cost)

  if upper_bounds:
    a_ub = [to_vector(d) for d, _ in upper_bounds]
    b_ub = [u for _, u in upper_bounds]
  else:
    a_ub = None
    b_ub = None

  if equality_constraints:
    a_eq = [to_vector(d) for d, _ in equality_constraints]
    b_eq = [a for _, a in equality_constraints]
  else:
    a_eq = None
    b_eq = None

  # Note: would like to set this to (None, None), but then the LP solver
  # fails when it shouldn't.
  bounds = [(0, None) for _ in all_vars]

  result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub,
                            A_eq=a_eq, b_eq=b_eq,
                            bounds=bounds)
  if not result.success:
    return None
  assert len(result.x) == len(varnames), (result.x, varnames)
  return dict(zip(varnames, result.x))


def argmin(value_list):
  # type: (Sequence) -> int
  """Returns index of minimum sequence element."""
  item = min(enumerate(value_list), key=lambda p: p[1])
  return item[0]

