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

"""Cartan matrices and Dynkin diagrams.

Generic code for weightspace decompositions and related operations.
"""

import collections
import itertools

import numpy


def get_eigval_for_eigvec(op, eigvec):
  """Given a matrix and an approximate eigenvector, estimates the eigenvalue."""
  return numpy.array(op.dot(eigvec).dot(eigvec.conj()) /
                     eigvec.dot(eigvec.conj()))


def weightspaces_quality(hermitean_operators, weightspace_by_weight):
  """Measures the numerical misalignment of a weightspace-decomposition.

  Args:
    hermitean_operators: [r, N, N]-numpy.ndarray, a collection of `r`
      mutually commuting hermitean [N, N]-matrices (as a 3-index array).
    weightspace_by_weight: {eigenvalues: [d, N]-numpy.ndarray}
      weight-decomposition.

  Returns:
    2-norm of the total violation of
      `hermitean_operators[k].dot(weightspace[n]) =
       = eigenvalues[k] * weightspace[n]`
    over all operator-indices `k` and weight-vector basis indices `n`
   (given the weight).
  """
  deviation = 0
  for weightspace in weightspace_by_weight.values():
    for wvec in weightspace:
      for eigval, op in zip(wvec, hermitean_operators):
        delta = eigval * wvec - op.dot(wvec)
        deviation += numpy.sum(delta * delta)
  return numpy.sqrt(deviation)


def _get_weightspaces0(hermitean_operators,
                       eigval_digits=5,
                       rng=None):
  """(Internal helper)."""
  num_operators = hermitean_operators.shape[0]
  rand_coeffs = rng.normal(size=num_operators)
  random_operator = numpy.einsum('r,rMN->MN', rand_coeffs, hermitean_operators)
  _, eigenbasis = numpy.linalg.eigh(random_operator)
  # Now, for every eigenvector in this eigenbasis, we need to know the vector
  # of eigenvalues for each operator. These are the weight-vectors ("wvecs").
  eigvecs_by_wvec = collections.defaultdict(list)
  for eigvec in eigenbasis.T:
    wvec = tuple(get_eigval_for_eigvec(op, eigvec).round(eigval_digits).item()
                 for op in hermitean_operators)
    eigvecs_by_wvec[wvec].append(eigvec)
  return {wvec: numpy.stack(eigvecs, axis=0)
          for wvec, eigvecs in eigvecs_by_wvec.items()}


def get_weightspaces(hermitean_operators,
                     *,
                     best_of=3,
                     eigval_digits=5,
                     rng=None,
                     check_rtol_atol=(1e-5, 1e-7)):
  """Weight-decomposes a linear space.

  Background (cf: https://en.wikipedia.org/wiki/Weight_(representation_theory) )

  If we have `r` commuting hermitean operators, these
  (physically/quantum mechanically speaking) represent a set of `r`
  "measurement operators" for `r` "charges" that "can be measured at
  the same time in a way so that measurements do not upset one
  another". While each of the `r` operators will in general have
  degenerate (perhaps highly degenerate) eigenvalues, it is possible
  to find a simultaneous eigenbasis of all `r` operators and tag each
  basis (eigen-)vector by the length-`r` "vector of eigenvalues"
  (indexed in parallel to the hermitean operators).
  These vectors-of-eigenvalues are called "weights", and the subspace
  associated with a particular weight generally is substantially
  lower-dimensional than the original space.

  One naive approach to compute weight spaces would start by splitting
  the vector space that the hermitean operators act on into subspaces
  that are eigenspaces of the first hermitean operator, then subdivide
  these according to eigenvalues of the second hermitean operator,
  and-so-on. The problem with this approach is that it leads to a
  "deep" numerical dependency-chain that requires high numerical
  accuracy (which might be hard to obtain). We hence here follow
  an alternative route, observing that a random linear combination
  of the hermitean operators is expected to not have accidental
  degeneracies - "unless we are very unlucky", which we address
  by looking for a pair of random operators that have least mismatch
  between their bases.

  Args:
    hermitean_operators: [r, N, N]-numpy.ndarray, a collection of `r`
      mutually commuting hermitean [N, N]-matrices (as a 3-index array).
    best_of: int, number of trials to produce a weightspace decomposition.
      In principle, determining the simultaneous eigenbasis from a random
      linear combination of operators might produce some accidental
      almost-near-degeneracy, which would negatively impact the numerical
      quality of the decomposition. We counteract this by retrying
      `best_of` times and picking the one that is "numerically best".
    eigval_digits: Rounding accuracy (decimal digits) for weights
      (= vectors-of-eigenvalues).
    rng: Optional random number generator to use for producing
     a random linear combination of generators.
    check_rtol_atol: Optional pair of relative/absolute tolerances.
      If a pair of numbers, these will be used as threshold
      tolerances for a numerical check of hermiticity and commutativity
      of the operators. If `None`, this check is not performed.

  Returns:
    Mapping `{weight: weightspace}`, where `weight` is a vector of eigenvalues
    (rounded to `eigval_digits` decimal digits) indexed in parallel to
    `hermitean_operators`, and `weightspace` is a [d, N]-numpy.ndarray holding
    `d` orthonormal simultaneous eigenvectors of all hermitean operators where
    the eigenvalues correspond to the weight.
  """
  if rng is None:
    # Use fixed seed if no RNG is provided, for reproducibility.
    rng = numpy.random.RandomState(seed=0)
  num_operators = hermitean_operators.shape[0]
  if check_rtol_atol is not None:
    rtol, atol = check_rtol_atol
    h_deviations = numpy.array([numpy.linalg.norm((h - h.T.conj()).ravel())
                                for h in hermitean_operators])
    if not all(numpy.allclose(1 + h_deviations, 1, rtol=rtol, atol=atol)
               for h in hermitean_operators):
      raise ValueError(f'Linear operators are not hermitean '
                       'within allowed tolerance. Deviations: '
                       f'{list(enumerate(h_deviations))}')
    for na, nb in itertools.permutations(range(num_operators), 2):
      ha = hermitean_operators[na]
      hb = hermitean_operators[nb]
      hab = ha @ hb
      hba = hb @ ha
      if not numpy.allclose(hab, hba, rtol=rtol, atol=atol):
        raise ValueError(f'Non-commuting operators: [h[{na}], h[{nb}]] ~ '
                         f'{numpy.linalg.norm((hab - hba).ravel()):.6g}')
  #
  best_quality = float('inf')
  best_pick = None
  for _ in range(best_of):
    ws = _get_weightspaces0(hermitean_operators, eigval_digits=eigval_digits,
                            rng=rng)
    quality = weightspaces_quality(hermitean_operators, ws)
    if quality < best_quality:
      best_pick = ws
      best_quality = quality
  return best_pick
