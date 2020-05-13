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

"""Analyzes residual symmetries of solutions.

As all critical points with a rank-2 simple Lie group symmetry have been
known for many years, we can restrict ourselves to a residual Lie symmetry of
Spin(3)^A x U(1)^B. This considerably simplifies the analysis.
"""

import cmath
import collections
import glob
import itertools
import math
import numpy
import os
import pprint
# CAUTION: scipy.linalg.eigh() will produce an orthonormal basis, while
# scipy.linalg.eig(), when used on a hermitean matrix, typically will not
# orthonormalize eigenvectors in degenerate eigenspaces.
# This behavior is not documented properly, but "obvious" when considering
# the underlying algorithm.
import scipy.linalg


from dim4.so8_supergravity_extrema.code import algebra


CanonicalizedSymmetry = collections.namedtuple(
    'CanonicalizedSymmetry',
    ['u1s',  # Sequence of U(1) generators, each as a 28-vector acting on [ik].
     'semisimple_part',  # [28, d]-array, semisimple part of the algebra.
     'spin3_cartan_gens'  # Cartan generators, one per spin(3) subalgebra.
    ])


# A `Spin8Action` tuple consists of an einsum reduction-string,
# typically of the form 'aij,aN->jiN', as well as the 1st tensor-argument
# to the corresponding contraction.
Spin8Action = collections.namedtuple(
    'Spin8Action', ['einsum', 'tensor'])


class BranchingFormatter(object):
  """Base class for branching-formatters."""

  def format(self, num_spin3s, branching):
    return self.sum_join(self.format_irreps(num_spin3s, b) for b in branching)

  def format_branching_tag(self, tag):
    """Formats tag (8, 'v') -> '8v' etc."""
    tag_dim, tag_subscript = tag
    return '%s%s' % (tag_dim, tag_subscript)

  def sum_join(self, formatted):
    return ' + '.join(formatted)

  def format_multiplicity(self, multiplicity, formatted_obj):
    """Adds a multiplicity prefix to a formatted object."""
    if multiplicity == 1:
      return formatted_obj
    return '%dx%s' % (multiplicity, formatted_obj)

  def format_irreps(self, num_spin3s, irreps_part):
    """Formats a group of identical irreducible representations."""
    charges, mult = irreps_part
    return self.format_multiplicity(mult,
                                    self.format_irrep(num_spin3s, charges))

  def format_irrep(self, num_spin3s, charges):
    """Formats a single irreducible representation."""
    if set(charges[:num_spin3s]) == {0}:
      spin3_part = ''
    else:
      spin3_part = 'x'.join('%s' % int(round(2 * c + 1))
                            for c in charges[:num_spin3s])
    assert all(c == int(c) for c in charges[num_spin3s:])
    u1_part = ', '.join(str(int(c)) for c in charges[num_spin3s:])
    if spin3_part:
      return ('[%s]{%s}' % (spin3_part, u1_part) if u1_part
              else '[%s]' % spin3_part)
    else:
      return '{%s}' % u1_part


class LaTeXBranchingFormatter(BranchingFormatter):
  """BranchingFormatter that generates LaTeX code."""

  def format_branching_tag(self, tag):
    """Formats tag (8, 'v') -> '8_{v}' etc."""
    tag_dim, tag_subscript = tag
    return '%s_{%s}' % (tag_dim, tag_subscript)

  def format_multiplicity(self, multiplicity, formatted_obj):
    if multiplicity == 1:
      return formatted_obj
    return r'%d\times%s' % (multiplicity, formatted_obj)

  def _format_charge(self, c, sub_super):
    assert c == int(c)
    if c == 0:
      return ''
    return r'%s{\scriptscriptstyle %s}' % (sub_super, '-+'[c > 0] * abs(int(c)))

  def format_irrep(self, num_spin3s, charges):
    # We use style such as 33^{+++}_{--},
    # i.e. 1st U(1) gets superscript charges,
    # 2nd U(1) gets subscript charges.
    assert all(c == int(c) for c in charges[num_spin3s:])
    if set(charges[:num_spin3s]) <= {0}:
      spin3_part = r'\mathbf{1}'  # No Spin3s, or only singlet.
    elif num_spin3s == 1:
      spin3_part = r'\mathbf{%s}' % int(round(2 * charges[0] + 1))
    else:
      spin3_part = '(%s)' % (
          ','.join(r'\mathbf{%d}' % int(round(2 * c + 1))
                   for c in charges[:num_spin3s]))
    num_u1s = len(charges) - num_spin3s
    u1a_part = u1b_part = ''
    if num_u1s >= 1:
      u1a_part = self._format_charge(charges[num_spin3s], '^')
    if num_u1s == 2:
      u1b_part = self._format_charge(charges[num_spin3s + 1], '_')
    return spin3_part + u1a_part + u1b_part


TEXT_FORMATTER = BranchingFormatter()
LATEX_FORMATTER = LaTeXBranchingFormatter()

# The Spin(8) structure constants.
_spin8_fabc = 2 * numpy.einsum('cik,abik->abc',
                               algebra.su8.m_28_8_8,
                               # We do not need to antisymmetrize [ik] here,
                               # as the above factor already does this.
                               numpy.einsum('aij,bjk->abik',
                                            algebra.su8.m_28_8_8,
                                            algebra.su8.m_28_8_8))

_spin8_action56 = numpy.einsum('aik,ABik->aAB',
                               algebra.su8.m_28_8_8,
                               algebra.su8.m_action_56_56_8_8)

# Branching-rules task specification, as used for the `decomposition_tasks`
# argument to spin3u1_decompose().
# One may generally want to pass an extended arg that adds tasks which also
# decompose e.g. degenerate mass-eigenstates w.r.t. symmetry.
# These are also used to find scaling for u(1) generators that makes all
# 8v, 8s, 8c charges integral.
SPIN8_ACTION_8V = Spin8Action(einsum='aij,aN->jiN',
                              tensor=algebra.su8.m_28_8_8)
SPIN8_ACTION_8V = Spin8Action(einsum='aij,aN->jiN',
                              tensor=algebra.su8.m_28_8_8)
SPIN8_ACTION_8S = Spin8Action(
    einsum='aAB,aN->BAN',
    tensor=numpy.einsum('aij,ijAB->aAB',
                        0.25 * algebra.su8.m_28_8_8,
                        algebra.spin8.gamma_vvss))
SPIN8_ACTION_8C = Spin8Action(
    einsum='aAB,aN->BAN',
    tensor=numpy.einsum('aij,ijAB->aAB',
                        0.25 * algebra.su8.m_28_8_8,
                        algebra.spin8.gamma_vvcc))
SPIN8_ACTION_AD = Spin8Action(einsum='aAB,aN->BAN', tensor=_spin8_fabc * 0.5)

SPIN8_ACTION_FERMIONS = Spin8Action(einsum='aAB,aN->BAN',
                                    tensor=_spin8_action56)

SPIN8_ACTION_SCALARS = Spin8Action(
    einsum='aAB,aN->BAN',
    tensor=0.5 * algebra.e7.spin8_action_on_v70o)

SPIN8_BRANCHINGS_VSC = (
    (SPIN8_ACTION_8V,
     [((8, 'v'), numpy.eye(8))]),
    (SPIN8_ACTION_8S,
     [((8, 's'), numpy.eye(8))]),
    (SPIN8_ACTION_8C,
     [((8, 'c'), numpy.eye(8))]))
# Extended branching-rules task speficication, adds 28->... branching.
SPIN8_BRANCHINGS = (
    SPIN8_BRANCHINGS_VSC +
    ((SPIN8_ACTION_AD, [((28, ''), numpy.eye(28))]),))


def round2(x):
  """Rounds number to 2 digits, canonicalizing -0.0 to 0.0."""
  return numpy.round(x, 2) or 0.0


def allclose2(p, q):
  """Determines if `p` and `q` match to two digits."""
  return numpy.allclose(p, q, rtol=0.01, atol=0.01)


def aggregate_eigenvectors(eigvals, eigvecs, tolerance=1e-6):
  """Collects eigenvectors by eigenvalue into eigenspaces.

  The `eigvals` and `eigvecs` arguments must be as produced by
  scipy.linalg.eigh().

  Args:
    eigvals, array of eigenvalues. Must be approximately-real.
    eigvecs, array of eigenvectors.
    tolerance, float. Tolerance threshold for considering eigenvalues
      as degenerate.

  Returns:
    List of the form [(eigenvalue, eigenspace), ...],
    where each `eigenspace` is a list of eigenvectors for the corresponding
    eigenvalue.

  Raises:
    ValueError, if reality requirements are violated.
  """
  if not numpy.allclose(eigvals, eigvals.real):
    raise ValueError('Non-real eigenvalues.')
  eigvalue_and_aggregated_eigvecs = []
  for eigvalue, eigvec in sorted(zip(eigvals.real,
                                     [tuple(v.astype(numpy.complex128))
                                      for v in eigvecs.T]),
                                 # Do not compare eigenvectors for degenerate
                                 # eigenvalues. Sort by descending order.
                                 key=lambda ev_evec: -ev_evec[0]):
    for eigvalue_known, eigvecs_known in eigvalue_and_aggregated_eigvecs:
      if abs(eigvalue - eigvalue_known) <= tolerance:
        eigvecs_known.append(eigvec)
        break
    else:  # Reached end of loop.
      eigvalue_and_aggregated_eigvecs.append((eigvalue, [eigvec]))
  return eigvalue_and_aggregated_eigvecs


def get_residual_gauge_symmetry(v70, threshold=0.05):
  """Maps scalar 70-vector to [a, n]-tensor of unbroken symmetry generators.

  Index `a` is a Spin(8)-adjoint index, `n` counts (orthonormal) basis vectors.

  Args:
    v70: The e7/su8 70-vector describing a point on the scalar manifold.
    threshold: Threshold on the generalized SVD-eigenvalue for considering
      a direction as belonging to the residual symmetry.
  """
  su, ss, svh = scipy.linalg.svd(
      numpy.einsum('avw,v->aw',
                   algebra.e7.spin8_action_on_v70,
                   v70))
  del svh  # Unused.
  # Select those columns for which the diagonal entry is essentially zero.
  return su.T[ss <= threshold].T


def get_simultaneous_eigenbasis(commuting_gens,
                                gen_action_einsum='abc,aN->cbN',
                                gen_action_tensor=_spin8_fabc,
                                initial_space=None,
                                checks=True,
                                tolerance=1e-6):
  """Finds a simultaneous eigenbasis for a collection of commuting generators.

  Args:
    commuting_gens: [28, N]-array of real and mutually orthogonal generators.
    gen_action_einsum: numpy.einsum() contraction specification that maps
      `gen_action_tensor` and `commuting_gens` to a set of N matrices given as
      [D, D, N]-array that represent the generators on the desired space.
    initial_space: [D, K]-dimensional initial space to decompose into
      eigenspaces, or `None`. If `None`, uses numpy.eye(D).
    checks: If True, perform internal consistency checks.
    tolerance: Tolerance difference-threshold for considering
      two eigenvalues as identical.

  Returns:
    Pair of (simultaneous_eigenbasis, charges), where `simultaneous_eigenbasis`
    is a [28, K]-dimensional array of eigenvectors, and `charges` is a list
    of corresponding charge-tuples.
  """
  # Map generators to endomorphisms. Our conventions are such that
  # the result of contracting with `gen_action_tensor` also gets multiplied
  # with 1j. For spin(8) action on 8v, 8s, 8c, 28, etc., this ensures that
  # with all-real generators and all-real action-tensor, we get hermitean
  # endomorphisms with all-real spectrum.
  gens_action = numpy.einsum(gen_action_einsum,
                             gen_action_tensor,
                             commuting_gens) * 1j
  if initial_space is None:
    initial_space = numpy.eye(gens_action.shape[0])
  #
  def recursively_split_eigenspaces(num_generator, charge_tagged_eigenspaces):
    """Recursively splits an eigenspace.

    Args:
      num_generator: The number of the commuting generator to use for the next
        splitting-step.
      charge_tagged_eigenspaces: List [(partial_charges, subspace), ...]
        where `partial_charges` is a tuple of charges w.r.t. the first
        `num_generator` generators (so, () for num_generator == 0),
        and `subspace` is a [D, K]-array of subspace directions.

    Returns:
      (Ultimately), fully split charge_tagged_eigenspaces, where the
      `partial_charges` tags list as many charges as there are generators.
    """
    if num_generator == gens_action.shape[-1]:
      return charge_tagged_eigenspaces
    gen_action = gens_action[:, :, num_generator]
    split_eigenspaces = []
    for charges, espace in charge_tagged_eigenspaces:
      if checks:
        eigenspace_sprod = numpy.einsum('aj,ak->jk', espace.conj(), espace)
        assert allclose2(
            eigenspace_sprod,
            numpy.eye(espace.shape[1])), (
                'Weird Eigenspace normalization: ' + repr(
                    numpy.round(eigenspace_sprod, 3)))
      gen_on_eigenspace = numpy.einsum(
          'aj,ak->jk',
          espace.conj(),
          numpy.einsum('ab,bj->aj', gen_action, espace))
      sub_eigvals, sub_eigvecs_T = scipy.linalg.eigh(gen_on_eigenspace)
      list_approx_eigval_and_eigvecs = []
      for sub_eigval, sub_eigvec in zip(sub_eigvals, sub_eigvecs_T.T):
        # Lift back to original space.
        eigvec = numpy.einsum('gs,s->g', espace, sub_eigvec)  # |v> <v| G |v>
        if checks:
          gv = numpy.dot(gen_action, eigvec)
          ev = sub_eigval * eigvec
          assert allclose2(gv, ev), (
              'Sub-Eigval is bad: g*v=%r, e*v=%r' % (
                  numpy.round(gv, 3), numpy.round(ev, 3)))
          assert allclose2(
              numpy.dot(eigvec.conj(), eigvec), 1.0), (
                  'Eigenvector is not normalized.')
        for seen_eigval, seen_eigvecs in list_approx_eigval_and_eigvecs:
          if abs(sub_eigval - seen_eigval) <= tolerance:
            assert all(allclose2(0, numpy.dot(s.conj(), eigvec))
                       for s in seen_eigvecs), 'Non-Orthogonality'
            seen_eigvecs.append(eigvec)
            break
        else:  # Reached end of list.
          list_approx_eigval_and_eigvecs.append(
            (sub_eigval,  # This is also the actual eigenvalue.
             [eigvec]))
      for eigval, eigvecs in list_approx_eigval_and_eigvecs:
        eigenspace = numpy.stack(eigvecs, axis=-1)
        assert allclose2(
            numpy.einsum('aj,ak->jk', eigenspace.conj(), eigenspace),
            numpy.eye(eigenspace.shape[-1])), 'Bad Eigenspace'
        split_eigenspaces.append((charges + (eigval,), eigenspace))
    return recursively_split_eigenspaces(num_generator + 1, split_eigenspaces)
  #
  charge_tagged_eigenspaces = recursively_split_eigenspaces(
      0, [((), initial_space)])
  simultaneous_eigenbasis = numpy.stack(
      [evec for _, espace in charge_tagged_eigenspaces for evec in espace.T],
      axis=-1)
  charges = [evec_charges
             for evec_charges, espace in charge_tagged_eigenspaces
             for evec in espace.T]
  return simultaneous_eigenbasis, charges


def scale_u1_generator_to_8vsc_integral_charges(u1_gen, round_to_digits=3):
  """Scales a generator such that all 8v, 8s, 8c charges are integers."""
  charges = []
  for spin8action, _ in SPIN8_BRANCHINGS_VSC:
    eigvals, _ = scipy.linalg.eigh(
        numpy.einsum(spin8action.einsum,
                     spin8action.tensor,
                     1j * u1_gen.reshape((28, 1)))[:, :, 0])
    assert numpy.allclose(eigvals, eigvals.real)
    for eigval in eigvals:
      charges.append(eigval)
  approx_charges = sorted(set(abs(numpy.round(c, 6)) for c in charges) - {0.0})
  factor = 1.0 / approx_charges[0]
  for n in range(1, 25):
    scaled_charges = [numpy.round(factor * n * c, round_to_digits)
                      for c in approx_charges]
    if all(x == int(x) for x in scaled_charges):
      return factor * n * u1_gen
  raise ValueError('Could not re-scale U(1)-generator.')


def canonicalize_u1s(u1s, tolerance=1e-3):
  """Canonicalizes a collection of up to two u(1) generators."""
  if u1s.shape[1] == 0:
    return numpy.zeros([28, 0])
  if u1s.shape[0] != 28:
    raise ValueError(
        'Each U(1) generator should be given as a 28-vector.')
  num_u1s = u1s.shape[1]
  if num_u1s > 2:
    raise ValueError('Cannot handle more than two U(1)s')
  if num_u1s == 1:
    return scale_u1_generator_to_8vsc_integral_charges(u1s[:, 0]).reshape(28, 1)
  eigvecs_T, evec_charges = get_simultaneous_eigenbasis(u1s)
  a_vecs_eigvals = numpy.array(evec_charges).T
  # Otherwise, we have exactly two U(1)s.
  # How to reduce the charge-lattice?
  zs = numpy.array([x + 1j * y for x, y in a_vecs_eigvals.T])
  zs_by_origin_distance = sorted([z for z in zs if abs(z) >= tolerance],
                                 key=abs)
  z1 = zs_by_origin_distance[0]
  angle = math.atan2(z1.imag, z1.real)
  cos_angle = math.cos(angle)
  sin_angle = math.sin(angle)
  u1a =  u1s[:, 0] * cos_angle + u1s[:, 1] * sin_angle
  u1b =  u1s[:, 0] * sin_angle - u1s[:, 1] * cos_angle
  canon_u1s = numpy.stack([
      scale_u1_generator_to_8vsc_integral_charges(u1a),
      scale_u1_generator_to_8vsc_integral_charges(u1b)], axis=1)
  return canon_u1s


def decompose_reductive_lie_algebra(residual_symmetry,
                                    threshold=0.05):
  """Decomposes a residual symmetry into semisimple and u(1) parts.

  Args:
    residual_symmetry: Residual symmetry as produced by
      `get_residual_gauge_symmetry()`.
    threshold: Threshold for SVD generalized commutator-eigenvalue to consider
      a generator as being part of the non-semisimple subalgebra.
  """
  no_symmetry = numpy.zeros([28, 0])
  if residual_symmetry.shape[1] == 0:
    return no_symmetry, no_symmetry
  commutators = numpy.einsum(
      'avc,cw->avw',
      numpy.einsum('abc,bv->avc', _spin8_fabc, residual_symmetry),
      residual_symmetry)
  su, ss, svh = scipy.linalg.svd(commutators.reshape(commutators.shape[0], -1))
  del svh  # Unused.
  # We want those commutators that do not go to zero.
  derivative_symmetry =  su.T[:len(ss)][ss >= threshold].T
  # By construction (via SVD), and using orthogonality of our spin(8) basis,
  # `derivative_symmetry` already consists of orthogonal spin(8) generators, i.e.
  # tr(AB) = 0 for basis vectors A != B.
  # The 'complement' consists of u(1) factors that have zero inner product with
  # `derivative_symmetry`.
  if derivative_symmetry.size:
    inner_products_with_input = numpy.einsum('av,aw->vw',
                                             residual_symmetry,
                                             derivative_symmetry)
    su, ss, svh = scipy.linalg.svd(inner_products_with_input)
    # Zero-pad the vector of 'generalized eigenvalues' to su's size.
    ss_ext = numpy.concatenate(
        [ss, numpy.zeros([max(0, su.shape[0] - len(ss))])])
    u1s = numpy.einsum('av,vn->an',
                       residual_symmetry,
                       su.T[ss_ext <= threshold].T)
  else:  # All residual symmetry is in u(1)-factors.
    return no_symmetry, residual_symmetry
  # Assert that our U1s are orthogonal.
  if u1s.size:
    # Check generator orthonormality.
    assert numpy.allclose(numpy.einsum('av,aw->vw', u1s, u1s),
                          numpy.eye(u1s.shape[1]), atol=1e-6)
  else:
    u1s = no_symmetry
  return derivative_symmetry, u1s


def find_raw_cartan_subalgebra(spin8_subalgebra_generators, threshold=1e-3):
  """Finds a Cartan subalgebra for an algebra if the form A*so(3) + B*u(1)."""
  if spin8_subalgebra_generators.shape[1] == 0:
    return numpy.zeros([28, 0])
  subalgebra_sprods = numpy.einsum(
      'aj,ak->jk', spin8_subalgebra_generators, spin8_subalgebra_generators)
  # Check that incoming subalgebra-generators really are reasonably orthonormal
  # (up to overall scaling) w.r.t. Cartan-Killing metric.
  assert numpy.allclose(subalgebra_sprods,
                        numpy.eye(spin8_subalgebra_generators.shape[1]))
  cartan_generators_found = []
  residual_charge_zero_subspace = spin8_subalgebra_generators
  while True:
    gen = residual_charge_zero_subspace[:, 0]
    cartan_generators_found.append(gen)
    assert numpy.allclose(gen, gen.real), 'Generator is not real!'
    orthogonal_subalgebra = residual_charge_zero_subspace[:, 1:]
    if not orthogonal_subalgebra.shape[1]:
      return numpy.stack(cartan_generators_found, axis=-1)
    gen_ad_action_on_spin8 = numpy.einsum('abc,a->cb', _spin8_fabc, gen)
    gen_action_on_orthogonal_subalgebra = numpy.einsum(
        'ai,aj->ij',
        orthogonal_subalgebra,
        numpy.einsum('bc,cj->bj',
                     gen_ad_action_on_spin8 * 1j,
                     orthogonal_subalgebra))
    assert numpy.allclose(gen_action_on_orthogonal_subalgebra +
                          gen_action_on_orthogonal_subalgebra.T,
                          numpy.zeros_like(gen_action_on_orthogonal_subalgebra))
    eigvals, eigvecs_T = scipy.linalg.eigh(gen_action_on_orthogonal_subalgebra)
    nullspace_gens = []
    for eigval, eigvec in zip(eigvals, eigvecs_T.T):
      if abs(eigval) <= threshold:
        assert numpy.allclose(eigvec, eigvec.real)
        nullspace_gens.append(
            numpy.einsum('ai,i->a', orthogonal_subalgebra, eigvec.real))
    if not len(nullspace_gens):
      return numpy.stack(cartan_generators_found, axis=-1)
    nullspace = numpy.stack(nullspace_gens, axis=1)
    assert numpy.allclose(nullspace, nullspace.real), 'Non-real nullspace'
    assert numpy.allclose(numpy.einsum('ai,aj->ij', nullspace, nullspace),
                          numpy.eye(nullspace.shape[1])), 'Non-Ortho Nullspace'
    residual_charge_zero_subspace = nullspace


def weightspace_decompose(generator_action,
                          cartan_subalgebra_generators,
                          space,
                          tolerance=1e-6):
  """Decomposes `space` into subspaces tagged by weight-vectors."""
  seq_cartan_generators = list(cartan_subalgebra_generators.T)
  def cartan_split(subspace_tagged_by_weight_prefix, num_cartan_generator):
    cartan_action = numpy.einsum(
        'aIJ,a->IJ',
        generator_action,
        seq_cartan_generators[num_cartan_generator] * 1j)
    result = []
    for weight_prefix, subspace in subspace_tagged_by_weight_prefix:
      assert numpy.allclose(
          numpy.einsum('aJ,aK->JK', subspace.conj(), subspace),
          numpy.eye(subspace.shape[1])), (
              'Non-orthonormalized subspace:\n' +
              repr(numpy.round(numpy.einsum('aJ,aK->JK',
                                            subspace.conj(),
                                            subspace), 3)))
      cartan_action_on_subspace = numpy.einsum(
          'Jm,Jn->mn', subspace.conj(),
          numpy.einsum('JK,Kn->Jn', cartan_action, subspace))
      eigvals, eigvecs_T = scipy.linalg.eigh(cartan_action_on_subspace)
      eigval_and_rel_eigenspace = aggregate_eigenvectors(eigvals, eigvecs_T)
      for eigval, rel_eigenspace in eigval_and_rel_eigenspace:
        ext_weight_prefix = (weight_prefix + (eigval,))
        result.append((ext_weight_prefix,
                       numpy.einsum('In,nj->Ij',
                                    subspace,
                                    numpy.stack(rel_eigenspace, axis=-1))))
    if num_cartan_generator == len(seq_cartan_generators) - 1:
      return result
    return cartan_split(result, num_cartan_generator + 1)
  return cartan_split([((), space)], 0)


def get_simple_roots_info(rootspaces, threshold=0.01):
  """Extracts simple roots from weightspace-decomposition of a Lie algebra."""
  # Finite-dimensional simple Lie algebras have one-dimensional root spaces.
  # We use this to eliminate the Cartan subalgebra at the zero-root.
  rank = len(rootspaces[0][0])
  null_root = (0.0,) * rank
  positive_roots = [root for root, subspace in rootspaces
                    if subspace.shape[1] == 1 and root > null_root]
  def root_length_squared(root):
    return sum(x * x for x in root)
  def root_distance(root1, root2):
    return max(abs(r1 - r2) for r1, r2 in zip(root1, root2))
  # If the root is 'clearly too long', drop it rightaway.
  # It does not hurt if we allow a large amount of slack,
  # as this is just for increased performance.
  threshold_root_length_squared = max(
      map(root_length_squared, positive_roots)) * (1 + threshold)
  sum_roots = []
  for root1 in positive_roots:
    for root2 in positive_roots:
      root12 = tuple(r1 + r2 for r1, r2 in zip(root1, root2))
      if root_length_squared(root12) > threshold_root_length_squared:
        continue
      for sum_root in sum_roots:
        if root_distance(sum_root, root12) <= threshold:
          break  # We already know this sum-root.
      else:  # Reached end of loop.
        sum_roots.append(root12)
  simple_roots = [root for root in positive_roots
                  if not any(root_distance(sum_root, root) < threshold
                             for sum_root in sum_roots)]
  a_simple_roots = numpy.array(simple_roots)
  simple_root_sprods = numpy.einsum('rj,rk->jk', a_simple_roots, a_simple_roots)
  # We always normalize the length-squared of the longest root to 2.
  scaling_factor_squared = 2.0 / max(
      simple_root_sprods[n, n] for n in range(simple_root_sprods.shape[0]))
  scaling_factor = math.sqrt(scaling_factor_squared)
  scaled_root_sprods = simple_root_sprods * scaling_factor_squared
  # For spin(3)^N, the roots have to be mutually orthogonal
  # with length-squared 2.
  assert numpy.allclose(scaled_root_sprods,
                        2 * numpy.eye(simple_root_sprods.shape[0]) )
  pos_simple_rootspaces = [(pos_root, scaling_factor * pos_rootspace)
                           for (pos_root, pos_rootspace) in rootspaces
                           for simple_root in simple_roots
                           if tuple(simple_root) == tuple(pos_root)]
  canonicalized_cartan_subalgebra_generators = []
  for pos_root, pos_rootspace in pos_simple_rootspaces:
    # For finite-dimensional Lie algebras, root spaces are one-dimensional.
    assert pos_rootspace.shape[1] == 1
    l_plus = pos_rootspace[:, 0]
    l_minus = l_plus.conj()
    cartan_h = -1j * numpy.einsum('abc,a,b->c', _spin8_fabc, l_plus, l_minus)
    canonicalized_cartan_subalgebra_generators.append(cartan_h)
  # TODO(tfish): Only return what we need, and *not* in a dict.
  return dict(simple_root_sprods=simple_root_sprods,
              canonicalized_cartan_subalgebra=numpy.stack(
                  canonicalized_cartan_subalgebra_generators, axis=-1),
              scaling_factor_squared=scaling_factor_squared,
              pos_simple_rootspaces=pos_simple_rootspaces,
              scaled_root_sprods=scaled_root_sprods,
              scaled_roots=a_simple_roots * math.sqrt(scaling_factor_squared))



def canonicalize_residual_spin3u1_symmetry(residual_symmetry):
  """Canonicalizes a residual so(3)^M u(1)^N symmetry."""
  semisimple_part, raw_u1s = decompose_reductive_lie_algebra(residual_symmetry)
  u1s = canonicalize_u1s(raw_u1s)
  spin3_cartan_gens_raw = find_raw_cartan_subalgebra(semisimple_part)
  return CanonicalizedSymmetry(u1s=u1s,
                               semisimple_part=semisimple_part,
                               spin3_cartan_gens=spin3_cartan_gens_raw)


def group_charges_into_spin3u1_irreps(num_spin3s, charge_vecs):
  """Groups observed charges into irreducible representations.

  Args:
    num_spin3s: Length of the prefix of the charge-vector that belongs to
      spin(3) angular momentum operators.
    charge_vecs: List of charge-tuple vectors.

  Returns:
    List [((tuple(highest_spin3_weights) + tuple(u1_charges)), multiplicity),
          ...] of irreducible-representation descriptions, sorted by descending
      combined-charge-vector.
  """
  def spin3_weights(highest_weight):
    """Computes a list of spin3 weights for a given irrep highest weight.

    E.g.: highest_weight = 1.5 -> [1.5, 0.5, -0.5, -1.5].

    Args:
      highest_weight: The highest weight (Element of [0, 0.5, 1.0, 1.5, ...]).

    Returns: List of weights, in descending order.
    """
    w2 = int(round(2 * highest_weight))
    return [highest_weight - n for n in range(1 + w2)]
  def descendants(cvec):
    for spin3_part in itertools.product(
        *[spin3_weights(w) for w in cvec[:num_spin3s]]):
      yield spin3_part + cvec[num_spin3s:]
  charges_todo = collections.Counter(charge_vecs)
  irreps = collections.defaultdict(int)
  while charges_todo:
    cvec, cvec_mult = sorted(charges_todo.items(), reverse=True)[0]
    for cvec_desc in descendants(cvec):
      charges_todo[cvec_desc] -= cvec_mult
      if charges_todo[cvec_desc] == 0:
        del charges_todo[cvec_desc]
    irreps[cvec] += cvec_mult
  return sorted(irreps.items(), reverse=True)  # Highest charges first.


def spin3u1_decompose(canonicalized_symmetry,
                      decomposition_tasks=SPIN8_BRANCHINGS,
                      simplify=round2):
  """Computes decompositions into so(3)^M x u(1)^N irreducible representations.

  Args:
    canonicalized_symmetry: A `CanonicalizedSymmetry` object.
    decomposition_tasks: Sequence of pairs (spin8action, tasks),
      where `tasks` is a sequence of pairs (tag, orthogonalized_subspace).
    simplify: The rounding function used to map approximately-integer charges
      to integers.
  """
  spin3_gens = (canonicalized_symmetry.spin3_cartan_gens.T
                if (canonicalized_symmetry.spin3_cartan_gens is not None
                    and len(canonicalized_symmetry.spin3_cartan_gens)) else [])
  u1_gens = (canonicalized_symmetry.u1s.T
             if (canonicalized_symmetry.u1s is not None
                 and len(canonicalized_symmetry.u1s)) else [])
  num_spin3s = len(spin3_gens)
  num_u1s = len(u1_gens)
  def grouped(charges):
    # Spin(3) angular momentum charges need to be half-integral.
    # For U(1) generators, we are not requiring this.
    assert all(round2(2 * c) == int(round2(2 * c))
               for charge_vec in charges
               for c in charge_vec[:num_spin3s])
    return group_charges_into_spin3u1_irreps(
        num_spin3s,
        [tuple(map(simplify, charge_vec)) for charge_vec in charges])
  if num_spin3s:
    rootspaces = weightspace_decompose(
        _spin8_fabc,
        spin3_gens.T,
        canonicalized_symmetry.semisimple_part)
    sroot_info = get_simple_roots_info(rootspaces)
    angular_momentum_u1s = list(sroot_info['canonicalized_cartan_subalgebra'].T)
  else:
    angular_momentum_u1s = []
  list_commuting_gens =  (
      [g for g in [angular_momentum_u1s, u1_gens] if len(g)])
  commuting_gens = (numpy.concatenate(list_commuting_gens).T
                    if list_commuting_gens else numpy.zeros([28, 0]))
  ret = []
  for spin8action, tasks in decomposition_tasks:
    ret.append([])
    for task_tag, space_to_decompose in tasks:
      _, charges = get_simultaneous_eigenbasis(
          commuting_gens,
          gen_action_einsum=spin8action.einsum,
          gen_action_tensor=spin8action.tensor,
          initial_space=space_to_decompose)
      ret[-1].append((task_tag, grouped(charges)))
  return ret


def spin3u1_branching_and_spectra(canonicalized_symmetry,
                                  decomposition_tasks=()):
  """Computes so(3)^M x u(1)^N spectra."""
  vsc_ad_branching = spin3u1_decompose(canonicalized_symmetry)
  spectra = spin3u1_decompose(canonicalized_symmetry,
                              decomposition_tasks)
  return vsc_ad_branching, spectra


def spin3u1_physics(
    canonicalized_symmetry,
    mass_tagged_eigenspaces_gravitinos=(),
    mass_tagged_eigenspaces_fermions=(),
    mass_tagged_eigenspaces_scalars=(),
    # Note that we see cases where we have very uneven parity-mixtures.
    parity_tolerance=1e-7):
  """Computes so(3)^M x u(1)^N spectra."""
  vsc_ad_branching = spin3u1_decompose(canonicalized_symmetry)
  decomposition_tasks = []
  # Gravitino tasks.
  gravitino_tasks = []
  for gravitino_mass, basis in mass_tagged_eigenspaces_gravitinos:
    subspace = numpy.array(basis).T
    task_tag = ('gravitinos', subspace.shape, gravitino_mass)
    gravitino_tasks.append((task_tag, subspace))
  decomposition_tasks.append(
      (SPIN8_ACTION_8V, gravitino_tasks))
  # Fermion tasks.
  fermion_tasks = []
  for fermion_mass, basis in mass_tagged_eigenspaces_fermions:
    subspace = numpy.array(basis).T
    task_tag = ('fermions', subspace.shape, fermion_mass)
    fermion_tasks.append((task_tag, subspace))
  decomposition_tasks.append(
      (SPIN8_ACTION_FERMIONS, fermion_tasks))
  # Scalar tasks.
  scalar_tasks = []
  # For scalars, we try to split off mass-eigenstates that are
  # 35s-only or 35c-only.
  p_op = numpy.eye(70)
  p_op[35:, 35:] *= -1
  for scalar_mass, basis in mass_tagged_eigenspaces_scalars:
    a_basis = numpy.array(basis)
    p_op_on_basis = numpy.einsum('jn,nm,km->jk', a_basis.conj(), p_op, a_basis)
    assert numpy.allclose(p_op_on_basis, p_op_on_basis.real)
    assert numpy.allclose(p_op_on_basis, p_op_on_basis.T)
    p_op_eigvals, p_op_eigvecs_T = numpy.linalg.eigh(p_op_on_basis)
    p_op_eigvals_re = p_op_eigvals.real
    assert numpy.allclose(p_op_eigvals, p_op_eigvals_re)
    # We have to lift the p_op_eigvecs_T to a_basis.
    subspace_eigvecs = numpy.einsum('vn,vV->Vn', p_op_eigvecs_T, a_basis)
    eigval_eigvecs = aggregate_eigenvectors(p_op_eigvals_re, subspace_eigvecs,
                                            tolerance=1e-4)
    # subspaces_35s and subspaces_35c each have <=1 entries.
    subspaces_35s = [eigvecs for eigval, eigvecs in eigval_eigvecs
                     if eigval > 1 - parity_tolerance]
    subspaces_35c = [eigvecs for eigval, eigvecs in eigval_eigvecs
                     if eigval < -1 + parity_tolerance]
    merged_subspaces_other = [
        eigvec for eigval, eigvecs in eigval_eigvecs
        for eigvec in eigvecs
        if -1 + parity_tolerance <= eigval <= 1 - parity_tolerance]
    for subspace in subspaces_35s:
      a_subspace = numpy.array(subspace).T
      task_tag = ('scalars', a_subspace.shape, scalar_mass, 's')
      scalar_tasks.append((task_tag, a_subspace))
    for subspace in subspaces_35c:
      a_subspace = numpy.array(subspace).T
      task_tag = ('scalars', a_subspace.shape, scalar_mass, 'c')
      scalar_tasks.append((task_tag, a_subspace))
    # "Mixture" states. While we do get them in terms of parity-eigenstates,
    # for 'weird' eigenvalues such as -1/3. Here, we just merge them all back
    # together into one space, i.e. forget about resolving the spectrum.
    # Why? Otherwise, we may see in the report
    # "0.000m{1}, 0.000m{1}, 0.000m{1}, ...", which is not overly informative.
    a_subspace = numpy.array(merged_subspaces_other).T
    if len(merged_subspaces_other):
      task_tag = ('scalars', a_subspace.shape, scalar_mass, 'm')
      scalar_tasks.append((task_tag, a_subspace))
  decomposition_tasks.append(
      (SPIN8_ACTION_SCALARS, scalar_tasks))
  spectra = spin3u1_decompose(canonicalized_symmetry,
                              decomposition_tasks)
  return vsc_ad_branching, spectra
