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

"""Lie algebra definitions relevant for supergravity."""


import itertools
import cached_property

from m_theory_lib import m_util as mu

import numpy
import scipy.linalg


# Naming deviates from PEP-8 conventions where this makes mathematics easier
# to read. Also, local variables may name-match module-global definitions,
# and we permit complex list comprehensions.
# pylint:disable=invalid-name
# pylint:disable=redefined-outer-name
# pylint: disable=g-complex-comprehension


nsum = mu.nsum  # Shorthand.


def get_normalizing_subalgebra(
    lie_gens_apq,
    subspace_ps,
    eigenvalue_threshold=1e-7):
  """Returns a [a, n]-basis for the subalgebra normalizing subspace_ps."""
  dim_large = lie_gens_apq.shape[1]
  dim_small = subspace_ps.shape[1]
  if ((not lie_gens_apq.shape[2] == dim_large == subspace_ps.shape[0]) or
      dim_small > dim_large):
    raise ValueError(
        'Bad shape of algebra generators / embedding: '
        f'lie_gens_apq.shape={lie_gens_apq.shape}, '
        f'subspace_ps.shape={subspace_ps.shape}')
  # The idea is very simple: If we start from vectors in subspace_ps.dot(ss),
  # and then project to some complement of the space spanned by the s vectors
  # of subspace_ps, this must be zero. Different complements differ by
  # adding to their basis vectors elements of subspace_ps,
  # but "as long as our bases are not too weird", such as when lengths of
  # and angles between basis vectors "are all reasonable", this is fine.
  svd_ps_u, svd_ps_s, svd_ps_vh = numpy.linalg.svd(subspace_ps,
                                                   full_matrices=True)
  del svd_ps_vh  # Unused, was only introduced for documentation completeness.
  onb_complement_s = (
      svd_ps_u[:, numpy.pad(svd_ps_s, [(0, dim_large - len(svd_ps_s))]) <=
               eigenvalue_threshold])
  act_project_rotated = nsum('apq,ps,qr->asr', lie_gens_apq, subspace_ps,
                             onb_complement_s)
  # We want to find those linear combinations of generators which,
  # when acting on vectors from the subspace, generate output
  # that, when projected to some complement of the subspace,
  # yields null. These are then generators which, when applied
  # to arbitrary subspace vectors, return other vectors whose
  # decomposition does not involve contributions going beyond that subspace.
  # This is the normalizer of the subspace.
  xu, xs, xvh = numpy.linalg.svd(
      act_project_rotated.reshape(act_project_rotated.shape[0], -1),
      full_matrices=True)
  del xvh  # Unused, named for documentation only.
  return xu[:, numpy.pad(xs, [(0, lie_gens_apq.shape[0] - len(xs))]) <=
            eigenvalue_threshold]


class Spin8:
  r"""Container class for Spin(8) tensor invariants.

  An instance essentially is just a namespace for constants.
  All attributes are to be considered as read-only by the user.

  Attributes:
    gamma_vsc:  The spin(8) gamma^i_{alpha,\dot\beta} gamma matrices, indexed by
      vector, spinor, co-spinor index.
    gamma_vvss: The spin(8) gamma^{ij}_{\alpha\beta}, indexed
      [i, j, \alpha, \beta].
    gamma_vvcc: The spin(8) gamma^{ij}_{\dot\alpha\dot\beta}, indexed
      [i, j, \alpha, \beta].
    gamma_sscc: The spin(8) gamma_{\alpha\beta\dot\delta\dot\epsilon}, indexed
      [\alpha, \beta, \dot\delta, \dot\epsilon].
    gamma_vvvvss: The spin(8) gamma^{ijkl}_{\alpha\beta}, indexed
      [i, j, k, l, \alpha, \beta].
    gamma_vvvvcc: The spin(8) gamma^{ijkl}_{\dot\alpha\dot\beta}, indexed
      [i, j, k, l, \dot\alpha, \dot\beta].
    gamma_sssscc: The spin(8)
      gamma^{\alpha\beta\gamma\delta}_{\dot\alpha\dot\beta}, indexed
      [\alpha, \beta, \gamma, \delta, \dot\alpha, \dot\beta].
    gamma_ccccvv: The spin(8)
      gamma^{\dot\alpha\dot\beta\dot\gamma\dot\delta}_{ij}, indexed
      [\dot\alpha, \dot\beta, \dot\gamma, \dot\delta, i, j].
    gamma_ssssvv: The spin(8)
      gamma^{\alpha\beta\gamma\delta}_{ij}, indexed
      [\alpha, \beta, \gamma, \delta, i, j].
    gamma_ccccss: The spin(8)
      gamma^{\dot\alpha\dot\beta\dot\gamma\dot\delta}_{\alpha\beta}, indexed
      [\dot\alpha, \dot\beta, \dot\gamma, \dot\delta, \alpha, \beta].
  """

  # Default conventions for lines-on-the-fano-plane.
  FANO_LINES = '124 156 137 235 267 346 457'
  # Their complements. The ordering of these is such that using them
  # to define SL(2)^7 inside E7 makes the holomorphic superpotential
  # of SO(8) supergravity match the 'Hamming code superpotential'
  # (7.5) in arXiv:1909.10969, with all coefficients +1, but with
  # the z-s relabeled such that we instead get the cubic terms
  # z_1 z_2 z_4 + z_1 z_5 z_6 + ..., in alignment with the above
  # FANO_LINES triplets that match the common 'octonion multiplication table'
  # definitions.
  FANO_QUARTETS = '1236 1257 1467 3567 2456 1345 2347'

  def __init__(self,
               conventions='gsw'
               ):
    """Initializes the instance.

    Args:
      conventions: str, the conventions to use for Gamma matrices. Options are:
        'gsw': Green, Schwarz, Witten, "Superstring Theory", Volume 1, (5.B.3).
        'octonionic': https://arxiv.org/abs/math/0105155, Table 1.
    """
    self._conventions = conventions
    self.gamma_vsc = gamma_vsc = self._get_gamma_vsc(conventions=conventions)
    #
    # The gamma^{ab}_{alpha beta} tensor that translates between antisymmetric
    # matrices over vectors [ij] and antisymmetric matrices over spinors [sS].
    # TODO(tfish): Change to mu.asymm2()
    self.gamma_vvss = 0.5 * (
        nsum('isc,jSc->ijsS', gamma_vsc, gamma_vsc) -
        nsum('jsc,iSc->ijsS', gamma_vsc, gamma_vsc))
    # The gamma^{ab}_{alpha* beta*} tensor that translates between antisymmetric
    # matrices over vectors [ij] and antisymmetric matrices over cospinors [cC].
    self.gamma_vvcc = 0.5 * (
        nsum('isc,jsC->ijcC', gamma_vsc, gamma_vsc) -
        nsum('jsc,isC->ijcC', gamma_vsc, gamma_vsc))
    # The gamma^{alpha beta}_{alpha* beta*} are not needed for the supergravity
    # computation per se, but we use these objects to determine the Spin(8)
    # rotation that brings E7 / SU(8)
    # scalar manifold coordinates into uniquely defined normal form.
    self.gamma_sscc = 0.5 * (
        nsum('vsc,vSC->sScC', gamma_vsc, gamma_vsc) -
        nsum('vsc,vSC->SscC', gamma_vsc, gamma_vsc))
    #
    # The gamma^{ijkl}_{alpha beta} tensor that translates between antisymmetric
    # 4-forms [ijkl] and symmetric traceless matrices over the spinors (sS),
    # as well as its co-spinor (cC) cousin.
    g_ijsS = nsum('isc,jSc->ijsS', self.gamma_vsc, self.gamma_vsc)
    g_ijcC = nsum('isc,jsC->ijcC', self.gamma_vsc, self.gamma_vsc)
    g_ijklsS = nsum('ijst,kltS->ijklsS', g_ijsS, g_ijsS)
    g_ijklcC = nsum('ijcd,kldC->ijklcC', g_ijcC, g_ijcC)
    gamma_vvvvss = numpy.zeros([8] * 6)
    gamma_vvvvcc = numpy.zeros([8] * 6)
    for perm in itertools.permutations(range(4)):
      perm_ijkl = ''.join('ijkl'[p] for p in perm)
      sign = mu.permutation_sign(perm)
      gamma_vvvvss += sign * nsum(perm_ijkl + 'sS->ijklsS', g_ijklsS)
      gamma_vvvvcc += sign * nsum(perm_ijkl + 'cC->ijklcC', g_ijklcC)
    self.gamma_vvvvss = gamma_vvvvss / 24.0
    self.gamma_vvvvcc = gamma_vvvvcc / 24.0
    # The other gamma_aaaabb are sometimes also useful, especially when
    # considering e7 subalgebras of sl(8) ~ so(8) + 35s.
    g_stuvcC = nsum('stcC,uvCd->stuvcd', self.gamma_sscc, self.gamma_sscc)
    gamma_sssscc = numpy.zeros([8] * 6)
    for perm in itertools.permutations(range(4)):
      perm_stuv = ''.join('stuv'[p] for p in perm)
      sign = mu.permutation_sign(perm)
      gamma_sssscc += sign * nsum(perm_stuv + 'cC->stuvcC', g_stuvcC)
    self.gamma_sssscc = gamma_sssscc / 24.0
    gamma_ccccvv = numpy.zeros([8] * 6)
    g_cdefvV = nsum('ijcd,jkef->cdefik', self.gamma_vvcc, self.gamma_vvcc)
    for perm in itertools.permutations(range(4)):
      perm_cdef = ''.join('cdef'[p] for p in perm)
      sign = mu.permutation_sign(perm)
      gamma_ccccvv += sign * nsum(perm_cdef + 'vV->cdefvV', g_cdefvV)
    self.gamma_ccccvv = gamma_ccccvv / 24.0
    #
    gamma_ssssvv = numpy.zeros([8] * 6)
    g_stuvij = nsum('ijst,jkuv->stuvik', self.gamma_vvss, self.gamma_vvss)
    for perm in itertools.permutations(range(4)):
      perm_stuv = ''.join('stuv' [p] for p in perm)
      sign = mu.permutation_sign(perm)
      gamma_ssssvv += sign * nsum(perm_stuv + 'ij->stuvij', g_stuvij)
    self.gamma_ssssvv = gamma_ssssvv / 24.0
    #
    gamma_ccccss = numpy.zeros([8] * 6)
    g_cdefsS = nsum('stcd,tuef->cdefsu', self.gamma_sscc, self.gamma_sscc)
    for perm in itertools.permutations(range(4)):
      perm_cdef = ''.join('cdef' [p] for p in perm)
      sign = mu.permutation_sign(perm)
      gamma_ccccss += sign * nsum(perm_cdef + 'sS->cdefsS', g_cdefsS)
    self.gamma_ccccss = gamma_ccccss / 24.0

  def _get_gamma_vsc(self, conventions='gsw'):
    r"""Computes Spin(8) gamma-matrices.

    Args:
      conventions: str, the conventions to use for Gamma matrices. Options are:
        'gsw': Green, Schwarz, Witten, "Superstring Theory", Volume 1, (5.B.3).
        'octonionic': https://arxiv.org/abs/math/0105155, Table 1.
        Adding a suffix '/svc', '/csv', '/vcs', etc. will apply a
        triality-rotation that re-labels the 8-dimensional representations, e.g.
        'gsw/vcs' will build Green-Schwarz-Witten Gamma matrices, but
        use 'vsc->vcs' re-labeling to swap the "spinor" and "co-spinor"
        representations (so, swapping the spinor and co-spinor gamma-eigenvalues
        (+1, -1) -> (-1, +1)).

    Returns:
      [8, 8, 8]-tensor gamma^i_\alpha\dot\beta.
    """
    ret = numpy.zeros([8, 8, 8])
    if conventions.startswith('gsw'):
      # Conventions for Spin(8) gamma matrices match Green, Schwarz, Witten,
      # but with indices shifted down by 1 to the range [0 .. 7].
      entries = (
          '007+ 016- 025- 034+ 043- 052+ 061+ 070- '
          '101+ 110- 123- 132+ 145+ 154- 167- 176+ '
          '204+ 215- 226+ 237- 240- 251+ 262- 273+ '
          '302+ 313+ 320- 331- 346- 357- 364+ 375+ '
          '403+ 412- 421+ 430- 447+ 456- 465+ 474- '
          '505+ 514+ 527+ 536+ 541- 550- 563- 572- '
          '606+ 617+ 624- 635- 642+ 653+ 660- 671- '
          '700+ 711+ 722+ 733+ 744+ 755+ 766+ 777+')
      for ijkc in entries.split():
        indices = tuple([int(m) for m in ijkc[:-1]])
        sign = 1 if ijkc[-1] == '+' else -1
        ret[indices] = sign
    elif conventions.startswith('octonionic'):
      for n in range(1, 8):
        ret[0, n, n] = -1
        ret[n, n, 0] = ret[n, 0, n] = 1
      ret[0, 0, 0] = 1
      for cijk in self.FANO_LINES.split():
        ijk = tuple(int(idx) for idx in cijk)
        for p, q, r in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
          # Note that we have to `go against the direction of the arrows'
          # to make the correspondence work.
          ret[ijk[r], ijk[p], ijk[q]] = -1
          ret[ijk[r], ijk[q], ijk[p]] = +1
    else:
      raise ValueError('Unknown spin(8) conventions: %r' % conventions)
    pos_slash = conventions.find('/')
    if pos_slash == -1:
      return ret
    return nsum('vsc->' + conventions[pos_slash + 1:], ret)

  def get_diagonalizing_rotation(self, m8x8):
    """Finds a SO(8) rotation that diagonalizes a symmetric 8x8 matrix."""
    eigvals, basis = scipy.linalg.eigh(m8x8)
    eigvecs = basis.T[[i for i, _ in sorted(enumerate(eigvals),
                                            key=lambda iv: -iv[1])]]
    det = numpy.linalg.det(eigvecs)
    assert abs(abs(det) - 1.0) < 1e-6, 'Eigenbasis not orthonormal'
    if det < 0:
      # Make eigenvector basis right-handed by reversing last eigenvector.
      eigvecs[-1] *= -1
    return eigvecs.T


class SU8:
  """Container class for su(8) tensor invariants.

  An instance essentially is just a namespace for constants.
  All attributes are to be considered as read-only by the user.

  Attributes:
    index56_and_coeff_by_ijk: dict mapping triplet (i, j, k) of three different
      su(8) indices to a pair of a 56-index and a sign factor (+1 or -1).
    ij_map: Lexicographically sorted list of pairs of su(8) indices
      (i, j) with i < j.
    inv_ij_map: Mapping that maps an (i, j)-key with i < j to the corresponding
      index of the pair in lexical ordering.
    m_35_8_8: [35, 8, 8]-array mapping a 35-index to a symmetric traceless
      matrix. Each such matrix has two entries of magnitude 1.
      The first 7 (8, 8) matrices are the lexicographically ordered matrices
      of the form diag(0, ..., 0, 1, -1, 0, ..., 0). The remaining 28 have
      a 1 in (i, j) and (j, i)-position and are zero otherwise.
    m_28_8_8: [28, 8, 8]-array mapping a 28-index to an antisymmetric traceless
      matrix. Each such matrix has one entry +1, one entry -1, and otherwise
      zeros. The locations of the +1s correspond to ij_map.
    ginv35: [35, 35]-array, inverse of the inner-products matrix for
      the 35-irrep of so(8) above that expands so(8) to su(8).
    m_56_8_8_8: [56, 8, 8, 8]-array mapping a 56-index to an antisymmetric
      [8, 8, 8]-array (or vice versa).
    m_action_56_56_8_8: [56, 56, 8, 8]-array mapping an 8x8 generator matrix
      to the corresponding action on the 56-irrep equivalent to [ijk].
    eps_56_56_8_8: epsilon^{ijklmnpq} with index groups (ijk) and (lmn) mapped
      to a 56-index.
    t_aij: su(8) generators (T_a)^j{}_i = t_aij[a, i, j].
  """

  def __init__(self):
    # Translates between adjoint indices 'a' and (vector) x (vector)
    # indices 'ij'.
    ij_map = [(i, j) for i in range(8) for j in range(8) if i < j]
    #
    # We also need the mapping between 8 x 8 and 35 representations of the
    # underlying spin(8), using common conventions for a basis of the
    # 35-representation, and likewise for 8 x 8 and 28.
    # These are used in various places, often with real-only quantities,
    # so we use dtype=float here, even though they also are used in complex
    # context.
    m_35_8_8 = numpy.zeros([35, 8, 8], dtype=numpy.float64)
    m_28_8_8 = numpy.zeros([28, 8, 8], dtype=numpy.float64)
    for n in range(7):
      m_35_8_8[n, n, n] = +1.0
      m_35_8_8[n, n + 1, n + 1] = -1.0
    for a, (m, n) in enumerate(ij_map):
      m_35_8_8[a + 7, m, n] = m_35_8_8[a + 7, n, m] = 1.0
      m_28_8_8[a, m, n] = 1.0
      m_28_8_8[a, n, m] = -1.0
    #
    # The su8 'Generator Matrices'.
    t_aij = numpy.zeros([63, 8, 8], dtype=numpy.complex128)
    t_aij[:35, :, :] = 1.0j * m_35_8_8
    print('IMPORTANT: Changed SO(8)-in-SU(8) definitions')
    # TODO(tfish): Apparently, with the definitions from the Big Paper,
    # the A1-tensor and hence gravitino masses come out wrong when
    # using the Theta-tensor formalism without this SO(8) sign change.
    # We will hence also need to fix so8_supergravity_extrema/:
    #  - Change distillation to diagonalize using an extra factor -1
    #    for so8 generators.
    #  - remove so8_supergravity_extrema/algebra.py - this is now unneeded.
    #  - add tests.
    for a, (i, j) in enumerate(ij_map):
      t_aij[a + 35, i, j] = -1.0  # Was: 1.0
      t_aij[a + 35, j, i] = +1.0  # Was: -1.0
    #
    # We also need to be able to map [ijk] to a linear 56-index.
    # Our choice of signs for the ijk-basis is essentially arbitrary here.
    # We lexicographically order triplets and attribute a + sign to
    # every first occurrence of a particular combination.
    index56_and_coeff_by_ijk = {}
    ijk_by_index56 = {}
    num_ijk = 0
    for i in range(8):
      for j in range(i + 1, 8):
        for k in range(j + 1, 8):
          ijk = (i, j, k)
          index56 = num_ijk
          ijk_by_index56[index56] = ijk
          num_ijk += 1
          for p in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for q_sign, q in ((1, (0, 1, 2)), (-1, (1, 0, 2))):
              pq_ijk = (ijk[p[q[0]]], ijk[p[q[1]]], ijk[p[q[2]]])
              index56_and_coeff_by_ijk[pq_ijk] = (index56, q_sign)
    # Let us also provide this as an (actually rather sparse) tensor.
    # We will only use this very occasionally.
    m_56_8_8_8 = numpy.zeros([56, 8, 8, 8])
    for ijk, (i56, sign) in index56_and_coeff_by_ijk.items():
      m_56_8_8_8[i56, ijk[0], ijk[1], ijk[2]] = sign
    # Supergravity has the "fermion mass" tensor
    # A3^ijk,lmn = (sqrt(2) / 144) * eps^ijkpqr[lm A2^n]_pqr
    # This structure suggests that it is numerically convenient
    # to have epsilon as a 56 x 56 x 8 x 8 tensor.
    eps_56_56_8_8 = numpy.zeros([56, 56, 8, 8])
    for p8 in itertools.permutations(range(8)):
      sign8 = mu.permutation_sign(p8)
      i56, coeff_i56 = index56_and_coeff_by_ijk[p8[:3]]
      j56, coeff_j56 = index56_and_coeff_by_ijk[p8[3: 6]]
      eps_56_56_8_8[i56, j56, p8[6], p8[7]] = sign8 * coeff_i56 * coeff_j56
    # Also, we need to know how su(8) elements given as 8x8 matrices act on this
    # 56x56-basis.
    m_action_56_56_8_8 = numpy.zeros([56, 56, 8, 8])
    for index56_left, ijk_left in ijk_by_index56.items():
      for index56_right, ijk_right in ijk_by_index56.items():
        common_indices = set(ijk_left) & set(ijk_right)
        if len(common_indices) != 2:
          continue
        # Two indices are the same, one gets transformed by the generator.
        transforming_index_left = [idx for idx in ijk_left
                                   if idx not in common_indices][0]
        transforming_index_right = [idx for idx in ijk_right
                                    if idx not in common_indices][0]
        transformed_ijk_left = [
            transforming_index_left if idx == transforming_index_right else idx
            for idx in ijk_right]
        sign = mu.permutation_sign([ijk_left.index(i)
                                    for i in transformed_ijk_left])
        m_action_56_56_8_8[
            index56_left, index56_right,
            transforming_index_left, transforming_index_right] = sign
    #
    self.index56_and_coeff_by_ijk = index56_and_coeff_by_ijk
    self.ij_map = ij_map
    self.inv_ij_map = {ij: num_ij for num_ij, ij in enumerate(ij_map)}
    self.m_35_8_8 = m_35_8_8
    self.ginv35 = numpy.linalg.inv(
        nsum('aij,bij->ab', m_35_8_8, m_35_8_8))
    self.m_28_8_8 = m_28_8_8
    self.m_56_8_8_8 = m_56_8_8_8
    self.eps_56_56_8_8 = eps_56_56_8_8
    self.m_action_56_56_8_8 = m_action_56_56_8_8
    self.t_aij = t_aij


class E7:
  r"""Container class for e7 tensor invariants.

  An instance essentially is just a namespace for constants.
  All attributes are to be considered as read-only by the user.

  Due to triality, we have freedom which 8-dimensional spin(8)
  representation to call the 'vector', 'spinor', and 'co-spinor'
  representation. For convenience, we here call the 8-dimensional
  representation whose symmetric product with itself provides
  compact directions in su(8) the 'vector' representation, and the other
  two representations the 'spinor' and 'co-spinor' representation, as this gives
  a mostly-symmetric role to spinors and co-spinors. These conventions deviate
  from some of the supergravity literature, but are convenient here.

  Attributes:
    t_a_ij_kl: [133, 56, 56]-array of e7(+7) generators
      (T_a)^{kl}{}_{ij} = t_aij[a, ij, kl] for the 56-dimensional fundamental
      irreducible representation. The 56-indices split into two pairs of
      28-indices that are antisymmetric index-pairs of the 8-dimensional su(8)
      representation. The first 70 of the 133 generators are the 35+35
      noncompact directions corresponding to the scalars of SO(8) supergravity.
      The subsequent 63 form the maximal compact subalgebra su(8).
    inv_gramian70: [70, 70]-array. Inverse inner product matrix of the first
      70 basis vectors. All entries in this matrix are exact (as integer
      multiples of 1/8) despite being float. This property can be relied on
      for high-accuracy computations.
    v70_as_sc8x8: [70, 2, 8, 8]-array that decomposes an e7 generator in
      e7/su(8) into two sets of symmetric traceless 8x8 matrices,
      (\alpha, \beta) and (\dot\alpha, \dot\beta), in this order.
    v70_from_sc8x8: Implements the inverse transform to v70_as_sc8x8.
    spin8_action_on_v70: [28, 70, 70]-array. For each spin(8) element
      in the 'vector' [i, j]-basis, provides the [70, 70] generator matrix
      when this generator acts on e7(7) / su(8). CAUTION: the output vector
      space's basis is dual (w.r.t. Killing form) to the input vector space's.
      This is useful for determining so(8)-invariant directions, but for
      computing mass matrices, spin8_action_on_v70o is much more appropriate.
    ijkl35: Tuple of all 35 lexicographically ordered 4-tuples of the form
      (0, j, k, l).
    fano_ijkl: Tuple of seven (i,j,k,l)-tuples that correspond to the quartic
      terms in (7.5) of arXiv:1909.10969 (with indexes shifted down to start
      from 0 rather than 1).
    sl2x7_88s: [7, 8, 8]-array, seven symmetric (8s x 8s) matrices that
      represent the 35s-components of seven commuting SL(2)s inside E7.
    sl2x7_88c: [7, 8, 8]-array, seven symmetric (8c x 8c) matrices that
      represent the 35c-components of seven commuting SL(2)s inside E7.
    sl2x7_88v: [7, 8, 8]-array, seven symmetric (8v x 8v) matrices that
      represent the 35v-components of seven commuting SL(2)s inside E7.
    sl2x7: [3, 7, 133]-array, indexed as [scv, num_sl2, e7_ad],
      generators for seven commuting SL(2)s inside E7. Leading index picks
      the (spinor, cospinor, vector)-35-irrep, middle index picks the
      specific SL(2), last index gives the e7-adjoint generator index.
      Indexing of the SL(2)s is in alignment with sl2x7_88{s,c,v}.
    v70_from_v70o: [70, 70]-array that maps orthonormal-basis-70-vectors to
      'common basis' 70-vectors.
      Commonly used as: v70 = v70_from_v70o.dot(v70o).
    v70o_from_v70: The inverse mapping of the above.
      Commonly used as: v70o = v70o_from_v70odot(v70).
    v133_from_v133o: [133, 133]-array that maps orthonormal-basis-133-vectors
      to 'common basis' 133-vectors.
    v133o_from_v133: The inverse mapping of the above.
    spin8_action_on_v70o: [28, 70, 70]-array. Like spin8_action_on_v70o,
      but with 70-vectors in the orthonormal basis.
    S_cr: [56, 56]-matrix that translates a "real sl(8)-basis"-index on the
      right to a "complex su(8)-basis" index on the left.
    S_rc: The inverse of S_cr.
    omega: [56, 56]-array, the symplectic E7 invariant.
    f_abc: [133, 133, 133]-array. The e7 structure constants with three
       adjoint indices.
    f_abC: [133, 133, 133]-array. The e7 structure constants with the last index
      in the co-adjoint representation.
    fo_abC: [133, 133, 133]-array. The e7 structure constants,
      using the orthonormal adjoint and co-adjoint basis.
    spin8: The Spin8 instance that was used to define E7.
    su8: The SU8 instance that was used to define E7.
  """

  def __init__(self, spin8, su8):
    """Initializes the instance.

    Args:
      spin8: The spin8 algebra to base this on. (May use varying conventions.)
      su8: The su(8) algebra to base this on.
    """
    self.spin8 = spin8
    self.su8 = su8
    t_a_ij_kl = numpy.zeros([133, 56, 56], dtype=numpy.complex128)
    t_a_ij_kl[:35, 28:, :28] = (1 / 8.0) * (
        nsum('ijklsS,qsS,Iij,Kkl->qIK',
             spin8.gamma_vvvvss, su8.m_35_8_8, su8.m_28_8_8, su8.m_28_8_8))
    t_a_ij_kl[:35, :28, 28:] = t_a_ij_kl[:35, 28:, :28]
    t_a_ij_kl[35:70, 28:, :28] = (1.0j / 8.0) * (
        nsum('ijklcC,qcC,Iij,Kkl->qIK',
             spin8.gamma_vvvvcc, su8.m_35_8_8, su8.m_28_8_8, su8.m_28_8_8))
    t_a_ij_kl[35:70, :28, 28:] = -t_a_ij_kl[35:70, 28:, :28]
    #
    # We need to find the action of the su(8) algebra on the
    # 28-representation.
    su8_28 = 2 * nsum('aij,mn,Iim,Jjn->aIJ',
                      su8.t_aij,
                      numpy.eye(8, dtype=numpy.complex128),
                      su8.m_28_8_8, su8.m_28_8_8)
    t_a_ij_kl[70:, :28, :28] = su8_28
    t_a_ij_kl[70:, 28:, 28:] = su8_28.conjugate()
    self.t_a_ij_kl = t_a_ij_kl
    m_35_8_8 = su8.m_35_8_8.real
    # Note that, due to the way our conventions work, the entries of this
    # matrix are all multiples of 1/8.0 = 0.125, which is an
    # exactly-representable floating point number. So, we are good to use this
    # even in conjunction with high-accuracy numerics(!). However,
    # we first have to 'sanitize away' numerical noise.
    raw_inv_gramian70 = nsum('AB,ab->AaBb', numpy.eye(2),
                             su8.ginv35).reshape(70, 70)
    self.inv_gramian70 = numpy.round(raw_inv_gramian70 * 8) / 8
    assert numpy.allclose(raw_inv_gramian70, self.inv_gramian70)
    # Assert that we only see 'good exact' numbers that are multiples of 1/8
    # with nonnegative values up to 16/8 = 2.
    assert set(abs(x * 8)
               for x in self.inv_gramian70.reshape(-1)) <= set(range(17))
    self.v70_as_sc8x8 = nsum('sc,xab->sxcab',
                             numpy.eye(2),
                             m_35_8_8).reshape(70, 2, 8, 8)
    self.v70_from_sc8x8 = nsum('vsab,vw->wsab',
                               self.v70_as_sc8x8,
                               self.inv_gramian70)
    # We also want to directly look at the action of the 28 Spin(8) generators
    # on the 70 scalars, both to determine residual gauge groups
    # (which we could also do in a 56-representation of E7),
    # and also to look for residual discrete subgroups of SO(8).
    spin8_action_on_s = 0.5 * nsum(
        'Aij,ijab->Aab', su8.m_28_8_8, spin8.gamma_vvss)
    spin8_action_on_c = 0.5 * nsum(
        'Aij,ijab->Aab', su8.m_28_8_8, spin8.gamma_vvcc)
    spin8_action_on_35s = (
        # [A,v,m,n]-array showing how acting with spin(8) generator A
        # changes a 35s element indexed by v, but with the change
        # expressed as a symmetric 8x8 matrix indexed (m, n).
        #
        # This could be simplified, exploiting symmetry, at the cost
        # of making the expression slightly less readable.
        nsum('Aab,van->Avbn', spin8_action_on_s,
             self.v70_as_sc8x8[:35, 0, :, :]) +
        nsum('Aab,vma->Avmb', spin8_action_on_s,
             self.v70_as_sc8x8[:35, 0, :, :]))
    spin8_action_on_35c = (
        # This could be simplified, exploiting symmetry, at the cost
        # of making the expression slightly less readable.
        nsum('Aab,van->Avbn', spin8_action_on_c,
             self.v70_as_sc8x8[35:, 1, :, :]) +
        nsum('Aab,vma->Avmb', spin8_action_on_c,
             self.v70_as_sc8x8[35:, 1, :, :]))
    spin8_action_on_35s35c = numpy.stack([spin8_action_on_35s,
                                          spin8_action_on_35c],
                                         axis=1)
    self.spin8_action_on_v70 = nsum(
        'Asvab,wsab->Asvw',
        spin8_action_on_35s35c,
        self.v70_from_sc8x8).reshape(28, 70, 70)
    # We also need an orthonormal basis for the 70 scalars.
    #
    # While we can find mass-eigenstates with the non-orthonormal basis
    # above (exercising a bit of care), these would be the eigenvalues of
    # a non-hermitean matrix operator. We do need orthonormal bases for
    # the mass-eigenstate subspaces so that subsequent automatic numerical
    # identification of charges can work (for which the code assumes that
    # charge-operators are represented as hermitean matrices, on which it
    # uses scipy.linalg.eigh() to produce orthonormal eigenbases).
    # We do not have to pay attention to define the mapping between these
    # 70-bases in a particularly elegant way.
    #
    # Also, it is important for high-accuracy calculations to have
    # exactly-representable matrix entries, while we can absorb an overall
    # (not-exactly-representable-at-finite-accuracy)
    # factor into the definition of the inner product.
    #
    # Furthermore, it is useful to have an orthonormal basis
    # for 56x133 Theta-tensors available. The primary reason here is that
    # we want to express things such as 'the projection to the linear
    # 912-constraint' in terms of proper coordinate-orthonormal bases.
    # If we used a non-orthonormal basis instead, this would lead to such
    # awkward situations as some important linear operations, such as
    # some projections, being conceptually expressible in terms of
    # symmetric matrices, but not appearing in that form - and consequently,
    # tf/numpy eigh() functions not being able to give us an orthonormal
    # eigenbasis (The corresponding .eig() functions do not give us
    # orthonormal bases and are numerically rather problematic to use).
    # So, we do form a "pseudoeuclidean-orthonormal" basis for the
    # entire 133-dimensional algebra for which the inner product
    # tr (g_a g_b) is delta_ab * {constant} * (+1 or -1).
    self.ijkl35 = tuple(ijkl for ijkl in itertools.combinations(range(8), 4)
                        if 0 in ijkl)
    v133_from_v133o = numpy.zeros([133, 133])
    # The SO(8) part stays unchanged.
    v133_from_v133o[-28:, -28:] = numpy.eye(28)
    for num_ijkl, ijkl in enumerate(self.ijkl35):
      v35s = nsum('mab,mM,ab->M', su8.m_35_8_8,
                  su8.ginv35,
                  spin8.gamma_vvvvss[ijkl[0], ijkl[1], ijkl[2], ijkl[3], :, :])
      v35c = nsum(
          'mab,mM,ab->M',
          su8.m_35_8_8, su8.ginv35,
          spin8.gamma_vvvvcc[ijkl[0], ijkl[1], ijkl[2], ijkl[3], :, :])
      v35v = nsum(
          'mab,mM,ab->M',
          su8.m_35_8_8, su8.ginv35,
          spin8.gamma_ssssvv[ijkl[0], ijkl[1], ijkl[2], ijkl[3], :, :])
      v133_from_v133o[0:35, num_ijkl] = 0.5 * v35s
      v133_from_v133o[35:70, 35 + num_ijkl] = 0.5 * v35c
      v133_from_v133o[70:105, 70 + num_ijkl] = 0.5 * v35v
    v70_from_v70o = v133_from_v133o[:70, :70]
    assert numpy.allclose(
        nsum('Vv,Wv->VW', v70_from_v70o, v70_from_v70o),
        2 * self.inv_gramian70)
    self.v133_from_v133o = v133_from_v133o
    self.v133o_from_v133 = numpy.linalg.inv(v133_from_v133o)
    #
    self.v70_from_v70o = v133_from_v133o[:70, :70]
    self.v70o_from_v70 = self.v133o_from_v133[:70, :70]
    self.spin8_action_on_v70o = nsum(
        'aVw,Ww->aVW',
        nsum('avw,vV->aVw',
             self.spin8_action_on_v70,
             self.v70_from_v70o),
        self.v70o_from_v70)
    #
    # Translating between the 'real' and 'complex' e7(7) basis.
    #
    self.S_cr = nsum('ab,AB->aAbB',
                     numpy.array([[1.0, 1.0j], [1.0, -1.0j]]),
                     numpy.eye(28)).reshape(56, 56) / 2**.5
    self.S_rc = numpy.linalg.inv(self.S_cr)
    assert numpy.allclose(self.S_rc @ self.S_cr, numpy.eye(56))
    #
    # The symplectic invariant of E7.
    #
    self.omega = nsum('ab,AB->aAbB',
                      numpy.array([[0.0, 1.0], [-1.0, 0.0]]),
                      numpy.eye(28)).reshape(56, 56)
    fano_quartets = Spin8.FANO_QUARTETS
    self.fano_ijkl = fano_ijkl = tuple(
        tuple(int(x) - 1 for x in xs)
        for xs in fano_quartets.split())
    hamming_factors = numpy.array(
        [-1, 1, 1, -1, 1, -1, -1])
    #
    # The 'seven commuting SL(2)s' submanifold.
    #
    self.sl2x7_88s = -numpy.stack([spin8.gamma_vvvvss[ijkl]
                                   for ijkl in fano_ijkl], axis=0)
    self.sl2x7_88c = -numpy.stack(
        [spin8.gamma_vvvvcc[ijkl]
         for ijkl in fano_ijkl], axis=0) * (
             hamming_factors[:, numpy.newaxis, numpy.newaxis])
    self.sl2x7_88v = numpy.stack(
        [numpy.diag([(+1.0, -1.0)[int(n in ijkl)]
                     for n in range(8)])
         for ijkl in fano_ijkl], axis=0) * (
             hamming_factors[:, numpy.newaxis, numpy.newaxis])
    def get_35s_from_88(n88):
      return 0.5 * nsum('nab,Aab,BA->nB',
                        n88, su8.m_35_8_8, su8.ginv35)
    sl2x7_35s = get_35s_from_88(self.sl2x7_88s)
    sl2x7_35c = get_35s_from_88(self.sl2x7_88c)
    sl2x7_35v = get_35s_from_88(self.sl2x7_88v)
    self.sl2x7 = numpy.stack(
        [numpy.pad(sl2x7_35s, [(0, 0), (0, 35 * 2 + 28)]),
         numpy.pad(sl2x7_35c, [(0, 0), (35, 35 + 28)]),
         numpy.pad(sl2x7_35v, [(0, 0), (35 * 2, 28)])],
        axis=0)

  def v70_from_35s35c(self, m35s, m35c):
    """Computes a v70-vector from 35s and 35c matrices."""
    return nsum('vsab,sab->v',
                self.v70_from_sc8x8,
                numpy.stack([m35s, m35c])).real

  def v70_as_35s35c(self, v70):
    m = nsum('v,vsab->sab', v70, self.v70_as_sc8x8).real
    return m[0], m[1]

  def get_cartan_subalgebra(self, conventions='SO8-SU8'):
    """Returns a Cartan Subalgebra of E7.

    The Cartan Subalgebra is embedded into the real E7 algebra,
    so if it consists of compact ("antisymmetric") generators,
    it lacks a factor 1j.

    Args:
      conventions: The conventions to use.
        Currently, this supports:
        - 'SO8-SU8': An all-compact-generators Cartan subalgebra
          whose first four generators form a Cartan subalgebra of
          SO(8)-in-E7.
        - 'SU8-diag7': A Cartan subalgebra consisting of seven
          commuting diag(0, 0, ..., 1, -1, 0, ..., 0) generators
          in 35v. These are not orthonormal w.r.t. the Killing form of E7.
        - 'SU8-fano': Seven noncompact generators from 35v that are in
          one-to-one alignment with lines on the Fano plane by
          taking the gamma_ssssvv elements of the 4-forms that are word-dual
          to the 3 indices on each line.
          If E7 is built with octionionic Spin(8)-conventions,
          this gives a particularly nice form of (SU(1,1)/U(1))^7.

    Returns:
      A float [7, 133]-ndarray of Cartan subalgebra generators.
    """
    su8 = self.su8
    inv_ij_map = su8.inv_ij_map
    cartan_gens = numpy.zeros([7, 133])
    if conventions == 'SO8-SU8':
      # The first four Cartan generators are the
      # [0, 1], [2, 3], [4, 5], [6, 7]-rotations of SO(8).
      # Overall normalization is such that Cartan-generators
      # have eigenvalues 1j * {-1, -0.5, 0, 0.5, 1}.
      for n in range(4):
        cartan_gens[n, 105 + inv_ij_map[(2 * n, 2 * n + 1)]] = 0.5
      # The next three Cartan generators must be identity matrices
      # on each [i, i+1]-subspace on which one of the first four
      # generators implements a SO(2)-rotation.
      diagonals = [[+1, +1, +1, +1, -1, -1, -1, -1],
                   [+1, +1, -1, -1, +1, +1, -1, -1],
                   [+1, +1, -1, -1, -1, -1, +1, +1]]
      # Scalar products with the [0, ..., 0, +1, -1, 0, ..., 0]
      # matrices:
      sprods = numpy.array([
          [diagonal[n] - diagonal[n + 1] for n in range(7)]
          for diagonal in diagonals])  # [3, 7]-array.
      # The inner products of the relevant 7 diagonal matrices
      # are available from self.inv_gramian70.
      cartan_gens[4:, 70:77] = nsum(
          'ab,na->nb',
          self.inv_gramian70[:7, :7],
          0.25 * sprods)
      return cartan_gens
    elif conventions == 'SU8-diag7':
      # This choice gives roots such as:
      # [ 0., -1.,  2., -1.,  0.,  0.,  0.]
      # [ 0., -0., -1., -0.,  0., -0.,  1.]
      for n in range(7):
        cartan_gens[n, 70 + n] = 0.5
      return cartan_gens
    elif conventions == 'SU8-fano':
      # Orthonormal choice in the '35v'.
      # Strategy: Out of the self-dual 4-forms over the spinors [abcd],
      # pick seven according to their complement being a
      # Fano-Plane-triplet.
      fano_lines = self.spin8.FANO_LINES
      fano_triplets = set(tuple(int(d) - 1 for d in ijk)
                          for ijk in fano_lines.split())
      abcds = [p7[3:] for p7 in itertools.permutations(range(7))
               if p7[:3] in fano_triplets and list(p7[3:]) == sorted(p7[3:])]
      for n, abcd in enumerate(abcds):
        v88 = self.spin8.gamma_ssssvv[abcd]
        # TODO(tfish): inv_gramian70[:35, :35] should also be available
        # separately.
        v35 = nsum('Aab,ab,BA->B',
                   self.su8.m_35_8_8, v88, self.inv_gramian70[:35, :35])
        cartan_gens[n, 70:105] = v35 * 0.25
      assert numpy.allclose(
          0, nsum('ma,nb,abC->mnC', cartan_gens, cartan_gens, self.f_abC)), (
              'Cartan-generators do not commute.')
      return cartan_gens

  def get_root_op(self, cartan_subalgebra, root):
    """Gets an E7 root-operator."""
    rs_plus = mu.get_simultaneous_eigenspace(
        (-1j) * nsum('na,abC->nCb', cartan_subalgebra, self.f_abC),
        root)
    if len(rs_plus) != 1:
      raise ValueError('Rootspace is not one-dimensional: root=%r' % root)
    # Subtlety: The eigenvector of a C^n -> C^n mapping, even if we
    # length-normalize it to 1, is only determined up to a complex phase -
    # and what we get may actually well depend on the eigenvalue library
    # implementation. So, we have to uniquely fix this phase here.
    # The most straightforward choice is to make the lexically-first
    # nonzero entry real-and-positive. While determining the 'lexically first'
    # vector entry of a numerically noisy vector is an ill-defined procedure,
    # we here can get away with it as long as our Cartan subalgebra choice
    # is reasonably nice.
    key_index, _ = next((n, c) for n, c in enumerate(rs_plus[0].round(7))
                        if abs(c))
    r_plus = rs_plus[0] * numpy.exp(-1j * numpy.log(rs_plus[0][key_index]).imag)
    r_minus = r_plus.conj()
    comm = nsum('abC,a,b->C', self.f_abC, r_plus, r_minus)
    # The commutator should be an element of the Cartan subalgebra (CSA).
    # We always pick the CSA such that it consists of all-compact or
    # all-noncompact generators, hence we normalize the magnitude of the root
    # such that it gives us an element of fixed-magnitude length-squared.
    #
    normalization = abs(
        nsum('AB,A,B->', (self.k133 / self.k133[-1, -1]), comm, comm))**.5 / 2.0
    return r_plus / normalization**.5

  @cached_property.cached_property
  def t56r(self):
    """The real "SL(8)-basis" e7(7) generator matrices."""
    # This only gets computed when actually needed.
    ret = nsum(
        'aRJ,JS->aRS',
        nsum('aIJ,IR->aRJ', self.t_a_ij_kl, self.S_cr),
        self.S_cr.conj())
    assert numpy.allclose(ret, ret.real)
    return ret.real

  @cached_property.cached_property
  def _f_abc_f_abC(self):
    # We here use the (smaller) real generator matrices.
    return mu.get_f_abc(self.t56r)

  @property
  def f_abc(self):
    return self._f_abc_f_abC[0]

  @property
  def f_abC(self):
    return self._f_abc_f_abC[1]

  @cached_property.cached_property
  def fo_abC(self):
    return nsum('abC,aA,bB,cC->ABc',
                self.f_abC,
                self.v133_from_v133o,
                self.v133_from_v133o,
                self.v133o_from_v133)

  @cached_property.cached_property
  def k133(self):
    return numpy.einsum('aBC,bCB->ab', self.f_abC, self.f_abC)


class E8:
  """Container class for e8 tensor invariants."""

  def __init__(self, spin8):
    self.spin8 = spin8
    gamma_vsc = spin8.gamma_vsc
    I_J_by_IJ = [(i, j) for i in range(16) for j in range(16) if i < j]
    IJ_by_I_J = {k: n for n, k in enumerate(I_J_by_IJ)}
    def paired(I, J):
      return (I, J) if I < J else (J, I)
    H_IJ_I_J = numpy.zeros([120, 16, 16], dtype=numpy.float64)
    for IJ, (I, J) in enumerate(I_J_by_IJ):
      H_IJ_I_J[IJ, I, J] = +0.5
      H_IJ_I_J[IJ, J, I] = -0.5
    gamma16_vsc = numpy.zeros([16, 128, 128], dtype=numpy.float64)
    gamma16_vsc[:8, :64, :64] = nsum(
        'ac,ibd->iabcd', numpy.eye(8), gamma_vsc).reshape(8, 64, 64)
    gamma16_vsc[:8, 64:, 64:] = nsum(
        'ac,idb->iabcd', numpy.eye(8), gamma_vsc).reshape(8, 64, 64)
    gamma16_vsc[8:, :64, 64:] = nsum(
        'bd,iac->iabcd', numpy.eye(8), gamma_vsc).reshape(8, 64, 64)
    gamma16_vsc[8:, 64:, :64] = -nsum(
        'bd,ica->iabcd', numpy.eye(8), gamma_vsc).reshape(8, 64, 64)
    #
    gamma16_vvss = mu.asymm2(
        nsum('IAC,JBC->IJAB', gamma16_vsc, gamma16_vsc), 'IJAB->JIAB')
    gamma16_IJ_A_B = nsum('IJAB,MIJ->MAB', gamma16_vvss, H_IJ_I_J)
    #
    f_ABC = numpy.zeros([248, 248, 248], dtype=numpy.float64)
    #
    def s(a, b, a_lt_b, a_gt_b):
      if a < b:
        return a_lt_b
      if a > b:
        return a_gt_b
      return 0.0
    for IJ, (I, J) in enumerate(I_J_by_IJ):
      # The IJ-rotation...
      for MN, (M, N) in enumerate(I_J_by_IJ):
        # ...when acting on MN, needs to have J=M, and produces IN,
        if J == M and I != N:
          f_ABC[128 + IJ,
                128 + MN,
                128 + IJ_by_I_J[paired(I, N)]] += s(I, N, -1, 1)
        # ...or needs to have J=N, and produces -IM,
        if J == N and I != M:
          f_ABC[128 + IJ,
                128 + MN,
                128 + IJ_by_I_J[paired(I, M)]] += s(I, M, +1, -1)
        # Or the negatives, for I<->J.
        if I == M and J != N:
          f_ABC[128 + IJ,
                128 + MN,
                128 + IJ_by_I_J[paired(J, N)]] += s(J, N, +1, -1)
        if I == N and J != M:
          f_ABC[128 + IJ,
                128 + MN,
                128 + IJ_by_I_J[paired(J, M)]] += s(J, M, -1, +1)
    f_ABC[128:, :128, :128] = +0.5 * gamma16_IJ_A_B
    f_ABC[:128, 128:, :128] = +0.5 * nsum('MAB->BMA', gamma16_IJ_A_B)
    f_ABC[:128, :128, 128:] = -0.5 * nsum('MAB->ABM', gamma16_IJ_A_B)
    self.I_J_by_IJ = I_J_by_IJ
    self.IJ_by_I_J = IJ_by_I_J
    self.gamma16_vsc = gamma16_vsc
    self.H_IJ_I_J = H_IJ_I_J
    self.f_ABC = f_ABC
    #
    self.spin16_I_J_CB = nsum(
        'ABC,AIJ->IJCB',
        f_ABC,
        numpy.pad(H_IJ_I_J, ((128, 0), (0, 0), (0, 0))))


class G:
  """Wrapper class for lazy-instantiation of only actually-used algebra definitions."""

  @cached_property.cached_property
  def spin8(self):
    return Spin8()

  @cached_property.cached_property
  def su8(self):
    return SU8()

  @cached_property.cached_property
  def e7(self):
    return E7(self.spin8, self.su8)

  @cached_property.cached_property
  def e8(self):
    return E8(self.spin8)


g = G()
