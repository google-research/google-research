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
import numpy
import scipy.linalg
import re


def np_esum(spec, *tensors, optimize='greedy'):
  """Shorthand for numpy.einsum with 'greedy' optimization default."""
  return numpy.einsum(spec, *tensors, optimize=optimize)


def dict_from_tensor(tensor, magnitude_threshold=0):
  """Converts a tensor to a dict of nonzero-entries keyed by index-tuple."""
  ret = {}
  for index_tuple in itertools.product(*(map(range, tensor.shape))):
    v = tensor[index_tuple]
    if abs(v) > magnitude_threshold:
      ret[index_tuple] = v
  return ret


def permutation_sign(p):
  """Determines the sign of a permutation, given as a sequence of integers."""
  q = list(p)  # Copy to list.
  sign = 1
  for n in range(len(p)):
    while n != q[n]:
      qn = q[n]
      q[n], q[qn] = q[qn], q[n]  # Flip to make q[qn] = qn.
      sign = -sign
  return sign


class Spin8(object):
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
  """

  def __init__(self, conventions='gsw'):
    """Initializes the instance.

    Args:
      conventions: str, the conventions to use for Gamma matrices. Options are:
        'gsw': Green, Schwarz, Witten, "Superstring Theory", Volume 1, (5.B.3).
        'octonionic': https://arxiv.org/abs/math/0105155, Table 1.
    """
    r8 = range(8)
    self._conventions = conventions
    self.gamma_vsc = gamma_vsc = self._get_gamma_vsc(conventions=conventions)
    #
    # The gamma^{ab}_{alpha beta} tensor that translates between antisymmetric
    # matrices over vectors [ij] and antisymmetric matrices over spinors [sS].
    self.gamma_vvss = 0.5 * (
        numpy.einsum('isc,jSc->ijsS', gamma_vsc, gamma_vsc) -
        numpy.einsum('jsc,iSc->ijsS', gamma_vsc, gamma_vsc))
    # The gamma^{ab}_{alpha* beta*} tensor that translates between antisymmetric
    # matrices over vectors [ij] and antisymmetric matrices over cospinors [cC].
    self.gamma_vvcc = 0.5 * (
        numpy.einsum('isc,jsC->ijcC', gamma_vsc, gamma_vsc) -
        numpy.einsum('jsc,isC->ijcC', gamma_vsc, gamma_vsc))
    # The gamma^{alpha beta}_{alpha* beta*} are not needed for the supergravity
    # computation per se, but we use these objects to determine the Spin(8)
    # rotation that brings E7 / SU(8)
    # scalar manifold coordinates into uniquely defined normal form.
    self.gamma_sscc = 0.5 * (
        numpy.einsum('vsc,vSC->sScC', gamma_vsc, gamma_vsc) -
        numpy.einsum('vsc,vSC->SscC', gamma_vsc, gamma_vsc))
    #
    # The gamma^{ijkl}_{alpha beta} tensor that translates between antisymmetric
    # 4-forms [ijkl] and symmetric traceless matrices over the spinors (sS),
    # as well as its co-spinor (cC) cousin.
    g_ijsS = numpy.einsum('isc,jSc->ijsS', self.gamma_vsc, self.gamma_vsc)
    g_ijcC = numpy.einsum('isc,jsC->ijcC', self.gamma_vsc, self.gamma_vsc)
    g_ijklsS = numpy.einsum('ijst,kltS->ijklsS', g_ijsS, g_ijsS)
    g_ijklcC = numpy.einsum('ijcd,kldC->ijklcC', g_ijcC, g_ijcC)
    gamma_vvvvss = numpy.zeros([8] * 6)
    gamma_vvvvcc = numpy.zeros([8] * 6)
    for perm in itertools.permutations(range(4)):
      perm_ijkl = ''.join('ijkl' [p] for p in perm)
      sign = permutation_sign(perm)
      gamma_vvvvss += sign * numpy.einsum(perm_ijkl + 'sS->ijklsS', g_ijklsS)
      gamma_vvvvcc += sign * numpy.einsum(perm_ijkl + 'cC->ijklcC', g_ijklcC)
    self.gamma_vvvvss = gamma_vvvvss / 24.0
    self.gamma_vvvvcc = gamma_vvvvcc / 24.0

  def _get_gamma_vsc(self, conventions='gsw'):
    """Computes Spin(8) gamma-matrices.

    Args:
      conventions: str, the conventions to use for Gamma matrices. Options are:
        'gsw': Green, Schwarz, Witten, "Superstring Theory", Volume 1, (5.B.3).
        'octonionic': https://arxiv.org/abs/math/0105155, Table 1.

    Returns:
      [8, 8, 8]-tensor gamma^i_\alpha\dot\beta.
    """
    ret = numpy.zeros([8, 8, 8])
    if conventions == 'gsw':
      # Conventions for Spin(8) gamma matrices match Green, Schwarz, Witten,
      # but with indices shifted down by 1 to the range [0 .. 7].
      entries = (
          "007+ 016- 025- 034+ 043- 052+ 061+ 070- "
          "101+ 110- 123- 132+ 145+ 154- 167- 176+ "
          "204+ 215- 226+ 237- 240- 251+ 262- 273+ "
          "302+ 313+ 320- 331- 346- 357- 364+ 375+ "
          "403+ 412- 421+ 430- 447+ 456- 465+ 474- "
          "505+ 514+ 527+ 536+ 541- 550- 563- 572- "
          "606+ 617+ 624- 635- 642+ 653+ 660- 671- "
          "700+ 711+ 722+ 733+ 744+ 755+ 766+ 777+")
      for ijkc in entries.split():
        indices = tuple([int(m) for m in ijkc[:-1]])
        sign = 1 if ijkc[-1] == '+' else -1
        ret[indices] = sign
    elif conventions == 'octonionic':
      fano_lines = "124 156 137 235 267 346 457"
      for n in range(1, 8):
        ret[0, n, n] = -1
        ret[n, n, 0] = ret[n, 0, n] = 1
      ret[0, 0, 0] = 1
      for cijk in fano_lines.split():
        ijk = tuple(int(idx) for idx in cijk)
        for p, q, r in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
          # Note that we have to `go against the direction of the arrows'
          # to make the correspondence work.
          ret[ijk[r], ijk[p], ijk[q]] = -1
          ret[ijk[r], ijk[q], ijk[p]] = +1
    else:
      raise ValueError('Unknown spin(8) conventions: %r' % conventions)
    return ret


class SU8(object):
  """Container class for su(8) tensor invariants.

  An instance essentially is just a namespace for constants.
  All attributes are to be considered as read-only by the user.

  Attributes:
    index56_and_coeff_by_ijk: dict mapping triplet (i, j, k) of three different
      su(8) indices to a pair of a 56-index and a sign factor (+1 or -1).
    ij_map: Lexicographically sorted list of pairs of su(8) indices
      (i, j) with i < j.
    m_35_8_8: [35, 8, 8]-array mapping a 35-index to a symmetric traceless
      matrix. Each such matrix has two entries of magnitude 1.
      The first 7 (8, 8) matrices are the lexicographically ordered matrices
      of the form diag(0, ..., 0, 1, -1, 0, ..., 0). The remaining 28 have
      a 1 in (i, j) and (j, i)-position and are zero otherwise.
    m_56_8_8_8: [56, 8, 8, 8]-array mapping a 56-index to an antisymmetric
      [8, 8, 8]-array (or vice versa).
    eps_56_56_8_8: epsilon^{ijklmnpq} with index groups (ijk) and (lmn) mapped
      to a 56-index.
    t_aij: su(8) generators (T_a)^j{}_i = t_aij[a, i, j].
  """

  def __init__(self):
    # Translates between adjoint indices 'a' and (vector) x (vector)
    # indices 'ij'.
    ij_map = [(i, j) for i in range(8) for j in range(8) if i < j]
    #
    # We also need the mapping between 8 x 8 and 35 representations, using
    # common conventions for a basis of the 35-representation, and likewise
    # for 8 x 8 and 28.
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
    for a, (i, j) in enumerate(ij_map):
      t_aij[a + 35, i, j] = 1.0
      t_aij[a + 35, j, i] = -1.0
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
      sign8 = permutation_sign(p8)
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
        sign = permutation_sign([ijk_left.index(i)
                                 for i in transformed_ijk_left])
        m_action_56_56_8_8[
            index56_left, index56_right,
            transforming_index_left, transforming_index_right] = sign
    #
    self.index56_and_coeff_by_ijk = index56_and_coeff_by_ijk
    self.ij_map = ij_map
    self.m_35_8_8 = m_35_8_8
    self.m_28_8_8 = m_28_8_8
    self.m_56_8_8_8 = m_56_8_8_8
    self.eps_56_56_8_8 = eps_56_56_8_8
    self.m_action_56_56_8_8 = m_action_56_56_8_8
    self.t_aij = t_aij


class E7(object):
  """Container class for e7 tensor invariants.

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
    v70_from_v70o: [70, 70]-array that maps orthonormal-basis-70-vectors to
      'common basis' 70-vectors.
    v70o_from_v70: The inverse mapping of the above.
    spin8_action_on_v70o: [28, 70, 70]-array. Like spin8_action_on_v70o,
      but with 70-vectors in the orthonormal basis.
  """

  def __init__(self, spin8, su8):
    self._spin8 = spin8
    self._su8 = su8
    ij_map = su8.ij_map
    t_a_ij_kl = numpy.zeros([133, 56, 56], dtype=numpy.complex128)
    t_a_ij_kl[:35, 28:, :28] = (1 / 8.0) * (
        np_esum('ijklsS,qsS,Iij,Kkl->qIK',
                spin8.gamma_vvvvss, su8.m_35_8_8, su8.m_28_8_8, su8.m_28_8_8))
    t_a_ij_kl[:35, :28, 28:] = t_a_ij_kl[:35, 28:, :28]
    t_a_ij_kl[35:70, 28:, :28] = (1.0j / 8.0) * (
        np_esum('ijklcC,qcC,Iij,Kkl->qIK',
                spin8.gamma_vvvvcc, su8.m_35_8_8, su8.m_28_8_8, su8.m_28_8_8))
    t_a_ij_kl[35:70, :28, 28:] = -t_a_ij_kl[35:70, 28:, :28]
    #
    # We need to find the action of the su(8) algebra on the
    # 28-representation.
    su8_28 = 2 * np_esum('aij,mn,Iim,Jjn->aIJ',
                         su8.t_aij,
                         numpy.eye(8, dtype=numpy.complex128),
                         su8.m_28_8_8, su8.m_28_8_8)
    t_a_ij_kl[70:, :28, :28] = su8_28
    t_a_ij_kl[70:, 28:, 28:] = su8_28.conjugate()
    self.t_a_ij_kl = t_a_ij_kl
    m_35_8_8 = su8.m_35_8_8.real
    inv_inner_products = numpy.linalg.inv(
        numpy.einsum('aij,bij->ab', m_35_8_8, m_35_8_8))
    # Note that, due to the way our conventions work, the entries of this
    # matrix are all multiples of 1/8.0 = 0.125, which is an
    # exactly-representable floating point number. So, we are good to use this
    # even in conjunction with high-accuracy numerics(!). However,
    # we first have to 'sanitize away' numerical noise.
    raw_inv_gramian70 = numpy.einsum('AB,ab->AaBb', numpy.eye(2),
                                     inv_inner_products).reshape(70, 70)
    self.inv_gramian70 = numpy.round(raw_inv_gramian70 * 8) / 8
    assert numpy.allclose(raw_inv_gramian70, self.inv_gramian70)
    # Assert that we only see 'good exact' numbers that are multiples of 1/8
    # with nonnegative values up to 16/8 = 2.
    assert set(abs(x * 8)
               for x in self.inv_gramian70.reshape(-1)) <= set(range(17))
    # Auxiliary constant to map [2, 8, 8] (sc, i, j)-data to 70-vectors.
    aux_35_from_8x8 = numpy.einsum('Aa,aij->Aij',
                                   inv_inner_products, m_35_8_8)
    self.v70_as_sc8x8 = numpy.einsum('sc,xab->sxcab',
                                     numpy.eye(2),
                                     m_35_8_8).reshape(70, 2, 8, 8)
    self.v70_from_sc8x8 = numpy.einsum('vsab,vw->wsab',
                                       self.v70_as_sc8x8,
                                       self.inv_gramian70)
    # We also want to directly look at the action of the 28 Spin(8) generators
    # on the 70 scalars, both to determine residual gauge groups
    # (which we could also do in a 56-representation of E7),
    # and also to look for residual discrete subgroups of SO(8).
    spin8_action_on_s = 0.5 * numpy.einsum(
        'Aij,ijab->Aab', su8.m_28_8_8, spin8.gamma_vvss)
    spin8_action_on_c = 0.5 * numpy.einsum(
        'Aij,ijab->Aab', su8.m_28_8_8, spin8.gamma_vvcc)
    spin8_action_on_35s = (
        # [A,v,m,n]-array showing how acting with spin(8) generator A
        # changes a 35s element indexed by v, but with the change
        # expressed as a symmetric 8x8 matrix indexed (m, n).
        #
        # This could be simplified, exploiting symmetry, at the cost
        # of making the expression slightly less readable.
        numpy.einsum('Aab,van->Avbn', spin8_action_on_s,
                     self.v70_as_sc8x8[:35, 0, :, :]) +
        numpy.einsum('Aab,vma->Avmb', spin8_action_on_s,
                     self.v70_as_sc8x8[:35, 0, :, :]))
    spin8_action_on_35c = (
        # This could be simplified, exploiting symmetry, at the cost
        # of making the expression slightly less readable.
        numpy.einsum('Aab,van->Avbn', spin8_action_on_c,
                     self.v70_as_sc8x8[35:, 1, :, :]) +
        numpy.einsum('Aab,vma->Avmb', spin8_action_on_c,
                     self.v70_as_sc8x8[35:, 1, :, :]))
    spin8_action_on_35s35c = numpy.stack([spin8_action_on_35s,
                                          spin8_action_on_35c],
                                         axis=1)
    self.spin8_action_on_v70 = numpy.einsum(
        'Asvab,wsab->Asvw',
        spin8_action_on_35s35c,
        self.v70_from_sc8x8).reshape(28, 70, 70)
    #
    # We also need an orthonormal basis for the 70 scalars.
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
    v70_from_v70o = numpy.zeros([70, 70])
    for num_ijkl, ijkl in enumerate(
        ijkl for ijkl in itertools.combinations(range(8), 4) if 0 in ijkl):
      v35a = numpy.einsum('vsab,s,ab->v',
                          self.v70_from_sc8x8,
                          numpy.array([1.0, 0.0]),
                          spin8.gamma_vvvvss[
                              ijkl[0], ijkl[1], ijkl[2], ijkl[3], :, :])
      v35b = numpy.einsum('vsab,s,ab->v',
                          self.v70_from_sc8x8,
                          numpy.array([0.0, 1.0]),
                          spin8.gamma_vvvvcc[
                              ijkl[0], ijkl[1], ijkl[2], ijkl[3], :, :])
      v70_from_v70o[:, num_ijkl] = 0.5 * v35a
      v70_from_v70o[:, 35 + num_ijkl] = 0.5 * v35b
    assert numpy.allclose(
        numpy.einsum('Vv,Wv->VW', v70_from_v70o, v70_from_v70o),
        2 * self.inv_gramian70)
    self.v70_from_v70o = v70_from_v70o
    self.v70o_from_v70 = numpy.linalg.inv(v70_from_v70o)
    self.spin8_action_on_v70o = numpy.einsum(
        'aVw,Ww->aVW',
        numpy.einsum('avw,vV->aVw',
                     self.spin8_action_on_v70,
                     self.v70_from_v70o),
        self.v70o_from_v70)

  def v70_from_35s35c(self, m35s, m35c):
    """Computes a v70-vector from 35s and 35c matrices."""
    return numpy.einsum('vsab,sab->v',
                        self.v70_from_sc8x8,
                        numpy.stack([m35s, m35c]))

  def v70_as_35s35c(self, v70):
    m = numpy.einsum('v,vsab->sab', v70, self.v70_as_sc8x8)
    return m[0], m[1]



spin8 = Spin8()
su8 = SU8()
e7 = E7(spin8, su8)
