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

"""Generic reduction of four-fermion terms arising from D=11 Supergravity.

Note: This code deliberately does not use Python type annotations,
for the following reasons:

 1. Physicists generally are not familiar with the underlying byzantine rules,
    so this would make the code less inclusive.
 2. They would be of little value, since many things would be `ArrayLike`.
"""

import abc
import itertools
import math

import numpy
import psi4_accum as _psi4_accum  # Module-internal use only!


# Linter has False Positives here.
# pylint:skip-file


# Short-hands.
iprod = itertools.product
range1_11 = range(1, 11)
range1 = range(1)
range32 = range(32)


# The Pauli Spin Matrices, as [3, 2, 2] complex array.
PAULI_SIGMA = numpy.stack(
    [numpy.eye(2),  # sigma_0 == identity.
     numpy.eye(2)[::-1],  # sigma_x
     numpy.diag([1j, -1j])[::-1],  # sigma_y
     numpy.diag([1, -1])],  # sigma_z
    axis=0)


# Default ranking polynomial for 4-tuples, converting four indices
# (a,b,c,d) in the range [0..319] to the linear index of the 4-tuple
# in lexicographic ordering. Must use dtype=int64 and machine
# byte order for use by _psi4_accum.collect_quartic_terms().
#
# Note: Having to deal with this is, in a way, due to laziness.
# We could also handle this by working out the ordered-quadruple
# ranking formula for indices in [0..N] once and for all.
#
# Entries are (where (a,b,c,d) is the lexically ordered 4-tuple
# of all different coefficients):
#   [denominator, 1-coeff, d-coeff, c-coeff,
#    c**2-coeff, b-coeff, b**2-coeff, b**3-coeff, a-coeff,
#    a**2-coeff, a**3-coeff, a**4-coeff].
RANKING_COEFFS_320_4 = numpy.array(
    [24, -1224984, 24, 7644, -12, 1213484, -3816, +4,
     +128022350, -604835, +1270, -1], dtype=numpy.int64)


def permutation_sign_d(perm):
  """Determines the sign of a permutation, destructively.

  Args:
   perm: list representation of a permutation, must satisfy
     `set(perm) == set(range(len(perm)))`. Will me mutated.

  Returns:
    +1 if `perm` can be reduced to list(range(len(perm)))
    by an even number of swaps, -1 otherwise.
  """
  sign = 1
  for n in range(len(perm)):
    while n != perm[n]:
      pn = perm[n]
      perm[n], perm[pn] = perm[pn], perm[n]  # Swap to make perm[pn] = pn.
      sign = -sign
  return sign


def get_sign_and_canonicalized(indices):
  """Sorts an index-sequence, determining the sorting permutation's sign.

  Args:
    indices: Arraylike of >= 0 indices. Must not have duplicates.

  Returns:
    Pair `(sorted(indices), {sign of the reordering permutation, +1 or -1})`.
  """
  p_sorted = sorted(indices)
  # O(N^2), but for N this small, this is still competitive.
  t_indices = tuple(indices)  # For calling .index().
  reordering = [t_indices.index(x) for x in p_sorted]
  return permutation_sign_d(reordering), p_sorted


# === D=10+1 Majorana Gammas ===


def get_gammas11d():
  """Computes the Spin(1,10) Gamma matrices, as real [11, 32, 32]-array."""
  s0, sx, sy, sz = PAULI_SIGMA
  # Kronecker product for five sigma matrices.
  def kp5(p, q, r, s, t):
    return numpy.einsum(
        'Aa,Bb,Cc,Dd,Ee->ABCDEabcde',
        p, q, r, s, t).reshape(32, 32)
  return numpy.stack(
      [kp5(1j*sy, sy, sy, sy, sy),
       kp5(sx, sy, sy, sy, sy),
       kp5(sz, sy, sy, sy, sy),
       kp5(s0, sx, s0, s0, s0),
       kp5(s0, sz, s0, s0, s0),
       kp5(s0, sy, sy, sx, s0),
       kp5(s0, sy, sy, sz, s0),
       kp5(s0, sy, sx, s0, sy),
       kp5(s0, sy, sz, s0, sy),
       kp5(s0, sy, s0, sy, sx),
       kp5(s0, sy, s0, sy, sz)],
      axis=0).real


def _gamma_cache_tag(vindices):
  # Hack: We use the byte-data representations of uint8
  # index-vectors as cache-keys. This is more compact and convenient
  # than using tuples (and faster to compare).
  return numpy.asarray(vindices, dtype=numpy.uint8).data.tobytes()


class Gamma11Family:
  """Family of Spin(1, 10) 1_AB, Gamma^I_AB, Gamma^IJ_AB etc. Gamma matrices.

  Must be based on a 'friendly representation'. For Spin(1, 10),
  we explicitly use that Gamma matrices are real/Majorana and multiply via
  2 Gamma^IJ_AB = Gamma^I_AC Gamma_J_CB - (J<->I) (but see text for
  simplification) - and likewise for higher products.

  Attributes:
    gamma_IAB: [11, 32, 32]-numpy.ndarray, the original Gamma matrix data.
  """

  def __init__(self, *, gamma_IAB=None):
    """Initializes the instance.

    Args:
      gamma_IAB: Optional Gamma matrices, as an [11, 32, 32] real
        array-like geometric constant. If `None`, then the default
        choice produced by get_gammas11d() will get used.
    """
    super().__init__()  # Chain __init__() via Python MRO.
    # If one were to generalize this family to e.g. Weyl spinors,
    # care would have to be taken about even-number-of-vector-index Gammas
    # coming in two variants: spinor/spinor and co-spinor/co-spinor.
    # Naturally, one would want to keep spinor-indices to the left
    # of cospinor-indices in products, but for D=11, we do not have
    # to concern ourselves with this.
    if gamma_IAB is not None:
      gamma_IAB = numpy.asarray(gamma_IAB)
    else:
      gamma_IAB = get_gammas11d()
    if gamma_IAB.shape != (11, 32, 32):
      raise ValueError(
          f'Gamma shape {gamma_IAB.shape} is not as expected for Spin(1,10).')
    if not (gamma_IAB.imag == 0).all():
      raise ValueError('D=11 Majorana Gamma-matrices should be all-real.')
    # Dimensionalities of the Spin(1,10) vector and spinor representations.
    dim_v, dim_s = 11, 32
    entries = set(gamma_IAB.ravel())
    if not entries <= {-1, 0, 1}:
      raise ValueError(
          f'Gamma-matrices must only have entries -1, 0, +1. Found: {entries}')
    # The compact representation of each Gamma^I uses a [2, dim_s] numpy-array,
    # where the [0, :]-part provides a permutation and the [1, :]-part
    # provides the +1/-1-factor to multiply with post-permutation.
    # Generalization note: For complex (Dirac) Gamma matrices in other
    # dimensions, it would be advantageous to count the number of 1j-factors
    # instead (mod-4, which would be automatic for uint-types), but we
    # here nevertheless go with signs, for simplicity.
    gdata = numpy.zeros([dim_v, 2, dim_s], dtype=numpy.int8)
    # Hack: taking the absolute magnitude of the gammas gives us entries 0, 1.
    # Here, we can conveniently use .argmax() to find the nonzero entries.
    abs_gamma_IAB = abs(gamma_IAB)
    # Quick consistency check: every gamma has dim_s many ones.
    if not ((abs_gamma_IAB.sum(axis=1) == 1).all() and
            (abs_gamma_IAB.sum(axis=2) == 1).all()):
      raise ValueError(
          'Gammas need to have a single +/-1 per row and column.')
    # Now, populate data per-matrix. We have a single +/-1 per row.
    # We henceforth call the left spinor-index on Gamma^I_AB the "row-index",
    # and the right spinor-index the "column-index", as customary.
    for idx_i in range(dim_v):
      abs_gamma = abs_gamma_IAB[idx_i]
      # nz_ == "nonzero_", "cix_" == "column_index_", "rix_" == "row_index_".
      nz_cix_by_rix = abs_gamma.argmax(axis=1)  # "row: which column?"
      # Only needed for expectation-check below.
      nz_rix_by_cix = abs_gamma.argmax(axis=0)  # "column: which row?"
      entry_by_row = numpy.take_along_axis(
          gamma_IAB[idx_i], nz_cix_by_rix[:, numpy.newaxis], 1).ravel()
      entry_by_col = numpy.take_along_axis(
          gamma_IAB[idx_i], nz_rix_by_cix[numpy.newaxis, :], 0).ravel()
      if not ((abs(entry_by_row) == 1).all() and
              (abs(entry_by_col) == 1).all()):
        # Let us make extra sure we will detect having been fed "bad"
        # Gamma-matrices.
        raise ValueError(
            'Gammas need to have a single +/-1 per row and column.')
      # The `entry_by_row` data is used if the row-index survives
      # and we sum over the column-index, and vice-versa.
      gdata[idx_i, 0, :] = nz_cix_by_rix
      gdata[idx_i, 1, :] = entry_by_row
    # It is good practice to only actually do instance state-mutations
    # once we know everything is OK. This way, even raising an exception
    # in __init__ will not leave us with half-initialized object-state.
    self.gamma_IAB = gamma_IAB
    self._gdata = gdata
    # Gamma-matrix by vstring tag.
    # Hack: We use the byte-data representations of uint8
    # index-vectors as cache-keys. This is more compact and convenient
    # than using tuples (and faster to compare).
    self._cached_gamma_by_tag = {
        _gamma_cache_tag([idx_i]): gdata[idx_i] for idx_i in range(dim_v)
    } | {
        # We also can straightaway register Gamma^[].
        # This is the identity matrix.
        _gamma_cache_tag([]): numpy.stack(
            [numpy.arange(32), numpy.ones([32])], axis=0).astype(numpy.int8)}
    # Hack: Looking up a method will first do a lookup on the instance,
    # which will fail, and then fall back to doing a lookup on the class,
    # where a callable result will get wrapped up into a method-object
    # for the original instance. We can speed this up by shadowing the
    # class-callables with instance-attributes that are the corresponding
    # method-objects. This saves both method-object instantiation and
    # the failing instance-lookup.
    # Technically, these attributes should then be documented as instance-
    # attributes, but they behave like methods (only faster).
    # They would however show up in self.__dict__.
    # pylint:disable=g-missing-from-attributes
    self._get_sign_and_gdata = self._get_sign_and_gdata
    self.apply_right = self.apply_right

  def _get_sign_and_gdata(self, vstring):
    """Registers a product of gamma-matrices."""
    sign, canonicalized = get_sign_and_canonicalized(vstring)
    vtag = _gamma_cache_tag(canonicalized)
    cache = self._cached_gamma_by_tag
    maybe_gdata = cache.get(vtag)  # overwritten if `None`.
    if maybe_gdata is not None:
      return sign, maybe_gdata
    # Otherwise, we do not have an entry.
    mid = len(vstring) // 2
    sign_le, gdata_le = self._get_sign_and_gdata(canonicalized[:mid])
    sign_ri, gdata_ri = self._get_sign_and_gdata(canonicalized[mid:])
    assert sign_le == sign_ri == 1  # Substrings are also canonicalized!
    gdata_product = numpy.zeros_like(gdata_le)
    # Indexing a numpy-vector with an int-vector performs index-gathering.
    # Explanation: see accompanying "Bit Twiddling Hacks" paper.
    gdata_product[0, :] = gdata_ri[0, :][gdata_le[0, :]]
    gdata_product[1, :] = gdata_le[1, :] * gdata_ri[1, :][gdata_le[0, :]]
    cache[vtag] = gdata_product
    return sign, gdata_product

  def apply_right(self, vstring, spinor):
    """Contracts a gamma-matrix on its right index with a spinor.

    Args:
      vstring: spacetime-vector-index-sequence denoting the gamma-matrix.
      spinor: the spinor to apply the gamma-matrix to.

    Returns:
      A pair (permuted, signs) of permuted-vector-entries and signs,
      both as new numpy.ndarray-s.
      We do not by default multiply the signs into the vector, since
      vector-entries may actually be element-tags rather than
      actual numbers.
    """
    sign, gdata = self._get_sign_and_gdata(vstring)
    g_times_vec = numpy.asarray(spinor)[gdata[0, :]]
    signs = sign * gdata[1, :]
    return g_times_vec, signs

  def as_matrix(self, vstring):
    """Returns a Gamma-matrix in matrix form.

    Args:
      vstring: spacetime-vector-index-sequence denoting the gamma-matrix.

    Returns:
      The Gamma matrix, as a [32, 32]-ndarray of the same kind as
      the original Gamma0.
    """
    # This is not only useful for debugging, but also if e.g. for some reason
    # want to do something with an explicit gamma0.
    ret = numpy.zeros_like(self.gamma_IAB[0])
    sign, gdata = self._get_sign_and_gdata(vstring)
    for n, (index, i_sign) in enumerate(zip(*gdata)):
      ret[n, index] = sign * i_sign
    return ret

  def as_sparse(self, vindices):
    """Returns a Gamma-matrix in sparse form.

    Args:
      vindices: spacetime-vector-index-sequence denoting the gamma-matrix.

    Returns:
      A new list of `(coefficient, (left_index, right_index))` tuples,
      one per nonzero entry of the Gamma matrix.
    """
    ret = []
    sign, gdata = self._get_sign_and_gdata(vindices)
    for n, (index, i_sign) in enumerate(zip(*gdata)):
      ret.append((sign * i_sign, (n, index)))
    return ret


# === D=10+1 Vector-Spinors ===

# We do need the default definitions in any case, even if specific
# applications might additionally go with other definitions.
# Convenient shorthands are useful here.
G11 = get_gammas11d()
G11F = Gamma11Family()


# We identify the basic anticommuting/fermionic variables with
# `psi[v={1..10}, s={0..31}]`; The 32 `psi[v=0,s=...]` are absent
# since the vector-spinor Gamma-tracelessness condition
# `Gamma_IAB psi^I_A=0`, i.e. representation-irreducibility,
# makes them depend on the other psi.
#
# The Gammas merely shuffle and sign-flip terms, so for much
# of the work, we can get away with using small integer numbers
# 1, 2, 3, ... to represent these basic fermionic variables,
# and then -1, -2, ... to represent their negatives.
# This allows for extremely compact in-memory representations
# where we never have to handle symbol-pointers(!)
#
# We use index 0 to indicate an 'invalid psi'.

# In name-tags, we let spinorial index-counting start at 1,
# to properly align with common conventions in the literature.
PSI_NAME_BY_TAG = ('BAD',) + tuple(
    f'psi_v{v:02d}s{s+1:02d}' for v, s in iprod(range1_11, range32))

# Inverse mapping of PSI_NAME_BY_TAG.
PSI_TAG_BY_NAME = {name: tag for tag, name in enumerate(PSI_NAME_BY_TAG)}

# All the psis, as a [11, 32]-array with invalid [0, :] entries.
PSIS = numpy.concatenate(
    [numpy.zeros((1, 32)),
     numpy.arange(1, 32*10+1).reshape(10, 32)], axis=0).astype(numpy.int16)

# "QBAR" = "Quasi-Barred"; we left-index-contract with Gamma0,
# but do not include a factor i. This allows us to keep the calculation real.
# Overall, in a quartic term of the form
# <bar-psi | gamma^... | psi> <bar-psi | gamma^... | psi>, the two omitted
# factors i introduce a factor -1 on the coefficient.
QBAR_PSIS = numpy.einsum('aB,BA->aA', PSIS, G11[0])


class GammaPsiTable:
  """Table of all |Gamma^..._AB | psi^I_B> expressions.

  Fermionic bilinear expressions <psi_... | Gamma^... | psi_...>
  are formed by taking scalar products of <psi_...| with cached
  lazily-computed | Gamma^... | psi_...>. Class instances handle
  the caching.

  Attributes:
    gamma11_family: The Gamma11Family instance used for this table.
    psis: The psi-table used for this family.
    psi_names: Mapping from psi-tag to variable name.
    qbar_psis: The "quasi-barred" psis.
    phis: The phi_IA := Gamma_IAB psi_IB
    qbar_phis: The "quasi-barred" phis.
  """

  def __init__(self, *,
               gamma11_family=None,
               psis=None,
               psi_names=None):
    """Initializes the instance.

    Args:
      gamma11_family: Optional Gamma11Family instance to use.
        If not provided, will default-construct one.
      psis: Optional [11,32] array-like, tags of the psi-definitions
        to use. The psi[0, :] generally will be 'invalid symbol' tags.
        If not provided, will use PSIS.
      psi_names: Optional psi-name-by-tag mapping.
        If not given, PSI_NAME_BY_TAG is used. Must be in lexicographic order.
    """
    super().__init__()  # Chain __init__() via Python MRO.
    self.gamma11_family = (gamma11_family if gamma11_family is not None
                           else Gamma11Family())
    gamma_IAB = self.gamma11_family.gamma_IAB
    self.psis = (numpy.array(psis, dtype=numpy.int16) if psis is not None
                 else PSIS)
    self.psi_names = psi_names or PSI_NAME_BY_TAG
    self.qbar_psis = numpy.einsum('aB,BA->aA',
                                  self.psis, gamma_IAB[0]).astype(numpy.int16)
    self.phis = numpy.einsum('aAB,aB->aA',
                             gamma_IAB[1:],
                             self.psis[1:, :]).astype(numpy.int16)
    self.qbar_phis = numpy.einsum('aB,BA->aA',
                                  self.phis, gamma_IAB[0]).astype(numpy.int16)
    self._gamma_psi_by_gamma_cache_tag = {}
    self._sizes_initialized = set()
    # Speedup hacks.
    # pylint:disable=g-missing-from-attributes
    self._ensure_have_size = self._ensure_have_size
    self.get = self.get

  def _ensure_have_size(self, num_vindices):
    """Ensures we have gamma^... | psi^a> table entries for given arity.

    Ensures that the table has an entry, keyed by the vector-indices
    on the Gamma-matrix, for gamma^i0,i1,...,ik_AB psi^J_B.

    Args:
      num_vindices: Number of vector-indices on the Gamma.
    """
    if not 0 <= num_vindices <= 11 or num_vindices in self._sizes_initialized:
      return
    gamma_rapply = self.gamma11_family.apply_right
    gamma_psi_by_gamma_cache_tag = self._gamma_psi_by_gamma_cache_tag
    psis = self.psis
    for gamma_indices in itertools.combinations(range(11), num_vindices):
      gamma_psi = numpy.zeros([11, 32], dtype=numpy.int16)
      for i in range(11):
        psi_i = psis[i]
        permuted, gamma_signs = gamma_rapply(gamma_indices, psi_i)
        # We can merge the signs into the tags, since tags start at 1.
        gamma_psi[i, :] = permuted * gamma_signs
      gamma_psi_by_gamma_cache_tag[_gamma_cache_tag(gamma_indices)] = (
          numpy.stack([gamma_psi, -gamma_psi], axis=0))
    self._sizes_initialized.add(num_vindices)

  def get(self, vindices, sign):
    """Returns `None` if `vindices` are invalid, else gamma x psi.

    Args:
      vindices: Vector-indices i0,...,ik on the gamma^i0,i1,...,ik_AB.
      sign: +1 or -1, overall sign to multiply the result with.

    Returns:
      A [11, 32]-numpy.ndarray, indexed `[J, B]`, holding the psi-tags
      (where a negative sign on the tag indicates a coefficient of -1) of
      `gamma^i0,i1,...,ik_AB psi^J_B`.
    """
    self._ensure_have_size(len(vindices))
    gamma_psi_by_gamma_cache_tag = self._gamma_psi_by_gamma_cache_tag
    # First, we have to see whether gamma_vstring is valid.
    gamma_vindices = numpy.asarray(vindices, dtype=numpy.uint8)
    sorted_vindices = gamma_vindices.copy()
    sorted_vindices.sort()
    if (sorted_vindices[1:] == sorted_vindices[:-1]).any():
      # There is a duplicate vector-index.
      return None
    if len(gamma_vindices) == 0:
      # Special case is needed since .argmax() will not handle empty sequences.
      perm_sign = +1
    else:
      # Here, we are using fast C-level NumPy code for an "O(N^2), but N<=11"
      # problem, which is a better idea than doing an O(N) approach in Python.
      perm = (gamma_vindices[:, numpy.newaxis] ==
              sorted_vindices[numpy.newaxis, :]).argmax(axis=0)
      perm_sign = permutation_sign_d(perm)
    entry = gamma_psi_by_gamma_cache_tag[sorted_vindices.data.tobytes()]
    return entry[int(sign != perm_sign)]


# === Collecting quartic terms ===


class PsiSubstitutionMapping:
  """Substitution-mapping for fundamental psi-variables."""

  def __init__(self, *,
               new_variables,
               psi_variables,
               ranking_polynomial,
               rules):
    """Initializes the instance.

    Args:
      new_variables: Sequence of string-names of the new variables
        to substitute to, in lexicographic order.
      psi_variables: Sequence of names of all the original
        psi-variables, in lexicographic order.
      ranking_polynomial: integer arraylike, coefficient data for
        the polynomial that converts the 0-based indices in ordered
        4-tuple to the lexicographic rank of the 4-tuple.
        If the tuple is `(a,b,c,d)`, and these coefficients are `r`,
        then the rank is computed as:
        (r[1]+d*r[2]+c*r[3]+c**2*r[4]+b*r[5]+b**2+r[6]+b**3*r[7]
         +a*r[8]+a**2*r[9]+a**3*r[10]+a**4*r[11])/r[0].
      rules: substitution-rules, as mapping from a psi-variable name
        (from `psi_variables`) to a sequence of
        `(coefficient, new_variable_name)`. None of the coefficients
        must be zero.
    """
    # We create copies of the data that the instance will own.
    # This is especially important for the ranking polynomial,
    # which must be passed to the FFI module as a specific kind
    # of numpy.ndarray.
    self._new_variables = list(new_variables)
    self._psi_variables = list(psi_variables)
    self._ranking_polynomial = numpy.array(ranking_polynomial,
                                           dtype=numpy.int64)
    self._rules = {psi: [tuple(cv) for cv in expansion]
                   for psi, expansion in rules.items()}

  def to_ffi_form(self):
    """Maps a substitution-table to FFI-understandable form.

    This provides substitution-information in a form that admits fast
    table lookup in the compiled-code extension module.

    Returns:
      An opaque object to be consumed only internally by
      a QuarticTermFastAccumulator instance.
    """
    max_expansion_width = max(map(len, self._rules.values()))
    subs_mapping = numpy.zeros(
        [len(self._psi_variables), max_expansion_width],
        dtype=numpy.int16)
    subs_coeffs = numpy.zeros(
        [len(self._psi_variables), max_expansion_width],
        dtype=numpy.float32)
    new_index_by_name = {name: k for k, name in
                         enumerate(self._new_variables)}
    psi_index_by_name = {name: k for k, name in
                         enumerate(self._psi_variables)}
    for psi_name, expansion in self._rules.items():
      psi_index = psi_index_by_name[psi_name]
      for num_expansion, (coeff, new_var_name) in enumerate(expansion):
        if coeff == 0:
          raise ValueError(
              'Invalid substitution-coefficient zero encountered.')
        subs_mapping[psi_index, num_expansion] = new_index_by_name[new_var_name]
        subs_coeffs[psi_index, num_expansion] = coeff
    return (self._ranking_polynomial,
            self._new_variables,
            (subs_mapping, subs_coeffs))


class QuarticTermAccumulator(abc.ABC):
  """An accumulator for quartic fermionic terms."""

  def __init__(self, *,
               gamma_psi_table=None,
               add_gamma0_ii_factor=True
               ):
    """Initializes the instance.

    Args:
      gamma_psi_table: The optional GammaPsiTable instance to use.
        If `None`, will use GammaPsiTable().
      add_gamma0_ii_factor: Whether to flip the sign on every coefficient
        due to having barred psi only with Gamma0 and not i*Gamma0.
    """
    super().__init__()  # Chain __init__() via Python MRO.
    self._gamma_psi_table = (gamma_psi_table if gamma_psi_table is not None
                             else GammaPsiTable())
    self._add_gamma0_ii_factor = add_gamma0_ii_factor

  def process_q4(self,
                 b0_le, b0_ri, b1_le, b1_ri,
                 b0v_psi_le, b0v_psi_ri, b1v_psi_le, b1v_psi_ri,
                 coefficient):
    """Processes a particular <psi|psi><psi|psi> combination.

    Args:
      b0_le: [11, 32]-numpy.int16 machine-order buffer with <psi| data for
        the first bilinear.
      b0_ri: [11, 32]-numpy.int16 machine-order buffer with |gamma |phi> data
        for the first bilinear.
      b1_le: [11, 32]-numpy.int16 machine-order buffer with <psi| data for
        the second bilinear.
      b1_ri: [11, 32]-numpy.int16 machine-order buffer with |gamma |phi> data
        for the second bilinear.
      b0v_psi_le: int vector index on the first bilinear's <psi|.
        If 0, this indicates "sum over 1..10".
        Otherwise, take data from b0_le[0, :].
      b0v_psi_ri: int vector index on the first bilinear's |gamma | psi>.
      b1v_psi_le: int vector index on the second bilinear's <psi|.
        If 0, this indicates "sum over 1..10".
        Otherwise, take data from b1_le[0, :].
      b1v_psi_ri: int vector index on the second bilinear's |gamma | psi>.
      coefficient: The overall multiplicative factor on terms.
    """
    # Base class implementation is a "null device" that does not
    # accumulate anything.
    del b0_le, b0_ri, b1_le, b1_ri
    del b0v_psi_le, b0v_psi_ri, b1v_psi_le, b1v_psi_ri
    del coefficient

  def collect(self, *,
              term_factor,
              ci):
    """Collects terms quartic in fermionic variables.

    This will repeatedly call `self.process_q4()` for fast collection of
    contributions.

    Args:
      term_factor: The overall scaling factor for this term.
      ci: Iterable of tuples of the form:
        `(coeff, (psi_index, vindices_first_gamma, psi_index),
                 (psi_index, vindices_second_gamma, psi_index))`
        Will get iterated over once.
    """
    effective_term_factor = (
        -term_factor if self._add_gamma0_ii_factor else term_factor)
    gamma_psi_table = self._gamma_psi_table
    qbar_psis = gamma_psi_table.qbar_psis
    process_q4 = self.process_q4
    # By-bilinear data-buffers.
    # Indexed [num_bilinear, is_right, vector_index, spinor_index].
    # Left-side and right-side buffers are used slightly differently:
    # For [:, is_right=0, :, :] left-side buffers, a vector-index of 0
    # does not select a specific spinor, but indicates
    # "iterate over 1..10 and sum".
    bb_buffers_compact = numpy.zeros([2, 2, 11, 32], dtype=numpy.int16)
    # The gamma_psi that currently live in the right-side-of-bilinear buffers,
    # tagged by vector-index strings.
    gamma_psi_have = [None, None]
    # Likewise, the data that is currently in the left-side-of-bilinear buffers.
    psi_bar_have = [None, None]
    # We only put the psi-bar data into the buffer if actually needed.
    initialized_barred = False
    # Individual views on the above compact data.
    # For more convenient data-loading.
    bb_buffers = list(bb_buffers_compact.reshape(4, 11, 32))
    # b0v_ / b1v_ are 0th / 1st bilinear vector-indices.
    for (combinatorial_factor,
         (b0v_psi_le, b0v_gammas, b0v_psi_ri),
         (b1v_psi_le, b1v_gammas, b1v_psi_ri)) in ci:
      if combinatorial_factor == 0:
        continue
      if not initialized_barred and (b0v_psi_le == 0 or b1v_psi_le == 0):
        bb_buffers_compact[:, 0, 1:, :] = (
            gamma_psi_table.qbar_phis[numpy.newaxis, :, :])
        initialized_barred = True
      # We only update the right-side-of-bilinear buffers if the
      # corresponding gamma changed since the previous iteration.
      if b0v_psi_le != 0 and b0v_psi_le != psi_bar_have[0]:
        bb_buffers[0][0, :] = qbar_psis[b0v_psi_le]
        psi_bar_have[0] = b0v_psi_le
      if b1v_psi_le != 0 and b1v_psi_le != psi_bar_have[1]:
        bb_buffers[2][0, :] = qbar_psis[b1v_psi_le]
        psi_bar_have[1] = b1v_psi_le
      # We correspondingly update right-side-of-bilinear
      # buffers as-needed.
      if b0v_gammas != gamma_psi_have[0]:
        gamma_psi = gamma_psi_table.get(b0v_gammas, 1)
        if gamma_psi is None:
          # Index-duplicate makes the gamma zero.
          continue
        bb_buffers[1][:, :] = gamma_psi
        gamma_psi_have[0] = b0v_gammas
      if b1v_gammas != gamma_psi_have[1]:
        gamma_psi = gamma_psi_table.get(b1v_gammas, 1)
        if gamma_psi is None:
          # Index-duplicate makes the gamma zero.
          continue
        bb_buffers[3][:, :] = gamma_psi
        gamma_psi_have[1] = b1v_gammas
      # Finally, generate and collect contributions.
      process_q4(
          *bb_buffers,
          b0v_psi_le, b0v_psi_ri,
          b1v_psi_le, b1v_psi_ri,
          effective_term_factor * combinatorial_factor)

  def collected(self, scale_by=1.0):
    """Yields collected contributions.

    Args:
      scale_by: Factor to scale coefficients with.

    Yields:
      Tuple `(coefficient, (var0_name, var1_name, var2_name var3_name))`.
    """
    del scale_by  # Unused by base class method.
    # Base class ("null device") implementation yields nothing, but still
    # must make the call evaluate to an iterator.
    yield from iter(())

  def reset(self):
    """Resets the accumulator."""
    # Base class ("null device") implementation does nothing.


class QuarticTermDictAccumulator(QuarticTermAccumulator):
  """Basic quartic-term accumulator that uses a dictionary."""
  # Note: This basic accumulator is mostly to illustrate the principle
  # with some simple and straightforward code.
  # It does not support linear substitution of the fundamental
  # fermionic variables.

  def __init__(self, *,
               gamma_psi_table=None,
               add_gamma0_ii_factor=True
               ):
    """Initializes the instance.

    Args:
      gamma_psi_table: The optional GammaPsiTable instance to use.
        If `None`, will use GammaPsiTable().
      add_gamma0_ii_factor: Whether to flip the sign on every coefficient
        due to having barred psi only with Gamma0 and not i*Gamma0.
    """
    super().__init__(gamma_psi_table=gamma_psi_table,
                     add_gamma0_ii_factor=add_gamma0_ii_factor)
    self._accumulator = {}

  def process_q4(self,
                 b0_le, b0_ri, b1_le, b1_ri,
                 b0v_psi_le, b0v_psi_ri, b1v_psi_le, b1v_psi_ri,
                 coefficient):
    # Docstring gets forwarded from base class method.
    accumulator = self._accumulator
    psi_factors = numpy.zeros([4], dtype=numpy.int16)
    for b0v_le in (range1_11 if b0v_psi_le == 0 else range1):
      for b0_s in range32:
        psi_factors[0] = b0_le[b0v_le, b0_s]
        psi_factors[1] = b0_ri[b0v_psi_ri, b0_s]
        for b1v_le in (range1_11 if b1v_psi_le == 0 else range1):
          for b1_s in range32:
            psi_factors[2] = b1_le[b1v_le, b1_s]
            psi_factors[3] = b1_ri[b1v_psi_ri, b1_s]
            abs_psi_factors = abs(psi_factors)
            # Are there any duplicates?
            if (abs_psi_factors[numpy.newaxis, :] ==
                abs_psi_factors[:, numpy.newaxis]).sum() != 4:
              continue
            psi_factors_sign = (1, -1)[(psi_factors !=
                                        abs_psi_factors).sum() & 1]
            reordering_sign, psi_factors_canonicalized = (
                get_sign_and_canonicalized(abs_psi_factors))
            # This blunt and straightforward code uses tag-tuples
            # as accumulator-keys.
            acc_key = tuple(psi_factors_canonicalized)
            contrib = (coefficient if psi_factors_sign == reordering_sign
                       else -coefficient)
            new_value = contrib + accumulator.get(acc_key, 0)
            # Let us ensure that cancellations are respected properly
            # by removing the key.
            if new_value == 0:
              del accumulator[acc_key]
            else:
              accumulator[acc_key] = new_value

  def collected(self, scale_by=1.0):
    # Docstring gets forwarded from base class method.
    accumulator = self._accumulator
    psi_names = self._gamma_psi_table.psi_names
    # We straightaway face a problem here: For simplifying comparisons,
    # we would like to iterate over items in sorted order,
    # but since there may be many entries, we do not sort the .items()
    # but the .keys(), which directly reference the key-objects.
    # The .items() would use another memory-costly key-value tuple
    # per entry. The cost is an extra key-lookup.
    keys = sorted(accumulator)
    for key in keys:
      coeff = accumulator[key]
      yield coeff * scale_by, tuple(psi_names[tag] for tag in key)

  def reset(self):
    # Docstring gets forwarded from base class method.
    self._accumulator.clear()


class QuarticTermFastAccumulator(QuarticTermAccumulator):
  """A fast accumulator for quartic fermionic terms.

  This accumulator delegates the generation of four-fermion
  contributions to a fast compiled extension module.
  """

  def __init__(self, *,
               gamma_psi_table=None,
               add_gamma0_ii_factor=True,
               psi_substitutions=None,
               ):
    """Initializes the instance.

    Note: This accumulator supports linear substitutions on psi-variables.

    Args:
      gamma_psi_table: The optional GammaPsiTable instance to use.
        If `None`, will use GammaPsiTable().
      add_gamma0_ii_factor: Whether to flip the sign on every coefficient
        due to having barred psi only with Gamma0 and not i*Gamma0.
      psi_substitutions: An optional `PsiSubstitutionMapping` instance that
        describes variable-mapping. If `None`, no linear substitution
        the original psi-variables is performed.
    """
    super().__init__(gamma_psi_table=gamma_psi_table,
                     add_gamma0_ii_factor=add_gamma0_ii_factor)
    if psi_substitutions is None:
      arr_ranking_polynomial = RANKING_COEFFS_320_4
      var_names = PSI_NAME_BY_TAG[1:]
      substitution_info = None
    else:
      arr_ranking_polynomial, var_names, substitution_info = (
          psi_substitutions.to_ffi_form())
    self._psi_substitutions = psi_substitutions
    self._var_names = var_names
    self._accumulator = accumulator = numpy.zeros(
        math.comb(len(var_names), 4), dtype=numpy.float32)
    # The foreign function interface implementation:
    ffi_collect_quartic_terms = _psi4_accum.collect_quartic_terms
    #
    # For the fast accumulator, `process_q4` is provided as a callable
    # instance-attribute that shadows the base class method.
    def process_q4(b0_le, b0_ri, b1_le, b1_ri,
                   b0v_psi_le, b0v_psi_ri, b1v_psi_le, b1v_psi_ri,
                   coefficient):
      success = ffi_collect_quartic_terms(
          substitution_info,
          arr_ranking_polynomial,
          accumulator,
          b0_le, b0_ri, b1_le, b1_ri,
          b0v_psi_le, b0v_psi_ri, b1v_psi_le, b1v_psi_ri,
          coefficient)
      if not success:
        raise ValueError(
            'Compiled code failed to collect terms, due to bad input data.')
    self.process_q4 = process_q4

  def collected(self, scale_by=1.0):
    # Docstring gets forwarded from base class method.
    accumulator = self._accumulator
    var_names = self._var_names
    for n, indices in enumerate(
        itertools.combinations(range(len(self._var_names)), 4)):
      coeff = accumulator[n]
      if coeff:
        yield coeff * scale_by, tuple(var_names[index] for index in indices)

  def reset(self):
    # Docstring gets forwarded from base class method.
    self._accumulator *= 0

  def save(self, filepath):
    """Saves the accumulator-state to filesystem.

    Accumulator will be dumped as a numpy file via numpy.save().

    Args:
      filepath: output string path.
    """
    # Let's be explicit about not allowing pickling.
    numpy.save(filepath, self._accumulator, allow_pickle=False)

  def load(self, filepath):
    """Loads the accumulator-state from filesystem.

    Accumulator will be loaded as a numpy file via numpy.load().

    Args:
      filepath: output string path.
    """
    self._accumulator = numpy.load(filepath, allow_pickle=False)
