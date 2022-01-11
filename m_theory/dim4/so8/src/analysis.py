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

"""SO(8) gauged D=4 supergravity.

Usage: python3 -i -m dim4.so8.src.analysis
"""


# Naming deviates from PEP-8 conventions where this makes mathematics easier
# to read. Also, local variables may name-match module-global definitions.
# pylint:disable=invalid-name
# pylint:disable=redefined-outer-name

import collections
import itertools
import os

from dim4.generic import a123
from m_theory_lib import algebra
from m_theory_lib import m_util as mu
from m_theory_lib import supergravity

import numpy
import tensorflow as tf


### Supergravity ###


class SO8_SUGRA(supergravity.SUGRA):
  """D=4 SO(8) Supergravity.

  In addition to the base class attributes, this class adds...:

  Attributes:
    e7: The e7 algebra that was used to define this supergravity.
  """

  signature = supergravity.SUGRASignature(
      name='SO8',
      gauge_algebra_name='so(8)',
      dim=4,
      generator_scaling=+1,
      dim_scalar_manifold=70,
      num_model_params=1,  # The 'dyonic angle'.
      scalar_masses_dV_from_right=False,
      scalar_masses_factor=36.0,
      gravitino_masses_factor=6.0,
      fermion_masses_factor=6.0,
      vector_masses_factor=6.0,
      num_spurious_vector_masses=28)

  def __init__(self,
               use_dwn_stationarity=True,
               e7=algebra.g.e7,
               squash_stationarity_tf_func=tf.math.asinh,
               **kwargs):
    """Initializes the instance.

    Args:
      use_dwn_stationarity: Whether to use the de A1/A2 formula from the
        de Wit-Nicolai 'SO(8) Supergravity' paper to compute the
        stationarity-violation (rather than taking the naive gradient-squared).
      e7: The e7 algebra to use.
      squash_stationarity_tf_func: Optional 'squashing' function that is used
        to squash the stationarity-violation ([]-tf.Tensor -> []-tf.Tensor).
      **kwargs: keyword parameters to be passed on to superclass __init__().
    """
    super().__init__(e7.t_a_ij_kl,
                     squash_stationarity_tf_func=squash_stationarity_tf_func,
                     **kwargs)
    self._use_dwn_stationarity = use_dwn_stationarity
    self.e7 = e7
    self._tc_28_8_8 = tf.constant(
        e7.su8.m_28_8_8.astype(numpy.complex128),
        dtype=tf.complex128)

  def _expand_ijkl(self, t_ab):
    """Index-expands 28, 28 -> [8, 8] [8, 8]."""
    return 0.5 * tf.einsum(
        'ijB,BIJ->ijIJ',
        tf.einsum('AB,Aij->ijB', t_ab, self._tc_28_8_8),
        self._tc_28_8_8)

  def _canonicalize_equilibrium_sc(self, v70, diagonalize_8x8s=True,
                                   rng=None, verbose=True):
    """Simplifies a location on the scalar manifold by rotation."""
    if rng is None:
      rng = numpy.random.RandomState()
    m8x8s = mu.nsum('Aij,A->ij', self.e7.su8.m_35_8_8.real, v70[:35])
    m8x8c = mu.nsum('Aij,A->ij', self.e7.su8.m_35_8_8.real, v70[35:])
    rot = self.e7.spin8.get_diagonalizing_rotation(
        m8x8s if diagonalize_8x8s else m8x8c)
    decomposed_rot = mu.product_decompose_rotation(rot)
    resynthesized_rot = mu.resynthesize_rotation_for_rep(
        8, 8, decomposed_rot, 'ab,->ab', numpy.ones([]))
    if not numpy.allclose(rot, resynthesized_rot, rtol=1e-3, atol=1e-5):
      raise ValueError(
          'Resynthesized rotation does not match original rotation.')
    generator_mapping_spec = 'sS,sScC->cC' if diagonalize_8x8s else 'cC,sScC->sS'
    rep_action = 0.25 * self.e7.spin8.gamma_sscc
    rot_other_rep = mu.resynthesize_rotation_for_rep(
        8, 8, decomposed_rot, generator_mapping_spec, rep_action)
    (rot_s, rot_c) = ((rot, rot_other_rep) if diagonalize_8x8s
                      else (rot_other_rep, rot))
    canon_m8x8s = rot_s.T @ m8x8s @ rot_s
    canon_m8x8c = rot_c.T @ m8x8c @ rot_c
    if diagonalize_8x8s:
      gens_postdiag = mu.get_generators_for_post_diagonalization_reduction(
          numpy.diag(canon_m8x8s), 'gsS,sScC->gcC', self.e7.spin8.gamma_sscc)
    else:
      gens_postdiag = mu.get_generators_for_post_diagonalization_reduction(
          numpy.diag(canon_m8x8c), 'gcC,sScC->gsS', self.e7.spin8.gamma_sscc)
    tc_rot_gens = mu.tff64(gens_postdiag)
    tc_8x8s = mu.tff64(canon_m8x8s)
    tc_8x8c = mu.tff64(canon_m8x8c)
    @tf.function
    def tf_rotated_8x8(t_rot_params):
      t_rot = mu.tf_expm(
          tf.einsum('gab,g->ab', tc_rot_gens, t_rot_params))
      if diagonalize_8x8s:
        tc_rotated_8x8 = tf.linalg.matmul(
            t_rot @ tc_8x8c, t_rot, transpose_b=True)
      else:
        tc_rotated_8x8 = tf.linalg.matmul(
            t_rot @ tc_8x8s, t_rot, transpose_b=True)
      return tc_rotated_8x8
    @tf.function
    def tf_loss(t_rot_params):
      t_8x8 = tf_rotated_8x8(t_rot_params)
      ret = tf.reduce_sum(tf.abs(t_8x8))
      return ret
    if gens_postdiag.shape[0] == 0:
      return self.e7.v70_from_35s35c(canon_m8x8s, canon_m8x8c)
    _, opt_rot_params = mu.tf_minimize_v2(
        tf_loss,
        rng.normal(scale=1.0, size=gens_postdiag.shape[0]),
        default_gtol=1e-14)
    opt_8x8 = tf_rotated_8x8(mu.tff64(opt_rot_params)).numpy()
    if diagonalize_8x8s:
      return self.e7.v70_from_35s35c(canon_m8x8s, opt_8x8)
    else:
      return self.e7.v70_from_35s35c(opt_8x8, canon_m8x8c)

  def canonicalize_equilibrium(self, v70, **kwargs):
    """Simplifies a location on the scalar manifold by rotation."""
    v70 = numpy.asarray(v70)
    canon_35s = self._canonicalize_equilibrium_sc(v70, **kwargs)
    canon_35c = self._canonicalize_equilibrium_sc(
        v70, diagonalize_8x8s=False, **kwargs)
    return min([canon_35s, canon_35c], key=lambda xs: (abs(xs) > 1e-5).sum())

  def tf_T(self, t_vielbein, t_omega=None):
    """Computes the SO(8) T-tensor."""
    t_omega = mu.tff64(0.0) if t_omega is None else t_omega
    t_u_ijIJ = self._expand_ijkl(t_vielbein[:28, :28])
    t_u_klKL = tf.math.conj(t_u_ijIJ)
    t_v_ijKL = self._expand_ijkl(t_vielbein[:28, 28:])
    t_v_klIJ = tf.math.conj(t_v_ijKL)
    t_cw = tf.math.cos(t_omega)
    t_sw = tf.math.sin(t_omega)
    tc_exp_w = tf.complex(t_cw, t_sw)
    tc_exp_nw = tf.complex(t_cw, -t_sw)
    t_uv = tc_exp_nw * t_u_klKL + tc_exp_w * t_v_klIJ
    t_uuvv = (
        tf.einsum('lmJK,kmKI->lkIJ', t_u_ijIJ, t_u_klKL) -
        tf.einsum('lmJK,kmKI->lkIJ', t_v_ijKL, t_v_klIJ))
    return tf.einsum('ijIJ,lkIJ->lkij', t_uv, t_uuvv)

  def tf_A123(self, t_T, want_A1=True, want_A2=True, want_A3=True):
    """See base class."""
    t_A1 = t_A2 = t_A3 = None
    if want_A1:
      t_A1 = mu.tfc128(-4 / 21) * tf.einsum('mijm->ij', t_T)
    if want_A2 or want_A3:
      t_A2 = mu.tfc128(-4 / (3 * 3)) * (
          # Antisymmetrize in last 3 indices, but using antisymmetry in last 2.
          # Note factor 1/3 above (in -4/(3*3) rather than -4/3).
          t_T + tf.einsum('lijk->ljki', t_T) + tf.einsum('lijk->lkij', t_T))
    if want_A3:
      t_A3 = a123.tf_A3_from_A2(t_A2)
    return t_A1, t_A2, t_A3

  def tf_ext_sugra_tensors(self, t_scalars, with_stationarity=True, **kwargs):
    """See base class."""
    if not self._use_dwn_stationarity or not with_stationarity:
      # If we are not using de Wit-Nicolai stationarity, or are not interested
      # in stationarity, just delegate to the base class method.
      return super().tf_ext_sugra_tensors(t_scalars,
                                          with_stationarity=with_stationarity,
                                          **kwargs)
    sugra_tensors = super().tf_ext_sugra_tensors(
        t_scalars, with_stationarity=False, **kwargs)
    *_, t_A1, t_A2, _ = sugra_tensors
    t_dwn_stationarity = self.dwn_stationarity(t_A1, t_A2, **kwargs)
    tf_squash = self._squash_stationarity_tf_func
    t_squashed_stationarity = (
        t_dwn_stationarity if tf_squash is None
        else tf_squash(t_dwn_stationarity))
    return sugra_tensors[:-1] + (t_squashed_stationarity,)

  def dwn_stationarity(self, t_A1, t_A2, **kwargs):
    """Computes stationarity-violation 'in the local frame'."""
    # Attention: The stationarity that we get from
    # self.position_and_stationarity() will typically be squashed
    # by the default asinh-squash!
    return a123.tf_dwn_stationarity(t_A1, t_A2)

  def show_position_tex(self, position, digits=6):
    """Returns a text-string that shows the position."""
    m35s = mu.nsum('Iij,I->ij', self.e7.su8.m_35_8_8, position[:35])
    m35c = mu.nsum('Iij,I->ij', self.e7.su8.m_35_8_8, position[35:70])
    fmt_num = lambda x: f'{x:.0{digits}f}'
    def dotified_m(sc, ij, is_positive):
      sign_str = '' if is_positive else '-'
      indices = (r'\dot{}\dot{}' if sc else '{}{}').format(*ij)
      return '%sM_{%s}' % (sign_str, indices)
    pos_sign_by_text = collections.defaultdict(list)
    for sc, m35 in ((0, m35s), (1, m35c)):
      for ij in itertools.product(range(8), range(8)):
        if not ij[0] <= ij[1]:
          # We only report entries of the upper-triangular part of these
          # symmetric matrices.
          continue
        num = m35[ij]
        abs_num_str = fmt_num(abs(num))
        if float(abs_num_str) == 0:  # Skip zeroes.
          continue
        pos_sign_by_text[abs_num_str].append((sc, ij, num > 0))
    groups = sorted(
        [(sorted(locations), abs_num_str)
         for abs_num_str, locations in pos_sign_by_text.items()])
    tex_pieces = []
    for raw_locations, abs_num_str in groups:
      is_plus_first = raw_locations[0][-1]
      if is_plus_first:
        locations = raw_locations
        num_str = abs_num_str
      else:
        # 'is_plus' gets replaced by relative sign w.r.t. first such entry.
        locations = [(sc, ij, not is_plus) for sc, ij, is_plus in raw_locations]
        num_str = '-' + abs_num_str
      tex_pieces.append(
          r'$\scriptstyle %s\approx%s$' % (
              r'{\approx}'.join(dotified_m(*sc_ij_relative_sign)
                                for sc_ij_relative_sign in locations),
              num_str))
    return (r'{\begin{minipage}[t]{10cm}'
            r'\begin{flushleft}%s\end{flushleft}\end{minipage}}\\' %
            ', '.join(tex_pieces))

  def tf_sugra_tensors_from_vielbein(self, t_vielbein, t_omega=None):
    """See base class."""
    t_T = self.tf_T(t_vielbein, t_omega=t_omega)
    t_A1, t_A2, _ = self.tf_A123(t_T, want_A3=False)
    t_potential_A1 = mu.tff64(-3 / 4) * tf.math.real(
        tf.einsum('ij,ij->', t_A1, tf.math.conj(t_A1)))
    t_potential_A2 = mu.tff64(1 / 24) * tf.math.real(
        tf.einsum('ijkl,ijkl->', t_A2, tf.math.conj(t_A2)))
    t_potential = t_potential_A1 + t_potential_A2
    return t_potential, t_T, t_A1, t_A2

  def tf_fermion_massmatrix(self, ts_A123, t_potential):
    """See base class."""
    *_, t_A3 = ts_A123
    return a123.tf_fermion_massmatrix(
        t_A3, t_potential,
        mu.tfc128(-self.signature.fermion_masses_factor))

  def tf_vector_massmatrix(self, ts_A123, t_potential):
    """See base class."""
    _, t_A2, _ = ts_A123
    return a123.tf_vector_massmatrix(
        t_A2, t_potential,
        mu.tfc128(-self.signature.vector_masses_factor))

  def get_residual_symmetry(self, v70, **kwargs):
    """See base class."""
    del kwargs  # Unused.
    unbroken_gg, d_unbroken_gg, u1s = mu.decompose_residual_symmetry(
        v70,
        self.e7.f_abC,
        numpy.pad(numpy.eye(28), [(0, 0), (105, 0)]))
    # TODO(tfish): Do we actually need the compact-group f_abC here?
    rank = mu.get_lie_algebra_rank(unbroken_gg, self.e7.f_abC)
    return supergravity.ResidualSymmetry(
        rank=rank,
        all_gens=unbroken_gg,
        semisimple_subgroup_gens=d_unbroken_gg,
        u1_gens=u1s)

  def show_position_text(self, position):
    """Returns a text-string that shows the position."""
    m288 = mu.nsum('a,axij->xij', position, self.e7.v70_as_sc8x8)
    m8x8s = m288[0].round(5)
    m8x8c = m288[1].round(5)
    def fmt8x8(m):
      # Pylint wrongly complains about `row` not being defined.
      # pylint: disable=undefined-loop-variable
      return '\n'.join(
          ', '.join('%+.5f' % x if x else ' 0      ' for x in row)
          for row in m)
      # pylint: enable=undefined-loop-variable
    return (f'=== 8x8s ===\n{fmt8x8(m8x8s)}\n'
            f'=== 8x8c ===\n{fmt8x8(m8x8c)}\n')

  def v70o_goldstone_basis_and_projector(self, v70o, ev_threshold=1e-5):
    """Computes a basis and a projector onto the 'goldstone directions'.

    Given a vector of 70 scalar field parameters in the orthonormal basis,
    determines a basis and projector for the subspace of directions that
    we get by applying so(8) generators to the vector (again, referring
    to the orthonormal basis.)

    Args:
      v70o: Optional [70]-numpy.ndarray, the scalar field parameters
        in the orthonormal basis.
      ev_threshold: Threshold for SVD singular values.

    Returns:
      A tuple (dim_goldstone_basis, basis, goldstone_projector)
      that gives us the dimensionality D == dim_goldstone_basis
      of the vector space V spanned by applying so(8) generators
      to the input vector, plus a [70, 70]-array B that provides
      an orthonormal basis for the scalars where B[:, :D] is an
      orthonormal basis of the D-dimensional subspace V, plus
      a [70, 70]-projector matrix that performs orthogonal projection
      onto V.
    """
    e7 = self.e7
    so8_rotated_v70o = mu.nsum('abC,b->Ca',
                               e7.fo_abC[105:, :70, :70], v70o)
    svd_so8_rot_u, svd_so8_rot_s, svd_so8_rot_vh = (
        numpy.linalg.svd(so8_rotated_v70o, full_matrices=True))
    del svd_so8_rot_vh  # Unused, named for documentation only.
    num_goldstone_directions = (svd_so8_rot_s > ev_threshold).sum()
    goldstone_directions = svd_so8_rot_u[:, :num_goldstone_directions]
    proj_goldstone = goldstone_directions.dot(goldstone_directions.T)
    return num_goldstone_directions, svd_so8_rot_u, proj_goldstone

  def get_subspace_aligner(self, target_subspace_an, rcond=1e-10):
    """Returns a closure that aligns a scalar vector with a target space."""
    target_subspace_an = numpy.asarray(target_subspace_an)
    if target_subspace_an.shape[0] != 70 or len(target_subspace_an.shape) != 2:
      raise ValueError(
          'Target subspace must be a [70, D]-array, '
          f'shape is: {target_subspace_an.shape}')
    tc_f_abC = mu.tff64(self.e7.f_abC)
    v70o_target_subspace_an = mu.nsum(
        'an,Aa->An', target_subspace_an, self.e7.v70o_from_v70)
    svd_u, svd_s, svd_vh = numpy.linalg.svd(v70o_target_subspace_an,
                                            full_matrices=False)
    del svd_vh  # Unused, named for documentation purposes only.
    v70o_target_subspace_an_basis = svd_u[:, svd_s > rcond]
    tc_v70o_proj_complement = mu.tff64(
        numpy.eye(70) -
        v70o_target_subspace_an_basis.dot(v70o_target_subspace_an_basis.T))
    tc_v70o_from_v70 = mu.tff64(self.e7.v70o_from_v70)
    #
    def f_do_align(v70, **kwargs):
      tc_v70 = mu.tff64(v70)
      def tf_loss(t_rot_params):
        t_gen_so8 = tf.einsum('abC,a->Cb', tc_f_abC[-28:, :70, :70],
                              t_rot_params)
        t_rot_so8 = tf.linalg.expm(t_gen_so8)
        t_rotated = tf.einsum('ab,b->a', t_rot_so8, tc_v70)
        t_deviation = tf.einsum(
            'a,Aa,BA->B', t_rotated, tc_v70o_from_v70, tc_v70o_proj_complement)
        return tf.reduce_sum(tf.math.square(t_deviation))
      return mu.tf_minimize_v2(tf_loss, v70, **kwargs)
    #
    return f_do_align


def known_solutions(potential_min, potential_max, csv_path=None):
  """Returns [potential, stationarity, *coords] rows for known solutions."""
  if csv_path is None:
    csv_path = os.path.join(os.path.dirname(__file__),
                            '../equilibria/SO8_SOLUTIONS.csv')
  return [numpy.asarray(row) for row in mu.csv_numdata(csv_path)
          if potential_min <= row[0] <= potential_max]


### Demo Functions ###

# demo_*() functions may come and go, and are generally not tested.
# They show some general use patterns of the above code and provide good
# starting points for exploring the capabilities of this code. They are
# not considered part of the stable API, however.


def demo_show_physics_so7():
  """Text-prints the physics of the known 'SO(7)+' solution."""
  sugra = SO8_SUGRA()
  rows = list(itertools.islice(mu.csv_numdata(
      'dim4/so8/equilibria/SO8_SOLUTIONS.csv'), 10))
  # phys = sugra.get_physics(rows[4][-70:],
  #                          dict(name='some_solution'))  # SU(3)xU(1) N=2
  # phys = sugra.get_physics(rows[8][-70:], dict(name='SO(3)xSO(3) S0880733'))
  phys = sugra.get_physics(rows[1][-70:], dict(name='SO(7)+'))
  print(sugra.show_physics_text(phys))


def demo_scan_sl2x7(seed=10, scale=1.0, verbosity=''):
  """Scans for equilibria on the (SL(2)/U(1))**7 submanifold."""
  sl2x7_embedding = algebra.g.e7.sl2x7[:2][Ellipsis, :70].reshape(-1, 70)
  sugra = SO8_SUGRA()
  x0s = sugra.get_generator_x0s(scale=scale, dim=14, seed=seed)
  for n, (pot, stat, a_pos) in enumerate(
      sugra.scan(x0s=x0s, submanifold_embedding=sl2x7_embedding,
                 verbosity=verbosity)):
    print(f'{(n, pot, stat, a_pos.round(7).tolist())!r}')


def demo_show_canonicalized():
  """Discovers and shows some canonicalized solutions."""
  sugra = SO8_SUGRA()
  rng = numpy.random.RandomState(seed=0)
  for n, sol in zip(range(10),
                    sugra.scan(x0s=sugra.get_generator_x0s(scale=0.2),
                               verbosity='SF')):
    pot, stat, pos = sol
    pos_canon = sugra.canonicalize_equilibrium(pos)
    print(f'### N={n}, P={pot}, S={stat}\n pos={pos_canon.round(6)}')
    pot_good, stat_good, pos_good = sugra.find_equilibrium(
        pos, minimize_kwargs=dict(strategy='N', mdnewton_maxsteps=2))
    del pot_good, stat_good  # Unused.
    pos_canon = sugra.canonicalize_equilibrium(pos_good, rng=rng)
    phys = sugra.get_physics(pos_canon, dict(name='some_solution'))
    print(sugra.show_physics_text(phys))
