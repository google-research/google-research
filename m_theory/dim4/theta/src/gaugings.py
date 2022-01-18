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

"""Generic Theta-tensor based code for gauged D=4 Supergravities."""


# Variable names are often dictated by the need to align with the
# published literature, hence do not always follow Python conventions.
# pylint:disable=invalid-name
# pylint:disable=redefined-outer-name

import collections
import dataclasses
import gc
import itertools
import time
from typing import Tuple

from dim4.generic import a123
from m_theory_lib import algebra
from m_theory_lib import m_util as mu
from m_theory_lib import supergravity
import numpy
import scipy.linalg
import tensorflow as tf


spin8 = algebra.g.spin8
e7 = algebra.g.e7


@dataclasses.dataclass(frozen=True)
class Gauging:
  """A gauging of omega-deformed D=4 Supergravity.

  Attributes:
    theta: [56, 133]-numpy.ndarray, the (potentially approximate)
      embedding tensor.
    gg_gens: [133, 28]-numpy.ndarray, generators of the gauge group.
    gg_ranges: int-tuple of the form (0, n1, n2, 28). Here,
     gg_gens[:, 0:n1] are the compact-semisimple generators of the gauge group,
     gg_gens[:, n1:n2] are the u(1) generators of the gauge group,
       and gg_gens[:, n2:28] are the extra generators not belonging
       to these subgroups.
    cartan_subalgebra: [n1, rank]-numpy.ndarray, a basis for a
       Cartan subalgebra of gg_gens[:, 0:n1].
  """
  theta: numpy.ndarray
  gg_gens: numpy.ndarray
  gg_ranges: Tuple[int, int, int, int]
  cartan_subalgebra: numpy.ndarray

  def __repr__(self):
    """Produces a string representation of the gauging."""
    _, n1, n2, _ = self.gg_ranges
    rank = self.cartan_subalgebra.shape[-1]
    return (f'<D=4 Gauging, num_u1s={n2 - n1}, num_null={28 - n2}, '
            f'semisimple_compact="dim{n1}_rank{rank}">')

  def save(self, filename):
    """Saves an instance to filesystem."""
    numpy.savez_compressed(
        filename,
        theta=self.theta,
        gg_gens=self.gg_gens,
        gg_ranges=numpy.array(self.gg_ranges, dtype=numpy.int32),
        cartan_subalgebra=self.cartan_subalgebra)

  @classmethod
  def load(cls, filename):
    """Loads an instance from filesystem."""
    data = numpy.load(filename)
    return cls(theta=data['theta'],
               gg_gens=data['gg_gens'],
               gg_ranges=tuple(data['gg_ranges']),
               cartan_subalgebra=data['cartan_subalgebra'])

  def get_compact_semisimple_proj133o(self, e7=e7, rcond=1e-5):
    """Returns 133o-projector onto the semisimple part, plus singular values."""
    gg_compact_semisimple = self.gg_gens[:, :self.gg_ranges[1]]
    gg_css_133o = e7.v133o_from_v133.dot(gg_compact_semisimple)
    go133 = numpy.diag([1.0] * 70 + [-1.0] * 63)
    sprod = gg_css_133o.T @ go133 @ gg_css_133o
    onb, onbi = mu.get_gramian_onb(sprod, eigenvalue_threshold=rcond)
    del onbi  # Unused, for documentation only.
    gg_onb = gg_css_133o.dot(onb.T)
    # gg_onb has an upper a-index, which we lower to the right.
    proj = gg_onb @ gg_onb.T @ go133
    evstats = mu.evstats(proj)
    return proj, evstats


def _get_e7r_912_inner(filename, verbose=True, e7=e7):
  """Computes a basis for the E7 912-irrep in 56 x 133."""
  # The 912-irrep is characterized by (2.14) in
  # https://arxiv.org/pdf/1112.3345.pdf /
  # (2.12) in https://arxiv.org/pdf/0705.2101.pdf
  #
  #     (t_b t^a)_M^N Theta_N^b = -0.5 Theta_M^a
  try:
    return numpy.load(filename)['arr_0']
  except (IOError, OSError):
    if verbose:
      print('Need to compute t912.')
  t0 = time.time()
  k_ab = mu.nsum('aMN,bNM->ab', e7.t56r, e7.t56r)
  # Notation in the papers is slightly ambiguous here.
  # We want '_bM^P,_aP^N' here or '_bP^N,_aM^P'?
  # Only the former option is a correct interpretation.
  M0 = mu.nsum('_bM^P,_aP^N,^aA->^M_A^N_b',
               e7.t56r, e7.t56r,
               numpy.linalg.inv(k_ab)).reshape(56 * 133, 56 * 133)
  for k in range(56 * 133):
    M0[k, k] += 0.5
  ret = scipy.linalg.null_space(M0.real, rcond=1e-5).T
  if verbose:
    # This computation typically takes ~5 minutes.
    # We hence cache the result to disk (~50M).
    print('Computing the 912-basis took %.3f sec.' % (time.time() - t0))
  numpy.savez_compressed(filename, ret)
  # Note that this null_space is orthonormal.
  return ret


def get_e7r_912(filename, verbose=True, e7=e7):
  """Computes a basis for the e7 912-irrep."""
  # Computing the 912 intermediately requires quite some memory.
  # Wrap this up in such a way that we are GC'ing immediately after.
  ret = _get_e7r_912_inner(filename, verbose=verbose, e7=e7)
  for level in (2, 1, 0):
    gc.collect(level)
  return ret


def align_proj133o(proj133o_target, proj133o, e7=e7, debug=True):
  """Finds an E7-split-133o-rotation that aligns two 133o-projectors."""
  # 'Split rotation' here means that on an ^a-index, we first apply a
  # 'compact SU(8)' and then a 'noncompact E7' rotation.
  tc_fo_abC = mu.tff64(e7.fo_abC)
  tc_proj133o_target = mu.tff64(proj133o_target)
  tc_proj133o = mu.tff64(proj133o)
  def tf_rotated(t_133o):
    t_gen_noncompact = tf.einsum('abC,a->Cb', tc_fo_abC[:70, :, :], t_133o[:70])
    t_gen_compact = tf.einsum('abC,a->Cb', tc_fo_abC[70:, :, :], t_133o[70:])
    t_rot = (
        tf.linalg.expm(t_gen_noncompact) @ tf.linalg.expm(t_gen_compact))
    t_rot_inv = (
        tf.linalg.expm(-t_gen_compact) @ tf.linalg.expm(-t_gen_noncompact))
    return t_rot @ tc_proj133o @ t_rot_inv
  def tf_rotation_loss(t_133o):
    t_rotated = tf_rotated(t_133o)
    t_loss = tf.math.reduce_sum(tf.math.square(t_rotated - tc_proj133o_target))
    if debug:
      print(f'Loss: {t_loss.numpy():.6g}')
    return t_loss
  opt_val, opt_rot = mu.tf_minimize_v2(
      tf_rotation_loss,
      mu.rng(0).normal(size=133, scale=1e-3))
  # Note that output-index of the projector is ^a. To this, we apply NC @ C.
  return opt_val, opt_rot, tf_rotated(mu.tff64(opt_rot)).numpy()


def e7_rotate_theta(theta,
                    e7_rotation,
                    order133_c_first=True,
                    e7=e7):
  """E7-rotates a Theta-tensor.

  Args:
    theta: [56, 133]-numpy.ndarray, the Theta-tensor to rotate.
    e7_rotation: [133]-numpy.ndarray, the e7 generator parameters
      to build the rotation from.
    order133_c_first: If true, transform the ^a index on the
      Theta-tensor via noncompact_cb compact_ba theta...^a, i.e.
      applying the compact rotation built from the su(8) part
      of e7_rotation first. If false, the noncompact rotation
      will get applied first. (Useful for applying an inverted
      rotation.)
    e7: The e7 algebra that provides the relevant definitions.

  Returns:
    A [56, 133]-numpy.ndarray, the rotated Theta-tensor.
  """
  # The 'rotation' is split here: We separately exponentiate
  # the 'compact' and 'noncompact' part, and, on an ^a-index,
  # apply noncompact @ compact.
  rot_nc133 = scipy.linalg.expm(
      mu.nsum('abC,a->Cb', e7.f_abC[:70], e7_rotation[:70]))
  rot_c133 = scipy.linalg.expm(
      mu.nsum('abC,a->Cb', e7.f_abC[70:], e7_rotation[70:]))
  rot_nc56 = scipy.linalg.expm(
      mu.nsum('aMN,a->NM', e7.t56r[:70], -e7_rotation[:70]))
  rot_c56 = scipy.linalg.expm(
      mu.nsum('aMN,a->NM', e7.t56r[70:], -e7_rotation[70:]))
  if order133_c_first:
    rot_133 = rot_nc133 @ rot_c133
    rot_56 = rot_c56 @ rot_nc56
  else:
    rot_133 = rot_c133 @ rot_nc133
    rot_56 = rot_nc56 @ rot_c56
  return mu.nsum('Ma,ba,MN->Nb', theta, rot_133, rot_56)


def tf_e7_turners_56_133o(e7=e7):
  """Returns closure that maps 70+63 e7-rotation to _56 and ^133o matrices."""
  # This is a somewhat technical helper function which however
  # is useful to have in quite a few places.
  tc_fo_abC = mu.tff64(e7.fo_abC)
  tc_to56r = mu.tff64(
      mu.nsum('aMN,ac->cMN', e7.t56r, e7.v133_from_v133o))
  def tf_turners(t_133o, order133_c_first=True):
    """Returns 133o x 133o and 56 x 56 rotation matrices.

    Args:
      t_133o: numpy.ndarray, rotation-parameters in the
        133o-irrep of e7 from which we build two rotations
        (per Theta-index, so four in total), one for the
        compact and one for the noncompact part of the algebra.
      order133_c_first: If true, the 133o-rotation is built
        by first performing the compact, then the noncompact
        rotation. (Useful for taking inverses.)

    Returns:
      A pair `(t_rot_133o, t_rot_56)` of a [133, 133]-float64-tf.Tensor
      `t_rot133o` rotating the 133o-index on Theta and a
      [56, 56]-float64-tf.Tensor rotating the 56-index on Theta.
    """
    t_70 = t_133o[:70]
    t_63 = t_133o[70:]
    t_rot_nc133o = tf.linalg.expm(
        tf.einsum('abC,a->Cb', tc_fo_abC[:70], t_70))
    t_rot_c133o = tf.linalg.expm(
        tf.einsum('abC,a->Cb', tc_fo_abC[70:], t_63))
    t_rot_nc56 = tf.linalg.expm(
        tf.einsum('aMN,a->NM', tc_to56r[:70], -t_70))
    t_rot_c56 = tf.linalg.expm(
        tf.einsum('aMN,a->NM', tc_to56r[70:], -t_63))
    if order133_c_first:
      t_rot_133o = t_rot_nc133o @ t_rot_c133o
      t_rot_56 = t_rot_c56 @ t_rot_nc56
    else:
      t_rot_133o = t_rot_nc133o @ t_rot_c133o
      t_rot_56 = t_rot_c56 @ t_rot_nc56
    return t_rot_133o, t_rot_56
  return tf_turners


def align_thetas(theta_to_align, thetas_target,
                 e7=e7, debug=True,
                 maxiter=10**4,
                 x0_hint=None,
                 strategy=(('BFGS', None),),
                 seed=11):
  """Aligns a linear combination of thetas_input with thetas_target."""
  # thetas_target.shape == (num_thetas, 56, 133).
  t0 = t_now = time.time()
  num_thetas = thetas_target.shape[0]
  tf_turners = tf_e7_turners_56_133o(e7=e7)
  # 'Split rotation' here means that on an ^a-index, we first apply a
  # 'compact SU(8)' and then a 'noncompact E7' rotation.
  theta_to_align_56x133o = mu.nsum(
      'Ma,ca->Mc', theta_to_align, e7.v133o_from_v133)
  tc_theta_to_align_56x133o = mu.tff64(
      theta_to_align_56x133o)
  thetas_target_56x133o = mu.nsum(
      'nMa,ca->nMc', thetas_target, e7.v133o_from_v133)
  tc_thetas_target_56x133o = mu.tff64(thetas_target_56x133o)
  def tf_rotated(t_133o):
    t_rot_133o, t_rot_56 = tf_turners(t_133o)
    return tf.einsum('Ma,ba,MN->Nb',
                     tc_theta_to_align_56x133o, t_rot_133o, t_rot_56)
  def tf_rotation_loss(t_133ox):
    nonlocal t_now
    t_rotated = tf_rotated(t_133ox[:133])
    t_target = mu.nsum('nMa,n->Ma', tc_thetas_target_56x133o, t_133ox[133:])
    t_loss = tf.math.reduce_sum(tf.math.square(t_rotated - t_target))
    if debug:
      t_next = time.time()
      print(f'T={t_next - t0:8.3f} sec (+{t_next - t_now:8.3f} sec) '
            f'Loss: {t_loss.numpy():.12g}')
      t_now = t_next
    return t_loss
  if x0_hint is not None:
    x0 = numpy.asarray(x0_hint)
  else:
    x0 = (mu.rng(seed).normal(size=133, scale=0.05).tolist() +
          mu.rng(seed + 1).normal(size=num_thetas, scale=0.25).tolist())
  opt_val, opt_133ox = mu.tf_minimize_v2(
      tf_rotation_loss,
      x0,
      strategy=strategy,
      default_maxiter=maxiter,
      default_gtol=1e-14)
  # Note that output-index of the projector is ^a.
  # To this, we apply NC @ C.
  return (opt_val,
          opt_133ox,
          numpy.concatenate(
              [e7.v133_from_v133o.dot(opt_133ox[:133]),
               opt_133ox[133:]], axis=0),
          mu.nsum('Ma,ba->Mb',
                  tf_rotated(mu.tff64(opt_133ox[:133])).numpy(),
                  e7.v133_from_v133o))


def get_invariant_thetas(e7_gens,
                         seed=0,
                         num_rounds_max=100,
                         threshold=1e-4,
                         thetas_start=None,
                         filename=None,
                         onb_form=True,  # Whether to transform to ONB.
                         e7=e7,  # The E7 algebra to use.
                         verbose=True):
  """Computes a basis for e7_gens-invariant Thetas in the 912-irrep."""
  # Index structure: Theta_M^a, with M a e7-real index;
  # returns a [N, 56, 133]-array.
  if filename is not None:
    try:
      return numpy.load(filename)
    except (IOError, OSError):
      pass
  # If we reach this point, loading cached thetas did not work.
  t0 = time.time()
  if verbose:
    print('Computing thetas.')
  rng = numpy.random.RandomState(seed=seed)
  def get_random_gen():
    return mu.nsum('na,n->a', e7_gens, rng.normal(size=len(e7_gens)))
  def get_d_thetas(thetas, e7_gen):
    return (
        -mu.nsum('Z_M^b,_aN^M,a->Z_N^b', thetas, e7.t56r, e7_gen)
        +mu.nsum('Z_M^b,_ab^c,a->Z_M^c', thetas, e7.f_abC, e7_gen))
  def reduce_thetas(thetas_now, e7_gen):
    # Z = batch-index.
    d_thetas = get_d_thetas(thetas_now, e7_gen)
    # We want to find those thetas that are invariant under this particular
    # rotation with some E7-element. Let us try to do this via a SVD.
    su, ss, svh = numpy.linalg.svd(d_thetas.reshape(-1, 56 * 133))
    del svh  # Unused.
    # d_thetas[k, :].reshape(...) gives us what the k-th Theta got mapped to.
    # We want to know those weight-vectors for the original Theta-s that give
    # us zeroes.
    if verbose:
      print('SVD ss:', sorted(collections.Counter(ss.round(3)).items()))
    kernel_basis = su.T[ss <= threshold]
    return mu.nsum('YZ,Z_M^b->Y_M^b', kernel_basis, thetas_now)
  #
  thetas_now = (
      thetas_start if thetas_start is not None
      else numpy.eye(56 * 133).reshape(-1, 56, 133))  # A 7448 x 7448 matrix!
  prev_num_thetas = -1  # Impossible int at start.
  for num_round in range(num_rounds_max):
    thetas_now = reduce_thetas(thetas_now, get_random_gen())
    num_thetas = thetas_now.shape[0]
    if verbose:
      print(f'Round {num_round}, num_thetas={num_thetas}')
    if num_thetas == prev_num_thetas:
      break
    prev_num_thetas = num_thetas
  gramian = mu.nsum('ZMa,WMb,ab->ZW', thetas_now, thetas_now, e7.k133)
  if verbose:
    t1 = time.time()
    print('Done computing thetas, T=%.3f sec' % (t1 - t0))
  if not onb_form:
    return thetas_now, gramian
  # Need to bring to ONB-form.
  onb, onbi = mu.get_gramian_onb(gramian)
  del onbi  # Unused.
  thetas_onb = mu.nsum('ZMa,WZ->WMa', thetas_now, onb)
  gramian_onb = mu.nsum('ZMa,WMb,ab->ZW', thetas_onb, thetas_onb, e7.k133)
  assert numpy.allclose(gramian_onb, numpy.diag(numpy.diag(gramian_onb)))
  if filename is not None:
    numpy.savez_compressed(filename, thetas_now, gramian_onb)
  return thetas_onb, gramian_onb


def gauge_group_gens_from_theta(theta, threshold=1e-4,
                                e7_gens=None):
  """Determines the gauge group generators, given the Theta-tensor."""
  # First, we need to know how each of the E7-generators acts on the
  # Theta-tensor. We can express this as a (56 x 133) x 133-matrix.
  if e7_gens is None:
    e7_gens = numpy.eye(133)
  d_theta = (
      -mu.nsum('M^b,_aN^M,an->N^bn', theta, e7.t56r, e7_gens)
      +mu.nsum('M^b,_ab^c,an->M^cn', theta, e7.f_abC, e7_gens))
  su, ss, svh = numpy.linalg.svd(d_theta.reshape(-1, e7_gens.shape[-1]))
  del su  # Unused.
  return svh[ss <= threshold].T


def get_X(Theta, e7=e7):
  """Computes the X-tensor from a Theta-tensor."""
  # The X_MN^P invariant. Let us (for now) do this quick and dirty, and later
  # consider ways to make this faster by exploiting sparsity.
  return numpy.einsum('aNP,Ma->MNP', e7.t56r, Theta)


# The thetas_so7s are "randomly dyonic".
# Overall metric is SO(p, q) - for so(7)-invariant solutions, SO(2, 2).
# We want to find a SO(p) x SO(q) rotation that "minimizes dyonic mixing".
# So, we can take the product of the sum-squared of "electric" and "magnetic"
# components as our "loss function".


def get_so_p_x_so_q_gens(p, q):
  """Produces generators of so(p) x so(q)."""
  num_gens_p = p * (p - 1) // 2
  num_gens_q = q * (q - 1) // 2
  gens = numpy.zeros([num_gens_p + num_gens_q, p + q, p + q])
  for n, (a, b) in enumerate(itertools.combinations(range(p), 2)):
    gens[n, a, b] = 1.0
    gens[n, b, a] = -1.0
  for n, (a, b) in enumerate(itertools.combinations(range(q), 2)):
    gens[num_gens_p + n, p + a, p + b] = 1.0
    gens[num_gens_p + n, p + b, p + a] = -1.0
  return gens


def de_dyonize(thetas_WMa, rotation_gens_W):
  """Rotates Theta-tensors to minimize mixing E/M charges."""
  # TODO(tfish): needs better naming (function and args).
  # TODO(tfish): Rewrite with tf_minimize.
  tc_rgens = tf.constant(rotation_gens_W, dtype=tf.float64)
  tc_thetas = tf.constant(thetas_WMa, dtype=tf.float64)
  def tf_get_rot_thetas(t_gs):
    t_rot = tf.linalg.expm(tf.einsum('aMN,a->MN', tc_rgens, t_gs))
    return tf.einsum('WMa,ZW->ZMa', tc_thetas, t_rot)
  def tf_get_loss(t_gs):
    t_rot_thetas = tf_get_rot_thetas(t_gs)
    t_l2_electric = tf.einsum(
        'ZMa,ZMa->Z', t_rot_thetas[:, :28, :], t_rot_thetas[:, :28, :])
    t_l2_magnetic = tf.einsum(
        'ZMa,ZMa->Z', t_rot_thetas[:, 28:, :], t_rot_thetas[:, 28:, :])
    return tf.reduce_sum(t_l2_electric * t_l2_magnetic)
  # TODO(tfish): Use mu.tf_minimize() here.
  def f_opt(gs):
    return tf_get_loss(tf.constant(gs, dtype=tf.float64)).numpy()
  def fprime_opt(gs):
    t_gs = tf.constant(gs, dtype=tf.float64)
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_gs)
      t_loss = tf_get_loss(t_gs)
    return tape.gradient(t_loss, t_gs).numpy()
  w_opt = scipy.optimize.fmin_bfgs(f_opt,
                                   numpy.zeros(len(rotation_gens_W)),
                                   gtol=1e-14,
                                   fprime=fprime_opt)
  rot_thetas = tf_get_rot_thetas(tf.constant(w_opt, dtype=tf.float64)).numpy()
  # Let's sort them lexically by (scalar_product, magnetic_l2).
  def theta_key(theta):
    return (mu.nsum('Ma,Mb,ab->', theta, theta, e7.k133),
            mu.nsum('Ma,Ma->', theta[28:], theta[28:]))
  ret = numpy.stack(sorted(rot_thetas, key=theta_key), axis=0)
  return ret


def get_Theta_so8(omega):
  """Returns a Theta-tensor for dyonic SO(8) gauging."""
  Theta = numpy.zeros([56, 133])
  cw, sw = numpy.cos(omega), numpy.sin(omega)
  Theta[:28, 3*35:] = +cw * numpy.eye(28) * 0.25
  Theta[28:, 3*35:] = +sw * numpy.eye(28) * 0.25
  return Theta


def get_gaugeability_condition_violations(
    Theta,
    checks=('omega', 'XX', 'linear_a', 'linear_b'),
    e7=e7,
    atol=1e-8,
    early_exit=True):
  """Checks whether the Theta-tensor satisfies the gaugeability constraints."""
  violations = {}
  X = get_X(Theta, e7=e7)
  if 'omega' in checks:  # (2.11) in arXiv:1112.3345
    omega_theta2 = mu.nsum('MN,Ma,Nb->ab', e7.omega, Theta, Theta)
    if not numpy.allclose(0, omega_theta2, atol=atol):
      violations['omega'] = sum(abs(omega_theta2).ravel()**2)**.5
      if early_exit:
        return violations
  if 'XX' in checks:  # (2.12) in arXiv:1112.3345
    XX_products = numpy.einsum('MPQ,NQR->MNPR', X, X)
    XX_comm = XX_products - numpy.einsum('MNPR->NMPR', XX_products)
    XX_MN = -numpy.einsum('MNP,PQR->MNQR', X, X)
    if not numpy.allclose(XX_comm, XX_MN, atol=atol):
      violations['XX'] = sum(abs(XX_comm - XX_MN).ravel()**2)**.5
      if early_exit:
        return violations
  if 'linear_a' in checks:  # (2.14) in arXiv:1112.3345
    deviation_a = numpy.einsum('Na,aMN->M', Theta, e7.t56r)
    if not numpy.allclose(0, deviation_a,
                          atol=atol):
      violations['linear_a'] = sum(abs(deviation_a)**2)
      if early_exit:
        return violations
  if 'linear_b' in checks:
    k56_inv = numpy.linalg.inv(
        numpy.einsum('aMN,bNM->ab', e7.t56r, e7.t56r))
    tt = numpy.einsum('bMP,APN->bAMN',
                      e7.t56r,
                      numpy.einsum('aPN,aA->APN', e7.t56r, k56_inv))
    tt_Theta = numpy.einsum('bAMN,Nb->MA', tt, Theta)
    deviation_b = tt_Theta +0.5 * Theta
    if not numpy.allclose(0, deviation_b, atol=atol):
      violations['linear_b'] = sum(abs(deviation_b.ravel())**2)**.5
  return frozenset(violations.items())


def f_MNP_from_Theta(Theta, e7=e7, rcond=1e-15):
  """Computes the embedded gauge group structure constants."""
  X = get_X(Theta, e7=e7)
  # [X_M, X_N] = f_MN^P X_P
  kX_ab = mu.nsum('MPQ,NQP->MN', X, X)
  # kX_ab will have rank 28, not 56. So, we need to use a pseudo-inverse here.
  kX_inv_ab = numpy.linalg.pinv(kX_ab, rcond=rcond)
  X_MN = mu.asymm2(mu.nsum('MPR,NRQ->MNPQ', X, X), 'MNPQ->NMPQ')
  X_MN_P = mu.nsum('MNRS,PSR->MNP', X_MN, X)
  # Problem: kX_ab has rank < 56.
  # Let's try getting the f_MNP from solving a least-squares problem.
  f_MNP = mu.nsum('MNQ,QP->MNP', X_MN_P, kX_inv_ab)
  return f_MNP


def show_theta(Theta, e7=e7):
  """Prints a Theta-tensor and the associated gauging's Killing form."""
  f_MNP = f_MNP_from_Theta(Theta, rcond=1e-15, e7=e7)
  k_ab = mu.nsum('MPQ,NQP->MN', f_MNP, f_MNP)
  mu.tprint(Theta, d=5, name='Theta')
  mu.tprint(k_ab, d=5, name='k_MN')
  print('K_ab eigvals:',
        sorted(
            collections.Counter(numpy.linalg.eigvals(k_ab).round(6)).items()))


def get_quadratic_constraints(thetas_in_912, e7=e7):
  """For a collection of Theta-tensors, returns the quadratic constraints.

  Note: This is typically a memory-wise expensive operation.

  Args:
    thetas_in_912: [k, 56, 133]-array of k Theta-tensors that must belong
      to the 912-irrep of potentially gaugeable Thetas.
      Here, the 56-index refers to the e7.t56r basis.
    e7: The algebra.E7 instance to use for E7-conventions.

  Returns:
   [c, k, k]-array of quadratic constraints `qcs`.
     The c-th entry represents the c-th quadratic constraint,
     i.e. einsum('pma,p->ma', thetas_in_912, coeffs) is gaugeable if
     einsum('kpq,p,q->k', qcs, coeffs, coeffs) vanishes.
  """
  # arXiv:1112.3345, below (2.15): For Theta-tensors that satisfy
  # the linear 912-constraint, the two quadratic constraints are equivalent, so
  # we only have to check the Omega-constraint: Theta_M^a Theta_N^b Omega^MN = 0
  thetas_qc = mu.nsum('W_M^a,Z_N^b,^MN->WZab',
                      thetas_in_912, thetas_in_912, e7.omega)
  # Symmetrize in W,Z, since we use the same linear combination of thetas
  # for both. Update entries for memory efficiency.
  thetas_qc += mu.nsum('WZab->ZWab', thetas_qc)
  ttsu, ttss, ttsvh = numpy.linalg.svd(thetas_qc.reshape(-1, 133 * 133))
  del ttsvh  # Unused.
  q_constraints = ttsu.T[ttss > 1e-9].reshape([-1] +
                                              [thetas_in_912.shape[0]] * 2)
  for level in (2, 1, 0):
    gc.collect(level)
  return q_constraints


def get_gauging_from_theta(theta, max_residuals=1e-4):
  """Analyzes a Theta-tensor and returns the corresponding Gauging."""
  gg_gens = gauge_group_gens_from_theta(theta, threshold=0.1)
  # Inner product on the gauge group generators,
  # induced by the e7-algebra's inner product.
  k_gg = mu.nsum('am,bn,ab->mn', gg_gens, gg_gens, e7.k133)
  gg_onb, gg_onbi = mu.get_gramian_onb(k_gg, eigenvalue_threshold=1.0)
  del gg_onbi  # Unused, named for documentation only.
  gg_gens_onb = mu.nsum('am,nm->an', gg_gens, gg_onb)
  #
  # Let us check that we indeed did it right.
  k_gg_onb = mu.nsum('am,bn,ab->mn', gg_gens_onb, gg_gens_onb, e7.k133)
  diag_k_gg_onb = numpy.diag(k_gg_onb)
  assert numpy.allclose(k_gg_onb, numpy.diag(diag_k_gg_onb), atol=0.01), (
      'Inner product on the gauge group is off in the orthonormal basis')
  rel_diag_k_gg_onb = diag_k_gg_onb / abs(diag_k_gg_onb).max()
  #
  gg_null = gg_gens_onb[:, abs(rel_diag_k_gg_onb) < 0.01]
  gg_compact = gg_gens_onb[:, rel_diag_k_gg_onb < -0.01]
  gg_compact_commutators = mu.nsum('abC,am,bn->Cmn',
                                   e7.f_abC, gg_compact, gg_compact)
  (gg_compact_commutators_decomp,
   gg_decomp_residuals, *_) = numpy.linalg.lstsq(
       gg_compact, gg_compact_commutators.reshape(133, -1), rcond=1e-5)
  assert abs(gg_decomp_residuals).max() < max_residuals, (
      '[Compact, Compact] ~ Compact decomposition has large residuals.')
  dim_compact = gg_compact.shape[-1]
  f_abC_compact = gg_compact_commutators_decomp[:dim_compact].reshape(
      [dim_compact] * 3)
  k_ab_compact = mu.nsum('mbc,ncb->mn', f_abC_compact, f_abC_compact)
  gg_compact_onb, gg_compact_onbi = mu.get_gramian_onb(
      k_ab_compact, eigenvalue_threshold=0.01)
  del gg_compact_onbi  # Unused, named for documentation only.
  gg_compact_gens_onb = mu.nsum('am,nm->an', gg_compact, gg_compact_onb)
  k_ab_compact_onb = mu.nsum('Mm,Nn,mn->MN',
                             gg_compact_onb, gg_compact_onb, k_ab_compact)
  k_ab_compact_onb_diag = numpy.diag(k_ab_compact_onb)
  gg_u1s = gg_compact_gens_onb[:, abs(k_ab_compact_onb_diag) < 0.5]
  gg_compact_semisimple = gg_compact_gens_onb[:,
                                              abs(k_ab_compact_onb_diag) >= 0.5]
  dim_compact_semisimple = gg_compact_semisimple.shape[-1]
  random_gg_semisimple_elem = gg_compact_semisimple.dot(
      mu.rng(0).normal(size=dim_compact_semisimple))
  # Let us see what the subspace of the compact-semisimple algebra
  # looks like that commutes with this random element.
  m_commute_with_random = mu.nsum('abC,a,bn->Cn',
                                  e7.f_abC,
                                  random_gg_semisimple_elem,
                                  gg_compact_semisimple)
  # 'csa' == 'Cartan Subalgebra'
  svd_csa_u, svd_csa_s, svd_csa_vh = numpy.linalg.svd(m_commute_with_random,
                                                      full_matrices=False)
  del svd_csa_u  # Unused, named for documentation only.
  csa_basis = svd_csa_vh[svd_csa_s < 1e-5, :].T
  gg_gens = numpy.concatenate([gg_compact_semisimple, gg_u1s, gg_null], axis=-1)
  gg_ranges = (0,
               gg_compact_semisimple.shape[1],
               gg_compact_semisimple.shape[1] + gg_u1s.shape[1],
               28)
  assert gg_gens.shape[1] == 28, 'Gauge group is not 28-dimensional.'
  return Gauging(
      theta=theta,
      gg_gens=gg_gens,
      gg_ranges=gg_ranges,
      cartan_subalgebra=csa_basis)


#### SUPERGRAVITY ###


class Dim4SUGRA(supergravity.SUGRA):
  """D=4 Supergravity."""

  signature = supergravity.SUGRASignature(
      name='Generic4D',
      dim=4,
      generator_scaling=-1.0,
      dim_scalar_manifold=70,
      num_model_params=0,
      scalar_masses_dV_from_right=True,
      scalar_masses_factor=36.0,
      gravitino_masses_factor=6.0,
      vector_masses_factor=6.0,
      fermion_masses_factor=6.0)

  # TODO(tfish): SUSY-tweak should be per-scan.
  def __init__(self, Theta,
               e7=e7,
               verbose=False,
               check_gaugeability=True,
               gaugeability_atol=1e-8,
               # Either `None`, or ('SUSY', None),
               # or ('M2G', {[8]-sequence of masses}).
               stationarity_tweak=None):
    super().__init__(e7.t56r, verbose=verbose)
    if check_gaugeability:
      if get_gaugeability_condition_violations(Theta, e7=e7,
                                               atol=gaugeability_atol):
        raise ValueError('Non-gaugeable Theta-tensor provided.')
    self._stationarity_tweak = stationarity_tweak
    self._opt_tc_stationarity_tweak = (
        None if stationarity_tweak is None or stationarity_tweak[1] is None
        else mu.tff64(stationarity_tweak[1]))
    self._tc_X = tc_X = mu.tff64(get_X(Theta, e7=e7))
    self._tc_XX = tf.einsum('MNQ,PQN->MP', tc_X, tc_X)
    self._tc_e7_S_rc = tf.constant(e7.S_rc, dtype=tf.complex128)
    self._tc_1j = mu.tfc128(0, 1)
    self._tc_28_8_8 = tf.constant(algebra.g.su8.m_28_8_8, dtype=tf.complex128)
    self._tc_56_888 = tf.constant(algebra.g.su8.m_56_8_8_8, dtype=tf.complex128)
    self._tc_eps_56_56_8_8 = tf.constant(algebra.g.su8.eps_56_56_8_8,
                                         dtype=tf.complex128)
    self._tc_omega = tf.constant(e7.omega, dtype=tf.complex128)

  def tf_stationarity_internal(self, t_potential, t_grad_potential, t_vielbein):
    """Computes the stationarity-violation."""
    t_stat = super().tf_stationarity_internal(t_potential, t_grad_potential,
                                              t_vielbein)
    if self._stationarity_tweak is None:
      return tf.math.asinh(t_stat)  # Squashed.
    else:
      t_A1, *_ = self.tf_A123(self.tf_T(t_vielbein),
                              want_A1=True, want_A2=False, want_A3=False)
      t_m2grav = self.tf_gravitino_masses(t_A1, t_potential)
      if self._stationarity_tweak[0] == 'SUSY':
        t_lightest_gravitino_mass = t_m2grav[-1]
        t_ret = t_stat * tf.clip_by_value(t_lightest_gravitino_mass, 1.0, 5.0)
        return tf.math.asinh(t_ret)  # Double-squash
      elif self._stationarity_tweak[0] == 'M2G':
        t_spectrum_mismatch = tf.math.reduce_sum(
            tf.math.square(t_m2grav - self._stationarity_tweak[1]))
        return (tf.math.asinh(t_stat) +
                tf.math.asinh(t_spectrum_mismatch) * mu.tff64(0.1))
      else:
        raise ValueError('Unknown stationarity-tweak.')

  # TODO(tfish): def canonicalize_equilibrium(self, v70, verbose=True):

  def tf_sugra_tensors_from_vielbein(self, t_V):
    """See base class."""
    # We also need the inverse Vielbein.
    # Here, we make use of the inverse of a symplectic matrix
    #   [[A, B], [C, D]]
    # being given by:
    #   [[D.T, -B.T], [-C.T, A.T]].
    t_Vi = tf.reshape(
        tf.stack(
            [tf.stack([tf.transpose(t_V[28:, 28:]),
                       -tf.transpose(t_V[:28, 28:])], axis=1),
             tf.stack([-tf.transpose(t_V[28:, :28]),
                       tf.transpose(t_V[:28, :28])], axis=1)], axis=0),
        (56, 56))
    t_M = tf.einsum('RX,SX->RS', t_V, t_V)
    t_Minv = tf.einsum('XR,XS->RS', t_Vi, t_Vi)
    # Potential (2.9) in arXiv:1112.3345 - also, (4.49) in arXiv:0705.2101.
    t_potential = mu.tff64(1 / 168.0) * (
        tf.einsum('mnr,mM,nN,rR,MNR->',
                  self._tc_X, t_Minv, t_Minv, t_M, self._tc_X) +
        7 * tf.einsum('MP,MP->', self._tc_XX, t_Minv))
    return (t_potential,)  # 1-tuple.

  def tf_T(self, t_vielbein):
    # Here, we do not actually compute the full T-tensor, but the T_i^jkl part.
    # This is all that we ever need here.
    V56_RC = tf.einsum('RS,SC->RC',
                       tf.cast(t_vielbein, tf.complex128),
                       self._tc_e7_S_rc)
    t_Q_Mijkl = self._tc_1j * tf.einsum(
        'NP,NI,Iij,MPQ,QK,Kkl->Mijkl',
        self._tc_omega,
        V56_RC[:, 28:],
        self._tc_28_8_8,
        tf.cast(self._tc_X, tf.complex128),
        V56_RC[:, :28],
        self._tc_28_8_8)
    t_Q_Mij = tf.einsum('Mijil->Mjl', t_Q_Mijkl) * mu.tfc128(-2 / 3.0)
    return mu.tfc128(0.0, 0.75 / 2**.5) * tf.einsum(
        # Note fudge-factor sqrt(2) above.
        # This makes the T-tensor match the colab notebook on SO(8) gauging.
        'MN,Mkl,NI,Iij->klij',
        self._tc_omega,
        t_Q_Mij,
        V56_RC[:, :28],
        self._tc_28_8_8)

  def tf_A123(self, t_T, want_A1=True, want_A2=True, want_A3=True):
    t_A1, t_A2, t_A3 = None, None, None
    if want_A1:
      t_A1 = tf.einsum('mijm->ij', t_T) * mu.tfc128(-4.0 / 21)
      # TODO(tfish): Do check if the SO(8)-origin-A1 is actually diag([1.0]*8)
      # or perhaps now has an extra factor 1j.
    if want_A2 or want_A3:
      # Over-satisfying is OK: When asked for A3 only, we also compute A2.
      t_A2 = mu.tfc128(-4.0 / (3 * 3)) * (
          t_T
          + tf.einsum('lijk->ljki', t_T)
          + tf.einsum('lijk->lkij', t_T))
    if want_A3:
      t_A3 = a123.tf_A3_from_A2(t_A2)
    return t_A1, t_A2, t_A3

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
