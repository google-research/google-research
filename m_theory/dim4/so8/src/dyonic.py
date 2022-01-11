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

"""Explores omega-deformation without detour through the stationarity-Hessian.

python3 -i -m dim4.so8.src.dyonic

Rather, we directly work with the scalar potential's gradient and Jacobian.


The linear algebra in this module is a bit involved.
It helps to clarify the role of some relevant linear spaces:

  - "v70" vectors are scalar field parameters in the conventions
    of https://arxiv.org/abs/1906.00207; with respect to the
    Cartan-Killing form of e7, this is a non-orthonormal basis.

  - "v70o" vectors are scalar field parameters in an alternate
    basis in which the e7 Cartan-Killing form is a multiple
    of the identity matrix. The relevance of this basis is that
    the "v70"-basis makes some relevant operators that are
    symmetric in the v70o-basis look non-symmetric.
    This prevents use of functions such as numpy.linalg.eigh()
    to get eigenspaces. However, if eigenvalues are degenerate,
    then the eigenbases produced by the more general
    numpy.linalg.eig() function are often not numerically
    well-behaved. So, numerical algorithms strongly suggest using
    orthonormal bases. However, since it is not yet quite clear
    what the most appropriate choice of v70o-basis is, while
    there are indications that some choices are indeed much
    more suitable to answer some relevant questions,
    the "v70o"-basis is to be considered 'internal', i.e. any data
    written to the filesystem should refer to the v70-basis ONLY.

 - "v71" vectors are v70-vectors extended with a final coordinate
   that provides the value of omega.

 - "v71o" vectors likewise are v70o-vectors extended with omega.

 - The `SO8c_SUGRA().subspace_an` attribute is an
   optional [70, d]-ndarray that holds the d basis vectors of the
   subspace to which we want to constrain the study of
   omega-deformation. In order to assess violation of the
   stationarity-condition, the gradient of the scalar potential
   is however taken in the full 70-dimensional manifold,
   so points where the stationarity-violation is "numerically zero"
   are equilibria on the full scalar manifold - we do not need this
   submanifold to be the invariant submanifold w.r.t. some subgroup
   of SO(8).

 - The `SO8c_SUGRA().subspace_normalizer_an` is a basis for
   the subalgebra of so(8) that normalizes `subspace_an`,
   as a [28, n]-ndarray.

 - The `SO8c_SUGRA().v71_from_wsspo` [71, d + 1]-ndarray consists
   of a basis of `subspace_an` that is orthonormal w.r.t.
   the e7 inner product, extended by a final basis vector that
   associates the final of the d+1 coordinates with the final
   omega-coordinate of v71. This basis is ad-hoc computed
   at SO8c_SUGRA instantiation time, and so may not be the same
   when running the same code on systems that use different
   TensorFlow / NumPy / SciPy / Python versions.
   Data written to the filesystem should always use the v70/v71 basis,
   but in some situations, it may make sense to temporarily work
   with `wsspo`-vectors, and then it matters to provide the specific
   choice of wsspo-basis when instantiating a new SO8c_SUGRA
   (to align with the one with which the earlier calculation was done).

 - The 'goldstone basis' returned by:
   `num_goldstone, wsspo_g_nong_basis, proj_wsspo_goldstone = (
        SO8c_SUGRA().wsspo_goldstone_basis_and_projector())`
   is a [71, dim_wsspo]-ndarray of dim_wsspo orthonormal v71o-vectors
   that span the same subspace as wsspo, but are such that the first
   num_goldstone such vectors form a basis of the 'goldstone'
   directions, i.e. the directions into which the input wsspo-vector
   can be rotated by acting with the subspace_normalizer_an generators.

"""


# Naming deviates from PEP-8 conventions where this makes mathematics easier
# to read. Also, local variables may name-match module-global definitions.
# pylint:disable=invalid-name
# pylint:disable=redefined-outer-name

import dataclasses
import datetime
import glob
import os
import re
import time

from dim4.generic import a123
from dim4.so8.src import analysis
from m_theory_lib import algebra
from m_theory_lib import m_util as mu
from m_theory_lib import supergravity
from matplotlib import pyplot
import numpy
import scipy.integrate
import scipy.interpolate
import tensorflow as tf


class SO8c_SUGRA(analysis.SO8_SUGRA):
  """SO(8) Supergravity with extra methods to study omega-deformation.

  Beyond the base class attributes, we additionally have these...

  Attributes:
    v70o_subspace_an: [70, n]-ndarray, a basis for the selected subspace
      of the noncompact generators of E7(7)/SU(8).
    v70o_subspace_normalizer_an: [28, d]-ndarray, a basis for the
      d-dimensional subalgebra of so(8) that normalizes subspace_an.
    v71o_from_wsspo: Mapping fro (o)rthonormal basis for omega(w)-extended
      (s)ub(sp)ace (wsspo) to v71o.
    v71_from_wsspo: Mapping fro (o)rthonormal basis for omega(w)-extended
      (s)ub(sp)ace (wsspo) to v71.
  """
  signature = dataclasses.replace(
      analysis.SO8_SUGRA.signature,
      name='SO8_omega')

  def __init__(self, *args,
               subspace_an=None,
               v71_from_wsspo=None,
               **kwargs):
    """Initializes the instance."""
    super().__init__(*args, **kwargs)
    # We need to add a function that gets us the Jacobian
    # of the stationarity-constraints.
    # Terminology:
    #   v71o = orthonormal basis of omega-extended E7(7)/SU(8) generators.
    #   wsspo = omega-extended subspace orthonormal basis.
    if subspace_an is None:
      subspace_an = numpy.eye(70)
    self.v70o_subspace_an = self.e7.v70o_from_v70.dot(subspace_an)
    if v71_from_wsspo is None:
      # The subspace basis will, in general, not be orthonormal.
      # We do need an orthonormal basis.
      k133 = self.e7.k133
      subspace_sprod = mu.nsum('am,bn,ab->mn',
                               subspace_an, subspace_an, k133[:70, :70] / 288.0)
      ssp_onb, ssp_onbi = mu.get_gramian_onb(subspace_sprod)
      del ssp_onbi  # Unused, named for documentation only.
      v71o_from_wsspo = numpy.zeros([71, 1 + self.v70o_subspace_an.shape[1]],
                                    dtype=numpy.float64)
      v71o_from_wsspo[-1, -1] = 1.0  # omega
      v71o_from_wsspo[:-1, :-1] = mu.nsum(
          'an,Nn->aN', self.v70o_subspace_an, ssp_onb)
    assert numpy.allclose(
        numpy.eye(v71o_from_wsspo.shape[1]),
        mu.nsum('aM,aN->MN', v71o_from_wsspo, v71o_from_wsspo)
    ), 'Omega-extended subspace-basis is not orthonormal.'
    self.v70o_subspace_normalizer_an = (
        algebra.get_normalizing_subalgebra(
            self.e7.fo_abC[-28:, :70, :70],
            self.v70o_subspace_an))
    self._tc_v70o_subspace_normalizer_an = mu.tff64(
        self.v70o_subspace_normalizer_an)
    self._tc_fo_abC = mu.tff64(self.e7.fo_abC)
    self.v71o_from_wsspo = v71o_from_wsspo
    self.dim_wsspo = v71o_from_wsspo.shape[1]
    self._tc_v70_from_v70o = mu.tff64(self.e7.v70_from_v70o)
    self.v71_from_wsspo = numpy.block(
        [[self.e7.v70_from_v70o.dot(v71o_from_wsspo[:-1, :-1]),
          numpy.zeros([70, 1])],
         [numpy.zeros([1, self.dim_wsspo - 1]), numpy.eye(1)]])
    self._tc_v71_from_wsspo = mu.tff64(self.v71_from_wsspo)

    @tf.function
    def tf_stationarity_constraints_wsspo(t_wsspo):
      """Computes the 70 stationarity-constraints."""
      t_omega = t_wsspo[-1]
      t_vielbein = self.tf_vielbein(
          tf.einsum('Aa,a->A', self._tc_v71_from_wsspo, t_wsspo)[:-1])
      t_T = self.tf_T(t_vielbein, t_omega=t_omega)
      t_A1, t_A2, _ = self.tf_A123(t_T, want_A3=False)
      return a123.tf_dwn_stationarity_vec(t_A1, t_A2)
    #
    self._jac_stationarity_constraints_wsspo = (
        mu.tf_jacobian(tf_stationarity_constraints_wsspo))

  def tf_v70_from_v70o(self, t_v70o):
    """Computes the v70 tf.Tensor given the v70o."""
    return tf.einsum('Aa,a->A', self._tc_v70_from_v70o, t_v70o)

  def wsspo_from_v71o(self, v71o, max_residuals=1e-10):
    """Expresses v71o-vector in wsspo-basis."""
    coeffs, residuals, *_ = scipy.linalg.lstsq(self.v71o_from_wsspo, v71o)
    if not max(abs(residuals.ravel()), default=0) <= max_residuals:
      raise ValueError(
          f'Could not wsspo-decompose v71o - residuals: {residuals.tolist()}')
    return coeffs

  def wsspo_from_v70_and_omega(self, v70, omega):
    """Returns wsspo-coeffs given v70-coeffs and omega."""
    return self.wsspo_from_v71o(
        self.e7.v70o_from_v70.dot(v70).tolist() + [omega])

  def jac_stationarity_singular_values(self, v70, omega):
    wsspo = self.wsspo_from_v70_and_omega(v70, omega)
    num_goldstone, *_ = (
        self.wsspo_goldstone_basis_and_projector(wsspo))
    jacobian = self._jac_stationarity_constraints_wsspo(mu.tff64(wsspo)).numpy()
    svd_u, svd_s, svd_vh = numpy.linalg.svd(jacobian)
    del svd_u, svd_vh  # Unused, only named for documentation.
    return num_goldstone, svd_s

  def wsspo_goldstone_basis_and_projector(self, wsspo, ev_threshold=1e-5):
    # Variant of super().v70o_goldstone_basis_and_projector
    # that respects the subspace embedding.
    e7 = self.e7
    v70o = self.v71o_from_wsspo.dot(wsspo)[:-1]
    # The v70o-vector, rotated by the normalizer-generators,
    # gives vectors that span the 'goldstone modes' subspace.
    v70os_normalizer_rotated = mu.nsum(
        'abC,an,b->Cn',
        e7.fo_abC[105:, :70, :70],
        self.v70o_subspace_normalizer_an,
        v70o)
    # We need `wsspo_g_nong_basis`, a basis for wsspo-space
    # in terms of dim_wsspo many 71-vectors, where the first
    # num_goldstone basis vectors correspond to the
    # 'goldstone mode subspace'.
    #
    # Conveniently, the self.v71o_from_wsspo vectors form
    # an orthonormal basis, so decomposition is straightforward.
    wsspo_normalizer_rotated = self.v71o_from_wsspo[:-1, :].T.dot(
        v70os_normalizer_rotated)
    # The vectors in `wsspo_normalizer_rotated` span a subspace,
    # but are not linearly independent. We want an orthonormal basis.
    svd_u, svd_s, svd_vh = numpy.linalg.svd(wsspo_normalizer_rotated[:-1, :],
                                            full_matrices=True)
    del svd_vh  # Unused, named for documentation only.
    svd_u_ext = mu.dstack(svd_u, numpy.eye(1))
    svd_sx = numpy.pad(svd_s, [(0, wsspo.size - svd_s.size)])
    onb_goldstone = svd_u_ext[:, svd_sx > ev_threshold]
    proj_goldstone = onb_goldstone.dot(onb_goldstone.T)
    return onb_goldstone.shape[1], svd_u_ext, proj_goldstone

  def get_wsspo_nongoldstone_omega_deformation_directions(
      self, wsspo,
      ev_threshold=1e-5,
      verbose=False,
      report=print):
    """Computes nontrivial deformable directions of an equilibrium."""
    # TODO(tfish): Document special-case ev_threshold=inf.
    # TODO(tfish): We do not need the full Jacobian - we can constrain
    # this computation to the 'nongoldstone' directions.
    # This may save up to 28/70 directions. Fix this once the code works.
    wsspo = numpy.asarray(wsspo)
    jac_stationarity_constraints = (
        self._jac_stationarity_constraints_wsspo(
            mu.tff64(wsspo)).numpy())
    num_goldstone, onb, proj_goldstone = (
        self.wsspo_goldstone_basis_and_projector(
            wsspo, ev_threshold=ev_threshold))
    del proj_goldstone  # Unused, named for documentation only.
    # We know that the stationarity-constraints will remain satisfied
    # when we move along the directions that we obtain from acting
    # on the 70-vector with the gauge group SO(8) - or, correspondingly,
    # for submanifolds, act with the submanifold's normalizer
    # subalgebra of SO(8).
    # These are the 'un-interesting' deformation directions.
    # We want the nullspace of the Jacobian on the orthogonal complement
    # of these directions.
    wsspo_dirs_nongoldstone = onb[:, num_goldstone:]
    if ev_threshold == numpy.inf:
      return wsspo_dirs_nongoldstone
    d_stationarity_constraints_by_d_nontrivial = (
        jac_stationarity_constraints.dot(wsspo_dirs_nongoldstone))
    # Let us actually use SVD here rather than scipy.linalg.null_space,
    # since the latter internally uses SVD anyhow, and we want access
    # to the singular values.
    svd_u, svd_s, svd_vh = numpy.linalg.svd(
        d_stationarity_constraints_by_d_nontrivial, full_matrices=False)
    del svd_u  # Unused, named for documentation only.
    if verbose:
      report(f'omega-directions SVD: s={svd_s.round(7).tolist()}')
    nontrivially_deformable_dirs_coords = svd_vh[svd_s < ev_threshold, :]
    # These are the nontrivially-deformable-directions.
    return mu.nsum('Ac,nc->An',
                   wsspo_dirs_nongoldstone,
                   nontrivially_deformable_dirs_coords), svd_s

  def get_extraflat_point(self, wsspo,
                          num_extraflat_directions=1,
                          ev_threshold=1e-5,
                          zoom_factor=1e5,
                          verbose=True,
                          report=print):
    wsspo = numpy.asarray(wsspo)
    num_goldstone, onb, proj_goldstone = (
        self.wsspo_goldstone_basis_and_projector(wsspo,
                                                 ev_threshold=ev_threshold))
    del onb, proj_goldstone  # Unused, named for documentation only.
    tc_wsspo_start = mu.tff64(wsspo)
    tc_zoom, tc_inv_zoom = mu.tff64(zoom_factor), mu.tff64(1 / zoom_factor)
    def tf_extraflatness(t_delta_wsspo):
      t_wsspo = tc_wsspo_start + tc_inv_zoom * t_delta_wsspo
      t_omega = t_wsspo[-1]
      t_pot, *_, t_stat = self.tf_ext_sugra_tensors(
          tf.einsum('Aa,a->A',
                    self._tc_v71_from_wsspo, t_wsspo)[:-1],
          t_omega=t_omega)
      # TODO(tfish): Refactor this to not re-do computation of the Vielbein.
      # Performance is not much of an issue here, but we still can improve this.
      t_jacobian = self._jac_stationarity_constraints_wsspo(t_wsspo)
      t_s, t_u, t_v = tf.linalg.svd(t_jacobian)
      del t_u, t_v  # Unused.
      # We impose 'flatness' on the 'Goldstone' and also
      # the requested-extra-flat directions.
      t_relevant_singular_values = t_s[
          -(num_goldstone + num_extraflat_directions):]
      t_loss = t_stat * tc_zoom + tf.math.reduce_sum(tf.math.square(
          t_relevant_singular_values))
      if verbose:
        report(f'L={t_loss.numpy():.6g} pot={t_pot.numpy():.10f} '
               f'stat={t_stat.numpy():.6g} '
               f'rsv={t_relevant_singular_values.numpy().round(7).tolist()}')
      return t_loss
    opt_val, opt_delta_wsspo = mu.tf_minimize_v2(
        tf_extraflatness, mu.tff64(numpy.zeros(self.dim_wsspo)),
        default_maxiter=10**4, default_gtol=1e-14)
    del opt_val  # Unused, named for documentation only.
    return wsspo + (1 / zoom_factor) * opt_delta_wsspo

  def refined_pot_stat_wsspo(
      self, wsspo,
      ev_threshold=1e-5):
    """Refines an equilibrium, keeping omega fixed."""
    wsspo = numpy.asarray(wsspo)
    t_omega = mu.tff64(wsspo[-1])
    num_goldstone, wsspo_g_nong_basis, proj_goldstone = (
        self.wsspo_goldstone_basis_and_projector(
            wsspo,
            ev_threshold=ev_threshold))
    del num_goldstone, proj_goldstone  # Unused, named for documentation only.
    # We want the non-Goldstone directions here,
    # have to orthonormal-project the wsspo to this subspace,
    # feed the mapping that gives us a v70 to
    # self.find_equilibrium(), and turn the result back into
    # a wsspo.
    # x0 uses the goldstone/non-goldstone-basis ('gng').
    v71o = self.v71o_from_wsspo.dot(wsspo)
    x0_gng = wsspo_g_nong_basis.T.dot(wsspo)
    # TODO(tfish): Perhaps introduce a private instance variable
    # for the transformation matrix relevant here?
    submanifold_embedding = self.v71_from_wsspo[:-1, :-1].dot(
        wsspo_g_nong_basis[:-1, :]).T
    pot_opt, stat_opt, xs_gng_opt = self.find_equilibrium(
        x0_gng,
        t_omega=t_omega,
        submanifold_embedding=submanifold_embedding,
        verbosity='',
        minimize_kwargs=dict(default_gtol=1e-15))
    v71o_opt = self.v71o_from_wsspo.dot(wsspo_g_nong_basis.dot(xs_gng_opt))
    # Now that we have a stationarity-approx-zero solution for this
    # particular value of omega, which furthermore sits on the
    # submanifold-under-study, we can still use the
    # submanifold-normalizer-subgroup-of-so(8) group N
    # of gauge rotations to keep a small distance to the
    # approximate input-equilibrium.
    #
    # This matters since, if we do have N-rotations that were used
    # in the previous optimization, it is quite possible that
    # these may have taken us far away in an effort to squeeze out
    # a minimally better stationarity-violation.
    dim_normalizer = self.v70o_subspace_normalizer_an.shape[1]
    if not dim_normalizer:
      # We do not have a subspace-normalizer, so, early-exit.
      wsspo_opt = self.wsspo_from_v71o(v71o_opt)
      return pot_opt, stat_opt, wsspo_opt
    # Otherwise, use another minimization step.
    # This is somewhat tricky:
    # - The main objective is that the re-minimization above
    #   does not take us 'far away' from the previous-omega
    #   point.
    # - Still, given that we have to make an ad-hoc choice here of
    #   'what makes a good pick of a solution among equivalent ones',
    #   this procedure will damage the smoothness of the trajectory.
    #   This will likely then show as a problem when trying to use
    #   higher-order convergence acceleration to determine the
    #   (gauge-coupling-scaled) limit Theta-tensor near a divergence.
    tc_v71o = mu.tff64(v71o)
    tc_v71o_opt = mu.tff64(v71o_opt)
    tc_v70o_subspace_normalizer_an = self._tc_v70o_subspace_normalizer_an
    tc_fo_abC = self._tc_fo_abC
    def tf_get_rotated_v70o(t_params):
      t_gen = tf.einsum('n,an,abC->Cb',
                        t_params,
                        tc_v70o_subspace_normalizer_an,
                        tc_fo_abC[-28:, :70, :70])
      return tf.einsum('Cb,b->C', tf.linalg.expm(t_gen), tc_v71o_opt[:-1])
    def tf_dist_rotated_original(t_params):
      t_v70o_opt_rotated = tf_get_rotated_v70o(t_params)
      return tf.math.reduce_sum(
          tf.math.square(t_v70o_opt_rotated - tc_v71o[:-1]))
    val_opt_rot, params_opt_rot = mu.tf_minimize_v2(
        tf_dist_rotated_original,
        numpy.zeros(dim_normalizer),
        default_gtol=1e-14)
    del val_opt_rot  # Unused, named for documentation only.
    v70o_opt_rot = tf_get_rotated_v70o(mu.tff64(params_opt_rot)).numpy()
    wsspo_opt_rot = self.wsspo_from_v71o(
        numpy.concatenate([v70o_opt_rot, v71o[-1:]], axis=0))
    return pot_opt, stat_opt, wsspo_opt_rot

  def get_physics_wsspo(self, wsspo, metadata):
    """Calls self.get_physics() on wsspo-location."""
    wsspo = numpy.asarray(wsspo)
    v71 = self.v71_from_wsspo.dot(wsspo)
    return self.get_physics(v71[:-1],
                            metadata,
                            t_omega=mu.tff64(wsspo[-1]))

  def refine_special_point(self, v71,
                           num_extraflat_directions=1,
                           point_type=None,
                           ev_threshold=1e-5, zoom_factor=1e5,
                           verbose=True, report=print):
    """Refines a special point."""
    wsspo = self.wsspo_from_v70_and_omega(v71[:-1], v71[-1])
    point_types = ('omega-extremum', 'flat')
    if point_type not in point_types:
      raise ValueError(f'Point type must be one of: {point_types}')
    num_goldstone, onb, proj_goldstone = (
        self.wsspo_goldstone_basis_and_projector(wsspo,
                                                 ev_threshold=ev_threshold))
    del onb, proj_goldstone  # Unused, was named for documentation only.
    tc_wsspo_start = mu.tff64(wsspo)
    tc_zoom, tc_inv_zoom = mu.tff64(zoom_factor), mu.tff64(1 / zoom_factor)
    def tf_loss(t_delta_wsspo):
      t_wsspo = tc_wsspo_start + tc_inv_zoom * t_delta_wsspo
      t_omega = t_wsspo[-1]
      t_pot, *_, t_stat = self.tf_ext_sugra_tensors(
          tf.einsum('Aa,a->A', self._tc_v71_from_wsspo, t_wsspo)[:-1],
          t_omega=t_omega)
      # TODO(tfish): Refactor this to not re-do computation of the Vielbein.
      # Performance is not much of an issue here, but we still can improve this.
      t_jacobian = self._jac_stationarity_constraints_wsspo(t_wsspo)
      t_s, t_u, t_v = tf.linalg.svd(t_jacobian)
      del t_u  # Unused, named for documentation only.
      if point_type == 'flat':
        # We impose 'flatness' on the 'Goldstone' and also
        # the requested-extra-flat directions.
        t_relevant_singular_values = t_s[
            -(num_goldstone + num_extraflat_directions):]
        t_loss = t_stat * tc_zoom + tf.math.reduce_sum(tf.math.square(
            t_relevant_singular_values))
        if verbose:
          report(f'L={t_loss.numpy():.6g} pot={t_pot.numpy():.10f} '
                 f'stat={t_stat.numpy():.6g} '
                 f'rsv={t_relevant_singular_values.numpy().round(7).tolist()}')
        return t_loss
      else:  # omega-extremum
        # Omega-extrema are characterized by none of the nullspace-directions
        # ("Goldstone plus omega-trajectory-tangent") of the Jacobian
        # having an omega-component, so we can minimize these alongside
        # stationarity-violation. Care has to be taken here, as TensorFlow's
        # SVD conventions differ from NumPy's in relevant ways. The right
        # singular vectors are the *columns* of t_v.
        t_nullspace_basis_omega_components = t_v[
            -1, -(num_goldstone + num_extraflat_directions):]
        t_omega_variation_loss = tf.math.reduce_sum(
            tf.math.square(t_nullspace_basis_omega_components))
        t_loss = t_stat * tc_zoom + tf.math.reduce_sum(tf.math.square(
            t_omega_variation_loss))
        if verbose:
          nbwc = t_nullspace_basis_omega_components.numpy().round(7).tolist()
          report(
              f'L={t_loss.numpy():.6g} pot={t_pot.numpy():.10f} '
              f'stat={t_stat.numpy():.6g} '
              f'nbwc={nbwc}')
        return t_loss
    opt_val, opt_delta_wsspo = mu.tf_minimize_v2(
        tf_loss, mu.tff64(numpy.zeros(self.dim_wsspo)),
        default_maxiter=10**4, default_gtol=1e-14)
    del opt_val  # Unused, named for documentation only.
    return self.v71_from_wsspo.dot(wsspo + (1 / zoom_factor) * opt_delta_wsspo)

  def wsspo_potential_and_stationarity(self, wsspo):
    """Computes potential and stationarity from wsspo-coordinates."""
    wsspo = numpy.asarray(wsspo)
    v71 = self.v71_from_wsspo.dot(wsspo)
    t_omega = mu.tff64(wsspo[-1])
    return self.potential_and_stationarity(v71[:-1], t_omega=t_omega)

  def v71o_potential_and_stationarity(self, v71o):
    """Computes potential and stationarity from v71o-coordinates."""
    v70 = self.e7.v70_from_v70o.dot(v71o[:-1])
    return self.potential_and_stationarity(v70, t_omega=mu.tff64(v71o[-1]))


def rk4_ode(df_ds, x0, s, ds, *f_args, f1=None):
  """Implements Runge-Kutta RK4 for numerically solving an ODE."""
  # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
  if f1 is None:
    f1 = df_ds(x0, s, *f_args)
  f2 = df_ds(x0 + 0.5 * ds * f1, s + 0.5 * ds, *f_args)
  f3 = df_ds(x0 + 0.5 * ds * f2, s + 0.5 * ds, *f_args)
  f4 = df_ds(x0 + ds * f3, s + ds, *f_args)
  return x0 + (ds / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)


def dyonic_ode_integrate(wsspo,
                         ds=0.01,
                         sugra=None,
                         stat_max=1e-12,
                         initial_walk_dir=None,
                         verbose=True,
                         report=print):
  """Integrates the omega-deformation ODE."""
  if sugra is None:
    sugra = SO8c_SUGRA()
  if initial_walk_dir is None:
    initial_walk_dir = numpy.array([0.0] * (len(wsspo) - 1) + [1.0])
  initial_walk_dir = numpy.asarray(initial_walk_dir)
  #
  def omega_step_details(wsspo, prev_marching_direction_wsspo,
                         verbose=verbose):
    nontrivial_directions, singular_values = (
        # Shape [dim_wsspo, n], usually [dim_wsspo, 1]
        sugra.get_wsspo_nongoldstone_omega_deformation_directions(wsspo))
    # Even if we have only a single nontrivial direction,
    # we need to correctly pick the sign of the way in which we walk.
    # So, let us rather use a method that should also work if we
    # encounter additional flat directions.
    # Let us use the convenient detail that we have an orthonormal
    # basis.
    proj_wsspo_nontrivial = nontrivial_directions.dot(nontrivial_directions.T)
    projected_marching_direction = proj_wsspo_nontrivial.dot(
        prev_marching_direction_wsspo)
    l2_length_projected_marching_direction = numpy.sqrt(
        projected_marching_direction.dot(projected_marching_direction))
    unit_marching_direction = projected_marching_direction / (
        l2_length_projected_marching_direction)
    # TODO(tfish): Might want to have an option to scale the marching-direction
    # differently here.
    if verbose:
      report(f'w/pi={wsspo[-1]/numpy.pi:.6f}, '
             f'dw/ds={unit_marching_direction[-1]:.6g}, '
             f'dv70/ds={(unit_marching_direction[:-1]**2).sum()**.5:.6g}')
    return unit_marching_direction, singular_values
  #
  def d_wsspo_ds(wsspo, s, prev_marching_direction_wsspo):
    unit_marching_direction, singular_values = omega_step_details(
        wsspo, prev_marching_direction_wsspo, verbose=(s == 0))
    del singular_values  # Unused, named for documentation only.
    return unit_marching_direction
  #
  pot_now, stat_now, wsspo_now = sugra.refined_pot_stat_wsspo(wsspo)
  walk_dir = initial_walk_dir
  while True:
    walk_dir, singular_values = omega_step_details(wsspo_now, walk_dir)
    #
    yield pot_now, stat_now, wsspo_now, singular_values  # <= Here is the yield!
    #
    wsspo_next = rk4_ode(d_wsspo_ds, wsspo_now, 0.0, ds, walk_dir, f1=walk_dir)
    pot, stat = sugra.wsspo_potential_and_stationarity(wsspo_next)
    if stat <= stat_max:
      wsspo_now, pot_now, stat_now = wsspo_next, pot, stat
    else:
      # Need to refine.
      pot_opt, stat_opt, wsspo_opt = sugra.refined_pot_stat_wsspo(wsspo_next)
      print(f'Realigned S={stat:.6g} -> S={stat_opt:.6g}')
      wsspo_now, pot_now, stat_now = wsspo_opt, pot_opt, stat_opt


def _incrementally_write_trajectory(
    sugra,
    data_filename,
    wsspo_start,
    initial_walk_dir,
    ds,
    stat_max,
    time_max_sec,
    abs_potential_max=500,
    verbose=True,
    report=print,
    re_raise_exceptions=False):
  """Internal, incrementally adds to a trajectory-logfile."""
  try:
    t0 = t_now = time.time()
    the_ode = dyonic_ode_integrate(
        wsspo_start, ds=ds, sugra=sugra,
        initial_walk_dir=initial_walk_dir,
        stat_max=stat_max)
    for n, (pot, stat, wsspo, singular_values) in enumerate(the_ode):
      t_prev, t_now = t_now, time.time()
      if verbose:
        report(f'# T={t_now - t0:.3f} s, +{t_now - t_prev:.3f} s: '
               f'w/pi={wsspo[-1]/numpy.pi:.6f}, P={pot:10f}, S={stat:.6g}')
      # We only resort to very robust methods for writing out data
      # at this point: Appending to a text file. The 'raw' log file data
      # will subsequently get processed losslessly into more compact
      # data storage format, and it then is ok to delete this text file.
      # Expected space consumption is:
      # ~200 numbers/row x ~20 bytes/number x one row per ~4 seconds x
      # 10k seconds x2 (for both directions) =
      # = order-of-magnitude 20 MB/solution, so for 200 solutions
      # about ~4 GB intermediate disk space (if analysis is done all in
      # one go, without intermediate repacking).
      v71 = sugra.v71_from_wsspo.dot(wsspo)
      numpy.save(data_filename + '_wsspo.npy', sugra.v71_from_wsspo)
      with open(data_filename, 'at') as h_out:  # Open-at-end every time.
        # General rule: Any data we write about solutions always
        # only refers to the 'classical' 35s+35c basis from
        # https://arxiv.org/abs/1906.00207
        # There may be more suitable orthonormal bases,
        # but the final verdict on which one is the most appropriate choice
        # has not been spoken yet.
        h_out.write(
            ','.join(
                map(repr,
                    [pot, stat, singular_values.size] +
                    v71.tolist() +
                    # We pad the singular values to 70,
                    # even though we will always have fewer of them.
                    # We need to do this since we may encounter points
                    # of different residual symmetry along a trajectory.
                    # If the number of 'goldstone' directions changes,
                    # so does the number of singular values. We however
                    # do need a homogeneous-row-length output.
                    [0.0] * (70 - singular_values.size) +
                    singular_values.tolist())) + '\n')
      if abs(pot) > abs_potential_max or t_now - t0 > time_max_sec:
        break
  except Exception as exn:  # pylint:disable=broad-except
    if verbose:
      report(f'Computation failed: {exn!r}')
    if re_raise_exceptions:
      raise


def continue_trajectory(
    data_filename,
    sugra=None,
    ds=0.01,
    stat_max=1e-12,
    time_max_sec=60 * 60 * 5,
    verbose=True,
    report=print,
    re_raise_exceptions=False):
  """Continues a trajectory, writing additional data to a new file."""
  if sugra is None:
    v71_from_wsspo = numpy.load(data_filename + '_wsspo.npy')
    sugra = SO8c_SUGRA(v71_from_wsspo=v71_from_wsspo)
  cont_filename = re.sub(
      # 'somefile.log' -> 'somefile_cont01.log' -> 'somefile_cont02.log'
      r'(.*?)(?:_cont(\d+))?[.]log$',
      lambda m: f'{m[1]}_cont{1 if m[2] is None else int(m[2]) + 1:02d}.log',
      data_filename)
  with open(data_filename, 'rt') as h_data:
    rows = list(h_data)
  final_rows = rows[-2:]
  final_rowdata = [[float(s) for s in row.split(',')] for row in final_rows]
  # We need to extract the final position, estimate the final walking-direction,
  # and check that the trajectory indeed only admits a single direction
  # to follow at its endpoint.
  v71_pre_final, v71_final = [numpy.array(rowdata[2: 2 + 71])
                              for rowdata in final_rowdata]
  final_last_singular_values = final_rowdata[-1][-2:]
  if (final_last_singular_values[0] < 1e-5 or
      final_last_singular_values[1] >= 1e-5):
    raise ValueError(
        'Cannot continue this trajectory. '
        'Final-point smallest restricted-Jacobian singular values are: '
        f'{final_last_singular_values}')
  wsspo_final, residuals_final, *_ = numpy.linalg.lstsq(sugra.v71_from_wsspo,
                                                        v71_final)
  assert (abs(residuals_final) < 1e-7).all(), (
      'Could not wsspo-decompose v71_final.')
  wsspo_pre_final, residuals_pre_final, *_ = (
      numpy.linalg.lstsq(sugra.v71_from_wsspo, v71_pre_final))
  assert (abs(residuals_pre_final) < 1e-10).all(), (
      'Could not wsspo-decompose v71_final.')
  delta_pos_wsspo = wsspo_final - wsspo_pre_final
  initial_walk_dir = delta_pos_wsspo / numpy.sqrt(
      delta_pos_wsspo.dot(delta_pos_wsspo))
  _incrementally_write_trajectory(
      sugra,
      cont_filename,
      wsspo_final,
      initial_walk_dir,
      ds,
      stat_max,
      time_max_sec,
      verbose=verbose,
      report=report,
      re_raise_exceptions=re_raise_exceptions)


def collect_trajectory_logs(trajectory_logs_glob,
                            trajectory_logs_glob_backwards=None):
  """Collects all the trajectories matching a glob.

  Args:
    trajectory_logs_glob: Shellglob pattern for trajectories.
      Files will be handled as if they were ordered lexically
      by (length, name), and then concatenated.
    trajectory_logs_glob_backwards: Optional shellglob
      for trajectory logs in the `backwards` direction.

  Returns:
    [num_entries, row_length]-numpy.ndarray, where every row is as logged, i.e.
    (potential, stationarity, num_relevant_singular_values,
     v70o[0], ..., v70o[69], omega, *singular_values),
    with `singular_values` being the singular values of the Jacobian,
    by decreasing magnitude, left-padded to a length of 70.
  """
  data = []
  for filename in sorted(glob.glob(trajectory_logs_glob),
                         key=lambda x: (len(x), x)):
    with open(filename, 'rt') as h_in:
      for row in h_in:
        data.append(numpy.array([float(x) for x in row.split(',')]))
  collected = numpy.stack(data, axis=0)
  if trajectory_logs_glob_backwards is None:
    return collected
  return numpy.concatenate(
      # We have to be careful here and remove the starting point from
      # the backwards-trajectory, as it matches the starting point
      # from the forwards-trajectory.
      [collect_trajectory_logs(trajectory_logs_glob_backwards)[:0:-1],
       collected],
      axis=0)


def trajectory_find_special_points(
    sugra,
    trajectory_data,
    refine=True):
  """Returns special points along a trajectory."""
  def find_special(column):
    vals = trajectory_data[:, column]
    delta_vals = vals[1:] - vals[:-1]
    # We want those points for which delta_vals change sign
    # from one entry to the next. Also, there, we would like to
    # then know these signs.
    delta_val_signs = delta_vals > 0
    change_places = delta_val_signs[1:] ^ delta_val_signs[:-1]
    change_indices = numpy.arange(1, len(vals) - 1)[change_places]
    local_val_extrema = trajectory_data[1:-1][change_places]
    # A `True` entry corresponds to a maximum, a `False` to a minimum.
    local_val_extrema_types = delta_val_signs[:-1][change_places]
    return tuple(zip(change_indices,
                     local_val_extrema_types,
                     local_val_extrema[:, column],
                     local_val_extrema))
  omega_local_extrema = find_special(3 + 70)
  # Extra-flat points are minima of the Jacobian
  jacobian_extraflat_direction_scale = trajectory_data[:, -2].mean()
  extraflat_points = [
      p for p in find_special(column=-2)  # Jacobian-extra-flat extrema.
      if p[2] < jacobian_extraflat_direction_scale * 0.1 and not p[1]]
  # Trick: If an index occurs both as an omega-extremum and an
  # extraflat point, make it an extraflat point by building
  # a type-by-index dict for the omega-extrema and then superseding
  # data in it with information from the correspinding
  # extraflat-point dict.
  all_special_points = sorted({
      **dict((idx, f'omega-{"max" if stype else "min"}')
             for idx, stype, *_ in omega_local_extrema),
      **dict((idx, 'flat') for idx, *_ in extraflat_points)}.items())
  result = []
  for idx, point_type in all_special_points:
    v71 = trajectory_data[idx, 3: 3 + 71]
    if refine:
      v71_refined = sugra.refine_special_point(
          v71, point_type='flat' if point_type == 'flat' else 'omega-extremum')
    else:
      v71_refined = v71  # 'This is refined enough'
    pot_refined, stat_refined = sugra.potential_and_stationarity(
        v71[:-1], t_omega=mu.tff64(v71[-1]))
    result.append(((idx, v71_refined[-1], point_type),
                   (pot_refined, stat_refined, v71_refined)))
  return result


def trajectory_v71_locate_omega(
    sugra,
    trajectory_data,
    omega,
    refine=True):
  """On a trajectory, locate the precise point for a specific omega-value.

  Args:
    sugra: The SO8c_SUGRA instance to use.
    trajectory_data: numpy.ndarray, trajectory data as returned by
      collect_trajectory_logs().
    omega: float, target omega to locate.
    refine: If true, refine interpolated position post-location.

  Returns:
    A pair `(locations, info)`, where `locations` is a numpy.ndarray
    of interpolated indices into the trajectory. If `locations` contains not
    exactly one value, then `info` is `None`. This case occurs both
    to signal that there is no such point, and also that the trajectory
    should be split to uniquely locate an omega-value. Otherwise, `info`
    is (potential, stationarity, v71).
  """
  omegas = trajectory_data[:, 3 + 70]
  omegas_lt = omegas < omega
  indices = numpy.arange(len(trajectory_data))
  transition_points = indices[:-1][omegas_lt[:-1] ^ omegas_lt[1:]]
  if len(transition_points) != 1:
    return transition_points + 0.5, None
  trajectory_spline = scipy.interpolate.interp1d(
      indices, trajectory_data.T, kind='cubic')
  omega_spline = scipy.interpolate.interp1d(
      indices, omegas, kind='cubic')
  # This is guaranteed to have smaller-than-reference omega.
  x_lower = transition_points[0]
  # Even if +1 were an an exact-match, this should have higher-than-reference
  # omega, as long as we are still approximately-linear.
  x_higher = transition_points[0] + 1.001
  x_omega = scipy.optimize.bisect(
      lambda x: omega_spline(x) - omega,
      x_lower, x_higher, xtol=1e-14)
  a_x_omega = numpy.array([x_omega])
  point_omega = trajectory_spline(x_omega)
  pot, stat, num_singular_values, *v71 = point_omega[:3 + 71]
  del num_singular_values  # Unused, named for documentation only.
  if not refine:
    return a_x_omega, (pot, stat, numpy.asarray(v71))
  pot_refined, stat_refined, wsspo_refined = sugra.refined_pot_stat_wsspo(
      sugra.wsspo_from_v70_and_omega(v71[:-1], x_omega))
  v71_refined = sugra.v71_from_wsspo.dot(wsspo_refined)
  # The index here is the interpolated index from which we refined
  # this solution.
  return a_x_omega, (pot_refined, stat_refined, v71_refined)


def trajectory_v71_locate_omegas(
    sugra,
    trajectory_data,
    omegas):
  """Locates specific omegas on a trajectory."""
  result = []
  for omega in omegas:
    located = trajectory_v71_locate_omega(
        sugra, trajectory_data, omega, refine=True)
    index_loci, optional_pot_stat_v71 = located
    if optional_pot_stat_v71 is not None:
      result.append(
          ((index_loci[0],
            optional_pot_stat_v71[-1][-1],
            f'omega/pi={omega/numpy.pi:.3f}'),
           optional_pot_stat_v71))
    elif index_loci.size:
      # We got multiple index-loci and need to split the trajectory
      # into pieces.
      splitting_index_loci = (
          [0] +
          [round(loc) for loc in (index_loci[:-1] + index_loci[1:]) / 2] +
          [len(trajectory_data)])
      for start_idx, end_idx in zip(splitting_index_loci[:-1],
                                    splitting_index_loci[1:]):
        piece_index_loci, the_point = trajectory_v71_locate_omega(
            sugra, trajectory_data[start_idx:end_idx], omega, refine=True)
        assert the_point is not None
        result.append(
            ((start_idx + piece_index_loci[0],
              omega,
              f'omega/pi={omega/numpy.pi:.3f}'),
             the_point))
  return sorted(result)  # We sort these special points by-index-on-trajectory.


def trajectory_get_story(sugra, trajectory_data,
                         # For writing .text, .tex, and .csv:
                         filename_out_stem=None):
  """Produces the relevant files 'to tell the story' of a trajectory."""
  pi = numpy.pi
  omega_min = trajectory_data[1:-1, 3 + 70].min()
  omega_max = trajectory_data[1:-1, 3 + 70].max()
  omegas_n_pi8 = numpy.arange(
      int(numpy.ceil(omega_min / (pi / 8))),
      1 + int(numpy.floor(omega_max / (pi / 8)))) * (pi / 8)
  special_omega_points = trajectory_v71_locate_omegas(
      sugra, trajectory_data, omegas_n_pi8)
  other_remarkable_points = trajectory_find_special_points(sugra,
                                                           trajectory_data)
  all_special_points = sorted(special_omega_points + other_remarkable_points)
  for ending in ('.tex', '.summary.tex', '.text', '.csv'):
    mu.rm(filename_out_stem + ending)
  result = []
  for (idx, omega, tag), (pot, stat, v71) in all_special_points:
    canon = sugra.canonicalize_equilibrium(v71[:-1])
    phys = sugra.get_physics(
        canon,
        {'tag': tag, 'omega/pi': omega / pi, 'trajectory_index': idx},
        t_omega=mu.tff64(v71[-1]))
    phys_text = sugra.show_physics_text(phys)
    omega_by_pi = omega / pi
    if omega == 0:
      in_brackets_text = None
    elif abs(round(omega_by_pi * 8) - omega_by_pi * 8) < 1e-8:
      in_brackets_text = f'{round(omega_by_pi * 8)}/8'
    else:
      in_brackets_text = f'{omega_by_pi:.6f}'
    phys_tex_summary, phys_tex_detail = sugra.show_physics_tex(
        phys, tag_kwargs=supergravity.default_tag_kwargs + (
            ('in_brackets', in_brackets_text),))
    if filename_out_stem:
      with open(filename_out_stem + '.csv', 'at') as h_csv:
        v71_str = ','.join(map(repr, v71))
        h_csv.write(f'{idx},{omega},"{tag}",{pot},{stat},{v71_str}\n')
      with open(filename_out_stem + '.text', 'at') as h_text:
        h_text.write(f'{phys_text}\n')
      # TODO(tfish):
      # The "summary" is not quite that useful yet, given that it
      # does not include omega in the table. Fix this.
      with open(filename_out_stem + '.summary.tex', 'at') as h_tex:
        h_tex.write(
            f'% index={idx}\n'
            f'{phys_tex_summary}\n')
      with open(filename_out_stem + '.tex', 'at') as h_tex:
        h_tex.write(
            f'\n\n%%% index={idx}, omega={omega:.8f}, '
            f'P={pot:.10f} S={stat:.6g}: {tag} %%%\n\n'
            f'{phys_tex_detail}\n')
    result.append(
        # Deliberately stay JSON-able here and use dict.
        dict(index=idx,
             tag=tag,
             potential=pot,
             stationarity=stat,
             v71=v71.tolist()))
  return result


def explore_around_special_point(
    sugra, v71o, csv_filename,
    radius=1e-4,
    seed=0,
    num_samples=100):
  """Explores a special point's neighborhood for connecting trajectories."""
  # TODO(tfish): This still needs some tweaking. It may well happen that
  # equivalent trajectories have somewhat-misaligned 'fingerprints'.
  num_goldstone, basis, proj_goldstone = (
      sugra.v70o_goldstone_basis_and_projector(v71o[:-1]))
  del proj_goldstone  # Unused below.
  basis_ext = numpy.pad(basis, [(0, 1), (0, 1)])
  basis_ext[-1, -1] = 1.0  # The omega-direction.
  tc_v71o_start = mu.tff64(v71o)
  tc_basis_ext = mu.tff64(basis_ext)
  # We take the radius as the inverse zoom factor, for it makes sense to
  # 'bring that to unit scale'.
  zoom_factor = 1 / radius
  tc_zoom, tc_inv_zoom = mu.tff64(zoom_factor), mu.tff64(1 / zoom_factor)
  #
  def tf_loss(t_params):
    # The rules of this optimization game are:
    # """With respect to non-goldstone (plus omega) directions,
    # we want to be on a sphere with prescribed radius.
    # With respect to the other ('trivial') directions,
    # we permit a small amount of moving-around, but we
    # multiplicatively punish going away from the starting point.
    # """
    # The role of this multiplicative factor is to eliminate
    # too much drift on the SO(8)-orbit due to numerical accidents
    # (such as better-after-rounding optimum with some large rotation).
    # This would be a problem for the subsequent step, where we then
    # have to make up our mind about 'which direction is `outgoing`'.
    t_delta71o = tf.einsum('an,n->a', tc_basis_ext, t_params)
    t_loss_factor_goldstone = (
        mu.tff64(1.0) +
        tf.math.reduce_sum(tf.math.square(t_params[:num_goldstone])))
    t_v71o = tc_v71o_start + tc_inv_zoom * t_delta71o
    t_v70o = t_v71o[:-1]
    t_omega = t_v71o[-1]
    *_, t_stat = sugra.tf_ext_sugra_tensors(
        sugra.tf_v70_from_v70o(t_v70o),
        t_omega=t_omega)
    # 'Distance from v71o must equal our radius'.
    t_relative_radius_sq = tf.math.reduce_sum(
        tf.math.square(t_params[num_goldstone:]))
    t_sphere_loss = tf.math.square(t_relative_radius_sq - mu.tff64(1))
    return (t_stat * tc_zoom + t_sphere_loss) * t_loss_factor_goldstone
  # TODO(tfish): Think about whether we may want to allow a tiny bit of motion
  # in the 'goldstone' directions to compensate for 'going off the SO(8) orbit'.
  # Intuitively, this seems to complicate things a lot, as we may then perhaps
  # encounter points that actually sit on the same trajectory, just a bit
  # spaced apart on it, and have difficulty telling them apart in terms
  # of fingerprinting. Also, thinking about linking, it actually should be
  # really unnecessary here.
  rng = mu.rng(seed)
  for num_sample in range(num_samples):
    del num_sample  # Unused, named for documentation only.
    num_params = 71 - num_goldstone
    opt_val, opt_params = mu.tf_minimize_v2(
        tf_loss,
        numpy.pad(
            rng.normal(size=num_params, scale=num_params**(-0.5)),
            [(num_goldstone, 0)]),
        default_maxiter=10**4, default_gtol=1e-14)
    del opt_val  # Unused, named for documentation only.
    v71o_sphere = v71o + (1 / zoom_factor) * basis_ext.dot(opt_params)
    pot_sphere, stat_sphere = sugra.v71o_potential_and_stationarity(v71o_sphere)
    with open(csv_filename, 'at') as h_csv:
      v71o_str = ','.join(map(repr, v71o_sphere))
      h_csv.write(
          f'{pot_sphere:.12f},{stat_sphere:.6g},'
          f'{v71o_sphere[-1]:.8f}{v71o_str}\n')


def plot_trajectory(
    tdata,
    title='',
    filename=None,
    show=True,
    show_legend=True):
  """Plots a trajectory."""
  # These trajectory-diagrams tend to become quite complicated.
  # Also, in order to work these out, we will generally have
  # changed step-size a few times. So, let us use
  # curve-length-in-omega-space as obtained from looking at samples
  # as the distance-measure.
  omegas = tdata[:, 3 + 70]
  omega_step_lengths = abs(omegas[1:] - omegas[:-1])
  omega_pi_walked = numpy.cumsum(
      numpy.pad(omega_step_lengths, [(1, 0)])) / numpy.pi
  pyplot.plot(omega_pi_walked, omegas / numpy.pi, '-k', label=r'$\omega/\pi$')
  pyplot.plot(omega_pi_walked, numpy.arcsinh(tdata[:, 0] / 6), '-g',
              label=r'$\operatorname{asinh}(V/6g^2)$')
  pyplot.plot(omega_pi_walked, numpy.arcsinh(tdata[:, -2]) / 10.0, '-b',
              label=r'$\operatorname{asinh}(E2)/10$')
  pyplot.grid()
  if show_legend:
    pyplot.legend(loc='best')
  pyplot.xlabel(r'$x=\int_0^s |d\omega/(\pi\,d\tilde s)|\,d\tilde s$')
  pyplot.title(title)
  if filename is not None:
    pyplot.savefig(filename)
  if show:
    pyplot.show()
  pyplot.close()


def analyze_omega_deformation(
    dest_dir,
    v70,
    sugra=None,
    omega0=0.0,
    ds=0.01,
    time_max_sec=60 * 60 * 6,
    abs_potential_max=500,
    stat_max=1e-12,
    re_raise_exceptions=False,
    verbose=True,
    report=print):
  """Analyzes the omega-deformation of SO(8) omega=0 solutions."""
  if sugra is None:
    sugra = SO8c_SUGRA()
  os.makedirs(dest_dir, exist_ok=True)
  v71o_start = numpy.concatenate(
      [sugra.e7.v70o_from_v70.dot(v70), [omega0]],
      axis=0)
  wsspo_start = sugra.wsspo_from_v71o(v71o_start)
  pot, stat = sugra.potential_and_stationarity(v70, t_omega=mu.tff64(omega0))
  if verbose:
    report(f'# Starting P={pot:.8f}, S={stat:.6g}')
  sol_id = supergravity.S_id(pot)
  sol_dir = os.path.join(dest_dir, sol_id)
  os.makedirs(sol_dir, exist_ok=True)
  timestamp = datetime.datetime.fromtimestamp(time.time())
  timestamp_filetag = timestamp.strftime('%Y_%m_%d__%H_%M_%S')
  for increasing_omega in (True, False):
    initial_walk_dir = numpy.array(
        [0.0] * (wsspo_start.size - 1) + [1.0 if increasing_omega else -1.0])
    data_filename = os.path.join(
        sol_dir,
        (f'omega_{omega0:7.4f}_{("neg", "pos")[int(increasing_omega)]}'
         f'_{timestamp_filetag}.log').replace(' ', ''))
    _incrementally_write_trajectory(
        sugra, data_filename, wsspo_start, initial_walk_dir, ds,
        stat_max, time_max_sec,
        abs_potential_max=abs_potential_max,
        verbose=verbose, report=report,
        re_raise_exceptions=re_raise_exceptions)
