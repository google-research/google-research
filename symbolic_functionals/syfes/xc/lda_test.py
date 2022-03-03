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

"""Tests for xc.lda."""

import tempfile
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from pyscf.dft import libxc
import pyscf.gto
from pyscf.lib import parameters

from symbolic_functionals.syfes.xc import lda

jax.config.update('jax_enable_x64', True)


class XCLDATest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    parameters.TMPDIR = tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)

    mol = pyscf.gto.M(
        atom='''O  0.   0.       0.
        H  0.   -0.757   0.587
        H  0.   0.757    0.587
        ''',
        basis='def2svpd',
        verbose=1)

    ks = pyscf.dft.RKS(mol)
    ks.xc = 'lda,lda'
    ks.kernel()

    ao = ks._numint.eval_ao(ks.mol, coords=ks.grids.coords, deriv=0)
    self.rho = ks._numint.eval_rho2(
        ks.mol, ao, mo_coeff=ks.mo_coeff, mo_occ=ks.mo_occ, xctype='LDA')

    # construct a spin polarized density to test spin polarized case
    zeta = 0.2
    self.rhoa = 0.5 * (1 + zeta) * self.rho
    self.rhob = 0.5 * (1 - zeta) * self.rho

  @parameterized.parameters(
      ('lda_x', lda.e_x_lda_unpolarized),
      ('lda_c_pw', lda.e_c_lda_unpolarized),
      )
  def test_lda_xc_unpolarized_against_libxc(self, xc_name, xc_fun):
    eps_xc_ref, (vrho_ref, _, _, _), _, _ = libxc.eval_xc(
        xc_name, self.rho, spin=0, relativity=0, deriv=1)
    e_xc_ref = eps_xc_ref * self.rho

    e_xc, vrho = jax.vmap(jax.value_and_grad(xc_fun))(self.rho)

    np.testing.assert_allclose(e_xc, e_xc_ref)
    np.testing.assert_allclose(vrho, vrho_ref)

  @parameterized.parameters(
      ('lda_x', lda.e_x_lda_polarized),
      ('lda_c_pw', lda.e_c_lda_polarized),
      )
  def test_lda_xc_polarized_against_libxc(self, xc_name, xc_fun):
    eps_xc_ref, (vrho_ref, _, _, _), _, _ = libxc.eval_xc(
        xc_name, (self.rhoa, self.rhob), spin=1, relativity=0, deriv=1)
    e_xc_ref = eps_xc_ref * self.rho
    vrhoa_ref, vrhob_ref = vrho_ref[:, 0], vrho_ref[:, 1]

    e_xc, (vrhoa, vrhob) = jax.vmap(jax.value_and_grad(xc_fun, argnums=(0, 1)))(
        self.rhoa, self.rhob)

    np.testing.assert_allclose(e_xc, e_xc_ref)
    np.testing.assert_allclose(vrhoa, vrhoa_ref)
    np.testing.assert_allclose(vrhob, vrhob_ref)

  # reference values are pre-computed using external codes
  @parameterized.parameters(
      (0., 0., 0.),
      (0.5, -0.01588696391353700, -0.0168974057764630),
      (1., -0.0345539265331996, -0.0366463870651906),
      )
  def test_decomposed_e_c_lda_unpolarized_use_jax(
      self, rho, expected_e_c_ss, expected_e_c_os):
    e_c_ss, e_c_os = lda.decomposed_e_c_lda_unpolarized(rho, use_jax=True)
    np.testing.assert_allclose(e_c_ss, expected_e_c_ss)
    np.testing.assert_allclose(e_c_os, expected_e_c_os)

  @parameterized.parameters(
      (0., 0., 0.),
      (0.5, -0.01588696391353700, -0.0168974057764630),
      (1., -0.0345539265331996, -0.0366463870651906),
      )
  def test_decomposed_e_c_lda_unpolarized_not_use_jax(
      self, rho, expected_e_c_ss, expected_e_c_os):
    e_c_ss, e_c_os = lda.decomposed_e_c_lda_unpolarized(rho, use_jax=False)
    np.testing.assert_allclose(e_c_ss, expected_e_c_ss)
    np.testing.assert_allclose(e_c_os, expected_e_c_os)

  # reference values are pre-computed using external codes
  @parameterized.parameters(
      (0., 0., 0., 0., 0.),
      (0., 0.5, 0., -0.0172769632665998, 0.),
      (0.5, 0., -0.0172769632665998, 0., 0.),
      (0.5, 1., -0.0172769632665998, -0.0374279447531902, -0.05301827865038937),
      (1., 0.5, -0.0374279447531902, -0.0172769632665998, -0.05301827865038937),
      (1., 1., -0.0374279447531902, -0.0374279447531902, -0.0791659658806080),
      )
  def test_decomposed_e_c_lda_polarized_use_jax(
      self, rhoa, rhob, expected_e_c_aa, expected_e_c_bb, expected_e_c_ab):
    e_c_aa, e_c_bb, e_c_ab = lda.decomposed_e_c_lda_polarized(
        rhoa, rhob, use_jax=True)
    np.testing.assert_allclose(e_c_aa, expected_e_c_aa)
    np.testing.assert_allclose(e_c_bb, expected_e_c_bb)
    np.testing.assert_allclose(e_c_ab, expected_e_c_ab)

  @parameterized.parameters(
      (0., 0., 0., 0., 0.),
      (0., 0.5, 0., -0.0172769632665998, 0.),
      (0.5, 0., -0.0172769632665998, 0., 0.),
      (0.5, 1., -0.0172769632665998, -0.0374279447531902, -0.05301827865038937),
      (1., 0.5, -0.0374279447531902, -0.0172769632665998, -0.05301827865038937),
      (1., 1., -0.0374279447531902, -0.0374279447531902, -0.0791659658806080),
      )
  def test_decomposed_e_c_lda_polarized_not_use_jax(
      self, rhoa, rhob, expected_e_c_aa, expected_e_c_bb, expected_e_c_ab):
    e_c_aa, e_c_bb, e_c_ab = lda.decomposed_e_c_lda_polarized(
        rhoa, rhob, use_jax=False)
    np.testing.assert_allclose(e_c_aa, expected_e_c_aa)
    np.testing.assert_allclose(e_c_bb, expected_e_c_bb)
    np.testing.assert_allclose(e_c_ab, expected_e_c_ab)


if __name__ == '__main__':
  absltest.main()
