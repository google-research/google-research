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

"""Tests for xc.gga."""

import tempfile
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from pyscf.dft import libxc
import pyscf.gto
from pyscf.lib import parameters

from symbolic_functionals.syfes.xc import gga
from symbolic_functionals.syfes.xc import utils


jax.config.update('jax_enable_x64', True)


class XCGGATest(parameterized.TestCase):

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
    ks.xc = 'pbe,pbe'
    ks.kernel()

    ao = ks._numint.eval_ao(ks.mol, coords=ks.grids.coords, deriv=1)
    self.rho_and_derivs = ks._numint.eval_rho2(
        ks.mol, ao, mo_coeff=ks.mo_coeff, mo_occ=ks.mo_occ, xctype='GGA')
    self.rho, rhogradx, rhogrady, rhogradz = self.rho_and_derivs
    self.sigma = rhogradx ** 2 + rhogrady ** 2 + rhogradz ** 2

    # construct a spin polarized density to test spin polarized case
    zeta = 0.2
    self.rho_and_derivs_a = 0.5 * (1 + zeta) * self.rho_and_derivs
    self.rho_and_derivs_b = 0.5 * (1 - zeta) * self.rho_and_derivs
    self.rhoa, self.rhob = self.rho_and_derivs_a[0], self.rho_and_derivs_b[0]
    self.sigma_aa = (0.5 * (1 + zeta)) ** 2 * self.sigma
    self.sigma_ab = ((0.5 * (1 + zeta))* (0.5 * (1 - zeta))) * self.sigma
    self.sigma_bb = (0.5 * (1 - zeta)) ** 2 * self.sigma

  @parameterized.parameters(
      ('gga_x_pbe', gga.e_x_pbe_unpolarized),
      ('gga_x_rpbe', gga.e_x_rpbe_unpolarized),
      ('gga_x_b88', gga.e_x_b88_unpolarized),
      ('gga_c_pbe', gga.e_c_pbe_unpolarized),
      ('hyb_gga_xc_b97',
       utils.function_sum(gga.e_x_b97_unpolarized, gga.e_c_b97_unpolarized)),
      )
  def test_gga_xc_unpolarized_against_libxc(self, xc_name, xc_fun):
    eps_xc_ref, (vrho_ref, vsigma_ref, _, _), _, _ = libxc.eval_xc(
        xc_name, self.rho, spin=0, relativity=0, deriv=1)
    e_xc_ref = eps_xc_ref * self.rho

    e_xc, (vrho, vsigma) = jax.vmap(jax.value_and_grad(xc_fun, argnums=(0, 1)))(
        self.rho, self.sigma)

    np.testing.assert_allclose(e_xc, e_xc_ref, atol=1e-12)
    np.testing.assert_allclose(vrho, vrho_ref, atol=1e-6)
    np.testing.assert_allclose(vsigma, vsigma_ref, atol=1e-6)

  @parameterized.parameters(
      ('gga_x_pbe', gga.e_x_pbe_polarized),
      ('gga_x_rpbe', gga.e_x_rpbe_polarized),
      ('gga_x_b88', gga.e_x_b88_polarized),
      ('gga_c_pbe', gga.e_c_pbe_polarized),
      ('hyb_gga_xc_b97',
       utils.function_sum(gga.e_x_b97_polarized, gga.e_c_b97_polarized)),
      )
  def test_gga_xc_polarized_against_libxc(self, xc_name, xc_fun):
    eps_xc_ref, (vrho_ref, vsigma_ref, _, _), _, _ = libxc.eval_xc(
        xc_name,
        (self.rho_and_derivs_a, self.rho_and_derivs_b),
        spin=1,
        relativity=0,
        deriv=1)
    e_xc_ref = eps_xc_ref * self.rho
    vrhoa_ref, vrhob_ref = vrho_ref[:, 0], vrho_ref[:, 1]
    vsigma_aa_ref, vsigma_ab_ref, vsigma_bb_ref = (
        vsigma_ref[:, 0], vsigma_ref[:, 1], vsigma_ref[:, 2])

    e_xc, grads = jax.vmap(jax.value_and_grad(xc_fun, argnums=(0, 1, 2, 3, 4)))(
        self.rhoa, self.rhob, self.sigma_aa, self.sigma_ab, self.sigma_bb)
    vrhoa, vrhob, vsigma_aa, vsigma_ab, vsigma_bb = grads

    np.testing.assert_allclose(e_xc, e_xc_ref, atol=1e-12)
    np.testing.assert_allclose(vrhoa, vrhoa_ref, atol=1e-6)
    np.testing.assert_allclose(vrhob, vrhob_ref, atol=1e-6)
    np.testing.assert_allclose(vsigma_aa, vsigma_aa_ref, atol=5e-5)
    np.testing.assert_allclose(vsigma_ab, vsigma_ab_ref, atol=5e-5)
    np.testing.assert_allclose(vsigma_bb, vsigma_bb_ref, atol=5e-5)

  # NOTE(htm): For wB97X-V, there is max absolute (relative) difference on
  # the order of .005 (0.01) between vsigma evaluated here and by libxc.
  # Since e_xc and vrho match well, and the comparison to B97 is good,
  # it is suspected that the error comes from libxc.
  @parameterized.parameters(
      ('hyb_gga_xc_wb97x_v', gga.e_xc_wb97xv_unpolarized)
      )
  def test_wb97xv_unpolarized_against_libxc(self, xc_name, xc_fun):
    eps_xc_ref, (vrho_ref, vsigma_ref, _, _), _, _ = libxc.eval_xc(
        xc_name, self.rho, spin=0, relativity=0, deriv=1)
    e_xc_ref = eps_xc_ref * self.rho

    e_xc, (vrho, vsigma) = jax.vmap(jax.value_and_grad(xc_fun, argnums=(0, 1)))(
        self.rho, self.sigma)

    np.testing.assert_allclose(e_xc, e_xc_ref, atol=1e-12)
    np.testing.assert_allclose(vrho, vrho_ref, atol=1e-6)
    np.testing.assert_allclose(vsigma, vsigma_ref, atol=1e-2)

  @parameterized.parameters(
      ('hyb_gga_xc_wb97x_v', gga.e_xc_wb97xv_polarized)
      )
  def test_wb97xv_polarized_against_libxc(self, xc_name, xc_fun):
    eps_xc_ref, (vrho_ref, vsigma_ref, _, _), _, _ = libxc.eval_xc(
        xc_name,
        (self.rho_and_derivs_a, self.rho_and_derivs_b),
        spin=1,
        relativity=0,
        deriv=1)
    e_xc_ref = eps_xc_ref * self.rho
    vrhoa_ref, vrhob_ref = vrho_ref[:, 0], vrho_ref[:, 1]
    vsigma_aa_ref, vsigma_ab_ref, vsigma_bb_ref = (
        vsigma_ref[:, 0], vsigma_ref[:, 1], vsigma_ref[:, 2])

    e_xc, grads = jax.vmap(jax.value_and_grad(xc_fun, argnums=(0, 1, 2, 3, 4)))(
        self.rhoa, self.rhob, self.sigma_aa, self.sigma_ab, self.sigma_bb)
    vrhoa, vrhob, vsigma_aa, vsigma_ab, vsigma_bb = grads

    np.testing.assert_allclose(e_xc, e_xc_ref, atol=1e-12)
    np.testing.assert_allclose(vrhoa, vrhoa_ref, atol=1e-6)
    np.testing.assert_allclose(vrhob, vrhob_ref, atol=1e-6)
    np.testing.assert_allclose(vsigma_aa, vsigma_aa_ref, atol=1.5e-2)
    np.testing.assert_allclose(vsigma_ab, vsigma_ab_ref, atol=1.5e-2)
    np.testing.assert_allclose(vsigma_bb, vsigma_bb_ref, atol=1.5e-2)

if __name__ == '__main__':
  absltest.main()
