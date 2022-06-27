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

"""Tests for xc.mgga."""

import tempfile
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from pyscf.dft import libxc
import pyscf.gto
from pyscf.lib import parameters

from symbolic_functionals.syfes.xc import mgga

jax.config.update('jax_enable_x64', True)


class XCMGGATest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    parameters.TMPDIR = tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)

    mol = pyscf.gto.M(
        atom="""O  0.   0.       0.
        H  0.   -0.757   0.587
        H  0.   0.757    0.587
        """,
        basis='def2svpd',
        verbose=1)

    ks = pyscf.dft.RKS(mol)
    ks.xc = 'pbe,pbe'
    ks.kernel()

    ao = ks._numint.eval_ao(ks.mol, coords=ks.grids.coords, deriv=2)
    self.weights = ks.grids.weights
    self.rho_and_derivs = ks._numint.eval_rho2(
        ks.mol, ao, mo_coeff=ks.mo_coeff, mo_occ=ks.mo_occ, xctype='MGGA')
    self.rho, gradx, grady, gradz, self.lapl, self.tau = self.rho_and_derivs
    self.sigma = gradx**2 + grady**2 + gradz**2

    # construct a spin polarized density to test spin polarized case
    zeta = 0.2
    self.rho_and_derivs_a = 0.5 * (1 + zeta) * self.rho_and_derivs
    self.rho_and_derivs_b = 0.5 * (1 - zeta) * self.rho_and_derivs
    self.rhoa, self.rhob = self.rho_and_derivs_a[0], self.rho_and_derivs_b[0]
    self.sigma_aa = (0.5 * (1 + zeta))**2 * self.sigma
    self.sigma_ab = ((0.5 * (1 + zeta)) * (0.5 * (1 - zeta))) * self.sigma
    self.sigma_bb = (0.5 * (1 - zeta))**2 * self.sigma
    self.tau_a, self.tau_b = self.rho_and_derivs_a[5], self.rho_and_derivs_b[5]

  # NOTE(htm): The parameters used in mgga.e_xc_wb97mv_* functions are checked
  # againast libxc, but the resulting e_xc shows small deviations from libxc
  # results (on the order of 1e-3). The reason for the deviation is uncertain.
  # The difference on the integrated E_xc energy is very small (< 1e-5 Hartree).
  # In the following tests the tolerance for assert_allclose has taken this
  # deviation into account.
  @parameterized.parameters(
      ('hyb_mgga_xc_wb97m_v', mgga.e_xc_wb97mv_unpolarized),
      ('mgga_xc_b97m_v', mgga.e_xc_b97mv_unpolarized)
      )
  def test_mgga_xc_unpolarized_against_libxc(self, xc_name, xc_fun):
    eps_xc_ref, rhograds_ref, _, _ = libxc.eval_xc(
        xc_name, self.rho, spin=0, relativity=0, deriv=1)
    vrho_ref, vsigma_ref, _, vtau_ref = rhograds_ref
    e_xc_ref = eps_xc_ref * self.rho

    e_xc = xc_fun(self.rho, self.sigma, self.tau)
    e_xc, grads = jax.vmap(jax.value_and_grad(xc_fun, argnums=(0, 1, 2)))(
        self.rho, self.sigma, self.tau)
    vrho, vsigma, vtau = grads

    np.testing.assert_allclose(
        np.sum(e_xc * self.weights), np.sum(e_xc_ref * self.weights), atol=1e-5)
    np.testing.assert_allclose(e_xc, e_xc_ref, rtol=1e-2, atol=1.e-12)
    np.testing.assert_allclose(vrho, vrho_ref, atol=2e-4)
    # TODO(leyley,htm): Fix vsigma with libxc 5.
    del vsigma, vsigma_ref
    # np.testing.assert_allclose(vsigma, vsigma_ref, atol=1e-2)
    np.testing.assert_allclose(vtau, vtau_ref, atol=1e-5)

  @parameterized.parameters(
      ('hyb_mgga_xc_wb97m_v', mgga.e_xc_wb97mv_polarized),
      ('mgga_xc_b97m_v', mgga.e_xc_b97mv_polarized)
      )
  def test_wb97xv_polarized_against_libxc(self, xc_name, xc_fun):
    eps_xc_ref, rhograds_ref, _, _ = libxc.eval_xc(
        xc_name, (self.rho_and_derivs_a, self.rho_and_derivs_b),
        spin=1, relativity=0, deriv=1)
    e_xc_ref = eps_xc_ref * self.rho
    vrho_ref, vsigma_ref, _, vtau_ref = rhograds_ref

    e_xc, grads = jax.vmap(jax.value_and_grad(
        xc_fun, argnums=(0, 1, 2, 3, 4, 5, 6)))(
            self.rhoa, self.rhob,
            self.sigma_aa, self.sigma_ab, self.sigma_bb,
            self.tau_a, self.tau_b)
    vrhoa, vrhob, vsigma_aa, vsigma_ab, vsigma_bb, vtau_a, vtau_b = grads

    np.testing.assert_allclose(
        np.sum(e_xc * self.weights), np.sum(e_xc_ref * self.weights), atol=1e-5)
    np.testing.assert_allclose(e_xc, e_xc_ref, rtol=1e-2, atol=1.e-12)
    np.testing.assert_allclose(vrhoa, vrho_ref[:, 0], atol=2e-4)
    np.testing.assert_allclose(vrhob, vrho_ref[:, 1], atol=2e-4)
    # TODO(leyley,htm): Fix vsigma with libxc 5.
    del vsigma_aa, vsigma_bb
    # np.testing.assert_allclose(vsigma_aa, vsigma_ref[:, 0], atol=1e-2)
    np.testing.assert_allclose(vsigma_ab, vsigma_ref[:, 1], atol=1e-2)
    # np.testing.assert_allclose(vsigma_bb, vsigma_ref[:, 2], atol=1e-2)
    np.testing.assert_allclose(vtau_a, vtau_ref[:, 0], atol=1e-5)
    np.testing.assert_allclose(vtau_b, vtau_ref[:, 1], atol=1e-5)


if __name__ == '__main__':
  absltest.main()
