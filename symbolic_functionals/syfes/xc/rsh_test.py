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
from pyscf import dft
from pyscf.lib import parameters

from symbolic_functionals.syfes.scf import scf
from symbolic_functionals.syfes.xc import lda
from symbolic_functionals.syfes.xc import rsh

jax.config.update('jax_enable_x64', True)


class XCRSHTest(parameterized.TestCase):

  # reference values are pre-computed using PySCF v1.7.1
  # with xc_code = 'lda_x_erf'
  @parameterized.parameters(
      (0., 0., 0.),
      (0.5, -0.21717557611164048, -0.62400285),
      (1., -0.5831486070796371, -0.82474832),
      (2., -1.5445435889264496, -1.07879391),
  )
  def test_e_x_lda_erf_unpolarized_use_jax(
      self, rho, expected_e_x, expected_vrho):

    def e_x_lda_erf(rho):
      return lda.e_x_lda_unpolarized(rho) * rsh.f_rsh(
          rho, omega=0.3, use_jax=True)

    e_x, vrho = jax.value_and_grad(e_x_lda_erf)(rho)
    np.testing.assert_allclose(e_x, expected_e_x)
    np.testing.assert_allclose(vrho, expected_vrho)

  @parameterized.parameters(
      (0., 0.),
      (0.5, -0.21717557611164048),
      (1., -0.5831486070796371),
      (2., -1.5445435889264496),
  )
  def test_e_x_lda_erf_unpolarized_not_use_jax(self, rho, expected_e_x):
    np.testing.assert_allclose(
        lda.e_x_lda_unpolarized(rho) * rsh.f_rsh(
            rho, omega=0.3, use_jax=False), expected_e_x)


class XCWB97MVTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    parameters.TMPDIR = tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)

  @parameterized.parameters((0, 0), (-1, 1))
  def test_wb97mv_xc_energy_decomposition(self, charge, spin):
    _, alpha, beta = dft.libxc.rsh_coeff('wb97m_v')
    res = scf.run_scf_for_mol(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        charge=charge,
        spin=spin,
        xc='wb97m_v',
        basis='def2svpd')
    rho = res['rho'][0] if spin == 0 else np.sum(res['rho'][:, 0, :], axis=0)

    eps_xc_sl = dft.libxc.eval_xc(
        'hyb_mgga_xc_wb97m_v', res['rho'], spin=spin, relativity=0, deriv=1)[0]
    sr_xc_energy = np.sum(res['weights'] * rho * eps_xc_sl)
    sr_xc_energy_from_decomposition = rsh.get_exc_sl(
        alpha=alpha,
        beta=beta,
        exc=res['Exc'],
        exx=res['Exx'],
        exx_lr=res['Exxlr'],
        enlc=res['Enlc'])

    self.assertAlmostEqual(sr_xc_energy, sr_xc_energy_from_decomposition)


if __name__ == '__main__':
  absltest.main()
