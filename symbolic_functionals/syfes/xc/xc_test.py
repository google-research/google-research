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

"""Tests for xc.xc."""

import tempfile
from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from pyscf import dft
from pyscf import gto
from pyscf.lib import parameters

from symbolic_functionals.syfes.symbolic import xc_functionals
from symbolic_functionals.syfes.xc import utils
from symbolic_functionals.syfes.xc import xc

jax.config.update('jax_enable_x64', True)

# NOTE(htm): In following tests VV10 nonlocal correlation is not included for
# functionals like wB97M-V to reduce computational cost


class CustomXCUnpolarizedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    parameters.TMPDIR = tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)
    self.mol = gto.M(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        basis='def2svpd',
        charge=0,
        spin=0)

  @parameterized.parameters(
      ('lda', 'lda_x,lda_c_pw'),
      ('pbe', 'pbe,pbe'),
      ('b97', 'hyb_gga_xc_b97'),
      ('wb97x_v', 'wb97x_v'),
      ('b97m_v', 'b97m_v'),
      ('wb97m_v', 'wb97m_v'),
      )
  def test_scf_calculations_with_custom_xc(self, xc_name, xc_code):
    hybrid_coeff, rsh_params = utils.get_hybrid_rsh_params(xc_name)
    ks_ref = dft.RKS(self.mol)
    ks_ref.xc = xc_code
    etot_ref = ks_ref.kernel()

    ks = dft.RKS(self.mol)
    ks.define_xc_(
        xc.make_eval_xc(xc_name),
        xctype='MGGA',
        hyb=hybrid_coeff,
        rsh=rsh_params)
    etot = ks.kernel()

    logging.info('Etot = %f, Etot_libxc = %f, diff = %f',
                 etot, etot_ref, abs(etot - etot_ref))
    self.assertAlmostEqual(etot, etot_ref, delta=2e-6)

  @parameterized.parameters(
      ('hyb_gga_xc_b97',
       xc_functionals.b97_x2,
       xc_functionals.B97_PARAMETERS),
      ('hyb_gga_xc_b97',
       xc_functionals.b97_x2_short,
       xc_functionals.B97_PARAMETERS_UTRANSFORM),
      ('wb97m_v',
       xc_functionals.wb97mv,
       xc_functionals.WB97MV_PARAMETERS),
      ('wb97m_v',
       xc_functionals.wb97mv_short,
       xc_functionals.WB97MV_PARAMETERS_UTRANSFORM),
      )
  def test_scf_calculations_with_symbolic_xc(self, xc_code, functional, params):
    hybrid_coeff, rsh_params = utils.get_hybrid_rsh_params(xc_code)
    ks_ref = dft.RKS(self.mol)
    ks_ref.xc = xc_code
    etot_ref = ks_ref.kernel()

    ks = dft.RKS(self.mol)
    ks.define_xc_(
        functional.make_eval_xc(omega=rsh_params[0], **params),
        xctype='MGGA',
        hyb=hybrid_coeff,
        rsh=rsh_params)
    etot = ks.kernel()

    logging.info('Etot = %f, Etot_libxc = %f, diff = %f',
                 etot, etot_ref, abs(etot - etot_ref))
    self.assertAlmostEqual(etot, etot_ref, delta=2e-6)


class CustomXCPolarizedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    parameters.TMPDIR = tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)
    self.mol = gto.M(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        basis='def2svpd',
        charge=-1,
        spin=1,
        verbose=1)

  @parameterized.parameters(
      ('lda', 'lda_x,lda_c_pw'),
      ('pbe', 'pbe,pbe'),
      ('b97', 'hyb_gga_xc_b97'),
      ('wb97x_v', 'wb97x_v'),
      ('b97m_v', 'b97m_v'),
      ('wb97m_v', 'wb97m_v'),
      )
  def test_scf_calculations_with_custom_xc(self, xc_name, xc_code):
    hybrid_coeff, rsh_params = utils.get_hybrid_rsh_params(xc_name)
    ks_ref = dft.UKS(self.mol)
    ks_ref.xc = xc_code
    etot_ref = ks_ref.kernel()

    ks = dft.UKS(self.mol)
    ks.define_xc_(
        xc.make_eval_xc(xc_name),
        xctype='MGGA',
        hyb=hybrid_coeff,
        rsh=rsh_params)
    etot = ks.kernel()

    logging.info('Etot = %f, Etot_libxc = %f, diff = %f',
                 etot, etot_ref, abs(etot - etot_ref))
    self.assertAlmostEqual(etot, etot_ref, delta=2e-6)

  @parameterized.parameters(
      ('hyb_gga_xc_b97',
       xc_functionals.b97_x2,
       xc_functionals.B97_PARAMETERS),
      ('hyb_gga_xc_b97',
       xc_functionals.b97_x2_short,
       xc_functionals.B97_PARAMETERS_UTRANSFORM),
      ('wb97m_v',
       xc_functionals.wb97mv,
       xc_functionals.WB97MV_PARAMETERS),
      ('wb97m_v',
       xc_functionals.wb97mv_short,
       xc_functionals.WB97MV_PARAMETERS_UTRANSFORM),
      )
  def test_scf_calculations_with_symbolic_xc(self, xc_code, functional, params):
    hybrid_coeff, rsh_params = utils.get_hybrid_rsh_params(xc_code)
    ks_ref = dft.UKS(self.mol)
    ks_ref.xc = xc_code
    etot_ref = ks_ref.kernel()

    ks = dft.UKS(self.mol)
    ks.define_xc_(
        functional.make_eval_xc(omega=rsh_params[0], **params),
        xctype='MGGA',
        hyb=hybrid_coeff,
        rsh=rsh_params)
    etot = ks.kernel()

    logging.info('Etot = %f, Etot_libxc = %f, diff = %f',
                 etot, etot_ref, abs(etot - etot_ref))
    self.assertAlmostEqual(etot, etot_ref, delta=2e-6)


if __name__ == '__main__':
  absltest.main()
