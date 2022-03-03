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

"""Tests for scf.scf."""

import os
import tempfile
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from pyscf.lib import parameters
import tensorflow.compat.v1 as tf

from symbolic_functionals.syfes.scf import scf
from symbolic_functionals.syfes.symbolic import xc_functionals
from symbolic_functionals.syfes.xc import mgga
from symbolic_functionals.syfes.xc import utils
from symbolic_functionals.syfes.xc import xc

jax.config.update('jax_enable_x64', True)


class SCFTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    parameters.TMPDIR = tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)

  def test_parse_xyz(self):
    xyz_path = os.path.join(flags.FLAGS.test_tmpdir, 'test.xyz')
    with tf.io.gfile.GFile(xyz_path, 'w') as f:
      f.write('\n0 1\nO  0. 0. 0.\nH  0. -0.757 0.587\nH 0. 0.757 0.587\n')

    atom, charge, spin = scf.parse_xyz(xyz_path)

    self.assertLen(atom.split(';'), 3)
    self.assertEqual(charge, 0)
    self.assertEqual(spin, 0)

  # expected values for Etot, Exc and Exx are computed by external PySCF
  @parameterized.parameters(
      ('pbe,pbe', 0, 0, (17812,), (6, 17812), {
          'Etot': -1.1601348451265638,
          'Exc': -0.6899737187913197,
          'Exx': -0.6583555393862027,
          'Exxlr': 0.0,
          'Enlc': 0.0
      }),
      ('pbe,pbe', -1, 1, (20048,), (2, 6, 20048), {
          'Etot': -1.0336723063997342,
          'Exc': -0.8723776781828819,
          'Exx': -0.8141180655850809,
          'Exxlr': 0.0,
          'Enlc': 0.0
      }),
      ('wb97m_v', 0, 0, (17812,), (6, 17812), {
          'Etot': -1.1537971220466094,
          'Exc': -0.6829720417857192,
          'Exx': -0.6577521311181448,
          'Exxlr': -0.29729171800068577,
          'Enlc': 0.00891190761270658
      }),
      ('wb97m_v', -1, 1, (20048,), (2, 6, 20048), {
          'Etot': -1.007161017404796,
          'Exc': -0.8544883116165207,
          'Exx': -0.8220018420576689,
          'Exxlr': -0.40928226576050875,
          'Enlc': 0.012704771979755908
      }),
  )
  def test_scf_calculation_with_pyscf(self, xc_name, charge, spin,
                                      expected_weights_shape,
                                      expected_rho_shape, expected_energies):
    res = scf.run_scf_for_mol(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        charge=charge,
        spin=spin,
        xc=xc_name,
        basis='def2svpd')

    self.assertCountEqual(
        list(res.keys()),
        scf.SCF_SCALAR_RESULTS + ['rho', 'weights'])
    self.assertTrue(res['converged'])
    self.assertEqual(res['weights'].shape, expected_weights_shape)
    self.assertEqual(res['rho'].shape, expected_rho_shape)
    for energy in ['Etot', 'Exc', 'Exx', 'Exxlr', 'Enlc']:
      np.testing.assert_allclose(res[energy], expected_energies[energy])

  @parameterized.parameters(
      ('lda', 'lda_x,lda_c_pw', 0, 0),
      ('lda', 'lda_x,lda_c_pw', -1, 1),
      ('pbe', 'pbe,pbe', 0, 0),
      ('pbe', 'pbe,pbe', -1, 1),
      ('b97', 'hyb_gga_xc_b97', 0, 0),
      ('b97', 'hyb_gga_xc_b97', -1, 1),
      ('wb97x_v', 'wb97x_v', 0, 0),
      ('wb97x_v', 'wb97x_v', -1, 1),
      ('b97m_v', 'b97m_v', 0, 0),
      ('b97m_v', 'b97m_v', -1, 1),
      ('wb97m_v', 'wb97m_v', 0, 0),
      ('wb97m_v', 'wb97m_v', -1, 1),
  )
  def test_scf_calculation_with_custom_xc_default_params(
      self, xc_name, xc_name_libxc, charge, spin):
    hybrid_coeff, rsh_params = utils.get_hybrid_rsh_params(xc_name)
    res_libxc = scf.run_scf_for_mol(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        charge=charge,
        spin=spin,
        xc=xc_name_libxc,
        basis='def2svpd')

    res_custom = scf.run_scf_for_mol(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        charge=charge,
        spin=spin,
        xc=xc_name,
        xc_fun=xc.make_eval_xc(xc_name),
        hybrid_coeff=hybrid_coeff,
        rsh_params=rsh_params,
        basis='def2svpd')

    for energy in ['Etot', 'Exc', 'Exx', 'Exxlr', 'Enlc']:
      self.assertAlmostEqual(res_libxc[energy], res_custom[energy], delta=2e-6)

  @parameterized.parameters((0, 0), (-1, 1),)
  def test_scf_calculation_with_custom_xc_custom_params(self, charge, spin):
    hybrid_coeff, rsh_params = utils.get_hybrid_rsh_params('b97m_v')
    res_libxc = scf.run_scf_for_mol(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        charge=charge,
        spin=spin,
        xc='b97m_v',
        basis='def2svpd')

    res_custom = scf.run_scf_for_mol(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        charge=charge,
        spin=spin,
        xc='b97m_v',
        xc_fun=xc.make_eval_xc('wb97m_v', params=mgga.B97MV_PARAMS),
        hybrid_coeff=hybrid_coeff,
        rsh_params=rsh_params,
        basis='def2svpd')

    for energy in ['Etot', 'Exc', 'Exx', 'Exxlr', 'Enlc']:
      self.assertAlmostEqual(res_libxc[energy], res_custom[energy], delta=2e-6)

  @parameterized.parameters((0, 0), (-1, 1),)
  def test_scf_calculation_with_symbolic_functional(self, charge, spin):
    hybrid_coeff, rsh_params = utils.get_hybrid_rsh_params('wb97m_v')
    res_libxc = scf.run_scf_for_mol(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        charge=charge,
        spin=spin,
        xc='wb97m_v',
        basis='def2svpd')

    res_custom = scf.run_scf_for_mol(
        atom='H  0. 0. 0.;H  0. 0. 0.74',
        charge=charge,
        spin=spin,
        xc='wb97m_v',
        xc_fun=xc_functionals.wb97mv_short.make_eval_xc(
            omega=rsh_params[0],
            **xc_functionals.WB97MV_PARAMETERS_UTRANSFORM),
        hybrid_coeff=hybrid_coeff,
        rsh_params=rsh_params,
        basis='def2svpd')

    for energy in ['Etot', 'Exc', 'Exx', 'Exxlr', 'Enlc']:
      self.assertAlmostEqual(res_libxc[energy], res_custom[energy], delta=2e-6)

if __name__ == '__main__':
  absltest.main()
