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

"""SCF calculation using PySCF."""

import time
from absl import logging
import numpy as np
from pyscf import dft
from pyscf import gto
import tensorflow.compat.v1 as tf


SCF_SCALAR_RESULTS = [
    'Etot', 'Exc', 'Exx', 'Exxlr', 'Enlc', 'converged', 'time']


def parse_xyz(xyz_path):
  """Parses atomic structure, charge and spin from xyz file.

  The second line of the xyz file is assumed to contains two integers,
  representing charge and spin multiplicity (spin + 1).

  Args:
    xyz_path: String, the path to the xyz file.

  Returns:
    atom: String, the atomic structure.
    charge: Integer, the total charge.
    spin: Integer, the difference of spin up and spin down electron numbers.
  """
  with tf.io.gfile.GFile(xyz_path, 'r') as f:
    xyz = f.readlines()
  charge, spin_mult = map(int, xyz[1].split())
  atom = '; '.join(atomline.strip() for atomline in xyz[2:])
  return atom, charge, spin_mult - 1


def run_scf_for_mol(atom,
                    charge,
                    spin,
                    xc,
                    basis,
                    xc_fun=None,
                    hybrid_coeff=None,
                    rsh_params=None,
                    conv_tol=None,
                    conv_tol_grad=None,
                    use_sg1_prune_for_nlc=True,
                    verbosity=4,
                    max_memory=4000):
  """Performs SCF calculation for a single molecule.

  Args:
    atom: String, the atomic structure.
    charge: Integer, the total charge of molecule.
    spin: Integer, the difference of spin up and spin down electron numbers.
    xc: String, the XC functional name.
    basis: String, the GTO basis.
    xc_fun: Function, custom XC functional. If xcfun is specified, the
      eval_xc method in PySCF will be overridden, and the argument 'xc' will
      only be used to determine VV10 NLC.
    hybrid_coeff: Float, the fraction of exact exchange for custom global
      hybrid functional.
    rsh_params: Tuple of (float, float, float), RSH parameters for custom
      range-separated hybrid functional.
    conv_tol: Float, the convergence threshold of total energy. If not
      specified, the PySCF default value 1e-9 au is used.
    conv_tol_grad: Float, the convergence threshold of orbital gradients. If not
      specified, the PySCF default value of sqrt(conv_tol) is used.
    use_sg1_prune_for_nlc: Boolean, whether use SG1 prune for NLC calculation.
    verbosity: Integer, the verbosity level for PySCF.
    max_memory: Float, the maximum memory in MB for PySCF.

  Returns:
    Dict, the results of the SCF calculation. Contains following keys:
        * Etot: Float, the DFT total energy.
        * Exc: Float, the exchange-correlation energy.
        * Exx: Float, the exact-exchange energy (equal to Exc if xc == 'HF').
        * rho: Float numpy array with shape (6, num_grids) for spin unpolarized
            case and (2, 6, num_grids) for spin polarized case, the density and
            its gradients. For spin polarized case, the first dimension
            represent spin index. The dimension with size 6 represents
            (density, gradient_x, gradient_y, gradient_z, laplacian, tau).
        * weights: Float numpy array with shape (num_grids,), the weights
            for numerical integration.
        * converged: Boolean, whether the SCF calculation is converged.
        * time: Float, the walltime elapsed for the calculation.
  """
  start = time.time()

  logging.info('Construct molecule.')
  mol = gto.M(
      atom=atom,
      basis=basis,
      charge=charge,
      spin=spin,
      verbose=verbosity,
      max_memory=max_memory
      )

  logging.info('Construct KS calculation.')
  if spin == 0:
    ks = dft.RKS(mol)
  else:
    ks = dft.UKS(mol)

  ks.xc = xc
  if xc_fun is not None:
    logging.info('Use Custom XC: %s', xc_fun)
    logging.info('hybrid_coeff = %s, rsh_params = %s', hybrid_coeff, rsh_params)
    ks.define_xc_(xc_fun, xctype='MGGA', hyb=hybrid_coeff, rsh=rsh_params)
  if xc.upper() in dft.libxc.VV10_XC:
    logging.info('Use VV10 NLC.')
    ks.nlc = 'VV10'
    if use_sg1_prune_for_nlc:
      # NOTE(htm): SG1 prune can be used to reduce the computational cost of
      # VV10 NLC. The use of SG1 prune has very little effect on the resulting
      # XC energy. SG1 prune is used in PySCF's example
      # pyscf/examples/dft/33-nlc_functionals.py and is also used in paper
      # 10.1080/00268976.2017.1333644. Note that SG1 prune in PySCF is not
      # available for some elements appeared in the MCGDB84 database.
      ks.nlcgrids.prune = dft.gen_grid.sg1_prune
    if xc_fun is not None:
      # NOTE(htm): It is necessary to override ks._numint._xc_type method to
      # let PySCF correctly use a custom XC functional with NLC. Also, note that
      # ks.xc is used to determine NLC parameters.
      ks._numint._xc_type = lambda code: 'NLC' if 'VV10' in code else 'MGGA'  # pylint: disable=protected-access

  if conv_tol is not None:
    ks.conv_tol = conv_tol
  if conv_tol_grad is not None:
    ks.conv_tol_grad = conv_tol_grad

  logging.info('Perform SCF calculation.')
  ks.kernel()

  logging.info('Compute rho and derivatives.')
  ao = dft.numint.eval_ao(ks.mol, coords=ks.grids.coords, deriv=2)
  if spin == 0:
    rho = dft.numint.eval_rho2(
        ks.mol,
        ao,
        mo_coeff=ks.mo_coeff,
        mo_occ=ks.mo_occ,
        xctype='MGGA')
  else:
    rhoa = dft.numint.eval_rho2(
        ks.mol,
        ao,
        mo_coeff=ks.mo_coeff[0],
        mo_occ=ks.mo_occ[0],
        xctype='MGGA')
    rhob = dft.numint.eval_rho2(
        ks.mol,
        ao,
        mo_coeff=ks.mo_coeff[1],
        mo_occ=ks.mo_occ[1],
        xctype='MGGA')
    rho = np.array([rhoa, rhob])

  logging.info('Compute exact exchange energy.')
  dm = ks.make_rdm1()
  vk = ks.get_k()
  if spin == 0:
    e_xx = -np.einsum('ij,ji', dm, vk) * .5 * .5
  else:
    e_xx = -(np.einsum('ij,ji', dm[0], vk[0]) +
             np.einsum('ij,ji', dm[1], vk[1])) * .5

  if xc_fun is None:
    omega = dft.libxc.rsh_coeff(ks.xc)[0]
  else:
    omega = rsh_params[0]

  if abs(omega) > 1e-10:
    vklr = dft.rks._get_k_lr(mol, dm, omega=omega)  # pylint: disable=protected-access
    # NOTE(htm): in PySCF v1.5a, _get_k_lr is protected. In PySCF >= 1.7.0
    # one can use vklr = ks.get_k(omega=omega)
    if spin == 0:
      e_xxlr = - np.einsum('ij,ji', dm, vklr) * .5 * .5
    else:
      e_xxlr = - (np.einsum('ij,ji', dm[0], vklr[0]) +
                  np.einsum('ij,ji', dm[1], vklr[1])) * .5
  else:
    e_xxlr = 0.

  if ks.nlc:
    logging.info('Compute VV10 nonlocal correlation energy.')
    _, e_nlc, _ = ks._numint.nr_rks(  # pylint: disable=protected-access
        # NOTE(htm): in PySCF v1.5a, dft.numint.nr_rks takes an _NumInt instance
        # as the first argument. Therefore it is difficult to circumvent the
        # access to protected attributes. In PySCF >= 1.7.0, NumInt is no longer
        # protected.
        mol=mol,
        grids=ks.nlcgrids,
        xc_code=ks.xc + '__' + ks.nlc,
        dms=(dm if spin == 0 else dm[0] + dm[1]))
  else:
    e_nlc = 0.

  results = {
      'Etot': ks.e_tot,
      'Exc': ks.get_veff(mol, dm=ks.make_rdm1()).exc,
      'Exx': e_xx,
      'Exxlr': e_xxlr,
      'Enlc': e_nlc,
      'rho': rho,
      'weights': ks.grids.weights,
      'converged': ks.converged,
      'time': time.time() - start,
      }
  logging.info('SCF finished. Results: %s', results)

  return results
