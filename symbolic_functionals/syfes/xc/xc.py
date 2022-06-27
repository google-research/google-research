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

"""Interface for using functionals developed here in PySCF."""

import functools
import jax
import numpy as np

from symbolic_functionals.syfes.xc import gga
from symbolic_functionals.syfes.xc import lda
from symbolic_functionals.syfes.xc import mgga
from symbolic_functionals.syfes.xc import utils


CUSTOM_FUNCTIONALS = {
    'lda': {
        'xc_type': 'LDA',
        'xc_fun_unpolarized': utils.function_sum(
            lda.e_x_lda_unpolarized, lda.e_c_lda_unpolarized),
        'xc_fun_polarized': utils.function_sum(
            lda.e_x_lda_polarized, lda.e_c_lda_polarized)
    },
    'pbe': {
        'xc_type': 'GGA',
        'xc_fun_unpolarized': utils.function_sum(
            gga.e_x_pbe_unpolarized, gga.e_c_pbe_unpolarized),
        'xc_fun_polarized': utils.function_sum(
            gga.e_x_pbe_polarized, gga.e_c_pbe_polarized),
    },
    'b97': {
        'xc_type': 'GGA',
        'xc_fun_unpolarized': utils.function_sum(
            gga.e_x_b97_unpolarized, gga.e_c_b97_unpolarized),
        'xc_fun_polarized': utils.function_sum(
            gga.e_x_b97_polarized, gga.e_c_b97_polarized),
    },
    'wb97x_v': {
        'xc_type': 'GGA',
        'xc_fun_unpolarized': gga.e_xc_wb97xv_unpolarized,
        'xc_fun_polarized': gga.e_xc_wb97xv_polarized,
    },
    'b97m_v': {
        'xc_type': 'MGGA',
        'xc_fun_unpolarized': mgga.e_xc_b97mv_unpolarized,
        'xc_fun_polarized': mgga.e_xc_b97mv_polarized,
    },
    'wb97m_v': {
        'xc_type': 'MGGA',
        'xc_fun_unpolarized': mgga.e_xc_wb97mv_unpolarized,
        'xc_fun_polarized': mgga.e_xc_wb97mv_polarized,
    },
}


def _to_numpy(array):
  """Converts a jax DeviceArray to numpy array."""
  if isinstance(array, jax.numpy.ndarray):
    return np.array(array)
  else:
    return array


def make_eval_xc(xc_name, params=None):
  """Finds functionals with given name and make the eval_xc function for PySCF.

  Args:
    xc_name: String, name of XC functional (case-insensitive). Possible values
      are given in CUSTOM_FUNCTIONALS.
    params: Dict, keyword arguments for the XC functional.

  Returns:
    Function, the eval_xc function to be use by define_xc_ method of RKS or UKS
      objects of PySCF. The xc_type will be MGGA regardless of actual type
      of XC functional.

  Raises:
    ValueError, if xc_name not found in CUSTOM_FUNCTIONALS.
  """
  if params is None:
    params = {}
  xc_name = xc_name.lower()

  if xc_name not in CUSTOM_FUNCTIONALS:
    raise ValueError(f'XC name {xc_name} not found in CUSTOM_FUNCTIONALS.')

  xc_type = CUSTOM_FUNCTIONALS[xc_name]['xc_type']
  xc_fun_unpolarized = CUSTOM_FUNCTIONALS[xc_name]['xc_fun_unpolarized']
  xc_fun_polarized = CUSTOM_FUNCTIONALS[xc_name]['xc_fun_polarized']

  if xc_type == 'MGGA':
    xc_fun_unpolarized_mgga = xc_fun_unpolarized
    xc_fun_polarized_mgga = xc_fun_polarized
  else:
    xc_fun_unpolarized_mgga = make_mgga(
        xc_fun_unpolarized, xc_type=xc_type, polarized=False)
    xc_fun_polarized_mgga = make_mgga(
        xc_fun_polarized, xc_type=xc_type, polarized=True)

  return make_eval_xc_mgga(
      xc_fun_unpolarized=xc_fun_unpolarized_mgga,
      xc_fun_polarized=xc_fun_polarized_mgga,
      params=params)


def make_mgga(xc_fun, xc_type, polarized):
  """Wrap a LDA or GGA functional as a MGGA functional.

  Args:
    xc_fun: Function, the LDA or GGA functional.
    xc_type: String, the type of xc_fun, possible values are 'LDA' and 'GGA'.
    polarized: Boolean, whether the function is for spin polarized systems.

  Returns:
    Function, the input functional wrapped as a MGGA functional.
  """
  if (xc_type, polarized) == ('LDA', False):

    def xc_fun_mgga(rho, sigma, tau, **params):
      del sigma, tau
      return xc_fun(rho, **params)

  elif (xc_type, polarized) == ('LDA', True):

    def xc_fun_mgga(rhoa, rhob, sigma_aa, sigma_ab, sigma_bb,
                    tau_a, tau_b, **params):
      del sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b
      return xc_fun(rhoa, rhob, **params)

  elif (xc_type, polarized) == ('GGA', False):

    def xc_fun_mgga(rho, sigma, tau, **params):
      del tau
      return xc_fun(rho, sigma, **params)

  elif (xc_type, polarized) == ('GGA', True):

    def xc_fun_mgga(rhoa, rhob, sigma_aa, sigma_ab, sigma_bb,
                    tau_a, tau_b, **params):
      del tau_a, tau_b
      return xc_fun(rhoa, rhob, sigma_aa, sigma_ab, sigma_bb, **params)

  else:
    raise ValueError(f'Unknown xc_type {xc_type}, expected LDA or GGA.')

  return xc_fun_mgga


def make_eval_xc_mgga(xc_fun_unpolarized, xc_fun_polarized, params=None):
  """Generates eval_xc functions for MGGA functionals.

  Args:
    xc_fun_unpolarized: Function, the XC functional for spin unpolarized case.
    xc_fun_polarized: Function, the XC functional for spin polarized case.
    params: Dict, keyword arguments for the XC functional.

  Returns:
    Function, the eval_xc function.
  """
  if params:
    xc_fun_unpolarized = functools.partial(xc_fun_unpolarized, **params)
    xc_fun_polarized = functools.partial(xc_fun_polarized, **params)

  def eval_xc(xc_code,
              rho_and_derivs,
              spin=0,
              relativity=0,
              deriv=1,
              verbose=None):
    """Evaluates exchange-correlation energy densities and derivatives.

    Args:
      xc_code: A dummy argument, not used.
      rho_and_derivs: Float numpy array with shape (6, num_grids) for spin
        unpolarized case; 2-tuple of float numpy array with shape (6, num_grids)
        for spin polarized case. Electron density and its derivatives. For
        spin unpolarized case, the 6 subarrays represent (density, gradient_x,
        gradient_y, gradient_z, laplacian, tau); for spin polarized case, the
        spin up and spin down densities and derivatives are each represented
        with a (6, num_grids) array.
      spin: Integer, 0 for spin unpolarized and 1 for spin polarized
        calculations.
      relativity: A dummy argument, not used.
      deriv: Integer, the order of derivatives evaluated for XC energy density.
      verbose: A dummy argument, not used.

    Returns:
      eps_xc: Float numpy array with shape (num_grids,), the XC energy density
        per particle.
      v_xc: Tuple of float numpy arrays, the first derivatives of XC energy
        density per volume to various quantities. See pyscf/dft/libxc.py for
        more details.
      f_xc: A dummy return value, not used.
      k_xc: A dummy return value, not used.

    Raises:
      NotImplementedError: If derivative order higher than one is requested
        (deriv > 1).
    """
    del xc_code, relativity, verbose

    if deriv != 1:
      raise NotImplementedError('Only deriv = 1 is implemented.')

    if spin == 0:
      rho, grad_x, grad_y, grad_z, _, tau = rho_and_derivs
      sigma = grad_x**2 + grad_y**2 + grad_z**2

      e_xc, grads = jax.jit(jax.vmap(jax.value_and_grad(
          xc_fun_unpolarized, argnums=(0, 1, 2))))(rho, sigma, tau)
      vrho, vsigma, vtau = grads

    else:
      rhoa, grad_x_a, grad_y_a, grad_z_a, _, tau_a = rho_and_derivs[0]
      rhob, grad_x_b, grad_y_b, grad_z_b, _, tau_b = rho_and_derivs[1]
      rho = rhoa + rhob
      sigma_aa = grad_x_a**2 + grad_y_a**2 + grad_z_a**2
      sigma_ab = grad_x_a * grad_x_b + grad_y_a * grad_y_b + grad_z_a * grad_z_b
      sigma_bb = grad_x_b**2 + grad_y_b**2 + grad_z_b**2

      e_xc, grads = jax.jit(jax.vmap(jax.value_and_grad(
          xc_fun_polarized, argnums=(0, 1, 2, 3, 4, 5, 6))))(
              rhoa, rhob, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b)

      vrhoa, vrhob, vsigma_aa, vsigma_ab, vsigma_bb, vtau_a, vtau_b = grads
      vrho = np.stack((vrhoa, vrhob), axis=1)
      vsigma = np.stack((vsigma_aa, vsigma_ab, vsigma_bb), axis=1)
      vtau = np.stack((vtau_a, vtau_b), axis=1)

    eps_xc = e_xc / (rho + utils.EPSILON)
    return (_to_numpy(eps_xc),
            (_to_numpy(vrho), _to_numpy(vsigma),
             np.zeros_like(vtau), _to_numpy(vtau)),
            None,
            None)

  return eval_xc
