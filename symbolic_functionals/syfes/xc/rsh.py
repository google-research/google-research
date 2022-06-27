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

"""Functions for evaluating Range-Separated-Hybrid (RSH) functionals."""

import jax.numpy as jnp
import jax.scipy.special
import numpy as onp
import scipy.special
from symbolic_functionals.syfes.xc import utils


def f_rsh(rho, omega, polarized=False, use_jax=True):
  """Enchancement factor for evaluating short-range semilocal exchange.

  10.1063/1.4952647 Eq. 11.

  Args:
    rho: Float numpy array with shape (num_grids,), the electron density.
    omega: Float, the range seperation parameter.
    polarized: Boolean, whether the system is spin polarized.
    use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
      numpy.

  Returns:
    Float numpy array with shape (num_grids,), the RSH enhancement factor.
  """
  if use_jax:
    np = jnp
    special = jax.scipy.special
  else:
    np = onp
    special = scipy.special
  spin_factor = 1 if polarized else 2
  # Fermi wave vector
  kf = (6 * jnp.pi**2 * rho / spin_factor + utils.EPSILON) ** (1 / 3)
  a = omega / kf + utils.EPSILON  # variable a in Eq. 11
  return (1 - 2 / 3 * a * (2 * jnp.pi ** (1 / 2) * special.erf(1 / a) - 3 * a
                           + a ** 3 + (2 * a - a ** 3) * np.exp(-1 / a ** 2)))


def get_exc_sl(alpha, beta, exc, exx, exx_lr, enlc=0.):
  """Evaluates the semilocal exchange-correlation energy.

  This function concerns the energies, which should not be confused with other
  functions in california.xc that evaluates energy densities.

  The total exchange energy Exc of range-separated hybrid (RSH) functionals
  can be decomposed into semilocal exchange-correlation Exc_sl, exact exchange
  Exx, long-range exact exchange Exx_lr and (optionally) nonlocal correlation
  Ec_nl:
    Exc = Exc_sl + (alpha + beta) * Exx - beta * Exx_lr + Ec_nl
  Therefore, the semilocal exchange-correlation energy is given by
    Exc_sl = Exc - (alpha + beta) * Exx + beta * Exx_lr - Ec_nl
  where alpha and beta are RSH parameters that can be obtained with the
  rsh_and_hybrid_coeff function in pyscf/dft/numint.py

  Exc_sl can be further decomposed into short-range semilocal exchange Ex_sr,
  long-range semilocal exchange Ex_lr and semilocal correlation Ec_sl:
    Exc_sl = (1 - (alpha + beta)) * Ex_sr + (1 - alpha) * Ex_lr + Ec_sl
  For RSH functionals with alpha = 1 (e.g. wB97M-V), the Exc_sl only contains
  short-range semilocal exchange and semilocal correlation:
    Exc_sl = -beta * Ex_sr + Ec_sl
  Note that for functionals like wB97M-V, the factor -beta does not explicitly
  appear, instead it is implicitly included in the definition of Ex_sr.

  Args:
    alpha: Float, RSH parameter.
    beta: Float, RSH parameter.
    exc: Float, the total exchange-correlation energy Exc.
    exx: Float, the exact-exchange energy Exx.
    exx_lr: Float, the long-range exact-exchange energy Exx_lr.
    enlc: Float, the nonlocal correlation energy Ec_nl.

  Returns:
    Float, the semilocal exchange-correlation energy Exc_sl.
  """
  return exc - (alpha + beta) * exx + beta * exx_lr - enlc
