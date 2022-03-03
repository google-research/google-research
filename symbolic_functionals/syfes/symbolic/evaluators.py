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

"""Evaluator for XC functionals."""

import json

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v1 as tf

from symbolic_functionals.syfes import loss
from symbolic_functionals.syfes.dataset import dataset
from symbolic_functionals.syfes.symbolic import xc_functionals
from symbolic_functionals.syfes.xc import gga
from symbolic_functionals.syfes.xc import lda
from symbolic_functionals.syfes.xc import mgga
from symbolic_functionals.syfes.xc import rsh


HARTREE_TO_KCALPERMOLE = 627.508

GAMMA_X_B97 = 0.004


def combine_arrays(*arrays):
  """Combines 1D arrays of equal length to a longer 1D array.

  [1, 2, 3], [4, 5, 6], [7, 8, 9] -> [1, 4, 7, 2, 5, 8, 3, 6, 9]

  Args:
    *arrays: List of numpy arrays with shape (num_grids,), the input arrays.

  Returns:
    Numpy array with shape (num_grids * num_arrays,), the combine array. A copy
      is made to ensure the resulting array is contiguous in memory.
  """
  return np.stack(arrays, axis=1).flatten().copy()


def parse_rho_and_derivs(rho_and_derivs, omega, polarized):
  """Computes common features and lda energies from density and derivatives.

  NOTE(htm): for spin unpolarized case, the rho_and_derivs should be multiplied
  with a spin factor 0.5 before passed to this function. This design avoids
  creating an array for 0.5 * rho_and_derivs within the function, which can cost
  a significant amount of memory for large sample sizes.

  Args:
    rho_and_derivs: Float numpy array with shape (6, num_grids) for spin
      unpolarized case; 2-tuple of float numpy array with shape (6, num_grids)
      for spin polarized case. Electron density and its derivatives. For
      spin unpolarized case, the 6 subarrays represent (density, gradient_x,
      gradient_y, gradient_z, laplacian, tau); for spin polarized case, the
      spin up and spin down densities and derivatives are each represented
      with a (6, num_grids) array.
    omega: Float, range separated parameter that affects the resulting LDA
      exchange energy.
    polarized: Boolean, whether the input density and derivatives are spin
      polarized.

  Returns:
    features: Dict {quantity_name: quantity_value} including the following
      keys:
      * rho: density
      * x2: x ** 2
      * w: (t - 1) / (t + 1)
      * u: B97 exchange u
      For spin polarized molecules, quantity_value is float numpy array with
      shape (3*num_grids,). Each grid point is repeated 3 times, with the 3
      elements representing spin up, spin down and spin-averaged quantities,
      respectively. For instance, the first three elements of rho denotes
      rho_a, rho_b, rho_ab = 0.5 * (rho_a + rho_b) on the first grid point.
      For spin unpolarized molecules, spin up, spin down and spin averaged
      quantities are equal, and quantity_value is float numpy array with shape
      (num_grids,) with each element representing the spin up quantity.
      NOTE(htm): the definition of spin-averaged quantities is not unique and
      can be adjusted as needed, as long as they are equal to spin up and spin
      down quantities for spin unpolarized molecules. Here we adopt following
      definitions:
      * rho_ab = 0.5 * (rho_a + rho_b)
      * x2_ab = 0.5 * (x2_a + x2_b)  # consistent with wB97M-V paper
      * w_ab: (t_ab - 1) / (t_ab + 1)  # consistent with wB97M-V paper
      * u_ab: gamma * x2_ab / (1 + gamma * x2_ab)  # consistent with B97 paper
    lda_energies: Dict {lda_energy: lda_energy_value}, the lda energy densities.
      lda_energy includes 'x' for exchange (short-range exchange if omega > 0),
      'css' for same-spin correlation and 'cos' for opposite-spin correlation.
      For spin polarized molecules, lda_energy_value is float numpy array with
      shape (3*num_grids,), and the elements are arranged such that when
      multiplied with a feature vector of spin polarized molecules,
      lda_energies['x'] and lda_energies['css'] are nonzero only for spin up and
      spin down elements, and lda_energies['cos'] is nonzero only for
      spin-averaged elements. For spin unpolarized molecules, the
      lda_energy_value is float numpy array with shape (num_grids,).

  Raises:
    ValueError, if the shape of rho_and_derivs is not (6, *) in spin unpolarized
      case or (2, 6, *) in spin polarized case.
  """
  if polarized:
    features, lda_energies = _parse_rho_and_derivs_polarized(
        rho_and_derivs=rho_and_derivs, omega=omega)
  else:
    features, lda_energies = _parse_rho_and_derivs_unpolarized(
        rho_and_derivs=rho_and_derivs, omega=omega)

  # NOTE(htm): H atom has zero spin polarized density and cause certain
  # arrays to contain NaNs
  return ({key: np.nan_to_num(array) for key, array in features.items()},
          {key: np.nan_to_num(array) for key, array in lda_energies.items()})


def _parse_rho_and_derivs_unpolarized(rho_and_derivs, omega):
  """Computes common features and lda energies for spin unpolarized case."""
  if rho_and_derivs.ndim != 2 or rho_and_derivs.shape[0] != 6:
    raise ValueError(
        f'Wrong shape for rho_and_derivs. Expected (6, *), '
        f'got {rho_and_derivs.shape}')

  rho_s, grad_x_s, grad_y_s, grad_z_s, _, tau_s = rho_and_derivs

  # first derivatives
  sigma_ss = grad_x_s ** 2 + grad_y_s ** 2 + grad_z_s ** 2
  x_s = gga.get_reduced_density_gradient(rho_s, sigma_ss, use_jax=False)
  x2_s = x_s ** 2
  u_s = gga.u_b97(x_s, gamma=GAMMA_X_B97, polarized=True)
  del x_s

  # second derivatives
  tau_s = tau_s.copy()
  t_s = mgga.get_mgga_t(rho_s, tau_s, polarized=True)
  w_s = (t_s - 1) / (t_s + 1)
  del t_s, tau_s

  # LDA energy densities
  e_lda_x = lda.e_x_lda_unpolarized(2 * rho_s)
  if omega > 1e-8:
    e_lda_x *= rsh.f_rsh(rho_s, omega=omega, polarized=True, use_jax=False)
  e_lda_css, e_lda_cos = lda.decomposed_e_c_lda_unpolarized(
      2 * rho_s, use_jax=False)

  return {
      'rho': rho_s,
      'x2': x2_s,
      'w': w_s,
      'u': u_s,
  }, {
      'x': e_lda_x,
      'css': e_lda_css,
      'cos': e_lda_cos,
  }


def _parse_rho_and_derivs_polarized(rho_and_derivs, omega):
  """Computes common features and lda energies for spin polarized case."""
  num_grids = rho_and_derivs.shape[-1]

  if rho_and_derivs.ndim != 3 or rho_and_derivs.shape[0:2] != (2, 6):
    raise ValueError(
        f'Wrong shape for rho_and_derivs. Expected (2, 6, *), '
        f'got {rho_and_derivs.shape}')

  rho_a, grad_x_a, grad_y_a, grad_z_a, _, tau_a = rho_and_derivs[0]
  rho_b, grad_x_b, grad_y_b, grad_z_b, _, tau_b = rho_and_derivs[1]

  # first derivatives
  sigma_aa = grad_x_a ** 2 + grad_y_a ** 2 + grad_z_a ** 2
  sigma_bb = grad_x_b ** 2 + grad_y_b ** 2 + grad_z_b ** 2
  x_a = gga.get_reduced_density_gradient(rho_a, sigma_aa, use_jax=False)
  x_b = gga.get_reduced_density_gradient(rho_b, sigma_bb, use_jax=False)
  x2_a = x_a ** 2
  x2_b = x_b ** 2
  u_a = gga.u_b97(x_a, gamma=GAMMA_X_B97, polarized=True)
  u_b = gga.u_b97(x_b, gamma=GAMMA_X_B97, polarized=True)
  u_ab = gga.u_b97(
      np.sqrt(0.5 * (x2_a + x2_b)), gamma=GAMMA_X_B97, polarized=True)
  del x_a, x_b

  # second derivatives
  t_a = mgga.get_mgga_t(rho_a, tau_a, polarized=True)
  t_b = mgga.get_mgga_t(rho_b, tau_b, polarized=True)
  t_ab = 0.5 * (t_a + t_b)
  w_a = (t_a - 1) / (t_a + 1)
  w_b = (t_b - 1) / (t_b + 1)
  w_ab = (t_ab - 1) / (t_ab + 1)
  del tau_a, tau_b, t_a, t_b

  # LDA energy densities
  e_lda_x_a = 0.5 * lda.e_x_lda_unpolarized(2 * rho_a)
  e_lda_x_b = 0.5 * lda.e_x_lda_unpolarized(2 * rho_b)
  if omega > 1e-8:
    e_lda_x_a *= rsh.f_rsh(rho_a, omega=omega, polarized=True, use_jax=False)
    e_lda_x_b *= rsh.f_rsh(rho_b, omega=omega, polarized=True, use_jax=False)
  e_lda_css_a, e_lda_css_b, e_lda_cos = lda.decomposed_e_c_lda_polarized(
      rho_a, rho_b, use_jax=False)

  return {
      'rho': combine_arrays(rho_a, rho_b, 0.5 * (rho_a + rho_b)),
      'x2': combine_arrays(x2_a, x2_b, 0.5 * (x2_a + x2_b)),
      'w': combine_arrays(w_a, w_b, w_ab),
      'u': combine_arrays(u_a, u_b, u_ab),
  }, {
      'x': combine_arrays(e_lda_x_a, e_lda_x_b, np.zeros(num_grids)),
      'css': combine_arrays(e_lda_css_a, e_lda_css_b, np.zeros(num_grids)),
      'cos': combine_arrays(
          np.zeros(num_grids), np.zeros(num_grids), e_lda_cos)
  }


def parse_ks_info(ks_info_unpolarized,
                  ks_info_polarized,
                  feature_names_x,
                  feature_names_css,
                  feature_names_cos,
                  omega):
  """Parses ks_info of dataset to obtain quantities for building Evaluator.

  Args:
    ks_info_unpolarized: List of float numpy arrays with shape (6, num_grids),
      the ks_info for spin unpolarized molecules.
    ks_info_polarized: List of float numpy arrays with shape (2, 6, num_grids),
      the ks_info for spin polarized molecules.
    feature_names_x: Sequence of strings, the feature names for evaluating
      exchange enhancement factor. See docstring of parse_rho_and_derivs for
      allowed feature names.
    feature_names_css: Sequence of strings, the feature names for evaluating
      same-spin correlation enhancement factor. See docstring of
      parse_rho_and_derivs for allowed feature names.
    feature_names_cos: Sequence of strings, the feature names for evaluating
      opposite-spin correlation enhancement factor. See docstring of
      parse_rho_and_derivs for allowed feature names.
    omega: Float, range separated parameter that affects the resulting LDA
      exchange energy.

  Returns:
    Dict, the quantities obtained from ks_info. Keys including
      * num_grids_for_mols
      * rho_weights
      * e_lda_x
      * e_lda_css
      * e_lda_cos
      * features_x
      * features_css
      * features_cos
      See constructor of Evaluator for documentation of these quantities.
      features_x, features_css and features_cos will share same arrays for
      same feature names.
  """

  def _remove_unused_features(features):
    """Removes unused features to reduce memory cost."""
    features_to_remove = []
    for feature_name in features:
      if all([feature_name not in feature_names_x,
              feature_name not in feature_names_css,
              feature_name not in feature_names_cos]):
        features_to_remove.append(feature_name)
    for feature_name in features_to_remove:
      del features[feature_name]

  num_mols_unpolarized = len(ks_info_unpolarized)
  num_mols_polarized = len(ks_info_polarized)

  # spin unpolarized molecules
  rho_and_derivs_unpolarized = 0.5 * np.concatenate(  # 0.5: spin factor
      [ks_info_mol['rho'] for ks_info_mol in ks_info_unpolarized]
      if num_mols_unpolarized else [np.zeros([6, 0])], axis=1)
  features_unpolarized, lda_energies_unpolarized = parse_rho_and_derivs(
      rho_and_derivs_unpolarized, omega=omega, polarized=False)
  _remove_unused_features(features_unpolarized)

  num_grids_for_mols_unpolarized = np.array(
      [ks_info_mol['rho'].shape[-1] for ks_info_mol in ks_info_unpolarized],
      dtype=int)
  rho_weights_unpolarized = np.concatenate(
      [ks_info_mol['weights'] for ks_info_mol in ks_info_unpolarized]
      if num_mols_unpolarized else [np.array([])])

  # spin polarized molecules
  rho_and_derivs_polarized = np.concatenate(
      [ks_info_mol['rho'] for ks_info_mol in ks_info_polarized]
      if num_mols_polarized else [np.zeros([2, 6, 0])], axis=2)
  features_polarized, lda_energies_polarized = parse_rho_and_derivs(
      rho_and_derivs_polarized, omega=omega, polarized=True)
  _remove_unused_features(features_polarized)

  num_grids_for_mols_polarized = 3 * np.array(  # 3: a, b, ab
      [ks_info_mol['rho'].shape[-1] for ks_info_mol in ks_info_polarized],
      dtype=int)
  rho_weights_polarized = np.repeat(np.concatenate(
      [ks_info_mol['weights'] for ks_info_mol in ks_info_polarized]
      if num_mols_polarized else [np.zeros([0])]), repeats=3)  # 3: a, b, ab

  # number of grids
  num_grids_all = (np.sum(num_grids_for_mols_unpolarized)
                   + np.sum(num_grids_for_mols_polarized))
  assert num_grids_all == (rho_and_derivs_unpolarized.shape[-1]
                           + 3 * rho_and_derivs_polarized.shape[-1])
  num_grids_for_mols = np.concatenate([
      num_grids_for_mols_unpolarized, num_grids_for_mols_polarized])
  assert np.sum(num_grids_for_mols) == num_grids_all

  # rho_weights
  rho_weights = np.concatenate([
      rho_weights_unpolarized, rho_weights_polarized])
  assert rho_weights.shape == (num_grids_all,)
  del rho_weights_unpolarized, rho_weights_polarized

  # lda energy densities
  e_lda_x = np.concatenate([
      lda_energies_unpolarized['x'], lda_energies_polarized['x']])
  e_lda_css = np.concatenate([
      lda_energies_unpolarized['css'], lda_energies_polarized['css']])
  e_lda_cos = np.concatenate([
      lda_energies_unpolarized['cos'], lda_energies_polarized['cos']])
  for e_lda in [e_lda_x, e_lda_css, e_lda_cos]:
    assert e_lda.shape == (num_grids_all,)
  del lda_energies_unpolarized, lda_energies_polarized

  # features
  features = {}
  for feature_name in set(
      feature_names_x + feature_names_css + feature_names_cos):
    features[feature_name] = np.concatenate(
        [features_unpolarized[feature_name], features_polarized[feature_name]])
  for feature_array in features.values():
    assert feature_array.shape == (num_grids_all,)

  return {
      'num_grids_for_mols': num_grids_for_mols,
      'rho_weights': rho_weights,
      'e_lda_x': e_lda_x,
      'e_lda_css': e_lda_css,
      'e_lda_cos': e_lda_cos,
      'features_x': {
          feature_name: features[feature_name]
          for feature_name in feature_names_x},
      'features_css': {
          feature_name: features[feature_name]
          for feature_name in feature_names_css},
      'features_cos': {
          feature_name: features[feature_name]
          for feature_name in feature_names_cos},
  }


class Evaluator:
  """Evaluator for the accuracies of exchange-correlation functionals."""

  def __init__(self,
               num_grids_for_mols,
               rho_weights,
               formula_matrix,
               targets,
               sample_weights,
               e_lda_x,
               e_lda_css,
               e_lda_cos,
               features=None,
               features_x=None,
               features_css=None,
               features_cos=None,
               eval_mode='jit',
               baselines=None):
    """Initializes evaluator.

    Args:
      num_grids_for_mols: List of num_mols integers, the number of grids for
        molecules.
      rho_weights: Sequence of num_mols float numpy arrays, each with shape
        (num_grids,) for the corresponding molecule, the weights for numerical
        integration of XC energy density.
      formula_matrix: Float numpy array with shape (num_targets, num_mols), the
        matrix to transform molecular data to properties.
      targets: Float numpy array with shape (num_targets,), the target XC
        energies.
      sample_weights: Float numpy array with shape (num_targets,), the sample
        weights for the calculation of loss.
      e_lda_x: Float numpy array with shape (num_grids_all,), the LDA exchange
        energy density.
      e_lda_css: Float numpy array with shape (num_grids_all,), the same-spin
        LDA correlation energy density.
      e_lda_cos: Float numpy array with shape (num_grids_all,), the opposite-
        spin LDA correlation energy density.
      features: Dict {feature_name: feature_value}, the features used for
        evaluating the XC functional. Feature values are float numpy arrays
        with shape (num_grids_all,).
      features_x: Dict {feature_name: feature_value}. If present, overrides
        features for the calculation of exchange enhancement factor.
      features_css: Dict {feature_name: feature_value}. If present, overrides
        features for the calculation of same-spin correlation enhancement
        factor.
      features_cos: Dict {feature_name: feature_value}. If present, overrides
        features for the calculation of opposite-spin correlation enhancement
        factor.
      eval_mode: String, mode for evaluation. Possible values are:
        * onp: original numpy.
        * jnp: jax.numpy.
        * jit: jax.numpy + jax.jit.
      baselines: Sequence of strings, baselines to be evaluated. Supported
        baselines are:
        * empty: empty functional (zero exchange correlation energies)
        * lda: LDA functional

    Raises:
      ValueError, if features, rho_weights, formula_matrix or sample_weights
        have wrong shapes, or eval_mode is not onp, jnp, or jit.
    """
    self.num_grids_for_mols = num_grids_for_mols
    self.grids_indices = [0] + list(
        np.cumsum(np.array(self.num_grids_for_mols)))

    self.num_mols = len(self.num_grids_for_mols)
    self.num_grids_all = sum(self.num_grids_for_mols)

    self.features_x = features_x if features_x is not None else features
    self.features_css = features_css if features_css is not None else features
    self.features_cos = features_cos if features_cos is not None else features

    for features in [self.features_x, self.features_css, self.features_cos]:
      for feature in features.values():
        if feature.shape != (self.num_grids_all,):
          raise ValueError(
              f'Wrong shape for features. Expected ({self.num_grids_all},), '
              f'got {feature.shape}')

    if rho_weights.shape != (self.num_grids_all,):
      raise ValueError(
          f'Wrong shape for rho_weights. Expected ({self.num_grids_all},), '
          f'got {rho_weights.shape}')
    self.rho_weights = rho_weights

    self.targets = np.array(targets)
    self.num_targets = len(self.targets)

    if formula_matrix.shape != (self.num_targets, self.num_mols):
      raise ValueError(
          f'Wrong shape for formula_matrix. Expected ({self.num_targets}, '
          f'{self.num_mols}), got {formula_matrix.shape}')
    self.formula_matrix = formula_matrix

    if sample_weights.shape != (self.num_targets,):
      raise ValueError(
          f'Wrong shape for sample_weights. Expected ({self.num_targets},), '
          f'got {sample_weights.shape}')
    self.sample_weights = sample_weights

    for e_lda in [e_lda_x, e_lda_css, e_lda_cos]:
      if e_lda.shape != (self.num_grids_all,):
        raise ValueError(
            f'Wrong shape for e_lda. Expected ({self.num_grids_all},), '
            f'got {e_lda.shape}')
    self.e_lda_x = e_lda_x
    self.e_lda_css = e_lda_css
    self.e_lda_cos = e_lda_cos

    if eval_mode not in ['onp', 'jnp', 'jit']:
      raise ValueError(f'Unknown eval_mode {eval_mode}, supported values '
                       'are onp, jnp and jit')
    self.eval_mode = eval_mode

    self.baselines = {}
    if baselines is not None:
      self.eval_baselines(baselines=baselines)

    self.validate()

  def validate(self):
    """Validates the evalutor by evaluating loss of existing functionals.

    Currently, this function will try to evaluate WRMSD of wB97M-V if features
    include x2 and w, and B97X if features include u. The resulting losses are
    only logged for validation purpose and not compared to any reference values.
    """
    if 'x2' in self.features_x and 'w' in self.features_x:
      functional_name = 'wB97M-V'
      functional = xc_functionals.wb97mv_short
      parameters = xc_functionals.WB97MV_PARAMETERS_UTRANSFORM
    elif 'u' in self.features_x:
      functional_name = 'B97X'
      functional = xc_functionals.b97x_u_short
      parameters = xc_functionals.B97_PARAMETERS
    else:
      return
    logging.info(
        'Evaluator.validate: WRMSD (kcal/mol) for %s functional: %s',
        functional_name, self.get_eval_wrmsd(functional)(**parameters))

  def eval_baselines(self, baselines):
    """Evaluates baselines."""
    for baseline in baselines:
      if baseline == 'empty':
        self.baselines['empty'] = float(self.get_eval_wrmsd(
            xc_functionals.empty_functional)({}, {}, {}))
      elif baseline == 'lda':
        self.baselines['lda'] = float(self.get_eval_wrmsd(
            xc_functionals.lda_functional)({}, {}, {}))
      else:
        raise ValueError(f'Unknown baseline: {baseline}')

  def eval_xc_energies_for_mols(self,
                                functional,
                                parameters_x,
                                parameters_css,
                                parameters_cos,
                                use_jax=True):
    """Evaluates XC energies for molecules.

    Args:
      functional: Instance of xc_functionals.XCFunctional, the exchange-
        correlation functional.
      parameters_x: Dict {parameter_name: parameter_value}, the scalar
        parameters for exchange enhancement factor.
      parameters_css: Dict {parameter_name: parameter_value}, the scalar
        parameters for same-spin correlation enhancement factor.
      parameters_cos: Dict {parameter_name: parameter_value}, the scalar
        parameters for opposite-spin correlation enhancement factor.
      use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
        numpy.

    Returns:
      Float numpy array with shape (num_mols,), the XC energyies for molecules.
    """
    exc_weighted = self.rho_weights * functional.eval_exc(
        parameters_x=parameters_x,
        parameters_css=parameters_css,
        parameters_cos=parameters_cos,
        e_lda_x=self.e_lda_x,
        e_lda_css=self.e_lda_css,
        e_lda_cos=self.e_lda_cos,
        features_x=self.features_x,
        features_css=self.features_css,
        features_cos=self.features_cos,
        use_jax=use_jax)

    if self.eval_mode == 'onp':
      xc_energies_for_mols = np.add.reduceat(
          exc_weighted, self.grids_indices[:-1])
    else:
      xc_energies_for_mols = jnp.array([
          jnp.sum(exc_weighted[self.grids_indices[i]:self.grids_indices[i + 1]])
          for i in range(self.num_mols)
      ])

    return xc_energies_for_mols

  def get_eval_wrmsd(self, functional):
    """Gets an eval_wrmsd function to evaluate a given functional.

    Args:
      functional: Instance of xc_functionals.XCFunctional, the exchange-
        correlation functional to be evaluated.

    Returns:
      Function, the resulting eval_wrmsd function. eval_wrmsd is a stateless
        pure function that can be jitted.
    """
    use_jax = self.eval_mode != 'onp'

    def eval_wrmsd(parameters_x, parameters_css, parameters_cos):
      """Evaluates weighted root mean square deviation for given parameters.

      Args:
        parameters_x: Dict {parameter_name: parameter_value}, the scalar
          parameters for exchange enhancement factor.
        parameters_css: Dict {parameter_name: parameter_value}, the scalar
          parameters for same-spin correlation enhancement factor.
        parameters_cos: Dict {parameter_name: parameter_value}, the scalar
          parameters for opposite-spin correlation enhancement factor.

      Returns:
        Float, the WRMSD value in kcal/mol.
      """
      xc_energies = self.formula_matrix @ self.eval_xc_energies_for_mols(
          functional,
          parameters_x=parameters_x,
          parameters_css=parameters_css,
          parameters_cos=parameters_cos,
          use_jax=use_jax)

      return HARTREE_TO_KCALPERMOLE * loss.weighted_root_mean_square_deviation(
          y_pred=xc_energies, y_true=self.targets, weights=self.sample_weights,
          use_jax=use_jax)

    if self.eval_mode == 'jit':
      return jax.jit(eval_wrmsd)
    else:
      return eval_wrmsd

  def __str__(self):
    return str({
        'num_mols': self.num_mols,
        'num_grids_all': self.num_grids_all,
        'num_targets': self.num_targets,
        'feature_names_x': list(self.features_x.keys()),
        'feature_names_css': list(self.features_css.keys()),
        'feature_names_cos': list(self.features_cos.keys()),
        'eval_mode': self.eval_mode,
        'baselines': self.baselines
    })

  def __repr__(self):
    return self.__str__()

  @staticmethod
  def from_dataset(subset,
                   feature_names_x,
                   feature_names_css,
                   feature_names_cos,
                   targets='mgcdb84_ref',
                   omega=0.3,
                   alpha=1.0,
                   beta=-0.85,
                   eval_mode='jit'):
    """Constructs an evaluator using a dataset containing SCF results.

    Args:
      subset: dataset.Dataset, input subset of MCGDB84.
      feature_names_x: Sequence of strings, the feature names for evaluating
        exchange enhancement factor. See docstring of parse_rho_and_derivs for
        allowed feature names.
      feature_names_css: Sequence of strings, the feature names for evaluating
        same-spin correlation enhancement factor. See docstring of
        parse_rho_and_derivs for allowed feature names.
      feature_names_cos: Sequence of strings, the feature names for evaluating
        opposite-spin correlation enhancement factor. See docstring of
        parse_rho_and_derivs for allowed feature names.
      targets: String, the targets used for evaluating WRMSD. Defaults to
        'mgcdb84_ref', which computes target values from reference values given
        by MCGDB84. Besides 'mgcdb84_ref', one can specify the targets to be
        the exchange-correlation energies evaluated by given functionals. In
        this case targets can be specified either through the functional name
        defined in xc_functionals, or through a path to a json file including
        the specification of functional form and parameters.
      omega: Float, RSH parameter for functional used in SCF calculations.
      alpha: Float, RSH parameter for functional used in SCF calculations.
      beta: Float, RSH parameter for functional used in SCF calculations.
        Default values of omega, alpha, beta are those of wB97M-V functional
        obtained with pyscf.dft.libxc.rsh_coeff('wb97m_v')
      eval_mode: String, mode for evaluation. Possible values are:
        * onp: original numpy
        * jnp: jax.numpy
        * jit: jax.numpy + jax.jit

    Returns:
      Evaluator, the evaluator generated with dataset and SCF results.
    """
    # sort dft_df based on spin to facilitate treating spin unpolarized
    # and spin polarized molecules separately
    dft_df_sorted = subset.dft_df.sort_values('spin')
    subset_sorted = dataset.Dataset(
        property_df=subset.property_df,
        dft_df=dft_df_sorted,
        geometries=subset.geometries,
        ks_info=subset.ks_info)

    num_mols_unpolarized = len(
        subset_sorted.dft_df[subset_sorted.dft_df['spin'] == 0])
    num_mols_polarized = len(
        subset_sorted.dft_df[subset_sorted.dft_df['spin'] != 0])

    logging.info(
        'Evaluator.from_dataset: %s', {
            'num_mols_unpolarized': num_mols_unpolarized,
            'num_mols_polarized': num_mols_polarized,
            'num_mols': subset_sorted.nrow_dft})
    assert num_mols_unpolarized + num_mols_polarized == subset_sorted.nrow_dft
    assert np.all(subset_sorted.dft_df.iloc[:num_mols_unpolarized]['spin'] == 0)
    assert np.all(subset_sorted.dft_df.iloc[num_mols_unpolarized:]['spin'] != 0)

    ks_info = list(subset_sorted.ks_info.values())
    quantities = parse_ks_info(
        ks_info_unpolarized=ks_info[:num_mols_unpolarized],
        ks_info_polarized=ks_info[num_mols_unpolarized:],
        feature_names_x=feature_names_x,
        feature_names_css=feature_names_css,
        feature_names_cos=feature_names_cos,
        omega=omega)

    if targets == 'mgcdb84_ref':
      # use reference values in MGCDB84

      def _get_exc_sl_ref_per_row(row):
        """Helper function to compute reference semilocal Exc for property_df."""
        return rsh.get_exc_sl(
            alpha=alpha,
            beta=beta,
            # reference total Exc computed based of ref_value
            exc=row['ref_value'] - (row['Etot'] - row['Exc']),
            exx=row['Exx'],
            exx_lr=row['Exxlr'],
            enlc=row['Enlc'])

      evaluator = Evaluator(
          **quantities,
          formula_matrix=subset_sorted.formula_matrix,
          targets=subset.property_df.apply(
              _get_exc_sl_ref_per_row, axis=1).to_numpy(),
          sample_weights=subset_sorted.property_df['mgcdb84_weight'].to_numpy(),
          eval_mode=eval_mode)

    else:
      # use reference values evaluated with a specified functional
      if 'json' in targets:
        with tf.io.gfile.GFile(targets, 'r') as f:
          functional_spec = json.load(f)
        functional = xc_functionals.XCFunctional.from_dict(functional_spec)
        parameters = functional_spec['parameters']
      else:
        functional = getattr(xc_functionals, targets)
        parameters = xc_functionals.DEFAULT_FUNCTIONAL_PARAMETERS[targets]

      evaluator = Evaluator(
          **quantities,
          formula_matrix=subset_sorted.formula_matrix,
          # dummy targets, will be overwritten
          targets=np.zeros(subset_sorted.nrow_property),
          sample_weights=subset_sorted.property_df['mgcdb84_weight'].to_numpy(),
          eval_mode=eval_mode)

      evaluator.targets = (
          evaluator.formula_matrix @ evaluator.eval_xc_energies_for_mols(
              functional=functional, **parameters, use_jax=False))

    evaluator.eval_baselines(['empty', 'lda'])
    return evaluator


class GridEvaluator:
  """Evaluator for exchange-correlation functionals on grids.

  Evaluator evaluates the WRMSD of functional signatures (exchange-correlation
  energy density or enhancement factor) on a given grids from reference values.
  """

  def __init__(self,
               features,
               weights,
               targets,
               e_lda_x,
               e_lda_css,
               e_lda_cos,
               signature,
               eval_mode='jit',
               baselines=('empty', 'lda')):
    """Initializes grid evaluator.

    Args:
      features: Dict {feature_name: feature_value}, the features used for
        evaluating the XC functional. Feature values are float numpy arrays
        with shape (num_grids,).
      weights: Float numpy array with shape (num_grids,), the weights on the
        grids of features.
      targets: Float numpy array with shape (num_targets,), the target XC
        enhancement factor.
      e_lda_x: Float numpy array with shape (num_grids,), the LDA exchange
        energy density.
      e_lda_css: Float numpy array with shape (num_grids,), the same-spin
        LDA correlation energy density.
      e_lda_cos: Float numpy array with shape (num_grids,), the opposite-
        spin LDA correlation energy density.
      signature: String, signature of functional to evaluate. Possible values
        include
        * e_xc: exchange-correlation energy density.
        * f_xc: exchange-correlation enhancement factor.
      eval_mode: String, mode for evaluation. Possible values are:
        * onp: original numpy.
        * jnp: jax.numpy.
        * jit: jax.numpy + jax.jit.
      baselines: Sequence of strings, baselines to be evaluated. Supported
        baselines are:
        * empty: empty functional (zero exchange correlation energies)
        * lda: LDA functional

    Raises:
      ValueError, if feature or e_lda has wrong shape,
        or if signature or eval_mode is unrecongnized.
    """
    self.num_grids = len(weights)

    if weights.shape != (self.num_grids,):
      raise ValueError
    self.weights = weights

    if targets.shape != (self.num_grids,):
      raise ValueError
    self.targets = targets

    for feature in features.values():
      if feature.shape != (self.num_grids,):
        raise ValueError(
            f'Wrong shape for features. Expected ({self.num_grids},), '
            f'got {feature.shape}')
    self.features = features

    for e_lda in [e_lda_x, e_lda_css, e_lda_cos]:
      if e_lda.shape != (self.num_grids,):
        raise ValueError(
            f'Wrong shape for e_lda. Expected ({self.num_grids},), '
            f'got {e_lda.shape}')
    self.e_lda_x = e_lda_x
    self.e_lda_css = e_lda_css
    self.e_lda_cos = e_lda_cos
    self.e_lda = self.e_lda_x + self.e_lda_css + self.e_lda_cos

    if eval_mode not in ['onp', 'jnp', 'jit']:
      raise ValueError(f'Unknown eval_mode {eval_mode}, supported values '
                       'are onp, jnp and jit')
    self.eval_mode = eval_mode

    if signature not in ['e_xc', 'f_xc']:
      raise ValueError(f'Unknown signature {signature}, supported values '
                       'are e_xc and f_xc')
    self.signature = signature

    self.baselines = {}
    if baselines is not None:
      self.eval_baselines(baselines=baselines)

  def get_eval_wrmsd(self, functional):
    """Gets an eval_wrmsd function to evaluate a given functional.

    Args:
      functional: Instance of xc_functionals.XCFunctional, the exchange-
        correlation functional to be evaluated.

    Returns:
      Function, the resulting eval_wrmsd function. eval_wrmsd is a stateless
        pure function that can be jitted.
    """
    use_jax = self.eval_mode != 'onp'

    def eval_wrmsd(parameters_x, parameters_css, parameters_cos):
      """Evaluates weighted root mean square deviation for given parameters.

      Args:
        parameters_x: Dict {parameter_name: parameter_value}, the scalar
          parameters for exchange enhancement factor.
        parameters_css: Dict {parameter_name: parameter_value}, the scalar
          parameters for same-spin correlation enhancement factor.
        parameters_cos: Dict {parameter_name: parameter_value}, the scalar
          parameters for opposite-spin correlation enhancement factor.

      Returns:
        Float, the WRMSD of XC energy density from target values.
      """
      e_xc = functional.eval_exc(
          parameters_x=parameters_x,
          parameters_css=parameters_css,
          parameters_cos=parameters_cos,
          e_lda_x=self.e_lda_x,
          e_lda_css=self.e_lda_css,
          e_lda_cos=self.e_lda_cos,
          features_x=self.features,
          features_css=self.features,
          features_cos=self.features,
          use_jax=use_jax)

      if self.signature == 'e_xc':
        return HARTREE_TO_KCALPERMOLE * loss.weighted_root_mean_square_deviation(
            y_pred=e_xc, y_true=self.targets, weights=self.weights ** 2,
            use_jax=use_jax)
      elif self.signature == 'f_xc':
        return loss.weighted_root_mean_square_deviation(
            y_pred=e_xc / (self.e_lda_x + self.e_lda_css + self.e_lda_cos),
            y_true=self.targets, weights=self.weights ** 2, use_jax=use_jax)

    if self.eval_mode == 'jit':
      return jax.jit(eval_wrmsd)
    else:
      return eval_wrmsd

  def eval_baselines(self, baselines):
    """Evaluates baselines."""
    for baseline in baselines:
      if baseline == 'empty':
        self.baselines['empty'] = float(self.get_eval_wrmsd(
            xc_functionals.empty_functional)({}, {}, {}))
      elif baseline == 'lda':
        self.baselines['lda'] = float(self.get_eval_wrmsd(
            xc_functionals.lda_functional)({}, {}, {}))
      else:
        raise ValueError(f'Unknown baseline: {baseline}')

  def __str__(self):
    return str({
        'num_grids': self.num_grids,
        'feature_names': list(self.features.keys()),
        'eval_mode': self.eval_mode,
        'baselines': self.baselines
    })
