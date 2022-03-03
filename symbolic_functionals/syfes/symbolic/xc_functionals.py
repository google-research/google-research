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

"""Exchange correlation functionals."""

import copy
import hashlib
import json
import jax
import numpy as np

from symbolic_functionals.syfes.symbolic import enhancement_factors
from symbolic_functionals.syfes.xc import gga
from symbolic_functionals.syfes.xc import lda
from symbolic_functionals.syfes.xc import mgga
from symbolic_functionals.syfes.xc import rsh
from symbolic_functionals.syfes.xc import xc


class XCFunctional:
  """Exchange-correlation functional.

  The exchange-correlation energy density e_xc is assumed to take the following
  form:
  e_xc(features) = e_x^LDA(rho) * f_x(features)
                 + e_css^LDA(rho) * f_css(features)
                 + e_cos^LDA(rho) * f_cos(features)
  where e_x^LDA, e_css^LDA and e_cos are LDA exchange, same-spin correlation and
  opposite-spin correlation energy densities, respectively; f_x, f_css and f_cos
  are exchange, same-spin correlation and opposite-spin correlation enhancement
  factors, respectively; features may include density, derivatives of density,
  and/or other custom features.
  """

  # representative values of common features, used to evaluate fingerprints
  _signature_features = {
      'rho': np.array([10., 1., 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]),
      'x2': np.array([10., 100., 1000.]),
      'w': np.array([-1., -0.5, 0., 0.5]),
      'u': np.array([0.0, 0.5, 0.9])
  }
  # 2: spin factor because rho as a feature is defined as rho_sigma
  _signature_e_lda_x = lda.e_x_lda_unpolarized(
      2 * _signature_features['rho'])
  _signature_e_lda_css, _signature_e_lda_cos = (
      lda.decomposed_e_c_lda_unpolarized(
          2 * _signature_features['rho'], use_jax=False))

  def __init__(self, f_x=None, f_css=None, f_cos=None):
    """Initialize an XC functional.

    Args:
      f_x: Instance of enhancement_factors.EnhancementFactor class, the
        exchange enhancement factor. If not specified, empty enhancement factor
        will be used.
      f_css: Instance of enhancement_factors.EnhancementFactor class, the
        same-spin correlation enhancement factor. If not specified, empty
        enhancement factor will be used.
      f_cos: Instance of enhancement_factors.EnhancementFactor class, the
        opposite-spin correlation enhancement factor. If not specified, empty
        enhancement factor will be used.
    """
    self.f_x = f_x or enhancement_factors.f_empty
    self.f_css = f_css or enhancement_factors.f_empty
    self.f_cos = f_cos or enhancement_factors.f_empty

    self.feature_names = sorted(set(
        self.f_x.feature_names + self.f_css.feature_names
        + self.f_cos.feature_names))

    self.num_parameters = (
        self.f_x.num_parameters + self.f_css.num_parameters
        + self.f_cos.num_parameters)

    self.num_used_parameters = (
        self.f_x.num_used_parameters + self.f_css.num_used_parameters
        + self.f_cos.num_used_parameters)

  @property
  def parameters_spec(self):
    """Parameter specification of the functional."""
    # NOTE(htm) parameters_spec cannot be stored as a regular data attribute of
    # XCFunctional as it prevents making deepcopies of XCFunctional instances
    return jax.tree_flatten({
        'parameters_x': {
            parameter_name: 0.
            for parameter_name in sorted(self.f_x.parameter_names)},
        'parameters_css': {
            parameter_name: 0.
            for parameter_name in sorted(self.f_css.parameter_names)},
        'parameters_cos': {
            parameter_name: 0.
            for parameter_name in sorted(self.f_cos.parameter_names)}})[1]

  def eval_exc(self,
               parameters_x,
               parameters_css,
               parameters_cos,
               e_lda_x,
               e_lda_css,
               e_lda_cos,
               features=None,
               features_x=None,
               features_css=None,
               features_cos=None,
               use_jax=True):
    """Evalutates exchange-correlation energy density on grids.

    Args:
      parameters_x: Dict {parameter_name: parameter_value}, the parameters
        for the exchange enhacement factor.
      parameters_css: Dict {parameter_name: parameter_value}, the parameters
        for the same-spin correlation enhacement factor.
      parameters_cos: Dict {parameter_name: parameter_value}, the parameters
        for the opposite-spin correlation enhacement factor.
      e_lda_x: Float numpy array of shape (num_grids_all,), the LDA exchange
        energy density on grids.
      e_lda_css: Float numpy array of shape (num_grids_all,), the LDA
        same-spin correlation energy density on grids.
      e_lda_cos: Float numpy array of shape (num_grids_all,), the LDA
        opposite-spin correlation energy density on grids.
      features: Dict {feature_name: feature_value}, the features for evaluating
        enhancement factors. feature_value's are float numpy array with shape
        (num_grids_all,).
      features_x: Dict {feature_name: feature_value}, if present, overrides
        features for evaluating the exchange enhancement factor.
      features_css: Dict {feature_name: feature_value}, if present, overrides
        features for evaluating the same-spin correlation enhancement factor.
      features_cos: Dict {feature_name: feature_value}, if present, overrides
        features for evaluating the opposite-spin correlation enhancement
        factor.
      use_jax: Boolean, if True, use jax.numpy for calculations, otherwise use
        numpy.

    Returns:
      Float numpy array with shape (num_grids_all,), the exchange-correlation
        energy density on grids.
    """
    features_x = features_x if features_x is not None else features
    features_css = features_css if features_css is not None else features
    features_cos = features_cos if features_cos is not None else features
    return (e_lda_x * self.f_x.eval(
                features_x, parameters_x, use_jax=use_jax) +
            e_lda_css * self.f_css.eval(
                features_css, parameters_css, use_jax=use_jax) +
            e_lda_cos * self.f_cos.eval(
                features_cos, parameters_cos, use_jax=use_jax))

  def eval_penalty(self, penalty_per_parameter):
    """Evaluates penalty of the functional based on number of used parameters.

    Args:
      penalty_per_parameter: Float, the penalty value per used parameter.

    Returns:
      Float, the penalty value.
    """
    return penalty_per_parameter * self.num_used_parameters

  def to_dict(self, parameters=None):
    """Converts the exchange-correlation functional to a dictionary.

    Args:
      parameters: Dict, the parameters of the functional. If present, parameters
        will be included in the resulting dict. This flag is intended to serve
        as a convenient option when storing a functional form together with a
        set of parameters; the XCFunctional instance itself is independent of
        parameters.

    Returns:
      Dict, the dictionary representation of exchange-correlation functional.
    """
    functional_dict = {
        'f_x': self.f_x.to_dict(),
        'f_css': self.f_css.to_dict(),
        'f_cos': self.f_cos.to_dict(),
    }

    if parameters is not None:
      functional_dict['parameters'] = copy.deepcopy(parameters)

    return functional_dict

  @staticmethod
  def from_dict(dictionary):
    """Loads an exchange-correlation functional from a dictionary.

    Args:
      dictionary: Dict, the dictionary representation of exchange-correlation
        functional.

    Returns:
      Instance of XCFunctional, the loaded exchange-correlation functional.
    """
    return XCFunctional(
        f_x=enhancement_factors.EnhancementFactor.from_dict(
            dictionary['f_x']),
        f_css=enhancement_factors.EnhancementFactor.from_dict(
            dictionary['f_css']),
        f_cos=enhancement_factors.EnhancementFactor.from_dict(
            dictionary['f_cos']),
        )

  def make_isomorphic_copy(self,
                           feature_names_x=None,
                           feature_names_css=None,
                           feature_names_cos=None,
                           num_shared_parameters=None,
                           num_variables=None):
    """Makes an isomorphic copy of the XCFunctional instance.

    Args:
      feature_names_x: List of strings, if present, specifies the features for
        exchange enhancement factor.
      feature_names_css: List of strings, if present, specifies the features for
        same-spin correlation enhancement factor.
      feature_names_cos: List of strings, if present, specifies the features for
        opposite-spin correlation enhancement factor.
      num_shared_parameters: Integer or sequence of 3 integers, if present,
        specifies the number of shared parameters for each enhancement factor.
      num_variables: Integer or sequence of 3 integers, if present, specifies
        the number of variables for each enhancement factor.

    Returns:
      XCFunctional instance, the isomorphic copy.
    """
    if num_shared_parameters is None or isinstance(num_shared_parameters, int):
      num_shared_parameters_x = num_shared_parameters
      num_shared_parameters_css = num_shared_parameters
      num_shared_parameters_cos = num_shared_parameters
    else:
      (num_shared_parameters_x, num_shared_parameters_css,
       num_shared_parameters_cos) = num_shared_parameters

    if num_variables is None or isinstance(num_variables, int):
      num_variables_x = num_variables
      num_variables_css = num_variables
      num_variables_cos = num_variables
    else:
      num_variables_x, num_variables_css, num_variables_cos = num_variables

    return XCFunctional(
        f_x=self.f_x.make_isomorphic_copy(
            feature_names=feature_names_x,
            num_shared_parameters=num_shared_parameters_x,
            num_variables=num_variables_x),
        f_css=self.f_css.make_isomorphic_copy(
            feature_names=feature_names_css,
            num_shared_parameters=num_shared_parameters_css,
            num_variables=num_variables_css),
        f_cos=self.f_cos.make_isomorphic_copy(
            feature_names=feature_names_cos,
            num_shared_parameters=num_shared_parameters_cos,
            num_variables=num_variables_cos),
    )

  def make_xc_fun_unpolarized(self, omega, **parameters):
    """Instantiates the functional form for spin unpolarized calculations.

    Args:
      omega: Float, the range separation parameter.
      **parameters: Dict, the parameters for the functional form.

    Returns:
      Function, the resulting function that evaluates exchange correlation
        energy density from rho, sigma and tau.
    """

    def xc_fun_unpolarized(rho, sigma, tau):
      """Evaluates XC energy density for spin unpolarized case.

      Args:
        rho: Float numpy array with shape (num_grids,), the electron density.
        sigma: Float numpy array with shape (num_grids,), the norm square of
          density gradient.
        tau: Float numpy array with shape (num_grids,), the kinetic energy
          density.

      Returns:
        Float numpy array with shape (num_grids,), the XC energy density.
      """
      rho_s = 0.5 * rho
      x_s = gga.get_reduced_density_gradient(rho_s, 0.25 * sigma, use_jax=True)
      x2_s = x_s ** 2
      t_s = mgga.get_mgga_t(rho_s, 0.5 * tau, polarized=True)
      w_s = (t_s - 1) / (t_s + 1)

      e_lda_x = lda.e_x_lda_unpolarized(rho) * rsh.f_rsh(
          rho, omega=omega, polarized=False, use_jax=True)
      e_lda_css, e_lda_cos = lda.decomposed_e_c_lda_unpolarized(
          rho, use_jax=True)

      return self.eval_exc(
          **parameters,
          e_lda_x=e_lda_x,
          e_lda_css=e_lda_css,
          e_lda_cos=e_lda_cos,
          features={'rho': rho_s, 'x2': x2_s, 'w': w_s},
          use_jax=True)

    return xc_fun_unpolarized

  def make_xc_fun_polarized(self, omega, **parameters):
    """Instantiates the functional form for spin polarized calculations.

    Args:
      omega: Float, the range separation parameter.
      **parameters: Dict, the parameters for the functional form.

    Returns:
      Function, the resulting function that evaluates exchange correlation
        energy density from rho, sigma and tau.
    """

    def xc_fun_polarized(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                         tau_a, tau_b):
      """Evaluates XC energy density for spin polarized case.

      Args:
        rho_a: Float numpy array with shape (num_grids,), the spin up electron
          density.
        rho_b: Float numpy array with shape (num_grids,), the spin down electron
          density.
        sigma_aa: Float numpy array with shape (num_grids,), the norm square of
          density gradient (aa component).
        sigma_ab: Float numpy array with shape (num_grids,), the norm square of
          density gradient (ab component).
        sigma_bb: Float numpy array with shape (num_grids,), the norm square of
          density gradient (bb component).
        tau_a: Float numpy array with shape (num_grids,), the spin up kinetic
          energy density.
        tau_b: Float numpy array with shape (num_grids,), the spin down kinetic
          energy density.

      Returns:
        Float numpy array with shape (num_grids,), the XC energy density.
      """
      del sigma_ab
      rho_ab = 0.5 * (rho_a + rho_b)

      x_a = gga.get_reduced_density_gradient(rho_a, sigma_aa, use_jax=True)
      x_b = gga.get_reduced_density_gradient(rho_b, sigma_bb, use_jax=True)
      x2_a = x_a ** 2
      x2_b = x_b ** 2
      x2_ab = 0.5 * (x2_a + x2_b)

      t_a = mgga.get_mgga_t(rho_a, tau_a, polarized=True)
      t_b = mgga.get_mgga_t(rho_b, tau_b, polarized=True)
      t_ab = 0.5 * (t_a + t_b)
      w_a = (t_a - 1) / (t_a + 1)
      w_b = (t_b - 1) / (t_b + 1)
      w_ab = (t_ab - 1) / (t_ab + 1)

      e_lda_x_a = 0.5 * lda.e_x_lda_unpolarized(2 * rho_a) * rsh.f_rsh(
          rho_a, omega=omega, polarized=True, use_jax=True)
      e_lda_x_b = 0.5 * lda.e_x_lda_unpolarized(2 * rho_b) * rsh.f_rsh(
          rho_b, omega=omega, polarized=True, use_jax=True)
      e_lda_css_a, e_lda_css_b, e_lda_cos = lda.decomposed_e_c_lda_polarized(
          rho_a, rho_b, use_jax=True)

      features_a = {'rho': rho_a, 'x2': x2_a, 'w': w_a}
      features_b = {'rho': rho_b, 'x2': x2_b, 'w': w_b}
      features_ab = {'rho': rho_ab, 'x2': x2_ab, 'w': w_ab}

      return (
          e_lda_x_a * self.f_x.eval(
              features_a, parameters['parameters_x'], use_jax=True)
          + e_lda_x_b * self.f_x.eval(
              features_b, parameters['parameters_x'], use_jax=True)
          + e_lda_css_a * self.f_css.eval(
              features_a, parameters['parameters_css'], use_jax=True)
          + e_lda_css_b * self.f_css.eval(
              features_b, parameters['parameters_css'], use_jax=True)
          + e_lda_cos * self.f_cos.eval(
              features_ab, parameters['parameters_cos'], use_jax=True)
          )

    return xc_fun_polarized

  def make_eval_xc(self, omega, **parameters):
    return xc.make_eval_xc_mgga(
        xc_fun_unpolarized=self.make_xc_fun_unpolarized(omega, **parameters),
        xc_fun_polarized=self.make_xc_fun_polarized(omega, **parameters))

  def __eq__(self, other):
    return all([self.f_x == other.f_x,
                self.f_css == other.f_css,
                self.f_cos == other.f_cos])

  def __str__(self):
    return json.dumps(self.to_dict(), indent=2)

  def __repr__(self):
    return self.__str__()

  def get_signature(self,
                    parameters_x,
                    parameters_css,
                    parameters_cos,
                    num_feature_samples=10,
                    signature='e_xc',
                    random_state=0):
    """Computes a signature vector of the functional form with given parameters.

    A random sample of features will be draw from self._signature_features,
    which includes representative values of features in real DFT calculations.
    The signature vector is defined as the exchange-correlation energy density
    (e_xc) or exchange-correlation enhancement factor (F_xc) on the random
    sample of features.

    Args:
      parameters_x: Dict {parameter_name: parameter_value}, the parameters
        for the exchange enhacement factor.
      parameters_css: Dict {parameter_name: parameter_value}, the parameters
        for the same-spin correlation enhacement factor.
      parameters_cos: Dict {parameter_name: parameter_value}, the parameters
        for the opposite-spin correlation enhacement factor.
      num_feature_samples: Integer, number of feature samples.
      signature: String, the signature values to be evaluated. Possible values
        are 'e_xc' for exchange-correlation energy density and 'F_xc' for
        exchange-correlation enhancement factor.
      random_state: Integer or instance of np.random.RandomState, the random
        state used for the calculation.

    Returns:
      Float numpy array with shape (num_feature_samples,), the signature vector.

    Raises:
      ValueError, if feature names not in self._signature_features.
    """
    for feature_name in self.feature_names:
      if feature_name not in self._signature_features:
        raise ValueError('Evaluating signature is not supported for '
                         f'feature {feature_name}')
    if isinstance(random_state, int):
      random_state = np.random.RandomState(random_state)

    # randomly sample rho and corresponding lda energies
    rho_sample_indices = random_state.choice(
        len(self._signature_features['rho']), size=num_feature_samples)
    rho_sample = self._signature_features['rho'][rho_sample_indices]
    e_lda_x_sample = self._signature_e_lda_x[rho_sample_indices]
    e_lda_css_sample = self._signature_e_lda_css[rho_sample_indices]
    e_lda_cos_sample = self._signature_e_lda_cos[rho_sample_indices]

    # randomly sample other features
    feature_samples = {'rho': rho_sample}
    for feature_name in self.feature_names:
      if feature_name == 'rho':
        continue
      feature_samples[feature_name] = random_state.choice(
          self._signature_features[feature_name], size=num_feature_samples)

    e_xc_sample = self.eval_exc(
        parameters_x=parameters_x,
        parameters_css=parameters_css,
        parameters_cos=parameters_cos,
        e_lda_x=e_lda_x_sample,
        e_lda_css=e_lda_css_sample,
        e_lda_cos=e_lda_cos_sample,
        features=feature_samples)

    if signature == 'e_xc':
      return e_xc_sample
    elif signature == 'F_xc':
      return e_xc_sample / (
          e_lda_x_sample + e_lda_css_sample + e_lda_cos_sample)
    else:
      raise ValueError(f'Unrecongnized signature flag {signature}')

  def get_fingerprint(self,
                      num_feature_samples=10,
                      num_parameter_samples=10,
                      num_decimals=5):
    """Gets a fingerprint for the functional.

    Fingerprint is evaluated as the MD5 hash value of functional singatures on
    a random sample of feature and parameter values. Signatures will be
    converted to strings with specified number of decimals.

    Args:
      num_feature_samples: Integer, number of samples of features.
      num_parameter_samples: Integer, number of samples of parameters.
      num_decimals: Integer, number of decimals when converting signatures to
        strings.

    Returns:
      String, the fingerprint of the functional.
    """
    format_string = f'{{:.{num_decimals}f}}'

    # fix random seed to have consistent sampling behavior when running the
    # code on different distributed workers
    random_state = np.random.RandomState(0)

    parameter_samples = random_state.rand(
        num_parameter_samples, self.num_parameters)

    signatures = []
    for parameters in parameter_samples:
      signatures.extend(self.get_signature(
          **jax.tree_unflatten(self.parameters_spec, parameters),
          num_feature_samples=num_feature_samples,
          random_state=random_state,
          signature='e_xc'))

    signature_string = ','.join(map(format_string.format, signatures))

    return hashlib.md5(signature_string.encode('utf-8')).hexdigest()


b97_u = XCFunctional(
    f_x=enhancement_factors.f_b97_u,
    f_css=enhancement_factors.f_b97_u,
    f_cos=enhancement_factors.f_b97_u)

b97_x2 = XCFunctional(
    f_x=enhancement_factors.f_b97_x2,
    f_css=enhancement_factors.f_b97_x2,
    f_cos=enhancement_factors.f_b97_x2)

b97_x2_short = XCFunctional(
    f_x=enhancement_factors.f_b97_x2_short,
    f_css=enhancement_factors.f_b97_x2_short,
    f_cos=enhancement_factors.f_b97_x2_short)

b97x_u_short = XCFunctional(
    f_x=enhancement_factors.f_b97_u_short,
    f_css=enhancement_factors.f_empty,
    f_cos=enhancement_factors.f_empty)

b97x_x2 = XCFunctional(
    f_x=enhancement_factors.f_b97_x2,
    f_css=enhancement_factors.f_empty,
    f_cos=enhancement_factors.f_empty)

b97x_x2_short = XCFunctional(
    f_x=enhancement_factors.f_b97_x2_short,
    f_css=enhancement_factors.f_empty,
    f_cos=enhancement_factors.f_empty)

wb97mv = XCFunctional(
    f_x=enhancement_factors.f_x_wb97mv,
    f_css=enhancement_factors.f_css_wb97mv,
    f_cos=enhancement_factors.f_cos_wb97mv)

wb97mv_short = XCFunctional(
    f_x=enhancement_factors.f_x_wb97mv_short,
    f_css=enhancement_factors.f_css_wb97mv_short,
    f_cos=enhancement_factors.f_cos_wb97mv_short)

empty_functional = XCFunctional()

lda_functional = XCFunctional(
    f_x=enhancement_factors.f_lda,
    f_css=enhancement_factors.f_lda,
    f_cos=enhancement_factors.f_lda)

B97_PARAMETERS = {
    'parameters_x': {
        'c0': 0.8094,
        'c1': 0.5073,
        'c2': 0.7481,
        'gamma': 0.004,
    },
    'parameters_css': {
        'c0': 0.1737,
        'c1': 2.3487,
        'c2': -2.4868,
        'gamma': 0.2,
    },
    'parameters_cos': {
        'c0': 0.9454,
        'c1': 0.7471,
        'c2': -4.5961,
        'gamma': 0.006,
    },
}

B97_PARAMETERS_UTRANSFORM = copy.deepcopy(B97_PARAMETERS)
for key in ['parameters_x', 'parameters_css', 'parameters_cos']:
  B97_PARAMETERS_UTRANSFORM[key]['gamma_utransform'] = (
      B97_PARAMETERS_UTRANSFORM[key].pop('gamma'))

WB97MV_PARAMETERS = {
    'parameters_x': {
        'c00': mgga.WB97MV_PARAMS['power_series_x'][0][2],
        'c10': mgga.WB97MV_PARAMS['power_series_x'][1][2],
        'c01': mgga.WB97MV_PARAMS['power_series_x'][2][2],
        'gamma': mgga.WB97MV_PARAMS['gamma_x']},
    'parameters_css': {
        'c00': mgga.WB97MV_PARAMS['power_series_ss'][0][2],
        'c10': mgga.WB97MV_PARAMS['power_series_ss'][1][2],
        'c20': mgga.WB97MV_PARAMS['power_series_ss'][2][2],
        'c43': mgga.WB97MV_PARAMS['power_series_ss'][3][2],
        'c04': mgga.WB97MV_PARAMS['power_series_ss'][4][2],
        'gamma': mgga.WB97MV_PARAMS['gamma_ss']},
    'parameters_cos': {
        'c00': mgga.WB97MV_PARAMS['power_series_os'][0][2],
        'c10': mgga.WB97MV_PARAMS['power_series_os'][1][2],
        'c20': mgga.WB97MV_PARAMS['power_series_os'][2][2],
        'c60': mgga.WB97MV_PARAMS['power_series_os'][3][2],
        'c21': mgga.WB97MV_PARAMS['power_series_os'][4][2],
        'c61': mgga.WB97MV_PARAMS['power_series_os'][5][2],
        'gamma': mgga.WB97MV_PARAMS['gamma_os']}}

WB97MV_PARAMETERS_UTRANSFORM = copy.deepcopy(WB97MV_PARAMETERS)
for key in ['parameters_x', 'parameters_css', 'parameters_cos']:
  WB97MV_PARAMETERS_UTRANSFORM[key]['gamma_utransform'] = (
      WB97MV_PARAMETERS_UTRANSFORM[key].pop('gamma'))

DEFAULT_FUNCTIONAL_PARAMETERS = {
    'b97_u': B97_PARAMETERS,
    'b97_x2': B97_PARAMETERS_UTRANSFORM,
    'b97_x2_short': B97_PARAMETERS_UTRANSFORM,
    'b97x_u_short': B97_PARAMETERS_UTRANSFORM,
    'b97x_x2': B97_PARAMETERS_UTRANSFORM,
    'b97x_x2_short': B97_PARAMETERS_UTRANSFORM,
    'wb97mv': WB97MV_PARAMETERS,
    'wb97mv_short': WB97MV_PARAMETERS_UTRANSFORM,
    'empty_functional': {
        'parameters_x': {}, 'parameters_css': {}, 'parameters_cos': {}},
    'lda_functional': {
        'parameters_x': {}, 'parameters_css': {}, 'parameters_cos': {}}
}
