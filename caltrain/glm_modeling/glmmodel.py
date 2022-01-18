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

"""Module containing GLMModel class."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brute

from caltrain.glm_modeling import get_beta_fit_data
from caltrain.glm_modeling import get_glm_fit_data
from caltrain.glm_modeling import Transforms
from caltrain.simulation.logistic import TrueLogisticLogOdds
from caltrain.simulation.logistic import TrueLogisticTwoParamFlipPolynomial
from caltrain.simulation.polynomial import TrueTwoParamFlipPolynomial
from caltrain.simulation.polynomial import TrueTwoParamPolynomial
from caltrain.utils import Enum


class GLMModel:
  """Class for GLM models of calibration function regression."""

  def __init__(self,
               link_fcn_pair=None,
               x_transform_fcn_pair=None,
               poly_signature=None,
               func_str=None,
               true_dataset_beta_prior_cls=None,
               name=None,
               data_dir=None):

    assert np.sum(poly_signature) > 0
    assert func_str is not None

    self.link_fcn_pair = link_fcn_pair
    self.x_transform_fcn_pair = x_transform_fcn_pair
    self.poly_signature = poly_signature
    self.func_str = func_str
    self.true_dataset_beta_prior_cls = true_dataset_beta_prior_cls
    self.name = name
    self.beta_fit_data = get_beta_fit_data(data_dir)
    self.glm_fit_data = get_glm_fit_data(data_dir)

  def get_true_dist(self,
                    alpha=None,
                    beta=None,
                    a=None,
                    b=None,
                    n_samples=None,
                    p1=None):
    """Get true distribution model object."""

    if not self.poly_signature[0]:
      assert (a in [0, None]) or np.isnan(a)
      a = 0

    if len(self.poly_signature) > 1:
      if not self.poly_signature[1]:
        assert (b in [0, None]) or np.isnan(b)
        b = 0
    else:
      assert (b in [0, None]) or np.isnan(b)
      b = 0

    assert alpha is not None
    assert beta is not None
    assert n_samples is not None

    dist = self.true_dataset_beta_prior_cls(
        alpha=alpha, beta=beta, a=a, b=b, p1=p1)
    dist.reset(n_samples)
    return dist

  def get_calibration_error_beta_dist(self, dataset, n_samples=None, norm='L2'):
    """Get the calibration error."""

    beta_hat_poly, _, _ = dataset.fit_glm(self)

    alpha = self.beta_fit_data['data'][dataset.model]['a']
    beta = self.beta_fit_data['data'][dataset.model]['b']
    p1 = self.beta_fit_data['data'][dataset.model]['p1']
    a = beta_hat_poly[0]
    b = beta_hat_poly[1]

    dist = self.get_true_dist(
        n_samples=n_samples, alpha=alpha, beta=beta, a=a, b=b, p1=p1)
    return dist.true_calib_error(norm=norm)

  @property
  def link_fcn_inv(self):
    return self.link_fcn_pair.value.finv

  @property
  def link_fcn(self):
    return self.link_fcn_pair.value.f

  @property
  def x_transform_fcn(self):
    return self.x_transform_fcn_pair.value.f

  @property
  def degrees_of_freedom(self):
    return np.sum(self.poly_signature)

  def nll(self, beta_hat, exog, endog):
    """Negative log-likelihood for optimization."""

    linear_arg = exog @ beta_hat
    p_hat = self.link_fcn_inv(linear_arg)
    a = endog * np.log(p_hat)
    b = (1 - endog) * np.log(1 - p_hat)
    obj = -np.sum(a + b)
    if np.isnan(obj):
      obj = float('inf')
    return obj

  def fit(self, x_data, y_data, n_s=50, ranges=None):
    """Fit GLM model."""

    if ranges is None:
      ranges = [(-5, 5)] * self.degrees_of_freedom

    # Build data for fits:
    endog = y_data
    exog = np.empty((len(x_data), self.degrees_of_freedom))
    ci = 0
    for pi, coeff_indicator in enumerate(self.poly_signature):
      if coeff_indicator:
        exog[:, ci] = np.power(self.x_transform_fcn(x_data), pi)
        ci += 1
      elif not coeff_indicator:
        pass
      else:
        raise RuntimeError(
            'Unrecognized value provided in poly_signature; expected True or False, received:{}'
            .format(coeff_indicator))

    beta_hat, nll, _, _ = brute(
        self.nll, ranges=ranges, args=(exog, endog), Ns=n_s, full_output=True)
    beta_hat_poly = np.zeros(len(self.poly_signature))
    ci = 0
    for pi, coeff_indicator in enumerate(self.poly_signature):
      if coeff_indicator:
        beta_hat_poly[pi] = beta_hat[ci]
        ci += 1
      elif not coeff_indicator:
        pass
      else:
        raise RuntimeError(
            'Unrecognized value provided in poly_signature; expected True or False, received:{}'
            .format(coeff_indicator))
    aic = 2 * self.degrees_of_freedom + 2 * nll
    return beta_hat_poly, nll, aic

  def plot_fit_sequence(self, dataset, figsize_single=3, fontsize=8):
    """Create a sequence of plots thaty illustrate the fit quality of the model."""

    beta_hat_poly, _, aic = dataset.fit_glm(self)

    n_plots = 6
    fig, ax = plt.subplots(
        1, n_plots, figsize=(n_plots * figsize_single, figsize_single))
    plot_yx = self.link_fcn_pair == self.x_transform_fcn_pair

    extra_lineplot_data = {}
    extra_lineplot_data['xfunc'] = lambda x: np.linspace(0, 1, 100001)
    extra_lineplot_data['yfunc'] = lambda x: self.link_fcn_inv(  # pylint: disable=g-long-lambda
        np.polyval(beta_hat_poly[::-1], self.x_transform_fcn(x)))
    extra_lineplot_data['label'] = 'AIC={AIC:.2f}'.format(AIC=aic)
    axi_nofit = 0, None
    axi_fit = n_plots - 1, [extra_lineplot_data]
    for axi, extra_lineplot_list in [axi_nofit, axi_fit]:
      emp_acc_label = 'binned acc.' if axi == 0 else None
      dataset.plot_emperical_accuracy(
          ax[axi],
          xlim=(0, 1),
          ylim=(0, 1),
          fontsize=fontsize,
          extra_lineplot_list=extra_lineplot_list,
          show_legend=True,
          emp_acc_label=emp_acc_label)

    extra_lineplot_data = {}
    extra_lineplot_data['xfunc'] = lambda x: np.linspace(  # pylint: disable=g-long-lambda
        x.min(), x.max(), 100001)
    extra_lineplot_data['yfunc'] = lambda x: self.link_fcn_inv(  # pylint: disable=g-long-lambda
        np.polyval(beta_hat_poly[::-1], x))
    axi_nofit = 1, None
    axi_fit = n_plots - axi_nofit[0] - 1, [extra_lineplot_data]
    for axi, extra_lineplot_list in [axi_nofit, axi_fit]:
      dataset.plot_emperical_accuracy(
          ax[axi],
          transform_x=self.x_transform_fcn,
          ylim=(0, 1),
          plot_yx=False,
          fontsize=fontsize,
          xlabel_formatter=self.x_transform_fcn_pair.value.str_formatter,
          extra_lineplot_list=extra_lineplot_list,
          show_legend=False)

    extra_lineplot_data = {}
    extra_lineplot_data['xfunc'] = lambda x: np.linspace(  # pylint: disable=g-long-lambda
        x.min(), x.max(), 100001)
    extra_lineplot_data['yfunc'] = lambda x: np.polyval(beta_hat_poly[::-1], x)
    axi_nofit = 2, None
    axi_fit = n_plots - axi_nofit[0] - 1, [extra_lineplot_data]
    for axi, extra_lineplot_list in [axi_nofit, axi_fit]:
      dataset.plot_emperical_accuracy(
          ax[axi],
          transform_x=self.x_transform_fcn,
          transform_y=self.link_fcn,
          plot_yx=plot_yx,
          fontsize=fontsize,
          xlabel_formatter=self.x_transform_fcn_pair.value.str_formatter,
          ylabel_formatter=self.link_fcn_pair.value.str_formatter,
          extra_lineplot_list=extra_lineplot_list,
          show_legend=False)

    fig.tight_layout(pad=.2, rect=[0, 0.03, 1, 0.9], w_pad=.5)
    fig.suptitle(
        self.get_title_string(beta_hat_poly), fontsize=int(1.2 * fontsize))

    return fig

  def get_title_string(self, beta_hat_poly):
    """Construct a title string from class attributes."""

    beta_hat_str_list = []
    for bi, beta_hat in enumerate(beta_hat_poly):
      if self.poly_signature[bi]:
        curr_beta_hat_str = f'$\\beta_{bi}={beta_hat:.2f}$'
        beta_hat_str_list.append(curr_beta_hat_str)
    beta_hat_str = '  '.join(beta_hat_str_list)

    title_string = '{func_str}    {beta_hat_str}'.format(
        func_str=self.func_str, beta_hat_str=beta_hat_str)

    return title_string

  def plot_calibration(self,
                       ax,
                       dataset,
                       plot_yx=True,
                       n_samples=10000,
                       color='b',
                       linestyle='-',
                       linewidth=1):
    """Create a plot of the calibration function."""

    x = np.linspace(0, 1., 1000)
    if plot_yx:
      ax.plot(x, x, linestyle='--', color='k', label='y=x')

    alpha = self.beta_fit_data['data'][dataset.model]['a']
    beta = self.beta_fit_data['data'][dataset.model]['b']
    a = self.glm_fit_data['data'][dataset.model][
        self.name]['b0']['mean']['value']
    b = self.glm_fit_data['data'][dataset.model][
        self.name]['b1']['mean']['value']

    dist = self.get_true_dist(
        n_samples=n_samples, alpha=alpha, beta=beta, a=a, b=b)
    true_ce = self.get_calibration_error_beta_dist(dataset, n_samples=n_samples)
    lines = ax.plot(
        x,
        dist.eval(x),
        label=dataset.model + ' CE= {:.2f}'.format(100 * true_ce),
        color=color)
    lines[0].set_color(color)
    lines[0].set_linestyle(linestyle)
    lines[0].set_linewidth(linewidth)


def get_glm_model_container(data_dir):
  """Get a container of GLMModels that can be fit to the calibration function."""

  class GLMModels(Enum):
    """Class containing all available GLMModels as attributes."""

    log_log_b1 = GLMModel(Transforms.log, Transforms.log, [False, True],
                          r'$\log (\mu) = \beta_1 \log (x)$',
                          TrueTwoParamPolynomial, 'log_log_b1', data_dir)
    log_log_b0 = GLMModel(Transforms.log, Transforms.log, [True],
                          r'$\log (\mu) = \beta_0$', TrueTwoParamPolynomial,
                          'log_log_b0', data_dir)
    log_log_b0_b1 = GLMModel(Transforms.log, Transforms.log, [True, True],
                             r'$\log (\mu) = \beta_0 + \beta_1 \log (x)$',
                             TrueTwoParamPolynomial, 'log_log_b0_b1', data_dir)

    logflip_logflip_b1 = GLMModel(Transforms.logflip, Transforms.logflip,
                                  [False, True],
                                  r'$\log (1-\mu) = \beta_1 \log (1-x)$',
                                  TrueTwoParamFlipPolynomial,
                                  'logflip_logflip_b1', data_dir)
    logflip_logflip_b0 = GLMModel(Transforms.logflip, Transforms.logflip,
                                  [True], r'$\log (1-\mu) = \beta_0',
                                  TrueTwoParamFlipPolynomial,
                                  'logflip_logflip_b0', data_dir)
    logflip_logflip_b0_b1 = GLMModel(
        Transforms.logflip, Transforms.logflip, [True, True],
        r'$\log (1-\mu) = \beta_0 + \beta_1 \log (1-x)$',
        TrueTwoParamFlipPolynomial, 'logflip_logflip_b0_b1', data_dir)

    logit_logit_b1 = GLMModel(
        Transforms.logit, Transforms.logit, [False, True],
        r'$\log \frac{\mu}{1-\mu} = \beta_1 \log \frac{x}{1-x}$',
        TrueLogisticLogOdds, 'logit_logit_b1', data_dir)
    logit_logit_b0 = GLMModel(Transforms.logit, Transforms.logit, [True],
                              r'$\log \frac{\mu}{1-\mu} = \beta_0',
                              TrueLogisticLogOdds, 'logit_logit_b0', data_dir)
    logit_logit_b0_b1 = GLMModel(
        Transforms.logit, Transforms.logit, [True, True],
        r'$\log \frac{\mu}{1-\mu} = \beta_0 + \beta_1 \log \frac{x}{1-x}$',
        TrueLogisticLogOdds, 'logit_logit_b0_b1', data_dir)

    logit_logflip_b1 = GLMModel(
        Transforms.logit, Transforms.logflip, [False, True],
        r'$\log \frac{\mu}{1-\mu} = \beta_1 \log (1-x)$',
        TrueLogisticTwoParamFlipPolynomial, 'logit_logflip_b1', data_dir)
    logit_logflip_b0 = GLMModel(Transforms.logit, Transforms.logflip, [True],
                                r'$\log \frac{\mu}{1-\mu} = \beta_0$',
                                TrueLogisticTwoParamFlipPolynomial,
                                'logit_logflip_b0', data_dir)
    logit_logflip_b0_b1 = GLMModel(
        Transforms.logit, Transforms.logflip, [True, True],
        r'$\log \frac{\mu}{1-\mu} = \beta_0 + \beta_1 \log (1-x)$',
        TrueLogisticTwoParamFlipPolynomial, 'logit_logflip_b0_b1', data_dir)

  return GLMModels
