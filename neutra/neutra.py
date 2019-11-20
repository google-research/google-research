# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python2, python3
# pylint: disable=invalid-name,g-bad-import-order,missing-docstring
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import copy
import time

import gin
import numpy as np
import scipy.optimize as sp_opt
import simplejson
from six.moves import range
from six.moves import zip
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple

from neutra import utils


tfd = tfp.distributions
tfb = tfp.bijectors


@gin.configurable("affine_bijector")
def MakeAffineBijectorFn(num_dims, train=False, use_tril=False):
  mu = tf.get_variable("mean", initializer=tf.zeros([num_dims]))
  if use_tril:
    tril_flat = tf.get_variable("tril_flat", [num_dims * (num_dims + 1) // 2])
    tril_raw = tfp.math.fill_triangular(tril_flat)
    sigma = tf.nn.softplus(tf.matrix_diag_part(tril_raw))
    tril = tf.linalg.set_diag(tril_raw, sigma)
    return tfb.Affine(shift=mu, scale_tril=tril)
  else:
    sigma = tf.nn.softplus(
        tf.get_variable("invpsigma", initializer=tf.zeros([num_dims])))
    return tfb.Affine(shift=mu, scale_diag=sigma)


@gin.configurable("rnvp_bijector")
def MakeRNVPBijectorFn(num_dims,
                       num_stages,
                       hidden_layers,
                       scale=1.0,
                       activation=tf.nn.elu,
                       train=False,
                       learn_scale=False,
                       dropout_rate=0.0):
  swap = tfb.Permute(permutation=np.arange(num_dims - 1, -1, -1))

  bijectors = []
  for i in range(num_stages):
    _rnvp_template = utils.DenseShiftLogScale(
        "rnvp_%d" % i,
        hidden_layers=hidden_layers,
        activation=activation,
        kernel_initializer=utils.L2HMCInitializer(factor=0.01),
        dropout_rate=dropout_rate,
        train=train)

    def rnvp_template(x, output_units, t=_rnvp_template):
      # TODO: I don't understand why the shape gets lost.
      x.set_shape([None, num_dims - output_units])
      return t(x, output_units)

    bijectors.append(
        tfb.RealNVP(
            num_masked=num_dims // 2, shift_and_log_scale_fn=rnvp_template))
    bijectors.append(swap)
  # Drop the last swap.
  bijectors = bijectors[:-1]
  if learn_scale:
    scale = tf.nn.softplus(
        tf.get_variable(
            "isp_global_scale", initializer=tfp.math.softplus_inverse(scale)))
  bijectors.append(tfb.Affine(scale_identity_multiplier=scale))

  bijector = tfb.Chain(bijectors)

  # Construct the variables
  _ = bijector.forward(tf.zeros([1, num_dims]))

  return bijector


@gin.configurable("iaf_bijector")
def MakeIAFBijectorFn(
    num_dims,
    num_stages,
    hidden_layers,
    scale=1.0,
    activation=tf.nn.elu,
    train=False,
    dropout_rate=0.0,
    learn_scale=False,
):
  swap = tfb.Permute(permutation=np.arange(num_dims - 1, -1, -1))

  bijectors = []
  for i in range(num_stages):
    _iaf_template = utils.DenseAR(
        "iaf_%d" % i,
        hidden_layers=hidden_layers,
        activation=activation,
        kernel_initializer=utils.L2HMCInitializer(factor=0.01),
        dropout_rate=dropout_rate,
        train=train)

    def iaf_template(x, t=_iaf_template):
      # TODO: I don't understand why the shape gets lost.
      x.set_shape([None, num_dims])
      return t(x)

    bijectors.append(
        tfb.Invert(
            tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=iaf_template)))
    bijectors.append(swap)
  # Drop the last swap.
  bijectors = bijectors[:-1]
  if learn_scale:
    scale = tf.nn.softplus(
        tf.get_variable(
            "isp_global_scale", initializer=tfp.math.softplus_inverse(scale)))
  bijectors.append(tfb.Affine(scale_identity_multiplier=scale))

  bijector = tfb.Chain(bijectors)

  # Construct the variables
  _ = bijector.forward(tf.zeros([1, num_dims]))

  return bijector


TargetSpec = collections.namedtuple(
    "TargetSpec", "name, num_dims,"
    "x_min, x_max, y_min, y_max, stats, bijector")


@gin.configurable("target_spec")
def GetTargetSpec(
    name,
    num_dims = 100,
    t_dof = 1.0,
    regression_dataset = "covertype",
    regression_num_points = 0,
    regression_normalize = False,
    regression_hier_type = "none",  # none, centered, non_centered
    regression_beta_prior = "normal",  # normal, student_t
    regression_type = "regular",  # regular, gamma_scales
    regression_use_beta_scales = True,
    eig_source = "linear",
    batch_size = 0,
    regression_stochastic_points = 0,
    gamma_shape = 0.5,
    precomputed_stats_path = None,
    **kwargs):
  if name == "funnel":
    spec = TargetSpec(
        name=name,
        num_dims=num_dims,
        x_min=-4.0,
        x_max=4.0,
        y_min=-10.0,
        y_max=10.0,
        stats=None,
        bijector=None)

    def funnel_forward(x):
      shift = tf.zeros_like(x)
      log_scale = tf.concat(
          [tf.zeros_like(x[Ellipsis, :1]),
           tf.tile(x[Ellipsis, :1], [1, num_dims - 1])], -1)
      return shift, log_scale

    mg = tfd.MultivariateNormalDiag(
        loc=tf.zeros(num_dims), scale_identity_multiplier=1.0)
    target = tfd.TransformedDistribution(
        mg, bijector=tfb.MaskedAutoregressiveFlow(funnel_forward))
  elif name == "ill_cond_gaussian":
    # For backwards compatibility with earlier experiments.
    spec = TargetSpec(
        name=name,
        num_dims=num_dims,
        x_min=-5.0,
        x_max=5.0,
        y_min=-5.0,
        y_max=5.0,
        stats=None,
        bijector=None)
    rng = np.random.RandomState(seed=10)
    diag_precisions = np.linspace(1., 1000., num_dims)**-1
    q, _ = np.linalg.qr(rng.randn(num_dims, num_dims))
    scg_prec = (q * diag_precisions).dot(q.T)
    scg_prec = scg_prec.astype(np.float32)
    scg_var = np.linalg.inv(scg_prec) / 1000.0
    target = tfd.MultivariateNormalFullCovariance(
        loc=tf.zeros(num_dims), covariance_matrix=scg_var)
  elif name == "new_ill_cond_gaussian":
    spec = TargetSpec(
        name=name,
        num_dims=num_dims,
        x_min=-5.0,
        x_max=5.0,
        y_min=-5.0,
        y_max=5.0,
        stats=None,
        bijector=None)
    rng = np.random.RandomState(seed=10)
    if eig_source == "linear":
      eigenvalues = np.linspace(1., 1000., num_dims)**-1
    elif eig_source == "gamma":
      eigenvalues = np.sort(
          rng.gamma(shape=gamma_shape, scale=1.,
                    size=num_dims)).astype(np.float32)
    q, _ = np.linalg.qr(rng.randn(num_dims, num_dims))
    covariance = (q * eigenvalues**-1).dot(q.T).astype(np.float32)
    target = tfd.MultivariateNormalFullCovariance(
        loc=tf.zeros(num_dims), covariance_matrix=covariance)
  elif name == "ill_cond_t":
    # For backwards compatibility with earlier experiments.
    spec = TargetSpec(
        name=name,
        num_dims=num_dims,
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        stats=None,
        bijector=None)
    rng = np.random.RandomState(seed=10)
    diag_precisions = np.linspace(1., 1000., num_dims)**-1
    q, _ = np.linalg.qr(rng.randn(num_dims, num_dims))
    scg_prec = (q * diag_precisions).dot(q.T)
    scg_prec = scg_prec.astype(np.float32)
    scg_var = np.linalg.inv(scg_prec) / 1000.0

    scale = tf.linalg.LinearOperatorFullMatrix(scg_var)
    target = tfd.MultivariateStudentTLinearOperator(
        loc=tf.zeros(num_dims), scale=scale, df=t_dof)
  elif name == "new_ill_cond_t":
    spec = TargetSpec(
        name=name,
        num_dims=num_dims,
        x_min=-5.0,
        x_max=5.0,
        y_min=-5.0,
        y_max=5.0,
        stats=None,
        bijector=None)
    rng = np.random.RandomState(seed=10)
    if eig_source == "linear":
      eigenvalues = np.linspace(1., 1000., num_dims)**-1
    elif eig_source == "gamma":
      eigenvalues = np.sort(rng.gamma(shape=0.5, scale=1.,
                                      size=num_dims)).astype(np.float32)
    q, _ = np.linalg.qr(rng.randn(num_dims, num_dims))
    covariance = (q * eigenvalues**-1).dot(q.T).astype(np.float32)

    scale = tf.linalg.LinearOperatorFullMatrix(covariance)
    target = tfd.MultivariateStudentTLinearOperator(
        loc=tf.zeros(num_dims), scale=scale, df=t_dof)
  elif name == "logistic_reg":
    if regression_hier_type == "none":
      extra_dims = 0
    else:
      extra_dims = 2

    if regression_dataset == "covertype":
      x, y = utils.LoadCovertype()
      if regression_num_points > 0:
        rng = np.random.RandomState(seed=10)
        chosen_rows = rng.choice(
            x.shape[0], regression_num_points, replace=False)
        x = x[chosen_rows]
        y = y[chosen_rows]

      num_features = x.shape[-1] + 1
      num_classes = 7
      num_dims = num_features * num_classes + extra_dims

      x = tf.to_float(x)
      y = tf.to_int32(y)
    elif regression_dataset == "german":
      x, y = utils.LoadGerman()

      num_features = int(x.shape[-1]) + 1
      num_classes = 2
      num_dims = num_features * num_classes + extra_dims

      x = tf.to_float(x)
      y = tf.to_int32(y)

      if regression_num_points > 0:
        rng = np.random.RandomState(seed=10)
        chosen_rows = rng.choice(
            x.shape[0], regression_num_points, replace=False)
        x = tf.gather(x, chosen_rows)
        y = tf.gather(y, chosen_rows)

    if regression_stochastic_points > 0:
      chosen_rows = tf.random.uniform([int(regression_stochastic_points)],
                                      0,
                                      int(x.shape[0]),
                                      dtype=tf.int32)
      x = tf.gather(x, chosen_rows)
      y = tf.gather(y, chosen_rows)

    if regression_normalize:
      x_min = tf.reduce_min(x, 0, keep_dims=True)
      x_max = tf.reduce_max(x, 0, keep_dims=True)

      x /= (x_max - x_min)
      x = 2.0 * x - 1.0

    x = tf.concat([x, tf.ones([int(x.shape[0]), 1])], -1)

    def regular_log_prob_fn(params):
      if regression_hier_type == "none":
        beta = params
        beta_scaled = beta
      elif regression_hier_type == "centered":
        mu_0 = params[Ellipsis, -1]
        tau_0 = tf.nn.softplus(params[Ellipsis, -2])
        beta = params[Ellipsis, :-2]
        beta_scaled = beta
      elif regression_hier_type == "non_centered":
        mu_0 = params[Ellipsis, -1]
        tau_0 = tf.nn.softplus(params[Ellipsis, -2])
        beta = params[Ellipsis, :-2]
        beta_scaled = beta / tf.expand_dims(tau_0, -1) + tf.expand_dims(
            mu_0, -1)
      else:
        raise ValueError("Unknown regression_hier_type:" + regression_hier_type)

      if batch_size:

        def body(_, i):
          y_dist = tfd.Categorical(
              logits=tf.einsum(
                  "ij,kjm->kim", x[i:i + batch_size],
                  tf.reshape(beta_scaled, [-1, num_features, num_classes])))
          return tf.reduce_sum(y_dist.log_prob(y[i:i + batch_size]), -1)

        log_prob = tf.reduce_sum(
            tf.scan(
                body,
                tf.range(0, x.shape[0], batch_size),
                initializer=tf.zeros(tf.shape(params)[:1]),
                parallel_iterations=1), 0)
      else:
        y_dist = tfd.Categorical(
            logits=tf.einsum(
                "ij,kjm->kim", x,
                tf.reshape(beta_scaled, [-1, num_features, num_classes])))
        log_prob = tf.reduce_sum(y_dist.log_prob(y), -1)

      def make_beta_dist(loc, scale):
        if regression_beta_prior == "normal":
          return tfd.Normal(loc=loc, scale=scale)
        else:
          if tf.convert_to_tensor(loc).shape.ndims == 0:
            loc = tf.fill(
                tf.stack([tf.shape(params)[0], num_features * num_classes]),
                loc)
          if tf.convert_to_tensor(scale).shape.ndims == 0:
            scale = tf.fill(
                tf.stack([tf.shape(params)[0], num_features * num_classes]),
                scale)

          scale = tf.linalg.LinearOperatorDiag(scale)
          return tfd.MultivariateStudentTLinearOperator(
              loc=loc, scale=scale, df=t_dof)

      if regression_hier_type == "none":
        beta_dist = make_beta_dist(loc=0.0, scale=10.0)
      else:
        mu_0_dist = tfd.Normal(loc=0.0, scale=10.0)
        tau_0_dist = tfd.Gamma(2.0, 1.0)
        log_prob += mu_0_dist.log_prob(mu_0) + tau_0_dist.log_prob(tau_0)

        if regression_hier_type == "centered":
          mu_0 = tf.tile(
              tf.expand_dims(mu_0, -1), [1, num_features * num_classes])
          tau_0 = tf.tile(
              tf.expand_dims(tau_0, -1), [1, num_features * num_classes])
          beta_dist = make_beta_dist(loc=mu_0, scale=1.0 / tau_0)
        elif regression_hier_type == "non_centered":
          beta_dist = make_beta_dist(loc=0.0, scale=1.0)
      log_prob += tf.reduce_sum(beta_dist.log_prob(beta), -1)
      return log_prob

    def gamma_scales_log_prob_fn(params):
      assert num_classes == 2

      def unmarshal(params):
        results = []
        n_dimensions_used = 0
        if regression_use_beta_scales:
          dim_list = [num_features, num_features, 1]
        else:
          dim_list = [num_features, 1]
        for n_to_add in dim_list:
          results.append(
              params[Ellipsis, n_dimensions_used:n_dimensions_used + n_to_add])
          n_dimensions_used += n_to_add
        return tuple(results)

      log_prob = 0.
      if regression_use_beta_scales:
        beta, beta_log_scales, overall_log_scale = unmarshal(params)
        # p(per-variable scales)
        log_prob += tf.reduce_sum(
            tfd.TransformedDistribution(
                tfd.Gamma(0.5, 0.5),
                tfb.Invert(tfb.Exp())).log_prob(beta_log_scales), -1)
      else:
        beta, overall_log_scale = unmarshal(params)
        beta_log_scales = 0.0
      # p(overall scale)
      log_prob += tf.reduce_sum(
          tfd.Normal(0., 10.).log_prob(overall_log_scale), -1)
      # p(beta)
      log_prob += tf.reduce_sum(tfd.Normal(0., 1.).log_prob(beta), -1)
      # p(y | x, beta)
      scaled_beta = beta * tf.exp(overall_log_scale) * tf.exp(beta_log_scales)
      if batch_size:

        def body(_, i):
          logits = tf.einsum("nd,md->mn", x[i:i + batch_size], scaled_beta)
          return tf.reduce_sum(
              tfd.Bernoulli(logits=logits).log_prob(y[i:i + batch_size]), -1)

        log_prob += tf.reduce_sum(
            tf.scan(
                body,
                tf.range(0, x.shape[0], batch_size),
                initializer=tf.zeros(tf.shape(params)[:1]),
                parallel_iterations=1), 0)
      else:
        logits = tf.einsum("nd,md->mn", x, scaled_beta)
        log_prob += tf.reduce_sum(tfd.Bernoulli(logits=logits).log_prob(y), -1)
      return log_prob

    def horseshoe_log_prob_fn(params):
      assert num_classes == 2

      (z, r1_local, r2_local, r1_global, r2_global) = tf.split(
          params, [num_features, num_features, num_features, 1, 1], axis=-1)

      def indep(d):
        return tfd.Independent(d, 1)

      zero = tf.zeros(num_features)
      one = tf.ones(num_features)
      half = 0.5 * one

      p_z = indep(tfd.Normal(zero, one))
      p_r1_local = indep(tfd.HalfNormal(one))
      p_r2_local = indep(tfd.InverseGamma(half, half))

      p_r1_global = indep(tfd.HalfNormal([1.]))
      p_r2_global = indep(tfd.InverseGamma([0.5], [0.5]))

      log_prob = (
          p_z.log_prob(z) + p_r1_local.log_prob(r1_local) +
          p_r2_local.log_prob(r2_local) +
          p_r1_global.log_prob(r1_global) +
          p_r2_global.log_prob(r2_global))

      lambda_ = r1_local * tf.sqrt(r2_local)
      tau = r1_global * tf.sqrt(r2_global)
      beta = z * lambda_ * tau

      if batch_size:

        def body(_, i):
          logits = tf.einsum("nd,md->mn", x[i:i + batch_size], beta)
          return tfd.Independen(tfd.Bernoulli(logits=logits),
                                1).log_prob(y[i:i + batch_size])

        log_prob += tf.reduce_sum(
            tf.scan(
                body,
                tf.range(0, x.shape[0], batch_size),
                initializer=tf.zeros(tf.shape(params)[:1]),
                parallel_iterations=1), 0)
      else:
        logits = tf.einsum("nd,md->mn", x, beta)
        log_prob += tfd.Independent(tfd.Bernoulli(logits=logits), 1).log_prob(y)
      return log_prob

    def gamma_scales2_log_prob_fn(params):
      assert num_classes == 2

      (z, local_scale, global_scale) = tf.split(
          params, [num_features, num_features, 1], axis=-1)

      def indep(d):
        return tfd.Independent(d, 1)

      zero = tf.zeros(num_features)
      one = tf.ones(num_features)
      half = 0.5 * one

      p_z = indep(tfd.Normal(zero, one))
      p_local_scale = indep(tfd.Gamma(half, half))
      p_global_scale = indep(tfd.Gamma([0.5], [0.5]))

      log_prob = (
          p_z.log_prob(z) + p_local_scale.log_prob(local_scale) +
          p_global_scale.log_prob(global_scale))

      beta = z * local_scale * global_scale

      if batch_size:

        def body(_, i):
          logits = tf.einsum("nd,md->mn", x[i:i + batch_size], beta)
          return tfd.Independen(tfd.Bernoulli(logits=logits),
                                1).log_prob(y[i:i + batch_size])

        log_prob += tf.reduce_sum(
            tf.scan(
                body,
                tf.range(0, x.shape[0], batch_size),
                initializer=tf.zeros(tf.shape(params)[:1]),
                parallel_iterations=1), 0)
      else:
        logits = tf.einsum("nd,md->mn", x, beta)
        log_prob += tfd.Independent(tfd.Bernoulli(logits=logits), 1).log_prob(y)
      return log_prob

    bijector = None
    if regression_type == "regular":
      log_prob_fn = regular_log_prob_fn
    elif regression_type == "gamma_scales":
      log_prob_fn = gamma_scales_log_prob_fn
      num_dims = num_features + 1
      if regression_use_beta_scales:
        num_dims += num_features
    elif regression_type == "horseshoe":
      log_prob_fn = horseshoe_log_prob_fn
      num_dims = num_features * 3 + 2
      bijector = tfb.Blockwise([tfb.Identity(), tfb.Exp()],
                               [num_features, num_features * 2 + 2])
    elif regression_type == "gamma_scales2":
      log_prob_fn = gamma_scales2_log_prob_fn
      num_dims = num_features * 2 + 1
      bijector = tfb.Blockwise([tfb.Identity(), tfb.Exp()],
                               [num_features, num_features + 1])


    target = utils.LogProbDist(num_dims=num_dims, log_prob_fn=log_prob_fn)
    spec = TargetSpec(
        name=name,
        num_dims=num_dims,
        x_min=0.10,
        x_max=0.15,
        y_min=0.10,
        y_max=0.15,
        stats=None,
        bijector=bijector)
  elif name == "mog":
    comp_1 = tfd.MultivariateNormalDiag(
        loc=[-1., 1.] + [0.] * (num_dims - 2), scale_identity_multiplier=2.)
    comp_2 = tfd.MultivariateNormalDiag(
        loc=[1., 1.] + [0.] * (num_dims - 2), scale_identity_multiplier=4.)
    comp_3 = tfd.MultivariateNormalDiag(
        loc=[0., 0.] + [0.] * (num_dims - 2), scale_identity_multiplier=2.)
    cat = tfd.Categorical(logits=[0] * 3)
    target = tfd.Mixture(cat=cat, components=[comp_1, comp_2, comp_3])
    spec = TargetSpec(
        name=name, num_dims=num_dims, x_min=-2., x_max=2., y_min=-2., y_max=2.,
        stats=None,
        bijector=None)
  elif name == "easy_gaussian":
    spec = TargetSpec(
        name=name,
        num_dims=num_dims,
        x_min=-5.0,
        x_max=5.0,
        y_min=-5.0,
        y_max=5.0,
        stats=None,
        bijector=None)
    rng = np.random.RandomState(seed=10)
    eigenvalues = np.linspace(0.5, 2., num_dims)**-1
    q, _ = np.linalg.qr(rng.randn(num_dims, num_dims))
    covariance = (q * eigenvalues**-1).dot(q.T).astype(np.float32)
    target = tfd.MultivariateNormalFullCovariance(
        loc=tf.zeros(num_dims), covariance_matrix=covariance)
  elif name == "gp_reg":
    x, y = utils.LoadCloud()

    if regression_num_points > 0:
      rng = np.random.RandomState(seed=10)
      chosen_rows = rng.choice(
          x.shape[0], regression_num_points, replace=False)
      x = x[chosen_rows]
      y = y[chosen_rows]

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    num_features = int(x.shape[-1])
    num_dims = num_features + 2

    def log_prob_fn(params):
      rho, alpha, sigma = tf.split(params, [num_features, 1, 1], -1)

      one = tf.ones(num_features)
      def indep(d):
        return tfd.Independent(d, 1)
      p_rho = indep(tfd.InverseGamma(5. * one, 5. * one))
      p_alpha = indep(tfd.HalfNormal([1.]))
      p_sigma = indep(tfd.HalfNormal([1.]))

      rho_shape = tf.shape(rho)
      alpha_shape = tf.shape(alpha)

      x1 = tf.expand_dims(x, -2)
      x2 = tf.expand_dims(x, -3)
      exp = -0.5 * tf.squared_difference(x1, x2)
      exp /= tf.reshape(tf.square(rho), tf.concat([rho_shape[:1], [1, 1], rho_shape[1:]], 0))
      exp = tf.reduce_sum(exp, -1, keep_dims=True)
      exp += 2. * tf.reshape(tf.log(alpha), tf.concat([alpha_shape[:1], [1, 1], alpha_shape[1:]], 0))
      exp = tf.exp(exp[Ellipsis, 0])
      exp += tf.matrix_diag(tf.tile(tf.square(sigma), [1, int(x.shape[0])]) + 1e-6)
      exp = tf.check_numerics(exp, "exp 2 has NaNs")
      with tf.control_dependencies([tf.print(exp[0], summarize=99999)]):
        exp = tf.identity(exp)

      p_y = tfd.MultivariateNormalFullCovariance(
          covariance_matrix=exp)

      log_prob = (
          p_rho.log_prob(rho) + p_alpha.log_prob(alpha) +
          p_sigma.log_prob(sigma) + p_y.log_prob(y))

      return log_prob

    bijector = tfb.Softplus()#tfb.Exp()
    target = utils.LogProbDist(num_dims=num_dims, log_prob_fn=log_prob_fn)
    spec = TargetSpec(
        name=name,
        num_dims=num_dims,
        x_min=0.10,
        x_max=0.15,
        y_min=0.10,
        y_max=0.15,
        stats=None,
        bijector=bijector)


  if precomputed_stats_path is not None:
    with tf.gfile.Open(precomputed_stats_path) as f:
      stats = simplejson.load(f)
      stats = {k: np.array(v) for k, v in stats.items()}
      spec = spec._replace(stats=stats)

  return target, spec._replace(**kwargs)


NeuTraOutputs = collections.namedtuple(
    "NeuTraOutputs", "x_chain, p_accept, kernel_results, log_accept_ratio")


@gin.configurable("neutra")
def MakeNeuTra(target,
               q,
               batch_size=32,
               num_steps=100,
               num_leapfrog_steps=2,
               step_size=0.1):
  x_init = q.sample(batch_size)

  kernel = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target.log_prob,
      step_size=step_size,
      num_leapfrog_steps=num_leapfrog_steps)

  kernel = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=kernel, bijector=q.bijector)

  x_chain, kernel_results = tfp.mcmc.sample_chain(
      num_results=num_steps, current_state=x_init, kernel=kernel)

  log_accept_ratio = kernel_results.inner_results.log_accept_ratio

  p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))

  x_fin = tf.stop_gradient(x_chain[-1, Ellipsis])

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))

  return NeuTraOutputs(
      x_chain=x_chain,
      p_accept=p_accept,
      kernel_results=kernel_results,
      log_accept_ratio=log_accept_ratio)


QStats = collections.namedtuple("QStats", "bias")


def ComputeQStats(q_samples, target_mean):
  return QStats(bias=target_mean - tf.reduce_mean(q_samples, 0))


ChainStats = collections.namedtuple(
    "ChainStats",
    "bias, inst_bias, variance, inst_variance, error_sq, ess, ess_per_grad,"
    "rhat, autocorr, warmupped_bias, warmupped_variance")


def ComputeChainStats(chain, target_mean, num_leapfrog_steps):
  # Chain is [num_steps, batch, num_dims]
  num_steps = tf.shape(chain)[0]
  counts = tf.to_float(tf.range(1, num_steps + 1))
  chain_mean = tf.cumsum(chain, 0) / counts[:, tf.newaxis, tf.newaxis]

  bias = target_mean - tf.reduce_mean(chain_mean, 1)
  variance = tf.reduce_mean(
      tf.square(chain_mean - tf.reduce_mean(chain_mean, 1, keep_dims=True)), 1)
  inst_bias = target_mean - tf.reduce_mean(chain, 1)
  inst_variance = tf.reduce_mean(tf.square(target_mean - chain), 1)

  def reducer(_, idx):
    chain_mean = tf.reduce_mean(chain[idx // 2:idx], 0)
    bias = tf.reduce_mean(target_mean - chain_mean, 0)
    variance = tf.reduce_mean(
        tf.square(chain_mean - tf.reduce_mean(chain_mean, 0)), 0)
    return bias, variance

  indices = 1 + tf.range(num_steps)
  warmupped_bias, warmupped_variance = tf.scan(
      reducer, indices, initializer=(chain[0, 0], chain[0, 0]))

  half_steps = num_steps // 2
  half_chain = chain[half_steps:]

  error_sq = tf.reduce_mean(
      tf.square(tf.reduce_mean(half_chain, 0) - target_mean), 0)

  ess = utils.EffectiveSampleSize(half_chain) / tf.to_float(half_steps)
  ess_per_grad = ess / tf.to_float(num_leapfrog_steps)
  rhat = tfp.mcmc.potential_scale_reduction(half_chain)
  autocorr = tf.reduce_mean(
      utils.SanitizedAutoCorrelation(half_chain, 0, max_lags=300), 1)

  return ChainStats(
      bias=bias,
      variance=variance,
      error_sq=error_sq,
      inst_bias=inst_bias,
      inst_variance=inst_variance,
      ess=ess,
      ess_per_grad=ess_per_grad,
      rhat=rhat,
      warmupped_bias=warmupped_bias,
      warmupped_variance=warmupped_variance,
      autocorr=autocorr)


TuneOutputs = collections.namedtuple("TuneOutputs",
                                     "num_leapfrog_steps, step_size")


@gin.configurable("neutra_experiment")
class NeuTraExperiment(object):

  def __init__(self,
               train_batch_size = 4096,
               test_chain_batch_size = 4096,
               bijector = "iaf",
               log_dir="/tmp/neutra",
               base_learning_rate=1e-3,
               q_base_scale=1.,
               learning_rate_schedule=[[6000, 1e-1]]):
    target, target_spec = GetTargetSpec()
    self.target = target
    self.target_spec = target_spec
    with gin.config_scope("train"):
      train_target, train_target_spec = GetTargetSpec()
      self.train_target = train_target
      self.train_target_spec = train_target_spec

    if bijector == "rnvp":
      bijector_fn = tf.make_template(
          "bijector",
          MakeRNVPBijectorFn,
          num_dims=self.target_spec.num_dims)
    elif bijector == "iaf":
      bijector_fn = tf.make_template(
          "bijector",
          MakeIAFBijectorFn,
          num_dims=self.target_spec.num_dims)
    elif bijector == "affine":
      bijector_fn = tf.make_template(
          "bijector",
          MakeAffineBijectorFn,
          num_dims=self.target_spec.num_dims)
    else:
      bijector_fn = lambda *args, **kwargs: tfb.Identity()

    self.train_bijector = bijector_fn(train=True)
    self.bijector = bijector_fn(train=False)
    if train_target_spec.bijector is not None:
      print("Using train target bijector")
      self.train_bijector = tfb.Chain(
          [train_target_spec.bijector, self.train_bijector])
    if target_spec.bijector is not None:
      print("Using target bijector")
      self.bijector = tfb.Chain([target_spec.bijector, self.bijector])

    q_base = tfd.Independent(
        tfd.Normal(
            loc=tf.zeros(self.target_spec.num_dims),
            scale=q_base_scale * tf.ones(self.target_spec.num_dims)), 1)
    self.q_x_train = tfd.TransformedDistribution(q_base, self.train_bijector)
    self.q_x = tfd.TransformedDistribution(q_base, self.bijector)

    # Params
    self.train_batch_size = int(train_batch_size)
    self.test_chain_batch_size = tf.placeholder_with_default(
        test_chain_batch_size, [], "test_chain_batch_size")
    self.test_batch_size = tf.placeholder_with_default(16384 * 8, [],
                                                       "test_batch_size")
    self.test_num_steps = tf.placeholder_with_default(1000, [],
                                                      "test_num_steps")
    self.test_num_leapfrog_steps = tf.placeholder_with_default(
        tf.to_int32(2), [], "test_num_leapfrog_steps")
    self.test_step_size = tf.placeholder_with_default(0.1, [], "test_step_size")

    # Test
    self.neutra_outputs = MakeNeuTra(
        target=self.target,
        q=self.q_x,
        batch_size=self.test_chain_batch_size,
        num_steps=self.test_num_steps,
        num_leapfrog_steps=self.test_num_leapfrog_steps,
        step_size=self.test_step_size,
    )
    self.z_chain = tf.reshape(
        self.bijector.inverse(
            tf.reshape(self.neutra_outputs.x_chain,
                       [-1, self.target_spec.num_dims])),
        tf.shape(self.neutra_outputs.x_chain))
    self.target_samples = self.target.sample(self.test_batch_size)
    self.target_z = self.bijector.inverse(self.target_samples)
    self.q_samples = self.q_x.sample(self.test_batch_size)

    self.target_cov = utils.Covariance(self.target_samples)
    self.target_eigvals, self.target_eigvecs = tf.linalg.eigh(self.target_cov)

    self.cached_target_eigvals = tf.get_local_variable(
        "cached_target_eigvals",
        self.target_eigvals.shape,
        initializer=tf.zeros_initializer())
    self.cached_target_eigvecs = tf.get_local_variable(
        "cached_target_eigvecs",
        self.target_eigvecs.shape,
        initializer=tf.zeros_initializer())
    self.cached_target_stats_update_op = [
        self.cached_target_eigvals.assign(self.target_eigvals),
        self.cached_target_eigvecs.assign(self.target_eigvecs),
        tf.print("Assigning target stats")
    ]

    def variance(x):
      x -= tf.reduce_mean(x, 0, keep_dims=True)
      x = tf.square(x)
      return x

    def rotated_variance(x):
      x2 = tf.reshape(x, [-1, self.target_spec.num_dims])
      x2 -= tf.reduce_mean(x2, 0, keep_dims=True)
      x2 = tf.matmul(x2, self.cached_target_eigvecs)
      x2 = tf.square(x2)
      return tf.reshape(x2, tf.shape(x))

    functions = [
        ("mean", tf.identity),
        #        ("var", variance),
        ("square", tf.square),
        #        ("rot_square", rot_square),
        #        ("rot_var", rotated_variance),
    ]

    self.cached_target_mean = {}
    self.cached_target_mean_update_op = [tf.print("Assigning target means.")]
    self.neutra_stats = {}
    self.q_stats = {}

    for name, f in functions:
      target_mean = tf.reduce_mean(f(self.target_samples), 0)
      cached_target_mean = tf.get_local_variable(name + "_cached_mean",
                                                 target_mean.shape)
      if self.target_spec.stats is not None:
        self.cached_target_mean_update_op.append(
            cached_target_mean.assign(self.target_spec.stats[name]))
      else:
        self.cached_target_mean_update_op.append(
            cached_target_mean.assign(target_mean))

      self.cached_target_mean[name] = cached_target_mean
      self.q_stats[name] = ComputeQStats(f(self.q_samples), cached_target_mean)
      self.neutra_stats[name] = ComputeChainStats(
          f(self.neutra_outputs.x_chain), cached_target_mean,
          self.test_num_leapfrog_steps)

    # Training
    self.train_q_samples = self.q_x_train.sample(self.train_batch_size)
    self.train_log_q_x = self.q_x_train.log_prob(self.train_q_samples)
    self.kl_q_p = tf.reduce_mean(self.train_log_q_x -
                                 self.target.log_prob(self.train_q_samples))

    loss = self.kl_q_p
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if reg_losses:
      tf.logging.info("Regularizing.")
      loss += tf.add_n(reg_losses)
    self.loss = tf.check_numerics(loss, "Loss has NaNs")

    self.global_step = tf.train.get_or_create_global_step()
    steps, factors = list(zip(*learning_rate_schedule))
    learning_rate = base_learning_rate * tf.train.piecewise_constant(
        self.global_step, steps, [1.0] + list(factors))

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    self.train_op = opt.minimize(self.loss, global_step=self.global_step)

    tf.contrib.summary.scalar("kl_q_p", self.kl_q_p)
    tf.contrib.summary.scalar("loss", self.loss)

    self.init = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
        tf.print("Initializing variables")
    ]

    self.saver = tf.train.Saver()
    self.log_dir = log_dir

  def Initialize(self, sess):
    sess.run(self.init)
    sess.run(self.cached_target_stats_update_op)
    sess.run(self.cached_target_mean_update_op)

  @gin.configurable("train_bijector")
  def TrainBijector(self,
                    sess,
                    num_steps,
                    summary_every=500,
                    feed={},
                    plot_callback=None):
    times = np.zeros(num_steps)
    q_errs = []

    def summarize():
      fetch, _ = sess.run([{
          "global_step": self.global_step
      },
                           tf.contrib.summary.all_summary_ops()])
      q_errs.append(sess.run(self.q_stats))
      if plot_callback:
        plot_callback()

    for i in range(num_steps):
      if i % summary_every == 0:
        summarize()

      start_time = time.time()
      sess.run(self.train_op, feed)
      times[i] = time.time() - start_time

    summarize()

    flat_q_errs = [tf.contrib.framework.nest.flatten(s) for s in q_errs]
    trans_q_errs = list(zip(*flat_q_errs))
    concat_q_errs = [np.stack(q, 0) for q in trans_q_errs]
    q_stats = tf.contrib.framework.nest.pack_sequence_as(
        q_errs[0], concat_q_errs)

    self.saver.save(
        sess,
        self.log_dir + "/model.ckpt",
        global_step=sess.run(self.global_step))
    return q_stats, times.mean()

  @gin.configurable("eval")
  def Eval(self, sess, feed={}, p_accept_only=False):
    if p_accept_only:
      return sess.run([self.neutra_stats, self.neutra_outputs.p_accept], feed)
    else:
      return sess.run([self.neutra_stats, self.neutra_outputs], feed)

  @gin.configurable("benchmark")
  def Benchmark(self, sess, feed={}):
    start_time = time.time()
    _, num_steps = sess.run([self.neutra_outputs.x_chain, self.test_num_steps],
                            feed)
    return (time.time() - start_time) / num_steps

  @gin.configurable("tune")
  def Tune(
      self,
      sess,
      method="scipy",
      min_step_size=1e-3,
      max_step_size=1.,
      max_leapfrog_steps=50,
      de_pop_size=5,
      f_name="square",
      scale_by_target=True,
      max_num_trials=50,
      obj_type="rhat",
      percentile=5,
      feed={},
  ):
    feed = copy.copy(feed)

    def Objective(num_leapfrog_steps, step_size):
      feed.update({
          self.test_num_leapfrog_steps: num_leapfrog_steps,
          self.test_step_size: step_size,
      })
      ess_per_grad, warmupped_bias, rhat, target_stats = sess.run([
          self.neutra_stats[f_name].ess_per_grad,
          self.neutra_stats[f_name].warmupped_bias[-1],
          self.neutra_stats[f_name].rhat,
          self.cached_target_mean[f_name]
      ], feed)

      bias_sq = warmupped_bias**2

      if scale_by_target:
        bias_sq /= target_stats**2

      ess_per_grad = np.percentile(ess_per_grad, percentile)
      bias_sq = np.percentile(bias_sq, 100 - percentile)
      rhat = np.percentile(rhat, 100 - percentile)

      tf.logging.info(
          "Evaluating step_size: %f, num_leapfrog_steps: %d, got ess_per_grad %f, bias_sq %f, rhat %f",
          step_size, num_leapfrog_steps, ess_per_grad, bias_sq, rhat)

      if obj_type == 'bias':
        return np.log(bias_sq) - ess_per_grad
      elif obj_type == 'rhat':
        # If it's above 1.4 or so, ess plays no role.
        ess_factor = np.exp(-(np.maximum(rhat - 1, 0))**2 / (2 * 0.1**2))
        return rhat - ess_per_grad * ess_factor

    def unconstrain(num_leapfrog_steps, step_size):
      return float(num_leapfrog_steps), np.log(step_size)

    def constrain(num_leapfrog_steps, step_size):
      return int(num_leapfrog_steps), np.exp(step_size)

    min_bounds = unconstrain(1, min_step_size)
    max_bounds = unconstrain(max_leapfrog_steps, max_step_size)
    unconstrained_obj = lambda x: Objective(*constrain(*x))

    if method == "scipy":
      opt_res = sp_opt.differential_evolution(
          unconstrained_obj,
          maxiter=max(max_num_trials // de_pop_size // 2, 1),
          popsize=de_pop_size,
          polish=False,
          disp=True,
          bounds=[(min_bounds[0], max_bounds[0]), (min_bounds[1], max_bounds[1])]
        )
      return TuneOutputs(*constrain(*opt_res.x))
    elif method == "random":
      trials = []
      for _ in range(max_num_trials):
        x = [np.random.uniform(min_bounds[0], max_bounds[0]), np.random.uniform(min_bounds[1], max_bounds[1])]
        trials.append((unconstrained_obj(x), x))
      trials.sort()
      return TuneOutputs(*constrain(*trials[0][1]))
