# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# python3
"""NeuTra implementation."""
# pylint: disable=invalid-name,missing-docstring

import time
from typing import Any, Text, Tuple, NamedTuple

from absl import logging
import gin
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from hmc_swindles import targets
from hmc_swindles import utils
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

# pylint: disable=g-import-not-at-top
USE_LOCAL_FUN_MCMC = True

if USE_LOCAL_FUN_MCMC:
  from discussion import fun_mcmc  # pylint: disable=reimported

tfd = tfp.distributions
tfb = tfp.bijectors
tfkl = tf.keras.layers


@gin.configurable("head_tail_bijector")
def MakeHeadTailBijectorFn(num_dims,
                           head_layers=(),
                           activation=tf.nn.elu,
                           train=False,
                           head_dims=3):
  """A RealNVP for stochastic volatility model."""
  # pylint: disable=no-value-for-parameter
  del train
  tail_dims = num_dims - head_dims

  @utils.MakeTFTemplate
  def head_bijector_fn(x):
    x.set_shape(list(x.shape)[:-1] + [head_dims])
    input_shape = x.shape
    for i, units in enumerate(head_layers):
      x = utils.MaskedDense(
          inputs=x,
          units=units,
          num_blocks=head_dims,
          exclusive=True if i == 0 else False,
          kernel_initializer=utils.L2HMCInitializer(factor=0.01),
          activation=activation,
      )

    x = utils.MaskedDense(
        inputs=x,
        units=2 * head_dims,
        num_blocks=head_dims,
        activation=None,
        kernel_initializer=utils.L2HMCInitializer(factor=0.01),
    )
    x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
    shift, log_scale = tf.unstack(x, num=2, axis=-1)
    return tfb.AffineScalar(shift=shift, log_scale=log_scale)

  @utils.MakeTFTemplate
  def head_to_tail_bijector_fn(x, _):
    for units in head_layers:
      x = tfkl.Dense(
          units=units,
          activation=activation,
          kernel_initializer=utils.L2HMCInitializer(factor=0.01),
      )(x,)
    shift = tfkl.Dense(
        units=1,
        activation=None,
        kernel_initializer=utils.L2HMCInitializer(factor=0.01),
    )(x,)
    return tfb.AffineScalar(shift=shift, log_scale=tf.Variable(0.))

  @utils.MakeTFTemplate
  def tail_to_head_bijector_fn(x, _):
    x = tf.reduce_mean(x, axis=-1, keepdims=True)
    for units in head_layers:
      x = tfkl.Dense(units=units, activation=activation)(x)
    shift = tfkl.Dense(units=head_dims, activation=None)(x)
    return tfb.AffineScalar(
        shift=shift, log_scale=tf.Variable(tf.zeros(head_dims)))

  b = tfb.Identity()

  b = tfb.Blockwise(
      [
          tfb.Invert(
              tfb.MaskedAutoregressiveFlow(
                  bijector_fn=head_bijector_fn("head"))),
          tfb.Identity(),
      ],
      [head_dims, tail_dims],
  )(b,)
  b = tfb.RealNVP(
      num_masked=head_dims,
      bijector_fn=head_to_tail_bijector_fn("head_to_tail"))(
          b)
  b = tfb.Permute(list(reversed(range(num_dims))))(b)
  b = tfb.RealNVP(
      num_masked=tail_dims,
      bijector_fn=tail_to_head_bijector_fn("tail_to_head"))(
          b)
  b = tfb.Permute(list(reversed(range(num_dims))))(b)
  b = tfb.Blockwise(
      [
          tfb.Identity(),
          tfb.AffineScalar(shift=tf.Variable(tf.zeros([tail_dims])))
      ],
      [head_dims, tail_dims],
  )(b,)

  # Construct the variables
  _ = b.forward(tf.zeros([1, num_dims]))
  return b


@gin.configurable("affine_bijector")
def MakeAffineBijectorFn(num_dims, train=False, use_tril=False):
  mu = tf.Variable(tf.zeros([num_dims]), name="mean", trainable=train)
  if use_tril:
    tril_flat = tf.Variable(
        tf.zeros([num_dims * (num_dims + 1) // 2]),
        name="tril_flat",
        trainable=train)
    tril_raw = tfp.math.fill_triangular(tril_flat)
    sigma = tf.nn.softplus(tf.linalg.diag_part(tril_raw))
    tril = tf.linalg.set_diag(tril_raw, sigma)
    return tfb.Affine(shift=mu, scale_tril=tril)
  else:
    sigma = tf.nn.softplus(
        tf.Variable(tf.zeros([num_dims]), name="invpsigma", trainable=train))
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
      # # TODO(siege): I don't understand why the shape gets lost.
      # x.set_shape([None, num_dims - output_units])
      return t(x, output_units)

    bijectors.append(
        tfb.RealNVP(
            num_masked=num_dims // 2, shift_and_log_scale_fn=rnvp_template))
    bijectors.append(swap)
  # Drop the last swap.
  bijectors = bijectors[:-1]
  if learn_scale:
    scale = tf.Variable(tfp.math.softplus_inverse(scale),
                        name="isp_global_scale")
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
      # # TODO(siege): I don't understand why the shape gets lost.
      # x.set_shape([None, num_dims])
      return t(x)

    bijectors.append(
        tfb.Invert(
            tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=iaf_template)))
    bijectors.append(swap)
  # Drop the last swap.
  bijectors = bijectors[:-1]
  if learn_scale:
    scale = tf.nn.softplus(
        tf.Variable(tfp.math.softplus_inverse(scale), name="isp_global_scale"))
  bijectors.append(tfb.AffineScalar(scale=scale))

  bijector = tfb.Chain(bijectors)

  # Construct the variables
  _ = bijector.forward(tf.zeros([1, num_dims]))

  return bijector


@utils.register_namedtuple
class TargetSpec(NamedTuple):
  name: Any
  num_dims: Any
  x_min: Any
  x_max: Any
  y_min: Any
  y_max: Any
  stats: Any
  bijector: Any
  transforms: Any


@gin.configurable("target_spec")
def GetTargetSpec(
    name,
    **kwargs):

  target_density = utils.VectorTargetDensity(getattr(targets, name)())
  num_dims = target_density.event_shape.num_elements()
  target = utils.LogProbDist(num_dims=num_dims, log_prob_fn=target_density)
  spec = TargetSpec(
      name=name,
      num_dims=num_dims,
      x_min=0.10,
      x_max=0.15,
      y_min=0.10,
      y_max=0.15,
      stats={
          k + "_mean": v.ground_truth_mean.astype(np.float32)
          for k, v in target_density.expectations.items()
          if v.ground_truth_mean is not None
      },
      transforms=list(target_density.expectations.items()),
      bijector=target_density.constraining_bijectors,
  )

  return target, spec._replace(**kwargs)


@utils.register_namedtuple
class MCMCOutputs(NamedTuple):
  x_chain: Any = ()
  xcv_chain: Any = ()
  xa_chain: Any = ()
  xcva_chain: Any = ()
  p_accept: Any = ()
  p_accept_cv: Any = ()
  is_accepted: Any = ()
  is_accepted_cv: Any = ()
  is_accepted_a: Any = ()
  log_accept_ratio: Any = ()
  num_leapfrog_steps: Any = ()
  step_size: Any = ()
  extra: Any = ()


def GetIntegrator(integrator, step_size, num_steps, target_log_prob_fn):
  integrators = {
      "leapfrog": (fun_mcmc.leapfrog_step, 1),
      "ruth4": (fun_mcmc.ruth4_step, 3),
      "blanes_3_stage": (fun_mcmc.blanes_3_stage_step, 3),
      "blanes_4_stage": (fun_mcmc.blanes_4_stage_step, 5),
  }
  integrator_step_fn, leapfrog_multiplier = integrators[integrator]

  kinetic_energy_fn = fun_mcmc.make_gaussian_kinetic_energy_fn(1)

  integrator_fn = lambda state: fun_mcmc.hamiltonian_integrator(  # pylint: disable=g-long-lambda
      state,
      num_steps=num_steps,
      integrator_step_fn=lambda state: integrator_step_fn(  # pylint: disable=g-long-lambda
          state,
          step_size=step_size,
          target_log_prob_fn=target_log_prob_fn,
          kinetic_energy_fn=kinetic_energy_fn),
      kinetic_energy_fn=kinetic_energy_fn)

  return integrator_fn, leapfrog_multiplier


@gin.configurable("cva_neutra")
def MakeCVANeuTra(target,
                  q,
                  batch_size=32,
                  num_steps=100,
                  num_leapfrog_steps=2,
                  step_size=0.1,
                  integrator="leapfrog",
                  x_init=None):
  if x_init is None:
    x_init = q.sample(batch_size)

  (transformed_log_prob_fn,
   z_init) = fun_mcmc.transform_log_prob_fn(lambda x: (target.log_prob(x), ()),
                                            q.bijector, x_init)

  def joint_log_prob_fn(z_zcv):
    # N.B. z is concatenated real + antithetic chain.
    z, zcv = tf.split(z_zcv, [batch_size * 2, batch_size], axis=0)
    lpz, (x, _) = transformed_log_prob_fn(z)
    lpzcv = q.distribution.log_prob(zcv)
    xcv = q.bijector.forward(zcv)
    x_xcv = tf.concat([x, xcv], axis=0)
    return tf.concat([lpz, lpzcv], axis=0), x_xcv

  integrator, leapfrog_multiplier = GetIntegrator(integrator, step_size,
                                                  num_leapfrog_steps,
                                                  joint_log_prob_fn)

  def transition_operator(hmc_state):
    momentum = tf.random.normal(tf.shape(x_init))
    log_uniform = tf.math.log(tf.random.uniform(shape=tf.shape(x_init)[:-1]))
    # Share the momentum for everything, but negate it for antithetic chain.
    # Share the log_uniform between z and zcv as is.
    momentum = tf.concat([momentum, -momentum, momentum], axis=0)
    log_uniform = tf.concat([log_uniform] * 3, axis=0)

    return fun_mcmc.hamiltonian_monte_carlo(
        hmc_state,
        target_log_prob_fn=joint_log_prob_fn,
        momentum=momentum,
        log_uniform=log_uniform,
        integrator_fn=integrator)

  def trace_fn(state, extra):
    x_xcv = state.state_extra
    zcva = -state.state[-batch_size:]
    return (x_xcv, zcva, extra.log_accept_ratio, extra.is_accepted)

  (_, (x_xcv_chain, zcva_chain, log_accept_ratio,
       is_accepted)) = fun_mcmc.trace(
           state=fun_mcmc.hamiltonian_monte_carlo_init(
               tf.concat([z_init, -z_init, z_init], axis=0), joint_log_prob_fn),
           fn=transition_operator,
           num_steps=num_steps,
           trace_fn=trace_fn)

  p_accept = tf.reduce_mean(
      tf.cast(is_accepted[:, :2 * batch_size], tf.float32))
  p_accept_cv = tf.reduce_mean(
      tf.cast(is_accepted[:, -batch_size:], tf.float32))
  is_accepted_a = is_accepted[:, batch_size:2 * batch_size]
  is_accepted_cv = is_accepted[:, -batch_size:]
  is_accepted = is_accepted[:, :batch_size]

  x_chain, xa_chain, xcv_chain = tf.split(x_xcv_chain, 3, axis=1)

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))
  xa_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], xa_chain], 0))
  xcv_chain = tf.stop_gradient(
      tf.concat([x_init[tf.newaxis, Ellipsis], xcv_chain], 0))
  zcva_chain = tf.concat([-z_init[tf.newaxis, Ellipsis], zcva_chain], 0)
  xcva_chain = tf.stop_gradient(q.bijector.forward(zcva_chain))

  return MCMCOutputs(
      x_chain=x_chain,
      xcv_chain=xcv_chain,
      xa_chain=xa_chain,
      xcva_chain=xcva_chain,
      p_accept=p_accept,
      p_accept_cv=p_accept_cv,
      is_accepted=is_accepted,
      is_accepted_cv=is_accepted_cv,
      is_accepted_a=is_accepted_a,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=num_leapfrog_steps * leapfrog_multiplier)


@gin.configurable("a_neutra")
def MakeANeuTra(target,
                q,
                batch_size=32,
                num_steps=100,
                num_leapfrog_steps=2,
                step_size=0.1,
                integrator="leapfrog",
                x_init=None):
  if x_init is None:
    x_init = q.sample(batch_size)

  (transformed_log_prob_fn,
   z_init) = fun_mcmc.transform_log_prob_fn(lambda x: (target.log_prob(x), ()),
                                            q.bijector, x_init)

  integrator, leapfrog_multiplier = GetIntegrator(integrator, step_size,
                                                  num_leapfrog_steps,
                                                  transformed_log_prob_fn)

  def transition_operator(hmc_state):
    momentum = tf.random.normal(tf.shape(x_init))
    log_uniform = tf.math.log(tf.random.uniform(shape=tf.shape(x_init)[:-1]))
    # Share the momentum for everything, but negate it for antithetic chain.
    momentum = tf.concat([momentum, -momentum], axis=0)
    log_uniform = tf.concat([log_uniform] * 2, axis=0)

    return fun_mcmc.hamiltonian_monte_carlo(
        hmc_state,
        target_log_prob_fn=transformed_log_prob_fn,
        momentum=momentum,
        log_uniform=log_uniform,
        integrator_fn=integrator)

  def trace_fn(state, extra):
    x_xa = state.state_extra[0]
    return (x_xa, extra.log_accept_ratio, extra.is_accepted)

  (_, (x_xa_chain, log_accept_ratio, is_accepted)) = fun_mcmc.trace(
      state=fun_mcmc.hamiltonian_monte_carlo_init(
          tf.concat([z_init, -z_init], axis=0), transformed_log_prob_fn),
      fn=transition_operator,
      num_steps=num_steps,
      trace_fn=trace_fn)

  p_accept = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
  is_accepted_a = is_accepted[:, -batch_size:]
  is_accepted = is_accepted[:, :batch_size]

  x_chain, xa_chain = tf.split(x_xa_chain, 2, axis=1)

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))
  xa_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], xa_chain], 0))

  return MCMCOutputs(
      x_chain=x_chain,
      xa_chain=xa_chain,
      p_accept=p_accept,
      is_accepted=is_accepted,
      is_accepted_a=is_accepted_a,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=num_leapfrog_steps * leapfrog_multiplier)


@gin.configurable("cv_neutra")
def MakeCVNeuTra(target,
                 q,
                 batch_size=32,
                 num_steps=100,
                 num_leapfrog_steps=2,
                 step_size=0.1,
                 integrator="leapfrog",
                 x_init=None):
  if x_init is None:
    x_init = q.sample(batch_size)

  (transformed_log_prob_fn,
   z_init) = fun_mcmc.transform_log_prob_fn(lambda x: (target.log_prob(x), ()),
                                            q.bijector, x_init)

  def joint_log_prob_fn(z_zcv):
    z, zcv = tf.split(z_zcv, 2, axis=0)
    lpz, (x, _) = transformed_log_prob_fn(z)
    lpzcv = q.distribution.log_prob(zcv)
    xcv = q.bijector.forward(zcv)
    x_xcv = tf.concat([x, xcv], axis=0)
    return tf.concat([lpz, lpzcv], axis=0), x_xcv

  integrator, leapfrog_multiplier = GetIntegrator(integrator, step_size,
                                                  num_leapfrog_steps,
                                                  joint_log_prob_fn)

  def transition_operator(hmc_state):
    momentum = tf.random.normal(tf.shape(x_init))
    momentum_cv = momentum

    log_uniform = tf.math.log(tf.random.uniform(shape=tf.shape(x_init)[:-1]))
    # Share the momentum and log_uniform between z and zcv
    momentum = tf.concat([momentum, momentum_cv], axis=0)
    log_uniform = tf.concat([log_uniform] * 2, axis=0)

    return fun_mcmc.hamiltonian_monte_carlo(
        hmc_state,
        target_log_prob_fn=joint_log_prob_fn,
        momentum=momentum,
        log_uniform=log_uniform,
        integrator_fn=integrator)

  def trace_fn(state, extra):
    x, xcv = tf.split(state.state_extra, 2, axis=0)
    return (x, xcv, extra.log_accept_ratio, extra.is_accepted)

  (_, (x_chain, xcv_chain, log_accept_ratio, is_accepted)) = fun_mcmc.trace(
      state=fun_mcmc.hamiltonian_monte_carlo_init(
          tf.concat([z_init] * 2, axis=0), joint_log_prob_fn),
      fn=transition_operator,
      num_steps=num_steps,
      trace_fn=trace_fn)

  p_accept = tf.reduce_mean(tf.cast(is_accepted[:, :batch_size], tf.float32))
  p_accept_cv = tf.reduce_mean(
      tf.cast(is_accepted[:, -batch_size:], tf.float32))
  is_accepted_cv = is_accepted[:, -batch_size:]
  is_accepted = is_accepted[:, :batch_size]

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))
  xcv_chain = tf.stop_gradient(
      tf.concat([x_init[tf.newaxis, Ellipsis], xcv_chain], 0))

  return MCMCOutputs(
      x_chain=x_chain,
      xcv_chain=xcv_chain,
      p_accept=p_accept,
      p_accept_cv=p_accept_cv,
      is_accepted=is_accepted,
      is_accepted_cv=is_accepted_cv,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=num_leapfrog_steps * leapfrog_multiplier)


@gin.configurable("neutra")
def MakeNeuTra(target,
               q,
               batch_size=32,
               num_steps=100,
               num_leapfrog_steps=2,
               step_size=0.1,
               integrator="leapfrog",
               x_init=None):
  if x_init is None:
    x_init = q.sample(batch_size)

  (transformed_log_prob_fn,
   z_init) = fun_mcmc.transform_log_prob_fn(lambda x: (target.log_prob(x), ()),
                                            q.bijector, x_init)

  integrator, leapfrog_multiplier = GetIntegrator(integrator, step_size,
                                                  num_leapfrog_steps,
                                                  transformed_log_prob_fn)

  def transition_operator(hmc_state):
    return fun_mcmc.hamiltonian_monte_carlo(
        hmc_state,
        target_log_prob_fn=transformed_log_prob_fn,
        integrator_fn=integrator)

  def trace_fn(state, extra):
    return (state.state_extra[0], extra.log_accept_ratio, extra.is_accepted)

  (_, (x_chain, log_accept_ratio, is_accepted)) = fun_mcmc.trace(
      state=fun_mcmc.hamiltonian_monte_carlo_init(z_init,
                                                  transformed_log_prob_fn),
      fn=transition_operator,
      num_steps=num_steps,
      trace_fn=trace_fn)

  p_accept = tf.reduce_mean(tf.cast(is_accepted, tf.float32))

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))

  return MCMCOutputs(
      x_chain=x_chain,
      p_accept=p_accept,
      is_accepted=is_accepted,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=num_leapfrog_steps * leapfrog_multiplier)


@gin.configurable("neutra_rwm")
def MakeNeuTraRWM(target,
                  q,
                  batch_size=32,
                  num_steps=100,
                  step_size=0.1,
                  x_init=None,
                  **_):
  if x_init is None:
    x_init = q.sample(batch_size)

  (transformed_log_prob_fn,
   z_init) = fun_mcmc.transform_log_prob_fn(lambda x: (target.log_prob(x), ()),
                                            q.bijector, x_init)

  def proposal_fn(x, seed):
    return x + step_size * tf.random.normal(x.shape, seed=seed), ((), 0.)

  def transition_operator(rwm_state):
    return fun_mcmc.random_walk_metropolis(
        rwm_state,
        target_log_prob_fn=transformed_log_prob_fn,
        proposal_fn=proposal_fn)

  def trace_fn(state, extra):
    return (state.state_extra[0], extra.log_accept_ratio, extra.is_accepted)

  (_, (x_chain, log_accept_ratio, is_accepted)) = fun_mcmc.trace(
      state=fun_mcmc.random_walk_metropolis_init(z_init,
                                                 transformed_log_prob_fn),
      fn=transition_operator,
      num_steps=num_steps,
      trace_fn=trace_fn)

  p_accept = tf.reduce_mean(tf.cast(is_accepted, tf.float32))

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))

  return MCMCOutputs(
      x_chain=x_chain,
      p_accept=p_accept,
      is_accepted=is_accepted,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=1)


@gin.configurable("cv_neutra_rwm")
def MakeCVNeuTraRWM(target,
                    q,
                    batch_size=32,
                    num_steps=100,
                    step_size=0.1,
                    x_init=None,
                    **_):
  if x_init is None:
    x_init = q.sample(batch_size)

  (transformed_log_prob_fn,
   z_init) = fun_mcmc.transform_log_prob_fn(lambda x: (target.log_prob(x), ()),
                                            q.bijector, x_init)

  def joint_log_prob_fn(z_zcv):
    z, zcv = tf.split(z_zcv, 2, axis=0)
    lpz, (x, _) = transformed_log_prob_fn(z)
    lpzcv = q.distribution.log_prob(zcv)
    xcv = q.bijector.forward(zcv)
    x_xcv = tf.concat([x, xcv], axis=0)
    return tf.concat([lpz, lpzcv], axis=0), x_xcv

  def transition_operator(rwm_state):
    log_uniform = tf.math.log(tf.random.uniform(shape=tf.shape(x_init)[:-1]))
    # Share the log_uniform and proposal between z and zcv
    log_uniform = tf.concat([log_uniform] * 2, axis=0)

    def proposal_fn(x, seed):
      proposal = tf.random.normal(tf.shape(x_init), seed=seed)
      proposal = tf.concat([proposal] * 2, axis=0)
      return x + step_size * proposal, ((), 0.)

    return fun_mcmc.random_walk_metropolis(
        rwm_state,
        target_log_prob_fn=joint_log_prob_fn,
        proposal_fn=proposal_fn,
        log_uniform=log_uniform)

  def trace_fn(state, extra):
    x, xcv = tf.split(state.state_extra, 2, axis=0)
    return (x, xcv, extra.log_accept_ratio, extra.is_accepted)

  (_, (x_chain, xcv_chain, log_accept_ratio, is_accepted)) = fun_mcmc.trace(
      state=fun_mcmc.random_walk_metropolis_init(
          tf.concat([z_init] * 2, axis=0), joint_log_prob_fn),
      fn=transition_operator,
      num_steps=num_steps,
      trace_fn=trace_fn)

  p_accept = tf.reduce_mean(tf.cast(is_accepted[:, :batch_size], tf.float32))
  p_accept_cv = tf.reduce_mean(
      tf.cast(is_accepted[:, -batch_size:], tf.float32))
  is_accepted_cv = is_accepted[:, -batch_size:]
  is_accepted = is_accepted[:, :batch_size]

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))
  xcv_chain = tf.stop_gradient(
      tf.concat([x_init[tf.newaxis, Ellipsis], xcv_chain], 0))

  return MCMCOutputs(
      x_chain=x_chain,
      xcv_chain=xcv_chain,
      p_accept=p_accept,
      p_accept_cv=p_accept_cv,
      is_accepted=is_accepted,
      is_accepted_cv=is_accepted_cv,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=1)


@gin.configurable("a_neutra_rwm")
def MakeANeuTraRWM(target,
                   q,
                   batch_size=32,
                   num_steps=100,
                   step_size=0.1,
                   x_init=None,
                   **_):
  if x_init is None:
    x_init = q.sample(batch_size)

  (transformed_log_prob_fn,
   z_init) = fun_mcmc.transform_log_prob_fn(lambda x: (target.log_prob(x), ()),
                                            q.bijector, x_init)

  def transition_operator(rwm_state):
    log_uniform = tf.math.log(tf.random.uniform(shape=tf.shape(x_init)[:-1]))
    # Share the log_uniform and proposal between z and zcv
    log_uniform = tf.concat([log_uniform] * 2, axis=0)

    def proposal_fn(x, seed):
      proposal = tf.random.normal(tf.shape(x_init), seed=seed)
      proposal = tf.concat([proposal, -proposal], axis=0)
      return x + step_size * proposal, ((), 0.)

    return fun_mcmc.random_walk_metropolis(
        rwm_state,
        target_log_prob_fn=transformed_log_prob_fn,
        proposal_fn=proposal_fn,
        log_uniform=log_uniform,
    )

  def trace_fn(state, extra):
    x_xa = state.state_extra[0]
    return (x_xa, extra.log_accept_ratio, extra.is_accepted)

  (_, (x_xa_chain, log_accept_ratio, is_accepted)) = fun_mcmc.trace(
      state=fun_mcmc.random_walk_metropolis_init(
          tf.concat([z_init, -z_init], axis=0), transformed_log_prob_fn),
      fn=transition_operator,
      num_steps=num_steps,
      trace_fn=trace_fn)

  p_accept = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
  is_accepted_a = is_accepted[:, -batch_size:]
  is_accepted = is_accepted[:, :batch_size]

  x_chain, xa_chain = tf.split(x_xa_chain, 2, axis=1)

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))
  xa_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], xa_chain], 0))

  return MCMCOutputs(
      x_chain=x_chain,
      xa_chain=xa_chain,
      p_accept=p_accept,
      is_accepted=is_accepted,
      is_accepted_a=is_accepted_a,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=1)


@gin.configurable("cva_neutra_rwm")
def MakeCVANeuTraRWM(target,
                     q,
                     batch_size=32,
                     num_steps=100,
                     step_size=0.1,
                     x_init=None):
  if x_init is None:
    x_init = q.sample(batch_size)

  (transformed_log_prob_fn,
   z_init) = fun_mcmc.transform_log_prob_fn(lambda x: (target.log_prob(x), ()),
                                            q.bijector, x_init)

  def joint_log_prob_fn(z_zcv):
    # N.B. z is concatenated real + antithetic chain.
    z, zcv = tf.split(z_zcv, [batch_size * 2, batch_size], axis=0)
    lpz, (x, _) = transformed_log_prob_fn(z)
    lpzcv = q.distribution.log_prob(zcv)
    xcv = q.bijector.forward(zcv)
    x_xcv = tf.concat([x, xcv], axis=0)
    return tf.concat([lpz, lpzcv], axis=0), x_xcv

  def transition_operator(rwm_state):
    log_uniform = tf.math.log(tf.random.uniform(shape=tf.shape(x_init)[:-1]))
    # Share the log_uniform between z and zcv as is.
    log_uniform = tf.concat([log_uniform] * 3, axis=0)

    # Share the proposal for everything, but negate it for antithetic chain.
    def proposal_fn(x, seed):
      proposal = tf.random.normal(tf.shape(x_init), seed=seed)
      proposal = tf.concat([proposal, -proposal, proposal], axis=0)
      return x + step_size * proposal, ((), 0.)

    return fun_mcmc.random_walk_metropolis(
        rwm_state,
        target_log_prob_fn=joint_log_prob_fn,
        proposal_fn=proposal_fn,
        log_uniform=log_uniform)

  def trace_fn(state, extra):
    x_xcv = state.state_extra
    zcva = -state.state[-batch_size:]
    return (x_xcv, zcva, extra.log_accept_ratio, extra.is_accepted)

  (_, (x_xcv_chain, zcva_chain, log_accept_ratio,
       is_accepted)) = fun_mcmc.trace(
           state=fun_mcmc.random_walk_metropolis_init(
               tf.concat([z_init, -z_init, z_init], axis=0), joint_log_prob_fn),
           fn=transition_operator,
           num_steps=num_steps,
           trace_fn=trace_fn)

  p_accept = tf.reduce_mean(
      tf.cast(is_accepted[:, :2 * batch_size], tf.float32))
  p_accept_cv = tf.reduce_mean(
      tf.cast(is_accepted[:, -batch_size:], tf.float32))
  is_accepted_a = is_accepted[:, batch_size:2 * batch_size]
  is_accepted_cv = is_accepted[:, -batch_size:]
  is_accepted = is_accepted[:, :batch_size]

  x_chain, xa_chain, xcv_chain = tf.split(x_xcv_chain, 3, axis=1)

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))
  xa_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], xa_chain], 0))
  xcv_chain = tf.stop_gradient(
      tf.concat([x_init[tf.newaxis, Ellipsis], xcv_chain], 0))
  zcva_chain = tf.concat([-z_init[tf.newaxis, Ellipsis], zcva_chain], 0)
  xcva_chain = tf.stop_gradient(q.bijector.forward(zcva_chain))

  return MCMCOutputs(
      x_chain=x_chain,
      xcv_chain=xcv_chain,
      xa_chain=xa_chain,
      xcva_chain=xcva_chain,
      p_accept=p_accept,
      p_accept_cv=p_accept_cv,
      is_accepted=is_accepted,
      is_accepted_cv=is_accepted_cv,
      is_accepted_a=is_accepted_a,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=1)


@gin.configurable("cv_hmc")
def MakeCVHMC(target,
              q,
              batch_size=32,
              num_steps=100,
              num_leapfrog_steps=2,
              step_size=0.1,
              integrator="leapfrog",
              x_init=None):
  if x_init is None:
    x_init = q.sample(batch_size)

  def joint_log_prob_fn(x_xcv):
    x, xcv = tf.split(x_xcv, 2, axis=0)
    lpx = target.log_prob(x)
    lpxcv = q.log_prob(xcv)
    return tf.concat([lpx, lpxcv], axis=0), ()

  integrator, leapfrog_multiplier = GetIntegrator(integrator, step_size,
                                                  num_leapfrog_steps,
                                                  joint_log_prob_fn)

  def transition_operator(hmc_state):
    momentum = tf.random.normal(tf.shape(x_init))
    log_uniform = tf.math.log(tf.random.uniform(shape=tf.shape(x_init)[:-1]))
    # Share the momentum and log_uniform between z and zcv
    momentum = tf.concat([momentum] * 2, axis=0)
    log_uniform = tf.concat([log_uniform] * 2, axis=0)

    return fun_mcmc.hamiltonian_monte_carlo(
        hmc_state,
        target_log_prob_fn=joint_log_prob_fn,
        momentum=momentum,
        log_uniform=log_uniform,
        integrator_fn=integrator)

  def trace_fn(state, extra):
    x, xcv = tf.split(state.state, 2, axis=0)
    return (x, xcv, extra.log_accept_ratio, extra.is_accepted)

  (_, (x_chain, xcv_chain, log_accept_ratio, is_accepted)) = fun_mcmc.trace(
      state=fun_mcmc.hamiltonian_monte_carlo_init(
          tf.concat([x_init] * 2, axis=0), joint_log_prob_fn),
      fn=transition_operator,
      num_steps=num_steps,
      trace_fn=trace_fn)

  p_accept = tf.reduce_mean(tf.cast(is_accepted[:, :batch_size], tf.float32))
  p_accept_cv = tf.reduce_mean(
      tf.cast(is_accepted[:, -batch_size:], tf.float32))
  is_accepted_cv = is_accepted[:, -batch_size:]
  is_accepted = is_accepted[:, :batch_size]

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))
  xcv_chain = tf.stop_gradient(
      tf.concat([x_init[tf.newaxis, Ellipsis], xcv_chain], 0))

  return MCMCOutputs(
      x_chain=x_chain,
      xcv_chain=xcv_chain,
      p_accept=p_accept,
      p_accept_cv=p_accept_cv,
      is_accepted=is_accepted,
      is_accepted_cv=is_accepted_cv,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=num_leapfrog_steps * leapfrog_multiplier)


@gin.configurable("hmc")
def MakeHMC(target,
            q,
            batch_size=32,
            num_steps=100,
            num_leapfrog_steps=2,
            step_size=0.1,
            integrator="leapfrog",
            x_init=None):
  if x_init is None:
    x_init = q.sample(batch_size)

  def joint_log_prob_fn(x):
    return target.log_prob(x), ()

  integrator, leapfrog_multiplier = GetIntegrator(integrator, step_size,
                                                  num_leapfrog_steps,
                                                  joint_log_prob_fn)

  def transition_operator(hmc_state):
    return fun_mcmc.hamiltonian_monte_carlo(
        hmc_state,
        target_log_prob_fn=joint_log_prob_fn,
        integrator_fn=integrator)

  def trace_fn(state, extra):
    return (state.state, extra.log_accept_ratio, extra.is_accepted)

  (_, (x_chain, log_accept_ratio, is_accepted)) = fun_mcmc.trace(
      state=fun_mcmc.hamiltonian_monte_carlo_init(x_init, joint_log_prob_fn),
      fn=transition_operator,
      num_steps=num_steps,
      trace_fn=trace_fn)

  p_accept = tf.reduce_mean(tf.cast(is_accepted, tf.float32))

  x_chain = tf.stop_gradient(tf.concat([x_init[tf.newaxis, Ellipsis], x_chain], 0))

  return MCMCOutputs(
      x_chain=x_chain,
      p_accept=p_accept,
      is_accepted=is_accepted,
      log_accept_ratio=log_accept_ratio,
      num_leapfrog_steps=num_leapfrog_steps * leapfrog_multiplier)


@utils.register_namedtuple
class ChainLossState(NamedTuple):
  z_state: Any
  step_size: Any


@utils.register_namedtuple
class ChainLossOutputs(NamedTuple):
  x_fin: Any
  loss: Any
  p_accept: Any


@gin.configurable("chain_loss")
def ChainLoss(chain_loss_state,
              target,
              q,
              batch_size=32,
              step_size=0.1,
              trajectory_length=1.,
              num_steps=1,
              target_accept_prob=0.9):
  if chain_loss_state is None:
    x_init = q.sample(batch_size)
    z_init = q.bijector.inverse(x_init)
    chain_loss_state = ChainLossState(
        z_state=z_init, step_size=tf.convert_to_tensor(step_size, tf.float32))

  transformed_log_prob_fn = fun_mcmc.transform_log_prob_fn(
      lambda x: (target.log_prob(x), ()), q.bijector)

  def transition_operator(hmc_state):
    num_leapfrog_steps = tf.cast(
        tf.math.ceil(trajectory_length / chain_loss_state.step_size), tf.int32)
    return fun_mcmc.hamiltonian_monte_carlo(
        hmc_state,
        target_log_prob_fn=transformed_log_prob_fn,
        step_size=chain_loss_state.step_size,
        num_integrator_steps=num_leapfrog_steps)

  def trace_fn(_state, extra):
    return (extra.log_accept_ratio, extra.is_accepted)

  (final_state, (_, is_accepted)) = fun_mcmc.trace(
      state=fun_mcmc.HamiltonianMonteCarloState(
          state=chain_loss_state.z_state,
          state_grads=None,
          target_log_prob=None,
          state_extra=None,
      ),
      fn=transition_operator,
      num_steps=num_steps,
      trace_fn=trace_fn)

  p_accept = tf.reduce_mean(tf.cast(is_accepted, tf.float32))

  step_size = fun_mcmc.sign_adaptation(
      control=chain_loss_state.step_size,
      output=p_accept,
      set_point=target_accept_prob,
      adaptation_rate=0.01)
  # step_size = chain_loss_state.step_size
  chain_loss_state = chain_loss_state._replace(
      z_state=final_state.state, step_size=step_size)
  x_fin = q.bijector.forward(final_state.state)
  x_fin = tf.stop_gradient(x_fin)
  loss = -tf.reduce_mean(q.log_prob(x_fin))

  return chain_loss_state, ChainLossOutputs(
      x_fin=x_fin, loss=loss, p_accept=p_accept)


@utils.register_namedtuple
class QStats(NamedTuple):
  bias: Any


def ComputeQStats(q_samples, target_mean):
  return QStats(bias=target_mean - tf.reduce_mean(q_samples, 0))


@utils.register_namedtuple
class ChainStats(NamedTuple):
  bias: Any
  inst_bias: Any
  variance: Any
  inst_variance: Any
  error_sq: Any
  ess: Any
  ess_per_grad: Any
  rhat: Any
  autocorr: Any
  warmupped_bias: Any
  warmupped_variance: Any
  overall_variance: Any
  per_chain_variance: Any


@gin.configurable("chain_stats")
@utils.compile
def ComputeChainStats(chain,
                      target_mean,
                      num_leapfrog_steps,
                      compute_stats_over_time=False,
                      target_variance=None):
  # Chain is [num_steps, batch, num_dims]
  num_steps = tf.shape(chain)[0]
  batch = tf.shape(chain)[1]

  if compute_stats_over_time:
    counts = tf.cast(tf.range(1, num_steps + 1), tf.float32)
    chain_mean = tf.cumsum(chain, 0) / counts[:, tf.newaxis, tf.newaxis]
    bias = target_mean - tf.reduce_mean(chain_mean, 1)
    variance = tf.math.reduce_variance(chain_mean, 1)
    inst_bias = target_mean - tf.reduce_mean(chain, 1)
    # XXX: This looks wrong, why are we using target_mean here?
    inst_variance = tf.reduce_sum(tf.square(target_mean - chain), 1) / tf.cast(
        batch - 1, tf.float32)

    def reducer(_, idx):
      chain_mean = tf.reduce_mean(chain[idx // 2:idx], 0)
      bias = tf.reduce_mean(target_mean - chain_mean, 0)
      variance = tf.math.reduce_variance(chain_mean, 0)
      return bias, variance

    indices = 1 + tf.range(num_steps)
    warmupped_bias, warmupped_variance = tf.scan(
        reducer, indices, initializer=(chain[0, 0], chain[0, 0]))

  half_steps = num_steps // 2
  half_chain = chain[half_steps:]

  error_sq = tf.reduce_mean(
      tf.square(tf.reduce_mean(half_chain, 0) - target_mean), 0)

  if target_variance is None:
    target_variance = tf.math.reduce_variance(half_chain, [0, 1])

  ess = utils.EffectiveSampleSize(
      half_chain / tf.sqrt(target_variance), use_geyer=True,
      normalize=False) / tf.cast(half_steps, tf.float32)
  ess_per_grad = ess / tf.cast(num_leapfrog_steps, tf.float32)
  rhat = tfp.mcmc.potential_scale_reduction(half_chain)
  autocorr = utils.SanitizedAutoCorrelationMean(
      half_chain, axis=0, reduce_axis=1, max_lags=300)

  # Brute ESS is computed as the ratio of these two, NB these are not normalized
  # by chain length.
  overall_variance = tf.math.reduce_variance(half_chain, [0, 1])
  per_chain_variance = tf.math.reduce_variance(tf.reduce_mean(half_chain, 0), 0)

  return ChainStats(
      bias=bias if compute_stats_over_time else (),
      variance=variance if compute_stats_over_time else (),
      error_sq=error_sq,
      inst_bias=inst_bias if compute_stats_over_time else (),
      inst_variance=inst_variance if compute_stats_over_time else (),
      ess=ess,
      ess_per_grad=ess_per_grad,
      rhat=rhat,
      warmupped_bias=warmupped_bias if compute_stats_over_time else (),
      warmupped_variance=warmupped_variance if compute_stats_over_time else (),
      autocorr=autocorr,
      overall_variance=overall_variance,
      per_chain_variance=per_chain_variance,
  )


@utils.register_namedtuple
class VRChainOutputs(NamedTuple):
  vr_chain: Any
  cv_beta: Any
  cv_rho: Any


def ChainCov(xs, ys):
  n = tf.shape(xs)[0]
  mx = tf.reduce_mean(xs, 0)
  my = tf.reduce_mean(ys, 0)
  return tf.einsum("abi,abj->ij", xs - mx, ys - my) / tf.cast(n - 1, tf.float32)


def ChainCovDiag(xs, ys):
  n = tf.shape(xs)[0]
  mx = tf.reduce_mean(xs, 0)
  my = tf.reduce_mean(ys, 0)
  return tf.einsum("abi,abi->i", xs - mx, ys - my) / tf.cast(n - 1, tf.float32)


def ChainCorr(xs, ys):
  cov_ys_ys = ChainCovDiag(ys, ys)
  cov_xs_ys = ChainCovDiag(xs, ys)
  cov_xs_xs = ChainCovDiag(xs, xs)
  return cov_xs_ys / tf.sqrt(cov_ys_ys * cov_xs_xs)


@utils.compile
def GetCVBeta(chain, cv_chain):
  num_steps = tf.shape(chain)[0]
  half_steps = num_steps // 2

  half_chain = chain[half_steps:]
  half_cv_chain = cv_chain[half_steps:]

  cov_cv_chain = ChainCov(half_cv_chain, half_chain)
  cov_cv_cv = ChainCov(half_cv_chain, half_cv_chain)
  cov_cv_cv += tf.eye(cov_cv_cv.shape[-1]) * 1e-6

  return tf.linalg.solve(cov_cv_cv, cov_cv_chain)


@utils.compile
def GetCVBetaVec(chain, cv_chain):
  num_steps = tf.shape(chain)[0]
  half_steps = num_steps // 2

  half_chain = chain[half_steps:]
  half_cv_chain = cv_chain[half_steps:]

  cov_cv_chain = ChainCovDiag(half_cv_chain, half_chain)
  cov_cv_cv = ChainCovDiag(half_cv_chain, half_cv_chain)

  beta_vec = cov_cv_chain / cov_cv_cv
  return beta_vec


@utils.compile
def GetVarianceReducedChain(chain, cv_chain, cv_mean, cv_beta):
  num_steps = tf.shape(chain)[0]
  half_steps = num_steps // 2

  half_chain = chain[half_steps:]
  half_cv_chain = cv_chain[half_steps:]

  if cv_beta.shape.rank == 2:
    vr_chain = chain - tf.einsum("abi,ij->abj", cv_chain - cv_mean, cv_beta)
  else:
    vr_chain = chain - (cv_chain - cv_mean) * cv_beta

  cv_rho = ChainCorr(half_chain, half_cv_chain)

  return VRChainOutputs(vr_chain=vr_chain, cv_beta=cv_beta, cv_rho=cv_rho)


@utils.compile
def GetMHDecoupleRate(is_accepted_1, is_accepted_2):
  return tf.reduce_mean(tf.cast(is_accepted_1 != is_accepted_2, tf.float32))


@utils.compile
def GetMHAgreeAcceptRate(is_accepted_1, is_accepted_2):
  return tf.reduce_mean(tf.cast(is_accepted_1 & is_accepted_2, tf.float32))


@utils.register_namedtuple
class MCMCStats(NamedTuple):
  chain_stats: Any = ()
  chain_stats_tune: Any = ()
  chain_stats_cv: Any = ()
  chain_stats_a: Any = ()
  chain_stats_vr_cv: Any = ()
  chain_stats_vr_a: Any = ()
  chain_stats_vr_cva: Any = ()
  chain_stats_vr_cv_one: Any = ()
  chain_stats_vr_cva_one: Any = ()
  chain_stats_vr_cv_vec: Any = ()
  chain_stats_vr_cva_vec: Any = ()
  vr_outputs_cva: Any = ()
  vr_outputs_cva_one: Any = ()
  vr_outputs_cva_vec: Any = ()
  vr_outputs_cv: Any = ()
  vr_outputs_cv_one: Any = ()
  vr_outputs_cv_vec: Any = ()
  p_accept: Any = ()
  p_accept_cv: Any = ()
  p_mh_agree_accept_cv: Any = ()
  p_mh_agree_accept_a: Any = ()
  p_mh_decouple_cv: Any = ()
  p_mh_decouple_a: Any = ()
  a_corr: Any = ()


def AverageStats(stats):

  def classify(path):
    if "ess" in path:
      return lambda x: 1. / np.mean(1. / np.array(x), 0)
    else:
      return lambda x: np.mean(x, 0)

  def to_numpy(t):
    if isinstance(t, tf.Tensor):
      return t.numpy()
    else:
      return t

  def is_sentinel(x):
    return isinstance(x, tuple) and not x

  stats = tf.nest.map_structure(to_numpy, stats)

  avg_type = [
      classify("".join(map(str, path)))  # pylint: disable=g-complex-comprehension
      for path, _ in nest.flatten_with_tuple_paths(stats[0])
  ]
  flat_stats = [tf.nest.flatten(r) for r in stats]
  trans_stats = zip(*flat_stats)

  trans_mean_stats = [
      r if is_sentinel(r[0]) else avg(r)
      for avg, r in zip(avg_type, trans_stats)
  ]
  mean_stats = tf.nest.pack_sequence_as(stats[0], trans_mean_stats)
  return mean_stats


@utils.register_namedtuple
class TuneOutputs(NamedTuple):
  num_leapfrog_steps: Any
  step_size: Any


@utils.register_namedtuple
class TuneObjective(NamedTuple):
  objective: Any
  step_size: Any
  num_leapfrog_steps: Any


@utils.register_namedtuple
class BenchmarkOutputs(NamedTuple):
  mcmc_secs_per_step: Any
  q_secs_per_sample: Any


@gin.configurable("neutra_experiment")
class NeuTraExperiment(tf.Module):

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      mcmc_type = "neutra",
      bijector = "iaf",
      log_dir="/tmp/neutra",
      base_learning_rate=1e-3,
      q_base_scale=1.,
      loss="kl_qp",
      learning_rate_schedule=[[6000, 1e-1]],
      do_polyak=False,
      polyak_start=0,
      polyak_rate=0.999,
  ):
    target, target_spec = GetTargetSpec()  # pylint: disable=no-value-for-parameter
    self._target = target
    self.target_spec = target_spec
    with gin.config_scope("train"):
      train_target, train_target_spec = GetTargetSpec()  # pylint: disable=no-value-for-parameter
      self.train_target = train_target
      self.train_target_spec = train_target_spec

    if bijector == "rnvp":
      bijector_fn = utils.Template(
          "bijector", MakeRNVPBijectorFn, num_dims=self.target_spec.num_dims)
    elif bijector == "iaf":
      bijector_fn = utils.Template(
          "bijector", MakeIAFBijectorFn, num_dims=self.target_spec.num_dims)
    elif bijector == "affine":
      bijector_fn = utils.Template(
          "bijector", MakeAffineBijectorFn, num_dims=self.target_spec.num_dims)
    elif bijector == "head_tail":
      bijector_fn = utils.Template(
          "bijector",
          MakeHeadTailBijectorFn,
          num_dims=self.target_spec.num_dims)
    else:
      bijector_fn = utils.Template("bijector",
                                   lambda *args, **kwargs: tfb.Identity())

    if self.train_target_spec.bijector is not None:
      print("Using train target bijector")
      # For var tracking.
      self.base_bijector_fn = bijector_fn
      bijector_fn = lambda train: tfb.Chain(  # pylint: disable=g-long-lambda
          [train_target_spec.bijector,
           self.base_bijector_fn(train=train)])

    self.bijector_fn = bijector_fn

    self.q_base_scale = q_base_scale

    # Training
    self.base_learning_rate = base_learning_rate
    self.learning_rate_schedule = learning_rate_schedule
    self.loss = loss
    self.mcmc_type = mcmc_type

    # Construct the variables
    self.bijector_fn(train=True)
    self.InitTargetStats()

    self.do_polyak = do_polyak
    self.polyak_rate = polyak_rate
    self.polyak_start = polyak_start
    if self.do_polyak:
      self.polyak_variables = []
      for v in self.trainable_variables:
        self.polyak_variables.append(
            tf.Variable(v, name=v.name[:-2] + "_poly", trainable=False))

    self.checkpoint = tf.train.Checkpoint(experiment=self)
    self.log_dir = log_dir
    self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

  @property
  def target(self):
    # Some bug with nested TF functions, need to re-construct the target
    # distribution to break the cache.
    return self._target.copy()

  def UpdatePolyVars(self, rate=None):
    if rate is None:
      rate = self.polyak_rate
    for pv, v in zip(self.polyak_variables, self.trainable_variables):
      pv.assign(rate * pv + (1. - rate) * v)
    return rate

  def UpdateFromPolyVars(self):
    for pv, v in zip(self.polyak_variables, self.trainable_variables):
      v.assign(pv)

  @utils.compile
  def QStats(self, num_samples=16384 * 8):
    q_stats = {}
    q_samples = self.Q().sample(num_samples)
    for name, f in self.functions:
      q_stats[name] = ComputeQStats(f(q_samples), self.target_mean[name])
    return q_stats

  def Q(self, bijector=None):
    if bijector is None:
      bijector = self.bijector_fn(train=False)
    q_base = tfd.Independent(
        tfd.Normal(
            loc=tf.zeros(self.target_spec.num_dims),
            scale=self.q_base_scale * tf.ones(self.target_spec.num_dims)), 1)
    return tfd.TransformedDistribution(q_base, bijector)

  @utils.compile
  def MCMC(self,
           batch_size=4096,
           test_num_steps=1000,
           test_num_leapfrog_steps=2,
           test_step_size=0.1,
           **kwargs):
    if self.mcmc_type == "hmc":
      mcmc_fn = MakeHMC
    elif self.mcmc_type == "cv_hmc":
      mcmc_fn = MakeCVHMC
    elif self.mcmc_type == "neutra":
      mcmc_fn = MakeNeuTra
    elif self.mcmc_type == "a_neutra":
      mcmc_fn = MakeANeuTra
    elif self.mcmc_type == "cv_neutra":
      mcmc_fn = MakeCVNeuTra
    elif self.mcmc_type == "cva_neutra":
      mcmc_fn = MakeCVANeuTra
    elif self.mcmc_type == "neutra_rwm":
      mcmc_fn = MakeNeuTraRWM
    elif self.mcmc_type == "a_neutra_rwm":
      mcmc_fn = MakeANeuTraRWM
    elif self.mcmc_type == "cv_neutra_rwm":
      mcmc_fn = MakeCVNeuTraRWM
    elif self.mcmc_type == "cva_neutra_rwm":
      mcmc_fn = MakeCVANeuTraRWM
    return mcmc_fn(
        target=self.target,
        q=self.Q(),
        batch_size=batch_size,
        num_steps=test_num_steps,
        num_leapfrog_steps=test_num_leapfrog_steps,
        step_size=test_step_size,
        **kwargs,
    )

  def InitTargetStats(self, batch_size=16384 * 8):
    target_samples = utils.compile(
        lambda: self.target.sample(batch_size))()

    def variance(x, mean_stat):
      x -= self.target_mean[mean_stat]
      return tf.square(x)

    transforms = []
    transforms.extend(self.target_spec.transforms)

    self.functions = []

    # This is because there is some horrid nonsense with lambdas and loops that
    # I couldn't figure out... I hate Python.
    def body(tname, transform):

      def get_name(fname):
        if tname is not None:
          return "_".join([tname, fname])
        return fname

      def make_fn(f):
        return lambda x: f(transform(x))

      self.functions.append((get_name("mean"), make_fn(tf.identity)))
      self.functions.append(
          (get_name("var"), make_fn(lambda x: variance(x, get_name("mean")))))

    for tname, transform in transforms:
      body(tname, transform)

    self.target_mean = {}
    for name, f in self.functions:
      if self.target_spec.stats is not None and name in self.target_spec.stats:
        target_mean = tf.convert_to_tensor(self.target_spec.stats[name])
      else:
        target_mean = tf.reduce_mean(f(target_samples), 0)
      self.target_mean[name] = target_mean

  @gin.configurable("mcmc_stats")
  def MCMCStats(self,
                neutra_outputs,
                return_vr_chains=False,
                num_q_samples=8192,
                compute_mat_beta=False,
                num_beta_chains=None):
    ret = MCMCStats(
        chain_stats={},
        chain_stats_cv={},
        chain_stats_a={},
        chain_stats_vr_cv={},
        chain_stats_vr_a={},
        chain_stats_vr_cva={},
        chain_stats_vr_cv_one={},
        chain_stats_vr_cva_one={},
        chain_stats_vr_cv_vec={},
        chain_stats_vr_cva_vec={},
        vr_outputs_cva={},
        vr_outputs_cva_vec={},
        vr_outputs_cva_one={},
        vr_outputs_cv={},
        vr_outputs_cv_vec={},
        vr_outputs_cv_one={},
        p_accept=neutra_outputs.p_accept,
        p_accept_cv=neutra_outputs.p_accept_cv,
        a_corr={},
    )
    ret = ret._replace(chain_stats_tune=ret.chain_stats)

    # TODO(siege): We should compute these only once...
    q_means = {}
    for name, f in self.functions:
      q_means[name] = tf.reduce_mean(f(self.Q().sample(num_q_samples)), 0)

    for name, f in self.functions:
      fx_chain = f(neutra_outputs.x_chain)
      half_steps = fx_chain.shape[0] // 2
      target_variance = tf.math.reduce_variance(fx_chain[half_steps:], [0, 1])
      ret.chain_stats[name] = ComputeChainStats(
          fx_chain, self.target_mean[name], neutra_outputs.num_leapfrog_steps)

      if self.mcmc_type in [
          "cv_neutra", "cv_hmc", "cv_neutra_rwm"
      ]:
        ret = ret._replace(
            p_mh_decouple_cv=GetMHDecoupleRate(
                neutra_outputs.is_accepted,
                neutra_outputs.is_accepted_cv,
            ),
            p_mh_agree_accept_cv=GetMHAgreeAcceptRate(
                neutra_outputs.is_accepted,
                neutra_outputs.is_accepted_cv,
            ),
            chain_stats_tune=ret.chain_stats_vr_cv_vec,
        )

        fxcv_chain = f(neutra_outputs.xcv_chain)
        ret.chain_stats_cv[name] = ComputeChainStats(
            fxcv_chain,
            self.target_mean[name],
            neutra_outputs.num_leapfrog_steps,
            target_variance=target_variance)

        if num_beta_chains is None:
          fx_chain_beta = fx_chain
          fxcv_chain_beta = fxcv_chain
        else:
          fx_chain_beta = fx_chain[:, :num_beta_chains]
          fxcv_chain_beta = fxcv_chain[:, :num_beta_chains]
          # TODO(siege): This introduces an annoying off-by-num_beta_chains
          # shift. We should only do this when computing the stats...
          fx_chain = fx_chain[:, num_beta_chains:]
          fxcv_chain = fxcv_chain[:, num_beta_chains:]

        cv_beta_vec = GetCVBetaVec(fx_chain_beta, fxcv_chain_beta)
        cv_beta_one = tf.ones(cv_beta_vec.shape[-1])
        vr_inputs = [
            (ret.chain_stats_vr_cv_vec, ret.vr_outputs_cv_vec, cv_beta_vec),
            (ret.chain_stats_vr_cv_one, ret.vr_outputs_cv_one, cv_beta_one),
        ]
        fxcv_mean = q_means[name]
        if compute_mat_beta:
          cv_beta = GetCVBeta(fx_chain_beta, fxcv_chain_beta)
          vr_inputs.append((ret.chain_stats_vr_cv, ret.vr_outputs_cv, cv_beta))

        for chain_stats_vr_cv, vr_outputs_cv, cv_beta_val in vr_inputs:
          vr_outputs_cv1 = GetVarianceReducedChain(fx_chain, fxcv_chain,
                                                   fxcv_mean, cv_beta_val)
          if return_vr_chains:
            vr_outputs_cv[name] = vr_outputs_cv1
          else:
            vr_outputs_cv[name] = vr_outputs_cv1._replace(vr_chain=())

          chain_stats_vr_cv[name] = ComputeChainStats(
              vr_outputs_cv1.vr_chain,
              self.target_mean[name],
              neutra_outputs.num_leapfrog_steps,
              target_variance=target_variance)
      elif self.mcmc_type in ["cva_neutra", "cva_neutra_rwm"]:
        ret = ret._replace(
            p_mh_decouple_cv=GetMHDecoupleRate(neutra_outputs.is_accepted,
                                               neutra_outputs.is_accepted_cv),
            p_mh_agree_accept_cv=GetMHAgreeAcceptRate(
                neutra_outputs.is_accepted,
                neutra_outputs.is_accepted_cv,
            ),
            p_mh_decouple_a=GetMHDecoupleRate(
                neutra_outputs.is_accepted,
                neutra_outputs.is_accepted_a,
            ),
            p_mh_agree_accept_a=GetMHAgreeAcceptRate(
                neutra_outputs.is_accepted,
                neutra_outputs.is_accepted_a,
            ),
            chain_stats_tune=ret.chain_stats_vr_cva_vec,
        )
        fxa_chain = f(neutra_outputs.xa_chain)
        fxcv_chain = f(neutra_outputs.xcv_chain)
        fxcva_chain = f(neutra_outputs.xcva_chain)
        ret.a_corr[name] = ChainCorr(fx_chain, fxa_chain)
        ret.chain_stats_a[name] = ComputeChainStats(
            fxa_chain,
            self.target_mean[name],
            neutra_outputs.num_leapfrog_steps,
            target_variance=target_variance)
        ret.chain_stats_cv[name] = ComputeChainStats(
            fxcv_chain,
            self.target_mean[name],
            neutra_outputs.num_leapfrog_steps,
            target_variance=target_variance)
        ret.chain_stats_vr_a[name] = ComputeChainStats(
            0.5 * (fx_chain + fxa_chain),
            self.target_mean[name],
            neutra_outputs.num_leapfrog_steps,
            target_variance=target_variance)

        if num_beta_chains is None:
          fx_chain_beta = fx_chain
          fxcv_chain_beta = fxcv_chain
          fxa_chain_beta = fxa_chain
          fxcva_chain_beta = fxcva_chain
        else:
          fx_chain_beta = fx_chain[:, :num_beta_chains]
          fxcv_chain_beta = fxcv_chain[:, :num_beta_chains]
          fxa_chain_beta = fxa_chain[:, :num_beta_chains]
          fxcva_chain_beta = fxcva_chain[:, :num_beta_chains]
          # TODO(siege): This introduces an annoying off-by-num_beta_chains
          # shift. We should only do this when computing the stats...
          fx_chain = fx_chain[:, num_beta_chains:]
          fxcv_chain = fxcv_chain[:, num_beta_chains:]
          fxa_chain = fxa_chain[:, num_beta_chains:]
          fxcva_chain = fxcva_chain[:, num_beta_chains:]

        fxcv_mean = q_means[name]
        cv_beta_vec = GetCVBetaVec(fx_chain_beta, fxcv_chain_beta)
        cv_beta_vec_a = GetCVBetaVec(fxa_chain_beta, fxcva_chain_beta)
        cv_beta_one = tf.ones(cv_beta_vec.shape[-1])
        vr_inputs = [
            (ret.chain_stats_vr_cv_vec, ret.chain_stats_vr_cva_vec,
             ret.vr_outputs_cv_vec, ret.vr_outputs_cva_vec, cv_beta_vec,
             cv_beta_vec_a),
            (ret.chain_stats_vr_cv_one, ret.chain_stats_vr_cva_one,
             ret.vr_outputs_cv_one, ret.vr_outputs_cva_one, cv_beta_one,
             cv_beta_one),
        ]
        if compute_mat_beta:
          cv_beta = GetCVBeta(fx_chain_beta, fxcv_chain_beta)
          cv_beta_a = GetCVBeta(fxa_chain_beta, fxcva_chain_beta)
          vr_inputs.append(
              (ret.chain_stats_vr_cv, ret.chain_stats_vr_cva, ret.vr_outputs_cv,
               ret.vr_outputs_cva, cv_beta, cv_beta_a))

        for (chain_stats_vr_cv, chain_stats_vr_cva, vr_outputs_cv,
             vr_outputs_cva, cv_beta_val, cv_beta_a_val) in vr_inputs:
          vr_outputs_cv1 = GetVarianceReducedChain(fx_chain, fxcv_chain,
                                                   fxcv_mean, cv_beta_val)
          vr_outputs_cv2 = GetVarianceReducedChain(fxa_chain, fxcva_chain,
                                                   fxcv_mean, cv_beta_a_val)

          if return_vr_chains:
            vr_outputs_cv[name] = vr_outputs_cv1
            vr_outputs_cva[name] = vr_outputs_cv2
          else:
            vr_outputs_cv[name] = vr_outputs_cv1._replace(vr_chain=())
            vr_outputs_cva[name] = vr_outputs_cv2._replace(vr_chain=())

          chain_stats_vr_cv[name] = ComputeChainStats(
              vr_outputs_cv1.vr_chain,
              self.target_mean[name],
              neutra_outputs.num_leapfrog_steps,
              target_variance=target_variance)
          chain_stats_vr_cva[name] = ComputeChainStats(
              0.5 * (vr_outputs_cv1.vr_chain + vr_outputs_cv2.vr_chain),
              self.target_mean[name],
              neutra_outputs.num_leapfrog_steps,
              target_variance=target_variance)
      elif self.mcmc_type in ["a_neutra", "a_neutra_rwm"]:
        ret = ret._replace(
            p_mh_decouple_a=GetMHDecoupleRate(
                neutra_outputs.is_accepted,
                neutra_outputs.is_accepted_a,
            ),
            p_mh_agree_accept_a=GetMHAgreeAcceptRate(
                neutra_outputs.is_accepted,
                neutra_outputs.is_accepted_a,
            ),
            chain_stats_tune=ret.chain_stats_vr_a,
        )
        fxa_chain = f(neutra_outputs.xa_chain)
        ret.a_corr[name] = ChainCorr(fx_chain, fxa_chain)
        ret.chain_stats_a[name] = ComputeChainStats(
            fxa_chain,
            self.target_mean[name],
            neutra_outputs.num_leapfrog_steps,
            target_variance=target_variance)

        ret.chain_stats_vr_a[name] = ComputeChainStats(
            0.5 * (fx_chain + fxa_chain),
            self.target_mean[name],
            neutra_outputs.num_leapfrog_steps,
            target_variance=target_variance)
    return ret

  @utils.compile
  def TrainLoss(self, batch_size=4096, step=None, state=()):
    bijector = self.bijector_fn(train=True)
    q_x_train = self.Q(bijector)

    if self.loss == "kl_qp":
      train_q_samples = q_x_train.sample(batch_size)
      train_log_q_x = q_x_train.log_prob(train_q_samples)
      kl_q_p = tf.reduce_mean(train_log_q_x -
                              self.target.log_prob(train_q_samples))

      loss = kl_q_p
      tf.summary.scalar("kl_qp", kl_q_p, step=step)
    elif self.loss == "kl_pq":
      state, out = ChainLoss(state, self.target, q_x_train, batch_size)
      loss = out.loss
      tf.summary.scalar("xent", loss, step=step)
      tf.summary.scalar("step_size", state.step_size, step=step)
      tf.summary.scalar("p_accept", out.p_accept, step=step)
    return loss, state

  @gin.configurable("train")
  def Train(self,
            num_steps,
            summary_every=500,
            batch_size=4096,
            plot_callback=None):
    times = np.zeros(num_steps)
    q_errs = []

    lr = tf.convert_to_tensor(self.base_learning_rate)
    steps, factors = zip(*self.learning_rate_schedule)
    learning_rate = utils.PiecewiseConstantDecay(
        [tf.cast(s, tf.float32) for s in steps],
        [lr * x for x in [1.0] + list(factors)])
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    global_step = self.global_step
    tf.summary.experimental.set_step(global_step)
    state = ()

    @utils.compile
    def summary_fn():
      with tf.summary.record_if(True):
        self.TrainLoss()
        return self.QStats()

    def summarize(t):
      # Grab the summaries from the loss computation.

      q_errs.append(summary_fn())
      if plot_callback:
        plot_callback(t)

    @utils.compile
    def minimize(state):
      with tf.summary.record_if(False):
        with tf.GradientTape() as tape:
          with utils.use_xla(False):
            loss, state = self.TrainLoss(batch_size, global_step, state)
        vals = tape.watched_variables()
        grads = tape.gradient(loss, vals)
        opt.apply_gradients(list(zip(grads, vals)))
        global_step.assign_add(1)
        return loss, state

    for t in range(num_steps):
      with utils.use_xla(False):
        if t % summary_every == 0:
          summarize(t)

      start_time = time.time()
      loss, state = minimize(state)
      if self.do_polyak:
        if t == self.polyak_start:
          self.UpdatePolyVars(rate=0.)
        elif t > self.polyak_start:
          self.UpdatePolyVars()
      times[t] = time.time() - start_time
      tf.debugging.assert_all_finite(loss, "Loss has NaNs at step %d" % t)

    if self.do_polyak:
      self.UpdateFromPolyVars()
    with utils.use_xla(False):
      summarize(num_steps)

    flat_q_errs = [tf.nest.flatten(s) for s in q_errs]
    trans_q_errs = zip(*flat_q_errs)
    concat_q_errs = [np.stack(q, 0) for q in trans_q_errs]
    q_stats = tf.nest.pack_sequence_as(q_errs[0], concat_q_errs)

    self.checkpoint.write(self.log_dir + "/model.ckpt")
    return q_stats, times[2:].mean()

  @gin.configurable("eval")
  def Eval(self, **neutra_args):
    neutra_outputs = self.MCMC(**neutra_args)
    return self.MCMCStats(neutra_outputs)

  @gin.configurable("benchmark")
  def Benchmark(self, test_num_steps, test_batch_size, **neutra_args):

    def bench_mcmc():
      res = self.MCMC(test_num_steps=test_num_steps, **neutra_args)
      ret = [res.x_chain]
      if res.xcv_chain is not None:
        ret.append(res.xcv_chain)
      elif res.xcva_chain is not None:
        ret.append(res.xcva_chain)
      return ret

    @utils.compile
    def bench_q():
      return self.Q().sample(test_batch_size)

    bench_mcmc()
    bench_q()
    start_time = time.time()
    bench_mcmc()
    mcmc_secs_per_step = (time.time() - start_time) / test_num_steps
    q_times = []
    for _ in range(10):
      start_time = time.time()
      bench_q()
      q_secs_per_sample = (time.time() - start_time) / test_batch_size
      q_times.append(q_secs_per_sample)
    q_secs_per_sample = np.mean(q_times)

    return BenchmarkOutputs(
        mcmc_secs_per_step=mcmc_secs_per_step,
        q_secs_per_sample=q_secs_per_sample)

  @gin.configurable("tune_objective")
  def TuneObjective(self,
                    num_leapfrog_steps,
                    step_size,
                    max_leapfrog_steps=50,
                    f_name="square",
                    percentile=5,
                    min_rhat=1.1,
                    max_rhat=1.3,
                    batch_size=4096,
                    minibatch_size=None,
                    use_new_obj=True,
                    **neutra_args):
    x_init = None

    res = TuneObjective(
        objective=1000,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size)
    if num_leapfrog_steps > max_leapfrog_steps:
      return res
    try:
      cur_neutra_args = neutra_args.copy()
      cur_neutra_args.update(
          test_num_leapfrog_steps=num_leapfrog_steps, test_step_size=step_size)

      if minibatch_size is None:
        minibatch_size = batch_size

      all_stats = []
      for _ in range(batch_size // minibatch_size):
        neutra_outputs = self.MCMC(
            x_init=x_init, batch_size=minibatch_size, **cur_neutra_args)
        mcmc_stats = self.MCMCStats(neutra_outputs)
        all_stats.append((mcmc_stats.p_accept, mcmc_stats.chain_stats_tune))
      p_accept, neutra_stats = AverageStats(all_stats)

      ess_per_grad = neutra_stats[f_name].ess_per_grad
      rhat = neutra_stats[f_name].rhat

      ess_per_grad = np.percentile(ess_per_grad, percentile)
      rhat = np.percentile(rhat, 100 - percentile)

      logging.info(
          "Evaluating step_size: %f, num_leapfrog_steps: %d, "
          "got ess_per_grad %f, rhat %f, p_accept %f", step_size,
          num_leapfrog_steps, ess_per_grad, rhat, p_accept)

      if p_accept < 0.25:
        return res

      if use_new_obj:
        # Linear between 1.1 and 1.3.
        factor = np.minimum(
            np.maximum(rhat - min_rhat, 0.) / (max_rhat - min_rhat), 1.)
        return res._replace(objective=rhat * factor - ess_per_grad *
                            (1. - factor))
      else:
        # If it's above 1.4 or so, ess plays no role.
        ess_factor = np.exp(-(np.maximum(rhat - 1, 0))**2 / (2 * 0.1**2))
        return res._replace(objective=rhat - ess_per_grad * ess_factor)
    except (TypeError, ValueError) as e:
      print(e)
      raise RuntimeError("Error in the objective...")
