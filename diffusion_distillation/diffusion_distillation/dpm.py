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

"""Simple diffusion implementation."""

# pylint:disable=missing-class-docstring,missing-function-docstring
# pylint:disable=logging-format-interpolation
# pylint:disable=g-long-lambda

from . import utils
from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as onp

### Basic diffusion process utilities


def diffusion_reverse(*, x, z_t, logsnr_s, logsnr_t, x_logvar):
  """q(z_s | z_t, x) (requires logsnr_s > logsnr_t (i.e. s < t))."""
  alpha_st = jnp.sqrt((1. + jnp.exp(-logsnr_t)) / (1. + jnp.exp(-logsnr_s)))
  alpha_s = jnp.sqrt(nn.sigmoid(logsnr_s))
  r = jnp.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
  one_minus_r = -jnp.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
  log_one_minus_r = utils.log1mexp(logsnr_s - logsnr_t)  # log(1-SNR(t)/SNR(s))

  mean = r * alpha_st * z_t + one_minus_r * alpha_s * x

  if isinstance(x_logvar, str):
    if x_logvar == 'small':
      # same as setting x_logvar to -infinity
      var = one_minus_r * nn.sigmoid(-logsnr_s)
      logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_s)
    elif x_logvar == 'large':
      # same as setting x_logvar to nn.log_sigmoid(-logsnr_t)
      var = one_minus_r * nn.sigmoid(-logsnr_t)
      logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_t)
    elif x_logvar.startswith('medium:'):
      _, frac = x_logvar.split(':')
      frac = float(frac)
      logging.info('logvar frac=%f', frac)
      assert 0 <= frac <= 1
      min_logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_s)
      max_logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_t)
      logvar = frac * max_logvar + (1 - frac) * min_logvar
      var = jnp.exp(logvar)
    else:
      raise NotImplementedError(x_logvar)
  else:
    assert isinstance(x_logvar, jnp.ndarray) or isinstance(
        x_logvar, onp.ndarray)
    assert x_logvar.shape == x.shape
    # start with "small" variance
    var = one_minus_r * nn.sigmoid(-logsnr_s)
    logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_s)
    # extra variance weight is (one_minus_r*alpha_s)**2
    var += jnp.square(one_minus_r) * nn.sigmoid(logsnr_s) * jnp.exp(x_logvar)
    logvar = jnp.logaddexp(
        logvar, 2. * log_one_minus_r + nn.log_sigmoid(logsnr_s) + x_logvar)
  return {'mean': mean, 'std': jnp.sqrt(var), 'var': var, 'logvar': logvar}


def diffusion_forward(*, x, logsnr):
  """q(z_t | x)."""
  return {
      'mean': x * jnp.sqrt(nn.sigmoid(logsnr)),
      'std': jnp.sqrt(nn.sigmoid(-logsnr)),
      'var': nn.sigmoid(-logsnr),
      'logvar': nn.log_sigmoid(-logsnr)
  }


def predict_x_from_eps(*, z, eps, logsnr):
  """x = (z - sigma*eps)/alpha."""
  logsnr = utils.broadcast_from_left(logsnr, z.shape)
  return jnp.sqrt(1. + jnp.exp(-logsnr)) * (
      z - eps * jax.lax.rsqrt(1. + jnp.exp(logsnr)))


def predict_xlogvar_from_epslogvar(*, eps_logvar, logsnr):
  """Scale Var[eps] by (1+exp(-logsnr)) / (1+exp(logsnr)) = exp(-logsnr)."""
  return eps_logvar - logsnr


def predict_eps_from_x(*, z, x, logsnr):
  """eps = (z - alpha*x)/sigma."""
  logsnr = utils.broadcast_from_left(logsnr, z.shape)
  return jnp.sqrt(1. + jnp.exp(logsnr)) * (
      z - x * jax.lax.rsqrt(1. + jnp.exp(-logsnr)))


def predict_epslogvar_from_xlogvar(*, x_logvar, logsnr):
  """Scale Var[x] by (1+exp(logsnr)) / (1+exp(-logsnr)) = exp(logsnr)."""
  return x_logvar + logsnr


def predict_x_from_v(*, z, v, logsnr):
  logsnr = utils.broadcast_from_left(logsnr, z.shape)
  alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
  sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
  return alpha_t * z - sigma_t * v


def predict_v_from_x_and_eps(*, x, eps, logsnr):
  logsnr = utils.broadcast_from_left(logsnr, x.shape)
  alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
  sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
  return alpha_t * eps - sigma_t * x


class Model:

  def __init__(self, model_fn, *, mean_type, logvar_type, logvar_coeff,
               target_model_fn=None):
    self.model_fn = model_fn
    self.mean_type = mean_type
    self.logvar_type = logvar_type
    self.logvar_coeff = logvar_coeff
    self.target_model_fn = target_model_fn

  def _run_model(self, *, z, logsnr, model_fn, clip_x):
    model_output = model_fn(z, logsnr)
    if self.mean_type == 'eps':
      model_eps = model_output
    elif self.mean_type == 'x':
      model_x = model_output
    elif self.mean_type == 'v':
      model_v = model_output
    elif self.mean_type == 'both':
      _model_x, _model_eps = jnp.split(model_output, 2, axis=-1)  # pylint: disable=invalid-name
    else:
      raise NotImplementedError(self.mean_type)

    # get prediction of x at t=0
    if self.mean_type == 'both':
      # reconcile the two predictions
      model_x_eps = predict_x_from_eps(z=z, eps=_model_eps, logsnr=logsnr)
      wx = utils.broadcast_from_left(nn.sigmoid(-logsnr), z.shape)
      model_x = wx * _model_x + (1. - wx) * model_x_eps
    elif self.mean_type == 'eps':
      model_x = predict_x_from_eps(z=z, eps=model_eps, logsnr=logsnr)
    elif self.mean_type == 'v':
      model_x = predict_x_from_v(z=z, v=model_v, logsnr=logsnr)

    # clipping
    if clip_x:
      model_x = jnp.clip(model_x, -1., 1.)

    # get eps prediction if clipping or if mean_type != eps
    if self.mean_type != 'eps' or clip_x:
      model_eps = predict_eps_from_x(z=z, x=model_x, logsnr=logsnr)

    # get v prediction if clipping or if mean_type != v
    if self.mean_type != 'v' or clip_x:
      model_v = predict_v_from_x_and_eps(
          x=model_x, eps=model_eps, logsnr=logsnr)

    return {'model_x': model_x,
            'model_eps': model_eps,
            'model_v': model_v}

  def predict(self, *, z_t, logsnr_t, logsnr_s, clip_x=None,
              model_output=None, model_fn=None):
    """p(z_s | z_t)."""
    assert logsnr_t.shape == logsnr_s.shape == (z_t.shape[0],)
    if model_output is None:
      assert clip_x is not None
      if model_fn is None:
        model_fn = self.model_fn
      model_output = self._run_model(
          z=z_t, logsnr=logsnr_t, model_fn=model_fn, clip_x=clip_x)

    logsnr_t = utils.broadcast_from_left(logsnr_t, z_t.shape)
    logsnr_s = utils.broadcast_from_left(logsnr_s, z_t.shape)

    pred_x = model_output['model_x']
    if self.logvar_type == 'fixed_small':
      pred_x_logvar = 'small'
    elif self.logvar_type == 'fixed_large':
      pred_x_logvar = 'large'
    elif self.logvar_type.startswith('fixed_medium:'):
      pred_x_logvar = self.logvar_type[len('fixed_'):]
    else:
      raise NotImplementedError(self.logvar_type)

    out = diffusion_reverse(
        z_t=z_t, logsnr_t=logsnr_t, logsnr_s=logsnr_s,
        x=pred_x, x_logvar=pred_x_logvar)
    out['pred_x'] = pred_x
    return out

  def vb(self, *, x, z_t, logsnr_t, logsnr_s, model_output):
    assert x.shape == z_t.shape
    assert logsnr_t.shape == logsnr_s.shape == (z_t.shape[0],)
    q_dist = diffusion_reverse(
        x=x,
        z_t=z_t,
        logsnr_t=utils.broadcast_from_left(logsnr_t, x.shape),
        logsnr_s=utils.broadcast_from_left(logsnr_s, x.shape),
        x_logvar='small')
    p_dist = self.predict(
        z_t=z_t, logsnr_t=logsnr_t, logsnr_s=logsnr_s,
        model_output=model_output)
    kl = utils.normal_kl(
        mean1=q_dist['mean'], logvar1=q_dist['logvar'],
        mean2=p_dist['mean'], logvar2=p_dist['logvar'])
    return utils.meanflat(kl) / onp.log(2.)

  def training_losses(self, *, x, rng, logsnr_schedule_fn,
                      num_steps, mean_loss_weight_type):
    assert x.dtype in [jnp.float32, jnp.float64]
    assert isinstance(num_steps, int)
    rng = utils.RngGen(rng)
    eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
    bc = lambda z: utils.broadcast_from_left(z, x.shape)

    # sample logsnr
    if num_steps > 0:
      logging.info('Discrete time training: num_steps=%d', num_steps)
      assert num_steps >= 1
      i = jax.random.randint(
          next(rng), shape=(x.shape[0],), minval=0, maxval=num_steps)
      u = (i+1).astype(x.dtype) / num_steps
    else:
      logging.info('Continuous time training')
      # continuous time
      u = jax.random.uniform(next(rng), shape=(x.shape[0],), dtype=x.dtype)
    logsnr = logsnr_schedule_fn(u)
    assert logsnr.shape == (x.shape[0],)

    # sample z ~ q(z_logsnr | x)
    z_dist = diffusion_forward(x=x, logsnr=bc(logsnr))
    z = z_dist['mean'] + z_dist['std'] * eps

    # get denoising target
    if self.target_model_fn is not None:  # distillation
      assert num_steps >= 1

      # two forward steps of DDIM from z_t using teacher
      teach_out_start = self._run_model(
          z=z, logsnr=logsnr, model_fn=self.target_model_fn, clip_x=False)
      x_pred = teach_out_start['model_x']
      eps_pred = teach_out_start['model_eps']

      u_mid = u - 0.5/num_steps
      logsnr_mid = logsnr_schedule_fn(u_mid)
      stdv_mid = bc(jnp.sqrt(nn.sigmoid(-logsnr_mid)))
      a_mid = bc(jnp.sqrt(nn.sigmoid(logsnr_mid)))
      z_mid = a_mid * x_pred + stdv_mid * eps_pred

      teach_out_mid = self._run_model(z=z_mid,
                                      logsnr=logsnr_mid,
                                      model_fn=self.target_model_fn,
                                      clip_x=False)
      x_pred = teach_out_mid['model_x']
      eps_pred = teach_out_mid['model_eps']

      u_s = u - 1./num_steps
      logsnr_s = logsnr_schedule_fn(u_s)
      stdv_s = bc(jnp.sqrt(nn.sigmoid(-logsnr_s)))
      a_s = bc(jnp.sqrt(nn.sigmoid(logsnr_s)))
      z_teacher = a_s * x_pred + stdv_s * eps_pred

      # get x-target implied by z_teacher (!= x_pred)
      a_t = bc(jnp.sqrt(nn.sigmoid(logsnr)))
      stdv_frac = bc(jnp.exp(
          0.5 * (nn.softplus(logsnr) - nn.softplus(logsnr_s))))
      x_target = (z_teacher - stdv_frac * z) / (a_s - stdv_frac * a_t)
      x_target = jnp.where(bc(i == 0), x_pred, x_target)
      eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)

    else:  # denoise to original data
      x_target = x
      eps_target = eps

    # also get v-target
    v_target = predict_v_from_x_and_eps(
        x=x_target, eps=eps_target, logsnr=logsnr)

    # denoising loss
    model_output = self._run_model(
        z=z, logsnr=logsnr, model_fn=self.model_fn, clip_x=False)
    x_mse = utils.meanflat(jnp.square(model_output['model_x'] - x_target))
    eps_mse = utils.meanflat(jnp.square(model_output['model_eps'] - eps_target))
    v_mse = utils.meanflat(jnp.square(model_output['model_v'] - v_target))
    if mean_loss_weight_type == 'constant':  # constant weight on x_mse
      loss = x_mse
    elif mean_loss_weight_type == 'snr':  # SNR * x_mse = eps_mse
      loss = eps_mse
    elif mean_loss_weight_type == 'snr_trunc':  # x_mse * max(SNR, 1)
      loss = jnp.maximum(x_mse, eps_mse)
    elif mean_loss_weight_type == 'v_mse':
      loss = v_mse
    else:
      raise NotImplementedError(mean_loss_weight_type)
    return {'loss': loss}

  def ddim_step(self, i, z_t, num_steps, logsnr_schedule_fn, clip_x):
    shape, dtype = z_t.shape, z_t.dtype
    logsnr_t = logsnr_schedule_fn((i + 1.).astype(dtype) / num_steps)
    logsnr_s = logsnr_schedule_fn(i.astype(dtype) / num_steps)
    model_out = self._run_model(
        z=z_t,
        logsnr=jnp.full((shape[0],), logsnr_t),
        model_fn=self.model_fn,
        clip_x=clip_x)
    x_pred_t = model_out['model_x']
    eps_pred_t = model_out['model_eps']
    stdv_s = jnp.sqrt(nn.sigmoid(-logsnr_s))
    alpha_s = jnp.sqrt(nn.sigmoid(logsnr_s))
    z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t
    return jnp.where(i == 0, x_pred_t, z_s_pred)

  def bwd_dif_step(self, rng, i, z_t, num_steps, logsnr_schedule_fn, clip_x):
    shape, dtype = z_t.shape, z_t.dtype
    logsnr_t = logsnr_schedule_fn((i + 1.).astype(dtype) / num_steps)
    logsnr_s = logsnr_schedule_fn(i.astype(dtype) / num_steps)
    z_s_dist = self.predict(
        z_t=z_t,
        logsnr_t=jnp.full((shape[0],), logsnr_t),
        logsnr_s=jnp.full((shape[0],), logsnr_s),
        clip_x=clip_x)
    eps = jax.random.normal(
        jax.random.fold_in(rng, i), shape=shape, dtype=dtype)
    return jnp.where(
        i == 0, z_s_dist['pred_x'], z_s_dist['mean'] + z_s_dist['std'] * eps)

  def sample_loop(self, *, rng, init_x, num_steps,
                  logsnr_schedule_fn, sampler, clip_x):
    if sampler == 'ddim':
      body_fun = lambda i, z_t: self.ddim_step(
          i, z_t, num_steps, logsnr_schedule_fn, clip_x)
    elif sampler == 'noisy':
      body_fun = lambda i, z_t: self.bwd_dif_step(
          rng, i, z_t, num_steps, logsnr_schedule_fn, clip_x)
    else: raise NotImplementedError(sampler)

    # loop over t = num_steps-1, ..., 0
    final_x = utils.reverse_fori_loop(
        lower=0, upper=num_steps, body_fun=body_fun, init_val=init_x)

    assert final_x.shape == init_x.shape and final_x.dtype == init_x.dtype
    return final_x
