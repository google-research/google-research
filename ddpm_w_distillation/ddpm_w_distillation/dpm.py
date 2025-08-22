# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""DPM."""

# pylint:disable=missing-class-docstring,missing-function-docstring
# pylint:disable=logging-format-interpolation,invalid-name,line-too-long
# pylint:disable=g-long-lambda,g-bad-todo,g-no-space-after-comment
# pylint: disable=invalid-unary-operand-type

from . import utils
from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as onp


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
    assert isinstance(x_logvar, jnp.ndarray)
    assert x_logvar.shape == x.shape
    # start with "small" variance
    var = one_minus_r * nn.sigmoid(-logsnr_s)
    logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_s)
    # extra variance weight is (one_minus_r*alpha_s)**2
    var += jnp.square(one_minus_r) * nn.sigmoid(logsnr_s) * jnp.exp(x_logvar)
    logvar = jnp.logaddexp(
        logvar, 2. * log_one_minus_r + nn.log_sigmoid(logsnr_s) + x_logvar)

  assert (mean.shape == var.shape == logvar.shape == x.shape == z_t.shape ==
          logsnr_s.shape == logsnr_t.shape)
  return {'mean': mean, 'std': jnp.sqrt(var), 'var': var, 'logvar': logvar}


def diffusion_forward(*, x, logsnr):
  """q(z_t | x)."""
  assert x.shape == logsnr.shape
  return {
      'mean': x * jnp.sqrt(nn.sigmoid(logsnr)),
      'std': jnp.sqrt(nn.sigmoid(-logsnr)),
      'var': nn.sigmoid(-logsnr),
      'logvar': nn.log_sigmoid(-logsnr)
  }


def predict_x_from_eps(*, z, eps, logsnr):
  """x = (z - sigma*eps)/alpha."""
  logsnr = utils.broadcast_from_left(logsnr, z.shape)
  assert z.shape == eps.shape == logsnr.shape
  return jnp.sqrt(1. + jnp.exp(-logsnr)) * (
      z - eps * jax.lax.rsqrt(1. + jnp.exp(logsnr)))


def predict_eps_from_x(*, z, x, logsnr):
  """eps = (z - alpha*x)/sigma."""
  logsnr = utils.broadcast_from_left(logsnr, z.shape)
  assert z.shape == x.shape == logsnr.shape
  return jnp.sqrt(1. + jnp.exp(logsnr)) * (
      z - x * jax.lax.rsqrt(1. + jnp.exp(-logsnr)))


def predict_v_from_x_and_eps(*, x, eps, logsnr):
  logsnr = utils.broadcast_from_left(logsnr, x.shape)
  alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
  sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
  return alpha_t * eps - sigma_t * x


def predict_x_from_v(*, z, v, logsnr):
  logsnr = utils.broadcast_from_left(logsnr, z.shape)
  alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
  sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
  return alpha_t * z - sigma_t * v


class Model:

  def __init__(
      self,
      model_fn,
      *,
      mean_type,
      logvar_type,
      # logvar_coeff,
      conditional_target_model_fn=None,
      unconditional_target_model_fn=None,
      cond_uncond_coefs=None,
      progressive_distill_loss=None,
      teacher_mean_type=None,
      use_ws=True,
      uncond_student_model=None,
      w_sample_const=None):
    self.model_fn = model_fn
    self.mean_type = mean_type
    self.logvar_type = logvar_type
    self.conditional_target_model_fn = conditional_target_model_fn
    self.unconditional_target_model_fn = unconditional_target_model_fn
    self.cond_uncond_coefs = cond_uncond_coefs
    self.progressive_distill_loss = progressive_distill_loss
    self.teacher_mean_type = teacher_mean_type
    self.use_ws = use_ws
    self.uncond_student_model = uncond_student_model
    self.w_sample_const = w_sample_const

  def _run_model(self,
                 *,
                 z,
                 logsnr,
                 model_fn,
                 clip_x,
                 ws=None,
                 use_teacher=False):
    if use_teacher:
      mean_type = self.teacher_mean_type
    else:
      mean_type = self.mean_type
    # NOTE: this part could cause error
    if not self.use_ws:  # added incorporating ws
      model_output = model_fn(z, logsnr)
    else:
      model_output = model_fn(z, logsnr, ws=ws)

    # NOTE changed all "self.mean_type" to "mean_type"
    if 'learned' in self.logvar_type:
      if mean_type == 'eps':
        model_eps, model_logvar = jnp.split(model_output, 2, axis=-1)
      elif mean_type == 'x':
        model_x, model_logvar = jnp.split(model_output, 2, axis=-1)
      elif mean_type == 'v':
        model_v, model_logvar = jnp.split(model_output, 2, axis=-1)
      elif mean_type == 'both':
        _model_x, _model_eps, model_logvar = jnp.split(model_output, 3, axis=-1)
      else:
        raise NotImplementedError(mean_type)
    else:
      model_logvar = None
      if mean_type == 'eps':
        model_eps = model_output
      elif mean_type == 'x':
        model_x = model_output
      elif mean_type == 'v':
        model_v = model_output
      elif mean_type == 'both':
        _model_x, _model_eps = jnp.split(model_output, 2, axis=-1)
      else:
        raise NotImplementedError(mean_type)

    # get prediction of x at t=0
    if mean_type == 'both':
      # reconcile the two predictions
      model_x_eps = predict_x_from_eps(z=z, eps=_model_eps, logsnr=logsnr)
      wx = utils.broadcast_from_left(nn.sigmoid(-logsnr), z.shape)
      model_x = wx * _model_x + (1. - wx) * model_x_eps
    elif mean_type == 'eps':
      model_x = predict_x_from_eps(z=z, eps=model_eps, logsnr=logsnr)
    elif mean_type == 'v':
      model_x = predict_x_from_v(z=z, v=model_v, logsnr=logsnr)

    # clipping
    if clip_x:
      model_x = jnp.clip(model_x, -1., 1.)

    # get eps prediction if clipping or if mean_type != eps
    if mean_type != 'eps' or clip_x:
      model_eps = predict_eps_from_x(z=z, x=model_x, logsnr=logsnr)

    # get v prediction if clipping or if mean_type != v
    if mean_type != 'v' or clip_x:
      model_v = predict_v_from_x_and_eps(
          x=model_x, eps=model_eps, logsnr=logsnr)

    return {
        'model_x': model_x,
        'model_eps': model_eps,
        'model_v': model_v,
        'model_logvar': model_logvar
    }

  def predict(self,
              *,
              z_t,
              logsnr_t,
              logsnr_s,
              clip_x=None,
              model_output=None,
              model_fn=None):
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

    # depends on t only
    # E[x | z_t]
    pred_x = model_output['model_x']

    # Var[x | z_t]
    if self.logvar_type == 'fixed_small':
      pred_x_logvar = 'small'
    elif self.logvar_type == 'fixed_large':
      pred_x_logvar = 'large'
    elif self.logvar_type.startswith('fixed_medium:'):  # this is used
      pred_x_logvar = self.logvar_type[len('fixed_'):]
    else:
      raise NotImplementedError(self.logvar_type)

    out = diffusion_reverse(
        z_t=z_t,
        logsnr_t=logsnr_t,
        logsnr_s=logsnr_s,
        x=pred_x,
        x_logvar=pred_x_logvar)
    out['pred_x'] = pred_x
    return out

  # added
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
        z_t=z_t,
        logsnr_t=logsnr_t,
        logsnr_s=logsnr_s,
        model_output=model_output)
    kl = utils.normal_kl(
        mean1=q_dist['mean'],
        logvar1=q_dist['logvar'],
        mean2=p_dist['mean'],
        logvar2=p_dist['logvar'])
    return utils.meanflat(kl) / onp.log(2.)

  def training_losses(self,
                      *,
                      x,
                      rng,
                      logsnr_schedule_fn,
                      num_steps,
                      mean_loss_weight_type,
                      logvar_loss_type,
                      w_schedule_fn=None,
                      clip_x=False):
    del logvar_loss_type
    assert x.dtype in [jnp.float32, jnp.float64]
    rng = utils.RngGen(rng)
    # added
    eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
    bc = lambda z: utils.broadcast_from_left(z, x.shape)

    # sample logsnr
    assert isinstance(num_steps, int)
    if num_steps > 0:
      logging.info('Discrete time training: num_steps=%d', num_steps)
      # assert num_steps >= 2
      assert num_steps >= 1
      t = jax.random.randint(
          next(rng), shape=(x.shape[0],), minval=0, maxval=num_steps)
      # u = t.astype(x.dtype) / (num_steps - 1.)
      u = (t + 1).astype(x.dtype) / num_steps
    else:
      logging.info('Continuous time training')
      # continuous time
      t = None
      u = jax.random.uniform(next(rng), shape=(x.shape[0],), dtype=x.dtype)
      # is_continuous_time = True
    logsnr = logsnr_schedule_fn(u)
    assert logsnr.shape == (x.shape[0],)

    # NOTE: check the random noise rng part is correct
    # (e.g., not the same noise)
    if w_schedule_fn is not None:
      u_w = jax.random.uniform(next(rng), shape=(x.shape[0],), dtype=x.dtype)
      ws = w_schedule_fn(u_w)
      ws = ws.reshape(x.shape[0], 1, 1, 1)
      assert ws.shape == (x.shape[0], 1, 1, 1)  # x.shape batch, img, img, 3
    else:
      ws = None

    # sample z ~ q(z_logsnr | x)
    z_dist = diffusion_forward(
        x=x, logsnr=utils.broadcast_from_left(logsnr, x.shape))
    eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
    z = z_dist['mean'] + z_dist['std'] * eps

    # added
    # get denoising target
    if self.conditional_target_model_fn is not None:  # 2->1 distillation
      # Stage 1
      if self.progressive_distill_loss is None:
        # NOTE: progressive_distill_loss is a flag
        # if self.teacher_mean_type == 'eps':
        # two forward steps of DDIM from z_t using teacher
        cond_teach_out_start = self._run_model(
            z=z,
            logsnr=logsnr,
            model_fn=self.conditional_target_model_fn,
            clip_x=False,
            use_teacher=True)
        cond_eps_pred = cond_teach_out_start['model_eps']
        uncond_teach_out_start = self._run_model(
            z=z,
            logsnr=logsnr,
            model_fn=self.unconditional_target_model_fn,
            clip_x=False,
            use_teacher=True)
        uncond_eps_pred = uncond_teach_out_start['model_eps']

        if w_schedule_fn is None:
          # need use the fixed w in config (cond_uncond_coefs)
          cond_coef, uncond_coef = self.cond_uncond_coefs
        else:
          cond_coef, uncond_coef = 1 + ws, -ws  # pytype: disable=unsupported-operands  # dataclasses-replace
        eps_target = cond_coef * cond_eps_pred + uncond_coef * uncond_eps_pred
        x_target = predict_x_from_eps(z=z, eps=eps_target, logsnr=logsnr)

        if clip_x:  # NOTE newly added
          x_target = jnp.clip(x_target, -1., 1.)
          eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)

        # clipping
        # if clip_x:
        # x_pred = jnp.clipx_pred, -1., 1.)

        # u_mid = u - 0.5/num_steps
        # logsnr_mid = logsnr_schedule_fn(u_mid)
        # stdv_mid = bc(jnp.sqrt(nn.sigmoid(-logsnr_mid)))
        # a_mid = bc(jnp.sqrt(nn.sigmoid(logsnr_mid)))
        # z_mid = a_mid * x_pred + stdv_mid * eps_pred

        # teach_out_mid = self._run_model(z=z_mid,
        #                                 logsnr=logsnr_mid,
        #                                 model_fn=self.target_model_fn,
        #                                 clip_x=False)
        # x_pred = teach_out_mid['model_x']
        # eps_pred = teach_out_mid['model_eps']

        # u_s = u - 1./num_steps
        # logsnr_s = logsnr_schedule_fn(u_s)
        # stdv_s = bc(jnp.sqrt(nn.sigmoid(-logsnr_s)))
        # a_s = bc(jnp.sqrt(nn.sigmoid(logsnr_s)))
        # z_teacher = a_s * x_pred + stdv_s * eps_pred

        # # get x-target implied by z_teacher (!= x_pred)
        # a_t = bc(jnp.sqrt(nn.sigmoid(logsnr)))
        # stdv_frac = bc(jnp.exp(
        #     0.5 * (nn.softplus(logsnr) - nn.softplus(logsnr_s))))
        # x_target = (z_teacher - stdv_frac * z) / (a_s - stdv_frac * a_t)
        # x_target = jnp.where(bc(i == 0), x_pred, x_target)
        # eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)

      # NOTE: Stage 2
      else:  # NOTE Stage 2: progressive distillation
        assert num_steps >= 1

        # two forward steps of DDIM from z_t using teacher
        if ws is None:
          teach_out_start = self._run_model(
              z=z,
              logsnr=logsnr,
              model_fn=self.conditional_target_model_fn,
              clip_x=False,
              ws=None,
              use_teacher=False)
        else:
          teach_out_start = self._run_model(
              z=z,
              logsnr=logsnr,
              model_fn=self.conditional_target_model_fn,
              clip_x=False,
              ws=ws.reshape(-1),
              use_teacher=False)
        x_pred = teach_out_start['model_x']
        eps_pred = teach_out_start['model_eps']

        u_mid = u - 0.5 / num_steps
        logsnr_mid = logsnr_schedule_fn(u_mid)
        stdv_mid = bc(jnp.sqrt(nn.sigmoid(-logsnr_mid)))
        a_mid = bc(jnp.sqrt(nn.sigmoid(logsnr_mid)))
        z_mid = a_mid * x_pred + stdv_mid * eps_pred

        if ws is None:
          teach_out_mid = self._run_model(
              z=z_mid,
              logsnr=logsnr_mid,
              model_fn=self.conditional_target_model_fn,
              clip_x=False,
              ws=None)
        else:
          teach_out_mid = self._run_model(
              z=z_mid,
              logsnr=logsnr_mid,
              model_fn=self.conditional_target_model_fn,
              clip_x=False,
              ws=ws.reshape(-1))

        x_pred = teach_out_mid['model_x']  # checked not NaN
        eps_pred = teach_out_mid['model_eps']

        u_s = u - 1. / num_steps
        logsnr_s = logsnr_schedule_fn(u_s)
        stdv_s = bc(jnp.sqrt(nn.sigmoid(-logsnr_s)))
        a_s = bc(jnp.sqrt(nn.sigmoid(logsnr_s)))
        z_teacher = a_s * x_pred + stdv_s * eps_pred  # has nan issue
        # NOTE: the nan issue is because u is redefined for the random seed
        # used in w sampling

        # get x-target implied by z_teacher (!= x_pred)
        a_t = bc(jnp.sqrt(nn.sigmoid(logsnr)))
        stdv_frac = bc(
            jnp.exp(0.5 * (nn.softplus(logsnr) - nn.softplus(logsnr_s))))
        x_target = (z_teacher - stdv_frac * z) / (a_s - stdv_frac * a_t)
        x_target = jnp.where(bc(t == 0), x_pred, x_target)
        eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)

        # # NOTE debugging purposes, remove
        # x_target = eps_pred #x_pred
        # eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)

    else:  # denoise to original data
      x_target = x
      eps_target = eps
    # end added. Note, add the v_target function here
    # also get v-target
    v_target = predict_v_from_x_and_eps(
        x=x_target, eps=eps_target, logsnr=logsnr)

    # denoise and calculate loss NOTE original
    # model_mean_param, model_logvar_param = self._run_model(z=z, logsnr=logsnr)
    # denoising loss
    if ws is None:
      model_output = self._run_model(
          z=z, logsnr=logsnr, model_fn=self.model_fn, clip_x=False, ws=None)
    else:
      model_output = self._run_model(
          z=z,
          logsnr=logsnr,
          model_fn=self.model_fn,
          clip_x=False,
          ws=ws.reshape(-1))

    x_mse = utils.meanflat(jnp.square(model_output['model_x'] - x_target))
    eps_mse = utils.meanflat(jnp.square(model_output['model_eps'] - eps_target))
    v_mse = utils.meanflat(jnp.square(model_output['model_v'] - v_target))
    if mean_loss_weight_type == 'constant':  # constant weight on x_mse
      loss_mean = x_mse
    elif mean_loss_weight_type == 'snr':  # SNR * x_mse = eps_mse
      loss_mean = eps_mse
    elif mean_loss_weight_type == 'snr_trunc':  # x_mse * max(SNR, 1)
      loss_mean = jnp.maximum(x_mse, eps_mse)
    elif mean_loss_weight_type == 'v_mse':
      loss_mean = v_mse
    else:
      raise NotImplementedError(mean_loss_weight_type)

    loss_var = jnp.zeros_like(loss_mean)

    assert loss_mean.shape == loss_var.shape == (x.shape[0],)
    loss = loss_mean + loss_var

    # prior bpd
    z1_dist = diffusion_forward(
        x=x, logsnr=jnp.full(x.shape, logsnr_schedule_fn(1.)))
    prior_bpd = (1. / onp.log(2.)) * utils.meanflat(
        utils.normal_kl(
            mean1=z1_dist['mean'],
            logvar1=z1_dist['logvar'],
            mean2=0.,
            logvar2=0.))

    return {
        'loss': loss,
        'prior_bpd': prior_bpd,
        'loss_mean': loss_mean,
        'loss_var': loss_var
    }

  def two_nvidia_training_losses(self,
                                 *,
                                 x,
                                 rng,
                                 logsnr_schedule_fn,
                                 num_steps,
                                 mean_loss_weight_type,
                                 logvar_loss_type,
                                 w_schedule_fn=None,
                                 clip_x=False):
    del clip_x
    assert x.dtype in [jnp.float32, jnp.float64]
    rng = utils.RngGen(rng)
    # added
    eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
    bc = lambda z: utils.broadcast_from_left(z, x.shape)

    # sample logsnr
    assert isinstance(num_steps, int)
    logging.info('Discrete time training: num_steps=%d', num_steps)
    # assert num_steps >= 2
    assert num_steps >= 1
    t = jax.random.randint(
        next(rng), shape=(x.shape[0],), minval=0, maxval=num_steps)
    # u = t.astype(x.dtype) / (num_steps - 1.)
    u = (t + 1).astype(x.dtype) / num_steps

    # is_continuous_time = True
    logsnr = logsnr_schedule_fn(u)
    assert logsnr.shape == (x.shape[0],)

    # NOTE: check the random noise rng part is correct
    # (e.g., not the same noise)
    if w_schedule_fn is not None:
      u_w = jax.random.uniform(next(rng), shape=(x.shape[0],), dtype=x.dtype)
      ws = w_schedule_fn(u_w)
      ws = ws.reshape(x.shape[0], 1, 1, 1)
      assert ws.shape == (x.shape[0], 1, 1, 1)  # x.shape batch, img, img, 3
    else:
      ws = None

    # sample z ~ q(z_logsnr | x)
    z_dist = diffusion_forward(
        x=x, logsnr=utils.broadcast_from_left(logsnr, x.shape))
    eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
    z = z_dist['mean'] + z_dist['std'] * eps

    # two forward steps of DDIM from z_t using teacher
    if ws is None:
      teach_out_start = self._run_model(
          z=z,
          logsnr=logsnr,
          model_fn=self.conditional_target_model_fn,
          clip_x=False,
          ws=None,
          use_teacher=False)
    else:
      teach_out_start = self._run_model(
          z=z,
          logsnr=logsnr,
          model_fn=self.conditional_target_model_fn,
          clip_x=False,
          ws=ws.reshape(-1),
          use_teacher=False)
    x_pred = teach_out_start['model_x']
    eps_pred = teach_out_start['model_eps']

    u_mid = u - 1. / num_steps  # u - 0.5/num_steps
    logsnr_mid = logsnr_schedule_fn(u_mid)
    stdv_mid = bc(jnp.sqrt(nn.sigmoid(-logsnr_mid)))
    a_mid = bc(jnp.sqrt(nn.sigmoid(logsnr_mid)))
    z_mid = a_mid * x_pred + stdv_mid * eps_pred

    # TODO fix this part
    # get x-target implied by z_teacher (!= x_pred)
    # A one-step update when i=0 (used when i=0 only)
    a_t = bc(jnp.sqrt(nn.sigmoid(logsnr)))
    stdv_frac = bc(
        jnp.exp(0.5 * (nn.softplus(logsnr) - nn.softplus(logsnr_mid))))
    x_target_init = (z_mid - stdv_frac * z) / (a_mid - stdv_frac * a_t)
    # x_target = jnp.where(bc(t == 0), x_pred, x_target)
    eps_target_init = predict_eps_from_x(z=z, x=x_target_init, logsnr=logsnr)

    # x_target_init = x_pred
    # eps_target_init = eps_pred

    if ws is None:
      teach_out_mid = self._run_model(
          z=z_mid,
          logsnr=logsnr_mid,
          model_fn=self.conditional_target_model_fn,
          clip_x=False,
          ws=None)
    else:
      teach_out_mid = self._run_model(
          z=z_mid,
          logsnr=logsnr_mid,
          model_fn=self.conditional_target_model_fn,
          clip_x=False,
          ws=ws.reshape(-1))

    x_pred = teach_out_mid['model_x']  # checked not NaN
    eps_pred = teach_out_mid['model_eps']

    # x_target_init = x_pred
    # eps_target_init = eps_pred

    # NOTE buggy (also check sampling)
    # if u >= 2./num_steps:
    u_s = u - 2. / num_steps  #u - 1./num_steps
    logsnr_s = logsnr_schedule_fn(u_s)
    stdv_s = bc(jnp.sqrt(nn.sigmoid(-logsnr_s)))
    a_s = bc(jnp.sqrt(nn.sigmoid(logsnr_s)))
    z_teacher = a_s * x_pred + stdv_s * eps_pred  # has nan issue
    # NOTE: the nan issue is because u is redefined for the random seed
    # used in w sampling

    # get x-target implied by z_teacher (!= x_pred)
    a_t = bc(jnp.sqrt(nn.sigmoid(logsnr)))
    stdv_frac = bc(jnp.exp(0.5 * (nn.softplus(logsnr) - nn.softplus(logsnr_s))))
    x_target = (z_teacher - stdv_frac * z) / (a_s - stdv_frac * a_t)
    x_target = jnp.where(bc(t == 0), x_pred, x_target)
    eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)
    # else:
    # x_target = x_pred
    # eps_target = eps_pred

    x_target = jnp.where((u >= 2. / num_steps).reshape(-1, 1, 1, 1), x_target,
                         x_target_init)
    eps_target = jnp.where((u >= 2. / num_steps).reshape(-1, 1, 1, 1),
                           eps_target, eps_target_init)

    # also get v-target
    v_target = predict_v_from_x_and_eps(
        x=x_target, eps=eps_target, logsnr=logsnr)

    # denoise and calculate loss NOTE original
    # model_mean_param, model_logvar_param = self._run_model(z=z, logsnr=logsnr)
    # denoising loss
    if ws is None:
      model_output = self._run_model(
          z=z, logsnr=logsnr, model_fn=self.model_fn, clip_x=False, ws=None)
    else:
      model_output = self._run_model(
          z=z,
          logsnr=logsnr,
          model_fn=self.model_fn,
          clip_x=False,
          ws=ws.reshape(-1))

    x_mse = utils.meanflat(jnp.square(model_output['model_x'] - x_target))
    eps_mse = utils.meanflat(jnp.square(model_output['model_eps'] - eps_target))
    v_mse = utils.meanflat(jnp.square(model_output['model_v'] - v_target))
    if mean_loss_weight_type == 'constant':  # constant weight on x_mse
      loss_mean = x_mse
    elif mean_loss_weight_type == 'snr':  # SNR * x_mse = eps_mse
      loss_mean = eps_mse
    elif mean_loss_weight_type == 'snr_trunc':  # x_mse * max(SNR, 1)
      loss_mean = jnp.maximum(x_mse, eps_mse)
    elif mean_loss_weight_type == 'v_mse':
      loss_mean = v_mse
    else:
      raise NotImplementedError(mean_loss_weight_type)

    if logvar_loss_type == 'vb':
      logsnr_s = logsnr_schedule_fn(jnp.maximum(0., u - 1. / num_steps))
      loss_var = self.vb(
          x=x_target,
          z_t=z,
          logsnr_t=logsnr,
          logsnr_s=logsnr_s,
          model_output=model_output)
    elif logvar_loss_type == 'none':
      loss_var = jnp.zeros_like(loss_mean)
    else:
      raise NotImplementedError(logvar_loss_type)
    # end modification

    assert loss_mean.shape == loss_var.shape == (x.shape[0],)
    loss = loss_mean + loss_var

    # prior bpd
    z1_dist = diffusion_forward(
        x=x, logsnr=jnp.full(x.shape, logsnr_schedule_fn(1.)))
    prior_bpd = (1. / onp.log(2.)) * utils.meanflat(
        utils.normal_kl(
            mean1=z1_dist['mean'],
            logvar1=z1_dist['logvar'],
            mean2=0.,
            logvar2=0.))

    return {
        'loss': loss,
        'prior_bpd': prior_bpd,
        'loss_mean': loss_mean,
        'loss_var': loss_var
    }

  # newly added loss, need to change main and include a new config file
  def encoding_training_losses(self,
                               *,
                               x,
                               rng,
                               logsnr_schedule_fn,
                               num_steps,
                               mean_loss_weight_type,
                               logvar_loss_type,
                               w_schedule_fn=None,
                               clip_x=False):
    del clip_x
    assert x.dtype in [jnp.float32, jnp.float64]
    rng = utils.RngGen(rng)
    # added
    eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
    bc = lambda z: utils.broadcast_from_left(z, x.shape)

    # sample logsnr
    assert isinstance(num_steps, int)
    logging.info('Discrete time training: num_steps=%d', num_steps)
    # assert num_steps >= 2
    assert num_steps >= 1
    t = jax.random.randint(
        next(rng), shape=(x.shape[0],), minval=0, maxval=num_steps)
    u = (t + 1).astype(x.dtype) / num_steps

    logsnr = logsnr_schedule_fn(u)
    assert logsnr.shape == (x.shape[0],)

    u_mid = u - 0.5 / num_steps
    logsnr_mid = logsnr_schedule_fn(u_mid)
    stdv_mid = bc(jnp.sqrt(nn.sigmoid(-logsnr_mid)))
    a_mid = bc(jnp.sqrt(nn.sigmoid(logsnr_mid)))

    u_s = u - 1. / num_steps
    logsnr_s = logsnr_schedule_fn(u_s)
    # stdv_s = bc(jnp.sqrt(nn.sigmoid(-logsnr_s)))
    a_s = bc(jnp.sqrt(nn.sigmoid(logsnr_s)))

    # NOTE: check the random noise rng part is correct
    # (e.g., not the same noise)
    if w_schedule_fn is not None:
      u_w = jax.random.uniform(next(rng), shape=(x.shape[0],), dtype=x.dtype)
      ws = w_schedule_fn(u_w)
      ws = ws.reshape(x.shape[0], 1, 1, 1)
      assert ws.shape == (x.shape[0], 1, 1, 1)  # x.shape batch, img, img, 3
    else:
      ws = None

    # sample z_start ~ q(z_logsnr_s | x) #NOTE logsnr_s not logsnr
    z_dist = diffusion_forward(
        x=x, logsnr=utils.broadcast_from_left(logsnr_s, x.shape))
    eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
    z = z_dist['mean'] + z_dist['std'] * eps

    # two forward steps of DDIM from z_t using teacher
    if ws is None:
      teach_out_start = self._run_model(
          z=z,
          logsnr=logsnr_s,
          model_fn=self.conditional_target_model_fn,
          clip_x=False,
          ws=None,
          use_teacher=False)
    else:
      teach_out_start = self._run_model(
          z=z,
          logsnr=logsnr_s,
          model_fn=self.conditional_target_model_fn,
          clip_x=False,
          ws=ws.reshape(-1),
          use_teacher=False)
    x_pred = teach_out_start['model_x']
    eps_pred = teach_out_start['model_eps']

    z_mid = a_mid * x_pred + stdv_mid * eps_pred

    if ws is None:
      teach_out_mid = self._run_model(
          z=z_mid,
          logsnr=logsnr_mid,
          model_fn=self.conditional_target_model_fn,
          clip_x=False,
          ws=None)
    else:
      teach_out_mid = self._run_model(
          z=z_mid,
          logsnr=logsnr_mid,
          model_fn=self.conditional_target_model_fn,
          clip_x=False,
          ws=ws.reshape(-1))

    x_pred = teach_out_mid['model_x']
    eps_pred = teach_out_mid['model_eps']

    stdv = bc(jnp.sqrt(nn.sigmoid(-logsnr)))  # largest noise level
    a = bc(jnp.sqrt(nn.sigmoid(logsnr)))  # largest noise level
    z_teacher = a * x_pred + stdv * eps_pred  # largest noise level

    # get x-target implied by z_teacher (!= x_pred)
    # TODO NOTE this part is buggy
    # a_t = bc(jnp.sqrt(nn.sigmoid(logsnr)))
    stdv_frac = bc(jnp.exp(0.5 * (nn.softplus(logsnr_s) - nn.softplus(logsnr))))
    x_target = (z_teacher - stdv_frac * z) / (a - stdv_frac * a_s)
    # x_target = jnp.where(bc(t == (num_steps-1)), x_pred, x_target) # NOTE remove this line
    eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)

    # end added. Note, add the v_target function here
    # also get v-target
    v_target = predict_v_from_x_and_eps(
        x=x_target, eps=eps_target, logsnr=logsnr)

    # denoise and calculate loss NOTE original
    # model_mean_param, model_logvar_param = self._run_model(z=z, logsnr=logsnr)
    # denoising loss
    if ws is None:
      model_output = self._run_model(
          z=z, logsnr=logsnr_s, model_fn=self.model_fn, clip_x=False, ws=None)
    else:
      model_output = self._run_model(
          z=z,
          logsnr=logsnr_s,
          model_fn=self.model_fn,
          clip_x=False,
          ws=ws.reshape(-1))

    x_mse = utils.meanflat(jnp.square(model_output['model_x'] - x_target))
    eps_mse = utils.meanflat(jnp.square(model_output['model_eps'] - eps_target))
    v_mse = utils.meanflat(jnp.square(model_output['model_v'] - v_target))
    if mean_loss_weight_type == 'constant':  # constant weight on x_mse
      loss_mean = x_mse
    elif mean_loss_weight_type == 'snr':  # SNR * x_mse = eps_mse
      loss_mean = eps_mse
    elif mean_loss_weight_type == 'snr_trunc':  # x_mse * max(SNR, 1)
      loss_mean = jnp.maximum(x_mse, eps_mse)
    elif mean_loss_weight_type == 'v_mse':
      loss_mean = v_mse
    else:
      raise NotImplementedError(mean_loss_weight_type)

    # var loss
    if logvar_loss_type == 'vb':
      raise NotImplementedError(logvar_loss_type)
    elif logvar_loss_type == 'none':
      loss_var = jnp.zeros_like(loss_mean)
    else:
      raise NotImplementedError(logvar_loss_type)
    # end modification

    assert loss_mean.shape == loss_var.shape == (x.shape[0],)
    loss = loss_mean + loss_var

    # prior bpd
    z1_dist = diffusion_forward(
        x=x, logsnr=jnp.full(x.shape, logsnr_schedule_fn(1.)))
    prior_bpd = (1. / onp.log(2.)) * utils.meanflat(
        utils.normal_kl(
            mean1=z1_dist['mean'],
            logvar1=z1_dist['logvar'],
            mean2=0.,
            logvar2=0.))

    return {
        'loss': loss,
        'prior_bpd': prior_bpd,
        'loss_mean': loss_mean,
        'loss_var': loss_var
    }


# NOTE: added two model distillation training loss

  def two_model_training_losses(self,
                                *,
                                x,
                                rng,
                                logsnr_schedule_fn,
                                num_steps,
                                mean_loss_weight_type,
                                logvar_loss_type,
                                w_schedule_fn=None,
                                clip_x=False):
    assert x.dtype in [jnp.float32, jnp.float64]
    rng = utils.RngGen(rng)
    # added
    eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
    bc = lambda z: utils.broadcast_from_left(z, x.shape)

    # sample logsnr
    assert isinstance(num_steps, int)
    if num_steps > 0:
      logging.info('Discrete time training: num_steps=%d', num_steps)
      # assert num_steps >= 2
      assert num_steps >= 1
      t = jax.random.randint(
          next(rng), shape=(x.shape[0],), minval=0, maxval=num_steps)
      # u = t.astype(x.dtype) / (num_steps - 1.)
      u = (t + 1).astype(x.dtype) / num_steps
    else:
      logging.info('Continuous time training')
      # continuous time
      t = None
      u = jax.random.uniform(next(rng), shape=(x.shape[0],), dtype=x.dtype)
      # is_continuous_time = True
    logsnr = logsnr_schedule_fn(u)
    assert logsnr.shape == (x.shape[0],)

    # NOTE: check the random noise rng part is correct (e.g., not the same noise)
    if w_schedule_fn is not None:
      u_w = jax.random.uniform(next(rng), shape=(x.shape[0],), dtype=x.dtype)
      ws = w_schedule_fn(u_w)
      ws = ws.reshape(x.shape[0], 1, 1, 1)
      assert ws.shape == (x.shape[0], 1, 1, 1)  # x.shape batch, img, img, 3
    else:
      ws = None

    # sample z ~ q(z_logsnr | x)
    z_dist = diffusion_forward(
        x=x, logsnr=utils.broadcast_from_left(logsnr, x.shape))
    eps = jax.random.normal(next(rng), shape=x.shape, dtype=x.dtype)
    z = z_dist['mean'] + z_dist['std'] * eps

    # added two student distillation model (one condition, one uncondition)
    # get denoising target
    # print("num_steps ", num_steps)
    assert num_steps >= 1  # NOTE need to be discrete steps
    if w_schedule_fn is None:  # need use the fixed w in config (cond_uncond_coefs)
      cond_coef, uncond_coef = self.cond_uncond_coefs
    else:
      cond_coef, uncond_coef = 1 + ws, -ws  # pytype: disable=unsupported-operands  # dataclasses-replace

    # two forward steps of DDIM from z_t using teacher
    cond_teach_out_start = self._run_model(
        z=z,
        logsnr=logsnr,
        model_fn=self.conditional_target_model_fn,
        clip_x=False,
        use_teacher=True)
    cond_eps_pred = cond_teach_out_start['model_eps']
    uncond_teach_out_start = self._run_model(
        z=z,
        logsnr=logsnr,
        model_fn=self.unconditional_target_model_fn,
        clip_x=False,
        use_teacher=True)
    uncond_eps_pred = uncond_teach_out_start['model_eps']
    eps_pred = cond_coef * cond_eps_pred + uncond_coef * uncond_eps_pred
    x_pred = predict_x_from_eps(z=z, eps=eps_pred, logsnr=logsnr)
    if clip_x:  # NOTE newly added
      raise NotImplementedError
      # x_pred = jnp.clip(x_pred, -1., 1.)
      # eps_pred = predict_eps_from_x(z=z, x=x_pred, logsnr=logsnr)

    u_mid = u - 0.5 / num_steps
    logsnr_mid = logsnr_schedule_fn(u_mid)
    stdv_mid = bc(jnp.sqrt(nn.sigmoid(-logsnr_mid)))
    a_mid = bc(jnp.sqrt(nn.sigmoid(logsnr_mid)))
    z_mid = a_mid * x_pred + stdv_mid * eps_pred

    # NOTE check if has applied _mid properly
    cond_teach_out_mid = self._run_model(
        z=z_mid,
        logsnr=logsnr_mid,
        model_fn=self.conditional_target_model_fn,
        clip_x=False,
        use_teacher=True)
    cond_eps_pred = cond_teach_out_mid['model_eps']
    cond_x_pred = predict_x_from_eps(
        z=z_mid, eps=cond_eps_pred, logsnr=logsnr_mid)

    uncond_teach_out_mid = self._run_model(
        z=z_mid,
        logsnr=logsnr_mid,
        model_fn=self.unconditional_target_model_fn,
        clip_x=False,
        use_teacher=True)
    uncond_eps_pred = uncond_teach_out_mid['model_eps']
    uncond_x_pred = predict_x_from_eps(
        z=z_mid, eps=uncond_eps_pred, logsnr=logsnr_mid)
    # eps_pred = cond_coef * cond_eps_pred + uncond_coef * uncond_eps_pred

    u_s = u - 1. / num_steps
    logsnr_s = logsnr_schedule_fn(u_s)
    stdv_s = bc(jnp.sqrt(nn.sigmoid(-logsnr_s)))
    a_s = bc(jnp.sqrt(nn.sigmoid(logsnr_s)))
    cond_z_teacher = a_s * cond_x_pred + stdv_s * cond_eps_pred  # has nan issue
    uncond_z_teacher = a_s * uncond_x_pred + stdv_s * uncond_eps_pred  # has nan issue

    # get x-target implied by z_teacher (!= x_pred)
    a_t = bc(jnp.sqrt(nn.sigmoid(logsnr)))
    stdv_frac = bc(jnp.exp(0.5 * (nn.softplus(logsnr) - nn.softplus(logsnr_s))))
    # x_target = (z_teacher - stdv_frac * z) / (a_s - stdv_frac * a_t)
    # x_target = jnp.where(bc(t == 0), x_pred, x_target)
    # eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)

    cond_x_target = (cond_z_teacher - stdv_frac * z) / (a_s - stdv_frac * a_t)
    cond_x_target = jnp.where(bc(t == 0), cond_x_pred, cond_x_target)
    cond_eps_target = predict_eps_from_x(
        z=z, x=cond_x_target, logsnr=logsnr)  # TODO check logsnr or logsnr_mid

    uncond_x_target = (uncond_z_teacher - stdv_frac * z) / (
        a_s - stdv_frac * a_t)
    uncond_x_target = jnp.where(bc(t == 0), uncond_x_pred, uncond_x_target)
    uncond_eps_target = predict_eps_from_x(
        z=z, x=uncond_x_target, logsnr=logsnr)
    # end added. Note, add the v_target function here
    # also get v-target
    cond_v_target = predict_v_from_x_and_eps(
        x=cond_x_target, eps=cond_eps_target, logsnr=logsnr)
    uncond_v_target = predict_v_from_x_and_eps(
        x=uncond_x_target, eps=uncond_eps_target, logsnr=logsnr)

    # denoise and calculate loss NOTE original
    # model_mean_param, model_logvar_param = self._run_model(z=z, logsnr=logsnr)
    # denoising loss
    if ws is None:
      cond_model_output = self._run_model(
          z=z, logsnr=logsnr, model_fn=self.model_fn, clip_x=False, ws=None)
      uncond_model_output = self._run_model(
          z=z,
          logsnr=logsnr,
          model_fn=self.uncond_student_model,
          clip_x=False,
          ws=None)
    else:
      cond_model_output = self._run_model(
          z=z,
          logsnr=logsnr,
          model_fn=self.model_fn,
          clip_x=False,
          ws=ws.reshape(-1))
      uncond_model_output = self._run_model(
          z=z,
          logsnr=logsnr,
          model_fn=self.uncond_student_model,
          clip_x=False,
          ws=ws.reshape(-1))

    cond_x_mse = utils.meanflat(
        jnp.square(cond_model_output['model_x'] - cond_x_target))
    cond_eps_mse = utils.meanflat(
        jnp.square(cond_model_output['model_eps'] - cond_eps_target))
    cond_v_mse = utils.meanflat(
        jnp.square(cond_model_output['model_v'] - cond_v_target))

    uncond_x_mse = utils.meanflat(
        jnp.square(uncond_model_output['model_x'] - uncond_x_target))
    uncond_eps_mse = utils.meanflat(
        jnp.square(uncond_model_output['model_eps'] - uncond_eps_target))
    uncond_v_mse = utils.meanflat(
        jnp.square(uncond_model_output['model_v'] - uncond_v_target))

    if mean_loss_weight_type == 'constant':  # constant weight on x_mse
      loss_mean = cond_x_mse + uncond_x_mse
    elif mean_loss_weight_type == 'snr':  # SNR * x_mse = eps_mse
      loss_mean = cond_eps_mse + uncond_eps_mse
    elif mean_loss_weight_type == 'snr_trunc':  # x_mse * max(SNR, 1)
      loss_mean = jnp.maximum(cond_x_mse, cond_eps_mse) + jnp.maximum(
          uncond_x_mse, uncond_eps_mse)
    elif mean_loss_weight_type == 'v_mse':
      loss_mean = cond_v_mse + uncond_v_mse
    else:
      raise NotImplementedError(mean_loss_weight_type)

    if logvar_loss_type == 'vb':
      raise NotImplementedError(logvar_loss_type)
    elif logvar_loss_type == 'none':
      loss_var = jnp.zeros_like(loss_mean)
    else:
      raise NotImplementedError(logvar_loss_type)
    # end modification

    assert loss_mean.shape == loss_var.shape == (x.shape[0],)
    loss = loss_mean + loss_var

    # prior bpd
    z1_dist = diffusion_forward(
        x=x, logsnr=jnp.full(x.shape, logsnr_schedule_fn(1.)))
    prior_bpd = (1. / onp.log(2.)) * utils.meanflat(
        utils.normal_kl(
            mean1=z1_dist['mean'],
            logvar1=z1_dist['logvar'],
            mean2=0.,
            logvar2=0.))

    return {
        'loss': loss,
        'prior_bpd': prior_bpd,
        'loss_mean': loss_mean,
        'loss_var': loss_var
    }

  #########
  # added
  def two_student_ddim_step(self, i, z_t, num_steps, logsnr_schedule_fn,
                            clip_x):
    shape, dtype = z_t.shape, z_t.dtype
    logsnr_t = logsnr_schedule_fn((i + 1.).astype(dtype) / num_steps)
    logsnr_s = logsnr_schedule_fn(i.astype(dtype) / num_steps)
    model_out = self._run_model(
        z=z_t,
        logsnr=jnp.full((shape[0],), logsnr_t),
        model_fn=self.model_fn,
        clip_x=clip_x)
    # cond_x_pred_t = model_out['model_x']
    cond_eps_pred_t = model_out['model_eps']

    # import pdb; pdb.set_trace()
    uncond_model_out = self._run_model(
        z=z_t,
        logsnr=jnp.full((shape[0],), logsnr_t),
        model_fn=self.uncond_student_model,
        clip_x=clip_x)
    # uncond_x_pred_t = uncond_model_out['model_x']
    uncond_eps_pred_t = uncond_model_out['model_eps']

    cond_coef, uncond_coef = 1. + self.w_sample_const, -self.w_sample_const
    eps_pred_t = cond_coef * cond_eps_pred_t + uncond_coef * uncond_eps_pred_t
    x_pred_t = predict_x_from_eps(
        z=z_t, eps=eps_pred_t, logsnr=jnp.full((shape[0],), logsnr_t))
    stdv_s = jnp.sqrt(nn.sigmoid(-logsnr_s))
    alpha_s = jnp.sqrt(nn.sigmoid(logsnr_s))
    z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t
    return jnp.where(i == 0, x_pred_t, z_s_pred)

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

  # added for style transfer
  def ddim_encoding_step(self, i, z_s, num_steps, logsnr_schedule_fn,
                         clip_x):
    shape, dtype = z_s.shape, z_s.dtype
    logsnr_t = logsnr_schedule_fn((i + 1.).astype(dtype) / num_steps)
    logsnr_s = logsnr_schedule_fn(i.astype(dtype) / num_steps)

    model_out = self._run_model(
        z=z_s,
        logsnr=jnp.full((shape[0],), logsnr_s),  # logsnr_t
        model_fn=self.model_fn,
        clip_x=clip_x)
    x_pred_s = model_out['model_x']
    eps_pred_s = model_out['model_eps']
    # z_s_pred = alpha_s * x_true + stdv_s * eps_pred_t

    stdv_t = jnp.sqrt(nn.sigmoid(-logsnr_t))
    alpha_t = jnp.sqrt(nn.sigmoid(logsnr_t))

    # if i == 0:
    # z_s_pred_init = alpha_t * z_t + stdv_t * eps_pred_t
    # # else:
    # eps_coff = jnp.sqrt(stdv_t **2 - (alpha_t * stdv_s / alpha_s) ** 2)
    # z_s_pred = alpha_t * z_t / alpha_s + eps_coff * eps_pred_t

    # z_s_pred = alpha_t * (z_t - stdv_s * eps_pred_t) / alpha_s + stdv_t * eps_pred_t
    z_t_pred = alpha_t * x_pred_s + stdv_t * eps_pred_s

    return z_t_pred
    # jnp.where(i == 0, z_s_pred_init, z_s_pred) # TODO check here jnp.where(i == 0, x_pred_t, z_s_pred)

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
    return jnp.where(i == 0, z_s_dist['pred_x'],
                     z_s_dist['mean'] + z_s_dist['std'] * eps)

  # NOTE ddim interpolated
  def added_noisy_step(self, rng, i, z_t, num_steps, logsnr_schedule_fn, clip_x,
                       interpolation):
    shape, dtype = z_t.shape, z_t.dtype
    logsnr_t = logsnr_schedule_fn((i + 1.).astype(dtype) / num_steps)
    # NOTE: need to use num_steps not train_steps! train_steps here refers to training start step
    logsnr_s = logsnr_schedule_fn(i.astype(dtype) / num_steps)
    # z_s_dist = self.predict(
    #     z_t=z_t,
    #     logsnr_t=jnp.full((shape[0],), logsnr_t),
    #     logsnr_s=jnp.full((shape[0],), logsnr_s),
    #     clip_x=clip_x)

    # predicted_x = z_s_dist['pred_x']
    # # sample z ~ q(z_logsnr | x)
    # z_dist = diffusion_forward(
    #     x=predicted_x,
    #     logsnr=utils.broadcast_from_left(logsnr_s, predicted_x.shape))
    # # eps = jax.random.normal(next(rng), shape=predicted_x.shape, dtype=predicted_x.dtype)
    # eps = jax.random.normal(
    #     jax.random.fold_in(rng, i), shape=shape, dtype=dtype)
    # # z = z_dist['mean'] + z_dist['std'] * eps
    # return jnp.where(
    #     i == 0, z_s_dist['pred_x'], z_dist['mean'] + z_dist['std'] * eps)
    model_out = self._run_model(
        z=z_t,
        logsnr=jnp.full((shape[0],), logsnr_t),
        model_fn=self.model_fn,
        clip_x=clip_x)
    x_pred_t = model_out['model_x']
    eps_pred_t = model_out['model_eps']
    eps_random = jax.random.normal(
        jax.random.fold_in(rng, i), shape=shape, dtype=dtype)
    stdv_s = jnp.sqrt(nn.sigmoid(-logsnr_s))
    alpha_s = jnp.sqrt(nn.sigmoid(logsnr_s))
    # TODO: need to change to sqrt(interpolaton), sqrt(1-interpolation)
    z_s_pred = alpha_s * x_pred_t + stdv_s * (
        jnp.sqrt(1. - interpolation) * eps_pred_t +
        jnp.sqrt(interpolation) * eps_random)
    return jnp.where(i == 0, x_pred_t, z_s_pred)

  # NOTE NVIDIA sampler
  def ddim_noisy_step(self, rng, i, z_t, num_steps, logsnr_schedule_fn,
                      clip_x):
    shape, dtype = z_t.shape, z_t.dtype
    logsnr_t = logsnr_schedule_fn((i + 1.).astype(dtype) / num_steps)
    logsnr_s = logsnr_schedule_fn(i.astype(dtype) / num_steps)

    logsnr_teacher_mid = logsnr_schedule_fn(
        i.astype(dtype) / num_steps + 0.5 / num_steps)

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

    stdv_mid = jnp.sqrt(nn.sigmoid(-logsnr_teacher_mid))
    alpha_mid = jnp.sqrt(nn.sigmoid(logsnr_teacher_mid))
    eps_random = jax.random.normal(
        jax.random.fold_in(rng, i), shape=shape, dtype=dtype)
    z_middle_noisy = alpha_mid * z_s_pred / alpha_s + jnp.sqrt(
        jnp.maximum(stdv_mid**2 - (stdv_s * alpha_mid / alpha_s)**2,
                    stdv_s * 0.)) * eps_random

    # z_middle_noisy = alpha_mid * x_pred_t + stdv_mid * eps_pred_t

    teacher_model_out = self._run_model(
        z=z_middle_noisy,
        logsnr=jnp.full((shape[0],), logsnr_teacher_mid),
        model_fn=self.conditional_target_model_fn,
        clip_x=clip_x)
    teacher_x_pred_t = teacher_model_out['model_x']
    teacher_eps_pred_t = teacher_model_out['model_eps']
    teacher_z_s_pred = alpha_s * teacher_x_pred_t + stdv_s * teacher_eps_pred_t

    return jnp.where(i == 0, teacher_x_pred_t, teacher_z_s_pred)
    # return jnp.where(i == 0, x_pred_t, z_s_pred)

  # NOTE todo
  def new_two_nvidia_step(self, rng, i, z_t, num_steps, logsnr_schedule_fn,
                          clip_x):
    shape, dtype = z_t.shape, z_t.dtype
    logsnr_t = logsnr_schedule_fn((i + 1.).astype(dtype) / num_steps)
    # TODO: add max (0, i-1) in the next line
    logsnr_s = logsnr_schedule_fn(
        jnp.maximum((i - 1), i * 0.).astype(dtype) / num_steps)
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

    # the next step (TODO)
    # if i > 1:
    logsnr_teacher_mid = logsnr_schedule_fn(i.astype(dtype) / num_steps)
    stdv_mid = jnp.sqrt(nn.sigmoid(-logsnr_teacher_mid))
    alpha_mid = jnp.sqrt(nn.sigmoid(logsnr_teacher_mid))
    eps_random = jax.random.normal(
        jax.random.fold_in(rng, i), shape=shape, dtype=dtype)
    z_middle_noisy = alpha_mid * z_s_pred / alpha_s + jnp.sqrt(
        jnp.maximum(stdv_mid**2 - (stdv_s * alpha_mid / alpha_s)**2,
                    stdv_s * 0.)) * eps_random

    teacher_model_out = self._run_model(
        z=z_middle_noisy,
        logsnr=jnp.full((shape[0],), logsnr_teacher_mid),
        model_fn=self.model_fn,
        clip_x=clip_x)
    teacher_x_pred_t = teacher_model_out['model_x']
    teacher_eps_pred_t = teacher_model_out['model_eps']
    teacher_z_s_pred = alpha_mid * teacher_x_pred_t + stdv_mid * teacher_eps_pred_t

    return jnp.where(i == 0, x_pred_t, teacher_z_s_pred)

  def sample_loop(self,
                  *,
                  rng,
                  init_x,
                  num_steps,
                  logsnr_schedule_fn,
                  sampler,
                  clip_x,
                  interpolation=0.):
    if sampler == 'ddim':
      body_fun = lambda i, z_t: self.ddim_step(i, z_t, num_steps,
                                               logsnr_schedule_fn, clip_x)
    elif sampler == 'two_student_ddim':
      body_fun = lambda i, z_t: self.two_student_ddim_step(
          i, z_t, num_steps, logsnr_schedule_fn, clip_x)
    elif sampler == 'noisy':
      body_fun = lambda i, z_t: self.bwd_dif_step(rng, i, z_t, num_steps,
                                                  logsnr_schedule_fn, clip_x)
    elif sampler == 'new_noisy':
      body_fun = lambda i, z_t: self.added_noisy_step(
          rng, i, z_t, num_steps, logsnr_schedule_fn, clip_x, interpolation)
    elif sampler == 'ddim_noisy':
      body_fun = lambda i, z_t: self.ddim_noisy_step(rng, i, z_t, num_steps,
                                                     logsnr_schedule_fn, clip_x)
    # encoding, not sampler
    elif sampler == 'ddim_encoder':
      body_fun = lambda i, z_t: self.ddim_encoding_step(
          i, z_t, num_steps, logsnr_schedule_fn, clip_x)
    elif sampler == 'new_two_step_nvidia':
      body_fun = lambda i, z_t: self.new_two_nvidia_step(
          rng, i, z_t, num_steps, logsnr_schedule_fn, clip_x)
    else:
      raise NotImplementedError(sampler)

    # loop over t = num_steps-1, ..., 0
    if sampler == 'ddim_encoder':
      # final eps, not sample x
      # TODO fix random seed
      # eps = jax.random.normal(jax.random.fold_in(rng, 100000), shape=init_x.shape, dtype=jnp.float32)
      final_x = jax.lax.fori_loop(
          lower=0, upper=num_steps, body_fun=body_fun, init_val=init_x)
    else:
      final_x = utils.reverse_fori_loop(
          lower=0, upper=num_steps, body_fun=body_fun, init_val=init_x)

    assert final_x.shape == init_x.shape and final_x.dtype == init_x.dtype
    return final_x
