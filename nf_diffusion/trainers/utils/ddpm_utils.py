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

"""Helper utils for DDPM."""
import abc
import functools

import jax
import jax.numpy as jnp


# TODO(guandao) adding post process
class DDPMProcesses(abc.ABC):
  """DDPM Langevin dynamic sampler and corruption process."""

  def __init__(self, cfg, rng, model):
    super().__init__()
    self.cfg = cfg
    self.rng = rng
    self.model = model
    self.max_t = int(self.cfg.get("max_t", 1000))
    self.jit_ldstep = self.cfg.get("jit_ldstep", True)

    # NOTE: this is discrete time version.
    self.beta_t_vect = jnp.array(
        [beta_t_fn(t + 1, self.max_t) for t in range(self.max_t)])
    self.sigma_t_vect = jnp.array(
        [sigma_t_fn(t + 1, self.max_t) for t in range(self.max_t)])
    self.alpha_t_vect = jnp.array(
        [alpha_t_fn(t + 1, self.max_t) for t in range(self.max_t)])
    self.alpha_t_bar_vect = jnp.cumprod(self.alpha_t_vect)
    # jnp.array(
    #     [alpha_t_bar_fn(t + 1) for t in range(self.max_t)])
    self.verbose = self.cfg.get("verbose", False)
    self.ld_step = None
    self.params = None

  def langevin_step(self, x, t, is_t1, rng, params, w=0., guided_grad_fn=None,
                    add_noise=True):
    """Langevin dynamic step."""
    z = jax.random.normal(rng, x.shape)
    sigma_t = self.sigma_t_vect[t] * is_t1

    alpha_t = self.alpha_t_vect[t]
    alpha_t_bar = self.alpha_t_bar_vect[t]

    rng, rng_d = jax.random.split(rng)
    t = t.reshape(x.shape[0])
    eps = self.model.apply(
        params,
        x,
        None,  # NOTE: cond = None
        t / float(self.max_t - 1),
        train=False,
        rngs={"dropout": rng_d})
    # Compute mu(x_t, t)
    mu_t = 1 / jnp.sqrt(alpha_t) * (
        x - (1 - alpha_t) / jnp.sqrt(1 - alpha_t_bar) * eps)

    if guided_grad_fn is not None:
      g = guided_grad_fn(x)
      mu_t = mu_t + g * sigma_t * w

    # x = 1 / jnp.sqrt(alpha_t) * (
    #     x - (1 - alpha_t) / jnp.sqrt(1 - alpha_t_bar) * eps)
    x = mu_t
    if add_noise:
      x = x + z * sigma_t
    return x

  def update_params(self, params, guided_grad_fn=None, guided_weight=0.):
    """Update parameter of the model: pls do this before running evaluation."""
    self.params = params
    # self.ld_step = jax.jit(functools.partial(
    #     self.langevin_step, params=self.params))
    self.ld_step = functools.partial(
        self.langevin_step, params=self.params,
        guided_grad_fn=guided_grad_fn, w=guided_weight)

    if self.jit_ldstep:
      self.ld_step = jax.jit(self.ld_step)

  # TODO(guandao) : make this learnable in the future.
  def forward_df(self, x_0, t, rng, return_diff=False):
    """Forward diffusion processing (i.e. adding noise)."""
    batch_size = x_0.shape[0]
    rng, key = jax.random.split(rng)
    eps = epsilon = jax.random.normal(key, shape=x_0.shape)

    alpha_t_bar_vect = self.alpha_t_bar_vect[t].reshape((batch_size, 1, 1, 1))
    x_t = x_0 * jnp.sqrt(alpha_t_bar_vect) + epsilon * jnp.sqrt(
        1 - alpha_t_bar_vect)

    if return_diff:  # otherwise return the original noise direction
      eps = x_t - x_0
    return x_t, eps

  def backward_df(self, x_t, t, rng_d, train=False, scaled=True, params=None):
    """Backward diffusion processing (i.e. denoising)."""
    if params is None:
      assert self.params is not None
      params = self.params
    assert t.shape[0] == 1 or t.shape[0] == x_t.shape[0]
    batch_size = x_t.shape[0]

    alpha_t_vect = self.alpha_t_vect[t].reshape((batch_size, 1, 1, 1))
    alpha_t_bar_vect = self.alpha_t_bar_vect[t].reshape((batch_size, 1, 1, 1))
    eps = self.model.apply(
        params,
        x_t,
        None,  # NOTE: cond = None
        t / float(self.max_t),
        train=train,
        rngs={"dropout": rng_d})

    if scaled:
      x_0 = 1 / jnp.sqrt(alpha_t_vect) * (
          x_t - (1 - alpha_t_vect) / jnp.sqrt(1 - alpha_t_bar_vect) * eps)
    else:
      x_0 = x_t + eps
    return x_0, eps

  def sample(self,
             input_shape=None,
             x_init=None,
             start_t=0,
             end_t=None,
             rng=None,
             params=None,
             guided_grad_fn=None,
             guided_weight=0.,
             verbose=False,
             tqdm=None):
    """Sampling an image with stored parameters."""
    if params is None:
      assert self.ld_step is not None
    else:
      self.update_params(params, guided_grad_fn=guided_grad_fn,
                         guided_weight=guided_weight)
    if rng is None:
      rng = self.rng

    x = x_init
    if x is None:
      assert input_shape is not None
      key, rng = jax.random.split(rng)
      x = jax.random.normal(key, input_shape)

    if end_t is None:
      end_t = self.max_t

    pbar = range(start_t, end_t)
    if verbose:
      # import tqdm.notebook as tqdm
      pbar = tqdm(pbar)
    for i in pbar:
      t = self.max_t - i
      t_vect = (jnp.ones((x.shape[0], 1, 1, 1)) * t).astype(int)
      rng, key = jax.random.split(rng)
      x = self.ld_step(
          x,
          t_vect,
          (jnp.ones_like(t_vect) if t > 1 else jnp.zeros_like(t_vect)),
          key,
      )

    return x


# TODO(guandao) : should allow changing the beta_t schedule :)
def beta_t_fn(step, max_t):
  ratio = (step - 1) / (max_t - 1)
  return 1e-4 * (1 - ratio) + 0.02 * ratio


def alpha_t_fn(step, max_t):
  return 1 - beta_t_fn(step, float(max_t))


def alpha_t_bar_fn(step, max_t):
  return jnp.prod(jnp.array([alpha_t_fn(s + 1, max_t) for s in range(step)]))


def sigma_t_fn(t, max_t, fix_x0 = False):
  # assert t > 1
  if fix_x0:
    # NOTE: according to DDPM, this is optimal for setting x_0 = a fixed point
    sig_t_2 = (1 - alpha_t_bar_fn(t - 1, max_t)) / (
        1 - alpha_t_bar_fn(t, max_t)) * beta_t_fn(t, max_t)
  else:
    # NOTE: according to DDPM, this is optimal for setting x_0 ~ N(0, I)
    sig_t_2 = beta_t_fn(t, float(max_t))
  return jnp.sqrt(sig_t_2)
