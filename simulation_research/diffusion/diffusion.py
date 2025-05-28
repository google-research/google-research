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

"""Core diffusion model implementation, noise schedule, type, and training."""
import time
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

from absl import logging
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
import jax
from jax import grad
from jax import jit
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

Scorefn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
PRNGKey = jnp.ndarray
TimeType = Union[float, jnp.ndarray]
ArrayShape = Sequence[int]
ParamType = Any


def unsqueeze_like(x,
                   *objs):
  """Append additional axes to each obj in objs for each extra axis in x.

  Example: x of shape (bs,n,c) and s,t both of shape (bs,),
  sp,tp = unsqueeze_like(x,s,t) has sp and tp of shape (bs,1,1)

  Args:
    x: ndarray with shape that to unsqueeze like
    *objs: ndarrays to unsqueeze to that shape

  Returns:
    unsqueeze_objs: unsqueezed versions of *objs
  """
  if len(objs) != 1:
    return [unsqueeze_like(x, obj) for obj in objs]  # pytype: disable=bad-return-type  # jax-ndarray
  elif hasattr(objs[0], 'shape') and len(objs[0].shape):  # broadcast to x shape
    return objs[0][(Ellipsis,) + len(x.shape[1:]) * (None,)]
  else:
    return objs[0]  # if it is a scalar, it already broadcasts to x shape  # pytype: disable=bad-return-type  # jax-ndarray


class StructuredCovariance(object):
  """Abstract base class for noise covariance matrices defined implicitly.

  The class is organized as a bijector, mapping white noise to structured
  noise, with forward, inverse, and logdet methods just like the interface
  of a normalizing flow.
  (StructuredCovariance is an instance of a rudimentary normalizing flow.)
  """

  @classmethod
  def forward(cls, v):
    """Multiplies the input vector v by Sigma^{1/2}."""
    raise NotImplementedError

  @classmethod
  def inverse(cls, v):
    """Multiplies the input vector v by Sigma^{-1/2}."""
    raise NotImplementedError

  @classmethod
  def logdet(cls, shape):
    """Computes the log determinant logdet(Sigma^{1/2})."""
    raise NotImplementedError

  @classmethod
  def sample(cls, key, shape):
    """Sample the structured noise by compting Sigma^{1/2}z."""
    return cls.forward(jax.random.normal(key, shape))


class Identity(StructuredCovariance):
  """Identity covariance matrix (equivalent to white noise)."""

  @classmethod
  def forward(cls, v):
    return v

  @classmethod
  def inverse(cls, v):
    return v

  @classmethod
  def logdet(cls, shape):
    return jnp.zeros(shape[0])


class FourierCovariance(StructuredCovariance):
  """Base class for covariance matrices which are diagonal in Fourier domain.

  Subclasses must implement spectrum(f) classmethod (of Sigma^{1/2})
  """

  @classmethod
  def spectrum(cls, f):
    """The spectrum (eigenvalues) of the Fourier covariance of Sigma^{1/2}."""
    raise NotImplementedError

  @classmethod
  def forward(cls, v, invert = False):
    """Maps v -> Sigma^{1/2}v.

    Args:
      v: of shape (b,n,c) or (b,h,w,c).
      invert: whether to use inverse transformation

    Returns:
      Sigma^{1/2}v
    """
    assert all(k % 2 == 0
               for k in v.shape[1:-1]), 'requires even lengths for fft for now'
    f = jnp.sqrt(
        sum(jnp.meshgrid(*[jnp.fft.rfftfreq(k)**2 for k in v.shape[1:-1]])))

    scaling = cls.spectrum(f)
    assert scaling.shape == f.shape, 'cls.spectrum should output same shape'

    if invert:
      scaling = 1 / scaling
    if len(v.shape) == 3:
      scaled_fft_v = jnp.fft.rfft(v, axis=1) * scaling[None, :, None]
      return jnp.fft.irfft(scaled_fft_v, axis=1)
    elif len(v.shape) == 4:
      scaled_fft_v = jnp.fft.rfft2(v, axes=(1, 2)) * scaling[None, :, :, None]
      return jnp.fft.irfft2(scaled_fft_v, axes=(1, 2))
    else:
      raise NotImplementedError

  @classmethod
  def inverse(cls, v):
    """Maps v -> Sigma^{-1/2}v.

    Args:
      v: of shape (b,n,c) or (b,h,w,c).

    Returns:
      Sigma^{-1/2}v
    """
    return cls.forward(v, invert=True)

  @classmethod
  def logdet(cls, shape):
    """Assumes input shape is (b,n,c) or (b,h,w,c) for 2d."""
    f = jnp.sqrt(
        sum(jnp.meshgrid(*[jnp.fft.fftfreq(k)**2 for k in shape[1:-1]])))
    return jnp.log(cls.spectrum(f)).sum() * shape[-1] + jnp.zeros(shape[0])


class WhiteCovariance(FourierCovariance):
  """White Noise Covariance matrix, equivalent to Identity."""
  multiplier: float = 1.

  @classmethod
  def spectrum(cls, f):
    return jnp.ones_like(f) * cls.multiplier


class BrownianCovariance(FourierCovariance):
  """Brown Noise Covariance matrix: (1/f) spectral noise."""
  multiplier: float = 30.  # Tuned scaling to use same scale as Identity

  @classmethod
  def spectrum(cls, f):
    scaling = jnp.where(f == 0, jnp.ones_like(f), 1. / f)
    scaling = scaling / jnp.max(scaling)
    return jnp.where(f == 0, jnp.ones_like(f), scaling) * cls.multiplier


class PinkCovariance(FourierCovariance):
  """Pink Noise Covariance matrix: 1/sqrt(f) spectral noise."""
  multiplier: float = 1.  # Tuned scaling to use same scale as Identity

  @classmethod
  def spectrum(cls, f):
    scaling = jnp.where(f == 0, jnp.ones_like(f), 1 / jnp.sqrt(f))
    scaling = scaling / jnp.max(scaling)
    return jnp.where(f == 0, jnp.ones_like(f), scaling) * cls.multiplier


class Diffusion(object):
  """Abstract class for diffusion types.

    Subclasses must implement sigma(t) and scale(t)
  """
  tmin = 1e-4
  tmax = 1.

  def __init__(self, covariance = Identity):
    super().__init__()
    self.covsqrt = covariance

  @classmethod
  def sigma(cls, t):
    """Noise schedule."""
    raise NotImplementedError

  @classmethod
  def scale(cls, t):
    """Scale schedule."""
    raise NotImplementedError

  @classmethod
  def f(cls, t):
    """Internal f func from https://arxiv.org/abs/2011.13456."""
    return grad(lambda s: jnp.log(cls.scale(s)))(t)

  @classmethod
  def g2(cls, t):
    """Internal g^2 func from https://arxiv.org/abs/2011.13456."""
    dsigma2 = grad(lambda s: cls.sigma(s)**2)(t)
    return dsigma2 - 2 * cls.f(t) * cls.sigma(t)**2

  @classmethod
  def dynamics(cls, score_fn, x, t):
    """Backwards probability flow ODE dynamics."""
    return cls.f(t) * x - .5 * cls.g2(t) * score_fn(x, t)

  @classmethod
  def drift(cls, score_fn, x, t):
    """Backwards SDE drift term."""
    return cls.f(t) * x - cls.g2(t) * score_fn(x, t)

  @classmethod
  def diffusion(cls, score_fn, x, t):  # pylint: disable=unused-argument
    """Backwards SDE diffusion term (independent of score_fn)."""
    return jnp.sqrt(cls.g2(t))

  @classmethod
  def noise_score(cls, xt, x0, t):
    r"""Actually the score times the cov matrix. `\Sigma\nabla\logp(xt)`."""
    s, sig = unsqueeze_like(x0, cls.scale(t), cls.sigma(t))
    return -(xt - s * x0) / sig**2

  def noise_input(self, x, t, key):
    """Apply the noise at scale sigma(t) and with covariance to the input."""
    s, sig = unsqueeze_like(x, self.scale(t), self.sigma(t))
    return s * x + sig * self.noise(key, x.shape)

  def noise(self, key, shape):
    """Sample from the structured noise covariance (without scale sigma(t))."""
    return self.covsqrt.sample(key, shape)


class VarianceExploding(Diffusion):
  """Variance exploding variant of Score-SDE diffusion models."""
  tmin = 1e-3  # smallest time to integrate to

  @classmethod
  def sigma(cls, t):
    sigma_max = 300
    sigma_min = 1e-3  # 1e-6#1e-3
    return sigma_min * jnp.sqrt((sigma_max / sigma_min)**(2 * t) - 1)

  @classmethod
  def scale(cls, t):
    return jnp.ones_like(t)


def int_b(t):
  """Integral b(t) for Variance preserving noise schedule."""
  bm = .1
  bd = 20
  return bm * t + (bd - bm) * t**2 / 2


class VariancePreserving(Diffusion):
  tmin = 1e-4

  @classmethod
  def sigma(cls, t):
    return jnp.sqrt(1 - jnp.exp(-int_b(t)))

  @classmethod
  def scale(cls, t):
    return jnp.exp(-int_b(t) / 2)


class SubVariancePreserving(Diffusion):
  tmin = 1e-4

  @classmethod
  def sigma(cls, t):
    return 1 - jnp.exp(-int_b(t))

  @classmethod
  def scale(cls, t):
    return jnp.exp(-int_b(t) / 2)


def nonefn(x):  # pylint: disable=unused-argument
  return None


def train_diffusion(
    model,
    dataloader,
    data_std,
    epochs = 100,
    lr = 1e-3,
    diffusion = VarianceExploding(),
    cond_fn = nonefn,  # function: array -> array or None
    num_ema_foldings = 5,
    writer = None,
    report = None,
    ckpt = None,
    seed = None,  # to avoid initing jax
):
  """Train diffusion model with score matching according to diffusion type.

  Minimizes score matching MSE loss between the model scores s(xₜ,t)
  and the data scores ∇log p(xₜ|x₀) over noised datapoints xₜ, with t sampled
  uniformly from 0 to 1, and x sampled from the training distribution.
  Produces score function s(xₜ,t) ≈ ∇log p(xₜ) which can be used for sampling.

  Loss = 𝔼[|s(xₜ,t) − ∇log p(xₜ|x₀)|²σₜ²]

  Args:
    model: UNet mapping (x,t,train,cond) -> x'
    dataloader: callable which produces an iterator for the minibatches
    data_std: standard deviation of training data for input normalization
    epochs: number of epochs to train
    lr: learning rate
    diffusion: diffusion object (VarianceExploding, VariancePreserving, etc)
    cond_fn: (optional) function cond_fn(x) to condition training on
    num_ema_foldings: number of ema timescales per total number of epochs
    writer: optional summary_writer to log to if not None
    report: optional report function to call if not None
    ckpt: optional clu.checkpoint to save the model. If None, does not save
    seed: random seed for model init and training

  Returns:
    score function (xt,t,cond)->scores (s(xₜ,t):=∇logp(xₜ))
  """
  # initialize model
  x = next(dataloader())
  t = np.random.rand(x.shape[0])
  key = random.PRNGKey(42) if seed is None else seed
  key, init_seed = random.split(key)
  params = model.init(init_seed, x=x, t=t, train=False, cond=cond_fn(x))
  logging.info(f"{count_params(params['params'])/1e6:.2f}M Params")  # pylint: disable=logging-fstring-interpolation

  def score(params,
            x,
            t,
            train = True,
            cond = None):
    """Score function with appropriate input and output scaling."""
    # scaling is equivalent to that in https://arxiv.org/abs/2206.00364
    sigma, scale = unsqueeze_like(x, diffusion.sigma(t), diffusion.scale(t))
    input_scale = 1 / jnp.sqrt(sigma**2 + (scale * data_std)**2)
    cond = cond / data_std if cond is not None else None
    out = model.apply(params, x=x * input_scale, t=t, train=train, cond=cond)
    return out / jnp.sqrt(sigma**2 + scale**2 * data_std**2)

  def loss(params, x, key):
    """Score matching MSE loss from Yang's Score-SDE paper."""
    key1, key2 = jax.random.split(key)
    u0 = jax.random.uniform(key1)
    # Use lowvar grid time sampling from https://arxiv.org/pdf/2107.00630.pdf
    u = jnp.remainder(u0 + jnp.linspace(0, 1, x.shape[0]), 1)
    t = u * (diffusion.tmax - diffusion.tmin) + diffusion.tmin
    xt = diffusion.noise_input(x, t, key2)
    target_score = diffusion.noise_score(xt, x, t)
    # weighting from Yang Song's https://arxiv.org/abs/2011.13456
    weighting = unsqueeze_like(x, diffusion.sigma(t)**2)
    error = score(params, xt, t, cond=cond_fn(x)) - target_score
    return jnp.mean((diffusion.covsqrt.inverse(error)**2) * weighting)

  tx = optax.adam(learning_rate=lr)
  opt_state = tx.init(params)
  ema_ts = epochs / num_ema_foldings  # number of ema timescales during training
  ema_params = params
  jloss = jit(loss)
  loss_grad_fn = jax.value_and_grad(loss)

  @jit
  def update_fn(params, ema_params, opt_state, key, data):
    loss_val, grads = loss_grad_fn(params, data, key)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    key, _ = random.split(key)
    ema_update = lambda p, ema: ema + (p - ema) / ema_ts
    ema_params = jax.tree.map(ema_update, params, ema_params)
    return params, ema_params, opt_state, key, loss_val

  for epoch in tqdm(range(epochs + 1)):
    for data in dataloader():
      params, ema_params, opt_state, key, loss_val = update_fn(
          params, ema_params, opt_state, key, data)
    if epoch % 25 == 0:
      ema_loss = jloss(ema_params, data, key)  # pylint: disable=undefined-loop-variable
      message = f'Loss epoch {epoch}: {loss_val:.3f} Ema {ema_loss:.3f}'
      logging.info(message)
      if writer is not None:
        metrics = {'loss': loss_val, 'ema_loss': ema_loss}
        eval_metrics_cpu = jax.tree.map(np.array, metrics)
        writer.write_scalars(epoch, eval_metrics_cpu)
        report(epoch, time.time())

  model_state = ema_params
  if ckpt is not None:
    ckpt.save(model_state)

  @jit
  def score_out(x,
                t,
                cond = None):
    """Trained score function s(xₜ,t):=∇logp(xₜ)."""
    if not hasattr(t, 'shape') or not t.shape:
      t = jnp.ones(x.shape[0]) * t
    return score(ema_params, x, t, train=False, cond=cond)

  return score_out


def count_params(params):
  """Count the number of parameters in the flax model param dict."""
  if isinstance(params, jax.numpy.ndarray):
    return np.prod(params.shape)
  elif isinstance(params, (dict, FrozenDict)):
    return sum([count_params(v) for v in params.values()])
  else:
    assert False, type(params)
