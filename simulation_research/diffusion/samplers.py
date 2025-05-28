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

"""Methods to sample and compute likelihoods of diffusion models.

Sampling and NLL are defined within the SDE and ODE formulations.
"""

from functools import partial  # pylint: disable=g-importing-member
from typing import Callable, List, Optional, Tuple

import jax
from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp
import numpy as np
import scipy.integrate
from tqdm.auto import tqdm

from simulation_research.diffusion.diffusion import Diffusion
from simulation_research.diffusion.diffusion import unsqueeze_like

# Function (x,t) -> x' such as scorefn or dynamics
Fnxt2x = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
Scorefn = Fnxt2x
PRNGKey = jnp.ndarray
ArrayShape = Tuple[int, Ellipsis]


def heun_integrate_loop(
    dynamics, x0,
    ts):
  """Integrate dynamics using Heun integrator with for loop (and tqdm)."""
  x = x0
  xs = []
  for t1, t2 in tqdm(zip(ts[:-1], ts[1:])):
    xdot = dynamics(x, t1)
    xp = x + (t2 - t1) * xdot
    x = x + (t2 - t1) * (dynamics(xp, t2) + xdot) / 2
    xs.append(x)
  xf = x
  return xf, xs


def heun_integrate(dynamics, x0,
                   ts):
  """Integrate dynamics using Heun integrator with lax.scan."""
  t12 = jnp.stack([ts[:-1], ts[1:]], -1)

  @jit
  def update(x, tpair):
    t1, t2 = tpair
    xdot = dynamics(x, t1)
    xp = x + (t2 - t1) * xdot
    x = x + (t2 - t1) * (dynamics(xp, t2) + xdot) / 2
    return x, x

  xf, xs = jax.lax.scan(update, x0, t12)
  return xf, xs


def euler_maruyama_integrate(
    diff, scorefn, x0, ts,
    key):
  """Integrate diffusion SDE with euler maruyama integrator."""
  t12 = jnp.stack([ts[:-1], ts[1:]], -1)
  drift = partial(diff.drift, scorefn)
  diffusion = partial(diff.diffusion, scorefn)

  @jit
  def update(xkey, tpair):
    x, key = xkey
    t1, t2 = tpair
    key = random.split(key)[0]
    xdot = drift(x, t1)
    noise = diff.noise(key, x0.shape)
    x = x + (t2 - t1) * xdot + diffusion(x, t1) * noise * jnp.sqrt(t1 - t2)
    return (x, key), x

  (x, _), xs = jax.lax.scan(update, (x0, key), t12)
  return x, xs


def compute_nll(diffusion,
                score_fn,
                key,
                data,
                num_probes = 1):
  """Returns negative log likelihood per dimension for data samples (bs,*).

  Uses connection to continuous normalizing flows spelled out in Score-SDEs
  in order to compute likelihood. Jacobian logdet = ∫₀¹ (∇ ⋅ ẋ) dt.
  Uses stochastic trace estimator for divergence calculation like in FFJORD

  Args:
    diffusion: diffusion type
    score_fn: score_fn that defines the probability
    key: rngkey to use for probe samples
    data: x data (bs,...,c) to compute model likelihood of
    num_probes: number of stochastic trace estimator probes, for a large dataset
      1 is enough, however for a single datapoint many may be needed.

  Returns:
    Negative log likelihood per dimension of data (bs,)
  """
  bs = data.shape[0]
  flat_data = data.reshape(bs, -1)
  # N(0,1) probe variables
  zs = jax.random.normal(key, (num_probes,) + (np.prod(flat_data.shape),))
  q0 = jnp.concatenate(
      [data.reshape(-1), jnp.zeros((bs,), data.dtype)], axis=-1)

  def get_xdot_div(x, t, z):  # pylint: disable=invalid-name
    """ẋ and divergence ∇⋅ẋ estimated using a single probe z."""

    def dynamics(x):
      return diffusion.dynamics(score_fn, x.reshape(*data.shape), t).reshape(-1)

    xdot, Jz = jax.jvp(dynamics, [x], [z])  # pylint: disable=invalid-name
    trace_est = (z * Jz).reshape(bs, -1).sum(-1)  # z^TDFz
    return xdot, trace_est

  @jit
  def prob_flow(q, t):
    x = q[:zs.shape[-1]]
    xdots, trace_ests = vmap(get_xdot_div, (None, None, 0), (None, 0))(x, t, zs)
    trace = jnp.mean(trace_ests, axis=0)
    return jnp.concatenate([xdots, trace], axis=-1)

  qf = odeint(
      prob_flow, q0, jnp.array([diffusion.tmin, diffusion.tmax]), rtol=1e-3)[-1]
  xf, logdet_dxf_dx0 = qf[:zs.shape[-1]].reshape(bs, -1), qf[zs.shape[-1]:]
  inv_sqrt_cov_xf = diffusion.covsqrt.inverse(xf.reshape(data.shape)).reshape(
      bs, -1)
  # Logdet correction from spatial covariance matrix
  inv_sqrt_logdet = -diffusion.covsqrt.logdet(data.shape)
  std_max = diffusion.sigma(diffusion.tmax)
  print(r'xf std {inv_sqrt_cov_xf.std()} and std_max: {std_max}')
  logpxf = -(inv_sqrt_cov_xf**2 / std_max**2 + jnp.log(
      2 * np.pi * std_max**2)).sum(-1) / 2  # N(0,smax^2) log likelihood
  logpx0 = (logpxf + logdet_dxf_dx0 + inv_sqrt_logdet)
  return -logpx0 / flat_data.shape[-1]


def probability_flow(diffusion, scores, x0,
                     t0, tf):
  """Integrates the probability flow ODE from t0 to tf (w/ adaptive solver)."""
  with tqdm(total=1000, desc='ODE Solve') as pbar:

    def dynamics_with_tqdm(t, x):
      delta = max(int(1000 * (jnp.abs((t - t0) / (tf - t0))) - pbar.n), 0)
      pbar.update(delta)
      return diffusion.dynamics(scores, x.reshape(*x0.shape), t).reshape(-1)

    return scipy.integrate.solve_ivp(
        dynamics_with_tqdm, (t0, tf), x0.reshape(-1), method='RK23',
        rtol=1e-3).y[:, -1].reshape(x0.shape)


def ode_sample(diffusion, scorefn, key,
               x_shape):
  """Sample deterministically from the model using the ODE (w/ adaptive solver)."""
  tmax = diffusion.tmax
  xf = diffusion.noise(key, x_shape) * diffusion.sigma(tmax)
  tmin = diffusion.tmin
  return probability_flow(diffusion, scorefn, xf, tmax, tmin)


def discrete_ode_sample(diffusion,
                        scorefn,
                        key,
                        x_shape,
                        nsteps = 500,
                        traj = False):
  """Sample deterministically from model using the ODE (w/o adaptive solver).

  Args:
    diffusion: diffusion type
    scorefn: function (x,t) -> x'
    key: PRNGKey for sampling
    x_shape: shape of stacked x (with batch size, e.g. (bs,n,c))
    nsteps: number of discrete timesteps to use in ODE solver
    traj: whether or not to return entire trajectory

  Returns:
    sample x0 if traj=False or [x0,x1,...] if traj=True
  """
  tmax = diffusion.tmax
  xf = diffusion.noise(key, x_shape) * diffusion.sigma(tmax)
  dynamics = jit(partial(diffusion.dynamics, scorefn))
  timesteps = (.5 + np.arange(nsteps)[::-1]) / nsteps
  x0, xs = heun_integrate(dynamics, xf, timesteps)
  return xs if traj else x0  # pytype: disable=bad-return-type  # jax-ndarray


def sde_sample(diffusion,
               scorefn,
               key,
               x_shape,
               nsteps = 500,
               traj = False):
  """Sample using SDE and Euler-Maruyama integrator.

  Args:
    diffusion: diffusion type
    scorefn: function (x,t) -> x'
    key: PRNGKey for sampling
    x_shape: shape of stacked x (with batch size, e.g. (bs,n,c))
    nsteps: number of discrete timesteps to use in ODE solver
    traj: whether or not to return entire trajectory

  Returns:
    sample x0 if traj=False or [x0,x1,...] if traj=True
  """
  timesteps = (.5 + np.arange(nsteps)[::-1]) / nsteps
  key0, key1 = random.split(key)
  xf = diffusion.noise(key0, x_shape) * diffusion.sigma(diffusion.tmax)
  samples, xt = euler_maruyama_integrate(diffusion, scorefn, xf, timesteps,
                                         key1)
  return xt if traj else samples  # pytype: disable=bad-return-type  # jax-ndarray


def inpainting_scores(diff,
                      scorefn,
                      observed_values,
                      slc,
                      alpha = 10):
  r"""Get conditional scores ∇logp(xₜ|xᵃ) of the diffusion model.

  Produces scores of entire trajectory conditioned on the values xᵃ which
  are the values of the trajectory at the given slice x[:,slc]. The alpha
  parameter controls the conditioning strength and should be tuned.

  Args:
    diff: diffusion type
    scorefn: ∇logp(xₜ) function (x,t) -> x'
    observed_values: the observed values xᵃ at the slice
    slc: the slice object determining where x is observed
    alpha: the conditioning strength. Should be tuned for best performance.

  Returns:
    Conditional scores ∇logp(xₜ|xᵃ)
  """
  b, _, c = observed_values.shape  # pylint: disable=invalid-name

  def conditioned_scores(xt, t):
    unflat_xt = xt.reshape(b, -1, c)

    observed_score = diff.noise_score(unflat_xt[:, slc], observed_values, t)
    unobserved_score = scorefn(xt, t).reshape(b, -1, c)

    def constraint(xt):
      one_step_xhat = (xt + diff.sigma(t)**2 * scorefn(xt, t)) / diff.scale(t)
      sliced = one_step_xhat.reshape(b, -1, c)[:, slc]
      return jnp.sum((sliced - observed_values)**2)

    scale = alpha * diff.scale(t)**2 / diff.sigma(t)**2
    unobserved_score -= scale * grad(constraint)(xt).reshape(unflat_xt.shape)
    combined_score = unobserved_score.at[:, slc].set(observed_score)
    return combined_score

  return jit(conditioned_scores)


def event_scores(diffusion,
                 scorefn,
                 constraint,
                 reg = 1e-3):
  """Model scores ∇log p(xₜ|E) conditioned on inequality constraint E=[C(x)>0]."""

  def xhat(xt, t):
    tt = unsqueeze_like(xt, t)
    score_xhat = (xt +
                  diffusion.sigma(tt)**2 * scorefn(xt, t)) / diffusion.scale(tt)
    return score_xhat

  def conditioned_scores(xt, t):
    b, _, c = xt.shape  # pylint: disable=invalid-name
    unobserved_score = scorefn(xt, t).reshape(b, -1, c)
    if not hasattr(t, 'shape') or not t.shape:
      array_t = t * jnp.ones(b)
    else:
      array_t = t

    def log_p(xt):  # log p(E|xt)
      xh = xhat(xt, array_t)
      C, DC = vmap(jax.value_and_grad(constraint))(xh)  # pylint: disable=invalid-name
      CoXhat = lambda x, t: constraint(xhat(x[None], t)[0])  # pylint: disable=invalid-name
      SigmaDC = vmap(jax.grad(CoXhat))(xt, array_t)  # pylint: disable=invalid-name
      # NOTE: will not work with img inputs
      std = ((DC * SigmaDC).sum((-1, -2)) * diffusion.scale(t))
      std = jnp.sqrt(jnp.abs(std) + reg) * (
          diffusion.sigma(t) / diffusion.scale(t))
      # use logit approximation of probit (guassian cdf)
      return jax.nn.log_sigmoid(1.6 * C / std).sum()

    unobserved_score += grad(log_p)(xt)
    return unobserved_score

  return jit(conditioned_scores)


def discrete_time_inverse(diffusion,
                          scorefn,
                          samples,
                          nsteps = 1000):
  """Integrate prob flow ODE from t=0 to t=1 using nsteps discrete timesteps."""
  timesteps = (.5 + jnp.arange(nsteps)) / nsteps
  z0, _ = heun_integrate(
      partial(diffusion.dynamics, scorefn), samples, timesteps)  # pylint: disable=bad-whitespace
  return z0


def discrete_time_likelihood(diffusion,
                             scorefn,
                             samples,
                             nsteps = 1000):
  """Compute likelihood logp(x) for samples x with discrete time ODE solver.

  Uses (non adaptive) Heun method and computes jacobian log determinant of the
  exact ODE steps, not of the continuous time process.
  Very slow when using the explicit logdet calculation
  May take 10 minutes for each data point, can only use small batch size ~5
  before running into memory constraints.
  Use this function over `compute_nll` when running on stiff dynamics
  encountered when conditioning on nonlinear or discrete events.

  Args:
    diffusion: diffusion process type
    scorefn: ∇logp(xₜ) function (x,t) -> x' (possibly conditional)
    samples: data x to compute the likelihood of, shape (bs,...,c)
    nsteps: number of integration steps to use for integrator

  Returns:
    Likelihoods logp(x) of shape (bs,)
  """
  F = partial(discrete_time_inverse, diffusion, scorefn, nsteps=nsteps)  # pylint: disable=bad-whitespace,invalid-name
  xf = F(samples)
  D = xf[0].size  # pylint: disable=invalid-name
  # TODO(finzi): replace with optional residual flows computation
  jacf = jit(vmap(jax.jacfwd(F)))
  jacobian = jacf(samples[:, None]).reshape(xf.shape[0], D, D)
  _, logdet_dxf_dx0 = jnp.linalg.slogdet(jacobian)
  inv_sqrt_cov_xf = diffusion.covsqrt.inverse(xf).reshape(xf.shape[0], -1)
  inv_sqrt_logdet = -diffusion.covsqrt.logdet(xf.shape)
  std_max = diffusion.sigma(diffusion.tmax)
  logpxf = -(inv_sqrt_cov_xf**2 / std_max**2 + jnp.log(
      2 * np.pi * std_max**2)).sum(-1) / 2  # N(0,smax^2) log likelihood
  logpx0 = (logpxf + logdet_dxf_dx0 + inv_sqrt_logdet)
  return logpx0


def marginal_logprob(diffusion,
                     scorefn,
                     constraint,
                     x_shape,
                     key = None,
                     num_samples = 5,
                     nsteps = 1000):
  """Compute the marginal logprob of event logP(E)=logP(x|E)-logP(x).

    Event E is defined as E=[C(x)>0]
    Quite slow, can take ~ 15 minutes to run.

  Args:
    diffusion: diffusion type (e.g. VarianceExploding, VariancePreserving, etc)
    scorefn: UNet scorefn mapping (x,t)->x'
    constraint: function C(x) specifying the event E=[C(x)>0]
    x_shape: shape of x excluding batch axis, e.g. (n,c) or (h,w,c)
    key: PRNGKey for generating the event conditioned samples x|E
    num_samples: number of event conditioned samples to generate and use
    nsteps: number of SDE integration steps to use for generation

  Returns:
    log P(E) estimate mean and standard error according to the model.
  """
  if key is None:
    key = random.PRNGKey(42)
  # sample event conditioned samples
  event_scorefn = event_scores(diffusion, scorefn, constraint)
  samples_shape = (num_samples,) + x_shape
  event_samples = sde_sample(
      diffusion, event_scorefn, key, samples_shape, nsteps=nsteps)
  print('Fraction of sampled which satisfy',
        (constraint(event_samples) > 0).mean())
  event_nats = []
  conditional_prob = jit(
      partial(
          discrete_time_likelihood, diffusion, event_scorefn, nsteps=nsteps))
  unconditional_prob = jit(
      partial(discrete_time_likelihood, diffusion, scorefn, nsteps=nsteps))
  for i in range(num_samples):
    # Need to for loop to not run out of memory
    xi = event_samples[i:i + 1]
    up = unconditional_prob(xi)
    cp = conditional_prob(xi)
    event_nats.append(up - cp)
  event_nats = jnp.concatenate(event_nats, axis=0)
  return event_nats.mean(), event_nats.std() / jnp.sqrt(num_samples)  # pytype: disable=bad-return-type  # jnp-type
