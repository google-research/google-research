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

"""Library for generating datasets based on general ODEs and hamiltonian systems."""
from typing import Callable, Optional, Tuple

from jax import device_count
from jax import grad
from jax import jit
from jax import vmap
from jax.experimental import mesh_utils
from jax.experimental.ode import odeint
from jax.experimental.pjit import pjit
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
from tqdm.auto import tqdm

from simulation_research.diffusion.animations import Animation
from simulation_research.diffusion.animations import PendulumAnimation


class ODEDataset(object):
  """An (abstract) dataset that generates trajectory chunks from an ODE.

  For a given dynamical system and initial condition distribution,
  each element ds[i] = ((ic,T),z_target) where ic (state_dim,) are the
  initial conditions, T are the evaluation timepoints,
  and z_target (T,state_dim) is the ground truth trajectory chunk.
  To use, one must specify both the dynamics and the initial condition
  distribution for a subclass.

  Class attributes:
    animator: which animator to use for the given dataset
    burnin_time: amount of time to discard for burnin for the given dataset

  Attributes:
    Zs: state variables z for each trajectory, of shape (N, L, C)
    T_long: full integration timesteps, of shape (L, )
    T: the integration timesteps for chunk if chunk_len is specified, same as
      T_long if chunk_len not specified, otherwise of shape (chunk_len,)
  """

  animator = Animation  # associated object to produce an animation of traj
  burnin_time: float = 0  # amount of time to discard to further mix the ICs

  def __init__(
      self,
      N = 30,  # pylint: disable=invalid-name
      chunk_len = None,
      dt = 0.1,
      integration_time = 30):
    """Constructor for the ODE dataset.

    Args:
        N: total number of trajectory chunks that makeup the dataset.
        chunk_len: the number of timepoints at which each chunk is evaluated
        dt: the spacing of the evaluation points (not the integrator step size
          which is set by tol=1e-4)
        integration_time: The integration time for evaluation rollouts and also
          the total integration time from which each trajectory chunk is
          randomly sampled
    """
    super().__init__()
    self.Zs = self.generate_trajectory_data(N, dt, integration_time)  # pylint: disable=invalid-name
    T = np.asarray(jnp.arange(0, integration_time, dt))  # pylint: disable=invalid-name
    self.T = self.T_long = T[T >= self.burnin_time]  # pylint: disable=invalid-name
    if chunk_len is not None:
      self.Zs = np.asarray(self.chunk_training_data(self.Zs, chunk_len))  # pytype: disable=wrong-arg-types  # jax-ndarray
      self.T = np.asarray(jnp.arange(0, chunk_len * dt, dt))

  def __len__(self):
    return self.Zs.shape[0]

  def __getitem__(self,
                  i):
    return (self.Zs[i, 0], self.T), self.Zs[i]  # pytype: disable=bad-return-type  # jax-ndarray

  def integrate(self,
                z0s,
                ts,
                tol = 1e-4):
    dynamics = jit(self.dynamics)
    return odeint(dynamics, z0s, ts, rtol=tol)

  def generate_trajectory_data(self,
                               trajectories,
                               dt,
                               integration_time,
                               bs = 10):
    """Returns ts: (N, traj_len) zs: (N, traj_len, z_dim)."""
    n_gen = 0
    bs = min(bs, trajectories)
    z_batches = []
    mesh = Mesh(mesh_utils.create_device_mesh((device_count(),)), ('data',))
    integrate = jit(
        vmap(lambda z0, t: odeint(self.dynamics, z0, t, rtol=1e-6), (0, None),
             0))
    jintegrate = pjit(integrate, (P('data', None), None), P('data', None, None))
    # batched_dynamics = jit(vmap(self.dynamics, (0, None)))
    k = len(mesh.devices)
    with mesh:
      for _ in tqdm(range(0, trajectories, bs * k)):
        z0s = self.sample_initial_conditions(bs * k)
        ts = jnp.arange(0, integration_time, dt)
        new_zs = jintegrate(z0s, ts)
        new_zs = new_zs[:, ts >= self.burnin_time]
        z_batches.append(new_zs)
        n_gen += bs
    zs = jnp.concatenate(z_batches, axis=0)[:trajectories]
    return zs

  def chunk_training_data(self, zs, chunk_len):
    """Helper function to separate the generated trajectories into chunks."""
    batch_size, traj_len, *_ = zs.shape
    n_chunks = traj_len // chunk_len
    chunk_idx = np.random.randint(0, n_chunks, (batch_size,))
    chunked_zs = np.stack(np.split(zs, n_chunks, axis=1))
    chosen_zs = chunked_zs[chunk_idx, np.arange(batch_size)]
    return chosen_zs

  def sample_initial_conditions(self, bs):
    """Initial condition distribution."""
    raise NotImplementedError

  def dynamics(self, z, t):
    """Implements the dynamics dz/dt = F(z,t). z is shape (d,)."""
    raise NotImplementedError

  def animate(self, zt = None):
    """Visualize the dynamical system, or given input trajectories.

        Usage
              from IPython.display import HTML
              HTML(dataset.animate())
              or
              from matplotlib import rc
              rc('animation',html='jshmtl')
              dataset.animate()
    Args:
      zt: array of shape (n,d)

    Returns:
      the animation object
    """
    if zt is None:
      zt = np.asarray(
          self.integrate(self.sample_initial_conditions(10)[0], self.T_long))
    anim = self.animator(zt)
    return anim.animate()


class LorenzDataset(ODEDataset):
  """ODEDataset generated from the Lorenz equations with dynamics.

      dx/dt = sigma * (y - x)
      dy/dt = x * (rho - z) - y
      dz/dt = x * y - beta * z
      where we have chosen rho=28, sigma=10, beta=8/3
  """
  burnin_time: float = 3

  def __init__(self,
               *args,
               dt = 0.1,
               integration_time = 7 + 3,
               **kwargs):
    super().__init__(*args, dt=dt, integration_time=integration_time, **kwargs)

  def dynamics(self, z, t):
    scale = 20.  # rescale dynamics so values are in range ~ (-3,3)
    x = scale * z
    rho, sigma, beta = 28, 10, 8 / 3
    zdot = jnp.array([
        sigma * (x[1] - x[0]), x[0] *
        (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]
    ],
                     dtype=x.dtype)
    return zdot / scale

  def sample_initial_conditions(self, bs):
    return np.random.randn(bs, 3)


class FitzHughDataset(ODEDataset):
  """FitzHugh dynamics from https://arxiv.org/pdf/1803.06277.pdf."""
  burnin_time = 1500

  def __init__(self,
               *args,
               dt = 6.,
               integration_time = 4000,
               **kwargs):
    super().__init__(*args, dt=dt, integration_time=integration_time, **kwargs)

  def dynamics(self, z, t):
    z = z / 5.
    a = jnp.ones(2) * (-.025794)
    c = jnp.ones(2) * .02
    b = jnp.array([.0065, .0135])
    k = .128
    coupling = 1
    n = z.shape[0] // 2
    assert n == 2, 'System should have 4 components'
    xs = z[:n]
    ys = z[n:]
    xdot = xs * (a - xs) * (xs - 1) - ys + k * coupling * (xs[::-1] - xs)
    ydot = b * xs - c * ys
    return jnp.concatenate([xdot, ydot]) * 5.

  def sample_initial_conditions(self, bs):
    return np.random.randn(bs, 4) * .2


def unpack(z):
  D = jnp.shape(z)[-1]  # pylint: disable=invalid-name,unused-variable
  assert D % 2 == 0, 'unpack requires even dimension'
  d = D // 2
  q, p_or_v = z[Ellipsis, :d], z[Ellipsis, d:]
  return q, p_or_v


def pack(q, p_or_v):
  return jnp.concatenate([q, p_or_v], axis=-1)


def symplectic_form(z):
  """Equivalent to multiplying z by the matrix J=[[0,I],[-I,0]]."""
  q, p = unpack(z)
  return pack(p, -q)


def hamiltonian_dynamics(hamiltonian,
                         z):
  """Computes hamiltonian dynamics dz/dt=J∇H.

  Args:
    hamiltonian: function state->scalar
    z: state vector (concatenation of q and momentum p)

  Returns:
    dz/dt
  """
  grad_h = grad(hamiltonian)  # ∇H
  gh = grad_h(z)  # ∇H(z)
  return symplectic_form(gh)  # J∇H(z)


class HamiltonianDataset(ODEDataset):
  """ODEDataset with dynamics given by a Hamiltonian system.

  q denotes the generalized coordinates and p denotes the momentum
  Hamiltonian is used along with an associated mass matrix M,
  which is used to convert the momentum p to velocity v
  after generating the trajectories.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # convert the momentum into velocity
    qs, ps = unpack(self.Zs)
    Ms = vmap(vmap(self.mass))(qs)  # pylint: disable=invalid-name
    vs = jnp.linalg.solve(Ms, ps[Ellipsis, None]).squeeze(-1)
    self.Zs = pack(qs, vs)

  def dynamics(self, z, t):
    return hamiltonian_dynamics(self.hamiltonian, z)

  def hamiltonian(self, z):  # pylint: disable=invalid-name
    """The Hamiltonian function, depending on z=pack(q,p)."""
    raise NotImplementedError

  def mass(self, q):  # pylint: disable=invalid-name
    """Mass matrix used for Kinetic energy T=vTM(q)v/2."""
    raise NotImplementedError

  def animate(self, zt = None):  # type: ignore  # jax-ndarray
    if zt is None:
      zt = np.asarray(
          self.integrate(self.sample_initial_conditions(10)[0], self.T_long))
    # bs, T, 2nd
    if len(zt.shape) == 3:
      j = np.random.randint(zt.shape[0])
      zt = zt[j]
    xt, _ = unpack(zt)
    anim = self.animator(xt)
    return anim.animate()


class SHO(HamiltonianDataset):
  """A basic simple harmonic oscillator."""

  def hamiltonian(self, z):
    ke = (z[Ellipsis, 1]**2).sum() / 2
    pe = (z[Ellipsis, 0]**2).sum() / 2
    return ke + pe

  def mass(self, q):
    return jnp.eye(1)

  def sample_initial_conditions(self, bs):  # pytype: disable=signature-mismatch  # jax-ndarray
    return np.random.randn(bs, 2)


class NPendulum(HamiltonianDataset):
  """An n-link (chaotic) pendulum.

  The generalized coordinates q are the angles (in radians) with respect to
  the vertical down orientation measured counterclockwise. ps are the
  conjugate momenta p = M(q)dq/dt.
  Mass matrix M(q) and Hamiltonian derived in https://arxiv.org/abs/2010.13581,
  page 20.
  """

  animator = PendulumAnimation

  def __init__(self, *args, n = 2, dt = .5, **kwargs):
    """NPendulum constructor.

    Uses additional arguments over base class.

    Args:
      *args: ODEDataset args
      n: number of pendulum links
      dt: timestep size (not for the integrator, but for the final subsampling)
      **kwargs: ODEDataset kwargs
    """
    self.n = n
    super().__init__(*args, dt=dt, **kwargs)

  def mass(self, q):
    # assume all ls are 1 and ms are 1
    ii = jnp.tile(jnp.arange(self.n), (self.n, 1))
    m = jnp.maximum(ii, ii.T)
    return jnp.cos(q[:, None] - q[None, :]) * (self.n - m + 1)

  def hamiltonian(self, z):
    """Energy H(q,p) = pTM(q)^-1p/2 + Sum(yi)."""
    q, p = unpack(z)
    kinetic = (p * jnp.linalg.solve(self.mass(q), p)).sum() / 2
    # assume all ls are 1 and ms are 1
    potential = -jnp.sum(jnp.cumsum(jnp.cos(q)))  # height of bobs
    return kinetic + potential

  def sample_initial_conditions(self, bs):  # pytype: disable=signature-mismatch  # jax-ndarray
    z0 = np.random.randn(bs, 2 * self.n)
    z0[:, self.n:] *= .2
    z0[:, -1] *= 1.5
    return z0
