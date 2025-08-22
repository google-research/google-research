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

"""NDP architecture."""

import diffrax
import flax.linen as nn
import jax
import jax.numpy as jnp


from hct.common import model_blocks
from hct.common import typing
from hct.common import utils


class NDPEncoder(nn.Module):
  """Encoder module for NDP.

  Implements the following maps:
    image, hf_obs --> (zs, g, W),
    where:
      zs: image-embedding,
      g: NDP-ODE goal vector.
      W: NDP-ODE weights matrix (flattened).
  """
  action_dim: int
  zs_dim: int = 64
  zs_width: int = 128
  num_basis_fncs: int = 4
  activation: typing.ActivationFunction = nn.relu

  def setup(self):

    self.image_map = model_blocks.ResNetBatchNorm(
        embed_dim=self.zs_dim, width=self.zs_width
    )

    self.goal_map = model_blocks.MLP([self.action_dim*2, self.action_dim],
                                     activate_final=False,
                                     activation=self.activation)

    num_weights = self.num_basis_fncs * self.action_dim
    self.weights_map = model_blocks.MLP([num_weights, num_weights],
                                        activate_final=False,
                                        activation=self.activation)

  def __call__(
      self,
      images,
      hf_obs,
      train = False):
    assert len(hf_obs.shape) == 2
    assert hf_obs.shape[0] == images.shape[0]

    zs = self.image_map(images, train)
    state_embedding = jnp.concatenate((self.activation(zs), hf_obs), axis=1)

    goals = self.goal_map(state_embedding)
    weights = self.weights_map(state_embedding)

    return zs, goals, weights


class NDPDecoder(nn.Module):
  """NDP Decoder.

  This models the map:
    zs, hf_obs --> u(0), u_dot(0),
    where:
      zs: image-embedding,
      u(0), u_dot(0): initial control & time-derivative for NDP-ODE.
  """
  action_dim: int
  zo_dim: int = 32
  activation: typing.ActivationFunction = nn.relu

  def setup(self):
    self.fusion_map = model_blocks.MLP([2*self.zo_dim]*3+[self.zo_dim],
                                       activate_final=True,
                                       activation=self.activation)
    out_dim = self.action_dim * 2
    self.out_map = model_blocks.MLP([out_dim*8, out_dim*4, out_dim*2, out_dim],
                                    activate_final=False,
                                    activation=self.activation)

  def __call__(self, zs, hf_obs):

    state_embedding = jnp.concatenate((self.activation(zs), hf_obs), axis=1)
    zo = self.fusion_map(state_embedding)
    zo_x = jnp.concatenate((zo, hf_obs), axis=1)
    return self.out_map(zo_x)


class NDP(nn.Module):
  """Full NDP architecture."""

  action_dim: int  # dimension of action space
  num_actions: int  # number of actions between two successive observations

  # Loss fnc
  loss_fnc: typing.LossFunction  # loss between true and predicted action

  activation: typing.ActivationFunction = nn.relu

  # Low-freq encoder
  zs_dim: int = 64  # image embedding dimension
  zs_width: int = 128  # width of image encoder network

  # Decoder network
  zo_dim: int = 32  # width of decoder network

  # NDP-ODE hyperparameters
  num_basis_fncs: int = 4
  alpha_p: float = 1.0
  alpha_u: float = 10.0
  beta: float = 2.5

  # Integrator
  ode_solver: diffrax.AbstractSolver = diffrax.Tsit5()
  ode_solver_dt: float = 1e-2
  adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()

  def setup(self):
    # Setup low-frequency encoder
    self.encoder = NDPEncoder(self.action_dim, self.zs_dim, self.zs_width,
                              self.num_basis_fncs, self.activation)

    # Setup decoder network.
    self.decoder = NDPDecoder(self.action_dim, self.zo_dim, self.activation)

    # Setup ODE params.
    rbf_centers = jnp.exp(
        -self.alpha_p * jnp.linspace(0, 1, self.num_basis_fncs))
    rbf_h = self.num_basis_fncs / rbf_centers

    def ndp_forcer(p, rbf_weights, goal, u0):
      psi = jnp.exp(-rbf_h * (p - rbf_centers)**2)
      weights_mat = jnp.reshape(rbf_weights, (self.action_dim,
                                              self.num_basis_fncs))
      return (1. / jnp.sum(psi)) * (weights_mat @ psi) * p * (goal - u0)
    self.ndp_forcer = ndp_forcer

    # Setup integration parameters
    self.step_delta = 1 / self.num_actions
    assert self.ode_solver_dt <= self.step_delta

    self.sample_times = jnp.arange(self.num_actions) * self.step_delta
    def interp_control(tau, u_samples):
      return jax.vmap(jnp.interp, in_axes=(None, None, 1))(
          tau, self.sample_times, u_samples)
    self.interp_control = interp_control

  def encode(self, images, hf_obs, train = False):
    """Encode observations (image & hf state) into latent variables.

    Args:
      images: (batch_size, height, width, channel)-ndarray.
      hf_obs: (batch_size, x_dim)-ndarray, current hf state observation.
      train: boolean for batchnorm

    Returns:
      zs: (batch_size, zs_dim): image embedding
      goals: (batch_size, action_dim): NDP goals.
      weights: (batch_size, num_weights): NDP weights.
    """
    return self.encoder(images, hf_obs, train)

  def decode(self, zs, hf_obs):
    """Decode image embedding & hf state into initial conditions for NDP-ODE.

    Args:
      zs: (batch_size, zs_dim): image embedding
      hf_obs: (batch_size, x_dim): current hf state observation.

    Returns:
      ndp_init: (batch_size, 2*action_dim): initial conditions for NDP ODE.
    """
    return self.decoder(zs, hf_obs)

  def __call__(self, images, hf_obs):
    """Main call function - computes NDP Flow."""
    return self.compute_ndp_flow(images, hf_obs, self.sample_times)

  def compute_ndp_flow(self, images, hf_obs,
                       pred_times):
    """Compute the NDP solution at the desired prediction times.

    Args:
      images: (batch_size, ....): images
      hf_obs: (batch_size, x_dim): concurrent hf observations
      pred_times: (num_times,): prediction times, starting at 0.

    Returns:
      u_pred: (batch_size, num_times, u_dim): predicted sequence of actions.
    """
    assert jnp.max(pred_times) <= 1.

    batch_size = images.shape[0]
    zs, goals, rbf_weights = self.encoder(images, hf_obs, train=False)
    init_u_udot = self.decoder(zs, hf_obs)
    u0s = init_u_udot[:, :self.action_dim]
    init_ps = jnp.ones((batch_size, 1))
    init_ndp_states = jnp.hstack((init_u_udot, init_ps))

    term = diffrax.ODETerm(self._ndp_ode)
    saveat = diffrax.SaveAt(ts=pred_times)

    def flow_one(init_state, weights, goal, u0):
      sol = diffrax.diffeqsolve(
          term, self.ode_solver, 0., pred_times[-1],
          self.ode_solver_dt, y0=init_state,
          args=(weights, goal, u0),
          adjoint=self.adjoint,
          saveat=saveat)
      return sol.ys[:, :self.action_dim]

    return jax.vmap(flow_one)(init_ndp_states, rbf_weights, goals, u0s)

  def compute_augmented_flow(self, images, hf_obs,
                             u_true, train = False):
    """Compute augmented flow for entire prediction period.

    Args:
      images: (batch_size, ....): images
      hf_obs: (batch_size, x_dim): concurrent hf observations
      u_true: (batch_size, num_actions, u_dim): observed control actions
      train: boolean for batchnorm.

    Returns:
      u_pred: (batch_size, num_actions, u_dim): predicted sequence of actions.
      net_loss: (batch_size,) integral of loss over the prediction period.
    """
    batch_size = images.shape[0]
    assert u_true.shape[1] == self.num_actions

    zs, goals, rbf_weights = self.encoder(images, hf_obs, train)
    init_u_udot = self.decoder(zs, hf_obs)
    u0s = init_u_udot[:, :self.action_dim]
    init_ps = jnp.ones((batch_size, 1))
    init_ndp_states = jnp.hstack((init_u_udot, init_ps))

    term = diffrax.ODETerm(self._aug_ode)
    saveat = diffrax.SaveAt(ts=self.sample_times)

    def flow_one(init_ndp_state, u_samples, weights, goal, u0):
      aug_state_0 = (init_ndp_state, 0.0)
      args = ((weights, goal, u0), u_samples)
      sol = diffrax.diffeqsolve(
          term, self.ode_solver, 0., self.sample_times[-1],
          self.ode_solver_dt, y0=aug_state_0,
          args=args,
          adjoint=self.adjoint,
          saveat=saveat)
      return sol.ys[0][:, :self.action_dim], sol.ys[1][-1]

    return jax.vmap(flow_one)(init_ndp_states, u_true, rbf_weights, goals, u0s)

  @nn.nowrap
  def _ndp_ode(self, tau, ndp_state, args):
    # Define the (unbatched) NDP ode over the state (u, u_dot, x).
    del tau
    rbf_weights, goal, u0 = args
    u, u_dot, p = jnp.split(ndp_state, [self.action_dim, 2*self.action_dim])
    force = self.ndp_forcer(p, rbf_weights, goal, u0)
    u_ddot = self.alpha_u*(self.beta * (goal - u) - u_dot) + force
    p_dot = -self.alpha_p * p
    return jnp.concatenate((u_dot, u_ddot, p_dot))

  def _step_ndp(self, ndp_state, tau, ndp_args):
    """Flow NDP forward by one prediction step; unbatched."""
    term = diffrax.ODETerm(self._ndp_ode)

    sol = diffrax.diffeqsolve(
        term, self.ode_solver, tau, tau + self.step_delta,
        self.ode_solver_dt, y0=ndp_state,
        args=ndp_args)
    return sol.ys[0], tau + self.step_delta

  @nn.nowrap
  def _aug_ode(self, tau, aug_state, args):
    # Define the (unbatched) NDP+cost ode over the state (ndp_state, cost).
    ndp_args, u_samples = args
    ndp_state, _ = aug_state
    ndp_out = self._ndp_ode(tau, ndp_state, ndp_args)
    u = ndp_state[:self.action_dim]
    # linearly interpolate true control samples
    u_true = self.interp_control(tau, u_samples)
    cost = self.loss_fnc(u_true, u)
    return ndp_out, cost

  @property
  def step_functions(self):
    """Return the 'low-frequency' and 'high-frequency' step functions."""

    def re_init(
        model_params, image, hf_obs
    ):
      """Re-initialize the NDP-ODE.

      Args:
        model_params: all params for the model.
        image: current image.
        hf_obs: concurrent hf observation.

      Returns:
        ndp_state: initialized NDP state (u(0), u_dot(0), phi(0)).
        ndp_args: ndp args (weights, goal, u0) for the NDP ODE.
      """
      zs, goal, weights = utils.unbatch_flax_fn(self.apply)(
          model_params, image, hf_obs, train=False, method=self.encode
      )
      ndp_state = utils.unbatch_flax_fn(self.apply)(
          model_params, zs, hf_obs, method=self.decode
      )
      u0 = ndp_state[: self.action_dim]
      return jnp.append(ndp_state, 1.0), (weights, goal, u0)

    def step_fwd(
        model_params,
        ndp_state,
        tau,
        ndp_args,
    ):
      """Flow NDP forward by one prediction step.

      Args:
        model_params: all params for the model.
        ndp_state: current NDP ODE state; (u(t), u_dot(t), phi(t)).
        tau: interpolation time, in between [0, 1 - (1/num_actions)].
        ndp_args: NDP-ODE args returned by re_init.

      Returns:
        ndp_state: ndp_state at (tau + step_delta).
        tau': tau + step_delta.
      """
      return self.apply(model_params, ndp_state, tau, ndp_args,
                        method=self._step_ndp)

    return re_init, step_fwd
