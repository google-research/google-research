# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Soft-Actor-Critic agent.

This is a cleanup of [1]. For the original algorithm, see [2].

References:
    [1]: https://github.com/denisyarats/pytorch_sac
    [2]: https://arxiv.org/abs/1801.01290
"""

import typing

import ml_collections
import numpy as np
from .replay_buffer import ReplayBuffer
import torch
from torch import distributions as pyd
from torch import nn
import torch.nn.functional as F

TensorType = torch.Tensor
InfoType = typing.Dict[str, TensorType]
TrainableType = typing.Union[nn.Parameter, nn.Module]


def orthogonal_init(m):
  """Orthogonal init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, "data"):
      m.bias.data.fill_(0.0)


def mlp(
    input_dim,
    hidden_dim,
    output_dim,
    hidden_depth,
    output_mod = None,
):
  """Construct an MLP module."""
  if hidden_depth == 0:
    mods = [nn.Linear(input_dim, output_dim)]
  else:
    mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
      mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    mods += [nn.Linear(hidden_dim, output_dim)]
  if output_mod is not None:
    mods += [output_mod]
  trunk = nn.Sequential(*mods)
  return trunk


class Critic(nn.Module):
  """Critic module."""

  def __init__(
      self,
      obs_dim,
      action_dim,
      hidden_dim,
      hidden_depth,
  ):
    super().__init__()

    self.model = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
    self.apply(orthogonal_init)

  def forward(self, obs, action):
    assert obs.size(0) == action.size(0)
    obs_action = torch.cat([obs, action], dim=-1)
    return self.model(obs_action)


class DoubleCritic(nn.Module):
  """DocubleCritic module."""

  def __init__(
      self,
      obs_dim,
      action_dim,
      hidden_dim,
      hidden_depth,
  ):
    super().__init__()

    self.critic1 = Critic(obs_dim, action_dim, hidden_dim, hidden_depth)
    self.critic2 = Critic(obs_dim, action_dim, hidden_dim, hidden_depth)

  def forward(self, *args):
    return self.critic1(*args), self.critic2(*args)


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
  """A tanh-squashed Normal distribution."""

  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale)
    transforms = [pyd.TanhTransform(cache_size=1)]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
      mu = tr(mu)
    return mu


class DiagGaussianActor(nn.Module):
  """A torch.distributions implementation of a diagonal Gaussian policy."""

  def __init__(
      self,
      obs_dim,
      action_dim,
      hidden_dim,
      hidden_depth,
      log_std_bounds,
  ):
    super().__init__()

    self.log_std_bounds = log_std_bounds
    self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)

    self.apply(orthogonal_init)

  def forward(self, obs):
    mu, log_std = self.trunk(obs).chunk(2, dim=-1)

    # Constrain log_std inside [log_std_min, log_std_max].
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std_range = log_std_max - log_std_min
    log_std = log_std_min + 0.5 * log_std_range * (log_std + 1)

    std = log_std.exp()
    return SquashedNormal(mu, std)


def soft_update_params(
    net,
    target_net,
    tau,
):
  for param, target_param in zip(net.parameters(), target_net.parameters()):
    val = tau * param.data + (1 - tau) * target_param.data
    target_param.data.copy_(val)


class SAC(nn.Module):
  """Soft-Actor-Critic."""

  def __init__(
      self,
      device,
      config,
  ):
    super().__init__()

    self.device = device
    self.config = config

    self.action_range = config.action_range
    self.discount = config.discount
    self.critic_tau = config.critic_tau
    self.actor_update_frequency = config.actor_update_frequency
    self.critic_target_update_frequency = (
        config.critic_target_update_frequency)
    self.batch_size = config.batch_size
    self.learnable_temperature = config.learnable_temperature

    self.critic = DoubleCritic(
        config.critic.obs_dim,
        config.critic.action_dim,
        config.critic.hidden_dim,
        config.critic.hidden_depth,
    ).to(self.device)
    self.critic_target = DoubleCritic(
        config.critic.obs_dim,
        config.critic.action_dim,
        config.critic.hidden_dim,
        config.critic.hidden_depth,
    ).to(self.device)
    self.critic_target.load_state_dict(self.critic.state_dict())

    self.actor = DiagGaussianActor(
        config.actor.obs_dim,
        config.actor.action_dim,
        config.actor.hidden_dim,
        config.actor.hidden_depth,
        config.actor.log_std_bounds,
    ).to(self.device)

    self.log_alpha = nn.Parameter(
        torch.as_tensor(np.log(config.init_temperature), device=self.device),
        requires_grad=True,
    )

    # Set target entropy to -|A|.
    self.target_entropy = -config.critic.action_dim

    # Optimizers.
    self.actor_optimizer = torch.optim.Adam(
        self.actor.parameters(),
        lr=config.actor_lr,
        betas=config.actor_betas,
    )
    self.critic_optimizer = torch.optim.Adam(
        self.critic.parameters(),
        lr=config.critic_lr,
        betas=config.critic_betas,
    )
    self.log_alpha_optimizer = torch.optim.Adam(
        [self.log_alpha],
        lr=config.alpha_lr,
        betas=config.alpha_betas,
    )

    self.train()
    self.critic_target.train()

  def train(self, training = True):
    self.training = training
    self.actor.train(training)
    self.critic.train(training)

  @property
  def alpha(self):
    return self.log_alpha.exp()

  @torch.no_grad()
  def act(self, obs, sample = False):
    obs = torch.as_tensor(obs, device=self.device)
    dist = self.actor(obs.unsqueeze(0))
    action = dist.sample() if sample else dist.mean
    action = action.clamp(*self.action_range)
    return action.cpu().numpy()[0]

  def update_critic(
      self,
      obs,
      action,
      reward,
      next_obs,
      mask,
  ):
    with torch.no_grad():
      dist = self.actor(next_obs)
      next_action = dist.rsample()
      log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
      target_q1, target_q2 = self.critic_target(next_obs, next_action)
      target_v = (
          torch.min(target_q1, target_q2) - self.alpha.detach() * log_prob)
      target_q = reward + (mask * self.discount * target_v)

    # Get current Q estimates.
    current_q1, current_q2 = self.critic(obs, action)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
        current_q2, target_q)

    # Optimize the critic.
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    return {"critic_loss": critic_loss}

  def update_actor_and_alpha(
      self,
      obs,
  ):
    dist = self.actor(obs)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    actor_q1, actor_q2 = self.critic(obs, action)

    actor_q = torch.min(actor_q1, actor_q2)
    actor_loss = (self.alpha.detach() * log_prob - actor_q).mean()
    actor_info = {
        "actor_loss": actor_loss,
        "entropy": -log_prob.mean(),
    }

    # Optimize the actor.
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Optimize the temperature.
    alpha_info = {}
    if self.learnable_temperature:
      self.log_alpha_optimizer.zero_grad()
      alpha_loss = (self.alpha *
                    (-log_prob - self.target_entropy).detach()).mean()
      alpha_loss.backward()
      self.log_alpha_optimizer.step()
      alpha_info["temperature_loss"] = alpha_loss
      alpha_info["temperature"] = self.alpha

    return actor_info, alpha_info

  def update(
      self,
      replay_buffer,
      step,
  ):
    obs, action, reward, next_obs, mask = replay_buffer.sample(self.batch_size)
    batch_info = {"batch_reward": reward.mean()}

    critic_info = self.update_critic(obs, action, reward, next_obs, mask)

    if step % self.actor_update_frequency == 0:
      actor_info, alpha_info = self.update_actor_and_alpha(obs)

    if step % self.critic_target_update_frequency == 0:
      soft_update_params(self.critic, self.critic_target, self.critic_tau)

    return {**batch_info, **critic_info, **actor_info, **alpha_info}

  def optim_dict(self):
    return {
        "actor_optimizer": self.actor_optimizer,
        "log_alpha_optimizer": self.log_alpha_optimizer,
        "critic_optimizer": self.critic_optimizer,
    }
