"""Soft-Actor-Critic agent.

This is a cleanup of [1]. For the original algorithm, see [2].

References:
    [1]: https://github.com/denisyarats/pytorch_sac
    [2]: https://arxiv.org/abs/1801.01290
"""

import math
import typing

import numpy as np

import ml_collections
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn

from .replay_buffer import ReplayBuffer

TensorType = torch.Tensor
InfoType = typing.Dict[str, TensorType]


def orthogonal_init(m: nn.Module) -> None:
  """Orthogonal init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, "data"):
      m.bias.data.fill_(0.0)


def mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int,
    output_mod: typing.Optional[nn.Module] = None,
) -> nn.Sequential:
  """Construct an MLP module."""
  if hidden_depth == 0:
    mods = [nn.Linear(input_dim, output_dim)]
  else:
    mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
      mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    mods.append(nn.Linear(hidden_dim, output_dim))
  if output_mod is not None:
    mods.append(output_mod)
  trunk = nn.Sequential(*mods)
  return trunk


class Critic(nn.Module):

  def __init__(
      self,
      obs_dim: int,
      action_dim: int,
      hidden_dim: int,
      hidden_depth: int,
  ) -> None:
    super().__init__()

    self.model = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
    self.apply(orthogonal_init)

  def forward(self, obs: TensorType, action: TensorType) -> TensorType:
    assert obs.size(0) == action.size(0)
    obs_action = torch.cat([obs, action], dim=-1)
    return self.model(obs_action)


class DoubleCritic(nn.Module):

  def __init__(
      self,
      obs_dim: int,
      action_dim: int,
      hidden_dim: int,
      hidden_depth: int,
  ) -> None:
    super().__init__()

    self.critic1 = Critic(obs_dim, action_dim, hidden_dim, hidden_depth)
    self.critic2 = Critic(obs_dim, action_dim, hidden_dim, hidden_depth)

  def forward(self, *args) -> typing.Tuple[TensorType, TensorType]:
    return self.critic1(*args), self.critic2(*args)


class TanhTransform(pyd.transforms.Transform):
  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size: int = 1) -> None:
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x: TensorType) -> TensorType:
    return 0.5 * (x.log1p() - (-x).log1p())

  @staticmethod
  def log_abs_det_jacobian(x: TensorType, y: TensorType) -> TensorType:
    # We use a formula that is more numerically stable, see details in the
    # following link https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    del y
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

  def __eq__(self, other):
    return isinstance(other, TanhTransform)

  def _call(self, x: TensorType) -> TensorType:
    return x.tanh()

  def _inverse(self, y: TensorType) -> TensorType:
    # We do not clamp to the boundary here as it may degrade the performance
    # of certain algorithms. One should use `cache_size=1` instead.
    return self.atanh(y)


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):

  def __init__(self, loc: TensorType, scale: TensorType) -> None:
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale)
    transforms = [TanhTransform()]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self) -> TensorType:
    mu = self.loc
    for tr in self.transforms:
      mu = tr(mu)
    return mu


class DiagGaussianActor(nn.Module):
  """A torch.distributions implementation of a diagonal Gaussian policy."""

  def __init__(
      self,
      obs_dim: int,
      action_dim: int,
      hidden_dim: int,
      hidden_depth: int,
      log_std_bounds: typing.Tuple[float, float],
  ) -> None:
    super().__init__()

    self.log_std_bounds = log_std_bounds
    self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)

    self.apply(orthogonal_init)

  def forward(self, obs: TensorType) -> TensorType:
    mu, log_std = self.trunk(obs).chunk(2, dim=-1)

    # Constrain log_std inside [log_std_min, log_std_max].
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std_range = log_std_max - log_std_min
    log_std = log_std_min + 0.5 * log_std_range * (log_std + 1)

    std = log_std.exp()
    return SquashedNormal(mu, std)


def soft_update_params(
    net: nn.Module,
    target_net: nn.Module,
    tau: float,
) -> None:
  for param, target_param in zip(net.parameters(), target_net.parameters()):
    val = tau * param.data + (1 - tau) * target_param.data
    target_param.data.copy_(val)


class SAC(nn.Module):
  """Soft-Actor-Critic."""

  def __init__(
      self,
      device: torch.device,
      config: ml_collections.ConfigDict,
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
        torch.tensor(np.log(config.init_temperature)).to(self.device),
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

  def train(self, training: bool = True) -> None:
    self.training = training
    self.actor.train(training)
    self.critic.train(training)

  @property
  def alpha(self) -> TensorType:
    return self.log_alpha.exp()

  @torch.no_grad()
  def act(self, obs: np.ndarray, sample: bool = False) -> np.ndarray:
    obs = torch.as_tensor(obs, device=self.device)
    dist = self.actor(obs.unsqueeze(0))
    action = dist.sample() if sample else dist.mean
    action = action.clamp(*self.action_range)
    return action.cpu().numpy()[0]

  def update_critic(
      self,
      obs: np.ndarray,
      action: np.ndarray,
      reward: float,
      next_obs: np.ndarray,
      mask: float,
  ) -> InfoType:
    dist = self.actor(next_obs)
    next_action = dist.rsample()
    log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
    target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
    target_V = (
        torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob)
    target_Q = reward + (mask * self.discount * target_V)
    target_Q = target_Q.detach()

    # Get current Q estimates.
    current_Q1, current_Q2 = self.critic(obs, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        current_Q2, target_Q)

    # Optimize the critic.
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    return {"critic_loss": critic_loss}

  def update_actor_and_alpha(
      self,
      obs: np.ndarray,
  ) -> typing.Tuple[InfoType, InfoType]:
    dist = self.actor(obs)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    actor_Q1, actor_Q2 = self.critic(obs, action)

    actor_Q = torch.min(actor_Q1, actor_Q2)
    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
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
      replay_buffer: ReplayBuffer,
      step: int,
  ) -> InfoType:
    obs, action, reward, next_obs, mask = replay_buffer.sample(self.batch_size)
    batch_info = {"batch_reward": reward.mean()}

    critic_info = self.update_critic(obs, action, reward, next_obs, mask)

    if step % self.actor_update_frequency == 0:
      actor_info, alpha_info = self.update_actor_and_alpha(obs)

    if step % self.critic_target_update_frequency == 0:
      soft_update_params(self.critic, self.critic_target, self.critic_tau)

    return {**batch_info, **critic_info, **actor_info, **alpha_info}

  def trainable_dict(self):
    return {
        "actor": self.actor,
        "log_alpha": self.log_alpha,
        "critic": self.critic,
        "critic_target": self.critic_target,
    }

  def optim_dict(self):
    return {
        "actor_optimizer": self.actor_optimizer,
        "log_alpha_optimizer": self.log_alpha_optimizer,
        "critic_optimizer": self.critic_optimizer,
    }
