# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Script for running continouous control experiments with ES/PES.
"""
import os
import sys
import ipdb
import copy
import time
import random
import argparse
import numpy as onp
import pickle as pkl

import jax
import jax.numpy as jnp

import gym
import mujoco_py
import haiku as hk

from logger import CSVLogger

available_envs = ['Swimmer-v2', 'Reacher-v1', 'Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Humanoid-v2']

parser = argparse.ArgumentParser(description='Using ES/PES for MuJoCo control tasks')
parser.add_argument('--iterations', type=int, default=1000,
                    help='How many gradient steps to perform')
parser.add_argument('--env_name', type=str, default='Swimmer-v2', choices=available_envs,
                    help='MuJoCo environment name')
parser.add_argument('--estimate', type=str, default='es', choices=['es', 'pes'],
                    help='Which gradient estimate to use')
parser.add_argument('--horizon', type=int, default=1000,
                    help='Total training horizon for an episode')
parser.add_argument('--K', type=int, default=1000,
                    help='Unroll length (-1 means use full unrolls)')
parser.add_argument('--lr', type=float, default=0.015,
                    help='Learning rate')
parser.add_argument('--N', type=int, default=40,
                    help='Number of particles for ES/PES')
parser.add_argument('--b', type=int, default=40,
                    help='Keep top-b best directions')
parser.add_argument('--normalize_state', action='store_true', default=False,
                    help='Whether to normalize states or not')
parser.add_argument('--clip_rewards', action='store_true', default=False,
                    help='Whether to clip the rewards to be in [-1, 1]')
parser.add_argument('--divide_by_variance', action='store_true', default=False,
                    help='Whether to divide the gradient by the variance of the rewards')
parser.add_argument('--noise', type=float, default=0.025,
                    help='Perturbation scale for ES/PES')
parser.add_argument('--shift', type=float, default=0.0,
                    help='Choose the shift amount for certain environments (like Humanoid)')
parser.add_argument('--log_every', type=int, default=1,
                    help='Log every T iterations')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('--save_dir', type=str, default='saves/control-mujoco',
                    help='Save directory')
args = parser.parse_args()

random.seed(args.seed)
onp.random.seed(args.seed)

exp_name = '{}-{}-lr:{}-sigma:{}-N:{}-T:{}-K:{}-n:{}-c:{}-d:{}'.format(
            args.estimate, args.env_name, args.lr, args.noise, args.N, args.horizon, args.K,
            int(args.normalize_state), int(args.clip_rewards), int(args.divide_by_variance))

save_dir = os.path.join(args.save_dir, exp_name, 'seed_{}'.format(args.seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

iteration_logger = CSVLogger(fieldnames=['time', 'iteration', 'total_steps', 'reward', 'theta_grad_norm'],
                             filename=os.path.join(save_dir, 'iteration.csv'))

total_count = 0

class Normalizer():
    def __init__(self, num_inputs):
        self.n = onp.zeros(num_inputs)
        self.mean = onp.zeros(num_inputs)
        self.mean_diff = onp.zeros(num_inputs)
        self.var = onp.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = onp.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


def get_action(state, params):
    return onp.dot(params, state)


def unroll(params, state, env, t, K, T, training=True, shift=0.0):
    global total_count

    reset = False
    total_reward = 0
    for i in range(K):
        if training:
            total_count += 1

        if args.normalize_state:
            if training:
                normalizer.observe(state)
            state = normalizer.normalize(state)

        action = get_action(state, params)
        state, reward, done, info = env.step(onp.array(action))
        total_reward += (reward - shift)
        t += 1
        if done or (t >= T):
            reset = True
            break
    return total_reward, state, env, reset, t


class ParticlePairPES(object):
  def __init__(self, key, sigma, T, K):
    self.key = key
    self.sigma = sigma
    self.T = T
    self.K = K

    self.envs = []
    for i in range(2):
      env = gym.make(args.env_name)  # Only needed here to get the observation_space shape below
      self.envs.append(env)

    self.observation_space_size = self.envs[0].observation_space.shape[0]
    self.action_space_size = self.envs[0].action_space.shape[0]
    self.reset()

  def reset(self):
    self.key, key_for_np_seed = jax.random.split(self.key)

    self.ts = []
    self.states = []
    np_seed = jax.random.randint(key_for_np_seed, (), 0, 1e6)
    for env in self.envs:
      env.seed(int(np_seed))
      state = env.reset()
      self.states.append(state)
      self.ts.append(0)

    self.pert_accums = jnp.zeros((2, self.action_space_size, self.observation_space_size))

  def compute_gradient(self, theta):
    self.key, skey = jax.random.split(self.key)
    pos_pert = jax.random.normal(skey, (1, *theta.shape)) * self.sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])

    ts = []
    resets = []
    new_envs = []
    new_states = []
    objectives = []
    for i in range(perts.shape[0]):
      objective, new_state, new_env, reset, t = unroll(theta + perts[i], self.states[i], self.envs[i], self.ts[i], self.K, self.T, shift=args.shift)
      ts.append(t)
      resets.append(reset)
      new_envs.append(new_env)
      new_states.append(new_state)
      objectives.append(objective)

    objective = jnp.stack(objectives)
    self.ts = ts
    self.envs = new_envs
    self.states = new_states

    self.pert_accums += perts
    gradient_estimate = jnp.mean(objective.reshape(-1, 1, 1) * self.pert_accums, axis=0) / (self.sigma**2)
    pert_accums_to_return = jnp.array(self.pert_accums)

    if any(resets):
      self.reset()

    return gradient_estimate, objective, pert_accums_to_return


class ParticlePairES(object):
  def __init__(self, key, sigma, T, K):
    self.key = key
    self.sigma = sigma
    self.T = T
    self.K = K

    self.env = gym.make(args.env_name)  # Only needed here to get the observation_space shape below
    self.observation_space_size = self.env.observation_space.shape
    self.reset()

  def reset(self):
    self.key, key_for_np_seed = jax.random.split(self.key)
    np_seed = jax.random.randint(key_for_np_seed, (), 0, 1e6)
    self.env.seed(int(np_seed))
    self.state = self.env.reset()
    self.t = 0

  def compute_gradient(self, theta):
    self.key, skey = jax.random.split(self.key)
    pos_pert = jax.random.normal(skey, (1, *theta.shape)) * self.sigma
    neg_pert = -pos_pert
    perts = jnp.concatenate([pos_pert, neg_pert])

    resets = []
    objectives = []
    for i in range(perts.shape[0]):
      env_copy = copy.deepcopy(self.env)
      objective, new_state, new_env, reset, _ = unroll(theta + perts[i], self.state, env_copy, self.t, self.K, self.T, shift=args.shift)
      resets.append(reset)
      objectives.append(objective)

    objective = jnp.stack(objectives)

    if (self.K < self.T) and (not any(resets)):
        _, self.state, self.env, _, self.t = unroll(theta, self.state, self.env, self.t, self.K, self.T, shift=args.shift)  # Compute new state/env using mean theta

    gradient_estimate = jnp.mean(objective.reshape(-1, 1, 1) * perts, axis=0) / (self.sigma**2)

    if any(resets):
      self.reset()

    return gradient_estimate, objective, perts


class MultiParticleEstimator(object):
  def __init__(self, key, num_pairs, sigma, T, K, estimate_type):
    pair_keys = jax.random.split(key, num_pairs)

    self.pairs = []
    for pair_key in pair_keys:
      if estimate_type == 'pes':
        self.pairs.append(ParticlePairPES(pair_key, sigma=sigma, T=T, K=K))
      elif estimate_type == 'es':
        self.pairs.append(ParticlePairES(pair_key, sigma=sigma, T=T, K=K))

  def compute_gradient(self, theta):
    pair_gradients = []
    perturbations = []
    objectives = []

    for pair in self.pairs:
      gradient, objective, perts = pair.compute_gradient(theta)
      pair_gradients.append(gradient)
      # Here we can aggregate all the objective values and compute the variance, so we can scale our update like in the ARS paper
      objectives.append(objective)
      perturbations.append(perts)

    objectives = jnp.concatenate(objectives)
    perturbations = jnp.concatenate(perturbations)
    sigma_r = objectives.std()

    if args.divide_by_variance:
        gradient_estimate = jnp.sum(objectives.reshape(-1, 1, 1) * (perturbations / args.noise), axis=0) / (sigma_r * args.N)
    else:
        gradient_estimate = jnp.sum(objectives.reshape(-1, 1, 1) * (perturbations / args.noise), axis=0) / (args.N)

    return gradient_estimate


env = gym.make(args.env_name)
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

key = jax.random.PRNGKey(args.seed)
key, key_for_params = jax.random.split(key)
theta = jnp.zeros((output_size, input_size))

normalizer = Normalizer(input_size)

# Using SGD as the outer optimizer because the ARS paper used vanilla SGD rather than
# Adam, and made a point about how this is simpler to understand/fewer hyperparameters
# than using Adam as the Sailimans ES paper from OpenAI did.
outer_optimizer = 'sgd'
if outer_optimizer == 'sgd':
    outer_optim_params = {
        'lr': args.lr,
    }

    @jax.jit
    def outer_optim_step(params, grads, optim_params):
        lr = optim_params['lr']
        updated_params = params - lr * grads
        return updated_params, optim_params


estimator = MultiParticleEstimator(key,
                                   num_pairs=args.N,
                                   sigma=args.noise,
                                   T=args.horizon,
                                   K=args.K,
                                   estimate_type=args.estimate)

elapsed_time = 0.0
start_time = time.time()

for iteration in range(args.iterations):
  theta_grad = estimator.compute_gradient(theta)
  theta, outer_optim_params = outer_optim_step(theta, -theta_grad, outer_optim_params)

  if iteration % args.log_every == 0:
      elapsed_time += time.time() - start_time
      fresh_env = gym.make(args.env_name)
      fresh_state = fresh_env.reset()
      total_reward, _, _, _, _ = unroll(theta, fresh_state, fresh_env, 0, args.horizon, args.horizon, training=False, shift=0.0)
      print('time: {} | i: {} | theta_grad_norm: {:6.4f} | total_reward: {}'.format(
             elapsed_time, iteration, jnp.linalg.norm(theta_grad), total_reward))
      sys.stdout.flush()

      iteration_logger.writerow({ 'time': elapsed_time,
                                  'iteration': iteration,
                                  'total_steps': total_count,
                                  'reward': total_reward,
                                  'theta_grad_norm': jnp.linalg.norm(theta_grad),
                                })
      start_time = time.time()
