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

"""Configurations and hyperparameters for training and testing."""

import getpass

import ml_collections


def get_config():
  """Helper to generate the ml_collections with hyperparameters and configs."""
  config = ml_collections.ConfigDict()

  # Training settings
  config.cell = ''
  config.user = getpass.getuser()
  config.run_name = 'runs/X'
  config.log_dir = '/tmp/robust_optim/{}'.format(config.run_name)
  config.seed = 0  # Random key generator seed
  config.log_interval = 10  # Interval of logging statistics to INFO stream
  # Keys to print in log printing
  config.log_keys = ('risk/train/zero_one', 'risk/train/adv/linf')

  # Dataset
  config.dataset = 'logistic'  # Training and test dataset
  config.dim = 10  # Input dimension
  config.num_train = 100  # Number of training samples
  config.num_test = 1000  # Number of test samples
  config.r = 1.  # The norm of the weight vector of the dataset
  config.temperature = 1.  # Label noise temperature

  # Optimizer
  config.optim = {
      'name': 'gd',  # Optimization method
      'lr': 0.1,  # Learning rate
      'niters': 2000,  # Number of training iterations
      'norm': 'l2',  # Weight norm in min-norm objective solved with cvxpy
      'bound_step': False,  # Bound the line search step by 1/B^2L(wt)
      'step_size': 1000.,  # Initial step size for line search
      # Adversarial attacks: train config
      'adv_train': {
          'enable': False,  # Train with adversarial examples
          'norm_type': 'linf',  # Norm-ball type
          'eps_iter': 0.1,  # Maximum norm of perturbation per iteration
          'eps_tot': 0.3,  # Maximum norm of total perturbation
          'lr': 0.1,  # Learning rate in projected gradient descent
          'niters': 1,  # Number of iterations in projected gradient descent
          'pre_normalize': False,  # Normalize weights before attack
          'step_dir': 'sign_grad',  # Direction of optimization step
      }
  }

  # Model
  config.model = {
      'arch': 'linear',  # Model name
      'nlayers': 1,  # Number of layers in a deep network
      'regularizer': 'none',  # Model regularization
      'reg_coeff': 1.,  # Regularizer coefficient
      'r': 1.,  # The norm of fixed weights e.g. w0 in two_linear_fixed_w0
  }

  # Adversarial attacks: test config
  config.adv = {
      'norm_type': 'linf',  # Norm-ball type
      'eps_iter': 0.1,  # Maximum norm of perturbation per iteration
      'eps_tot': 0.3,  # Maximum norm of total perturbation
      'lr': 0.1,  # Learning rate in projected gradient descent
      'niters': 10,  # Number of iterations in projected gradient descent
      'pre_normalize': False,  # Normalize weights before attack
      'step_dir': 'sign_grad',  # Direction of optimization step
      'eps_from_cvxpy': False,  # Use the min-norm solution to determine eps
  }

  config.available_norm_types = ('linf', 'l2', 'l1', 'l4', 'l1.5', 'dft1')

  config.enable_cvxpy = False  # Print distance to the minimum norm solution

  return config
