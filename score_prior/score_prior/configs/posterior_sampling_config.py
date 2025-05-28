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

"""DPI default config."""
# pylint:disable=line-too-long
import ml_collections


def get_config():
  """Returns the default hyperparameter configuration for RealNVP / Glow."""
  config = ml_collections.ConfigDict()

  config.model = model = ml_collections.ConfigDict()
  model.bijector = 'RealNVP'
  model.n_flow = 32  # num. of flow steps in RealNVP
  model.include_softplus = False  # whether to include softplus layer for positivity
  model.batch_norm = True
  model.init_std = 0.05

  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 64
  training.n_iters = 20000
  training.log_freq = 100
  training.snapshot_freq = 500
  training.n_saved_checkpoints = 5

  config.optim = optim = ml_collections.ConfigDict()
  optim.learning_rate = 2e-4
  optim.grad_clip = 1.
  optim.lambda_data = 1.
  optim.lambda_prior = 1.
  optim.lambda_entropy = 1.
  optim.prior = 'score'
  optim.adam_beta1 = 0.9
  optim.adam_beta2 = 0.999
  optim.adam_eps = 1e-8
  optim.lambda_data_start_order = 0  # initial data weight = 10**(-start_order)
  optim.lambda_data_decay_steps = 1000  # num. steps to decrease data weight by one order of magnitude

  config.likelihood = likelihood = ml_collections.ConfigDict()
  likelihood.likelihood = ''
  likelihood.noise_scale = 0.1
  likelihood.n_dft = 8
  likelihood.eht_image_path = ''
  likelihood.eht_matrix_path = ''
  likelihood.eht_sigmas_path = ''

  config.data = data = ml_collections.ConfigDict()
  data.dataset = ''
  data.image_size = 32
  data.num_channels = 1
  data.centered = False
  data.shuffle_seed = 0

  config.prob_flow = prob_flow = ml_collections.ConfigDict()
  prob_flow.score_model_dir = ''
  prob_flow.n_trace_estimates = 16
  # ODE solver.
  prob_flow.solver = 'Dopri5'
  prob_flow.stepsize_controller = 'PIDController'
  prob_flow.dt0 = 0.001
  prob_flow.rtol = 1e-3  # rtol for diffrax.PIDController
  prob_flow.atol = 1e-5  # atol for diffrax.PIDController
  # Adjoint ODE solver.
  prob_flow.adjoint_method = 'BacksolveAdjoint'
  prob_flow.adjoint_solver = 'Dopri5'
  prob_flow.adjoint_stepsize_controller = 'PIDController'
  prob_flow.adjoint_rms_seminorm = True  # seminorm can reduce speed of backprop
  prob_flow.adjoint_rtol = 1e-3
  prob_flow.adjoint_atol = 1e-5

  config.seed = 42

  return config
