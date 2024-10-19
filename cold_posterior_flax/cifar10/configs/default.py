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

"""A config for training CIFAR10."""

import ml_collections


def get_config():
  """Config for training on CIFAR10. Should be used on a 2x2 TPU donut."""
  config = ml_collections.ConfigDict()

  # Run without JIT on a single device.
  config.debug_run = False
  # Debugging flag for reporting intermediate output statistics.
  config.report_metrics = 0
  # Use (0) no residuals, (1) residuals, (2) renormed residuals.
  config.use_residual = 2
  # The learning rate for the adam optimizer.
  config.learning_rate = 0.001
  # Learning rate scheduler warmup epochs.
  config.warmup_epochs = 0
  # Learning rate/step_size_factor schedule type; constant, stepped or cosine'
  config.lr_schedule = 'cosine'
  # Temperature schedule type; constant, ramp_up
  config.temp_schedule = 'ramp_up'
  # Learning rate schedule steps as a Python list;
  # Format: '[[step1_epoch, step1_lr_scale], [step2_epoch, step2_lr_scale],...]'
  config.lr_sched_steps = '[[60, 0.2], [120, 0.04], [160, 0.008]]'
  # 'Temperature (ramp_start, ramp_end) for `--temp_schedule=ramp_up`.
  config.temp_ramp = '(100,150)'
  # The decay rate used for the momentum optimizer.
  config.momentum = 0.98
  # The amount of L2-regularization to apply.
  config.l2_reg = 0.0
  # The amount of Prior-regularization to apply, if used.
  config.prior_reg = 1.0
  # Number of batches to estimate the preconditioner every epoch.
  config.precon_est_batches = 32
  # Epsilon for the preconditioner estimator.
  config.precon_est_eps = 1e-7
  # Batch size for training.
  config.batch_size = 128
  # Batch size for evaluation, should divide 10000 and number devices.
  config.eval_batch_size = 80
  # Number of training epochs.
  config.num_epochs = 300
  # Epochs per cycle, for SGMCMC.
  config.cycle_length = 10
  # Network architecture.
  config.arch = 'rnv1_20'
  # Wide ResNet Dropout rate.
  config.wrn_dropout_rate = 0.0
  # Random seed for network initialization.
  config.seed = 0
  # Network normalization style.
  # Values: 'bn', 'bn_sync', 'none', 'gn_4', 'gn_16', 'bcn_sync'.
  config.normalization = 'bn'
  # Benchmark to run.
  # Values: 'cifar10', 'cifar100', 'imagenet'.
  config.benchmark = 'cifar10'
  # KL loss on activations.
  config.std_penalty_mult = 0.0
  # Algorithm to run, 'sgd' or 'sgmcmc'.
  config.algorithm = 'sgd'
  # Optimizer/Sampler to use for SGD/SGMCMC respectively.
  # Values: ['adam', 'sym_euler', 'momentum']
  config.optimizer = 'momentum'

  # Activation function used in network
  # Values: 'relu', 'tanh', 'tlu', 'none', 'tldu', 'tlduz', 'tlum', 'relu_norm',
  # 'relu_unitvar', 'bias_relu_norm', 'selu', 'bias_SELU_norm',
  # 'bias_scale_relu_norm', 'SELU_norm_rebias', 'bias_scale_SELU_norm', 'swish'.
  config.activation_f = 'relu'
  # Weight Normalization type.
  # Values: 'none', 'learned', 'fixed', 'ws_sqrt', 'learned_b', 'ws'.
  config.weight_norm = 'none'
  # Scale of normal initialization of bias.
  config.bias_scale = 1e-5
  # Use softplus parametrization of scale parameter in normalization layers.
  config.softplus_scale = 1
  # Compensate for padding to ensure normal output.
  config.compensate_padding = 1
  # Prior on the kernels.
  # Values: 'none', 'he_normal', 'normal'.
  config.kernel_prior = 'normal'
  # Prior on scale parameters.
  # Values: 'none', 'he_normal', 'normal'.
  config.scale_prior = 'normal'
  # Prior on the bias.
  # Values: 'none', 'normal'.
  config.bias_prior = 'normal'

  # Bias prior scale.
  config.bias_prior_scale = 1.
  # Kernel prior scale.
  config.kernel_prior_scale = 1.
  # Scale prior scale (std-dev of prior on the scale parameter).
  config.scale_prior_scale = 1.
  # Base temperature for SGMCMC sampler.
  config.base_temp = 1.
  # Number of samples in ensemble when using SGMCMC.
  config.ensemble_size = 27
  # Max norm of gradient, 0.0 to turn off gradient clipping.
  config.gradient_clipping = 5.0
  # Run evaluation, turn off for faster training.
  config.do_eval = True

  return config


def get_hyper(h):
  return h.product([
      h.sweep('seed', range(1)),
  ], name='config')
