# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Config file for reproducing the results of DDPM on church."""

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 64
  training.n_epochs = 2400000
  training.n_iters = 2400001
  training.snapshot_freq = 20000
  training.snapshot_freq_for_preemption = 1000
  training.snapshot_sampling = True
  training.anneal_power = 2
  training.loss = 'ddpm'
  # shared configs for sample generation
  step_size = 0.0000062
  n_steps_each = 2
  final_only = True
  target_snr = 0.15
  noise_removal = True
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'ddpm'
  sampling.noise_removal = noise_removal
  sampling.step_size = step_size
  sampling.n_steps_each = n_steps_each
  sampling.final_only = final_only
  sampling.target_snr = target_snr
  # eval
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = 256
  evaluate.num_samples = 50000
  evaluate.step_size = step_size
  evaluate.n_steps_each = n_steps_each
  evaluate.begin_ckpt = 40
  evaluate.end_ckpt = 50
  evaluate.verbose = False
  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'LSUN'
  data.category = 'church_outdoor'
  data.image_size = 256
  data.centered = True
  data.random_flip = True
  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.sigma_begin = 50
  model.num_classes = 232
  model.ema_rate = 0.9999
  model.sigma_dist = 'geometric'
  model.sigma_end = 0.01
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 4, 4)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.dropout = 0.
  model.resamp_with_conv = True
  model.conditional = True
  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-5
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  return config
