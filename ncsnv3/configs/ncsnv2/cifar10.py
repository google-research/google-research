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
"""Config file for training NCSNv2 on CIFAR-10."""

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 128
  training.n_epochs = 500000
  training.n_iters = 300001
  training.snapshot_freq = 5000
  training.snapshot_freq_for_preemption = 2500
  training.snapshot_sampling = True
  training.anneal_power = 2
  training.loss = 'ncsnv2'
  # shared configs for sample generation
  step_size = 0.0000062
  n_steps_each = 5
  ckpt_id = 300000
  final_only = True
  noise_removal = False
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'ald'
  sampling.step_size = step_size
  sampling.n_steps_each = n_steps_each
  sampling.ckpt_id = ckpt_id
  sampling.final_only = final_only
  sampling.noise_removal = noise_removal
  # eval
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = 1024
  evaluate.num_samples = 50000
  evaluate.step_size = step_size
  evaluate.n_steps_each = n_steps_each
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 62
  evaluate.verbose = False
  evaluate.ensemble = False
  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.centered = False
  data.image_size = 32
  data.random_flip = True
  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ncsnv2_64'
  model.scale_by_sigma = True
  model.sigma_begin = 50
  model.num_classes = 232
  model.ema_rate = 0.999
  model.sigma_dist = 'geometric'
  model.sigma_end = 0.01
  model.normalization = 'InstanceNorm++'
  model.nonlinearity = 'elu'
  model.nf = 128
  model.interpolation = 'bilinear'
  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 0
  optim.grad_clip = -1.

  config.seed = 24

  return config
