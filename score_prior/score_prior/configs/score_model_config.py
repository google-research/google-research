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

"""Default config file."""

import ml_collections


def get_config():
  """Returns the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  # training
  # need to define: training.n_iters, training.batch_size,
  # training.snapshot_freq, training.log_freq, training.eval_freq
  config.training = training = ml_collections.ConfigDict()
  training.sde = 'vpsde'
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.n_jitted_steps = 1
  training.n_iters = 100000
  training.batch_size = 256
  training.snapshot_freq = 500
  training.log_freq = 100
  training.eval_freq = 100

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.17

  # data
  # need to define: data.dataset, data.image_size, data.num_channels
  config.data = data = ml_collections.ConfigDict()
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.dataset = ''
  data.image_size = 32
  data.num_channels = 3
  data.n_bits = 8
  data.standardize = False

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ncsnpp'
  model.num_scales = 1000
  model.sigma_min = 0.002
  model.sigma_max = 50.
  model.beta_min = 0.1
  model.beta_max = 10.
  model.dropout = 0.1
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 64
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.embedding_type = 'positional'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.interpolation = 'bilinear'  # NCSNv2

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  # evaluation
  config.eval = evaluation = ml_collections.ConfigDict()
  evaluation.batch_size = 256

  config.seed = 42

  return config
