# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Training NCSNv3 on CIFAR-10 with class label information."""

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 128
  training.n_epochs = 500000
  training.n_iters = 1300001
  training.snapshot_freq = 50000
  training.snapshot_freq_for_preemption = 5000
  training.snapshot_sampling = True
  training.anneal_power = 2
  training.loss = 'ncsnv2_conditional'
  step_size = 0.0000062
  n_steps_each = 1
  ckpt_id = 300000
  final_only = True
  noise_removal = True
  begin_ckpt = 5
  end_ckpt = 27
  target_snr = 0.16
  eval_batch_size = 1024
  conditional = True
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'ald_fix_snr_reverse_diffusion'
  sampling.target_snr = target_snr
  sampling.step_size = step_size
  sampling.n_steps_each = n_steps_each
  sampling.ckpt_id = ckpt_id
  sampling.final_only = final_only
  sampling.noise_removal = noise_removal
  # eval
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = eval_batch_size
  evaluate.num_samples = 50000
  evaluate.step_size = step_size
  evaluate.n_steps_each = n_steps_each
  evaluate.begin_ckpt = begin_ckpt
  evaluate.end_ckpt = end_ckpt
  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.centered = False
  data.num_classes = 10
  data.random_flip = True
  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ncsnv3'
  model.scale_by_sigma = True
  model.sigma_begin = 50
  model.num_classes = 1000
  model.ema_rate = 0.999
  model.sigma_dist = 'geometric'
  model.sigma_end = 0.01
  model.normalization = 'GroupNorm'
  model.input_normalize = False
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.dropout = 0.1
  model.resamp_with_conv = True
  model.conditional = conditional
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.0
  model.embedding_type = 'positional'
  model.conv_size = 3

  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config
