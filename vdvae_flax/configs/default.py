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

"""Config file for training VDVAE."""

import ml_collections


def d(**kwargs):
  """Helper of creating a config dict."""
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.exp_name = 'experiment'
  config.model_name = 'vdvae'
  config.seed = 42
  config.precision = 'highest'

  config.optimizer = d(
      name='adamw',
      args=d(
          b1=0.9,
          b2=0.9,
          eps=1e-8,
          weight_decay=.01,
      ),
      base_learning_rate=3e-4,  # initial learning rate
      gradient_skip_norm=500.,
      gradient_clip_norm=250.,
  )
  config.encoder = d(
      num_blocks=33,
      num_channels=384,
      bottlenecked_num_channels=96,
      # downsampling_rates is a list of tuples of the form
      # (block_index, downsampling_rates).
      downsampling_rates=[(11, 2), (18, 2), (25, 2), (29, 4)],
      precision=config.get_oneway_ref('precision'),
  )
  config.decoder = d(
      num_blocks=43,
      num_channels=384,
      bottlenecked_num_channels=96,
      latent_dim=16,
      # upsampling_rates is a list of tuples of the form
      # (block_index, upsampling_rates).
      upsampling_rates=[(1, 4), (4, 2), (10, 2), (21, 2)],
      output_image_resolution=32,
      precision=config.get_oneway_ref('precision'),
  )
  config.sampler = d(
      num_mixtures=10,
      low=0,
      high=255,
      num_output_channels=3,
      precision=config.get_oneway_ref('precision'),
  )
  config.training = d(
      batch_size=128,  # training batch size
      warmup_iters=25,  # lr warmup
      num_train_steps=500_000,
      substeps=100,
  )
  config.data = d(task='cifar10',)
  config.evaluation = d(
      subset='test',
      ema_rate=0.9996,
      batch_size=128,
      num_data=10_000,
  )
  config.logs = d(
      log_loss_every_steps=500,
      eval_batch_every_steps=1000,
      eval_full_every_steps=5000,
      checkpoint_every_steps=5000,
  )
  config.trial = 0  # Dummy for repeated runs.

  return config


def get_hyper(h):
  return h.product([
      h.sweep('trial', [0]),
  ], name='config')
