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

"""Configuration."""

import ml_collections



def config_dict(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():

  return config_dict(
      seed=0,
      dataset=config_dict(
          name='CIFAR10',
          args=config_dict(
              class_conditional=False,
              randflip=True,
          ),
      ),
      model=config_dict(
          # architecture, see main.py and model.py
          name='unet0',
          args=config_dict(
              ch=128,
              out_ch=3,
              ch_mult=[1, 2, 2, 2],
              num_res_blocks=2,
              attn_resolutions=[16],
              num_heads=1,
              dropout=0.1,
              model_output='logistic_pars',  # logits or logistic_pars
          ),
          # diffusion betas, see diffusion_categorical.get_diffusion_betas
          diffusion_betas=config_dict(
              type='linear',
              # start, stop only relevant for linear, power, jsdtrunc schedules.
              start=1.e-4,  # 1e-4 gauss, 0.02 uniform
              stop=0.02,  # 0.02, gauss, 1. uniform
              num_timesteps=1000,
          ),
          # Settings used in diffusion_categorical.py
          model_prediction='x_start',  # 'x_start','xprev'
          # 'gaussian','uniform','absorbing'
          transition_mat_type='gaussian',
          transition_bands=None,
          loss_type='hybrid',  # kl,cross_entropy_x_start, hybrid
          hybrid_coeff=0.001,  # only used for hybrid loss type.
      ),
      train=config_dict(
          # optimizer
          batch_size=128,
          optimizer='adam',
          learning_rate=2e-4,
          learning_rate_warmup_steps=0,
          weight_decay=0.0,
          ema_decay=0.9999,
          grad_clip=1.0,
          substeps=10,
          num_train_steps=1500000,  # multiple of substeps
          # logging
          log_loss_every_steps=1000,
          checkpoint_every_secs=900,  # 15 minutes
          retain_checkpoint_every_steps=100000,
          eval_every_steps=50000,
      ))
