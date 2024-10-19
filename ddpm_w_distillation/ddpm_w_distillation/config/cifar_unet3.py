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

# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=g-no-space-after-comment,g-bad-todo
# pylint: disable=invalid-name,line-too-long

import ml_collections


class hyper:
  pass


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  return D(
      launch=D(
          sweep=hyper.product([
              hyper.sweep('config.model.args.dropout', [0.1, 0.2]),
              hyper.sweep('config.seed', list(range(5))),
          ]),),
      seed=0,
      main_class='Model',
      dataset=D(
          name='CIFAR10',
          args=D(
              class_conditional=False,
              randflip=True,
          ),
      ),
      model=D(
          # architecture
          name='unet3',
          args=D(
              ch=256,
              emb_ch=1024,  # default is ch * 4
              ch_mult=[1, 1, 1],
              num_res_blocks=3,
              attn_resolutions=[8, 16],
              num_heads=1,
              dropout=0.2,
              logsnr_input_type='inv_cos',
              resblock_resample=True,
          ),
          mean_type='eps',
          logvar_type='fixed_large',
          mean_loss_type='mse',
          logvar_loss_type='none',
          # not using the dpm_distill2 configuration below:
          # mean_type='both',  # eps, x, both
          # logvar_type='fixed_medium',
          # mean_loss_weight_type='snr_trunc',  # constant, snr, snr_trunc
          # logvar_loss_type='none',

          # logsnr schedule
          train_num_steps=0,  # train in continuous time
          eval_sampling_num_steps=1024,
          train_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_clip_denoised=True,
      ),
      train=D(
          # optimizer
          batch_size=128,
          optimizer='adam',
          learning_rate=2e-4,
          learning_rate_warmup_steps=1000,
          weight_decay=0.001,
          ema_decay=0.9999,
          grad_clip=1.0,
          substeps=10,
          enable_update_skip=False,
          # logging
          log_loss_every_steps=100,
          checkpoint_every_secs=900,  # 15 minutes
          eval_every_steps=10000,
      ),
      eval=D(
          batch_size=128,
          num_inception_samples=50000,
      ),
  )
