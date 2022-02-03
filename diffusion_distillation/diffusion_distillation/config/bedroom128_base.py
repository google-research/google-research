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

"""LSUN bedroom 128x128."""

# pylint: disable=invalid-name,line-too-long

import ml_collections


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  return D(
      seed=0,
      dataset=D(
          name='LSUN',
          args=D(
              subset='bedroom',
              image_size=128,
              randflip=True,
          ),
      ),
      sampler='ddim',
      model=D(
          # architecture
          name='unet_iddpm',
          args=D(
              ch=64,
              emb_ch=768,  # default is ch * 4
              ch_mult=[1, 2, 4, 6, 8],
              num_res_blocks=3,
              attn_resolutions=[8, 16, 32],
              num_heads=None,
              head_dim=64,
              dropout=0.1,
              logsnr_input_type='inv_cos',
              resblock_resample=True,
          ),
          mean_type='both',  # eps, x, both, v
          logvar_type='fixed_large',
          mean_loss_weight_type='snr_trunc',  # constant, snr, snr_trunc

          # logsnr schedule
          train_num_steps=0,  # train in continuous time
          eval_sampling_num_steps=1024,
          train_logsnr_schedule=D(name='cosine',
                                  logsnr_min=-20., logsnr_max=20.),
          eval_logsnr_schedule=D(name='cosine',
                                 logsnr_min=-20., logsnr_max=20.),
          eval_clip_denoised=True,
      ),
      train=D(
          # optimizer
          batch_size=2048,
          optimizer='adam',
          learning_rate=3e-4,
          learning_rate_warmup_steps=1000,
          weight_decay=0.0,
          ema_decay=0.9999,
          grad_clip=1.0,
          substeps=10,
          enable_update_skip=False,
          # logging
          log_loss_every_steps=100,
          checkpoint_every_secs=900,  # 15 minutes
          retain_checkpoint_every_steps=20000,  # old checkpoints won't get deleted
          eval_every_steps=10000,
      ),
  )
