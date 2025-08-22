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
              hyper.sweep('config.model.args.uncond_prob', [0.1]),
          ]),),
      seed=0,
      main_class='Model',
      dataset=D(
          name='ImageNet',
          args=D(
              image_size=128,
              class_conditional=True,
              randflip=True,
          ),
      ),
      model=D(
          # architecture
          name='unet3',
          args=D(
              ch=256,
              emb_ch=1024,  # default is ch * 4
              ch_mult=[1, 1, 2, 3, 4],
              num_res_blocks=2,
              attn_resolutions=[8, 16, 32],
              num_heads=4,
              head_dim=None,
              dropout=0.0,
              logsnr_input_type='inv_cos',
              resblock_resample=True,
              uncond_prob=0.,
          ),
          mean_type='eps',
          logvar_type='fixed_medium:0.2',
          mean_loss_type='mse',
          logvar_loss_type='none',

          # logsnr schedule
          train_num_steps=0,  # train in continuous time
          eval_sampling_num_steps=256,
          train_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_clip_denoised=True,
      ),
      train=D(
          # optimizer
          batch_size=256,
          optimizer='adam',
          learning_rate=1e-4,
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
      eval=D(
          batch_size=128,
          num_inception_samples=50000,
          sampling_fixed_classes=[249, 284],  # visualize malamute, siamese cat
      ),
  )
