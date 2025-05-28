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


w_sample_const = 4  #0.3 #4.
single_model_path = 'projects/diffusion/cifar10_retrain_v_stage0_43438125/1/retained_checkpoints/checkpoint_480000'
sampler = 'ddim'


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  return D(
      launch=D(
          sweep=hyper.product([
              # hyper.sweep('config.model.args.uncond_prob', [0.01, 0.02, 0.05]),
              # hyper.sweep('config.model.eval_sampling_num_steps', [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
              hyper.sweep('config.seed', [1]),
              hyper.sweep('config.model.args.uncond_prob', [0.1]),
              hyper.sweep(
                  'config.model.eval_cond_uncond_coefs',
                  [[1., 0.], [1.3, -0.3], [2., -1.], [3., -2.], [5., -4.]]),
          ]),),
      seed=0,
      main_class='Model',
      dataset=D(
          name='CIFAR10',
          args=D(
              # image_size=64,
              class_conditional=True,  # NOTE
              randflip=True,
          ),
      ),
      sampler=sampler,  #'ddim', # 'noisy', # added
      ##
      #together
      use_sample_single_ckpt=True,
      sample_single_ckpt_path=single_model_path,
      #together
      model=D(
          # architecture
          name='unet3',
          args=D(
              ch=256,  #192,
              emb_ch=1024,  #768,  # default is ch * 4
              ch_mult=[1, 1, 1],  #[1, 2, 3, 4],
              num_res_blocks=3,
              attn_resolutions=[8, 16],  #[8, 16, 32],
              num_heads=1,  #None,
              # head_dim=64, # NOTE check not used
              dropout=0.2,  #0.1,
              logsnr_input_type='inv_cos',
              resblock_resample=True,
              uncond_prob=0.1,
          ),
          mean_type='v',  #NOTE: change to 'v', #'v', #'eps',
          logvar_type='fixed_medium:0.3',
          mean_loss_weight_type='snr',  #'snr_trunc', #'mse',
          logvar_loss_type='none',

          # logsnr schedule
          train_num_steps=0,  # train in continuous time
          eval_sampling_num_steps=1024,  #256,
          train_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_clip_denoised=True,
          eval_cond_uncond_coefs=[1. + w_sample_const, -w_sample_const
                                 ]  #[1.3, -0.3], # [cond_coef, uncond_coef]
      ),
      train=D(
          # optimizer
          batch_size=128,  #2048, #256,
          optimizer='adam',
          learning_rate=2e-4,  #3e-4,
          learning_rate_warmup_steps=10000,  # used to be 1k, but 10k helps with stability
          weight_decay=0.001,  #0.0,
          ema_decay=0.9999,
          grad_clip=1.0,
          substeps=10,
          enable_update_skip=False,
          # logging
          log_loss_every_steps=100,
          checkpoint_every_secs=900,  # 15 minutes
          retain_checkpoint_every_steps=20000,  # old checkpoints won't get deleted
          eval_every_steps=20000,
      ),
      eval=D(
          batch_size=128,
          num_inception_samples=50000,
          sampling_fixed_classes=[0, 5],
          w_sample_const=w_sample_const,
      ),
  )
