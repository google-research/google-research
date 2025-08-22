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

####################################################
# NOTE fixed w only
####################################################

import ml_collections


class hyper:
  pass


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


# added
# end_num_steps = 1  # eventual number of steps in the distilled sampler
# start_num_steps = 1024  # number of steps in baseline sampler
distill_steps_per_iter = 1000000
# TODO, change the teacher ckpt path
teacher_ckpt_path = 'projects/diffusion/retrain_snr_2048_42804866/1/retained_checkpoints/checkpoint_220000'

use_sample_single_ckpt = False  # only used for fid evaluation
single_model_path = 'projects/diffusion/stage1_2048_42848231/1/retained_checkpoints/checkpoint_520000'
eval_sampling_num_steps = 512  #128
train_batch_size = 256
train_clip_x = True


def get_config():
  return D(
      launch=D(
          sweep=hyper.product([
              # hyper.sweep('config.model.args.uncond_prob', [0.01, 0.02, 0.05]),
              # hyper.sweep('config.model.args.uncond_prob', [0.1, 0.2, 0.5]),
              hyper.sweep('config.seed', [0]),  #TODO [1, 2, 3] change to [0]
              hyper.sweep('config.model.args.uncond_prob', [0.1]),  # check
          ]),),
      # added
      distillation=D(
          # teacher checkpoint is used for teacher and initial params of student
          teacher_checkpoint_path=teacher_ckpt_path,
          steps_per_iter=distill_steps_per_iter,  # number of distillation training steps per halving of sampler steps
          only_finetune_temb=False,
          # start_num_steps=start_num_steps,
          # end_num_steps=end_num_steps,
      ),
      # added
      seed=0,
      main_class='Model',
      dataset=D(
          name='ImageNet',
          args=D(
              image_size=64,
              class_conditional=True,
              randflip=True,
          ),
      ),
      sampler='ddim',  # added

      ##
      #together
      use_sample_single_ckpt=use_sample_single_ckpt,
      sample_single_ckpt_path=single_model_path,
      #together
      model=D(
          # architecture
          name='unet3',
          args=D(
              ch=192,
              emb_ch=768,  # default is ch * 4
              ch_mult=[1, 2, 3, 4],
              num_res_blocks=3,
              attn_resolutions=[8, 16, 32],
              num_heads=None,
              head_dim=64,
              dropout=0,  #NOTE: 0.1,
              logsnr_input_type='inv_cos',
              resblock_resample=True,
              uncond_prob=0.1,  #NOTE: default, but as sweep 0.,
          ),
          mean_type='v',  #'eps',
          logvar_type='fixed_medium:0.3',  # TODO: check
          mean_loss_weight_type='snr',  #'constant', #constant='mse', snr, snr_trunc
          logvar_loss_type='none',

          # logsnr schedule
          train_num_steps=0,  # train in continuous time
          eval_sampling_num_steps=eval_sampling_num_steps,
          train_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_clip_denoised=True,

          # sample interpolation
          cond_uncond_coefs=[
              5, -4
          ],  #[1.3, -0.3], #[1.3, -0.3], # [cond_coef, uncond_coef]
          # eval_cond_uncond_coefs # NOTE: never have it for distillation!, it does not make sense
      ),
      train=D(
          # optimizer
          batch_size=train_batch_size,  #2048, # TODO: change back 2048,
          optimizer='adam',
          learning_rate=3e-4,  #1e-4, #edited 3e-4,
          learning_rate_warmup_steps=0,  #edited 10000,  # used to be 1k, but 10k helps with stability
          weight_decay=0.0,
          ema_decay=0.9999,
          grad_clip=1.0,
          substeps=10,
          enable_update_skip=False,
          # logging
          log_loss_every_steps=100,
          checkpoint_every_secs=900,  # 15 minutes
          retain_checkpoint_every_steps=20000,  # old checkpoints won't get deleted
          eval_every_steps=20000,

          # added
          train_clip_x=train_clip_x,
      ),
      eval=D(
          batch_size=128,
          num_inception_samples=50000,
          sampling_fixed_classes=[249, 284],  # visualize malamute, siamese cat
      ),
  )
