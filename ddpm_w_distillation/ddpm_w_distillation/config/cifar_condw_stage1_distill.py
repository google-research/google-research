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


distill_steps_per_iter = 1000000

# initialization only
another_teacher_ckpt_path = 'projects/diffusion/stage1_2048_42848231/1/retained_checkpoints/checkpoint_520000'

teacher_ckpt_path = 'projects/diffusion/cifar10_retrain_v_stage0_43438125/1/retained_checkpoints/checkpoint_480000'

single_model_path = 'projects/diffusion/stage1_2048_42848231/1/retained_checkpoints/checkpoint_520000'
eval_sampling_num_steps = 256  #128 #512 #256 #512 #128
train_batch_size = 128

use_sample_single_ckpt = False  # for eval_w single ckpt only True #False
use_retained_ckpt = False  #True only when using single ckpt
train_clip_x = False
# sampler = 'ddim', # 'noisy'


def get_config():
  return D(
      launch=D(
          sweep=hyper.product([
              # hyper.sweep('config.model.args.uncond_prob', [0.01, 0.02, 0.05]),
              # hyper.sweep('config.model.args.uncond_prob', [0.1, 0.2, 0.5]),
              hyper.sweep('config.seed', [0]),  #TODO [1, 2, 3] change to [0]
              hyper.sweep(
                  'config.model.args.uncond_prob', [0.1]
              ),  # NOTE: not used for w_unet model check NOTE change from 0.1 to 0
              # hyper.sweep(config.model.acond_uncond_coefs)
          ]),),
      # added
      distillation=D(
          # teacher checkpoint is used for teacher and initial params of student
          teacher_checkpoint_path=teacher_ckpt_path,
          steps_per_iter=distill_steps_per_iter,  # number of distillation training steps per halving of sampler steps
          only_finetune_temb=False,  #TODO!! False,
          another_teacher_init=False,  # NOTE used for continue training
          another_teacher_path=another_teacher_ckpt_path,
          # start_num_steps=start_num_steps,
          # end_num_steps=end_num_steps,
      ),
      # added
      seed=0,
      main_class='Model',
      dataset=D(
          name='CIFAR10',
          args=D(
              # image_size=64,
              class_conditional=True,
              randflip=True,
          ),
      ),
      sampler='noisy',  #'ddim', # 'noisy', # added

      ##
      #together
      use_sample_single_ckpt=use_sample_single_ckpt,  #True,
      sample_single_ckpt_path=single_model_path,
      #together
      model=D(
          # architecture
          name='w_unet3',
          args=D(
              ch=256,
              emb_ch=1024,  # default is ch * 4
              ch_mult=[1, 1, 1],
              num_res_blocks=3,
              attn_resolutions=[8, 16],
              num_heads=1,
              # head_dim=64,
              dropout=0.,  #NOTE!! changed previously 0.1,
              logsnr_input_type='inv_cos',
              w_input_type='inv_cos',  # w embedding added
              resblock_resample=True,
              uncond_prob=0.1,  #NOTE: default, but as sweep 0.,
          ),
          teacher_extra_class=True,  #NOTE added
          mean_type='v',  #'eps', #'v', #NOTE: change to v 'eps',
          teacher_mean_type='v',  #"eps", # added
          logvar_type='fixed_medium:0.3',  # TODO: check
          mean_loss_weight_type='snr',  #'constant', #'snr', #'snr_trunc', #note not 'constant', #constant='mse', snr, snr_trunc
          logvar_loss_type='none',

          # logsnr schedule
          train_num_steps=0,  # train in continuous time
          eval_sampling_num_steps=eval_sampling_num_steps,
          train_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_clip_denoised=True,
          # added
          train_w_schedule=D(
              name='uniform',
              # logsnr_min=0., logsnr_max=0.5),
              # logsnr_min=0., logsnr_max=1.0),
              # logsnr_min=0., logsnr_max=2.0),
              logsnr_min=0.,
              logsnr_max=4.),
          # NOTE can set logsnr_max=logsnr_min for a single w value

          # sample interpolation
          # cond_uncond_coefs=[1.3, -0.3], # [cond_coef, uncond_coef]
          # eval_cond_uncond_coefs # NOTE: never have it for distillation!, it does not make sense
      ),
      train=D(
          # optimizer
          batch_size=train_batch_size,  #2048, # 256 #2048, # TODO: change back 2048,
          optimizer='adam',
          learning_rate=1e-4,  #3e-4, #edited 3e-4,
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
          train_clip_x=train_clip_x,  # NOTE added
          w_conditoned_training=True,  # added
          w_warmup_steps=10000,  #1, #10000, # added to worm up w embedding
      ),
      eval=D(
          batch_size=128,  # TODO change to 128,
          num_inception_samples=50000,
          sampling_fixed_classes=[0, 5],  # visualize malamute, siamese cat
          sampling_fixed_w=[0.1, 0.3, 0.5],
          w_sample_const=0.3,
          noisy_sampler_interpolation=0.5,  #0.2, # NOTE: need to change
          use_retained_ckpt=use_retained_ckpt,  #True,
      ),
  )
