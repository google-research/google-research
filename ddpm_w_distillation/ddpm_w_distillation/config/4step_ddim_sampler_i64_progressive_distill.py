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


teacher_ckpt_path = 'projects/diffusion/stage1_batch256_dropout0_continue_43202616/1/retained_checkpoints/checkpoint_20000'

ddim_noisy_sampler_extra_step_teacher_path = 'projects/diffusion/i64_stage2_snr_truc_128_1_50k_42688273/1/retained_checkpoints/checkpoint_150000'
# NOTE: NOT used

# NOTE: todo change the following path
single_model_path = 'projects/diffusion/stage2_ddim_1024_1_50k_1e-4_new_teacher_43219921/1/retained_checkpoints/checkpoint_400000'

eval_sampling_num_steps = 4
train_num_steps = 4
sampling_num_steps_train_start = 128  # not used
distill_steps_per_iter = 50000  # not used 10000 #10000 #1000 #50000
w_sample_const = 0  #4.0 #0.3 #0 #4


def get_config():
  return D(
      launch=D(
          sweep=hyper.product([
              hyper.sweep('config.seed', [0]),  #TODO [1, 2, 3] change to [0]
              hyper.sweep('config.model.args.uncond_prob',
                          [0.]),  # check NOTE not 0.1
              # hyper.sweep(config.model.acond_uncond_coefs)
          ]),),
      # added
      distillation=D(
          # teacher checkpoint is used for teacher and initial params of student
          teacher_checkpoint_path=teacher_ckpt_path,
          # steps_per_iter=distill_steps_per_iter,  # number of distillation training steps per halving of sampler steps
          # only_finetune_temb=False,
          # start_num_steps=start_num_steps,
          # end_num_steps=end_num_steps,

          # another_teacher_init=False, #True, # added
      ),
      # added
      seed=0,
      progressive_distill=True,  # a flag for stage 2 training
      main_class='Model',
      dataset=D(
          name='ImageNet',
          args=D(
              image_size=64,
              class_conditional=True,
              randflip=True,
          ),
      ),
      sampler='ddim',
      ddim_noisy_sampler_extra_step_teacher=ddim_noisy_sampler_extra_step_teacher_path,

      ##
      #together
      use_sample_single_ckpt=True,
      sample_single_ckpt_path=single_model_path,
      #together
      model=D(
          # architecture
          name='w_unet3',
          args=D(
              ch=192,
              emb_ch=768,  # default is ch * 4
              ch_mult=[1, 2, 3, 4],
              num_res_blocks=3,
              attn_resolutions=[8, 16, 32],
              num_heads=None,
              head_dim=64,
              dropout=0.1,
              logsnr_input_type='inv_cos',
              w_input_type='inv_cos',  # w embedding added
              resblock_resample=True,
              uncond_prob=0.1,  #NOTE: default, but as sweep 0.,
              # extra_class=True,
          ),
          teacher_extra_class=True,  #NOTE added
          mean_type='v',  #'eps', #both might not work since the teach model uses eps, 'both', #NOTE: need to implement 'eps',
          teacher_mean_type='v',  # added
          logvar_type='fixed_medium:0.3',  # TODO: check
          mean_loss_weight_type='snr_trunc',  #'constant', #NOTE changed defalut 'snr_trunc', #constant='mse', snr, snr_trunc
          logvar_loss_type='none',

          # logsnr schedule
          train_num_steps=train_num_steps,
          eval_sampling_num_steps=eval_sampling_num_steps,
          train_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_clip_denoised=True,
          # added
          # eval_sampling_num_steps_train_start=8, # NOTE: need to change
          train_w_schedule=D(name='uniform', logsnr_min=0., logsnr_max=4.),
      ),
      train=D(
          # optimizer
          batch_size=256,  #2048, # TODO: change back 2048,
          optimizer='adam',
          learning_rate=1e-4,  #3e-4, #NOTE: todo #1e-4, #edited 3e-4,
          learning_rate_warmup_steps=0,  #edited 10000,  # used to be 1k, but 10k helps with stability
          learning_rate_anneal_type='linear',  # TODO: checked
          learning_rate_anneal_steps=distill_steps_per_iter,  # TODO: checked
          weight_decay=0.0,
          ema_decay=0.9999,  #0.,
          grad_clip=1.0,
          substeps=10,
          enable_update_skip=False,
          # logging
          log_loss_every_steps=100,
          checkpoint_every_secs=900,  # 15 minutes
          retain_checkpoint_every_steps=20000,  # old checkpoints won't get deleted
          eval_every_steps=20000,
          w_conditoned_training=True,  # added
          w_warmup_steps=0,  # NOTE, set 0 10000, # added to worm up w embedding
      ),
      eval=D(
          batch_size=128,  # TODO change to 128,
          num_inception_samples=50000,
          sampling_fixed_classes=[249, 284],  # visualize malamute, siamese cat
          sampling_fixed_w=[0.1, 0.3, 0.5],  # NOTE, not used
          w_sample_const=w_sample_const,  #4.0, #0.3,
          noisy_sampler_interpolation=0.2,  #0.5, #0.2, # NOTE: need to change
          use_retained_ckpt=True,
      ),
  )
