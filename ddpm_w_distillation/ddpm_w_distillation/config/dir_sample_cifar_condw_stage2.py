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


# added, edited
end_num_steps = 1  # eventual number of steps in the distilled sampler
start_num_steps = 1024  #512 #1024 #1024 #512 #1024 #NOTE: todo change to #1024  # number of steps in baseline sampler
distill_steps_per_iter = 50000  #50000 #10000 #10000 #1000 #50000

# NOTE alithough it's stage1 it's actually stage 2
teacher_ckpt_path = 'projects/diffusion/cifar_stage1_43492025/1/retained_checkpoints/checkpoint_620000'

train_batch = 128  #2048 # 256
lr = 1e-4  #3e-4 #1e-4
sampling_num_steps_train_start = 128
use_eval_ckpt_dir = True

# # # our two stage approach
# eval_ckpt_dir = 'projects/diffusion/cifar10_stage2_50k_ema_decay0_43877018/1/retained_checkpoints/'
# sampler='ddim'

# nvidia_2step approach
eval_ckpt_dir = 'projects/diffusion/cifar_stage2_twostep_nvidia_50k_43902200/1/retained_checkpoints/'
sampler = 'new_two_step_nvidia'

# previous runs with ema-decay 0.9999
# 'projects/diffusion/cifar_stage1_43555434/1/retained_checkpoints/'
use_retained_ckpt = True
w_sample_const = 1.0  #0.
waiting_for_new_ckpt = False

progressive_sampling_step = True


def get_config():
  config = D(
      launch=D(
          sweep=hyper.product([
              hyper.sweep('config.seed', [0]),  #TODO [1, 2, 3] change to [0]
              hyper.sweep('config.model.args.uncond_prob',
                          [0.]),  # check NOTE not 0.1
              hyper.sweep('config.eval.w_sample_const', [0., 0.3, 1., 2., 4.])
          ]),),
      # added
      distillation=D(
          # teacher checkpoint is used for teacher and initial params of student
          teacher_checkpoint_path=teacher_ckpt_path,
          steps_per_iter=distill_steps_per_iter,  # number of distillation training steps per halving of sampler steps
          only_finetune_temb=False,
          start_num_steps=start_num_steps,
          end_num_steps=end_num_steps,
          another_teacher_init=False,  #True, # added
      ),
      # added
      seed=0,
      progressive_distill=True,  # a flag for stage 2 training
      main_class='Model',
      dataset=D(
          name='CIFAR10',
          args=D(
              # image_size=64,
              class_conditional=True,
              randflip=True,
          ),
      ),
      sampler=sampler,  #'noisy',
      use_eval_ckpt_dir=use_eval_ckpt_dir,
      eval_ckpt_dir=eval_ckpt_dir,
      waiting_for_new_ckpt=waiting_for_new_ckpt,
      progressive_sampling_step=progressive_sampling_step,
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
              dropout=0.,  # NOTE changes 0.1,
              logsnr_input_type='inv_cos',
              w_input_type='inv_cos',  # w embedding added
              resblock_resample=True,
              uncond_prob=0.1,  #NOTE: default, but as sweep 0.,
              # extra_class=True,
          ),
          teacher_extra_class=True,  #NOTE added
          mean_type='v',  #'eps', #both might not work since the teach model uses eps, 'both', #NOTE: need to implement 'eps',
          teacher_mean_type='v',  # added
          logvar_type='fixed_large',  #'fixed_medium:0.3', # TODO: check
          mean_loss_weight_type='snr_trunc',  #NOTE:default 'snr_trunc', 'snr' performs worse #'constant', #NOTE changed defalut 'snr_trunc', #constant='mse', snr, snr_trunc
          logvar_loss_type='none',

          # logsnr schedule
          train_num_steps=end_num_steps,
          eval_sampling_num_steps=end_num_steps,
          train_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_logsnr_schedule=D(
              name='cosine', logsnr_min=-20., logsnr_max=20.),
          eval_clip_denoised=True,
          # added
          eval_sampling_num_steps_train_start=sampling_num_steps_train_start,  # NOTE: need to change
          noisy_sampler_interpolation=0.2,
          train_w_schedule=D(
              name='uniform',
              # logsnr_min=0., logsnr_max=0.5),
              # logsnr_min=0., logsnr_max=1.0),
              # logsnr_min=0., logsnr_max=2.0),
              logsnr_min=0.,
              logsnr_max=4.),
      ),
      train=D(
          # optimizer
          batch_size=train_batch,  #2048, # TODO: change back 2048,
          optimizer='adam',
          learning_rate=lr,  # 1e-4 for 50k, 2e-4 for 10k #3e-4, #NOTE: todo #1e-4, #edited 3e-4,
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
          sampling_fixed_classes=[0, 5],  # visualize malamute, siamese cat
          sampling_fixed_w=[0.1, 0.3, 0.5],  # NOTE, not used
          use_retained_ckpt=use_retained_ckpt,
          w_sample_const=w_sample_const,
      ),
  )

  # added
  if hasattr(config, 'progressive_distill'):
    # run some sanity checks on inputs
    assert config.distillation.start_num_steps > 0
    assert config.distillation.end_num_steps > 0
    assert config.distillation.start_num_steps % config.distillation.end_num_steps == 0

  return config
