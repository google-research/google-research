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

"""CIFAR distillation."""

# pylint: disable=invalid-name,line-too-long

import ml_collections

end_num_steps = 1  # eventual number of steps in the sampler
start_num_steps = 8192  # number of steps in baseline sampler
distill_steps_per_iter = 50000
teacher_ckpt_path = ''  # checkpoints to be released later


def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  config = D(
      distillation=D(
          # teacher checkpoint is used for teacher and initial params of student
          teacher_checkpoint_path=teacher_ckpt_path,
          steps_per_iter = distill_steps_per_iter,  # number of distillation training steps per halving of sampler steps
          only_finetune_temb = False,
          start_num_steps = start_num_steps,
          end_num_steps = end_num_steps,
      ),
      seed=0,
      dataset=D(
          name='CIFAR10',
          args=D(
              class_conditional=False,
              randflip=True,
          ),
      ),
      sampler='ddim',
      model=D(
          # architecture, same as teacher but with 0 dropout!
          name='unet_iddpm',
          args=D(
              ch=256,
              emb_ch=1024,  # default is ch * 4
              ch_mult=[1, 1, 1],
              num_res_blocks=3,
              attn_resolutions=[8, 16],
              num_heads=1,
              dropout=0.,
              logsnr_input_type='inv_cos',
              resblock_resample=True,
          ),
          mean_type='x',  # eps, x, both, v
          logvar_type='fixed_large',
          mean_loss_weight_type='snr_trunc',  # constant, snr, snr_trunc

          # logsnr schedule
          train_num_steps=end_num_steps,
          eval_sampling_num_steps=end_num_steps,
          train_logsnr_schedule=D(name='cosine',
                                  logsnr_min=-20., logsnr_max=20.),
          eval_logsnr_schedule=D(name='cosine',
                                 logsnr_min=-20., logsnr_max=20.),
          eval_clip_denoised=True,
      ),
      train=D(
          # optimizer
          batch_size=128,
          optimizer='adam',
          learning_rate=5e-5,
          learning_rate_warmup_steps=0,
          learning_rate_anneal_type='linear',
          learning_rate_anneal_steps=distill_steps_per_iter,
          weight_decay=0.,
          ema_decay=0.,
          grad_clip=1.0,
          substeps=10,
          enable_update_skip=False,
          # logging
          log_loss_every_steps=100,
          checkpoint_every_secs=900,  # 15 minutes
          eval_every_steps=10000,
      ),
  )

  if hasattr(config, 'distillation'):
    # run some sanity checks on inputs
    assert config.distillation.start_num_steps > 0
    assert config.distillation.end_num_steps > 0
    assert config.distillation.start_num_steps % config.distillation.end_num_steps == 0

  return config
