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

"""Script to generate commands."""

import argparse
import os

from utils import dp_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_name', type=str, default='chatbot_arena_instructions_train180k'
)
parser.add_argument('--model_name', type=str, default='yahma/llama-7b-hf')
parser.add_argument(
    '--job_sess',
    type=str,
    default='debug',
    help='session of this job, will be prefix of sess',
)
parser.add_argument(
    '--perdevice_bs', type=int, default=4, help='number of batchsize per gpu'
)
parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
parser.add_argument(
    '--max_seq_len', type=int, default=2048, help='max sequence length'
)
parser.add_argument('--total_bs', type=int, default=32, help='total batchsize')
parser.add_argument(
    '--num_epochs', type=int, default=3, help='number of epochs'
)
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument(
    '--lr_scheduler',
    type=str,
    default='constant',
    help='learning rate scheduler',
)
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument(
    '--clip', type=float, default=-1, help='per-example gradient clip norm'
)
parser.add_argument(
    '--eps',
    type=float,
    default=-1,
    help='target epsilon value for (eps,delta)-DP',
)
parser.add_argument(
    '--delta',
    type=float,
    default=5e-7,
    help='target delta value for (eps,delta)-DP',
)
parser.add_argument(
    '--prompt_style',
    type=str,
    default=None,
    help=(
        'style of the prompt, could be vicuna style (vicuna) or empty prompt'
        ' for unconditional generation (uncond_generation)'
    ),
)
parser.add_argument(
    '--no_eval_at_start',
    action='store_true',
    help='whether not to run evaluation at epoch 0',
)


args = parser.parse_args()


NUM_TRAIN_SAMPLES = 180000


prefix = './'

os.makedirs(prefix + 'logs', exist_ok=True)

if args.gpus == 1:
  accelerate_cfg_file = (
      prefix + 'accelerate_configs/accelerate_config_nofsdp.cfg'
  )
else:
  accelerate_cfg_file = (
      prefix + f'accelerate_configs/accelerate_config_nofsdp_gpu{args.gpus}.cfg'
  )

dataset_name = args.dataset_name
model_name = args.model_name
perdevice_bs = args.perdevice_bs
max_seq_len = args.max_seq_len
total_bs = args.total_bs
accumulation_steps = total_bs // (perdevice_bs * args.gpus)
assert total_bs % perdevice_bs == 0
num_epochs = args.num_epochs
lr = args.lr
wd = args.wd
clip = args.clip
eps = args.eps
delta = args.delta
gpus = args.gpus
job_sess = args.job_sess
prompt_style = args.prompt_style
lr_scheduler = args.lr_scheduler

steps = int(num_epochs / (total_bs / NUM_TRAIN_SAMPLES))

if eps > 0:
  np = dp_utils.get_noise_multiplier(
      eps, delta, steps, total_bs / NUM_TRAIN_SAMPLES
  )
  print(
      f'Noise multiplier is {np} for ({eps,delta})-DP. Config:'
      f' batchsize={total_bs}, dataset size={NUM_TRAIN_SAMPLES},'
      f' epochs={num_epochs}.'
  )
else:
  np = -1

model_name_in_file = model_name.split('/')[-1]

if dataset_name == 'chatbot_arena_instructions_train180k':
  assert prompt_style == 'uncond_generation'
  sess = 'gensyn_arena180k_noredacted_model{}_eps{}_delta{}_bs{}_maxseq{}_epoch{}_lr{}_clip{}_np{}_gpus{}'.format(
      model_name_in_file,
      eps,
      delta,
      total_bs,
      max_seq_len,
      num_epochs,
      lr,
      clip,
      np,
      gpus,
  )
elif 'labelled' in dataset_name:
  assert prompt_style == 'vicuna'
  sess = 'sft_noredacted_model{}_eps{}_delta{}_bs{}_maxseq{}_epoch{}_lr{}_clip{}_np{}_gpus{}'.format(
      model_name_in_file,
      eps,
      delta,
      total_bs,
      max_seq_len,
      num_epochs,
      lr,
      clip,
      np,
      gpus,
  )
else:
  raise NotImplementedError
sess = job_sess + '_' + sess

prepend = ''
hf_login_str = ''  # add your huggingface login token here
if not hf_login_str:
  raise ValueError(
      'Please add your huggingface login token to hf_login_str in'
      ' generate_train_command.py'
  )

add_str = ''
if np > 0 or args.no_eval_at_start:
  add_str = '--no_eval_at_start'


command = f"""{prepend} accelerate launch --config_file {accelerate_cfg_file} {prefix}/train_clm.py --dataset_name {dataset_name}    \
    --model_name_or_path {model_name}  --per_device_train_batch_size {perdevice_bs} --output_dir {prefix}/outputs/{sess} \
    --block_size {max_seq_len}  --gradient_ckpt --lr_scheduler_type {lr_scheduler} --log_freq 10 --gradient_accumulation_steps {accumulation_steps} --clip_norm {clip} --delta {delta} --noise_multiplier {np} \
    --num_train_epochs {num_epochs} --learning_rate {lr} --num_warmup_steps 30 --exp_path {prefix}  --weight_decay {wd} --prompt_style {prompt_style} \
    --access_token {hf_login_str} --qbits 16 {add_str} 2>&1 | tee -a {prefix}/logs/{sess}.txt """

os.system(command)
