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

"""Generate inference command."""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--instruction_file', type=str, default='utils/syn_instruct.txt'
)
parser.add_argument('--model_name', type=str, default='yahma/llama-7b-hf')
parser.add_argument('--adapter_path', type=str, default=None)
parser.add_argument(
    '--job_sess',
    type=str,
    default='debug',
    help='session of this job, will be prefix of sess',
)
parser.add_argument('--perdevice_bs', type=int, default=1)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--max_seq_len', type=int, default=2048)

# generation config
parser.add_argument(
    '--prompt_style',
    type=str,
    default='uncond_generation',
    help='Prompt style.',
)
parser.add_argument(
    '--no_sample',
    action='store_true',
    help='use sampling or not. When false greedy decoding is used.',
)
parser.add_argument(
    '--top_k', type=int, default=0, help='top k words for generation'
)
parser.add_argument(
    '--top_p', type=float, default=1.0, help='top p probability'
)
parser.add_argument(
    '--temperature', type=float, default=1.0, help='sampling temperature'
)
parser.add_argument(
    '--repetition_penalty',
    type=float,
    default=1.0,
    help='The parameter for repetition penalty. 1.0 means no penalty.',
)
parser.add_argument(
    '--access_token', type=str, default=None, help='Huggingface access token'
)
parser.add_argument(
    '--enforce_min_new_tokens',
    action='store_true',
    help='enforce min new tokens or not',
)
parser.add_argument(
    '--main_process_port',
    type=int,
    help='main process port for accelerate launch',
    default=29500,
)

args = parser.parse_args()


prefix = './'

os.makedirs(prefix + 'inference_outputs', exist_ok=True)

if args.gpus == 1:
  accelerate_cfg_file = (
      prefix + 'accelerate_configs/accelerate_config_nofsdp.cfg'
  )
else:
  accelerate_cfg_file = (
      prefix + f'accelerate_configs/accelerate_config_nofsdp_gpu{args.gpus}.cfg'
  )

instruction_file = args.instruction_file
model_name = args.model_name
adapter_path = args.adapter_path
job_sess = args.job_sess
perdevice_bs = args.perdevice_bs
gpus = args.gpus
max_seq_len = args.max_seq_len
prompt_style = args.prompt_style
no_sample = args.no_sample
top_k = args.top_k
top_p = args.top_p
temperature = args.temperature
repetition_penalty = args.repetition_penalty
enforce_min_new_tokens = args.enforce_min_new_tokens


model_name_in_file = model_name.split('/')[-1]
if prompt_style == 'vicuna':
  sess = 'syn_answers_{}_genmaxseq{}_sample{}_topk{}_topp{}_temp{}_rep{}_enforce{}_syngpus{}'.format(
      model_name_in_file,
      max_seq_len,
      not no_sample,
      top_k,
      top_p,
      temperature,
      repetition_penalty,
      enforce_min_new_tokens,
      gpus,
  )
elif (
    prompt_style == 'len_cond_generation' or prompt_style == 'uncond_generation'
):
  assert (
      'syn_instruct' in instruction_file
  ), 'instruction file must be syn_instruct.txt'
  sess = 'syn_instructions_{}_genmaxseq{}_sample{}_topk{}_topp{}_temp{}_rep{}_enforce{}_syngpus{}'.format(
      model_name_in_file,
      max_seq_len,
      not no_sample,
      top_k,
      top_p,
      temperature,
      repetition_penalty,
      enforce_min_new_tokens,
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
      ' generate_inference_command.py'
  )

add_str = ''
if enforce_min_new_tokens:
  add_str += ' --enforce_min_new_tokens '
if not no_sample:
  add_str += ' --sample '
if adapter_path is not None:
  add_str += f' --lora_weights_path {prefix}/{adapter_path} '

command = f"""{prepend}accelerate launch --main_process_port {args.main_process_port} --config_file {accelerate_cfg_file} {prefix}/inference_with_instruction.py  --model_name_or_path {model_name}  \
    --instruction_file {prefix}/{instruction_file} --output_dir {prefix}/inference_outputs/{sess} --max_length {max_seq_len} --prompt_style {prompt_style} \
    --temperature {temperature} --top_k {top_k} --top_p {top_p} --exp_path {prefix} --repetition_penalty {repetition_penalty} {add_str} \
    --access_token {hf_login_str} --qbits 16 """


os.system(command)
