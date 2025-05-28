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

import argparse
import os
import subprocess
import sys

import yaml


def format_value(value):
  """Converts Python types to strings suitable for command line arguments."""
  if isinstance(value, bool):
    return str(value)  # Pass True/False as strings
  return str(value)


def main():
  parser = argparse.ArgumentParser(
      description='Run training using accelerate and a YAML config file.'
  )
  parser.add_argument(
      'config_path',
      type=str,
      help='Path to the YAML configuration file.',
      nargs='?',  # Make the argument optional
      default=(  # Default to config.yaml if no path is provided
          'config/kd_train.yaml'
      ),
  )
  args = parser.parse_args()

  # --- Load Configuration ---
  print(f'Loading configuration from: {args.config_path}')
  try:
    with open(args.config_path, 'r') as f:
      config = yaml.safe_load(f)
  except FileNotFoundError:
    print(f'Error: Configuration file not found at {args.config_path}')
    sys.exit(1)
  except yaml.YAMLError as e:
    print(f'Error parsing YAML file: {e}')
    sys.exit(1)

  # --- Extract Parameters ---
  task_p = config.get('task_params', {})
  kd_p = config.get('kd_params', {})
  train_p = config.get('training_params', {})
  model_p = config.get('model_params', {})
  res_p = config.get('resource_params', {})
  exec_p = config.get('exec_params', {})

  # --- Validate Required Params ---
  required_top_level = [
      'task_params',
      'kd_params',
      'training_params',
      'model_params',
      'resource_params',
      'exec_params',
  ]
  required_task = ['task_type', 'inp_length', 'max_new_tokens']
  required_kd = [
      'kd_type',
      'top_k',
      'student_temperature',
      'student_top_p',
      'teacher_temperature',
      'teacher_top_p',
      'distance_metric',
  ]
  required_train = [
      'seed',
      'lr',
      'grad_acc_size',
      'num_epoch',
      'eval_step',
      'early_stop_epoch',
      'mixed_precision',
  ]
  required_model = [
      'checkpoint_template',
      'tokenizer_name',
  ]  # assistant_checkpoint is handled below
  required_res = [
      'gpu_group',
      'num_processes',
      'main_process_port',
      'user',
      'wandb_key',
      'wandb_proj',
  ]
  required_exec = ['debug_enable', 'enable_stop_token', 'ckpt_prefix']

  missing = []
  for key in required_top_level:
    if key not in config:
      missing.append(f"Top-level key '{key}'")
  for key in required_task:
    if key not in task_p:
      missing.append(f'task_params.{key}')
  for key in required_kd:
    if key not in kd_p:
      missing.append(f'kd_params.{key}')
  for key in required_train:
    if key not in train_p:
      missing.append(f'training_params.{key}')
  for key in required_model:
    if key not in model_p:
      missing.append(f'model_params.{key}')
  for key in required_res:
    if key not in res_p:
      missing.append(f'resource_params.{key}')
  for key in required_exec:
    if key not in exec_p:
      missing.append(f'exec_params.{key}')

  if missing:
    print('Error: Missing required configuration parameters:')
    for m in missing:
      print(f' - {m}')
    sys.exit(1)

  # --- Calculate Derived Parameters ---
  max_length = task_p['inp_length'] + task_p['max_new_tokens']

  # Construct prefix
  prefix = (
      f"{task_p['task_type']}_{kd_p['kd_type']}_{kd_p['top_k']}_"
      f"kd_from_sft_teacher_t_{kd_p['teacher_temperature']}_p_{kd_p['teacher_top_p']}_"
      f"student_t_{kd_p['student_temperature']}_p_{kd_p['student_top_p']}_"
      f"{kd_p['distance_metric']}"
  )

  # Construct checkpoint path
  checkpoint = model_p['checkpoint_template'].format(
      user=res_p['user'], task_type=task_p['task_type']
  )

  # Construct assistant checkpoint path (handle override)
  if (
      'assistant_checkpoint_override' in model_p
      and model_p['assistant_checkpoint_override']
  ):
    assistant_checkpoint = model_p['assistant_checkpoint_override']
  elif 'assistant_checkpoint_template' in model_p:
    assistant_checkpoint = model_p['assistant_checkpoint_template'].format(
        task_type=task_p['task_type'],
        seed=train_p['seed'],
        # Add other formatting keys if your template needs them
    )
  else:
    print(
        "Error: Missing 'assistant_checkpoint_template' or"
        " 'assistant_checkpoint_override' in model_params"
    )
    sys.exit(1)

  # --- Set Environment Variables ---
  env = os.environ.copy()
  env['CUDA_VISIBLE_DEVICES'] = format_value(res_p['gpu_group'])
  env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
  env['TOKENIZERS_PARALLELISM'] = 'false'
  # Optional: Add WANDB keys if accelerate doesn't pick them up automatically
  # env["WANDB_API_KEY"] = res_p['wandb_key']
  # env["WANDB_PROJECT"] = res_p['wandb_proj']

  # --- Construct accelerate Command ---
  cmd = ['accelerate', 'launch']

  # Accelerate options
  cmd.extend([
      '--multi_gpu',
      f"--mixed_precision={train_p['mixed_precision']}",
      '--num_processes',
      format_value(res_p['num_processes']),
      '--main_process_port',
      format_value(res_p['main_process_port']),
      '--offload_optimizer_device',
      'cpu',  # Keep these hardcoded as in original script
      '--offload_param_device',
      'cpu',  # Keep these hardcoded as in original script
  ])

  # Training script
  cmd.append('train/ddp_skd.py')  # Assuming it's in a 'train' subdirectory

  # Script arguments (mapping from config keys to cmd flags)
  arg_map = {
      'lr': ('-lr', train_p['lr']),
      'top_k': ('-top_k', kd_p['top_k']),
      'task_type': ('-task_type', task_p['task_type']),
      'grad_acc_size': ('-grad_acc_size', train_p['grad_acc_size']),
      'kd_type': ('-kd_type', kd_p['kd_type']),
      'distance_metric': ('-distance_metric', kd_p['distance_metric']),
      'debug_enable': ('-debug_enable', exec_p['debug_enable']),
      'n_epoch': ('-n_epoch', train_p['num_epoch']),
      'seed': ('-seed', train_p['seed']),
      'eval_step': ('-eval_step', train_p['eval_step']),
      'prefix': ('-prefix', prefix),
      'assistant_checkpoint': ('-assistant_checkpoint', assistant_checkpoint),
      'checkpoint': ('-checkpoint', checkpoint),
      'inp_length': ('-inp_length', task_p['inp_length']),
      'tokenizer_name': ('-tokenizer_name', model_p['tokenizer_name']),
      'max_new_tokens': ('-max_new_tokens', task_p['max_new_tokens']),
      'max_length': ('-max_length', max_length),
      'wandb_key': ('-wandb_key', res_p['wandb_key']),
      'wandb_proj': ('-wandb_proj', res_p['wandb_proj']),
      'enable_stop_token': ('-enable_stop_token', exec_p['enable_stop_token']),
      'student_temperature': (
          '-student_temperature',
          kd_p['student_temperature'],
      ),
      'student_top_p': ('-student_top_p', kd_p['student_top_p']),
      'teacher_temperature': (
          '-teacher_temperature',
          kd_p['teacher_temperature'],
      ),
      'teacher_top_p': ('-teacher_top_p', kd_p['teacher_top_p']),
      'ckpt_prefix': ('-ckpt_prefix', exec_p['ckpt_prefix']),
      'early_stop_epoch': ('-early_stop_epoch', train_p['early_stop_epoch']),
      'expected_seq_len': (
          '-expected_seq_len',
          task_p.get('expected_seq_len', 0),
      ),  # Handle optional
  }

  # Add optional mixed_ratio if present
  if 'mixed_ratio' in kd_p:
    arg_map['mixed_ratio'] = ('-mixed_ratio', kd_p['mixed_ratio'])

  for _, (flag, value) in arg_map.items():
    cmd.extend([flag, format_value(value)])

  # --- Execute Command ---
  print('\n--- Running Command ---')
  # Print command in a readable format
  print(' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
  print('-----------------------\n')

  print(
      f"Train {task_p['task_type']} from {assistant_checkpoint} with"
      f" {kd_p['kd_type']} with top k {kd_p['top_k']}\n"
  )

  try:
    # Using Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Read and print output line by line
    for line in process.stdout:
      print(line, end='')

    process.wait()  # Wait for the process to complete

    if process.returncode != 0:
      print(f'\nError: Command exited with status {process.returncode}')
      sys.exit(process.returncode)
    else:
      print('\nTraining command completed successfully.')

  except FileNotFoundError:
    print(
        "Error: 'accelerate' command not found. Make sure accelerate is "
        'installed and in your PATH.'
    )
    sys.exit(1)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'An error occurred while running the command: {e}')
    sys.exit(1)


if __name__ == '__main__':
  main()
