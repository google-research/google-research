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

"""General utility functions."""

import os
import torch


def print_trainable_parameters(model):
  """Prints the number of trainable parameters in the model."""
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    if len(param.shape) > 1:
      all_param += param.shape[0] * param.shape[1]
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f'trainable params: {trainable_params} || all params: {all_param} ||'
      f' trainable%: {100 * trainable_params / all_param} || Note: loading with'
      ' bits and bytes will havle the number of all parameters'
  )


def get_num_training_points(dataset_name):
  if '180k' in dataset_name:
    num_training_points = 180000
  else:
    raise ValueError('Dataset size not specified for DP training.')
  return num_training_points


def save_checkpoint(
    path, peft_model, optimizer, lr_scheduler, completed_steps, epoch
):
  """Save the model, optimizer, lr_scheduler, and epoch to the given path."""
  peft_model.save_pretrained(path + f'/peftmodel_epoch{epoch}')

  optimizer_state_dict = optimizer.state_dict()
  lr_scheduler_state_dict = lr_scheduler.state_dict()

  optimizer_state_dict['completed_steps'] = completed_steps

  os.makedirs(path + f'/checkpoint{epoch}', exist_ok=True)
  torch.save(optimizer_state_dict, path + f'/checkpoint{epoch}/optimizer.pt')
  torch.save(
      lr_scheduler_state_dict, path + f'/checkpoint{epoch}/lr_scheduler.pt'
  )


def load_training_states(path, optimizer, lr_scheduler, epoch):
  """Returns completed_steps from optimizer, lr_scheduler, and epoch."""
  optimizer_state_dict = torch.load(path + f'/checkpoint{epoch}/optimizer.pt')
  lr_scheduler_state_dict = torch.load(
      path + f'/checkpoint{epoch}/lr_scheduler.pt'
  )

  completed_steps = optimizer_state_dict.pop('completed_steps')

  optimizer.load_state_dict(optimizer_state_dict)
  lr_scheduler.load_state_dict(lr_scheduler_state_dict)

  return completed_steps


def find_newest_checkpoint_epoch(path):
  """Find the newest checkpoint epoch in the given path."""
  checkpoint_epochs = []

  for file in os.listdir(path):
    if os.path.isdir(os.path.join(path, file)):
      if 'checkpoint' in file:
        checkpoint_epochs.append(int(file.split('checkpoint')[-1]))
  if checkpoint_epochs:
    return max(checkpoint_epochs)
  else:
    return -1
