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

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
# optimizer
from torch.optim import SGD, Adam
import torch_optimizer as optim
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .warmup_scheduler import GradualWarmupScheduler

from .visualization import *


def get_parameters(models):
  """Get all model parameters recursively."""
  parameters = []
  if isinstance(models, list):
    for model in models:
      parameters += get_parameters(model)
  elif isinstance(models, dict):
    for model in models.values():
      parameters += get_parameters(model)
  else:  # models is actually a single pytorch model
    parameters += list(models.parameters())
  return parameters


def get_optimizer(hparams, models):
  eps = 1e-8
  parameters = get_parameters(models)
  if hparams.optimizer == 'sgd':
    optimizer = SGD(
        parameters,
        lr=hparams.lr,
        momentum=hparams.momentum,
        weight_decay=hparams.weight_decay)
  elif hparams.optimizer == 'adam':
    optimizer = Adam(
        parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay)
  elif hparams.optimizer == 'radam':
    optimizer = optim.RAdam(
        parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay)
  elif hparams.optimizer == 'ranger':
    optimizer = optim.Ranger(
        parameters, lr=hparams.lr, eps=eps, weight_decay=hparams.weight_decay)
  else:
    raise ValueError('optimizer not recognized!')

  return optimizer


def get_scheduler(hparams, optimizer):
  eps = 1e-8
  if hparams.lr_scheduler == 'steplr':
    scheduler = MultiStepLR(
        optimizer, milestones=hparams.decay_step, gamma=hparams.decay_gamma)
  elif hparams.lr_scheduler == 'cosine':
    scheduler = CosineAnnealingLR(
        optimizer, T_max=hparams.num_epochs, eta_min=eps)
  elif hparams.lr_scheduler == 'poly':
    scheduler = LambdaLR(
        optimizer, lambda epoch:
        (1 - epoch / hparams.num_epochs)**hparams.poly_exp)
  else:
    raise ValueError('scheduler not recognized!')

  if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=hparams.warmup_multiplier,
        total_epoch=hparams.warmup_epochs,
        after_scheduler=scheduler)

  return scheduler


def get_learning_rate(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


def extract_model_state_dict(ckpt_path,
                             model_name='model',
                             prefixes_to_ignore=[]):
  checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
  checkpoint_ = {}
  if 'state_dict' in checkpoint:  # if it's a pytorch-lightning checkpoint
    checkpoint = checkpoint['state_dict']
  for k, v in checkpoint.items():
    if not k.startswith(model_name):
      continue
    k = k[len(model_name) + 1:]
    for prefix in prefixes_to_ignore:
      if k.startswith(prefix):
        print('ignore', k)
        break
    else:
      checkpoint_[k] = v
  return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
  if not ckpt_path:
    return
  model_dict = model.state_dict()
  checkpoint_ = extract_model_state_dict(ckpt_path, model_name,
                                         prefixes_to_ignore)
  model_dict.update(checkpoint_)
  model.load_state_dict(model_dict)
