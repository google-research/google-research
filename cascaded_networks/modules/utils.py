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

"""General utils.

  See GCS api for upload / download functionality:
  https://github.com/googleapis/python-storage/blob/master/google/cloud/storage/blob.py # pylint: disable=line-too-long
"""
import datetime
import os
import random
import numpy as np
import torch


def make_reproducible(random_seed):
  """Make experiments reproducible."""
  print(f'Making reproducible on seed {random_seed}')
  random.seed(random_seed)
  np.random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  os.environ['PYTHONHASHSEED'] = str(random_seed)


def get_timestamp(datetime_format='%YY_%mM_%dD-%Hh_%Mm_%Ss'):
  """Create timestamp."""
  timestamp = datetime.datetime.now().strftime(datetime_format)
  return timestamp


def save_metrics(metrics, output_dir, config):
  """Save metrics and upload it to GCS."""
  if config.debug:
    return
  save_path = os.path.join(output_dir, 'metrics.pt')
  torch.save(metrics, save_path)


def save_model(model, optimizer, output_dir, epoch_i, config):
  """Save model and upload it to GCS."""
  if config.debug:
    return

  save_dict = {
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
  }

  ckpt_dir = os.path.join(output_dir, 'ckpts')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  save_path = os.path.join(ckpt_dir, f'ckpt__epoch_{epoch_i:04d}.pt')
  torch.save(save_dict, save_path)


def save_model_config(model_config, output_dir, config):
  """Save model and upload it to GCS."""
  if config.debug:
    return
  save_path = os.path.join(output_dir, 'model_config.pt')
  torch.save(model_config, save_path)


def save_flags(flags, output_dir, config):
  """Save flags and upload it to GCS."""
  if config.debug:
    return
  save_path = os.path.join(output_dir, 'flagfile.txt')
  flags.append_flags_into_file(save_path)

