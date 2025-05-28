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

"""Helper functions."""

import hashlib
import os
import random
import sys
import zipfile
from absl import logging
import numpy as np
import tensorflow as tf
import torch



def set_seed(seed):
  """Sets seed."""
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.multiprocessing.set_sharing_strategy('file_system')


def prepare_environ_for_pytorch():
  """Sets appropriate environment variables for using pytorch on borg."""
  if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = '/tmp'
  os.environ['TMPDIR'] = '/tmp'
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    print('>>> Tensorflow found GPUs:', gpus)
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for d in visible_devices:
      assert d.device_type != 'GPU'
    print('>>> Disabled GPUs for Tensorflow')
  else:
    print('>>> No GPUs found for Tensorflow')


def prepare_folders(args):
  folders_util = [
      args.output_dir,
      os.path.join(args.output_dir, args.output_folder_name),
  ]
  for folder in folders_util:
    if not os.path.exists(folder):
      print(f'===> Creating folder: {folder}')
      os.makedirs(folder)




def seed_hash(*args):
  """Derive an integer hash from all args, for use as a random seed."""
  args_str = str(args)
  return int(hashlib.md5(args_str.encode('utf-8')).hexdigest(), 16) % (2**31)


def print_separator():
  print('=' * 80)


def print_row(row, colwidth=10, latex=False):
  """Prints a row of values."""
  if latex:
    sep = ' & '
    end_ = '\\\\'
  else:
    sep = '  '
    end_ = ''

  def format_val(x):
    if np.issubdtype(type(x), np.floating):
      x = '{:.4f}'.format(x)
    return str(x).ljust(colwidth)[:colwidth]

  print(sep.join([format_val(x) for x in row]), end_)


class Tee:
  """Helper class to print and write to file at the same time."""

  def __init__(self, fname, mode='a'):
    self.stdout = sys.stdout
    self.file = open(fname, mode)

  def write(self, message):
    self.stdout.write(message)
    self.file.write(message)
    logging.info(message)
    self.flush()

  def flush(self):
    self.stdout.flush()
    self.file.flush()
