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

"""Checkpoint utils."""

import logging
import os
import pathlib

import torch

Path = pathlib.Path


PROJ_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_ckpt(eval_ckpt_dir, prefix='', ckpt_type='last', run_id=None):
  """Find checkpoint."""
  if prefix:
    prefix = prefix + '-'
  if run_id is not None:
    path = Path(os.path.join(PROJ_ROOT, 'multirun'))
    wildcard = path.glob('**/{}/checkpoints/*.ckpt'.format(run_id))
  else:
    if eval_ckpt_dir:
      path = Path(os.path.join(PROJ_ROOT, eval_ckpt_dir))
    else:
      cwd = os.getcwd()
      cwd = cwd.replace(',exp.eval_only=True', '')
      cwd = cwd.replace(',exp.interpret_only=True', '')
      path = Path(os.path.join(cwd, 'causal_over_time'))
    wildcard = path.glob('**/*.ckpt')
  for ckptfile in wildcard:
    filename = ckptfile.stem
    if ckpt_type == 'last' and filename == f'{prefix}last':
      return str(ckptfile)
    elif ckpt_type == 'best' and filename.startswith(f'{prefix}epoch='):
      return str(ckptfile)
  raise FileNotFoundError(
      f'Ckpt file not found in {str(path)}, with prefix={prefix},'
      f' ckpt_type={ckpt_type}!'
  )


def load_checkpoint(model, ckpt_path, load_ema=False, strict=True):
  """Load checkpoint."""
  ckpt = torch.load(ckpt_path)
  model.load_state_dict(ckpt['state_dict'], strict=strict)
  if load_ema and model.hparams.exp.weights_ema:
    if not hasattr(model, 'ema_treatment'):
      model.configure_optimizers()
    if 'ema_state_dict' in ckpt:
      model.ema_treatment.load_state_dict(
          ckpt['ema_state_dict']['ema_treatment']
      )
      model.ema_non_treatment.load_state_dict(
          ckpt['ema_state_dict']['ema_non_treatment']
      )
    else:
      logger.warning('No ema_state_dict found in ckpt!')
  return model


if __name__ == '__main__':
  print(PROJ_ROOT)
