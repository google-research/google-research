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

"""miscellaneous utils."""
from collections import defaultdict  # pylint: disable=g-importing-member
import json
import os

import torch


def find_best_checkpoint(
    ckpt_dir, start_from=None, end_on=None, metric_name='fid5k_full'
):
  """find checkpoint with best metric value and return path."""
  # based on stylegan training-runs outputs
  metric_file = os.path.join(ckpt_dir, f'metric-{metric_name}.jsonl')
  fids = []
  with open(metric_file) as f:
    for line in f:
      fids.append((json.loads(line.strip())))
  metric = []
  for item in fids:
    metric.append((item['results'][metric_name], item['snapshot_pkl']))
  if start_from is not None:
    metric = metric[start_from:]
  if end_on is not None:
    metric = metric[:end_on]
  ckpt_metric = min(metric)
  print('best checkpoint:')
  print(ckpt_metric)
  ckpt_path = os.path.join(ckpt_dir, ckpt_metric[1])
  print(ckpt_path)
  print('final checkpoint: %s' % metric[-1][1])
  print('final checkpoint idx: %s' % len(metric))
  return ckpt_path


def interpolate(x, size, mode='bilinear'):
  out = torch.nn.functional.interpolate(
      x, size, mode=mode, align_corners=False, antialias=True
  )
  return out


def concat_dict(input_list, dim=1):
  # input: list of dictionaries
  # output: dictionary with values concatenated from input list
  output_dict = defaultdict(list)
  for item in input_list:
    for k, v in item.items():
      output_dict[k].append(v)
  return {k: torch.cat(v, dim=dim) for k, v in output_dict.items()}


def count_parameters(model, all_params=False):
  return sum(
      p.numel() for p in model.parameters() if p.requires_grad or all_params
  )
