# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Model utils."""
import torch


def apply_weight_decay(net, weight_decay):
  """Apply weight decay."""
  if weight_decay == 0:
    return
  for _, param in net.named_parameters():
    if param.grad is None:
      continue
    param.grad = param.grad.add(param, alpha=weight_decay)


def load_model(net, kwargs):
  """Load pretrained model."""
  pretrained_path = kwargs.get('pretrained_path', False)
  assert pretrained_path, 'Could not find pretrained_path!'
  print(f'Loading model from {pretrained_path}')
  state_dict = torch.load(pretrained_path)['state_dict']
  net.load_state_dict(state_dict)
  return net
