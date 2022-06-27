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

"""RM3 optimizer."""

import functools

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import required


class RM3(Optimizer):
  """Implements RM3 (median of 3) algorithm.

  It's based on SGD implementation from torch.optim package.

  Args:
  params (iterable): iterable of parameters to optimize or dicts defining
    parameter groups
  lr (float): learning rate
  weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
  """

  def __init__(self, params, lr=required, weight_decay=0):
    if lr is not required and lr < 0.0:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if weight_decay < 0.0:
      raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))

    defaults = dict(lr=lr, weight_decay=weight_decay)
    super(RM3, self).__init__(params, defaults)
    self.ringbuffer_size = 3

  def step(self, closure=None):
    for group in self.param_groups:
      weight_decay = group['weight_decay']

      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data
        if weight_decay != 0:
          d_p = d_p.add(weight_decay, p.data)
        param_state = self.state[p]
        if 'gradients_buffer' not in param_state:
          param_state['gradients_buffer'] = []
          for _ in range(self.ringbuffer_size):
            param_state['gradients_buffer'].append(torch.clone(d_p).detach())
          param_state['gradients_counter'] = 1
        else:
          param_state['gradients_buffer'][
              param_state['gradients_counter'] % 3].copy_(d_p)
          param_state['gradients_counter'] += 1

        gradients_buffer = param_state['gradients_buffer']

        # If we have 3 gradients already, compute median and update weights
        if param_state['gradients_counter'] >= self.ringbuffer_size:
          sum_gradients = functools.reduce(torch.add, gradients_buffer)
          min_gradients = functools.reduce(torch.min, gradients_buffer)
          max_gradients = functools.reduce(torch.max, gradients_buffer)
          median_of_3 = sum_gradients - min_gradients - max_gradients
          p.data.add_(-group['lr'], median_of_3)
