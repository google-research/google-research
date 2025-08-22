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
"""Defines optimizer class for constrained optimization based Adam algorithm."""

import torch
from torch.optim import Optimizer

from editable_graph_temporal.optim import functional


class AdamCons(Optimizer):
  r"""Implements constrained optimization based Adam algorithm.

  We modified the original implementation of Adam algorithm by adding
  parameter updating constraints. Specifically, for any parameter matrix
  \theta_0, let the parameter matrix after updating is \theta, this optimizer
  will force the norm of \theta-\theta_0 be smaller than
  square_root(N) * cons_thre, where N is the number of parameters in \theta_0,
  cons_thre is a specified threshold.
  """

  def __init__(self,
               params,
               lr=1e-3,
               betas=(0.9, 0.999),
               eps=1e-8,
               weight_decay=0,
               cons_thre=1e-3,
               amsgrad=False):
    r"""Instantiates the optimizer.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups.
      lr (float, optional): learning rate (default: 1e-3).
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999)).
      eps (float, optional): term added to the denominator to improve numerical
        stability (default: 1e-8).
      weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
      cons_thre (float, optional): the threshold for constrained optimization
        (default: 1e-3).
      amsgrad (boolean, optional): whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False). .. _Constrained optimization\: Modifying Memories in
        Transformer Models:
        https://arxiv.org/abs/2012.00363.
    """
    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= eps:
      raise ValueError('Invalid epsilon value: {}'.format(eps))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    if not 0.0 <= weight_decay:
      raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
    defaults = dict(
        lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    super(AdamCons, self).__init__(params, defaults)

    self.init_param_copy = []
    self.cons_thre = cons_thre
    for group in self.param_groups:
      new_param_group = []
      for param in group['params']:
        new_param_group.append(param.detach().clone().requires_grad_(False))
      self.init_param_copy.append(new_param_group)

  def __setstate__(self, state):
    super(AdamCons, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('amsgrad', False)

  @torch.no_grad()
  def step(self):
    """Performs a single optimization step."""

    for i, group in enumerate(self.param_groups):
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']
      init_params = self.init_param_copy[i]

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          if p.grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, '
                               'please consider SparseAdam instead.')
          grads.append(p.grad)

          state = self.state[p]
          # Lazy state initialization
          if not state:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            if group['amsgrad']:
              # Maintains max of all exp. moving avg. of sq. grad. values
              state['max_exp_avg_sq'] = torch.zeros_like(
                  p, memory_format=torch.preserve_format)

          exp_avgs.append(state['exp_avg'])
          exp_avg_sqs.append(state['exp_avg_sq'])

          if group['amsgrad']:
            max_exp_avg_sqs.append(state['max_exp_avg_sq'])

          # update the steps for each param group update
          state['step'] += 1
          # record the step after step update
          state_steps.append(state['step'])

      functional.adam_cons(
          params_with_grad,
          init_params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          state_steps,
          amsgrad=group['amsgrad'],
          beta1=beta1,
          beta2=beta2,
          lr=group['lr'],
          weight_decay=group['weight_decay'],
          eps=group['eps'],
          cons_thre=self.cons_thre)


class AdamConsGlobalEmbeds(Optimizer):
  """Constrained optimization for global embedding matrix in GATRNN model.

  This algorightm is based off of Adam.

  It keeps the global embedding vectors corresponding to the indexs in
  frozen_idx unchanged, and update other global embedding vectors with
  constrained optimization based Adam algorithm.
  """

  def __init__(self,
               params,
               lr=1e-3,
               betas=(0.9, 0.999),
               eps=1e-8,
               weight_decay=0,
               cons_thre=1e-3,
               frozen_idx=None,
               amsgrad=False):
    r"""Instantiates the optimizer.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups.
      lr (float, optional): learning rate (default: 1e-3)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999)).
      eps (float, optional): term added to the denominator to improve numerical
        stability (default: 1e-8).
      weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
      cons_thre (float, optional): the threshold for constrained optimization
        (default: 1e-3).
      frozen_idx (list, optional): the list of idxs for global embedding vectors
        that will not be changed (default: None).
      amsgrad (boolean, optional): whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False). .. _Constrained optimization\: Modifying Memories in
        Transformer Models:
        https://arxiv.org/abs/2012.00363.
    """
    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= eps:
      raise ValueError('Invalid epsilon value: {}'.format(eps))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    if not 0.0 <= weight_decay:
      raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
    defaults = dict(
        lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    super(AdamConsGlobalEmbeds, self).__init__(params, defaults)

    self.init_param_copy = []
    self.cons_thre = cons_thre
    self.frozen_idx = frozen_idx
    for group in self.param_groups:
      new_param_group = []
      for param in group['params']:
        new_param_group.append(param.detach().clone().requires_grad_(False))
      self.init_param_copy.append(new_param_group)

  def __setstate__(self, state):
    super(AdamConsGlobalEmbeds, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('amsgrad', False)

  @torch.no_grad()
  def step(self):
    """Performs a single optimization step."""

    for i, group in enumerate(self.param_groups):
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']
      init_params = self.init_param_copy[i]

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          if p.grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, '
                               'please consider SparseAdam instead')
          grads.append(p.grad)

          state = self.state[p]
          # Lazy state initialization
          if not state:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            if group['amsgrad']:
              # Maintains max of all exp. moving avg. of sq. grad. values
              state['max_exp_avg_sq'] = torch.zeros_like(
                  p, memory_format=torch.preserve_format)

          exp_avgs.append(state['exp_avg'])
          exp_avg_sqs.append(state['exp_avg_sq'])

          if group['amsgrad']:
            max_exp_avg_sqs.append(state['max_exp_avg_sq'])

          # update the steps for each param group update
          state['step'] += 1
          # record the step after step update
          state_steps.append(state['step'])

      functional.adam_cons_with_idx(
          params_with_grad,
          init_params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          state_steps,
          amsgrad=group['amsgrad'],
          beta1=beta1,
          beta2=beta2,
          lr=group['lr'],
          weight_decay=group['weight_decay'],
          eps=group['eps'],
          cons_thre=self.cons_thre,
          frozen_idx=self.frozen_idx)
