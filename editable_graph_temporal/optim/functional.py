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
"""Implements the concrete optimization computation for AdamCons and AdamCons_global_embed."""

import math
from typing import List
import torch
from torch import Tensor


def adam_cons(params, init_params,
              grads, exp_avgs,
              exp_avg_sqs, max_exp_avg_sqs,
              state_steps, *, amsgrad, beta1,
              beta2, lr, weight_decay, eps,
              cons_thre):
  r"""Performs constrained optimization based Adam algorithm computation.

  Args:
    params (iterable): iterable of parameters to optimize or dicts defining
      parameter groups.
    init_params (iterable): initial model parameters.
    grads (iterable): gradients of each model parameter.
    exp_avgs (iterable): exponential average of gradients up to now.
    exp_avg_sqs (iterable): exponential average of gradient squares up to now.
    max_exp_avg_sqs (iterable): maximum gradient square average of gradients.
    state_steps (iterable): total number of optimization steps of each model
      parameter.
    amsgrad (boolean, optional): whether to use the AMSGrad variant of this
      algorithm from the paper `On the Convergence of Adam and Beyond`_
      (default: False).
    beta1 (float, optional): the first coefficient used for computing running
      averages of gradient and its square (default: 0.9).
    beta2 (float, optional): the second coefficient used for computing running
      averages of gradient and its square (0.999).
    lr (float, optional): learning rate (default: 1e-3).
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
    eps (float, optional): term added to the denominator to improve numerical
      stability (default: 1e-8).
    cons_thre (float, optional): the threshold for constrained optimization
      (default: 1e-3). .. _Constrained optimization\: Modifying Memories in
      Transformer Models:
      https://arxiv.org/abs/2012.00363.
  """
  for i, param in enumerate(params):

    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    if weight_decay != 0:
      grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    if amsgrad:
      # Maintains the maximum of all 2nd moment running avg. till now
      torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
      # Use the max. for normalizing running avg. of gradient
      denom = (max_exp_avg_sqs[i].sqrt() /
               math.sqrt(bias_correction2)).add_(eps)
    else:
      denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1

    new_param_by_grad = torch.addcdiv(param, exp_avg, denom, value=-step_size)
    grad_norm = torch.linalg.norm(new_param_by_grad - init_params[i])
    grad_factor = min(
        cons_thre * torch.sqrt(torch.prod(torch.tensor(param.shape))) /
        grad_norm, 1.0)
    param.copy_(init_params[i] +
                (new_param_by_grad - init_params[i]) * grad_factor)


def adam_cons_with_idx(params, init_params,
                       grads, exp_avgs,
                       exp_avg_sqs, max_exp_avg_sqs,
                       state_steps, *, amsgrad, beta1,
                       beta2, lr, weight_decay, eps,
                       cons_thre, frozen_idx):
  r"""Performs constrained optimization for global embedding matrix.

  This functional API is based off of the Adam algorithm.

  Args:
    params (iterable): iterable of parameters to optimize or dicts defining
      parameter groups.
    init_params (iterable): initial model parameters.
    grads (iterable): gradients of each model parameter.
    exp_avgs (iterable): exponential average of gradients up to now.
    exp_avg_sqs (iterable): exponential average of gradient squares up to now.
    max_exp_avg_sqs (iterable): maximum gradient square average of gradients.
    state_steps (iterable): total number of optimization steps of each model
      parameter.
    amsgrad (boolean, optional): whether to use the AMSGrad variant of this
      algorithm from the paper `On the Convergence of Adam and Beyond`_
      (default: False).
    beta1 (float, optional): the first coefficient used for computing running
      averages of gradient and its square (default: 0.9).
    beta2 (float, optional): the second coefficient used for computing running
      averages of gradient and its square (0.999).
    lr (float, optional): learning rate (default: 1e-3).
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
    eps (float, optional): term added to the denominator to improve numerical
      stability (default: 1e-8).
    cons_thre (float, optional): the threshold for constrained optimization
      (default: 1e-3).
    frozen_idx (list, optional): the list of idxs for global embedding vectors
      that will not be changed (default: None). .. _Constrained optimization\:
      Modifying Memories in Transformer Models:
      https://arxiv.org/abs/2012.00363.
  """
  for i, param in enumerate(params):

    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    if weight_decay != 0:
      grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    if amsgrad:
      # Maintains the maximum of all 2nd moment running avg. till now
      torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
      # Use the max. for normalizing running avg. of gradient
      denom = (max_exp_avg_sqs[i].sqrt() /
               math.sqrt(bias_correction2)).add_(eps)
    else:
      denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1

    new_param_by_grad = torch.addcdiv(param, exp_avg, denom, value=-step_size)
    new_param_by_grad[frozen_idx] = init_params[i][frozen_idx]
    grad_norm = torch.linalg.norm(new_param_by_grad -
                                  init_params[i]).cpu().item()
    grad_factor = min(
        cons_thre *
        math.sqrt(float(
            (param.shape[0] - len(frozen_idx)) * param.shape[1])) / grad_norm,
        1.0)
    new_param_by_grad[frozen_idx] = param[frozen_idx]
    grad_factor = torch.tensor([grad_factor],
                               device=param.device).repeat_interleave(
                                   len(param))
    grad_factor[frozen_idx] = 1.0
    param.copy_(init_params[i] +
                (new_param_by_grad - init_params[i]) * grad_factor.view(-1, 1))
