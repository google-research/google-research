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

"""Utils for DP."""

from prv_accountant import PRVAccountant
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism
import torch


def linear_forward_hook(module, intsr, outtsr):  # pylint: disable=unused-argument
  module.input = intsr[0]


# pylint: disable=invalid-name
def linear_backward_hook(layer, grad_input, grad_output):  # pylint: disable=unused-argument
  """Backward hook for network layer."""
  grad_output = grad_output[0]  # n, len, outdim
  grad_input = layer.input  # n, len, indim

  layer_batch_dim = 0

  A = grad_input
  B = grad_output
  if A.dtype != B.dtype:
    # during forward, some activates are casted to fp32 for stability.
    # Convert them back for gradient computation
    A = A.to(B.dtype)

  # Compute per-sequence gradients
  # The gradients of tokens in the same sequence are summed up
  # k: tokens-per-sample
  # n: batch size
  if layer_batch_dim == 1:
    gs = torch.einsum('kn...i,kn...j->nij', B, A)
    if layer.bias is not None:
      gs_bias = torch.einsum('kn...i->ni', B)
  else:
    gs = torch.einsum('n...i,n...j->nij', B, A)
    if layer.bias is not None:
      gs_bias = torch.einsum('n...k->nk', B)

  layer.weight.grad_sample = gs.float()
  if layer.bias is not None:
    layer.bias.grad_sample = gs_bias.float()


def make_lora_model_dp(model):
  # register forward and backward hooks for lora branch
  for module in model.modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
      module.lora_A['default'].register_forward_hook(linear_forward_hook)
      module.lora_A['default'].register_backward_hook(linear_backward_hook)
      module.lora_B['default'].register_forward_hook(linear_forward_hook)
      module.lora_B['default'].register_backward_hook(linear_backward_hook)


def get_grad_norm(params):
  """Get the gradient norm of each example."""
  # params: all trainable parameters
  # when lora is enabled, the params only contain lora parameters
  for p in params:
    if hasattr(p, 'grad_sample'):
      # n is the batch size
      n = p.grad_sample.shape[0]
      break
  grad_norm_list = torch.zeros(n).cuda()
  for p in params:
    if hasattr(p, 'grad_sample'):
      flat_g = p.grad_sample.reshape(n, -1)
      current_norm_list = torch.norm(flat_g, dim=1)
      grad_norm_list += torch.square(current_norm_list)
    else:
      raise ValueError('DP enabled but no grad_sample found')
  grad_norm_list = torch.sqrt(grad_norm_list)

  return grad_norm_list


def clip_grad_sample(params, clipping):
  """Clip the gradient of each example."""
  for p in params:
    if hasattr(p, 'grad_sample'):
      n = p.grad_sample.shape[0]
      break
  grad_norm_list = torch.zeros(n).cuda()
  for p in params:
    if hasattr(p, 'grad_sample'):
      flat_g = p.grad_sample.reshape(n, -1)
      current_norm_list = torch.norm(flat_g, dim=1)
      grad_norm_list += torch.square(current_norm_list)
  grad_norm_list = torch.sqrt(grad_norm_list)
  scaling = clipping / grad_norm_list
  scaling[scaling > 1] = 1

  for p in params:
    if hasattr(p, 'grad_sample'):
      p_dim = len(p.shape)
      scaling = scaling.view([n] + [1] * p_dim)
      p.grad_sample *= scaling

  return grad_norm_list


def get_epsilon_prv(noise_multiplier, delta, steps, sampling_prob):
  """Get the epsilon for running dp-sgd."""
  prv = PoissonSubsampledGaussianMechanism(
      noise_multiplier=noise_multiplier, sampling_probability=sampling_prob
  )
  accountant = PRVAccountant(
      prvs=[prv],
      max_self_compositions=[steps],
      eps_error=0.1,
      delta_error=delta / 10,
  )
  _, _, eps_up = accountant.compute_epsilon(
      delta=delta, num_self_compositions=[steps]
  )
  return eps_up


def search_for_sigma(
    current_sigma, eps, delta, steps, sampling_prob, precision
):
  """Search for the sigma that gives the closest epsilon to the target."""
  while current_sigma > 0:
    current_eps = get_epsilon_prv(current_sigma, delta, steps, sampling_prob)
    if current_eps < eps:
      current_sigma -= precision
    else:
      current_sigma += precision
      return current_sigma
  return precision


def get_noise_multiplier(eps, delta, steps, sampling_prob, init_sigma=10):
  """Get the noise multiplier for running dp-sgd."""
  current_sigma = init_sigma
  current_sigma = search_for_sigma(
      current_sigma, eps, delta, steps, sampling_prob, precision=1
  )
  current_sigma = search_for_sigma(
      current_sigma, eps, delta, steps, sampling_prob, precision=0.1
  )
  current_sigma = search_for_sigma(
      current_sigma, eps, delta, steps, sampling_prob, precision=0.01
  )

  if current_sigma == 0.01:
    raise ValueError(
        'Cannot find a valid sigma for the given epsilon and delta.'
    )

  return current_sigma


# eps = 5.94
# delta = 5e-7
# steps = int(10/(4096/180000))
# sampling_prob = 4096/180000


def clip_and_accumulate_perexample_grads(
    require_grad_params, accumulated_steps, clip_norm, accelerator
):
  """Clip and accumulate per-example gradients."""
  if accelerator.scaler is not None:
    # get the scale of mixed precision training
    mixed_precision_scale = accelerator.scaler.get_scale()
  else:
    mixed_precision_scale = 1.0
  for p in require_grad_params:
    if hasattr(p, 'grad_sample'):
      # convert to fp32
      p.grad_sample = p.grad_sample.float()
      # undo mixed precision scaling
      p.grad_sample /= mixed_precision_scale
    else:
      raise RuntimeError('DP enabled but no grad_sample found')

  # clip gradients
  grad_norms = clip_grad_sample(require_grad_params, clip_norm)

  # accumulate gradients
  for p in require_grad_params:
    if hasattr(p, 'grad_sample'):
      if accumulated_steps == 0:
        p.accumulated_grad = torch.sum(p.grad_sample, dim=0)
      else:
        p.accumulated_grad += torch.sum(p.grad_sample, dim=0)
      p.grad_sample = None

  return grad_norms
