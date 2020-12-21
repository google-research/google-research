# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: disable=invalid-name, unused-argument
"""Utilities for numerator and denominator computation in the prefix sum in an autoregressive Performer."""
import torch

_ITER_CHUNK_SIZE = 64


class _NumIter(torch.autograd.Function):
  """Custom gradient for numerator computation in prefix sum."""

  @staticmethod
  def forward(ctx, qs, ks, vs, sums):
    result = []

    for start_index in range(0, qs.shape[0], _ITER_CHUNK_SIZE):
      end_index = min(qs.shape[0], start_index + _ITER_CHUNK_SIZE)
      chunk = torch.einsum('sijk,sijl->sijkl', ks[start_index:end_index],
                           vs[start_index:end_index])
      chunk = sums[None, Ellipsis] + torch.cumsum(chunk, 0)
      sums = chunk[-1]
      result_elem = torch.einsum('sijkl,sijk->sijl', chunk,
                                 qs[start_index:end_index])
      result.append(result_elem)

    result = torch.cat(result, 0)
    ctx.save_for_backward(qs, ks, vs, sums)
    return result, sums

  @staticmethod
  def backward(ctx, res_grad, grads):
    qs, ks, vs, sums = ctx.saved_tensors

    q_grads = []
    k_grads = []
    v_grads = []

    inverse_index = torch.arange(qs.shape[0] - 1, -1, -1, device=qs.device)

    res_grad = res_grad[inverse_index]
    qs = qs[inverse_index]
    ks = ks[inverse_index]
    vs = vs[inverse_index]

    for start_index in range(0, qs.shape[0], _ITER_CHUNK_SIZE):
      end_index = min(qs.shape[0], start_index + _ITER_CHUNK_SIZE)
      chunk = torch.einsum('sijk,sijl->sijkl', ks[start_index:end_index - 1],
                           vs[start_index:end_index - 1])
      chunk = torch.cat([torch.zeros_like(sums[None, Ellipsis]), chunk], 0)
      chunk = sums[None, Ellipsis] - torch.cumsum(chunk, 0)
      sums = chunk[-1] - torch.einsum('ijk,ijl->ijkl', ks[end_index - 1],
                                      vs[end_index - 1])
      q_grads.append(
          torch.einsum('sijkl,sijl->sijk', chunk,
                       res_grad[start_index:end_index]))
      grad_chunk = torch.einsum('sijk,sijl->sijkl', qs[start_index:end_index],
                                res_grad[start_index:end_index])
      grad_chunk = grads[None, Ellipsis] + torch.cumsum(grad_chunk, 0)
      grads = grad_chunk[-1]
      k_grads.append(
          torch.einsum('sijkl,sijl->sijk', grad_chunk,
                       vs[start_index:end_index]))
      v_grads.append(
          torch.einsum('sijkl,sijk->sijl', grad_chunk,
                       ks[start_index:end_index]))

    q_grads = torch.cat(q_grads, 0)[inverse_index]
    k_grads = torch.cat(k_grads, 0)[inverse_index]
    v_grads = torch.cat(v_grads, 0)[inverse_index]
    return q_grads, k_grads, v_grads, grads


num_iter = _NumIter.apply


class _DenIter(torch.autograd.Function):
  """Custom gradient for denominator computation in prefix sum."""

  @staticmethod
  def forward(ctx, qs, ks, sums):
    result = []

    for start_index in range(0, qs.shape[0], _ITER_CHUNK_SIZE):
      end_index = min(qs.shape[0], start_index + _ITER_CHUNK_SIZE)
      chunk = ks[start_index:end_index]
      chunk = sums[None, Ellipsis] + torch.cumsum(chunk, 0)
      sums = chunk[-1]
      result_elem = (qs[start_index:end_index] * chunk).sum(3)
      result.append(result_elem)

    result = torch.cat(result, 0)
    ctx.save_for_backward(qs, ks, sums)
    return result, sums

  @staticmethod
  def backward(ctx, res_grad, k_grad):
    qs, ks, sums = ctx.saved_tensors

    q_grads = []
    k_grads = []

    inverse_index = torch.arange(qs.shape[0] - 1, -1, -1, device=qs.device)

    res_grad = res_grad[inverse_index]
    qs = qs[inverse_index]
    ks = ks[inverse_index]

    for start_index in range(0, qs.shape[0], _ITER_CHUNK_SIZE):
      end_index = min(qs.shape[0], start_index + _ITER_CHUNK_SIZE)

      chunk = ks[start_index:end_index - 1]
      chunk = torch.cat([torch.zeros_like(sums[None, Ellipsis]), chunk], axis=0)
      chunk = sums[None, Ellipsis] - torch.cumsum(chunk, 0)
      sums = chunk[-1] - ks[end_index - 1]
      q_grads.append(
          torch.einsum('sijk,sij->sijk', chunk,
                       res_grad[start_index:end_index]))
      k_grad_chunk = torch.einsum('sijk,sij->sijk', qs[start_index:end_index],
                                  res_grad[start_index:end_index])
      k_grad_chunk = k_grad[None, Ellipsis] + torch.cumsum(k_grad_chunk, 0)
      k_grad = k_grad_chunk[-1]
      k_grads.append(k_grad_chunk)

    q_grads = torch.cat(q_grads, 0)[inverse_index]
    k_grads = torch.cat(k_grads, 0)[inverse_index]
    return q_grads, k_grads, k_grad


den_iter = _DenIter.apply


def num_reverse_sums_iter(qs, ks, vs, sums):
  for start_index in range(0, qs.shape[0], _ITER_CHUNK_SIZE):
    end_index = min(qs.shape[0], start_index + _ITER_CHUNK_SIZE)
    chunk = torch.einsum('sijk,sijl->sijkl', ks[start_index:end_index],
                         vs[start_index:end_index])
    sums = sums - chunk.sum(0)
  return sums


def den_reverse_sums_iter(qs, ks, sums):
  for start_index in range(0, qs.shape[0], _ITER_CHUNK_SIZE):
    end_index = min(qs.shape[0], start_index + _ITER_CHUNK_SIZE)
    chunk = ks[start_index:end_index]
    sums = sums - chunk.sum(0)
  return sums


def num_ps(qs, ks, vs, init_sums, on_parallel):
  R = torch.einsum('sijk,sijl->sijkl', ks, vs)

  if on_parallel:
    R = prefix_sum(R)
  else:
    R = torch.cumsum(R, 0)

  R = R + init_sums[None, Ellipsis]
  return torch.einsum('sijkl,sijk->sijl', R, qs), R[-1]


def den_ps(qs, ks, init_sums, on_parallel):

  if on_parallel:
    s = prefix_sum(ks)
  else:
    s = torch.cumsum(ks, 0)

  s = s + init_sums[None, Ellipsis]
  return torch.einsum('sijk,sijk->sij', s, qs), s[-1]


def num_reverse_sums_ps(qs, ks, vs, final_sums):
  R = torch.einsum('sijk,sijl->sijkl', ks, vs)
  return final_sums - R.sum(0)


def den_reverse_sums_ps(qs, ks, final_sums):
  return final_sums - ks.sum(0)


def prefix_sum(x):
  """Computes prefix sum for autoregressive mode."""
  if x.shape[0] == 1:
    return x

  if x.shape[0] % 2 == 1:
    z = x[:-1]
    z_ps = prefix_sum(z)
    return torch.cat([z_ps, z_ps[-1:] + x[-1:]], 0)

  y = x[::2] + x[1::2]
  y_ps = prefix_sum(y)

  result = torch.cat([(y_ps - x[1::2])[:, None], y_ps[:, None]], 1)
  result = result.reshape(x.shape)

  return result
