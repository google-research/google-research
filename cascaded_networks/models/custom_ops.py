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

"""Custom BatchNorm operations.

BatchNorm is modified to handle temporal operations. Specifically, activation
statistics of each layer are changing over time, and to account for this we
add a time dimension to BatchNorm's running mean and std. The operation now
accepts a time argument which must be passed in the forward call.
"""
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def conv3x3(planes, out_planes, stride=1):
  """3x3 convolution with padding."""
  return nn.Conv2d(planes,
                   out_planes,
                   kernel_size=3,
                   stride=stride,
                   padding=1,
                   bias=False)


def conv1x1(planes, out_planes, stride=1):
  """1x1 convolution."""
  return nn.Conv2d(planes,
                   out_planes,
                   kernel_size=1,
                   stride=stride,
                   bias=False)


class _NormBase(nn.Module):
  """Common base of _InstanceNorm and _BatchNorm.

  Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d #pylint: disable=line-too-long
  """

  _version = 2
  __constants__ = [
      'bn_opts', 'track_running_stats', 'momentum', 'eps', 'num_features',
      'affine'
  ]

  def __init__(self,
               bn_opts,
               num_features,
               eps=1e-5,
               momentum=0.1,
               affine=True,
               track_running_stats=True):
    super(_NormBase, self).__init__()
    self.bn_opts = bn_opts
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_running_stats = track_running_stats
    if self.affine:
      if self.bn_opts['temporal_affine']:
        self.weight = Parameter(
            torch.Tensor(self.bn_opts['n_timesteps'], num_features))
        self.bias = Parameter(
            torch.Tensor(self.bn_opts['n_timesteps'], num_features))
      else:
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
    else:
      self.register_parameter('weight', None)
      self.register_parameter('bias', None)
    if self.track_running_stats:
      if self.bn_opts['temporal_stats']:
        self.register_buffer(
            'running_mean',
            torch.zeros(self.bn_opts['n_timesteps'], num_features))
        self.register_buffer(
            'running_var', torch.ones(self.bn_opts['n_timesteps'],
                                      num_features))
      else:
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

      self.register_buffer('num_batches_tracked',
                           torch.tensor(0, dtype=torch.long))
    else:
      self.register_parameter('running_mean', None)
      self.register_parameter('running_var', None)
      self.register_parameter('num_batches_tracked', None)

    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)
      self.num_batches_tracked.zero_()

  def reset_parameters(self):
    self.reset_running_stats()
    if self.affine:
      init.ones_(self.weight)
      init.zeros_(self.bias)

  def _check_input_dim(self, x):
    raise NotImplementedError

  def extra_repr(self):
    return ('{num_features}, eps={eps}, momentum={momentum}, affine={affine}, '
            'track_running_stats={track_running_stats}'.format(**self.__dict__))

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    version = local_metadata.get('version', None)

    if (version is None or version < 2) and self.track_running_stats:
      # at version 2: added num_batches_tracked buffer
      #               this should have a default value of 0
      num_batches_tracked_key = prefix + 'num_batches_tracked'
      if num_batches_tracked_key not in state_dict:
        state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

    super(_NormBase,
          self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)


class _BatchNorm(_NormBase):
  """BatchNorm base class.

  Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d #pylint: disable=line-too-long
  """

  def __init__(self,  # pylint: disable=useless-super-delegation
               num_features,
               eps=1e-5,
               momentum=0.1,
               affine=True,
               track_running_stats=True):
    super(_BatchNorm, self).__init__(num_features, eps, momentum, affine,
                                     track_running_stats)

  def forward(self, x, t):
    # Limit t to max
    t = min(t, self.bn_opts['n_timesteps'] - 1)

    # Check input dim
    self._check_input_dim(x)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
      exponential_average_factor = 0.0
    else:
      exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
      if self.num_batches_tracked is not None:
        self.num_batches_tracked = self.num_batches_tracked + 1
        if self.momentum is None:  # use cumulative moving average
          exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
          exponential_average_factor = self.momentum

    # Potential time-dependent params
    running_mean = self.running_mean
    running_var = self.running_var
    weight = self.weight
    bias = self.bias

    # Extract time-dependent params where appropriate
    if self.bn_opts['temporal_stats']:
      running_mean = running_mean[t]
      running_var = running_var[t]

    if self.bn_opts['temporal_affine']:
      weight = weight[t]
      bias = bias[t]

    return F.batch_norm(x, running_mean, running_var, weight, bias,
                        self.training or not self.track_running_stats,
                        exponential_average_factor, self.eps)


class BatchNorm2d(_BatchNorm):
  """BatchNorm 2D class.

  Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d #pylint: disable=line-too-long
  """

  def _check_input_dim(self, x):
    if x.dim() != 4:
      raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))
