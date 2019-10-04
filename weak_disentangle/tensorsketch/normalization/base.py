# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# python3
"""Normalization base modules."""

# pylint: disable=g-importing-member, g-bad-import-order
from weak_disentangle.tensorsketch.modules.base import Module


class Norm(Module):
  """Abstract class for normalization modules."""

  NAME = "norm"

  @staticmethod
  def normalize(*inputs):
    raise NotImplementedError("Implement normalize.")

  @staticmethod
  def add(module, norm, use_out_hook=True):
    hooks = module.select_hooks_dict(use_out_hook)
    assert not hasattr(module, norm.NAME), (
        "{} already in module attributes".format(norm.NAME))
    assert norm.NAME not in hooks, (
        "{} already in module hooks".format(norm.NAME))

    # pylint: disable=unused-argument
    def hook(self, *inputs):
      return norm(*inputs)

    setattr(module, norm.NAME, norm)
    hooks.update({norm.NAME: hook})

  @staticmethod
  def remove(module, norm_name, use_out_hook):
    norm = getattr(module, norm_name)
    hooks = module.select_hooks_dict(use_out_hook)

    delattr(module, norm.NAME)
    del hooks[norm.NAME]


class KernelNorm(Module):
  """Abstract class for kernel normalization modules."""

  NAME = "kernel_norm"

  @staticmethod
  def normalize(*inputs):
    raise NotImplementedError("Implement normalize.")

  @staticmethod
  def add(module, kn):
    assert not hasattr(module, kn.NAME), (
        "{} already in module attributes".format(kn.NAME))
    assert kn.NAME not in module.kernel_normalizers, (
        "{} already in module.kernel_normalizers".format(kn.NAME))
    setattr(module, kn.NAME, kn)
    module.kernel_normalizers.update({kn.NAME: kn})

  @staticmethod
  def remove(module, kn_name):
    delattr(module, kn_name)
    del module.kernel_normalizers[kn_name]
