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

"""Sparse/dense MobileNetV1 model configurations."""


class Config(object):
  """Config base class."""

  def __init__(self, fuse_bnbr=False):
    self.num_blocks = 13
    self.width = None
    self.block_nonzeros = None
    self.block_config = ["dense"] * self.num_blocks
    self.fuse_bnbr = fuse_bnbr
    self.num_classes = 1000


class W14(Config):
  """Width 1.4."""

  def __init__(self, fuse_bnbr=False):
    super(W14, self).__init__(fuse_bnbr)
    self.width = 1.4


class W12(Config):
  """Width 1.2."""

  def __init__(self, fuse_bnbr=False):
    super(W12, self).__init__(fuse_bnbr)
    self.width = 1.2


class W10(Config):
  """Width 1.0."""

  def __init__(self, fuse_bnbr=False):
    super(W10, self).__init__(fuse_bnbr)
    self.width = 1.0


class W18S90D0(Config):
  """Width 1.8, sparsity 90%, dense first block."""

  def __init__(self, fuse_bnbr=False):
    super(W18S90D0, self).__init__(fuse_bnbr)
    self.block_nonzeros = [
        6440,
        2668,
        5382,
        10765,
        21530,
        42688,
        84640,
        84640,
        84640,
        84640,
        84640,
        169280,
        338560,
    ]
    self.block_config = ["dense"] + ["sparse"] * (self.num_blocks - 1)
    self.width = 1.8


class W17S90D0(W18S90D0):
  """Width 1.7, sparsity 90%, dense first block."""

  def __init__(self, fuse_bnbr=False):
    super(W17S90D0, self).__init__(fuse_bnbr)
    self.block_nonzeros = [
        6048,
        2333,
        4666,
        9331,
        18662,
        37670,
        76038,
        76038,
        76038,
        76038,
        76038,
        152077,
        304154,
    ]
    self.width = 1.7


class W16S90D0(W18S90D0):
  """Width 1.6, sparsity 90%, dense first block."""

  def __init__(self, fuse_bnbr=False):
    super(W16S90D0, self).__init__(fuse_bnbr)
    self.block_nonzeros = [
        4896,
        2122,
        4326,
        8486,
        16646,
        33293,
        66586,
        66586,
        66586,
        66586,
        66586,
        133824,
        268960,
    ]
    self.width = 1.6


class W15S90D0(W18S90D0):
  """Width 1.5, sparsity 90%, dense first block."""

  def __init__(self, fuse_bnbr=False):
    super(W15S90D0, self).__init__(fuse_bnbr)
    self.block_nonzeros = [
        4608,
        1843,
        3686,
        7373,
        14746,
        29491,
        58982,
        58982,
        58982,
        58982,
        58982,
        117965,
        235930,
    ]
    self.width = 1.5


class W14S90D0(W18S90D0):
  """Width 1.4, sparsity 90%, dense first block."""

  def __init__(self, fuse_bnbr=False):
    super(W14S90D0, self).__init__(fuse_bnbr)
    self.block_nonzeros = [
        4272,
        1566,
        3098,
        6336,
        12960,
        25920,
        51840,
        51840,
        51840,
        51840,
        51840,
        103104,
        205062,
    ]
    self.width = 1.4


class W13S90D0(W18S90D0):
  """Width 1.3, sparsity 90%, dense first block."""

  def __init__(self, fuse_bnbr=False):
    super(W13S90D0, self).__init__(fuse_bnbr)
    self.block_nonzeros = [
        3320,
        1394,
        2822,
        5645,
        11290,
        22310,
        44090,
        44090,
        44090,
        44090,
        44090,
        88179,
        176358,
    ]
    self.width = 1.3


def get_config(width, sparsity):
  """Getter for model configuration."""
  tag = "_".join(["mb"] + [str(int(x)) for x in [width * 10, sparsity * 100]])
  configs = {
      "mb_10_0": W10,
      "mb_12_0": W12,
      "mb_14_0": W14,
      "mb_13_90": W13S90D0,
      "mb_14_90": W14S90D0,
      "mb_15_90": W15S90D0,
      "mb_16_90": W16S90D0,
      "mb_17_90": W17S90D0,
      "mb_18_90": W18S90D0,
  }
  if tag not in configs:
    raise ValueError("Could not find config for width {}, sparsity {}.".format(
        width, sparsity))
  return configs[tag]
