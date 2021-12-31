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

"""Gives the interface of the rans coder."""


class Bitstream():
  """Bitstream object that manages encoding/decoding."""

  def __init__(self, scale_bits):
    self.scale_bits = scale_bits

  def encode_cat(self, x, probs):
    raise NotImplementedError

  def decode_cat(self, probs):
    raise NotImplementedError

  def __len__(self):
    """Should return the length of the bitstream (in bits)."""
    raise NotImplementedError
