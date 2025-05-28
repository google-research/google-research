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

"""Tests for checkpoint."""

import os

from absl.testing import absltest

 import resources
from scaling_transformer_inference_efficiency import checkpoint

_TOY_HPARAMS = checkpoint.HParams(
    layers=3,
    embed=128,
    ff=256,
    heads=2,
    qkv=32,
    max_len=128,
    vocab=32128,
)

# Test relies on internal checkpoints


class CheckpointTest(absltest.TestCase):

  def test_init_zero(self):
    c = checkpoint.Checkpoint.init_zero(_TOY_HPARAMS)
    shapes = checkpoint.Checkpoint.make_shaped_arrays(_TOY_HPARAMS)
    self.assertEqual(c.q_wi.shape, shapes.q_wi.shape)
    self.assertEqual(c.q_wi.dtype, shapes.q_wi.dtype)


if __name__ == '__main__':
  absltest.main()
