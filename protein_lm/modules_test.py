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

# Lint as: python3
"""Tests for flax modules."""

import functools
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from protein_lm import domains
from protein_lm import models
from protein_lm import modules

lm_cls = functools.partial(
    models.FlaxLM,
    num_layers=1,
    num_heads=1,
    emb_dim=64,
    mlp_dim=64,
    qkv_dim=64)


class ModulesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (modules.AddLearnedPositionalEncodings,),
      (modules.AddSinusoidalPositionalEncodings,))
  def test_positional_encodings(self, positional_encoding_module):
    """Tests that the model runs with both types of positional encodings."""
    domain = domains.FixedLengthDiscreteDomain(vocab_size=2, length=2)
    lm = lm_cls(domain=domain,
                positional_encoding_module=positional_encoding_module)
    lm.sample(1)


if __name__ == '__main__':
  tf.test.main()
