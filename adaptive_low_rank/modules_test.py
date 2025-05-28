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

# Copyright 2024 Google LLC
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

import modules
import tensorflow as tf


class AdaptiveModuleTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # n represents number of tokens, d the latent dimension.
    equation = "...nd,dc->...nc"
    self.batch_size = 8
    self.n_tokens = 32
    self.hidden_dim = 16
    # There are two non-batch dimensions.
    self.output_shape = (None, self.hidden_dim)
    dense_layer = tf.keras.layers.EinsumDense(
        equation, self.output_shape, trainable=False
    )
    self.inputs = tf.random.normal(
        (self.batch_size, self.n_tokens, self.hidden_dim)
    )

    dense_layer(self.inputs)
    rank = 2
    self.adaptive_layer = modules.AdaptiveEinsumDense(dense_layer, rank)

  def test_adaptive_einsum_dense(
      self,
  ):
    output = self.adaptive_layer(self.inputs)

    with self.subTest("Test last dimension."):
      self.assertEqual(
          output.shape[-1],
          self.hidden_dim,
          msg="Last dimension must equal hidden dimension.",
      )

    with self.subTest("Test first dimension."):
      self.assertEqual(
          output.shape[0],
          self.batch_size,
          msg="First dimension must equal batch size.",
      )


if __name__ == "__main__":
  tf.test.main()
