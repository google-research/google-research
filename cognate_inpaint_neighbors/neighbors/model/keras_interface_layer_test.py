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

"""Tests for KerasInterfaceLayer."""

from keras_interface_layer import KerasInterfaceLayer

from lingvo import compat as tf
from lingvo.core import test_utils


class KerasInterfaceLayerTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    self.params = KerasInterfaceLayer.Params()
    self.params.name = "keras_interface_layer"

  def testKerasInterfaceLayer(self):
    layer = self.params.Instantiate()
    layer.dense = layer.AddVariable(
        tf.keras.layers.Dense(20), input_shape=(10, 20))
    layer.gru = layer.AddVariable(
        tf.keras.layers.GRU(
            256,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"),
        input_shape=(10, 20, 30))
    layer.emb = layer.AddVariable(
        tf.keras.layers.Embedding(20, 30), input_shape=(10,))
    layer.emb2 = layer.AddVariable(
        tf.keras.layers.Embedding(20, 30),
        input_shape=(10,),
        keras_scope="boerenkaas")
    layer.var = layer.AddVariable(tf.Variable([[1.], [2.]]))
    layer.var2 = layer.AddVariable(tf.Variable([[1.], [2.]], name="foo"))
    self.assertSameElements(
        layer.activated_var_names,
        [
            "bias",  # From Dense
            "embeddings",
            "boerenkaas/embeddings",
            "foo",
            "gru_cell/kernel",
            "gru_cell/recurrent_kernel",
            "gru_cell/bias",
            "kernel",  # From Dense
            "Variable"
        ])
    # Verifying that these work as intended:
    _ = layer.dense(tf.zeros([10, 20]))
    _ = layer.emb(tf.zeros([10]))
    _ = layer.gru(tf.zeros([10, 20, 30]))


if __name__ == "__main__":
  tf.test.main()
