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

"""Keras-based attention layer."""
import tensorflow as tf

EinsumDense = tf.keras.layers.experimental.EinsumDense


class LightMultiHeadAttention(tf.keras.layers.MultiHeadAttention):

  def call(self, query, value, key=None, attention_mask=None,
           return_attention_scores=False, shared_attention_scores=None):

    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    # `value_tensor` = [B, S, N, H]
    value_tensor = self._value_dense(value)

    if shared_attention_scores is None:
      attention_output = value_tensor
    else:
      attention_output = tf.einsum(self._combine_equation,
                                   shared_attention_scores, value_tensor)

    attention_scores = shared_attention_scores

    attention_output = self._output_dense(attention_output)

    if return_attention_scores:
      return attention_output, attention_scores
    return attention_output
