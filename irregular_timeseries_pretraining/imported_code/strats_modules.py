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

"""Modules copied directly from the original STraTS repo: https://github.com/sindhura97/STraTS/blob/main/strats_notebook.ipynb."""

from imported_code.smart_cond import smart_cond
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from tensorflow import nn
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import array_ops

# Note: "smart_cond" is a deprecated function from tensorflow;  as a workaround,
# we wget the module from a specific version of tensorflow and import it here
# (see README)


class CVE(Layer):
  """Source: https://github.com/sindhura97/STraTS/blob/main/strats_notebook.ipynb.

  Continuous value embedding layer, which maps continuous input data to
  output_dim dimensions
  """

  def __init__(self, hid_units, output_dim):
    self.hid_units = hid_units
    self.output_dim = output_dim
    super().__init__()

  def build(self, input_shape):
    self.w1 = self.add_weight(
        name='CVE_W1',
        shape=(1, self.hid_units),
        initializer='glorot_uniform',
        trainable=True,
    )
    self.b1 = self.add_weight(
        name='CVE_b1',
        shape=(self.hid_units,),
        initializer='zeros',
        trainable=True,
    )
    self.w2 = self.add_weight(
        name='CVE_W2',
        shape=(self.hid_units, self.output_dim),
        initializer='glorot_uniform',
        trainable=True,
    )
    super().build(input_shape)

  def call(self, x):
    x = K.expand_dims(x, axis=-1)
    x = K.dot(K.tanh(K.bias_add(K.dot(x, self.w1), self.b1)), self.w2)
    return x

  def compute_output_shape(self, input_shape):
    return input_shape + (self.output_dim,)


class Attention(Layer):
  """Source: https://github.com/sindhura97/STraTS/blob/main/strats_notebook.ipynb.

  Fusion self-attention layer - provides weighted average of representations
  across timesteps for final time series representation.
  """

  def __init__(self, hid_dim):
    self.hid_dim = hid_dim
    super().__init__()

  def build(self, input_shape):
    d = input_shape.as_list()[-1]
    self.w = self.add_weight(
        shape=(d, self.hid_dim),
        name='Att_W',
        initializer='glorot_uniform',
        trainable=True,
    )
    self.b = self.add_weight(
        shape=(self.hid_dim,), name='Att_b', initializer='zeros', trainable=True
    )
    self.u = self.add_weight(
        shape=(self.hid_dim, 1),
        name='Att_u',
        initializer='glorot_uniform',
        trainable=True,
    )
    super().build(input_shape)

  def call(self, x, mask, mask_value=-1e30):
    attn_weights = K.dot(K.tanh(K.bias_add(K.dot(x, self.w), self.b)), self.u)
    mask = K.expand_dims(mask, axis=-1)
    attn_weights = mask * attn_weights + (1 - mask) * mask_value
    attn_weights = K.softmax(attn_weights, axis=-2)
    return attn_weights

  def compute_output_shape(self, input_shape):
    return input_shape[:-1] + (1,)


class Transformer(Layer):
  """Source: https://github.com/sindhura97/STraTS/blob/main/strats_notebook.ipynb.

  Defines transformer architecture with N blocks of MHA attention with h heads.
  """

  def __init__(self, num_blocks=2, h=8, dk=None, dv=None, dff=None, dropout=0):
    self.num_heads, self.h, self.dk, self.dv, self.dff, self.dropout = (
        num_blocks,
        h,
        dk,
        dv,
        dff,
        dropout,
    )
    self.epsilon = K.epsilon() * K.epsilon()
    super().__init__()

  def build(self, input_shape):
    d = input_shape.as_list()[-1]
    if self.dk is None:
      self.dk = d // self.h
    if self.dv is None:
      self.dv = d // self.h
    if self.dff is None:
      self.dff = 2 * d
    self.wq = self.add_weight(
        shape=(self.num_heads, self.h, d, self.dk),
        name='Wq',
        initializer='glorot_uniform',
        trainable=True,
    )
    self.wk = self.add_weight(
        shape=(self.num_heads, self.h, d, self.dk),
        name='Wk',
        initializer='glorot_uniform',
        trainable=True,
    )
    self.wv = self.add_weight(
        shape=(self.num_heads, self.h, d, self.dv),
        name='Wv',
        initializer='glorot_uniform',
        trainable=True,
    )
    self.wo = self.add_weight(
        shape=(self.num_heads, self.dv * self.h, d),
        name='Wo',
        initializer='glorot_uniform',
        trainable=True,
    )
    self.w1 = self.add_weight(
        shape=(self.num_heads, d, self.dff),
        name='W1',
        initializer='glorot_uniform',
        trainable=True,
    )
    self.b1 = self.add_weight(
        shape=(self.num_heads, self.dff),
        name='b1',
        initializer='zeros',
        trainable=True,
    )
    self.w2 = self.add_weight(
        shape=(self.num_heads, self.dff, d),
        name='W2',
        initializer='glorot_uniform',
        trainable=True,
    )
    self.b2 = self.add_weight(
        shape=(self.num_heads, d),
        name='b2',
        initializer='zeros',
        trainable=True,
    )
    self.gamma = self.add_weight(
        shape=(2 * self.num_heads,),
        name='gamma',
        initializer='ones',
        trainable=True,
    )
    self.beta = self.add_weight(
        shape=(2 * self.num_heads,),
        name='beta',
        initializer='zeros',
        trainable=True,
    )
    super().build(input_shape)

  def call(self, x, mask, mask_value=-1e-30):
    mask = K.expand_dims(mask, axis=-2)
    for i in range(self.num_heads):
      # MHA
      mha_ops = []
      for j in range(self.h):
        q = K.dot(x, self.wq[i, j, :, :])
        k = K.permute_dimensions(K.dot(x, self.wk[i, j, :, :]), (0, 2, 1))
        v = K.dot(x, self.wv[i, j, :, :])
        a = K.batch_dot(q, k)
        # Mask unobserved steps.
        a = mask * a + (1 - mask) * mask_value

        # Mask for attention dropout.
        def dropped_a():
          dp_mask = K.cast(
              # pylint: disable=cell-var-from-loop
              (K.random_uniform(shape=array_ops.shape(a)) >= self.dropout),
              K.floatx(),
          )
          # pylint: disable=cell-var-from-loop
          return a * dp_mask + (1 - dp_mask) * mask_value

        # pylint: disable=cell-var-from-loop
        a = smart_cond(
            K.learning_phase(), dropped_a, lambda: array_ops.identity(a)
        )
        a = K.softmax(a, axis=-1)
        mha_ops.append(K.batch_dot(a, v))
      conc = K.concatenate(mha_ops, axis=-1)
      proj = K.dot(conc, self.wo[i, :, :])
      # Dropout.
      proj = smart_cond(
          K.learning_phase(),
          # pylint: disable=cell-var-from-loop
          lambda: array_ops.identity(nn.dropout(proj, rate=self.dropout)),
          lambda: array_ops.identity(proj),
      )
      # def drp(x):
      #   return array_ops.identity(nn.dropout(x, rate=self.dropout))
      # def notdrp(x):
      #   return array_ops.identity(x)
      # proj = smart_cond(
      #     K.learning_phase(),
      #     lambda: drp(proj),
      #     lambda: notdrp(proj),
      # )
      # Add & LN
      x = x + proj
      mean = K.mean(x, axis=-1, keepdims=True)
      variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
      std = K.sqrt(variance + self.epsilon)
      x = (x - mean) / std
      x = x * self.gamma[2 * i] + self.beta[2 * i]
      # FFN
      ffn_op = K.bias_add(
          K.dot(
              K.relu(K.bias_add(K.dot(x, self.w1[i, :, :]), self.b1[i, :])),
              self.w2[i, :, :],
          ),
          self.b2[
              i,
              :,
          ],
      )
      # Dropout.
      ffn_op = smart_cond(
          K.learning_phase(),
          # pylint: disable=cell-var-from-loop
          lambda: array_ops.identity(nn.dropout(ffn_op, rate=self.dropout)),
          lambda: array_ops.identity(ffn_op),
      )
      # Add & LN
      x = x + ffn_op
      mean = K.mean(x, axis=-1, keepdims=True)
      variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
      std = K.sqrt(variance + self.epsilon)
      x = (x - mean) / std
      x = x * self.gamma[2 * i + 1] + self.beta[2 * i + 1]
    return x

  def compute_output_shape(self, input_shape):
    return input_shape


def get_res(y_true, y_pred):
  """Source: https://github.com/sindhura97/STraTS/blob/main/strats_notebook.ipynb."""
  precision, recall, _ = precision_recall_curve(y_true, y_pred)
  pr_auc = auc(recall, precision)
  minrp = np.minimum(precision, recall).max()
  roc_auc = roc_auc_score(y_true, y_pred)
  return [roc_auc, pr_auc, minrp]
