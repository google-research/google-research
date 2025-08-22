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

"""This file contains the TiDE model  code."""

from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

EPS = 1e-7

train_loss = keras.losses.MeanSquaredError()


class MLPResidual(keras.layers.Layer):
  """Simple one hidden state residual network."""

  def __init__(
      self, hidden_dim, output_dim, layer_norm=False, dropout_rate=0.0
  ):
    super(MLPResidual, self).__init__()
    self.lin_a = tf.keras.layers.Dense(
        hidden_dim,
        activation='relu',
    )
    self.lin_b = tf.keras.layers.Dense(
        output_dim,
        activation=None,
    )
    self.lin_res = tf.keras.layers.Dense(
        output_dim,
        activation=None,
    )
    if layer_norm:
      self.lnorm = tf.keras.layers.LayerNormalization()
    self.layer_norm = layer_norm
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs):
    """Call method."""
    h_state = self.lin_a(inputs)
    out = self.lin_b(h_state)
    out = self.dropout(out)
    res = self.lin_res(inputs)
    if self.layer_norm:
      return self.lnorm(out + res)
    return out + res


def _make_dnn_residual(hidden_dims, layer_norm=False, dropout_rate=0.0):
  """Multi-layer DNN residual model."""
  if len(hidden_dims) < 2:
    return keras.layers.Dense(
        hidden_dims[-1],
        activation=None,
    )
  layers = []
  for i, hdim in enumerate(hidden_dims[:-1]):
    layers.append(
        MLPResidual(
            hdim,
            hidden_dims[i + 1],
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )
    )
  return keras.Sequential(layers)


class TideModel(keras.Model):
  """Main class for multi-scale DNN model."""

  def __init__(
      self,
      model_config,
      pred_len,
      cat_sizes,
      num_ts,
      transform=False,
      cat_emb_size=4,
      layer_norm=False,
      dropout_rate=0.0,
  ):
    """Tide model.

    Args:
      model_config: configurations specific to the model.
      pred_len: prediction horizon length.
      cat_sizes: number of categories in each categorical covariate.
      num_ts: number of time-series in the dataset
      transform: apply reversible transform or not.
      cat_emb_size: embedding size of categorical variables.
      layer_norm: use layer norm or not.
      dropout_rate: level of dropout.
    """
    super().__init__()
    self.model_config = model_config
    self.transform = transform
    if self.transform:
      self.affine_weight = self.add_weight(
          name='affine_weight',
          shape=(num_ts,),
          initializer='ones',
          trainable=True,
      )

      self.affine_bias = self.add_weight(
          name='affine_bias',
          shape=(num_ts,),
          initializer='zeros',
          trainable=True,
      )
    self.pred_len = pred_len
    self.encoder = _make_dnn_residual(
        model_config.get('hidden_dims'),
        layer_norm=layer_norm,
        dropout_rate=dropout_rate,
    )
    self.decoder = _make_dnn_residual(
        model_config.get('hidden_dims')[:-1]
        + [
            model_config.get('decoder_output_dim') * self.pred_len,
        ],
        layer_norm=layer_norm,
        dropout_rate=dropout_rate,
    )
    self.linear = tf.keras.layers.Dense(
        self.pred_len,
        activation=None,
    )
    self.time_encoder = _make_dnn_residual(
        model_config.get('time_encoder_dims'),
        layer_norm=layer_norm,
        dropout_rate=dropout_rate,
    )
    self.final_decoder = MLPResidual(
        hidden_dim=model_config.get('final_decoder_hidden'),
        output_dim=1,
        layer_norm=layer_norm,
        dropout_rate=dropout_rate,
    )
    self.cat_embs = []
    for cat_size in cat_sizes:
      self.cat_embs.append(
          tf.keras.layers.Embedding(input_dim=cat_size, output_dim=cat_emb_size)
      )
    self.ts_embs = tf.keras.layers.Embedding(input_dim=num_ts, output_dim=16)

  @tf.function
  def _assemble_feats(self, feats, cfeats):
    """assemble all features."""
    all_feats = [feats]
    for i, emb in enumerate(self.cat_embs):
      all_feats.append(tf.transpose(emb(cfeats[i, :])))
    return tf.concat(all_feats, axis=0)

  @tf.function
  def call(self, inputs):
    """Call function that takes in a batch of training data and features."""
    past_data = inputs[0]
    future_features = inputs[1]
    bsize = past_data[0].shape[0]
    tsidx = inputs[2]
    past_feats = self._assemble_feats(past_data[1], past_data[2])
    future_feats = self._assemble_feats(future_features[0], future_features[1])
    past_ts = past_data[0]
    if self.transform:
      affine_weight = tf.gather(self.affine_weight, tsidx)
      affine_bias = tf.gather(self.affine_bias, tsidx)
      batch_mean = tf.math.reduce_mean(past_ts, axis=1)
      batch_std = tf.math.reduce_std(past_ts, axis=1)
      batch_std = tf.where(
          tf.math.equal(batch_std, 0.0), tf.ones_like(batch_std), batch_std
      )
      past_ts = (past_ts - batch_mean[:, None]) / batch_std[:, None]
      past_ts = affine_weight[:, None] * past_ts + affine_bias[:, None]
    encoded_past_feats = tf.transpose(
        self.time_encoder(tf.transpose(past_feats))
    )
    encoded_future_feats = tf.transpose(
        self.time_encoder(tf.transpose(future_feats))
    )
    enc_past = tf.repeat(tf.expand_dims(encoded_past_feats, axis=0), bsize, 0)
    enc_past = tf.reshape(enc_past, [bsize, -1])
    enc_fut = tf.repeat(
        tf.expand_dims(encoded_future_feats, axis=0), bsize, 0
    )  # batch x fdim x H
    enc_future = tf.reshape(enc_fut, [bsize, -1])
    residual_out = self.linear(past_ts)
    ts_embs = self.ts_embs(tsidx)
    encoder_input = tf.concat([past_ts, enc_past, enc_future, ts_embs], axis=1)
    encoding = self.encoder(encoder_input)
    decoder_out = self.decoder(encoding)
    decoder_out = tf.reshape(
        decoder_out, [bsize, -1, self.pred_len]
    )  # batch x d x H
    final_in = tf.concat([decoder_out, enc_fut], axis=1)
    out = self.final_decoder(tf.transpose(final_in, (0, 2, 1)))  # B x H x 1
    out = tf.squeeze(out, axis=-1)
    out += residual_out
    if self.transform:
      out = (out - affine_bias[:, None]) / (affine_weight[:, None] + EPS)
      out = out * batch_std[:, None] + batch_mean[:, None]
    return out

  @tf.function
  def train_step(self, past_data, future_features, ytrue, tsidx, optimizer):
    """One step of training."""
    with tf.GradientTape() as tape:
      all_preds = self((past_data, future_features, tsidx), training=True)
      loss = train_loss(ytrue, all_preds)

    grads = tape.gradient(loss, self.trainable_variables)
    optimizer.apply_gradients(zip(grads, self.trainable_variables))
    return loss

  def get_all_eval_data(self, data, mode, num_split=1):
    y_preds = []
    y_trues = []
    all_test_loss = 0
    all_test_num = 0
    idxs = np.arange(0, self.pred_len, self.pred_len // num_split).tolist() + [
        self.pred_len
    ]
    for i in range(len(idxs) - 1):
      indices = (idxs[i], idxs[i + 1])
      logging.info('Getting data for indices: %s', indices)
      all_y_true, all_y_pred, test_loss, test_num = (
          self.get_eval_data_for_split(data, mode, indices)
      )
      y_preds.append(all_y_pred)
      y_trues.append(all_y_true)
      all_test_loss += test_loss
      all_test_num += test_num
    return np.hstack(y_preds), np.hstack(y_trues), all_test_loss / all_test_num

  def get_eval_data_for_split(self, data, mode, indices):
    iterator = data.tf_dataset(mode=mode)

    all_y_true = None
    all_y_pred = None

    def set_or_concat(a, b):
      if a is None:
        return b
      return tf.concat((a, b), axis=1)

    all_test_loss = 0
    all_test_num = 0
    ts_count = 0
    ypreds = []
    ytrues = []
    for all_data in tqdm(iterator):
      past_data = all_data[:3]
      future_features = all_data[4:6]
      y_true = all_data[3]
      tsidx = all_data[-1]
      all_preds = self((past_data, future_features, tsidx), training=False)
      y_pred = all_preds
      y_pred = y_pred[:, 0 : y_true.shape[1]]
      id1 = indices[0]
      id2 = min(indices[1], y_true.shape[1])
      y_pred = y_pred[:, id1:id2]
      y_true = y_true[:, id1:id2]
      loss = train_loss(y_true, y_pred)
      all_test_loss += loss
      all_test_num += 1
      ts_count += y_true.shape[0]
      ypreds.append(y_pred)
      ytrues.append(y_true)
      if ts_count >= len(data.ts_cols):
        ts_count = 0
        ypreds = tf.concat(ypreds, axis=0)
        ytrues = tf.concat(ytrues, axis=0)
        all_y_true = set_or_concat(all_y_true, ytrues)
        all_y_pred = set_or_concat(all_y_pred, ypreds)
        ypreds = []
        ytrues = []
    return (
        all_y_true.numpy(),
        all_y_pred.numpy(),
        all_test_loss.numpy(),
        all_test_num,
    )

  def eval(self, data, mode, num_split=1):
    all_y_pred, all_y_true, test_loss = self.get_all_eval_data(
        data, mode, num_split
    )

    result_dict = {}
    for metric in METRICS:
      eval_fn = METRICS[metric]
      result_dict[metric] = np.float64(eval_fn(all_y_pred, all_y_true))

    logging.info(result_dict)
    logging.info('Loss: %f', test_loss)

    return (
        result_dict,
        (all_y_pred, all_y_true),
        test_loss,
    )


def mape(y_pred, y_true):
  abs_diff = np.abs(y_pred - y_true).flatten()
  abs_val = np.abs(y_true).flatten()
  idx = np.where(abs_val > EPS)
  mpe = np.mean(abs_diff[idx] / abs_val[idx])
  return mpe


def mae_loss(y_pred, y_true):
  return np.abs(y_pred - y_true).mean()


def wape(y_pred, y_true):
  abs_diff = np.abs(y_pred - y_true)
  abs_val = np.abs(y_true)
  wpe = np.sum(abs_diff) / (np.sum(abs_val) + EPS)
  return wpe


def smape(y_pred, y_true):
  abs_diff = np.abs(y_pred - y_true)
  abs_mean = (np.abs(y_true) + np.abs(y_pred)) / 2
  smpe = np.mean(abs_diff / (abs_mean + EPS))
  return smpe


def rmse(y_pred, y_true):
  return np.sqrt(np.square(y_pred - y_true).mean())


def nrmse(y_pred, y_true):
  mse = np.square(y_pred - y_true)
  return np.sqrt(mse.mean()) / np.abs(y_true).mean()


METRICS = {
    'mape': mape,
    'wape': wape,
    'smape': smape,
    'nrmse': nrmse,
    'rmse': rmse,
    'mae': mae_loss,
}
