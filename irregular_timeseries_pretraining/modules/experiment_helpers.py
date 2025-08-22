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

"""Contains many helper functions for experiments, including loss functions, callbacks, data loading."""
import ast
import copy
import gc
import json
import os
from imported_code.strats_modules import Attention
from imported_code.strats_modules import CVE
from imported_code.strats_modules import Transformer
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Multiply


# Model training callbacks
class ClearMemory(Callback):
  """Callback to clear unused memory."""

  def on_epoch_end(self, epoch, logs=None):
    gc.collect()
    K.clear_session()


# Saving training logs
class CustomCallbackSupervised(Callback):
  """Callback for saving results for classfication task."""

  def __init__(
      self, training_data, validation_data, batch_size, tr_epochs_log_path
  ):
    self.train_x, self.train_y = training_data
    self.val_x, self.val_y = validation_data
    self.batch_size = batch_size
    self.tr_epochs_log_path = tr_epochs_log_path
    # pylint: disable=bad-super-call
    super(Callback, self).__init__()

  # pylint: disable=dangerous-default-value
  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model.predict(
        self.val_x, verbose=0, batch_size=self.batch_size
    )
    if isinstance(y_pred, list):
      y_pred = y_pred[0]
    precision, recall, _ = precision_recall_curve(self.val_y, y_pred)
    pr_auc = auc(recall, precision)
    if np.sum(self.val_y) > 0:
      roc_auc = roc_auc_score(self.val_y, y_pred)
    else:
      roc_auc = np.nan
    logs["custom_metric"] = pr_auc + roc_auc
    print("val_aucs:", pr_auc, roc_auc)

    tr_y_pred = self.model.predict(
        self.train_x, verbose=0, batch_size=self.batch_size
    )
    if isinstance(y_pred, list):
      tr_y_pred = tr_y_pred[0]
    tr_precision, tr_recall, _ = precision_recall_curve(self.train_y, tr_y_pred)
    tr_pr_auc = auc(tr_recall, tr_precision)
    if np.sum(self.train_y) > 0:
      tr_roc_auc = roc_auc_score(self.train_y, tr_y_pred)
    else:
      tr_roc_auc = np.nan
    with open(self.tr_epochs_log_path, "a") as f:
      f.write("{}, {}, {}, {}\n".format(tr_pr_auc, tr_roc_auc, pr_auc, roc_auc))


#  Defining model architecture
# Adapted from STraTS codebase:
# https://github.com/sindhura97/STraTS/blob/main/strats_notebook.ipynb
def build_model(
    static_dims, max_len, num_feats, d, blocks, he, dropout, pretraining=True
):
  """Defining model architecture."""

  # STRATS ENCODER:
  values = Input(shape=(max_len,))
  times = Input(shape=(max_len,))
  varis = Input(shape=(max_len,))
  varis_emb = Embedding(num_feats + 2, d)(varis)
  cve_units = int(np.sqrt(d))
  values_emb = CVE(cve_units, d)(values)
  times_emb = CVE(cve_units, d)(times)
  comb_emb = Add()([varis_emb, values_emb, times_emb])  # b, L, d
  padding_mask = Lambda(lambda x: K.clip(x, 0, 1))(varis)  # b, L
  val_mask = Lambda(
      # pylint: disable=g-long-lambda
      lambda x: tf.where(
          tf.greater(x, -99), tf.ones((max_len,)), tf.zeros((max_len,))
      )
  )(values)
  mask = Multiply()([padding_mask, val_mask])
  cont_emb = Transformer(
      blocks, he, dk=None, dv=None, dff=None, dropout=dropout
  )(comb_emb, mask=mask)

  # Decoders for masked sequence prediction:
  values_dec_hidden0 = Dense(d, activation="relu")(cont_emb)
  values_dec_hidden1 = Dense(d, activation="relu")(values_dec_hidden0)
  values_dec_hidden = Dense(cve_units, activation="relu")(values_dec_hidden1)
  values_out = Dense(1)(values_dec_hidden)

  # STraTS-like forecasting + binary prediction:
  demo = Input(shape=(static_dims,))
  demo_enc = Dense(2 * d, activation="tanh")(demo)
  demo_enc = Dense(d, activation="tanh")(demo_enc)
  attn_weights = Attention(2 * d)(cont_emb, mask=mask)
  fused_emb = Lambda(lambda x: K.sum(x[0] * x[1], axis=-2))(
      [cont_emb, attn_weights]
  )
  conc = Concatenate(axis=-1)([fused_emb, demo_enc])
  fore_op = Dense(num_feats)(conc)
  op = Dense(1, activation="sigmoid")(fore_op)
  model = Model([demo, times, values, varis], op)

  if pretraining:
    fore_encdec_model = Model(
        [demo, times, values, varis], [fore_op, values_out]
    )
    return [model, fore_encdec_model]
  return model


#  Loss functions


# MORTALITY LOSS
# Changed from original STraTS codebase so that it can be called with
# class_weights as an input
# original strats code needed it hardcoded based on data:
def classweighted_mortality_loss(class_weights):
  """Given set class weights, creates class-weighted loss function."""

  def tmp_mortality_loss(y_true, y_pred):
    sample_weights = (1 - y_true) * class_weights[0] + y_true * class_weights[1]
    bce = K.binary_crossentropy(y_true, y_pred)
    return K.mean(sample_weights * bce, axis=-1)

  return tmp_mortality_loss


### FORECASTING LOSS:
# Changed from STraTS codebase so that it can be called with a specific
# V rather than hardcoded
# pylint: disable=invalid-name
def forecast_loss_V(num_feats):
  """Generates a forecasting loss function based on the number of features expected."""

  def forecast_loss(y_true, y_pred):
    """Forecasting loss function (ignores missing values)."""

    return K.sum(
        y_true[:, num_feats:] * (y_true[:, :num_feats] - y_pred) ** 2, axis=-1
    )
  return forecast_loss


# pylint: disable=invalid-name
def masked_MSE_loss_SUM(seq_len):
  """Creates loss function for reconstruction task given a set sequence length."""

  # pylint: disable=invalid-name
  def masked_MSE_loss(y_true, y_pred):
    """Reconstruction loss function.

    Args:
      y_true:  #samples x 2*seqlen (first seqlen elements are true data,
      and second is the mask)
      y_pred:   #samples x seqlen x 1 (model's predicted output)
    Returns:
      loss_agg:  summed MSEs over predictions (excluding padding)
    """

    if len(y_true.shape) == 1:
      y_true_data = y_true[:seq_len]
      mask = tf.cast(y_true[seq_len:], tf.float64)
      y_pred_reshaped = tf.reshape(y_pred, y_true_data.shape)

    else:
      y_true_data = y_true[:, :seq_len]
      mask = tf.cast(y_true[:, seq_len:], tf.float64)
      y_pred_reshaped = tf.reshape(y_pred, (-1, seq_len))

    all_loss = K.square(y_true_data - y_pred_reshaped)
    loss_agg = K.sum(tf.multiply(tf.cast(all_loss, tf.float64), mask), axis=-1)
    return loss_agg

  return masked_MSE_loss


#  Processing hyperparameter settings
def strategy_dict_from_string(FT_save_string, args_dict_literaleval):
  tmp_args_dict = {}
  for elt in FT_save_string.split("~"):
    [name, val] = elt.split("-")
    if args_dict_literaleval[name] and val != "NA":
      tmp_args_dict[name] = ast.literal_eval(val)
    else:
      tmp_args_dict[name] = val
  return tmp_args_dict


def load_data(path_to_data, raw_times):
  """Loads data from path_to_data folder and normalizes times if applicable."""

  loaded_data = {}
  file_names = [
      "fore_train_ip",
      "fore_valid_ip",
      "train_ip",
      "valid_ip",
      "test_ip",
      "fore_train_op",
      "fore_valid_op",
      "train_op",
      "valid_op",
      "test_op",
  ]
  for key in file_names:
    with open(os.path.join(path_to_data, key + ".json"), "r") as openfile:
      loaded_data[key] = json.load(openfile)
  fore_train_ip = [np.array(x) for x in loaded_data["fore_train_ip"]]
  fore_valid_ip = [np.array(x) for x in loaded_data["fore_valid_ip"]]
  train_ip = [np.array(x) for x in loaded_data["train_ip"]]
  valid_ip = [np.array(x) for x in loaded_data["valid_ip"]]
  test_ip = [np.array(x) for x in loaded_data["test_ip"]]
  fore_train_op = np.array(loaded_data["fore_train_op"])
  fore_valid_op = np.array(loaded_data["fore_valid_op"])
  train_op = np.array(loaded_data["train_op"])
  valid_op = np.array(loaded_data["valid_op"])
  test_op = np.array(loaded_data["test_op"])
  del loaded_data

  if not raw_times:
    # default is False, so times usually WILL be normalized
    # compute mean and variance of times in training set while ignoring padding
    missing_idx = fore_train_ip[3] == 0
    tmp_times = copy.deepcopy(fore_train_ip[1])
    tmp_times[missing_idx] = np.nan
    time_mean = np.nanmean(tmp_times)
    time_stddev = np.nanstd(tmp_times)
    tmp_times = (tmp_times - time_mean) / time_stddev
    tmp_times[missing_idx] = 0
    fore_train_ip = [
        fore_train_ip[0],
        tmp_times,
        fore_train_ip[2],
        fore_train_ip[3],
    ]

    # normalize val set times
    missing_idx = fore_valid_ip[3] == 0
    tmp_times = copy.deepcopy(fore_valid_ip[1])
    tmp_times[missing_idx] = np.nan
    tmp_times = (tmp_times - time_mean) / time_stddev
    tmp_times[missing_idx] = 0
    fore_valid_ip = [
        fore_valid_ip[0],
        tmp_times,
        fore_valid_ip[2],
        fore_valid_ip[3],
    ]

    # normalize labeled datasets
    for tmp_ip in [train_ip, valid_ip, test_ip]:
      missing_idx = tmp_ip[3] == 0
      tmp_times = copy.deepcopy(tmp_ip[1])
      tmp_times[missing_idx] = np.nan
      tmp_times = (tmp_times - time_mean) / time_stddev
      tmp_times[missing_idx] = 0
      tmp_ip[1] = tmp_times
  else:
    time_mean = time_stddev = None

  return (
      fore_train_ip,
      fore_train_op,
      fore_valid_ip,
      fore_valid_op,
      train_ip,
      train_op,
      valid_ip,
      valid_op,
      test_ip,
      test_op,
      time_mean,
      time_stddev,
  )


def downsample_data(
    downsampled_frac,
    fore_train_ip,
    fore_train_op,
    fore_valid_ip,
    fore_valid_op,
    train_ip,
    train_op,
    valid_ip,
    valid_op,
    test_ip,
    test_op,
):
  """Downsamples the entire dataset for quick experimental testing."""

  np.random.seed(2023)
  tmp_tr_id = np.random.choice(
      len(fore_train_op),
      int(len(fore_train_op) * downsampled_frac),
      replace=False,
  )
  np.random.seed(2023)
  tmp_val_id = np.random.choice(
      len(fore_valid_op),
      int(len(fore_valid_op) * downsampled_frac),
      replace=False,
  )

  fore_train_ip = [x[tmp_tr_id] for x in fore_train_ip]
  fore_train_op = fore_train_op[tmp_tr_id]
  fore_valid_ip = [x[tmp_val_id] for x in fore_valid_ip]
  fore_valid_op = fore_valid_op[tmp_val_id]

  np.random.seed(2023)
  tmp_tr_id = np.random.choice(
      len(train_op), int(len(train_op) * downsampled_frac), replace=False
  )
  np.random.seed(2023)
  tmp_val_id = np.random.choice(
      len(valid_op), int(len(valid_op) * downsampled_frac), replace=False
  )
  np.random.seed(2023)
  tmp_test_id = np.random.choice(
      len(test_op), int(len(test_op) * downsampled_frac), replace=False
  )

  train_ip = [x[tmp_tr_id] for x in train_ip]
  train_op = train_op[tmp_tr_id]
  valid_ip = [x[tmp_val_id] for x in valid_ip]
  valid_op = valid_op[tmp_val_id]
  test_ip = [x[tmp_test_id] for x in test_ip]
  test_op = test_op[tmp_test_id]

  return (
      fore_train_ip,
      fore_train_op,
      fore_valid_ip,
      fore_valid_op,
      train_ip,
      train_op,
      valid_ip,
      valid_op,
      test_ip,
      test_op,
  )
