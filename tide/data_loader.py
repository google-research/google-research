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

"""TF dataloaders for general timeseries datasets.

The expected input format is csv file with a datetime index.
"""


from absl import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import time_features


class TimeSeriesdata(object):
  """Data loader class."""

  def __init__(
      self,
      data_path,
      datetime_col,
      num_cov_cols,
      cat_cov_cols,
      ts_cols,
      train_range,
      val_range,
      test_range,
      hist_len,
      pred_len,
      batch_size,
      freq='H',
      normalize=True,
      epoch_len=None,
      holiday=False,
      permute=True,
  ):
    """Initialize objects.

    Args:
      data_path: path to csv file
      datetime_col: column name for datetime col
      num_cov_cols: list of numerical global covariates
      cat_cov_cols: list of categorical global covariates
      ts_cols: columns corresponding to ts
      train_range: tuple of train ranges
      val_range: tuple of validation ranges
      test_range: tuple of test ranges
      hist_len: historical context
      pred_len: prediction length
      batch_size: batch size (number of ts in a batch)
      freq: freq of original data
      normalize: std. normalize data or not
      epoch_len: num iters in an epoch
      holiday: use holiday features or not
      permute: permute ts in train batches or not

    Returns:
      None
    """
    self.data_df = pd.read_csv(open(data_path, 'r'))
    if not num_cov_cols:
      self.data_df['ncol'] = np.zeros(self.data_df.shape[0])
      num_cov_cols = ['ncol']
    if not cat_cov_cols:
      self.data_df['ccol'] = np.zeros(self.data_df.shape[0])
      cat_cov_cols = ['ccol']
    self.data_df.fillna(0, inplace=True)
    self.data_df.set_index(
        pd.DatetimeIndex(self.data_df[datetime_col]), inplace=True
    )
    self.num_cov_cols = num_cov_cols
    self.cat_cov_cols = cat_cov_cols
    self.ts_cols = ts_cols
    self.train_range = train_range
    self.val_range = val_range
    self.test_range = test_range
    data_df_idx = self.data_df.index
    date_index = data_df_idx.union(
        pd.date_range(
            data_df_idx[-1] + pd.Timedelta(1, freq=freq),
            periods=pred_len + 1,
            freq=freq,
        )
    )
    self.time_df = time_features.TimeCovariates(
        date_index, holiday=holiday
    ).get_covariates()
    self.hist_len = hist_len
    self.pred_len = pred_len
    self.batch_size = batch_size
    self.freq = freq
    self.normalize = normalize
    self.data_mat = self.data_df[self.ts_cols].to_numpy().transpose()
    self.data_mat = self.data_mat[:, 0 : self.test_range[1]]
    self.time_mat = self.time_df.to_numpy().transpose()
    self.num_feat_mat = self.data_df[num_cov_cols].to_numpy().transpose()
    self.cat_feat_mat, self.cat_sizes = self._get_cat_cols(cat_cov_cols)
    self.normalize = normalize
    if normalize:
      self._normalize_data()
    logging.info(
        'Data Shapes: %s, %s, %s, %s',
        self.data_mat.shape,
        self.time_mat.shape,
        self.num_feat_mat.shape,
        self.cat_feat_mat.shape,
    )
    self.epoch_len = epoch_len
    self.permute = permute

  def _get_cat_cols(self, cat_cov_cols):
    """Get categorical columns."""
    cat_vars = []
    cat_sizes = []
    for col in cat_cov_cols:
      dct = {x: i for i, x in enumerate(self.data_df[col].unique())}
      cat_sizes.append(len(dct))
      mapped = self.data_df[col].map(lambda x: dct[x]).to_numpy().transpose()  # pylint: disable=cell-var-from-loop
      cat_vars.append(mapped)
    return np.vstack(cat_vars), cat_sizes

  def _normalize_data(self):
    self.scaler = StandardScaler()
    train_mat = self.data_mat[:, self.train_range[0] : self.train_range[1]]
    self.scaler = self.scaler.fit(train_mat.transpose())
    self.data_mat = self.scaler.transform(self.data_mat.transpose()).transpose()

  def train_gen(self):
    """Generator for training data."""
    num_ts = len(self.ts_cols)
    perm = np.arange(
        self.train_range[0] + self.hist_len,
        self.train_range[1] - self.pred_len,
    )
    perm = np.random.permutation(perm)
    hist_len = self.hist_len
    logging.info('Hist len: %s', hist_len)
    if not self.epoch_len:
      epoch_len = len(perm)
    else:
      epoch_len = self.epoch_len
    for idx in perm[0:epoch_len]:
      for _ in range(num_ts // self.batch_size + 1):
        if self.permute:
          tsidx = np.random.choice(num_ts, size=self.batch_size, replace=False)
        else:
          tsidx = np.arange(num_ts)
        dtimes = np.arange(idx - hist_len, idx + self.pred_len)
        (
            bts_train,
            bts_pred,
            bfeats_train,
            bfeats_pred,
            bcf_train,
            bcf_pred,
        ) = self._get_features_and_ts(dtimes, tsidx, hist_len)

        all_data = [
            bts_train,
            bfeats_train,
            bcf_train,
            bts_pred,
            bfeats_pred,
            bcf_pred,
            tsidx,
        ]
        yield tuple(all_data)

  def test_val_gen(self, mode='val'):
    """Generator for validation/test data."""
    if mode == 'val':
      start = self.val_range[0]
      end = self.val_range[1] - self.pred_len + 1
    elif mode == 'test':
      start = self.test_range[0]
      end = self.test_range[1] - self.pred_len + 1
    else:
      raise NotImplementedError('Eval mode not implemented')
    num_ts = len(self.ts_cols)
    hist_len = self.hist_len
    logging.info('Hist len: %s', hist_len)
    perm = np.arange(start, end)
    if self.epoch_len:
      epoch_len = self.epoch_len
    else:
      epoch_len = len(perm)
    for idx in perm[0:epoch_len]:
      for batch_idx in range(0, num_ts, self.batch_size):
        tsidx = np.arange(batch_idx, min(batch_idx + self.batch_size, num_ts))
        dtimes = np.arange(idx - hist_len, idx + self.pred_len)
        (
            bts_train,
            bts_pred,
            bfeats_train,
            bfeats_pred,
            bcf_train,
            bcf_pred,
        ) = self._get_features_and_ts(dtimes, tsidx, hist_len)
        all_data = [
            bts_train,
            bfeats_train,
            bcf_train,
            bts_pred,
            bfeats_pred,
            bcf_pred,
            tsidx,
        ]
        yield tuple(all_data)

  def _get_features_and_ts(self, dtimes, tsidx, hist_len=None):
    """Get features and ts in specified windows."""
    if hist_len is None:
      hist_len = self.hist_len
    data_times = dtimes[dtimes < self.data_mat.shape[1]]
    bdata = self.data_mat[:, data_times]
    bts = bdata[tsidx, :]
    bnf = self.num_feat_mat[:, data_times]
    bcf = self.cat_feat_mat[:, data_times]
    btf = self.time_mat[:, dtimes]
    if bnf.shape[1] < btf.shape[1]:
      rem_len = btf.shape[1] - bnf.shape[1]
      rem_rep = np.repeat(bnf[:, [-1]], repeats=rem_len)
      rem_rep_cat = np.repeat(bcf[:, [-1]], repeats=rem_len)
      bnf = np.hstack([bnf, rem_rep.reshape(bnf.shape[0], -1)])
      bcf = np.hstack([bcf, rem_rep_cat.reshape(bcf.shape[0], -1)])
    bfeats = np.vstack([btf, bnf])
    bts_train = bts[:, 0:hist_len]
    bts_pred = bts[:, hist_len:]
    bfeats_train = bfeats[:, 0:hist_len]
    bfeats_pred = bfeats[:, hist_len:]
    bcf_train = bcf[:, 0:hist_len]
    bcf_pred = bcf[:, hist_len:]
    return bts_train, bts_pred, bfeats_train, bfeats_pred, bcf_train, bcf_pred

  def tf_dataset(self, mode='train'):
    """Tensorflow Dataset."""
    if mode == 'train':
      gen_fn = self.train_gen
    else:
      gen_fn = lambda: self.test_val_gen(mode)
    output_types = tuple(
        [tf.float32] * 2 + [tf.int32] + [tf.float32] * 2 + [tf.int32] * 2
    )
    dataset = tf.data.Dataset.from_generator(gen_fn, output_types)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
