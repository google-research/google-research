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

"""Input pipeline for the synthetic data settings we consider."""

import numpy as np
import tensorflow as tf
from moe_models_implicit_bias.semi_distributed import models


def create_split(config, local_batch_size, train, w, mu, stats=None):
  """Create train or eval TF dataset for the cluster setting according to the config.

  Args:
    config: specifies the information according to which data is to be
            generated.
    local_batch_size: the local batch size.
    train: boolean indicating whether to create train split or eval split.
    w: the ground truth vector used to average the outputs from clusters.
    mu: numpy array containing the cluster centers.
    stats: statistics of the training dataset such as mean and standard dev.

  Returns:
    ds: A Tensorflow dataset.
    stats: if train flag is set to True, returns statistics of the train set.

  """
  ot = (tf.float32, tf.float32)
  os = (tf.TensorShape([config.dim]), tf.TensorShape([config.out_dim]))
  if train:
    od = OnlineDataset(config, train, w, mu)
    ds = tf.data.Dataset.from_generator(od, output_types=ot, output_shapes=os)
    stats = od.get_stats()
  else:
    od = OnlineDataset(config, train, w, mu, stats)
    ds = tf.data.Dataset.from_generator(od, output_types=ot, output_shapes=os)
    stats = None
  options = tf.data.Options()
  options.autotune.enabled = True
  ds = ds.with_options(options)
  if train:
    ds = ds.repeat()
    ds = ds.shuffle(8 * local_batch_size, seed=0)
  ds = ds.batch(local_batch_size, drop_remainder=True)
  if not train:
    ds = ds.repeat()
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds, stats


class OnlineDataset:
  """Synthetic dataset generating class.

  Used to generate Mixture of Gaussians and Mixture of Subspaces data with
  different types of output functions.
  """

  def __init__(self, config, train, w, mu, stats=None):
    self.config = config
    self.train = train
    self.mu = mu
    if train:
      self.len = config.num_samples
    else:
      self.len = (1 + (50 // config.batch_size)) * config.batch_size
    self.x = np.random.randn(self.len, config.dim)
    self.w = w
    self.y = np.random.randn(self.len, 1)
    if config.out_type == 'con':
      self.y = np.random.randn(self.len, config.out_dim)
    if train:
      np.random.seed(config.seed)
    else:
      np.random.seed(config.seed + 1)
    if config.num_clusters == 1 and config.out_type == 'nn' and config.inp_type == 'mog':
      print('I need to find this', flush=True)
      b_sz = 128
      for ind in range(self.len // b_sz):
        v = np.random.randn(b_sz, config.dim)
        self.x[ind * b_sz:(ind + 1) * b_sz] = self.mu[0][None, :] + v
        label_model = models.LabelModel(config)
        self.y[ind * b_sz:(ind + 1) * b_sz] = label_model.apply(self.w[0], v)
    else:
      for ind in range(self.len):
        v = np.random.randn(config.dim)
        if config.inp_type == 'mos':
          u = np.random.rand(config.rank)
          u = u / np.linalg.norm(u)
          self.x[ind] = np.einsum('jk,j->k', self.mu[ind % config.num_clusters],
                                  u) + v  # -mu_mean
        elif config.inp_type == 'mog':
          self.x[ind] = self.mu[ind % config.num_clusters] + v
        if config.out_type == 'hyp':
          self.y[ind][0] = v @ self.w[ind % config.num_clusters] / (
              config.dim**.5)
        elif config.out_type == 'con':
          self.y[ind] = self.w[ind % config.num_clusters]
        elif config.out_type == 'nn':
          label_model = models.LabelModel(config)
          v = v.reshape((1, -1))
          self.y[ind][0] = label_model.apply(self.w[ind % config.num_clusters],
                                             v)[0][0]
    if train and config.out_dim < 2:
      self.y_mean = np.mean(self.y)
      self.y_std = np.std(self.y)
    elif config.out_dim < 2:
      (self.y_mean, self.y_std) = stats
    if config.out_dim < 2:
      self.y -= self.y_mean
      self.y /= self.y_std
    if train:
      self.y += np.random.randn(self.len, 1) * config.noise

  def get_stats(self):
    if self.config.out_dim < 2:
      return (self.y_mean, self.y_std)
    else:
      return (0, 1)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    return tf.convert_to_tensor(self.x[idx]), tf.convert_to_tensor(self.y[idx])

  def __call__(self):
    for i in range(self.__len__()):
      yield self.__getitem__(i)
