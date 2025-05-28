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

"""Input pipeline for the underparameterized MoE experiment."""

import numpy as np
import tensorflow as tf
from moe_models_implicit_bias.underparam_moe import models


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
  os = (tf.TensorShape([config.dim]), tf.TensorShape([1]))
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
  """Synthetic Dataset for generating a Mixture of Gaussians data.
  """

  def __init__(self, config, train, w, e_dir, stats=None):
    self.config = config
    self.train = train
    if train:
      self.len = config.num_samples
    else:
      self.len = (max(2 * config.num_samples, 2000) //
                  config.batch_size) * config.batch_size
    self.x = np.random.randn(self.len * config.num_experts, config.dim)
    self.x /= np.linalg.norm(self.x, axis=-1, keepdims=True)
    dots = self.x @ e_dir
    dots_argsort = np.argsort(dots)
    self.w = w
    self.y = np.random.randn(self.len * config.num_experts, 1)
    if train:
      np.random.seed(config.seed)
    else:
      np.random.seed(config.seed + 1)
    b_sz = 1024
    for ind in range(self.len * config.num_experts // b_sz):
      label_model = models.LabelModel(config)
      self.y[ind * b_sz:(ind + 1) * b_sz] = label_model.apply(
          self.w, self.x[ind * b_sz:(ind + 1) * b_sz])
    if train:
      self.y_mean = np.mean(self.y)
      self.y_std = np.std(self.y)
    else:
      (self.y_mean, self.y_std) = stats

    self.x = self.x[dots_argsort[-self.len:]]
    self.y = self.y[dots_argsort[-self.len:]]

    self.y -= self.y_mean
    self.y /= self.y_std
    if train:
      self.y += np.random.randn(self.len, 1) * config.noise

  def get_stats(self):
    return (self.y_mean, self.y_std)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    return tf.convert_to_tensor(self.x[idx]), tf.convert_to_tensor(self.y[idx])

  def __call__(self):
    for i in range(self.__len__()):
      yield self.__getitem__(i)
