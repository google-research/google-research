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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import nn
from . import transformers
from . import utils
from . import worker_util

from absl import logging
import numpy as np
import tensorflow as tf


def log_trainable_variables(trainable_variables):
  logging.info('trainable variables:')
  for v in trainable_variables:
    logging.info(repr(v))
  logging.info('num params {:,}'.format(
      sum(int(np.prod(v.shape)) for v in trainable_variables)))


class SlicedChannelModel(worker_util.BaseModel):

  def __init__(self, config):
    self.config = config
    logging.info('SlicedChannelModel config: {}'.format(config))

    # decoder
    self.ar_decoder = transformers.Transformer3d(
        name='ar_decoder',
        img_height=self.config.img_size,
        img_width=self.config.img_size,
        img_channels=3,
        num_embs=256,
        **self.config.ardec.values())
    self._trainable_variables = self.ar_decoder.trainable_variables
    log_trainable_variables(self._trainable_variables)

    # optimizer

    self.global_step = tf.train.get_or_create_global_step()
    self.lr = utils.get_warmed_up_lr(
        max_lr=self.config.optim.max_lr,
        warmup=self.config.optim.warmup,
        global_step=self.global_step)
    self.non_sharded_optimizer = tf.train.AdamOptimizer(
        self.lr,
        beta1=self.config.optim.adam_beta1,
        beta2=self.config.optim.adam_beta2)
    self._ema = tf.train.ExponentialMovingAverage(
        decay=tf.where(
            tf.less(self.global_step, 1), 1e-10, self.config.optim.ema))
    self._ema_op = self._ema.apply(self.trainable_variables)

  @property
  def trainable_variables(self):
    return self._trainable_variables

  @property
  def ema(self):
    return self._ema

  def train_fn(self, x_bhwc):
    loss = tf.reduce_mean(
        self.ar_decoder.compute_random_slice_nll(
            x_bhwc, cond=None, dropout=self.config.dropout)) / np.log(2.0)

    # Training+EMA op
    train_op, gnorm = utils.make_train_op(
        optimizer=self.non_sharded_optimizer,
        loss=loss,
        trainable_variables=self.trainable_variables,
        global_step=self.global_step,
        grad_clip_norm=self.config.optim.grad_clip_norm)
    with tf.control_dependencies([train_op]):
      train_op = tf.group(self._ema_op)

    return {
        'lr': self.lr,
        'loss': loss,
        'grad_norm': gnorm,
        'param_rms': utils.rms(self.trainable_variables)
    }, train_op

  def eval_fn(self, x_bhwc):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.ar_decoder.compute_all_slice_logits(
            x_bhwc, cond=None, dropout=0.0),
        labels=x_bhwc)
    losses = tf.reduce_mean(nn.flatten(losses), axis=1) / np.log(2.0)
    return {'losses': losses}

  def samples_fn(self, x_bhwc):
    noise_shape = x_bhwc.shape + [self.ar_decoder.num_embs]
    cond = None
    out = {}
    out['samples_1.0'] = self.ar_decoder.sample_fast(
        utils.gumbel(shape=noise_shape, temperature=1.0),
        cond=cond,
        dropout=0.0)
    out['samples_0.99'] = self.ar_decoder.sample_fast(
        utils.gumbel(shape=noise_shape, temperature=0.99),
        cond=cond,
        dropout=0.0)
    return out
