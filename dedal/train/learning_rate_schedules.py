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

"""Implements custom learning rate schedules."""

import gin
import tensorflow as tf


@gin.configurable
class InverseSquareRootDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Implements the learning rate schedule in Vaswani et al. 2017."""

  def __init__(
      self,
      lr_max = 1e-3,
      warmup_init_lr = 0.0,
      warmup_steps = 4000,
      **kwargs):
    super().__init__(**kwargs)
    self._lr_max = lr_max
    self._warmup_init_lr = warmup_init_lr
    self._warmup_steps = warmup_steps

  def __call__(self, step):
    norm_step = step / self._warmup_steps

    def true_fn():
      return (self._warmup_init_lr +
              (self._lr_max - self._warmup_init_lr) * norm_step)

    def false_fn():
      return self._lr_max * tf.math.rsqrt(norm_step)

    return tf.cond(norm_step <= 1.0, true_fn, false_fn)


@gin.configurable
class NoDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Implements a constant learning rate with a warmup period."""

  def __init__(
      self,
      lr_max = 1e-3,
      warmup_init_lr = 0.0,
      warmup_steps = 4000,
      **kwargs):
    super().__init__(**kwargs)
    self._lr_max = lr_max
    self._warmup_init_lr = warmup_init_lr
    self._warmup_steps = warmup_steps

  def __call__(self, step):
    norm_step = step / self._warmup_steps

    def true_fn():
      return (self._warmup_init_lr +
              (self._lr_max - self._warmup_init_lr) * norm_step)

    def false_fn():
      return tf.cast(self._lr_max, tf.float32)

    return tf.cond(norm_step <= 1.0, true_fn, false_fn)
