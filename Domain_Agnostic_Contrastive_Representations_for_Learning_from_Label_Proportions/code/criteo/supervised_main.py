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

"""Supervised training main."""
import os

from autoint_utils import AutoInt
from batch_generator import MiniBatchGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint


class SuperviseTrain:
  """Main class for training."""

  def __init__(self,
               train_dir,
               test_dir,
               train_batch_size=1024,
               test_batch_size=75000,
               nb_testbatch=5,
               steps_per_epoch=1000,
               epoches=50):
    autoint = AutoInt(train_dir + os.listdir(train_dir)[0])
    self.model = autoint.construct(model_type='supervised')
    self.train_gen = MiniBatchGenerator(train_dir, train_batch_size,
                                        autoint.dense_feats,
                                        autoint.sparse_feats)
    self.test_gen = MiniBatchGenerator(test_dir, test_batch_size,
                                       autoint.dense_feats,
                                       autoint.sparse_feats)
    self.epoches = epoches
    self.steps_per_epoch = steps_per_epoch
    self.nb_testbatch = nb_testbatch

  def lr_scheduler(self, x):
    return 1e-3 if x < 25 else 1e-4 if x < 75 else 1e-5 if x < 100 else 1e-6

  def fit_model(self, model_path):
    """Training the model."""
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    self.model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['binary_crossentropy',
                 tf.keras.metrics.AUC(name='auc')])
    reduce_lr = LearningRateScheduler(self.lr_scheduler)
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_auc',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        mode='max')
    self.model.fit(
        self.train_gen,
        steps_per_epoch=self.steps_per_epoch,
        epochs=self.epoches,
        validation_data=self.test_gen,
        validation_steps=self.nb_testbatch,
        callbacks=[model_checkpoint, reduce_lr],
        verbose=2)


if __name__ == '__main__':
  train_main = SuperviseTrain('./data/train/', './data/test/')
  train_main.fit_model('autoint_supervised.h5')

