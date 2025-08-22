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

"""Supervised training code for MovieLens_1M dataset.

To execute this file, run the following command:
CUDA_VISIBLE_DEVICES=1 nohup python -m supervised_main > log_supervised.log &
"""

from autoint_utils import AutoInt
from batch_generator import MiniBatchGenerator
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint


class SuperviseTrain:
  """Main class for training."""

  def __init__(self,
               total_train,
               total_test,
               steps_per_epoch=1000,
               epoch=50):
    self.total_train = total_train
    self.train_data = self.total_train.iloc[:, 1:]
    self.train_label = self.total_train['rating 2']
    self.total_test = total_test
    self.test_data = self.total_test.iloc[:, 1:]
    self.test_label = self.total_test['rating 2']
    self.steps_per_epoch = steps_per_epoch
    self.epoch = epoch

  def training_preprocess(self, nb_trainbatch=1024, nb_testbatch=147803):
    autoint = AutoInt(my_data=self.train_data)
    self.model = autoint.construct(model_type='supervised')
    self.batch_train = MiniBatchGenerator(self.total_train, nb_trainbatch,
                                          autoint.sparse_feats)
    self.batch_test = MiniBatchGenerator(self.total_test, nb_testbatch,
                                         autoint.sparse_feats)
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    self.model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['binary_crossentropy',
                 tf.keras.metrics.AUC(name='auc')])

  def lr_scheduler(self, x):
    return 1e-4 if x < 5 else 1e-5 if x < 25 else 1e-6

  def fit_model(self, model_path):
    """Training the model."""
    reduce_lr = LearningRateScheduler(self.lr_scheduler)
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_auc',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        mode='max')

    print('i am here ...', flush=True)
    self.model.fit(
        self.batch_train,
        steps_per_epoch=self.steps_per_epoch,
        epochs=self.epoch,
        validation_data=self.batch_test,
        validation_steps=1,
        callbacks=[model_checkpoint, reduce_lr],
        verbose=2)

if __name__ == '__main__':
  print('Loading Data ....', flush=True)
  ml_train = pd.read_csv('movie_test.csv')
  ml_test = pd.read_csv('movie_test.csv')

  print('Training Iterations ....', flush=True)
  train_main = SuperviseTrain(ml_train, ml_test)
  train_main.training_preprocess()
  train_main.fit_model('supervised_autoint_ml.h5')
