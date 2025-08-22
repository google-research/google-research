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

"""DLLP LLP training main.

To execute this file, run the following command:
CUDA_VISIBLE_DEVICES=1 nohup python -m  dllp_main > log_dllp.log &
"""

import warnings

from autoint_utils import AutoInt
from batch_generator import BagGenerator
from batch_generator import MiniBatchGenerator
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler

warnings.filterwarnings('ignore')


class ROCCallback(keras.callbacks.Callback):
  """Customized call-back for testing model performance."""

  def __init__(self, model, model_path, data_gen):
    self.best_auroc = 0.
    self.model = model
    self.model_path = model_path
    self.data_gen = data_gen

  def get_auroc(self):
    for (x, y) in self.data_gen:
      y_out = self.model.predict_on_batch(x)
      return tf.py_function(roc_auc_score, (np.squeeze(y), np.squeeze(y_out)),
                            tf.double).numpy()

  def on_epoch_end(self, epoch, logs=None):
    auroc = 0.
    auroc = self.get_auroc()
    print('Validation AUROC score: {}'.format(auroc), end='')
    if self.best_auroc <= auroc:
      print('... best Validation AUROC.')
      self.best_auroc = auroc
    else:
      print('Did not improve from: {}'.format(self.best_auroc))
    print('Saving the last model at: ' + self.model_path)
    self.model.save(self.model_path)


class DLLPTrain:
  """Main class for training."""

  def __init__(self,
               total_train,
               total_test,
               nb_bags=5,
               nb_testbatch=147803,
               steps_per_epoch=1000,
               epoch=35):
    self.total_train = total_train
    self.train_data = self.total_train.iloc[:, 1:]
    self.train_label = self.total_train['rating 2']
    self.total_test = total_test
    self.test_data = self.total_test.iloc[:, 1:]
    self.test_label = self.total_test['rating 2']
    self.steps_per_epoch = steps_per_epoch
    self.epoch = epoch
    self.nb_bags = nb_bags
    self.nb_testbatch = nb_testbatch

  @tf.function
  def bag_bce_loss(self, correct, predicted):
    """Bag level cross entropy loss."""
    start = 0
    loss = 0.
    for _ in range(self.nb_bags):
      my_bag_size = int(tf.reduce_sum(correct[start]))
      y_pos_gt = tf.abs(tf.reduce_sum(correct[start+1])) + 1e-7
      y_neg_gt = tf.abs(1-tf.reduce_sum(correct[start+1])) + 1e-7
      y_pos_pred = tf.reduce_mean(predicted[start: start+ my_bag_size])
      y_pos_pred = tf.abs(y_pos_pred) + 1e-7
      y_neg_pred = tf.abs(1 - y_pos_pred) + 1e-7
      loss += -(
          y_pos_gt * tf.math.log(y_pos_pred) +
          y_neg_gt * tf.math.log(y_neg_pred))
      start = start + my_bag_size
    return loss

  def lr_scheduler(self, x):
    return 5e-5 if x < 5 else 1e-5 if x < 25 else 1e-6

  def fit_model(self, model_path):
    """Training function."""
    autoint = AutoInt(my_data=self.train_data)
    model = autoint.construct(model_type='dllp')
    bag_train = BagGenerator(
        self.total_train, self.nb_bags, autoint.sparse_feats, model_type='dllp')
    batch_test = MiniBatchGenerator(self.total_test, self.nb_testbatch,
                                    autoint.sparse_feats)
    opt = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=opt, loss=self.bag_bce_loss)
    reduce_lr = LearningRateScheduler(self.lr_scheduler)
    model_checkpoint = ROCCallback(model, model_path, batch_test)
    model.fit(
        bag_train,
        steps_per_epoch=self.steps_per_epoch,
        epochs=self.epoch,
        callbacks=[model_checkpoint, reduce_lr],
        verbose=2)


if __name__ == '__main__':
  print('Loading Data ....', flush=True)
  ml_train = pd.read_csv('movie_train.csv')
  ml_test = pd.read_csv('movie_test.csv')

  print('Training Iterations ....', flush=True)
  train_main = DLLPTrain(ml_train, ml_test)
  train_main.fit_model('supervised_autoint_ml.h5')
