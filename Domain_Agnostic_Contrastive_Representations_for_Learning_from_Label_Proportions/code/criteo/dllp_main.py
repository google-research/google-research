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

"""DLLP LLP training main."""
import os

from autoint_utils import AutoInt
from batch_generator import BagGenerator
from batch_generator import MiniBatchGenerator
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler


class ROCCallback(keras.callbacks.Callback):
  """Customized call-back for testing model performance."""

  def __init__(self, model, model_path, data_gen, nb_testbatch):
    self.best_auroc = 0.
    self.model = model
    self.model_path = model_path
    self.my_epoch = 0.
    self.data_gen = data_gen
    self.nb_testbatch = nb_testbatch

  def get_auroc(self):
    cnt = 0
    y_pred = None
    y_true = None
    for (x, y) in self.data_gen:
      y_out = self.model.predict_on_batch(x)
      if cnt == 0:
        y_pred = y_out
        y_true = y
      else:
        y_pred = np.concatenate([y_pred, y_out], 0)
        y_true = np.concatenate([y_true, y], 0)
      cnt += 1
      if cnt >= self.nb_testbatch:
        break
    return tf.py_function(roc_auc_score,
                          (np.squeeze(y_true), np.squeeze(y_pred)),
                          tf.double).numpy()

  def on_epoch_end(self, epoch, logs=None):
    auroc = self.get_auroc()
    print('Test AUROC score: {}'.format(auroc), end='')
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
               train_dir,
               test_dir,
               nb_bag=5,
               test_batch_size=75000,
               nb_testbatch=15,
               steps_per_epoch=1000,
               epoches=50):
    autoint = AutoInt(train_dir + os.listdir(train_dir)[0])
    self.model = autoint.construct(model_type='dllp')
    self.bag_gen = BagGenerator(train_dir, nb_bag, autoint.dense_feats,
                                autoint.sparse_feats, 'dllp')
    self.test_gen = MiniBatchGenerator(test_dir, test_batch_size,
                                       autoint.dense_feats,
                                       autoint.sparse_feats)
    self.epoches = epoches
    self.steps_per_epoch = steps_per_epoch
    self.nb_testbatch = nb_testbatch

  @tf.function
  def bag_bce_loss(self, correct, predicted):
    """Bag-level cross-entropy loss for DLLP."""
    start, loss = 0, 0.
    for _ in range(self.bag_gen.nb_bag):
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
    return 1e-5 if x < 35 else 1e-6

  def fit_model(self, model_path):
    opt = keras.optimizers.Adam(learning_rate=1e-5)
    self.model.compile(optimizer=opt, loss=self.bag_bce_loss)
    reduce_lr = LearningRateScheduler(self.lr_scheduler)
    model_checkpoint = ROCCallback(self.model, model_path, self.test_gen,
                                   self.nb_testbatch)
    self.model.fit(
        self.bag_gen,
        steps_per_epoch=self.steps_per_epoch,
        epochs=self.epoches,
        callbacks=[model_checkpoint, reduce_lr],
        verbose=2)

if __name__ == '__main__':
  train_main = DLLPTrain('./data/train/', './data/test/')
  train_main.fit_model('autoint_dllp.h5')
