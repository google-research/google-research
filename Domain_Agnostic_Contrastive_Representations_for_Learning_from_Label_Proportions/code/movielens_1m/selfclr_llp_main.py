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

"""Selfclr LLP training main.

To execute this file, run the following command:
CUDA_VISIBLE_DEVICES=1 nohup python -m  selfclr_llp > log_selfclr.log &
"""

import warnings

from autoint_utils import AutoInt
from batch_generator import BagGenerator
from batch_generator import MiniBatchGenerator
import pandas as pd
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler

warnings.filterwarnings('ignore')


class ROCCallback(keras.callbacks.Callback):
  """Customized call-back for testing model performance."""

  def __init__(self, model, model_path, cluster_per_class, nb_class, data_gen):
    self.best_auroc = 0.
    self.model = model
    self.model_path = model_path
    self.my_epoch = 0.
    self.cluster_per_class = cluster_per_class
    self.nb_class = nb_class
    self.data_gen = data_gen

  def get_auroc(self):
    for (x, y) in self.data_gen:
      y_out = self.model.predict_on_batch(x)
      shp = tf.shape(y_out)[0]
      y_out = tf.reshape(y_out, [shp, self.cluster_per_class, self.nb_class])
      y_out = tf.nn.softmax(tf.reduce_max(y_out, axis=-2))
      y_out = y_out[:, 0:1]
      return tf.py_function(roc_auc_score, (y, y_out), tf.double).numpy()

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


class SelfCLRTrain:
  """Main class for training."""

  def __init__(self,
               total_train,
               total_test,
               cluster_per_class=500,
               nb_bags=5,
               nb_testbatch=147803,
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
    self.cluster_per_class = cluster_per_class
    self.nb_class = 2
    self.nb_bags = nb_bags
    self.nb_testbatch = nb_testbatch

  @tf.function
  def bag_loss(self, correct, predicted):
    """Bag-level cross-entropy loss: same as DLLP."""
    y_soft = tf.nn.softmax(predicted)
    start, loss = 0, 0.0
    for _ in range(self.bag_train.nb_bags):
      bag_size = int(tf.reduce_sum(correct[start]))
      y_true = correct[start+1]
      y_pred_bag = tf.reduce_mean(y_soft[start:start+bag_size], [0])
      loss += -tf.reduce_sum(y_true*(tf.math.log(y_pred_bag + 1e-7)))
      start = start + bag_size
    return loss/self.bag_train.nb_bags

  @tf.function
  def add_cd(self, activation):
    """New divergence loss  ---> Same half; different augmentation."""
    batch_size = int(tf.shape(activation)[0])
    labels_ = tf.one_hot(tf.range(batch_size), batch_size)
    logits_ = tf.matmul(activation, activation, transpose_b=True)
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_, logits=logits_))

  @tf.function
  def total_loss(self, correct, predict):
    loss_1 = self.add_cd(predict)
    shp = tf.shape(predict)[0]
    tf_b = tf.reshape(predict, [shp, self.cluster_per_class, self.nb_class])
    logits = tf.reduce_max(tf_b, axis=-2)
    y_true = correct[:, :self.nb_class]
    loss_2 = self.bag_loss(y_true, logits)
    loss_3 = tf.reduce_mean(tf.reduce_max(tf.nn.sigmoid(logits), [1]))
    return 0.5*loss_1 + loss_2 + 0.001* loss_3

  def lr_scheduler(self, x):
    return 5e-5 if x < 5 else 1e-5 if x < 25 else 1e-6

  def fit_model(self, model_path):
    """Training function."""
    autoint = AutoInt(my_data=self.train_data)
    self.model = autoint.construct(
        cluster_per_class=self.cluster_per_class, model_type='selfclr_llp')
    self.bag_train = BagGenerator(self.total_train, self.nb_bags,
                                  autoint.sparse_feats)
    self.batch_test = MiniBatchGenerator(self.total_test, self.nb_testbatch,
                                         autoint.sparse_feats)
    opt = keras.optimizers.Adam(learning_rate=1e-5)
    self.model.compile(optimizer=opt, loss=self.total_loss)
    reduce_lr = LearningRateScheduler(self.lr_scheduler)
    model_checkpoint = ROCCallback(self.model, model_path,
                                   self.cluster_per_class, self.nb_class,
                                   self.batch_test)
    self.model.fit(
        self.bag_train,
        steps_per_epoch=self.steps_per_epoch,
        epochs=self.epoch,
        callbacks=[model_checkpoint, reduce_lr],
        verbose=2)

if __name__ == '__main__':
  print('Loading Data ....', flush=True)
  ml_train = pd.read_csv('movie_train.csv')
  ml_test = pd.read_csv('movie_test.csv')

  print('Training Iterations ....', flush=True)
  train_main = SelfCLRTrain(ml_train, ml_test)
  train_main.fit_model('selfclr_llp_autoint_ml.h5')
