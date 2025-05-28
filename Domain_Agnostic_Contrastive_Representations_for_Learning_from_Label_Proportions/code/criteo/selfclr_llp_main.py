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

"""SelfCLR-LLP training main."""
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

  def __init__(self, model, model_path, cluster_per_class, nb_class, data_gen,
               nb_testbatch):
    self.best_auroc = 0.
    self.model = model
    self.model_path = model_path
    self.my_epoch = 0.
    self.cluster_per_class = cluster_per_class
    self.nb_class = nb_class
    self.data_gen = data_gen
    self.nb_testbatch = nb_testbatch

  def get_auroc(self):
    cnt, y_pred, y_true = 0, None, None
    for (x, y) in self.data_gen:
      y_out = self.model.predict_on_batch(x)
      shp = tf.shape(y_out)[0]
      y_out = tf.reshape(y_out, [shp, self.cluster_per_class, self.nb_class])
      y_out = tf.nn.softmax(tf.reduce_max(y_out, axis=-2))
      y_out = y_out[:, 0:1]
      if cnt == 0:
        y_pred = y_out
        y_true = y
      else:
        y_pred = np.concatenate([y_pred, y_out], 0)
        y_true = np.concatenate([y_true, y], 0)
      cnt += 1
      if cnt >= self.nb_testbatch:
        break
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double).numpy()

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


class SelfCLRTrain:
  """Main class for training."""

  def __init__(self,
               train_dir,
               test_dir,
               nb_bag=5,
               cluster_per_class=500,
               test_batch_size=75000,
               nb_testbatch=15,
               steps_per_epoch=1000,
               epoches=50):
    autoint = AutoInt(train_dir + os.listdir(train_dir)[0])
    self.model = autoint.construct()
    self.bag_train = BagGenerator(train_dir, nb_bag, autoint.dense_feats,
                                  autoint.sparse_feats, 'selfclr_llp')
    self.test_gen = MiniBatchGenerator(test_dir, test_batch_size,
                                       autoint.dense_feats,
                                       autoint.sparse_feats)
    self.epoches = epoches
    self.cluster_per_class = cluster_per_class
    self.nb_class = 2
    self.steps_per_epoch = steps_per_epoch
    self.nb_testbatch = nb_testbatch

  @tf.function
  def bag_loss(self, correct, predicted):
    """Bag-level cross-entropy loss: same as DLLP."""
    y_soft = tf.nn.softmax(predicted)
    start, loss = 0, 0.0
    for _ in range(self.bag_train.nb_bag):
      bag_size = int(tf.reduce_sum(correct[start]))
      y_true = correct[start+1]
      y_pred_bag = tf.reduce_mean(y_soft[start:start+bag_size], [0])
      loss += -tf.reduce_sum(y_true*(tf.math.log(y_pred_bag + 1e-7)))
      start = start + bag_size
    return loss/self.bag_train.nb_bag

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
    return 0.5*loss_1 + loss_2 + 0.01* loss_3

  def lr_scheduler(self, x):
    return 1e-5 if x < 35 else 1e-6

  def fit_model(self, model_path):
    """Training function."""
    opt = keras.optimizers.Adam(learning_rate=1e-5)
    self.model.compile(optimizer=opt, loss=self.total_loss)
    reduce_lr = LearningRateScheduler(self.lr_scheduler)
    model_checkpoint = ROCCallback(self.model, model_path,
                                   self.cluster_per_class, self.nb_class,
                                   self.test_gen, self.nb_testbatch)
    self.model.fit(
        self.bag_train,
        steps_per_epoch=self.steps_per_epoch,
        epochs=self.epoches,
        callbacks=[model_checkpoint, reduce_lr],
        verbose=2)

if __name__ == '__main__':
  train_main = SelfCLRTrain('./data/train/', './data/test/')
  train_main.fit_model('autoint_selfclr.h5')
