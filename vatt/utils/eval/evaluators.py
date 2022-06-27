# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Main evaluators classes/functions for MMV/VATT models."""

from absl import logging
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.neural_network
import sklearn.svm
import tensorflow as tf

from vatt.utils.eval import measures


class LinearClsHead(object):
  """A TF-based linear classifier with one hidden unit."""

  def __init__(self,
               hidden_size=128,
               batch_size=8,
               num_epochs=10,
               learning_rate=0.001,
               dropout_rate=0.5,
               seed=None,
               strategy=None):
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate
    self.strategy = strategy
    with self.strategy.scope():
      self.optimizer = tf.keras.optimizers.Adam(learning_rate)
      self.loss = tf.keras.losses.CategoricalCrossentropy()
      self.metric = tf.keras.metrics.CategoricalAccuracy()
      self.model = tf.keras.Sequential()
    if seed:
      tf.random.set_seed(seed)

  def fit(self,
          inputs,
          labels):
    """Receives features and labels and trains a network from scratch."""

    labels = tf.keras.utils.to_categorical(labels)
    _, self.num_classes = labels.shape
    with self.strategy.scope():
      self.model = tf.keras.Sequential()
      self.model.add(
          tf.keras.layers.Dense(
              self.hidden_size,
              kernel_initializer='glorot_normal',
              name='projection'))
      self.model.add(
          tf.keras.layers.Dropout(
              self.dropout_rate,
              name='dropout'))
      self.model.add(
          tf.keras.layers.Dense(
              self.num_classes,
              kernel_initializer='glorot_normal',
              kernel_regularizer=tf.keras.regularizers.L2(0.1),
              activation='softmax',
              name='cls'))
      self.model.compile(optimizer=self.optimizer,
                         loss=self.loss,
                         metrics=self.metric)
    self.model.fit(inputs,
                   labels,
                   batch_size=self.batch_size,
                   epochs=self.num_epochs,
                   shuffle=True,
                   verbose=0)

  def decision_function(self,
                        inputs):
    outputs = self.model.predict(inputs,
                                 batch_size=self.batch_size)

    return outputs


def linear_classifier(train_features,
                      test_features,
                      train_labels,
                      test_labels,
                      dataset_id,
                      num_windows_test,
                      strategy):
  """Trains a linear classifier on the features."""

  # classifier = sklearn.svm.LinearSVC(C=6e-5)
  classifier = LinearClsHead(batch_size=64,
                             hidden_size=128,
                             num_epochs=1000,
                             learning_rate=0.0005,
                             dropout_rate=0.9,
                             seed=1,
                             strategy=strategy)

  # Training.
  n_sample = len(train_features)
  logging.info('Training linear model on %d clips of %s.',
               n_sample,
               dataset_id)
  classifier.fit(train_features, train_labels)
  logging.info('Training done !')

  # Evaluation.
  n_sample = len(test_features)
  logging.info('Running classifier inference on %d clips.', n_sample)
  pred_train = classifier.decision_function(train_features)
  pred_test = classifier.decision_function(test_features)

  if num_windows_test > 1:
    pred_test = np.reshape(pred_test,
                           (len(test_labels), -1, pred_test.shape[1]))
    pred_test = pred_test.mean(axis=1)

  train_metrics = measures.compute_accuracy_metrics(pred_train, train_labels)
  test_metrics = measures.compute_accuracy_metrics(pred_test, test_labels)
  return train_metrics, test_metrics


def vid_mlp_classifier(train_features,
                       test_features,
                       train_labels,
                       test_labels,
                       dataset_id,
                       num_windows_test,
                       strategy):
  """Trains a linear svm on the video features."""

  del strategy
  classifier = sklearn.neural_network.MLPClassifier(
      hidden_layer_sizes=(),
      random_state=1,
      max_iter=200,
      solver='sgd',
      activation='identity',
      learning_rate='adaptive',
      learning_rate_init=0.01,
      alpha=0.05,
      )

  # Training.
  n_sample = len(train_features)
  logging.info('Training linear model on %d clips of %s.',
               n_sample,
               dataset_id)
  classifier.fit(train_features, train_labels)
  logging.info('Training done !')

  # Evaluation.
  n_sample = len(test_features)
  logging.info('Running classifier inference on %d clips.', n_sample)
  pred_train = classifier.predict_proba(train_features)
  pred_test = classifier.predict_proba(test_features)

  if num_windows_test > 1:
    pred_test = np.reshape(pred_test,
                           (len(test_labels), -1, pred_test.shape[1]))
    pred_test = pred_test.mean(axis=1)

  metrics = measures.compute_accuracy_metrics(
      pred_train, train_labels, prefix='train_')
  metrics.update(measures.compute_accuracy_metrics(
      pred_test, test_labels, prefix='test_'))
  return metrics


def modality_similarity(eval_outputs,
                        has_text,
                        has_audio,
                        n_windows):
  """Calculates similarities/retrievals between embeddings in common space."""

  # Report recall metrics based on cosine similarity when
  #    1. embeddings are averaged.
  #    2. similarities are averaged.
  metrics = {}
  if has_text:
    embd_sim_txt2vid = measures.compute_similarity_eval(
        embd=eval_outputs['test_txt2vid_embd'],
        video_embd=eval_outputs['test_vid2txt_embd'],
        n_windows=n_windows,
        normalize=True,
        average_similarities=False,
        average_embeddings=True)
    metrics.update(measures.compute_retrieval_metrics(
        embd_sim_txt2vid, prefix='txt2vid_embd_'))
    cos_sim_txt2vid = measures.compute_similarity_eval(
        embd=eval_outputs['test_txt2vid_embd'],
        video_embd=eval_outputs['test_vid2txt_embd'],
        n_windows=n_windows,
        normalize=True,
        average_similarities=True,
        average_embeddings=False)
    metrics.update(measures.compute_retrieval_metrics(
        cos_sim_txt2vid, prefix='txt2vid_cosine_'))

  if has_audio:
    embd_sim_audio2vid = measures.compute_similarity_eval(
        embd=eval_outputs['test_aud2vid_embd'],
        video_embd=eval_outputs['test_vid2aud_embd'],
        n_windows=n_windows,
        normalize=True,
        average_similarities=False,
        average_embeddings=True)
    metrics.update(measures.compute_retrieval_metrics(
        embd_sim_audio2vid, prefix='audio2vid_embd_'))
    cos_sim_audio2vid = measures.compute_similarity_eval(
        embd=eval_outputs['test_aud2vid_embd'],
        video_embd=eval_outputs['test_vid2aud_embd'],
        n_windows=n_windows,
        normalize=True,
        average_similarities=True,
        average_embeddings=False)
    metrics.update(measures.compute_retrieval_metrics(
        cos_sim_audio2vid, prefix='audio2vid_cosine_'))

  return metrics
