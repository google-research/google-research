# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Helper file to run the discover concept algorithm in the toy dataset."""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import app

import keras
from keras.activations import sigmoid
import keras.backend as K
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Layer
from keras.models import Model

from keras.optimizers import Adam
from keras.optimizers import SGD
import numpy as np
from numpy import inf
from numpy.random import seed
from scipy.special import comb
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import set_random_seed

seed(0)
set_random_seed(0)

# global variables
init = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
batch_size = 128

step = 200
min_weight_arr = []
min_index_arr = []
concept_arr = {}


class Weight(Layer):
  """Simple Weight class."""

  def __init__(self, dim, **kwargs):
    self.dim = dim
    super(Weight, self).__init__(**kwargs)

  def build(self, input_shape):
    # creates a trainable weight variable for this layer.
    self.kernel = self.add_weight(
        name='proj', shape=self.dim, initializer=init, trainable=True)
    super(Weight, self).build(input_shape)

  def call(self, x):
    return self.kernel

  def compute_output_shape(self, input_shape):
    return self.dim


def reduce_var(x, axis=None, keepdims=False):
  """Returns variance of a tensor, alongside the specified axis."""
  m = tf.reduce_mean(x, axis=axis, keep_dims=True)
  devs_squared = tf.square(x - m)
  return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def concept_loss(cov, cov0, i, n_concept, lmbd=5.):
  """Creates a concept loss based on reconstruction loss."""

  def loss(y_true, y_pred):
    if i == 0:
      return tf.reduce_mean(
          tf.keras.backend.binary_crossentropy(y_true, y_pred))
    else:
      return tf.reduce_mean(
          tf.keras.backend.binary_crossentropy(y_true, y_pred)
      ) + lmbd * K.mean(cov - np.eye(n_concept)) + lmbd * K.mean(cov0)

  return loss


def concept_variance(cov, cov0, i, n_concept):
  """Creates a concept loss based on reconstruction variance."""

  def loss(_, y_pred):
    if i == 0:
      return 1. * tf.reduce_mean(reduce_var(y_pred, axis=0))
    else:
      return 1. * tf.reduce_mean(reduce_var(y_pred, axis=0)) + 10. * K.mean(
          cov - np.eye(n_concept)) + 10. * K.mean(cov0)

  return loss


def ipca_model(concept_arraynew2,
               dense2,
               predict,
               f_train,
               y_train,
               f_val,
               y_val,
               n_concept,
               verbose=False,
               epochs=20,
               metric='binary_accuracy'):
  """Returns main function of ipca."""
  pool1f_input = Input(shape=(f_train.shape[1],), name='pool1_input')
  cluster_input = K.variable(concept_arraynew2)
  proj_weight = Weight((f_train.shape[1], n_concept))(pool1f_input)
  proj_weight_n = Lambda(lambda x: K.l2_normalize(x, axis=0))(proj_weight)
  eye = K.eye(n_concept) * 1e-5
  proj_recon_t = Lambda(
      lambda x: K.dot(x, tf.linalg.inv(K.dot(K.transpose(x), x) + eye)))(
          proj_weight)
  proj_recon = Lambda(lambda x: K.dot(K.dot(x[0], x[2]), K.transpose(x[1])))(
      [pool1f_input, proj_weight, proj_recon_t])
  # proj_recon2 = Lambda(lambda x: x[0] - K.dot(K.dot(x[0],K.dot(x[1],
  # tf.linalg.inv(K.dot(K.transpose(x[1]), x[1]) + 1e-5 * K.eye(n_concept)))),
  # K.transpose(x[1])))([pool1f_input, proj_weight])

  cov1 = Lambda(lambda x: K.mean(K.dot(x[0], x[1]), axis=1))(
      [cluster_input, proj_weight_n])
  cov0 = Lambda(lambda x: x - K.mean(x, axis=0, keepdims=True))(cov1)
  cov0_abs = Lambda(lambda x: K.abs(K.l2_normalize(x, axis=0)))(cov0)
  cov0_abs_flat = Lambda(lambda x: K.reshape(x, (-1, n_concept)))(cov0_abs)
  cov = Lambda(lambda x: K.dot(K.transpose(x), x))(cov0_abs_flat)
  fc2_pr = dense2(proj_recon)
  softmax_pr = predict(fc2_pr)
  # fc2_pr2 = dense2(proj_recon2)
  # softmax_pr2 = predict(fc2_pr2)

  finetuned_model_pr = Model(inputs=pool1f_input, outputs=softmax_pr)
  # finetuned_model_pr2 = Model(inputs=pool1f_input, outputs=softmax_pr2)
  # finetuned_model_pr2.compile(loss=
  #                             concept_loss(cov,cov0_abs,0),
  #                             optimizer = sgd(lr=0.),
  #                             metrics=['binary_accuracy'])
  finetuned_model_pr.layers[-1].activation = sigmoid
  print(finetuned_model_pr.layers[-1].activation)
  finetuned_model_pr.layers[-1].trainable = False
  # finetuned_model_pr2.layers[-1].trainable = False
  finetuned_model_pr.layers[-2].trainable = False
  finetuned_model_pr.layers[-3].trainable = False
  # finetuned_model_pr2.layers[-2].trainable = False
  finetuned_model_pr.compile(
      loss=concept_loss(cov, cov0_abs, 0, n_concept),
      optimizer=Adam(lr=0.001),
      metrics=[metric])
  # finetuned_model_pr2.compile(
  #    loss=concept_variance(cov, cov0_abs, 0),
  #    optimizer=SGD(lr=0.0),
  #    metrics=['binary_accuracy'])

  if verbose:
    print(finetuned_model_pr.summary())
  # finetuned_model_pr2.summary()

  finetuned_model_pr.fit(
      f_train,
      y_train,
      batch_size=50,
      epochs=epochs,
      validation_data=(f_val, y_val),
      verbose=verbose)
  finetuned_model_pr.layers[-1].trainable = False
  finetuned_model_pr.layers[-2].trainable = False
  finetuned_model_pr.layers[-3].trainable = False
  finetuned_model_pr.compile(
      loss=concept_loss(cov, cov0_abs, 1, n_concept),
      optimizer=Adam(lr=0.001),
      metrics=[metric])

  return finetuned_model_pr  # , finetuned_model_pr2


def ipca_model_shap(dense2, predict, n_concept, input_size, concept_matrix):
  """returns model that calculates of SHAP."""
  pool1f_input = Input(shape=(input_size,), name='cluster1')
  concept_mask = Input(shape=(n_concept,), name='mask')
  proj_weight = Weight((input_size, n_concept))(pool1f_input)
  concept_mask_r = Lambda(lambda x: K.mean(x, axis=0, keepdims=True))(
      concept_mask)
  proj_weight_m = Lambda(lambda x: x[0] * x[1])([proj_weight, concept_mask_r])
  eye = K.eye(n_concept) * 1e-10
  proj_recon_t = Lambda(
      lambda x: K.dot(x, tf.linalg.inv(K.dot(K.transpose(x), x) + eye)))(
          proj_weight_m)
  proj_recon = Lambda(lambda x: K.dot(K.dot(x[0], x[2]), K.transpose(x[1])))(
      [pool1f_input, proj_weight_m, proj_recon_t])
  fc2_pr = dense2(proj_recon)
  softmax_pr = predict(fc2_pr)
  finetuned_model_pr = Model(
      inputs=[pool1f_input, concept_mask], outputs=softmax_pr)
  finetuned_model_pr.compile(
      loss='categorical_crossentropy',
      optimizer=SGD(lr=0.000),
      metrics=['accuracy'])
  finetuned_model_pr.summary()
  finetuned_model_pr.layers[-7].set_weights([concept_matrix])
  return finetuned_model_pr


def get_acc(binary_sample, f_val, y_val_logit, shap_model, verbose=False):
  """Returns accuracy."""
  acc = shap_model.evaluate(
      [f_val, np.tile(np.array(binary_sample), (f_val.shape[0], 1))],
      y_val_logit,
      verbose=verbose)[1]
  return acc


def shap_kernel(n, k):
  """Returns kernel of shapley in KernelSHAP."""
  return (n-1)*1.0/((n-k)*k*comb(n, k))


def get_shap(nc, f_val, y_val_logit, shap_model, full_acc, null_acc, n_concept):
  """Returns ConceptSHAP."""
  inputs = list(itertools.product([0, 1], repeat=n_concept))
  outputs = [(get_acc(k, f_val, y_val_logit, shap_model)-null_acc)/
             (full_acc-null_acc) for k in inputs]
  kernel = [shap_kernel(nc, np.sum(ii)) for ii in inputs]
  x = np.array(inputs)
  y = np.array(outputs)
  k = np.array(kernel)
  k[k == inf] = 0
  xkx = np.matmul(np.matmul(x.transpose(), np.diag(k)), x)
  xky = np.matmul(np.matmul(x.transpose(), np.diag(k)), y)
  expl = np.matmul(np.linalg.pinv(xkx), xky)
  return expl


def main(_):
  return


if __name__ == '__main__':
  app.run(main)
