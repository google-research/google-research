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

"""Contains network definitions (for siamese net, and cnc_net)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import time

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.layers import Input
from tensorflow.compat.v1.keras.layers import Lambda
from tensorflow.compat.v1.keras.models import Model

from clustering_normalized_cuts import affinities
from clustering_normalized_cuts import train
from clustering_normalized_cuts import util
from clustering_normalized_cuts.layer import stack_layers


class SiameseNet(object):
  """Class for Siamese Network."""

  def __init__(self, inputs, arch, siam_reg, main_path, y_true):
    self.orig_inputs = inputs
    # set up inputs
    self.inputs = {
        'A': inputs['Unlabeled'],
        'B': Input(shape=inputs['Unlabeled'].get_shape().as_list()[1:]),
        'Labeled': inputs['Labeled'],
    }

    self.main_path = os.path.join(main_path, 'siemese/')
    self.y_true = y_true

    # generate layers
    self.layers = []
    self.layers += util.make_layer_list(arch, 'siamese', siam_reg)

    # create the siamese net
    self.outputs = stack_layers(self.inputs, self.layers)

    # add the distance layer
    self.distance = Lambda(
        affinities.euclidean_distance,
        output_shape=affinities.eucl_dist_output_shape)(
            [self.outputs['A'], self.outputs['B']])

    # create the distance model for training
    self.net = Model([self.inputs['A'], self.inputs['B']], self.distance)

    # compile the siamese network
    self.net.compile(
        loss=affinities.get_contrastive_loss(m_neg=1, m_pos=0.05),
        optimizer='rmsprop')

  def train(self,
            pairs_train,
            dist_train,
            pairs_val,
            dist_val,
            lr,
            drop,
            patience,
            num_epochs,
            batch_size,
            dset,
            load=True):
    """Train the Siamese Network."""
    if load:
      # load weights into model
      output_path = os.path.join(self.main_path, dset)
      load_model(self.net, output_path, '_siamese')
      return
    # create handler for early stopping and learning rate scheduling
    self.lh = util.LearningHandler(
        lr=lr, drop=drop, lr_tensor=self.net.optimizer.lr, patience=patience)

    # initialize the training generator
    train_gen_ = util.train_gen(pairs_train, dist_train, batch_size)

    # format the validation data for keras
    validation_data = ([pairs_val[:, 0], pairs_val[:, 1]], dist_val)

    # compute the steps per epoch
    steps_per_epoch = int(len(pairs_train) / batch_size)

    # train the network
    self.net.fit_generator(
        train_gen_,
        epochs=num_epochs,
        validation_data=validation_data,
        steps_per_epoch=steps_per_epoch,
        callbacks=[self.lh])

    model_json = self.net.to_json()
    output_path = os.path.join(self.main_path, dset)
    save_model(self.net, model_json, output_path, '_siamese')

  def predict(self, x, batch_sizes):
    # compute the siamese embeddings of the input data
    return train.predict(
        self.outputs['A'],
        x_unlabeled=x,
        inputs=self.orig_inputs,
        y_true=self.y_true,
        batch_sizes=batch_sizes)


class CncNet(object):
  """Class for CNC Network."""

  def __init__(self,
               inputs,
               arch,
               cnc_reg,
               y_true,
               y_train_labeled_onehot,
               n_clusters,
               affinity,
               scale_nbr,
               n_nbrs,
               batch_sizes,
               result_path,
               dset,
               siamese_net=None,
               x_train=None,
               lr=0.01,
               temperature=1.0,
               bal_reg=0.0):
    self.y_true = y_true
    self.y_train_labeled_onehot = y_train_labeled_onehot
    self.inputs = inputs
    self.batch_sizes = batch_sizes
    self.result_path = result_path
    self.lr = lr
    self.temperature = temperature
    # generate layers
    self.layers = util.make_layer_list(arch[:-1], 'cnc', cnc_reg)

    print('Runing with CNC loss')
    self.layers += [{
        'type': 'None',
        'size': n_clusters,
        'l2_reg': cnc_reg,
        'name': 'cnc_{}'.format(len(arch))
    }]

    # create CncNet
    self.outputs = stack_layers(self.inputs, self.layers)
    self.net = Model(
        inputs=self.inputs['Unlabeled'], outputs=self.outputs['Unlabeled'])

    # DEFINE LOSS

    # generate affinity matrix W according to params
    if affinity == 'siamese':
      input_affinity = tf.concat(
          [siamese_net.outputs['A'], siamese_net.outputs['Labeled']], axis=0)
      x_affinity = siamese_net.predict(x_train, batch_sizes)
    elif affinity in ['knn', 'full']:
      input_affinity = tf.concat(
          [self.inputs['Unlabeled'], self.inputs['Labeled']], axis=0)
      x_affinity = x_train

    # calculate scale for affinity matrix
    scale = util.get_scale(x_affinity, self.batch_sizes['Unlabeled'], scale_nbr)

    # create affinity matrix
    if affinity == 'full':
      weight_mat = affinities.full_affinity(input_affinity, scale=scale)
    elif affinity in ['knn', 'siamese']:
      weight_mat = affinities.knn_affinity(
          input_affinity, n_nbrs, scale=scale, scale_nbr=scale_nbr)

    # define loss
    self.tau = tf.Variable(self.temperature, name='temperature')
    self.outputs['Unlabeled'] = util.gumbel_softmax(self.outputs['Unlabeled'],
                                                    self.tau)
    num_nodes = self.batch_sizes['Unlabeled']
    cluster_size = tf.reduce_sum(self.outputs['Unlabeled'], axis=0)
    ground_truth = [num_nodes / float(n_clusters)] * n_clusters
    bal = tf.losses.mean_squared_error(ground_truth, cluster_size)

    degree = tf.expand_dims(tf.reduce_sum(weight_mat, axis=1), 0)
    vol = tf.matmul(degree, self.outputs['Unlabeled'], name='vol')
    normalized_prob = tf.divide(
        self.outputs['Unlabeled'], vol[tf.newaxis, :],
        name='normalized_prob')[0]
    gain = tf.matmul(
        normalized_prob,
        tf.transpose(1 - self.outputs['Unlabeled']),
        name='res2')
    self.loss = tf.reduce_sum(gain * weight_mat) + bal_reg * bal

    # create the train step update
    self.learning_rate = tf.Variable(self.lr, name='cnc_learning_rate')
    self.train_step = tf.train.RMSPropOptimizer(
        learning_rate=self.learning_rate).minimize(
            self.loss, var_list=self.net.trainable_weights)
    # initialize cnc_net variables
    K.get_session().run(tf.global_variables_initializer())
    K.get_session().run(tf.variables_initializer(self.net.trainable_weights))
    if affinity == 'siamese':
      output_path = os.path.join(self.main_path, dset)
      load_model(siamese_net, output_path, '_siamese')

  def train(self,
            x_train_unlabeled,
            x_train_labeled,
            x_val_unlabeled,
            drop,
            patience,
            min_tem,
            num_epochs,
            load=False):
    """Train the CNC network."""
    file_name = 'cnc_net'
    if load:
      # load weights into model
      print('load pretrain weights of the CNC network.')
      load_model(self.net, self.result_path, file_name)
      return

    # create handler for early stopping and learning rate scheduling
    self.lh = util.LearningHandler(
        lr=self.lr,
        drop=drop,
        lr_tensor=self.learning_rate,
        patience=patience,
        tau=self.temperature,
        tau_tensor=self.tau,
        min_tem=min_tem,
        gumble=True)

    losses = np.empty((num_epochs,))
    val_losses = np.empty((num_epochs,))

    # begin cnc_net training loop
    self.lh.on_train_begin()
    for i in range(num_epochs):
      # train cnc_net
      losses[i] = train.train_step(
          return_var=[self.loss],
          updates=self.net.updates + [self.train_step],
          x_unlabeled=x_train_unlabeled,
          inputs=self.inputs,
          y_true=self.y_true,
          batch_sizes=self.batch_sizes,
          x_labeled=x_train_labeled,
          y_labeled=self.y_train_labeled_onehot,
          batches_per_epoch=100)[0]

      # get validation loss
      val_losses[i] = train.predict_sum(
          self.loss,
          x_unlabeled=x_val_unlabeled,
          inputs=self.inputs,
          y_true=self.y_true,
          x_labeled=x_train_unlabeled[0:0],
          y_labeled=self.y_train_labeled_onehot,
          batch_sizes=self.batch_sizes)

      # do early stopping if necessary
      if self.lh.on_epoch_end(i, val_losses[i]):
        print('STOPPING EARLY')
        break

      # print training status
      print('Epoch: {}, loss={:2f}, val_loss={:2f}'.format(
          i, losses[i], val_losses[i]))
      with gfile.Open(self.result_path + 'losses', 'a') as f:
        f.write(str(i) + ' ' + str(losses[i]) + ' ' + str(val_losses[i]) + '\n')

    model_json = self.net.to_json()
    save_model(self.net, model_json, self.result_path, file_name)

  def predict(self, x):
    # test inputs do not require the 'Labeled' input
    inputs_test = {'Unlabeled': self.inputs['Unlabeled']}
    return train.predict(
        self.outputs['Unlabeled'],
        x_unlabeled=x,
        inputs=inputs_test,
        y_true=self.y_true,
        x_labeled=x[0:0],
        y_labeled=self.y_train_labeled_onehot[0:0],
        batch_sizes=self.batch_sizes)


def save_model(net, model_json, output_path, file_name):
  """serialize weights to HDF5."""
  with gfile.Open(output_path + file_name + '.json', 'w') as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  weight_path = os.path.join(output_path, file_name, '.h5')
  local_filename = weight_path.split('/')[-1]
  tmp_filename = os.path.join(tempfile.gettempdir(),
                              str(int(time.time())) + '_' + local_filename)
  net.save_weights(tmp_filename)
  gfile.Copy(tmp_filename, weight_path, overwrite=True)
  gfile.Remove(tmp_filename)


def load_model(net, output_path, file_name):
  weights_path = os.path.join(output_path, file_name, '.h5')
  local_filename = weights_path.split('/')[-1]
  tmp_filename = os.path.join(tempfile.gettempdir(),
                              str(int(time.time())) + '_' + local_filename)
  gfile.Copy(weights_path, tmp_filename)
  net.load_weights(tmp_filename)
  gfile.Remove(tmp_filename)
