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

"""Runs an experiment on the 'after kernel'."""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import sklearn.svm
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

parser = argparse.ArgumentParser()

parser.add_argument("--max_num_epochs",
                    type=int,
                    default=1000)

parser.add_argument("--training_loss_target",
                    type=float,
                    default=None)

parser.add_argument("--learning_rate",
                    type=float,
                    default=None)

parser.add_argument("--test_accuracy_output",
                    type=str,
                    default="/tmp/ak.txt")

parser.add_argument("--model_summary_output",
                    type=str,
                    default="/tmp/ak_model.txt")

parser.add_argument("--filter_size",
                    type=int,
                    default=3)

parser.add_argument("--pool_size",
                    type=int,
                    default=2)

parser.add_argument("--num_parameters",
                    type=int,
                    default=1e5)

parser.add_argument("--num_blocks",
                    type=int,
                    default=2)

parser.add_argument("--num_layers",
                    type=int,
                    default=4)

parser.add_argument("--num_runs",
                    type=int,
                    default=5)

parser.add_argument("--num_translations",
                    type=int,
                    default=0)

parser.add_argument("--num_zooms",
                    type=int,
                    default=0)

parser.add_argument("--num_swaps",
                    type=int,
                    default=0)

parser.add_argument("--num_rotations",
                    type=int,
                    default=0)

parser.add_argument("--ker_alignment_sample_size",
                    type=int,
                    default=0)

parser.add_argument("--effective_rank_sample_size",
                    type=int,
                    default=0)

parser.add_argument("--rotation_angle",
                    type=float,
                    default=0.25)

parser.add_argument("--use_augmentation",
                    type=bool,
                    default=False)

parser.add_argument("--train_linear_model",
                    type=bool,
                    default=False)

parser.add_argument("--use_conjugate_kernel",
                    type=bool,
                    default=False)

parser.add_argument("--plus_class",
                    type=int,
                    default=8)

parser.add_argument("--minus_class",
                    type=int,
                    default=3)

parser.add_argument("--dataset",
                    type=str,
                    default="mnist",
                    choices=["mnist", "cifar10"])

parser.add_argument("--nn_architecture",
                    type=str,
                    default="VGG",
                    choices=["VGG", "MLP", "Sum"])

parser.add_argument("--svm_solver",
                    type=str,
                    default="jst1",
                    choices=["sklearn", "jst1", "jst2"])

parser.add_argument("--svm_time_limit",
                    type=float,
                    default=120.0)

args = parser.parse_args()

LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True)
SHUFFLE_SIZE = 10000
FEATURE_COMPUTATION_BATCH_SIZE = 32
WRITE_TRY_LIMIT = 3
SLEEP_TIME = 60
EPSILON = 0.0001
# for training, in degrees
ROTATION_RANGE = 15
# for training, in pixels
WIDTH_SHIFT_RANGE = 1
HEIGHT_SHIFT_RANGE = 1
# for measuring zoom invariance
CROP_FRACTION = 13.0/14.0
# when training to a particular loss target, start with this number of
# epochs, and double each time you fail to achieve the loss target
FIRST_PASS_EPOCHS = 10
# VERBOSITY 0 is silent, 1 is progress bar, and 2 is one line per epoch
VERBOSITY = 1


def margin(w, x_matrix, y):
  wnorm = np.linalg.norm(w)
  if wnorm < EPSILON:
    return 0.0
  else:
    yx_matrix = np.expand_dims((2.0*y.astype(float)-1.0), axis=1)*x_matrix
    margins = np.dot(yx_matrix, w)/wnorm
    return np.min(margins)


class ScikitLearnWithTimeLimit:
  """Try to learn a hard-margin SVM using sklearn, with a time limit."""

  def __init__(self, time_limit):
    self.time_limit = time_limit
    self.submodel = sklearn.svm.LinearSVC(C=100, fit_intercept=False)

  def fit(self, x_matrix, y):
    start_time = time.time()
    while time.time() - start_time < self.time_limit:
      self.submodel.fit(x_matrix, y)

  def predict(self, x_matrix):
    return self.submodel.predict(x_matrix)

  def weights(self):
    return np.squeeze(self.submodel.coef_)


class JSTtwo:
  """Implements Algorithm 2 from https://arxiv.org/abs/2107.00595.

  Uses the hyperparameters that worked the best in Figure 3
  of that paper: beta_t = t/(t+1) and theta_t = 1.

  Each call to fit starts from scratch.
  """

  def __init__(self, time_limit, clip_threshold=16.0):
    self.w = None
    self.g = None
    self.time_limit = time_limit
    self.clip_threshold = clip_threshold

  def fit(self, x_matrix, y):
    """Fit training data."""
    start_time = time.time()
    n, p = x_matrix.shape
    yx_matrix = -(2.0*y.astype(float)-1.0).reshape(n, 1)*x_matrix
    self.w = np.zeros(p)
    g = np.zeros(p)
    q = np.ones(n)/float(n)
    t = 0.0
    while time.time() - start_time < self.time_limit:
      i = np.random.choice(n, p=q)
      g = (t/(t+1.0))*(g + yx_matrix[i])
      self.w -= (g + yx_matrix[i])
      unnormalized_q = np.exp(np.clip(np.dot(yx_matrix, self.w),
                                      -self.clip_threshold,
                                      self.clip_threshold))
      q = unnormalized_q/np.sum(unnormalized_q)
      t += 1.0

  def predict(self, x_matrix):
    return (np.dot(x_matrix, self.w) > 0.0).astype(int)

  def weights(self):
    return self.w


class JSTone:
  """Implements Algorithm 1 from https://arxiv.org/abs/2107.00595.

  Adapted from code provided by Ziwei Ji.
  """

  def __init__(self, time_limit):
    self.w = None
    self.time_limit = time_limit

  def fit(self, x_matrix, y):
    """Fix the training data."""
    start_time = time.time()
    n, p = x_matrix.shape
    # transform x_matrix so that all classifications are 1
    x_matrix = ((2.0*y.astype(float)-1.0).reshape(n, 1))*x_matrix
    x_matrix_tf = tf.constant(x_matrix, dtype=tf.float32)
    w_tf = tf.zeros(p)
    g = tf.zeros(p)
    t = 0.0
    while time.time() - start_time < self.time_limit:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(w_tf)
        pred = tf.linalg.matvec(x_matrix_tf, w_tf)
        loss = tf.math.reduce_logsumexp(-pred)

      w_grad = tape.gradient(loss, w_tf)
      g = (g + w_grad) * t / (t + 1)
      w_tf = w_tf - (g + w_grad)

      t += 1.0

    self.w = w_tf.numpy()

  def predict(self, x_matrix):
    return (np.dot(x_matrix, self.w) > 0.0).astype(int)

  def weights(self):
    return self.w


class ModelWithFeatures(tf.keras.Model):
  """Neural network model with additional functions that output features."""

  def __init__(self, nn_architecture, width):
    super(ModelWithFeatures, self).__init__()
    self.nn_architecture = nn_architecture
    if nn_architecture == "VGG":
      self.build_VGG_model(width)
    elif nn_architecture == "MLP":
      self.build_MLP_model(width)
    else:
      self.build_Sum_model(width)

  def build_VGG_model(self, width):
    layers_per_block = int(args.num_layers/args.num_blocks)
    self.layer_list = list()
    num_channels = width
    for _ in range(args.num_blocks):
      for _ in range(layers_per_block):
        self.layer_list.append(tf.keras.layers.Conv2D(num_channels,
                                                      args.filter_size,
                                                      activation=tf.nn.relu,
                                                      padding="same"))
      self.layer_list.append(tf.keras.layers.MaxPool2D(
          pool_size=(args.pool_size,
                     args.pool_size),
          strides=(args.pool_size,
                   args.pool_size),
          padding="same"))
      num_channels *= 2
    self.layer_list.append(tf.keras.layers.Flatten())
    self.layer_list.append(tf.keras.layers.Dense(1))

  def build_MLP_model(self, width):
    self.layer_list = list()
    self.layer_list.append(tf.keras.layers.Flatten())
    for _ in range(args.num_layers):
      self.layer_list.append(tf.keras.layers.Dense(width,
                                                   activation=tf.nn.relu))
    self.layer_list.append(tf.keras.layers.Dense(1))

  def build_Sum_model(self, width):
    self.layer_list = [tf.keras.layers.Flatten(),
                       tf.keras.layers.Dense(width,
                                             activation=tf.nn.relu),
                       tf.keras.layers.Dense(1)
                       ]

  # given data, and a function for computing features on
  # a batch, compute features for all of the examples in x
  def all_features(self, x, phi):
    num_examples = x.shape[0]
    begin_index = 0
    while begin_index < num_examples:
      end_index = min(begin_index + FEATURE_COMPUTATION_BATCH_SIZE,
                      num_examples)
      batch_features = phi(x[begin_index:end_index, :, :, :])
      if begin_index == 0:
        features = np.zeros([num_examples, batch_features.shape[1]])
      features[begin_index:end_index, :] = batch_features
      begin_index = end_index
    return features

  def conjugate_ker_features(self, x):
    def batch_ck_features(x_batch):
      hidden = x_batch
      for ell in range(len(self.layer_list)-1):
        hidden = (self.layer_list[ell])(hidden)
      return hidden
    return self.all_features(x, batch_ck_features)

  def grad_features(self, x):
    def batch_grad_features(x_batch):
      num_examples = x_batch.shape[0]
      with tf.GradientTape() as g:
        yhat = self(x_batch, training=False)
      features_with_original_shape = g.jacobian(yhat,
                                                self.variables)
      num_features = 0
      for feature_tensor in features_with_original_shape:
        if feature_tensor is not None:
          num_features += np.prod(feature_tensor.shape.as_list()[1:])
      features = np.zeros([num_examples, num_features])

      begin_index = 0
      for feature_tensor in features_with_original_shape:
        if feature_tensor is not None:
          this_size = np.prod(feature_tensor.shape.as_list()[1:])
          end_index = begin_index + this_size
          feature_slice = np.reshape(feature_tensor.numpy(),
                                     [num_examples, -1])
          features[:, begin_index:end_index] = feature_slice
          begin_index = end_index
      return features

    # call to make sure that the weights are initialized
    _ = self(x[:1], training=False)
    return self.all_features(x, batch_grad_features)

  def call(self, inputs):
    if self.nn_architecture == "Sum":
      hidden = tf.reduce_sum(inputs, axis=[1, 2, 3], keepdims=True)
    else:
      hidden = inputs
    for layer in self.layer_list:
      hidden = layer(hidden)
    return hidden


def transform_dataset(raw_x, raw_y):
  mask = np.logical_or(raw_y == args.minus_class,
                       raw_y == args.plus_class)
  temp_x = raw_x[mask, :, :, :]
  temp_x = temp_x/255.0
  temp_y = np.zeros(raw_y[mask].size, dtype="int")
  temp_y[raw_y[mask] == args.plus_class] = 1
  return temp_x, temp_y


# measure the alignment between the gradient kernel and the
# conjugate kernel
def kalignment(model, x):
  gradient_features = extract_features(model, x, False)
  conjugate_features = extract_features(model, x, True)
  gradient_gram = np.matmul(gradient_features,
                            np.transpose(gradient_features))
  conjugate_gram = np.matmul(conjugate_features,
                             np.transpose(conjugate_features))
  return (np.sum(gradient_gram*conjugate_gram)/
          (np.linalg.norm(gradient_gram)
           *np.linalg.norm(conjugate_gram)))


def eff_rank(model, x):
  features = extract_features(model, x, args.use_conjugate_kernel)
  gram_matrix = np.matmul(features, np.transpose(features))
  eigenvalues = np.linalg.eigvalsh(gram_matrix)
  return np.sum(eigenvalues)/np.max(eigenvalues)


# this extracts features from a model, either using the NTK, or using the
# conjugate kernel
def extract_features(model, x, use_conjugate_kernel):
  if use_conjugate_kernel:
    return model.conjugate_ker_features(x)
  else:
    return model.grad_features(x)


# This is only an approximate
# count
def calc_vgg_param_count(image_size,
                         num_input_channels,
                         width,
                         n_classes):

  """Returns number of parameters in a VGG model.

  Args:
    image_size: height and width of input images
    num_input_channels: number of channels in input images
    width: number of channels in the first hidden layer of the VGG net
    n_classes: number of classes predicted by the model.
  """
  filter_size = args.filter_size
  running_count = 0
  num_channels = num_input_channels
  layers_per_block = int(args.num_layers/args.num_blocks)
  field_size = image_size
  for b in range(args.num_blocks):
    if b == 0:
      new_channels = width
    else:
      new_channels = 2*num_channels
    running_count += (num_channels*new_channels
                      *(1 + filter_size*filter_size))
    num_channels = new_channels
    for _ in range(1, layers_per_block):
      running_count += (num_channels*num_channels
                        *(1 + filter_size*filter_size))
    field_size /= 2
  if n_classes == 2:
    running_count += (field_size*field_size*num_channels+1)
  else:
    running_count += (field_size*field_size*num_channels*n_classes+1)
  return running_count


def normalize_rows(x):
  return x / (EPSILON + np.linalg.norm(x, axis=1).reshape(x.shape[0], 1))


# given two matrices of the same shape, compute
# the average of the cosine similarities between
# their rows
def ave_cosine(xone, xtwo):
  normalized_xone = normalize_rows(xone)
  normalized_xtwo = normalize_rows(xtwo)
  return np.average(np.sum(normalized_xone * normalized_xtwo, axis=1))


def calc_trans_inv(model, x):
  """Returns the translation invariance of the kernel of a model.

  Args:
    model: neural network model
    x: a (mini)batch of inputs
  """

  original_features = extract_features(model, x, args.use_conjugate_kernel)
  translated_right_x = tfa.image.translate(x, [0, 1])
  translated_right_features = extract_features(model,
                                               translated_right_x,
                                               args.use_conjugate_kernel)
  translated_down_x = tfa.image.translate(x, [1, 0])
  translated_down_features = extract_features(model,
                                              translated_down_x,
                                              args.use_conjugate_kernel)
  return 0.5*(ave_cosine(original_features,
                         translated_right_features)
              + ave_cosine(original_features,
                           translated_down_features))


# this measures the zoom invariance of the kernel of a model
def calc_zoom_inv(model, x, image_size):
  original_features = extract_features(model,
                                       x,
                                       args.use_conjugate_kernel)
  cropped_x = tf.image.central_crop(x, CROP_FRACTION)
  zoomed_x = tf.image.resize(cropped_x, [image_size, image_size])
  zoomed_features = extract_features(model,
                                     zoomed_x,
                                     args.use_conjugate_kernel)
  return ave_cosine(original_features, zoomed_features)


# this measures the rotation invariance of the kernel associated with a
# model
def calc_rotation_inv(model, x):
  """Returns the rotation invariance of the kernel of a model.

  Args:
    model: neural network model
    x: a (mini)batch of inputs
  """
  original_features = extract_features(model,
                                       x,
                                       args.use_conjugate_kernel)
  rotated_pos_x = tfa.image.rotate(x, args.rotation_angle)
  rotated_pos_features = extract_features(model,
                                          rotated_pos_x,
                                          args.use_conjugate_kernel)
  rotated_neg_x = tfa.image.rotate(x, -args.rotation_angle)
  rotated_neg_features = extract_features(model,
                                          rotated_neg_x,
                                          args.use_conjugate_kernel)
  return 0.5*(ave_cosine(original_features,
                         rotated_pos_features)
              + ave_cosine(original_features,
                           rotated_neg_features))


def calc_swap_inv(model, x, image_size):
  """Measures 'locality' of features."""
  original_features = extract_features(model,
                                       x,
                                       args.use_conjugate_kernel)
  slice_size = int(image_size/2)
  x_copy = x.copy()
  upper_left = x_copy[:, :slice_size, :slice_size, :].copy()
  lower_right = x_copy[:, slice_size:, slice_size:, :].copy()
  x_copy[:, :slice_size, :slice_size, :] = lower_right
  x_copy[:, slice_size:, slice_size:, :] = upper_left
  perturbed_x = tf.constant(x_copy, dtype=tf.float32)
  perturbed_features = extract_features(model,
                                        perturbed_x,
                                        args.use_conjugate_kernel)
  return ave_cosine(original_features, perturbed_features)


def stddev(sum_of_squares, plain_sum, n):
  n = float(n)
  if sum_of_squares > plain_sum*plain_sum/n:
    return math.sqrt((sum_of_squares - plain_sum*plain_sum/n)/
                     (n-1.0))
  else:
    return 0.0

# suppress retracing warnings
tf.get_logger().setLevel("ERROR")
np.set_printoptions(precision=3, suppress=True)

if args.dataset == "mnist":
  dataset_image_size = 28
  ds_num_input_channels = 1
  ds_train = tfds.load("mnist",
                       split="train",
                       as_supervised=True,
                       batch_size=-1
                       )
  ds_test = tfds.load("mnist",
                      split="test",
                      as_supervised=True,
                      batch_size=-1
                      )
elif args.dataset == "cifar10":
  dataset_image_size = 32
  ds_num_input_channels = 3
  ds_train = tfds.load("cifar10",
                       split="train",
                       as_supervised=True,
                       batch_size=-1
                       )
  ds_test = tfds.load("cifar10",
                      split="test",
                      as_supervised=True,
                      batch_size=-1
                      )
x_train, y_train = tfds.as_numpy(ds_train)
x_test, y_test = tfds.as_numpy(ds_test)
x_train, y_train = transform_dataset(x_train, y_train)
x_test, y_test = transform_dataset(x_test, y_test)
num_classes = 2

sum_nn_error = 0.0
sumsq_nn_error = 0.0
if args.train_linear_model:
  sum_linear_error = 0.0
  sumsq_linear_error = 0.0
  sum_margin = 0.0
  sumsq_margin = 0.0
if args.num_translations > 0:
  sum_ntk_trans_inv = 0.0
  sumsq_ntk_trans_inv = 0.0
  sum_ak_trans_inv = 0.0
  sumsq_ak_trans_inv = 0.0
if args.num_zooms > 0:
  sum_ntk_zoom_inv = 0.0
  sumsq_ntk_zoom_inv = 0.0
  sum_ak_zoom_inv = 0.0
  sumsq_ak_zoom_inv = 0.0
if args.num_swaps > 0:
  sum_ntk_swap_inv = 0.0
  sumsq_ntk_swap_inv = 0.0
  sum_ak_swap_inv = 0.0
  sumsq_ak_swap_inv = 0.0
if args.num_rotations > 0:
  sum_ntk_rotation_inv = 0.0
  sumsq_ntk_rotation_inv = 0.0
  sum_ak_rotation_inv = 0.0
  sumsq_ak_rotation_inv = 0.0
if args.ker_alignment_sample_size > 0:
  sum_ntk_kalignment = 0.0
  sumsq_ntk_kalignment = 0.0
  sum_ak_kalignment = 0.0
  sumsq_ak_kalignment = 0.0
if args.effective_rank_sample_size > 0:
  sum_ntk_eff_rank = 0.0
  sumsq_ntk_eff_rank = 0.0
  sum_ak_eff_rank = 0.0
  sumsq_ak_eff_rank = 0.0
for r in range(args.num_runs):
  if args.nn_architecture == "VGG":
    nn_width = 4
    parameter_count = calc_vgg_param_count(dataset_image_size,
                                           ds_num_input_channels,
                                           nn_width,
                                           num_classes)
    while parameter_count < args.num_parameters:
      nn_width += 1
      parameter_count = calc_vgg_param_count(dataset_image_size,
                                             ds_num_input_channels,
                                             nn_width,
                                             num_classes)
    nn_model = ModelWithFeatures("VGG", nn_width)
  elif args.nn_architecture == "MLP":
    input_size = dataset_image_size*dataset_image_size*ds_num_input_channels
    nn_width = 4
    parameter_count = 0
    while parameter_count < args.num_parameters:
      nn_width += 1
      parameter_count = (input_size*nn_width
                         +(args.num_layers-1)*(nn_width+1)*nn_width
                         +nn_width+1)
    nn_model = ModelWithFeatures("MLP", nn_width)
  else:
    nn_width = args.num_parameters
    nn_model = ModelWithFeatures("Sum", nn_width)
  if args.num_translations > 0:
    this_ntk_trans_inv = calc_trans_inv(nn_model,
                                        x_test[:args.num_translations])
    sum_ntk_trans_inv += this_ntk_trans_inv
    sumsq_ntk_trans_inv += (this_ntk_trans_inv
                            *this_ntk_trans_inv)
  if args.num_zooms > 0:
    this_ntk_zoom_inv = calc_zoom_inv(nn_model,
                                      x_test[:args.num_zooms],
                                      dataset_image_size)
    sum_ntk_zoom_inv += this_ntk_zoom_inv
    sumsq_ntk_zoom_inv += (this_ntk_zoom_inv*this_ntk_zoom_inv)
  if args.num_swaps > 0:
    this_ntk_swap_inv = calc_swap_inv(nn_model,
                                      x_test[:args.num_swaps],
                                      dataset_image_size)
    sum_ntk_swap_inv += this_ntk_swap_inv
    sumsq_ntk_swap_inv += (this_ntk_swap_inv*this_ntk_swap_inv)
  if args.num_rotations > 0:
    this_ntk_rotation_inv = calc_rotation_inv(nn_model,
                                              x_test[:args.num_rotations])
    sum_ntk_rotation_inv += this_ntk_rotation_inv
    sumsq_ntk_rotation_inv += (this_ntk_rotation_inv*this_ntk_rotation_inv)
  if args.ker_alignment_sample_size > 0:
    this_ntk_kalignment = kalignment(nn_model,
                                     x_test[:args.ker_alignment_sample_size])
    sum_ntk_kalignment += this_ntk_kalignment
    sumsq_ntk_kalignment += (this_ntk_kalignment*this_ntk_kalignment)
  if args.effective_rank_sample_size > 0:
    this_ntk_eff_rank = eff_rank(nn_model,
                                 x_test[:args.eff_rank_sample_size])
    sum_ntk_eff_rank += this_ntk_eff_rank
    sumsq_ntk_eff_rank += (this_ntk_eff_rank*this_ntk_eff_rank)
  if args.learning_rate is not None:
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
  else:
    optimizer = tf.keras.optimizers.SGD()
  nn_model.compile(optimizer=optimizer,
                   loss=LOSS,
                   metrics=["accuracy"])

  if args.max_num_epochs > 0:
    if args.use_augmentation:
      print("Using data augmentation")
      idg = tf.keras.preprocessing.image.ImageDataGenerator
      datagen = idg(rotation_range=ROTATION_RANGE,
                    width_shift_range=WIDTH_SHIFT_RANGE,
                    height_shift_range=HEIGHT_SHIFT_RANGE)
      datagen.fit(x_train)

    if args.training_loss_target:
      total_epochs = 0
      training_loss = None
      this_num_epochs = FIRST_PASS_EPOCHS
      while ((not training_loss)
             or ((total_epochs < args.max_num_epochs)
                 and training_loss > args.training_loss_target)):
        this_num_epochs = min(this_num_epochs,
                              args.max_num_epochs-total_epochs)
        if args.use_augmentation:
          nn_model.fit(datagen.flow(x_train,
                                    y_train,
                                    batch_size=32),
                       epochs=this_num_epochs,
                       verbose=VERBOSITY)
        else:
          nn_model.fit(x_train,
                       y_train,
                       epochs=this_num_epochs,
                       verbose=VERBOSITY)
        training_loss, _ = nn_model.evaluate(x_train,
                                             y_train,
                                             verbose=VERBOSITY)
        total_epochs += this_num_epochs
        this_num_epochs *= 2
    else:
      if args.use_augmentation:
        nn_model.fit(datagen.flow(x_train, y_train, batch_size=32),
                     epochs=args.max_num_epochs,
                     verbose=VERBOSITY)
      else:
        nn_model.fit(x_train,
                     y_train,
                     epochs=args.max_num_epochs,
                     verbose=VERBOSITY)

  if args.num_translations > 0:
    this_ak_trans_inv = calc_trans_inv(nn_model,
                                       x_test[:args.num_translations])
    sum_ak_trans_inv += this_ak_trans_inv
    sumsq_ak_trans_inv += (this_ak_trans_inv*this_ak_trans_inv)
  if args.num_zooms > 0:
    this_ak_zoom_inv = calc_zoom_inv(nn_model,
                                     x_test[:args.num_zooms],
                                     dataset_image_size)
    sum_ak_zoom_inv += this_ak_zoom_inv
    sumsq_ak_zoom_inv += (this_ak_zoom_inv*this_ak_zoom_inv)
  if args.num_swaps > 0:
    this_ak_swap_inv = calc_swap_inv(nn_model,
                                     x_test[:args.num_swaps],
                                     dataset_image_size)
    sum_ak_swap_inv += this_ak_swap_inv
    sumsq_ak_swap_inv += (this_ak_swap_inv*this_ak_swap_inv)
  if args.num_rotations > 0:
    this_ak_rotation_inv = calc_rotation_inv(nn_model,
                                             x_test[:args.num_rotations])
    sum_ak_rotation_inv += this_ak_rotation_inv
    sumsq_ak_rotation_inv += (this_ak_rotation_inv*this_ak_rotation_inv)
  if args.ker_alignment_sample_size > 0:
    this_ak_kalignment = kalignment(nn_model,
                                    x_test[:args.ker_alignment_sample_size])
    sum_ak_kalignment += this_ak_kalignment
    sumsq_ak_kalignment += (this_ak_kalignment*this_ak_kalignment)
  if args.effective_rank_sample_size > 0:
    this_ak_eff_rank = eff_rank(nn_model,
                                x_test[:args.eff_rank_sample_size])
    sum_ak_eff_rank += this_ak_eff_rank
    sumsq_ak_eff_rank += (this_ak_eff_rank*this_ak_eff_rank)

  if args.train_linear_model:
    print("Extracting features on the training data")
    phi_train = extract_features(nn_model, x_train, args.use_conjugate_kernel)
    print("Extracting features on the test data")
    phi_test = extract_features(nn_model, x_test, args.use_conjugate_kernel)
    print("Normalizing feature vectors")
    Z = EPSILON + np.average(np.linalg.norm(phi_train, axis=1))
    phi_train /= Z
    phi_test /= Z
    print("Done normalizing")

    if args.svm_solver == "sklearn":
      linear_model = ScikitLearnWithTimeLimit(args.svm_time_limit)
    elif args.svm_solver == "jst1":
      linear_model = JSTone(args.svm_time_limit)
    elif args.svm_solver == "jst2":
      linear_model = JSTtwo(args.svm_time_limit)

    linear_model.fit(phi_train, y_train)
    skl_score = sklearn.metrics.accuracy_score
    this_linear_error = 1.0 - skl_score(linear_model.predict(phi_test),
                                        y_test)
    sum_linear_error += this_linear_error
    sumsq_linear_error += (this_linear_error*this_linear_error)
    this_margin = margin(linear_model.weights(), phi_train, y_train)
    sum_margin += this_margin
    sumsq_margin += (this_margin*this_margin)

  _, nn_accuracy = nn_model.evaluate(x_test, y_test)
  this_nn_error = 1.0 - nn_accuracy

  sum_nn_error += this_nn_error
  sumsq_nn_error += (this_nn_error*this_nn_error)

  print("running_nn_error = {}".format(sum_nn_error/(r+1)))
  if args.train_linear_model:
    print("running_linear_error = {}".format(sum_linear_error/(r+1)))
    print("running_margin = {}".format(sum_margin/(r+1)))
  if args.num_translations > 0:
    print("running_ntk_trans_inv = {}".format(sum_ntk_trans_inv/(r+1)))
    print("running_ak_trans_inv = {}".format(sum_ak_trans_inv/(r+1)))
  if args.num_zooms > 0:
    print("running_ntk_zoom_inv = {}".format(sum_ntk_zoom_inv/(r+1)))
    print("running_ak_zoom_inv = {}".format(sum_ak_zoom_inv/(r+1)))
  if args.num_swaps > 0:
    print("running_ntk_swap_inv = {}".format(sum_ntk_swap_inv/(r+1)))
    print("running_ak_swap_inv = {}".format(sum_ak_swap_inv/(r+1)))
  if args.num_rotations > 0:
    print("running_ntk_rotation_inv = {}".format(sum_ntk_rotation_inv/(r+1)))
    print("running_ak_rotation_inv = {}".format(sum_ak_rotation_inv/(r+1)))
  if args.ker_alignment_sample_size > 0:
    print("running_ntk_kalignment = {}".format(sum_ntk_kalignment/(r+1)))
    print("running_ak_kalignment = {}".format(sum_ak_kalignment/(r+1)))
  if args.effective_rank_sample_size > 0:
    print("running_ntk_eff_rank = {}".format(sum_ntk_eff_rank/(r+1)))
    print("running_ak_eff_rank = {}".format(sum_ak_eff_rank/(r+1)))

ave_nn_error = sum_nn_error/args.num_runs
if args.num_runs > 1:
  stddev_nn_error = stddev(sumsq_nn_error,
                           sum_nn_error,
                           args.num_runs)
  ciwidth_nn_error = stddev_nn_error/math.sqrt(args.num_runs)
if args.train_linear_model:
  ave_linear_error = sum_linear_error/args.num_runs
  if args.num_runs > 1:
    stddev_linear_error = stddev(sumsq_linear_error,
                                 sum_linear_error,
                                 args.num_runs)
    ciwidth_linear_error = stddev_linear_error/math.sqrt(args.num_runs)
  ave_margin = sum_margin/args.num_runs
  if args.num_runs > 1:
    stddev_margin = stddev(sumsq_margin,
                           sum_margin,
                           args.num_runs)
    ciwidth_margin = stddev_margin/math.sqrt(args.num_runs)
if args.num_translations > 0:
  ave_ntk_trans_inv = sum_ntk_trans_inv/args.num_runs
  if args.num_runs > 1:
    stddev_ntk_trans_inv = stddev(sumsq_ntk_trans_inv,
                                  sum_ntk_trans_inv,
                                  args.num_runs)
    ciwidth_ntk_trans_inv = stddev_ntk_trans_inv/math.sqrt(args.num_runs)
  ave_ak_trans_inv = sum_ak_trans_inv/args.num_runs
  if args.num_runs > 1:
    stddev_ak_trans_inv = stddev(sumsq_ak_trans_inv,
                                 sum_ak_trans_inv,
                                 args.num_runs)
    ciwidth_ak_trans_inv = stddev_ak_trans_inv/math.sqrt(args.num_runs)
if args.num_zooms > 0:
  ave_ntk_zoom_inv = sum_ntk_zoom_inv/args.num_runs
  if args.num_runs > 1:
    stddev_ntk_zoom_inv = stddev(sumsq_ntk_zoom_inv,
                                 sum_ntk_zoom_inv,
                                 args.num_runs)
    ciwidth_ntk_zoom_inv = stddev_ntk_zoom_inv/math.sqrt(args.num_runs)
  ave_ak_zoom_inv = sum_ak_zoom_inv/args.num_runs
  if args.num_runs > 1:
    stddev_ak_zoom_inv = stddev(sumsq_ak_zoom_inv,
                                sum_ak_zoom_inv,
                                args.num_runs)
    ciwidth_ak_zoom_inv = stddev_ak_zoom_inv/math.sqrt(args.num_runs)
if args.num_rotations > 0:
  ave_ntk_rotation_inv = sum_ntk_rotation_inv/args.num_runs
  if args.num_runs > 1:
    stddev_ntk_rotation_inv = stddev(sumsq_ntk_rotation_inv,
                                     sum_ntk_rotation_inv,
                                     args.num_runs)
    ciwidth_ntk_rotation_inv = stddev_ntk_rotation_inv/math.sqrt(args.num_runs)
  ave_ak_rotation_inv = sum_ak_rotation_inv/args.num_runs
  if args.num_runs > 1:
    stddev_ak_rotation_inv = stddev(sumsq_ak_rotation_inv,
                                    sum_ak_rotation_inv,
                                    args.num_runs)
    ciwidth_ak_rotation_inv = stddev_ak_rotation_inv/math.sqrt(args.num_runs)
if args.ker_alignment_sample_size > 0:
  ave_ntk_kalignment = sum_ntk_kalignment/args.num_runs
  if args.num_runs > 1:
    stddev_ntk_kalignment = stddev(sumsq_ntk_kalignment,
                                   sum_ntk_kalignment,
                                   args.num_runs)
    ciwidth_ntk_kalignment = stddev_ntk_kalignment/math.sqrt(args.num_runs)
  ave_ak_kalignment = sum_ak_kalignment/args.num_runs
  if args.num_runs > 1:
    stddev_ak_kalignment = stddev(sumsq_ak_kalignment,
                                  sum_ak_kalignment,
                                  args.num_runs)
    ciwidth_ak_kalignment = stddev_ak_kalignment/math.sqrt(args.num_runs)
if args.num_swaps > 0:
  ave_ntk_swap_inv = sum_ntk_swap_inv/args.num_runs
  if args.num_runs > 1:
    stddev_ntk_swap_inv = stddev(sumsq_ntk_swap_inv,
                                 sum_ntk_swap_inv,
                                 args.num_runs)
    ciwidth_ntk_swap_inv = stddev_ntk_swap_inv/math.sqrt(args.num_runs)
  ave_ak_swap_inv = sum_ak_swap_inv/args.num_runs
  if args.num_runs > 1:
    stddev_ak_swap_inv = stddev(sumsq_ak_swap_inv,
                                sum_ak_swap_inv,
                                args.num_runs)
    ciwidth_ak_swap_inv = stddev_ak_swap_inv/math.sqrt(args.num_runs)
if args.effective_rank_sample_size > 0:
  ave_ntk_eff_rank = sum_ntk_eff_rank/args.num_runs
  if args.num_runs > 1:
    stddev_ntk_eff_rank = stddev(sumsq_ntk_eff_rank,
                                 sum_ntk_eff_rank,
                                 args.num_runs)
    ciwidth_ntk_eff_rank = stddev_ntk_eff_rank/math.sqrt(args.num_runs)
  ave_ak_eff_rank = sum_ak_eff_rank/args.num_runs
  if args.num_runs > 1:
    stddev_ak_eff_rank = stddev(sumsq_ak_eff_rank,
                                sum_ak_eff_rank,
                                args.num_runs)
    ciwidth_ak_eff_rank = stddev_ak_eff_rank/math.sqrt(args.num_runs)

wrote = False
write_tries = 0
while (not wrote) and (write_tries < WRITE_TRY_LIMIT):
  # provisionally assume that we wrote
  wrote = True
  try:
    fout = open(args.test_accuracy_output, mode="a")
    format_string = "{}\t"*37 + "{}\n"
    fout.write(format_string.format(
        args.max_num_epochs,
        args.use_augmentation,
        args.use_conjugate_kernel,
        ave_nn_error,
        (ciwidth_nn_error
         if args.num_runs > 1
         else ""),
        ave_linear_error if args.train_linear_model else "",
        (ciwidth_linear_error
         if args.train_linear_model and args.num_runs > 1
         else ""),
        ave_margin if args.train_linear_model else "",
        (ciwidth_margin
         if args.train_linear_model and args.num_runs > 1
         else ""),
        ave_ntk_trans_inv if args.num_translations > 0 else "",
        (ciwidth_ntk_trans_inv
         if args.num_translations > 0 and args.num_runs > 1
         else ""),
        ave_ak_trans_inv if args.num_translations > 0 else "",
        (ciwidth_ak_trans_inv
         if args.num_translations > 0 and args.num_runs > 1
         else ""),
        ave_ntk_zoom_inv if args.num_zooms > 0 else "",
        (ciwidth_ntk_zoom_inv
         if args.num_zooms > 0 and args.num_runs > 1
         else ""),
        ave_ak_zoom_inv if args.num_zooms > 0 else "",
        (ciwidth_ak_zoom_inv
         if args.num_zooms > 0 and args.num_runs > 1
         else ""),
        ave_ntk_rotation_inv if args.num_rotations > 0 else "",
        (ciwidth_ntk_rotation_inv
         if args.num_rotations > 0 and args.num_runs > 1
         else ""),
        ave_ak_rotation_inv if args.num_rotations > 0 else "",
        (ciwidth_ak_rotation_inv
         if args.num_rotations > 0 and args.num_runs > 1
         else ""),
        (ave_ntk_kalignment
         if args.ker_alignment_sample_size > 0
         else ""),
        (ciwidth_ntk_kalignment
         if (args.ker_alignment_sample_size > 0 and args.num_runs > 1)
         else ""),
        (ave_ak_kalignment
         if args.ker_alignment_sample_size > 0
         else ""),
        (ciwidth_ak_kalignment
         if (args.ker_alignment_sample_size > 0 and args.num_runs > 1)
         else ""),
        ave_ntk_swap_inv if args.num_swaps > 0 else "",
        (ciwidth_ntk_swap_inv
         if args.num_swaps > 0 and args.num_runs > 1
         else ""),
        ave_ak_swap_inv if args.num_swaps > 0 else "",
        (ciwidth_ak_swap_inv
         if args.num_swaps > 0 and args.num_runs > 1
         else ""),
        (ave_ntk_eff_rank
         if args.effective_rank_sample_size > 0
         else ""),
        (ciwidth_ntk_eff_rank
         if (args.effective_rank_sample_size > 0 and args.num_runs > 1)
         else ""),
        (ave_ak_eff_rank
         if args.effective_rank_sample_size > 0
         else ""),
        (ciwidth_ak_eff_rank
         if (args.effective_rank_sample_size > 0 and args.num_runs > 1)
         else ""),
        nn_model.count_params(),
        args.nn_architecture,
        args.svm_solver if args.train_linear_model else "",
        args.svm_time_limit if args.train_linear_model else "",
        args.learning_rate))
    fout.close()
  except OSError as e:
    print("Write failed with exception {}, retrying".format(e))
    sys.stdout.flush()
    wrote = False
    write_tries += 1
    time.sleep(random.randrange(SLEEP_TIME))

if not os.path.exists(args.model_summary_output):
  fmout = open(args.model_summary_output, mode="a")
  nn_model.summary(print_fn=lambda x: fmout.write(x + "\n"))
  fmout.close()
