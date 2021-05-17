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

parser.add_argument('--num_epochs',
                    type=int,
                    default=1000)

parser.add_argument('--test_accuracy_output',
                    type=str,
                    default='/tmp/ak.txt')

parser.add_argument('--model_summary_output',
                    type=str,
                    default='/tmp/ak_model.txt')

parser.add_argument('--filter_size',
                    type=int,
                    default=3)

parser.add_argument('--pool_size',
                    type=int,
                    default=2)

# This is only approximate
parser.add_argument('--num_parameters',
                    type=int,
                    default=100000)

parser.add_argument('--num_blocks',
                    type=int,
                    default=2)

parser.add_argument('--layers',
                    type=int,
                    default=4)

parser.add_argument('--num_runs',
                    type=int,
                    default=5)

parser.add_argument('--num_translations',
                    type=int,
                    default=0)

parser.add_argument('--num_zooms',
                    type=int,
                    default=0)

parser.add_argument('--num_swaps',
                    type=int,
                    default=0)

parser.add_argument('--num_rotations',
                    type=int,
                    default=0)

parser.add_argument('--ker_alignment_sample_size',
                    type=int,
                    default=0)

parser.add_argument('--effective_rank_sample_size',
                    type=int,
                    default=0)

parser.add_argument('--rotation_angle',
                    type=float,
                    default=0.25)

parser.add_argument('--use_augmentation',
                    type=bool,
                    default=False)

parser.add_argument('--train_linear_model',
                    type=bool,
                    default=False)

parser.add_argument('--use_conjugate_kernel',
                    type=bool,
                    default=False)

parser.add_argument('--plus_class',
                    type=int,
                    default=8)

parser.add_argument('--minus_class',
                    type=int,
                    default=3)

parser.add_argument('--dataset',
                    type=str,
                    default='mnist',
                    choices=['mnist', 'cifar10'])

parser.add_argument('--nn_architecture',
                    type=str,
                    default='VGG',
                    choices=['VGG', 'MLP', 'Sum'])

args = parser.parse_args()

LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# This batch size is not for training, but for calculating features
BATCH_SIZE = 32
WRITE_TRY_LIMIT = 3
SLEEP_TIME = 10
EPSILON = 0.0001
# for training, in degrees
ROTATION_RANGE = 15
# for training, in pixels
WIDTH_SHIFT_RANGE = 1
HEIGHT_SHIFT_RANGE = 1
# for measuring zoom invariance
CROP_FRACTION = 13.0/14.0


class ModelWithFeatures(tf.keras.Model):
  """Neural network model with additional functions that output features."""

  def __init__(self, nn_architecture, model_width):
    super(ModelWithFeatures, self).__init__()
    self.nn_architecture = nn_architecture
    if nn_architecture == 'VGG':
      self.build_VGG_model(model_width)
    elif nn_architecture == 'MLP':
      self.build_MLP_model(model_width)
    else:
      self.build_Sum_model(model_width)

  def build_VGG_model(self, model_width):
    layers_per_block = int(args.layers/args.num_blocks)
    self.layer_list = list()
    num_channels = model_width
    pool_size = args.pool_size
    for _ in range(args.num_blocks):
      for _ in range(layers_per_block):
        self.layer_list.append(tf.keras.layers.Conv2D(num_channels,
                                                      args.filter_size,
                                                      activation=tf.nn.relu,
                                                      padding='same'))
      self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(pool_size,
                                                                  pool_size),
                                                       strides=(pool_size,
                                                                pool_size),
                                                       padding='same'))
      num_channels *= 2
    self.layer_list.append(tf.keras.layers.Flatten())
    self.layer_list.append(tf.keras.layers.Dense(1))

  def build_MLP_model(self, model_width):
    self.layer_list = list()
    self.layer_list.append(tf.keras.layers.Flatten())
    for _ in range(args.layers):
      self.layer_list.append(tf.keras.layers.Dense(model_width,
                                                   activation=tf.nn.relu))
    self.layer_list.append(tf.keras.layers.Dense(1))

  def build_Sum_model(self, model_width):
    self.layer_list = [tf.keras.layers.Flatten(),
                       tf.keras.layers.Dense(model_width,
                                             activation=tf.nn.relu),
                       tf.keras.layers.Dense(1)
                       ]

  def conjugate_kernel_features(self, x):
    hidden = x
    for ell in range(len(self.layer_list)-1):
      hidden = (self.layer_list[ell])(hidden)
    return hidden

  def grad_features(self, x):
    def batch_grad_features(x_batch):
      num_examples = x_batch.shape[0]
      with tf.GradientTape() as g:
        yhat = self(x_batch, training=False)
      features_with_original_shape = g.jacobian(yhat, self.variables)
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
          reshaped_features = np.reshape(feature_tensor.numpy(),
                                         [num_examples, -1])
          features[:, begin_index:end_index] = reshaped_features
          begin_index = end_index
      return features

    # call to make sure that the weights are initialized
    _ = self(x[:1], training=False)
    num_examples = x.shape[0]
    begin_index = 0
    while begin_index < num_examples:
      end_index = min(begin_index + BATCH_SIZE, num_examples)
      batch_features = batch_grad_features(x[begin_index:end_index, :, :, :])
      if begin_index == 0:
        features = np.zeros([num_examples, batch_features.shape[1]])
      features[begin_index:end_index, :] = batch_features
      begin_index = end_index
    return features

  def call(self, inputs):
    if self.nn_architecture == 'Sum':
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
  temp_y = np.zeros(raw_y[mask].size, dtype='int')
  temp_y[raw_y[mask] == args.plus_class] = 1
  temp_x = tf.constant(temp_x, dtype=tf.float32)
  temp_y = tf.constant(temp_y, dtype=tf.float32)
  return temp_x, temp_y


# measure the alignment between the gradient kernel and the
# conjugate kernel
def ker_alignment(model, x):
  gradient_features = extract_features(model, x, False)
  conjugate_features = extract_features(model, x, True)
  gradient_gram = np.matmul(gradient_features,
                            np.transpose(gradient_features))
  conjugate_gram = np.matmul(conjugate_features,
                             np.transpose(conjugate_features))
  return (np.sum(gradient_gram*conjugate_gram)/
          (np.linalg.norm(gradient_gram)*np.linalg.norm(conjugate_gram)))


def effective_rank(model, x):
  features = extract_features(model, x, args.use_conjugate_kernel)
  gram_matrix = np.matmul(features, np.transpose(features))
  eigenvalues = np.linalg.eigvalsh(gram_matrix)
  return np.sum(eigenvalues)/np.max(eigenvalues)


# this extracts features from a model, either using the NTK, or using the
# conjugate kernel
def extract_features(model, x, use_conjugate_kernel):
  if use_conjugate_kernel:
    return model.conjugate_kernel_features(x)
  else:
    return model.grad_features(x)


# This is only an approximate count
def calculate_vgg_parameter_count(image_size,
                                  num_input_channels,
                                  model_width,
                                  num_classes):
  """Returns number of parameters in a VGG model.

  Args:
    image_size: height and width of input images
    num_input_channels: number of channels in input images
    model_width: number of channels in the first hidden layer of the VGG net
    num_classes: number of classes predicted by the model
  """
  filter_size = args.filter_size
  running_count = 0
  num_channels = num_input_channels
  layers_per_block = int(args.layers/args.num_blocks)
  field_size = image_size
  for b in range(args.num_blocks):
    if b == 0:
      new_channels = model_width
    else:
      new_channels = 2*num_channels
    running_count += (num_channels*new_channels
                      *(1 + filter_size*filter_size))
    num_channels = new_channels
    for _ in range(1, layers_per_block):
      running_count += (num_channels*num_channels
                        *(1 + filter_size*filter_size))
    field_size /= 2
  if num_classes == 2:
    running_count += (field_size*field_size*num_channels+1)
  else:
    running_count += (field_size*field_size*num_channels*num_classes+1)
  return running_count


def normalize_rows(x):
  return x / (EPSILON + np.linalg.norm(x, axis=1).reshape(x.shape[0], 1))


# given two matrices of the same shape, compute
# the average of the cosine similarities between
# their rows
def average_cosine(xone, xtwo):
  normalized_xone = normalize_rows(xone)
  normalized_xtwo = normalize_rows(xtwo)
  return np.average(np.sum(normalized_xone * normalized_xtwo, axis=1))


# this measures the translation invariance of the kernel associated with a
# model
def measure_translation_invariance(model, x):
  """Returns measurement of the translation invariance of a model on x."""
  original_features = extract_features(model, x, args.use_conjugate_kernel)
  translated_right_x = tfa.image.translate(x, [0, 1])
  translated_right_features = extract_features(model,
                                               translated_right_x,
                                               args.use_conjugate_kernel)
  translated_down_x = tfa.image.translate(x, [1, 0])
  translated_down_features = extract_features(model,
                                              translated_down_x,
                                              args.use_conjugate_kernel)
  return 0.5*(average_cosine(original_features, translated_right_features)
              + average_cosine(original_features, translated_down_features))


# this measures the zoom invariance of the kernel associated with a
# model
def measure_zoom_invariance(model, x, image_size):
  original_features = extract_features(model, x, args.use_conjugate_kernel)
  cropped_x = tf.image.central_crop(x, CROP_FRACTION)
  zoomed_x = tf.image.resize(cropped_x, [image_size, image_size])
  zoomed_features = extract_features(model, zoomed_x, args.use_conjugate_kernel)
  return average_cosine(original_features, zoomed_features)


# this measures the rotation invariance of the kernel associated with a
# model
def measure_rotation_invariance(model, x):
  """Returns measurement of the rotation invariance of a model on some data."""
  original_features = extract_features(model, x, args.use_conjugate_kernel)
  rotated_pos_x = tfa.image.rotate(x, args.rotation_angle)
  rotated_pos_features = extract_features(model,
                                          rotated_pos_x,
                                          args.use_conjugate_kernel)
  rotated_neg_x = tfa.image.rotate(x, -args.rotation_angle)
  rotated_neg_features = extract_features(model,
                                          rotated_neg_x,
                                          args.use_conjugate_kernel)
  return 0.5*(average_cosine(original_features, rotated_pos_features)
              + average_cosine(original_features, rotated_neg_features))


# this measures the "locality" of features -- invariance to swaps between the
# upper right and lower-left quadrants of the image
def measure_swap_invariance(model, x, image_size):
  """Returns measurement of "swap invariance".

  Args:
    model: a member of the ModelWithFeatures class
    x: the data to use to evaluate invariance
    image_size: the height and width of the input images
  """
  original_features = extract_features(model, x, args.use_conjugate_kernel)
  # tf.tensor_scatter_nd_update is too confusing.  Going back and forth from
  # numpy instead
  x_np = x.numpy()
  slice_size = int(image_size/2)
  upper_left = x_np[:, :slice_size, :slice_size, :].copy()
  lower_right = x_np[:, slice_size:, slice_size:, :].copy()
  x_np[:, :slice_size, :slice_size, :] = lower_right
  x_np[:, slice_size:, slice_size:, :] = upper_left
  perturbed_x = tf.constant(x_np, dtype=tf.float32)
  perturbed_features = extract_features(model,
                                        perturbed_x,
                                        args.use_conjugate_kernel)
  return average_cosine(original_features, perturbed_features)

# suppress retracing warnings
tf.get_logger().setLevel('ERROR')

np.set_printoptions(precision=3, suppress=True)

if args.dataset == 'mnist':
  dataset_image_size = 28
  ds_num_input_channels = 1
  ds_train = tfds.load('mnist',
                       split='train',
                       as_supervised=True,
                       batch_size=-1
                       )
  ds_test = tfds.load('mnist',
                      split='test',
                      as_supervised=True,
                      batch_size=-1
                      )
elif args.dataset == 'cifar10':
  dataset_image_size = 32
  ds_num_input_channels = 3
  ds_train = tfds.load('cifar10',
                       split='train',
                       as_supervised=True,
                       batch_size=-1
                       )
  ds_test = tfds.load('cifar10',
                      split='test',
                      as_supervised=True,
                      batch_size=-1
                      )
x_train, y_train = tfds.as_numpy(ds_train)
x_test, y_test = tfds.as_numpy(ds_test)
x_train, y_train = transform_dataset(x_train, y_train)
x_test, y_test = transform_dataset(x_test, y_test)
dataset_num_classes = 2

running_nn_error = 0.0
running_linear_error = 0.0
if args.num_translations > 0:
  running_ntk_trans_inv = 0.0
  running_ak_trans_inv = 0.0
if args.num_zooms > 0:
  running_ntk_zoom_inv = 0.0
  running_ak_zoom_inv = 0.0
if args.num_swaps > 0:
  running_ntk_swap_inv = 0.0
  running_ak_swap_inv = 0.0
if args.num_rotations > 0:
  running_ntk_rotat_inv = 0.0
  running_ak_rotat_inv = 0.0
if args.ker_alignment_sample_size > 0:
  running_ntk_ker_alignment = 0.0
  running_ak_ker_alignment = 0.0
if args.effective_rank_sample_size > 0:
  running_ntk_effective_rank = 0.0
  running_ak_effective_rank = 0.0
for r in range(args.num_runs):
  if args.nn_architecture == 'VGG':
    width = 4
    parameter_count = calculate_vgg_parameter_count(dataset_image_size,
                                                    ds_num_input_channels,
                                                    width,
                                                    dataset_num_classes)
    while parameter_count < args.num_parameters:
      width += 1
      parameter_count = calculate_vgg_parameter_count(dataset_image_size,
                                                      ds_num_input_channels,
                                                      width,
                                                      dataset_num_classes)
    nn_model = ModelWithFeatures('VGG', width)
  elif args.nn_architecture == 'MLP':
    input_size = dataset_image_size*dataset_image_size*ds_num_input_channels
    width = 4
    parameter_count = 0
    while parameter_count < args.num_parameters:
      width += 1
      parameter_count = (input_size*width
                         +(args.layers-1)*(width+1)*width
                         +width+1)
    nn_model = ModelWithFeatures('MLP', width)
  else:
    width = int(args.num_parameters)
    nn_model = ModelWithFeatures('Sum', width)
  if args.num_translations > 0:
    nti = measure_translation_invariance(nn_model,
                                         x_test[:args.num_translations])
    running_ntk_trans_inv += ((1.0/(r+1)) * (nti
                                             - running_ntk_trans_inv))
  if args.num_zooms > 0:
    this_ntk_zoom_inv = measure_zoom_invariance(nn_model,
                                                x_test[:args.num_zooms],
                                                dataset_image_size)
    running_ntk_zoom_inv += ((1.0/(r+1)) * (this_ntk_zoom_inv
                                            - running_ntk_zoom_inv))
  if args.num_swaps > 0:
    this_ntk_swap_inv = measure_swap_invariance(nn_model,
                                                x_test[:args.num_swaps],
                                                dataset_image_size)
    running_ntk_swap_inv += ((1.0/(r+1)) * (this_ntk_swap_inv
                                            - running_ntk_swap_inv))
  if args.num_rotations > 0:
    nri = measure_rotation_invariance(nn_model, x_test[:args.num_rotations])
    running_ntk_rotat_inv += ((1.0/(r+1)) * (nri
                                             - running_ntk_rotat_inv))
  if args.ker_alignment_sample_size > 0:
    nka = ker_alignment(nn_model, x_test[:args.ker_alignment_sample_size])
    running_ntk_ker_alignment += ((1.0/(r+1)) * (nka
                                                 - running_ntk_ker_alignment))
  if args.effective_rank_sample_size > 0:
    ner = effective_rank(nn_model,
                         x_test[:args.effective_rank_sample_size])
    running_ntk_effective_rank += ((1.0/(r+1)) * (ner
                                                  - running_ntk_effective_rank))
  nn_model.compile(optimizer=tf.keras.optimizers.SGD(),
                   loss=LOSS,
                   metrics=['accuracy'])
  optimizer = tf.keras.optimizers.SGD()
  nn_model.compile(optimizer=optimizer,
                   loss=LOSS,
                   metrics=['accuracy'])

  if args.num_epochs > 0:
    if args.use_augmentation:
      print('Using data augmentation')
      idg = tf.keras.preprocessing.image.ImageDataGenerator
      # idg defined on the preceding line (to keep line length <= 80)
      datagen = idg(rotation_range=ROTATION_RANGE,
                    width_shift_range=WIDTH_SHIFT_RANGE,
                    height_shift_range=HEIGHT_SHIFT_RANGE)
      datagen.fit(x_train)
      nn_model.fit(datagen.flow(x_train, y_train, batch_size=32),
                   epochs=args.num_epochs)
    else:
      nn_model.fit(x_train, y_train, epochs=args.num_epochs)

  if args.num_translations > 0:
    ati = measure_translation_invariance(nn_model,
                                         x_test[:args.num_translations])
    running_ak_trans_inv += ((1.0/(r+1)) * (ati
                                            - running_ak_trans_inv))
  if args.num_zooms > 0:
    this_ak_zoom_inv = measure_zoom_invariance(nn_model,
                                               x_test[:args.num_zooms],
                                               dataset_image_size)
    running_ak_zoom_inv += ((1.0/(r+1)) * (this_ak_zoom_inv
                                           - running_ak_zoom_inv))
  if args.num_swaps > 0:
    this_ak_swap_inv = measure_swap_invariance(nn_model,
                                               x_test[:args.num_swaps],
                                               dataset_image_size)
    running_ak_swap_inv += ((1.0/(r+1)) * (this_ak_swap_inv
                                           - running_ak_swap_inv))
  if args.num_rotations > 0:
    this_ak_rotat_inv = measure_rotation_invariance(nn_model,
                                                    x_test[:args.num_rotations])
    running_ak_rotat_inv += ((1.0/(r+1)) * (this_ak_rotat_inv
                                            - running_ak_rotat_inv))
  if args.ker_alignment_sample_size > 0:
    aka = ker_alignment(nn_model,
                        x_test[:args.ker_alignment_sample_size])
    running_ak_ker_alignment += ((1.0/(r+1)) * (aka
                                                - running_ak_ker_alignment))
  if args.effective_rank_sample_size > 0:
    aer = effective_rank(nn_model, x_test[:args.effective_rank_sample_size])
    running_ak_effective_rank += ((1.0/(r+1)) * (aer
                                                 - running_ak_effective_rank))

  if args.train_linear_model:
    print('Extracting features on the training data')
    phi_train = extract_features(nn_model, x_train, args.use_conjugate_kernel)
    print('Extracting features on the test data')
    phi_test = extract_features(nn_model, x_test, args.use_conjugate_kernel)
    print('Done extracting features')
    Z = EPSILON + np.average(np.linalg.norm(phi_train, axis=1))
    phi_train /= Z
    phi_test /= Z

    linear_model = sklearn.svm.LinearSVC(max_iter=10000, C=100)
    linear_model.fit(phi_train, y_train)
    lin_acc = sklearn.metrics.accuracy_score(linear_model.predict(phi_test),
                                             y_test)
    this_linear_error = 1.0 - lin_acc
    running_linear_error += ((1.0/(r+1))*(this_linear_error
                                          - running_linear_error))

  _, nn_accuracy = nn_model.evaluate(x_test, y_test)
  this_nn_error = 1.0 - nn_accuracy

  running_nn_error += ((1.0/(r+1))*(this_nn_error - running_nn_error))

  print('running_nn_error = {}'.format(running_nn_error))
  if args.train_linear_model:
    print('running_linear_error = {}'.format(running_linear_error))
  if args.num_translations > 0:
    print('running_ntk_trans_inv = {}'.format(running_ntk_trans_inv))
    print('running_ak_trans_inv = {}'.format(running_ak_trans_inv))
  if args.num_zooms > 0:
    print('running_ntk_zoom_inv = {}'.format(running_ntk_zoom_inv))
    print('running_ak_zoom_inv = {}'.format(running_ak_zoom_inv))
  if args.num_swaps > 0:
    print('running_ntk_swap_inv = {}'.format(running_ntk_swap_inv))
    print('running_ak_swap_inv = {}'.format(running_ak_swap_inv))
  if args.num_rotations > 0:
    print('running_ntk_rotat_inv = {}'.format(running_ntk_rotat_inv))
    print('running_ak_rotat_inv = {}'.format(running_ak_rotat_inv))
  if args.ker_alignment_sample_size > 0:
    print('running_ntk_ker_alignment = {}'.format(running_ntk_ker_alignment))
    print('running_ak_ker_alignment = {}'.format(running_ak_ker_alignment))
  if args.effective_rank_sample_size > 0:
    print('running_ntk_effective_rank = {}'.format(running_ntk_effective_rank))
    print('running_ak_effective_rank = {}'.format(running_ak_effective_rank))

wrote = False
write_tries = 0
while (not wrote) and (write_tries < WRITE_TRY_LIMIT):
  # provisionally assume that we wrote
  wrote = True
  try:
    fout = open(args.test_accuracy_output, mode='a')
    format_string = ('{}\t' * 16) + '\n'
    fout.write(format_string.format(args.num_epochs,
                                    args.use_augmentation,
                                    args.use_conjugate_kernel,
                                    running_nn_error,
                                    (running_linear_error
                                     if args.train_linear_model
                                     else ''),
                                    (running_ntk_trans_inv
                                     if args.num_translations > 0
                                     else ''),
                                    (running_ak_trans_inv
                                     if args.num_translations > 0
                                     else ''),
                                    (running_ntk_zoom_inv
                                     if args.num_zooms > 0
                                     else ''),
                                    (running_ak_zoom_inv
                                     if args.num_zooms > 0
                                     else ''),
                                    (running_ntk_rotat_inv
                                     if args.num_rotations > 0
                                     else ''),
                                    (running_ak_rotat_inv
                                     if args.num_rotations > 0
                                     else ''),
                                    (running_ntk_ker_alignment
                                     if args.ker_alignment_sample_size > 0
                                     else ''),
                                    (running_ak_ker_alignment
                                     if args.ker_alignment_sample_size > 0
                                     else ''),
                                    (running_ntk_swap_inv
                                     if args.num_swaps > 0
                                     else ''),
                                    (running_ak_swap_inv
                                     if args.num_swaps > 0
                                     else ''),
                                    (running_ntk_effective_rank
                                     if args.effective_rank_sample_size > 0
                                     else ''),
                                    (running_ak_effective_rank
                                     if args.effective_rank_sample_size > 0
                                     else '')))
    fout.close()
  except OSError as e:
    print('Write failed with exception {}, retrying'.format(e))
    sys.stdout.flush()
    wrote = False
    write_tries += 1
    time.sleep(random.randrange(SLEEP_TIME))

if not os.path.exists(args.model_summary_output):
  fmout = open(args.model_summary_output, mode='a')
  nn_model.summary(print_fn=lambda x: fmout.write(x + '\n'))
  fmout.close()
