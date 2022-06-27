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

"""other code used for approxNN project (adapted from CNN Zoo github)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

CNN_KERNEL_SIZE = 3


def _preprocess_batch(batch,
                      normalize,
                      to_grayscale,
                      augment=False):
  """Preprocessing function for each batch of data.

  Args:
    batch: the batch of samples to be procesed, of dims (N, H, W, D).
    normalize: boolean indicating whether or not to normalize the batch samples.
    to_grayscale: boolean indicating whether or not to convert batch samples
                  to grayscale values rather than RGB.
    augment: boolean indicating whether or not to augment the batch samples.

  Returns:
    Processed batch images according to the flags above,
    and the corresponding labels (unchanged).
  """
  min_out = -1.0
  max_out = 1.0
  image = tf.cast(batch['image'], tf.float32)
  image /= 255.0

  if augment:
    shape = image.shape
    image = tf.image.resize_with_crop_or_pad(image, shape[1] + 2, shape[2] + 2)
    image = tf.image.random_crop(image, size=shape)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)

  if normalize:
    image = min_out + image * (max_out - min_out)
  if to_grayscale:
    image = tf.math.reduce_mean(image, axis=-1, keepdims=True)
  return image, batch['label']


def get_dataset(dataset_name,
                batch_size,
                to_grayscale=True,
                train_fraction=1.0,
                shuffle_buffer=1024,
                seed=None,
                normalize=True,
                augmentation=False):
  """Load and preprocess the dataset.

  Args:
    dataset_name: The dataset name. Either 'toy' or a TFDS dataset
    batch_size: the desired batch size.
    to_grayscale: if True, all images will be converted into grayscale.
    train_fraction: what fraction of the overall training set should we use.
    shuffle_buffer: size of the shuffle.buffer for tf.data.Dataset.shuffle.
    seed: random seed for shuffling operations.
    normalize: whether to normalize the data into [-1, 1].
    augmentation: use data augmentation on the training set.

  Returns:
    tuple (training_dataset, test_dataset, info), where info is a dictionary
    with some relevant information about the dataset.
  """
  data_train, ds_info = tfds.load(dataset_name, split='train', with_info=True)
  effective_train_size = ds_info.splits['train'].num_examples

  if train_fraction < 1.0:
    effective_train_size = int(effective_train_size * train_fraction)
    data_train = data_train.shuffle(shuffle_buffer, seed=seed)
    data_train = data_train.take(effective_train_size)

  fn_tr = lambda b: _preprocess_batch(b, normalize, to_grayscale, augmentation)
  data_train = data_train.shuffle(shuffle_buffer, seed=seed)
  data_train = data_train.batch(batch_size, drop_remainder=True)
  data_train = data_train.map(fn_tr, tf.data.experimental.AUTOTUNE)
  data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

  fn_te = lambda b: _preprocess_batch(b, normalize, to_grayscale, False)
  data_test = tfds.load(dataset_name, split='test')
  data_test = data_test.batch(batch_size)
  data_test = data_test.map(fn_te, tf.data.experimental.AUTOTUNE)
  data_test = data_test.prefetch(tf.data.experimental.AUTOTUNE)

  dataset_information = {
      'num_classes': ds_info.features['label'].num_classes,
      'data_shape': ds_info.features['image'].shape,
      'train_num_examples': effective_train_size
  }
  return data_train, data_test, dataset_information


def build_cnn(num_layers, num_hidden, num_outputs, dropout_rate, activation,
              stride, w_regularizer, w_init, b_init, use_batchnorm):
  """Convolutional deep neural network."""
  model = tf.keras.Sequential()
  for _ in range(num_layers):
    model.add(
        tf.keras.layers.Conv2D(
            num_hidden,
            kernel_size=CNN_KERNEL_SIZE,
            strides=stride,
            activation=activation,
            kernel_regularizer=w_regularizer,
            kernel_initializer=w_init,
            bias_initializer=b_init))
    if dropout_rate > 0.0:
      model.add(tf.keras.layers.Dropout(dropout_rate))
    if use_batchnorm:
      model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.GlobalAveragePooling2D())
  model.add(
      tf.keras.layers.Dense(
          num_outputs,
          kernel_regularizer=w_regularizer,
          kernel_initializer=w_init,
          bias_initializer=b_init,
          activation='softmax'))
  return model


def get_model_wireframe(dataset_name='mnist'):
  """Construct the base model architecture as used in the CNN Zoo paper.

  Paper: https://arxiv.org/pdf/2002.11448.pdf

  Args:
    dataset_name: a string representing the dataset to be loaded using tfds

  Returns:
    A tf.keras.Sequential() object 3 convolutional layers with 16 filters each,
    followed by global average pooling and a fully connected layer, for a total
    of 4970 learnable weights.
  """

  ds_info = get_dataset_info(dataset_name)

  n_layers = 3
  n_hiddens = 16
  n_outputs = ds_info['num_classes']
  # epochs = 18
  # epochs_between_checkpoints = 6
  cnn_stride = 2
  dropout_rate = 0.0
  l2_penalty = 0.0
  init_std = 0.05
  learning_rate = 0.01
  optimizer_name = 'sgd'
  activation_name = 'relu'
  w_init_name = 'he_normal'
  b_init_name = 'zero'
  # reduce_learningrate = False
  dnn_architecture = 'cnn'
  # verbosity = 0
  # use_tpu = False
  # master = 'local'

  optimizer = tf.keras.optimizers.get(optimizer_name)
  optimizer.learning_rate = learning_rate
  w_init = tf.keras.initializers.get(w_init_name)
  if w_init_name.lower() in ['truncatednormal', 'randomnormal']:
    w_init.stddev = init_std
  b_init = tf.keras.initializers.get(b_init_name)
  if b_init_name.lower() in ['truncatednormal', 'randomnormal']:
    b_init.stddev = init_std
  w_reg = tf.keras.regularizers.l2(l2_penalty) if l2_penalty > 0 else None

  model_wireframe = build_cnn(n_layers, n_hiddens, n_outputs, dropout_rate,
                              activation_name, cnn_stride, w_reg, w_init,
                              b_init, dnn_architecture == 'cnnbn')

  model_wireframe.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy', 'mse', 'sparse_categorical_crossentropy'])

  input_shape = (None,) + ds_info['data_shape']
  model_wireframe.build(input_shape)
  model_wireframe.summary()
  return model_wireframe


def get_dataset_info(dataset_name='mnist'):
  """Method to return dataset information for a specific dataset_name.

  Args:
    dataset_name: a string representing the dataset to be loaded using tfds

  Returns:
    A dictionary of relevant information for the loaded dataset.
  """

  ds_info = tfds.builder(dataset_name).info
  dataset_information = {
      'num_classes': ds_info.features['label'].num_classes,
      'data_shape': ds_info.features['image'].shape,
      'train_num_examples': ds_info.splits['train'].num_examples,

  }
  return dataset_information
