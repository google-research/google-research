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

"""Code to load, preprocess and train on Patch Camelyon dataset."""
from absl import app
from absl import flags
from absl import logging

import functools
import os
import pickle
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from do_wide_and_deep_networks_learn_the_same_things.resnet_cifar import ResNet_CIFAR
from do_wide_and_deep_networks_learn_the_same_things.shake_shake import build_shake_shake_model
from tensorflow.keras import backend as K

tf.enable_v2_behavior()

FLAGS = flags.FLAGS
#Define training hyperparameters
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train for')
flags.DEFINE_float('weight_decay', 0.0001, 'L2 regularization')
#Define model & data hyperparameters
flags.DEFINE_integer('depth', 56, 'No. of layers to use in the ResNet model')
flags.DEFINE_integer(
    'width_multiplier', 1,
    'How much to scale the width of the standard ResNet model by')
flags.DEFINE_integer(
    'copy', 0,
    'If the same model configuration has been run before, train another copy with a different random initialization'
)
flags.DEFINE_string('base_dir', None,
                    'Where the trained model will be saved')
flags.DEFINE_string('dataset_name', 'cifar10',
                    'Name of dataset used (CIFAR-10 of CIFAR-100)')
flags.DEFINE_boolean('use_residual', True,
                     'Whether to include residual connections in the model')
flags.DEFINE_boolean('randomize_labels', False,
                     'Whether to randomize labels during training')
flags.DEFINE_string('pretrain_dir', '',
                    'Directory where the pretrained model is saved')
flags.DEFINE_boolean(
    'partial_init', False,
    'Whether to initialize only the first few layers with pretrained weights')
flags.DEFINE_boolean('shake_shake', False, 'Whether to use shake shake model')
flags.DEFINE_integer('epoch_save_freq', 0, 'Frequency at which ckpts are saved')


def find_stack_markers(model):
  """Finds the layers where a new stack starts."""
  stack_markers = []
  old_shape = None
  for i, layer in enumerate(model.layers):
    if i == 0:
      continue
    if 'conv' in layer.name:
      conv_weights_shape = layer.get_weights()[0].shape
      if conv_weights_shape[-1] != conv_weights_shape[-2] and conv_weights_shape[
          0] != 1 and conv_weights_shape[-2] % 16 == 0:
        stack_markers.append(i)
  assert (len(stack_markers) == 2)
  return stack_markers


def preprocess_data(images, labels, is_training):
  """Patch Camelyon data preprocessing"""
  images = tf.image.convert_image_dtype(images, tf.float32)

  if is_training:
    images = tf.image.random_crop(images, [32, 32, 3])
    images = tf.image.random_flip_up_down(images)
  else:
    images = tf.image.resize_with_crop_or_pad(images, 32, 32)  # central crop
  return images, labels


def load_train_data(batch_size, dataset_name, n_data):
  """Load Patch Camelyon training data"""
  train_dataset = tfds.load(
      name=dataset_name, split='train', as_supervised=True)
  train_dataset = train_dataset.shuffle(buffer_size=n_data)
  train_dataset = train_dataset.map(
      functools.partial(preprocess_data, is_training=True))
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return train_dataset


def load_val_data(batch_size, dataset_name):
  """Load Patch Camelyon val data"""
  val_dataset = tfds.load(
      name=dataset_name, split='validation', as_supervised=True)
  val_dataset = val_dataset.map(
      functools.partial(preprocess_data, is_training=False))
  val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
  val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return val_dataset


def load_test_data(batch_size, dataset_name, n_data=10000, shuffle=False):
  """Load Patch Camelyon test data"""
  test_dataset = tfds.load(name=dataset_name, split='test', as_supervised=True)
  test_dataset = test_dataset.map(
      functools.partial(preprocess_data, is_training=False))
  if shuffle:
    test_dataset = test_dataset.shuffle(buffer_size=n_data)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return test_dataset


def main(argv):
  n_data = 262144
  train_dataset = load_train_data(
      FLAGS.batch_size, dataset_name=FLAGS.dataset_name, n_data=n_data)
  val_dataset = load_val_data(FLAGS.batch_size, dataset_name=FLAGS.dataset_name)
  test_dataset = load_test_data(
      FLAGS.batch_size, dataset_name=FLAGS.dataset_name, n_data=10000)
  steps_per_epoch = n_data // FLAGS.batch_size  #tf.data.experimental.cardinality(train_dataset).numpy()
  optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate, momentum=0.9)
  schedule = tf.keras.experimental.CosineDecay(FLAGS.learning_rate,
                                               FLAGS.epochs)
  lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

  num_classes = 2
  if FLAGS.shake_shake:
    model = build_shake_shake_model(
        num_classes,
        FLAGS.depth,
        FLAGS.width_multiplier,
        FLAGS.weight_decay,
        image_shape=(32, 32, 3))
  else:
    model = ResNet_CIFAR(
        FLAGS.depth,
        FLAGS.width_multiplier,
        FLAGS.weight_decay,
        num_classes=num_classes,
        input_shape=(32, 32, 3),
        use_residual=FLAGS.use_residual)

  if FLAGS.pretrain_dir:
    pretrained_model = tf.keras.models.load_model(FLAGS.pretrain_dir)
    n_layers = len(model.layers)
    if FLAGS.partial_init:
      stack_marker = find_stack_markers(pretrained_model)[0]
      for i in range(
          stack_marker
      ):  # use pretrained weights for only layers from the first stage
        model.layers[i].set_weights(pretrained_model.layers[i].get_weights())
    else:
      for i in range(
          n_layers -
          1):  # use pretrained weights for all layers except the last
        model.layers[i].set_weights(pretrained_model.layers[i].get_weights())

  model.compile(
      optimizer,
      tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['acc'])

  model_dir = 'depth-%d-width-%d-bs-%d-lr-%f-reg-%f/' % \
      (FLAGS.depth, FLAGS.width_multiplier, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.weight_decay)
  experiment_dir = os.path.join(FLAGS.base_dir, model_dir)
  if FLAGS.copy > 0:
    experiment_dir = '%s/depth-%d-width-%d-bs-%d-lr-%f-reg-%f-copy-%d/' % \
      (FLAGS.base_dir, FLAGS.depth, FLAGS.width_multiplier, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.weight_decay, FLAGS.copy)

  if FLAGS.epoch_save_freq > 0:
    #Save initialization
    tf.keras.models.save_model(
        model, experiment_dir, overwrite=True, include_optimizer=False)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=experiment_dir + 'weights.{epoch:02d}.ckpt',
        monitor='val_acc',
        verbose=1,
        save_best_only=False,
        save_freq='epoch',
        period=FLAGS.epoch_save_freq)
  else:
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=experiment_dir,
        monitor='val_acc',
        verbose=1,
        save_best_only=True)
  hist = model.fit(
      train_dataset,
      batch_size=FLAGS.batch_size,
      epochs=FLAGS.epochs,
      validation_data=val_dataset,
      verbose=1,
      steps_per_epoch=steps_per_epoch,
      callbacks=[checkpoint, lr_scheduler])

  best_model = tf.keras.models.load_model(experiment_dir)
  best_model.compile(
      'sgd',
      tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['acc'])
  test_metrics = best_model.evaluate(test_dataset, verbose=1)


  logging.info('Test accuracy: %.4f', test_metrics[1])


if __name__ == '__main__':
  app.run(main)
