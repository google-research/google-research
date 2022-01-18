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

"""Code to load, preprocess and train on CIFAR-10, with first PCs regularized
"""
from absl import app
from absl import flags
from absl import logging

import functools
import os
import pickle
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from do_wide_and_deep_networks_learn_the_same_things.resnet_cifar_pc_regularize import ResNet_CIFAR
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
flags.DEFINE_integer('width_multiplier', 1, 'How much to scale the width of the standard ResNet model by')
flags.DEFINE_integer('copy', 0, 'If the same model configuration has been run before, train another copy with a different random initialization')
flags.DEFINE_string('base_dir', None, 'Where the trained model will be saved')
flags.DEFINE_string('data_path', '', 'Directory where CIFAR subsampled dataset is stored')
flags.DEFINE_string('dataset_name', 'cifar10', 'Name of dataset used (CIFAR-10 of CIFAR-100)')
flags.DEFINE_boolean('use_residual', True, 'Whether to include residual connections in the model')
flags.DEFINE_boolean('randomize_labels', False, 'Whether to randomize labels during training')
flags.DEFINE_string('pretrain_dir', '', 'Directory where the pretrained model is saved')
flags.DEFINE_boolean('partial_init', False, 'Whether to initialize only the first few layers with pretrained weights')
flags.DEFINE_boolean('shake_shake', False, 'Whether to use shake shake model')
flags.DEFINE_float('pc_reg_strength', 10, 'First principal component regularization strength')
flags.DEFINE_boolean('residual_reg_only', False, 'Whether to only regularize the principal components after residual connections')
flags.DEFINE_boolean('relu_reg_only', False, 'Whether to only regularize the principal components after activation layers')
flags.DEFINE_boolean('last_2_stages', False, 'Whether to only regularize the activations from the last 2 stages of the network')
flags.DEFINE_float('threshold', 0.1, 'Maximum variance ratio explained before penalty is applied')


def find_stack_markers(model):
  """Finds the layers where a new stack starts."""
  stack_markers = []
  old_shape = None
  for i, layer in enumerate(model.layers):
    if i == 0:
      continue
    if 'conv' in layer.name:
      conv_weights_shape = layer.get_weights()[0].shape
      if conv_weights_shape[-1] != conv_weights_shape[-2] and conv_weights_shape[0] != 1 and conv_weights_shape[-2] % 16 == 0:
        stack_markers.append(i)
  assert(len(stack_markers) == 2)
  return stack_markers


def preprocess_data(images, labels, is_training):
  """CIFAR data preprocessing"""
  images = tf.image.convert_image_dtype(images, tf.float32)

  if is_training:
    crop_padding = 4
    images = tf.pad(images, [[crop_padding, crop_padding],
                             [crop_padding, crop_padding], [0, 0]], 'REFLECT')
    images = tf.image.random_crop(images, [32, 32, 3])
    images = tf.image.random_flip_left_right(images)
  return images, labels


def load_train_data(batch_size, data_path='', dataset_name='cifar10', n_data=50000, randomize_labels=False):
  """Load CIFAR training data"""
  if not data_path:
    train_dataset = tfds.load(name=dataset_name, split='train', as_supervised=True)
  else:
    if 'tiny' in data_path:
      train_dataset = tfds.load(name=dataset_name, split='train[:6%]', as_supervised=True)
    elif 'half' in data_path:
      train_dataset = tfds.load(name=dataset_name, split='train[:50%]', as_supervised=True)
    else:
      train_dataset = tfds.load(name=dataset_name, split='train[:25%]', as_supervised=True)

  if randomize_labels:
    all_labels = []
    all_images = []
    for images, labels in train_dataset:
      all_labels.extend([labels.numpy()])
      all_images.append(images.numpy()[np.newaxis, :, :, :])
    all_images = np.vstack(all_images)
    np.random.seed(FLAGS.copy)
    np.random.shuffle(all_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(all_images, dtype=tf.float32), tf.convert_to_tensor(all_labels, dtype=tf.int64)))

  train_dataset = train_dataset.shuffle(buffer_size=n_data)
  train_dataset = train_dataset.map(functools.partial(preprocess_data, is_training=True))
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return train_dataset


def load_test_data(batch_size, shuffle=False, data_path='', dataset_name='cifar10', n_data=10000):
  """Load CIFAR test data"""
  if not data_path:
    test_dataset = tfds.load(name=dataset_name, split='test', as_supervised=True)
  else:
    test_data = pickle.load(tf.io.gfile.GFile(os.path.join(data_path, 'test_data.pkl'), 'rb')).astype(np.uint8)
    test_labels = pickle.load(tf.io.gfile.GFile(os.path.join(data_path, 'test_labels.pkl'), 'rb'))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

  test_dataset = test_dataset.map(functools.partial(preprocess_data, is_training=False))
  if shuffle:
    test_dataset = test_dataset.shuffle(buffer_size=n_data)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  return test_dataset


def main(argv):
  if FLAGS.data_path:
    if 'tiny' in FLAGS.data_path:
      n_data = int(50000 * 6/100)
    elif 'half' in FLAGS.data_path:
      n_data = 50000//2
    elif 'subsampled' in FLAGS.data_path:
      n_data = 50000//4
  else:
    n_data = 50000
  train_dataset = load_train_data(FLAGS.batch_size, dataset_name=FLAGS.dataset_name, n_data=n_data, data_path=FLAGS.data_path, randomize_labels=FLAGS.randomize_labels)
  test_dataset = load_test_data(FLAGS.batch_size, dataset_name=FLAGS.dataset_name, n_data=10000)
  steps_per_epoch = n_data // FLAGS.batch_size
  optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate, momentum=0.9)
  schedule = tf.keras.experimental.CosineDecay(FLAGS.learning_rate,
                                               FLAGS.epochs)
  lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

  if FLAGS.dataset_name == 'cifar100':
    num_classes = 100
  else:
    num_classes = 100

  if FLAGS.shake_shake:
    model = build_shake_shake_model(num_classes, FLAGS.depth, FLAGS.width_multiplier, FLAGS.weight_decay)
  else:
    model = ResNet_CIFAR(FLAGS.depth, FLAGS.width_multiplier, FLAGS.weight_decay,
                         num_classes=num_classes, use_residual=FLAGS.use_residual,
                         reg_strength=FLAGS.pc_reg_strength,
                         residual_reg_only=FLAGS.residual_reg_only, relu_reg_only=FLAGS.relu_reg_only,
                         threshold=FLAGS.threshold, last_2_stages=FLAGS.last_2_stages)

  if FLAGS.pretrain_dir:
    pretrained_model = tf.keras.models.load_model(FLAGS.pretrain_dir)
    n_layers = len(model.layers)
    if FLAGS.partial_init:
      stack_marker = find_stack_markers(pretrained_model)[0]
      for i in range(stack_marker): # use pretrained weights for only layers from the first stage
        model.layers[i].set_weights(pretrained_model.layers[i].get_weights())
    else:
      for i in range(n_layers-1): # use pretrained weights for all layers except the last
        model.layers[i].set_weights(pretrained_model.layers[i].get_weights())

  model.compile(optimizer,
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['acc'])

  model_dir = 'cifar-depth-%d-width-%d-bs-%d-lr-%f-reg-%f-pc-reg-%f/' % \
      (FLAGS.depth, FLAGS.width_multiplier, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.weight_decay, FLAGS.pc_reg_strength)
  experiment_dir = os.path.join(FLAGS.base_dir, model_dir)
  if FLAGS.copy > 0:
    experiment_dir = '%s/cifar-depth-%d-width-%d-bs-%d-lr-%f-reg-%f-pc-reg-%f-copy-%d/' % \
      (FLAGS.base_dir, FLAGS.depth, FLAGS.width_multiplier, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.weight_decay, FLAGS.pc_reg_strength, FLAGS.copy)

  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=experiment_dir,
                                        monitor='val_acc',
                                        verbose=1,
                                        save_best_only=True)
  hist = model.fit(train_dataset,
            batch_size=FLAGS.batch_size,
            epochs=FLAGS.epochs,
            validation_data=test_dataset,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            callbacks=[checkpoint, lr_scheduler])

  best_model = tf.keras.models.load_model(experiment_dir)
  best_model.compile('sgd',
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])
  test_metrics = best_model.evaluate(test_dataset, verbose=1)


  logging.info('Test accuracy: %.4f', test_metrics[1])


if __name__ == '__main__':
  app.run(main)
