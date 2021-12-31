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

"""Code to load, preprocess and train on CIFAR-10."""
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
flags.DEFINE_string('data_path', '',
                    'Directory where CIFAR subsampled dataset is stored')
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
flags.DEFINE_boolean('distort_color', False,
                     'Whether to apply color distortion augmentation')
flags.DEFINE_integer('epoch_save_freq', 0, 'Frequency at which ckpts are saved')
flags.DEFINE_boolean(
    'save_image', False,
    'Whether to save metadata of images used for each minibatch')


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


def random_apply(transform_fn, image, p):
  """Randomly apply with probability p a transformation to an image"""
  if tf.random.uniform([]) < p:
    return transform_fn(image)
  else:
    return image


def color_distortion(image, s=1.0):
  """Color distortion data augmentation"""
  # image is a tensor with value range in [0, 1].
  # s is the strength of color distortion.
  def color_jitter(x):
    # one can also shuffle the order of following augmentations
    # each time they are applied.
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_hue(x, max_delta=0.2 * s)
    x = tf.clip_by_value(x, 0, 1)
    return x

  def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x

  # randomly apply transformation with probability p.
  image = random_apply(color_jitter, image, p=0.8)
  image = random_apply(color_drop, image, p=0.2)
  return image


def preprocess_data(image, label, is_training):
  """CIFAR data preprocessing"""
  image = tf.image.convert_image_dtype(image, tf.float32)

  if is_training:
    crop_padding = 4
    image = tf.pad(image, [[crop_padding, crop_padding],
                           [crop_padding, crop_padding], [0, 0]], 'REFLECT')
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    if FLAGS.distort_color:
      image = color_distortion(image, s=1.0)
  else:
    image = tf.image.resize_with_crop_or_pad(image, 32, 32)  # central crop
  return image, label


def preprocess_data_with_id(data, is_training):
  """CIFAR data preprocessing when image ids are included in the data loader"""
  image = data['image']
  image = tf.image.convert_image_dtype(image, tf.float32)

  if is_training:
    crop_padding = 4
    image = tf.pad(image, [[crop_padding, crop_padding],
                           [crop_padding, crop_padding], [0, 0]], 'REFLECT')
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
  else:
    image = tf.image.resize_with_crop_or_pad(image, 32, 32)  # central crop
  return data['id'], image, data['label']


def load_train_data(batch_size,
                    data_path='',
                    dataset_name='cifar10',
                    n_data=50000,
                    randomize_labels=False,
                    as_supervised=True):
  """Load CIFAR training data"""
  if not data_path:
    train_dataset = tfds.load(
        name=dataset_name, split='train', as_supervised=as_supervised)
  else:
    if 'tiny' in data_path:  # load about 1/16 of the data
      train_dataset = tfds.load(
          name=dataset_name, split='train[:6%]', as_supervised=as_supervised)
    elif 'half' in data_path:  # load half of the data
      train_dataset = tfds.load(
          name=dataset_name, split='train[:50%]', as_supervised=as_supervised)
    else:  # load 1/4 of the data
      train_dataset = tfds.load(
          name=dataset_name, split='train[:25%]', as_supervised=as_supervised)

  if randomize_labels:
    all_labels = []
    all_images = []
    for images, labels in train_dataset:
      all_labels.extend([labels.numpy()])
      all_images.append(images.numpy()[np.newaxis, :, :, :])
    all_images = np.vstack(all_images)
    np.random.seed(FLAGS.copy)
    np.random.shuffle(all_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(all_images, dtype=tf.float32),
         tf.convert_to_tensor(all_labels, dtype=tf.int64)))

  train_dataset = train_dataset.shuffle(buffer_size=n_data)
  if as_supervised:
    train_dataset = train_dataset.map(
        functools.partial(preprocess_data, is_training=True))
  else:
    train_dataset = train_dataset.map(
        functools.partial(preprocess_data_with_id, is_training=True))
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return train_dataset


def load_test_data(batch_size,
                   shuffle=False,
                   data_path='',
                   dataset_name='cifar10',
                   n_data=10000,
                   as_supervised=True):
  """Load CIFAR test data"""
  if 'random' in dataset_name:
    np.random.seed(0)
    test_labels = np.zeros((n_data,), dtype=np.int64)
    test_data = np.random.rand(n_data, 32, 32, 3)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
  else:
    test_dataset = tfds.load(
        name=dataset_name, split='test', as_supervised=as_supervised)
    if as_supervised:
      test_dataset = test_dataset.map(
          functools.partial(preprocess_data, is_training=False))
    else:
      test_dataset = test_dataset.map(
          functools.partial(preprocess_data_with_id, is_training=False))

  if shuffle:
    test_dataset = test_dataset.shuffle(buffer_size=n_data)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  return test_dataset


def main(argv):
  if FLAGS.data_path:
    if 'tiny' in FLAGS.data_path:
      n_data = int(50000 * 6 / 100)
    elif 'half' in FLAGS.data_path:
      n_data = 50000 // 2
    elif 'subsampled' in FLAGS.data_path:
      n_data = 50000 // 4
  else:
    n_data = 50000

  train_dataset = load_train_data(
      FLAGS.batch_size,
      dataset_name=FLAGS.dataset_name,
      n_data=n_data,
      data_path=FLAGS.data_path,
      randomize_labels=FLAGS.randomize_labels,
      as_supervised=not FLAGS.save_image)

  test_dataset = load_test_data(
      FLAGS.batch_size, dataset_name=FLAGS.dataset_name, n_data=10000)
  steps_per_epoch = n_data // FLAGS.batch_size
  optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate, momentum=0.9)
  schedule = tf.keras.experimental.CosineDecay(FLAGS.learning_rate,
                                               FLAGS.epochs)
  lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

  if FLAGS.dataset_name == 'cifar100':
    num_classes = 100
  else:
    num_classes = 10

  if FLAGS.shake_shake:
    model = build_shake_shake_model(num_classes, FLAGS.depth,
                                    FLAGS.width_multiplier, FLAGS.weight_decay)
  else:
    model = ResNet_CIFAR(
        FLAGS.depth,
        FLAGS.width_multiplier,
        FLAGS.weight_decay,
        num_classes=num_classes,
        use_residual=FLAGS.use_residual,
        save_image=FLAGS.save_image)

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

  if FLAGS.save_image:
    model.compile(
        optimizer,
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'],
        run_eagerly=True)
  else:
    model.compile(
        optimizer,
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

  model_dir = 'cifar-depth-%d-width-%d-bs-%d-lr-%f-reg-%f/' % \
      (FLAGS.depth, FLAGS.width_multiplier, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.weight_decay)
  experiment_dir = os.path.join(FLAGS.base_dir, model_dir)
  if FLAGS.copy > 0:
    experiment_dir = '%s/cifar-depth-%d-width-%d-bs-%d-lr-%f-reg-%f-copy-%d/' % \
      (FLAGS.base_dir, FLAGS.depth, FLAGS.width_multiplier, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.weight_decay, FLAGS.copy)

  if FLAGS.epoch_save_freq > 0:
    tf.keras.models.save_model(
        model, experiment_dir, overwrite=True,
        include_optimizer=False)  # Save initialization
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
      validation_data=test_dataset,
      verbose=1,
      steps_per_epoch=steps_per_epoch,
      callbacks=[checkpoint, lr_scheduler])

  if FLAGS.save_image:
    pickle.dump(model.all_ids,
                tf.io.gfile.GFile(
                    os.path.join(experiment_dir, 'image_ids.pkl'), 'wb'))

  best_model = tf.keras.models.load_model(experiment_dir)
  best_model.compile(
      'sgd',
      tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['acc'])
  test_metrics = best_model.evaluate(test_dataset, verbose=1)


  logging.info('Test accuracy: %.4f', test_metrics[1])


if __name__ == '__main__':
  app.run(main)
