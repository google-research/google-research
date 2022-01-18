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

"""Code to load, preprocess and train on CIFAR-10."""
import functools
import os

from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from do_wide_and_deep_networks_learn_the_same_things.large_scale_training import train_lib
from do_wide_and_deep_networks_learn_the_same_things.resnet_cifar import ResNet_CIFAR

tf.enable_v2_behavior()

FLAGS = flags.FLAGS
# Define training hyperparameters.
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train for')
flags.DEFINE_integer('epochs_between_evals', 10,
                     'Number of epochs between evals and checkpoints.')
flags.DEFINE_float('weight_decay', 0.001, 'L2 regularization')
# Define model & data hyperparameters.
flags.DEFINE_integer('depth', 56, 'No. of layers to use in the ResNet model')
flags.DEFINE_float(
    'width_multiplier', 1,
    'How much to scale the width of the standard ResNet model by')
flags.DEFINE_integer(
    'copy', 0,
    'If the same model configuration has been run before, train another copy '
    'with a different random initialization')
flags.DEFINE_string('base_dir', None,
                    'Where the trained model will be saved')
flags.DEFINE_string('data_path', None,
                    'Directory where CIFAR-10 subsampled dataset is stored')
flags.DEFINE_boolean('distort_color', False,
                     'Whether to apply color distortion augmentation')


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


def preprocess_data(images, labels, is_training):
  """CIFAR data preprocessing"""
  images = tf.image.convert_image_dtype(images, tf.float32)

  if is_training:
    crop_padding = 4
    images = tf.pad(images, [[crop_padding, crop_padding],
                             [crop_padding, crop_padding], [0, 0]], 'REFLECT')
    images = tf.image.random_crop(images, [32, 32, 3])
    images = tf.image.random_flip_left_right(images)
    if FLAGS.distort_color:
      images = color_distortion(images, s=1.0)
  return {'input_1': images, 'label': labels}


def load_train_data(batch_size,
                    data_path='',
                    dataset_name='cifar10',
                    n_data=50000):
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

  train_dataset = train_dataset.shuffle(buffer_size=n_data)
  train_dataset = train_dataset.map(
      functools.partial(preprocess_data, is_training=True),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return train_dataset


def load_test_data(batch_size,
                   shuffle=False,
                   data_path='',
                   dataset_name='cifar10',
                   n_data=10000):
  """Load CIFAR test data"""
  if 'random' in dataset_name:
    np.random.seed(0)
    test_labels = np.zeros((n_data,), dtype=np.int64)
    test_data = np.random.rand(n_data, 32, 32, 3)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
  else:
    test_dataset = tfds.load(
        name=dataset_name, split='test', as_supervised=as_supervised)
    test_dataset = test_dataset.map(
        functools.partial(preprocess_data, is_training=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if shuffle:
    test_dataset = test_dataset.shuffle(buffer_size=n_data)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  return test_dataset


def main(argv):
  del argv

  train_dataset = load_train_data(FLAGS.batch_size, data_path=FLAGS.data_path)
  if FLAGS.data_path:
    if 'subsampled-tiny' in FLAGS.data_path:
      train_dataset_size = 50000 // 16
    elif 'subsampled' in FLAGS.data_path:
      train_dataset_size = 50000 // 4
  else:
    train_dataset_size = 50000

  # Always use full test dataset as validation.
  test_dataset = load_test_data(FLAGS.batch_size)
  test_dataset_size = 10000

  steps_per_epoch = train_dataset_size // FLAGS.batch_size
  steps_between_evals = int(FLAGS.epochs_between_evals * steps_per_epoch)
  train_steps = FLAGS.epochs * steps_per_epoch
  eval_steps = ((test_dataset_size - 1) // FLAGS.batch_size) + 1

  model_dir_name = 'cifar-depth-%d-width-%s-bs-%d-lr-%f-reg-%f' % \
      (FLAGS.depth, FLAGS.width_multiplier, FLAGS.batch_size,
       FLAGS.learning_rate, FLAGS.weight_decay)
  if FLAGS.copy > 0:
    model_dir_name += '-copy-%d' % FLAGS.copy
  experiment_dir = os.path.join(FLAGS.base_dir, model_dir_name)

  def model_optimizer_fn():
    model = ResNet_CIFAR(
        FLAGS.depth, FLAGS.width_multiplier, FLAGS.weight_decay, sync_bn=True)
    schedule = tf.keras.experimental.CosineDecay(FLAGS.learning_rate,
                                                 train_steps)
    optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9)
    return model, optimizer

  train_lib.train(
      model_optimizer_fn=model_optimizer_fn,
      train_steps=train_steps,
      eval_steps=eval_steps,
      steps_between_evals=steps_between_evals,
      train_dataset=train_dataset,
      test_dataset=test_dataset,
      experiment_dir=experiment_dir)


if __name__ == '__main__':
  app.run(main)
