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

"""Train a feed forward model to predict the label of a given mnist image.

The script loads MNIST Dataset and trains and a keras model to predict
the labels.
"""
import json
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
import tensorflow_datasets as tfds

from smug_saliency.mnist_models import mnist_constants

tf.compat.v1.enable_eager_execution()

# model saving options
flags.DEFINE_string('model_path', '',
                    'Path to save model weights and configurations.')
flags.DEFINE_integer('save_period', 10,
                     'Number of epochs to save a new model checkpoint.')

# model hyperparameters for Dense layers
flags.DEFINE_string('num_dense_units', '32',
                    'Comma-separated string describing the number of units in '
                    'each Dense hidden layer.')

# training hyperparameters
flags.DEFINE_integer('epochs', 2, 'Number of epochs for Keras model.fit()')
flags.DEFINE_float('learning_rate', 0.001,
                   'Initial learning rate for Keras optimizers.')
flags.DEFINE_float('dropout', 0.0,
                   'Dropout probability following each Dense hidden layer.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for Keras model.fit().')

FLAGS = flags.FLAGS


def process_images(data_point):
  """Reshape features and convert labels to one-hot vectors.

  Args:
    data_point: dictionary, attributes = 'image', 'label'
        'image' has a shape (28 * 28 * 1), dtype='utf-8' and
        'label' is a scalar with dtype='int64'.

  Returns:
    A tuple of features and label
  """
  image = tf.image.convert_image_dtype(data_point['image'], tf.float32)
  # Flatten the input image. Original shape = (28 * 28 * 1)
  image = tf.reshape(image, [-1])
  label = tf.one_hot(data_point['label'], depth=10)
  return image, label


def dataloader(loader, mode):
  """Sets batchsize and repeat for the train, valid, and test iterators.

  Args:
    loader: tfds.load instance, a train, valid, or test iterator.
    mode: string, set to 'train' for use during training;
        set to anything else for use during validation/test

  Returns:
    An iterator for features and labels tensors.
  """
  loader = loader.map(process_images)
  repeat = 1
  if mode == 'train':
    repeat = None
    loader = loader.shuffle(1000 * FLAGS.batch_size)
  return loader.batch(
      FLAGS.batch_size).repeat(repeat).prefetch(tf.data.experimental.AUTOTUNE)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # Verify that it is actually using GPU
  logging.info('Num GPUs Available: %d',
               len(tf.config.experimental.list_physical_devices('GPU')))
  model_config = dict({
      'num_dense_units': [
          # create empty list for empty string.
          int(value) for value in FLAGS.num_dense_units.split(',') if value],
      'dropout': FLAGS.dropout
  })

  # Stage 1: Load Dataset
  train_ds = dataloader(
      tfds.load(
          'mnist',
          split='train[:{}%]'.format(mnist_constants.TRAIN_DATA_PERCENT)),
      mode='train')
  valid_ds = dataloader(
      tfds.load(
          'mnist',
          split='train[{}%:]'.format(mnist_constants.TRAIN_DATA_PERCENT)),
      mode='valid')
  test_ds = dataloader(tfds.load('mnist', split='test'), mode='test')

  num_hidden_units_list = model_config['num_dense_units']

  # Stage 2: Create Model
  layers = [
      tf.keras.layers.Dense(
          units=num_hidden_units_list[0],
          input_shape=(mnist_constants.NUM_FLATTEN_FEATURES,),
          activation='relu'),
      tf.keras.layers.Dropout(FLAGS.dropout)
  ]
  for hidden_units in num_hidden_units_list[1:]:
    layers.append(
        tf.keras.layers.Dense(
            units=hidden_units,
            activation='relu'))
    layers.append(tf.keras.layers.Dropout(FLAGS.dropout))

  layers.append(
      tf.keras.layers.Dense(
          units=mnist_constants.NUM_OUTPUTS, activation='softmax'))
  model = tf.keras.Sequential(layers)

  # Stage 3: configure checkpoints and save model metadata.
  model_dir = FLAGS.model_path

  if not gfile.Exists(model_dir):
    gfile.MakeDirs(model_dir)
  logging.info('Model will be saved to: %s', model_dir)

  # Constantly save model weight checkpoints during training.
  model_weight_path = os.path.join(model_dir, 'weights_epoch{epoch:04d}')
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          model_weight_path,
          period=FLAGS.save_period,
          save_weights_only=False),
      tf.keras.callbacks.TensorBoard(log_dir=model_dir)
  ]
  # Also need model configuration to fully reconstruct model later.
  model_config_path = os.path.join(model_dir, 'model_config.json')
  logging.info('Write model configuration to: %s', model_config_path)
  with gfile.Open(model_config_path, 'wb') as f:
    f.write(json.dumps(model_config, indent=4, sort_keys=True))

  # Stage 4: train model and save weights.
  model.compile(
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=['accuracy'])

  model.fit(
      train_ds,
      validation_data=valid_ds,
      epochs=FLAGS.epochs,
      steps_per_epoch=mnist_constants.NUM_TRAIN_EXAMPLES // FLAGS.batch_size,
      callbacks=callbacks)

  _, test_accuracy = model.evaluate(test_ds)

  # Stage 5: save losses, accuracies, and result
  with gfile.Open(os.path.join(model_dir, 'test_accuracy.txt'), 'wb') as f:
    f.write('{:.2f}%'.format(100*test_accuracy))

if __name__ == '__main__':
  app.run(main)
