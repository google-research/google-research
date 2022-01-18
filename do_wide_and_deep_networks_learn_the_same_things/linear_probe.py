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

"""Train and evaluate linear probes for Add and Batch Norm right before Add layers."""

from absl import app
from absl import flags

import pickle
import functools
import os
import numpy as np
from skimage.measure import block_reduce

import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds


FLAGS = flags.FLAGS
flags.DEFINE_string('trained_model', '',
                    'Path to where the trained model is saved')
flags.DEFINE_integer('layer_idx', 0,
                     'Layer index from which activations are extracted')
flags.DEFINE_string('data_dir', '',
                    'Directory where CIFAR-10 subsampled dataset is stored')
flags.DEFINE_integer('batch', 128, 'Batch size')
flags.DEFINE_float('lr', 0.01, 'Learning rate')
flags.DEFINE_integer('num_epochs', 300, 'Number of epochs to train for')
flags.DEFINE_float('l2_reg', 0, 'L2 regularization strength')
flags.DEFINE_boolean(
    'pooling', False,
    'Whether to apply spatial pooling on activations to reduce dimensions')
flags.DEFINE_string('data_sample', '',
                    'Directory where CIFAR subsampled dataset is stored')


def get_layer_activations(images, model, layer_idx):
  """Returns activations obtained from a model at layer_idx on a set of images."""
  if len(images.get_shape().as_list()) == 3:
    images = tf.expand_dims(images, axis=0)
  input_layer = model.input
  layer_output = model.layers[layer_idx].output
  get_layer_outputs = K.function(input_layer, layer_output)
  activations = get_layer_outputs(images)
  return activations


def preprocess_linear_probe_data(images,
                                 labels,
                                 is_training,
                                 model,
                                 layer_idx,
                                 pooling=False):
  """Preprocesses input images as in standard training and returns activations extracted from a given layer_idx."""
  images = tf.image.convert_image_dtype(images, tf.float32)

  if is_training:
    crop_padding = 4
    images = tf.pad(images, [[crop_padding, crop_padding],
                             [crop_padding, crop_padding], [0, 0]], 'REFLECT')
    images = tf.image.random_crop(images, [32, 32, 3])
    images = tf.image.random_flip_left_right(images)

  activations = get_layer_activations(images, model, layer_idx)
  if pooling:
    pooling_scale = np.ones(len(activations.shape), dtype=np.int8)
    pooling_scale[1] = 2
    pooling_scale[2] = 2
    activations = block_reduce(activations, tuple(pooling_scale), np.max)
  activations = activations.flatten()
  return activations, labels


def load_linear_probe_train_data(model,
                                 layer_idx,
                                 input_shape,
                                 batch_size,
                                 data_path=None):
  """Loads train data for linear probe experiments."""
  buffer_size = 50000
  if 'tiny' in data_path:
    train_dataset = tfds.load(
        name='cifar10', split='train[:6%]', as_supervised=True)
  else:
    train_dataset = tfds.load(name='cifar10', split='train', as_supervised=True)

  if 'tiny' in data_path:
    buffer_size //= 16

  train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
  processing_fn = lambda x, y: tf.py_function(
      inp=(x, y),
      func=functools.partial(
          preprocess_linear_probe_data,
          model=model,
          layer_idx=layer_idx,
          is_training=True,
          pooling=FLAGS.pooling),
      Tout=[tf.float32, tf.int64])
  train_dataset = train_dataset.map(processing_fn)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return train_dataset


def load_linear_probe_test_data(model,
                                layer_idx,
                                input_shape,
                                batch_size,
                                shuffle=False,
                                data_path=None,
                                dataset_name='cifar10'):
  """Loads test data for linear probe experiments."""
  buffer_size = 10000
  if data_path is None:
    test_dataset = tfds.load(
        name=dataset_name, split='test', as_supervised=True)
  else:
    test_data = pickle.load(
        tf.io.gfile.GFile(os.path.join(data_path, 'test_data.pkl'),
                   'rb')).astype(np.uint8)
    test_labels = pickle.load(
        tf.io.gfile.GFile(os.path.join(data_path, 'test_labels.pkl'), 'rb'))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    if 'tiny' in data_path:
      buffer_size //= 16
    else:
      buffer_size //= 4

  processing_fn = lambda x, y: tf.py_function(
      inp=(x, y),
      func=functools.partial(
          preprocess_linear_probe_data,
          model=model,
          layer_idx=layer_idx,
          is_training=False,
          pooling=FLAGS.pooling),
      Tout=[tf.float32, tf.int64])
  test_dataset = test_dataset.map(processing_fn)
  if shuffle:
    test_dataset = test_dataset.shuffle(buffer_size=buffer_size)
  test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
  return test_dataset


def main(argv):
  if FLAGS.data_dir:
    if 'subsampled-tiny' in FLAGS.data_dir:
      n_data = 50000 // 16
    elif 'subsampled' in FLAGS.data_dir:
      n_data = 50000 // 4
  else:
    n_data = 50000
  steps_per_epoch = n_data // FLAGS.batch
  optimizer = tf.keras.optimizers.SGD(FLAGS.lr, momentum=0.9)

  trained_model = tf.keras.models.load_model(FLAGS.trained_model)
  layer_output = trained_model.layers[FLAGS.layer_idx].output
  out_dim = np.array(
      layer_output.get_shape().as_list()[1:])  # remove batch dimension
  if FLAGS.pooling:
    out_dim[0] /= 2
    out_dim[1] /= 2
  total_dim = np.prod(out_dim)
  train_dataset = load_linear_probe_train_data(
      trained_model,
      FLAGS.layer_idx, (total_dim,),
      FLAGS.batch,
      data_path=FLAGS.data_sample)
  test_dataset = load_linear_probe_test_data(
      trained_model, FLAGS.layer_idx, (total_dim,),
      FLAGS.batch)  #use full test dataset as validation

  #Define linear model
  inputs = tf.keras.Input(shape=(total_dim,))
  outputs = tf.keras.layers.Dense(
      10,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2_reg))(
          inputs)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer,
      tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['acc'])
  save_dir = 'layer-%d-bs-%d-lr-%f-reg-%f' % \
      (FLAGS.layer_idx, FLAGS.batch, FLAGS.lr, FLAGS.l2_reg)
  if FLAGS.pooling:
    save_dir += '-pooling'
  experiment_dir = os.path.join(FLAGS.trained_model, save_dir)


  # Resume training in case of preemption
  optimizer_weights_set = True
  ckpt_path = os.path.join(experiment_dir, 'ckpt')
  opt_path = os.path.join(ckpt_path, 'optimizer_weights.pkl')
  metadata_path = os.path.join(ckpt_path, 'metadata.pkl')

  if not tf.io.gfile.exists(ckpt_path):
    tf.io.gfile.makedirs(ckpt_path)
  if tf.io.gfile.listdir(ckpt_path):
    opt_weights = pickle.load(tf.io.gfile.GFile(opt_path, 'rb'))
    optimizer_weights_set = False
    #optimizer.set_weights(opt_weights)
    model = tf.keras.models.load_model(ckpt_path)

  if tf.io.gfile.exists(metadata_path):
    metadata = pickle.load(tf.io.gfile.GFile(metadata_path, 'rb'))
    start_epoch = metadata['latest_epoch'] + 1
    best_val_acc = metadata['best_acc']
  else:
    best_val_acc = 0
    start_epoch = 0

  # Start training
  for epoch in range(start_epoch, FLAGS.num_epochs):
    for (batch_id, (images, labels)) in enumerate(train_dataset.take(-1)):
      if batch_id >= steps_per_epoch:
        continue

      with tf.GradientTape(persistent=True) as tape:
        logits = model(images, training=True)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)(labels, logits)

      grads = tape.gradient(loss_fn, model.trainable_variables)

      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      if not optimizer_weights_set:  # optimizer weights are only created during the first step
        optimizer.set_weights(opt_weights)
        optimizer_weights_set = True

    #Evaluate the model and print results
    n_correct_preds, n_val = 0, 10000
    for (_, (images, labels)) in enumerate(test_dataset.take(-1)):
      logits = model(images, training=False)
      correct_preds = tf.equal(tf.argmax(input=logits, axis=1), labels)
      n_correct_preds += correct_preds.numpy().sum()
    val_accuracy = n_correct_preds / n_val

    if val_accuracy > best_val_acc:
      best_val_acc = val_accuracy
      tf.keras.models.save_model(
          model, experiment_dir, overwrite=True, include_optimizer=False)

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
      tf.keras.models.save_model(
          model, ckpt_path, overwrite=True, include_optimizer=False)
      metadata = {'latest_epoch': epoch, 'best_acc': best_val_acc}
      pickle.dump(metadata, tf.io.gfile.GFile(metadata_path, 'wb'))
      opt_weights = optimizer.get_weights()
      pickle.dump(opt_weights, tf.io.gfile.GFile(opt_path, 'wb'))


if __name__ == '__main__':
  app.run(main)
