# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Train DNN of a specified architecture on a specified data set."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import os
import sys
import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS
CNN_KERNEL_SIZE = 3

flags.DEFINE_integer('num_layers', 3, 'Number of layers in the network.')
flags.DEFINE_integer('num_units', 16, 'Number of units in a dense layer.')
flags.DEFINE_integer('batchsize', 512, 'Size of the mini-batch.')
flags.DEFINE_float(
    'train_fraction', 1.0, 'How much of the dataset to use for'
    'training [as fraction]: eg. 0.15, 0.5, 1.0')
flags.DEFINE_integer('epochs', 18, 'How many epochs to train for')
flags.DEFINE_integer('epochs_between_checkpoints', 6,
                     'How many epochs to train between creating checkpoints')
flags.DEFINE_integer('random_seed', 42, 'Random seed.')
flags.DEFINE_integer('cnn_stride', 2, 'Stride of the CNN')
flags.DEFINE_float('dropout', 0.0, 'Dropout Rate')
flags.DEFINE_float('l2reg', 0.0, 'L2 regularization strength')
flags.DEFINE_float('init_std', 0.05, 'Standard deviation of the initializer.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_string('optimizer', 'sgd',
                    'Optimizer algorithm: sgd / adam / momentum.')
flags.DEFINE_string('activation', 'relu',
                    'Nonlinear activation: relu / tanh / sigmoind / selu.')
flags.DEFINE_string(
    'w_init', 'he_normal', 'Initialization for weights. '
    'see tf.keras.initializers for options')
flags.DEFINE_string(
    'b_init', 'zero', 'Initialization for biases.'
    'see tf.keras.initializers for options')
flags.DEFINE_boolean('grayscale', True, 'Convert input images to grayscale.')
flags.DEFINE_boolean('augment_traindata', False, 'Augmenting Training data.')
flags.DEFINE_boolean('reduce_learningrate', False,
                     'Reduce LR towards end of training.')
flags.DEFINE_string('dataset', 'mnist', 'Name of the dataset compatible '
                    'with TFDS.')
flags.DEFINE_string('dnn_architecture', 'cnn',
                    'Architecture of the DNN [fc, cnn, cnnbn]')
flags.DEFINE_string(
    'workdir', '/tmp/dnn_science_workdir', 'Base working directory for storing'
    'checkpoints, summaries, etc.')
flags.DEFINE_integer('verbose', 0, 'Verbosity')
flags.DEFINE_bool('use_tpu', False, 'Whether running on TPU or not.')
flags.DEFINE_string('master', 'local',
                    'Name of the TensorFlow master to use. "local" for GPU.')
flags.DEFINE_string(
    'tpu_job_name', 'tpu_worker',
    'Name of the TPU worker job. This is required when having multiple TPU '
    'worker jobs.')


def _get_workunit_params():
  """Get command line parameters of the current process as dict."""
  main_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  params = {'config.' + k.name: k.value for k in main_flags}
  return params


def store_results(info_dict, filepath):
  """Save results in the json file."""
  with gfile.GFile(filepath, 'w') as json_fp:
    json.dump(info_dict, json_fp)


def restore_results(filepath):
  """Retrieve results in the json file."""
  with gfile.GFile(filepath, 'r') as json_fp:
    info = json.load(json_fp)
  return info


def _preprocess_batch(batch,
                      normalize,
                      to_grayscale,
                      augment=False):
  """Preprocessing function for each batch of data."""
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


def get_dataset(dataset,
                batchsize,
                to_grayscale=True,
                train_fraction=1.0,
                shuffle_buffer=1024,
                random_seed=None,
                normalize=True,
                augment=False):
  """Load and preprocess the dataset.

  Args:
    dataset: The dataset name. Either 'toy' or a TFDS dataset
    batchsize: the desired batch size
    to_grayscale: if True, all images will be converted into grayscale
    train_fraction: what fraction of the overall training set should we use
    shuffle_buffer: size of the shuffle.buffer for tf.data.Dataset.shuffle
    random_seed: random seed for shuffling operations
    normalize: whether to normalize the data into [-1, 1]
    augment: use data augmentation on the training set.

  Returns:
    tuple (training_dataset, test_dataset, info), where info is a dictionary
    with some relevant information about the dataset.
  """
  data_tr, ds_info = tfds.load(dataset, split='train', with_info=True)
  effective_train_size = ds_info.splits['train'].num_examples

  if train_fraction < 1.0:
    effective_train_size = int(effective_train_size * train_fraction)
    data_tr = data_tr.shuffle(shuffle_buffer, seed=random_seed)
    data_tr = data_tr.take(effective_train_size)

  fn_tr = lambda b: _preprocess_batch(b, normalize, to_grayscale, augment)
  data_tr = data_tr.shuffle(shuffle_buffer, seed=random_seed)
  data_tr = data_tr.batch(batchsize, drop_remainder=True)
  data_tr = data_tr.map(fn_tr, tf.data.experimental.AUTOTUNE)
  data_tr = data_tr.prefetch(tf.data.experimental.AUTOTUNE)

  fn_te = lambda b: _preprocess_batch(b, normalize, to_grayscale, False)
  data_te = tfds.load(dataset, split='test')
  data_te = data_te.batch(batchsize)
  data_te = data_te.map(fn_te, tf.data.experimental.AUTOTUNE)
  data_te = data_te.prefetch(tf.data.experimental.AUTOTUNE)

  dataset_info = {
      'num_classes': ds_info.features['label'].num_classes,
      'data_shape': ds_info.features['image'].shape,
      'train_num_examples': effective_train_size
  }
  return data_tr, data_te, dataset_info


def build_cnn(n_layers, n_hidden, n_outputs, dropout_rate, activation, stride,
              w_regularizer, w_init, b_init, use_batchnorm):
  """Convolutional deep neural network."""
  model = tf.keras.Sequential()
  for _ in range(n_layers):
    model.add(
        tf.keras.layers.Conv2D(
            n_hidden,
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
          n_outputs,
          kernel_regularizer=w_regularizer,
          kernel_initializer=w_init,
          bias_initializer=b_init))
  return model


def build_fcn(n_layers, n_hidden, n_outputs, dropout_rate, activation,
              w_regularizer, w_init, b_init, use_batchnorm):
  """Fully Connected deep neural network."""
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten())
  for _ in range(n_layers):
    model.add(
        tf.keras.layers.Dense(
            n_hidden,
            activation=activation,
            kernel_regularizer=w_regularizer,
            kernel_initializer=w_init,
            bias_initializer=b_init))
    if dropout_rate > 0.0:
      model.add(tf.keras.layers.Dropout(dropout_rate))
    if use_batchnorm:
      model.add(tf.keras.layers.BatchNormalization())
  model.add(
      tf.keras.layers.Dense(
          n_outputs,
          kernel_regularizer=w_regularizer,
          kernel_initializer=w_init,
          bias_initializer=b_init))
  return model


def eval_model(model, data_tr, data_te, info, logger, cur_epoch, workdir):
  """Runs Model Evaluation."""
  # get training set metrics in eval-mode (no dropout etc.)
  metrics_te = model.evaluate(data_te, verbose=0)
  res_te = dict(zip(model.metrics_names, metrics_te))
  metrics_tr = model.evaluate(data_tr, verbose=0)
  res_tr = dict(zip(model.metrics_names, metrics_tr))
  metrics = {
      'train_accuracy': res_tr['accuracy'],
      'train_loss': res_tr['loss'],
      'test_accuracy': res_te['accuracy'],
      'test_loss': res_te['loss'],
  }
  for k in metrics:
    info[k][cur_epoch] = float(metrics[k])
  metrics['epoch'] = cur_epoch  # so it's included in the logging output
  print(metrics)
  savepath = os.path.join(workdir, 'permanent_ckpt-%d' % cur_epoch)
  model.save(savepath)


def run(workdir,
        data,
        strategy,
        architecture,
        n_layers,
        n_hiddens,
        activation,
        dropout_rate,
        l2_penalty,
        w_init_name,
        b_init_name,
        optimizer_name,
        learning_rate,
        n_epochs,
        epochs_between_checkpoints,
        init_stddev,
        cnn_stride,
        reduce_learningrate=False,
        verbosity=0):
  """Runs the whole training procedure."""
  data_tr, data_te, dataset_info = data
  n_outputs = dataset_info['num_classes']

  with strategy.scope():
    optimizer = tf.keras.optimizers.get(optimizer_name)
    optimizer.learning_rate = learning_rate
    w_init = tf.keras.initializers.get(w_init_name)
    if w_init_name.lower() in ['truncatednormal', 'randomnormal']:
      w_init.stddev = init_stddev
    b_init = tf.keras.initializers.get(b_init_name)
    if b_init_name.lower() in ['truncatednormal', 'randomnormal']:
      b_init.stddev = init_stddev
    w_reg = tf.keras.regularizers.l2(l2_penalty) if l2_penalty > 0 else None

    if architecture == 'cnn' or architecture == 'cnnbn':
      model = build_cnn(n_layers, n_hiddens, n_outputs, dropout_rate,
                        activation, cnn_stride, w_reg, w_init, b_init,
                        architecture == 'cnnbn')
    elif architecture == 'fcn':
      model = build_fcn(n_layers, n_hiddens, n_outputs, dropout_rate,
                        activation, w_reg, w_init, b_init, False)
    else:
      assert False, 'Unknown architecture: ' % architecture

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', 'mse', 'sparse_categorical_crossentropy'])

  # force the model to set input shapes and init weights
  for x, _ in data_tr:
    model.predict(x)
    if verbosity:
      model.summary()
    break

  ckpt = tf.train.Checkpoint(
      step=optimizer.iterations, optimizer=optimizer, model=model)
  ckpt_dir = os.path.join(workdir, 'temporary-ckpt')
  ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
  if ckpt_manager.latest_checkpoint:
    logging.info('restoring checkpoint: %s', ckpt_manager.latest_checkpoint)
    print('restoring from %s' % ckpt_manager.latest_checkpoint)
    with strategy.scope():
      ckpt.restore(ckpt_manager.latest_checkpoint)
    info = restore_results(os.path.join(workdir, '.intermediate-results.json'))
    print(info, flush=True)
  else:
    info = {
        'steps': 0,
        'start_time': time.time(),
        'train_loss': dict(),
        'train_accuracy': dict(),
        'test_loss': dict(),
        'test_accuracy': dict(),
    }
    info.update(_get_workunit_params())  # Add command line parameters.

  logger = None
  starting_epoch = len(info['train_loss'])
  cur_epoch = starting_epoch
  for cur_epoch in range(starting_epoch, n_epochs):
    if reduce_learningrate and cur_epoch == n_epochs - (n_epochs // 10):
      optimizer.learning_rate = learning_rate / 10
    elif reduce_learningrate and cur_epoch == n_epochs - 2:
      optimizer.learning_rate = learning_rate / 100

    # Train until we reach the criterion or get NaNs
    try:
      # always keep checkpoints for the first few epochs
      # we evaluate first and train afterwards so we have the at-init data
      if cur_epoch < 4 or (cur_epoch % epochs_between_checkpoints) == 0:
        eval_model(model, data_tr, data_te, info, logger, cur_epoch, workdir)

      model.fit(data_tr, epochs=1, verbose=verbosity)
      ckpt_manager.save()
      store_results(info, os.path.join(workdir, '.intermediate-results.json'))

      dt = time.time() - info['start_time']
      logging.info('epoch %d (%3.2fs)', cur_epoch, dt)

    except tf.errors.InvalidArgumentError as e:
      # We got NaN in the loss, most likely gradients resulted in NaNs
      logging.info(str(e))
      info['status'] = 'NaN'
      logging.info('Stop training because NaNs encountered')
      break

  eval_model(model, data_tr, data_te, info, logger, cur_epoch+1, workdir)
  store_results(info, os.path.join(workdir, 'results.json'))

  # we don't need the temporary checkpoints anymore
  gfile.rmtree(os.path.join(workdir, 'temporary-ckpt'))
  gfile.remove(os.path.join(workdir, '.intermediate-results.json'))


def main(unused_argv):
  workdir = FLAGS.workdir


  if not gfile.isdir(workdir):
    gfile.makedirs(workdir)

  tf.random.set_seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  data = get_dataset(
      FLAGS.dataset,
      FLAGS.batchsize,
      to_grayscale=FLAGS.grayscale,
      train_fraction=FLAGS.train_fraction,
      random_seed=FLAGS.random_seed,
      augment=FLAGS.augment_traindata)

  # Figure out TPU related stuff and create distribution strategy
  use_remote_eager = FLAGS.master and FLAGS.master != 'local'
  if FLAGS.use_tpu:
    logging.info("Use TPU at %s with job name '%s'.", FLAGS.master,
                 FLAGS.tpu_job_name)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.master, job_name=FLAGS.tpu_job_name)
    if use_remote_eager:
      tf.config.experimental_connect_to_cluster(resolver)
      logging.warning('Remote eager configured. Remote eager can be slow.')
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    if use_remote_eager:
      tf.config.experimental_connect_to_host(
          FLAGS.master, job_name='gpu_worker')
      logging.warning('Remote eager configured. Remote eager can be slow.')
    gpus = tf.config.experimental.list_logical_devices(device_type='GPU')
    if gpus:
      logging.info('Found GPUs: %s', gpus)
      strategy = tf.distribute.MirroredStrategy()
    else:
      logging.info('Devices: %s', tf.config.list_logical_devices())
      strategy = tf.distribute.OneDeviceStrategy('CPU')
  logging.info('Devices: %s', tf.config.list_logical_devices())
  logging.info('Distribution strategy: %s', strategy)
  logging.info('Model directory: %s', workdir)

  run(workdir,
      data,
      strategy,
      architecture=FLAGS.dnn_architecture,
      n_layers=FLAGS.num_layers,
      n_hiddens=FLAGS.num_units,
      activation=FLAGS.activation,
      dropout_rate=FLAGS.dropout,
      l2_penalty=FLAGS.l2reg,
      w_init_name=FLAGS.w_init,
      b_init_name=FLAGS.b_init,
      optimizer_name=FLAGS.optimizer,
      learning_rate=FLAGS.learning_rate,
      n_epochs=FLAGS.epochs,
      epochs_between_checkpoints=FLAGS.epochs_between_checkpoints,
      init_stddev=FLAGS.init_std,
      cnn_stride=FLAGS.cnn_stride,
      reduce_learningrate=FLAGS.reduce_learningrate,
      verbosity=FLAGS.verbose)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)
