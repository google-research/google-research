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

"""Main module for VAE OOD experiments."""

import os
import pickle
import sys

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from vae_ood import dataset_utils
from vae_ood import network
from vae_ood import utils

# pylint: disable=g-bad-import-order


_DO_TRAIN = flags.DEFINE_bool('do_train', default=False, help='Do training')
_DO_EVAL = flags.DEFINE_bool('do_eval', default=False, help='Do evaluaion')
_DATASET = flags.DEFINE_string(
    'dataset', default='fashion_mnist', help='Name of the dataset to train on')
_NORMALIZE = flags.DEFINE_string(
    'normalize', default=None, help='Normalization to apply')
_TEST_NORMALIZE = flags.DEFINE_string(
    'test_normalize',
    default='same',
    help='Normalization to apply at test time')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', default=64, help='Batch size')
_EVAL_EVERY = flags.DEFINE_integer('eval_every', default=100, help='Eval every')
_NUM_FILTERS = flags.DEFINE_integer(
    'num_filters', default=32, help='Number of conv filters')
_LATENT_DIM = flags.DEFINE_integer(
    'latent_dim', default=100, help='Latent dimension')
_LR = flags.DEFINE_float('lr', default=5e-4, help='Learning rate')
_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs', default=100, help='Number of epochs')
_VISIBLE_DIST = flags.DEFINE_string(
    'visible_dist', default='cont_bernoulli', help='Visible distribution')
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir', default='test', help='Model checkpoint directory')


CIFAR10_X = [f'cifar10-{x}' for x in range(10)]
COLOR_DATASETS = [
    'svhn_cropped',
    'cifar10',
    'celeb_a',
    'gtsrb',
    'compcars',
    'noise'
]
GRAYSCALE_DATASETS = [
    'mnist',
    'fashion_mnist',
    'emnist/letters',
    'sign_lang',
    'noise'
]
ALL_DATASETS = GRAYSCALE_DATASETS[:-1] + COLOR_DATASETS
# COLOR_DATASETS = COLOR_DATASETS + CIFAR10_X


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  assert _VISIBLE_DIST.value in [
      'cont_bernoulli', 'categorical', 'gaussian', 'bernoulli']
  if _TEST_NORMALIZE.value == 'same':
    test_normalize = _NORMALIZE.value
  else:
    test_normalize = _TEST_NORMALIZE.value
  print(test_normalize)

  if _DATASET.value in COLOR_DATASETS:
    mode = 'color'
    input_shape = (32, 32, 3)
    datasets = COLOR_DATASETS
  elif _DATASET.value in GRAYSCALE_DATASETS:
    mode = 'grayscale'
    input_shape = (32, 32, 1)
    datasets = GRAYSCALE_DATASETS
  else:
    logging.error('%s dataset not supported', _DATASET.value)
    sys.exit(1)

  suffix = (f'{_DATASET.value.replace("/", "_")}-{_NORMALIZE.value}'
            f'-zdim_{_LATENT_DIM.value}-lr_{_LR.value}-bs_{_BATCH_SIZE.value}'
            f'-nf_{_NUM_FILTERS.value}')

  model_dir = os.path.join(_EXPERIMENT_DIR.value, suffix)
  log_dir = os.path.join(_EXPERIMENT_DIR.value, 'logs', suffix)

  ds_train, ds_val, _ = dataset_utils.get_dataset(
      _DATASET.value,
      _BATCH_SIZE.value,
      mode,
      normalize=_NORMALIZE.value,
      dequantize=False,
      visible_dist=_VISIBLE_DIST.value
  )

  logging.info('%s loaded', _DATASET.value)

  model = network.CVAE(
      input_shape=input_shape,
      num_filters=_NUM_FILTERS.value,
      latent_dim=_LATENT_DIM.value,
      visible_dist=_VISIBLE_DIST.value
  )

  logging.info('%s model initialized', model.__class__.__name__)

  optimizer = tf.optimizers.Adam(learning_rate=_LR.value)
  model.compile(
      optimizer=optimizer,
      loss={'posterior': model.kl_divergence_loss,
            'decoder_ll': model.decoder_nll_loss}
  )

  if _DO_TRAIN.value:
    tf.io.gfile.makedirs(model_dir)
    callbacks = [
        utils.TensorBoardWithLLStats(
            _EVAL_EVERY.value,
            _DATASET.value,
            datasets,
            mode,
            _NORMALIZE.value,
            _VISIBLE_DIST.value,
            log_dir=log_dir,
            update_freq='epoch'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'weights.hdf5'),
            verbose=1, save_weights_only=True, save_best_only=True
            )
    ]

    logging.info('Training...')
    model.fit(
        ds_train,
        epochs=_NUM_EPOCHS.value,
        validation_data=ds_val,
        callbacks=callbacks
    )

  if _DO_EVAL.value:
    assert tf.io.gfile.exists(model_dir), 'Model directory does not exist'
    fname = 'probs.pkl'
    if tf.io.gfile.exists(os.path.join(model_dir, fname)):
      sys.exit(0)
    weights = tf.io.gfile.listdir(model_dir)
    weights.sort()
    logging.info('Loading weights from %s',
                 os.path.join(model_dir, weights[-1]))
    model.build([None]+list(input_shape))

    weights_path = os.path.join(model_dir, weights[-1])
    model.load_weights(weights_path)

    model.compute_corrections(ds_train)
    logging.info('Evaluating...')
    model.evaluate(ds_val)
    probs_res = utils.get_probs(datasets,
                                model,
                                mode,
                                test_normalize,
                                n_samples=5,
                                split='test',
                                training=False,
                                visible_dist=_VISIBLE_DIST.value)
    with tf.io.gfile.GFile(os.path.join(model_dir, fname), 'wb') as f:
      pickle.dump(probs_res, f)

if __name__ == '__main__':
  app.run(main)
