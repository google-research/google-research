# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Model trainer."""

from collections.abc import Sequence
import functools
import json

from absl import app
from absl import flags
from absl import logging
from dataset_generator import bag_dataset_generator
from dataset_generator import generate_dataset
from losses import dllp_loss
from losses import dllp_loss_graph
from losses import dllp_loss_graph_regression
from losses import easy_llp_loss
from losses import erot_loss
from losses import genbags_loss
from losses import genbags_loss_regression
from losses import ot_llp_loss
from losses import sim_llp_loss
from losses import sim_llp_loss_regression
from network import CustomModel
from network import CustomModelRegression
from network import MeanMapModel
from network import my_model
import tensorflow as tf
import train_constants


tfk = tf.keras
tfkl = tf.keras.layers

_WHICH_METHOD = flags.DEFINE_string('method', None, 'which method to use?')
_WHICH_OPTIMIZER = flags.DEFINE_string(
    'optimizer', 'adam', 'which optimizer to use?'
)
# _WHICH_MODEL = flags.DEFINE_string('model', None, 'which model to use?')
# _WHICH_LOSS = flags.DEFINE_string('loss', None, 'which loss to use?')
_C1 = flags.DEFINE_integer('c1', -1, 'c1?')
_C2 = flags.DEFINE_integer('c2', -1, 'c2?')
_SPLIT = flags.DEFINE_integer('split', -1, 'split?')

_EPOCHS = flags.DEFINE_integer('epochs', 50, 'epochs?')
_WARM_START_EPOCHS = flags.DEFINE_integer(
    'warm_start_epochs', 50, 'warm_start_epochs?'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 8, 'batch_size?')
_VALIDATION_BATCH_SIZE = flags.DEFINE_integer(
    'validation_batch_size', 1024, 'validation_batch_size?'
)
_BLOCK_SIZE = flags.DEFINE_integer('block_size', 4, 'block_size?')
_NUM_GEN_BAGS_PER_BLOCK = flags.DEFINE_integer(
    'num_gen_bags_per_block', 60, 'num_gen_bags_per_block?'
)
_LAMBDA = flags.DEFINE_float('lambda', 2.0, 'lambda?')
_SIM_LOSS_SIZE = flags.DEFINE_integer('sim_loss_size', 400, 'sim_loss_size?')
_REG = flags.DEFINE_float('reg', 0.2, 'reg?')
_RANDOM_BAGS = flags.DEFINE_bool('random_bags', False, 'random_bags?')
_BAG_SIZE = flags.DEFINE_integer(
    'bag_size', 64, 'bag_size?'
)  # specified only for random bags or feature random bags
_FEATURE_RANDOM_BAGS = flags.DEFINE_bool(
    'feature_random_bags', False, 'feature_random_bags?'
)
_WHICH_DATASET = flags.DEFINE_enum(
    'which_dataset',
    'criteo_ctr',
    ['criteo_ctr', 'criteo_sscl'],
    'Which dataset to preprocess.',
)


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Program Started')
  if _WHICH_DATASET.value == 'criteo_ctr':
    train_ds, x_test, y_test, multihot_dim, p = generate_dataset(
        _RANDOM_BAGS.value,
        _FEATURE_RANDOM_BAGS.value,
        _BAG_SIZE.value,
        'C' + str(_C1.value),
        'C' + str(_C2.value),
        str(_SPLIT.value),
        _BATCH_SIZE.value,
    )
    logging.info('Datasets Loaded')
    model = my_model(multihot_dim)
    optimizer = {
        'adam': tfk.optimizers.Adam(learning_rate=1e-5),
        'adam_legacy': tfk.optimizers.legacy.Adam(learning_rate=1e-5),
    }[_WHICH_OPTIMIZER.value]
    genbags_mean = tf.zeros(_BLOCK_SIZE.value, dtype=tf.float32)
    genbags_variance = 1.331 * tf.eye(
        _BLOCK_SIZE.value, dtype=tf.float32
    ) - 0.33 * tf.ones(
        shape=(_BLOCK_SIZE.value, _BLOCK_SIZE.value), dtype=tf.float32
    )
    loss = {
        'dllp_bce': functools.partial(
            dllp_loss,
            _BATCH_SIZE.value,
            tfk.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM
            ),
        ),
        'dllp_mse': functools.partial(
            dllp_loss,
            _BATCH_SIZE.value,
            tfk.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM
            ),
        ),
        'genbags': functools.partial(
            genbags_loss,
            genbags_mean,
            genbags_variance,
            _BATCH_SIZE.value,
            _BLOCK_SIZE.value,
            _NUM_GEN_BAGS_PER_BLOCK.value,
        ),
        'easy_llp': functools.partial(
            easy_llp_loss,
            _BATCH_SIZE.value,
            p,
            tfk.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM
            ),
        ),
        'ot_llp': functools.partial(
            ot_llp_loss,
            _BATCH_SIZE.value,
            tfk.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM
            ),
        ),
        'sim_llp': functools.partial(
            dllp_loss_graph,
            _BATCH_SIZE.value,
            tfk.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM
            ),
        ),
        'soft_erot_llp': functools.partial(
            erot_loss,
            False,
            _REG.value,
            _BATCH_SIZE.value,
            tfk.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM
            ),
        ),
        'hard_erot_llp': functools.partial(
            erot_loss,
            True,
            _REG.value,
            _BATCH_SIZE.value,
            tfk.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM
            ),
        ),
        'mean_map': None,
    }[_WHICH_METHOD.value]
    logging.info('Training Started')
    if _WHICH_METHOD.value == 'mean_map':
      del model
      if _RANDOM_BAGS.value:
        mean_map_file = (
            '../results/mean_map_vectors/rand_full_vector_'
            + str(_SPLIT.value)
            + '_'
            + 'random_'
            + str(_BAG_SIZE.value)
            + '.json'
        )
      elif _FEATURE_RANDOM_BAGS.value:
        mean_map_file = (
            '../results/mean_map_vectors/feat_rand_full_vector_'
            + str(_SPLIT.value)
            + '_'
            + str(_BAG_SIZE.value)
            + '_'
            + 'C'
            + str(_C1.value)
            + '_'
            + 'C'
            + str(_C2.value)
            + '.json'
        )
      else:
        mean_map_dir = '../results/mean_map_vectors/full_vector_'
        mean_map_file = (
            mean_map_dir
            + str(_SPLIT.value)
            + '_C'
            + str(_C1.value)
            + '_C'
            + str(_C2.value)
            + '.json'
        )
      with open(mean_map_file, 'r') as fp:
        mu_xy = json.load(fp)
      mu_xy = tf.reshape(
          tf.convert_to_tensor(mu_xy, dtype=tf.float32), shape=(1, -1)
      )
      model = MeanMapModel(
          multihot_dim, _REG.value * _BATCH_SIZE.value * 400.0, mu_xy
      )
      model.compile(
          optimizer=optimizer,
          metrics=[tf.keras.metrics.AUC(name='auc')],
      )
      earlystopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_auc', patience=3, mode='max', restore_best_weights=True
      )
      hist = model.fit(
          train_ds,
          epochs=_EPOCHS.value,
          validation_data=(x_test, y_test),
          validation_batch_size=_VALIDATION_BATCH_SIZE.value,
          callbacks=[earlystopping_callback],
      )
      best_auc = max(hist.history['val_auc'])
      best_auc_warm_start = '-'
    elif (
        _WHICH_METHOD.value == 'hard_erot_llp'
        or _WHICH_METHOD.value == 'soft_erot_llp'
    ):
      del model
      model = CustomModel(multihot_dim)
      warm_start_loss = functools.partial(
          dllp_loss_graph,
          _BATCH_SIZE.value,
          tfk.losses.BinaryCrossentropy(
              reduction=tf.keras.losses.Reduction.SUM
          ),
      )
      model.compile(
          optimizer=optimizer,
          bag_loss=warm_start_loss,
          metrics=[tf.keras.metrics.AUC(name='auc')],
      )
      earlystopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_auc', patience=3, mode='max', restore_best_weights=True
      )
      hist = model.fit(
          train_ds,
          epochs=_EPOCHS.value,
          validation_data=(x_test, y_test),
          validation_batch_size=_VALIDATION_BATCH_SIZE.value,
          callbacks=[earlystopping_callback],
      )
      best_auc_warm_start = max(hist.history['val_auc'])
      model.compile(
          optimizer=optimizer,
          bag_loss=loss,
          metrics=[tf.keras.metrics.AUC(name='auc')],
      )
      hist = model.fit(
          train_ds,
          epochs=_EPOCHS.value,
          initial_epoch=len(hist.history['val_auc']) + 1,
          validation_data=(x_test, y_test),
          validation_batch_size=_VALIDATION_BATCH_SIZE.value,
          callbacks=[earlystopping_callback],
      )
      best_auc = max(hist.history['val_auc'])
    elif _WHICH_METHOD.value == 'ot_llp':
      warm_start_loss = functools.partial(
          dllp_loss,
          _BATCH_SIZE.value,
          tfk.losses.BinaryCrossentropy(
              reduction=tf.keras.losses.Reduction.SUM
          ),
      )
      model.compile(
          optimizer=optimizer,
          loss=warm_start_loss,
          metrics=[tf.keras.metrics.AUC(name='auc')],
      )
      earlystopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_auc', patience=3, mode='max', restore_best_weights=True
      )
      hist = model.fit(
          train_ds,
          epochs=_EPOCHS.value,
          validation_data=(x_test, y_test),
          validation_batch_size=_VALIDATION_BATCH_SIZE.value,
          callbacks=[earlystopping_callback],
      )
      best_auc_warm_start = max(hist.history['val_auc'])
      model.compile(
          optimizer=optimizer,
          loss=loss,
          metrics=[tf.keras.metrics.AUC(name='auc')],
      )
      hist = model.fit(
          train_ds,
          epochs=_EPOCHS.value,
          initial_epoch=len(hist.history['val_auc']) + 1,
          validation_data=(x_test, y_test),
          validation_batch_size=_VALIDATION_BATCH_SIZE.value,
          callbacks=[earlystopping_callback],
      )
      best_auc = max(hist.history['val_auc'])
    elif _WHICH_METHOD.value == 'sim_llp':
      del model
      model1 = CustomModel(multihot_dim)
      model2 = CustomModel(multihot_dim)
      model1.compile(
          optimizer=optimizer,
          bag_loss=loss,
          metrics=[tf.keras.metrics.AUC(name='warm_start_auc')],
          # run_eagerly=True,
      )
      earlystopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_warm_start_auc',
          patience=3,
          mode='max',
          restore_best_weights=True,
      )
      hist = model1.fit(
          train_ds,
          epochs=_WARM_START_EPOCHS.value,
          validation_data=(x_test, y_test),
          validation_batch_size=_VALIDATION_BATCH_SIZE.value,
          callbacks=[earlystopping_callback],
      )
      best_auc_warm_start = max(hist.history['val_warm_start_auc'])
      rep = model1.get_rep()
      new_loss = functools.partial(
          sim_llp_loss,
          _SIM_LOSS_SIZE.value,
          loss,
          _LAMBDA.value,
          rep,
      )
      new_optimizer = tfk.optimizers.Adam(learning_rate=1e-5)
      model2.compile(
          optimizer=new_optimizer,
          bag_loss=new_loss,
          metrics=[tf.keras.metrics.AUC(name='auc')],
          # run_eagerly=True,
      )
      earlystopping_callback_1 = tf.keras.callbacks.EarlyStopping(
          monitor='val_auc',
          patience=3,
          mode='max',
          restore_best_weights=True,
      )
      hist = model2.fit(
          train_ds,
          epochs=_EPOCHS.value,
          validation_data=(x_test, y_test),
          validation_batch_size=_VALIDATION_BATCH_SIZE.value,
          callbacks=[earlystopping_callback_1],
      )
      best_auc = max(hist.history['val_auc'])
    else:
      model.compile(
          optimizer=optimizer,
          loss=loss,
          metrics=[tf.keras.metrics.AUC(name='auc')],
      )
      earlystopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_auc', patience=3, mode='max', restore_best_weights=True
      )
      hist = model.fit(
          train_ds,
          epochs=_EPOCHS.value,
          validation_data=(x_test, y_test),
          validation_batch_size=_VALIDATION_BATCH_SIZE.value,
          callbacks=[earlystopping_callback],
      )
      best_auc = max(hist.history['val_auc'])
      best_auc_warm_start = '-'
    logging.info('Training Complete')
    if _RANDOM_BAGS.value:
      result_dict = {
          'random': True,
          'split': _SPLIT.value,
          'bag_size': _BAG_SIZE.value,
          'method': _WHICH_METHOD.value,
          'auc': best_auc,
          'auc_warm_start': best_auc_warm_start,
      }
    elif _FEATURE_RANDOM_BAGS.value:
      result_dict = {
          'feature_random': True,
          'split': _SPLIT.value,
          'bag_size': _BAG_SIZE.value,
          'c1': train_constants.C + str(_C1.value),
          'c2': train_constants.C + str(_C2.value),
          'method': _WHICH_METHOD.value,
          'auc': best_auc,
          'auc_warm_start': best_auc_warm_start,
      }
    else:
      result_dict = {
          'c1': train_constants.C + str(_C1.value),
          'c2': train_constants.C + str(_C2.value),
          'split': _SPLIT.value,
          'method': _WHICH_METHOD.value,
          'auc': best_auc,
          'auc_warm_start': best_auc_warm_start,
      }
    if _RANDOM_BAGS.value:
      saving_dir = (
          '../results/training_dicts/random_bags_ds/'
          + _WHICH_METHOD.value
          + '/'
      )
      saving_file = (
          saving_dir
          + str(_SPLIT.value)
          + '_bs-'
          + str(_BAG_SIZE.value)
          + '.json'
      )
    elif _FEATURE_RANDOM_BAGS.value:
      saving_dir = (
          '../results/training_dicts/fixed_size_feature_bags_ds/'
          + _WHICH_METHOD.value
          + '/'
      )
      saving_file = (
          saving_dir
          + str(_SPLIT.value)
          + '_bs-'
          + str(_BAG_SIZE.value)
          + '_C'
          + str(_C1.value)
          + '_C'
          + str(_C2.value)
          + '.json'
      )
    else:
      saving_dir = (
          '../results/training_dicts/feature_bags_ds/'
          + _WHICH_METHOD.value
          + '/'
      )
      saving_file = (
          saving_dir
          + str(_SPLIT.value)
          + '_C'
          + str(_C1.value)
          + '_C'
          + str(_C2.value)
          + '.json'
      )
    with open(saving_file, 'w') as fp:
      json.dump(result_dict, fp)
    logging.info('Results Saved')
  else:
    c1 = train_constants.C + str(_C1.value)
    c2 = train_constants.C + str(_C2.value)
    split = str(_SPLIT.value)
    train_ds, test_ds = bag_dataset_generator(
        c1,
        c2,
        split,
        _RANDOM_BAGS.value,
        _BAG_SIZE.value,
        _BATCH_SIZE.value,
        _VALIDATION_BATCH_SIZE.value,
    )
    logging.info('Datasets Created')
    vocab_sizes = train_constants.SSCL_VOCAB_SIZES
    if _WHICH_METHOD.value == 'sim_llp':
      rep_model = CustomModelRegression(
          n_catg=17,
          embed_size=train_constants.EMBED_SIZE,
          vocab_sizes=vocab_sizes,
      )
      rep_opt = tfk.optimizers.Adam(learning_rate=1e-4)
      rep_loss = functools.partial(
          dllp_loss_graph_regression,
          _BATCH_SIZE.value,
          tfk.losses.MeanSquaredError(
              reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
          ),
      )
      rep_model.compile(
          optimizer=rep_opt,
          bag_loss=rep_loss,
          metrics=[tfk.metrics.MeanSquaredError(name='mse')],
      )
      rep_earlystopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_mse', patience=5, mode='min', restore_best_weights=True
      )
      rep_model.fit(
          train_ds,
          validation_data=test_ds,
          epochs=_EPOCHS.value,
          callbacks=[rep_earlystopping_callback],
      )
    else:
      rep_model = None
    model = CustomModelRegression(
        n_catg=17,
        embed_size=train_constants.EMBED_SIZE,
        vocab_sizes=vocab_sizes,
    )
    logging.info('Model Created')
    opt = tfk.optimizers.Adam(learning_rate=1e-4)
    genbags_mean = tf.zeros(_BLOCK_SIZE.value, dtype=tf.float32)
    genbags_variance = 1.331 * tf.eye(
        _BLOCK_SIZE.value, dtype=tf.float32
    ) - 0.33 * tf.ones(
        shape=(_BLOCK_SIZE.value, _BLOCK_SIZE.value), dtype=tf.float32
    )
    loss = {
        'dllp_mse': functools.partial(
            dllp_loss_graph_regression,
            _BATCH_SIZE.value,
            tfk.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            ),
        ),
        'dllp_mae': functools.partial(
            dllp_loss_graph_regression,
            _BATCH_SIZE.value,
            tfk.losses.MeanAbsoluteError(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            ),
        ),
        'genbags': functools.partial(
            genbags_loss_regression,
            genbags_mean,
            genbags_variance,
            _BATCH_SIZE.value,
            _BLOCK_SIZE.value,
            _NUM_GEN_BAGS_PER_BLOCK.value,
        ),
        'sim_llp': functools.partial(
            sim_llp_loss_regression,
            _SIM_LOSS_SIZE.value,
            functools.partial(
                dllp_loss_graph,
                _BATCH_SIZE.value,
                tfk.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                ),
            ),
            _LAMBDA.value,
            rep_model,
        ),
    }[_WHICH_METHOD.value]
    model.compile(
        optimizer=opt,
        bag_loss=loss,
        metrics=[tfk.metrics.MeanSquaredError(name='mse')],
    )
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mse', patience=5, mode='min', restore_best_weights=True
    )
    hist = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=_EPOCHS.value,
        callbacks=[earlystopping_callback],
    )
    logging.info(
        'Model Training Complete, min val mse: %d', min(hist.history['val_mse'])
    )
    result_dict = {
        'loss': _WHICH_METHOD.value,
        'c1': c1,
        'c2': c2,
        'split': _SPLIT.value,
        'batch_size': _BATCH_SIZE.value,
        'validation_batch_size': _VALIDATION_BATCH_SIZE.value,
        'epochs': _EPOCHS.value,
        'best_mse': min(hist.history['val_mse']),
        'random_bags': _RANDOM_BAGS.value,
        'bag_size': _BAG_SIZE.value,
    }
    if _RANDOM_BAGS.value:
      saving_dir = (
          '../results/training_dicts/random_bags_ds/'
          + _WHICH_METHOD.value
          + '_sscl/'
      )
      saving_file = (
          saving_dir
          + str(_SPLIT.value)
          + '_bs-'
          + str(_BAG_SIZE.value)
          + '.json'
      )
    else:
      saving_dir = (
          '../results/training_dicts/feature_bags_ds/'
          + _WHICH_METHOD.value
          + '_sscl/'
      )
      saving_file = (
          saving_dir
          + str(_SPLIT.value)
          + '_C'
          + str(_C1.value)
          + '_C'
          + str(_C2.value)
          + '.json'
      )
    with open(saving_file, 'w') as fp:
      json.dump(result_dict, fp)
    logging.info('Results Saved')
  logging.info('Program Finished')


if __name__ == '__main__':
  app.run(main)
