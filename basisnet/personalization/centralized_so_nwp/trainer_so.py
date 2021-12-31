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

"""Runs Stackoverflow next word prediction.

It first learn a global model on the training clients, then adapt to each of
the test client. It evaluates the personalized accuracy for each client on the
individual test sets.
"""
import functools

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from basisnet.personalization.centralized_so_nwp import so_nwp_eval
from basisnet.personalization.centralized_so_nwp import so_nwp_preprocessing
from basisnet.personalization.centralized_so_nwp import stackoverflow_basis_models


# Stack Overflow NWP flags
flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                     'Number of out of vocabulary buckets.')
flags.DEFINE_integer('so_nwp_sequence_length', 20,
                     'Max sequence length to use.')
flags.DEFINE_integer('so_nwp_max_elements_per_user', 1000, 'Max number of '
                     'training sentences to use per user.')

flags.DEFINE_string('modeldir', '/tmp/basisnet/centralized_so_nwp',
                    'The dir for saving checkpoints and logs.')

flags.DEFINE_integer(
    'fine_tune_epoch', 20, 'number of epochs for fine-tuning'
    'to use from test set for per-round validation.')
flags.DEFINE_integer('max_num_ft_clients', 1000,
                     'number of clients fot personalized evaluation.')

flags.DEFINE_integer('num_basis', 1,
                     'number of basis to learn, 1 = original model.')
flags.DEFINE_integer(
    'num_lstm_units', -1,
    'number of LSTM hidden size, -1 to use default value for each task.')
flags.DEFINE_string('experiment_name', '',
                    'Experiment name string')
flags.DEFINE_integer('fine_tune_batch_size', 20,
                     'Batch size for fine-tuning.')

FLAGS = flags.FLAGS


def main(argv):
  tf.compat.v2.enable_v2_behavior()
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  if 'debug' in FLAGS.experiment_name:
    total_iterations = 100
  else:
    total_iterations = 400000

  # Avoid a long line
  clientdata = tff.simulation.datasets.stackoverflow.load_data()
  (train_clientdata, valid_clientdata, _) = clientdata

  vocab = so_nwp_preprocessing.create_vocab(FLAGS.so_nwp_vocab_size)
  sample_client_ids = np.random.choice(valid_clientdata.client_ids,
                                       FLAGS.max_num_ft_clients)

  # id = 0 for global embedding
  ids = np.arange(
      len(train_clientdata.client_ids) + len(valid_clientdata.client_ids),
      dtype=np.int64) + 1
  str_ids = train_clientdata.client_ids + valid_clientdata.client_ids
  client_id_encodings = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(str_ids, ids),
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets)

  def to_embedding_id(client_id):
    return client_id_encodings.lookup(client_id)

  preprocess_fn = so_nwp_preprocessing.build_preprocess_fn(
      vocab,
      so_nwp_sequence_length=FLAGS.so_nwp_sequence_length,
      so_nwp_num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
      debug='debug' in FLAGS.experiment_name)

  train_dataset, val_dataset = so_nwp_preprocessing.create_centralized_datasets(
      preprocess_fn,
      to_embedding_id,
      sample_client_ids)

  special_tokens = so_nwp_preprocessing.get_special_tokens(
      vocab_size=FLAGS.so_nwp_vocab_size,
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets)

  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos
  mask_vocab_id = [pad_token, eos_token] + oov_tokens

  # Create train set, split train set, test set for personalization.
  # Create train/test sets by dates.
  per_tuples_by_date = so_nwp_preprocessing.build_split_centralized_dataset(
      valid_clientdata,
      preprocess_fn,
      to_embedding_id,
      sample_client_ids,
      split_by='date')
  # Create train/test sets randomly.
  per_tuples_random = so_nwp_preprocessing.build_split_centralized_dataset(
      valid_clientdata,
      preprocess_fn,
      to_embedding_id,
      sample_client_ids,
      split_by='random')

  stackoverflow_models_fn = stackoverflow_basis_models.create_basis_recurrent_model
  model_builder = functools.partial(
      stackoverflow_models_fn,
      vocab_size=FLAGS.so_nwp_vocab_size,
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
      num_basis=FLAGS.num_basis)

  # Compile
  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  metrics = [
      so_nwp_eval.SubsetInVocabAccuracy(
          name='mask_accuracy',
          non_vocabulary_classes=[pad_token, eos_token] + oov_tokens,
          masked_classes=[pad_token, eos_token])
  ]

  basisnet = model_builder()
  basisnet.summary()
  basisnet.compile(
      loss=loss_builder(), optimizer='adam', metrics=metrics)

  history = basisnet.fit(
      train_dataset, epochs=1, validation_data=val_dataset, verbose=1,
      steps_per_epoch=total_iterations,
      workers=16,
      use_multiprocessing=True)
  logging.info(history)

  if 'debug' not in FLAGS.experiment_name:
    basisnet.save_weights(
        FLAGS.modeldir+'/so_%s_basis_%d.ckpt' %
        (FLAGS.experiment_name, FLAGS.num_basis))

  # model_builder for the global embedding
  global_model_builder = functools.partial(
      stackoverflow_models_fn,
      vocab_size=FLAGS.so_nwp_vocab_size,
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
      num_basis=FLAGS.num_basis,
      global_embedding_only=True)

  # Personalization
  def online_evaluation(fix_basis=True):
    def _create_full_dataset_with_id(client_id):
      def add_id(x):
        x['client_id'] = 0
        return x

      # pylint: disable=protected-access
      client_ds = so_nwp_preprocessing.sort_by_date_pipe(
          valid_clientdata._create_dataset(client_id)).map(add_id)
      return client_ds

    all_clients_acc_before = []
    all_clients_acc = []
    for clnt_id in sample_client_ids:
      local_basisnet = model_builder()
      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

      local_basisnet.compile(
          optimizer=optimizer,
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=metrics)
      local_basisnet.set_weights(basisnet.get_weights())

      if fix_basis:
        # only fine-tune the embedding
        logging.info('Fix basis')
        for layer in local_basisnet.layers:
          if layer.name != 'client_embedding':
            layer.trainable = False

      ds = _create_full_dataset_with_id(clnt_id)
      ds = preprocess_fn(ds).unbatch().batch(FLAGS.fine_tune_batch_size)

      all_clients_acc_before.append(local_basisnet.evaluate(ds)[1])

      num_batches = so_nwp_preprocessing.count_batches(ds)

      all_val_acc = []
      for idx in range(1, num_batches):
        train_data_time, test_data_time = so_nwp_preprocessing.split_time(
            ds, idx)
        history = local_basisnet.fit(
            train_data_time,
            epochs=1,
            validation_data=test_data_time,
            verbose=0)
        all_val_acc.append(history.history['val_mask_accuracy'])
      all_clients_acc.append(np.mean(all_val_acc))

    logging.info(all_clients_acc_before)
    logging.info(np.mean(all_clients_acc_before))

    logging.info(all_clients_acc)
    logging.info(np.mean(all_clients_acc))

  logging.info('=====Start evaluation split by dates=====')
  so_nwp_eval.per_evaluation(
      basisnet,
      per_tuples_by_date,
      global_model_builder,
      model_builder,
      mask_vocab_id,
      fix_basis=FLAGS.num_basis > 1)
  logging.info('=====Start evaluation split by random=====')
  so_nwp_eval.per_evaluation(
      basisnet,
      per_tuples_random,
      global_model_builder,
      model_builder,
      mask_vocab_id,
      fix_basis=FLAGS.num_basis > 1)
  logging.info('=====Start online evaluation=====')
  online_evaluation(fix_basis=FLAGS.num_basis > 1)

if __name__ == '__main__':
  app.run(main)
