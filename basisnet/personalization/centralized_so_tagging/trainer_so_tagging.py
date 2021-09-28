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

"""Runs centralized training of Stackoverflow tag predictions.

We create tagging dataset with client ids, and learn it as a regression task.
"""
import functools

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from basisnet.personalization.centralized_so_tagging import so_preprocessing
from basisnet.personalization.centralized_so_tagging import so_tagging_model

flags.DEFINE_string('experiment_name', '', 'User provided name for experiment')
flags.DEFINE_integer('num_basis', 4,
                     'number of basis to learn, 1 = original model.')
flags.DEFINE_string(
    'modeldir', '/tmp/basisnet/centralized_so_tagging',
    'Work dir for the experiment.')

FLAGS = flags.FLAGS
np.random.seed(0)


def main(argv):
  tf.compat.v2.enable_v2_behavior()
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  vocab_size = 10000
  tag_size = 500
  batch_size = 128
  shuffle_buffer_size = 1000
  max_element_per_client = 1000
  logging.info('experiment_name')
  logging.info(FLAGS.experiment_name)
  steps_per_epoch = 10 if 'debug' in FLAGS.experiment_name else 500000
  max_validation_client = 10 if 'debug' in FLAGS.experiment_name else 1000

  # Avoid a long line
  client_datasets = tff.simulation.datasets.stackoverflow.load_data()
  (train_clientdata, valid_clientdata, test_clientdata) = client_datasets

  word_vocab = so_preprocessing.create_word_vocab(vocab_size)
  tag_vocab = so_preprocessing.create_tag_vocab(tag_size)

  ids = np.arange(
      len(train_clientdata.client_ids) + len(valid_clientdata.client_ids),
      dtype=np.int64) + 1
  str_ids = train_clientdata.client_ids + valid_clientdata.client_ids
  client_id_encodings = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(str_ids, ids),
      num_oov_buckets=1)

  preprocess_fn = so_preprocessing.create_preprocess_fn(
      word_vocab, tag_vocab, shuffle_buffer_size=1000)

  def to_embedding_id(client_id):
    return client_id_encodings.lookup(client_id)

  def parse_dataset(clientdata):
    def _create_dataset_with_id(client_id):
      client_number_id = to_embedding_id(client_id)
      def add_id(x):
        x['client_id'] = client_number_id
        return x

      # pylint: disable=protected-access
      return clientdata._create_dataset(client_id).take(
          max_element_per_client).map(add_id)

    client_ids = clientdata.client_ids
    nested_dataset = tf.data.Dataset.from_tensor_slices(client_ids)
    centralized_train = nested_dataset.flat_map(_create_dataset_with_id)

    centralized_train = preprocess_fn(centralized_train)
    print(centralized_train.element_spec)
    return centralized_train

  train_dataset = parse_dataset(train_clientdata).shuffle(
      shuffle_buffer_size).batch(batch_size)
  test_dataset = parse_dataset(test_clientdata).batch(batch_size).take(1000)

  model_builder = functools.partial(
      so_tagging_model.create_logistic_basis_model,
      vocab_tokens_size=vocab_size,
      vocab_tags_size=tag_size,
      num_basis=FLAGS.num_basis,
      )

  loss_builder = functools.partial(
      tf.keras.losses.BinaryCrossentropy,
      from_logits=False,
      reduction=tf.keras.losses.Reduction.SUM)

  metrics = [tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(top_k=5, name='recall_at_5')]

  basisnet = model_builder()
  basisnet.summary()
  basisnet.compile(optimizer='adam', loss=loss_builder(), metrics=metrics)

  def _create_split_dataset_with_id(client_id):
    def add_id(x):
      x['client_id'] = to_embedding_id(client_id)
      return x

    # pylint: disable=protected-access
    client_ds = valid_clientdata._create_dataset(client_id).map(add_id)
    total_size = client_ds.reduce(0, lambda x, _: x + 1)

    num_elements_ten_percent = tf.cast((total_size - 1) / 10, dtype=tf.int64)
    num_elements_half = tf.cast((total_size - 1) / 2, dtype=tf.int64)

    train_set = client_ds.take(num_elements_half)
    train_split_set = client_ds.take(num_elements_ten_percent)
    test_set = client_ds.skip(num_elements_half)

    train_dataset = preprocess_fn(train_set).batch(batch_size)
    train_split_dataset = preprocess_fn(train_split_set).batch(batch_size)
    test_dataset = preprocess_fn(test_set).batch(batch_size)

    return train_dataset, train_split_dataset, test_dataset

  sample_client_ids = np.random.choice(valid_clientdata.client_ids,
                                       max_validation_client)

  def per_evaluation(basisnet, model_builder, fix_basis=True):
    all_clients_acc_before = []
    all_clients_acc = []
    all_clients_split_acc = []

    for clnt_id in sample_client_ids[:100]:
      local_basisnet = model_builder()
      local_basisnet.compile(
          optimizer='adam', loss=loss_builder(), metrics=metrics)
      local_basisnet.set_weights(basisnet.get_weights())

      if fix_basis:
        # only fine-tune the embedding
        logging.info('Fix basis')
        for layer in local_basisnet.layers:
          if layer.name != 'client_embedding':
            layer.trainable = False

        tf_clnt_id = tf.constant(clnt_id)
        # Avoid a long line
        datasets = _create_split_dataset_with_id(tf_clnt_id)
        train_dataset, train_split_dataset, test_dataset = datasets

      bf_acc = local_basisnet.evaluate(test_dataset)[-1]
      all_clients_acc_before.append(bf_acc)
      logging.info(bf_acc)

      local_basisnet.fit(
          train_dataset, epochs=10, verbose=0)
      all_clients_acc.append(local_basisnet.evaluate(test_dataset)[-1])

      local_basisnet.set_weights(basisnet.get_weights())
      local_basisnet.fit(
          train_split_dataset, epochs=10, verbose=0)
      # Fine-tune with a smaller split of the training data. Here is 20%.
      all_clients_split_acc.append(local_basisnet.evaluate(test_dataset)[-1])

    logging.info(all_clients_acc)
    logging.info(np.mean(all_clients_acc))

    logging.info(all_clients_split_acc)
    logging.info(np.mean(all_clients_split_acc))

    logging.info(all_clients_split_acc)
    logging.info(np.mean(all_clients_split_acc))

  for ep in range(2):
    basisnet.fit(
        train_dataset,
        epochs=1,
        validation_data=test_dataset,
        steps_per_epoch=steps_per_epoch
    )
    basisnet.save_weights(
        FLAGS.modeldir+'/so_tagging%s_basis_%d_ep%d.ckpt' %
        (FLAGS.experiment_name, FLAGS.num_basis, ep))
    per_evaluation(basisnet, model_builder, fix_basis=True)

if __name__ == '__main__':
  app.run(main)
