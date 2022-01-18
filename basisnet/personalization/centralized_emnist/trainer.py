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

"""Runs centralied training and personalization on EMNIST."""
import collections

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from basisnet.personalization.centralized_emnist import data_processing
from basisnet.personalization.centralized_emnist import emnist_models
from basisnet.personalization.centralized_emnist import training_specs

# Training hyperparameters
flags.DEFINE_integer('client_datasets_random_seed', 1,
                     'Random seed for client sampling.')
flags.DEFINE_float('client_learning_rate', 1e-3,
                   'learning rate for client training.')
# Training loop configuration
flags.DEFINE_string(
    'experiment_name', 'test',
    'The name of this experiment. Will be append to '
    '--root_output_dir to separate experiment results.')
flags.mark_flag_as_required('experiment_name')
flags.DEFINE_string('root_output_dir', '/tmp/basisnet/centralized_emnist',
                    'Root directory for writing experiment output.')
flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
flags.DEFINE_integer(
    'rounds_per_eval', 100,
    'How often to evaluate the global model on the validation dataset.')
flags.DEFINE_integer('rounds_per_checkpoint', 100,
                     'How often to checkpoint the global model.')

flags.DEFINE_string('modeldir', '', 'The dir for saving checkpoints and logs.')
flags.DEFINE_bool('debug', False, 'If true, reduce batch size and do not use'
                  'tf_function.')


# For personalization
flags.DEFINE_integer(
    'fine_tune_epoch', 20, 'number of epochs for fine-tuning'
    'to use from test set for per-round validation.')

flags.DEFINE_integer('num_basis', 4,
                     'number of basis to learn, 1 = original model.')

flags.DEFINE_float(
    'num_filters_expand', 1,
    'number of expanding Conv channel size.')

flags.DEFINE_float(
    'temp', 1.0, 'temperature for softmax of generating the client embedding.')

_SUPPORTED_EMBEDDING_TYPE = ['lookup']

flags.DEFINE_enum('embedding_type', 'lookup', _SUPPORTED_EMBEDDING_TYPE,
                  'The type of the client embedding.')

flags.DEFINE_boolean('run_sweep', False, 'Whether to'
                     ' run hyper parameter tunning with sweep.')

flags.DEFINE_boolean('digit_only', False, 'digit_only for emnist')
flags.DEFINE_boolean('global_embedding', False,
                     'train with global_embedding only')
flags.DEFINE_boolean('with_dist', False, 'use label distribution as the inputs')

FLAGS = flags.FLAGS


def main(argv):
  tf.compat.v2.enable_v2_behavior()
  # necessary to enable hyperparameter explorations.
  # xm.setup_work_unit()

  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=FLAGS.digit_only)

  if 'test' in FLAGS.experiment_name:
    logging.info('Test run ...')
    num_client = 20
    num_test_client = 20
    epochs = 1
  else:
    num_client = 2500
    num_test_client = 900
    epochs = 40

  train_batch_size = 256

  cliend_encodings = {}
  for i, idx in enumerate(emnist_train.client_ids):
    cliend_encodings[idx] = i

  all_client_ids = np.array(emnist_train.client_ids)
  np.random.shuffle(all_client_ids)

  train_client_ids = all_client_ids[:num_client]
  test_client_ids = all_client_ids[num_client:num_client + num_test_client]

  train_tuple, _, test_tuple = data_processing.parse_data(
      emnist_train,
      emnist_test,
      train_client_ids,
      cliend_encodings,
      with_dist=FLAGS.with_dist)
  ft_train_tuple, ft_sp_train_tuple, ft_test_tuple = data_processing.parse_data(
      emnist_train,
      emnist_test,
      test_client_ids,
      cliend_encodings,
      with_dist=FLAGS.with_dist)

  dataset = data_processing.pack_dataset(
      train_tuple, mode='train', with_dist=FLAGS.with_dist)
  val_dataset = data_processing.pack_dataset(
      test_tuple, mode='test', with_dist=FLAGS.with_dist)

  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  task_spec = training_specs.TaskSpec(
      fine_tune_epoch=FLAGS.fine_tune_epoch,
      num_basis=FLAGS.num_basis,
      num_filters_expand=FLAGS.num_filters_expand,
      temp=FLAGS.temp,
      embedding_type=FLAGS.embedding_type)

  model_builder = emnist_models.get_model_builder(
      task_spec,
      only_digits=FLAGS.digit_only,
      batch_size=train_batch_size,
      with_dist=FLAGS.with_dist,
      global_embedding_only=FLAGS.global_embedding)

  basisnet = model_builder()
  basisnet.summary()

  learning_rate = FLAGS.client_learning_rate
  logging.info(learning_rate)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  basisnet.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
      metrics=['accuracy'])

  basisnet.fit(
      dataset, epochs=epochs, validation_data=val_dataset, verbose=2)
  results = basisnet.evaluate(val_dataset)
  acc = results[1]
  logging.info(acc)

  checkpoint_path = FLAGS.modeldir + 'emnist_basis_%d_lr%f_%s.ckpt' % (
      FLAGS.num_basis, FLAGS.client_learning_rate, FLAGS.experiment_name)
  basisnet.save_weights(checkpoint_path)

  # Personalization
  per_batch_size = 20

  def eval_per_acc(preds, dataset):
    pred_cls = np.argmax(preds, -1)
    dataset = dataset.unbatch()

    per_acc_dict = collections.OrderedDict()
    for y_hat, (x, y)in zip(pred_cls, dataset):
      clnt_id = str(x['input_id'])
      if clnt_id not in per_acc_dict:
        per_acc_dict[clnt_id] = {'cnt': 0, 'correct': 0}
      per_acc_dict[clnt_id]['cnt'] += 1
      per_acc_dict[clnt_id]['correct'] += int(y_hat == y.numpy())

    per_acc_list = [d['correct'] / d['cnt'] for d in per_acc_dict.values()]
    return per_acc_list

  def finetuning(mode,
                 ft_dataset,
                 ft_dataset_test,
                 train_size=1,
                 fix_basis=True,
                 global_exp=False):
    logging.info('==============')
    logging.info(mode)
    logging.info(train_size)
    logging.info('Bases fixed' if fix_basis else 'Bases not fixed')
    logging.info(
        'Global experiment' if global_exp else 'Personalized experiment')
    logging.info('==============')

    per_model_builder = emnist_models.get_model_builder(
        task_spec,
        only_digits=FLAGS.digit_only,
        batch_size=per_batch_size,
        with_dist=FLAGS.with_dist,
        global_embedding_only=global_exp)

    local_basisnet = per_model_builder()
    local_basisnet.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    local_basisnet.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])
    local_basisnet.set_weights(basisnet.get_weights())

    if fix_basis:
      if FLAGS.global_embedding or FLAGS.num_basis == 1:
        # local fine-tune the whole network
        pass
      else:
        # only fine-tune the embedding
        logging.info('Fix basis')
        for layer in local_basisnet.layers:
          if layer.name != 'embedding':
            layer.trainable = False

    preds = local_basisnet.predict(ft_dataset_test)
    per_acc_list = eval_per_acc(preds, ft_dataset_test)
    logging.info('Before fine-tuning')
    logging.info(np.nanmean(per_acc_list))
    logging.info(per_acc_list)

    for ep in range(FLAGS.fine_tune_epoch):
      local_basisnet.fit(
          ft_dataset, epochs=1, verbose=0, validation_data=ft_dataset_test)
      preds = local_basisnet.predict(ft_dataset_test)
      post_acc_list = eval_per_acc(preds, ft_dataset_test)
      logging.info('Fine-tune epoch%d', ep)
      logging.info(np.nanmean(post_acc_list))
      logging.info(post_acc_list)

    return local_basisnet

  ft_dataset = data_processing.pack_dataset(
      ft_train_tuple,
      mode='train',
      batch_size=per_batch_size,
      with_dist=FLAGS.with_dist)
  sp_ft_dataset = data_processing.pack_dataset(
      ft_sp_train_tuple,
      mode='train',
      batch_size=per_batch_size,
      with_dist=FLAGS.with_dist)
  ft_val_dataset = data_processing.pack_dataset(
      ft_test_tuple,
      mode='test',
      batch_size=per_batch_size,
      with_dist=FLAGS.with_dist)

  # Not fix bases
  finetuning(
      mode='test',
      ft_dataset=ft_dataset,
      ft_dataset_test=ft_val_dataset,
      fix_basis=False)
  finetuning(
      mode='test',
      ft_dataset=sp_ft_dataset,
      ft_dataset_test=ft_val_dataset,
      fix_basis=False,
      train_size=0.1)

  if FLAGS.num_basis == 1:
    return

  # Fix bases
  finetuning(mode='test', ft_dataset=ft_dataset, ft_dataset_test=ft_val_dataset)
  finetuning(
      mode='test',
      ft_dataset=sp_ft_dataset,
      ft_dataset_test=ft_val_dataset,
      train_size=0.1)

  # Global Acc
  local_basisnet = finetuning(
      mode='test',
      ft_dataset=ft_dataset,
      ft_dataset_test=ft_val_dataset,
      global_exp=True)
  finetuning(
      mode='test',
      ft_dataset=sp_ft_dataset,
      ft_dataset_test=ft_val_dataset,
      train_size=0.1,
      global_exp=True)

  global_embedding = local_basisnet.get_layer('embedding').get_weights()[0][0]
  new_embedding = np.tile(global_embedding, (3402, 1))
  basisnet.get_layer('embedding').set_weights([new_embedding])

  finetuning(mode='test', ft_dataset=ft_dataset, ft_dataset_test=ft_val_dataset)
  finetuning(
      mode='test',
      ft_dataset=sp_ft_dataset,
      ft_dataset_test=ft_val_dataset,
      train_size=0.1)


if __name__ == '__main__':
  app.run(main)


