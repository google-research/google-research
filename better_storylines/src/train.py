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

"""Tries to classify which sentence is the correct fifth sentence."""

import os
import time

from absl import app
from absl import flags
from absl import logging
import gin
import gin.tf
import models
import rocstories_sentence_embeddings
import tensorflow.compat.v2 as tf
import tensorflow.compat.v2.keras.backend as K
import tensorflow_datasets.public_api as tfds
import utils

gfile = tf.io.gfile


tf.enable_v2_behavior()

FLAGS = flags.FLAGS
flags.DEFINE_string('save_dir', None, 'Where to save model.')
flags.DEFINE_integer('index', None, 'Optional index of this experiment.')
flags.DEFINE_string('data_dir', None, 'TFDS dataset directory.')
flags.DEFINE_multi_string('gin_config', None, 'Gin config file.')
flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')


def cur_learning_rate(optimizer):
  """Copied from keras/optimizers.py."""
  lr = optimizer.lr * (1. / (1. + optimizer.decay * K.cast(
      optimizer.iterations, K.dtype(optimizer.decay))))
  return lr


@gin.configurable('dataset')
def prepare_datasets(dataset_name=gin.REQUIRED,
                     shuffle_input_sentences=False,
                     num_eval_examples=2000,
                     batch_size=32):
  """Create batched, properly-formatted datasets from the TFDS datasets.

  Args:
    dataset_name: Name of TFDS dataset.
    shuffle_input_sentences: If True, the order of the input sentences is
      randomized.
    num_eval_examples: Number of examples to use during evaluation. For the
      nolabel evaluation, this is also the number of distractors we choose
      between.
    batch_size: Batch size.

  Returns:
    A dictionary mapping from the dataset split to a Dataset object.
  """
  splits_to_load = {
      'valid_nolabel': 'train[:2%]',
      'train': 'train[2%:]',
      'train_nolabel': 'train[2%:4%]',
      'valid2018': rocstories_sentence_embeddings.VALIDATION_2018,
      'valid2016': rocstories_sentence_embeddings.VALIDATION_2016}

  datasets = tfds.load(
      dataset_name,
      data_dir=FLAGS.data_dir,
      split=splits_to_load,
      download=False)

  emb_matrices = {}
  # Convert datasets to expected training data format, and build of the
  # embedding matrices.
  train_ds = utils.build_train_style_dataset(
      datasets['train'], batch_size, shuffle_input_sentences)
  datasets['train'], emb_matrices['train'] = train_ds

  valid_nolabel_ds = utils.build_train_style_dataset(
      datasets['valid_nolabel'], batch_size, False,
      num_examples=num_eval_examples)
  datasets['valid_nolabel'], emb_matrices['valid_nolabel'] = valid_nolabel_ds

  train_nolabel_ds = utils.build_train_style_dataset(
      datasets['train_nolabel'], batch_size, False,
      num_examples=num_eval_examples)
  datasets['train_nolabel'], emb_matrices['train_nolabel'] = train_nolabel_ds

  # Convert official evaluation datasets to validation data format. There are no
  # embedding matrices involved here since the task has only two possible next
  # sentences to pick between for each example.
  datasets['valid2018'] = utils.build_validation_dataset(
      datasets['valid2018']).take(num_eval_examples)
  datasets['valid2016'] = utils.build_validation_dataset(
      datasets['valid2016']).take(num_eval_examples)

  logging.info('EMBEDDING MATRICES CREATED:')
  for key in emb_matrices:
    logging.info('%s: %s', key, emb_matrices[key].shape)

  return datasets, emb_matrices


@gin.configurable(blacklist=['save_dir'])
def train(save_dir, num_epochs=300,
          learning_rate=0.0001, save_every_n_epochs=25):
  """Train pipeline for next sentence embedding prediction on ROCStories."""
  #### LOAD DATA ####
  datasets, embedding_matrices = prepare_datasets()

  #### CREATE MODEL AND OPTIMIZER ####
  num_input_sentences = tf.compat.v1.data.get_output_shapes(
      datasets['train'])[0][1]
  model = models.build_model(
      num_input_sentences=num_input_sentences,
      embedding_matrix=embedding_matrices['train'])
  metrics = model.create_metrics()
  optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

  num_train_steps = 0

  #### DO TRAINING ####
  summary_writer = tf.summary.create_file_writer(
      os.path.join(save_dir, 'summaries'))
  with summary_writer.as_default():
    logging.info('Starting training.')
    for epoch in range(1, num_epochs+1):
      for x, labels in datasets['train']:
        utils.train_step(model, optimizer, x, labels, metrics)
        num_train_steps += 1

      start_time = time.time()
      utils.do_evaluation(model, metrics, datasets, embedding_matrices)
      logging.info('Eval took %f seconds.', (time.time() - start_time))

      to_log = ['%s=%f, ' % (m.name, m.result()) for m in metrics.values()]
      logging.info('Epoch %d, %s ', epoch, ''.join(to_log))

      # Add each metric to the TensorBoard and then reset it for the next epoch.
      for metric in metrics.values():
        tf.summary.scalar(
            metric.name, metric.result(), step=optimizer.iterations)
        metric.reset_states()

      # lr = cur_learning_rate(optimizer)
      # tf.summary.scalar('learning_rate', lr, step=optimizer.iterations)

      if epoch % save_every_n_epochs == 0:
        prefix = os.path.join(
            save_dir, 'ep%04d_step%05d.ckpt' % (epoch, num_train_steps))
        logging.info('Saving checkpoint: %s', prefix)
        checkpoint.save(file_prefix=prefix)

  #### SAVE HYPERPARAMETERS AND FINAL EVAL RESULTS TO FILE ####
  to_save = {}
  for metric in metrics.values():
    metric.reset_states()
  utils.do_evaluation(model, metrics, datasets, embedding_matrices)
  for metric in metrics.values():
    to_save['metric_' + metric.name] = metric.result().numpy()
  results_file_path = os.path.join(save_dir, 'final_eval.tsv')
  with gfile.GFile(results_file_path, 'w') as f:
    for name, value in to_save.iteritems():
      f.write('%s\t%s\n' % (name, str(value)))


def main(argv):
  del argv

  save_dir = FLAGS.save_dir
  if FLAGS.index is not None:
    save_dir = os.path.join(save_dir, str(FLAGS.index))
  logging.info('SAVE DIR: %s', save_dir)

  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  logging.info('CONFIG DIRS: %s', str(FLAGS.gin_config))

  gfile.makedirs(save_dir)
  with gfile.GFile(os.path.join(save_dir, 'config.gin'), 'w') as f:
    f.write(gin.config_str())

  train(save_dir)

if __name__ == '__main__':
  flags.mark_flag_as_required('save_dir')
  flags.mark_flag_as_required('gin_config')
  app.run(main)
