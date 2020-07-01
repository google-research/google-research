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

"""Outputs the overall validation accuracy on the 2016 and 2018 validation sets.

Also outputs accuracy on train set and train-style valid set.
"""

import collections
import csv
import os
import re
import time

from absl import app
from absl import flags
from absl import logging
import gin
import gin.tf
import models
import rocstories_sentence_embeddings
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
import utils

gfile = tf.io.gfile


FLAGS = flags.FLAGS
flags.DEFINE_string('base_dir', '/tmp/model',
                    'Base directory containing checkpoints and .gin config.')
flags.DEFINE_string('checkpoint_name', None,
                    'Specific checkpoint to run one-time eval on. If set, '
                    'state of FLAGS.continuous is ignored.')
flags.DEFINE_string('output_dir', None,
                    'Directory in which to save evaluation results.')
flags.DEFINE_bool('continuous', False,
                  'If True, infintely loops over base_dir looking for new '
                  'checkpoints. If False, only loops once.')
flags.DEFINE_bool('sharded_eval', False,
                  'If True, break the dataset into shards and perform eval '
                  'separately on each. This is intended to be used to be able '
                  'to compute error bounds on accuracies.')
flags.DEFINE_float('timeout', 9600, 'If greater than 0, time out after this '
                   'many seconds.')

flags.DEFINE_string('data_dir', None, 'Where to look for TFDS datasets.')


tf.enable_v2_behavior()

METRICS_TO_SAVE = [
    # Acc of predicting 5th sentence out of 2000 from valid set.
    'valid_nolabel_acc',
    # Acc of predicting 5th sentence out of 2000 from train set.
    'train_subset_acc',
    'valid_spring2016_acc',  # Acc on 2016 Story Cloze task.
    'valid_winter2018_acc',  # Acc on 2018 Story Cloze task.
]


@gin.configurable('dataset')
def prepare_datasets(dataset_name=gin.REQUIRED,
                     shuffle_input_sentences=False,
                     num_eval_examples=2000,
                     batch_size=32):
  """Create batched, properly-formatted datasets from the TFDS datasets.

  Args:
    dataset_name: Name of TFDS dataset.
    shuffle_input_sentences: Not used during evaluation, but arg still needed
      for gin compatibility.
    num_eval_examples: Number of examples to use during evaluation. For the
      nolabel evaluation, this is also the number of distractors we choose
      between.
    batch_size: Batch size.

  Returns:
    A dictionary mapping from the dataset split to a Dataset object.
  """
  del shuffle_input_sentences

  splits_to_load = {
      'valid_nolabel': 'train[:2%]',
      'train_nolabel': 'train[2%:4%]',
      'valid2018': rocstories_sentence_embeddings.VALIDATION_2018,
      'valid2016': rocstories_sentence_embeddings.VALIDATION_2016}

  datasets = tfds.load(
      dataset_name,
      data_dir=FLAGS.data_dir,
      split=splits_to_load,
      download=False)

  emb_matrices = {}

  valid_nolabel_ds = utils.build_train_style_dataset(
      datasets['valid_nolabel'], batch_size, False,
      num_examples=num_eval_examples, is_training=False)
  datasets['valid_nolabel'], emb_matrices['valid_nolabel'] = valid_nolabel_ds

  train_nolabel_ds = utils.build_train_style_dataset(
      datasets['train_nolabel'], batch_size, False,
      num_examples=num_eval_examples, is_training=False)
  datasets['train_nolabel'], emb_matrices['train_nolabel'] = train_nolabel_ds

  # Convert official evaluation datasets to validation data format. There are no
  # embedding matrices involved here since the task has only two possible next
  # sentences to pick between for each example. Ignore num_eval_examples and use
  # the full datasets for these.
  datasets['valid2018'] = utils.build_validation_dataset(
      datasets['valid2018'])
  datasets['valid2016'] = utils.build_validation_dataset(
      datasets['valid2016'])

  return datasets, emb_matrices


def eval_single_checkpoint(
    ckpt_name, output_path, model, datasets, embedding_matrices):
  """Runs quantitative evaluation on a single checkpoint."""
  if gfile.exists(output_path):
    logging.info('Skipping already exists: "%s"', output_path)
    return

  metrics = model.create_metrics()

  logging.info('Evaluating: "%s"', ckpt_name)
  utils.do_evaluation(model, metrics, datasets, embedding_matrices)

  # This code assumed the checkpoint name contains the epoch and step in the
  # following format.
  path_search = re.search(r'ep(\w+)_step(\w+)', ckpt_name)
  epoch = int(path_search.group(1))
  step = int(path_search.group(2))

  to_write = collections.OrderedDict()
  to_write['checkpoint'] = ckpt_name
  to_write['epoch'] = epoch
  to_write['step'] = step
  for metric in metrics.values():
    if metric.name in METRICS_TO_SAVE:
      tf.summary.scalar(metric.name, metric.result(), step=step)
      to_write[metric.name] = metric.result().numpy()
    metric.reset_states()

  # Save the results to a text file.
  with gfile.GFile(output_path, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=to_write.keys())
    writer.writeheader()
    writer.writerow(to_write)


def do_eval(checkpoint_paths, eval_dir, datasets,
            embedding_matrices, sharded_eval=False):
  """Runs quantitative eval for each checkpoint in list."""
  num_input_sentences = tf.compat.v1.data.get_output_shapes(
      datasets['valid2018'])[0][1]

  embedding_dim = tf.compat.v1.data.get_output_shapes(
      datasets['valid2018'])[0][2]

  for checkpoint_path in sorted(checkpoint_paths):
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    logging.info('Processing checkpoint %s', checkpoint_name)

    model = models.build_model(
        num_input_sentences=num_input_sentences,
        embedding_dim=embedding_dim)

    checkpoint = tf.train.Checkpoint(model=model)
    result = checkpoint.restore(checkpoint_path).expect_partial()
    result.assert_nontrivial_match()

    if sharded_eval:
      num_shards = 10
      for i in range(num_shards):
        sharded_datasets = {
            name: ds.shard(num_shards, i) for name, ds in datasets.items()
        }
        output_path = os.path.join(
            eval_dir, '%s_metrics_shard.%02d.csv' % (checkpoint_name, i))
        eval_single_checkpoint(
            checkpoint_name, output_path, model,
            sharded_datasets, embedding_matrices)
    else:
      eval_path = os.path.join(eval_dir, '%s_metrics.csv' % checkpoint_name)
      eval_single_checkpoint(
          checkpoint_name, eval_path, model, datasets, embedding_matrices)


def create_single_results_file(eval_dir):
  """Merges quantitative result files for each checkpoint into a single file."""
  header = ''
  to_save = []
  for fpath in gfile.glob(os.path.join(eval_dir, '*metrics*.csv')):
    if 'all_metrics' not in fpath:
      with gfile.GFile(fpath, 'r') as f:
        header = next(f)
        to_save.append(next(f))

  if to_save:
    merged_metrics_file_path = os.path.join(eval_dir, 'all_metrics.csv')
    with gfile.GFile(merged_metrics_file_path, 'w') as f:
      f.write(header)
      for data_line in to_save:
        f.write(data_line)


def run_eval():
  """Evaluate the ROCSTories next-sentence prediction model."""

  base_dir = FLAGS.base_dir
  if FLAGS.output_dir:
    eval_dir = FLAGS.output_dir
  else:
    eval_dir = os.path.join(base_dir, 'eval')
  gfile.makedirs(eval_dir)

  datasets, embedding_matrices = prepare_datasets()

  if FLAGS.checkpoint_name is not None:
    logging.info('Evaluating single checkpoint: %s', FLAGS.checkpoint_name)
    checkpoint_paths = [os.path.join(base_dir, FLAGS.checkpoint_name)]
    do_eval(checkpoint_paths, eval_dir, datasets,
            embedding_matrices, FLAGS.sharded_eval)
  elif not FLAGS.continuous:
    logging.info('Evaluating all checkpoints currently in %s', base_dir)
    checkpoint_paths = gfile.glob(os.path.join(base_dir, '*ckpt*.index'))
    checkpoint_paths = [p.replace('.index', '') for p in checkpoint_paths]
    do_eval(checkpoint_paths, eval_dir, datasets,
            embedding_matrices, FLAGS.sharded_eval)
    create_single_results_file(eval_dir)
  else:
    logging.info('Continuous evaluation in %s', base_dir)
    checkpoint_iter = tf.train.checkpoints_iterator(
        base_dir, timeout=FLAGS.timeout)

    summary_writer = tf.summary.create_file_writer(
        os.path.join(base_dir, 'summaries_eval'))
    with summary_writer.as_default():
      for checkpoint_path in checkpoint_iter:
        do_eval([checkpoint_path], eval_dir, datasets,
                embedding_matrices, FLAGS.sharded_eval)

      # Save a file with the results from all the checkpoints
      create_single_results_file(eval_dir)
  logging.info('Results written to %s', eval_dir)


def main(argv):
  del argv

  # Load gin.config settings stored in model directory. It is possible to run
  # this script concurrently with the train script. In this case, wait for the
  # train script to start up and actually write out a gin config file.
  # Wait 10 minutes (periodically checking for file existence) before giving up.
  gin_config_path = os.path.join(FLAGS.base_dir, 'config.gin')
  num_tries = 0
  while not gfile.exists(gin_config_path):
    num_tries += 1
    if num_tries >= 10:
      raise ValueError('Could not find config.gin in "%s"' % FLAGS.base_dir)
    time.sleep(60)

  gin.parse_config_file(gin_config_path, skip_unknown=True)
  gin.finalize()

  run_eval()


if __name__ == '__main__':
  app.run(main)
