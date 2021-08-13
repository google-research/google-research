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

# pylint: disable=line-too-long
"""Train a model to predict protein labels."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import hparams_sets
import pandas as pd
import protein_dataset
import protein_model
import tensorflow.compat.v1 as tf
import utils

flags.DEFINE_string(
    'data_base_path', None,
    'Directory path containing tfrecords named like "train", "dev" and "test"')

flags.DEFINE_string('label_vocab_path', None,
                    'Relative path (from this file) to csv file of labels.')

flags.DEFINE_string('output_dir', '/tmp/protein_model',
                    'Path to save checkpoints.')

flags.DEFINE_string('hparams_set', hparams_sets.small_test_model.__name__,
                    'Hyperparameters to use (see hparams_sets module).')

flags.DEFINE_enum(
    'train_fold', protein_dataset.TRAIN_FOLD, protein_dataset.DATA_FOLD_VALUES,
    'Fold to use for training data '
    '(one of protein_dataset.DATA_FOLD_VALUES)')

flags.DEFINE_enum(
    'eval_fold', protein_dataset.TEST_FOLD, protein_dataset.DATA_FOLD_VALUES,
    'Fold to use for training data '
    '(one of protein_dataset.DATA_FOLD_VALUES)')

FLAGS = flags.FLAGS

_VOCAB_ITEM_COLUMN_NAME = 'vocab_item'
_VOCAB_INDEX_COLUMN_NAME = 'vocab_index'


def _make_estimator(hparams, label_vocab, output_dir):
  """Create a tf.estimator.Estimator.

  Args:
    hparams: tf.contrib.training.HParams.
    label_vocab: list of string.
    output_dir: str. Path to save checkpoints.

  Returns:
    tf.estimator.Estimator.
  """
  model_fn = protein_model.make_model_fn(
      label_vocab=label_vocab, hparams=hparams)
  run_config = tf.estimator.RunConfig(model_dir=output_dir)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      params=hparams,
      config=run_config,
  )

  return estimator


def get_serving_input_fn():
  """Create an input function for serving."""

  def serving_input_fn():
    """Input function for serving."""
    batched_one_hot_sequences = tf.placeholder(
        tf.float32,
        shape=[None, None, len(utils.AMINO_ACID_VOCABULARY)],
        name='batched_one_hot_sequences_placeholder')

    sequence_lengths = tf.placeholder(
        tf.int32,
        shape=[None],
        name='sequence_length_placeholder',
    )

    receivers = {
        protein_dataset.SEQUENCE_KEY: batched_one_hot_sequences,
        protein_dataset.SEQUENCE_LENGTH_KEY: sequence_lengths
    }
    features = {
        protein_dataset.SEQUENCE_KEY: batched_one_hot_sequences,
        protein_dataset.SEQUENCE_LENGTH_KEY: sequence_lengths
    }

    input_receiver = tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=receivers)
    return input_receiver

  return serving_input_fn


def _make_estimator_and_inputs(hparams, label_vocab, data_base_path, output_dir,
                               train_fold, eval_fold):
  """Makes Estimator and input_fn for train and eval.

  Args:
    hparams: tf.contrib.training.HParams.
    label_vocab: list of string.
    data_base_path: str. Directory path containing tfrecords named like "train",
      "dev" and "test"
    output_dir: str. Path to save checkpoints.
    train_fold: fold to use for training data (one of
          protein_dataset.DATA_FOLD_VALUES)
    eval_fold: fold to use for training data (one of
          protein_dataset.DATA_FOLD_VALUES)
  Returns:
    A tuple of estimator, train_spec and eval_spec
  """

  estimator = _make_estimator(
      hparams=hparams, label_vocab=label_vocab, output_dir=output_dir)

  logging.info('Loading data from %s', data_base_path)
  logging.info('Writing to directory %s', output_dir)
  train_input_fn = protein_dataset.make_input_fn(
      data_file_pattern=data_base_path,
      batch_size=hparams.batch_size,
      label_vocab=label_vocab,
      train_dev_or_test=train_fold)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=hparams.train_steps)

  eval_input_fn = protein_dataset.make_input_fn(
      data_file_pattern=data_base_path,
      batch_size=hparams.batch_size,
      label_vocab=label_vocab,
      train_dev_or_test=eval_fold)
  savedmodel_exporters = [
      tf.estimator.LatestExporter(
          name='saved_model',
          serving_input_receiver_fn=get_serving_input_fn(),
          exports_to_keep=1)
  ]
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      throttle_secs=1,
      exporters=savedmodel_exporters,
  )

  return estimator, train_spec, eval_spec


def get_hparams(hparams_set_name):
  """Retrieves a tf.contrib.training.HParams from the hparam_sets module.

  Args:
    hparams_set_name: name of a function in the hparams_sets module returning a
      tf.contrib.training.HParams object.

  Returns:
    tf.contrib.training.HParams.
  """
  return getattr(hparams_sets, hparams_set_name)()


def parse_label_vocab(label_vocab_path):
  """Returns np.array of strings (labels).

  Args:
    label_vocab_path: str. Path to tsv file containing columns
      _VOCAB_ITEM_COLUMN_NAME and _VOCAB_INDEX_COLUMN_NAME. See
      testdata/label_vocab.tsv for an example.

  Returns:
    np.array of str. Labels are sorted by values in column
    _VOCAB_INDEX_COLUMN_NAME.
  """
  with tf.gfile.GFile(label_vocab_path) as f:
    label_df = pd.read_csv(f, sep='\t')

  available_indexes = label_df[_VOCAB_INDEX_COLUMN_NAME].values
  if set(available_indexes) != set(range(len(available_indexes))):
    raise ValueError('Vocab indexes were not the consecutive integers between '
                     '0 (inclusive) and len(vocab) (exclusive). '
                     'Got {}.'.format(sorted(available_indexes)))

  return label_df.sort_values(
      [_VOCAB_INDEX_COLUMN_NAME])[_VOCAB_ITEM_COLUMN_NAME].values


def train(data_base_path, output_dir, label_vocab_path, hparams_set_name,
          train_fold, eval_fold):
  """Constructs trains, and evaluates a model on the given input data.

  Args:
    data_base_path: str. Directory path containing tfrecords named like "train",
      "dev" and "test"
    output_dir: str. Path to save checkpoints.
    label_vocab_path: str. Path to tsv file containing columns
      _VOCAB_ITEM_COLUMN_NAME and _VOCAB_INDEX_COLUMN_NAME. See
      testdata/label_vocab.tsv for an example.
    hparams_set_name: name of a function in the hparams module which returns a
      tf.contrib.training.HParams object.
    train_fold: fold to use for training data (one of
          protein_dataset.DATA_FOLD_VALUES)
    eval_fold: fold to use for training data (one of
          protein_dataset.DATA_FOLD_VALUES)

  Returns:
    A tuple of the evaluation metrics, and the exported objects from Estimator.
  """
  hparams = get_hparams(hparams_set_name)
  label_vocab = parse_label_vocab(label_vocab_path)
  (estimator, train_spec, eval_spec) = _make_estimator_and_inputs(
      hparams=hparams,
      label_vocab=label_vocab,
      data_base_path=data_base_path,
      output_dir=output_dir,
      train_fold=train_fold,
      eval_fold=eval_fold)
  return tf.estimator.train_and_evaluate(
      estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
  train(
      data_base_path=FLAGS.data_base_path,
      output_dir=FLAGS.output_dir,
      label_vocab_path=FLAGS.label_vocab_path,
      hparams_set_name=FLAGS.hparams_set,
      train_fold=FLAGS.train_fold,
      eval_fold=FLAGS.eval_fold)


if __name__ == '__main__':
  FLAGS.alsologtostderr = True  # Shows training output.
  flags.mark_flags_as_required(['data_base_path', 'label_vocab_path'])

  app.run(main)
