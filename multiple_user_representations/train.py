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

"""Trains the next-item prediction model for recommender sytems."""

import functools
import os
from typing import Any, Dict, Sequence, Text

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import yaml

from multiple_user_representations import dataloader
from multiple_user_representations.models import ItemModelMLP
from multiple_user_representations.models import RETRIEVAL_MODELS
from multiple_user_representations.models import task
from multiple_user_representations.models import USER_MODELS
from multiple_user_representations.models import util

FLAGS = flags.FLAGS

flags.DEFINE_string('config_path', None, 'Path to the config file.')
flags.DEFINE_string(
    'results_dir', None,
    'Results directory. Overrides the `results_dir` parameter in config.')
flags.DEFINE_integer('seed', 1234, 'Random seed for tf.')
flags.DEFINE_integer(
    'epochs', -1,
    'Number of epochs. If greater than zero, it overrides the epochs specified in config.')  # pylint: disable=line-too-long
flags.DEFINE_integer(
    'num_representations', -1,
    'Number of user representations. If greater than zero, it overrides the epochs specified in config.')  # pylint: disable=line-too-long
flags.DEFINE_bool('use_disagreement', None,
                  'Use cosine disagreement loss for query heads.')
flags.DEFINE_bool('save_embeddings', False,
                  'Save the embedding space. Used for visualization.')
flags.DEFINE_string('dataset_path', None, 'If given, will override the config.')
flags.DEFINE_string('root_dir', None, 'If given, will override the config.')
flags.DEFINE_bool('use_projection_layer', None, 'Use projection layer or not.')
flags.DEFINE_list('metrics_k', [10, 50, 100, 200],
                  'The values for K when computing metrics@K')
flags.DEFINE_enum('retrieval_model_type', None, ['standard_retrieval',
                                                 'density_smoothed_retrieval'],
                  ('Retrieval model to train. The supported retrieval models '
                   'are `standard_retrieval` and `density_smoothed_retrieval`.'
                   'The `standard_retrieval` implements the standard two tower'
                   'retrieval model, and `density_smoothed_retrieval` is the'
                   'extension proposed in http://shortn/_uPej1Fh7Jq#heading=h.q00w2gg84yzw'))  # pylint: disable=line-too-long
flags.DEFINE_bool('finetuning', False,
                  ('Finetune the model. This is used to calibrate the user tower on the interest distribution of users in the train data keeping the user tower fixed.'))  # pylint: disable=line-too-long
flags.DEFINE_float('delta', 0.0002,
                   ('The stopping threshold for iterative density smoothing. See http://shortn/_uPej1Fh7Jq#heading=h.q00w2gg84yzw for details'))  # pylint: disable=line-too-long


def load_config():
  """Loads and returns the config for the experiment.

  Returns:
    config: The experiment config.
  """

  config = yaml.load(open(FLAGS.config_path, 'r'), Loader=yaml.FullLoader)

  model_config = config['model_config']

  if FLAGS.results_dir is not None:
    config['results_dir'] = FLAGS.results_dir
  if FLAGS.root_dir is not None:
    config['root_dir'] = FLAGS.root_dir
  if FLAGS.epochs > 0:
    config['epochs'] = FLAGS.epochs
  if FLAGS.num_representations > 0:
    model_config['user_model_config'][
        'num_representations'] = FLAGS.num_representations
  if FLAGS.use_disagreement is not None:
    model_config['use_disagreement_loss'] = FLAGS.use_disagreement
  if FLAGS.use_projection_layer is not None:
    model_config['user_model_config'][
        'use_projection_layer'] = FLAGS.use_projection_layer
  if FLAGS.retrieval_model_type is not None:
    model_config['retrieval_model_type'] = FLAGS.retrieval_model_type
  if FLAGS.dataset_path is not None:
    config['dataset_path'] = FLAGS.dataset_path

  root_dir = config['root_dir']
  num_heads = model_config['user_model_config']['num_representations']
  if num_heads == 1 and FLAGS.use_projection_layer:
    model_str = 'SUR-P'
    model_config['use_disagreement_loss'] = False
  elif num_heads == 1:
    model_str = 'SUR'
    model_config['use_disagreement_loss'] = False
  elif model_config['use_disagreement_loss']:
    model_str = 'MUR-D_{}'.format(num_heads)
  else:
    model_str = 'MUR_{}'.format(num_heads)

  results_dir = os.path.join(root_dir, config['results_dir'],
                             config['dataset_path'],
                             'seed_{}'.format(FLAGS.seed), model_str)
  config['results_dir'] = results_dir
  config['dataset_path'] = os.path.join(root_dir, config['dataset_path'])

  return config


def setup_retrieval_model(model_config,
                          item_dataset,
                          temperature = 1.0):
  """Sets up the retrieval model.

  Args:
    model_config: Dictionary containing model specific configuration. See the
      `model_config` field in configs/synthetic_experiment.yaml for expected
      parameters.
    item_dataset: Dataset containing all the items.
    temperature: Softmax temperature used for training.

  Returns:
    retrieval_model: The retrieval model.
  """

  user_model = USER_MODELS[model_config['user_model']](
      output_dimension=model_config['output_dimension'],
      input_embedding_dimension=model_config['input_embedding_dimension'],
      max_sequence_size=model_config['max_seq_size'],
      vocab_size=model_config['vocab_size'],
      mask_zero=model_config['mask_zero'],
      **model_config['user_model_config'])
  item_model_config = model_config['item_model_config']
  num_layers = item_model_config.get('num_layers', 0)
  dropout = item_model_config.get('dropout', 0.0)

  item_model = ItemModelMLP(model_config['output_dimension'],
                            model_config['vocab_size'],
                            model_config['input_embedding_dimension'],
                            num_layers, dropout)

  def setup_metrics(item_dataset,
                    item_model):

    def item_map(batched_items):
      return tf.squeeze(item_model(tf.expand_dims(batched_items, axis=1)))

    candidates = item_dataset.batch(500).map(item_map)
    metrics_k = map(int, FLAGS.metrics_k)
    factorized_metrics = []
    for x in metrics_k:
      factorized_metrics.append(
          tf.keras.metrics.TopKCategoricalAccuracy(k=x, name=f'HR@{x}'))
      factorized_metrics.append(
          tf.keras.metrics.TopKCategoricalAccuracy(k=x, name=f'Head_HR@{x}'))
      factorized_metrics.append(
          tf.keras.metrics.TopKCategoricalAccuracy(k=x, name=f'Tail_HR@{x}'))

    candidates = task.MultiQueryStreaming(k=256).index_from_dataset(candidates)
    metrics = task.MultiQueryFactorizedTopK(
        candidates=candidates, metrics=factorized_metrics, k=256)
    retrieval_task = task.MultiShotRetrievalTask(
        metrics=metrics, temperature=temperature)

    return retrieval_task

  retrieval_task = setup_metrics(item_dataset, item_model)
  num_items = model_config['vocab_size']

  retrieval_type = model_config.get('retrieval_model_type',
                                    'standard_retrieval')
  assert retrieval_type in RETRIEVAL_MODELS, ('{:s} not a valid retrieval model'
                                              ' type.').format(retrieval_type)

  retrieval_model = RETRIEVAL_MODELS[retrieval_type](
      user_model,
      item_model,
      retrieval_task,
      num_items,
      use_disagreement_loss=model_config['use_disagreement_loss'])

  return retrieval_model


def train_retrieval_model(
    retrieval_model,
    train_dataset,
    val_dataset,
    use_early_stopping,
    batch_size,
    num_epochs,
    early_stopping_patience = 5,
    early_stopping_criterion = 'val_total_loss'):
  """Trains the retrieval model.

  Args:
    retrieval_model: The tf keras model that will be trained.
    train_dataset: The train dataset.
    val_dataset: The validation dataset.
    use_early_stopping: True, if early stopping is used for validation dataset.
    batch_size: The training batch size.
    num_epochs: The number of epochs for which the retrieval model is trained.
      When use_early_stopping is True, then num_epochs is the maximum number of
      epochs the model is trained for.
    early_stopping_patience: Early stop patience for keras callback.
    early_stopping_criterion: Early stop criterion.

  Returns:
    Training history.
  """

  if use_early_stopping:
    if 'loss' in early_stopping_criterion:
      mode = 'min'
    elif 'HR' in early_stopping_criterion:
      mode = 'max'
    else:
      mode = 'auto'

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_criterion,
            min_delta=1e-5,
            patience=early_stopping_patience,
            mode=mode,
            verbose=1)
    ]

    return retrieval_model.fit(
        train_dataset.batch(batch_size),
        epochs=num_epochs,
        validation_data=val_dataset.batch(batch_size),
        callbacks=callbacks)
  else:
    return retrieval_model.fit(
        train_dataset.batch(batch_size),
        epochs=num_epochs,
        validation_data=val_dataset.batch(batch_size))


def setup_and_train(config):
  """Sets up the experiment and trains the model.

  Args:
    config: The config of the experiment. See configs/synthetic_experiment.yaml
      for the template.

  Returns:
    results: A dictionary of metrics evaluated on test dataset.
  """

  data = dataloader.load_dataset(config['dataset_name'], config['dataset_path'],
                                 use_validation=True)
  results = dict()

  train_dataset = data['train_dataset']
  val_dataset = data['valid_dataset']
  model_config = config['model_config']
  model_config['max_seq_size'] = data['max_seq_size']
  model_config['vocab_size'] = data['num_items']
  model_config['mask_zero'] = data.get('mask_zero', False)
  temperature = config.get('softmax_temperature', 1.0)

  retrieval_model = setup_retrieval_model(model_config, data['item_dataset'],
                                          temperature)
  retrieval_model_type = model_config.get('retrieval_model_type',
                                          'standard_retrieval')
  item_count_weights = data['item_count_probs']
  retrieval_model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']))
  if retrieval_model_type == 'density_smoothed_retrieval':

    train_dataset = util.update_train_dataset_with_sample_weights(
        train_dataset, item_count_weights)

    train_retrieval_model(
        retrieval_model=retrieval_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['train_batch_size'],
        num_epochs=config['epochs'],
        use_early_stopping=True)

    early_stopping_criterion = f'val_Tail_HR@{max(FLAGS.metrics_k)}'
    fit_retrieval_model_iteration = functools.partial(
        train_retrieval_model,
        retrieval_model=retrieval_model,
        val_dataset=val_dataset,
        batch_size=config['train_batch_size'],
        use_early_stopping=True,
        num_epochs=5,
        early_stopping_criterion=early_stopping_criterion,
        early_stopping_patience=1)

    # Consider resetting the optimizer params after each iteration.
    retrieval_model.iterative_training(
        fit_retrieval_model_iteration,
        train_dataset,
        data['item_dataset'],
        item_count_weights,
        config['results_dir'],
        max_iterations=5,
        delta=FLAGS.delta,
        momentum=0.8)
  elif retrieval_model_type == 'standard_retrieval':
    train_retrieval_model(
        retrieval_model=retrieval_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['train_batch_size'],
        num_epochs=config['epochs'],
        use_early_stopping=True)
  else:
    raise ValueError(
        ('{:s} not a valid retrieval model type.'.format(retrieval_model_type)))

  finetune_user_tower = FLAGS.finetuning
  if finetune_user_tower:
    logging.info('Finetuning the model.')

    # Reset the optimizer params before finetuning.
    retrieval_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['learning_rate'] / 10))

    # Refine the user tower keeping the item tower fixed.
    # Consider calibration/finetuning on validation dataset.
    retrieval_model.candidate_model.trainable = False
    train_retrieval_model(
        retrieval_model=retrieval_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['train_batch_size'],
        num_epochs=config['epochs'],
        use_early_stopping=True,
        early_stopping_patience=1)

  # Test
  logging.info('Evaluating on test data.')
  test_dataset = data['test_dataset']
  results['eval_result'] = retrieval_model.evaluate(
      test_dataset.batch(config['test_batch_size']), return_dict=True)

  if FLAGS.save_embeddings:
    item_embeddings = data['item_dataset'].batch(100).map(
        retrieval_model.candidate_model).unbatch()
    item_embeddings = np.array(list(item_embeddings.as_numpy_iterator()))
    results['item_embeddings'] = item_embeddings

    # Save the user_embeddings for test dataset
    user_model = retrieval_model.user_model
    user_embeddings = np.array(
        list(
            test_dataset.batch(
                100).map(lambda x: user_model(x['user_item_sequence'])).unbatch(
                ).as_numpy_iterator()))  # Shape: [N, H, D]
    user_embeddings = np.squeeze(user_embeddings)
    results['user_embeddings'] = user_embeddings

  return results


def save_results(config, results):
  """Saves results.

  Args:
    config: Config of the experiment.
    results: Results after evaluating the model.
  """

  results_dir = config['results_dir']
  tf.io.gfile.makedirs(results_dir)
  with tf.io.gfile.GFile(os.path.join(results_dir, 'config.yaml'), 'w') as fout:
    yaml.dump(config, fout, default_flow_style=False, allow_unicode=True)

  with tf.io.gfile.GFile(os.path.join(results_dir, 'eval_result.yaml'), 'w') as fout:
    yaml.dump(
        results['eval_result'],
        fout,
        default_flow_style=False,
        allow_unicode=True)

  if FLAGS.save_embeddings:
    for arr_key, arr in results.items():
      embeddings_file = os.path.join(results_dir, '{}.npy'.format(arr_key))
      with tf.io.gfile.GFile(embeddings_file, 'wb') as fout:
        np.save(fout, arr)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  config = load_config()
  results = setup_and_train(config)
  save_results(config, results)


if __name__ == '__main__':
  app.run(main)
