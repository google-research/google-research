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

# Lint as: python3
"""Train and eval a sklearn model."""

import itertools
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable
from absl import logging

import numpy as np
import tensorflow as tf

from non_semantic_speech_benchmark.eval_embedding import metrics
from non_semantic_speech_benchmark.eval_embedding.sklearn import models
from non_semantic_speech_benchmark.eval_embedding.sklearn import sklearn_utils


def train_and_get_score(
    embedding_name,
    label_name,
    label_list,
    train_glob,
    eval_glob,
    test_glob,
    model_name,
    l2_normalization,
    speaker_id_name = None,
    save_model_dir = None,
    save_predictions_dir = None,
    eval_metrics = ('accuracy',)
):
  """Train and eval sklearn models on data.

  Args:
    embedding_name: Name of embedding.
    label_name: Name of label to use.
    label_list: Python list of all values for label.
    train_glob: Location of training data, as tf.Examples.
    eval_glob: Location of eval data, as tf.Examples.
    test_glob: Location of test data, as tf.Examples.
    model_name: Name of model.
    l2_normalization: Python bool. If `True`, normalize embeddings by L2 norm.
    speaker_id_name: `None`, or name of speaker ID field.
    save_model_dir: If not `None`, write sklearn models to this directory.
    save_predictions_dir: If not `None`, write numpy array of predictions on
      train, eval, and test into this directory.
    eval_metrics: Iterable of string names of the desired evaluation metrics.

  Returns:
    A dict: {metric name: (eval metric, test metric)}
  """
  def _cur_s(s):
    return time.time() - s
  def _cur_m(s):
    return (time.time() - s) / 60.0

  # Read and validate data.
  def _read_glob(glob, name):
    logging.info('Starting to read %s: %s', name, glob)
    s = time.time()
    npx, npy, _ = sklearn_utils.tfexamples_to_nps(glob, embedding_name,
                                                  label_name, label_list,
                                                  l2_normalization,
                                                  speaker_id_name)
    logging.info('Finished reading %s %s data: %.2f sec.', embedding_name, name,
                 _cur_s(s))
    return npx, npy
  npx_train, npy_train = _read_glob(train_glob, 'train')
  npx_eval, npy_eval = _read_glob(eval_glob, 'eval')
  npx_test, npy_test = _read_glob(test_glob, 'test')

  # Sanity check npx_*.
  assert npx_train.size > 0
  assert npx_eval.size > 0
  assert npx_test.size > 0

  # Sanity check npy_train.
  assert npy_train.size > 0
  assert np.unique(npy_train).size > 1
  # Sanity check npy_eval.
  assert npy_eval.size > 0
  assert np.unique(npy_eval).size > 1
  # Sanity check npy_test.
  assert npy_test.size > 0
  assert np.unique(npy_test).size > 1

  # If `save_model_dir` is present and the model exists, load the model instead
  # of training.
  if save_model_dir:
    cur_models_dir = os.path.join(save_model_dir, embedding_name)
    tf.io.gfile.makedirs(cur_models_dir)
    model_filename = os.path.join(cur_models_dir, f'{model_name}.pickle')
    train_model = not tf.io.gfile.exists(model_filename)
  else:
    train_model = True

  # Train models.
  if train_model:
    d = models.get_sklearn_models()[model_name]()
    logging.info('Made model: %s.', model_name)
    s = time.time()
    d.fit(npx_train, npy_train)
    logging.info('Trained model: %s, %s: %.2f min', model_name, embedding_name,
                 _cur_m(s))
    # If `save_model_dir` is present and the model exists, write model to this
    # directory.
    if save_model_dir:
      with tf.io.gfile.GFile(model_filename, 'wb') as f:
        pickle.dump(d, f)
  else:  # Load model.
    with tf.io.gfile.GFile(model_filename, 'rb') as f:
      d = pickle.load(f)

  scores = {}
  for eval_metric in eval_metrics:
    eval_score, test_score = _calc_scores(eval_metric, d, npx_eval, npy_eval,
                                          npx_test, npy_test, label_list)
    logging.info('Finished eval: %s: %.3f', model_name, eval_score)
    logging.info('Finished test: %s: %.3f', model_name, test_score)
    scores[eval_metric] = (eval_score, test_score)

  if save_predictions_dir:
    cur_preds_dir = os.path.join(save_predictions_dir, embedding_name)
    tf.io.gfile.makedirs(cur_preds_dir)
    for dat_name, dat_x, dat_y in [('train', npx_train, npy_train),
                                   ('eval', npx_eval, npy_eval),
                                   ('test', npx_test, npy_test)]:
      pred_filename = os.path.join(cur_preds_dir,
                                   f'{model_name}_{dat_name}_pred.npz')
      pred_y = d.predict(dat_x)
      with tf.io.gfile.GFile(pred_filename, 'wb') as f:
        np.save(f, pred_y)
      y_filename = os.path.join(cur_preds_dir,
                                f'{model_name}_{dat_name}_y.npz')
      with tf.io.gfile.GFile(y_filename, 'wb') as f:
        np.save(f, dat_y)

  return scores


def _calc_scores(eval_metric, d, npx_eval,
                 npy_eval, npx_test,
                 npy_test,
                 label_list):
  """Compute desired metric on eval and test."""
  if eval_metric == 'equal_error_rate':
    # Eval.
    regression_output = d.predict_proba(npx_eval)[:, 1]  # Prob of class 1.
    eval_score = metrics.calculate_eer(npy_eval, regression_output)
    # Test.
    regression_output = d.predict_proba(npx_test)[:, 1]  # Prob of class 1.
    test_score = metrics.calculate_eer(npy_test, regression_output)
  elif eval_metric == 'accuracy':
    # Eval.
    eval_score = d.score(npx_eval, npy_eval)
    # Test.
    test_score = d.score(npx_test, npy_test)
  elif eval_metric == 'balanced_accuracy':
    # Eval.
    pred_eval = d.predict(npx_eval)
    eval_score = metrics.balanced_accuracy(npy_eval, pred_eval)
    # Test.
    pred_test = d.predict(npx_test)
    test_score = metrics.balanced_accuracy(npy_test, pred_test)
  elif eval_metric == 'unweighted_average_recall':
    # The accuracy per class divided by the number of classes without
    # considerations of instances per class.
    def _class_scores(npx, npy):
      class_scores = []
      for lbl in np.unique(npy):
        i = npy == lbl
        class_scores.append(d.score(npx[i], npy[i]))
      return class_scores
    eval_score = np.mean(_class_scores(npx_eval, npy_eval))
    test_score = np.mean(_class_scores(npx_test, npy_test))
  elif eval_metric == 'auc':
    binary_classification = (len(label_list) == 2)
    eval_score = metrics.calculate_auc(
        labels=npy_eval,
        predictions=regression_output,
        binary_classification=binary_classification)
    test_score = metrics.calculate_auc(
        labels=npy_test,
        predictions=regression_output,
        binary_classification=binary_classification)
  elif eval_metric == 'dprime':
    binary_classification = (len(label_list) == 2)
    eval_auc = metrics.calculate_auc(
        labels=npy_eval,
        predictions=regression_output,
        binary_classification=binary_classification)
    test_auc = metrics.calculate_auc(
        labels=npy_test,
        predictions=regression_output,
        binary_classification=binary_classification)
    eval_score = metrics.dprime_from_auc(eval_auc)
    test_score = metrics.dprime_from_auc(test_auc)
  else:
    raise ValueError(f'`eval_metric` not recognized: {eval_metric}')
  return eval_score, test_score


def experiment_params(
    embedding_list,
    speaker_id_name,
    label_name,
    label_list,
    train_glob,
    eval_glob,
    test_glob,
    save_model_dir,
    save_predictions_dir,
    eval_metrics,
    comma_escape_char = '?',
):
  """Get experiment params."""
  # Sometimes we want commas to appear in `embedding_modules`,
  # `embedding_names`, or `module_output_key`. However, commas get split out in
  # Google's Python `DEFINE_list`. We compromise by introducing a special
  # character, which we replace with commas here.
  embedding_list = _maybe_add_commas(embedding_list, comma_escape_char)

  # Enumerate the configurations we want to run.
  exp_params = []
  model_names = models.get_sklearn_models().keys()
  for elem in itertools.product(*[embedding_list, model_names]):

    def _params_dict(l2_normalization,
                     speaker_id_name=speaker_id_name,
                     elem=elem):
      return {
          'embedding_name': elem[0],
          'model_name': elem[1],
          'label_name': label_name,
          'label_list': label_list,
          'train_glob': train_glob,
          'eval_glob': eval_glob,
          'test_glob': test_glob,
          'l2_normalization': l2_normalization,
          'speaker_id_name': speaker_id_name,
          'save_model_dir': save_model_dir,
          'save_predictions_dir': save_predictions_dir,
          'eval_metrics': eval_metrics,
      }

    exp_params.append(_params_dict(l2_normalization=True))
    exp_params.append(_params_dict(l2_normalization=False))
    if speaker_id_name is not None:
      exp_params.append(
          _params_dict(l2_normalization=True, speaker_id_name=None))
      exp_params.append(
          _params_dict(l2_normalization=False, speaker_id_name=None))

  return exp_params


def format_text_line(k_v):
  """Convert params and score to human-readable format."""
  # p, (eval_score, test_score) = k_v
  p, scores = k_v
  line_list = [
      f'Embed: {p["embedding_name"]}',
      f'Label: {p["label_name"]}',
      f'Model: {p["model_name"]}',
      f'L2 normalization: {p["l2_normalization"]}',
      f'Speaker normalization: {p["speaker_id_name"] is not None}', '\n'
  ]
  logging.info('Scores: %s', scores)
  for metric_name, (eval_score, test_score) in scores.items():
    line_list.append(f'Eval score {metric_name}: {eval_score}')
    line_list.append(f'Test score {metric_name}: {test_score}')
  cur_elem = ', '.join(line_list)
  logging.info('Finished formatting: %s', cur_elem)
  return cur_elem


def _maybe_add_commas(list_obj, comma_escape_char):
  return [x.replace(comma_escape_char, ',') for x in list_obj]


def validate_flags(train_glob, eval_glob, test_glob,
                   output_file):
  """Validate flags."""
  if not tf.io.gfile.glob(train_glob):
    raise ValueError(f'Files not found: {train_glob}')
  if not tf.io.gfile.glob(eval_glob):
    raise ValueError(f'Files not found: {eval_glob}')
  if not tf.io.gfile.glob(test_glob):
    raise ValueError(f'Files not found: {test_glob}')

  outputs = tf.io.gfile.glob(f'{output_file}*')
  if outputs:
    raise ValueError(f'Output file already exists: {outputs}')

  # Create output directory if it doesn't already exist.
  outdir = os.path.dirname(output_file)
  tf.io.gfile.makedirs(outdir)
