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

# Lint as: python3
"""Train and eval sklearn model on downstream embeddings."""

import os
import time

from typing import List, Optional, Set, Tuple
from absl import app
from absl import flags
from absl import logging


import tensorflow as tf
from non_semantic_speech_benchmark.trillsson import eval_downstream_embedding_fidelity as edef


from non_semantic_speech_benchmark.eval_embedding.sklearn import train_and_eval_sklearn

FLAGS = flags.FLAGS

# Controls the file watchdog.
# Note: We borrow some flags from edef, like `embeddings_output_dir`.
flags.DEFINE_integer(
    'folder_waittime_seconds', 300,
    'Time to wait between checking for new embedding dumps, in seconds.')
flags.DEFINE_integer(
    'max_error_count', -1,
    'Max number of errors to have before quitting.')

# Flags just for the sklearn model
flags.DEFINE_string('model_name', None, 'Sklearn model type.')
flags.DEFINE_bool('l2_normalization', False, 'l2_normalization')
flags.DEFINE_string('label_name', None, 'Key of label.')
flags.DEFINE_list('label_list', None, 'Python list of possible label values.')
flags.DEFINE_string('eval_metric', 'accuracy', 'Eval metric for sklearn.')


def _most_recent_embeddings_dump(embedding_filenames,
                                 prefix_dir):
  """Get the filename of the most recent embedding dump."""
  # Filenames are the result of `listdir`, so they should just be integers.
  # Be sure to sort by value, and not lexigraphically, so that 100 > 99.
  return [
      os.path.join(prefix_dir, str(f))
      for f in sorted(embedding_filenames, reverse=True)
  ]


def _train_and_get_score(
    train_glob,
    eval_glob,
    test_glob,
    embedding_name,
    label_name,
    speaker_id,
    ):
  """Wrapper for `train_and_get_score` that uses defaults."""
  logging.info('Running `train_and_get_score` with %s, %s, %s, %s, %s',
               embedding_name, label_name, FLAGS.model_name,
               FLAGS.l2_normalization, speaker_id)
  score_dict = train_and_eval_sklearn.train_and_get_score(
      embedding_name=embedding_name,
      label_name=label_name,
      label_list=FLAGS.label_list,
      train_glob=train_glob,
      eval_glob=eval_glob,
      test_glob=test_glob,
      model_name=FLAGS.model_name,
      l2_normalization=FLAGS.l2_normalization,
      speaker_id_name=speaker_id,
      eval_metrics=[FLAGS.eval_metric])
  assert len(score_dict) == 1, score_dict
  eval_score, test_score = list(score_dict.values())[0]
  return (eval_score, test_score)


def _get_sklearn_scores(
    filename, watched_dir,
    label_name):
  """Trains a sklearn model and returns scores.

  Random exceptions should be handled outside this function.

  Args:
    filename: A string filename.
    watched_dir: The directory being watched.
    label_name: Name of label to use.

  Returns:
    The (step, eval score, test score).
  """
  step = int(filename.split('/')[-1])
  logging.info('Found step for sklearn eval: %i', step)
  def _glob(split):
    return os.path.join(watched_dir, str(step), f'{split}*')

  eval_score, test_score = _train_and_get_score(
      train_glob=_glob('train'),
      eval_glob=_glob('validation'),
      test_glob=_glob('test'),
      embedding_name=edef.EMBEDDING_KEY_,
      label_name=label_name,
      speaker_id=FLAGS.speaker_id_key)

  return step, eval_score, test_score


def sklearn_eval(watched_dir,
                 file_watchdog_waittime_seconds,
                 eval_dir,
                 max_error_count,
                 baseline_score_eval,
                 baseline_score_test):
  """Check fidelity of a dataset."""
  logging.info('watched_dir: %s', watched_dir)

  if not tf.io.gfile.exists(watched_dir):
    tf.io.gfile.makedirs(watched_dir)

  writer = tf.summary.create_file_writer(eval_dir)
  if baseline_score_eval or baseline_score_test:
    assert baseline_score_eval
    assert baseline_score_test
    baseline_writer = tf.summary.create_file_writer(
        os.path.join(eval_dir, 'baseline'))
  else:
    baseline_writer = None
  # Count the number of errors, and exit if too many occur.
  error_count = 0

  prev_files = set([])
  while True:
    cur_files = set(tf.io.gfile.listdir(watched_dir))
    if cur_files == prev_files:
      logging.info(
          'Found the same files as before. Sleeping for %i seconds: %s',
          file_watchdog_waittime_seconds, cur_files)
      time.sleep(file_watchdog_waittime_seconds)  # Seconds.
      continue

    # If files are empty, skip everything.
    if not cur_files:
      logging.info(
          'No files found. Sleeping for %i seconds.',
          file_watchdog_waittime_seconds)
      time.sleep(file_watchdog_waittime_seconds)  # Seconds.
      continue

    # See expected file structure in
    # `eval_downstream_embedding_fidelity._get_embedding_filename`.
    possible_new_files = cur_files - prev_files
    cur_filenames = _most_recent_embeddings_dump(possible_new_files,
                                                 watched_dir)
    step, eval_score, test_score = None, None, None  # Everything failed if None
    for cur_filename in cur_filenames:
      try:
        step, eval_score, test_score = _get_sklearn_scores(
            cur_filename, watched_dir, edef.LABEL_KEY_)
      except Exception as e:  # pylint:disable=broad-except
        logging.warning('Reading embeddings failed: %s', str(e))
        continue
    # If all recent checkpoints failed, register an error, sleep, try again.
    if step is None:
      assert eval_score is None
      assert test_score is None
      error_count += 1
      if max_error_count >= 0 and error_count > max_error_count:
        raise ValueError(f'Failed {error_count} times, exiting.')
      time.sleep(file_watchdog_waittime_seconds)  # Seconds.
      continue

    # Some files were new, so process the newest checkpoint embeddings.
    prev_files = cur_files  # So we don't process the same files again.

    with writer.as_default():
      s = FLAGS.eval_suffix
      tf.summary.scalar(f'downstream_acc_eval_{s}', eval_score, step=step)
      tf.summary.scalar(f'downstream_acc_test_{s}', test_score, step=step)
    if baseline_writer:
      with baseline_writer.as_default():
        tf.summary.scalar(
            f'downstream_acc_eval_{s}', baseline_score_eval, step=step)
        tf.summary.scalar(
            f'downstream_acc_test_{s}', baseline_score_test, step=step)


def main(unused_argv):
  assert FLAGS.embeddings_output_dir
  assert FLAGS.folder_waittime_seconds
  assert FLAGS.eval_dir
  # sklearn flags.
  assert FLAGS.model_name
  assert FLAGS.label_list

  if (FLAGS.file_pattern_train or FLAGS.file_pattern_validation or
      FLAGS.file_pattern_test):
    assert FLAGS.file_pattern_train
    assert FLAGS.file_pattern_validation
    assert FLAGS.file_pattern_test
    assert FLAGS.target_key
    baseline_score_eval, baseline_score_test = _train_and_get_score(
        FLAGS.file_pattern_train,
        FLAGS.file_pattern_validation,
        FLAGS.file_pattern_test,
        embedding_name=FLAGS.target_key[len('embedding/'):],
        label_name=FLAGS.label_name,
        speaker_id=FLAGS.speaker_id_key,
    )
    logging.info('Baseline performance: %f, %f', baseline_score_eval,
                 baseline_score_test)
  else:
    baseline_score_eval, baseline_score_test = None, None

  sklearn_eval(
      watched_dir=FLAGS.embeddings_output_dir,
      file_watchdog_waittime_seconds=FLAGS.folder_waittime_seconds,
      eval_dir=FLAGS.eval_dir,
      max_error_count=FLAGS.max_error_count,
      baseline_score_eval=baseline_score_eval,
      baseline_score_test=baseline_score_test)


if __name__ == '__main__':
  app.run(main)
