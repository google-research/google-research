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
"""Eval a model on downstream task and to measure fidelity."""

import os
import time

from typing import Any, List, Optional, Tuple, Union

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from non_semantic_speech_benchmark.trillsson import get_data
from non_semantic_speech_benchmark.trillsson import models

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', None, 'Directory where the model was written to.')
flags.DEFINE_string('eval_dir', None,
                    'Directory where the results are saved to.')

flags.DEFINE_string('file_pattern_train', None, 'Dataset location.')
flags.DEFINE_string('file_pattern_validation', None, 'Dataset location.')
flags.DEFINE_string('file_pattern_test', None, 'Dataset location.')

flags.DEFINE_string('eval_suffix', None, 'Prefix for tensorboard.')
flags.DEFINE_integer('eval_batch_size', None, 'The number of eval batches.')

flags.DEFINE_string(
    'label_key', None,
    'Labels in the dataset on disk. Will be dumped to disk for '
    '`downstream_sklearn_eval` in a different format.')
flags.DEFINE_string(
    'target_key', None, 'Teacher embedding key in precomputed tf.Examples.')

# Flags for dumping embeddings to disk for more analysis.
flags.DEFINE_string(
    'embeddings_output_dir', None,
    'Optional directory to write embeddings to disk.')
flags.DEFINE_string('speaker_id_key', None, 'Optional')

# Teacher / student network flags.
flags.DEFINE_string('model_type', None, 'Specification for student model.')
flags.DEFINE_alias('mt', 'model_type')

flags.DEFINE_float('lr', None, 'not used')

flags.DEFINE_integer('take_fixed_data', None,
                     'If not `None`, take a fixed number of data elements.')
flags.DEFINE_integer('timeout', 7200, 'Wait-for-checkpoint timeout.')

# Not used.
flags.DEFINE_integer('max_sample_length', -1, 'Max samples length.')
flags.DEFINE_alias('msl', 'max_sample_length')
flags.DEFINE_integer('tbs', None, 'The number of images in each batch.')

# Constants for writing embedding data dump.
AUDIO_KEY_ = 'audio'
LABEL_KEY_ = 'label'
EMBEDDING_KEY_ = 'emb'
SPLIT_NAMES_ = ['train', 'validation', 'test']


def _get_embedding_filename(base_dir, split_name, step):
  """Create the filename for embeddings."""
  return os.path.join(base_dir, str(step), f'{split_name}-embeddings.tfrecord')


def _get_ds(file_patterns, step):
  """Gets a tf.Dataset for a file."""
  ds = get_data.get_data(
      file_patterns=file_patterns,
      reader=tf.data.TFRecordDataset,
      samples_key=AUDIO_KEY_,
      batch_size=FLAGS.eval_batch_size,
      loop_forever=False,
      shuffle=False,
      target_key=FLAGS.target_key,
      label_key=FLAGS.label_key,
      speaker_id_key=FLAGS.speaker_id_key,
      samples_are_float=False,
      max_samples_length=None)
  logging.info('Got dataset for eval step: %s.', step)
  if FLAGS.take_fixed_data:
    ds = ds.take(FLAGS.take_fixed_data)
  return ds


def _get_splits(
    names, file_patterns,
    embeddings_output_dir, step
):
  """Returns a list of (name, dataset, OPTIONAL tfrecord writer)."""
  assert len(names) == len(file_patterns)

  dss = [_get_ds(file_pattern, step) for file_pattern in file_patterns]
  logging.info('[_get_splits]: Got dss: %s', dss)
  if embeddings_output_dir:
    emb_writers = []
    for n in names:
      emb_filename = _get_embedding_filename(embeddings_output_dir, n, step)
      if not tf.io.gfile.exists(os.path.dirname(emb_filename)):
        logging.info('Creating dir: %s', os.path.dirname(emb_filename))
        tf.io.gfile.makedirs(os.path.dirname(emb_filename))
      emb_writers.append(tf.io.TFRecordWriter(emb_filename))
  else:
    emb_writers = [None, None, None]

  return list(zip(names, dss, emb_writers))


def process_single_checkpoint(
    writer,
    model,
    checkpoint,
    ckpt,
    output_dim,
    model_output_key,
    embeddings_output_dir,
    file_pattern_train,
    file_pattern_validation,
    file_pattern_test):
  """Perform all the actions associated with a single checkpoint."""
  assert 'ckpt-' in ckpt, ckpt
  step = int(ckpt.split('ckpt-')[-1])
  logging.info(
      '[process_single_checkpoint] Starting to evaluate step: %i.', step)

  checkpoint.restore(ckpt)
  logging.info(
      '[process_single_checkpoint] Loaded weights for eval step: %i.', step)

  splits_metadata = _get_splits(
      SPLIT_NAMES_,
      [file_pattern_train, file_pattern_validation, file_pattern_test],
      embeddings_output_dir, step)
  logging.info('[process_single_checkpoint] Got splits metadata.')
  # Track MSE and MAE both per-split and overall.
  mse_ms = {n: tf.keras.metrics.MeanSquaredError() for n in SPLIT_NAMES_}
  mae_ms = {n: tf.keras.metrics.MeanAbsoluteError() for n in SPLIT_NAMES_}
  mse_all = tf.keras.metrics.MeanSquaredError()
  mae_all = tf.keras.metrics.MeanAbsoluteError()

  logging.info('Starting the ds loop...')
  count, ex_count = 0, 0
  s = time.time()
  for split_name, ds, emb_writer in splits_metadata:
    for outs in ds:
      if FLAGS.speaker_id_key:
        wav_samples, targets, labels, speaker_ids = outs
      else:
        wav_samples, targets, labels = outs
        speaker_ids = [None] * wav_samples.shape[0]
      wav_samples.shape.assert_is_compatible_with([None, None])
      targets.shape.assert_is_compatible_with([None, output_dim])

      embs = model(wav_samples, training=False)[model_output_key]
      embs.shape.assert_is_compatible_with(targets.shape)
      embs = tf.debugging.check_numerics(
          embs, message='Nans', name='check_numerics')

      # Update the split-specific metric and overall metric.
      for met in (mse_ms[split_name], mae_ms[split_name], mse_all, mae_all):
        met.update_state(y_true=targets, y_pred=embs)

      ex_count += embs.shape[0]
      count += 1
      logging.info('Saw %i examples after %i iterations as %.2f secs...',
                   ex_count, count,
                   time.time() - s)

      # Rather than store all embeddings in memory and write them to disk at
      # the end, let's write embeddings to disk as we generate them, if we
      # need to.
      if emb_writer:
        logging.info('Starting to write %i embeddings to disk...', ex_count)
        for emb, lbl, speaker_id in zip(embs, labels, speaker_ids):
          make_tfexample_and_write(emb, lbl, speaker_id, FLAGS.speaker_id_key,
                                   emb_writer)
        logging.info('Write %i embeddings to disk.', ex_count)
  with writer.as_default():
    suff = FLAGS.eval_suffix
    tf.summary.scalar(f'mse_all_{suff}', mse_all.result().numpy(), step=step)
    tf.summary.scalar(f'mae_all_{suff}', mae_all.result().numpy(), step=step)
    for split in SPLIT_NAMES_:
      tf.summary.scalar(
          f'mse_{split}_{suff}', mse_ms[split].result().numpy(), step=step)
      tf.summary.scalar(
          f'mae_{split}_{suff}', mae_ms[split].result().numpy(), step=step)
  for _, _, emb_writer in splits_metadata:
    if emb_writer:
      emb_writer.close()
  logging.info('Done with eval step: %i in %.2f secs.', step, time.time() - s)


def eval_and_report(output_dim = 1024,
                    model_output_key = 'embedding'):
  """Check fidelity of a dataset."""
  logging.info('Logdir: %s', FLAGS.logdir)

  writer = tf.summary.create_file_writer(FLAGS.eval_dir)
  model = models.get_keras_model(
      model_type=FLAGS.model_type, frame_hop=FLAGS.frame_hop)
  checkpoint = tf.train.Checkpoint(model=model)

  for ckpt in tf.train.checkpoints_iterator(
      FLAGS.logdir, timeout=FLAGS.timeout):
    process_single_checkpoint(
        writer=writer,
        model=model,
        checkpoint=checkpoint,
        ckpt=ckpt,
        output_dim=output_dim,
        model_output_key=model_output_key,
        embeddings_output_dir=FLAGS.embeddings_output_dir,
        file_pattern_train=FLAGS.file_pattern_train,
        file_pattern_validation=FLAGS.file_pattern_validation,
        file_pattern_test=FLAGS.file_pattern_test)


def make_tfexample_and_write(emb, onehot_lbl,
                             speaker_id,
                             speaker_id_key,
                             tfrecord_writer):
  """Create and write tf.Example from an embedding.

  This output should be able to be read by `train_and_get_score`.`

  Args:
    emb: An embedding Tensor.
    onehot_lbl: The onehot label for this embedding.
    speaker_id: Optionally, the speaker ID for this embedding.
    speaker_id_key: Optionally, the key for speaker ID.
    tfrecord_writer: An open tfrecord writer.
  """
  # New tf.Example.
  ex = tf.train.Example()

  # Add the embedding.
  ex.features.feature[f'embedding/{EMBEDDING_KEY_}'].float_list.value.extend(
      emb.numpy())

  # Add the label.
  ex.features.feature[LABEL_KEY_].bytes_list.value.append(onehot_lbl.numpy())

  # Optionally add the speaker ID.
  if speaker_id:
    ex.features.feature[speaker_id_key].bytes_list.value.append(
        speaker_id.numpy())

  tfrecord_writer.write(ex.SerializeToString())


def main(unused_argv):
  assert FLAGS.model_type
  assert FLAGS.file_pattern_train
  assert FLAGS.file_pattern_validation
  assert FLAGS.file_pattern_test
  assert FLAGS.logdir
  assert FLAGS.eval_batch_size

  assert FLAGS.target_key

  assert tf.executing_eagerly()
  eval_and_report()


if __name__ == '__main__':
  app.run(main)
