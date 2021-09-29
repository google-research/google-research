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
# pylint:disable=line-too-long
r"""Counts average audio length.

"""
# pylint:enable=line-too-long

import os
from typing import Any, Dict, Iterable, List, Tuple

from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
import numpy as np
import tensorflow as tf

from non_semantic_speech_benchmark.data_prep import audio_to_embeddings_beam_utils

flags.DEFINE_string('output_file', None, 'Output file.')
flags.DEFINE_boolean('debug', False, 'Whether to debug.')
flags.DEFINE_list(
    'audio_keys', ['audio', 'processed/audio_samples', 'audio_waveform',
                   'WAVEFORM/feature/floats'],
    'Possible audio keys in tf.Examples.')
flags.DEFINE_list(
    'sr_keys', [], 'Possible sample rate keys in tf.Examples.')

FLAGS = flags.FLAGS


def duration_from_tfex(k_v):
  """Duration from a tf.Example."""
  k, ex = k_v

  audio_vals = []
  for possible_audio_key in FLAGS.audio_keys:
    if possible_audio_key in ex.features.feature:
      logging.info('Found audio key: %s', possible_audio_key)
      audio_feats = ex.features.feature[possible_audio_key]
      cur_audio_vals = (audio_feats.int64_list.value or
                        audio_feats.float_list.value)
      assert cur_audio_vals
      audio_vals.append(cur_audio_vals)
  assert len(audio_vals) == 1, ex
  audio_vals = audio_vals[0]
  logging.info('%s audio: %s', k, audio_vals)

  sr_vals = []
  for possible_sr_key in FLAGS.sr_keys:
    if possible_sr_key in ex.features.feature:
      logging.info('Found sample rate key: %s', possible_sr_key)
      cur_audio = ex.features.feature[possible_sr_key].int64_list.value[0]
      sr_vals.append(cur_audio)
  assert len(sr_vals) in [0, 1], ex
  if len(sr_vals) == 1:
    sr = sr_vals[0]
  else:
    logging.info('Used default sr.')
    sr = 16000

  return len(audio_vals) / float(sr)


def durations(root, ds_file, ds_name,
              reader_type, suffix):
  """Beam pipeline for durations from a particular file or glob."""
  logging.info('Reading from %s: (%s, %s)', reader_type, ds_name, ds_file)
  input_examples = audio_to_embeddings_beam_utils.reader_functions[reader_type](
      root, ds_file, f'Read-{suffix}')
  return input_examples | f'Lens-{suffix}' >> beam.Map(duration_from_tfex)


def duration_and_num_examples(
    root, ds_files, ds_name,
    reader_type):
  """Beam pipeline for durations from a list of files or globs."""
  durations_l = []
  for i, ds_file in enumerate(ds_files):
    cur_dur = durations(
        root, ds_file, ds_name, reader_type, suffix=f'{ds_name}_{i}')
    durations_l.append(cur_dur)
  def _mean_and_count(durs):
    return np.mean(durs), len(durs)
  return (durations_l
          | f'Flatten-{ds_name}' >> beam.Flatten()
          | f'ToList-{ds_name}' >> beam.combiners.ToList()
          | f'Stats-{ds_name}' >> beam.Map(_mean_and_count))


def get_dataset_info_dict(debug):
  """Get dictionary of dataset info."""
  def _tfds_fns(ds_name):
    fns = [
        x  # pylint:disable=g-complex-comprehension
        for s in ('train', 'validation', 'test')
        for x in audio_to_embeddings_beam_utils._tfds_filenames(ds_name, s)]  # pylint:disable=protected-access
    fns = [fns]  # TFRecords require a list.
    return (fns, 'tfrecord')

  if debug:
    dss = {'savee': _tfds_fns('savee')}
  else:
    dss = {
        'crema_d': _tfds_fns('crema_d'),
        'savee': _tfds_fns('savee'),
        'speech_commands': _tfds_fns('speech_commands'),
        'voxceleb': _tfds_fns('voxceleb'),
    }

  return dss


def main(unused_argv):
  dss = get_dataset_info_dict(FLAGS.debug)

  out_file = FLAGS.output_file
  assert not tf.io.gfile.exists(out_file)
  if not tf.io.gfile.exists(os.path.dirname(out_file)):
    tf.io.gfile.makedirs(os.path.dirname(out_file))

  pipeline_option = None

  with beam.Pipeline(pipeline_option) as root:
    stats = []  # (ds name, avg duration, num examples)
    for ds_name, (ds_files, reader_type) in dss.items():
      cur_stat = duration_and_num_examples(root, ds_files, ds_name, reader_type)
      cur_stat = cur_stat | f'AddName-{ds_name}' >> beam.Map(
          lambda x, name: (name, x[0], x[1]), name=ds_name)
      stats.append(cur_stat)
    # Write output.
    _ = (
        stats
        | 'CombineDSes' >> beam.Flatten()
        | 'ToStr' >> beam.Map(lambda x: ','.join([str(r) for r in x]))
        | 'WriteOutput' >> beam.io.WriteToText(out_file, num_shards=1))


if __name__ == '__main__':
  flags.mark_flag_as_required('output_file')
  app.run(main)
