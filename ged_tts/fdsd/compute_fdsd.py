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

"""Frechet Deep Speech Distance metric calculator.

This is our re-implementation of the metrics introduced in Bińkowski *et al.*
"High fidelity speech synthesis with adversarial networks." arXiv preprint
arXiv:1909.11646 (2019).
"""

import os
from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf
import tqdm
import ged_tts.fdsd.eval as mod_eval


flags.DEFINE_string(
    'sample1', None, 'Directory with the first set of audio samples.')
flags.DEFINE_string(
    'sample2', None, 'Directory with the second set of audio samples.')
flags.DEFINE_string('ds2_ckpt',
                    'ds2_large/model.ckpt-54800',
                    'Path to the DeepSpeech2 checkpoint.')
flags.DEFINE_integer('batch_size', 64, 'Batch size for the evaluator.')


FLAGS = flags.FLAGS


def make_wave_dataset(path, batch_size):
  """Create a WAV dataset from audio in a give directory."""
  file_pattern = os.path.join(path, '*.wav')
  files = tf.data.Dataset.list_files(file_pattern, shuffle=False, seed=1234)

  def _parse_file(filename):
    """Read and parse an audio file."""
    wave_bytes = tf.io.read_file(filename)
    wave, _ = tf.audio.decode_wav(wave_bytes)
    return wave

  n_samples = tf.data.experimental.cardinality(files).numpy()
  ds = files.map(_parse_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=False)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds, n_samples


def collect_activations(scorer, ds, n_samples, name):
  """Collect activations using a scorer and an input dataset."""
  @tf.function
  def eval_fn(inputs):
    """Single evaluation step of the generator."""
    return scorer.infer(inputs, training=False)

  samples = {name: [] for name in scorer.collect_names}
  it = iter(ds)

  with tqdm.tqdm(total=n_samples, desc=name) as pbar:
    while True:
      try:
        activations = eval_fn(next(it))
      except StopIteration:
        break

      n_activations = -1
      activations = {name: value.numpy() for name, value in activations.items()
                     if name in scorer.collect_names}
      activations = {name: np.split(value, value.shape[0], axis=0)
                     for name, value in activations.items()}
      for name, value in activations.items():
        samples[name].extend([sample[0, Ellipsis] for sample in value])
        n_activations = len(value)

      # Update the progress bar.
      pbar.update(n_activations)
  return samples


def main(argv):
  """Run the main scorer."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Init the IO pipelines.
  if FLAGS.sample1 is None or FLAGS.sample2 is None:
    raise app.UsageError('Both sample set directories must be specified.')

  print('[i] Constructing datasets.')
  ds_sample1, n_sample1 = make_wave_dataset(FLAGS.sample1, FLAGS.batch_size)
  ds_sample2, n_sample2 = make_wave_dataset(FLAGS.sample2, FLAGS.batch_size)

  if n_sample1 != n_sample2:
    raise app.UsageError('Both sample sets must have the same number of waves '
                         '(%d and %d).' % (n_sample1, n_sample2))

  # Create a scorer instance and load the checkpoint.
  print('[i] Restoring DS2 checkpoint.')
  scorer = mod_eval.DS2Scorer(FLAGS.ds2_ckpt)
  scorer.restore()

  # Collect samples.
  print('[i] Collecting samples.')
  sample1 = collect_activations(scorer, ds_sample1, n_sample1, 'Sample 1')
  sample2 = collect_activations(scorer, ds_sample2, n_sample2, 'Sample 2')

  print('[i] Computing metrics.')
  metrics = scorer.compute_scores(sample1, sample2)
  print('    [i] FDSD: %.3f' % metrics['FDSD'])


if __name__ == '__main__':
  app.run(main)
