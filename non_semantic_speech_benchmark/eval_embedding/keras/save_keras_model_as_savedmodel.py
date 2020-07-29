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

# pylint:disable=line-too-long
r"""Saves a trained model as a SavedModel.

Takes the result of a model trained from this directory and saves it as a
SavedModel, suitable for running inference on wavs.

"""
# pylint:enable=line-too-long

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

from non_semantic_speech_benchmark.eval_embedding.keras import models


FLAGS = flags.FLAGS


flags.DEFINE_integer('num_classes', None, 'Number of classes.')
flags.DEFINE_integer('num_clusters', None, 'num_clusters')
flags.DEFINE_boolean('use_batch_normalization', None, 'Whether to normalize')
flags.DEFINE_string('checkpoint_filename', None,
                    'Location of the checkpoint. Of the form '
                    '`/dir/ckpt-83800`.')
flags.DEFINE_string(
    'embedding_tfhub_handle',
    'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3',
    'Hub handle for generating embeddings.')
flags.DEFINE_string(
    'output_dir', None, 'Directory to save final SavedModel.')


def _load_keras_model(num_classes, use_batch_normalization, num_clusters,
                      checkpoint_filename):
  """Load the model, and make sure weights have been loaded properly."""
  dummy_input = tf.random.uniform([3, 100, 2048])

  agg_model = models.get_keras_model(
      num_classes, use_batch_normalization, num_clusters=num_clusters)
  checkpoint = tf.train.Checkpoint(model=agg_model)

  o1 = agg_model(dummy_input)
  checkpoint.restore(checkpoint_filename)
  o2 = agg_model(dummy_input)

  assert not np.allclose(o1, o2)

  return agg_model


def _combine_keras_model_with_trill(embedding_tfhub_handle, aggregating_model):
  """Combines keras model with TRILL model."""
  trill_layer = hub.KerasLayer(
      handle=embedding_tfhub_handle,
      trainable=False,
      arguments={'sample_rate': 16000},
      output_key='embedding',
      output_shape=[None, 2048]
  )
  input1 = tf.keras.Input([None])
  trill_output = trill_layer(input1)
  final_out = aggregating_model(trill_output)

  final_model = tf.keras.Model(
      inputs=input1,
      outputs=final_out)

  return final_model


def main(unused_argv):
  aggregating_model = _load_keras_model(
      FLAGS.num_classes, FLAGS.use_batch_normalization,
      FLAGS.num_clusters, FLAGS.checkpoint_filename)
  logging.info('Created and loaded aggregating Keras model.')

  combined_model = _combine_keras_model_with_trill(
      FLAGS.embedding_tfhub_handle, aggregating_model)
  logging.info('Created and combined model with TRILL.')

  tf.keras.models.save_model(
      combined_model,
      FLAGS.output_dir,
      include_optimizer=False)
  logging.info('Saved model to disk.')

  # Sanity check the output directory.
  m = tf.keras.models.load_model(FLAGS.output_dir)
  m(tf.random.uniform([5, 32000]))  # Just check that this doesn't crash.
  logging.info('Sanity checked saved model.')


if __name__ == '__main__':
  tf.enable_v2_behavior()
  assert tf.executing_eagerly()
  flags.mark_flags_as_required([
      'num_classes', 'num_clusters', 'use_batch_normalization',
      'checkpoint_filename', 'embedding_tfhub_handle', 'output_dir'])
  app.run(main)
