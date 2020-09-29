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

# Lint as: python3
# pylint:disable=line-too-long
r"""Exports a graph as a saved model.

"""
# pylint:enable=line-too-long

import os
from absl import app
from absl import flags
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from non_semantic_speech_benchmark.export_model import tf_frontend


flags.DEFINE_string('export_dir', None, 'Location and name of SavedModel.')

flags.DEFINE_string('trill_model_location', None,
                    'Location of TRILL SavedModel, with no frontend.')
flags.DEFINE_string('trill_distilled_model_location', None,
                    'Location of TRILL-distilled SavedModel, with no frontend.')


FLAGS = flags.FLAGS


@tf.function
def _sample_to_features(x):
  return tf_frontend.compute_frontend_features(x, 16000, overlap_seconds=79)


class TRILLModule(tf.train.Checkpoint):
  """TRILL module for TF 1 and 2.

  """

  def __init__(self, savedmodel_dir, distilled_output_keys):
    super(TRILLModule, self).__init__()
    self.trill_module = hub.load(savedmodel_dir)
    assert len(self.trill_module.signatures.keys()) == 1
    self.sig_key = list(self.trill_module.signatures.keys())[0]
    self.variables = self.trill_module.variables
    self.trainable_variables = self.trill_module.variables

    self.distilled_output_keys = distilled_output_keys

  @tf.function
  def __call__(self, samples, sample_rate):
    """Runs model.

    Args:
      samples: A 1-D or 2-D array. If integers, is automatically cast as a
        float.
      sample_rate: Sample rate. Must be 16 kHz.

    Returns:
      A dictionary of embeddings.
    """
    tf.debugging.assert_equal(
        sample_rate, 16000, message='Sample rate must be 16kHz. '
        'Instead, was %s' % sample_rate)
    if samples.shape.ndims > 2:
      raise ValueError('Samples must be 1 or 2 dimensional. Instead, found %s' %
                       samples.shape.ndims)
    has_batchdim = samples.shape.ndims == 2

    # Compute frontend features.
    assert isinstance(samples, tf.Tensor)
    if has_batchdim:
      features = tf.map_fn(
          _sample_to_features,
          samples,
          dtype=tf.float64)
      assert features.shape.rank == 4
      f_shape = tf.shape(features)
    else:
      features = _sample_to_features(samples)
      assert features.shape.rank == 3
      f_shape = tf.shape(features)

    # Cast features to tf.float32, if necessary.
    if features.dtype == tf.float64:
      features = tf.cast(features, tf.float32)

    # Reshape batch dimension, if necessary, and run inference.
    def _maybe_unbatch(f):
      if has_batchdim:
        return tf.reshape(f, [f_shape[0] * f_shape[1], f_shape[2], f_shape[3]])
      else:
        return f
    def _maybe_batch(n):
      assert n.shape.rank == 2
      if has_batchdim:
        feat_dim = n.shape[-1]
        out = tf.reshape(n, [f_shape[0], -1, feat_dim])
        out.set_shape([None, None, feat_dim])
        return out
      else:
        return n
    net_endpoints = self.trill_module.signatures[self.sig_key](
        _maybe_unbatch(features))
    if self.distilled_output_keys:
      emb = net_endpoints['tower0/network/layer26/embedding']
      out_dict = dict(embedding=_maybe_batch(emb))
    else:
      layer19 = tf.keras.backend.batch_flatten(
          net_endpoints['tower0/network/layer19/chain1/layer0/conv/BiasAdd'])
      layer19.set_shape([None, 12288])
      layer19 = _maybe_batch(layer19)
      out_dict = dict(
          layer19=layer19,
          embedding=_maybe_batch(net_endpoints['normalizing']))

    return out_dict


def _make_and_export_trill(savedmodel_dir, name, distilled_output_keys):
  """Make and export TRILL and TRILL distilled."""
  trill_mod = TRILLModule(savedmodel_dir, distilled_output_keys)
  for dtype in (tf.float32, tf.float64, tf.int16):
    trill_mod.__call__.get_concrete_function(
        tf.TensorSpec([None], dtype),
        tf.constant(16000))
    trill_mod.__call__.get_concrete_function(
        tf.TensorSpec([None, None], dtype),
        tf.constant(16000))

  out_dir = os.path.join(FLAGS.export_dir, name)
  tf.saved_model.save(trill_mod, out_dir)

  return out_dir


def _test_module(out_dir, allow_batchdim=False, has_variables=False):
  """Test that the exported doesn't crash."""
  model = hub.load(out_dir)
  sr = tf.constant(16000)
  proper_shape = tf.random.uniform([32000], -1.0, 1.0, tf.float32)
  model(proper_shape, sr)
  if allow_batchdim:
    proper_shape = tf.random.uniform([5, 32000], -1.0, 1.0, tf.float32)
    model(proper_shape, sr)

  proper_shape = tf.random.uniform([32000], -1.0, 1.0, tf.float64)
  model(proper_shape, sr)
  if allow_batchdim:
    proper_shape = tf.random.uniform([5, 32000], -1.0, 1.0, tf.float64)
    model(proper_shape, sr)

  proper_shape = np.random.randint(0, high=10000, size=(32000), dtype=np.int16)
  model(proper_shape, sr)
  if allow_batchdim:
    proper_shape = np.random.randint(
        0, high=10000, size=(5, 32000), dtype=np.int16)
    model(proper_shape, sr)

  short_shape = np.random.randint(0, high=10000, size=(5000), dtype=np.int16)
  model(short_shape, sr)
  if allow_batchdim:
    short_shape = np.random.randint(
        0, high=10000, size=(5, 5000), dtype=np.int16)
    model(short_shape, sr)

  try:
    model(short_shape, tf.constant(8000))
    assert False
  except tf.errors.InvalidArgumentError:
    pass

  if has_variables:
    assert model.variables
    assert model.trainable_variables
  assert not model._is_hub_module_v1   # pylint:disable=protected-access


def main(unused_argv):

  out_dir = _make_and_export_trill(
      FLAGS.trill_model_location, 'trill',
      distilled_output_keys=False)
  _test_module(out_dir, allow_batchdim=True, has_variables=True)

  out_dir = _make_and_export_trill(
      FLAGS.trill_distilled_model_location, 'trill-distilled',
      distilled_output_keys=True)
  _test_module(out_dir, allow_batchdim=True, has_variables=True)


if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  app.run(main)
