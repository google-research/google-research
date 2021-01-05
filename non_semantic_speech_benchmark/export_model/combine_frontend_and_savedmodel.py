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
r"""Exports a graph as a saved model.

"""
# pylint:enable=line-too-long

import os
from absl import app
from absl import flags
from absl import logging
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
def _sample_to_features(x, export_tflite=False):
  return tf_frontend.compute_frontend_features(
      x, 16000, overlap_seconds=79, tflite=export_tflite)


class TRILLModule(tf.train.Checkpoint):
  """TRILL module for TF 1 and 2.

  """

  def __init__(self, savedmodel_dir, distilled_output_keys, tflite):
    super(TRILLModule, self).__init__()
    self.trill_module = hub.load(savedmodel_dir)
    assert len(self.trill_module.signatures.keys()) == 1
    self.sig_key = list(self.trill_module.signatures.keys())[0]
    self.variables = self.trill_module.variables
    self.trainable_variables = self.trill_module.variables
    self.tflite = tflite

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
      if self.tflite:
        features = tf.map_fn(
            _sample_to_features, (samples, True), dtype=tf.float64)
      else:
        features = tf.map_fn(_sample_to_features, samples, dtype=tf.float64)
      assert features.shape.rank == 4
      f_shape = tf.shape(features)
    else:
      features = _sample_to_features(samples, self.tflite)
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


def make_and_export_trill(savedmodel_dir,
                          distilled_output_keys,
                          allow_batch_dimension,
                          fixed_length_input,
                          tflite_only):
  """Make and export TRILL or TRILL distilled.

  Args:
    savedmodel_dir: Directory with frontend-less SavedModel.
    distilled_output_keys: Boolean. Whether exporting the distilled model.
    allow_batch_dimension: Whether to allow batch dimensions.
    fixed_length_input: Length of input, or `None` for variable length.
    tflite_only: Whether to export models suitable for mobile inference with
      TensorFlow Lite.

  Returns:
    (signatures, module)
  """
  trill_mod = TRILLModule(
      savedmodel_dir, distilled_output_keys, tflite_only)

  signature = None
  if tflite_only:
    # For TFLite inference, we only generate float32 models with fixed input
    # and no batch dim.
    signature = trill_mod.__call__.get_concrete_function(
        tf.TensorSpec([fixed_length_input], tf.float32), tf.constant(16000))
    signatures = {'inference': signature}
  else:
    for dtype in (tf.int16, tf.float32, tf.float64):
      signature = trill_mod.__call__.get_concrete_function(
          tf.TensorSpec([fixed_length_input], dtype), tf.constant(16000))
      if allow_batch_dimension:
        trill_mod.__call__.get_concrete_function(
            tf.TensorSpec([None, fixed_length_input], dtype),
            tf.constant(16000))
    signatures = None

  return signatures, trill_mod


def convert_tflite_file(model_dir):
  """Make and export TRILL and TRILL distilled."""
  converter = tf.lite.TFLiteConverter.from_saved_model(
      saved_model_dir=model_dir, signature_keys=['inference'])

  # TODO(srjoglekar): Explore quantization later.
  converter.optimizations = []
  converter.post_training_quantize = False
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet
      .TFLITE_BUILTINS,  # enable TensorFlow Lite builtin ops only.
  ]

  output_data = converter.convert()
  output_path = os.path.join(model_dir, 'model.tflite')
  if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)
  with tf.io.gfile.GFile(output_path, 'wb') as f:
    f.write(output_data)
  return output_path


def construct_savedmodel_dir(export_dir, distilled_model, tflite,
                             allow_batch_dimension, fixed_length_input):
  name = 'trill-distilled' if distilled_model else 'trill'
  suffix = '_tflite' if tflite else ''
  bd_str = 'wbatchdim' if allow_batch_dimension else 'nobatchdim'
  fl_str = f'fixedlen{fixed_length_input}' if fixed_length_input else 'nofixedlen'
  return os.path.join(export_dir, f'{name}{suffix}_{bd_str}_{fl_str}')


def test_module(out_dir,
                allow_batch_dimension=True,
                fixed_length_input=None,
                tflite_only=False):
  """Test that the exported doesn't crash."""
  if out_dir.endswith('tflite'):
    # TODO(joelshor, srjoglekar): Load TFLite model here.
    return
  else:
    model = hub.load(out_dir)
  sr = tf.constant(16000)
  input_len = fixed_length_input or 320000
  logging.info('Input length: %s', input_len)

  proper_shape = tf.random.uniform([input_len], -1.0, 1.0, tf.float32)
  model(proper_shape, sr)
  if allow_batch_dimension:
    proper_shape = tf.random.uniform([5, input_len], -1.0, 1.0, tf.float32)
    model(proper_shape, sr)

  if not tflite_only:
    # TfLite does not support these types, and uses fixed sizes.
    proper_shape = tf.random.uniform([input_len], -1.0, 1.0, tf.float64)
    model(proper_shape, sr)
    if allow_batch_dimension:
      proper_shape = tf.random.uniform([5, input_len], -1.0, 1.0, tf.float64)
      model(proper_shape, sr)

    proper_shape = np.random.randint(
        0, high=10000, size=(input_len), dtype=np.int16)
    model(proper_shape, sr)
    if allow_batch_dimension:
      proper_shape = np.random.randint(
          0, high=10000, size=(5, input_len), dtype=np.int16)
      model(proper_shape, sr)

    if fixed_length_input is None:
      short_shape = np.random.randint(
          0, high=10000, size=(5000), dtype=np.int16)
      model(short_shape, sr)
      if allow_batch_dimension:
        short_shape = np.random.randint(
            0, high=10000, size=(5, 5000), dtype=np.int16)
        model(short_shape, sr)

      try:
        model(short_shape, tf.constant(8000))
        assert False
      except tf.errors.InvalidArgumentError:
        pass

  # Check variables.
  assert model.variables
  assert model.trainable_variables
  assert not model._is_hub_module_v1   # pylint:disable=protected-access


def main(unused_argv):

  # pylint: disable=bad-whitespace
  t_loc = FLAGS.trill_model_location
  d_loc = FLAGS.trill_distilled_model_location
  model_params = [
      # loc,  distilled  batch dim, input len,  tflite
      (t_loc, False,     True,      None,       False),
      (t_loc, False,     False,     16000,      False),
      (d_loc, True,      True,      None,       False),
      (d_loc, True,      False,     16000,      False),
      (d_loc, True,      True,      None,       True),
      (d_loc, True,      False,     None,       True),
      (d_loc, True,      False,     16000,      True),
  ]
  # pylint: enable=bad-whitespace

  for (model_location, distilled_output_keys, allow_batch_dimension,
       fixed_length_input, export_tflite) in model_params:
    signatures, saved_mod = make_and_export_trill(
        model_location,
        distilled_output_keys,
        allow_batch_dimension,
        fixed_length_input,
        export_tflite)
    if not export_tflite:
      assert signatures is None, signatures
    out_dir = construct_savedmodel_dir(
        FLAGS.export_dir, distilled_output_keys, export_tflite,
        allow_batch_dimension, fixed_length_input)
    tf.saved_model.save(saved_mod, out_dir, signatures)
    if export_tflite:
      tflite_out_file = convert_tflite_file(out_dir)
      test_module(
          tflite_out_file,
          allow_batch_dimension,
          fixed_length_input,
          tflite_only=True)
    test_module(
        out_dir,
        allow_batch_dimension,
        fixed_length_input,
        tflite_only=export_tflite)


if __name__ == '__main__':
  flags.mark_flag_as_required('export_dir')
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  app.run(main)
