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
"""Scorer for speech data based on DeepSpeech2."""

from . import scorer

from absl import logging
from . ds2_encoder import DS2Encoder
import tensorflow.compat.v2 as tf


def load_gru_cell(cell, reader, prefix):
  """Load weights of a CuDNN GRU checkpoint into a K.layers.GRUCell.

  Args:
    cell: K.layers.GRUCell. Cell to load weights into.
    reader: CheckpointReader. Checkpoint reader used for loading individual
      tensors by name.
    prefix: Str. Prefix for the cell's tensors in the checkpoint reader.
  """
  # Load GRU-style weights.
  gates_kernel = reader.get_tensor(prefix + '/gates/kernel')
  gates_bias = reader.get_tensor(prefix + '/gates/bias')
  inputs_kernel = reader.get_tensor(
      prefix + '/candidate/input_projection/kernel')
  inputs_bias = reader.get_tensor(prefix + '/candidate/input_projection/bias')
  hidden_kernel = reader.get_tensor(
      prefix + '/candidate/hidden_projection/kernel')
  hidden_bias = reader.get_tensor(prefix + '/candidate/hidden_projection/bias')

  # Construct new weights.
  num_units = cell.units
  kernel = tf.concat([gates_kernel[:-num_units, num_units:],
                      gates_kernel[:-num_units, :num_units],
                      inputs_kernel], axis=-1)
  recurrent_kernel = tf.concat([gates_kernel[-num_units:, num_units:],
                                gates_kernel[-num_units:, :num_units],
                                hidden_kernel], axis=-1)
  bias_0 = tf.concat(
      [gates_bias[num_units:], gates_bias[:num_units], inputs_bias], axis=-1)
  bias_1 = tf.concat([tf.zeros_like(gates_bias), hidden_bias], axis=-1)
  bias = tf.stack([bias_0, bias_1])

  kernel = tf.cast(kernel, tf.float32)
  recurrent_kernel = tf.cast(recurrent_kernel, tf.float32)
  bias = tf.cast(bias, tf.float32)

  # Assign weights.
  cell.kernel.assign(kernel)
  cell.recurrent_kernel.assign(recurrent_kernel)
  cell.bias.assign(bias)


def load_d2_encoder(encoder, reader):
  """Load the DS2 encoder model weights from a checkpoint reader."""
  # Convolutions and batch norm.
  for i in range(2):
    prefix = 'ForwardPass/ds2_encoder/conv%d' % (i + 1)

    kernel = reader.get_tensor(prefix + '/kernel')
    encoder.conv[i].kernel.assign(kernel)

    moving_mean = reader.get_tensor(prefix + '/bn/moving_mean')
    moving_variance = reader.get_tensor(prefix + '/bn/moving_variance')
    beta = reader.get_tensor(prefix + '/bn/beta')
    gamma = reader.get_tensor(prefix + '/bn/gamma')

    encoder.bn[i].moving_mean.assign(moving_mean)
    encoder.bn[i].moving_variance.assign(moving_variance)
    encoder.bn[i].beta.assign(beta)
    encoder.bn[i].gamma.assign(gamma)

  # GRUs.
  gru_prefix = 'ForwardPass/ds2_encoder/cudnn_gru/stack_bidirectional_rnn'
  fw_prefix = (
      gru_prefix + '/cell_%i/bidirectional_rnn/fw/cudnn_compatible_gru_cell')
  bw_prefix = (
      gru_prefix + '/cell_%i/bidirectional_rnn/bw/cudnn_compatible_gru_cell')

  for i in range(5):
    load_gru_cell(
        encoder.gru_stack[i].forward_layer.cell, reader, fw_prefix % i)
    load_gru_cell(
        encoder.gru_stack[i].backward_layer.cell, reader, bw_prefix % i)

  # Fully-connected.
  kernel = reader.get_tensor('ForwardPass/ds2_encoder/fully_connected/kernel')
  bias = reader.get_tensor('ForwardPass/ds2_encoder/fully_connected/bias')
  kernel, bias = tf.cast(kernel, tf.float32), tf.cast(bias, tf.float32)

  encoder.fully_connected.kernel.assign(kernel)
  encoder.fully_connected.bias.assign(bias)

  # Fully-connected CTC.
  kernel = reader.get_tensor(
      'ForwardPass/fully_connected_ctc_decoder/fully_connected/kernel')
  bias = reader.get_tensor(
      'ForwardPass/fully_connected_ctc_decoder/fully_connected/bias')
  kernel, bias = tf.cast(kernel, tf.float32), tf.cast(bias, tf.float32)

  encoder.fully_connected_ctc.kernel.assign(kernel)
  encoder.fully_connected_ctc.bias.assign(bias)


class DS2Scorer(scorer.NoOpScorer):
  """Speech scorer based on DeepSpeech2."""

  def __init__(self, ckpt_filename):
    """Constructor.

    Args:
      ckpt_filename: Str. Checkpoint filename.
    """
    super(DS2Scorer, self).__init__()

    self.infer = DS2Encoder()
    self.ckpt_filename = ckpt_filename
    self.is_restored = False

    self.collect_names = ['pooled_activations']

  def restore(self, ckpt_filename=None):
    """Restore the model from a checkpoint.

    Args:
      ckpt_filename: Str. Checkpoint filename.

    Returns:
      Nothing.
    """
    ckpt_filename = ckpt_filename or self.ckpt_filename

    # Ensure that all variables were created by pushing a batch through.
    # TODO(agritsenko): Creating the module using a hardcoded shape may not work
    # for all models. We should instead create and restore at call - not
    # sure how to do that.
    self.infer(waves=tf.ones([16, 48000, 1]))

    # Restore the variables.
    reader = tf.train.load_checkpoint(ckpt_filename)
    load_d2_encoder(self.infer, reader)

    self.is_restored = True

  def compute_scores(self, real_samples=None, fake_samples=None):
    """Computes the Frechet DeeepSpeech Distance (FDSD).

    Args:
      real_samples: Dictionary of numpy arrays. Selected output of the
        inferer on real samples to be used for score computation.
      fake_samples: Dictionary of numpy arrays. Selected output of the
        inferer on fake samples to be used for score computation.

    Returns:
      Tuple of dictionaries with scores computer for real and fake examples.
    """
    if real_samples is None or fake_samples is None:
      raise ValueError(
          'Both sample set must be provided for FDSD score computation.')

    for name in self.collect_names:
      if name not in real_samples:
        raise ValueError('Missing output (%s) in provided real samples.' % name)
      if name not in fake_samples:
        raise ValueError('Missing output (%s) in provided fake samples.' % name)

    if not self.is_restored:
      self.restore()

    # (Un)conditional Frechet DeepSpeech distance.
    n_fake_samples = len(fake_samples['pooled_activations'])
    n_real_samples = len(real_samples['pooled_activations'])
    n_samples = min(n_fake_samples, n_real_samples)
    if n_real_samples != n_fake_samples:
      logging.warning(
          'Number of real (%d) and fake (%d) samples do not match. '
          'Computing FDSD from the first (%s) samples.',
          n_real_samples, n_fake_samples, n_samples)

    mu_real, sigma_real = scorer.compute_mean_and_covariance(
        real_samples['pooled_activations'][:n_samples])
    mu_fake, sigma_fake = scorer.compute_mean_and_covariance(
        fake_samples['pooled_activations'][:n_samples])

    return {'FDSD': scorer.compute_frechet_inception_distance(
        mu_fake, sigma_fake, mu_real, sigma_real)}
