# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for entropy_model."""

from typing import Optional
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf

from vct.src import bottlenecks
from vct.src import entropy_model
from vct.src import transformer_layers


def add_batch_channel(t, num_chan):
  return tf.constant(t, tf.float32)[:, tf.newaxis] * tf.ones(
      (4, 1, num_chan))


def fake_shift_to_the_right(x,
                            pad = None):
  """Version of shift_to_the_right that does not actually shift."""
  del pad  # Unused.
  return x


def _base_test_config(**kwargs):
  base = dict(
      num_channels=8,
      context_len=2,
      window_size_enc=6,
      window_size_dec=4,
      num_layers_encoder_sep=1,
      num_layers_encoder_joint=1,
      num_layers_decoder=2,
      d_model=16,
      num_head=2,
      mlp_expansion=2)
  base.update(**kwargs)
  return base


def _test_configs():
  yield _base_test_config()


class FastConditionalLocScaleShiftBottleneck(
    bottlenecks.ConditionalLocScaleShiftBottleneck):
  """Fast version of ConditionalLocScaleShiftBottleneck.

  The default kwargs will induce a O(minutes) wait time to make compression
  tables. Here, we reduce the number of entries in the table to 15.
  """

  def __init__(self, *args, **kwargs):
    kwargs["num_means"] = 3
    kwargs["num_scales"] = 5
    super().__init__(*args, **kwargs)


class SmokeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.product(training=(True, False), config=list(_test_configs()))
  def test_call(self, training, config):
    model = entropy_model.VCTEntropyModel(**config)
    d = 0 if training else 7  # Test padding.
    latent = tf.random.uniform((8, 16 + d, 16 + d, 8))
    previous = [tf.random.uniform((8, 16 + d, 16 + d, 8)),
                tf.random.uniform((8, 16 + d, 16 + d, 8))]
    previous = [
        model.process_previous_latent_q(p, training) for p in previous
    ]
    model.validate_causal(latent, previous)
    out = tf.function(model.__call__)(
        latent_unquantized=latent, previous_latents=previous, training=training)
    self.assertEqual(latent.shape, out.perturbed_latent.shape)

    @tf.function
    def validate_causal_tffunction(latent, previous):
      return model.validate_causal(latent, previous)

    validate_causal_tffunction(latent, previous)


class VCTEntropyModelTest(parameterized.TestCase, tf.test.TestCase):

  @mock.patch.object(bottlenecks, "ConditionalLocScaleShiftBottleneck",
                     FastConditionalLocScaleShiftBottleneck)
  def test_range_code(self):
    model = entropy_model.VCTEntropyModel(**_base_test_config())
    latent = tf.random.uniform((1, 8, 8, 8))
    previous = [
        tf.random.uniform((1, 8, 8, 8)),
        tf.random.uniform((1, 8, 8, 8))
    ]
    previous = [
        model.process_previous_latent_q(p, training=False) for p in previous
    ]
    model.prepare_for_range_coding()
    out = model.range_code(latent_unquantized=latent, previous_latents=previous,
                           run_decode=True)
    self.assertEqual(latent.shape, out.perturbed_latent.shape)

  def test_causal_no_mask(self):
    """Make sure causality test raises if we do not use a mask."""
    def no_mask(size, free=0):
      del free  # Unused.
      return tf.zeros((size, size))

    with mock.patch.object(
        transformer_layers, "create_look_ahead_mask", new=no_mask):
      model = entropy_model.VCTEntropyModel(**_base_test_config())

    latent = tf.random.uniform((8, 16, 16, 8))
    previous = [
        model.process_previous_latent_q(
            tf.random.uniform((8, 16, 16, 8)), training=False)
    ]
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "Expected.+to be true"):
      model.validate_causal(latent, previous)


if __name__ == "__main__":
  tf.test.main()
