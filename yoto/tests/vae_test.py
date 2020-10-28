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

"""Tests for yoto.problems.vae."""

from absl.testing import parameterized

import tensorflow.compat.v1 as tf

from yoto.problems import vae as vae_mod


class SimpleMlpModel(tf.keras.Model):

  def __init__(self, dimensions):
    super(SimpleMlpModel, self).__init__()
    self._model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(dimension) for dimension in dimensions])

  def call(self, inputs, inputs_extra):
    if inputs_extra is not None:
      inputs = tf.concat((inputs, inputs_extra), axis=1)
    return self._model(inputs), {}


class VaeYotoTest(parameterized.TestCase,
                  tf.test.TestCase):

  @parameterized.parameters(
      dict(observation_dims=5,
           latent_dims=2,
           conditioning_size=12,
           batch_size=12,
           should_fail=True),
      dict(observation_dims=5,
           latent_dims=2,
           conditioning_size=12,
           batch_size=12,
           should_fail=False),)
  def test_sizes(self,
                 observation_dims,
                 latent_dims,
                 conditioning_size,
                 batch_size,
                 should_fail):

    def create_encoder():
      return SimpleMlpModel((observation_dims + conditioning_size,
                             2 * latent_dims))

    def create_decoder():
      return SimpleMlpModel((latent_dims + conditioning_size,
                             2 * observation_dims))

    vae = vae_mod.ConditionalVaeProblem(
        create_encoder, create_decoder, latent_dims)
    vae.initialize_model()
    input_data = tf.ones((batch_size, observation_dims))
    if conditioning_size:
      if should_fail:
        conditioning_data = tf.ones((batch_size + 1, conditioning_size))
      else:
        conditioning_data = tf.ones((batch_size, conditioning_size))
    else:
      conditioning_data = None
    if should_fail:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        vae.losses_and_metrics({"image": input_data}, conditioning_data)
    else:
      losses, _ = vae.losses_and_metrics({"image": input_data},
                                         conditioning_data)
      self.assertIn("reconstruction_loss", losses)
      self.assertIn("kl_loss", losses)
      self.assertEqual(losses["reconstruction_loss"].shape.as_list(),
                       [batch_size,])
      self.assertEqual(losses["kl_loss"].shape.as_list(),
                       [batch_size,])


if __name__ == "__main__":
  tf.test.main()
