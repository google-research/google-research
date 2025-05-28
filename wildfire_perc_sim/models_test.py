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

"""Tests for models."""
import functools as ft
from typing import Any, Callable

from absl.testing import absltest

import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random

from wildfire_perc_sim import models


def _create_model(prng, model_fn,
                  *inputs):
  prng, key = random.split(prng)
  model = model_fn()
  variables = model.init(key, *inputs)
  states, params = flax.core.pop(variables, 'params')

  def apply_fn(params, *x, **kwargs):
    return model.apply({'params': params, **states}, *x, **kwargs)

  return apply_fn, model, params, states


class ModelsTest(absltest.TestCase):

  def test_ResNetEncoder(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    x = jnp.zeros((1, 128, 128, 3))
    apply_fn, model, params, _ = _create_model(
        key,
        ft.partial(
            models.ResNetEncoder, stage_sizes=(2, 2, 2, 2), latent_dim=128), x)

    y = apply_fn(params, x)
    self.assertEqual(y.shape, (y.shape[0], model.latent_dim))

    apply_fn, model, params, _ = _create_model(
        key,
        ft.partial(
            models.ResNetEncoder,
            stage_sizes=(2, 2, 2, 2),
            embedding_dimension=(5, 5, 16)), x)

    y = apply_fn(params, x)
    self.assertEqual(y.shape, (y.shape[0], 5, 5, 16))

  def test_ResNetDecoder(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    x = jnp.zeros((1, 128))
    image_size = (128, 128, 3)
    apply_fn, _, params, _ = _create_model(
        key,
        ft.partial(
            models.ResNetDecoder,
            stage_sizes=(2, 2, 2, 2),
            image_size=image_size,
            num_filters=512), x)

    y = apply_fn(params, x)
    self.assertEqual(y.shape, (x.shape[0], *image_size))

  def test_EncoderConvLSTM(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    x = (jnp.zeros((1, 128, 128, 3)), jnp.zeros(
        (1, 128, 128, 3)), jnp.zeros((1, 128, 128, 3)))
    apply_fn, model, params, _ = _create_model(
        key,
        ft.partial(
            models.EncoderConvLSTM,
            features=3,
            encoding_dim=128,
            memory_shape=(128, 128, 3),
            kernel_shape=(3, 3)), x)

    y = apply_fn(params, x)
    self.assertEqual(y.shape, (y.shape[0], model.encoding_dim))

  def test_ConditionalEncoderConv(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    x = jnp.zeros((1, 128, 128, 3))
    c = jnp.zeros((1, 128))
    encoder_block = ft.partial(
        models.ResNetEncoder, stage_sizes=(2, 2, 2, 2), latent_dim=128)
    apply_fn, model, params, _ = _create_model(
        key,
        ft.partial(
            models.ConditionalEncoderConv,
            latent_dim=128,
            encoder_block=encoder_block), x, c)

    y = apply_fn(params, x, c)
    self.assertEqual(y.shape, (y.shape[0], model.latent_dim))

  def test_ConditionalDecoderConv(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    x = jnp.zeros((1, 128))
    c = jnp.zeros((1, 128))
    image_size = (128, 128, 3)
    decoder_block = ft.partial(
        models.ResNetDecoder,
        stage_sizes=(2, 2, 2, 2),
        image_size=image_size,
        num_filters=512)
    apply_fn, _, params, _ = _create_model(
        key,
        ft.partial(models.ConditionalDecoderConv, decoder_block=decoder_block),
        x, c)

    y = apply_fn(params, x, c)
    self.assertEqual(y.shape, (y.shape[0], *image_size))

  def test_ConditionalVAEConv(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    x = jnp.zeros((2, 128, 128, 3))
    c = (jnp.zeros((2, 128, 128, 3)), jnp.zeros(
        (2, 128, 128, 3)), jnp.zeros((2, 128, 128, 3)))
    latent_dim = 128

    encoder_block = ft.partial(
        models.ResNetEncoder, stage_sizes=(2, 2, 2, 2), latent_dim=128)
    conditional_encoder_block = ft.partial(
        models.ConditionalEncoderConv,
        latent_dim=latent_dim,
        encoder_block=encoder_block)

    conditional_block = ft.partial(
        models.EncoderConvLSTM,
        features=3,
        encoding_dim=latent_dim,
        memory_shape=x.shape[1:],
        kernel_shape=(3, 3))

    decoder_block = ft.partial(
        models.ResNetDecoder,
        stage_sizes=(2, 2, 2, 2),
        image_size=x.shape[1:],
        num_filters=512)
    conditional_decoder_block = ft.partial(
        models.ConditionalDecoderConv, decoder_block=decoder_block)

    prng, key = random.split(prng)
    apply_fn, model, params, states = _create_model(
        key,
        ft.partial(
            models.ConditionalVAEConv,
            latent_dim=latent_dim,
            conditional_encoder_block=conditional_encoder_block,
            conditional_decoder_block=conditional_decoder_block,
            conditional_block=conditional_block), x, c, key)

    y, mean, logvar = apply_fn(params, x, c, key)
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(mean.shape, (x.shape[0], latent_dim // 2))
    self.assertEqual(logvar.shape, (x.shape[0], latent_dim // 2))

    prng, key = random.split(prng)
    nsamples = 16
    variables = {'params': params, **states}
    samples = model.apply(variables, nsamples, c, key, method=model.sample)

    self.assertEqual(samples.shape, (nsamples, *x.shape))

  def test_ConvolutionOperatorPredictor(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    x = jnp.zeros((2, 128, 128, 3))

    predictor = ft.partial(models.ResNetEncoder, stage_sizes=(2, 2, 2, 2))
    op_predictor = ft.partial(
        models.ConvolutionOperatorPredictor,
        num_kernels=5,
        window_size=(3, 3),
        predictor=predictor,
        num_channels=16)

    apply_fn, _, params, _ = _create_model(key, op_predictor, x)

    y = apply_fn(params, x)
    self.assertEqual(y.shape, (5, x.shape[0], 3, 3, x.shape[3], 16))

  def test_PercolationPropagator(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    x = jnp.zeros((2, 128, 128, 3))

    predictor = ft.partial(models.ResNetEncoder, stage_sizes=(2, 2, 2, 2))
    op_predictor = ft.partial(
        models.ConvolutionOperatorPredictor,
        num_kernels=5,
        window_size=(3, 3),
        predictor=predictor,
        num_channels=x.shape[3])

    percolation_model_fn = ft.partial(
        models.PercolationPropagator,
        convolution_operator_predictor=op_predictor,
        static_kernel=True)

    apply_fn, _, params, _ = _create_model(key, percolation_model_fn, x, 10)

    for unroll_steps in (1, 5, 10):
      states, kernel = apply_fn(params, x, unroll_steps)

      self.assertIsInstance(kernel, jnp.ndarray)
      self.assertEqual(kernel.shape,
                       (5, x.shape[0], 3, 3, x.shape[3], x.shape[3]))
      self.assertLen(states, unroll_steps)
      for state in states:
        self.assertEqual(state.shape, x.shape)

    percolation_model_fn = ft.partial(
        models.PercolationPropagator,
        convolution_operator_predictor=op_predictor,
        static_kernel=False)

    apply_fn, _, params, _ = _create_model(key, percolation_model_fn, x, 10)

    for unroll_steps in (1, 5, 10):
      states, kernels = apply_fn(params, x, unroll_steps)

      self.assertIsInstance(kernels, (list, tuple))
      self.assertLen(kernels, unroll_steps)
      for kernel in kernels:
        self.assertEqual(kernel.shape,
                         (5, x.shape[0], 3, 3, x.shape[3], x.shape[3]))
      self.assertLen(states, unroll_steps)
      for state in states:
        self.assertEqual(state.shape, x.shape)

  def test_StandardPropagator(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    x = jnp.zeros((2, 128, 128, 3))

    def single_step_fn():
      return nn.Sequential([
          nn.Conv(6, (3, 3), padding='SAME'),
          nn.GroupNorm(num_groups=3),
          nn.relu,
          nn.Conv(3, (3, 3), padding='SAME'),
      ])

    observation_fn = lambda: (lambda x: x[Ellipsis, :2])

    propagation_fn = ft.partial(
        models.StandardPropagator,
        observation_predictor=observation_fn,
        hidden_state_predictor=single_step_fn)

    apply_fn, _, params, _ = _create_model(key, propagation_fn, x, 10)

    for unroll_steps in (1, 5, 10):
      hstates, observations = apply_fn(params, x, unroll_steps)

      self.assertIsInstance(hstates, (list, tuple))
      self.assertIsInstance(observations, (list, tuple))
      self.assertLen(hstates, unroll_steps)
      self.assertLen(observations, unroll_steps)

      for hstate in hstates:
        self.assertEqual(hstate.shape, x.shape)

      for observation in observations:
        self.assertEqual(observation.shape, x.shape[:3] + (2,))

  def test_DeterministicPredictorPropagator(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    c = (jnp.zeros((2, 128, 128, 3)), jnp.zeros(
        (2, 128, 128, 3)), jnp.zeros((2, 128, 128, 3)))

    model_fn = ft.partial(
        models.DeterministicPredictorPropagator,
        field_shape=(128, 128),
        observation_channels=3)

    apply_fn, model, params, _ = _create_model(key, model_fn, c)

    for unroll_steps in (None, 1, 5, 10):
      hstates, observations = apply_fn(params, c, unroll_steps)

      if unroll_steps is None:
        self.assertLen(hstates, len(c))
        self.assertLen(observations, len(c))
      else:
        self.assertLen(hstates, unroll_steps)
        self.assertLen(observations, unroll_steps)

      for x in hstates:
        self.assertEqual(
            x.shape,
            (c[0].shape[0], *model.field_shape, model.hidden_state_channels))
      for x in observations:
        self.assertEqual(
            x.shape,
            (c[0].shape[0], *model.field_shape, model.observation_channels))

  def test_VariationalPredictorPropagator(self):
    prng = random.PRNGKey(0)
    prng, key = random.split(prng)
    c = (jnp.zeros((2, 128, 128, 3)), jnp.zeros(
        (2, 128, 128, 3)), jnp.zeros((2, 128, 128, 3)))
    h = jnp.zeros((2, 128, 128, 9))

    model_fn = ft.partial(
        models.VariationalPredictorPropagator,
        field_shape=(128, 128),
        observation_channels=3)

    apply_fn, model, params, _ = _create_model(key, model_fn, h, c, key)

    for unroll_steps in (None, 1, 5, 10):
      prng, key = random.split(prng)
      hstate, hstates, observations, mean, logvar = apply_fn(
          params, h, c, key, unroll_steps)

      self.assertEqual(mean.shape, (c[0].shape[0], model.latent_dim // 2))
      self.assertEqual(logvar.shape, (c[0].shape[0], model.latent_dim // 2))
      self.assertEqual(hstate.shape, h.shape)

      if unroll_steps is None:
        self.assertLen(hstates, len(c))
        self.assertLen(observations, len(c))
      else:
        self.assertLen(hstates, unroll_steps)
        self.assertLen(observations, unroll_steps)

      for x in hstates:
        self.assertEqual(
            x.shape,
            (c[0].shape[0], *model.field_shape, model.hidden_state_channels))
      for x in observations:
        self.assertEqual(
            x.shape,
            (c[0].shape[0], *model.field_shape, model.observation_channels))

    prng, key = random.split(prng)
    hstates, observations = model.apply({'params': params},
                                        10,
                                        c,
                                        key,
                                        method=model.sample)
    for x in hstates:
      self.assertEqual(
          x.shape,
          (c[0].shape[0], *model.field_shape, model.hidden_state_channels))
    for x in observations:
      self.assertEqual(
          x.shape,
          (c[0].shape[0], *model.field_shape, model.observation_channels))


if __name__ == '__main__':
  absltest.main()
