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

"""Test for end-to-end standard first and second moments."""
from absl.testing import absltest
import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as onp

from cold_posterior_flax.cifar10.models import activations
from cold_posterior_flax.cifar10.models import conv_layers
from cold_posterior_flax.cifar10.models import evonorm
from cold_posterior_flax.cifar10.models import new_initializers
from cold_posterior_flax.cifar10.models import wideresnet


def normal(mean=0., stddev=1e-2, dtype=jnp.float32):
  """Normal Initializer with mean parameter."""
  zero_mean_normal = initializers.normal(stddev, dtype)

  def init(key, shape, dtype=dtype):
    return zero_mean_normal(key, shape, dtype) + mean

  return init


class NormalizedTest(absltest.TestCase):

  def test_resnetv1(self):
    rng = random.PRNGKey(10)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (128, 32, 32, 3))

    activation_f = 'bias_scale_SELU_norm'
    model_def = wideresnet.ResnetV1.partial(
        depth=20,
        num_outputs=10,
        activation_f=activation_f,
        normalization='none',
        dropout_rate=0,
        std_penalty_mult=0,
        use_residual=2,  # TODO(basv): test with residual.
        bias_scale=0.0,
        weight_norm='none',
        no_head=True,
        report_metrics=True,
    )
    (y, _, metrics), _ = model_def.create(
        key2,
        x,
    )
    mean = jnp.mean(y, axis=(0, 1, 2))
    std = jnp.std(
        y, axis=(
            0,
            1,
            2,
        ))
    mean_x = jnp.mean(x, axis=(0, 1, 2))
    std_x = jnp.std(x, axis=(0, 1, 2))

    onp.testing.assert_allclose(mean_x, jnp.zeros_like(mean_x), atol=0.1)
    onp.testing.assert_allclose(std_x, jnp.ones_like(std_x), atol=0.1)

    for metric_key, metric_value in metrics.items():
      if 'postnorm' in metric_key or 'postact' in metric_key or 'postres' in metric_key:
        if 'std' in metric_key:
          onp.testing.assert_allclose(
              metric_value,
              jnp.ones_like(metric_value),
              atol=0.1,
              err_msg=metric_key)
        elif 'mean' in metric_key:
          onp.testing.assert_allclose(
              metric_value,
              jnp.zeros_like(metric_value),
              atol=0.3,
              err_msg=metric_key)

    onp.testing.assert_allclose(mean, jnp.zeros_like(mean), atol=0.2)
    onp.testing.assert_allclose(std, jnp.ones_like(std), atol=0.3)

  def test_resnet_imagenet(self):
    rng = random.PRNGKey(10)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (128, 32, 32, 3))
    activation_f = 'bias_scale_SELU_norm'

    model_def = wideresnet.ResNetImageNet50.partial(
        num_classes=1000,
        activation_f=activation_f,
        normalization='none',
        std_penalty_mult=0,
        use_residual=2,
        bias_scale=0.0,
        weight_norm='fixed',
        softplus_scale=1,
        compensate_padding=True,
        no_head=True,
    )
    (y, _, metrics), _ = model_def.create(
        key2,
        x,
        train=True,
    )
    mean = jnp.mean(y, axis=(0, 1, 2))
    std = jnp.std(
        y, axis=(
            0,
            1,
            2,
        ))
    mean_x = jnp.mean(x, axis=(0, 1, 2))
    std_x = jnp.std(x, axis=(0, 1, 2))

    onp.testing.assert_allclose(mean_x, jnp.zeros_like(mean_x), atol=0.1)
    onp.testing.assert_allclose(std_x, jnp.ones_like(std_x), atol=0.1)

    for metric_key, metric_value in metrics.items():
      if 'postnorm' in metric_key or 'postact' in metric_key or 'postres' in metric_key:
        if 'std' in metric_key:
          onp.testing.assert_allclose(
              metric_value,
              jnp.ones_like(metric_value),
              atol=0.1,
              err_msg=metric_key)
        elif 'mean' in metric_key:
          onp.testing.assert_allclose(
              metric_value,
              jnp.zeros_like(metric_value),
              atol=0.1,
              err_msg=metric_key)

    onp.testing.assert_allclose(std, jnp.ones_like(std), atol=0.4)
    onp.testing.assert_allclose(mean, jnp.zeros_like(mean), atol=0.6)

  def test_wrn26_4(self):
    rng = random.PRNGKey(10)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (128, 32, 32, 3))

    for activation_f in ['bias_scale_SELU_norm']:
      model_def = wideresnet.WideResnet.partial(
          blocks_per_group=4,
          channel_multiplier=4,
          num_outputs=10,
          activation_f=activation_f,
          normalization='none',
          dropout_rate=0,
          std_penalty_mult=0,
          use_residual=2,  # TODO(basv): test with residual.
          bias_scale=0.0,
          weight_norm='learned',
          no_head=True,
      )
      (y, _, metrics), _ = model_def.create(
          key2,
          x,
      )
      mean = jnp.mean(jnp.abs(jnp.mean(y, axis=(0, 1, 2))))
      std = jnp.mean(jnp.std(y, axis=(0, 1, 2)))
      mean_x = jnp.mean(x, axis=(0, 1, 2))
      std_x = jnp.std(x, axis=(0, 1, 2))

      onp.testing.assert_allclose(mean_x, jnp.zeros_like(mean_x), atol=0.1)
      onp.testing.assert_allclose(std_x, jnp.ones_like(std_x), atol=0.1)

      for metric_key, metric_value in metrics.items():
        if 'postnorm' in metric_key or 'postact' in metric_key or 'postres' in metric_key:
          if 'std' in metric_key:
            onp.testing.assert_allclose(
                metric_value,
                jnp.ones_like(metric_value),
                atol=0.2,
                err_msg=metric_key)
          elif 'mean' in metric_key:
            onp.testing.assert_allclose(
                metric_value,
                jnp.zeros_like(metric_value),
                atol=0.2,
                err_msg=metric_key)

      onp.testing.assert_allclose(mean, jnp.zeros_like(mean), atol=0.1)
      onp.testing.assert_allclose(std, jnp.ones_like(std), atol=0.1)

  def test_weight_norm_standard(self):

    rng = random.PRNGKey(5)
    key1, key2 = random.split(rng)
    for k in [3, 5]:
      for padding in ['VALID', 'SAME']:
        for layer in [
            conv_layers.Conv, conv_layers.ConvWS, conv_layers.ConvFixedScale
        ]:
          x = random.normal(key1, (512, 32, 32, 128))
          y = x
          for i in range(5):
            y, _ = layer.create(
                key2,
                y,
                features=128,
                kernel_size=(k, k),
                bias=False,
                padding=padding,
                kernel_init=jax.nn.initializers.lecun_normal())

            mean = jnp.mean(y)
            std = jnp.std(y)

            err_msg = 'layer %s, padding %s, kernel_size %d, depth %d' % (
                layer.__name__, padding, k, i)
            onp.testing.assert_allclose(
                mean, jnp.zeros_like(mean), atol=0.1, err_msg=err_msg)
            onp.testing.assert_allclose(
                std, jnp.ones_like(std), atol=0.1, err_msg=err_msg)

  def test_bias_selu_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (100000, 128))
    y, _ = activations.BiasSELUNorm.create(
        key2,
        x,
        features=128,
        bias_init=jax.nn.initializers.normal(stddev=.5),
        scale_init=normal(mean=new_initializers.inv_softplus(1), stddev=.5),
    )
    mean = jnp.mean(y, axis=0)
    std = jnp.std(y, axis=0)

    onp.testing.assert_allclose(mean, jnp.zeros_like(mean), atol=1e-1)
    onp.testing.assert_allclose(std, jnp.ones_like(mean), atol=1e-1)

  def test_bias_relu_norm_stable(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (100000, 32)) * 4
    y, _ = activations.BiasReluNorm.create(
        key2,
        x,
        features=32,
        bias_init=normal(mean=0.0, stddev=4),
        scale_init=normal(mean=onp.log(onp.exp(1.0) - 1), stddev=4),
        scale=True,
        bias=True,
    )

    assert onp.all(onp.isfinite(y))

  def test_bias_relu_norm(self):
    rng = random.PRNGKey(0)
    x = random.normal(rng, (100000, 32))
    y, _ = activations.BiasReluNorm.create(
        random.PRNGKey(0),
        x,
        features=32,
        bias_init=normal(mean=0.0, stddev=.5),
        scale_init=normal(mean=1.0, stddev=.5),
        scale=True,
        bias=True,
    )
    mean = jnp.mean(y, axis=0)
    std = jnp.std(y, axis=0)

    onp.testing.assert_allclose(mean, jnp.zeros_like(mean), atol=.1)
    onp.testing.assert_allclose(std, jnp.ones_like(std), atol=.1)

  def test_evonorm_b0(self):
    rng = random.PRNGKey(0)
    x = random.normal(rng, (100000, 32))
    y, _ = evonorm.EvoNorm.create(
        random.PRNGKey(0),
        x,
        num_groups=4,
        bias_init=normal(mean=0.0, stddev=.5),
        scale_init=normal(mean=1.0, stddev=.5),
        scale=True,
        bias=True,
    )
    self.assertTrue(jnp.all(~jnp.isnan(y)), 'Should be valid.')

  def test_evonorm_s0(self):
    rng = random.PRNGKey(0)
    x = random.normal(rng, (100000, 32))
    y, _ = evonorm.EvoNorm.create(
        random.PRNGKey(0),
        x,
        layer=evonorm.LAYER_EVONORM_S0,
        num_groups=4,
        bias_init=normal(mean=0.0, stddev=.5),
        scale_init=normal(mean=1.0, stddev=.5),
        scale=True,
        bias=True,
    )
    self.assertTrue(jnp.all(~jnp.isnan(y)), 'Should be valid.')


if __name__ == '__main__':
  absltest.main()
