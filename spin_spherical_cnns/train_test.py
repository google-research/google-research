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

"""Tests for spin_spherical_cnns.train."""

import functools
from unittest import mock

from absl import logging
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import tensorflow as tf

from spin_spherical_cnns import train
from spin_spherical_cnns.configs import default


class TrainTest(tf.test.TestCase, parameterized.TestCase):
  """Test cases for ImageNet library."""

  def setUp(self):
    super().setUp()
    tf.config.experimental.set_visible_devices([], "GPU")

  @property
  def is_tpu(self):
    return jax.local_devices()[0].platform == "tpu"

  @parameterized.parameters(
      (0, 0.0),  #
      (1, 6.410256901290268e-05),  #
      (1000, 0.06410256773233414),  #
      (1560, 0.10000000149011612),  #
      (3000, 0.09927429258823395),  #
      (6000, 0.09324192255735397),  #
      (10000, 0.077022984623909))
  def test_get_learning_rate(self, step, expected_lr):
    actual_lr = train.get_learning_rate(
        step, base_learning_rate=0.1, steps_per_epoch=312, num_epochs=90)
    self.assertAllClose(expected_lr, actual_lr)

  @parameterized.parameters(
      (0, 0.0),  #
      (1, 6.410256901290268e-05),  #
      (1000, 0.06410256773233414),  #
      (1560, 0.10000000149011612),  #
      (3000, 0.09927429258823395),  #
      (6000, 0.09324192255735397),  #
      (10000, 0.077022984623909))
  def test_get_learning_rate_jitted(self, step, expected_lr):
    lr_fn = jax.jit(
        functools.partial(
            train.get_learning_rate,
            base_learning_rate=0.1,
            steps_per_epoch=312,
            num_epochs=90))
    actual_lr = lr_fn(jnp.array(step))
    self.assertAllClose(expected_lr, actual_lr)

  def test_evaluate(self):
    n = jax.device_count()
    assert n in (1, 2), f"Expected 1 or 2 devices, got {n}."
    # Dimensions : [1, devices, batch_size, ...] - first dim consumed by the
    # function .from_tensor_slices().
    eval_ds = tf.data.Dataset.from_tensor_slices(dict(
        input=tf.zeros(shape=(1, n, 2 // n, 8, 8, 1, 1)),
        label=tf.reshape(tf.constant([0, 9]), (1, n, 2 // n)),
    ))

    logits = (jnp.arange(10.).reshape(1, -1),)

    class MockedState:

      def __init__(self):
        self.optimizer = mock.Mock()
        self.optimizer.target.side_effect = logits
        self.batch_stats = mock.MagicMock()

    jax.tree_util.register_pytree_node(MockedState, lambda _: ((), None),
                                       lambda *_: MockedState())

    model = mock.Mock()
    model.apply.side_effect = logits

    # Disable type checking of the mock object.
    eval_metrics = train.evaluate(model, MockedState(), eval_ds)  # pytype: disable=wrong-arg-types
    self.assertIsNotNone(eval_metrics)
    # Disable pytype because we just asserted that eval_metrics is not None.
    metrics = eval_metrics.compute()   # pytype: disable=attribute-error
    logging.info("eval_metrics: %s", metrics)
    self.assertAllClose(
        {"accuracy": 0.5, "eval_loss": 4.9586296},
        metrics,
        # Lower precision to cover both single-device GPU and
        # multi-device TPU loss that are slightly different.
        atol=1e-4,
        rtol=1e-4)

  def test_train_and_evaluate(self):
    config = default.get_config()
    config.model_name = "tiny_classifier"
    config.dataset = "tiny_dummy"
    config.per_device_batch_size = 1
    config.num_train_steps = 2
    config.num_eval_steps = 1
    config.num_epochs = 1
    config.warmup_epochs = 0
    config.eval_pad_last_batch = False
    workdir = self.create_tempdir().full_path
    train.train_and_evaluate(config, workdir)
    logging.info("workdir content: %s", tf.io.gfile.listdir(workdir))

  def test_jax_conjugate_gradient(self):
    """Check that current JAX version conjugates the gradients.

    The spin-weighted model has complex weights, and the training loop
    explicitly conjugates all gradients in order to make backprop work. JAX will
    possibly change this default (see https://github.com/google/jax/issues/4891)
    which may silently break our training. This checks if the convention is
    still the one we assume.
    """
    # A function that returns the imaginary part has gradient 1j in TensorFlow
    # convention, which can be directly used for gradient descent. In JAX, the
    # convention returns the conjugate of that, -1j.
    imaginary_part = lambda x: x.imag
    gradient = jax.grad(imaginary_part)
    self.assertAllClose(gradient(0.1 + 0.2j), -1j)

if __name__ == "__main__":
  tf.test.main()
