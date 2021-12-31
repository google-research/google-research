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

"""Tests for f_net.train_utils."""

import functools

from typing import Any, Dict, Mapping, Tuple

from absl.testing import absltest
from flax import jax_utils
from flax import optim
from flax.training import common_utils
import jax
from jax import numpy as jnp
import ml_collections

from f_net import models
from f_net import train_utils
from f_net.configs import base as default_config

# Type Stubs
PRNGKey = Any


def dummy_frozen_config():
  """Creates a dummy model config that can be used by all tests."""
  config = default_config.get_config()
  config.model_arch = default_config.ModelArchitecture.FF_ONLY
  config.d_emb = 4
  config.d_model = 4
  config.d_ff = 4
  config.max_seq_length = 8
  config.num_layers = 1
  config.vocab_size = 1000
  config.train_batch_size = 2
  return ml_collections.FrozenConfigDict(config)


def dummy_metrics():
  """Creates simple metrics."""
  return {"very_helpful_metric": jnp.array([4])}


def dummy_loss_and_metrics(
    params,
    batch,
    rng,
):
  """Computes dummy loss and metrics."""
  del batch, params, rng  # We return fixed, dummy loss and metrics
  dummy_loss = 0.1
  return dummy_loss, dummy_metrics()


def create_optimizer(key, config):
  """Creates optimizer for models.EncoderModel."""
  model = models.EncoderModel(config=config)

  init_batch = {
      "input_ids": jnp.ones((1, config.max_seq_length), jnp.int32),
      "input_mask": jnp.ones((1, config.max_seq_length), jnp.int32),
      "type_ids": jnp.ones((1, config.max_seq_length), jnp.int32)
  }

  key, dropout_key = jax.random.split(key)

  jit_init = jax.jit(model.init)
  initial_variables = jit_init({
      "params": key,
      "dropout": dropout_key
  }, **init_batch)
  params = initial_variables["params"]

  optimizer_def = optim.Adam(learning_rate=1e-4)
  return optimizer_def.create(params)


class TrainUtilsTest(absltest.TestCase):

  def test_learning_rate_scheduler(self):
    num_steps = 6
    warmup_steps = 2
    learning_rate_fn = train_utils.create_learning_rate_scheduler(
        factors="constant * linear_warmup * linear_decay",
        base_learning_rate=1,
        warmup_steps=warmup_steps,
        decay_steps=num_steps - warmup_steps,
    )

    _ = learning_rate_fn(0)

    for step, expected_rate in zip(
        range(num_steps), [0, 0.5, 1, 0.75, 0.5, 0.25]):
      self.assertAlmostEqual(learning_rate_fn(step), expected_rate, delta=1e-7)

  def test_train_step(self):
    num_steps = 2
    learning_rate_fn = train_utils.create_learning_rate_scheduler(
        factors="constant * linear_decay",
        base_learning_rate=1,
        warmup_steps=0,
        decay_steps=num_steps - 1,
    )

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, jax.device_count())

    config = dummy_frozen_config()
    optimizer = create_optimizer(rng, config)
    p_optimizer = jax_utils.replicate(optimizer)

    p_train_step = jax.pmap(
        functools.partial(
            train_utils.train_step,
            loss_and_metrics_fn=dummy_loss_and_metrics,
            learning_rate_fn=learning_rate_fn,
            clipped_grad_norm=1.0),
        axis_name="batch")

    batch = jax.random.randint(
        rng, (config.train_batch_size, config.max_seq_length),
        minval=0,
        maxval=10)
    batch = common_utils.shard(batch)

    for _ in range(num_steps):
      p_optimizer, metrics, rngs = p_train_step(
          optimizer=p_optimizer, batch=batch, rng=rngs)
      self.assertSetEqual(
          set(metrics.keys()), {
              "very_helpful_metric", "clipped_grad_l2_sum",
              "unclipped_grad_l2_sum"
          })


if __name__ == "__main__":
  absltest.main()
