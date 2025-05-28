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

"""Tests for train_utils."""

import functools
import os
# Restrict tests to a single device setting. Import before JAX.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
from typing import Callable, Tuple  # pylint: disable=g-import-not-at-top

from absl.testing import absltest
from flax import jax_utils
from flax.training import common_utils
import jax
from jax import numpy as jnp
import ml_collections
import optax

from sparse_mixers import core_utils
from sparse_mixers import models
from sparse_mixers import train_utils
from sparse_mixers.configs import base as default_config

# Type Stubs
Batch = train_utils.Batch
ClassificationStats = models.ClassificationStats
DiversityMetrics = train_utils.DiversityMetrics
Loss = train_utils.Loss
Params = train_utils.Params
PretrainingStats = models.PretrainingStats
PRNGKey = train_utils.PRNGKey
Stats = train_utils.Stats
FlaxTrainState = train_utils.FlaxTrainState


def frozen_config(
    sharded_params = False):
  """Creates a dummy model config that can be used by all tests."""
  config = default_config.get_config()
  config.model_arch = default_config.ModelArchitecture.LINEAR.name
  config.num_attention_layers = 0
  config.d_emb = 4
  config.d_model = 4
  config.d_ff = 4
  config.max_seq_length = 8
  config.num_layers = 1
  config.vocab_size = 16
  config.train_batch_size = 4
  config.dtype = jnp.float32
  config.pad_id = 3
  # MoE layers contain sharded parameters.
  config.num_moe_layers = 1 if sharded_params else 0
  config.num_experts = 1 if sharded_params else 0
  config.auxiliary_loss_factor = 0.01
  config.router_z_loss_factor = 0.01
  return ml_collections.FrozenConfigDict(config)


def dummy_loss_and_metrics(
    params,
    batch,
    rng,
):
  """Computes dummy loss and metrics."""
  del params, batch, rng  # Dummy parameters to confirm to loss fn API
  dummy_loss = 0.1
  return dummy_loss, ClassificationStats(
      batch_loss=dummy_loss, num_labels=2, correct_predictions=1)


def create_flax_train_state(key,
                            config,
                            num_steps):
  """Creates train state for models.EncoderModel."""
  model = models.EncoderModel(config=config)

  init_batch = {
      "input_ids": jnp.ones((1, config.max_seq_length), jnp.int32),
      "type_ids": jnp.ones((1, config.max_seq_length), jnp.int32)
  }

  dropout_key, jitter_key = jax.random.split(key)

  jit_init = jax.jit(model.init)
  initial_variables = jit_init(
      {
          "params": key,
          "dropout": dropout_key,
          "jitter": jitter_key
      }, **init_batch)
  params = initial_variables["params"]

  tx = optax.adamw(learning_rate=simple_lr_fn(num_steps=num_steps))
  return FlaxTrainState.create(apply_fn=model.apply, params=params, tx=tx)


def simple_lr_fn(num_steps):
  return train_utils.create_learning_rate_scheduler(
      factors="constant * linear_decay",
      base_learning_rate=1,
      warmup_steps=0,
      decay_steps=num_steps - 1,
  )


def dummy_batch(rng, batch_size, max_seq_length):
  dummy_data = jax.random.randint(
      rng, (batch_size, max_seq_length), minval=0, maxval=10)
  return {"a": dummy_data, "b": dummy_data}


class TrainUtilsTest(absltest.TestCase):

  def test_validate_correct_config(self):
    config = default_config.get_config()
    train_utils.validate_config(config)

  def test_validate_incorrect_configs(self):
    config = default_config.get_config()
    config.train_batch_size = 6
    config.gradient_accum_steps = 4
    with self.assertRaisesRegex(
        ValueError,
        "training batch size must be divisible by gradient_accum_steps"):
      train_utils.validate_config(config)

  def test_collect_metrics(self):

    def create_dummy_stats(n):
      return PretrainingStats(
          masked_lm_loss=0.7 * n,
          next_sentence_loss=0.2 * n,
          masked_lm_correct=n // 2,
          masked_lm_normalization=n,
          masked_lm_total=n,
          next_sentence_correct=n // 2,
          num_next_sentence_labels=n,
          expert_metrics=DiversityMetrics(
              auxiliary_loss=0.1 * n,
              router_z_loss=0.1 * n,
              fraction_tokens_left_behind=0.2 * n,
              expert_usage=0.7 * n,
              router_confidence=0.4 * n))

    stats = [
        jax.pmap(create_dummy_stats)(jnp.arange(1,
                                                jax.local_device_count() + 1))
    ]

    expected_collected_stats = PretrainingStats(
        masked_lm_loss=jnp.array([0.7]),
        next_sentence_loss=jnp.array([0.2]),
        masked_lm_correct=jnp.array([0]),
        masked_lm_normalization=jnp.array([1]),
        masked_lm_total=jnp.array([1]),
        next_sentence_correct=jnp.array([0]),
        num_next_sentence_labels=jnp.array([1]),
        expert_metrics=DiversityMetrics(
            auxiliary_loss=jnp.array([0.1]),
            router_z_loss=jnp.array([0.1]),
            fraction_tokens_left_behind=jnp.array([0.2]),
            expert_usage=jnp.array([0.7]),
            router_confidence=jnp.array([0.4])))

    self.assertEqual(
        train_utils.collect_metrics(stats), expected_collected_stats)

  def test_compute_pretraining_metrics(self):
    stats = PretrainingStats(
        masked_lm_loss=jnp.array([0.7, 0.3]),
        next_sentence_loss=jnp.array([0.2, 0.2]),
        masked_lm_correct=jnp.array([0, 1]),
        masked_lm_normalization=jnp.array([1, 3]),
        masked_lm_total=jnp.array([1, 3]),
        next_sentence_correct=jnp.array([0, 1]),
        num_next_sentence_labels=jnp.array([1, 1]),
        grad_l2_sum=jnp.array([100, 100]),
        expert_metrics=DiversityMetrics(
            auxiliary_loss=jnp.array([0.1, 0.9]),
            router_z_loss=jnp.array([0.1, 0.3]),
            fraction_tokens_left_behind=jnp.array([0.2, 0.1]),
            expert_usage=jnp.array([0.7, 0.8]),
            router_confidence=jnp.array([0.4, 0.2])))

    self.assertEqual(
        train_utils.compute_pretraining_metrics(stats), {
            "loss": 1.15,
            "masked_lm_accuracy": 0.25,
            "masked_lm_loss": 0.25,
            "next_sentence_accuracy": 0.5,
            "next_sentence_loss": 0.2,
            "grad_l2_norm": 14.142136,
            "auxiliary_loss": 0.5,
            "router_z_loss": 0.2,
            "expert_usage": 0.75,
            "fraction_tokens_left_behind": 0.15,
            "router_confidence": 0.3
        })

  def test_compute_classification_metrics(self):
    stats = ClassificationStats(
        batch_loss=jnp.array([1, 3]),
        correct_predictions=jnp.array([0, 1]),
        num_labels=jnp.array([1, 3]),
        expert_metrics=DiversityMetrics(
            auxiliary_loss=jnp.array([0.1, 0.9]),
            router_z_loss=jnp.array([0.05, 0.05]),
            fraction_tokens_left_behind=jnp.array([0.2, 0.1]),
            expert_usage=jnp.array([0.7, 0.8]),
            router_confidence=jnp.array([0.4, 0.2])))

    self.assertEqual(
        train_utils.compute_classification_metrics(
            stats, is_regression_task=False), {
                "loss": 1.55,
                "accuracy": 0.25,
                "auxiliary_loss": 0.5,
                "router_z_loss": 0.05,
                "expert_usage": 0.75,
                "fraction_tokens_left_behind": 0.15,
                "router_confidence": 0.3
            })

  def test_summarize_expert_metrics(self):
    state = {
        "intermediates": {
            "encoder": {
                "moe_0": {
                    "diversity_metrics":
                        DiversityMetrics(
                            auxiliary_loss=1.,
                            router_z_loss=2.,
                            fraction_tokens_left_behind=0.,
                            expert_usage=1.,
                            router_confidence=0.5)
                },
                "moe_1": {
                    "diversity_metrics":
                        DiversityMetrics(
                            auxiliary_loss=2.,
                            router_z_loss=4.,
                            fraction_tokens_left_behind=0.1,
                            expert_usage=1.,
                            router_confidence=0.7)
                }
            }
        }
    }
    expected_metrics = DiversityMetrics(
        auxiliary_loss=0.3,
        router_z_loss=0.6,
        fraction_tokens_left_behind=0.05,
        expert_usage=1.,
        router_confidence=0.6)
    self.assertEqual(
        train_utils.summarize_expert_metrics(
            state=state, auxiliary_loss_factor=0.1, router_z_loss_factor=0.1),
        expected_metrics)

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

  def test_replicated_train_step(self):
    num_steps = 2

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, jax.local_device_count())

    config = frozen_config()
    train_state = create_flax_train_state(rng, config, num_steps)
    p_train_state = jax_utils.replicate(train_state)

    p_train_step = jax.pmap(
        functools.partial(
            train_utils.pmap_train_step,
            loss_and_metrics_fn=dummy_loss_and_metrics,
            axis_name="batch"),
        axis_name="batch")

    batch = dummy_batch(rng, config.train_batch_size, config.max_seq_length)
    batch = common_utils.shard(batch)

    expected_metrics = ClassificationStats(
        batch_loss=0.1, num_labels=2, correct_predictions=1, grad_l2_sum=0.)

    for _ in range(num_steps):
      p_train_state, metrics, rngs = p_train_step(
          train_state=p_train_state, batch=batch, rng=rngs)
      self.assertEqual(metrics, expected_metrics)

  def test_sharded_train_step(self):
    num_steps = 2

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, jax.device_count())

    config = frozen_config(sharded_params=True)
    sharded_match_fn = core_utils.match_fn(r".*expert.*")

    train_state = create_flax_train_state(rng, config, num_steps)
    p_train_state = jax_utils.replicate(train_state)

    p_train_step = jax.pmap(
        functools.partial(
            train_utils.pmap_train_step,
            loss_and_metrics_fn=dummy_loss_and_metrics,
            axis_name="batch",
            sharded_match_fn=sharded_match_fn),
        axis_name="batch")

    batch = dummy_batch(rng, config.train_batch_size, config.max_seq_length)
    batch = common_utils.shard(batch)

    expected_metrics = ClassificationStats(
        batch_loss=0.1, num_labels=2, correct_predictions=1, grad_l2_sum=0.)

    for _ in range(num_steps):
      p_train_state, metrics, rngs = p_train_step(
          train_state=p_train_state, batch=batch, rng=rngs)
      self.assertEqual(metrics, expected_metrics)

  def test_accumulate_gradient(self):
    rng = jax.random.PRNGKey(0)
    config = frozen_config()

    opt_rng, batch_rng, loss_rng = jax.random.split(rng, num=3)
    train_state = create_flax_train_state(opt_rng, config, num_steps=1)
    batch = dummy_batch(batch_rng, config.train_batch_size,
                        config.max_seq_length)
    loss_fn = functools.partial(dummy_loss_and_metrics, rng=loss_rng)

    # Accumulated gradient and metrics over mini-batches should match results
    # over single (large) batch.
    expected_grad, expected_metrics = jax.grad(
        loss_fn, has_aux=True)(train_state.params, batch)

    actual_grad, actual_metrics = train_utils._accumulate_gradient(
        train_state.params, batch, loss_fn, accum_steps=2)

    self.assertEqual(
        jax.tree.map(jnp.shape, actual_grad),
        jax.tree.map(jnp.shape, expected_grad))
    self.assertEqual(actual_metrics, expected_metrics)


if __name__ == "__main__":
  absltest.main()
