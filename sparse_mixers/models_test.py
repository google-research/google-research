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

import functools
from typing import Any, Dict, Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
from jax import numpy as jnp
import ml_collections
import numpy as np

from sparse_mixers import models
from sparse_mixers.configs import base as base_config

jax.config.update("jax_threefry_partitionable", False)

# Type Stubs
ClassificationStats = models.ClassificationStats
DispatchAlgorithm = base_config.DispatchAlgorithm
FrozenDict = flax.core.frozen_dict.FrozenDict
LayerLayout = base_config.LayerLayout
Model = Any
ModelArchitecture = base_config.ModelArchitecture
PRNGKey = models.PRNGKey


def dummy_config(model_arch):
  """Creates a dummy model config that can be used by all tests."""
  config = base_config.get_config()
  config.model_arch = model_arch.name
  config.d_emb = 2
  config.d_model = 2
  config.d_ff = 2
  config.max_seq_length = 4
  config.num_heads = 1
  config.num_layers = 2
  config.vocab_size = 16
  config.pad_id = 0
  config.train_batch_size = 3
  config.eval_batch_size = 2
  config.use_fft = True
  config.num_experts = 2
  config.num_moe_layers = 0
  config.num_attention_layers = 0
  config.max_group_size = 2
  config.auxiliary_loss_factor = 0.01
  config.router_z_loss_factor = 0.01
  config.dispatch_algorithm = DispatchAlgorithm.MASK_TOKENS_CHOOSE
  config.dtype = jnp.float32

  return config


def dummy_inputs(
    key,
    config):
  """Creates a dummy model base inputs."""
  return {
      "input_ids":
          jax.random.randint(
              key, (config.train_batch_size, config.max_seq_length),
              minval=0,
              maxval=10),
      "type_ids":
          jax.random.randint(
              key, (config.train_batch_size, config.max_seq_length),
              minval=0,
              maxval=config.type_vocab_size)
  }


def init_encoder_batch(
    config):
  """Creates a batch of inputs used to initialize the models.EncoderModel."""
  return {
      "input_ids": jnp.ones((1, config.max_seq_length), jnp.int32),
      "type_ids": jnp.ones((1, config.max_seq_length), jnp.int32)
  }


def init_model_params(
    key, model,
    init_batch):
  """Initializes model parameters."""
  dropout_key, jitter_key = jax.random.split(key)
  jit_init = jax.jit(model.init)
  initial_variables = jit_init(
      {
          "params": key,
          "dropout": dropout_key,
          "jitter": jitter_key
      }, **init_batch)
  return initial_variables["params"]


class ModelsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(model_arch=ModelArchitecture.F_NET),
      dict(model_arch=ModelArchitecture.H_NET))
  def test_unparametrized_mixing(self, model_arch):
    config = dummy_config(model_arch=model_arch)
    frozen_config = ml_collections.FrozenConfigDict(config)

    encoder = models.EncoderModel(config=frozen_config)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config=frozen_config)
    params = init_model_params(rng, encoder, init_batch)
    # Unparameterized mixing encoders do not have any parameters in their mixing
    # layers, so their mixing layer names do not show up in params.
    expected_keys = {
        "embedder", "encoder_0", "encoder_1", "feed_forward_0",
        "feed_forward_1", "pooler"
    }
    self.assertEqual(params.keys(), expected_keys)

    inputs = dummy_inputs(rng, config=frozen_config)
    encoder_output = encoder.apply({"params": params},
                                   rngs={"dropout": rng},
                                   **inputs)
    expected_sequence_output_shape = (config.train_batch_size,
                                      config.max_seq_length, config.d_model)
    self.assertEqual(encoder_output.sequence_output.shape,
                     expected_sequence_output_shape)
    expected_pooled_output_shape = (config.train_batch_size, config.d_model)
    self.assertEqual(encoder_output.pooled_output.shape,
                     expected_pooled_output_shape)

  def test_f_net_bad_long_seq(self):
    config = dummy_config(model_arch=ModelArchitecture.F_NET)
    with config.unlocked():
      config.use_fft = True
      config.max_seq_length = 8194  # Long seq but not power of 2
    frozen_config = ml_collections.FrozenConfigDict(config)

    encoder = models.EncoderModel(config=frozen_config)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config=frozen_config)

    with self.assertRaisesRegex(
        ValueError,
        "must be a power of 2 to take advantage of FFT optimizations"):
      _ = init_model_params(rng, encoder, init_batch)

  @parameterized.parameters(
      dict(model_arch=ModelArchitecture.BERT, mixing_layer_name="attention"),
      dict(
          model_arch=ModelArchitecture.LINEAR,
          mixing_layer_name="linear_transform"),
      dict(
          model_arch=ModelArchitecture.C_NET,
          mixing_layer_name="circulant_transform"),
      dict(
          model_arch=ModelArchitecture.T_NET,
          mixing_layer_name="toeplitz_transform"))
  def test_parameterized_mixing(self, model_arch,
                                mixing_layer_name):
    config = dummy_config(model_arch=model_arch)
    with config.unlocked():
      config.use_fft = True
    frozen_config = ml_collections.FrozenConfigDict(config)

    encoder = models.EncoderModel(config=frozen_config)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config=frozen_config)
    params = init_model_params(rng, encoder, init_batch)
    expected_keys = {
        "embedder", "encoder_0", "encoder_1", "feed_forward_0",
        "feed_forward_1", f"{mixing_layer_name}_0", f"{mixing_layer_name}_1",
        "pooler"
    }
    self.assertEqual(params.keys(), expected_keys)

    inputs = dummy_inputs(rng, config=frozen_config)
    encoder_output = encoder.apply({"params": params},
                                   rngs={"dropout": rng},
                                   **inputs)
    expected_sequence_output_shape = (config.train_batch_size,
                                      config.max_seq_length, config.d_model)
    self.assertEqual(encoder_output.sequence_output.shape,
                     expected_sequence_output_shape)
    expected_pooled_output_shape = (config.train_batch_size, config.d_model)
    self.assertEqual(encoder_output.pooled_output.shape,
                     expected_pooled_output_shape)

  @parameterized.parameters(
      dict(
          attention_layout=LayerLayout.BOTTOM,
          num_attention_layers=0,
          expected_attention_layers=[]),
      dict(
          attention_layout=LayerLayout.MIDDLE,
          num_attention_layers=2,
          expected_attention_layers=[1, 2]),
      dict(
          attention_layout=LayerLayout.MIXED,
          num_attention_layers=2,
          expected_attention_layers=[0, 2]),
      dict(
          attention_layout=LayerLayout.TOP,
          num_attention_layers=1,
          expected_attention_layers=[3]))
  def test_hybrid(self, attention_layout,
                  num_attention_layers,
                  expected_attention_layers):
    config = dummy_config(model_arch=ModelArchitecture.F_NET)
    with config.unlocked():
      config.num_layers = 4  # More layers so we can test different layouts
      config.attention_layout = attention_layout
      config.num_attention_layers = num_attention_layers
    frozen_config = ml_collections.FrozenConfigDict(config)

    encoder = models.EncoderModel(config=frozen_config)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config=frozen_config)
    params = init_model_params(rng, encoder, init_batch)

    expected_keys = {
        "embedder", "encoder_0", "encoder_1", "encoder_2", "encoder_3",
        "feed_forward_0", "feed_forward_1", "feed_forward_2", "feed_forward_3",
        "pooler"
    }
    for expected_attention_layer in expected_attention_layers:
      expected_keys.add(f"attention_{expected_attention_layer}")

    self.assertEqual(params.keys(), expected_keys)

    inputs = dummy_inputs(rng, config=frozen_config)
    encoder_output = encoder.apply({"params": params},
                                   rngs={"dropout": rng},
                                   **inputs)
    expected_sequence_output_shape = (config.train_batch_size,
                                      config.max_seq_length, config.d_model)
    self.assertEqual(encoder_output.sequence_output.shape,
                     expected_sequence_output_shape)
    expected_pooled_output_shape = (config.train_batch_size, config.d_model)
    self.assertEqual(encoder_output.pooled_output.shape,
                     expected_pooled_output_shape)

  @parameterized.parameters(
      dict(
          moe_layout=LayerLayout.BOTTOM,
          num_moe_layers=0,
          expected_ff_layer_keys=[
              "feed_forward_0", "feed_forward_1", "feed_forward_2",
              "feed_forward_3"
          ],
          expected_moe_layer_keys=[],
          expected_moe_keys=[]),
      dict(
          moe_layout=LayerLayout.MIDDLE,
          num_moe_layers=1,
          expected_ff_layer_keys=[
              "feed_forward_0", "feed_forward_1", "feed_forward_3"
          ],
          expected_moe_layer_keys=["moe_2"],
          expected_moe_keys=["expert_2", "router_weights_2"]),
      dict(
          moe_layout=LayerLayout.MIXED,
          num_moe_layers=2,
          expected_ff_layer_keys=["feed_forward_1", "feed_forward_3"],
          expected_moe_layer_keys=["moe_0", "moe_2"],
          expected_moe_keys=[
              "expert_0", "expert_2", "router_weights_0", "router_weights_2"
          ]),
      dict(
          moe_layout=LayerLayout.TOP,
          num_moe_layers=2,
          expected_ff_layer_keys=["feed_forward_0", "feed_forward_1"],
          expected_moe_layer_keys=["moe_2", "moe_3"],
          expected_moe_keys=[
              "expert_2", "expert_3", "router_weights_2", "router_weights_3"
          ]))
  def test_moe(self, moe_layout, num_moe_layers,
               expected_ff_layer_keys,
               expected_moe_layer_keys,
               expected_moe_keys):
    config = dummy_config(model_arch=ModelArchitecture.T_NET)
    with config.unlocked():
      config.num_layers = 4  # More layers so we can test different layouts
      config.moe_layout = moe_layout
      config.num_moe_layers = num_moe_layers
    frozen_config = ml_collections.FrozenConfigDict(config)

    encoder = models.EncoderModel(config=frozen_config)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config=frozen_config)
    params = init_model_params(rng, encoder, init_batch)

    expected_keys = {
        "embedder", "encoder_0", "encoder_1", "encoder_2", "encoder_3",
        "pooler", "toeplitz_transform_0", "toeplitz_transform_1",
        "toeplitz_transform_2", "toeplitz_transform_3"
    }
    expected_keys.update(expected_ff_layer_keys)
    expected_keys.update(expected_moe_keys)

    self.assertEqual(params.keys(), expected_keys)

    rng, dropout_key, jitter_key = jax.random.split(rng, num=3)
    inputs = dummy_inputs(rng, config=frozen_config)
    encoder_output, state = encoder.apply({"params": params},
                                          rngs={
                                              "dropout": dropout_key,
                                              "jitter": jitter_key
                                          },
                                          mutable=["intermediates"],
                                          **inputs)

    expected_sequence_output_shape = (config.train_batch_size,
                                      config.max_seq_length, config.d_model)
    self.assertEqual(encoder_output.sequence_output.shape,
                     expected_sequence_output_shape)
    expected_pooled_output_shape = (config.train_batch_size, config.d_model)
    self.assertEqual(encoder_output.pooled_output.shape,
                     expected_pooled_output_shape)

    if num_moe_layers > 0:
      self.assertIn("intermediates", state)
      for layer in expected_moe_layer_keys:
        self.assertIn(layer, state["intermediates"])
        self.assertIn("diversity_metrics", state["intermediates"][layer])
    else:
      self.assertNotIn("intermediates", state)


class PretrainingModelTest(absltest.TestCase):

  def test_pretraining_model(self):
    config = dummy_config(model_arch=ModelArchitecture.F_NET)
    with config.unlocked():
      config.max_predictions_per_seq = 2
      config.num_attention_layers = 1
      config.num_moe_layers = 1
      num_tokens = config.train_batch_size * config.max_seq_length
      config.max_group_size = num_tokens
    frozen_config = ml_collections.FrozenConfigDict(config)

    model = models.PreTrainingModel(config=frozen_config)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config=frozen_config)
    # Pre-training model needs MLM and NSP inputs to be initialized.
    init_batch.update({
        "masked_lm_positions":
            jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
        "masked_lm_labels":
            jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
        "masked_lm_weights":
            jnp.ones((1, config.max_predictions_per_seq), jnp.float32),
        "next_sentence_labels":
            jnp.ones((1, 1), jnp.int32)
    })

    params = init_model_params(rng, model, init_batch)
    expected_keys = {
        "encoder", "predictions_dense", "predictions_output", "classification",
        "predictions_layer_norm"
    }
    self.assertEqual(params.keys(), expected_keys)

    rng, dropout_key, jitter_key = jax.random.split(rng, num=3)
    inputs = dummy_inputs(rng, config=frozen_config)
    inputs.update({
        "masked_lm_positions":
            jnp.ones((config.train_batch_size, config.max_predictions_per_seq),
                     jnp.int32),
        "masked_lm_labels":
            jnp.ones((config.train_batch_size, config.max_predictions_per_seq),
                     jnp.int32),
        "masked_lm_weights":
            jnp.ones((config.train_batch_size, config.max_predictions_per_seq),
                     jnp.int32),
        "next_sentence_labels":
            jnp.ones((config.train_batch_size, 1), jnp.int32)
    })
    metrics, state = model.apply({"params": params},
                                 rngs={
                                     "dropout": dropout_key,
                                     "jitter": jitter_key
                                 },
                                 mutable=["intermediates"],
                                 **inputs)

    self.assertAlmostEqual(metrics.masked_lm_loss, 16.818085, places=5)
    self.assertAlmostEqual(metrics.next_sentence_loss, 2.0803921)
    self.assertEqual(metrics.masked_lm_correct, 0)
    self.assertEqual(metrics.masked_lm_normalization, 6)
    self.assertEqual(metrics.masked_lm_total, 6)
    self.assertEqual(metrics.next_sentence_correct, 0)
    self.assertEqual(metrics.num_next_sentence_labels, 3.)

    self.assertIn("intermediates", state)
    jax.tree_util.tree_map(
        functools.partial(np.testing.assert_allclose, rtol=1e-6),
        FrozenDict(state["intermediates"]),
        FrozenDict({
            "encoder": {
                "moe_1": {
                    "diversity_metrics":
                        models.DiversityMetrics(
                            auxiliary_loss=1.0073242,
                            router_z_loss=0.50146484,
                            fraction_tokens_left_behind=0.25,
                            expert_usage=0.75,
                            router_confidence=0.51171875)
                }
            }
        }))


class SequenceClassificationModelTest(absltest.TestCase):

  def test_classification_model(self):
    n_classes = 2

    config = dummy_config(model_arch=ModelArchitecture.BERT)
    with config.unlocked():
      config.dataset_name = "dummy/classification_dataset"
    frozen_config = ml_collections.FrozenConfigDict(config)

    model = models.SequenceClassificationModel(
        config=frozen_config, n_classes=n_classes)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config=frozen_config)
    params = init_model_params(rng, model, init_batch)
    self.assertEqual(params.keys(), {"encoder", "classification"})

    # Logits for eval/prediction (no labels supplied).
    eval_inputs = dummy_inputs(rng, config=frozen_config)
    eval_inputs["deterministic"] = True
    logits = model.apply({"params": params}, **eval_inputs)
    expected_logits_shape = (config.train_batch_size, n_classes)
    self.assertEqual(jnp.shape(logits), expected_logits_shape)

    # Metrics for training (labels supplied).
    train_inputs = dummy_inputs(rng, config=frozen_config)
    train_inputs["labels"] = jnp.ones(config.train_batch_size, jnp.int32)
    metrics = model.apply({"params": params},
                          rngs={"dropout": rng},
                          **train_inputs)

    expected_metrics = ClassificationStats(
        batch_loss=0.6932529,
        num_labels=3,
        correct_predictions=1,
        expert_metrics=None)
    self.assertEqual(metrics, expected_metrics)

  def test_regression_model(self):
    n_classes = 1  # Only one label for regression

    config = dummy_config(model_arch=ModelArchitecture.F_NET)
    with config.unlocked():
      config.dataset_name = "glue/stsb"  # regression task dataset
      config.num_moe_layers = 1  # Add moe layer to verify expert metrics
      num_tokens = config.train_batch_size * config.max_seq_length
      config.max_group_size = num_tokens
    frozen_config = ml_collections.FrozenConfigDict(config)

    model = models.SequenceClassificationModel(
        config=frozen_config, n_classes=n_classes)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config=frozen_config)
    params = init_model_params(rng, model, init_batch)
    self.assertEqual(params.keys(), {"encoder", "classification"})

    # Logits for eval/prediction (no labels supplied).
    eval_inputs = dummy_inputs(rng, config=frozen_config)
    eval_inputs["deterministic"] = True
    logits = model.apply({"params": params}, **eval_inputs)
    expected_logits_shape = (config.train_batch_size, n_classes)
    self.assertEqual(jnp.shape(logits), expected_logits_shape)

    # Metrics for training (labels supplied).
    train_inputs = dummy_inputs(rng, config=frozen_config)
    label_key, dropout_key, jitter_key = jax.random.split(rng, num=3)
    train_inputs["labels"] = jax.random.uniform(
        label_key, (config.train_batch_size,), minval=0., maxval=1.)

    metrics, state = model.apply({"params": params},
                                 rngs={
                                     "dropout": dropout_key,
                                     "jitter": jitter_key
                                 },
                                 mutable=["intermediates"],
                                 **train_inputs)

    self.assertAlmostEqual(metrics.batch_loss, 1.0100806, places=6)
    self.assertEqual(metrics.num_labels, 3)

    self.assertIn("intermediates", state)
    jax.tree_util.tree_map(
        functools.partial(np.testing.assert_allclose, rtol=1e-6),
        FrozenDict(state["intermediates"]),
        FrozenDict({
            "encoder": {
                "moe_1": {
                    "diversity_metrics":
                        models.DiversityMetrics(
                            auxiliary_loss=1.0073242,
                            router_z_loss=0.45751953,
                            fraction_tokens_left_behind=0.25,
                            expert_usage=0.75,
                            router_confidence=0.51171875)
                }
            }
        }))


if __name__ == "__main__":
  absltest.main()
