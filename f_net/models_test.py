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

"""Tests for f_net.models."""

from typing import Any, Dict, Mapping

from absl.testing import absltest
import jax
from jax import numpy as jnp
import ml_collections

from f_net import models
from f_net.configs import base as base_config

# Type Stubs
PRNGKey = Any
Model = Any


def dummy_frozen_config(
    model_arch,
    max_seq_length = 16,
    use_tpu_fourier_optimizations = False,
    dataset_name = "dummy/data"):
  """Creates a dummy model config that can be used by all tests."""
  config = base_config.get_config()
  config.model_arch = model_arch
  config.use_tpu_fourier_optimizations = use_tpu_fourier_optimizations
  config.d_emb = 8
  config.d_model = 8
  config.d_ff = 8
  config.max_seq_length = max_seq_length
  config.num_heads = 1
  config.num_layers = 2
  config.vocab_size = 28000
  config.train_batch_size = 3
  config.eval_batch_size = 2
  config.dataset_name = dataset_name

  # Pre-training only.
  config.max_predictions_per_seq = 7

  return ml_collections.FrozenConfigDict(config)


def dummy_inputs(
    key,
    config):
  """Creates a dummy model base inputs."""
  input_ids = jax.random.randint(
      key, (config.train_batch_size, config.max_seq_length),
      minval=0,
      maxval=10)
  return {
      "input_ids":
          input_ids,
      "input_mask": (input_ids > 0).astype(jnp.int32),
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
      "input_mask": jnp.ones((1, config.max_seq_length), jnp.int32),
      "type_ids": jnp.ones((1, config.max_seq_length), jnp.int32)
  }


def init_model_params(
    key, model,
    init_batch):
  """Initializes model parameters."""
  key, dropout_key = jax.random.split(key)
  jit_init = jax.jit(model.init)
  initial_variables = jit_init({
      "params": key,
      "dropout": dropout_key
  }, **init_batch)
  return initial_variables["params"]


class ModelsTest(absltest.TestCase):

  def test_f_net_encoder_short_seq(self):
    config = dummy_frozen_config(
        model_arch=base_config.ModelArchitecture.F_NET, max_seq_length=16)
    encoder = models.EncoderModel(config=config)
    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config)
    params = init_model_params(rng, encoder, init_batch)
    # Fourier sublayers have no parameters so do not show up in params.
    expected_keys = {
        "embedder", "encoder_0", "encoder_1", "feed_forward_0",
        "feed_forward_1", "pooler"
    }
    self.assertEqual(params.keys(), expected_keys)

    inputs = dummy_inputs(rng, config)
    hidden_states, pooled_output = encoder.apply({"params": params},
                                                 rngs={"dropout": rng},
                                                 **inputs)
    expected_hidden_states_shape = (config.train_batch_size,
                                    config.max_seq_length, config.d_model)
    self.assertEqual(hidden_states.shape, expected_hidden_states_shape)
    expected_pooled_output_shape = (config.train_batch_size, config.d_model)
    self.assertEqual(pooled_output.shape, expected_pooled_output_shape)

  def test_f_net_encoder_bad_long_seq(self):
    config = dummy_frozen_config(
        model_arch=base_config.ModelArchitecture.F_NET,
        use_tpu_fourier_optimizations=True,
        max_seq_length=8194)
    encoder = models.EncoderModel(config=config)
    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config)

    with self.assertRaisesRegex(
        ValueError,
        "must be a power of 2 to take advantage of FFT optimizations"):
      _ = init_model_params(rng, encoder, init_batch)

  def test_bert_encoder(self):
    config = dummy_frozen_config(model_arch=base_config.ModelArchitecture.BERT)
    encoder = models.EncoderModel(config=config)
    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config)
    params = init_model_params(rng, encoder, init_batch)
    expected_keys = {
        "embedder", "encoder_0", "encoder_1", "feed_forward_0",
        "feed_forward_1", "self_attention_0", "self_attention_1", "pooler"
    }
    self.assertEqual(params.keys(), expected_keys)

    inputs = dummy_inputs(rng, config)
    hidden_states, pooled_output = encoder.apply({"params": params},
                                                 rngs={"dropout": rng},
                                                 **inputs)
    expected_hidden_states_shape = (config.train_batch_size,
                                    config.max_seq_length, config.d_model)
    self.assertEqual(hidden_states.shape, expected_hidden_states_shape)
    expected_pooled_output_shape = (config.train_batch_size, config.d_model)
    self.assertEqual(pooled_output.shape, expected_pooled_output_shape)

  def test_linear_encoder(self):
    config = dummy_frozen_config(
        model_arch=base_config.ModelArchitecture.LINEAR)
    encoder = models.EncoderModel(config=config)
    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config)
    params = init_model_params(rng, encoder, init_batch)
    expected_keys = {
        "embedder", "encoder_0", "encoder_1", "feed_forward_0",
        "feed_forward_1", "linear_transform_0", "linear_transform_1", "pooler"
    }
    self.assertEqual(params.keys(), expected_keys)

    inputs = dummy_inputs(rng, config)
    hidden_states, pooled_output = encoder.apply({"params": params},
                                                 rngs={"dropout": rng},
                                                 **inputs)
    expected_hidden_states_shape = (config.train_batch_size,
                                    config.max_seq_length, config.d_model)
    self.assertEqual(hidden_states.shape, expected_hidden_states_shape)
    expected_pooled_output_shape = (config.train_batch_size, config.d_model)
    self.assertEqual(pooled_output.shape, expected_pooled_output_shape)

  def test_ff_only_encoder(self):
    config = dummy_frozen_config(
        model_arch=base_config.ModelArchitecture.FF_ONLY)
    encoder = models.EncoderModel(config=config)
    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config)
    params = init_model_params(rng, encoder, init_batch)
    # Identity sublayers (for FF-only architecture) have no parameters so do not
    # show up in params.
    expected_keys = {
        "embedder", "encoder_0", "encoder_1", "feed_forward_0",
        "feed_forward_1", "pooler"
    }
    self.assertEqual(params.keys(), expected_keys)

    inputs = dummy_inputs(rng, config)
    hidden_states, pooled_output = encoder.apply({"params": params},
                                                 rngs={"dropout": rng},
                                                 **inputs)
    expected_hidden_states_shape = (config.train_batch_size,
                                    config.max_seq_length, config.d_model)
    self.assertEqual(hidden_states.shape, expected_hidden_states_shape)
    expected_pooled_output_shape = (config.train_batch_size, config.d_model)
    self.assertEqual(pooled_output.shape, expected_pooled_output_shape)

  def test_random_encoder(self):
    config = dummy_frozen_config(
        model_arch=base_config.ModelArchitecture.RANDOM)
    encoder = models.EncoderModel(config=config, random_seed=123)
    init_batch = init_encoder_batch(config)
    rng = jax.random.PRNGKey(0)
    params = init_model_params(rng, encoder, init_batch)
    expected_keys = {
        "embedder", "encoder_0", "encoder_1", "feed_forward_0",
        "feed_forward_1", "pooler"
    }
    self.assertEqual(params.keys(), expected_keys)

    inputs = dummy_inputs(rng, config)
    hidden_states, pooled_output = encoder.apply({"params": params},
                                                 rngs={"dropout": rng},
                                                 **inputs)
    expected_hidden_states_shape = (config.train_batch_size,
                                    config.max_seq_length, config.d_model)
    self.assertEqual(hidden_states.shape, expected_hidden_states_shape)
    expected_pooled_output_shape = (config.train_batch_size, config.d_model)
    self.assertEqual(pooled_output.shape, expected_pooled_output_shape)

  def test_pretraining_model(self):
    config = dummy_frozen_config(model_arch=base_config.ModelArchitecture.F_NET)
    model = models.PreTrainingModel(config=config)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config)
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

    inputs = dummy_inputs(rng, config)
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
    metrics = model.apply({"params": params}, rngs={"dropout": rng}, **inputs)
    expected_metrics = {
        "loss", "masked_lm_loss", "masked_lm_normalization",
        "masked_lm_correct", "masked_lm_total", "next_sentence_loss",
        "num_next_sentence_labels", "next_sentence_correct"
    }
    self.assertEqual(metrics.keys(), expected_metrics)

    # Because model is randomly initialized, we can only check the sign of most
    # metrics.
    self.assertGreater(metrics["loss"], 0.0)
    self.assertGreater(metrics["masked_lm_loss"], 0.0)
    self.assertGreater(metrics["next_sentence_loss"], 0.0)
    self.assertGreater(metrics["masked_lm_normalization"], 0.0)
    self.assertGreater(metrics["num_next_sentence_labels"], 0.0)
    self.assertGreater(metrics["masked_lm_total"], 0.0)

    # Number of correct labels is bound by the batch size.
    self.assertLessEqual(
        metrics["masked_lm_correct"],
        config.train_batch_size * config.max_predictions_per_seq)
    self.assertLessEqual(metrics["num_next_sentence_labels"],
                         config.train_batch_size)

  def test_classification_model(self):
    n_classes = 2

    config = dummy_frozen_config(model_arch=base_config.ModelArchitecture.BERT)
    model = models.SequenceClassificationModel(
        config=config, n_classes=n_classes)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config)
    params = init_model_params(rng, model, init_batch)
    self.assertEqual(params.keys(), {"encoder", "classification"})

    # Logits for eval/prediction (no labels supplied).
    eval_inputs = dummy_inputs(rng, config)
    eval_inputs["deterministic"] = True
    logits = model.apply({"params": params}, **eval_inputs)
    expected_logits_shape = (config.train_batch_size, n_classes)
    self.assertEqual(jnp.shape(logits), expected_logits_shape)

    # Metrics for training (labels supplied).
    train_inputs = dummy_inputs(rng, config)
    train_inputs["labels"] = jnp.ones(config.train_batch_size, jnp.int32)
    metrics = model.apply({"params": params},
                          rngs={"dropout": rng},
                          **train_inputs)
    self.assertEqual(metrics.keys(),
                     {"loss", "correct_predictions", "num_labels"})

  def test_regression_model(self):
    n_classes = 1  # Only one label for regression

    # "glue/stsb" is a regression task dataset.
    config = dummy_frozen_config(
        model_arch=base_config.ModelArchitecture.F_NET,
        dataset_name="glue/stsb")
    model = models.SequenceClassificationModel(
        config=config, n_classes=n_classes)

    rng = jax.random.PRNGKey(0)
    init_batch = init_encoder_batch(config)
    params = init_model_params(rng, model, init_batch)
    self.assertEqual(params.keys(), {"encoder", "classification"})

    # Logits for eval/prediction (no labels supplied).
    eval_inputs = dummy_inputs(rng, config)
    eval_inputs["deterministic"] = True
    logits = model.apply({"params": params}, **eval_inputs)
    expected_logits_shape = (config.train_batch_size, n_classes)
    self.assertEqual(jnp.shape(logits), expected_logits_shape)

    # Metrics for training (labels supplied).
    train_inputs = dummy_inputs(rng, config)
    _, label_key = jax.random.split(rng)
    train_inputs["labels"] = jax.random.uniform(
        label_key, (config.train_batch_size,), minval=0., maxval=1.)

    metrics = model.apply({"params": params},
                          rngs={"dropout": rng},
                          **train_inputs)
    self.assertEqual(metrics.keys(), {"loss", "num_labels"})


if __name__ == "__main__":
  absltest.main()
