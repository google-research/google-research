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

"""Language modeling tasks using RNN."""

from typing import Dict, Text, Any
import numpy as np
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
from task_set.tasks import utils
import tensorflow.compat.v1 as tf

LMConfig = Dict[Text, Any]


@registry.task_registry.register_sampler("char_rnn_language_model_family")
def sample_char_rnn_language_model_family_cfg(seed):
  """Samples a character NN language modeling task."""
  rng = np.random.RandomState(seed)
  cfg = {}
  cfg["embed_dim"] = utils.sample_log_int(rng, 8, 128)
  cfg["w_init"] = utils.sample_initializer(rng)

  full_vocab = utils.sample_bool(rng, 0.8)
  if full_vocab:
    cfg["vocab_size"] = 256
  else:
    # only operate on some subset of full words.
    cfg["vocab_size"] = utils.sample_log_int(rng, 100, 256)
  cfg["core"] = utils.sample_rnn_core(rng)
  cfg["trainable_init"] = bool(rng.choice([True, False]))

  cfg["dataset"] = utils.sample_char_lm_dataset(rng)
  return cfg


def _build_lm_task(cfg, dataset):
  """Builds a language modeling task."""

  def _build(batch):
    """Builds the sonnet module."""
    # Shape is [batch size, sequence length]
    inp = batch["text"]

    # Clip the vocab to be at most vocab_size.
    inp = tf.minimum(inp,
                     tf.to_int64(tf.reshape(cfg["vocab_size"] - 1, [1, 1])))

    embed = snt.Embed(vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"])
    embedded_chars = embed(inp)

    rnn = utils.get_rnn_core(cfg["core"])
    batch_size = inp.shape.as_list()[0]

    state = rnn.initial_state(batch_size, trainable=cfg["trainable_init"])

    outputs, state = tf.nn.dynamic_rnn(rnn, embedded_chars, initial_state=state)

    w_init = utils.get_initializer(cfg["w_init"])
    pred_logits = snt.BatchApply(
        snt.Linear(cfg["vocab_size"], initializers={"w": w_init}))(
            outputs[:, :-1])
    actual_output_tokens = inp[:, 1:]

    flat_s = [pred_logits.shape[0] * pred_logits.shape[1], pred_logits.shape[2]]
    flat_pred_logits = tf.reshape(pred_logits, flat_s)
    flat_actual_tokens = tf.reshape(actual_output_tokens, [flat_s[0]])

    loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_actual_tokens, logits=flat_pred_logits)
    return tf.reduce_mean(loss_vec)

  return base.DatasetModelTask(lambda: snt.Module(_build), dataset)


@registry.task_registry.register_getter("char_rnn_language_model_family")
def get_char_language_model_family(cfg):
  dataset_fn = utils.get_char_lm_dataset(cfg["dataset"])
  return _build_lm_task(cfg, dataset_fn)


@registry.task_registry.register_sampler("word_rnn_language_model_family")
def sample_word_language_model_family_cfg(seed):
  """Sample a word language model config."""
  rng = np.random.RandomState(seed)
  cfg = {}
  cfg["embed_dim"] = utils.sample_log_int(rng, 8, 128)
  cfg["w_init"] = utils.sample_initializer(rng)
  cfg["vocab_size"] = utils.sample_log_int(rng, 1000, 30000)
  cfg["core"] = utils.sample_rnn_core(rng)
  cfg["trainable_init"] = bool(rng.choice([True, False]))

  cfg["dataset"] = utils.sample_word_lm_dataset(rng)
  return cfg


@registry.task_registry.register_getter("word_rnn_language_model_family")
def get_word_language_model_family(cfg):
  dataset_fn = utils.get_word_lm_dataset(cfg["dataset"])
  return _build_lm_task(cfg, dataset_fn)
