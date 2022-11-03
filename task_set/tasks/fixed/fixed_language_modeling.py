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

"""Fixed language modeling tasks."""

import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
import tensorflow.compat.v1 as tf


def lm1b_byte(batch_size, patch_length):
  return datasets.random_slice_text_data(
      dataset_name="lm1b/bytes",
      batch_size=batch_size,
      patch_length=patch_length,
      cache_dataset=True,
      shuffle_buffer=100000)


def teacher_force_language_modeling(core_fn, embed_dim=32):
  """Helper for teacher forced language modeling.

  Args:
    core_fn: callable callable that returns a sonnet RNN core.
    embed_dim: int size of the embedding table.

  Returns:
    callable that returns a sonnet module representing the loss.
  """

  def _fn(batch):
    """Compute the loss from the given batch."""
    # Shape is [bs, seq len, features]
    inp = batch["text"]
    mask = batch["mask"]
    embed = snt.Embed(vocab_size=256, embed_dim=embed_dim)
    embedded_chars = embed(inp)

    rnn = core_fn()
    bs = inp.shape.as_list()[0]

    state = rnn.initial_state(bs, trainable=True)

    outputs, state = tf.nn.dynamic_rnn(rnn, embedded_chars, initial_state=state)

    pred_logits = snt.BatchApply(snt.Linear(256))(outputs[:, :-1])
    actual_tokens = inp[:, 1:]

    flat_s = [pred_logits.shape[0] * pred_logits.shape[1], pred_logits.shape[2]]
    f_pred_logits = tf.reshape(pred_logits, flat_s)
    f_actual_tokens = tf.reshape(actual_tokens, [flat_s[0]])
    f_mask = tf.reshape(mask[:, 1:], [flat_s[0]])

    loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=f_actual_tokens, logits=f_pred_logits)
    total_loss = tf.reduce_sum(f_mask * loss_vec)
    mean_loss = total_loss / tf.reduce_sum(f_mask)

    return mean_loss

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed(
    "FixedLM_lm1b_patch128_GRU64_embed64_avg_bs128")
def _():
  base_model_fn = teacher_force_language_modeling(
      lambda: snt.GRU(64), embed_dim=64)
  dataset = lm1b_byte(128, 128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedLM_lm1b_patch128_GRU128_embed64_avg_bs128")
def _():
  base_model_fn = teacher_force_language_modeling(
      lambda: snt.GRU(128), embed_dim=64)
  dataset = lm1b_byte(128, 128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedLM_lm1b_patch128_GRU256_embed64_avg_bs128")
def _():
  base_model_fn = teacher_force_language_modeling(
      lambda: snt.GRU(256), embed_dim=64)
  dataset = lm1b_byte(128, 128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedLM_lm1b_patch128_LSTM128_embed64_avg_bs128")
def _():
  base_model_fn = teacher_force_language_modeling(
      lambda: snt.LSTM(128), embed_dim=64)
  dataset = lm1b_byte(128, 128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedLM_lm1b_patch128_LSTM256_embed64_avg_bs128")
def _():
  base_model_fn = teacher_force_language_modeling(
      lambda: snt.LSTM(256), embed_dim=64)
  dataset = lm1b_byte(128, 128)
  return base.DatasetModelTask(base_model_fn, dataset)
