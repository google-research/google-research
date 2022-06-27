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

"""Fixed tasks containing text classification using RNN."""
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
import tensorflow.compat.v1 as tf


def imdb_subword(batch_size, patch_length):
  return datasets.random_slice_text_data(
      dataset_name="imdb_reviews/subwords8k",
      batch_size=batch_size,
      cache_dataset=True,
      patch_length=patch_length)


def rnn_classification(core_fn,
                       vocab_size=10000,
                       embed_dim=64,
                       aggregate_method="last"):
  """Helper for RNN based text classification tasks.

  Args:
    core_fn: callable callable that returns a sonnet RNN core
    vocab_size: int number of words to use for the embedding table. All index
      higher than this will be clipped
    embed_dim: int size of the embedding dim
    aggregate_method: str how to aggregate the sequence of features. If 'last'
      grab the last hidden features. If 'avg' compute the average over the full
      sequence.

  Returns:
    a callable that returns a sonnet module representing the loss.
  """

  def _build(batch):
    """Build the loss sonnet module."""
    # TODO(lmetz) these are dense updates.... so keeping this small for now.
    tokens = tf.minimum(batch["text"],
                        tf.to_int64(tf.reshape(vocab_size - 1, [1, 1])))
    embed = snt.Embed(vocab_size=vocab_size, embed_dim=embed_dim)
    embedded_tokens = embed(tokens)
    rnn = core_fn()
    bs = tokens.shape.as_list()[0]

    state = rnn.initial_state(bs, trainable=True)

    outputs, state = tf.nn.dynamic_rnn(
        rnn, embedded_tokens, initial_state=state)
    if aggregate_method == "last":
      last_output = outputs[:, -1]  # grab the last output
    elif aggregate_method == "avg":
      last_output = tf.reduce_mean(outputs, [1])  # average over length
    else:
      raise ValueError("not supported aggregate_method")

    logits = snt.Linear(2)(last_output)

    loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=batch["label_onehot"], logits=logits)

    return tf.reduce_mean(loss_vec)

  return lambda: snt.Module(_build)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch32_LSTM128_bs128")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.LSTM(128), embed_dim=64, aggregate_method="last")
  dataset = imdb_subword(128, 32)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch128_LSTM128_bs64")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.LSTM(128), embed_dim=64, aggregate_method="last")
  dataset = imdb_subword(64, 128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch32_LSTM128_E128_bs128")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.LSTM(128), embed_dim=128, aggregate_method="last")
  dataset = imdb_subword(128, 32)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch128_LSTM128_embed128_bs64")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.LSTM(128), embed_dim=128, aggregate_method="last")
  dataset = imdb_subword(64, 128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch128_LSTM128_avg_bs64")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.LSTM(128), embed_dim=64, aggregate_method="avg")
  dataset = imdb_subword(64, 128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch32_GRU128_bs128")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.GRU(128), embed_dim=64, aggregate_method="last")
  dataset = imdb_subword(128, 32)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.GRU(64), embed_dim=64, aggregate_method="avg")
  dataset = imdb_subword(128, 32)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch32_VRNN128_tanh_bs128")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.VanillaRNN(128), embed_dim=64, aggregate_method="last")
  dataset = imdb_subword(128, 32)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch32_VRNN64_tanh_avg_bs128")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.VanillaRNN(64), embed_dim=64, aggregate_method="avg")
  dataset = imdb_subword(128, 32)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch32_VRNN64_relu_avg_bs128")
def _():
  base_model_fn = rnn_classification(
      lambda: snt.VanillaRNN(64, activation=tf.nn.relu),
      embed_dim=64,
      aggregate_method="avg")
  dataset = imdb_subword(128, 32)
  return base.DatasetModelTask(base_model_fn, dataset)


def _get_irnn_cell_fn(num_unit):
  init = {snt.VanillaRNN.HIDDEN_TO_HIDDEN: {"w": tf.initializers.identity(1.0)}}

  def rnn():
    return snt.VanillaRNN(num_unit, activation=tf.nn.relu, initializers=init)

  return rnn


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_avg_bs128")
def _():
  base_model_fn = rnn_classification(
      _get_irnn_cell_fn(64), embed_dim=64, aggregate_method="avg")
  dataset = imdb_subword(128, 32)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_last_bs128")
def _():
  base_model_fn = rnn_classification(
      _get_irnn_cell_fn(64), embed_dim=64, aggregate_method="last")
  dataset = imdb_subword(128, 32)
  return base.DatasetModelTask(base_model_fn, dataset)
