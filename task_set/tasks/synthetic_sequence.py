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

# python3
"""Synthetic sequence problems.

These problems take a sequence of onehot encoded tokens, and predict another
sequence of tokens.

See copy_sequence, and associative_sequence in ../datasets.py for a description
of the problems used.
"""

import functools
from typing import Callable
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
import tensorflow.compat.v1 as tf


def sequence_to_sequence_rnn(
    core_fn):
  """A RNN based model for sequence to sequence prediction.

  This module takes a batch of data containing:
  * input: a [batch_size, seq_lenth, feature] onehot tensor.
  * output : a [batch_size, seq_lenth, feature] onehot tensor.
  * loss_mask: a [batch_size, seq_lenth] tensor.

  The input sequence encoded is passed it through a RNN, then a linear layer to
  the prediction dimension specified by the output. A cross entropy loss is then
  done comparing the predicted output with the actual outputs. A weighted
  average is then done using weights specified by the loss_mask.

  Args:
    core_fn: A fn that returns a sonnet RNNCore.

  Returns:
    A Callable that returns a snt.Module.
  """

  def _build(batch):
    """Build the sonnet module."""
    rnn = core_fn()

    initial_state = rnn.initial_state(batch["input"].shape[0])
    outputs, _ = tf.nn.dynamic_rnn(
        rnn,
        batch["input"],
        initial_state=initial_state,
        dtype=tf.float32,
        time_major=False)

    pred_logits = snt.BatchApply(snt.Linear(batch["output"].shape[2]))(outputs)

    flat_shape = [
        pred_logits.shape[0] * pred_logits.shape[1], pred_logits.shape[2]
    ]
    flat_pred_logits = tf.reshape(pred_logits, flat_shape)
    flat_actual_tokens = tf.reshape(batch["output"], flat_shape)
    flat_mask = tf.reshape(batch["loss_mask"], [flat_shape[0]])

    loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=flat_actual_tokens, logits=flat_pred_logits)
    total_loss = tf.reduce_sum(flat_mask * loss_vec)
    mean_loss = total_loss / tf.reduce_sum(flat_mask)

    return mean_loss

  return lambda: snt.Module(_build)

_rnn_mod_map = {
    "LSTM": snt.LSTM,
    "GRU": snt.GRU,
    "VRNN": snt.VanillaRNN,
}

# pylint: disable=bad-whitespace
_cfgs = [
    ("LSTM",  128, 128, 5,  20,),
    ("LSTM",  128, 128, 20, 50,),
    ("LSTM",  256, 128, 40, 100,),

    ("LSTM",  128, 128, 10, 50,),
    ("GRU",   128, 128, 10, 50,),
    ("VRNN",  128, 128, 10, 50,),

    ("LSTM",  256, 128, 20, 50,),
    ("GRU",   256, 128, 20, 50,),
    ("VRNN",  256, 128, 20, 50,),
]
# pylint: enable=bad-whitespace


def _make_associative_name(c):
  return "Associative_%s%d_BS%d_Pairs%d_Tokens%d" % c


def associative_fn(c):
  base_model_fn = sequence_to_sequence_rnn(lambda: _rnn_mod_map[c[0]](c[1]))
  return base.DatasetModelTask(
      base_model_fn,
      datasets.associative_sequence(c[2], num_pairs=c[3], num_tokens=c[4]))


for _cfg in _cfgs:
  registry.task_registry.register_fixed(_make_associative_name(_cfg))(
      functools.partial(associative_fn, _cfg))


# pylint: disable=bad-whitespace
_cfgs = [
    ("LSTM",  128, 128, 5, 10,),
    ("LSTM",  128, 128, 20, 20,),
    ("LSTM",  128, 128, 50, 5,),

    ("LSTM",  128, 128, 20, 10,),
    ("GRU",   128, 128, 20, 10,),
    ("VRNN",  128, 128, 20, 10,),

    ("LSTM",  256, 128, 40, 50,),
    ("GRU",   256, 128, 40, 50,),
    ("VRNN",  256, 128, 40, 50,),
]
# pylint: enable=bad-whitespace


def _make_copy_name(c):
  return "Copy_%s%d_BS%d_Length%d_Tokens%d" % c


def copy_fn(c):
  base_model_fn = sequence_to_sequence_rnn(lambda: _rnn_mod_map[c[0]](c[1]))
  return base.DatasetModelTask(
      base_model_fn,
      datasets.copy_sequence(
          c[2], sequence_length=c[3], num_separator=1, num_tokens=c[4]))


for _cfg in _cfgs:
  registry.task_registry.register_fixed(_make_copy_name(_cfg))(
      functools.partial(copy_fn, _cfg))
