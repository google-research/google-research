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

"""Fast clipping profiling tools."""
import time

import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.fast_gradient_clipping import clip_grads



def compute_sample_gradient_norms(model, batch_size, registry):
  """Computes gradient norms on a model using dummy data."""
  inputs = []
  for spec in model.input_spec:
    input_shape = (batch_size,) + spec.shape[1:]
    inputs.append(tf.ones(shape=input_shape))
  output_shape = (batch_size,) + model.output_shape[1:]
  outputs = tf.zeros(shape=output_shape)
  return clip_grads.compute_gradient_norms(model, registry, inputs, outputs)


def get_compute_profile(model, batch_size, registry, repeats=20):
  """Gets runtime (sec) and peak memory usage (MB) on norm computation."""
  times = []
  memories = []
  for _ in range(repeats):
    t0 = time.time()
    _ = compute_sample_gradient_norms(model, batch_size, registry)
    t1 = time.time()
    times.append(t1 - t0)
    # Memory profiling currently uses Google internal tools; use a stub if we
    # are in open-source mode.
    if not memories:
      memories = [0.0]

  return np.median(times), np.median(memories)


def compute_sample_gradient_norms_with_vocab(
    model, vocab_size, num_queries, registry, seed=777
):
  """Computes gradient norms on a model using dummy data."""
  tf.keras.utils.set_random_seed(seed)
  inputs = tf.sort(
      tf.random.uniform(
          shape=(1, num_queries), minval=0, maxval=vocab_size, dtype=tf.int32
      )
  )
  outputs = tf.zeros(shape=(1, 1))
  return clip_grads.compute_gradient_norms(model, registry, inputs, outputs)


def get_compute_profile_with_vocab(
    model, vocab_size, num_queries, registry, repeats=20
):
  """Gets runtime (sec) and peak memory usage (MB) on norm computation."""
  times = []
  memories = []
  for _ in range(repeats):
    t0 = time.time()
    compute_sample_gradient_norms_with_vocab(
        model, vocab_size, num_queries, registry
    )
    t1 = time.time()
    times.append(t1 - t0)
    # Memory profiling currently uses Google internal tools; use a stub if we
    # are in open-source mode.
    if not memories:
      memories = [0.0]
  return np.median(times), np.median(memories)


def train_bert_model(
    model,
    batch_size,
    vocab_size,
    query_size,
    num_epochs,
    num_steps,
):
  """Trains a BERT model using dummy data."""
  sample_size = batch_size * num_epochs * num_steps
  x_batch = [
      tf.random.uniform(
          shape=(sample_size, query_size),
          minval=0,
          maxval=vocab_size,
          dtype=tf.int32,
      ),
      tf.ones(shape=(sample_size, query_size)),
      tf.random.uniform(
          shape=(sample_size, query_size),
          minval=0,
          maxval=vocab_size,
          dtype=tf.int32,
      ),
  ]
  y_batch = tf.ones(shape=(sample_size, 1))
  model.fit(
      x_batch,
      y_batch,
      batch_size=batch_size,
      epochs=num_epochs,
      steps_per_epoch=num_steps,
      verbose=0,
  )


def get_train_bert_model_compute_profile(
    model,
    batch_size,
    vocab_size,
    query_size,
    num_epochs,
    num_steps,
    repeats=1,
):
  """Gets runtime (sec) and peak memory usage (MB) on model fitting."""
  times = []
  memories = []
  for _ in range(repeats):
    t0 = time.time()
    train_bert_model(
        model,
        batch_size,
        vocab_size,
        query_size,
        num_epochs,
        num_steps,
    )
    t1 = time.time()
    times.append(t1 - t0)
    # Memory profiling currently uses Google internal tools; use a stub if we
    # are in open-source mode.
  return np.median(times), np.median(memories)
