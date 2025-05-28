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

"""Test for HUGE-TPU graph embedding model."""
import os
import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from graph_embedding.huge import io
from graph_embedding.huge import model as huge_model_lib


# Using the empty string, the TPUClusterResolver will try to automatically
# connect to a TPU cluster and will find the (linked) simulated device.
_TPU = ""


class ModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.strategy = huge_model_lib.initialize_tpu(_TPU)
    self.walk_length = 4

    self.positive_batch_size = 2
    self.num_negs_per_pos = 1
    self.total_batch_size = huge_model_lib.compute_total_batch_size(
        self.positive_batch_size, self.num_negs_per_pos
    )
    self.vocabulary_size = 3
    self.embedding_dim = 2
    self.embedding_values = np.array(
        list(range(self.vocabulary_size * self.embedding_dim)), dtype=np.float64
    )
    self.embedding_initializer = tf.constant_initializer(self.embedding_values)

    self.base_dir = os.getcwd()

  def tearDown(self):
    tf.tpu.experimental.shutdown_tpu_system()
    super().tearDown()

  def _create_dense_distributed_dataset(
      self,
  ):
    src = tf.random.uniform(
        shape=(self.positive_batch_size,),
        minval=0,
        maxval=self.vocabulary_size,
        dtype=tf.int64,
    )
    dst = tf.random.uniform(
        shape=(self.positive_batch_size,),
        minval=0,
        maxval=self.vocabulary_size,
        dtype=tf.int64,
    )
    co_occurrence = tf.random.uniform(
        shape=(self.positive_batch_size, self.walk_length),
        minval=0.0,
        maxval=25.0,
        dtype=tf.float32,
    )

    def input_fn(ctx):
      if ctx:
        batch_size = ctx.get_per_replica_batch_size(self.positive_batch_size)
      else:
        batch_size = self.positive_batch_size

      ds = (
          tf.data.Dataset.from_tensor_slices((src, dst, co_occurrence))
          .repeat()
          .batch(batch_size, drop_remainder=True)
      )
      ds = io.add_uniform_random_negatives(
          ds,
          num_nodes=self.vocabulary_size,
          num_negs_per_pos=self.num_negs_per_pos,
      )
      ds = io.add_expected_edge_score(
          ds, weights=tf.constant([1.0] * self.walk_length, dtype=tf.float32)
      )
      return ds

    return self.strategy.distribute_datasets_from_function(
        input_fn,
        options=tf.distribute.InputOptions(experimental_fetch_to_device=False),
    )

  @parameterized.named_parameters(
      (
          "trivial",
          [
              1.0,
              1.0,
              1.0,
          ],
          4.158,
          2.079,
          2.079,
      ),
      ("random", [2.0, 4.0, 0], 6.238, 4.158, 2.079),
  )
  def test_nlgl_loss(self, edgescores, tl, pl, nl):
    positive_logits = tf.constant([0.5, 0.5, 0.5], dtype=tf.dtypes.float32)
    negative_logits = tf.constant([0.5, 0.5, 0.5], dtype=tf.dtypes.float32)
    expected_edge_scores = tf.constant(edgescores, dtype=tf.dtypes.float32)

    total_loss, positive_loss, negative_loss = (
        huge_model_lib.negative_log_graph_likelihood(
            positive_logits, negative_logits, expected_edge_scores
        )
    )

    self.assertAllClose(tl, total_loss, atol=1e-3)
    self.assertAllClose(pl, positive_loss, atol=1e-3)
    self.assertAllClose(nl, negative_loss, atol=1e-3)

  @parameterized.named_parameters(
      (
          "sgd",
          "sgd",
          {"learning_rate": 0.1},
      ),
      (
          "warmup",
          "warmup_with_poly_decay",
          {
              "warmup_steps": 200,
              "warmup_power": 0.5,
              "warmup_end_lr": 0.01,
              "warmup_decay_steps": 50,
              "warmup_decay_power": 0.5,
              "warmup_decay_end_lr": 0.001,
          },
      ),
  )
  def test_runs_ok(self, optimizer_name, optimizer_kwargs):
    """Test the optimizer/model on simulated TPUs.

    Args:
      optimizer_name: A name of an optimizer
      optimizer_kwargs: Optimizer keyword arguments.
    """
    optimizer = huge_model_lib.create_optimizer(
        optimizer_name, self.strategy, **optimizer_kwargs
    )

    model = huge_model_lib.huge_model(
        num_nodes=self.vocabulary_size,
        embedding_dim=self.embedding_dim,
        total_batch_size=self.total_batch_size,
        strategy=self.strategy,
        optimizer=optimizer,
        initializer=self.embedding_initializer,
    )

    ds = self._create_dense_distributed_dataset()
    ds_iter = iter(ds)

    @tf.function
    def test_fn():
      def step(inputs):
        src, dst, _ = inputs
        return model(inputs={"src": src, "dst": dst})

      return self.strategy.run(step, args=(next(ds_iter),))

    logits = test_fn()

    # Check we get some values out of the model with the expected shape and
    # type.
    self.assertLen(logits.values, self.strategy.num_replicas_in_sync)
    for i in range(self.strategy.num_replicas_in_sync):
      self.assertIsInstance(logits.values[i], tf.Tensor)
      self.assertEqual(
          logits.values[i].shape,
          (self.total_batch_size // self.strategy.num_replicas_in_sync,),
      )
      self.assertEqual(logits.values[i].dtype, tf.float32)

  @parameterized.named_parameters(
      (
          "sgd",
          "sgd",
          {"learning_rate": 0.1},
      ),
      (
          "warmup",
          "warmup_with_poly_decay",
          {
              "warmup_steps": 200,
              "warmup_power": 0.5,
              "warmup_end_lr": 0.01,
              "warmup_decay_steps": 50,
              "warmup_decay_power": 0.5,
              "warmup_decay_end_lr": 0.001,
          },
      ),
  )
  def test_training_runs_ok(self, optimizer_name, optimizer_kwargs):
    """Test training for a few steps on simulated hardware."""
    positive_batch_size = 2
    num_negs_per_pos = 1
    epochs = 1
    train_steps = 2
    nhost_steps = 1
    total_batch_size = huge_model_lib.compute_total_batch_size(
        positive_batch_size, num_negs_per_pos
    )
    optimizer = huge_model_lib.create_optimizer(
        optimizer_name, self.strategy, **optimizer_kwargs
    )

    model = huge_model_lib.huge_model(
        num_nodes=self.vocabulary_size,
        embedding_dim=self.embedding_dim,
        total_batch_size=total_batch_size,
        strategy=self.strategy,
        optimizer=optimizer,
        initializer=self.embedding_initializer,
    )

    ds = self._create_dense_distributed_dataset()
    ds_iter = iter(ds)
    model_dir = str(tempfile.mkdtemp())
    huge_model_lib.train(
        model,
        optimizer,
        self.strategy,
        ds_iter,
        model_dir,
        epochs=epochs,
        train_steps=train_steps,
        nhost_steps=nhost_steps,
        positive_batch_size=positive_batch_size,
        num_negs_per_pos=num_negs_per_pos,
    )


