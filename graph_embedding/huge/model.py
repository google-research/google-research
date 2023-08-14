# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""HUGE-TPU embedding model."""
from typing import Any, Mapping, Optional, Tuple

from absl import logging
import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_recommenders as tfrs


def initialize_tpu(
    tpu_address = "",
):
  """Initialize a TPU strategy given a particular machine address.

  Args:
    tpu_address: Optional string address of a TPU system. Will attempt to find a
      local TPU if not provided.

  Returns:
    A tf.distribute.experimental.TPUStrategy instance.
  """
  logging.info("Using TPU Strategy")
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
  if tpu.cluster_spec() is not None:
    logging.info("Running on TPU: %s", tpu.cluster_spec().as_dict()["worker"])
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.TPUStrategy(tpu)
  embedding_feature = strategy.extended.tpu_hardware_feature.embedding_feature
  assert (
      embedding_feature
      == tf.tpu.experimental.HardwareFeature.EmbeddingFeature.V1
  ), "Make sure that you have the right TPU Hardware"
  return strategy


def initialize_cpu():
  """Initialize the default CPU strategy.

  Returns:
    A tf.distribute.experimental.Strategy instance.
  """
  logging.info("Using default (CPU) strategy")
  return tf.distribute.get_strategy()


def load_txt_compression_map(filename):
  """Load a txt file compression map."""
  with tf.io.gfile.GFile(filename) as f:
    return [line.strip() for line in f]


def compute_total_batch_size(
    positive_batch_size, num_neg_per_pos
):
  """Compute the total batch size.

  Args:
    positive_batch_size: The number of positive examples per batch.
    num_neg_per_pos: The number of random negative examples per positive
      example.

  Returns:
    The global batch size after the positive examples are augmented with random
      uniform negative sampling.
  """
  return positive_batch_size * (1 + num_neg_per_pos)


def clip_positive_tensor(x, eps = 1e-5):
  """x is assumed to be positive. Eps is used iff x < eps.

  Args:
    x: A tensor, the max clipping value will be x.dtype.max
    eps: A minimal value for clipping

  Returns:
    A tensor representing the x clipped to [eps, x.dtype.max]
  """
  return tf.clip_by_value(x, eps, x.dtype.max)


def negative_log_graph_likelihood(
    positive_logits,
    negative_logits,
    expected_edge_score,
):
  """Compute the Negative Log Graph Likelihood loss.

  See "Watch Your Step: Learning Node Embeddings via Graph Attention":
    https://arxiv.org/pdf/1710.09599.pdf

  Args:
    positive_logits: Logits from positive node samples - source and destination
      pairs that are neighbors within the context window of the random walk.
    negative_logits: Logits from negative node samples - source and destination
      pairs that are not connected within the context window of the random walk.
    expected_edge_score: Expected Value of D_{u,v} as described in the "Watch
      your Step" paper.

  Returns:
    total_loss: Aggregated loss over positive and negative node sample pairs.
    positive_loss: A loss value (Tensor) associated with the positive
      source/destination pairs.
    negative_loss: Loss value (Tensor) associated with the set of negative
      source/destination pairs.
  """
  positive_loss = -tf.math.multiply(
      expected_edge_score, tf.math.log(clip_positive_tensor(positive_logits))
  )

  negative_loss = -tf.math.log(clip_positive_tensor(1 - negative_logits))

  all_loss = tf.concat([positive_loss, negative_loss], axis=0)

  total_loss = tf.math.reduce_sum(all_loss)

  return (
      total_loss,
      tf.math.reduce_sum(positive_loss),
      tf.math.reduce_sum(negative_loss),
  )


# The TF2 TPU Embedding layer can't currently handle nested learning rate
# schedules. Providing this wrapper which instantiates and calls the nested
# learning rate schedule avoids the issue.
@tf.keras.utils.register_keras_serializable("tensorflow_models.optimization")
class PolyWarmupWithPolyDecaySchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
  """Polynomial warmup learning rate schedule with optional decay."""

  def __init__(
      self,
      warmup_steps,
      warmup_power,
      warmup_end_lr,
      warmup_decay_steps = 0,
      warmup_decay_power = 1.0,
      warmup_decay_end_lr = 0.001,
      name = "PolyWarmupWithPolyDecaySchedule",
  ):
    """Polynomial warmup learning rate schedule.

    Defaults to linear warmup without decay.

    Args:
      warmup_steps: Intenger number of warmup steps
      warmup_power: Exponent applied to warmup rate. Defaults to 1.0
      warmup_end_lr: The endpoint for the warmup learning rate.
      warmup_decay_steps: The number of decay steps after completing warmup. No
        polynomial decay will be added if `warmup_decay_steps` is zero or less
        and the learning rate will remain constant at `warmup_end_lr`.
      warmup_decay_power: Exponent applied to decay rate. Defaults to 1.0.
      warmup_decay_end_lr: Final learning rate after decay period.
      name: String name learning rate schedule config.

    Returns:
      PolyWarmupWithPolyDecaySchedule instance.
    """
    super().__init__()

    self._warmup_steps = warmup_steps
    self._warmup_power = warmup_power
    self._warmup_end_lr = warmup_end_lr
    self._warmup_decay_steps = warmup_decay_steps
    self._warmup_decay_power = warmup_decay_power
    self._warmup_decay_end_lr = warmup_decay_end_lr

    self._after_warmup_lr_schedule = warmup_end_lr
    if warmup_decay_steps is not None and warmup_decay_steps > 0:
      self._after_warmup_lr_schedule = (
          tf.keras.optimizers.schedules.PolynomialDecay(
              initial_learning_rate=warmup_end_lr,
              decay_steps=warmup_decay_steps,
              power=warmup_decay_power,
              end_learning_rate=warmup_decay_end_lr,
          )
      )

    self._lr_schedule = tfm.optimization.PolynomialWarmUp(
        after_warmup_lr_sched=self._after_warmup_lr_schedule,
        warmup_steps=self._warmup_steps,
        power=self._warmup_power,
    )

    self._name = name

  def get_config(self):
    config = {
        "warmup_steps": self._warmup_steps,
        "warmup_power": self._warmup_power,
        "warmup_end_lr": self._warmup_end_lr,
        "name": self._name,
    }

    if self._warmup_decay_steps is not None and self._warmup_decay_steps > 0:
      config.update({
          "warmup_decay_steps": self._warmup_decay_steps,
          "warmup_decay_power": self._warmup_decay_power,
          "warmup_decay_end_lr": self._warmup_decay_end_lr,
      })

    return config

  def __call__(self, step):
    return self._lr_schedule(step)


def create_optimizer(
    name, strategy, **kwargs
):
  """Create a strategy-enabled optimizer.

  Args:
    name: String, only "SGD" or "WARMUP_WITH_POLY_DECAY" are currently
      supported.
    strategy: A tf.distribute.Strategy object.
    **kwargs: Additional optimizer-specific keyword arguments.

  Returns:
    An instantiated keras optimizer.

  Raises:
    ValueError if an unsupported opitmizer name is specified.
  """
  with strategy.scope():
    if name.lower() == "sgd":
      learning_rate = kwargs.get("learning_rate", 0.001)
      logging.info("Optimizer: SGD with learning rate %f", learning_rate)
      return tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

    elif name.lower() == "warmup_with_poly_decay":
      learning_rate = PolyWarmupWithPolyDecaySchedule(
          kwargs.get("warmup_steps", 1000),
          kwargs.get("warmup_power", 1.0),
          kwargs.get("warmup_end_lr", 0.01),
          kwargs.get("warmup_decay_steps", 0),
          kwargs.get("warmup_decay_power", 1.0),
          kwargs.get("warmup_decay_end_lr", 0.001),
      )
      logging.info(
          "Optimizer: SGD with PolynomialWarmUp: %s", learning_rate.get_config()
      )
      return tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

    else:
      raise ValueError(f"Unsupported optimizer name: {name}")


def huge_model(
    num_nodes,
    embedding_dim,
    total_batch_size,
    strategy,
    optimizer,
    cosine_adjustment = None,
    pipeline_execution_with_tensor_core = False,
    initializer = None,
):
  """Build a HUGE-TPU Keras model.

  Args:
    num_nodes: Number of nodes in the input graph. This is equal to the
      embedding vocabulary size.
    embedding_dim: The desired embedding dimension, typically a power of two.
    total_batch_size: Total batch size, the number of examples that will be
      input into the embedding layers.
    strategy: A tf.distribute.Strategy object.
    optimizer: A TPU-supported optimizer instance.
    cosine_adjustment: Optional cosine adjustment factor. It has been observed
      that scaling the cosine similarity prior to passing through a sigmoid
      function may help the expressivity of the model. If supplied, the cosine
      similarity will be scaled by `cosine_adjustment`.
    pipeline_execution_with_tensor_core: Option to pipeline (overlap) SparseCore
      lookups with TensorCore execution. This may result in speed improvments
      but may (or may not) degrade performance due to a number of factors.
      Consult the tpu_embedding_layer documentation for further details.
    initializer: An optional tf.keras.initializer.Initializer. Defaults to
      tf.initializer.TruncatedNormal(mean=0.0, stddev=0.02). Useful for
      tests/debugging.

  Returns:
    A tf.keras.Model
  """
  with strategy.scope():
    table_config = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=num_nodes,
        dim=embedding_dim,
        combiner="mean",
        initializer=initializer,
        name="embedding",
    )

    feature_config = {
        "src": tf.tpu.experimental.embedding.FeatureConfig(table=table_config),
        "dst": tf.tpu.experimental.embedding.FeatureConfig(table=table_config),
    }

    inputs = {
        "src": tf.keras.layers.Input(
            batch_size=total_batch_size,
            shape=(),
            name="src",
            dtype=tf.int64,
        ),
        "dst": tf.keras.layers.Input(
            batch_size=total_batch_size,
            shape=(),
            name="dst",
            dtype=tf.int64,
        ),
    }

    embedded_features = tfrs.layers.embedding.TPUEmbedding(
        feature_config,
        optimizer,
        pipeline_execution_with_tensor_core=pipeline_execution_with_tensor_core,
    )(inputs)

    src_embeddings = embedded_features["src"]
    dst_embeddings = embedded_features["dst"]

    sim = tf.math.reduce_sum(
        tf.multiply(
            tf.nn.l2_normalize(src_embeddings, 1),
            tf.nn.l2_normalize(dst_embeddings, 1),
        ),
        axis=1,
    )

    if cosine_adjustment:
      sim *= tf.constant(cosine_adjustment, dtype=tf.float32)

    logits = tf.sigmoid(sim)

  return tf.keras.Model(inputs=inputs, outputs=logits)
