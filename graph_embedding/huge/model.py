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

"""HUGE-TPU embedding model."""
import os
from typing import Any, Iterator, Mapping, Optional, Tuple

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


def create_or_restore_checkpoint_manager(
    model,
    optimizer,
    strategy,
    ckpt_dir,
    max_to_keep = 2,
    checkpoint_interval = 1000,
):
  """Create a checkpoint manager that may instantiate or restore a model.

  Args:
    model: A HUGE-TPU model as a tf.keras.Model instance.
    optimizer: A TPUEmbedding compatible optimizer.
    strategy: A tf.distribute.Strategy instance.
    ckpt_dir: String checkpoint path specification.
    max_to_keep: Integer number of checkpoints to keeps.
    checkpoint_interval: Integer checkpoint interval.

  Returns:
    A tf.train.CheckpointManager object.
  """
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      ckpt_dir,
      max_to_keep=max_to_keep,
      step_counter=optimizer.iterations,
      checkpoint_interval=checkpoint_interval,
  )

  if checkpoint_manager.latest_checkpoint:
    logging.info(
        "Restoring from checkpoint: %s", checkpoint_manager.latest_checkpoint
    )

    with strategy.scope():
      checkpoint.restore(checkpoint_manager.latest_checkpoint)

    logging.info(
        "Restored checkpoint at step: %d", optimizer.iterations.numpy()
    )

  return checkpoint_manager


class HugeMetrics:
  """Huge-TPU metrics suitable for distributed TPU training."""

  def __init__(
      self,
      strategy,
      logs_dir,
      num_negs_per_pos = None,
  ):
    """Initialize a HugeMetrics object.

    Args:
      strategy: A tf.distribute.Strategy object.
      logs_dir: Path specification for writing tensorboard metrics.
      num_negs_per_pos: Integer number of negative samples per positive samples
        in a training batch.
    """
    self.metrics = {}
    self.logs_dir = logs_dir
    self.writer = tf.summary.create_file_writer(self.logs_dir)
    self.strategy = strategy
    if num_negs_per_pos is not None:
      assert num_negs_per_pos > 0, "num_negs_per_pos must be greater than 0."
    self.num_negs_per_pos = num_negs_per_pos

    with strategy.scope():
      self.metrics["total_loss"] = tf.keras.metrics.Mean(name="total_loss")
      self.metrics["positive_loss"] = tf.keras.metrics.Mean(
          name="positive_loss"
      )
      self.metrics["negative_loss"] = tf.keras.metrics.Mean(
          name="negative_loss"
      )
      self.metrics["accuracy"] = tf.keras.metrics.Mean(name="accuracy")
      self.metrics["precision"] = tf.keras.metrics.Mean(name="precision")
      self.metrics["recall"] = tf.keras.metrics.Mean(name="recall")
      self.metrics["f1"] = tf.keras.metrics.Mean(name="f1")
      self.metrics["tp"] = tf.keras.metrics.Mean(name="TP")
      self.metrics["tn"] = tf.keras.metrics.Mean(name="TN")
      self.metrics["fp"] = tf.keras.metrics.Mean(name="FP")
      self.metrics["fn"] = tf.keras.metrics.Mean(name="FN")
      self.metrics["steps_per_second"] = tf.keras.metrics.Mean(
          name="steps_per_second"
      )

  def update(
      self,
      total_loss,
      positive_loss,
      negative_loss,
      positive_logits,
      negative_logits,
  ):
    """Update the huge-tpu metrics.

    Args:
      total_loss: A tf.Tensor containing the total model loss.
      positive_loss: A tf.Tensor containing the positive portion of the loss.
      negative_loss: A tf.Tensor containing the negative portion of the loss.
      positive_logits: Model output corresponding to positive examples.
      negative_logits: Model output corresponding to negative examples.
    """
    # Approximate in-batch confusion matrix.
    tp = tf.math.count_nonzero(positive_logits > 0.5, dtype=tf.float32)
    fn = tf.math.count_nonzero(positive_logits <= 0.5, dtype=tf.float32)
    tn = tf.math.count_nonzero(negative_logits <= 0.5, dtype=tf.float32)
    fp = tf.math.count_nonzero(negative_logits > 0.5, dtype=tf.float32)

    # There are many more negatives than positives: scale the negatives s.t.
    # accuracy and recall are more intelligible.
    if self.num_negs_per_pos:
      tn = tn / self.num_negs_per_pos
      fp = fp / self.num_negs_per_pos

    accuracy = tf.divide(tf.add(tp, tn), tf.add(tp, tf.add(tn, tf.add(fp, fn))))
    precision = tf.divide(tp, tf.add(tp, fp))
    recall = tf.divide(tp, tf.add(tp, fn))
    f1 = tf.divide(
        tf.multiply(tp, 2), tf.add(tf.multiply(tp, 2), tf.add(fp, fn))
    )

    self.metrics["total_loss"].update_state(total_loss)
    self.metrics["positive_loss"].update_state(positive_loss)
    self.metrics["negative_loss"].update_state(negative_loss)
    self.metrics["tp"].update_state(tp)
    self.metrics["fn"].update_state(fn)
    self.metrics["tn"].update_state(tn)
    self.metrics["fp"].update_state(fp)
    self.metrics["accuracy"].update_state(accuracy)
    self.metrics["precision"].update_state(precision)
    self.metrics["recall"].update_state(recall)
    self.metrics["f1"].update_state(f1)

  def reset_states(self):
    """Reset the internal states of the metric tracking variables."""
    _ = [m.reset_states() for m in self.metrics.values()]

  def summarize(self, current_step):
    """Write metric summaries.

    Args:
      current_step: A tf.Tensor containing the current step, typically provided
        by an optimizer.
    """
    with self.writer.as_default():
      with tf.name_scope("Loss"):
        tf.summary.scalar(
            "Total", self.metrics["total_loss"].result(), current_step
        )
        tf.summary.scalar(
            "Positive", self.metrics["positive_loss"].result(), current_step
        )
        tf.summary.scalar(
            "Negative", self.metrics["negative_loss"].result(), current_step
        )

      with tf.name_scope("TrainingMetrics"):
        tf.summary.scalar(
            "accuracy", self.metrics["accuracy"].result(), current_step
        )
        tf.summary.scalar(
            "precision", self.metrics["precision"].result(), current_step
        )
        tf.summary.scalar(
            "recall", self.metrics["recall"].result(), current_step
        )
        tf.summary.scalar("f1", self.metrics["f1"].result(), current_step)

      tp = self.metrics["tp"].result()
      fp = self.metrics["fp"].result()
      fn = self.metrics["fn"].result()
      tn = self.metrics["tn"].result()
      tpr = tp / (tp + fn)
      tnr = tn / (tn + fp)
      fpr = fp / (fp + tn)
      fnr = fn / (fp + tn)
      with tf.name_scope("TrainingConfusionMatrix"):
        tf.summary.scalar("TPR", tpr, current_step)
        tf.summary.scalar("TNR", tnr, current_step)
        tf.summary.scalar("FPR", fpr, current_step)
        tf.summary.scalar("FNR", fnr, current_step)

      # TODO(bmayer): Is this really necessary?
      self.writer.flush()

  def __str__(self):
    return (
        f"total_loss: {self.metrics['total_loss'].result().numpy()}, "
        f"positive_loss: {self.metrics['positive_loss'].result().numpy()}, "
        f"negative_loss: {self.metrics['negative_loss'].result().numpy()}"
    )


def train(
    model,
    optimizer,
    strategy,
    ds_iter,
    model_dir,
    epochs,
    train_steps,
    nhost_steps,
    positive_batch_size,
    num_negs_per_pos,
    logs_dir = None,
    async_checkpoint = False,
):
  """Train a HUGE-TPU model.

  Args:
    model: A huge-tpu model.
    optimizer: A tf.keras.optimizer.Optimizer class.
    strategy: a tf.distribute.Strategy object.
    ds_iter: An iterator over a tf.data.Dataset or tf.distributed.Dataset.
    model_dir: A path specification for writing metric summaries and saving
      checkpoints. If there are checkpoints under `model_dir`, the model and
      optimizer will be reloaded and training will resume from
      `step=optimizer.iterations`.
    epochs: Integer number of desired epochs.
    train_steps: Integer number of training steps per epoch.
    nhost_steps: Integer number of host loops per train_steps. Note the
      constraint that the `train_steps` must be a multiple of `nhost_stesp`,
      i.e.,`train_steps % nhost_steps == 0`.
    positive_batch_size: Integer number of positive examples per training batch.
    num_negs_per_pos: Integer number of random negative samples to draw for each
      positive example. E.g., `total_batch_size = positive_batch_size * (1 +
      num_negs_per_pos)`.
    logs_dir: Optional log directory for tensorboard summaries. If not provided,
      will write summaries to `model_dir`.
    async_checkpoint: Boolean option to enable async checkpoint writing. This
      will allow training to continue while a model. While this may result in
      significant wall clock savings per epoch, this will consume extra host
      memory, beware of OOMs. Defaults to False.

  Raises:
    ValueError if `positive_batch_size` is not divisible by
    `strategy.num_replicas_in_synce` or if `train_steps` is not divisible by
    `nhost_steps`.
  """

  if positive_batch_size % strategy.num_replicas_in_sync != 0:
    raise ValueError(
        f"positive_batch_size: {positive_batch_size} should be divisible by"
        f" strategy.num_replicas_in_sync: {strategy.num_replicas_in_sync}"
    )

  if train_steps % nhost_steps != 0:
    raise ValueError(
        f"train_steps: {train_steps} should be divisible by nhost_steps:"
        f" {nhost_steps}"
    )

  # Turn num hosts steps into a tensor, this is needed as sometimes passing
  # non-tensor python objects to tf.functions causes re-tracing.
  nhost_steps_t = tf.constant(nhost_steps, dtype=tf.int64)

  per_replica_positive_batch_size = (
      positive_batch_size // strategy.num_replicas_in_sync
  )

  logging.info(
      "per_replica_positive_batch_size: %s", per_replica_positive_batch_size
  )

  ckpt_dir = os.path.join(model_dir, "ckpt")
  if not tf.io.gfile.exists(ckpt_dir):
    logging.info("Creating checkpoint directory: %s", ckpt_dir)
    tf.io.gfile.makedirs(ckpt_dir)

  logging.info("ckpt_dir: %s", ckpt_dir)
  checkpoint_options = tf.train.CheckpointOptions(enable_async=async_checkpoint)
  checkpoint_manager = create_or_restore_checkpoint_manager(
      model, optimizer, strategy, ckpt_dir
  )

  metrics_path = logs_dir if logs_dir else model_dir
  logging.info("Writing metrics to: %s", metrics_path)
  metrics = HugeMetrics(strategy, metrics_path, num_negs_per_pos)

  starting_epoch = 0
  if optimizer.iterations.numpy() > 0:
    starting_epoch = optimizer.iterations.numpy() // train_steps
  logging.info(
      "starting_epoch: %s (Should be > 0 if restored from checkpoint).",
      starting_epoch,
  )

  @tf.function
  def train_step(iterator, nhost_steps):
    """Perform HUGE-TPU training steps."""

    def step_fn(inputs):
      src, dst, expected_edge_scores = inputs[:3]

      with tf.GradientTape() as tape:
        logits = model(inputs={"src": src, "dst": dst})
        positive_logits = logits[:per_replica_positive_batch_size]
        negative_logits = logits[per_replica_positive_batch_size:]
        tl, pl, nl = negative_log_graph_likelihood(
            positive_logits, negative_logits, expected_edge_scores
        )

      gradients = tape.gradient(tl, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      metrics.update(tl, pl, nl, positive_logits, negative_logits)

    # Each TPU core will run the step_fn, when this loop completes,
    # execution will be handed back to the coordinator.
    for _ in tf.range(nhost_steps):
      strategy.run(step_fn, args=(next(iterator),))

    metrics.summarize(optimizer.iterations)

  for epoch in range(starting_epoch, epochs):
    current_step = optimizer.iterations.numpy()
    logging.info(
        "Starting epoch: %d, step %d", current_step // train_steps, current_step
    )
    epoch_start = tf.timestamp()
    with tf.experimental.async_scope():
      for step in range(train_steps // nhost_steps):
        logging.info(
            "\tQueueing train_step: %d of %d",
            current_step + step,
            train_steps // nhost_steps,
        )
        train_step(ds_iter, nhost_steps_t)

    # metrics.summarize(optimizer.iterations)
    logging.info("Epoch: %d, %s", epoch, metrics)
    metrics.reset_states()

    epoch_duration = tf.timestamp() - epoch_start

    steps_per_second = (
        tf.constant(train_steps, dtype=epoch_duration.dtype) / epoch_duration
    )
    positive_examples_per_second = (
        tf.constant(positive_batch_size, dtype=epoch_duration.dtype)
        / epoch_duration
    )
    with metrics.writer.as_default():
      with tf.name_scope("Timing"):
        tf.summary.scalar(
            "StepsPerSecond", steps_per_second, optimizer.iterations
        )

        tf.summary.scalar(
            "PositiveExamplesPerSecond",
            positive_examples_per_second,
            optimizer.iterations,
        )
    latest_checkpoint = checkpoint_manager.save(
        check_interval=True, options=checkpoint_options
    )
    if latest_checkpoint:
      logging.info("Wrote checkpoint: %s", latest_checkpoint)
    else:
      logging.warning(
          "Failed to save checkpoint: %d", optimizer.iterations.numpy()
      )

  # Make sure we save the last checkpoint
  latest_checkpoint = checkpoint_manager.save(
      check_interval=False, options=checkpoint_options
  )
  if latest_checkpoint:
    logging.info("Wrote final checkpoint: %s", latest_checkpoint)
  else:
    logging.warning("Failed to save final checkpoint.")
