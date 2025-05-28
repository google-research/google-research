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

"""Layout Base Trainer."""

import abc
import functools
import os
from typing import Dict

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import linen as nn
import flax.jax_utils as flax_utils
from flax.training import common_utils
import jax
import jax.numpy as jnp
from layout_blt import input_pipeline
from layout_blt.utils import metrics
from layout_blt.utils import task_manager
import ml_collections
import numpy as np
import tensorflow as tf


class LayoutBaseTrainer(abc.ABC):
  """Base Trainer for layout generation."""

  def __init__(self, config, workdir):
    self.config = config
    self.workdir = workdir
    self.rng = jax.random.PRNGKey(config.seed)
    self.dtype, self.data_dtype = self.get_dtype()
    self.layout_dim = self.config.layout_dim
    self.total_dim = self.layout_dim * 2 + 1

  def get_dtype(self):
    if self.config.dtype == "bfloat16":
      return jnp.bfloat16, tf.bfloat16
    else:
      return jnp.float32, tf.float32

  def merge_model_state(self, state):
    if jax.tree.leaves(state.model_state):
      cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, "x"), "x")
      return state.replace(model_state=cross_replica_mean(state.model_state))
    else:
      return state

  @abc.abstractmethod
  def create_train_state(
      self,
      rng,
      inputs,
  ):
    pass

  def create_optimizer(self):
    """Creates optimizers separated for weights using decay.

    Returns:
      An instance of `Optimizer`.
    """
    opt_def = flax.optim.Adam(
        learning_rate=self.config.optimizer.lr,
        beta1=self.config.optimizer.beta1,
        beta2=self.config.optimizer.beta2,
        weight_decay=self.config.optimizer.weight_decay)
    return opt_def

  def make_mask(self, vocab_size, pos_info, seq_len, layout_dim):
    """Creates masking for logits at each training step and token offset.

    Our vocabulary is the combination of special symbols, asset classes,
    positions and sizes. At each step, only a part of vocabulary is possible.
    For example, the first step is asset type and only the asset candidates
    could be generated.

    Args:
      vocab_size: vocabulary size.
      pos_info: start indexs and number of candidates for each vocabulary
        segment. For example, in the following sample, [[2, 3], [6, 2]],
        denotes that for the first segment, its start index in the vocab is 2
        and there are 3 elements in the first segment.
      seq_len: the total length of input sequence.
      layout_dim: the layout dimension.
    Returns:
      asset_logit_masks: [1, seq_len, vocab_size]: logits mask for each step.
      asset_offset:  [1, seq_len]: offset to map the output token ids back to
        its original ids.
    """
    total_dim = layout_dim * 2 + 1
    logit_masks = []
    offset = jnp.array([pi[0] for pi in pos_info])
    offset = jnp.expand_dims(offset, 0)

    asset_offset = jnp.tile(offset, (1, seq_len // total_dim))
    # In our current model, the order of asset reprentation is
    # [asset, width, height, x, y]. We create masks for each of them.
    for idx, pi in enumerate(pos_info):
      # Creates a logit mask for the current segment. The logit shape from model
      # is [batch size, seq_len, vocab_size]. At a given step, the logit shape
      # is [batch size, 1, vocab_size], since for a given position, all logits
      # in the batch has the same masking, we just create a [1, 1, vocab_size]
      # mask which can broadcast to the whole batch.
      logit_mask = jnp.ones((1, 1, vocab_size))
      # pi[1] denotes the number of elements in the current segment.
      # The shape of pos_mask is [1, 1, #elements in this current segument].
      # For example, we have four possible asset classes, the pos_mask should be
      # [1, 1, 4].
      pos_mask = jnp.zeros((1, 1, pi[1]))
      # pi[0] means the start index of the current segment in the vocabulary.
      # Here, we update index [pi[0]: pi[0] + pi[1]] to zero.
      logit_mask = jax.lax.dynamic_update_slice(logit_mask, pos_mask,
                                                (0, 0, pi[0]))
      # At asset positions, we could also produce eos symbol.
      if idx == 0:
        logit_mask = logit_mask.at[:, :, 2].set(0)
      logit_masks.append(logit_mask)
    # We have creates masks for all segments and concatenate them into the mask
    # for a asset. [1, 5, vocab_size]
    logit_masks = jnp.concatenate(logit_masks, axis=1)
    # We extend the above mask to all positions in the sequences.
    asset_logit_masks = jnp.tile(logit_masks, (1, seq_len // total_dim, 1))
    # Concatenates all others positions.
    if seq_len % total_dim > 0:
      asset_logit_masks = jnp.concatenate(
          (asset_logit_masks, logit_masks[:, :(seq_len % total_dim), :]),
          axis=1)
      asset_offset = jnp.concatenate(
          (asset_offset, offset[:, :(seq_len % total_dim)]), axis=1)

    return asset_logit_masks, asset_offset

  def create_learning_rate_scheduler(self, learning_rate=1., warmup_steps=4000):
    """Creates learning rate scheduler for transformer.

    First increases the learning rate linearly for the first warmup_steps
    training steps, and decreases it thereafter proportionally to the inverse
    square root of the step number.
    Args:
      learning_rate: float, the starting constant for the lr schedule.
      warmup_steps: int, how many steps to warm up (> 0).
    Returns:
      A function to calculate the learing rate given current step.
    """
    def step_fn(step):
      cur_lr = learning_rate * jnp.minimum(1.0, step / warmup_steps) / jnp.sqrt(
          jnp.maximum(step, warmup_steps))
      return jnp.asarray(cur_lr, dtype=jnp.float32)

    return step_fn

  def compute_weighted_cross_entropy(self,
                                     logits,
                                     targets,
                                     mask=None,
                                     label_smoothing=0.0,
                                     logits_mask=None):
    """Computes weighted cross entropy between logits and targets.

    Args:
     logits: [batch, length, vocab_size] float array.
     targets: [batch, length] int array.
     mask: None or array of shape [batch, length].
     label_smoothing: label smoothing constant, used to determine the on and off
       values.
     logits_mask: [batch, length, vocab_size] float array: logits masking to
       ignore impossible candidates at each step.

    Returns:
      loss: float scalar.
    """
    if logits_mask is not None:
      logits = jnp.where(logits_mask > 0, -1e7, logits)
    if logits.ndim != targets.ndim + 1:
      raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                       (str(logits.shape), str(targets.shape)))
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = label_smoothing / (vocab_size - 1)
    soft_targets = common_utils.onehot(
        targets, vocab_size, on_value=confidence, off_value=low_confidence)

    loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
    # Calculates the best (lowest) possible value of cross entropy, and
    # subtract from the cross entropy loss.
    normalizing_constant = -(
        confidence * jnp.log(confidence) +
        (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
    loss = loss - normalizing_constant

    normalizing_factor = np.prod(targets.shape)
    if mask is not None:
      loss = loss * mask
      normalizing_factor = mask.sum()

    return loss.sum(), normalizing_factor

  def evaluate(self,
               p_eval_step,
               state,
               rng,
               eval_ds,
               batch_size=0,
               use_vertical=False,
               num_eval_steps=None,
               dataset="RICO"):
    """Evaluate the target an return a dictionary with the metrics."""
    logging.info("Gathering evaluation metrics.")
    eval_metrics = None
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    for _, eval_batch in zip(range(num_eval_steps), eval_iter):
      eval_batch = jax.tree.map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
      eval_batch, eval_label = self.preprocess_batch(eval_batch, batch_size,
                                                     dataset, use_vertical)
      if eval_batch is None:
        continue
      # eval_batch = common_utils.shard(eval_batch)
      metrics_update = p_eval_step(rng, state, eval_batch, eval_label)
      metrics_update = flax.jax_utils.unreplicate(metrics_update)
      eval_metrics = (
          metrics_update
          if eval_metrics is None else eval_metrics.merge(metrics_update))
    return eval_metrics

  def train(self):
    """Training loop."""

    tf.io.gfile.makedirs(self.workdir)
    checkpoint_dir = os.path.join(self.workdir, "checkpoints")
    n_devices = jax.local_device_count()
    task_manager_csv = task_manager.TaskManagerWithCsvResults(checkpoint_dir)
    rng, data_rng = jax.random.split(self.rng)
    # Make sure each host uses a different RNG for the training data.
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    batch_size = self.config.batch_size
    dataset = self.config.dataset
    use_vertical = self.config.use_vertical_info

    train_ds, eval_ds, _, vocab_size, pos_info = input_pipeline.get_all_dataset(
        batch_size,
        self.config.dataset_path,
        n_devices,
        add_bos=self.config.autoregressive,
        max_length=self.config.max_length,
        dataset_name=dataset)
    train_ds = train_ds.repeat()
    self.config.vocab_size = vocab_size
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    num_train_steps = self.config.num_train_steps
    logging.info("total_steps=%d", num_train_steps)

    rng, model_rng = jax.random.split(rng)
    init_batch = jnp.ones(
        (batch_size, self.config.max_length))
    init_label = jnp.ones((batch_size, 1))
    init_batch = dict(inputs=init_batch, labels=init_label)
    model_dict, state = self.create_train_state(model_rng, init_batch)
    learning_rate_fn = self.create_learning_rate_scheduler(
        learning_rate=self.config.optimizer.lr,
        warmup_steps=self.config.optimizer.warmup_steps)
    state = task_manager.restore_checkpoint(state, checkpoint_dir)
    # Creates logits mask.
    logits_mask, _ = self.make_mask(vocab_size, pos_info,
                                    self.config.max_length,
                                    self.config.layout_dim)
    initial_step = int(state.step) + 1
    # Warm-start from a checkpoint
    if initial_step == 1 and self.config.get(
        "checkpoint_path") and self.config.checkpoint_path:
      state = task_manager.restore_from_path(
          state, self.config.checkpoint_path)

    state = flax_utils.replicate(state)

    writer = metric_writers.create_default_writer(
        self.workdir, just_logging=jax.process_index() > 0)
    if initial_step == 1:

      writer.write_hparams(dict(self.config))

    logging.info("Starting training loop at step %d.", initial_step)
    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=self.config.num_train_steps, writer=writer)
    if jax.process_index() == 0:
      hooks += [
          report_progress,
          periodic_actions.Profile(num_profile_steps=5, logdir=self.workdir)
      ]

    p_train_step = jax.pmap(
        functools.partial(
            self.train_step,
            model_dict=model_dict,
            learning_rate_fn=learning_rate_fn,
            logits_mask=logits_mask
        ),
        axis_name="batch")

    p_eval_step = jax.pmap(
        functools.partial(
            self.eval_step,
            model_dict=model_dict,
            logits_mask=logits_mask
        ),
        axis_name="batch")

    train_metrics = None
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, train_rng, sample_rng = jax.random.split(rng, 3)  # pylint: disable=unused-variable

    with metric_writers.ensure_flushes(writer):
      for step in range(initial_step, num_train_steps + 1):
        # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
        # devices.
        is_last_step = step == self.config.num_train_steps
        with jax.profiler.StepTraceContext("train", step_num=step):
          batch = jax.tree.map(np.asarray, next(train_iter))
          batch, label = self.preprocess_batch(batch, batch_size, dataset,
                                               use_vertical)
          if batch is None:
            continue

          step_rng = jax.random.fold_in(train_rng, step)
          step_rngs = jax.random.split(step_rng, jax.local_device_count())
          state, metrics_update = p_train_step(step_rngs, state, batch, label)
          metric_update = flax.jax_utils.unreplicate(metrics_update)
          train_metrics = (
              metric_update
              if train_metrics is None else train_metrics.merge(metric_update))

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
        for h in hooks:
          h(step)

        if step % self.config.log_every_steps == 0:
          logging.info("Finish training step %d.", step)
          writer.write_scalars(step, train_metrics.compute())
          train_metrics = None
        if step % self.config.eval_every_steps == 0 or is_last_step:
          logging.info("eval step")
          state = self.merge_model_state(state)
          sample_rngs = jax.random.split(sample_rng, jax.local_device_count())
          eval_metrics = self.evaluate(p_eval_step, state, sample_rngs, eval_ds,
                                       batch_size,
                                       use_vertical,
                                       self.config.eval_num_steps,
                                       dataset)
          if eval_metrics is not None:
            writer.write_scalars(step, eval_metrics.compute())

        if step % self.config.checkpoint_every_steps == 0 or is_last_step:
          with report_progress.timed("checkpoint"):
            state = self.merge_model_state(state)
            task_manager.save_checkpoint(state, checkpoint_dir)
      logging.info("Finishing training at step %d", num_train_steps)
    if jax.process_index() == 0:
      task_manager_csv.mark_training_done()

  def evaluate_metrics(self,
                       generated_samples,
                       real_samples,
                       eos_id=-2,
                       conditional="a+s"):
    """Computing metrics."""
    def convert_format(layouts, eos_id):
      new_layouts = []
      for sample in layouts:
        sample = np.array(sample)
        if np.nonzero(sample == eos_id)[0].shape[0] > 0:
          real_len = np.nonzero(sample == eos_id)[0][0]
          sample = sample[:real_len]
          new_layouts.append(sample.reshape(-1, 5))
      return new_layouts
    generated_samples = convert_format(generated_samples, eos_id)

    iou = []
    overlap = []
    alignment = []
    for sample in generated_samples:
      iou.append(metrics.get_layout_iou(sample))
      overlap.append(metrics.get_overlap_index(sample))
      align_loss = metrics.get_alignment_loss(sample)
      if align_loss > 0:
        alignment.append(align_loss)
    def avg(a):
      return sum(a)/len(a)
    rst = {
        "iou": avg(iou),
        "overlap": avg(overlap),
        "alignment": avg(alignment)
    }

    if conditional != "unconditional":
      real_samples = convert_format(real_samples, eos_id)
      similarity = metrics.conditional_distance(generated_samples, real_samples,
                                                conditional)
      rst["similarity"] = similarity
    else:
      diveristy = metrics.diveristy(generated_samples)
      rst["diversity"] = diveristy

    return rst

  @abc.abstractmethod
  def train_step(self, rng, state, batch, model_dict, logits_mask):
    pass

  @abc.abstractmethod
  def eval_step(self, rng, state, batch, model_dict, logits_mask):
    pass

  @abc.abstractmethod
  def preprocess_batch(self, batch, batch_size, dataset, use_vertical):
    pass
