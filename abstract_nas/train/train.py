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

"""Functions for training Models and Subgraphs.

Forked from //third_party/py/big_vision/train.py
"""

import dataclasses
from functools import partial  # pylint: disable=g-importing-member so standard
import multiprocessing
import time
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from big_vision import input_pipeline
from big_vision import optax as bv_optax
from big_vision import utils as bv_utils
import flax
import flax.jax_utils as flax_utils
import jax
from jax import numpy as jnp
import numpy as np
import optax
import tensorflow as tf  # pylint: disable=unused-import (for ThreadPool, weird)

from abstract_nas import utils
from abstract_nas.model import Model
from abstract_nas.model.subgraph import inherit_params
from abstract_nas.model.subgraph import replace_subgraph
from abstract_nas.train import perf_tools
from abstract_nas.train import preprocess
from abstract_nas.train import utils as train_utils
from abstract_nas.train.config import Config
import tensorflow.io.gfile as gfile  # pylint: disable=consider-using-from-import


@dataclasses.dataclass
class Metrics:
  loss: float
  acc: float
  num_params: int
  flops: int
  im_sec_core_train: float
  im_sec_core_infer: float


GFILE_TRIES = 100
GFILE_SLEEP_SEC = 5


def train_and_eval(
    config,
    eval_perf = True,
    callback = None
):
  """Training loop + eval.

  Args:
    config: The training config.
    eval_perf: Whether to collect performance metrics.
    callback: A callback which accepts the current epoch number and performance
      metrics, and returns True to continue training or False to early stop.

  Returns:
    A tuple of the final metrics, number of epochs trained, and the state dict.
  """
  if "train" in config.config_dict:
    train_config = config.config_dict.train
  else:
    train_config = config.config_dict
  rng = jax.random.PRNGKey(train_config.seed)

  is_host = jax.process_index() == 0

  if is_host:
    # The pool is used to perform operations such as checkpointing in async way.
    pool = multiprocessing.pool.ThreadPool(2)
  else:
    pool = None

  # set up output directory
  if config.output_dir is not None:
    if is_host:
      if gfile.exists(config.output_dir):
        logging.warn("Output directory %s already exists.", config.output_dir)
      else:
        gfile.makedirs(config.output_dir)
      utils.write_to_store(config, f"{config.output_dir}/config")
    else:
      ready = False
      for _ in range(GFILE_TRIES):
        ready = gfile.exists(config.output_dir)
        if ready: break
        time.sleep(GFILE_SLEEP_SEC)
      if not ready:
        raise ValueError(f"Output directory {config.output_dir} was not "
                         f"created within {GFILE_SLEEP_SEC * GFILE_TRIES} "
                         "secs.")

  # get data
  num_devices = jax.device_count()
  batch_size = train_config.device_batch_size * num_devices
  if batch_size % num_devices != 0:
    raise ValueError("JAX num_devices {num_devices} does not divide batch_size "
                     f"{batch_size}.")
  local_batch_size = batch_size // jax.process_count()
  local_batch_size_eval = local_batch_size * 8

  if is_host:
    logging.info(
        "Global batch size %d on %d hosts results in %d local batch size. "
        "With %d dev per host (%d dev total), that's a %d per-device batch "
        "size.",
        batch_size, jax.process_count(), local_batch_size,
        jax.local_device_count(), jax.device_count(),
        local_batch_size // jax.local_device_count())

  train_pp = preprocess.get_preprocess_fn(
      train_config.dataset_name, train_config.dataset.train_split,
      **train_config.dataset.get("preprocess_kwargs", {}))
  train_ds = input_pipeline.make_for_train(
      dataset=train_config.dataset_name,
      split=train_config.dataset.train_split,
      preprocess_fn=train_pp,
      batch_size=local_batch_size,
      shuffle_buffer_size=250_000,
      prefetch=2,
      cache_raw=False)

  train_iter = input_pipeline.start_input_pipeline(
      train_ds, n_prefetch=1)

  ntrain_img = input_pipeline.get_num_examples(train_config.dataset_name,
                                               train_config.dataset.train_split)
  steps_per_epoch = ntrain_img / batch_size
  total_steps = int(steps_per_epoch * train_config.epochs)

  eval_pp = preprocess.get_preprocess_fn(train_config.dataset_name,
                                         train_config.dataset.val_split)
  eval_ds, eval_steps = input_pipeline.make_for_inference(
      dataset=train_config.dataset_name,
      split=train_config.dataset.val_split,
      preprocess_fn=eval_pp,
      batch_size=local_batch_size_eval,
      cache_final=True,
      cache_raw=False,
      data_dir=None)
  eval_it = input_pipeline.start_input_pipeline(eval_ds, n_prefetch=1)

  # set up model
  graph = config.graph
  if isinstance(graph, tuple):
    graph, constants = graph[0], graph[1]
  else:
    constants = None
  if config.subgraph is not None:
    graph = replace_subgraph(graph, config.subgraph)
    if (config.inherit_weights and
        config.freeze_inherited and
        config.train_subg_outputs):
      # TODO(charlesjin) finish training with weight inheritance
      output_names = sum([node.output_names for node in config.subgraph], [])
      graph.output_names = output_names
      raise NotImplementedError
  model = Model(graph, constants)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @partial(jax.jit, backend="cpu")
  def init(rng):
    image_size = tuple(train_ds.element_spec["image"].shape[1:])
    dummy_input = jnp.zeros((1,) + image_size, jnp.float32)
    return flax.core.unfreeze(model.init(rng, dummy_input))

  rng, rng_init = jax.random.split(rng)
  state_cpu = init(rng_init)
  params_cpu = state_cpu["params"]
  if "batch_stats" in state_cpu:
    # Non-param variable collections. Currently we only support the additional
    # collection batch_stats, which is Flax's convention for batchnorm.
    coll_cpu = {"batch_stats": state_cpu["batch_stats"]}
  else:
    coll_cpu = {}

  # weight inheritance
  if config.inherit_weights:
    if config.init_dir is None:
      raise ValueError("Cannot inherit weights without parent directory.")

    parent_state = bv_utils.load_checkpoint(None, f"{config.init_dir}/state")
    parent_params = parent_state["params"]
    old_params, new_params = inherit_params(params_cpu, parent_params)

    if config.freeze_inherited:
      trainable_params = new_params
      frozen_params = old_params
    else:
      trainable_params = {**old_params, **new_params}
      frozen_params = {}
  else:
    trainable_params = params_cpu
    frozen_params = {}

  if is_host:
    if trainable_params:
      logging.info("trainable params:")
      for key in trainable_params.keys():
        logging.info("  %s", key)
    else:
      logging.warn("WARNING: no trainable params!")
    if frozen_params:
      logging.info("frozen params:")
      for key in frozen_params.keys():
        logging.info("  %s", key)

  # training step
  @partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1, 3,))
  def train_step(opt, params, other_params, coll, data, labels, rng):
    """Trains for a single step."""
    # Get device-specific loss rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index("batch"))
    def loss_fn(params):
      all_params = {**params, **other_params}
      logits, new_coll = Model(graph, constants).apply(
          flax.core.freeze({
              "params": all_params,
              **coll
          }),
          data,
          rngs={"dropout": rng_model_local},
          mutable=list(coll.keys()),
          deterministic=False,
          training=True)

      loss = jnp.mean(
          bv_utils.softmax_xent(
              logits=logits, labels=labels))
      return loss, (logits, loss, new_coll)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(params)
    _, loss, new_coll = aux[1]
    grads = jax.lax.pmean(grads, axis_name="batch")

    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)

    return opt, params, new_coll, jax.lax.psum(loss, axis_name="batch"), rng

  cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, axis_name="batch"),
                                axis_name="batch")

  # eval step
  @partial(jax.pmap, axis_name="batch")
  def eval_step(params, coll, data, labels, mask):
    mask *= labels.max(axis=1)
    logits = Model(graph, constants).apply(
        flax.core.freeze({
            "params": params,
            **coll
        }),
        data,
        deterministic=True,
        training=False)
    loss = jnp.mean(
        bv_utils.softmax_xent(
            logits=logits, labels=labels))

    top1_idx = jnp.argmax(logits, axis=1)
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    correct = top1_correct * mask
    return (jax.lax.psum(correct, axis_name="batch"),
            jax.lax.psum(loss, axis_name="batch"),
            jax.lax.psum(mask, axis_name="batch"))

  def eval_model(params, coll, eval_it):
    total_correct = 0
    total_loss = 0
    total = 0
    eval_time = 0
    eval_start = time.time()
    for _, batch in zip(range(eval_steps), eval_it):
      correct, loss, neval = eval_step(params, coll, batch["image"],
                                       batch["labels"], batch["_mask"])
      total_correct += jnp.sum(correct[0])
      total_loss += jnp.sum(loss[0])
      total += jnp.sum(neval[0])
    if total: total.block_until_ready()
    eval_time += time.time() - eval_start
    return total_correct, total_loss, total, eval_time

  if eval_perf and is_host:
    num_params = perf_tools.compute_num_params(params_cpu)
    image_size = tuple(train_ds.element_spec["image"].shape[1:])
    dummy_input = jnp.zeros((1,) + image_size, jnp.float32)
    apply_fn = lambda v, inp: model.apply(  # pylint: disable=g-long-lambda
        v, inp, deterministic=True, training=False)
    flops = perf_tools.compute_num_flops(
        apply_fn,
        True,  # optimize
        flax.core.freeze({
            "params": params_cpu,
            **coll_cpu
        }), dummy_input)
    print(f"num_params: {num_params} | flops: {flops}")
  else:
    num_params = 0
    flops = 0

  im_sec_core_eval_measurements = np.array([])
  im_sec_core_train_measurements = np.array([])
  last_step = 0
  checkpoint_extra = dict(
      im_sec_core_eval_measurements=im_sec_core_eval_measurements,
      im_sec_core_train_measurements=im_sec_core_train_measurements,
      step=last_step)

  if config.output_dir is not None:
    checkpoint_path = f"{config.output_dir}/checkpoint.npz"
  else:
    checkpoint_path = None

  if trainable_params:
    tx, _ = bv_optax.make(train_config.optim, params_cpu, sched_kw=dict(
        global_batch_size=batch_size,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch))
    opt_cpu = jax.jit(tx.init, backend="cpu")(trainable_params)

    # EMA
    ema_decay = train_config.get("ema_decay", 0)
    if ema_decay:
      end_warmup_step = train_config.get("ema_warmup_steps", 1560)
      ema_state_cpu = {"params": params_cpu, "coll": coll_cpu}
      ema_manager = train_utils.ExponentialMovingAverage(ema_state_cpu,
                                                         ema_decay,
                                                         end_warmup_step)

      @partial(jax.pmap, axis_name="batch")
      def update_ema(step, params, collection, ema):
        ema_state = {"params": params, "coll": collection}
        return ema.update_moving_average(ema_state, step)

    else:
      update_ema = ema_manager = ema_state_cpu = None

    # Load checkpoint if already exists
    if checkpoint_path and gfile.exists(checkpoint_path):
      checkpoint = {
          "opt": opt_cpu,
          "coll": coll_cpu,
          "params": params_cpu,
          "ema_state": ema_state_cpu,
          "extra": checkpoint_extra
      }
      checkpoint_tree = jax.tree.structure(checkpoint)
      loaded = bv_utils.load_checkpoint(checkpoint_tree, checkpoint_path)
      # bfloat16 type gets lost when data is saved to disk, so we recover it.
      checkpoint = jax.tree.map(bv_utils.recover_dtype, loaded)
      opt_cpu, coll_cpu, params_cpu, ema_state_cpu, checkpoint_extra = (
          checkpoint["opt"], checkpoint["coll"], checkpoint["params"],
          checkpoint["ema_state"], checkpoint["extra"])
      im_sec_core_eval_measurements = checkpoint_extra[
          "im_sec_core_eval_measurements"]
      im_sec_core_train_measurements = checkpoint_extra[
          "im_sec_core_train_measurements"]
      last_step = checkpoint_extra["step"]
      if ema_manager and ema_state_cpu:
        ema_manager = ema_manager.replace(state=ema_state_cpu)

      logging.info("Loaded checkpoint at step %d (%d total).", last_step,
                   total_steps)
  else:
    opt_cpu = None
    update_ema = ema_manager = None

  do_last_eval = True
  eval_is_compiled = False
  last_step = bv_optax.get_count(opt_cpu)
  if trainable_params and last_step < total_steps:
    trainable_params_repl = flax_utils.replicate(trainable_params)
    opt_repl = flax_utils.replicate(opt_cpu)
    coll_repl = flax_utils.replicate(coll_cpu)
    rng, rng_loop = jax.random.split(rng, 2)
    rngs_loop = flax_utils.replicate(rng_loop)
    frozen_repl = flax_utils.replicate(frozen_params)

    if ema_manager:
      ema_manager_repl = flax_utils.replicate(ema_manager)
    else:
      ema_manager_repl = None

    def ema_repl_to_state_cpu(ema_manager_repl):
      if ema_manager_repl is None:
        return None
      ema_trainable_params_repl = ema_manager_repl.state["params"]
      ema_trainable_params_cpu = jax.tree.map(lambda x: np.array(x[0]),
                                              ema_trainable_params_repl)
      ema_coll_repl = ema_manager_repl.state["coll"]
      ema_coll_cpu = jax.tree.map(lambda x: np.array(x[0]), ema_coll_repl)
      ema_state_cpu = {"params": ema_trainable_params_cpu,
                       "coll": ema_coll_cpu}
      return ema_state_cpu

    write_checkpoints = (
        is_host and checkpoint_path is not None and config.checkpoint_steps)

    step = last_step
    epoch = int(last_step / steps_per_epoch) + 1
    loss = 0
    train_time = 0
    checkpoint_writer = None

    if is_host:
      logging.info(
          "Training on dataset %s for %d total epochs (starting from %d).",
          train_config.dataset_name, train_config.epochs, epoch)

    for step, train_batch in zip(
        range(last_step + 1, total_steps + 1), train_iter):
      step_start = time.time()
      do_last_eval = True
      (opt_repl, trainable_params_repl, coll_repl, loss_repl,
       rngs_loop) = train_step(opt_repl, trainable_params_repl, frozen_repl,
                               coll_repl, train_batch["image"],
                               train_batch["labels"], rngs_loop)

      if update_ema is not None:
        step_repl = flax_utils.replicate(step)
        ema_manager_repl = update_ema(step_repl, trainable_params_repl,
                                      coll_repl, ema_manager_repl)

      loss += loss_repl[0]
      if step > steps_per_epoch * epoch or step == total_steps:
        line = (f"epoch {epoch:d}"
                f" | train loss {loss / steps_per_epoch:.1f}")
        if coll_cpu:
          coll_repl = cross_replica_mean(coll_repl)
        train_time += time.time() - step_start
        if epoch > 1:
          train_im = steps_per_epoch * batch_size
          im_sec_core_train = train_im / num_devices / train_time
          im_sec_core_train_measurements = np.append(
              im_sec_core_train_measurements, im_sec_core_train)

        if epoch % train_config.log_epochs == 0:
          if ema_manager_repl is not None:
            trainable_params_repl_eval = ema_manager_repl.state["params"]
            coll_repl_eval = ema_manager_repl.state["coll"]
          else:
            trainable_params_repl_eval = trainable_params_repl
            coll_repl_eval = coll_repl
          params_repl = {**trainable_params_repl_eval, **frozen_repl}
          correct, loss, n_eval, eval_time = eval_model(params_repl,
                                                        coll_repl_eval,
                                                        eval_it)
          if eval_is_compiled:
            eval_im = int(n_eval)
            im_sec_core_eval = eval_im / num_devices / eval_time
            im_sec_core_eval_measurements = np.append(
                im_sec_core_eval_measurements, im_sec_core_eval)
          eval_is_compiled = True
          line += (f" | val loss {loss:.2f}"
                   f" | val acc {correct / n_eval * 100:.3f}%"
                   f" ({int(correct)} / {int(n_eval)})")
          do_last_eval = False
          if step < total_steps and callback:
            metrics = Metrics(
                loss=loss,
                acc=correct / n_eval,
                num_params=num_params,
                flops=flops,
                im_sec_core_infer=(np.median(im_sec_core_eval_measurements) if
                                   len(im_sec_core_eval_measurements) else 0),
                im_sec_core_train=(np.median(im_sec_core_train_measurements) if
                                   len(im_sec_core_train_measurements) else 0))
            if not callback(epoch, metrics):
              line += " | EARLY STOPPED"
              if is_host:
                logging.info(line)
              break
        if is_host:
          logging.info(line)
          logging.info("Train measurements stddev: %.2f",
                       np.std(im_sec_core_train_measurements))
          logging.info("Eval measurements stddev: %.2f",
                       np.std(im_sec_core_eval_measurements))
        loss = 0
        epoch += 1
        train_time = 0
      train_time += time.time() - step_start
      if write_checkpoints and pool and step % config.checkpoint_steps == 0:
        assert pool is not None
        bv_utils.checkpointing_timeout(checkpoint_writer, 10)
        checkpoint_extra[
            "im_sec_core_eval_measurements"] = im_sec_core_eval_measurements
        checkpoint_extra[
            "im_sec_core_train_measurements"] = im_sec_core_train_measurements
        checkpoint_extra["step"] = step
        # We need to transfer the weights over now or else we risk keeping them
        # alive while they'll be updated in a future step, creating hard to
        # debug memory errors (see b/160593526). Also, takes device 0's params
        # only.
        opt_cpu = jax.tree.map(lambda x: np.array(x[0]), opt_repl)
        coll_cpu = jax.tree.map(lambda x: np.array(x[0]), coll_repl)
        trainable_params_cpu = jax.tree.map(lambda x: np.array(x[0]),
                                            trainable_params_repl)
        params_cpu = {**trainable_params_cpu, **frozen_params}
        ema_state_cpu = ema_repl_to_state_cpu(ema_manager_repl)

        # Checkpoint should be a nested dictionary or FLAX datataclasses from
        # `flax.struct`. Both can be present in a checkpoint.
        checkpoint = {
            "opt": opt_cpu,
            "coll": coll_cpu,
            "params": params_cpu,
            "ema_state": ema_state_cpu,
            "extra": checkpoint_extra
        }
        checkpoint_writer = pool.apply_async(bv_utils.save_checkpoint,
                                             (checkpoint, checkpoint_path))
    coll_cpu = jax.tree.map(lambda x: np.array(x[0]), coll_repl)
    opt_cpu = jax.tree.map(lambda x: np.array(x[0]), opt_repl)
    trainable_params_cpu = jax.tree.map(lambda x: np.array(x[0]),
                                        trainable_params_repl)
    params_cpu = {**trainable_params_cpu, **frozen_params}
    params_repl = {**trainable_params_repl, **frozen_repl}
    ema_state_cpu = ema_repl_to_state_cpu(ema_manager_repl)
    if ema_manager:
      coll_repl_eval = ema_manager_repl.state["coll"]
      params_repl_eval = {**ema_manager_repl.state["params"], **frozen_repl}
    else:
      coll_repl_eval = coll_repl
      params_repl_eval = params_repl
  else:
    epoch = 0
    coll_repl = flax_utils.replicate(coll_cpu)
    params_cpu = frozen_params
    params_repl = flax_utils.replicate(params_cpu)
    ema_state_cpu = None
    coll_repl_eval = coll_repl
    params_repl_eval = params_repl

  if do_last_eval:
    correct, loss, n_eval, eval_time = eval_model(params_repl_eval,
                                                  coll_repl_eval,
                                                  eval_it)
    if eval_is_compiled:
      eval_im = int(n_eval)
      im_sec_core_eval = eval_im / num_devices / eval_time
      im_sec_core_eval_measurements = np.append(im_sec_core_eval_measurements,
                                                im_sec_core_eval)
    eval_is_compiled = True

  if eval_perf and not len(im_sec_core_eval_measurements):  # pylint: disable=g-explicit-length-test (can't check len on numpy arrays)
    assert eval_is_compiled
    correct, loss, n_eval, eval_time = eval_model(params_repl_eval,
                                                  coll_repl_eval,
                                                  eval_it)
    eval_im = int(n_eval)
    im_sec_core_eval = eval_im / num_devices / eval_time
    im_sec_core_eval_measurements = np.append(im_sec_core_eval_measurements,
                                              im_sec_core_eval)

  checkpoint_extra[
      "im_sec_core_eval_measurements"] = im_sec_core_eval_measurements
  checkpoint_extra[
      "im_sec_core_train_measurements"] = im_sec_core_train_measurements
  checkpoint_extra["step"] = step
  checkpoint = {
      "opt": opt_cpu,
      "coll": coll_cpu,
      "params": params_cpu,
      "ema_state": ema_state_cpu,
      "extra": checkpoint_extra
  }
  if checkpoint_path is not None and is_host and pool:
    checkpoint_writer = pool.apply_async(bv_utils.save_checkpoint,
                                         (checkpoint, checkpoint_path))

  metrics = Metrics(
      loss=loss,
      acc=correct / n_eval,
      num_params=num_params,
      flops=flops,
      im_sec_core_infer=(np.median(im_sec_core_eval_measurements)
                         if len(im_sec_core_eval_measurements) else 0),
      im_sec_core_train=(np.median(im_sec_core_train_measurements)
                         if len(im_sec_core_train_measurements) else 0))

  if ema_state_cpu:
    state = ema_state_cpu
  else:
    state = {"coll": coll_cpu, "params": params_cpu}
  return metrics, epoch, state
