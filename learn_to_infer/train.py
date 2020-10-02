# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Code implementing train loops."""
import functools
from functools import partial
import sys
import timeit
import traceback

from . import util

from flax import jax_utils
from flax import optim
from flax.metrics.tensorboard import SummaryWriter
from flax.training import checkpoints
import jax
from jax import jit
from jax.lib import xla_bridge
import jax.numpy as jnp


def default_summarize(writer, step, params, key):
  pass


def can_train_parallel():
  return (xla_bridge.get_backend().platform == "tpu" and
          len(xla_bridge.devices()) > 1)


def train_loop(
    key,
    init_params,
    loss_fn,
    parallel=True,
    summarize_fn=default_summarize,
    lr=1e-4,
    num_steps=int(1e5),
    summarize_every=100,
    checkpoint_every=5000,
    clobber_checkpoint=False,
    logdir="/tmp/lda_inference"):

  if not parallel:
    train_fn = local_train_loop
  elif parallel and can_train_parallel():
    train_fn = parallel_train_loop
  else:
    print(
        "Platform is %s and num devices is %d, defaulting to local training." %
        (xla_bridge.get_backend().platform, len(xla_bridge.devices())))
    train_fn = local_train_loop

  train_fn(
      key, init_params, loss_fn,
      summarize_fn=summarize_fn,
      lr=lr,
      num_steps=num_steps,
      summarize_every=summarize_every,
      checkpoint_every=checkpoint_every,
      clobber_checkpoint=clobber_checkpoint,
      logdir=logdir)


def local_train_loop(
    key,
    init_params,
    loss_fn,
    summarize_fn=default_summarize,
    lr=1e-4,
    num_steps=int(1e5),
    summarize_every=100,
    checkpoint_every=5000,
    clobber_checkpoint=False,
    logdir="/tmp/lda_inference"):

  optimizer_def = optim.Adam()
  optimizer = optimizer_def.create(init_params)
  optimizer = util.maybe_load_checkpoint(logdir, optimizer,
                                         clobber_checkpoint=clobber_checkpoint)
  lr_fn = util.create_learning_rate_scheduler(
      base_learning_rate=lr)

  def train_step(optimizer, key):
    loss_val, loss_grad = jax.value_and_grad(
        loss_fn, argnums=0)(optimizer.target, key)
    new_optimizer = optimizer.apply_gradient(
        loss_grad, learning_rate=lr_fn(optimizer.state.step))
    return loss_val, new_optimizer

  train_step = jit(train_step)

  sw = SummaryWriter(logdir)

  start = timeit.default_timer()
  first_step = optimizer.state.step
  for t in range(optimizer.state.step, num_steps):
    if t % checkpoint_every == 0 and t != first_step:
      checkpoints.save_checkpoint(logdir,
                                  optimizer,
                                  optimizer.state.step, keep=3)
      print("Checkpoint saved for step %d" % optimizer.state.step)
    key, subkey = jax.random.split(key)
    try:
      loss_val, new_optimizer = train_step(optimizer, subkey)
    except FloatingPointError as e:
      print("Exception on step %d" % t)
      print(e)
      traceback.print_exc()
      checkpoints.save_checkpoint(logdir,
                                  optimizer,
                                  optimizer.state.step, keep=3)
      print("Checkpoint saved for step %d" % optimizer.state.step)
      print("key ", subkey)
      sys.stdout.flush()
      sys.exit(1)
    optimizer = new_optimizer
    if t % summarize_every == 0:
      key, subkey = jax.random.split(key)
      print("Step %d loss: %0.4f" % (t, loss_val))
      sw.scalar("loss", loss_val, step=t)
      summarize_fn(sw, t, optimizer.target, subkey)
      end = timeit.default_timer()
      if t == 0:
        steps_per_sec = 1. / (end - start)
      else:
        steps_per_sec = summarize_every / (end - start)
      print("Steps/sec: %0.2f" % steps_per_sec)
      sw.scalar("steps_per_sec", steps_per_sec, step=t)
      start = end
      sw.flush()
      sys.stdout.flush()


def parallel_train_loop(key,
                        init_params,
                        loss_fn,
                        summarize_fn=default_summarize,
                        lr=1e-4,
                        num_steps=int(1e5),
                        summarize_every=100,
                        checkpoint_every=5000,
                        clobber_checkpoint=False,
                        logdir="/tmp/lda_inference"):

  loss_fn = jax.jit(loss_fn)

  optimizer_def = optim.Adam()
  local_optimizer = optimizer_def.create(init_params)
  local_optimizer = util.maybe_load_checkpoint(
      logdir, local_optimizer, clobber_checkpoint=clobber_checkpoint)
  first_step = local_optimizer.state.step
  repl_optimizer = jax_utils.replicate(local_optimizer)

  lr_fn = util.create_learning_rate_scheduler(base_learning_rate=lr)

  @functools.partial(jax.pmap, axis_name="batch")
  def train_step(optimizer, key):
    key, subkey = jax.random.split(key)
    loss_grad = jax.grad(loss_fn, argnums=0)(optimizer.target, key)
    loss_grad = jax.lax.pmean(loss_grad, "batch")
    new_optimizer = optimizer.apply_gradient(
        loss_grad, learning_rate=lr_fn(optimizer.state.step))
    return new_optimizer, subkey

  sw = SummaryWriter(logdir)

  repl_key = jax.pmap(jax.random.PRNGKey)(jnp.arange(jax.local_device_count()))
  start = timeit.default_timer()
  for t in range(first_step, num_steps):
    if t % checkpoint_every == 0 and t != first_step:
      optimizer = jax_utils.unreplicate(repl_optimizer)
      checkpoints.save_checkpoint(logdir,
                                  optimizer,
                                  optimizer.state.step, keep=3)
      print("Checkpoint saved for step %d" % optimizer.state.step)

    repl_optimizer, repl_key = train_step(repl_optimizer, repl_key)

    if t % summarize_every == 0:
      key, subkey = jax.random.split(jax_utils.unreplicate(repl_key))
      optimizer = jax_utils.unreplicate(repl_optimizer)
      loss_val = loss_fn(optimizer.target, key)
      print("Step %d loss: %0.4f" % (t, loss_val))
      sw.scalar("loss", loss_val, step=t)
      summarize_fn(sw, t, optimizer.target, subkey)
      end = timeit.default_timer()
      if t == 0:
        steps_per_sec = 1. / (end - start)
      else:
        steps_per_sec = summarize_every / (end - start)
      print("Steps/sec: %0.2f" % steps_per_sec)
      sw.scalar("steps_per_sec", steps_per_sec, step=t)
      start = end
      sw.flush()
      sys.stdout.flush()

