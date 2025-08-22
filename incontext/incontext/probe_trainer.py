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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains probe model."""
import functools
from typing import Any, Callable, Tuple

from absl import flags
from absl import logging
from flax import jax_utils
from flax.training import common_utils
from flax.training import train_state
from incontext import predictor_flax
from incontext import probe_flax
from incontext import sampler_lib
from incontext import transformer_lib_flax
from incontext import utils
import jax
from jax import random
import jax.numpy as jnp
import optax

flags.DEFINE_float(
    "probe_learning_rate", default=0.001, help="probe_learning_rate")
flags.DEFINE_integer("probe_epochs", default=20, help="n_epochs")
flags.DEFINE_integer("probe_iters", default=100, help="n_epochs")
flags.DEFINE_string(
    "probe_lr_scheduler_type",
    default="cosine",
    help="Use learning rate scheduler")


def train(
    rng,
    model,
    state,
    probe,
    probe_state,
    p_train_step,
    x_dim = 3,
    num_exemplars = 9,
    n_epochs = 100,
    n_iter_per_epoch = 100,
    batch_size = 16,
    hidden_size = 512,
    x_distribution_str = "normal*1+0",
    w_distribution_str = "normal*1+0",
):
  """Trains the transformer model for in-context learning.

  Args:
    rng:
    model:
    state:
    probe:
    probe_state:
    p_train_step:
    x_dim:
    num_exemplars:
    n_epochs:
    n_iter_per_epoch:
    batch_size:
    hidden_size:
    x_distribution_str:
    w_distribution_str:

  Returns:
    state of the model.

  """
  del probe
  rng, new_rng = jax.random.split(rng)
  dropout_rngs = random.split(new_rng, jax.local_device_count())

  sampler = sampler_lib.Sampler(
      num_exemplars,
      x_dim,
      hidden_size,
      x_distribution_fn=sampler_lib.str_to_distribution_fn(x_distribution_str),
      w_distribution_fn=sampler_lib.str_to_distribution_fn(w_distribution_str),
  )

  for epoch in range(n_epochs):
    metrics_all = []

    for _ in range(n_iter_per_epoch):
      seqs, coefficients, *_ = sampler.sample(n=batch_size)
      seqs = jnp.array(seqs)
      coefficients = jnp.array(coefficients)
      _, (_, _, _, seq_hiddens) = model.apply({"params": state.params},
                                              inputs=seqs,
                                              train=False)
      seq_hiddens = jnp.array(seq_hiddens).transpose(1, 0, 2, 3)
      seq_hiddens = common_utils.shard(seq_hiddens)
      coefficients = common_utils.shard(coefficients)
      probe_state, metrics = p_train_step(
          probe_state, seq_hiddens, coefficients, dropout_rng=dropout_rngs)
      metrics_all.append(metrics)

    logging.info("Epoch %d is finished.", epoch)
    metrics_all = common_utils.get_metrics(metrics_all)
    logging.info(metrics_all["lr"][-1])
    loss = jnp.mean(metrics_all["loss"], axis=(0, 1)) / batch_size
    logging.info(loss)
  return probe_state


def train_step_probe(state,
                     seq_hiddens,
                     coefficients,
                     model,
                     learning_rate_fn,
                     dropout_rng=None):
  """Train step for probe network."""
  dropout_rng = jax.random.fold_in(dropout_rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    output = model.apply(
        {"params": params},
        seq_hiddens=seq_hiddens,
        coefficients=coefficients,
        train=True,
        rngs={"dropout": dropout_rng},
    )

    return output.mean(), output

  lr = learning_rate_fn(state.step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, extras), grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, "batch")
  new_state = state.apply_gradients(grads=grads)
  loss = jax.lax.pmean(extras, "batch")
  metrics = {"loss": loss, "lr": lr}
  return new_state, metrics


def get_probe_model(rng, args):
  """Initialize the probe model and returns the model, state and step fn."""
  config = probe_flax.ProbeConfig(
      num_layers=args.n_layers, hidden_size=args.hidden_size, x_dim=args.x_dim)

  model = probe_flax.ProbeModel(config)

  rng, init_rng = random.split(rng)

  @jax.jit
  def initialize_variables_probe(init_rng):
    init_batch = jnp.ones(
        (1, config.num_layers + 1, config.max_len, config.hidden_size),
        dtype=jnp.float32,
    )
    coefficients = jnp.ones((1, config.x_dim), dtype=jnp.float32)
    init_variables = model.init(
        init_rng,
        seq_hiddens=init_batch,
        coefficients=coefficients,
        train=False)
    return init_variables

  init_variables = initialize_variables_probe(init_rng)

  if args.probe_lr_scheduler_type == "cosine":
    scheduler = transformer_lib_flax.create_learning_rate_scheduler(
        base_learning_rate=args.probe_learning_rate,
        num_warmup_steps=(args.probe_epochs // 5) * args.probe_iters,
        num_training_steps=args.probe_epochs * args.probe_iters,
    )
  elif args.lr_scheduler_type == "warmup":
    scheduler = transformer_lib_flax.create_learning_rate_scheduler_v2(
        factors="constant * linear_warmup",
        base_learning_rate=args.learning_rate,
        warmup_steps=(args.n_epochs // 5) * args.n_iter_per_epoch,
    )
  else:

    def scheduler(_):
      return args.probe_learning_rate

  opt = optax.adamw(
      scheduler,
      b1=args.adam_b1,
      b2=args.adam_b2,
      eps=args.adam_eps,
      weight_decay=args.weight_decay,
  )

  state = train_state.TrainState.create(
      apply_fn=model.apply, params=init_variables["params"], tx=opt)

  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(
          train_step_probe, model=model, learning_rate_fn=scheduler),
      axis_name="batch",
  )

  return model, state, p_train_step
