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
"""Trains a linear model with SGD."""
import functools
import os

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax.training import train_state
from incontext import linear_model
from incontext import optax
from incontext import sampler_lib
from incontext import utils
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.io import gfile

FLAGS = flags.FLAGS


def train_step(state, x, y, *, model, learning_rate_fn, dropout_rng=None):
  """Perform a single training step."""
  del learning_rate_fn
  dropout_rng = jax.random.fold_in(dropout_rng, state.step)

  # x, y = inputs[..., :-1], inputs[..., -1]

  def loss_fn(params):
    """loss function used for training."""
    ypred = model.apply({"params": params},
                        inputs=x,
                        train=True,
                        rngs={"dropout": dropout_rng})

    y_loss = ((ypred - y)**2).mean()

    reg_loss = sum(
        linear_model.l2_loss(w, alpha=model.config.alpha)
        for w in jax.tree.leaves(params))

    loss = y_loss + reg_loss

    return loss, (loss, y_loss, reg_loss, ypred)

  # lr = learning_rate_fn(state.step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  _, grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, "batch")
  new_state = state.apply_gradients(grads=grads)
  # loss = jax.lax.pmean(extras[0], "batch")
  # y_errors = jax.lax.pmean(extras[1], "batch")
  metrics = {}
  return new_state, metrics


def get_next_vector_loss(
    model,
    params,
    xs,
    ys,
):
  ypred = model.apply({"params": params}, inputs=xs, train=False)
  y_loss = ((ypred - ys)**2).mean()
  return float(y_loss)


def run_exp(
    xs,
    ys,
    seed = 0,
    x_dim = 3,
    learning_rate = 0.001,
    weight_decay = 0.001,
    gd = False,
):
  """Runs SGD on linear layer."""
  utils.set_seed(seed)
  config = linear_model.LinearConfig(alpha=weight_decay)
  model = linear_model.LinearModel(config)
  rng = random.PRNGKey(seed)
  rng, init_rng = random.split(rng)

  # call a jitted initialization function to get the initial parameter tree

  @jax.jit
  def initialize_variables(init_rng):
    init_batch = jnp.ones((1, x_dim), jnp.float32)
    init_variables = model.init(init_rng, inputs=init_batch, train=False)
    return init_variables

  init_variables = initialize_variables(init_rng)

  def scheduler(_):
    return learning_rate

  opt = optax.sgd(scheduler)

  state = train_state.TrainState.create(
      apply_fn=model.apply, params=init_variables["params"], tx=opt)

  # Replicate optimizer.
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(train_step, model=model, learning_rate_fn=scheduler),
      axis_name="batch",
      donate_argnums=(0,),
  )  # pytype: disable=wrong-arg-types

  dropout_rngs = random.split(rng, jax.local_device_count())

  logging.info("Running model training")

  avg_losses = []
  for i in range(xs.shape[0] - 1):
    if gd:
      x = xs[:i + 1, :]
      y = ys[:i + 1, :]
    else:
      x = xs[i:i + 1, :]
      y = ys[i:i + 1, :]

    # inputs = jnp.concatenate([x, y], axis=-1)
    state, _ = p_train_step(state, x, y, dropout_rng=dropout_rngs)
    y_loss = get_next_vector_loss(
        model,
        jax_utils.unreplicate(state).params,
        xs[i + 1:i + 2, :],
        ys[i + 1:i + 2, :],
    )
    avg_losses.append(y_loss)
  return avg_losses


def main(_):
  utils.set_seed(FLAGS.seed)
  x_distribution_str = "normal*1+0"
  w_distribution_str = "normal*1+5.0"

  sampler = sampler_lib.Sampler(
      FLAGS.num_exemplars,
      FLAGS.x_dim,
      FLAGS.x_dim + 1,
      x_distribution_fn=sampler_lib.str_to_distribution_fn(x_distribution_str),
      w_distribution_fn=sampler_lib.str_to_distribution_fn(w_distribution_str),
  )
  data = sampler.sample(n=FLAGS.batch_size)
  # [np.repeat(d, batch_size, axis=0) for d in data]
  seqs, coefficients, xs, ys = data
  seqs = jnp.array(seqs)
  coefficients = jnp.array(coefficients)
  avg_losses = []
  for _ in range(xs.shape[0]):
    avg_loss = run_exp(
        xs[0],
        ys[0],
        x_dim=FLAGS.x_dim,
        seed=FLAGS.seed,
        gd=FLAGS.gd,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
    )
    avg_losses.append(np.array(avg_loss))
  avg_losses = np.stack(avg_losses, axis=0)
  avg_losses = np.mean(avg_losses, axis=0)
  print(avg_losses)
  plt.figure()
  plt.plot(np.arange(len(avg_losses)), avg_losses)
  with gfile.Open(os.path.join(FLAGS.exp_folder, "losses.png"), "wb") as handle:
    plt.savefig(handle)
  plt.close()


if __name__ == "__main__":
  flags.DEFINE_integer("seed", default=0, help="seed")
  flags.DEFINE_integer("x_dim", default=10, help="x_dim")
  flags.DEFINE_float("learning_rate", default=0.01, help="learnning_rate")
  flags.DEFINE_float("weight_decay", default=0.0, help="weight_decay")
  flags.DEFINE_string("exp_folder", default="exp", help="exp_folder")
  flags.DEFINE_bool("gd", default=False, help="gd or sgd")
  flags.DEFINE_integer("num_exemplars", default=32, help="x_dim")
  flags.DEFINE_integer("batch_size", default=32, help="batch_size")
  flags.DEFINE_bool(
      "debug", default=False, help="debug predictions and posterior dist.")

  app.run(main)
