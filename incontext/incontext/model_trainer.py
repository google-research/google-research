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
"""Trains transformer model for in-context learning."""

import functools
import os
import pickle
from typing import Any, Callable, Mapping, Tuple

from absl import flags
from absl import logging
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from incontext import plotting
from incontext import predictor_flax
from incontext import sampler_lib
from incontext import transformer_lib_flax
from incontext import utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow.io import gfile

flags.DEFINE_integer("n_epochs", default=5001, help="n_epochs")
flags.DEFINE_integer("n_iter_per_epoch", default=100, help="n_iter_per_epoch")
flags.DEFINE_float("learning_rate", default=1e-4, help="learnning_rate")
flags.DEFINE_float("weight_decay", default=0, help="weight_decay")
flags.DEFINE_string(
    "lr_scheduler_type", default="cosine", help="Use learning rate scheduler")
flags.DEFINE_float("adam_b1", default=0.9, help="Adam b1")
flags.DEFINE_float("adam_b2", default=0.98, help="Adam b2")
flags.DEFINE_float("adam_eps", default=1e-9, help="Adam eps")
flags.DEFINE_string(
    "x_distribution_str",
    default="normal*1.0+0.0",
    help="Training distribution for xs")

flags.DEFINE_string(
    "w_distribution_str",
    default="normal*1.0+0.0",
    help="Training distribution for ws")

flags.DEFINE_float("noise_std", default=0.0, help="Noise std")


def test_empirical_statistics(
    model,
    params,
    x_dim = 3,
    num_exemplars = 9,
    hidden_size = 512,
    n_eval = 1,
    batch_size = 32,
    path_prefix = "plots/distribution/0/",
    x_distribution_str = "normal*1+0",
    w_distribution_str = "normal*1+0",
    plot_w_path = True,
    plot_average_stats = True,
    eval_noise_std = 0.0,
    **test_args,
):
  """Plots models predictions and empirical weights."""
  # W are sampling one less than usual, otherwise the position encodings
  # used for the last example will be untrained parameter.
  sampler = sampler_lib.Sampler(
      num_exemplars - 1,
      x_dim,
      hidden_size,
      w_distribution_fn=sampler_lib.str_to_distribution_fn(w_distribution_str),
      x_distribution_fn=sampler_lib.str_to_distribution_fn(x_distribution_str),
      noise_std=eval_noise_std,
  )

  for itr in range(n_eval):
    data = sampler.sample(n=1)
    # We keep the W same so that we can get the average stats for a single W
    seqs, coefficients, xs, ys = [
        np.repeat(d, batch_size, axis=0) for d in data
    ]
    seqs = jnp.array(seqs)
    coefficients = jnp.array(coefficients)
    path = os.path.join(path_prefix, f"{itr}.jpeg")
    plotting.plot_empirical_distribution(
        model,
        params,
        seqs,
        sampler,
        coefficients,
        xs=xs,
        ys=ys,
        path=path,
        plot=True,
        **test_args,
    )
    if plot_w_path and itr == n_eval - 1:
      path = os.path.join(path_prefix, f"{itr}_implicit_name.jpeg")
      plotting.plot_implicit_w(
          model, params, seqs, xs, ys, sampler, coefficients, path=path)
      path = os.path.join(path_prefix, f"{itr}_basis/")
      gfile.makedirs(path)
      plotting.plot_basis_image(
          model, params, seqs, sampler, coefficients, path=path)

  if plot_average_stats:
    data = sampler.sample(n=batch_size)
    seqs, coefficients, xs, ys = data
    # [np.repeat(d, batch_size, axis=0) for d in data]
    seqs = jnp.array(seqs)
    coefficients = jnp.array(coefficients)
    path = os.path.join(path_prefix, "average_stats.jpeg")
    x, y, scores, _ = plotting.plot_empirical_distribution(
        model,
        params,
        seqs,
        sampler,
        coefficients,
        xs=xs,
        ys=ys,
        path=path,
        plot=True,
        plot_dots=False,
        plot_planes=False,
        plot_fake_least_square_errors=False,
        plot_errors=True,
        plot_least_square_errors=True,
    )

    with gfile.GFile(path_prefix + "average_stats.pkl", "wb") as f:
      pickle.dump(
          {
              "x": x,
              "y": y,
              "scores": scores,
              "xs": np.array(xs),
              "ys": np.array(ys)
          }, f)

    path = path_prefix + "average_stats_v2.jpeg"
    x, y, scores, _ = plotting.plot_empirical_distribution(
        model,
        params,
        seqs,
        sampler,
        coefficients,
        xs=xs,
        ys=ys,
        path=path,
        plot=True,
        plot_dots=False,
        plot_knn_errors=True,
        plot_planes=False,
        plot_fake_least_square_errors=False,
        plot_errors=True,
        plot_least_square_errors=True,
        save_predictions=True,
    )

    # attention_vis
    seq0 = seqs[0:1, Ellipsis]
    _, (_, _, _, _, attn_weights) = model.apply({"params": params},
                                                inputs=seq0,
                                                train=False,
                                                return_attention=True)

    attn_weights = jax.tree_util.tree_map(np.array, attn_weights)

    with gfile.GFile(path_prefix + "attention_stats.pkl", "wb") as f:
      pickle.dump({"scores": attn_weights}, f)


def train_step(state, seq, model, learning_rate_fn, dropout_rng=None):
  """Perform a single training step."""

  dropout_rng = jax.random.fold_in(dropout_rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    output = model.apply({"params": params},
                         inputs=seq,
                         train=True,
                         rngs={"dropout": dropout_rng})

    return output[0].mean(), output

  lr = learning_rate_fn(state.step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, extras), grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, "batch")
  new_state = state.apply_gradients(grads=grads)
  loss = jax.lax.pmean(extras[0], "batch")
  y_errors = jax.lax.psum(extras[1][0], "batch").sum(axis=0)
  metrics = {"loss": loss, "lr": lr, "y_errors": y_errors}
  return new_state, metrics


def save_checkpoint(state, exp_folder):
  """Save model checkpoints."""

  def get_array(x):
    try:
      x = np.array(x)
    except:
      x = None
    return x

  # TODO(ekina): jax.device_get()?
  state = jax.tree_util.tree_map(get_array, jax_utils.unreplicate(state))

  ckpt_dir = os.path.join(exp_folder, "ckpt/")

  gfile.makedirs(ckpt_dir)

  try:
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir, target=state, step=state.step, overwrite=True)
  except:
    logging.warn("Issue in checkpointing")


def get_model(rng, args):
  """Initialize model and optimizer states."""
  rng, init_rng = random.split(rng)

  config = transformer_lib_flax.TransformerConfig(
      num_heads=args.n_heads,
      num_layers=args.n_layers,
      hidden_size=args.hidden_size,
      loss_on_x_steps=args.loss_on_x_steps,
      norm_first=args.norm_first,
      disable_layer_norms=args.disable_layer_norms,
      final_layer_norm=args.final_layer_norm,
      kernel_init=transformer_lib_flax.nn_init_parser(args.kernel_init),
      bias_init=transformer_lib_flax.nn_init_parser(args.bias_init),
      linear_w_init=transformer_lib_flax.nn_init_parser(args.linear_w_init),
      linear_bias_init=transformer_lib_flax.nn_init_parser(
          args.linear_bias_init),
      posemb_init=transformer_lib_flax.nn_init_parser(args.posemb_init),
      max_len=(args.num_exemplars + 1) * 2,
      inner_dim=None,
      activation_fn=transformer_lib_flax.nn_activation_parser(
          args.activation_fn),
  )

  model = predictor_flax.CausalLM(config)

  @jax.jit
  def initialize_variables(init_rng):
    init_batch = jnp.ones((1, config.max_len, args.x_dim + 1), jnp.float32)
    init_variables = model.init(init_rng, inputs=init_batch, train=False)
    return init_variables

  init_variables = initialize_variables(init_rng)

  if args.lr_scheduler_type == "cosine":
    scheduler = transformer_lib_flax.create_learning_rate_scheduler(
        base_learning_rate=args.learning_rate,
        num_warmup_steps=(args.n_epochs // 5) * args.n_iter_per_epoch,
        num_training_steps=args.n_epochs * args.n_iter_per_epoch,
    )
  elif args.lr_scheduler_type == "warmup":
    scheduler = transformer_lib_flax.create_learning_rate_scheduler_v2(
        factors="constant * linear_warmup",
        base_learning_rate=args.learning_rate,
        warmup_steps=(args.n_epochs // 5) * args.n_iter_per_epoch,
    )
  else:

    def scheduler(_):
      return args.learning_rate

  opt = optax.adamw(
      scheduler,
      b1=args.adam_b1,
      b2=args.adam_b2,
      eps=args.adam_eps,
      weight_decay=args.weight_decay,
  )

  # opt = optax.adamw(
  #     scheduler, b1=0.9, b2=0.999, eps=1e-8, weight_decay=weight_decay)

  state = train_state.TrainState.create(
      apply_fn=model.apply, params=init_variables["params"], tx=opt)

  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(train_step, model=model, learning_rate_fn=scheduler),
      axis_name="batch",
  )

  return model, state, p_train_step


def eval_model(
    model,
    params,
    *,
    batch_size,
    num_exemplars,
    hidden_size,
    exp_folder,
    x_dim,
    epoch = 1,
    plot_w_path = False,
    plot_average_stats = False,
    test_distributions = ("normal*1+0", "normal*2.5+0",
                                           "normal*1+2.5"),
    **test_kwargs,
):
  """Eval models on different distributions."""

  for distribution_str in test_distributions:
    distribution_str_to_save = distribution_str.replace("*", "x")
    folder = f"{exp_folder}/plots/distribution/w_{distribution_str_to_save}/{epoch+1}/"
    gfile.makedirs(folder)
    test_empirical_statistics(
        model,
        params,
        x_dim=x_dim,
        num_exemplars=num_exemplars,
        hidden_size=hidden_size,
        batch_size=batch_size,
        path_prefix=folder,
        w_distribution_str=distribution_str,
        plot_w_path=plot_w_path,
        plot_average_stats=plot_average_stats,
        **test_kwargs,
    )

  for distribution_str in test_distributions:
    distribution_str_to_save = distribution_str.replace("*", "x")
    folder = f"{exp_folder}/plots/distribution/x_{distribution_str_to_save}/{epoch+1}/"
    gfile.makedirs(folder)
    test_empirical_statistics(
        model,
        params,
        x_dim=x_dim,
        num_exemplars=num_exemplars,
        hidden_size=hidden_size,
        batch_size=batch_size,
        path_prefix=folder,
        x_distribution_str=distribution_str,
        plot_w_path=plot_w_path,
        plot_average_stats=plot_average_stats,
        **test_kwargs,
    )
    eval_noise_std = x_dim / 20
    folder = f"{exp_folder}/plots/distribution/x_{distribution_str_to_save}/{epoch+1}/noise_{eval_noise_std}/"
    gfile.makedirs(folder)
    test_empirical_statistics(
        model,
        params,
        x_dim=x_dim,
        num_exemplars=num_exemplars,
        hidden_size=hidden_size,
        batch_size=batch_size,
        path_prefix=folder,
        x_distribution_str=distribution_str,
        plot_w_path=plot_w_path,
        plot_average_stats=plot_average_stats,
        eval_noise_std=eval_noise_std,
        **test_kwargs,
    )


def train(
    rng,
    model,
    state,
    p_train_step,
    exp_folder = "exp/",
    x_dim = 3,
    x_distribution_str = "normal*1+0",
    w_distribution_str = "normal*1+0",
    num_exemplars = 9,
    n_epochs = 100,
    n_iter_per_epoch = 100,
    batch_size = 64,
    hidden_size = 512,
    eval_every_n_epochs = 1000,
    plot_w_path = True,
    plot_average_stats = True,
    noise_std = 0.0,
    **test_kwargs,
):
  """Trains models."""
  rng, new_rng = jax.random.split(rng)

  dropout_rngs = random.split(new_rng, jax.local_device_count())

  sampler = sampler_lib.Sampler(
      num_exemplars,
      x_dim,
      hidden_size,
      x_distribution_fn=sampler_lib.str_to_distribution_fn(x_distribution_str),
      w_distribution_fn=sampler_lib.str_to_distribution_fn(w_distribution_str),
      noise_std=noise_std,
  )

  if len(x_distribution_str.split(",")) > 1:
    test_distributions = (
        "normal*1+0",
        "normal*2.5+0",
        "normal*1+2.5",
        "normal*1+-2.5",
        "normal*1+5.0",
    )
  else:
    test_distributions = (
        "normal*1+0",
        "normal*2.5+0",
        "normal*1+2.5",
    )

  metrics_full = []
  for epoch in range(n_epochs):
    metrics_all = []

    for _ in range(n_iter_per_epoch):
      seqs, coefficients, *_ = sampler.sample(n=batch_size)
      seqs = jnp.array(seqs)
      coefficients = jnp.array(coefficients)
      seqs = common_utils.shard(seqs)
      state, metrics = p_train_step(state, seqs, dropout_rng=dropout_rngs)
      metrics = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], metrics))
      metrics_all.append(metrics)
      metrics_full.append(metrics)

    if epoch == n_epochs - 1 or (epoch + 1) % eval_every_n_epochs == 0:
      if epoch == n_epochs - 1:
        plot_average_stats = True
        plot_w_path = True
      else:
        # plot_average_stats = False
        # plot_w_path = False
        plot_average_stats = True
        plot_w_path = True

      eval_model(
          model,
          jax_utils.unreplicate(state).params,
          batch_size=batch_size,
          num_exemplars=num_exemplars,
          hidden_size=hidden_size,
          exp_folder=exp_folder,
          x_dim=x_dim,
          epoch=epoch,
          plot_w_path=plot_w_path,
          plot_average_stats=plot_average_stats,
          test_distributions=test_distributions,
          **test_kwargs,
      )

    logging.info("Epoch %d is finished.", epoch)
    metrics_all = common_utils.stack_forest(metrics_all)
    logging.info(metrics_all["lr"][-1])
    y_errors = jnp.mean(metrics_all["y_errors"], axis=0) / batch_size
    logging.info(y_errors)

  metrics_full = common_utils.stack_forest(metrics_full)
  metrics_full["y_errors"] = jnp.mean(
      metrics_full["y_errors"], axis=0) / batch_size

  return state, metrics_full
