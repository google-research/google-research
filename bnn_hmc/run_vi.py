# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Run Variational Inference."""

import os
import numpy as onp
from jax import numpy as jnp
import jax
import tensorflow.compat.v2 as tf
import argparse

from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import logging_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import optim_utils
from bnn_hmc.utils import script_utils
from bnn_hmc.core import vi

parser = argparse.ArgumentParser(description="Run MFVI training")
cmd_args_utils.add_common_flags(parser)
cmd_args_utils.add_sgd_flags(parser)
parser.add_argument(
    "--optimizer",
    type=str,
    default="Adam",
    choices=["SGD", "Adam"],
    help="Choice of optimizer; (SGD or Adam; default: SGD)")
parser.add_argument(
    "--vi_sigma_init",
    type=float,
    default=1e-3,
    help="Initial value of the standard deviation over the "
    "weights in MFVI (default: 1e-3)")
parser.add_argument(
    "--vi_ensemble_size",
    type=int,
    default=20,
    help="Size of the ensemble sampled in the VI evaluation "
    "(default: 20)")
parser.add_argument(
    "--mean_init_checkpoint",
    type=str,
    default=None,
    help="SGD checkpoint to use for initialization of the "
    "mean of the MFVI approximation")

args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)


def get_optimizer(lr_schedule, args):
  if args.optimizer == "SGD":
    optimizer = optim_utils.make_sgd_optimizer(
        lr_schedule, momentum_decay=args.momentum_decay)
  elif args.optimizer == "Adam":
    optimizer = optim_utils.make_adam_optimizer(lr_schedule)
  return optimizer


def get_dirname_tfwriter(args):
  method_name = "mfvi_initsigma_{}".format(args.vi_sigma_init)
  if args.mean_init_checkpoint:
    method_name += "_meaninit"
  if args.optimizer == "SGD":
    optimizer_name = "opt_sgd_{}".format(args.momentum_decay)
  elif args.optimizer == "Adam":
    optimizer_name = "opt_adam"
  lr_schedule_name = "lr_sch_i_{}".format(args.init_step_size)
  hypers_name = "_epochs_{}_wd_{}_batchsize_{}_temp_{}".format(
      args.num_epochs, args.weight_decay, args.batch_size, args.temperature)
  subdirname = "{}__{}__{}__{}__seed_{}".format(method_name, optimizer_name,
                                                lr_schedule_name, hypers_name,
                                                args.seed)
  dirname, tf_writer = script_utils.prepare_logging(subdirname, args)
  return dirname, tf_writer


def make_vi_ensemble_predict_fn(predict_fn, ensemble_upd_fn, args):

  def vi_ensemble_predict_fn(net_apply, params, net_state, ds):
    net_state, all_preds = jax.lax.scan(
        lambda state, _: predict_fn(net_apply, params, state, ds),
        init=net_state,
        xs=jnp.arange(args.vi_ensemble_size))

    ensemble_predictions = None
    num_ensembled = 0
    for pred in all_preds:
      ensemble_predictions = ensemble_upd_fn(ensemble_predictions,
                                             num_ensembled, pred)
      num_ensembled += 1
    return net_state, ensemble_predictions

  return vi_ensemble_predict_fn


def train_model():
  # Initialize training directory
  dirname, tf_writer = get_dirname_tfwriter(args)

  # Initialize data, model, losses and metrics
  (train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn, _,
   _, predict_fn, ensemble_upd_fn, metrics_fns,
   tabulate_metrics) = script_utils.get_data_model_fns(args)

  # Convert the model to MFVI parameterization
  net_apply, mean_apply, _, params, net_state = vi.get_mfvi_model_fn(
      net_apply, params, net_state, seed=0, sigma_init=args.vi_sigma_init)
  prior_kl = vi.make_kl_with_gaussian_prior(args.weight_decay, args.temperature)
  vi_ensemble_predict_fn = make_vi_ensemble_predict_fn(predict_fn,
                                                       ensemble_upd_fn, args)

  # Initialize step-size schedule and optimizer
  num_batches, total_steps = script_utils.get_num_batches_total_steps(
      args, train_set)
  num_devices = len(jax.devices())
  lr_schedule = optim_utils.make_cosine_lr_schedule(args.init_step_size,
                                                    total_steps)
  optimizer = get_optimizer(lr_schedule, args)

  # Initialize variables
  opt_state = optimizer.init(params)
  net_state = jax.pmap(lambda _: net_state)(jnp.arange(num_devices))
  key = jax.random.split(key, num_devices)
  init_dict = checkpoint_utils.make_sgd_checkpoint_dict(-1, params, net_state,
                                                        opt_state, key)
  init_dict = script_utils.get_initialization_dict(dirname, args, init_dict)
  start_iteration, params, net_state, opt_state, key = (
      checkpoint_utils.parse_sgd_checkpoint_dict(init_dict))
  start_iteration += 1

  # Loading mean checkpoint
  if args.mean_init_checkpoint:
    print("Initializing VI mean from the provided checkpoint")
    ckpt_dict = checkpoint_utils.load_checkpoint(args.mean_init_checkpoint)
    mean_params = checkpoint_utils.parse_sgd_checkpoint_dict(ckpt_dict)[1]
    params["mean"] = mean_params

  # Define train epoch
  sgd_train_epoch = script_utils.time_fn(
      train_utils.make_sgd_train_epoch(net_apply, log_likelihood_fn, prior_kl,
                                       optimizer, num_batches))

  # Train
  for iteration in range(start_iteration, args.num_epochs):

    (params, net_state, opt_state, elbo_avg, key), iteration_time = (
        sgd_train_epoch(params, net_state, opt_state, train_set, key))

    # Evaluate the model
    train_stats = {"ELBO": elbo_avg, "KL": prior_kl(params)}
    test_stats, ensemble_stats = {}, {}
    if (iteration % args.eval_freq == 0) or (iteration == args.num_epochs - 1):
      # Evaluate the mean
      _, test_predictions, train_predictions, test_stats, train_stats_ = (
          script_utils.evaluate(mean_apply, params, net_state, train_set,
                                test_set, predict_fn, metrics_fns, prior_kl))
      train_stats.update(train_stats_)
      del train_stats["prior"]

      # Evaluate the ensemble
      net_state, ensemble_predictions = onp.asarray(
          vi_ensemble_predict_fn(net_apply, params, net_state, test_set))
      ensemble_stats = train_utils.evaluate_metrics(ensemble_predictions,
                                                    test_set[1], metrics_fns)

    # Save checkpoint
    if iteration % args.save_freq == 0 or iteration == args.num_epochs - 1:
      checkpoint_name = checkpoint_utils.make_checkpoint_name(iteration)
      checkpoint_path = os.path.join(dirname, checkpoint_name)
      checkpoint_dict = checkpoint_utils.make_sgd_checkpoint_dict(
          iteration, params, net_state, opt_state, key)
      checkpoint_utils.save_checkpoint(checkpoint_path, checkpoint_dict)

    # Log results
    other_logs = script_utils.get_common_logs(iteration, iteration_time, args)
    other_logs["hypers/step_size"] = lr_schedule(opt_state[-1].count)
    other_logs["hypers/momentum"] = args.momentum_decay
    logging_dict = logging_utils.make_logging_dict(train_stats, test_stats,
                                                   ensemble_stats)
    logging_dict.update(other_logs)
    script_utils.write_to_tensorboard(tf_writer, logging_dict, iteration)
    # Add a histogram of MFVI stds
    with tf_writer.as_default():
      stds = jax.tree_map(jax.nn.softplus, params["inv_softplus_std"])
      stds = jnp.concatenate([std.reshape(-1) for std in jax.tree_leaves(stds)])
      tf.summary.histogram("MFVI/param_stds", stds, step=iteration)

    tabulate_dict = script_utils.get_tabulate_dict(tabulate_metrics,
                                                   logging_dict)
    tabulate_dict["lr"] = lr_schedule(opt_state[-1].count)
    table = logging_utils.make_table(tabulate_dict, iteration - start_iteration,
                                     args.tabulate_freq)
    print(table)


if __name__ == "__main__":
  script_utils.print_visible_devices()
  train_model()
