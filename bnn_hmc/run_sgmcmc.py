# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Run an SGLD chain on a cloud TPU. We are not using data augmentation."""

import os
from jax import numpy as jnp
import jax
import argparse
from collections import OrderedDict

from bnn_hmc.core import sgmcmc
from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import logging_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import optim_utils
from bnn_hmc.utils import script_utils

parser = argparse.ArgumentParser(description="Run SG-MCMC training")
cmd_args_utils.add_common_flags(parser)
cmd_args_utils.add_sgd_flags(parser)

parser.add_argument(
    "--save_all_ensembled",
    action="store_true",
    help="Save all the networks that are ensembled")
parser.add_argument(
    "--ensemble_freq",
    type=int,
    default=10,
    help="Frequency of checkpointing (epochs; default: 10)")

parser.add_argument(
    "--preconditioner",
    type=str,
    default="None",
    choices=["None", "RMSprop"],
    help="Choice of preconditioner; None or RMSprop "
    "(default: None)")

# Step size schedule
parser.add_argument(
    "--step_size_schedule",
    type=str,
    default="constant",
    choices=["constant", "cyclical"],
    help="Choice step size schedule;"
    "constant sets the step size to final_step_size "
    "after a cosine burn-in for num_burnin_epochs epochs."
    "Cyclical uses a constant burn-in for num_burnin_epochs "
    "epochs and then a cosine cyclical schedule"
    "(default: constant)")
parser.add_argument(
    "--num_burnin_epochs",
    type=int,
    default=300,
    help="Number of epochs before final lr is reached")
parser.add_argument(
    "--final_step_size",
    type=float,
    default=None,
    help="Final step size (used only with constant schedule; "
    "default: init_step_size)")
parser.add_argument(
    "--step_size_cycle_length_epochs",
    type=float,
    default=50,
    help="Cycle length "
    "(epochs; used only with cyclic schedule; default: 50)")

args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)


def get_dirname_tfwriter(args):
  method_name = "sgld_mom_{}_preconditioner_{}".format(args.momentum_decay,
                                                       args.preconditioner)
  lr_schedule_name = "lr_sch_{}_i_{}_f_{}_c_{}_bi_{}".format(
      args.step_size_schedule, args.init_step_size, args.final_step_size,
      args.step_size_cycle_length_epochs, args.num_burnin_epochs)
  hypers_name = "_epochs_{}_wd_{}_batchsize_{}_temp_{}".format(
      args.num_epochs, args.weight_decay, args.batch_size, args.temperature)
  subdirname = "{}__{}__{}__seed_{}".format(method_name, lr_schedule_name,
                                            hypers_name, args.seed)
  dirname, tf_writer = script_utils.prepare_logging(subdirname, args)
  return dirname, tf_writer


def get_lr_schedule(num_batches, args):
  burnin_steps = num_batches * args.num_burnin_epochs
  if args.step_size_schedule.lower() == "constant":
    final_step_size = args.final_step_size or args.init_step_size
    lr_schedule = optim_utils.make_constant_lr_schedule_with_cosine_burnin(
        args.init_step_size, final_step_size, burnin_steps)
  else:
    # Use cyclical schedule
    cycle_steps = args.step_size_cycle_length_epochs * num_batches
    lr_schedule = (
        optim_utils.make_cyclical_cosine_lr_schedule_with_const_burnin(
            args.init_step_size, burnin_steps, cycle_steps))
  return lr_schedule


def get_preconditioner(args):
  if args.preconditioner == "None":
    preconditioner = None
  else:
    preconditioner = sgmcmc.get_rmsprop_preconditioner()
  return preconditioner


def is_eval_ens_save_epoch(iteration, args):
  is_evaluation_epoch = ((iteration % args.eval_freq == 0) or
                         (iteration == args.num_epochs - 1))
  is_ensembling_epoch = ((iteration > args.num_burnin_epochs) and (
      (iteration - args.num_burnin_epochs + 1) % args.ensemble_freq == 0))
  if args.save_all_ensembled:
    is_save_epoch = is_ensembling_epoch
  else:
    is_save_epoch = (
        iteration % args.save_freq == 0 or iteration == args.num_epochs - 1)
  return is_evaluation_epoch, is_ensembling_epoch, is_save_epoch


def train_model():
  # Initialize training directory
  dirname, tf_writer = get_dirname_tfwriter(args)

  # Initialize data, model, losses and metrics
  (train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn,
   log_prior_fn, _, predict_fn, ensemble_upd_fn, metrics_fns,
   tabulate_metrics) = script_utils.get_data_model_fns(args)

  # Initialize step-size schedule and optimizer
  num_batches, total_steps = script_utils.get_num_batches_total_steps(
      args, train_set)
  num_devices = len(jax.devices())
  lr_schedule = get_lr_schedule(num_batches, args)
  preconditioner = get_preconditioner(args)
  optimizer = sgmcmc.sgld_gradient_update(
      lr_schedule,
      momentum_decay=args.momentum_decay,
      seed=args.seed,
      preconditioner=preconditioner)

  # Initialize variables
  opt_state = optimizer.init(params)
  net_state = jax.pmap(lambda _: net_state)(jnp.arange(num_devices))
  key = jax.random.split(key, num_devices)
  init_dict = checkpoint_utils.make_sgmcmc_checkpoint_dict(
      -1, params, net_state, opt_state, key, 0, None, None)
  init_dict = script_utils.get_initialization_dict(dirname, args, init_dict)
  (start_iteration, params, net_state, opt_state, key, num_ensembled, _,
   ensemble_predictions) = (
       checkpoint_utils.parse_sgmcmc_checkpoint_dict(init_dict))
  start_iteration += 1

  # Define train epoch
  sgmcmc_train_epoch = script_utils.time_fn(
      train_utils.make_sgd_train_epoch(net_apply, log_likelihood_fn,
                                       log_prior_fn, optimizer, num_batches))

  # Train
  for iteration in range(start_iteration, args.num_epochs):

    (params, net_state, opt_state, logprob_avg, key), iteration_time = (
        sgmcmc_train_epoch(params, net_state, opt_state, train_set, key))

    is_evaluation_epoch, is_ensembling_epoch, is_save_epoch = (
        is_eval_ens_save_epoch(iteration, args))

    # Evaluate the model
    train_stats, test_stats = {"log_prob": logprob_avg}, {}
    if is_evaluation_epoch or is_ensembling_epoch:
      _, test_predictions, train_predictions, test_stats, train_stats_ = (
          script_utils.evaluate(net_apply, params, net_state, train_set,
                                test_set, predict_fn, metrics_fns,
                                log_prior_fn))
      train_stats.update(train_stats_)

    # Ensemble predictions
    if is_ensembling_epoch:
      ensemble_predictions = ensemble_upd_fn(ensemble_predictions,
                                             num_ensembled, test_predictions)
      ensemble_stats = train_utils.evaluate_metrics(ensemble_predictions,
                                                    test_set[1], metrics_fns)
      num_ensembled += 1
    else:
      ensemble_stats = {}
      test_predictions = None

    # Save checkpoint
    if is_save_epoch:
      checkpoint_name = checkpoint_utils.make_checkpoint_name(iteration)
      checkpoint_path = os.path.join(dirname, checkpoint_name)
      checkpoint_dict = checkpoint_utils.make_sgmcmc_checkpoint_dict(
          iteration, params, net_state, opt_state, key, num_ensembled,
          test_predictions, ensemble_predictions)
      checkpoint_utils.save_checkpoint(checkpoint_path, checkpoint_dict)

    # Log results
    other_logs = script_utils.get_common_logs(iteration, iteration_time, args)
    other_logs["hypers/step_size"] = lr_schedule(opt_state.count)
    other_logs["hypers/momentum"] = args.momentum_decay
    other_logs["telemetry/num_ensembled"] = num_ensembled
    logging_dict = logging_utils.make_logging_dict(train_stats, test_stats,
                                                   ensemble_stats)
    logging_dict.update(other_logs)
    script_utils.write_to_tensorboard(tf_writer, logging_dict, iteration)

    tabulate_dict = script_utils.get_tabulate_dict(tabulate_metrics,
                                                   logging_dict)
    tabulate_dict["lr"] = lr_schedule(opt_state.count)
    table = logging_utils.make_table(tabulate_dict, iteration - start_iteration,
                                     args.tabulate_freq)
    print(table)


if __name__ == "__main__":
  script_utils.print_visible_devices()
  train_model()
