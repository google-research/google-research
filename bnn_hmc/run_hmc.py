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
# coding=utf-8
"""Run an Hamiltonian Monte Carlo chain on a cloud TPU."""
import argparse
import os

import jax
from jax import numpy as jnp

from utils import checkpoint_utils
from utils import cmd_args_utils
from utils import logging_utils
from utils import script_utils
from utils import train_utils
from utils import tree_utils

parser = argparse.ArgumentParser(description="Run an HMC chain on a cloud TPU")
cmd_args_utils.add_common_flags(parser)
parser.add_argument(
    "--step_size", type=float, default=1.e-4, help="HMC step size")
parser.add_argument(
    "--trajectory_len", type=float, default=1.e-3, help="HMC trajectory length")
parser.add_argument(
    "--num_iterations",
    type=int,
    default=1000,
    help="Total number of HMC iterations")
parser.add_argument(
    "--max_num_leapfrog_steps",
    type=int,
    default=10000,
    help="Maximum number of leapfrog steps allowed; increase to"
    "run longer trajectories")
parser.add_argument(
    "--num_burn_in_iterations",
    type=int,
    default=0,
    help="Number of burn-in iterations")
parser.add_argument(
    "--no_mh",
    default=False,
    action="store_true",
    help="If set, Metropolis Hastings correction is ignored")

args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)


def get_dirname_tfwriter(cmd_args):
  subdirname = ("model_{}_wd_{}_stepsize_{}_trajlen_{}_burnin_{}_mh_{}_temp_{}_"
                "seed_{}".format(cmd_args.model_name, cmd_args.weight_decay,
                                 cmd_args.step_size, cmd_args.trajectory_len,
                                 cmd_args.num_burn_in_iterations,
                                 not cmd_args.no_mh, cmd_args.temperature,
                                 cmd_args.seed))
  dirname, tf_writer = script_utils.prepare_logging(subdirname, cmd_args)
  return dirname, tf_writer


def train_model():
  """Trains model via HMC."""
  dirname, tf_writer = get_dirname_tfwriter(args)

  # Initialize data, model, losses and metrics
  (train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn,
   log_prior_fn, log_prior_diff_fn, predict_fn, ensemble_upd_fn, metrics_fns,
   tabulate_metrics) = (
       script_utils.get_data_model_fns(args))

  # Initialize variables
  num_devices = len(jax.devices())
  net_state = jax.pmap(lambda _: net_state)(jnp.arange(num_devices))
  step_size = args.step_size
  trajectory_len = args.trajectory_len
  init_dict = checkpoint_utils.make_hmc_checkpoint_dict(-1, params, net_state,
                                                        key, step_size, None, 0,
                                                        None)
  init_dict = script_utils.get_initialization_dict(dirname, args, init_dict)
  (start_iteration, params, net_state, key, step_size, _, num_ensembled,
   ensemble_predictions) = (
       checkpoint_utils.parse_hmc_checkpoint_dict(init_dict))
  start_iteration += 1

  # manually convert all params to dtype
  dtype = script_utils.get_dtype(args)
  params = jax.tree_map(lambda p: p.astype(dtype), params)

  update, get_log_prob_and_grad = train_utils.make_hmc_update(
      net_apply, log_likelihood_fn, log_prior_fn, log_prior_diff_fn,
      args.max_num_leapfrog_steps, 1., 0.)

  update = script_utils.time_fn(update)

  log_prob, state_grad, log_likelihood, net_state = (
      get_log_prob_and_grad(train_set, params, net_state))

  assert log_prob.dtype == dtype, (
      "log_prob data type {} does not match specified data type {}".format(
          log_prob.dtype, dtype))

  grad_types = tree_utils.tree_get_types(state_grad)
  assert all([
      g_type == dtype for g_type in grad_types
  ]), ("Gradient data types {} do not match specified data type {}".format(
      grad_types, dtype))

  for iteration in range(start_iteration, args.num_iterations):

    in_burnin = (iteration < args.num_burn_in_iterations)
    do_mh_correction = (not args.no_mh) and (not in_burnin)

    (params, net_state, log_likelihood, state_grad, step_size, key, accept_prob,
     accepted), iteration_time = (
         update(train_set, params, net_state, log_likelihood, state_grad, key,
                step_size, trajectory_len, do_mh_correction))

    # Evaluation
    _, test_predictions, train_predictions, test_stats, train_stats = (
        script_utils.evaluate(net_apply, params, net_state, train_set, test_set,
                              predict_fn, metrics_fns, log_prior_fn))
    del train_predictions  # unused

    # Ensembling
    if ((not in_burnin) and accepted) or args.no_mh:
      ensemble_predictions = ensemble_upd_fn(ensemble_predictions,
                                             num_ensembled, test_predictions)
      ensemble_stats = train_utils.evaluate_metrics(ensemble_predictions,
                                                    test_set[1], metrics_fns)
      num_ensembled += 1
    else:
      ensemble_stats = {}

    # Save the checkpoint
    checkpoint_name = checkpoint_utils.make_checkpoint_name(iteration)
    checkpoint_path = os.path.join(dirname, checkpoint_name)
    checkpoint_dict = checkpoint_utils.make_hmc_checkpoint_dict(
        iteration, params, net_state, key, step_size, accepted, num_ensembled,
        ensemble_predictions)
    checkpoint_utils.save_checkpoint(checkpoint_path, checkpoint_dict)

    # Logging
    other_logs = script_utils.get_common_logs(iteration, iteration_time, args)
    other_logs.update({
        "telemetry/accept_prob": accept_prob,
        "telemetry/accepted": accepted,
        "telemetry/num_ensembled": num_ensembled,
        "hypers/step_size": step_size,
        "hypers/trajectory_len": trajectory_len,
        "debug/do_mh_correction": float(do_mh_correction),
        "debug/in_burnin": float(in_burnin)
    })
    logging_dict = logging_utils.make_logging_dict(train_stats, test_stats,
                                                   ensemble_stats)
    logging_dict.update(other_logs)

    script_utils.write_to_tensorboard(tf_writer, logging_dict, iteration)
    tabulate_dict = script_utils.get_tabulate_dict(tabulate_metrics,
                                                   logging_dict)
    tabulate_dict["accept_p"] = accept_prob
    tabulate_dict["accepted"] = accepted

    table = logging_utils.make_table(tabulate_dict, iteration - start_iteration,
                                     args.tabulate_freq)
    print(table)


if __name__ == "__main__":
  script_utils.print_visible_devices()
  train_model()
