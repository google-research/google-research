# coding=utf-8
# Copyright 2025 REDACTED
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

# Copyright 2024 The Google Research Authors.
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
"""Ensemble multiple predictions."""

import os
import numpy as onp
from jax import numpy as jnp
import jax
import tensorflow.compat.v2 as tf
import argparse
import logging

from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import logging_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import optim_utils
from bnn_hmc.utils import script_utils
from bnn_hmc.core import vi

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Ensemble multiple predictions")
cmd_args_utils.add_common_flags(parser)
cmd_args_utils.add_sgd_flags(parser)
parser.add_argument(
    "--optimizer",
    type=str,
    default="Adam",
    choices=["SGD", "Adam"],
    help="Choice of optimizer; (SGD or Adam; default: SGD)")
parser.add_argument(
    "--vi_ensemble_size",
    type=int,
    default=20,
    help="Size of the ensemble sampled in the VI evaluation "
    "(default: 20)")
parser.add_argument(
    "--save_ensembled_preds",
    action="store_true",
    help="Save final ensembled predictions for further access")
parser.add_argument(
    "--sgd_checkpoints",
    action="store_true",
    help="Ensemble SGD checkpoints")
parser.add_argument(
    "--vi_checkpoints",
    action="store_true",
    help="Ensemble VI checkpoints")
parser.add_argument(
    "remainder_args",
    nargs=argparse.REMAINDER,
    help="If starting with '--' interpret any following args as the list of checkpoints. "
         "Otherwise, read them from stdin.")
# TODO potentially bootstrap the args here with a flag

args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)
assert args.sgd_checkpoints != args.vi_checkpoints, \
  "Specify one of --sgd_checkpoints or --vi_checkpoints"
if args.remainder_args:
  assert args.remainder_args[0] == '--', \
    "Extra args should begin with '--' followed by checkpoints"
  checkpoints = args.remainder_args[1:]
else:
  class StdinIterator:
      def __iter__(self):
          return self

      def __next__(self):
          line = sys.stdin.readline()
          if line == '':
              raise StopIteration
          return line.strip()

  checkpoints = StdinIterator()


def get_dirname_tfwriter(args):
  subdirname = "{}_ensembled_{}".format(
    'vi' if args.vi_checkpoints else 'sgd' if args.sgd_checkpoints else '',
    checkpoint_utils.hexdigest(str(checkpoints), ndigits=12))
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

  if args.vi_checkpoints:
    # Convert the model to MFVI parameterization
    net_apply, mean_apply, _, params, net_state = vi.get_mfvi_model_fn(
        net_apply, params, net_state, seed=0, sigma_init=onp.nan)
    predict_fn = make_vi_ensemble_predict_fn(predict_fn, ensemble_upd_fn, args)

  ensemble_predictions = None
  for checkpoint_index, checkpoint in enumerate(checkpoints):
    checkpoint_dict = checkpoint_utils.load_checkpoint(checkpoint)
    iteration_num, params, net_state, _, _ = (
        checkpoint_utils.parse_sgd_checkpoint_dict(checkpoint_dict))
    logger.info(f"Ensembling checkpoint {checkpoint_index} with trained iteration no {iteration_num}")

    # Evaluate the model
    net_state, test_predictions = predict_fn(net_apply, params, net_state, test_set)
    test_stats = train_utils.evaluate_metrics(test_predictions, test_set[1],
                                              metrics_fns)

    ensemble_predictions = ensemble_upd_fn(ensemble_predictions,
                                           checkpoint_index, test_predictions)
    ensemble_stats = train_utils.evaluate_metrics(ensemble_predictions,
                                                  test_set[1], metrics_fns)

    # Log results
    logging_dict = logging_utils.make_logging_dict({}, test_stats, ensemble_stats)
    script_utils.write_to_tensorboard(tf_writer, logging_dict, checkpoint_index)
    if args.vi_checkpoints:
      # TODO ensemble E[theta] and E[theta^2] for computing ensembled second moment
      pass
      # # Add a histogram of MFVI stds
      # with tf_writer.as_default():
      #   stds = jax.tree.map(jax.nn.softplus, params["inv_softplus_std"])
      #   stds = jnp.concatenate([std.reshape(-1) for std in jax.tree.leaves(stds)])
      #   tf.summary.histogram("MFVI/param_stds", stds, step=checkpoint_index)

    logging_dict["telemetry/iteration"], logging_dict["telemetry/iteration_time"] = None, None
    tabulate_dict = script_utils.get_tabulate_dict(tabulate_metrics,
                                                   logging_dict)
    table = logging_utils.make_table(tabulate_dict, checkpoint_index, args.tabulate_freq)
    print(table)

  if args.save_ensembled_preds:
    logger.info("Saving ensembled predictions...")
    onp.save(os.path.join(dirname, "ensembled_preds.npy"), ensemble_predictions)


if __name__ == "__main__":
  script_utils.print_visible_devices()
  train_model()
