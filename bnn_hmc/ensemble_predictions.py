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
    "--save_ensembled_preds",
    action="store_true",
    help="Save final ensembled predictions for further access")
parser.add_argument(
    "remainder_args",
    nargs=argparse.REMAINDER,
    help="If starting with '--' interpret any following args as the list of predictions. "
         "Otherwise, read them from stdin.")

args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)
if args.remainder_args:
  assert args.remainder_args[0] == '--', \
    "Extra args should begin with '--' followed by predictions"
  pred_files = args.remainder_args[1:]
else:
  class StdinIterator:
      def __iter__(self):
          return self

      def __next__(self):
          line = sys.stdin.readline()
          if line == '':
              raise StopIteration
          return line.strip()

  pred_files = StdinIterator()


def get_dirname_tfwriter(args):
  subdirname = "ensembled_preds_{}".format(
    checkpoint_utils.hexdigest(str(pred_files), ndigits=12))
  dirname, tf_writer = script_utils.prepare_logging(subdirname, args)
  return dirname, tf_writer


def main():
  # Initialize training directory
  dirname, tf_writer = get_dirname_tfwriter(args)

  # Initialize data, model, losses and metrics
  (train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn, _,
   _, predict_fn, ensemble_upd_fn, metrics_fns,
   tabulate_metrics) = script_utils.get_data_model_fns(args)

  ensemble_predictions = None
  for pred_index, pred_file in enumerate(pred_files):
    predictions = onp.load(pred_file)
    logger.info(f"Ensembling predictions {pred_index}")

    test_stats = train_utils.evaluate_metrics(predictions, test_set[1],
                                              metrics_fns)

    ensemble_predictions = ensemble_upd_fn(ensemble_predictions,
                                           pred_index, predictions)
    ensemble_stats = train_utils.evaluate_metrics(ensemble_predictions,
                                                  test_set[1], metrics_fns)

    # Log results
    logging_dict = logging_utils.make_logging_dict({}, test_stats, ensemble_stats)
    script_utils.write_to_tensorboard(tf_writer, logging_dict, pred_index)

    logging_dict["telemetry/iteration"], logging_dict["telemetry/iteration_time"] = None, None
    tabulate_dict = script_utils.get_tabulate_dict(tabulate_metrics,
                                                   logging_dict)
    table = logging_utils.make_table(tabulate_dict, pred_index, args.tabulate_freq)
    print(table)

  if args.save_ensembled_preds:
    logger.info("Saving ensembled predictions...")
    onp.save(os.path.join(dirname, "ensembled_preds.npy"), ensemble_predictions)


if __name__ == "__main__":
  script_utils.print_visible_devices()
  main()
