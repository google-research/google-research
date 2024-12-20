# coding=utf-8
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
"""Run Variational Inference."""

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

parser = argparse.ArgumentParser(description="Sample synthetic labels from posterior")
cmd_args_utils.add_common_flags(parser)
parser.add_argument(
    "--params_checkpoint",
    type=str,
    default=None,
    help="SGD/MCMC checkpoint for the setting of parameters to sample labels from")
parser.add_argument(
    "--vi_checkpoint",
    type=str,
    default=None,
    help="VI checkpoint for sampling the setting of parameters to sample labels from")
parser.add_argument(
    "--append_synthetic_dataset_to",
    type=str,
    default=None,
    help="If specified, the synthetic dataset will be appended to this dataset, "
         "instead of being saved down on its own.")

args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)


def get_dirname_tfwriter(args):
  method_name = "sample"
  if args.params_checkpoint:
    method_name += "_from_checkpoint_{}".format(
      checkpoint_utils.hexdigest(args.params_checkpoint))
  elif args.vi_checkpoint:
    method_name += "_from_vi_{}".format(
      checkpoint_utils.hexdigest(args.vi_checkpoint))

  if args.subset_train_to:
    method_name += "_subset_{}".format(args.subset_train_to)
  if args.sequential_training:
    method_name += "_split_{}_of_{}".format(
      1+args.index_sequential_training_fold, args.num_sequential_training_folds)
  if args.stratified_folds:
    method_name += "_{}".format(args.stratified)
  subdirname = "{}__seed_{}".format(method_name, args.seed)
  dirname, tf_writer = script_utils.prepare_logging(subdirname, args)
  return dirname, tf_writer


def load_checkpoint_params(args):
  try:
    checkpoint = checkpoint_utils.load_checkpoint(args.params_checkpoint)
  except Exception as e:
    raise ValueError("Could not load specified params checkpoint", args.vi_checkpoint, e)
  return checkpoint["params"]


def sample_vi_params(args, net_apply, params, net_state, key):
  # Convert the model to MFVI parameterization
  (net_apply, mean_apply, sample_params_fn, vi_params, net_state
  ) = vi.get_mfvi_model_fn(
      net_apply, params, net_state, seed=0, sigma_init=0.)
  net_state.update({
      "mfvi_key": key
  })

  try:
    vi_checkpoint = checkpoint_utils.load_checkpoint(args.vi_checkpoint)
  except Exception as e:
    raise ValueError("Could not load specified VI checkpoint", args.vi_checkpoint, e)
  vi_params = vi_checkpoint["params"]

  params_sampled, new_mfvi_state = sample_params_fn(vi_params, net_state)
  return params_sampled


def main():
  # Initialize training directory
  dirname, tf_writer = get_dirname_tfwriter(args)

  # Initialize data, model, losses and metrics
  (train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn, _,
   _, predict_fn, ensemble_upd_fn, metrics_fns,
   tabulate_metrics) = script_utils.get_data_model_fns(args)

  if args.params_checkpoint:
    params = load_checkpoint_params(args)
  elif args.vi_checkpoint:
    key, vi_sample_key = jax.random.split(key, 2)
    params = sample_vi_params(args, net_apply, params, net_state, key=vi_sample_key)
  else:
    raise ValueError("Either params_checkpoint or vi_checkpoint needs to be "
                     "specifed to sample params configuration from")

  # Initialize variables
  num_devices = len(jax.devices())
  net_state = jax.pmap(lambda _: net_state)(jnp.arange(num_devices))

  new_net_state, predictions = predict_fn(net_apply, params, net_state, train_set)
  synthetic_labels = jax.random.categorical(key, jnp.log(predictions))
  logger.info("Synthetic labels sampled.")

  synth_x_train_chunks, synth_y_train_chunks = [], []
  outfilename = "synth"
  synth_data_info = {
    "num_classes": predictions.shape[-1],
    "train_shape": tuple(t.shape for t in train_set),
    "test_shape": tuple(t.shape for t in test_set),
  }
  if args.append_synthetic_dataset_to:
    try:
      archive = onp.load(args.append_synthetic_dataset_to, allow_pickle=True)
      assert onp.equal(test_set[0], archive["x_test"]).all(), "Test x mismatch with appendee"
      assert onp.equal(test_set[1], archive["y_test"]).all(), "Test x mismatch with appendee"
      assert archive["data_info"].item()["num_classes"] == synth_data_info["num_classes"]
      synth_x_train_chunks.append(archive["x_train"])
      synth_y_train_chunks.append(archive["y_train"])
    except Exception as e:
      raise FileNotFoundError(f"Dataset to append could not be read: {e}")
    outfilename += "_appended_{}".format(
      checkpoint_utils.hexdigest(args.append_synthetic_dataset_to))
  synth_x_train_chunks.append(train_set[0])
  synth_y_train_chunks.append(synthetic_labels)
  try:
    synth_x_train = onp.concatenate(synth_x_train_chunks, axis=1)
    synth_y_train = onp.concatenate(synth_y_train_chunks, axis=1)
  except Exception as e:
    raise ValueError("Could not stack appendee and synthetic datasets.", e)

  onp.savez_compressed(
    os.path.join(dirname, f"{outfilename}.npz"),
    x_train=synth_x_train,
    y_train=synth_y_train,
    x_test=test_set[0],
    y_test=test_set[1],
    data_info=synth_data_info,
  )
  logger.info("Synthetic dataset successfully saved.")


if __name__ == "__main__":
  script_utils.print_visible_devices()
  main()
