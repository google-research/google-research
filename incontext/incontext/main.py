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
"""Runs in-context learning experiments."""
import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging
from incontext import model_trainer
from incontext import probe_trainer
from incontext import utils
from jax import random
from tensorflow.io import gfile

# import matplotlib.pyplot as plt

# plt.style.use(".mplstyle")

FLAGS = flags.FLAGS


def main(_):
  args = utils.flags_to_args()

  gfile.makedirs(args.exp_folder)

  with gfile.GFile(os.path.join(args.exp_folder, "config.json"), "w") as handle:
    json.dump(args.initial_dict, handle)

  logging.info(args)

  utils.set_seed(args.seed)
  rng = random.PRNGKey(args.seed)
  rng, new_rng = random.split(rng)

  model, state, p_train_step = model_trainer.get_model(new_rng, args)

  rng, new_rng = random.split(rng)

  logging.info("Running model training")

  state, metrics = model_trainer.train(
      new_rng,
      model,
      state,
      p_train_step,
      exp_folder=args.exp_folder,
      num_exemplars=args.num_exemplars,
      n_epochs=args.n_epochs,
      x_dim=args.x_dim,
      n_iter_per_epoch=args.n_iter_per_epoch,
      batch_size=args.batch_size,
      hidden_size=args.hidden_size,
      x_distribution_str=args.x_distribution_str,
      w_distribution_str=args.w_distribution_str,
      plot_w_path=True,
      plot_planes=args.x_dim == 2,
      plot_dots=args.x_dim == 2,
      plot_fake_least_square_errors=False,
      plot_errors=True,
      plot_least_square_errors=True,
      noise_std=args.noise_std,
  )

  with gfile.GFile(os.path.join(args.exp_folder, "metrics.pickle"),
                   "wb") as handle:
    pickle.dump(metrics, handle)

  model_trainer.save_checkpoint(state, args.exp_folder)

  if args.train_probe:
    rng, new_rng = random.split(rng)
    probe_model, probe_state, probe_p_train_step = probe_trainer.get_probe_model(
        new_rng, args)

    rng, new_rng = random.split(rng)

    probe_state = probe_trainer.train(
        new_rng,
        model,
        state,
        probe_model,
        probe_state,
        probe_p_train_step,
        x_distribution_str=args.x_distribution_str,
        w_distribution_str=args.w_distribution_str,
        x_dim=args.x_dim,
        n_epochs=args.probe_epochs,
        n_iter_per_epoch=args.probe_iters,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
    )


if __name__ == "__main__":
  flags.DEFINE_integer("seed", default=0, help="seed")
  flags.DEFINE_integer("batch_size", default=64, help="batch_size")
  flags.DEFINE_integer("x_dim", default=20, help="x_dim")
  flags.DEFINE_integer("num_exemplars", default=40, help="x_dim")
  flags.DEFINE_string("exp_folder", default="exp", help="exp_folder")
  flags.DEFINE_bool(
      "debug", default=False, help="debug predictions and posterior dist.")
  flags.DEFINE_bool(
      "train_probe", default=False, help="Train probe model after")

  app.run(main)
