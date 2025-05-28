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

"""Running training and evaluation on 4 synthetic autoregressive datasets."""

import datetime
import os
import random

from absl import app
from absl import flags
import analyze_experiments
import datasets
import matplotlib.pyplot as plt
import model_library
from models import model_utils
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

now = datetime.datetime.now()
launch_time = now.strftime("%H:%M:%S")

flags.DEFINE_integer(
    "gpu_index", -1,
    "GPU index to run the job among the available GPUs, if it is -1, use CPUs.")
flags.DEFINE_integer("num_trials", 100,
                     "Number of hyperparameter trials to search for.")
flags.DEFINE_integer("seed", 2, "Random seed")
flags.DEFINE_string("model_type", "tft_saf", "Proposed forecasting method.")
flags.DEFINE_bool("display_all_models", "True",
                  "Whether to print all the models or on the best model.")
flags.DEFINE_integer("len_total", 750, "Number of samples.")
flags.DEFINE_integer("synthetic_data_option", 1, "Synthethic data choice.")
flags.DEFINE_string("filename", "experiment_synthetic" + launch_time,
                    "Filename to save the model artifacts.")


def main(args):
  """Orchestrates dataset creation, model training and evaluation.

  Args:
    args: Not used.
  """
  del args  # Not used.

  tf.keras.backend.set_floatx("float32")
  tf.autograph.set_verbosity(0)

  if not os.path.exists("figures"):
    os.makedirs("figures")

  model_utils.set_seed(FLAGS.seed)

  # Set the GPU index.
  if FLAGS.gpu_index >= 0:
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    tf.config.experimental.set_visible_devices(
        devices=gpus[FLAGS.gpu_index], device_type="GPU")
    tf.config.experimental.set_memory_growth(
        device=gpus[FLAGS.gpu_index], enable=True)

  (train_dataset, valid_dataset, test_dataset,
   dataset_params) = datasets.synthetic_autoregressive(
       synthetic_data_option=FLAGS.synthetic_data_option,
       len_total=FLAGS.len_total)

  # Hyperparameter search
  use_nowcast_errors_candidates = [True, False]
  temporal_batch_size_eval = dataset_params["num_items"]
  batch_size_candidates = [32, 64, 128, 256]
  learning_rate_candidates = [0.0001, 0.0003, 0.001, 0.003]
  learning_rate_adaptation_candidates = [0.0003, 0.001, 0.003, 0.01]
  num_units_candidates = [16, 32, 64]
  iterations_candidates = [3000]
  num_encode_candidates = [10, 30, 50]
  keep_prob_candidates = [0.5, 0.8, 1.0]
  num_heads_candidates = [1, 2]
  representation_combination_candidates = ["concatenation", "addition"]
  reset_weights_each_eval_step_candidates = [True]

  best_valid_metric = 1e128
  best_hparams = []
  if FLAGS.display_all_models:
    all_val_mae = []
    all_test_mae = []
    all_val_mape = []
    all_test_mape = []
    all_val_wmape = []
    all_test_wmape = []
    all_val_mse = []
    all_test_mse = []

  for ni in range(FLAGS.num_trials):
    # Try setting the random seed each trial so that we tend to get repeatable
    # hyper-parameters. If you add one at the end the previous ones should
    # remain unchanged.
    model_utils.set_seed(FLAGS.seed + ni)

    chosen_hparams = {
        "batch_size":
            random.sample(batch_size_candidates, 1)[0],
        "learning_rate":
            random.sample(learning_rate_candidates, 1)[0],
        "learning_rate_adaptation":
            random.sample(learning_rate_adaptation_candidates, 1)[0],
        "num_units":
            random.sample(num_units_candidates, 1)[0],
        "iterations":
            random.sample(iterations_candidates, 1)[0],
        "num_encode":
            random.sample(num_encode_candidates, 1)[0],
        "keep_prob":
            random.sample(keep_prob_candidates, 1)[0],
        "num_heads":
            random.sample(num_heads_candidates, 1)[0],
        "representation_combination":
            random.sample(representation_combination_candidates, 1)[0],
        "reset_weights_each_eval_step":
            random.sample(reset_weights_each_eval_step_candidates, 1)[0],
        "use_nowcast_errors":
            random.sample(use_nowcast_errors_candidates, 1)[0],
        "target_index":
            dataset_params["target_index"],
        "static_index_cutoff":
            dataset_params["static_index_cutoff"],
        "display_iterations":
            250,
        "forecast_horizon":
            dataset_params["forecast_horizon"],
        "num_features":
            dataset_params["num_features"],
        "num_static":
            dataset_params["num_static"],
        "num_val_splits":
            (dataset_params["num_items"] * dataset_params["len_val"] //
             temporal_batch_size_eval),
        "num_test_splits":
            (dataset_params["num_items"] * dataset_params["len_test"] //
             temporal_batch_size_eval),
        "temporal_batch_size_eval":
            temporal_batch_size_eval,
    }

    model = model_library.get_model_type(
        FLAGS.model_type, chosen_hparams, loss_form="MSE")

    batched_train_dataset = iter(
        train_dataset.shuffle(1000).repeat(100000000).batch(
            chosen_hparams["batch_size"]))
    batched_valid_dataset = iter(
        valid_dataset.repeat(100000000).batch(temporal_batch_size_eval))
    batched_test_dataset = iter(
        test_dataset.repeat(100000000).batch(temporal_batch_size_eval))

    eval_metrics = model.run_train_eval_pipeline(batched_train_dataset,
                                                 batched_valid_dataset,
                                                 batched_test_dataset)

    if FLAGS.display_all_models and not np.isnan(eval_metrics["val_mse"][-1]):
      print("Best hyperparameter combination: ", flush=True)
      print(best_hparams, flush=True)

      # Select the model iteration based on the validation performance
      model_selection_index = np.argmin(eval_metrics["val_mse"])

      all_val_mae.append(eval_metrics["val_mae"][model_selection_index])
      all_test_mae.append(eval_metrics["test_mae"][model_selection_index])
      all_val_mape.append(eval_metrics["val_mape"][model_selection_index])
      all_test_mape.append(eval_metrics["test_mape"][model_selection_index])
      all_val_wmape.append(eval_metrics["val_wmape"][model_selection_index])
      all_test_wmape.append(eval_metrics["test_wmape"][model_selection_index])
      all_val_mse.append(eval_metrics["val_mse"][model_selection_index])
      all_test_mse.append(eval_metrics["test_mse"][model_selection_index])

      analyze_experiments.display_metrics(
          all_val_mse, all_test_mse, "MSE", 100,
          "figures/" + FLAGS.filename + "_all_hparam_runs_MSE.png")

      print("Best test mae: ", flush=True)
      print(all_test_mae[np.argmin(all_val_mae)], flush=True)

      print("Best test mape: ", flush=True)
      print(all_test_mape[np.argmin(all_val_mape)], flush=True)

      print("Best test wmape: ", flush=True)
      print(all_test_wmape[np.argmin(all_val_wmape)], flush=True)

      print("Best test mse: ", flush=True)
      print(all_test_mse[np.argmin(all_val_mse)], flush=True)

      print("Correlation: ", flush=True)
      print(str(np.corrcoef(all_val_mse, all_test_mse)[0, 1]), flush=True)

      print("Average val/test performance: ", flush=True)
      print(
          np.mean(np.asarray(all_val_mae) / np.asarray(all_test_mae)),
          flush=True)

    current_valid_metric = eval_metrics["val_mse"][model_selection_index]
    if current_valid_metric < best_valid_metric:

      best_hparams = chosen_hparams

      plt.figure()
      plt.plot(
          eval_metrics["display_iterations"],
          eval_metrics["val_mae"],
          "-r",
          label="Val")
      plt.plot(
          eval_metrics["display_iterations"],
          eval_metrics["test_mae"],
          "-b",
          label="Test")
      plt.xlabel("Iterations")
      plt.ylabel("MAE")
      plt.legend()
      plt.savefig("figures/" + FLAGS.filename + "_mae_convergence.png")

      plt.figure()
      plt.plot(
          eval_metrics["display_iterations"],
          eval_metrics["val_mse"],
          "-r",
          label="Val")
      plt.plot(
          eval_metrics["display_iterations"],
          eval_metrics["test_mse"],
          "-b",
          label="Test")
      plt.xlabel("Iterations")
      plt.ylabel("MSE")
      plt.legend()
      plt.savefig("figures/" + FLAGS.filename + "_mse_convergence.png")

      if "meta_self_adapting" in FLAGS.model_type:
        plt.figure()
        plt.plot(
            eval_metrics["display_iterations"],
            eval_metrics["val_self_adaptation"],
            "-b",
            label="Valid self-adapting")
        plt.plot(
            eval_metrics["display_iterations"],
            eval_metrics["test_self_adaptation"],
            "-r",
            label="Test self-adapting")
        plt.xlabel("Iterations")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.savefig("figures/" + FLAGS.filename +
                    "_self_adaptation_convergence.png")

      best_valid_metric = current_valid_metric


if __name__ == "__main__":
  app.run(main)
