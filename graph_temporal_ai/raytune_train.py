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

"""Train file with automatic hyper-parameter tuning using Ray Tune.

Contains the train file to start training the latent graph forecaster model.
"""

import argparse
import os
import sys
from typing import Any, Dict

import numpy as np
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

sys.path.insert(0, "./src")
from src.data import DataModule  # pylint: disable=g-import-not-at-top
from src.lgf_model import LGF

import torch
from torch import save

import yaml

AVAIL_GPUS = min(0, torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# grid search config
config = {
    "hidden_dim": tune.grid_search([8, 16]),
    # "learning_rate": tune.loguniform(1e-4, 1e-1),
    # "batch_size": tune.choice([32, 64, 128]),
}


def train_func(tune_config,
               input_data = None,
               train_args = None):
  """Trains the forecasting model with specified configuration.

  Args:
    tune_config: hyperparameters that need tuning
    input_data: numpy array for the input data
    train_args: other hyperparameters
  """
  if train_args.filter_type == "learned":
    # intialize
    adj_mx = torch.eye(train_args.num_nodes)
  else:
    # load input graphs
    adj_mx = np.load(train_args.graph_data_file)

  # Create deep learning model
  model = LGF(adj_mx, train_args, tune_config)

  # Create data module
  datamodule = DataModule(input_data, train_args)

  # Create the Tune Reporting Callback
  metrics = {"loss": "val_loss"}
  early_stop_callback = EarlyStopping(
      monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")
  tune_report_callback = TuneReportCallback(metrics, on="validation_end")
  callbacks = [early_stop_callback, tune_report_callback]

  # Model Training
  trainer = Trainer(
      gpus=AVAIL_GPUS,
      max_epochs=args.num_epochs,
      auto_lr_find=True,
      auto_scale_batch_size=False,
      callbacks=callbacks)

  trainer.fit(model, datamodule)

  # Save
  save(model.state_dict(),
       train_args.save_dir + train_args.model_name + "_model")

  result = trainer.test(model, datamodule)

  out = trainer.predict(model, datamodule)
  pred = out[0][0]
  truth = out[0][1]
  truth = truth.detach().numpy().reshape(train_args.batch_size,
                                         train_args.output_len, -1)
  pred = pred.detach().numpy().reshape(train_args.batch_size,
                                       train_args.output_len, -1)
  np.save(train_args.save_dir + train_args.model_name + "_results.npy", {
      "test_loss": result,
      "pred": pred,
      "truth": truth
  })


if __name__ == "__main__":
  seed_everything(42)

  parser = argparse.ArgumentParser()
  parser.add_argument("-id", "--input-dim", type=int, default=30)
  parser.add_argument("-od", "--output-dim", type=int, default=10)
  parser.add_argument("-hd", "--hidden-dim", type=int, default=32)
  parser.add_argument("-n", "--num-nodes", type=int, default=500)
  parser.add_argument("-il", "--input-len", type=int, default=14)
  parser.add_argument("-ol", "--output-len", type=int, default=14)
  parser.add_argument("-dr", "--dropout", type=float, default=0.5)

  parser.add_argument(
      "-dc", "--data-config", type=str, default="./experiments/Sine.yaml")
  parser.add_argument("-f", "--filter-type", type=str, default="learned")
  parser.add_argument("-ds", "--max-diffusion-step", type=int, default=2)
  parser.add_argument(
      "-cl", "--use-curriculum-learning", type=bool, default=True)
  parser.add_argument("-cld", "--cl-decay-steps", type=int, default=1000)
  parser.add_argument("-gcu", "--use-gc-ru", type=bool, default=False)
  parser.add_argument("-gcc", "--use-gc-c", type=bool, default=True)

  parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
  parser.add_argument("-b", "--batch-size", type=int, default=2)
  parser.add_argument("-m", "--model-name", type=str, default="lgf")
  parser.add_argument("-e", "--num-epochs", type=int, default=1)
  parser.add_argument("-l", "--num-layers", type=int, default=2)
  parser.add_argument("-a", "--activation", type=str, default="linear")

  args = parser.parse_args()

  # load data config
  config_path = args.data_config
  if not os.path.isabs(config_path):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(file_dir, config_path)

  with open(config_path, "r") as stream:
    try:
      data_config = yaml.safe_load(stream)
      # overwrite args
      vars(args).update(data_config)
    except yaml.YAMLError as exc:
      print(exc)

  # load data
  data_path = args.data_file
  if not os.path.isabs(data_path):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(file_dir, data_path)
  data = np.load(data_path)
  print("data loaded! shape:", data.shape)

  # reshape data if not graph temporal
  if args.model_name == "lstm":
    # Reshape for LSTM
    num_time = data.shape[0]
    data = np.reshape(data, (num_time, -1), order="F")  # fotran-like indexing
    args.input_dim = data.shape[-1]
    args.output_dim = args.num_nodes * args.output_dim

  # create log directory
  if not os.path.exists(args.save_dir):
    # Create a new directory because it does not exist
    os.makedirs(args.save_dir)
    print("The new directory " + args.save_dir + " is created!")

  trainable = tune.with_parameters(train_func, input_data=data, train_args=args)

  cpu_count = len(os.sched_getaffinity(0))
  grid_size = 4
  resources_per_trial = {
      "cpu": cpu_count / grid_size,
      "gpu": AVAIL_GPUS / grid_size
  }

  analysis = tune.run(
      trainable,
      resources_per_trial=resources_per_trial,
      local_dir=args.save_dir,
      metric="loss",
      mode="min",
      config=config,
      name="tune_test")

  print("Best hyperparameters found were: ", analysis.best_config)
