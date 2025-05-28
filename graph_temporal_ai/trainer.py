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
import fsspec

import numpy as np
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from src.data import DataModule  # pylint: disable=g-import-not-at-top
from src.lgf_model import LGF
# from src.mpnn_model import MPNN

import torch
from torch import save

import yaml

sys.path.insert(0, "../src")

AVAIL_GPUS = min(1, torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = "gpu" if torch.cuda.is_available() else "cpu"


def train_func(input_data = None, train_args=None):
  """Trains the forecasting model with specified configuration.

  Args:
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
  if train_args.model_name == "lgf":
    model = LGF(adj_mx, train_args)
  # elif train_args.model_name == "mpnn":
  #   model = MPNN(adj_mx, train_args)

  # Transfer data and model to device
  model.to(device)

  # Create data module
  datamodule = DataModule(input_data, train_args)

  # Create the Tune Reporting Callback
  early_stop_callback = EarlyStopping(
      monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min"
  )

  callbacks = [early_stop_callback]

  # Change log directory
  tb_logger = pl_loggers.TensorBoardLogger(save_dir=train_args.save_dir)

  # Model Training
  trainer = Trainer(
      accelerator=accelerator,
      gpus=AVAIL_GPUS,
      strategy="ddp",
      max_epochs=train_args.num_epochs,
      auto_lr_find=True,
      auto_scale_batch_size=False,
      callbacks=callbacks,
      logger=tb_logger,
  )

  trainer.fit(model, datamodule)

  # Save
  save(
      model.state_dict(), train_args.save_dir + train_args.model_name + "_model"
  )

  state_dict = model.state_dict()
  adj_mx = state_dict["adj_mx"].cpu()
  np.save(train_args.save_dir + train_args.model_name + "_graph.npy", adj_mx)

  test_loss = trainer.test(model, datamodule)

  out = trainer.predict(model, datamodule)
  pred = out[0][0]
  truth = out[0][1]
  truth = (
      truth.detach()
      .numpy()
      .reshape(train_args.batch_size, train_args.output_len, -1)
  )
  pred = (
      pred.detach()
      .numpy()
      .reshape(train_args.batch_size, train_args.output_len, -1)
  )

  results = {"test_loss": test_loss, "pred": pred, "truth": truth}

  np.save(train_args.save_dir + train_args.model_name + "_results.npy", results)


if __name__ == "__main__":
  seed_everything(42)

  parser = argparse.ArgumentParser()
  parser.add_argument("-id", "--input-dim", type=int, default=30)
  parser.add_argument("-od", "--output-dim", type=int, default=10)
  parser.add_argument("-hd", "--hidden-dim", type=int, default=128)
  parser.add_argument("-fd", "--fc-dim", type=int, default=16512)
  parser.add_argument("-n", "--num-nodes", type=int, default=500)
  parser.add_argument("-il", "--input-len", type=int, default=14)
  parser.add_argument("-ol", "--output-len", type=int, default=14)

  parser.add_argument("-dn", "--data-name", type=str, default="Sine")
  parser.add_argument("-df", "--data-file", type=str, default="./data/sine.npy")
  parser.add_argument(
      "-gdf", "--graph-data-file", type=str, default="./data/sine_graph.npy"
  )

  parser.add_argument(
      "-sd", "--save-dir", type=str, default="./lightning_logs/"
  )

  parser.add_argument("-f", "--filter-type", type=str, default="learned")
  parser.add_argument("-ds", "--max-diffusion-step", type=int, default=2)
  parser.add_argument(
      "-cl", "--use-curriculum-learning", type=bool, default=True
  )
  parser.add_argument("-cld", "--cl-decay-steps", type=int, default=1000)
  parser.add_argument(
      "-pr", "--use-profiler", default=False, choices=("True", "False")
  )
  parser.add_argument(
      "-gcu", "--use-gc-ru", default=False, choices=("True", "False")
  )
  parser.add_argument(
      "-gcc", "--use-gc-c", default=True, choices=("True", "False")
  )
  parser.add_argument("-dr", "--dropout", type=float, default=0.5)

  parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
  parser.add_argument("-b", "--batch-size", type=int, default=2)
  parser.add_argument("-m", "--model-name", type=str, default="lgf")
  parser.add_argument("-e", "--num-epochs", type=int, default=10)
  parser.add_argument("-l", "--num-layers", type=int, default=2)
  parser.add_argument("-a", "--activation", type=str, default="linear")
  parser.add_argument("-lf", "--loss-func", type=str, default="MAE")

  args = parser.parse_args()

  config_file = "./experiments/" + args.data_name + ".yaml"
  args.save_dir = args.save_dir + args.data_name + "/"

  # load data config
  with open(config_file, "r") as stream:
    try:
      data_config = yaml.safe_load(stream)
      # overwrite args
      vars(args).update(data_config)
    except yaml.YAMLError as exc:
      print(exc)

  args.use_gc_ru = args.use_gc_ru == "True"
  args.use_gc_c = args.use_gc_c == "True"

  # load data from the bucket
  with fsspec.open(args.data_file, "rb") as f:
    data = np.load(f)
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
    os.makedirs(args.save_dir, mode=0o777)
    print("The new directory "+args.save_dir+ " is created!")

  train_func(data, args)
