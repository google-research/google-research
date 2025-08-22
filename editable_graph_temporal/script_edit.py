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
"""Script for running the model editing."""

import argparse
import os
from editable_graph_temporal import editer


def get_args():
  """Parses the input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("-il", "--input-len", type=int, default=20)
  parser.add_argument("-ol", "--output-len", type=int, default=20)
  # Ratios for splitting the whole dataset to train/test/val sets, the first
  # number is the ratio of training data, the second number is the ratio of
  # test data, one minus the summation of these two numbers is the ratio of
  # validation data.
  parser.add_argument("-sp", "--splits", type=list, default=[0.7, 0.2])

  parser.add_argument("-hd", "--hidden-dim", type=int, default=512)
  parser.add_argument("-l", "--num-layers", type=int, default=1)
  # Output activation function for encoder and decoder of GATRNN model,
  # "linear", "relu", or "tanh".
  parser.add_argument("-a", "--activation", type=str, default="linear")
  parser.add_argument("-dr", "--dropout", type=float, default=0.1)
  # Total number of relation types, i.e., graph edge types,
  # existed in the dataset.
  parser.add_argument("-nr", "--num-relation-types", type=int, default=2)
  # Negative slope value for the leaky ReLU in GAT convolution.
  parser.add_argument("-ns", "--negative-slope", type=float, default=0.2)
  # Whether parameters for computing Query and Key are shared in GAT
  # convolution.
  parser.add_argument("-sa", "--share-attn-weights", type=bool, default=False)
  # Temperature for sampling from Gumbel-Softmax distribution.
  parser.add_argument("-t", "--temperature", type=float, default=0.5)
  # Adjacency matrix type, "fixed" for a fixed ground truth matrix,
  # "learned" for learned matrix, "empty" for identity matrix.
  parser.add_argument("-at", "--adj-type", type=str, default="learned")

  # Edge prediction loss coefficient, multiplied with edge prediction loss.
  parser.add_argument("-ec", "--edge-co", type=float, default=0.1)
  parser.add_argument("-b", "--batch-size", type=int, default=32)
  parser.add_argument("-nw", "--num-workers", type=int, default=16)
  # Learning rates of global embeddings (first), graph prediction model
  # parameters (second), and time series forecast model parameters (third).
  parser.add_argument(
      "-lr", "--learning-rates", type=float, default=[5.0e-4, 6.0e-4, 3.0e-4])
  parser.add_argument("-uc", "--use-cons-optim", type=bool, default=False)
  # If using constrained optimization, thresholds for global embeddings (first),
  # graph prediction model parameters (second), and time series forecast model
  # parameters (third).
  parser.add_argument(
      "-ct", "--cons-thres", type=float, default=[1.0e-1, 1.0e-1, 1.0e-2])
  parser.add_argument("-e", "--num-max-fine-tune-iters", type=int, default=1500)

  parser.add_argument("-mp", "--model-path", type=str, default="./model.ckpt")
  parser.add_argument("-s", "--save-dir", type=str, default="./test/")
  parser.add_argument(
      "-d",
      "--data-path",
      type=str,
      default="./spring_data/spring_n10t10000.npz")
  parser.add_argument("-sm", "--save-model", type=bool, default=False)

  return parser.parse_args()


if __name__ == "__main__":
  args = get_args()
  if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

  model_editer = editer.Editer(args)
  model_editer.train()
