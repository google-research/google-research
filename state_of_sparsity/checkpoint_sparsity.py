# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Calculates weight sparsity for a model checkpoint."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf


flags.DEFINE_string(
    "checkpoint",
    None,
    "Path to checkpoint."
)
flags.DEFINE_enum(
    "sparsity_technique",
    "magnitude_pruning",
    [
        "magnitude_pruning",
        "random_pruning",
        "variational_dropout",
        "l0_regularization"
    ],
    "Technique used to produce model checkpoint."
)
flags.DEFINE_enum(
    "model",
    "transformer",
    ["transformer", "rn50"],
    "Model saved in checkpoint."
)
flags.DEFINE_float(
    "log_alpha_threshold",
    3.0,
    "log alpha threshold for variational dropout checkpoint."
)

FLAGS = flags.FLAGS
EPSILON = 1e-8
GAMMA = -0.1
ZETA = 1.1


def get_sparsity(checkpoint, suffixes, mask_fn):
  """Helper function to calculate and print sparsity from a checkpoint.

  Args:
    checkpoint: path to checkpoint.
    suffixes: possible suffixes of mask variables in the checkpoint.
    mask_fn: helper function to calculate the weight mask from a saved
      tensor.
  """
  ckpt_reader = tf.train.NewCheckpointReader(checkpoint)

  # Create a list of variable names to process.
  all_names = ckpt_reader.get_variable_to_shape_map().keys()

  # Gather all variables ending with the specified suffixes
  tensor_names = []
  for s in suffixes:
    tensor_names += [x for x in all_names if x.endswith(s)]

  sorted_list = sorted(tensor_names)
  nnz = 0.0
  total = 0.0
  for s in sorted_list:
    tensor = ckpt_reader.get_tensor(s)
    mask = mask_fn(tensor)
    nnz += np.count_nonzero(mask)
    total += mask.size
  print("{} global sparsity = {}%".format(checkpoint, 100 * (1 - nnz / total)))


def l0_mask(log_alpha, gamma=GAMMA, zeta=ZETA):
  """Helper to get weight mask for an l0-regularized tensor."""
  def sigmoid(x):
    return 1/(1+np.exp(-x))
  stretched_values = sigmoid(log_alpha) * (zeta - gamma) + gamma
  return np.clip(stretched_values, a_max=1.0, a_min=0.0)


# Specialization of 'get_sparsity' for l0-regularized models.
l0_sparsity = functools.partial(
    get_sparsity,
    suffixes=["log_alpha", "_aux"],
    mask_fn=l0_mask)


# Specialization of 'get_sparsity' for magnitude & random pruning models.
pruning_sparsity = functools.partial(
    get_sparsity,
    suffixes=["mask"],
    mask_fn=lambda x: x)


def compute_log_alpha(log_sigma2, theta, eps=EPSILON):
  """Compute the log-alpha values for tensor trained with variational dropout."""
  return log_sigma2 - np.log(np.square(theta) + eps)


def vd_sparsity(checkpoint, log_alpha_threshold, model):
  """Calculate and print global sparsity for variational dropout checkpoint.

  Args:
    checkpoint: path to checkpoint.
    log_alpha_threshold: log alpha threshold to calculate sparsity with.
    model: either 'transformer' or 'rn50'.
  """
  weight_suffix = "kernel"
  if model == "rn50":
    weight_suffix = "weights"

  ckpt_reader = tf.train.NewCheckpointReader(checkpoint)

  # Create a list of variable names to process.
  all_names = ckpt_reader.get_variable_to_shape_map().keys()

  # Gather all variables ending with the specified suffixes
  tensor_names = [x for x in all_names if x.endswith("log_sigma2")]
  tensor_names += [x for x in all_names if x.endswith("_aux")]

  sorted_list = sorted(tensor_names)
  nnz = 0.0
  total = 0.0
  for s in sorted_list:
    log_sigma2 = ckpt_reader.get_tensor(s)

    if s.endswith("log_sigma2"):
      theta_name = s.replace("log_sigma2", weight_suffix)
    else:
      theta_name = s.replace("_aux", "")
    theta = ckpt_reader.get_tensor(theta_name)
    mask = np.less(compute_log_alpha(log_sigma2, theta), log_alpha_threshold)

    nnz += np.count_nonzero(mask)
    total += mask.size
  print("{} global sparsity = {}%".format(checkpoint, 100 * (1 - nnz / total)))


def main(_):
  flags.mark_flag_as_required("checkpoint")

  if "pruning" in FLAGS.sparsity_technique:
    pruning_sparsity(FLAGS.checkpoint)
  elif FLAGS.sparsity_technique == "l0_regularization":
    l0_sparsity(FLAGS.checkpoint)
  elif FLAGS.sparsity_technique == "variational_dropout":
    vd_sparsity(FLAGS.checkpoint, FLAGS.log_alpha_threshold, FLAGS.model)
  else:
    raise ValueError("Invalid sparsity_technique argument.")


if __name__ == "__main__":
  app.run(main)
