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

"""Feature selection experiments."""

import json
import os
import pathlib
import random

from absl import app
from absl import flags
import numpy as np
from sequential_attention.experiments.datasets.dataset import get_dataset
from sequential_attention.experiments.models.mlp_lly import LiaoLattyYangModel
from sequential_attention.experiments.models.mlp_omp import OrthogonalMatchingPursuitModel
from sequential_attention.experiments.models.mlp_sa import SequentialAttentionModel
from sequential_attention.experiments.models.mlp_seql import SequentialLASSOModel
from sequential_attention.experiments.models.mlp_sparse import SparseModel
import tensorflow as tf


os.environ["TF_DETERMINISTIC_OPS"] = "1"

FLAGS = flags.FLAGS

# Experiment parameters
flags.DEFINE_integer("seed", 2023, "Random seed")
flags.DEFINE_enum(
    "data_name",
    "mnist",
    ["mnist", "fashion", "isolet", "mice", "coil", "activity"],
    "Data name",
)
flags.DEFINE_string(
    "model_dir",
    "./model_dir",
    "Checkpoint directory for feature selection model",
)

# Feature selection hyperparameters
flags.DEFINE_integer(
    "num_selected_features", 50, "Number of features to select"
)
flags.DEFINE_enum("algo", "sa", ["sa", "lly", "seql", "gl", "omp"], "Algorithm")
flags.DEFINE_integer(
    "num_inputs_to_select_per_step", 1, "Number of features to select at a time"
)

# Hyperparameters
flags.DEFINE_float(
    "val_ratio", 0.125, "How much of the training data to split for validation."
)
flags.DEFINE_list("deep_layers", "67", "Layers in MLP model")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate")
flags.DEFINE_integer("decay_steps", 250, "Decay steps")
flags.DEFINE_float("decay_rate", 1.0, "Decay rate")
flags.DEFINE_float("alpha", 0, "Leaky ReLU alpha")
flags.DEFINE_bool("enable_batch_norm", False, "Enable batch norm")
flags.DEFINE_float("group_lasso_scale", 0.01, "Group LASSO scale")

# Finer control if needed
flags.DEFINE_integer("num_epochs_select", -1, "Number of epochs to fit")
flags.DEFINE_integer("num_epochs_fit", -1, "Number of epochs to select")


ALGOS = {
    "sa": SequentialAttentionModel,
    "lly": LiaoLattyYangModel,
    "seql": SequentialLASSOModel,
    "gl": SequentialLASSOModel,
    "omp": OrthogonalMatchingPursuitModel,
}


def run_trial(
    batch_size=256,
    num_epochs_select=250,
    num_epochs_fit=250,
    learning_rate=0.0002,
    decay_steps=100,
    decay_rate=1.0,
):
  """Run a feature selection experiment with a given set of hyperparameters."""
  datasets = get_dataset(FLAGS.data_name, FLAGS.val_ratio, batch_size)
  ds_train = datasets["ds_train"]
  ds_val = datasets["ds_val"]
  ds_test = datasets["ds_test"]
  is_classification = datasets["is_classification"]
  num_classes = datasets["num_classes"]
  num_features = datasets["num_features"]
  num_train_steps_select = num_epochs_select * len(ds_train)
  loss_fn = (
      tf.keras.losses.CategoricalCrossentropy()
      if is_classification
      else tf.keras.losses.MeanAbsoluteError()
  )

  model_dir = pathlib.Path(FLAGS.model_dir)
  model_dir_select = model_dir / "select"
  model_dir_fit = model_dir / "fit"
  model_dir_select.mkdir(exist_ok=True, parents=True)
  model_dir_fit.mkdir(exist_ok=True, parents=True)

  mlp_args = {
      "layer_sequence": [int(i) for i in FLAGS.deep_layers],
      "is_classification": is_classification,
      "num_classes": num_classes,
      "learning_rate": learning_rate,
      "decay_steps": decay_steps,
      "decay_rate": decay_rate,
      "alpha": FLAGS.alpha,
      "batch_norm": FLAGS.enable_batch_norm,
  }
  fs_args = {
      "num_inputs": num_features,
      "num_inputs_to_select": FLAGS.num_selected_features,
  }
  if FLAGS.algo == "sa":
    fs_args["num_inputs_to_select_per_step"] = (
        FLAGS.num_inputs_to_select_per_step
    )
    fs_args["num_train_steps"] = num_train_steps_select
  if FLAGS.algo == "seql":
    fs_args["num_train_steps"] = num_train_steps_select
    fs_args["group_lasso_scale"] = FLAGS.group_lasso_scale
  if FLAGS.algo == "gl":
    fs_args["num_inputs_to_select_per_step"] = FLAGS.num_selected_features
    fs_args["num_train_steps"] = num_train_steps_select
    fs_args["group_lasso_scale"] = FLAGS.group_lasso_scale
  if FLAGS.algo == "omp":
    fs_args["num_train_steps"] = num_train_steps_select
  if FLAGS.algo == "lly":
    del fs_args["num_inputs_to_select"]

  ########### Feature Selection ##########
  print("Starting selecting features...")

  if FLAGS.algo in ALGOS:
    args = {**mlp_args, **fs_args}
    mlp_select = ALGOS[FLAGS.algo](**args)
    mlp_select.compile(loss=loss_fn, metrics=["accuracy"])
    mlp_select.fit(
        ds_train, validation_data=ds_val, epochs=num_epochs_select, verbose=2
    )

    ########### Get Features ##########
    if FLAGS.algo == "sa":
      selected_features = mlp_select.seqatt.selected_features
      _, selected_indices = tf.math.top_k(
          selected_features, k=FLAGS.num_selected_features
      )
      selected_indices = selected_indices.numpy()
    elif FLAGS.algo == "lly":
      x_train = datasets["x_train"]
      attention_logits = mlp_select.lly(tf.convert_to_tensor(x_train))
      _, selected_indices = tf.math.top_k(
          attention_logits, k=FLAGS.num_selected_features
      )
      selected_indices = selected_indices.numpy()
    elif FLAGS.algo in ["gl", "seql"]:
      selected_indices = (
          mlp_select.seql.selected_features_history.numpy().tolist()
      )
    elif FLAGS.algo == "omp":
      selected_indices = (
          mlp_select.omp.selected_features_history.numpy().tolist()
      )
    assert (
        len(selected_indices) == FLAGS.num_selected_features
    ), f"Selected: {selected_indices}"

  print("Finished selecting features...")

  selected_features = tf.math.reduce_sum(
      tf.one_hot(selected_indices, num_features, dtype=tf.int32), 0
  ).numpy()
  with open(model_dir_select / "selected_features.txt", "w") as fp:
    fp.write(",".join([str(i) for i in selected_indices]))
  tf.print("Selected", tf.reduce_sum(selected_features), "features")
  tf.print("Selected mask:", selected_features, summarize=-1)
  selected_features = tf.where(selected_features)[:, 0].numpy().tolist()
  selected_features = ",".join([str(i) for i in selected_features])
  print("Selected indices:", selected_features)

  selected_features = [int(i) for i in selected_features.split(",")]
  selected_features = tf.math.reduce_sum(
      tf.one_hot(selected_features, num_features, dtype=tf.float32), 0
  )

  ########### Model Training ##########

  print("Starting retraining...")

  mlp_fit = SparseModel(selected_features=selected_features, **mlp_args)
  mlp_fit.compile(loss=loss_fn, metrics=["accuracy"])
  mlp_fit.fit(
      ds_train, validation_data=ds_val, epochs=num_epochs_fit, verbose=2
  )

  print("Finished retraining...")
  ########### Evaluation ##########

  results = dict()
  results_val = mlp_fit.evaluate(ds_val, return_dict=True)
  results_test = mlp_fit.evaluate(ds_test, return_dict=True)
  results["val"] = round(results_val["accuracy"], 4)
  results["test"] = round(results_test["accuracy"], 4)

  with open(model_dir_fit / "results.json", "w") as fp:
    json.dump(results, fp)

  print(results)

  return results["val"]


def main(args):
  del args  # Not used.

  os.environ["PYTHONHASHSEED"] = str(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)

  tf.keras.backend.clear_session()

  num_epochs_select = FLAGS.num_epochs
  num_epochs_fit = FLAGS.num_epochs
  if FLAGS.num_epochs_select > 0:
    num_epochs_select = FLAGS.num_epochs_select
  if FLAGS.num_epochs_fit > 0:
    num_epochs_fit = FLAGS.num_epochs_fit
  run_trial(
      batch_size=FLAGS.batch_size,
      num_epochs_select=num_epochs_select,
      num_epochs_fit=num_epochs_fit,
      learning_rate=FLAGS.learning_rate,
      decay_steps=FLAGS.decay_steps,
      decay_rate=FLAGS.decay_rate,
  )


if __name__ == "__main__":
  app.run(main)
