# coding=utf-8
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

# Lint as: python3
"""Main model trainer from which a number of robust-learning models can be trained.

Currently we support the following robust-learning approaches:
  - robust_learning: proposed adversarial re-weighting robust learning approach
  - baseline: a simple baseline model, which implements a fully connected
    feedforward network with standard ERM objective.
  - inverse_propensity_weighting: a naive re-weighting baseline using
    inverse_propensity_weighting.
  - adversarial_subgroup_reweighting: (modified) version of our proposed
    approach wherein adversary has access only to protected features,
    and learner has access to all features.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from group_agnostic_fairness import adversarial_subgroup_reweighting_model
from group_agnostic_fairness import baseline_model
from group_agnostic_fairness import ips_reweighting_model
from group_agnostic_fairness import robust_learning_model
from group_agnostic_fairness.data_utils.compas_input import CompasInput
from group_agnostic_fairness.data_utils.law_school_input import LawSchoolInput
from group_agnostic_fairness.data_utils.uci_adult_input import UCIAdultInput
from group_agnostic_fairness.fairness_metrics import RobustFairnessMetrics

FLAGS = flags.FLAGS

MODEL_KEYS = ["baseline",
              "inverse_propensity_weighting",
              "robust_learning",
              "adversarial_subgroup_reweighting"]

# pylint: disable=line-too-long
# Flags for creating and running a model
flags.DEFINE_string("model_name", "robust_learning", "Name of the model to run")
flags.DEFINE_string("base_dir", "/tmp", "Base directory for output.")
flags.DEFINE_string("model_dir", None, "Model directory for output.")
flags.DEFINE_string("output_file_name", "results.txt",
                    "Output file where to write metrics to.")
flags.DEFINE_string("print_dir", None, "directory for tf.print output_stream.")

# Flags for training and evaluation
flags.DEFINE_integer("train_steps", 20, "Number of training steps.")
flags.DEFINE_integer("test_steps", 5, "Number of evaluation steps.")
flags.DEFINE_integer("min_eval_frequency", 100,
                     "How often (steps) to run evaluation.")

# Flags for loading dataset
flags.DEFINE_string("dataset_base_dir", "./group_agnostic_fairness/data/toy_data", "(string) path to dataset directory")
flags.DEFINE_string("dataset", "uci_adult", "Name of the dataset to run")
flags.DEFINE_multi_string("train_file", ["./group_agnostic_fairness/data/toy_data/train.csv"], "List of (string) path(s) to training file(s).")
flags.DEFINE_multi_string("test_file", ["./group_agnostic_fairness/data/toy_data/test.csv"], "List of (string) path(s) to evaluation file(s).")

# # If the model has an adversary, the features for adversary are constructed
# # in the corresponding custom estimator implementation by filtering feature_columns passed to the learner.
flags.DEFINE_bool("include_sensitive_columns", False,
                  "Set the flag to include protected features in the feature_columns of the learner.")

# Flags for setting common model parameters for all approaches
flags.DEFINE_multi_integer("primary_hidden_units", [64, 32],
                           "Hidden layer sizes of main learner.")
flags.DEFINE_integer("embedding_dimension", 32,
                     "Embedding size; if 0, use one hot.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_float("primary_learning_rate", 0.001,
                   "learning rate for main learner.")
flags.DEFINE_string("optimizer", "Adagrad", "Name of the optimizer to use.")
flags.DEFINE_string("activation", "relu", "Name of the activation to use.")

# # Flags for approaches that have an adversary
# # Currently only for ''robust_learning'' Model and ''adversarial_subgroup_reweighting'' Model.
flags.DEFINE_multi_integer("adversary_hidden_units", [32],
                           "Hidden layer sizes of adversary.")
flags.DEFINE_float("adversary_learning_rate", 0.001,
                   "learning rate for adversary.")

# # Flags for robust_learning model
flags.DEFINE_string("adversary_loss_type", "ce_loss",
                    "Type of adversary loss function to be used. Takes values in [``ce_loss'',''hinge_loss'']. ``ce loss`` stands for cross-entropy loss")
flags.DEFINE_bool("upweight_positive_instance_only", False,
                  "Set the flag to weight only positive examples if in adversarial loss. Only used when adversary_loss_type parameter of robust_learning model is set to hinge_loss")
flags.DEFINE_bool(
    "adversary_include_label", True,
    "Set the flag to add label as a feature to adversary in the model.")
flags.DEFINE_integer(
    "pretrain_steps", 250,
    "Number of steps to train primary before alternating with adversary.")

# # Flags for inverse_propensity_weighting Model
flags.DEFINE_string("reweighting_type", "IPS_without_label",
                    "Type of reweighting to be performed. Takes values in [''IPS_with_label'', ''IPS_without_label'']")

TENSORFLOW_BOARD_BINARY = "learning/brain/tensorboard/tensorboard.sh"
tf.logging.set_verbosity(tf.logging.INFO)


def get_estimator(model_dir,
                  model_name,
                  feature_columns,
                  protected_groups,
                  label_column_name):
  """Instantiates and returns a model estimator.

  Args:
    model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into an estimator
        to continue training a previously saved model.
    model_name: (string) name of the estimator to instantiate.
    feature_columns: list of feature_columns.
    protected_groups: list of protected_groups. For example, ["sex","race"].
    label_column_name: (string) name of the target variable.

  Returns:
    An instance of `tf.estimator.Estimator'.

  Raises:
    ValueError: if estimator for model_name is not implemented.
    ValueError: if activation function is not implemented.
  """
  # Defines activation function to be used for the model. Append activation
  # functions that we want to use here.
  if FLAGS.activation == "relu":
    activation_fn = tf.nn.relu
  elif FLAGS.activation == "linear":
    activation_fn = tf.nn.linear
  else:
    raise ValueError("Activation {} is not supported.".format(FLAGS.activation))

  kwargs = {
      "feature_columns": feature_columns,
      "label_column_name": label_column_name,
      "config": tf.estimator.RunConfig(
          model_dir=model_dir,
          save_checkpoints_steps=FLAGS.min_eval_frequency),
      "model_dir": model_dir,
      "batch_size": FLAGS.batch_size,
      "activation": activation_fn,
      "optimizer": FLAGS.optimizer
  }

  # Instantiates estimators to be used. Append new estimators that we want to use here.
  if model_name == "baseline":
    estimator = baseline_model.get_estimator(
        hidden_units=FLAGS.primary_hidden_units,
        learning_rate=FLAGS.primary_learning_rate,
        **kwargs)
  elif model_name == "inverse_propensity_weighting":
    estimator = ips_reweighting_model.get_estimator(
        reweighting_type=FLAGS.reweighting_type,
        hidden_units=FLAGS.primary_hidden_units,
        learning_rate=FLAGS.primary_learning_rate,
        **kwargs)
  elif model_name == "robust_learning":
    estimator = robust_learning_model.get_estimator(
        adversary_loss_type=FLAGS.adversary_loss_type,
        adversary_include_label=FLAGS.adversary_include_label,
        upweight_positive_instance_only=FLAGS.upweight_positive_instance_only,
        pretrain_steps=FLAGS.pretrain_steps,
        primary_hidden_units=FLAGS.primary_hidden_units,
        adversary_hidden_units=FLAGS.adversary_hidden_units,
        primary_learning_rate=FLAGS.primary_learning_rate,
        adversary_learning_rate=FLAGS.adversary_learning_rate,
        **kwargs)
  elif model_name == "adversarial_subgroup_reweighting":
    estimator = adversarial_subgroup_reweighting_model.get_estimator(
        protected_column_names=protected_groups,
        pretrain_steps=FLAGS.pretrain_steps,
        primary_hidden_units=FLAGS.primary_hidden_units,
        adversary_hidden_units=FLAGS.adversary_hidden_units,
        primary_learning_rate=FLAGS.primary_learning_rate,
        adversary_learning_rate=FLAGS.adversary_learning_rate,
        **kwargs)
  else:
    raise ValueError("Model {} is not implemented.".format(model_name))
  return estimator


def write_to_output_file(eval_results, output_file_path):
  """Serializes eval_results dictionary and writes json to directory.

  Args:
    eval_results: dictionary of evaluation metrics results.
    output_file_path: (string) filepath to write output to.
  """
  to_save = {}
  for key, val in eval_results.items():
    if isinstance(val, np.ndarray):
      to_save[key] = val.tolist()
    else:
      to_save[key] = float(val)
  tf.logging.info("Evaluation metrics saved:{}".format(to_save))
  with tf.gfile.Open(output_file_path, mode="w") as output_file:
    print("writing output to:{}".format(output_file_path))
    output_file.write(json.dumps(to_save))
    output_file.close()


def _initialize_model_dir():
  """Initializes model_dir. Deletes the model directory folder."""
  if FLAGS.model_dir:
    model_dir = FLAGS.model_dir
    model_name = FLAGS.model_name
  else:
    base_dir = FLAGS.base_dir
    model_name = FLAGS.model_name
    if model_name == "inverse_propensity_weighting":
      setting_name = "{}/{}/{}_{}_{}".format(FLAGS.dataset, model_name,
                                             FLAGS.reweighting_type,
                                             str(FLAGS.batch_size),
                                             str(FLAGS.primary_learning_rate))
    elif model_name == "robust_learning":
      setting_name = "{}/{}/{}_{}_{}_{}_{}".format(
          FLAGS.dataset, model_name, FLAGS.adversary_loss_type,
          FLAGS.adversary_include_label, str(FLAGS.batch_size),
          str(FLAGS.primary_learning_rate), str(FLAGS.adversary_learning_rate))
    elif model_name == "adversarial_subgroup_reweighting":
      setting_name = "{}/{}/{}_{}_{}".format(
          FLAGS.dataset, model_name, str(FLAGS.batch_size),
          str(FLAGS.primary_learning_rate), str(FLAGS.adversary_learning_rate))
    elif model_name == "baseline":
      setting_name = "{}/{}/{}_{}".format(
          FLAGS.dataset, model_name, str(FLAGS.batch_size),
          str(FLAGS.primary_learning_rate))
    else:
      raise ValueError("Model {} is not implemented.".format(model_name))
    model_dir = os.path.join(base_dir, setting_name)

  if tf.gfile.Exists(model_dir):
    tf.gfile.DeleteRecursively(model_dir)

  # Creates a printing directory. This argument is passed to the "output_steam" option of tf.print".
  # # Each tensorflow variable with a corresponding tf.print op will be saved in a seprate file in the print_dir directory.
  # # If print_dir set to None, no variables are saved to directory.
  if (FLAGS.print_dir is not None and FLAGS.model_name == "robust_learning"):
    print_dir = FLAGS.print_dir
    if tf.gfile.Exists(print_dir):
      tf.gfile.DeleteRecursively(print_dir)
    tf.gfile.MakeDirs(print_dir)
  else:
    print_dir = None

  return model_dir, model_name, print_dir


def run_model():
  """Instantiate and run model.

  Raises:
    ValueError: if model_name is not implemented.
    ValueError: if dataset is not implemented.
  """
  if FLAGS.model_name not in MODEL_KEYS:
    raise ValueError("Model {} is not implemented.".format(FLAGS.model_name))
  else:
    model_dir, model_name, print_dir = _initialize_model_dir()

  tf.logging.info(
      "Creating experiment, storing model files in {}".format(model_dir))

  # Instantiates dataset and gets input_fn
  if FLAGS.dataset == "law_school":
    load_dataset = LawSchoolInput(dataset_base_dir=FLAGS.dataset_base_dir,
                                  train_file=FLAGS.train_file,
                                  test_file=FLAGS.test_file)
  elif FLAGS.dataset == "compas":
    load_dataset = CompasInput(
        dataset_base_dir=FLAGS.dataset_base_dir,
        train_file=FLAGS.train_file,
        test_file=FLAGS.test_file)
  elif FLAGS.dataset == "uci_adult":
    load_dataset = UCIAdultInput(
        dataset_base_dir=FLAGS.dataset_base_dir,
        train_file=FLAGS.train_file,
        test_file=FLAGS.test_file)
  else:
    raise ValueError("Input_fn for {} dataset is not implemented.".format(
        FLAGS.dataset))

  train_input_fn = load_dataset.get_input_fn(
      mode=tf.estimator.ModeKeys.TRAIN, batch_size=FLAGS.batch_size)
  test_input_fn = load_dataset.get_input_fn(
      mode=tf.estimator.ModeKeys.EVAL, batch_size=FLAGS.batch_size)

  feature_columns, _, protected_groups, label_column_name = (
      load_dataset.get_feature_columns(
          embedding_dimension=FLAGS.embedding_dimension,
          include_sensitive_columns=FLAGS.include_sensitive_columns))

  # Constructs a int list enumerating the number of subgroups in the dataset.
  # # For example, if the dataset has two (binary) protected_groups. The dataset has 2^2 = 4 subgroups, which we enumerate as [0, 1, 2, 3].
  # # If the  dataset has two protected features ["race","sex"] that are cast as binary features race=["White"(0), "Black"(1)], and sex=["Male"(0), "Female"(1)].
  # # We call their catesian product ["White Male" (00), "White Female" (01), "Black Male"(10), "Black Female"(11)] as subgroups  which are enumerated as [0, 1, 2, 3].
  subgroups = np.arange(len(protected_groups))

  # Instantiates tf.estimator.Estimator object
  estimator = get_estimator(
      model_dir,
      model_name,
      feature_columns=feature_columns,
      protected_groups=protected_groups,
      label_column_name=label_column_name)

  # Adds additional fairness metrics
  fairness_metrics = RobustFairnessMetrics(
      label_column_name=label_column_name,
      protected_groups=protected_groups,
      subgroups=subgroups,
      print_dir=print_dir)
  eval_metrics_fn = fairness_metrics.create_fairness_metrics_fn()
  estimator = tf.estimator.add_metrics(estimator, eval_metrics_fn)

  # Creates training and evaluation specifications
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=FLAGS.train_steps)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=test_input_fn, steps=FLAGS.test_steps)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  tf.logging.info("Training completed.")

  eval_results = estimator.evaluate(
      input_fn=test_input_fn, steps=FLAGS.test_steps)

  eval_results_path = os.path.join(model_dir, FLAGS.output_file_name)
  write_to_output_file(eval_results, eval_results_path)


def main(_):
  run_model()


if __name__ == "__main__":
  app.run(main)
