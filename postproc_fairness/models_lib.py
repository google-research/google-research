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

"""Utilities for creating model instances."""
import json
import os
import shutil
import tempfile
import time
import tensorflow as tf
from postproc_fairness.utils import utils

EPSILON = 1e-3


def create_base_mlp_model(
    feature_name_list, sensitive_attribute, num_hidden_units_list
):
  """Create baseline keras sequential model."""
  inputs = {
      k: tf.keras.Input(shape=(1,), name=k)
      for k in feature_name_list
      if k != sensitive_attribute
  }
  x = tf.keras.layers.concatenate(tf.nest.flatten(inputs), axis=1)
  for i, num_units in enumerate(utils.csv_str_to_list(num_hidden_units_list)):
    x = tf.keras.layers.Dense(
        num_units, activation=tf.nn.relu, name=f"hidden_{i}"
    )(x)
  logits = tf.keras.layers.Dense(1, activation=None, name="logits")(x)
  outputs = tf.keras.layers.Activation(activation="sigmoid", name="outputs")(
      logits
  )
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model


def create_postproc_model(
    base_model,
    feature_name_list,
    num_hidden_units_list,
    regularizer_name=None,
    regularization_strength=0.0,
):
  """Create post-processing MLP model that uses an arbitrary pretrained model as base model."""
  inputs = {k: tf.keras.Input(shape=(1,), name=k) for k in feature_name_list}

  if base_model is not None:
    base_logits_model = tf.keras.Model(
        inputs=base_model.inputs, outputs=base_model.get_layer("logits").output
    )
    base_logits = base_logits_model(inputs)
    base_outputs = base_model(inputs)
    x = tf.keras.layers.concatenate(tf.nest.flatten(inputs), axis=1)
  else:
    # base_logits = inputs["log_probs"]
    # Get logits from probabilities, such that I recover the original
    # probabilities if I pass them through sigmoid.
    base_logits = tf.log(
        EPSILON + inputs["probs"] / (EPSILON + 1 - inputs["probs"])
    )
    base_outputs = inputs["probs"]
    x = tf.keras.layers.concatenate(
        [v for k, v in inputs.items() if k not in ["probs", "log_probs"]],
        axis=1,
    )

  for i, num_units in enumerate(utils.csv_str_to_list(num_hidden_units_list)):
    print("hidden", i, num_units)
    x = tf.keras.layers.Dense(
        num_units, activation=tf.nn.relu, name=f"pp_hidden_{i}"
    )(x)
  multiplier = tf.keras.layers.Dense(1, activation=None, name="pp_multiplier")(
      x
  )
  pp_logits = tf.keras.layers.Add(name="pp_logits")([base_logits, multiplier])
  pp_outputs = tf.keras.layers.Activation(
      activation="sigmoid", name="pp_outputs"
  )(pp_logits)
  model = tf.keras.Model(inputs=inputs, outputs=pp_outputs)

  if regularizer_name == "None":
    loss = None
  elif regularizer_name == "multiplier_l2":
    loss = tf.linalg.norm(multiplier) ** 2
  elif regularizer_name == "kl":
    # Note: this needs to be changed for multiclass to be sum along softmax
    # output axis.

    # # Take binary outputs, instead of continuous scores.
    # # This allows the pretrained model to be more flexible.
    # base_outputs = tf.where(base_outputs > 0.5, 1.0, 0.0)
    # pp_outputs = tf.where(pp_outputs > 0.5, 1.0, 0.0)
    loss = tf.reduce_mean(
        base_outputs * tf.log(base_outputs / (pp_outputs + EPSILON) + EPSILON)
        + (1 - base_outputs)
        * tf.log((1 - base_outputs) / (1 - pp_outputs + EPSILON) + EPSILON)
    )
  else:
    raise RuntimeError(f"Undefined regularizer {regularizer_name}.")
  model.add_loss(regularization_strength * loss)

  return model


def load_model_from_saved_weights(
    base_model_path,
    feature_name_list,
    num_hidden_units_list,
    sensitive_attribute,
    freeze_weights=True,
):
  """Loads a Keras model from saved weights."""
  model = create_base_mlp_model(
      feature_name_list=feature_name_list,
      num_hidden_units_list=num_hidden_units_list,
      sensitive_attribute=sensitive_attribute,
  )
  model = load_weights_from_file(model, base_model_path)
  if freeze_weights:
    model.trainable = False
  return model


def get_pretrained_model_path(config_path, num_hidden_units_list_str):
  """Gets the pretrained model path from config file."""
  with open(config_path, "r") as f:
    data = f.read()
    model_path_dict = json.loads(data)
  return model_path_dict[num_hidden_units_list_str]


def load_weights_from_file(
    model, filepath, skip_mismatch=False, by_name=False, options=None
):
  """Loads the model weights for a Keras instance that has been saved to file."""
  local_filename = filepath.split("/")[-1]
  tmp_filename = os.path.join(
      tempfile.gettempdir(), str(int(time.time())) + "_" + local_filename
  )
  shutil.copyfile(filepath, tmp_filename)
  model.load_weights(
      tmp_filename,
      skip_mismatch=skip_mismatch,
      by_name=by_name,
      options=options,
  )
  os.remove(tmp_filename)
  return model


def save_weights_to_file(
    model,
    filepath,
    save_format=None,
    options=None,
):
  """Saves a model's weights to file."""
  local_filename = filepath.split("/")[-1]
  tmp_filename = os.path.join(
      tempfile.gettempdir(), str(int(time.time())) + "_" + local_filename
  )
  model.save_weights(
      tmp_filename,
      overwrite=True,
      save_format=save_format,
      options=options,
  )
  shutil.copyfile(tmp_filename, filepath)
  os.remove(tmp_filename)
