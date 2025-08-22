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

"""Experimental code for "Distributino Embedding Network for Meta Learning" on OpenML data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl

# Data preparation hparams
_DATA_DIRECTORY = flags.DEFINE_string(
    "openml_data_directory", None,
    "Directory that stores the training data. This dicretory should only"
    "include training csv files, and nothing else.")
_MAX_NUM_CLASSES = flags.DEFINE_integer("max_num_classes", 2,
                                        "max number of classes across tasks")
_OPENML_TEST_ID = flags.DEFINE_integer("openml_test_id", 0,
                                       "id of openml data to be used as test.")
_MISSING_VALUE = flags.DEFINE_float("missing_value", -1.0,
                                    "missing value in real data.")

# Data generation hp arams
_NUM_FINE_TUNE = flags.DEFINE_integer("num_fine_tune", 50,
                                      "number of fine-tuning examples.")
_NUM_INPUTS = flags.DEFINE_integer("num_inputs", 25,
                                   "max number input dimensions across tasks.")
_PAD_VALUE = flags.DEFINE_integer("pad_value", -10, "value used for padding.")
_R = flags.DEFINE_integer("r", 2, "r value in embedding.")

# Training hparams
_PRETRAIN_BATCHES = flags.DEFINE_integer(
    "pretrain_batches", 10000, "number of steps to pretrain the model.")
_TUNE_EPOCHS = flags.DEFINE_integer("tune_epochs", 10,
                                    "number of fine-tuning epochs.")
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 64, "batch size for pretraining and finetuning.")

# DEN hparams
_NUM_CALIB_KEYS = flags.DEFINE_integer(
    "num_calib_keys", 10, "number of keypoints in the calibration layer.")
_HIDDEN_LAYER = flags.DEFINE_integer("hidden_layer", 2,
                                     "depth of the h, phi and psi functions.")
_DISTRIBUTION_REPRESENTATION_DIM = flags.DEFINE_integer(
    "distribution_representation_dim", 16, "width of the h function.")
_DEEPSETS_LAYER_UNITS = flags.DEFINE_integer("deepsets_layer_units", 10,
                                             "width of the phi function.")
_OUTPUT_LAYER_UNITS = flags.DEFINE_integer("output_layer_units", 8,
                                           "width of the psi function.")


#############################################################################
# Utils
#############################################################################
def load_openml_data():
  """Loads data from CNS.

  Output a dictionary of the form {name: data}. Here data is a list of numpy
  arrays with the first column being labels and the rest being features.

  Returns:
    datasets: dictionary. Each element is (name, val) pair where name is the
      dataset name and val is a list containing binary classification tasks
      within this dataset.
    files: list of files in the directory.
  """
  datasets = dict()
  files = os.listdir(_DATA_DIRECTORY.value)
  for file_name in files:
    with open(_DATA_DIRECTORY.value + file_name, "r") as ff:
      task = np.loadtxt(ff, delimiter=",", skiprows=1)
      np.random.shuffle(task)
      datasets[file_name] = [task]
  return datasets, files


def truncate_data(data, indices):
  """Truncates data using indices provided."""
  truncated_data = []
  for task in data:
    truncated_data.append(task[indices])
  return truncated_data


def compute_metrics(labels, predictions):
  """Computes metrics."""
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  res = [loss(labels, predictions).numpy()]

  metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
  for m in metrics:
    _ = m.update_state(labels, predictions)
    res.append(m.result().numpy())
  return res


def average_metrics(metrics):
  """Average metrics."""
  avg_metrics = dict()
  for name, metric in metrics.items():
    avg_metrics[name] = []
    for m in metric:
      avg_metrics[name].append({
          "mean": np.mean(np.array(m), axis=0),
          "std": np.std(np.array(m), axis=0)
      })
  return avg_metrics


def print_metrics(metrics, data_name):
  """Print metrics."""
  metrics_name = ["loss", "accuracy", "auc"]
  for i, metric in enumerate(metrics):
    for mean, std, name in zip(metric["mean"], metric["std"], metrics_name):
      print(f"[metric] task{i}_{data_name}_{name}_mean={mean}")
      print(f"[metric] task{i}_{data_name}_{name}_std={std}")


def pad_features(features, size, axis=1, pad_value=None):
  """Pad features."""
  if pad_value is None:  # Repeat columns
    num = features.shape[axis]
    repeat_indices = random.sample(range(num), size - num)
    repeat_features = tf.gather(features, repeat_indices, axis=axis)
    new_features = tf.concat([features, repeat_features], axis=axis)
  else:  # Add padding values
    paddings = [[0, 0] for _ in features.shape]
    paddings[axis] = [0, size - features.shape[axis]]
    new_features = tf.pad(
        features, tf.constant(paddings), constant_values=pad_value)
  return new_features


def get_pairwise_inputs(inputs):
  """Reform inputs to pairwise format."""
  # [BATCH_SIZE, NUM_INPUTS] --> [BATCH_SIZE, NUM_INPUTS**2, _R.value]
  num_features = inputs.shape[1]
  feature = []
  np.random.seed(seed=np.mod(round(time.time() * 1000), 2**31))
  for _ in range(_R.value):
    random_indices = np.random.choice(range(num_features), num_features**2)
    feature.append(tf.gather(inputs, random_indices, axis=1))
  pairwise_inputs = tf.stack(feature, axis=-1)
  return pairwise_inputs


def copy_keras_model(model):
  """Copy Keras model."""
  new_model = tf.keras.models.clone_model(model)
  for layer, new_layer in zip(model.layers, new_model.layers):
    weights = layer.get_weights()
    new_layer.set_weights(weights)
  return new_model


def freeze_keras_model(model):
  """Freeze part of the keras model."""
  model.trainable = True
  for layer in model.layers[::-1]:
    if "input_calibration" not in layer.name:
      layer.trainable = False  # freeze this layer


def compile_keras_model(model, init_lr=0.001):
  """Compile Keras model."""
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
      optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr))


def build_deepsets_joint_representation_model():
  """Build a pairwise joint distribution representation model."""
  # We first create the embedding model
  test_input = tf.keras.layers.Input(shape=(_NUM_INPUTS.value,))
  train_input = tf.keras.layers.Input(shape=(_NUM_INPUTS.value,))
  train_label = tf.keras.layers.Input(shape=(1,))

  # Obtain a mask variable. Output dimension [1, _NUM_INPUTS.value]
  mask = tf.ones((1, _NUM_INPUTS.value))
  one_row = tf.reshape(tf.gather(train_input, [0], axis=0), [-1])
  mask = mask * tf.cast(tf.not_equal(one_row, _PAD_VALUE.value), tf.float32)

  # Calibrate input if haven't done so
  calibrated_train_input = train_input
  calibrated_test_input = test_input
  calibration = tfl.layers.PWLCalibration(
      input_keypoints=np.linspace(0.0, 1.0, _NUM_CALIB_KEYS.value),
      units=_NUM_INPUTS.value,
      output_min=0.0,
      output_max=1.0,
      impute_missing=True,
      missing_input_value=_MISSING_VALUE.value,
      name="input_calibration")
  calibrated_train_input = calibration(train_input)
  calibrated_test_input = calibration(test_input)

  # Reshape the input to pair-wise format.
  # Output dimension [_BATCH_SIZE.value, _NUM_INPUTS.value**2, 2]
  pairwise_train_input = get_pairwise_inputs(calibrated_train_input)
  pairwise_test_input = get_pairwise_inputs(calibrated_test_input)

  # Obtain pairwise masks. Output dimesion [_NUM_INPUTS.value**2,]
  pairwise_mask = get_pairwise_inputs(mask)
  pairwise_mask = tf.reshape(tf.reduce_prod(pairwise_mask, axis=-1), [-1])

  # Obtain pairwise labels.
  # Output dimension
  # [_BATCH_SIZE.value, _NUM_INPUTS.value**2, _MAX_NUM_CLASSES.value]
  one_hot_train_label = tf.one_hot(
      tf.cast(train_label, tf.int32), _MAX_NUM_CLASSES.value)
  pairwise_train_label = tf.tile(one_hot_train_label,
                                 tf.constant([1, _NUM_INPUTS.value**2, 1]))

  # Concatenate pairwise inputs and labels.
  # Output dimension
  # [_BATCH_SIZE.value, _NUM_INPUTS.value**2, _MAX_NUM_CLASSES.value + 2]
  pairwise_train_input = tf.concat([pairwise_train_input, pairwise_train_label],
                                   axis=-1)

  # Obtain distribution representation. Output dimension
  # [_BATCH_SIZE.value, _NUM_INPUTS.value**2,
  #  _DISTRIBUTION_REPRESENTATION_DIM.value]
  batch_embedding = tf.keras.layers.Dense(
      _DISTRIBUTION_REPRESENTATION_DIM.value, activation="relu")(
          pairwise_train_input)
  for _ in range(_HIDDEN_LAYER.value - 1):
    batch_embedding = tf.keras.layers.Dense(
        _DISTRIBUTION_REPRESENTATION_DIM.value, activation="relu")(
            batch_embedding)

  # Average embeddings over the batch. Output dimension
  # [_NUM_INPUTS.value**2, _DISTRIBUTION_REPRESENTATION_DIM.value].
  mean_distribution_embedding = tf.reduce_mean(batch_embedding, axis=0)

  outputs = []
  for pairwise_input in [pairwise_test_input, pairwise_train_input]:
    # [_NUM_INPUTS.value**2, _DISTRIBUTION_REPRESENTATION_DIM.value] ->
    # [_BATCH_SIZE.value, _NUM_INPUTS.value**2,
    #  _DISTRIBUTION_REPRESENTATION_DIM.value] via repetition.
    distribution_embedding = tf.tile(
        [mean_distribution_embedding],
        tf.stack([tf.shape(pairwise_input)[0],
                  tf.constant(1),
                  tf.constant(1)]))
    # Concatenate pairwise inputs and embeddings. Output shape
    # [_BATCH_SIZE.value, _NUM_INPUTS.value**2,
    #  2 + _DISTRIBUTION_REPRESENTATION_DIM.value]
    concat_input = tf.concat([pairwise_input, distribution_embedding], axis=-1)

    # Apply a common function to each pair. Output shape
    # [_BATCH_SIZE.value, _NUM_INPUTS.value**2, _DEEPSETS_LAYER_UNITS.value]
    pairwise_output = tf.keras.layers.Dense(
        _DEEPSETS_LAYER_UNITS.value, activation="relu")(
            concat_input)
    for _ in range(_HIDDEN_LAYER.value - 1):
      pairwise_output = tf.keras.layers.Dense(
          _DEEPSETS_LAYER_UNITS.value, activation="relu")(
              pairwise_output)

    # Average pair-wise outputs across valid pairs.
    # Output shape [_BATCH_SIZE.value, _DEEPSETS_LAYER_UNITS.value]
    average_outputs = tf.tensordot(pairwise_mask, pairwise_output, [[0], [1]])
    average_outputs = average_outputs / tf.reduce_sum(pairwise_mask)

    # Use several dense layers to get the final output
    final_output = tf.keras.layers.Dense(
        _OUTPUT_LAYER_UNITS.value, activation="relu")(
            average_outputs)
    for i in range(_HIDDEN_LAYER.value - 1):
      final_output = tf.keras.layers.Dense(
          _OUTPUT_LAYER_UNITS.value, activation="relu")(
              final_output)
    outputs.append(final_output)

  test_outputs = tf.math.l2_normalize(outputs[0], axis=1)
  train_outputs = tf.math.l2_normalize(outputs[1], axis=1)
  similarity_matrix = tf.exp(
      tf.matmul(test_outputs, tf.transpose(train_outputs)))

  similarity_list = []
  for i in range(_MAX_NUM_CLASSES.value):
    mask = tf.cast(tf.squeeze(tf.equal(train_label, i)), tf.float32)
    similarity_list.append(similarity_matrix * mask)

  similarity = [
      tf.reduce_mean(s, axis=1, keepdims=True) for s in similarity_list
  ]
  sum_similarity = tf.reduce_sum(
      tf.concat(similarity, axis=1), axis=1, keepdims=True)
  final_output = [similarity / sum_similarity for similarity in similarity_list]
  final_output = tf.concat(final_output, axis=1)

  keras_model = tf.keras.models.Model(
      inputs=[test_input, train_input, train_label], outputs=final_output)
  compile_keras_model(keras_model)
  return keras_model


#############################################################################
# Training and evaluation
#############################################################################
def prepare_training_examples(data):
  """Prepare training examples."""
  features = tf.convert_to_tensor(data[:, 1:], dtype=tf.float32)
  labels = tf.convert_to_tensor(data[:, 0], dtype=tf.float32)

  # Pad features
  features = pad_features(
      features, _NUM_INPUTS.value, pad_value=_PAD_VALUE.value)

  half = int(features.shape[0] / 2)
  inputs = [features[half:], features[:half], labels[:half]]
  true_labels = labels[half:]

  return inputs, true_labels


def prepare_testing_examples(data, test_data):
  """Prepare testing examples."""
  features = tf.convert_to_tensor(data[:, 1:], dtype=tf.float32)
  labels = tf.convert_to_tensor(data[:, 0], dtype=tf.float32)
  test_features = tf.convert_to_tensor(test_data[:, 1:], dtype=tf.float32)
  test_labels = tf.convert_to_tensor(test_data[:, 0], dtype=tf.float32)

  # Pad features
  features = pad_features(
      features, _NUM_INPUTS.value, pad_value=_PAD_VALUE.value)
  test_features = pad_features(
      test_features, _NUM_INPUTS.value, pad_value=_PAD_VALUE.value)

  test_inputs = [test_features, features, labels]
  return test_inputs, test_labels


def train_model(models,
                data,
                batch_size,
                epochs,
                init_lr=0.001,
                compile_model=False):
  """Train model."""
  inputs, labels = prepare_training_examples(data)

  if compile_model:
    compile_keras_model(models, init_lr)
  model = models

  model.fit(inputs, labels, batch_size=batch_size, epochs=epochs, verbose=0)


def fine_tune_model(pretrain_models, fine_tune_task, tune_epochs, init_lr):
  """Fine-tune pretrained model."""
  # Make a copy of the pretrained model
  models = copy_keras_model(pretrain_models)
  freeze_keras_model(models)

  train_model(
      models,
      fine_tune_task,
      min(_BATCH_SIZE.value, _NUM_FINE_TUNE.value),
      tune_epochs,
      init_lr=init_lr,
      compile_model=True)
  return models


def evaluate_model(models, data, eval_data):
  """Evaluate model."""
  inputs, labels = prepare_testing_examples(data, eval_data)

  predictions = models(inputs)
  metrics = compute_metrics(labels, predictions)
  return np.array(metrics)


def pretrain_model(models, datasets, batch_size):
  """Pretrain model."""
  batch = 0  # Initialize batch counter
  while batch < _PRETRAIN_BATCHES.value:
    # Randomly select a task
    data_id = random.sample(range(len(datasets)), 1)[0]
    data = datasets[data_id]
    task_id = random.sample(range(len(data)), 1)[0]
    task = data[task_id]
    # Randomly select a batch
    batch_task = task[random.sample(range(task.shape[0]), batch_size)]
    # Train the model on this batch
    train_model(models, batch_task, batch_size, 1, init_lr=0.001)
    batch += 1


def run_simulation(pretrain_data, fine_tune_data, test_data, test_tasks):
  """Run simulation."""
  metrics = {"test": [list() for _ in range(len(test_data))]}

  # Pretraining
  pretrain_models = build_deepsets_joint_representation_model()
  pretrain_model(pretrain_models, pretrain_data, _BATCH_SIZE.value)

  for row in test_tasks:
    col_indices = [0] + [ind + 1 for ind in row[1:]]

    # Fine-tuning/direct training
    tune_task = fine_tune_data[row[0]][:, col_indices]
    models = fine_tune_model(pretrain_models, tune_task, _TUNE_EPOCHS.value,
                             0.001)
    # Evaluation
    metrics["test"][row[0]].append(
        evaluate_model(models, tune_task, test_data[row[0]][:, col_indices]))

  # Print metrics
  avg_metrics = average_metrics(metrics)
  print_metrics(avg_metrics["test"], "test")


def main(_):
  openml_datasets, openml_data_names = load_openml_data()

  # Prepare datasets
  target_names = openml_data_names[_OPENML_TEST_ID.value]
  target_data = openml_datasets[target_names]

  pretrain_data = [
      val for key, val in openml_datasets.items() if key not in target_names
  ]
  fine_tune_data = truncate_data(target_data, range(_NUM_FINE_TUNE.value))

  test_range = range(_NUM_FINE_TUNE.value, target_data[0].shape[0])
  num_features = target_data[0].shape[1] - 1
  test_tasks = [[task_id] + list(range(num_features))
                for task_id in range(len(target_data))]
  test_data = truncate_data(target_data, test_range)

  run_simulation(pretrain_data, fine_tune_data, test_data, test_tasks)


if __name__ == "__main__":
  app.run(main)
