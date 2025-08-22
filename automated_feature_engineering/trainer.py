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

"""Automated feature selection and engineering."""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import bq_data
import data_loader
import feature_engineering
import feature_selection
from google import auth
from google.cloud import bigquery
import tensorflow as tf
import utils

# Needed to make GPU training deterministic
# (per https://github.com/NVIDIA/framework-determinism/blob/
# master/doc/tensorflow.md)
# os.environ["TF_DETERMINISTIC_OPS"] = "1"


FLAGS = flags.FLAGS

# Experiment parameters
_SEED = flags.DEFINE_integer("seed", 21, "Random seed")
_DATA_NAME = flags.DEFINE_string("data_name", "isolet", "Data name")

# Number of features amplification for discovery model.
_N_FEATURES_AMPLIFIER = 3
_USE_SOFTMAX_MASK = True
_LOGGING_STEPS = 300
_MIN_NUM_SELECTED_FEATURES = 30
_REQUIRED_FLAGS = ["target", "project_id", "train_table_name", "dataset_name"]

_NUM_SELECTED_FEATURES = flags.DEFINE_integer(
    "num_selected_features", None, "Number of features for feature selection"
)

_MODEL_TYPE = flags.DEFINE_string(
    "model_type",
    "discovery",
    "Model type can be selection or discovery.",
)
_TASK_TYPE = flags.DEFINE_string(
    "task_type",
    "classification",
    "Task type can be classification or regression.",
)
_LOGGING_FILENAME = flags.DEFINE_string(
    "logging_filename",
    "features.json",
    "Name of the file used for logging discovered or selected features.",
)

_TARGET = flags.DEFINE_string(
    "target", None, "Name for the training target feature.", required=False
)

_NUM_MLP_LAYERS = flags.DEFINE_integer(
    "num_mlp_layers",
    2,
    "Number of MLP layers in MLP model",
)

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 2048, "Batch size")
_FEATURE_DIM = flags.DEFINE_integer("feature_dim", None, "Feature dimension")

_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
_NUM_STEPS = flags.DEFINE_integer("num_steps", 50, "Number of training steps")
_DECAY_STEPS = flags.DEFINE_integer("decay_steps", 500, "Decay steps")
_DECAY_RATE = flags.DEFINE_float("decay_rate", 0.5, "Decay rate")
_DATA_BUFFER_SIZE = flags.DEFINE_integer(
    "data_buffer_size", 4096, "Dataset buffer size."
)
_BATCH_BUFFER_SIZE = flags.DEFINE_integer(
    "batch_buffer_size", 32, "Number of batches held in shuffling buffer."
)
_PROJECT_ID = flags.DEFINE_string(
    "project_id", None, "The BigQuery project ID.", required=False
)
_DATASET_NAME = flags.DEFINE_string(
    "dataset_name",
    None,
    "BigQuery dataset name for train and test.",
    required=False,
)
_TRAIN_TABLE_NAME = flags.DEFINE_string(
    "train_table_name",
    None,
    "Table name of the training dataset.",
    required=False,
)
_TEST_TABLE_NAME = flags.DEFINE_string(
    "test_table_name", None, "Table name of the test dataset."
)
_CONFIG = flags.DEFINE_string(
    "config", None, "Configuration string for running pipeline from container."
)
_UPLOAD_FEATURES_TO_BQ = flags.DEFINE_bool(
    "upload_features_to_bq", True, "Whether to upload features to BQ table."
)
_GCS_OUTPUT_PATH = flags.DEFINE_string(
    "gcs_output_path", None, "GCS output path."
)


def _parse_config_string():
  """Parses config string when running pipeline from container."""
  if _CONFIG.value is not None:
    container_config = json.loads(_CONFIG.value)
    for flag in _REQUIRED_FLAGS:
      if flag not in container_config:
        raise ValueError(f"Required flag {flag} not found under --config.")

    for flag, flag_val in container_config.items():
      setattr(FLAGS, flag, flag_val)


def check_and_set_flags(num_features, n_feature_threshold = 20):
  """Sets certain optional flag values if not specified by the user."""
  # Past experiments have shown that the performance is not very sensitive to
  # these parameter settings, as long as they are within a reasonable range.
  # I.e. the model learning adjusts for the difference in these parameters.

  if _FEATURE_DIM.value is None:
    FLAGS.feature_dim = 64 if num_features <= n_feature_threshold else 128

  if _NUM_SELECTED_FEATURES.value is None:
    FLAGS.num_selected_features = max(
        _MIN_NUM_SELECTED_FEATURES, num_features // 10
    )


def process_dataset(
    bq_client, bq_info, training = True
):
  """Retrieves and preprocesses dataset from BQ."""

  # Use all batches since some datasets are small.
  drop_remainder = False
  dataset, table_metadata = bq_data.get_data_from_bq_with_bq_info(
      bq_client,
      bq_info,
      batch_size=_BATCH_SIZE.value,
      drop_remainder=drop_remainder,
  )
  # Should not count label
  num_features = len(table_metadata) - 1

  if _TASK_TYPE.value == "classification":
    # The +1 here is for the out-of-vocabulary bin created during category
    # lookup.
    num_classes = (
        table_metadata.get_metadata_by_name(_TARGET.value).cardinality + 1
    )
  else:
    num_classes = None

  cat_transform_fn, cat_features, numerical_features = (
      bq_data.make_categorical_transform_fn(
          table_metadata, target_key=_TARGET.value, task_type=_TASK_TYPE.value
      )
  )
  dataset = dataset.map(cat_transform_fn, num_parallel_calls=tf.data.AUTOTUNE)

  # dataset is already batched.
  if training:
    # Cache Dataset on disk.
    filename = tempfile.mkdtemp()
    # This gives a false warning: b/194670791
    dataset = dataset.cache(filename)
    dataset = dataset.shuffle(
        buffer_size=_BATCH_BUFFER_SIZE.value,
        seed=_SEED.value,
        reshuffle_each_iteration=True,
    )
    dataset = dataset.repeat()

  return (
      dataset.prefetch(tf.data.AUTOTUNE),
      num_features,
      cat_features,
      numerical_features,
      num_classes,
  )


def load_data_bq(
    bq_client,
):
  """Loads and returns training and test datasets from BQ."""

  train_bq_info = bq_data.BQInfo(
      _PROJECT_ID.value, _DATASET_NAME.value, _TRAIN_TABLE_NAME.value
  )
  if _TEST_TABLE_NAME.value:
    test_bq_info = bq_data.BQInfo(
        _PROJECT_ID.value, _DATASET_NAME.value, _TEST_TABLE_NAME.value
    )
  else:
    test_bq_info = train_bq_info
    logging.info("Using training data for evaluation.")

  train_dataset, num_features, cat_features, numerical_features, num_classes = (
      process_dataset(bq_client, train_bq_info, training=True)
  )
  test_dataset = process_dataset(bq_client, test_bq_info, training=False)[0]
  # TODO(yihed): obtain num_features and num_classes.

  return (
      train_dataset,
      test_dataset,
      num_features,
      cat_features,
      numerical_features,
      num_classes,
  )


def load_data():
  """Loads datasets for training.

  Returns:
    Datasets for training.

  Raises:
    ValueError: if specified dataset is not supported.
  """
  if _DATA_NAME.value == "isolet":
    (x_train, x_test, y_train, y_test, _, num_classes, _) = (
        data_loader.load_isolet()
    )
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
  else:
    raise ValueError(f"Dataset {_DATA_NAME.value} not supported.")

  ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  ds_train = ds_train.shuffle(
      buffer_size=_DATA_BUFFER_SIZE.value,
      seed=_SEED.value,
      reshuffle_each_iteration=True,
  )
  ds_train = ds_train.repeat(_NUM_STEPS.value)
  ds_train = ds_train.batch(_BATCH_SIZE.value, drop_remainder=True)
  ds_test = ds_test.batch(_BATCH_SIZE.value, drop_remainder=False)
  ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
  return (ds_train, ds_test, x_train.shape[-1], num_classes)


def get_predictor():
  """Generates model predictor head."""
  mlp_sequence = [
      tf.keras.layers.Dense(_FEATURE_DIM.value, activation="relu")
      for _ in range(_NUM_MLP_LAYERS.value)
  ]
  mlp_model = tf.keras.Sequential(mlp_sequence)
  return mlp_model


def define_discovery_model(
    num_features, num_cat_features = 0
):
  """Define model for feature discovery.

  Args:
    num_features: total number of features.
    num_cat_features: number of selected features.

  Returns:
    Model for feature discovery.
  """
  discovery_model = feature_engineering.FeatureDiscoveryModel(
      num_features,
      _NUM_SELECTED_FEATURES.value,
      _FEATURE_DIM.value,
      _NUM_MLP_LAYERS.value,
      n_temporal_features=0,
      num_cat_features=num_cat_features,
  )
  return discovery_model


def define_feature_selector(
    num_features,
    num_selected_features,
    num_feature_scaler = None,
):
  """Define module for feature selection."""
  feature_selector = feature_selection.FeatureSelectionSparseMasks(
      num_features=num_features,
      num_selected_features=(
          num_selected_features * _N_FEATURES_AMPLIFIER
          if _MODEL_TYPE.value == "discovery"
          else num_selected_features
      ),
      num_feature_scaler=num_feature_scaler,
      use_softmax_mask=_USE_SOFTMAX_MASK,
  )
  return feature_selector


def eval_on_prediction(
    data, prediction_model
):
  """Evaluates the trained model."""
  predictions = []
  labels = []
  for batch in data:
    batch_features = []
    batch_features.extend(batch[:-1])
    labels.append(batch[-1])

    pred = prediction_model(batch_features, training=False)[0]
    predictions.append(pred)

  predictions = tf.concat(predictions, axis=0)
  return predictions, labels


def train(
    ds_train,
    ds_test,
    is_classification,
    num_features,
    cat_features,
    numerical_features,
    num_classes,
    bq_client,
):
  """Trains model for the user-supplied task type."""

  # TODO(yihed): account for embedding dimension in this shape.
  input_features = tf.keras.Input(shape=(num_features,))
  embed_idx_input = tf.keras.Input(shape=(len(cat_features),))
  # TODO(yihed): update this definition.
  num_feature_scaler = None

  num_selected_features = min(_NUM_SELECTED_FEATURES.value, num_features)

  feature_selector = define_feature_selector(
      num_features, num_selected_features, num_feature_scaler
  )
  selected_features = feature_selector(input_features)
  if _MODEL_TYPE.value == "discovery":
    discovery_model = define_discovery_model(num_features, len(cat_features))
    representation, _ = discovery_model(
        selected_features, idx_inputs=embed_idx_input
    )
  else:
    discovery_model = None
    representation = selected_features

  dense_model = get_predictor()
  latents = dense_model(representation)

  if is_classification:
    predictor = tf.keras.layers.Dense(num_classes)
  else:
    predictor = tf.keras.layers.Dense(1)

  prediction = predictor(latents)
  prediction_model = tf.keras.Model(
      inputs=[input_features, embed_idx_input],
      outputs=prediction,
      name="prediction_model",
  )
  lr = tf.keras.optimizers.schedules.ExponentialDecay(
      _LEARNING_RATE.value,
      decay_steps=_DECAY_STEPS.value,
      decay_rate=_DECAY_RATE.value,
      staircase=False,
  )
  optimizer = tf.keras.optimizers.Adam(lr)
  if is_classification:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  else:
    loss_fn = tf.keras.losses.MeanAbsoluteError()

  do_training(
      ds_train,
      prediction_model=prediction_model,
      discovery_model=discovery_model,
      loss_fn=loss_fn,
      optimizer=optimizer,
      cat_features=cat_features,
      numerical_features=numerical_features,
      num_feature_scaler=num_feature_scaler,
  )

  # TODO(yihed): return string of selected features when model_type = selection.
  if discovery_model:
    infer_and_upload_features(
        prediction_model,
        discovery_model,
        ds_test,
        bq_client,
        cat_features,
        numerical_features,
    )


def _upload_transforms_to_gcs(transforms):
  """Uploads discovered transforms or selected features to GCS."""
  if _GCS_OUTPUT_PATH.value:
    gcs_feature_path = os.path.join(
        _GCS_OUTPUT_PATH.value, _LOGGING_FILENAME.value
    )
    with tf.io.gfile.GFile(gcs_feature_path, "w") as f:
      if isinstance(transforms, dict):
        transforms = {
            key.decode("utf-8"): str(value) for key, value in transforms.items()
        }
      f.write(json.dumps(transforms))
  else:
    logging.info("Not logging features, as no GCS output path is specified.")


def do_training(
    ds_train,
    prediction_model,
    discovery_model,
    loss_fn,
    optimizer,
    cat_features,
    numerical_features,
    num_feature_scaler = None,
):
  """Performs training.

  Args:
    ds_train: dataset for training.
    prediction_model: model for training.
    discovery_model: module for feature discovery, used during feature
      inference.
    loss_fn: Loss function for training.
    optimizer: Optimizer for training.
    cat_features: categorical feature names.
    numerical_features: numerical feature names.
    num_feature_scaler: Object used for gradually scaling number of features.
  """

  for step, step_data in enumerate(ds_train):
    if step >= _NUM_STEPS.value:
      break
    x = step_data[bq_data.X_KEY]
    cat_embed_idx = step_data[bq_data.EMBED_IDX_KEY]
    labels = step_data[bq_data.TARGET_KEY]

    with tf.GradientTape() as tape:

      logits = prediction_model([x, cat_embed_idx], training=True)
      loss = loss_fn(labels, logits)
      if step % _LOGGING_STEPS == 0:
        logging.info("step %d loss %f", step, loss)
        if discovery_model:
          transforms, _, _ = feature_engineering.recover_transforms(
              discovery_model,
              cat_features=cat_features,
              numerical_features=numerical_features,
          )
          _upload_transforms_to_gcs(transforms)

    grads = tape.gradient(loss, prediction_model.trainable_variables)

    optimizer.apply_gradients(zip(grads, prediction_model.trainable_variables))
    if num_feature_scaler:
      num_feature_scaler.add_step()


def infer_and_upload_features(
    model,
    discovery_model,
    dataset,
    bq_client,
    cat_features,
    numerical_features,
):
  """Infers discovered features on the given dataset."""
  transforms, ranked_feature_names, feature_ranking_idx = (
      feature_engineering.recover_transforms(
          discovery_model,
          cat_features=cat_features,
          numerical_features=numerical_features,
      )
  )
  _upload_transforms_to_gcs(transforms)
  logging.info(
      "Discovered feature transforms ordered by importance: %s", transforms
  )
  feature_table_name = utils.infer_and_upload_discovered_features(
      dataset,
      model,
      bq_client,
      _PROJECT_ID.value,
      _DATASET_NAME.value,
      _TRAIN_TABLE_NAME.value,
      feature_names=ranked_feature_names,
      feature_ranking=feature_ranking_idx,
      cat_features=cat_features,
      numerical_features=numerical_features,
      upload_to_bq=_UPLOAD_FEATURES_TO_BQ.value,
  )
  logging.info("Feature table name: %s", feature_table_name)

  return feature_table_name, transforms


def main(args):
  del args
  _parse_config_string()
  logging.info("Flags: %s", FLAGS.flag_values_dict())
  credentials, _ = auth.default()
  bq_client = bigquery.Client(
      project=_PROJECT_ID.value, credentials=credentials
  )
  (
      ds_train,
      ds_test,
      num_features,
      cat_features,
      numerical_features,
      num_classes,
  ) = load_data_bq(bq_client)
  check_and_set_flags(num_features)
  is_classification = _TASK_TYPE.value == "classification"
  train(
      ds_train,
      ds_test,
      is_classification,
      num_features,
      cat_features,
      numerical_features,
      num_classes,
      bq_client,
  )


if __name__ == "__main__":
  app.run(main)
