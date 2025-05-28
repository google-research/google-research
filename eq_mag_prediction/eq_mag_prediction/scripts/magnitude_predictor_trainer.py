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

r"""Train and evaluate the magnitude prediction model.

Example of execution line:
python3 path/to/magnitude_predictor_trainer.py
  --gin_config='path/to/gin/config.gin'
  --output_dir='path/fo/saving/the/model/and/results'
  --gin_bindings='custom.gin.bindings=value'
"""

import os
from typing import Callable, Optional, Sequence, Union

from absl import app
from absl import flags
import gin
import tensorflow as tf

from eq_mag_prediction.forecasting import external_configurations
from eq_mag_prediction.forecasting import head_models
from eq_mag_prediction.forecasting import metrics
from eq_mag_prediction.forecasting import one_region_model
from eq_mag_prediction.forecasting import training_examples
from eq_mag_prediction.utilities import data_utils


_GIN_CONFIG = flags.DEFINE_multi_string('gin_config', None, 'Gin config file.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', None, 'Newline separated list of Gin parameter bindings.'
)
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory.')


@gin.configurable(denylist=['output_dir'])
def train_and_evaluate_magnitude_prediction_model(
    output_dir,
    *,
    labels_constructor = training_examples.magnitude_prediction_labels,
    loss_function = None,
    learning_rate = gin.REQUIRED,
    batch_size = gin.REQUIRED,
    epochs = gin.REQUIRED,
    pdf_support_stretch = 7,
    metric_functions = tuple(),
):
  """Trains and evaluates the magnitude prediction model.

  Args:
    output_dir: Directory for storing the trained model.
    labels_constructor:  A method to build labels.
    loss_function: A string identifier of a Keras loss function or a loss
      function. Defaults to metrics.MinusLoglikelihoodLoss.
    learning_rate: The learning rate of the optimizer.
    batch_size: The size of the batch per training step.
    epochs: The number of epochs for training.
    pdf_support_stretch: factor by which the result pdf will be stretched.
    metric_functions: The metrics that we want to evaluate on every epoch.

  Returns:
    The history object, tracking metrics over epochs, and the final model.
    Additionally, stores the history on the work unit (if present), and the
    model and the encoders in the output directory.
  """
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  domain = training_examples.CatalogDomain()
  labels = labels_constructor(domain)
  all_encoders = one_region_model.build_encoders(domain)
  features_and_models = one_region_model.load_features_and_construct_models(
      domain, all_encoders, output_dir
  )

  if loss_function is None:
    loss_function = metrics.MinusLoglikelihoodConstShiftStretchLoss(
        metrics.kumaraswamy_mixture_instance,
        domain.magnitude_threshold,
        pdf_support_stretch,
    )

  head_model = head_models.magnitude_prediction_model(
      spatially_dependent_models=features_and_models.spatially_dependent_models,
      spatially_independent_models=features_and_models.spatially_independent_models,
  )
  head_model.compile(optimizer, loss_function, metric_functions)

  train_features = one_region_model.features_in_order(features_and_models, 0)
  valid_features = one_region_model.features_in_order(features_and_models, 1)
  test_features = one_region_model.features_in_order(features_and_models, 2)

  one_region_model.store_everything_in_folder(
      output_dir=output_dir,
      encoders_dict=all_encoders,
      train_features=train_features,
      validation_features=valid_features,
      test_features=test_features,
      loss_function=loss_function,
      domain=domain,
  )

  history = head_model.fit(
      x=train_features,
      y=labels.train_labels,
      validation_data=(valid_features, labels.validation_labels),
      epochs=epochs,
      batch_size=batch_size,
      verbose='auto',
  )

  one_region_model.store_everything_in_folder(
      output_dir=output_dir, model=head_model, history=history
  )

  return history, head_model


def main(_):
  output_dir = _OUTPUT_DIR.value

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  gin.parse_config_files_and_bindings(_GIN_CONFIG.value, _GIN_BINDINGS.value)
  train_and_evaluate_magnitude_prediction_model(output_dir)


if __name__ == '__main__':
  app.run(main)
