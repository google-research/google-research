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

"""Main file used for approxNN project."""

from typing import Sequence
import warnings

from absl import app

from invariant_explanations import config
from invariant_explanations import other
from invariant_explanations import utils


warnings.simplefilter('ignore')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  utils.create_experimental_folders()

  # utils.analyze_accuracies_of_base_models()

  utils.process_and_resave_cnn_zoo_data(
      config.RANDOM_SEED,
      other.get_model_wireframe(),
      config.COVARIATES_SETTINGS,
  )

  # utils.plot_treatment_effect_values()

  # utils.train_meta_model_over_different_setups(config.RANDOM_SEED)

  # utils.save_heat_map_of_meta_model_results()

  # utils.process_per_class_explanations(config.RANDOM_SEED)

  # utils.measure_prediction_explanation_variance(config.RANDOM_SEED)


if __name__ == '__main__':
  app.run(main)
