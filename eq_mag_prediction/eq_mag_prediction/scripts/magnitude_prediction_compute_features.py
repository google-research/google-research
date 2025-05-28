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

"""Calculates features, and stores them in a cache."""

import os

from absl import app
from absl import flags
from absl import logging
import gin

from eq_mag_prediction.forecasting import one_region_model
from eq_mag_prediction.forecasting import training_examples


DEFAULT_FEATURE_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "../..", "results/cached_features"
)

_FORCE_RECOMPUTE = flags.DEFINE_boolean(
    "force_recompute",
    False,
    "Force recomputation of features even if existing.",
)

_CACHE_DIR = flags.DEFINE_string(
    "cache_dir", DEFAULT_FEATURE_CACHE_DIR, "Features cache directory."
)

_GIN_PATH = flags.DEFINE_string("gin_path", None, "gin config file path.")


def features_cache_dir():
  assert_msg = (
      "cache_dir is empty, define in command line or in"
      " eq_mag_prediction/scripts/magnitude_prediction_compute_features.py"
  )
  assert _CACHE_DIR.value, assert_msg
  return _CACHE_DIR.value


def compute_and_cache_magnitude_prediction_features():
  """Computes and caches magnitude prediction features."""
  domain = training_examples.CatalogDomain()
  all_encoders = one_region_model.build_encoders(domain)
  one_region_model.compute_and_cache_features_scaler_encoder(
      domain,
      all_encoders,
      cache_dir=features_cache_dir(),
      force_recalculate=_FORCE_RECOMPUTE.value,
  )


def main(_):
  gin.parse_config_file(_GIN_PATH.value, skip_unknown=True)
  logging.info("Computing features for gin config.")
  compute_and_cache_magnitude_prediction_features()
  logging.info("Done.")


if __name__ == "__main__":
  app.run(main)
