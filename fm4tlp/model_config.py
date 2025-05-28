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

"""Helpers to define models for xmanager runs.
"""

import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from fm4tlp.models import model_config_pb2


_MODEL_CONFIG_PATHS = [
    "google_research/fm4tlp/models/configs/dyrep.pbtxt",
    "google_research/fm4tlp/models/configs/dyrep_structmap.pbtxt",
    "google_research/fm4tlp/models/configs/dyrep_structmap_alpha10.pbtxt",
    "google_research/fm4tlp/models/configs/dyrep_structmap_alpha100.pbtxt",
    "google_research/fm4tlp/models/configs/jodie.pbtxt",
    "google_research/fm4tlp/models/configs/jodie_structmap.pbtxt",
    "google_research/fm4tlp/models/configs/jodie_structmap_alpha10.pbtxt",
    "google_research/fm4tlp/models/configs/jodie_structmap_alpha100.pbtxt",
    "google_research/fm4tlp/models/configs/tgn.pbtxt",
    "google_research/fm4tlp/models/configs/tgn_structmap.pbtxt",
    "google_research/fm4tlp/models/configs/tgn_structmap_alpha10.pbtxt",
    "google_research/fm4tlp/models/configs/tgn_structmap_alpha100.pbtxt",
    "google_research/fm4tlp/models/configs/edgebank.pbtxt",
]


def get_model_config(model_name):
  """Returns a model config from the specified model name."""
  model_configs = {}
  for model_config_path in _MODEL_CONFIG_PATHS:
    model_config = model_config_pb2.TlpModelConfig()
    filepath = str(model_config_path)
    with tf.io.gfile.GFile(filepath, "r") as f:
      text_format.Parse(f.read(), model_config)
    if model_config.model_name in model_configs:
      raise ValueError(
          f"Duplicate model name: {model_config.model_name}"
      )
    model_configs[model_config.model_name] = model_config
  return model_configs[model_name]
