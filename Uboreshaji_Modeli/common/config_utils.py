# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Utilities for loading and processing Poly-Sense2 configurations."""

import importlib.util
import os
import sys

import ml_collections


def load_config(path):
  """Loads and parses a Python config file into a ConfigDict object.

  Args:
    path: The path to the Python config file.

  Returns:
    A ml_collections.ConfigDict object loaded from the file.

  Raises:
    FileNotFoundError: If the config file does not exist.
    ImportError: If the config file cannot be loaded as a module.
    AttributeError: If the config file does not define a 'get_config()'
      or 'get_base_config()' function.
  """

  resolved_path = path


  module_name = "config_module"
  try:
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
  except FileNotFoundError:
    raise FileNotFoundError(
        f"Config file not found: {resolved_path!r}"
    ) from None
  else:
    if spec is None or spec.loader is None:
      raise ImportError(f"Could not load config from {resolved_path!r}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "get_config"):
      return module.get_config()
    # Fallback/convenience: look for get_base_config if get_config is missing
    if hasattr(module, "get_base_config"):
      return module.get_base_config()
    raise AttributeError(
        f"Config file {resolved_path!r} must define a 'get_config()' or"
        " 'get_base_config()' function."
    )

_OWL_DEFAULTS = {
    "/path/to/models/owlv2",
}
_OWL_DEFAULTS = frozenset(_OWL_DEFAULTS)


def derive_paths(cfg):
  """Derives dataset_path and model_id from their component fields."""
  modality_val = cfg.get("task_modality", "VISION")
  modality = (
      modality_val.value if hasattr(modality_val, "value") else modality_val
  )

  if modality == "VISION":
    if not cfg.dataset.get("dataset_path") and cfg.dataset.get("dataset_base"):
      cfg.dataset.dataset_path = (
          f"{cfg.dataset.dataset_base}/{cfg.dataset.dataset_uri}"
          f"/huggingface_dataset/{cfg.dataset.dataset_version}/"
      )
    if (
        not cfg.get("model_id") or cfg.get("model_id") in _OWL_DEFAULTS
    ) and cfg.get("model_base") and cfg.get("model_name"):
      cfg.model_id = f"{cfg.model_base}/{cfg.model_name}"
  else:
    # For TEXT and AUDIO modalities, explicit model_id and dataset_path take
    # precedence. We only derive them if they are not explicitly set, are empty,
    # or match legacy defaults.
    if (
        not cfg.dataset.get("dataset_path")
        and cfg.dataset.get("dataset_uri")
    ):
      base = cfg.dataset.get("dataset_base") or ""
      cfg.dataset.dataset_path = os.path.join(base, cfg.dataset.dataset_uri)
    if (
        (not cfg.get("model_id") or cfg.get("model_id") in _OWL_DEFAULTS)
        and cfg.get("model_base")
        and cfg.get("model_name")
    ):
      cfg.model_id = os.path.join(cfg.model_base, cfg.model_name)
