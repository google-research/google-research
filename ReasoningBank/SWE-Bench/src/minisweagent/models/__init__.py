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

"""This file provides convenience functions for selecting models.
You can ignore this file completely if you explicitly set your model in your run script.
"""

import copy
import importlib
import os
import threading

from minisweagent import Model


class GlobalModelStats:
    """Global model statistics tracker with optional limits."""

    def __init__(self):
        self._cost = 0.0
        self._n_calls = 0
        self._lock = threading.Lock()
        self.cost_limit = float(os.getenv("MSWEA_GLOBAL_COST_LIMIT", "0"))
        self.call_limit = int(os.getenv("MSWEA_GLOBAL_CALL_LIMIT", "0"))
        if (self.cost_limit > 0 or self.call_limit > 0) and not os.getenv("MSWEA_SILENT_STARTUP"):
            print(f"Global cost/call limit: ${self.cost_limit:.4f} / {self.call_limit}")

    def add(self, cost):
        """Add a model call with its cost, checking limits."""
        with self._lock:
            self._cost += cost
            self._n_calls += 1
        if 0 < self.cost_limit < self._cost or 0 < self.call_limit < self._n_calls + 1:
            raise RuntimeError(f"Global cost/call limit exceeded: ${self._cost:.4f} / {self._n_calls + 1}")

    @property
    def cost(self):
        return self._cost

    @property
    def n_calls(self):
        return self._n_calls


GLOBAL_MODEL_STATS = GlobalModelStats()


def get_model(input_model_name = None, config = None):
    """Get an initialized model object from any kind of user input or settings."""
    resolved_model_name = get_model_name(input_model_name, config)
    if config is None:
        config = {}
    config = copy.deepcopy(config)
    config["model_name"] = resolved_model_name

    model_class = get_model_class(resolved_model_name, config.pop("model_class", ""))

    if (from_env := os.getenv("MSWEA_MODEL_API_KEY")) and not str(type(model_class)).endswith("DeterministicModel"):
        config.setdefault("model_kwargs", {})["api_key"] = from_env

    return model_class(**config)


def get_model_name(input_model_name = None, config = None):
    """Get a model name from any kind of user input or settings."""
    if config is None:
        config = {}
    if input_model_name:
        return input_model_name
    if from_config := config.get("model_name"):
        return from_config
    if from_env := os.getenv("MSWEA_MODEL_NAME"):
        return from_env
    raise ValueError("No default model set. Please run `mini-extra config setup` to set one.")


_MODEL_CLASS_MAPPING = {
    "anthropic": "minisweagent.models.anthropic.AnthropicModel",
    "litellm": "minisweagent.models.litellm_model.LitellmModel",
    "deterministic": "minisweagent.models.test_models.DeterministicModel",
}


def get_model_class(model_name, model_class = ""):
    """Select the best model class.

    If a model_class is provided (as shortcut name, or as full import path,
    e.g., "anthropic" or "minisweagent.models.anthropic.AnthropicModel"),
    it takes precedence over the `model_name`.
    Otherwise, the model_name is used to select the best model class.
    """
    if model_class:
        full_path = _MODEL_CLASS_MAPPING.get(model_class, model_class)
        try:
            module_name, class_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError):
            msg = f"Unknown model class: {model_class} (resolved to {full_path}, available: {_MODEL_CLASS_MAPPING})"
            raise ValueError(msg)

    if any(s in model_name.lower() for s in ["anthropic", "sonnet", "opus", "claude"]):
        from minisweagent.models.anthropic import AnthropicModel

        return AnthropicModel

    # Default to LitellmModel
    from minisweagent.models.litellm_model import LitellmModel

    return LitellmModel
