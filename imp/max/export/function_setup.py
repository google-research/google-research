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

"""A minimal interface for evaluating IMP."""

import functools
import importlib
from typing import Any

import flax
from flax import traverse_util
import jax

from imp.max.config import registry
from imp.max.core import constants
from imp.max.execution import executors
from imp.max.utils import typing

Registrar = registry.Registrar
DataFeatureType = constants.DataFeatureType


def infer_model_with_params(
    input_data,
    model_params,
    mutables,
    executor,
    postprocess_fn = None):
  """Main function for model inference."""
  inference_fn = executor.create_inference_step(postprocess_fn)
  outputs, _, _ = inference_fn(model_params, mutables, input_data)

  # Metadata is not convertible to a tensor so remove it from outputs.
  if DataFeatureType.METADATA in outputs:
    del outputs[DataFeatureType.METADATA]

  return traverse_util.flatten_dict(outputs, sep='/')


def configure_function(
    qualified_fn,
    experiment_name,
    checkpoint_path = None,
    is_flax_checkpoint = False,
    function_kwargs = None,
):
  """Returns configured function that only receives input tensors and params."""

  module_name, function_name = qualified_fn.rsplit('.', 1)
  module = importlib.import_module(module_name)
  configurable_function = getattr(module, function_name)

  config = Registrar.get_config_by_name(experiment_name)
  config.path = '/tmp/summary/'

  model = Registrar.get_class_by_name(config.model.name)(
      **config.model.as_dict()
  )
  executor = executors.BaseExecutor(model=model, dataloaders=(), config=config)
  state, _, _ = executor.initialize_states()

  params = dict(state.params)
  if checkpoint_path is not None:
    if is_flax_checkpoint:
      params = flax.training.checkpoints.restore_checkpoint(
          checkpoint_path, target=None, step=0
      )
    else:
      state = executor.ckpt_manager.restore(checkpoint_path, state)
      params = dict(state.params)

  function_kwargs = function_kwargs or {}
  return (
      functools.partial(
          configurable_function,
          config=config,
          executor=executor,
          **function_kwargs,
      ),
      params,
  )


def set_const_model_params(
    function, params
):
  """Returns function with data input only, model_params fixed to params."""
  return functools.partial(function, model_params=params)
