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

"""Base model definition for VAL models.

This module defines the base class decorator for VAL models. This is a
lightweight abstraction we use to obtain models that are compatible with the
train/eval loops.
"""
import enum
import inspect

from flax import linen as nn
from flax.struct import dataclasses
import gin
from jax.example_libraries import optimizers as jax_opt
import typing_extensions
from utils.types import Array
from utils.types import ModelFn
from utils.types import TextDict


@gin.constants_from_enum
class ExecutionMode(enum.Enum):
  """Defines the model execution mode."""
  TRAIN = 1
  EVAL = 2
  PREDICT = 3


class VALModelFn(typing_extensions.Protocol):
  """The type of a model_fn.

  This is the main interface of a model_fn we use throughout the codebase. Each
  model_fn should accept a `mode` value and return a
  `flax.deprecated.nn.Module`.
  """

  def __call__(self, mode, **kwargs):
    Ellipsis


class BaseModel(nn.Module):
  """Base class for VAL models.

  This base class defines a `mode` attribute that the train/eval loop needs.
  NOTE: do not use this base class for non-model modules that are only supposed
  to be invoked from other modules.
  """
  # NOTE: We cannot use a default value here since a flax `Module` is a
  # dataclass and dataclasses cannot have fields with default values in parent
  # class and fields with no default values in children
  mode: ExecutionMode

  def record_norm(self, name, x):
    """Record the l2 norm of the input array."""
    self.sow(
        'activation_norm', name, jax_opt.l2_norm(x), reduce_fn=lambda a, b: b)


def filter_attrs(model_fn,
                 module_attrs,
                 use_signature = True,
                 **model_init_kwargs):
  """Filter the dictionary attributes for initializing a module.

  We avoid creating the module here by directly inspecting the attributes of
  the modules using dataclasses and inspect. Creating objects also works as
  objects are cheap to create in Flax.
  Args:
    model_fn: A function that returns an nn.Module.
    module_attrs: A dictionary of attributes to set.
    use_signature: A bool to control whether to filter the attributes by
      creating an instance of model or using just the input signatures.
    **model_init_kwargs: Keyword arguments to initialize the module Only used
      when use_signature = False.

  Returns:
    valid_module_attrs: A dictionary of valid attributes for this module.

  Raises:
    KeyError: The module requires more attributes to be provided than what
      exist in module_attrs.
  """
  if use_signature:
    try:
      params = dataclasses.fields(model_fn)
      param_names = [v.name for v in params]
    except TypeError:  # Handle functools.partial.
      params = inspect.signature(model_fn).parameters
      param_names = params.keys()

    valid_module_attrs = {
        k: module_attrs[k] for k in param_names if k in module_attrs
    }
    return valid_module_attrs

  if 'mode' not in module_attrs:
    raise KeyError('Mode not present in module attributes!')

  obj_filter_attrs = (
      lambda x: {k: v for k, v in module_attrs.items() if hasattr(x, k)})
  return obj_filter_attrs(model_fn(**model_init_kwargs))
