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

"""Base mode class."""

from absl import logging
from flax.deprecated import nn


class BaseModel(nn.Module):
  """Base class for Models."""

  @classmethod
  def build_flax_module(cls, hparams=None, dataset_metadata=None):
    """Build flax module (partially build by passing the hparams).

    This method should be overriden by the children classes.

    Args:
      hparams: ConfigDict; Contains the hyperparams of the model architecture.
      dataset_metadata: dict; If hparams is None, dataset_meta data should be
        passed to provide the output_dim for the default hparam set.

    Returns:
      hparams (in children classes it also returns the partially built Module.)
    """
    if hparams is None:
      logging.warning('You are creating the model with default hparams.')
      if dataset_metadata is None:
        raise ValueError('Both hparams and dataset_metadata are None.')
      hparams = cls.default_flax_module_hparams(dataset_metadata)
    elif dataset_metadata is not None:
      logging.warning('The argument dataset_metadata is not None but not used.')

    return hparams

  @classmethod
  def default_flax_module_hparams(cls, dataset_metadata):
    """Default hparams for the flax module that is built in `build_flax_module`.

    This function in particular serves the testing functions and supposed to
    provide hparams that are passed to the flax_module when it's built in
    `build_flax_module` function, e.g., `model_dtype_str`.

    Args:
      dataset_metadata: dict; Passed to provide output dim.

    Returns:
      default hparams.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_module_hparams().')
