# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Functions to load/save the hparams to/from a config dict."""

import json
import os
import typing
from typing import Any, Dict, Optional, Type, TypeVar

import dacite
import dataclasses
import jax
import ml_collections


from aqt.jax import quant_config
from aqt.jax import quantization
from aqt.jax.flax import struct as flax_struct

T = TypeVar('T')

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


@dataclass
class HParamsMetadata:
  """Metadata associated with an experiment configuration."""

  # Human-readable description of this hparams configuration. Mainly
  # useful for hand inspection of serialized JSON files.
  description: str

  # Creation time of the configuration in the format of seconds from epoch.
  # Used for versioning different hyperparameter settings for the same
  # model configuration.
  last_updated_time: Optional[float]

  # By default, it is used to name the model directory and label the
  # experiment in tensorboard.
  hyper_str: Optional[str] = None


# TODO(abdolrashidi): Add unit tests for the functions below.
def save_dataclass_to_disk(data, path):
  """Serializes the given dataclass to a JSON file on disk.

  Args:
    data: A dataclass instance.
    path: Path to save the dataclass to.
  """

  data_dict = dataclasses.asdict(data)
  with open(path, 'w') as file:
    json.dump(data_dict, file, indent=2)


def write_hparams_to_file_with_host_id_check(hparams,
                                             output_dir):
  """Writes hparams to file for master host.

  Args:
    hparams: Hparams.
    output_dir: Output directory to save hparams to, saves as output_dir /
      'hparams_config.json.
  """
  if jax.host_id() == 0 and output_dir is not None:
    # The directory is usually created automatically by the time we reach here,
    # but on some training runs it appears not to be.
    # MakeDirs will create the directory if it doesn't already exist and is a
    # no-op if it already exists.


    os.makedirs(output_dir, exist_ok=True)
    save_dataclass_to_disk(hparams,
                           os.path.join(output_dir, 'hparams_config.json'))


def load_dataclass_from_dict(dataclass_name,
                             data_dict):
  """Converts parsed dictionary from JSON into a dataclass.

  Args:
    dataclass_name: Name of the dataclass.
    data_dict: Dictionary parsed from JSON.

  Returns:
    An instance of `dataclass` populated with the data from `data_dict`.
  """
  # Some fields in TrainingHParams are formal Python enums, but they are stored
  # as plain text in the json. Dacite needs to be given a list of which classes
  # to convert from a string into an enum. The classes of all enum values which
  # are stored in a TrainingHParams instance (directly or indirectly) should be
  # listed here. See https://github.com/konradhalas/dacite#casting.
  enum_classes = [
      quantization.QuantOps.ActHParams.InputDistribution,
      quantization.QuantType, quant_config.QuantGranularity
  ]
  data_dict = _convert_lists_to_tuples(data_dict)
  return dacite.from_dict(
      data_class=dataclass_name,
      data=data_dict,
      config=dacite.Config(cast=enum_classes))


T = TypeVar('T')


def _convert_lists_to_tuples(node):
  """Recursively converts all lists to tuples in a nested structure.

  Recurses into all lists and dictionary values referenced by 'node',
  converting all lists to tuples.

  Args:
    node: A Python structure corresponding to JSON (a dictionary, a list,
      scalars, and compositions thereof)

  Returns:
    A Python structure identical to the input, but with lists replaced by
      tuples.
  """

  if isinstance(node, dict):
    return {key: _convert_lists_to_tuples(value) for key, value in node.items()}
  elif isinstance(node, (list, tuple)):
    return tuple([_convert_lists_to_tuples(value) for value in node])
  else:
    return node


def load_dataclass_from_json(dataclass_name, json_data):
  """Creates a dataclass instance from JSON.

  Args:
    dataclass_name: Name of the dataclass to deserialize the JSON into.
    json_data: A Python string containing JSON.

  Returns:
    An instance of 'dataclass' populated with the JSON data.
  """

  data_dict = json.loads(json_data)
  return load_dataclass_from_dict(dataclass_name, data_dict)


# TODO(shivaniagrawal): functionality `load_hparams_from_file` is created for a
# generic (model hparams independent) train_hparams class; either we should move
# towards shared TrainHparams or remove the following functionalities.
def load_hparams_from_config_dict(hparams_classname,
                                  model_classname,
                                  config_dict):
  """Loads hparams from a configdict, and populates its model object.

  Args:
    hparams_classname: Name of the hparams class.
    model_classname: Name of the model class within the hparams class
    config_dict: A config dict mirroring the structure of hparams.

  Returns:
    An instance of 'hparams_classname' populated with the data from
    'config_dict'.
  """

  hparams = load_dataclass_from_config_dict(hparams_classname, config_dict)
  hparams.model_hparams = load_dataclass_from_dict(model_classname,
                                                   hparams.model_hparams)
  return hparams


def load_dataclass_from_config_dict(
    dataclass_name, config_dict):
  """Creates a dataclass instance from a configdict.

  Args:
    dataclass_name: Name of the dataclass to deserialize the configdict into.
    config_dict: A config dict mirroring the structure of 'dataclass_name'.

  Returns:
    An instance of 'dataclass_name' populated with the data from 'config_dict'.
  """

  # We convert the config dicts to JSON instead of a dictionary to force all
  # recursive field references to fully resolve in a way that Dacite can
  # consume.
  json_data = config_dict.to_json()
  return load_dataclass_from_json(dataclass_name, json_data)
