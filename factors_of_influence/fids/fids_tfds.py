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

"""Define the FIDS sourceset collection."""
import abc
from typing import Dict, Generator, List, Optional, Text, Tuple

import tensorflow_datasets as tfds

from factors_of_influence.fids import fids_dataset


class FIDSConfig(tfds.core.BuilderConfig):
  """Tfds config. Wraps FIDSDataset."""

  def __init__(self,
               dataset,
               version,
               release_notes = None,
               **kwargs):
    """Config."""
    super().__init__(
        name=dataset.config_name,
        description=dataset.config_description,
        version=version,
        release_notes=release_notes,
        **kwargs)
    self.dataset = dataset

  def get_tfds_features_dict(self):
    """Returns feature dictionary to construct tfds DatasetInfo."""
    return self.dataset.feature_utils.get_tfds_features_dict(
        self.dataset.feature_names)

  def info(self):
    return self.dataset.info()

  def generate_examples(self, split):
    return self.dataset.generate_examples(split)

  @property
  def splits(self):
    return self.dataset.splits


class FIDSTFDS(tfds.core.GeneratorBasedBuilder):
  """Base class for factor of influence TFDS dataset collection.

  Implements all TFDS dependent parts.
  """

  @property
  @classmethod
  @abc.abstractmethod
  def BUILDER_CONFIGS(cls):
    # An abstract class property to ensure this class is not registered by tfds,
    # while the child classes only need to define BUILDER_CONFIGS = [ ... ]
    # as is done in fids_tfds_builders.
    pass

  def _info(self):
    """Define DatasetInfo."""

    return tfds.core.DatasetInfo(
        builder=self,
        features=self.builder_config.get_tfds_features_dict(),
        supervised_keys=None,
        homepage='https://github.com/google-research/google-research/tree/master/factors_of_influence',
        metadata=tfds.core.MetadataDict(self.builder_config.info()))

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators, each calling _generate_examples(split)."""
    return [
        tfds.core.SplitGenerator(name=split, gen_kwargs={'split': split})
        for split in self.builder_config.splits
    ]

  def _generate_examples(
      self, split
  ):
    """Returns examples (data with annotations).

    This function assumes that each feature implements its own getter function.
    The getter function returns a dictionary mapping keys to examples (examples
    may be image paths). The getter function is named f'_get_{feature}_dict'.

    Args:
      split: Split of the dataset.

    Returns:
      key, examples pairs.
    """
    return self.builder_config.generate_examples(split)
