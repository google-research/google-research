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

"""Parent dataset class that all datasets inherit from."""
import abc


class PretextDataset(abc.ABC):
  """Parent class with methods that all children must implment.

  For the research, we always create datasets using the Dataset class, which
  inherits from this class, but it is useful to have a pretext dataset class
  that has fewer methods in must implment when playing with pretext algos.
  """

  @abc.abstractmethod
  def get_pretext_ds(self,):
    """Returns the dataset for pretext training.

    Each call to the dataset should return a batch of data, and the iterator
    terminates at the end of the epoch.
    """

  @abc.abstractmethod
  def get_example_features(self,):
    """Returns a batch of data.

    Useful for initializing models.
    """

  def get_pretext_validation_ds(self):
    """Returns the pretext validation dataset if it exists.

    Not all datasets are required to have this.
    """
    return None


class Dataset(PretextDataset):
  """Parent class with methods that all children must implement."""

  @abc.abstractmethod
  def get_num_classes(self,):
    """Returns the number of classes in the dataset."""

  @abc.abstractmethod
  def get_train_ds(self,):
    """Returns the supervised training dataset.

    Each call to the dataset should return a batch of data, and the iterator
    terminates at the end of the epoch.
    """

  @abc.abstractmethod
  def get_validation_ds(self,):
    """Returns the validation dataset for supervised training."""

  @abc.abstractmethod
  def get_test_epoch_iterator(self,):
    """Returns the supervsied test dataset that only can iterate one epoch."""

  def get_train_val_ds(self):
    """Returns the final train-val dataset if it exists.

    Not all datasets are required to have this.
    """
    return None

  def get_imputation_validation_ds(self):
    """Returns the imputation validation dataset if it exists.

    Not all datasets are required to have this.
    """
    return None

  def get_imputation_train_ds(self):
    """Returns the imputation train dataset if it exists.

    Not all datasets are required to have this.
    """
    return None

