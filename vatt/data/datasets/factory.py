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

"""Defines internal data_loader of YouTube datasets."""

import functools
from typing import Any, Dict, Optional, Text

from vatt.data.datasets import toy_dataset


# DMVR-based factories
DS_TO_FACTORY = {
    #########################################
    #### put your dataset factories here ####
    #########################################
    'TOY_DS': toy_dataset.ToyFactory,
}


def get_ds_factory(dataset_name = 'toy_ds',
                   override_args = None):
  """Gets dataset source and name and returns its factory class."""

  dataset_name = dataset_name.upper()

  ds = DS_TO_FACTORY[dataset_name]

  if override_args:
    return functools.partial(ds, **override_args)
  else:
    return ds
