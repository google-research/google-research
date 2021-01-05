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

"""Learned Interpreters adapters.

Adapters bridge the gap between datasets and models. They provide the
implementation details of workflows so that workflows can be dataset and model
agnostic.
"""

from ipagnn.adapters import common_adapters
from ipagnn.adapters import gat_adapters
from ipagnn.adapters import ipagnn_adapters

ADAPTER_MODULES = [common_adapters, gat_adapters, ipagnn_adapters]


def get_default_adapter(info, config):
  """Gets the adapter for the indicated model and dataset."""
  target_adapter_name = f'{config.model.name}Adapter'
  for module in ADAPTER_MODULES:
    if hasattr(module, target_adapter_name):
      adapter_cls = getattr(module, target_adapter_name)
      return adapter_cls(info, config)

  # Fall back on a common adapter.
  return common_adapters.SequenceAdapter(info, config)
