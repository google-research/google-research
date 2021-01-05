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

"""Learned Interpreters models."""

from ipagnn.models import gat
from ipagnn.models import ggnn
from ipagnn.models import ipagnn
from ipagnn.models import ipagnn_interpolants
from ipagnn.models import rnn_models

MODEL_MODULES = [
    gat,
    ggnn,
    ipagnn,
    ipagnn_interpolants,
    rnn_models,
]
ADDITIONAL_MODELS = {}


def get_model(info, config):
  model_name = config.model.name
  for module in MODEL_MODULES:
    model_cls = getattr(module, model_name, None)
    if model_cls is not None:
      return model_cls.partial(info=info, config=config)

  if model_name in ADDITIONAL_MODELS:
    return ADDITIONAL_MODELS[model_name]
  raise ValueError('Could not create model.', model_name)
