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

"""Registry for the available models we can train."""

from gift.models import resnet
from gift.models import wide_resnet

ALL_MODELS = {}

CLASSIFICATION_MODELS = {
    'resnet': resnet.ResNet,
    'wide_resnet': wide_resnet.WideResnet
}

ALL_MODELS.update(CLASSIFICATION_MODELS)


def get_model_class(model_name):
  """Get the corresponding model class based on the model string.

  API:
  model_builder= get_model('fully_connected')
  model = model_builder(hparams, num_classes)

  Args:
    model_name: str; Name of the model, e.g. 'fully_connected'.

  Returns:
    The model architecture (a flax Model) along with its default hparams.
  Raises:
    ValueError if model_name is unrecognized.
  """
  if model_name not in ALL_MODELS.keys():
    raise ValueError('Unrecognized model: {}'.format(model_name))
  return ALL_MODELS[model_name]
