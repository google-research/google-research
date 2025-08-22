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

"""Factory for models."""
from typing import Optional

from q_match.models.pretext_and_main import ImixClassifier
from q_match.models.pretext_and_main import ResnetClassifier


NAME_TO_MODEL = {
    'iMixClassifier': {
        'constructor': ImixClassifier,
    },
    'ResnetClassifier': {
        'constructor': ResnetClassifier,
    },
}


def get_models(model_name,
               num_classes,
               algorithm = None):
  """Returns training and eval model pair."""
  if model_name not in NAME_TO_MODEL:
    raise ValueError('%s not supported yet.' % model_name)
  model = NAME_TO_MODEL[model_name]['constructor']
  constructor_args = {'num_classes': num_classes, 'algorithm': algorithm}
  training_model = model(**constructor_args, training=True)  # pytype: disable=wrong-keyword-args
  eval_model = model(**constructor_args, training=False)  # pytype: disable=wrong-keyword-args
  return training_model, eval_model
