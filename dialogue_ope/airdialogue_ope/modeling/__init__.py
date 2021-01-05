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

from transformers.modeling_auto import *
from .modeling_roberta import RobertaForAirOPE

MODEL_FOR_AIROPE_MAPPING = OrderedDict([(RobertaConfig, RobertaForAirOPE)])


class AutoModelForAirOPE:

  def __init__(self):
    raise EnvironmentError(
        "AutoModel is designed to be instantiated "
        "using the `AutoModelForAirOPE.from_pretrained(pretrained_model_name_or_path)` or "
        "`AutoModelForAirOPE.from_config(config)` methods.")

  @classmethod
  def from_config(cls, config):
    for config_class, model_class in MODEL_FOR_AIROPE_MAPPING.items():
      if isinstance(config, config_class):
        return model_class(config)
    raise ValueError(
        "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
        "Model type should be one of {}.".format(
            config.__class__, cls.__name__,
            ", ".join(c.__name__ for c in MODEL_FOR_AIROPE_MAPPING.keys())))

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                      **kwargs):
    config = kwargs.pop("config", None)
    if not isinstance(config, PretrainedConfig):
      config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                          **kwargs)

    for config_class, model_class in MODEL_FOR_AIROPE_MAPPING.items():
      if isinstance(config, config_class):
        return model_class.from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, **kwargs)
    raise ValueError(
        "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
        "Model type should be one of {}.".format(
            config.__class__, cls.__name__,
            ", ".join(c.__name__ for c in MODEL_FOR_AIROPE_MAPPING.keys())))
