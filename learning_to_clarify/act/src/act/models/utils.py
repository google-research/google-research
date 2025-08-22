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

import os
from pathlib import Path
from typing import List, Optional, Union
import warnings

from act.config.base_config import BaseConfig
from act.config.base_config import BaseInitializationConfig
from act.config.model.hf_model_config import HFModelConfig
from act.config.model.model_config import ModelConfig
from act.models.action_classifier_model import ActionClassifierModel
from act.models.generative_model import GoogleGenerativeModel
from act.models.hf_model import HFModel
from act.models.intent_model import UserIntentModel
from act.models.preference_model import RejectedSampleModel
from act.models.simulator_model import SimulatorModel
from peft import LoraConfig, PeftConfig
import torch
from transformers.trainer_utils import get_last_checkpoint
import google.generativeai as genai


# I'm initializing this with the models that I've tested so far, but in
# principle there's no particular reason to limit it to these models.
_HF_MODELS = [
    'google/gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'HuggingFaceH4/zephyr-7b-beta',
    'HuggingFaceH4/zephyr-7b-gemma-sft-v0.1',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'mistralai/Mistral-Nemo-Instruct-2407',
]


# I'm initializing this with the models that I've tested so far, but in
# principle there's no particular reason to limit it to these models.
_GOOGLE_MODELS = [
    'gemini-ultra',
    'gemini-1.5-pro-001',
    'gemini-1.5-pro-002',
    'gemini-2.0-flash-001',
    'gemini-2.0-flash-exp',
]


# Prefix of models we will load from Google Storage.
_GS_MODEL_PREFIXES = ['gs://learning-to-clarify-staging']


_CLASSES = {
    'action': ActionClassifierModel,
    'user_simulator': SimulatorModel,
    'preference': RejectedSampleModel,
    'intent': UserIntentModel,
}


_SUPPORTED_POLICY_MODELS = _HF_MODELS
_SUPPORTED_ACTION_MODELS = _GOOGLE_MODELS
_SUPPORTED_USER_SIMULATOR_MODELS = _GOOGLE_MODELS
_SUPPORTED_INTENT_MODELS = _GOOGLE_MODELS
_SUPPORTED_REF_MODELS = _HF_MODELS


def initialize_env() -> None:
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
      raise ValueError('GOOGLE_API_KEY environment variable not set.')

    genai.configure(api_key=api_key)


def get_checkpoint(output_dir: str) -> Path | None:
  last_checkpoint = None
  if os.path.isdir(output_dir):
    last_checkpoint = get_last_checkpoint(output_dir)
  return last_checkpoint


def get_peft_config(model_config) -> PeftConfig | None:
  if model_config.use_peft is False:
    return None

  peft_config = LoraConfig(
      r=model_config.lora_r,
      lora_alpha=model_config.lora_alpha,
      lora_dropout=model_config.lora_dropout,
      bias='none',
      task_type='CAUSAL_LM',
      target_modules=model_config.lora_target_modules,
      modules_to_save=model_config.lora_modules_to_save,
  )

  return peft_config


def _starts_with_any(model_path: Optional[str], prefixes: List[str]) -> bool:
  if model_path is None:
    return False

  for prefix in prefixes:
    if model_path.startswith(prefix):
      return True

  return False


def _check_supported_model_configs(
    policy_model: Optional[ModelConfig],
    ref_model: Optional[ModelConfig],
    action_model: ModelConfig,
    user_simulator: ModelConfig,
    intent_model: ModelConfig,
):
  """Check that the model configs are supported."""
  if policy_model and (
      policy_model.model_id not in _SUPPORTED_POLICY_MODELS
      and not policy_model.model_path
  ):
    raise ValueError(
        f'Policy model {policy_model.model_id} is not supported. Please use one'
        f' of the following models: {_SUPPORTED_POLICY_MODELS}.'
    )
  if ref_model and (
      ref_model.model_id not in _SUPPORTED_REF_MODELS
      and not ref_model.model_path
  ):
    raise ValueError(
        f'Ref model {ref_model.model_id} is not supported. Please use one'
        f' of the following models: {_SUPPORTED_REF_MODELS}.'
    )
  if action_model.model_id not in _SUPPORTED_ACTION_MODELS:
    raise ValueError(
        f'Action model {action_model.model_id} is not supported. Please use'
        f' one of the following models: {_SUPPORTED_ACTION_MODELS}'
    )
  if user_simulator.model_id not in _SUPPORTED_USER_SIMULATOR_MODELS:
    raise ValueError(
        f'User simulator model {user_simulator.model_id} is not supported.'
        ' Please use one of the following models:'
        f' {_SUPPORTED_USER_SIMULATOR_MODELS}'
    )
  if intent_model.model_id not in _SUPPORTED_INTENT_MODELS:
    raise ValueError(
        f'Intent model {intent_model.model_id} is not supported. Please use'
        f' one of the following models: {_SUPPORTED_INTENT_MODELS}'
    )


def _route_and_load_model(
    config: Union[BaseConfig, BaseInitializationConfig],
    model_config: Union[ModelConfig, HFModel],
    model_type: str,
) -> Union[
    SimulatorModel,
    UserIntentModel,
    ActionClassifierModel,
    GoogleGenerativeModel,
    HFModel,
]:
  """Route the model to the appropriate loader."""
  if model_config.model_id in _HF_MODELS or model_config.model_path:
    return load_hf_model(config, model_config)
  elif model_config.model_id in _GOOGLE_MODELS:
    return load_google_model(config, model_config, model_type)
  else:
    raise ValueError(
        f'Model {model_config.model_id} is not supported. Please use'
        f' one of the following models: {_HF_MODELS} or {_GOOGLE_MODELS}'
    )


def load_google_model(
    config: BaseConfig, model_config: ModelConfig, model_type: str
) -> GoogleGenerativeModel:
  """Load a model from Vertex."""
  return _CLASSES[model_type](config, model_config)


def load_hf_model(
    config: Union[BaseConfig, BaseInitializationConfig],
    model_config: HFModelConfig,
):
  """Load a model from Hugging Face."""
  tokenizer, model = HFModel.construct_hf_model(config, model_config)
  return HFModel(config, tokenizer, model, model_config)


def load_models(
    config: BaseConfig,
    load_policy_from_checkpoint: bool = False,
):
  _check_supported_model_configs(
      None if load_policy_from_checkpoint else config.policy_model_config,
      None if load_policy_from_checkpoint else config.policy_model_config,
      config.action_model_config,
      config.user_simulator_config,
      config.intent_model_config,
  )
  policy_model = (
      load_hf_model(config, config.policy_model_config)
      if load_policy_from_checkpoint
      else _route_and_load_model(
          config, config.policy_model_config, model_type='policy'
      )
  )
  ref_model = (
      load_hf_model(config, config.policy_model_config)
      if load_policy_from_checkpoint
      else _route_and_load_model(
          config, config.policy_model_config, model_type='policy'
      )
  )
  action_model = _route_and_load_model(
      config, config.action_model_config, model_type='action'
  )
  user_simulator = _route_and_load_model(
      config, config.user_simulator_config, model_type='user_simulator'
  )
  intent_summarization_model = _route_and_load_model(
      config, config.intent_model_config, model_type='intent'
  )
  return (
      policy_model,
      ref_model,
      action_model,
      user_simulator,
      intent_summarization_model,
  )


def load_sft_models(config: BaseInitializationConfig, load_policy_from_checkpoint: bool = False):
  policy_model = (
      load_hf_model(config, config.policy_model_config)
      if load_policy_from_checkpoint
      else _route_and_load_model(
          config, config.policy_model_config, model_type='policy'
      )
  )
  return (
      policy_model,
  )
