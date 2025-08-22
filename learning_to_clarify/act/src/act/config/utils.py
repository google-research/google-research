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
import tokenize

from act.config.base_config import BaseConfig, BaseInitializationConfig
from act.config.data.data_config import DataConfig
from act.config.model.hf_model_config import HFModelConfig
from act.config.model.model_config import ModelConfig
from act.config.training.training_config import (
    ACTConfig,
    ACTInitializationConfig,
)
from act.utils.storage_utils import read_json

_DEFAULT_CONFIG = "/act/src/act/config/demo_config.json"
_DEFAULT_SFT_CONFIG = "/act/src/act/config/sft_demo_config.json"

def get_default_config():
  """Returns a default config object."""
  return get_config_from_json_path(_DEFAULT_CONFIG)


def get_default_sft_config():
  """Returns a default config object for SFTTrainer."""
  return get_sft_config_from_json_path(_DEFAULT_SFT_CONFIG)


def get_sft_config_from_flags(flags):
  """Get SFT config from flags."""
  policy_model_config = get_policy_model_config(flags)
  training_config = get_sft_training_config(flags)
  preference_model_config = get_preference_model_config(flags)
  data_config = get_data_config(flags)
  return BaseInitializationConfig(
      policy_model_config=policy_model_config,
      training_config=training_config,
      preference_model_config=preference_model_config,
      data_config=data_config,
  )


def get_config_from_flags(flags):
  """Get data from flags."""
  policy_model_config = get_policy_model_config(flags)
  action_model_config = get_action_model_config(flags)
  user_simulator_config = get_user_simulator_config(flags)
  intent_model_config = get_intent_model_config(flags)
  training_config = get_training_config(flags)
  preference_model_config = get_preference_model_config(flags)
  data_config = get_data_config(flags)
  return BaseConfig(
      policy_model_config=policy_model_config,
      action_model_config=action_model_config,
      user_simulator_config=user_simulator_config,
      intent_model_config=intent_model_config,
      training_config=training_config,
      preference_model_config=preference_model_config,
      data_config=data_config,
  )


def get_preference_model_config(flags):
  """Get preference model config from flags."""
  return ModelConfig(
      model_id=flags.preference_model_id,
  )


def get_policy_model_config(flags):
  """Get policy model config from flags."""
  conf = HFModelConfig(
      model_id=flags.policy_model_id,
      model_path=flags.policy_model_path,
      token=flags.policy_model_token,
      revision=flags.policy_model_revision,
      trust_remote_code=flags.policy_model_hf_trust_remote_code,
      use_flash_attention_2=flags.policy_model_use_flash,
      model_code_revision=flags.policy_model_code_revision,
      torch_dtype=flags.policy_model_torch_dtype,
      tokenizer_name_or_path=flags.policy_model_tokenizer_path,
  )
  return conf


def get_sft_training_config(flags):
  """Get training config from flags."""
  if flags.policy_model_gradient_checkpointing:
    gradient_checkpointing_kwargs = {
        "use_reentrant": flags.use_reentrant
    }
  else:
    gradient_checkpointing_kwargs = {}
  return ACTInitializationConfig(
      project_id=flags.project_id,
      location=flags.location,
      staging_bucket=flags.staging_bucket,
      machine_type=flags.machine_type,
      accelerator_type=flags.accelerator_type,
      bf16=flags.bf16,
      num_train_epochs=flags.num_train_epochs,
      seed=flags.seed,
      task=flags.task,
      do_eval=flags.do_eval,
      resume_from_checkpoint=flags.resume_from_checkpoint,
      output_dir=flags.output_dir,
      gradient_checkpointing=flags.policy_model_gradient_checkpointing,
      gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
      max_length=flags.max_length,
      is_preference=flags.is_preference_task,
      target_label=flags.target_label,
      icl_examples=flags.icl_examples,
      optim=flags.optim,
      learning_rate=flags.learning_rate,
      lr_scheduler_type=flags.lr_scheduler_type,
      per_device_train_batch_size=flags.per_device_train_batch_size,
      per_device_eval_batch_size=flags.per_device_eval_batch_size,
  )


def get_training_config(flags):
  """Get training config from flags."""
  if flags.policy_model_gradient_checkpointing:
    gradient_checkpointing_kwargs = {
        "use_reentrant": flags.use_reentrant
    }
  else:
    gradient_checkpointing_kwargs = {}
  return ACTConfig(
      project_id=flags.project_id,
      location=flags.location,
      staging_bucket=flags.staging_bucket,
      machine_type=flags.machine_type,
      accelerator_type=flags.accelerator_type,
      bf16=flags.bf16,
      num_train_epochs=flags.num_train_epochs,
      seed=flags.seed,
      task=flags.task,
      do_eval=flags.do_eval,
      resume_from_checkpoint=flags.resume_from_checkpoint,
      output_dir=flags.output_dir,
      beta=flags.beta,
      gradient_checkpointing=flags.policy_model_gradient_checkpointing,
      gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
      max_prompt_length=flags.max_prompt_length,
      max_length=flags.max_length,
      is_preference=flags.is_preference_task,
      target_label=flags.target_label,
      icl_examples=flags.icl_examples,
      optim=flags.optim,
      loss_type=flags.loss_type,
      sample_frequency=flags.sample_frequency,
      learning_rate=flags.learning_rate,
      lr_scheduler_type=flags.lr_scheduler_type,
      per_device_train_batch_size=flags.per_device_train_batch_size,
      per_device_eval_batch_size=flags.per_device_eval_batch_size,
  )


def get_data_config(flags):
  """Get data config from flags."""
  return DataConfig(
      train_path=flags.train_path,
      validation_path=flags.validation_path,
      train_preference_filename=flags.train_preference_filename,
      validation_preference_filename=flags.validation_preference_filename,
      truncation_side=flags.truncation_side,
      has_context_metadata=flags.has_context_metadata,
      eval_data=flags.eval_data,
      eval_result_output_path=flags.eval_result_output_path,
      eval_sample_output_path=flags.eval_sample_output_path,
      eval_max_input_length=flags.eval_max_input_length,
      preference_batch_generation=flags.preference_batch_generation,
  )


def get_action_model_config(flags):
  """Get policy model config from flags."""
  return ModelConfig(
      model_id=flags.action_model_id,
  )


def get_user_simulator_config(flags):
  """Get user simulator config from flags."""
  return ModelConfig(
      model_id=flags.simulator_model_id,
  )


def get_intent_model_config(flags):
  """Get intent model config from flags."""
  return ModelConfig(
      model_id=flags.intent_model_id,
  )


def get_config_from_json_path(path):
  """Returns a config object from a json path."""
  config = read_json(_DEFAULT_CONFIG)
  config.update(read_json(path))
  return get_config_from_dict(config)

def get_sft_config_from_json_path(path):
  """Returns a config object from a json path."""
  config = read_json(_DEFAULT_SFT_CONFIG)
  config.update(read_json(path))
  return get_sft_config_from_dict(config)

def get_sft_config_from_dict(config):
  """Returns a config object from a dictionary."""
  policy_model_cfg = HFModelConfig(**config["policy_model_config"])
  training_cfg = ACTInitializationConfig(**config["training_config"])
  preference_model_cfg = ModelConfig(**config["preference_model_config"])
  data_cfg = DataConfig(**config["data_config"])
  final_cfg = BaseInitializationConfig(
      policy_model_config=policy_model_cfg,
      training_config=training_cfg,
      preference_model_config=preference_model_cfg,
      data_config=data_cfg,
  )
  return final_cfg


def get_config_from_dict(config):
  """Returns a config object from a dictionary."""
  policy_model_cfg = HFModelConfig(**config["policy_model_config"])
  action_model_cfg = ModelConfig(**config["action_model_config"])
  user_simulator_cfg = ModelConfig(**config["user_simulator_config"])
  intent_model_cfg = ModelConfig(**config["intent_model_config"])
  training_cfg = ACTConfig(**config["training_config"])
  preference_model_cfg = ModelConfig(**config["preference_model_config"])
  data_cfg = DataConfig(**config["data_config"])
  final_cfg = BaseConfig(
      policy_model_config=policy_model_cfg,
      action_model_config=action_model_cfg,
      user_simulator_config=user_simulator_cfg,
      intent_model_config=intent_model_cfg,
      training_config=training_cfg,
      preference_model_config=preference_model_cfg,
      data_config=data_cfg,
  )
  return final_cfg
