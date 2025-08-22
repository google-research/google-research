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

from typing import Any, Optional, Union

from absl import flags
from act.config.base_config import BaseConfig, BaseInitializationConfig


def initialize_flags(
    config,
):
  ##########################################################################
  #                                                                        #
  #                             Config flags                               #
  #                                                                        #
  ##########################################################################

  flags.DEFINE_string(
      'config',
      '',
      'The path to the config file. If not empty use the config file instead of'
      ' using the flag values.',
  )

  flags.DEFINE_bool('debug', False, 'Debug.'),

  ##########################################################################
  #                                                                        #
  #                             Data flags                                 #
  #                                                                        #
  ##########################################################################

  flags.DEFINE_string(
      'train_path',
      config.data_config.train_path,
      'The path to the training data.',
  )

  flags.DEFINE_string(
      'validation_path',
      config.data_config.validation_path,
      'The path to the validation data.',
  )

  flags.DEFINE_string(
      'train_preference_filename',
      config.data_config.train_preference_filename,
      'Train preference filename.',
  )

  flags.DEFINE_string(
      'validation_preference_filename',
      config.data_config.validation_preference_filename,
      'Validation preference filename.',
  )

  flags.DEFINE_string(
      'truncation_side',
      config.data_config.truncation_side,
      'The side to truncate the data on.',
  )

  flags.DEFINE_bool(
      'has_context_metadata',
      config.data_config.has_context_metadata,
      'Whether the data has context metadata.',
  )

  flags.DEFINE_string(
      'eval_data', config.data_config.eval_data, 'Evaluation preference data.'
  ),

  flags.DEFINE_string(
      'eval_result_output_path',
      config.data_config.eval_result_output_path,
      'Evaluation result output path.',
  ),

  flags.DEFINE_string(
      'eval_sample_output_path',
      config.data_config.eval_sample_output_path,
      'Evaluation sample output path.',
  ),

  flags.DEFINE_integer(
      'eval_max_input_length',
      config.data_config.eval_max_input_length,
      'Evaluation max input length.',
  ),

  flags.DEFINE_bool(
      'preference_batch_generation',
      config.data_config.preference_batch_generation,
      'Generate preference data in a batch job',
  ),

  ##########################################################################
  #                                                                        #
  #                            Training flags                              #
  #                                                                        #
  ##########################################################################

  flags.DEFINE_string(
      'project_id',
      config.training_config.project_id,
      'The project id to use.',
  )

  flags.DEFINE_string(
      'location',
      config.training_config.location,
      'The location to use.',
  )

  flags.DEFINE_integer(
      'per_device_train_batch_size',
      config.training_config.per_device_train_batch_size,
      'The per device train batch size to use.',
  )

  flags.DEFINE_integer(
      'per_device_eval_batch_size',
      config.training_config.per_device_eval_batch_size,
      'The per device eval batch size to use.',
  )

  flags.DEFINE_string(
      'staging_bucket',
      config.training_config.staging_bucket,
      'The staging bucket to use.',
  )

  flags.DEFINE_string(
      'machine_type',
      config.training_config.machine_type,
      'The machine type to use.',
  )

  flags.DEFINE_string(
      'accelerator_type',
      config.training_config.accelerator_type,
      'The accelerator type to use.',
  )

  flags.DEFINE_bool(
      'bf16',
      config.training_config.bf16,
      'Whether to use bf16.',
  )

  flags.DEFINE_integer(
      'num_train_epochs',
      config.training_config.num_train_epochs,
      'The number of epochs to train for.',
  )

  flags.DEFINE_integer(
      'seed',
      config.training_config.seed,
      'The seed for the training.',
  )

  flags.DEFINE_string(
      'task',
      config.training_config.task,
      'The task type to use.',
  )

  flags.DEFINE_bool(
      'do_eval',
      config.training_config.do_eval,
      'Whether to do evaluation.',
  )

  flags.DEFINE_string(
      'resume_from_checkpoint',
      config.training_config.resume_from_checkpoint,
      'The checkpoint to resume from.',
  )

  flags.DEFINE_string(
      'output_dir',
      config.training_config.output_dir,
      'The output directory to use.',
  )

  flags.DEFINE_float(
      'beta',
      config.training_config.beta
      if hasattr(config.training_config, 'beta')
      else None,
      'The beta value for the DPO algorithm.',
  )

  flags.DEFINE_integer(
      'max_prompt_length',
      config.training_config.max_prompt_length
      if hasattr(config.training_config, 'max_prompt_length')
      else None,
      'The maximum prompt length.',
  )

  flags.DEFINE_integer(
      'max_length',
      config.training_config.max_length,
      'The maximum length of the data.',
  )

  flags.DEFINE_float(
      'learning_rate',
      config.training_config.learning_rate,
      'The learning rate to use.',
  )

  flags.DEFINE_string(
      'lr_scheduler_type',
      config.training_config.lr_scheduler_type,
      'The learning rate scheduler type to use.',
  )

  flags.DEFINE_bool(
      'is_preference_task',
      config.training_config.is_preference,
      'Whether the data is already preference data.',
  )

  flags.DEFINE_string(
      'target_label',
      config.training_config.target_label,
      'The target label for the data.',
  )

  flags.DEFINE_integer(
      'icl_examples',
      config.training_config.icl_examples,
      'The number of ICL examples to use.',
  )

  flags.DEFINE_string(
      'optim',
      config.training_config.optim,
      'The optimizer to use.',
  )

  flags.DEFINE_bool(
      'mixed_precision',
      config.training_config.mixed_precision,
      'Whether to use mixed precision.',
  )

  flags.DEFINE_string(
      'loss_type',
      config.training_config.loss_type
      if hasattr(config.training_config, 'loss_type')
      else None,
      'The type of loss: sigmoid is default , Hinge and IPO are also'
      ' supported.',
  )

  flags.DEFINE_integer(
      'sample_frequency',
      config.training_config.sample_frequency
      if hasattr(config.training_config, 'sample_frequency')
      else None,
      'The frequency to sample the data.',
  )

  flags.DEFINE_bool(
      'policy_model_gradient_checkpointing',
      config.training_config.gradient_checkpointing,
      'Whether to use gradient checkpointing for the policy model.',
  )

  flags.DEFINE_bool(
      'use_reentrant',
      config.training_config.gradient_checkpointing_kwargs['use_reentrant'],
      'Whether to use gradient checkpointing for the policy model.',
  )

  ##########################################################################
  #                                                                        #
  #                           Policy Model Flags                           #
  #                                                                        #
  ##########################################################################

  flags.DEFINE_string(
      'policy_model_id',
      config.policy_model_config.model_id,
      'The id of the policy model to use.',
  )

  flags.DEFINE_string(
      'policy_model_path',
      config.policy_model_config.model_path,
      'The path of the policy model to use.',
  )

  flags.DEFINE_string(
      'policy_model_token',
      config.policy_model_config.token,
      'The hugging face token of the policy model to use.',
  )

  flags.DEFINE_string(
      'policy_model_revision',
      config.policy_model_config.revision,
      'The revision of the policy model to use.',
  )

  flags.DEFINE_bool(
      'policy_model_hf_trust_remote_code',
      config.policy_model_config.trust_remote_code,
      'Whether to trust the remote for the policy model.',
  )

  flags.DEFINE_bool(
      'policy_model_use_flash',
      config.policy_model_config.use_flash_attention_2,
      'Whether to use flash attention 2for the policy model.',
  )

  flags.DEFINE_string(
      'policy_model_code_revision',
      config.policy_model_config.model_code_revision,
      'The HF model code revision of the policy model to use.',
  )

  flags.DEFINE_string(
      'policy_model_torch_dtype',
      config.policy_model_config.torch_dtype,
      'The torch dtype of the policy model to use.',
  )

  flags.DEFINE_string(
      'policy_model_tokenizer_path',
      config.policy_model_config.tokenizer_name_or_path,
      'The HF path to the tokenizer of the policy model to use.',
  )

  ##########################################################################
  #                                                                        #
  #                           Action Model Flags                           #
  #                                                                        #
  ##########################################################################

  flags.DEFINE_string(
      'action_model_id',
      config.action_model_config.model_id
      if hasattr(config, 'action_model_config')
      else None,
      'The id of the policy model to use.',
  )

  ##########################################################################
  #                                                                        #
  #                        Preference Model Flags                          #
  #                                                                        #
  ##########################################################################

  flags.DEFINE_string(
      'preference_model_id',
      config.preference_model_config.model_id,
      'The id of the policy model to use.',
  )

  ##########################################################################
  #                                                                        #
  #                          Simulator Model Flags                         #
  #                                                                        #
  ##########################################################################

  flags.DEFINE_string(
      'simulator_model_id',
      config.user_simulator_config.model_id
      if hasattr(config, 'user_simulator_config')
      else None,
      'The id of the simulator model to use.',
  )

  ##########################################################################
  #                                                                        #
  #                            Intent Model Flags                          #
  #                                                                        #
  ##########################################################################

  flags.DEFINE_string(
      'intent_model_id',
      config.intent_model_config.model_id
      if hasattr(config, 'intent_model_config')
      else None,
      'The id of the intent model to use.',
  )


FLAGS = flags.FLAGS
