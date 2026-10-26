# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Synthetic data training script."""

from collections.abc import Sequence
import functools
import os
import random

from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from privacy_auditing import audit_lib
from privacy_auditing import prepare_data
import qwix
import tunix


# Use the GEMMA3_TOKENIZER path from environment or config:
class _ParamsLib:
  GEMMA3_TOKENIZER = os.environ.get(
      'GEMMA3_TOKENIZER_PATH', '<PATH_TO_GEMMA3_TOKENIZER>'
  )


params_lib = _ParamsLib()

# Flags for the data loading.
_SYNTHETIC_DATA_DIR = flags.DEFINE_string(
    'synthetic_data_dir',
    None,
    'Path to the synthetic data .bagz file.',
    required=True,
)

_MAX_TARGET_LENGTH = flags.DEFINE_integer(
    'max_target_length',
    1024,
    'Maximum length of the target sequence.',
)

_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    'train_batch_size',
    64,
    'Train batch size.',
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    None,
    'Learning rate.',
)

_SAVE_INTERVAL_SECS = flags.DEFINE_integer(
    'save_interval_secs',
    180,
    'Checkpoint save interval in seconds.',
)

_SAVE_INTERVAL_STEPS = flags.DEFINE_integer(
    'save_interval_steps',
    None,
    'Checkpoint save interval in steps. If None, save based on time interval.',
)

_MAX_CHECKPOINTS_TO_KEEP = flags.DEFINE_integer(
    'max_checkpoints_to_keep',
    1,
    'Maximum number of checkpoints to keep.',
)

_WORK_DIR = flags.DEFINE_string(
    'work_dir',
    None,
    'Working directory. All artifacts will be saved here.',
    required=True,
)

_MODEL_VERSION = flags.DEFINE_enum_class(
    'model_version',
    None,
    audit_lib.ModelVersion,
    'Version of the model used.',
    required=True,
)

_NUM_TRAINING_EPOCHS = flags.DEFINE_integer(
    'num_training_epochs',
    1,
    'Number of training epochs.',
)

_LORA_RANK = flags.DEFINE_integer(
    'lora_rank',
    None,
    'Rank of the LoRA adapter. If None, full finetuning is used.',
)

_ALPHA_SCALE_FACTOR = flags.DEFINE_float(
    'alpha_scale_factor',
    1.0,
    'Alpha scale factor for LoRA.',
)
_MIN_FSDP_AXIS_SIZE = flags.DEFINE_integer(
    'min_fsdp_axis_size',
    1,
    'Minimum size of the FSDP axis. Must evenly divide train_batch_size, or '
    'microbatch_size if using DP. Increase this number to avoid HBM OOMs.',
)

_SEED = flags.DEFINE_integer(
    'seed',
    None,
    'Random seed for model initialization, data shuffling, and DP noise.  '
    'If None, a random seed will be generated.',
)

_USE_LR_SCHEDULE = flags.DEFINE_bool(
    'use_lr_schedule',
    False,
    'Whether to use a learning rate schedule.',
)

# 12B LoRA 64 tuned Learning Rate.

LEARNING_RATE_LORA = 0.0002


def _create_sharded_model(mesh):
  """Creates the sharded model from the pretrained checkpoint."""
  model_config, ckpt_path = audit_lib.get_model_config_and_pretrained_ckpt_path(
      _MODEL_VERSION.value
  )
  # NOTE: you must implement `audit_lib.load_pretrained_model` for your
  # environment before running.
  return audit_lib.load_pretrained_model(model_config, mesh, ckpt_path)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  seed = _SEED.value
  if seed is None:
    seed = random.randint(0, 2**31 - 1)
    logging.info('No seed provided. Generated primary random seed: %d', seed)
  else:
    logging.info('Using provided primary random seed: %d', seed)

  rng = np.random.default_rng(seed)
  seeds = rng.integers(2**31 - 1, size=3).tolist()
  init_params_seed, dataset_seed, noise_seed = seeds
  logging.info(
      'init_params_seed: %s, dataset_seed: %s, noise_seed: %s',
      init_params_seed,
      dataset_seed,
      noise_seed,
  )

  if ocp.utils.is_pathways_backend():
    # Using Pathways as the backend.
    logging.info('Registering Pathways handlers for Orbax checkpointing')
    ocp.pathways.register_type_handlers()
  logging.info('Devices: %s', jax.devices())

  # Creates dataset iterators.
  logging.info('Number of training epochs: %s', _NUM_TRAINING_EPOCHS.value)

  tokenizer_initializer = functools.partial(
      tunix.generate.tokenizer_adapter.Tokenizer,
      tokenizer_path=params_lib.GEMMA3_TOKENIZER,
      add_bos=False,
      add_eos=False,
  )
  train_ds, num_steps = prepare_data.create_synthetic_dataset(
      file_path=_SYNTHETIC_DATA_DIR.value,
      train_batch_size=_TRAIN_BATCH_SIZE.value,
      num_epochs=_NUM_TRAINING_EPOCHS.value,
      tokenizer_initializer=tokenizer_initializer,
      max_seq_len=_MAX_TARGET_LENGTH.value,
      it_formatting=_MODEL_VERSION.value.is_it,
  )

  # Allow `clip` axis to be as big as the batch.
  clip_axis_max_size = _TRAIN_BATCH_SIZE.value
  try:
    # Put as many devices as possible on the clip axis, without exceeding batch
    # size, while keeping the FSDP axis at least large enough to prevent OOMs.
    mesh = audit_lib.create_mesh(
        clip_axis_max_size, min_fsdp_axis_size=_MIN_FSDP_AXIS_SIZE.value
    )
  except ValueError as e:
    logging.warn('Mesh creation failed.')
    raise e
  logging.info('Mesh: %s', mesh)

  with mesh:
    transformer = _create_sharded_model(mesh)

  if _LORA_RANK.value:
    lora_provider = qwix.LoraProvider(
        module_path='.*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj',
        rank=_LORA_RANK.value,
        alpha=_ALPHA_SCALE_FACTOR.value * _LORA_RANK.value,
    )
    model_input = transformer.get_model_input()
    rngs_for_lora = nnx.Rngs(init_params_seed)
    transformer = qwix.apply_lora_to_model(
        transformer, lora_provider, **model_input, rngs=rngs_for_lora
    )

  # Creates the trainer.
  if _SAVE_INTERVAL_STEPS.value is not None:
    save_decision_policy = ocp.checkpoint_managers.FixedIntervalPolicy(
        interval=_SAVE_INTERVAL_STEPS.value
    )
    save_steps = _SAVE_INTERVAL_STEPS.value
  else:
    save_decision_policy = (
        ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
            minimum_interval_secs=_SAVE_INTERVAL_SECS.value
        )
    )
    save_steps = 500
  config = tunix.sft.peft_trainer.TrainingConfig(
      eval_every_n_steps=save_steps,
      max_steps=num_steps,
      checkpoint_root_directory=(_WORK_DIR.value),
      checkpointing_options=ocp.CheckpointManagerOptions(
          save_decision_policy=save_decision_policy,
          max_to_keep=_MAX_CHECKPOINTS_TO_KEEP.value,
          async_options=ocp.AsyncOptions(timeout_secs=600),
      ),
      metrics_logging_options=tunix.sft.metrics_logger.MetricsLoggerOptions(
          flush_every_n_steps=save_steps,
          log_dir=os.path.join(_WORK_DIR.value, 'metrics'),
      ),
      # pyrefly: ignore[bad-argument-type]
      data_sharding_axis=(('clip', 'fsdp'),),
  )

  if _LEARNING_RATE.value is None:
    if _LORA_RANK.value:
      learning_rate = LEARNING_RATE_LORA
    else:
      raise ValueError(
          'learning_rate must be set when full finetuning is used.'
      )
  else:
    learning_rate = _LEARNING_RATE.value

  if _USE_LR_SCHEDULE.value:
    warmup_steps = min(100, max(0, num_steps - 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
        end_value=learning_rate * 0.1,
    )
  else:
    schedule = learning_rate

  trainer = tunix.sft.peft_trainer.PeftTrainer(
      model=transformer,
      optimizer=optax.adamw(schedule),
      training_config=config,
  )

  # Creates the model input function.
  tokenizer = tokenizer_initializer()
  pad_id = tokenizer.pad_id()

  def gen_model_input_fn(x):
    pad_mask = x.input_tokens != pad_id
    return {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': tunix.rl.common.build_positions_from_mask(pad_mask),
        'attention_mask': tunix.rl.common.make_causal_attn_mask(pad_mask),
    }

  trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

  with mesh:
    trainer.train(train_ds, cache_nnx_graph=True)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
