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

"""Main training script."""

from collections.abc import Sequence
import functools
import os
import random

from absl import app
from absl import flags
from absl import logging
import dp_accounting
from flax import nnx
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from privacy_auditing import audit_lib
from privacy_auditing import dp_trainer
from privacy_auditing import prepare_data
import qwix
import tunix


# Use the GEMMA3_TOKENIZER path from environment or config:
class _ParamsLib:
  GEMMA3_TOKENIZER = os.environ.get(
      'GEMMA3_TOKENIZER_PATH', '<PATH_TO_GEMMA3_TOKENIZER>'
  )


params_lib = _ParamsLib()

# the privatizer_state is not an explicit argument to train_step, but rather
# a closed over constant. This can lead to large compiled programs and long
# compilation times without this config flag below.
jax.config.update('jax_use_simplified_jaxpr_constants', True)


_DEFAULT_EXPECTED_BATCH_SIZE_RATIO = 0.85

# Flags for the data loading.
_DATASET_NAME = flags.DEFINE_string(
    'dataset_name',
    'enron',
    'Name of the dataset to use for training.',
)

_MAX_TARGET_LENGTH = flags.DEFINE_integer(
    'max_target_length',
    1024,
    'Maximum length of the target sequence.',
)

_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    'train_batch_size',
    64,
    'Train batch size. With DP, this is the batch size including padding '
    'examples, and expected_batch_size controls the number of non-padding '
    'examples per batch.',
)

_VALIDATION_BATCH_SIZE = flags.DEFINE_integer(
    'validation_batch_size',
    64,
    'Validation batch size. Set to 0 to disable validation.',
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate',
    None,
    'Learning rate.',
)

_MAX_STEPS = flags.DEFINE_integer(
    'max_steps',
    10000,
    'Maximum number of training steps.',
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
    10,
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
    'Number of training epochs. If DP is enabled, this flag is ignored.',
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

_CANARY_PATH = flags.DEFINE_string(
    'canary_path',
    None,
    'Base path to the canary dataset (without the .seen/.unseen.arrayrecord'
    ' suffix). If None, no canaries are used.',
)

_CANARY_REPS = flags.DEFINE_integer(
    'canary_reps',
    1,
    'Number of canary repetitions to use.',
)

_NUM_UNIQUE_CANARIES = flags.DEFINE_integer(
    'num_unique_canaries',
    3000,
    'Number of unique seen/unseen canaries.',
)


# DP Flags.

_ENABLE_DP = flags.DEFINE_bool(
    'enable_dp',
    False,
    'Whether to enable DP. This turns on Poisson sampling, gradient clipping '
    'and noise addition.',
)

_CLIP_NORM = flags.DEFINE_float(
    'clip_norm',
    1.0,
    'Clip norm to use for DP.',
)

_NOISE_MULTIPLIER = flags.DEFINE_float(
    'noise_multiplier',
    None,
    'Noise multiplier to use for DP. If not None, epsilon and delta will be '
    'ignored. Otherwise, it will be automatically calibrated based on epsilon '
    'and delta',
)

_EPSILON = flags.DEFINE_float(
    'epsilon',
    None,
    'Epsilon to use for DP. Only used to calibrate noise_multiplier if it is '
    'not manually set.',
)

_DELTA = flags.DEFINE_float(
    'delta',
    None,
    'Delta to use for DP. Only used to calibrate noise_multiplier if it is '
    'not manually set.',
)

_EXPECTED_BATCH_SIZE = flags.DEFINE_float(
    'expected_batch_size',
    None,
    'Expected batch size for Poisson sampling. If DP is enabled and this is '
    f'None, it will be set to {_DEFAULT_EXPECTED_BATCH_SIZE_RATIO} * '
    'train_batch_size.',
)

_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples',
    465_661,
    'Number of examples to use for training. Default is the number of '
    'examples in the Enron dataset train split. Used for DP accounting.',
)

_MICROBATCH_SIZE = flags.DEFINE_integer(
    'microbatch_size',
    64,
    'Microbatch size to use when training with DP. Smaller microbatch sizes '
    'require less memory, but more sequential computation. Must evenly divide '
    'train_batch_size.',
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

# Tuned Learning Rates.

LEARNING_RATE_LORA = 0.0005
LEARNING_RATE_FULL = 0.0001


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
  expected_batch_size = None
  if _ENABLE_DP.value:
    expected_batch_size = (
        _DEFAULT_EXPECTED_BATCH_SIZE_RATIO * _TRAIN_BATCH_SIZE.value
        if _EXPECTED_BATCH_SIZE.value is None
        else _EXPECTED_BATCH_SIZE.value
    )
    poisson_config = prepare_data.PoissonConfig(
        training_steps=_MAX_STEPS.value,
        expected_batch_size=expected_batch_size,
        seed=dataset_seed,
    )
  else:
    if _EXPECTED_BATCH_SIZE.value is not None:
      raise ValueError(
          'expected_batch_size must be None when DP is disabled. '
          f'Got {_EXPECTED_BATCH_SIZE.value=}.'
      )
    poisson_config = None

  tokenizer_initializer = functools.partial(
      tunix.generate.tokenizer_adapter.Tokenizer,
      tokenizer_path=params_lib.GEMMA3_TOKENIZER,
      add_bos=False,
      add_eos=False,
  )
  train_ds, validation_ds = prepare_data.create_datasets(
      train_batch_size=_TRAIN_BATCH_SIZE.value,
      validation_batch_size=_VALIDATION_BATCH_SIZE.value,
      num_epochs=_NUM_TRAINING_EPOCHS.value,
      tokenizer_initializer=tokenizer_initializer,
      max_seq_len=_MAX_TARGET_LENGTH.value,
      num_unique_canaries=_NUM_UNIQUE_CANARIES.value,
      canary_path=_CANARY_PATH.value,
      canary_reps=_CANARY_REPS.value,
      poisson_config=poisson_config,
      it_formatting=_MODEL_VERSION.value.is_it,
      data_shuffle_seed=dataset_seed,
  )

  # Allow `clip` axis to be as big as the batch, or microbatch.
  if _ENABLE_DP.value:
    if _MICROBATCH_SIZE.value is not None and _MICROBATCH_SIZE.value > 0:
      clip_axis_max_size = _MICROBATCH_SIZE.value
    else:
      clip_axis_max_size = _TRAIN_BATCH_SIZE.value
  else:
    clip_axis_max_size = _TRAIN_BATCH_SIZE.value
  try:
    # Put as many devices as possible on the clip axis, without exceeding batch
    # size, while keeping the FSDP axis at least large enough to prevent OOMs.
    mesh = audit_lib.create_mesh(
        clip_axis_max_size, min_fsdp_axis_size=_MIN_FSDP_AXIS_SIZE.value
    )
  except ValueError as e:
    logging.warn(
        'Mesh creation failed. If running with DP, ensure that the number of '
        'devices is a multiple of microbatch_size.'
    )
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
    eval_steps = _SAVE_INTERVAL_STEPS.value
  else:
    save_decision_policy = (
        ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
            minimum_interval_secs=_SAVE_INTERVAL_SECS.value
        )
    )
    eval_steps = 500
  config = tunix.sft.peft_trainer.TrainingConfig(
      eval_every_n_steps=eval_steps,
      max_steps=_MAX_STEPS.value,
      checkpoint_root_directory=(_WORK_DIR.value),
      checkpointing_options=ocp.CheckpointManagerOptions(
          save_decision_policy=save_decision_policy,
          max_to_keep=_MAX_CHECKPOINTS_TO_KEEP.value,
          async_options=ocp.AsyncOptions(timeout_secs=600),
      ),
      metrics_logging_options=tunix.sft.metrics_logger.MetricsLoggerOptions(
          flush_every_n_steps=eval_steps,
          log_dir=os.path.join(_WORK_DIR.value, 'metrics'),
      ),
      data_sharding_axis=('clip',),
  )
  if _LEARNING_RATE.value is None:
    if _LORA_RANK.value:
      learning_rate = LEARNING_RATE_LORA
    else:
      learning_rate = LEARNING_RATE_FULL
  else:
    learning_rate = _LEARNING_RATE.value
  if _ENABLE_DP.value:
    if _TRAIN_BATCH_SIZE.value % _MICROBATCH_SIZE.value != 0:
      raise ValueError(
          'train_batch_size must be a multiple of microbatch_size. Got '
          f'train_batch_size: {_TRAIN_BATCH_SIZE.value}, '
          f'microbatch_size: {_MICROBATCH_SIZE.value}.'
      )
    if _NOISE_MULTIPLIER.value is None:
      noise_multiplier = dp_accounting.calibrate_dp_mechanism(
          dp_accounting.pld.PLDAccountant,
          lambda nm: dp_accounting.SelfComposedDpEvent(
              dp_accounting.PoissonSampledDpEvent(
                  expected_batch_size / _NUM_EXAMPLES.value,
                  dp_accounting.GaussianDpEvent(nm),
              ),
              _MAX_STEPS.value,
          ),
          _EPSILON.value,
          _DELTA.value,
      )
    else:
      if _EPSILON.value is not None or _DELTA.value is not None:
        raise ValueError(
            'epsilon and delta must be None when noise_multiplier is manually '
            f'set. Got epsilon: {_EPSILON.value}, delta: {_DELTA.value}.'
        )
      noise_multiplier = _NOISE_MULTIPLIER.value

    trainer = dp_trainer.DPPeftTrainer(
        model=transformer,
        optimizer=optax.adamw(learning_rate),
        training_config=config,
        clip_norm=_CLIP_NORM.value,
        noise_multiplier=noise_multiplier,
        microbatch_size=_MICROBATCH_SIZE.value,
        expected_batch_size=expected_batch_size,
        seed=noise_seed,
        clip_axis_name='clip',
    )
  else:
    trainer = tunix.sft.peft_trainer.PeftTrainer(
        model=transformer,
        optimizer=optax.adamw(learning_rate),
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
    trainer.train(train_ds, validation_ds, cache_nnx_graph=True)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
