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

"""Evaluation script: computes held-out loss and perplexity from a checkpoint.

Loads a finetuned (or LoRA-adapted) model from a checkpoint and runs a full
pass over the held-out validation split of the Enron dataset using
PeftTrainer._run_eval, reporting cross-entropy loss and perplexity.

Results are written to ``{work_dir}/eval_metrics.json``.
"""

from collections.abc import Sequence
import functools
import json
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
import jax
import optax
import orbax.checkpoint as ocp
from privacy_auditing import audit_lib
from privacy_auditing import filesystem
from privacy_auditing import prepare_data
import tunix


# Use the GEMMA3_TOKENIZER path from environment or config:
class _ParamsLib:
  GEMMA3_TOKENIZER = os.environ.get(
      'GEMMA3_TOKENIZER_PATH', '<PATH_TO_GEMMA3_TOKENIZER>'
  )


params_lib = _ParamsLib()

_CKPT_PATH = flags.DEFINE_string(
    'ckpt_path',
    None,
    'Path to the finetuned model checkpoint.',
    required=True,
)

_WORK_DIR = flags.DEFINE_string(
    'work_dir',
    None,
    'Working directory.  eval_metrics.json will be written here.',
    required=True,
)

_MODEL_VERSION = flags.DEFINE_enum_class(
    'model_version',
    None,
    audit_lib.ModelVersion,
    'Version of the model used.',
    required=True,
)

_LORA_RANK = flags.DEFINE_integer(
    'lora_rank',
    None,
    'Rank of the LoRA adapter.  If None, full finetuning assumed.',
)

_ALPHA_SCALE_FACTOR = flags.DEFINE_float(
    'alpha_scale_factor',
    1.0,
    'Alpha scale factor for LoRA.',
)

_VALIDATION_BATCH_SIZE = flags.DEFINE_integer(
    'validation_batch_size',
    64,
    'Batch size used when iterating over the validation split.',
)

_MAX_TARGET_LENGTH = flags.DEFINE_integer(
    'max_target_length',
    1024,
    'Maximum sequence length (must match the value used during training).',
)

_MIN_FSDP_AXIS_SIZE = flags.DEFINE_integer(
    'min_fsdp_axis_size',
    1,
    'Minimum FSDP axis size.  Increase to avoid HBM OOMs on large models.',
)


def _run_eval(
    model_version,
    ckpt_path,
    work_dir,
    lora_rank,
    alpha_scale_factor,
    validation_batch_size,
    max_target_length,
    min_fsdp_axis_size,
):
  """Runs evaluation over the held-out validation split of the Enron dataset.

  Args:
    model_version: The model version to evaluate.
    ckpt_path: The path to the finetuned model checkpoint.
    work_dir: The working directory to write the results to.
    lora_rank: The rank of the LoRA adapter.
    alpha_scale_factor: The alpha scale factor for LoRA.
    validation_batch_size: The batch size to use for validation.
    max_target_length: The maximum sequence length to use for validation.
    min_fsdp_axis_size: The minimum FSDP axis size to use for validation.
  """
  model_config, pretrained_ckpt_path = (
      audit_lib.get_model_config_and_pretrained_ckpt_path(
          model_version,
          audit_sharding=True,
      )
  )

  # clip_axis_max_size=1 puts all chips on FSDP, avoiding HBM OOMs from
  # large logits tensors (bf16[batch, seq, vocab]) on 12B models.
  mesh = audit_lib.create_mesh(
      clip_axis_max_size=1,
      min_fsdp_axis_size=min_fsdp_axis_size,
  )
  logging.info('Mesh: %s', mesh)

  with mesh:
    model = audit_lib.load_finetuned_model(
        model_config=model_config,
        mesh=mesh,
        finetuned_ckpt_path=ckpt_path,
        pretrained_ckpt_path=pretrained_ckpt_path,
        lora_rank=lora_rank,
        alpha_scale_factor=alpha_scale_factor,
    )

  tokenizer_initializer = functools.partial(
      tunix.generate.tokenizer_adapter.Tokenizer,
      tokenizer_path=params_lib.GEMMA3_TOKENIZER,
      add_bos=False,
      add_eos=False,
  )

  # create_datasets returns (train_ds, validation_ds); we only need validation.
  _, validation_ds = prepare_data.create_datasets(
      train_batch_size=validation_batch_size,
      validation_batch_size=validation_batch_size,
      num_epochs=1,
      tokenizer_initializer=tokenizer_initializer,
      max_seq_len=max_target_length,
      num_unique_canaries=0,  # no canaries needed for eval
      canary_path=None,
      it_formatting=model_version.is_it,
  )

  # Build a minimal PeftTrainer — no checkpointing, no metrics file output.
  # data_sharding_axis='fsdp' (no clip axis needed for eval).
  config = tunix.sft.peft_trainer.TrainingConfig(
      eval_every_n_steps=1,
      max_steps=0,
      data_sharding_axis=('fsdp',),
  )
  trainer = tunix.sft.peft_trainer.PeftTrainer(
      model=model,
      optimizer=optax.sgd(0.0),  # unused; eval does not update weights
      training_config=config,
  )

  # Matches gen_model_input_fn in train.py.
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

  # _run_eval handles nnx.jit, stop_gradient, sharding, and loss aggregation.
  logging.info('Starting evaluation over the held-out validation split ...')
  _, eval_step = trainer.jit_train_and_eval_step(cache_nnx_graph=True)
  with mesh:
    trainer._run_eval(validation_ds, eval_step)  # pylint: disable=protected-access

  mean_loss = float(trainer.metrics_logger.get_metric('', 'loss', 'eval'))
  perplexity = float(
      trainer.metrics_logger.get_metric('', 'perplexity', 'eval')
  )

  logging.info('=== Evaluation complete ===')
  logging.info('  Mean loss (nats)  : %.6f', mean_loss)
  logging.info('  Perplexity        : %.4f', perplexity)

  metrics = {
      'ckpt_path': ckpt_path,
      'mean_loss': mean_loss,
      'perplexity': perplexity,
  }

  filesystem.makedirs(work_dir)
  output_path = os.path.join(work_dir, 'eval_metrics.json')
  with filesystem.open_file(output_path, 'w') as f:
    json.dump(metrics, f, indent=2)
  logging.info('Wrote eval metrics to %s', output_path)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Devices: %s', jax.devices())

  if ocp.utils.is_pathways_backend():
    logging.info('Registering Pathways handlers for Orbax checkpointing')
    ocp.pathways.register_type_handlers()

  _run_eval(
      _MODEL_VERSION.value,
      _CKPT_PATH.value,
      _WORK_DIR.value,
      _LORA_RANK.value,
      _ALPHA_SCALE_FACTOR.value,
      _VALIDATION_BATCH_SIZE.value,
      _MAX_TARGET_LENGTH.value,
      _MIN_FSDP_AXIS_SIZE.value,
  )


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
