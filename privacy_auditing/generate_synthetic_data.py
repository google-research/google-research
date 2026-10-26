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

"""Generates synthetic data from a finetuned model."""

from collections.abc import Sequence
import functools
import os
import random
import re

from absl import app
from absl import flags
from absl import logging
import bagz
import jax
import orbax.checkpoint as ocp
from privacy_auditing import audit_lib
from privacy_auditing import filesystem
from privacy_auditing import prepare_data as data_lib
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
    'Checkpoint path.',
    required=True,
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
_NUM_EXAMPLES_TO_GENERATE = flags.DEFINE_integer(
    'num_examples_to_generate',
    1000,
    'Number of examples to generate.',
)
_TEMPERATURE = flags.DEFINE_float(
    'temperature',
    1.0,
    'Temperature for sampling.',
)
_TOP_P = flags.DEFINE_float(
    'top_p',
    1.0,
    'Top-p for sampling.',
)
_MAX_GENERATION_STEPS = flags.DEFINE_integer(
    'max_generation_steps',
    1024,
    'Maximum number of steps for generation.',
)
_SAMPLING_BATCH_SIZE = flags.DEFINE_integer(
    'sampling_batch_size',
    16,
    'Batch size for sampling.',
)
_SAVE_STEPS = flags.DEFINE_integer(
    'save_steps',
    8192,
    'Number of examples to generate before saving a checkpoint shard.',
)
_MIN_FSDP_AXIS_SIZE = flags.DEFINE_integer(
    'min_fsdp_axis_size',
    1,
    'Minimum size of the FSDP axis. Increase this number to avoid HBM OOMs.',
)
_SEED = flags.DEFINE_integer(
    'seed',
    None,
    'Seed for the random number generator. If None, generation is '
    'non-deterministic.',
)


def num_batches():
  q, r = divmod(_NUM_EXAMPLES_TO_GENERATE.value, _SAMPLING_BATCH_SIZE.value)
  return (q + 1) if r else q


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _SAVE_STEPS.value % _SAMPLING_BATCH_SIZE.value != 0:
    raise ValueError(
        f'save_steps ({_SAVE_STEPS.value}) must be a multiple of '
        f'sampling_batch_size ({_SAMPLING_BATCH_SIZE.value}).'
    )

  filesystem.makedirs(_WORK_DIR.value)

  seed_file = f'{_WORK_DIR.value}/seed.txt'
  if _SEED.value is not None:
    seed = _SEED.value
  elif filesystem.exists(seed_file):
    with filesystem.open_file(seed_file, 'r') as f:
      seed = int(f.read())
  else:
    seed = random.randint(0, 2**31 - 1)
    with filesystem.open_file(seed_file, 'w') as f:
      f.write(str(seed))
    logging.info('Seed: %d', seed)

  logging.info('Devices: %s', jax.devices())
  # Use audit_sharding=True so the activation batch axis is not sharded over
  # the fsdp axis. Without this, qwix.apply_lora_to_model traces the model
  # with a tiny batch (size 2), which is indivisible by fsdp=64+ and raises
  # an IndivisibleError. The audit sharding config leaves act_btd batch axes
  # as None, keeping only weight sharding over fsdp/tp.
  model_config, pretrained_ckpt_path = (
      audit_lib.get_model_config_and_pretrained_ckpt_path(
          _MODEL_VERSION.value, audit_sharding=True
      )
  )

  if ocp.utils.is_pathways_backend():
    # Using Pathways as the backend.
    logging.info('Registering Pathways handlers for Orbax checkpointing')
    ocp.pathways.register_type_handlers()

  mesh = audit_lib.create_mesh(
      clip_axis_max_size=jax.device_count(),
      min_fsdp_axis_size=_MIN_FSDP_AXIS_SIZE.value,
  )
  logging.info('Mesh: %s', mesh)
  with mesh:
    finetuned_model = audit_lib.load_finetuned_model(
        model_config=model_config,
        mesh=mesh,
        finetuned_ckpt_path=_CKPT_PATH.value,
        pretrained_ckpt_path=pretrained_ckpt_path,
        lora_rank=_LORA_RANK.value,
        alpha_scale_factor=_ALPHA_SCALE_FACTOR.value,
    )

  tokenizer_initializer = functools.partial(
      tunix.generate.tokenizer_adapter.Tokenizer,
      tokenizer_path=params_lib.GEMMA3_TOKENIZER,
      add_bos=False,
      add_eos=False,
  )
  tokenizer = tokenizer_initializer()

  sampler = audit_lib.create_sampler(
      model=finetuned_model,
      tokenizer=tokenizer,
      model_config=model_config,
      cache_size=_MAX_GENERATION_STEPS.value + 128,  # prompt length is small
  )

  prompt = data_lib.get_input_prompt(_MODEL_VERSION.value.is_it)

  # Resume from existing shards. Order does not matter.
  shard_pattern = re.compile(r'shard_(\d+)\.bagz')
  all_found_files = filesystem.glob(f'{_WORK_DIR.value}/shard_*.bagz')
  existing_shards = sorted(
      f for f in all_found_files if shard_pattern.search(f)
  )

  generations = []
  if existing_shards:
    for shard_path in existing_shards:
      with bagz.BagReader(shard_path) as reader:
        generations.extend(reader)

    if len(generations) % _SAVE_STEPS.value != 0:
      raise ValueError(
          f'Number of examples in existing shards ({len(generations)}) must be '
          f'a multiple of save_steps ({_SAVE_STEPS.value}).'
      )

    logging.info(
        'Resumed %d examples from %d shards.',
        len(generations),
        len(existing_shards),
    )
  num_saved = len(generations)

  batch_idx = len(generations) // _SAMPLING_BATCH_SIZE.value
  total_batches = num_batches()

  while len(generations) < _NUM_EXAMPLES_TO_GENERATE.value:
    batch_size = min(
        _SAMPLING_BATCH_SIZE.value,
        _NUM_EXAMPLES_TO_GENERATE.value - len(generations),
    )
    logging.info(
        'Generating batch %d/%d of size %d.',
        batch_idx,
        total_batches,
        batch_size,
    )
    batch_result = sampler(
        input_strings=[prompt] * batch_size,
        max_generation_steps=_MAX_GENERATION_STEPS.value,
        temperature=_TEMPERATURE.value,
        top_p=_TOP_P.value,
        seed=seed + batch_idx,
    )

    generations.extend(
        tokenizer.decode(t.tolist()).encode('utf-8')
        for t in batch_result.tokens
    )

    # Save checkpoint shard periodically.
    if len(generations) - num_saved == _SAVE_STEPS.value:
      shard_idx = num_saved // _SAVE_STEPS.value
      shard_path = f'{_WORK_DIR.value}/shard_{shard_idx}.bagz'
      with bagz.BagWriter(shard_path) as writer:
        for item in generations[num_saved : num_saved + _SAVE_STEPS.value]:
          writer.write(item)
      existing_shards.append(shard_path)
      num_saved += _SAVE_STEPS.value

    batch_idx += 1

  output_path = f'{_WORK_DIR.value}/synthetic_data.bagz'
  logging.info('Writing %d examples to %s.', len(generations), output_path)
  with bagz.BagWriter(output_path) as writer:
    for item in generations:
      writer.write(item)

  # Clean up the shard files.
  for p in existing_shards:
    filesystem.remove(p)

  logging.info(
      'Finished writing %d examples to %s.', len(generations), output_path
  )


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
