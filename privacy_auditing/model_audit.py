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

"""Model auditing script."""

import collections
from collections.abc import Sequence
import functools
import json
import math
import os

from absl import app
from absl import flags
from absl import logging
import editdistance
from gemma.models.gemma3 import model as model_lib
import grain
import jax
import jax.numpy as jnp
import numpy as np
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

_CANARY_PATH = flags.DEFINE_string(
    'canary_path',
    None,
    'Base path to the canary dataset (without the .seen/.unseen.arrayrecord'
    ' suffix).',
    required=True,
)

_NUM_UNIQUE_CANARIES = flags.DEFINE_integer(
    'num_unique_canaries',
    3000,
    'Number of unique seen/unseen canaries.',
)

_MAX_TARGET_LENGTH = flags.DEFINE_integer(
    'max_target_length',
    1024,
    'Maximum length of the target sequence.',
)

_RECONS_SUFFIX_LEN = flags.DEFINE_integer(
    'recons_suffix_len',
    10,
    'Suffix length for reconstruction metrics.',
)

_CANARY_LEN = flags.DEFINE_integer(
    'canary_len',
    60,
    'Length of canary examples.',
)

_SAMPLING_BATCH_SIZE = flags.DEFINE_integer(
    'sampling_batch_size',
    64,
    'Batch size for sampling.',
)

_SCORING_BATCH_SIZE = flags.DEFINE_integer(
    'scoring_batch_size',
    64,
    'Number of canaries scored per forward pass during log-likelihood scoring.',
)

_ALPHA_SCALE_FACTOR = flags.DEFINE_float(
    'alpha_scale_factor',
    1.0,
    'Alpha scale factor for LoRA.',
)

_MIN_FSDP_AXIS_SIZE = flags.DEFINE_integer(
    'min_fsdp_axis_size',
    1,
    'Minimum size of the FSDP axis. Increase this number to avoid HBM OOMs.'
    ' For 12B models, set to 32 to fully shard weights. The audit sharding'
    ' config leaves activation batch axes un-sharded, so this does not cause'
    ' IndivisibleErrors when scoring single canaries.',
)


def _next_power_of_2(x):
  """Returns the smallest power of 2 that is >= x."""
  if x <= 1:
    return 1
  return int(pow(2, int(math.ceil(math.log2(x)))))


def _log_score_stats(score_str, scores):
  """Logs score stats for a given suffix length."""
  logging.info('%s min score: %f', score_str, np.min(scores))
  logging.info('%s max score: %f', score_str, np.max(scores))
  logging.info('%s mean score: %f', score_str, np.mean(scores))
  logging.info('%s median score: %f', score_str, np.median(scores))
  logging.info('%s std score: %f', score_str, np.std(scores))
  filesystem.makedirs(_WORK_DIR.value)
  scores_path = f'{_WORK_DIR.value}/{score_str}_raw_scores.npy'
  with filesystem.open_file(scores_path, 'wb') as f:
    np.save(f, scores)


def _compute_reconstruction_counts(
    model_config,
    model,
    tokenizer,
    seen_canaries,
    unseen_canaries,
):
  """Computes the reconstruction lifts for the given model."""

  sampler = audit_lib.create_sampler(
      model=model,
      tokenizer=tokenizer,
      model_config=model_config,
      cache_size=512,
  )

  # Reconstruct the last ten tokens that are not one of these.
  disallowed_tokens = set()
  for t in ['<end_of_turn>', '\n', '\n\n']:
    disallowed_tokens.update(tokenizer.encode(t))
  disallowed_tokens.add(tokenizer.eos_id())
  logging.info('Disallowed tokens: %s', disallowed_tokens)

  def split(prompt, canary):
    canary = canary[: _CANARY_LEN.value]
    while canary.size > 0 and canary[-1] in disallowed_tokens:
      canary = canary[:-1]
    if len(canary) < _RECONS_SUFFIX_LEN.value:
      logging.fatal(
          'Canary suffix is too short: %d < %d',
          len(canary),
          _RECONS_SUFFIX_LEN.value,
      )
    prefix_tokens = jnp.concatenate(
        [prompt, canary[: -_RECONS_SUFFIX_LEN.value]]
    )
    suffix_tokens = canary[-_RECONS_SUFFIX_LEN.value :]
    prefix = tokenizer.decode(prefix_tokens.tolist())
    return prefix, suffix_tokens.tolist()

  seen_canaries_list = list(seen_canaries)
  unseen_canaries_list = list(unseen_canaries)
  all_canaries = seen_canaries_list + unseen_canaries_list
  n_seen = len(seen_canaries_list)
  n_unseen = len(unseen_canaries_list)
  assert n_seen == n_unseen

  prefixes, all_targets = zip(
      *[split(prompt, canary) for prompt, canary in all_canaries]
  )

  logging.info('Generating suffixes for %d canaries.', len(all_canaries))
  predictions = []
  batch_start = 0
  while batch_start < len(prefixes):
    batch_end = min(batch_start + _SAMPLING_BATCH_SIZE.value, len(prefixes))
    batch_result = sampler(
        input_strings=prefixes[batch_start:batch_end],
        max_generation_steps=_RECONS_SUFFIX_LEN.value,
        forbidden_tokens=[tokenizer.eos_id()],
    )
    predictions.extend([t.tolist() for t in batch_result.tokens])
    batch_start = batch_end

  seen_targets = all_targets[:n_seen]
  unseen_targets = all_targets[n_seen:]
  seen_predictions = predictions[:n_seen]
  unseen_predictions = predictions[n_seen:]

  # Counter to see how many of the predictions are shorter than the suffix len.
  length_counter = collections.Counter()

  def get_recons_counts(targets, predictions):
    # recons_count_at_edit_distance[d] will store the number of reconstructions
    # with an edit distance <= d.
    recons_count_at_edit_distance = np.zeros(
        _RECONS_SUFFIX_LEN.value + 1, dtype=np.int32
    )
    for target, prediction in zip(targets, predictions):
      length_counter[f'prediction length {len(prediction)}'] += 1
      edit_distance = editdistance.distance(target, prediction)
      recons_count_at_edit_distance[edit_distance:] += 1
    return recons_count_at_edit_distance

  seen_recons_counts = get_recons_counts(seen_targets, seen_predictions)
  unseen_recons_counts = get_recons_counts(unseen_targets, unseen_predictions)

  logging.info('Canary reconstruction length stats:')
  for key, value in length_counter.items():
    logging.info('%s: %d', key, value)

  return seen_recons_counts, unseen_recons_counts


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Devices: %s', jax.devices())

  model_config, pretrained_ckpt_path = (
      audit_lib.get_model_config_and_pretrained_ckpt_path(
          _MODEL_VERSION.value,
          audit_sharding=True,
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
    # The finetuned model (the one being audited) is restored from --ckpt_path.
    finetuned_model = audit_lib.load_finetuned_model(
        model_config=model_config,
        mesh=mesh,
        finetuned_ckpt_path=_CKPT_PATH.value,
        pretrained_ckpt_path=pretrained_ckpt_path,
        lora_rank=_LORA_RANK.value,
        alpha_scale_factor=_ALPHA_SCALE_FACTOR.value,
    )
    # The base model is the pretrained model before finetuning; it is used as
    # the reference when scoring canaries. NOTE: you must implement
    # `audit_lib.load_pretrained_model` for your environment before running.
    base_model = audit_lib.load_pretrained_model(
        model_config, mesh, pretrained_ckpt_path
    )

  tokenizer_initializer = functools.partial(
      tunix.generate.tokenizer_adapter.Tokenizer,
      tokenizer_path=params_lib.GEMMA3_TOKENIZER,
      add_bos=False,
      add_eos=False,
  )
  seen_canaries, unseen_canaries = data_lib.create_audit_datasets(
      tokenizer_initializer=tokenizer_initializer,
      canary_path=_CANARY_PATH.value,
      max_seq_len=_MAX_TARGET_LENGTH.value,
      it_formatting=_MODEL_VERSION.value.is_it,
      num_unique_canaries=_NUM_UNIQUE_CANARIES.value,
  )

  def score_canaries_in_batches(canaries):
    """Scores canaries in batches and returns a flat list of scores."""
    canary_list = list(canaries)
    scores = []
    batch_size = _SCORING_BATCH_SIZE.value
    for start in range(0, len(canary_list), batch_size):
      batch = canary_list[start : start + batch_size]
      batch_scores = audit_lib.score_canaries_batch(
          batch,
          base_model=base_model,
          finetuned_model=finetuned_model,
          canary_len=_CANARY_LEN.value,
      )
      scores.extend(jax.device_get(batch_scores).tolist())
    return scores

  logging.info('Starting to score seen canaries with full suffix.')
  seen_canary_scores = score_canaries_in_batches(seen_canaries)
  logging.info('Scored %d seen canaries.', len(seen_canary_scores))

  _log_score_stats('seen', seen_canary_scores)

  logging.info('Starting to score unseen canaries with full suffix.')
  unseen_canary_scores = score_canaries_in_batches(unseen_canaries)
  logging.info('Scored %d unseen canaries.', len(unseen_canary_scores))

  _log_score_stats('unseen', unseen_canary_scores)

  # Compute reconstruction counts.
  tokenizer = tokenizer_initializer()
  seen_recons_counts, unseen_recons_counts = _compute_reconstruction_counts(
      model_config,
      finetuned_model,
      tokenizer,
      seen_canaries,
      unseen_canaries,
  )
  logging.info('Seen reconstruction counts: %s', seen_recons_counts)
  logging.info('Unseen reconstruction counts: %s', unseen_recons_counts)

  recons_stats = {
      'seen_recons_counts': seen_recons_counts.tolist(),
      'unseen_recons_counts': unseen_recons_counts.tolist(),
  }
  with filesystem.open_file(f'{_WORK_DIR.value}/recons_stats.json', 'w') as f:
    f.write(json.dumps(recons_stats))


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
