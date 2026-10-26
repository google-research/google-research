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

"""Common utilities for privacy auditing."""

import enum
from typing import Any

from absl import logging
from flax import nnx

from gemma.models.gemma3 import model as model_lib
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import qwix
import tunix


class ModelVersion(enum.Enum):
  GEMMA3_1B = 'gemma3_1b'
  GEMMA3_1B_IT = 'gemma3_1b_it'
  GEMMA3_12B = 'gemma3_12b'
  GEMMA3_12B_IT = 'gemma3_12b_it'

  @property
  def is_it(self) -> bool:
    """Whether the model version is IT."""
    return self in (ModelVersion.GEMMA3_1B_IT, ModelVersion.GEMMA3_12B_IT)


def create_mesh(
    clip_axis_max_size: int,
    min_fsdp_axis_size: int = 1,
) -> jax.sharding.Mesh:
  """Creates the mesh for distributed computation."""
  devices = jax.devices()
  original_num_devices = len(devices)
  num_slices = len(set(d.slice_index for d in devices))
  remaining_num_devices = original_num_devices
  if remaining_num_devices == 1:
    # Single-device case: return a trivial 1-device Mesh so callers always
    # receive a real jax.sharding.Mesh (required by nnx.get_named_sharding).
    return jax.sharding.Mesh(
        np.array(devices).reshape((1,)),
        axis_names=('clip',),
    )

  # First reserve any necessary devices to the FSDP axis.
  remaining_num_devices, r = divmod(remaining_num_devices, min_fsdp_axis_size)
  if r != 0:
    raise ValueError(
        'num_devices must be evenly divisible by min_fsdp_axis_size. Got'
        f' num_devices: {original_num_devices}, min_fsdp_axis_size:'
        f' {min_fsdp_axis_size}.'
    )

  # Now we have two cases, either the clip parallelism axis ("replica" axis in
  # other codebases) is larger than the number of devices, or it is not.
  #
  # If it is larger, put all devices on the clip parallelism axis, and shard the
  # batch axis over the maximum possible devices. This will leave no devices for
  # the other axes (unless min_fsdp_axis_size is greater than 1).
  #
  # Otherwise, assign the maximum clip parallelism axis and put the remaining
  # devices on the fsdp axis.
  if clip_axis_max_size > remaining_num_devices:
    if clip_axis_max_size % remaining_num_devices != 0:
      raise ValueError(
          'clip_axis_max_size must be evenly divisible by num_devices. Got'
          f' clip_axis_max_size: {clip_axis_max_size}, num_devices:'
          f' {remaining_num_devices}.'
      )
    # Put all devices on the clip axis if the maximum size it can be is larger
    # than the number of devices. This will shard the batch axis over the
    # maximum possible devices.
    clip_axis_size = remaining_num_devices
    remaining_num_devices //= clip_axis_size
  else:
    if remaining_num_devices % clip_axis_max_size != 0:
      raise ValueError(
          'clip_axis_max_size must evenly divide num_devices. Got'
          f' clip_axis_max_size: {clip_axis_max_size}, num_devices:'
          f' {remaining_num_devices}.'
      )
    clip_axis_size = clip_axis_max_size
    remaining_num_devices //= clip_axis_size

  fsdp_axis_size = remaining_num_devices * min_fsdp_axis_size
  tp_axis_size = 1

  # Try to split the left most axes across the slices as much as possible.
  if clip_axis_size >= num_slices:
    dcn_shape = (num_slices, 1, 1)
    ici_shape = (clip_axis_size // num_slices, fsdp_axis_size, tp_axis_size)
  elif fsdp_axis_size >= num_slices:
    dcn_shape = (1, num_slices, 1)
    ici_shape = (clip_axis_size, fsdp_axis_size // num_slices, tp_axis_size)
  else:
    logging.fatal(
        'with axis sizes (clip: %d, fsdp: %d, tp: %d), we cannot split across'
        ' %d slices cleanly.',
        clip_axis_size,
        fsdp_axis_size,
        tp_axis_size,
        num_slices,
    )
  axis_names = ('clip', 'fsdp', 'tp')
  logging.info(
      'Constructing mesh %r:\n\tici: %r\n\tdcn: %r',
      axis_names,
      ici_shape,
      dcn_shape,
  )
  device_mesh = mesh_utils.create_hybrid_device_mesh(
      mesh_shape=ici_shape,
      dcn_mesh_shape=dcn_shape,
      devices=jax.devices(),
      allow_split_physical_axes=True,
  )
  return jax.sharding.Mesh(
      devices=device_mesh,
      axis_names=axis_names,
      axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
  )


def create_audit_sharding_config() -> model_lib.ShardingConfig:
  """Returns a ShardingConfig suitable for auditing (inference with tiny batches).

  The default ShardingConfig assigns 'fsdp' to the batch axis in activations
  (act_btd, act_btf, act_btnh). During auditing, we score one canary at a time
  so the batch size is 1-2. With fsdp=16+ this causes an IndivisibleError.

  This config instead leaves activation batch axes un-sharded (None), keeping
  only weight sharding over 'fsdp' and 'tp'.
  """
  return model_lib.ShardingConfig(
      emb_vd=('tp', 'fsdp'),
      q_weight_ndh=('tp', 'fsdp', None),
      kv_weight_cndh=(None, 'tp', 'fsdp', None),
      qkv_weight_cndh=(None, 'tp', 'fsdp', None),
      o_weight_nhd=('tp', None, 'fsdp'),
      ffw_weight_df=('fsdp', 'tp'),
      ffw_weight_fd=('tp', 'fsdp'),
      rms_norm_weight=('tp',),
      # Leave batch axis (dim 0) un-sharded so that tiny canary batches (1-2)
      # are not partitioned across the fsdp axis (which can be 16+).
      act_btd=(None, None, 'tp'),
      act_btf=(None, None, 'tp'),
      act_btnh=(None, None, 'tp', None),
      vision_proj=('fsdp', 'tp'),
      vision_soft_emb_norm_weight=('tp',),
      siglip=None,
  )


def get_model_config_and_pretrained_ckpt_path(
    model_version: ModelVersion,
    *,
    audit_sharding: bool = False,
) -> tuple[model_lib.ModelConfig, str]:
  """Returns the model config and pretrained checkpoint path.

  Args:
    model_version: The model version to use.
    audit_sharding: If True, uses an audit-specific ShardingConfig that avoids
      sharding activations on the batch axis. This prevents IndivisibleErrors
      when the canary batch size (1-2) is smaller than fsdp (e.g. 16).
  """
  sharding_config = (
      create_audit_sharding_config()
      if audit_sharding
      else model_lib.ShardingConfig.get_default_sharding()
  )
  match model_version:
    case ModelVersion.GEMMA3_1B:
      model_config = model_lib.ModelConfig.gemma3_1b_pt(sharding_config)
      ckpt_path = '<PATH_TO_GEMMA3_1B_PT_CHECKPOINT>'
    case ModelVersion.GEMMA3_1B_IT:
      model_config = model_lib.ModelConfig.gemma3_1b_it(sharding_config)
      ckpt_path = '<PATH_TO_GEMMA3_1B_IT_CHECKPOINT>'
    case ModelVersion.GEMMA3_12B:
      model_config = model_lib.ModelConfig.gemma3_12b_pt(sharding_config)
      ckpt_path = '<PATH_TO_GEMMA3_12B_PT_CHECKPOINT>'
    case ModelVersion.GEMMA3_12B_IT:
      model_config = model_lib.ModelConfig.gemma3_12b_it(sharding_config)
      ckpt_path = '<PATH_TO_GEMMA3_12B_IT_CHECKPOINT>'
    case _:
      raise ValueError(f'Unsupported model version: {model_version!r}')
  return model_config, ckpt_path


def restore_from_checkpoint(
    ckpt_path: str,
    abs_state: nnx.Module,
    sharding: jax.sharding.NamedSharding,
) -> Any:
  """Restores model state from a checkpoint."""
  mapped_state = jax.tree.map(
      lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s),
      abs_state,
      sharding,
  )
  with ocp.StandardCheckpointer() as checkpointer:
    return checkpointer.restore(ckpt_path, target=mapped_state)


def load_pretrained_model(
    model_config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh,
    pretrained_ckpt_path: str,
) -> model_lib.Gemma3:
  """Loads the pretrained (base) model, sharded over `mesh`.

  This must be implemented by the user before running training or auditing,
  since loading the pretrained Gemma 3 checkpoint depends on the environment.
  It is used to initialize training (`train.py`), as the reference model when
  auditing (`model_audit.py`), and as the base for LoRA finetuned models
  (`load_finetuned_model`).

  Args:
    model_config: Config of the model to load.
    mesh: Mesh to shard the model parameters over.
    pretrained_ckpt_path: Path to the pretrained checkpoint.

  Returns:
    The pretrained model.
  """
  # Not yet implemented: replace with the OSS tunix model-loading API.
  # See https://github.com/google-deepmind/tunix for the correct call.
  # Example: return tunix.load_and_shard_model(
  #     pretrained_ckpt_path, model_config, mesh)
  del model_config, mesh  # Unused until implemented.
  raise NotImplementedError(
      'Implement pretrained model loading in `audit_lib.load_pretrained_model`'
      ' from checkpoint path: ' + pretrained_ckpt_path
  )


def load_finetuned_model(
    model_config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh,
    finetuned_ckpt_path: str,
    *,
    pretrained_ckpt_path: str | None = None,
    lora_rank: int | None = None,
    alpha_scale_factor: float = 1.0,
) -> model_lib.Gemma3:
  """Loads a finetuned model (either full or LoRA)."""
  if lora_rank is not None:
    if pretrained_ckpt_path is None:
      raise ValueError('pretrained_ckpt_path is required for LoRA models.')

    transformer = load_pretrained_model(
        model_config, mesh, pretrained_ckpt_path
    )

    lora_provider = qwix.LoraProvider(
        module_path='.*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj',
        rank=lora_rank,
        alpha=alpha_scale_factor * lora_rank,
    )
    model_input = transformer.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        transformer, lora_provider, **model_input
    )

    lora_params_state = nnx.state(lora_model, nnx.LoRAParam)
    lora_params_sharding = nnx.get_named_sharding(lora_params_state, mesh)

    restored_lora_params = restore_from_checkpoint(
        finetuned_ckpt_path, lora_params_state, lora_params_sharding
    )

    nnx.update(
        lora_model,
        jax.tree.map(
            lambda a, b: b,
            nnx.state(lora_model, nnx.LoRAParam),
            restored_lora_params,
        ),
    )
    return lora_model

  # Full finetuning.
  abs_gemma: nnx.Module = nnx.eval_shape(
      lambda: model_lib.Gemma3(model_config, rngs=nnx.Rngs(params=0))
  )

  abs_state = nnx.state(abs_gemma)
  sharding = nnx.get_named_sharding(abs_state, mesh)

  restored_params = restore_from_checkpoint(
      finetuned_ckpt_path, abs_state, sharding
  )
  graph_def, _ = nnx.split(abs_gemma)
  return nnx.merge(graph_def, restored_params)


def create_sampler(
    model: model_lib.Gemma3,
    tokenizer: tunix.generate.tokenizer_adapter.Tokenizer,
    model_config: model_lib.ModelConfig,
    cache_size: int = 1024,
) -> tunix.Sampler:
  """Creates a sampler for the given model."""
  return tunix.Sampler(
      transformer=model,
      tokenizer=tokenizer,
      cache_config=tunix.CacheConfig(
          cache_size=cache_size,
          num_layers=model_config.num_layers,
          num_kv_heads=model_config.num_kv_heads,
          head_dim=model_config.head_dim,
      ),
  )


def make_causal_attn_mask(
    input_mask: jax.Array,
) -> jax.Array:
  """Creates a causal attention mask."""
  seq_len = input_mask.shape[-1]
  attn_mask = input_mask[..., None, :]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  # Prefixes can be attended by all tokens.
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def compute_log_likelihood(
    model: model_lib.Gemma3,
    *,
    prefix: jax.Array | np.ndarray,
    suffix: jax.Array | np.ndarray,
) -> float | jax.Array:
  """Computes the log-likelihood of a suffix given a prefix."""
  full_sequence = jnp.concatenate([prefix, suffix])
  actual_seq_len = len(full_sequence)

  if actual_seq_len == 0 or len(prefix) >= actual_seq_len:
    return 0.0

  batched_input_tokens = full_sequence[None, :]
  positions = jnp.arange(actual_seq_len)[None, :]
  input_mask_for_attn = jnp.ones_like(full_sequence, dtype=jnp.bool_)[None, :]
  attention_mask = make_causal_attn_mask(input_mask_for_attn)

  model.eval()
  next_token_logits, _ = model(
      last_tokens=batched_input_tokens,
      positions=positions,
      cache=None,
      attention_mask=attention_mask,
  )

  next_token_logits = next_token_logits[0]
  next_token_logits = next_token_logits.astype(jnp.float32)
  log_probs = jax.nn.log_softmax(next_token_logits, axis=-1)

  if prefix.size == 0:
    if actual_seq_len <= 1:
      return 0.0
    else:
      selected_log_probs = log_probs[
          jnp.arange(0, actual_seq_len - 1),
          full_sequence[1:],
      ]
  else:
    selected_log_probs = log_probs[
        jnp.arange(len(prefix) - 1, actual_seq_len - 1),
        full_sequence[len(prefix) :],
    ]
  return jnp.sum(selected_log_probs)


def compute_log_likelihood_batch(
    model: model_lib.Gemma3,
    *,
    prefixes: list[np.ndarray],
    suffixes: list[np.ndarray],
) -> jax.Array:
  """Computes log-likelihoods for a batch of (prefix, suffix) pairs.

  All prefixes must have the same length. Suffixes can have variable lengths.

  Args:
    model: The language model to score with.
    prefixes: List of prefix token arrays, all the same length.
    suffixes: List of suffix token arrays.

  Returns:
    A 1-D array of shape (batch_size,) containing the summed log-likelihood
    of each suffix conditioned on its prefix.
  """
  if len(suffixes) != len(prefixes):
    raise ValueError(
        f'Number of prefixes ({len(prefixes)}) does not match number of'
        f' suffixes ({len(suffixes)}).'
    )
  if not prefixes:
    return jnp.array([])
  batch_size = len(prefixes)

  if any(s.size == 0 for s in suffixes):
    raise ValueError('All suffixes must have at least one token.')

  prefix_len = len(prefixes[0])
  if any(len(p) != prefix_len for p in prefixes):
    raise ValueError(
        'Prefix lengths must be the same for all examples in a batch.'
    )

  suffix_lengths = np.array([len(s) for s in suffixes])
  max_suffix_len = np.max(suffix_lengths)
  seq_len = prefix_len + max_suffix_len

  tokens = np.zeros((batch_size, seq_len), dtype=np.int32)
  input_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)

  for i, (p, s) in enumerate(zip(prefixes, suffixes)):
    tokens[i, :prefix_len] = p
    tokens[i, prefix_len : prefix_len + len(s)] = s
    input_mask[i, : prefix_len + len(s)] = True

  positions = jnp.broadcast_to(
      jnp.arange(seq_len)[None, :], (batch_size, seq_len)
  )
  attention_mask = make_causal_attn_mask(jnp.array(input_mask))

  model.eval()
  next_token_logits, _ = model(
      last_tokens=jnp.array(tokens),
      positions=positions,
      cache=None,
      attention_mask=attention_mask,
  )

  # At position t the model predicts token t+1. The suffix occupies tokens
  # [prefix_len, prefix_len + suffix_len), so the relevant logit positions are
  # [prefix_len - 1, prefix_len + suffix_len - 1).
  # When prefix_len == 0 there is no prefix context, so the first token cannot
  # be scored; start from position 0 and score tokens [1, seq_len), matching
  # compute_log_likelihood.
  logit_start = max(0, prefix_len - 1)
  target_start = max(1, prefix_len)
  num_targets = seq_len - target_start
  if num_targets <= 0:
    return jnp.zeros((batch_size,), dtype=jnp.float32)

  suffix_logits = next_token_logits[:, logit_start : seq_len - 1, :]
  suffix_logits = suffix_logits.astype(jnp.float32)  # (B, num_targets, V)
  log_probs = jax.nn.log_softmax(suffix_logits, axis=-1)

  # Target tokens are from target_start up to seq_len.
  suffix_targets = jnp.array(tokens[:, target_start:])  # (B, num_targets)

  selected = log_probs[
      jnp.arange(batch_size)[:, None],
      jnp.arange(num_targets)[None, :],
      suffix_targets,
  ]  # (B, num_targets)

  # Mask out log-probs from padding.
  suffix_mask = input_mask[:, target_start:]  # (B, num_targets)
  selected = jnp.where(suffix_mask, selected, 0.0)

  return jnp.sum(selected, axis=-1)  # (B,)


def _trim_suffix(suffix: np.ndarray, canary_len: int) -> np.ndarray:
  """Trims the suffix to the specified length."""
  if suffix.size == 0:
    raise ValueError('Canary has an empty suffix.')
  if suffix.size < canary_len:
    logging.warning(
        'Suffix length %d is less than the specified canary_len %d.',
        suffix.size,
        canary_len,
    )
  return suffix[:canary_len]


def score_canaries_batch(
    canaries: list[tuple[np.ndarray, np.ndarray]],
    *,
    base_model: model_lib.Gemma3,
    finetuned_model: model_lib.Gemma3,
    canary_len: int,
) -> jax.Array:
  """Scores a batch of canaries by the difference in log-likelihoods.

  Args:
    canaries: A list of (prefix_tokens, suffix_tokens) tuples.
    base_model: The model before fine-tuning.
    finetuned_model: The model after fine-tuning.
    canary_len: Only the first `canary_len` suffix tokens are used.

  Returns:
    A 1-D array of shape (len(canaries),) with finetuned - base log-likelihood
    differences.
  """
  prefixes, suffixes = zip(*canaries)
  suffixes = [_trim_suffix(suffix, canary_len) for suffix in suffixes]

  finetuned_ll = compute_log_likelihood_batch(
      finetuned_model, prefixes=list(prefixes), suffixes=list(suffixes)
  )
  base_ll = compute_log_likelihood_batch(
      base_model, prefixes=list(prefixes), suffixes=list(suffixes)
  )
  return finetuned_ll - base_ll


def score_canary(
    canary: tuple[np.ndarray, np.ndarray],
    *,
    base_model: model_lib.Gemma3,
    finetuned_model: model_lib.Gemma3,
    canary_len: int,
) -> float | jax.Array:
  """Scores a canary by the difference in log-likelihoods.

  Args:
    canary: A tuple of `(prefix_tokens, suffix_tokens)` representing the canary.
    base_model: The model before fine-tuning.
    finetuned_model: The model after fine-tuning.
    canary_len: Only the first `canary_len` suffix tokens are used.

  Returns:
    The difference between the log-likelihood of the canary under the
    finetuned model and the base model.

  Raises:
    ValueError: If the suffix is empty.
  """
  prefix, suffix = canary
  suffix = _trim_suffix(suffix, canary_len)

  return compute_log_likelihood(
      finetuned_model, prefix=prefix, suffix=suffix
  ) - compute_log_likelihood(base_model, prefix=prefix, suffix=suffix)
