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

"""Gemma text engine for composed causal language model SFT."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import ml_collections
import peft
import torch
from torch import nn
import transformers

from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import data
from . import base
from . import decoders


class GemmaTextTransform(data.PicklableProcessorMixin):
  """Picklable transform callable for Gemma Text data loader workers.

  When used with ``dataloader_num_workers > 0``, PyTorch pickles this
  object into each worker process.  The HuggingFace tokenizer wraps a
  SentencePiece Rust backend that does not survive Python pickling.

  To work around this, ``__getstate__`` strips the live tokenizer and
  stores its ``model_id`` path, and ``__setstate__`` re-loads a fresh
  tokenizer from disk in the new worker process.
  """

  def __init__(
      self,
      processor,
      max_length,
      cfg,
      model_id = None,
  ):
    self.processor = processor
    self.max_length = max_length
    self.cfg = cfg
    self._model_id = model_id or getattr(processor, "name_or_path", None)
    # Preserve the chat_template so it survives pickle round-trips
    # across DataLoader worker processes (num_workers > 0).
    self._chat_template = getattr(processor, "chat_template", None)

  def _load_processor(self, model_id):
    return transformers.AutoTokenizer.from_pretrained(model_id)

  def _post_load_processor(self):
    if self.processor.pad_token is None:
      self.processor.pad_token = self.processor.eos_token
    if self._chat_template is not None:
      self.processor.chat_template = self._chat_template

  def __call__(
      self,
      batch,
  ):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    if (
        isinstance(batch["messages"], list)
        and batch["messages"]
        and isinstance(batch["messages"][0], dict)
    ):
      messages_list = [batch["messages"]]
    else:
      messages_list = batch["messages"]

    for messages in messages_list:
      text = self.processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=False
      )
      encoded = self.processor(
          text,
          truncation=True,
          max_length=self.max_length,
          padding=False,
          return_tensors=None,
      )
      input_ids = encoded["input_ids"]
      attention_mask = encoded["attention_mask"]

      labels = [-100] * len(input_ids)

      if self.cfg and self.cfg.model_flavor == config.ModelFlavor.GEMMA_4_TEXT:
        assistant_marker_str = "<|turn>model\n"
        end_marker_str = "<turn|>\n"
      else:
        assistant_marker_str = "<start_of_turn>model\n"
        end_marker_str = "<end_of_turn>\n"

      assistant_marker_ids = self.processor.encode(
          assistant_marker_str, add_special_tokens=False
      )
      end_marker_ids = self.processor.encode(
          end_marker_str, add_special_tokens=False
      )

      model_turn_starts = []
      for j in range(len(input_ids) - len(assistant_marker_ids) + 1):
        if (
            input_ids[j : j + len(assistant_marker_ids)]
            == assistant_marker_ids
        ):
          model_turn_starts.append(j)

      for start_idx in model_turn_starts:
        content_start = start_idx + len(assistant_marker_ids)
        content_end = len(input_ids)
        for k in range(
            content_start, len(input_ids) - len(end_marker_ids) + 1
        ):
          if input_ids[k : k + len(end_marker_ids)] == end_marker_ids:
            content_end = k
            break

        for idx in range(content_start, content_end):
          if idx < len(input_ids):
            labels[idx] = input_ids[idx]

      all_input_ids.append(input_ids)
      all_attention_masks.append(attention_mask)
      all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


class GemmaTextCollate:
  """Picklable collate callable for Gemma Text data loader workers."""

  def __init__(
      self,
      is_tpu = False,
      max_seq_length = 256,
      pad_token_id = 0,
  ):
    self.is_tpu = is_tpu
    self.max_seq_length = max_seq_length
    self.pad_token_id = pad_token_id

  def __call__(self, batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_masks = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    if self.is_tpu:
      max_len = self.max_seq_length
    else:
      max_len = max(ids.shape[0] for ids in input_ids)

    padded_input_ids = torch.full(
        (len(batch), max_len), self.pad_token_id, dtype=torch.long
    )
    padded_attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (ids, att, lab) in enumerate(
        zip(input_ids, attention_masks, labels)
    ):
      seq_len = min(ids.shape[0], max_len)
      padded_input_ids[i, :seq_len] = ids[:seq_len]
      padded_attention[i, :seq_len] = att[:seq_len]
      padded_labels[i, :seq_len] = lab[:seq_len]

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention,
        "labels": padded_labels,
    }


class GemmaTextPreprocessor(base.DataPreprocessor):
  """Preprocessor for Gemma causal language SFT."""

  def get_transform_fn(
      self,
      processor,  # This is the tokenizer.
      cfg = None,
      *,
      is_train = False,
      **kwargs,
  ):
    """Returns a transform function converting batch examples to Gemma text SFT inputs."""
    del self  # Unused.
    max_length = cfg.get("max_seq_length", 512) if cfg else 512
    temp_dir = data.stage_processor_locally(
        processor, prefix="local_tokenizer_"
    )

    return GemmaTextTransform(
        processor=processor,
        max_length=max_length,
        cfg=cfg,
        model_id=temp_dir,
    )

  def get_collate_fn(
      self,
      cfg = None,
      *,
      pad_token_id = 0,
      **kwargs,
  ):
    """Returns a collation function to dynamically pad causal text datasets."""
    del self
    device_type = cfg.training.get("device", "cpu") if cfg else "cpu"
    is_tpu = device_type == "tpu"
    max_seq_length = cfg.get("max_seq_length", 256) if cfg else 256

    return GemmaTextCollate(
        is_tpu=is_tpu,
        max_seq_length=max_seq_length,
        pad_token_id=pad_token_id,
    )

  def get_sft_config_overrides(
      self, cfg
  ):
    """Returns SFT overrides configured for this text preprocessor.

    Args:
      cfg: Config dictionary.

    Returns:
      A dictionary containing dataset and gradient checkpoint overrides.
    """
    del self  # Unused.
    return {
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        **(
            {"max_seq_length": cfg.max_seq_length}
            if cfg.get("max_seq_length")
            else {}
        ),
    }


class GemmaTextEngine(base.ModelEngine):
  """Coordinated causal language SFT setup for the Gemma text pipeline."""

  def __init__(self):
    """Initializes SFT preprocessor, text decoder and baseline handlers."""
    super().__init__(
        preprocessor=GemmaTextPreprocessor(),
        loss_handler=None,
        decoder=decoders.TextDecoder(),
    )

  def load_model_and_processor(
      self,
      model_id,
      device,
      **kwargs,
  ):
    """Loads the causal Gemma model and its tokenizer.

    Args:
      model_id: Model ID or direct checkpoint directory path.
      device: PyTorch target hardware device mapping.
      **kwargs: Additional configuration arguments.

    Returns:
      A tuple (model, tokenizer), where model is the initialized
      AutoModelForCausalLM model on the device, and tokenizer is the
      AutoTokenizer instance.
    """
    del self  # Unused.
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

    cfg = kwargs.get("cfg")

    if getattr(tokenizer, "chat_template", None) is None:
      if cfg and cfg.model_flavor == config.ModelFlavor.GEMMA_3_TEXT:
        tokenizer.chat_template = (
            "{% for message in messages %}{% if message['role'] == 'user' %}{{"
            " '<start_of_turn>user\\n' + message['content'] +"
            " '<end_of_turn>\\n' }}{% elif message['role'] == 'assistant' %}{{"
            " '<start_of_turn>model\\n' + message['content'] +"
            " '<end_of_turn>\\n' }}{% endif %}{% endfor %}"
        )
      elif cfg and cfg.model_flavor == config.ModelFlavor.GEMMA_4_TEXT:
        tokenizer.chat_template = (
            "{% for message in messages %}{% if message['role'] == 'user' %}{{"
            " '<|turn>user\\n' + message['content'] + '<turn|>\\n' }}{% elif"
            " message['role'] == 'assistant' %}{{ '<|turn>model\\n' +"
            " message['content'] + '<turn|>\\n' }}{% endif %}{% endfor %}"
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)

    # LoRA wrapping.
    if cfg and cfg.training.get("use_lora", False):
      lora_cfg = cfg.training.lora
      target_modules = list(lora_cfg.target_modules)

      adjusted_targets = []
      for tm in target_modules:
        needs_sublinear = False
        for name, module in model.named_modules():
          if (
              name.endswith(tm)
              and hasattr(module, "linear")
              and not isinstance(module, nn.Linear)
          ):
            needs_sublinear = True
            break
        if needs_sublinear:
          adjusted_targets.append(f"{tm}.linear")
        else:
          adjusted_targets.append(tm)

      peft_config = peft.LoraConfig(
          r=lora_cfg.r,
          lora_alpha=lora_cfg.alpha,
          lora_dropout=lora_cfg.dropout,
          target_modules=adjusted_targets,
      )
      model = peft.get_peft_model(model, peft_config)

    model.to(device)
    return model, tokenizer
