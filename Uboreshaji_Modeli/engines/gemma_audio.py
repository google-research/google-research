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

"""Gemma audio engine for composed multimodal speech causal SFT."""

from collections.abc import Mapping, Sequence
import functools
from typing import Any, Callable

from absl import logging
import ml_collections
import peft
import torch
from torch import nn
import transformers

from Uboreshaji_Modeli.common import audio_utils
from . import base
from . import decoders


def _mask_labels(labels, processor):
  """Masks padding and special tokens in labels for loss computation."""
  labels = labels.clone()
  if not hasattr(processor, "tokenizer"):
    return labels

  tokenizer = processor.tokenizer
  mask_attrs = [
      "pad_token_id",
      "image_token_id",
      "audio_token_id",
      "boi_token_id",
      "eoi_token_id",
  ]
  mask_ids = [
      getattr(tokenizer, attr)
      for attr in mask_attrs
      if getattr(tokenizer, attr, None) is not None
  ]

  if mask_ids:
    labels[torch.isin(labels, torch.tensor(mask_ids, device=labels.device))] = (
        -100
    )

  return labels


def _collate_fn(
    examples,
    processor,
    cfg = None,
):
  """Collates a batch of examples for training with Gemma audio.

  Calls the full processor in one shot to produce all tensors with correct
  dtypes, following the waxal/gemma3n pattern.

  Args:
    examples: A list of dicts, each with 'messages' and 'audio' keys.
    processor: The AutoProcessor instance for tokenization and audio features.
    cfg: Optional configuration dictionary.

  Returns:
    A dictionary containing 'input_ids', 'attention_mask',
    'input_features', 'input_features_mask', and 'labels'.
  """
  texts = [
      processor.apply_chat_template(
          example["messages"], tokenize=False, add_generation_prompt=False
      )
      for example in examples
  ]
  audios = [example["audio"]["array"] for example in examples]

  device_type = cfg.training.get("device", "cuda") if cfg else "cuda"
  # Transformers v5 processors reject a top-level `max_length`.
  if device_type == "tpu":
    max_length = cfg.get("max_seq_length", 512) if cfg else 512
    kwargs = {
        "padding": "max_length",
        "text_kwargs": {"max_length": max_length},
    }
  else:
    kwargs = {"padding": True}

  batch = processor(text=texts, audio=audios, return_tensors="pt", **kwargs)
  # Detach and clone to avoid issues with in-place operations
  batch = {
      k: v.detach().clone() if isinstance(v, torch.Tensor) else v
      for k, v in batch.items()
  }

  batch["labels"] = _mask_labels(batch["input_ids"], processor)
  return batch


class GemmaAudioTransform:
  """Picklable transform callable for Gemma Audio data loader workers."""

  def __init__(
      self,
      prompt,
      cfg,
  ):
    self.prompt = prompt
    self.cfg = cfg

  def __call__(
      self,
      batch,
  ):
    batch_dict = dict(batch)
    audio_column = batch_dict["audio"]
    if isinstance(audio_column, dict):
      audio_inputs = [audio_column]
    else:
      audio_inputs = audio_column

    sampling_rate = 16000
    if (
        self.cfg
        and "training" in self.cfg
        and "audio_sample_rate" in self.cfg.training
    ):
      sampling_rate = self.cfg.training.audio_sample_rate

    if "text" in batch_dict:
      text_key = "text"
    elif "transcription" in batch_dict:
      text_key = "transcription"
    else:
      text_key = None

    text_column = batch_dict.get(text_key) if text_key else None
    text_inputs = (
        ([text_column] if isinstance(text_column, str) else text_column)
        if text_column is not None
        else [None] * len(audio_inputs)
    )

    messages_list = []
    audio_list = []
    for i, (audio_input, text) in enumerate(zip(audio_inputs, text_inputs)):
      arr = audio_utils.get_audio_array(audio_input, target_sr=sampling_rate)
      if arr is None:
        logging.warning("Skipping bad audio record at index %d", i)
        continue

      audio_list.append({"array": arr, "sampling_rate": sampling_rate})

      user_content = [
          {"type": "audio", "audio": arr},
          {"type": "text", "text": self.prompt},
      ]
      assistant_content = [
          {"type": "text", "text": text if text else ""},
      ]
      messages_list.append([
          {"role": "user", "content": user_content},
          {"role": "assistant", "content": assistant_content},
      ])

    return {"messages": messages_list, "audio": audio_list}


class GemmaAudioPreprocessor(base.DataPreprocessor):
  """Preprocessor for Gemma audio-speech SFT tasks."""

  def get_transform_fn(
      self,
      processor,
      cfg = None,
      *,
      is_train = False,
      **kwargs,
  ):
    """Returns a transform function for Gemma audio models."""
    del self, processor  # Unused; processor is used at collation time.
    prompt = (
        cfg.prompt
        if cfg and "prompt" in cfg
        else "Transpose the following audio:"
    )

    return GemmaAudioTransform(
        prompt=prompt,
        cfg=cfg,
    )

  def get_collate_fn(
      self,
      cfg = None,
      **kwargs,
  ):
    """Returns a collation function that calls the full processor in one shot.

    This follows the waxal/gemma3n pattern: the processor handles tokenization,
    audio feature extraction, and padding together, ensuring all tensors have
    correct dtypes.

    Args:
      cfg: Optional configuration dictionary containing hyper-parameters.
      **kwargs: Additional config keyword options including "processor".

    Returns:
      A collation callable that produces model-ready tensors.
    """
    del self  # Unused.
    processor = kwargs.get("processor")
    if processor is None:
      raise ValueError(
          "processor is required in kwargs for GemmaAudioPreprocessor"
          " collate_fn."
      )
    return functools.partial(_collate_fn, processor=processor, cfg=cfg)

  def get_sft_config_overrides(
      self, cfg
  ):
    """Returns SFTConfig overrides for the custom Gemma preprocessor.

    Args:
      cfg: Config dictionary container.

    Returns:
      A dictionary containing training and checkpointing overrides.
    """
    del self, cfg  # Unused.
    return {
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }


class GemmaAudioEngine(base.ModelEngine):
  """Coordinated multimodal training setup for the Gemma speech pipeline."""

  def __init__(self):
    """Initializes the composed speech preprocessor and prediction decoder."""
    super().__init__(
        preprocessor=GemmaAudioPreprocessor(),
        loss_handler=None,
        decoder=decoders.TextDecoder(),
    )

  def load_model_and_processor(
      self,
      model_id,
      device,
      **kwargs,
  ):
    """Loads the pretrained Gemma audio visual model and its processor.

    Args:
      model_id: Pretrained repository ID or local checkpoint path.
      device: Target torch device mapping.
      **kwargs: Additional config overrides.

    Returns:
      A tuple (model, processor), where model is the initialized
      AutoModelForCausalLM model on the device, and processor is the
      AutoProcessor instance.
    """
    del self  # Unused.
    cfg = kwargs.get("cfg")
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    # Gemma 3n trains with eager attention.
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        model_id, attn_implementation="eager"
    )
    if cfg and cfg.training.get("use_lora", False):
      lora = cfg.training.lora
      model = peft.get_peft_model(
          model,
          peft.LoraConfig(
              task_type="CAUSAL_LM",
              r=lora.r,
              lora_alpha=lora.alpha,
              lora_dropout=lora.dropout,
              target_modules=list(lora.target_modules),
          ),
      )
    model.to(device)
    return model, processor
