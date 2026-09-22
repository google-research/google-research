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

"""Whisper engine for composed ASR fine-tuning."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from absl import logging
import ml_collections
import torch
from torch import nn
import transformers

from Uboreshaji_Modeli.common import audio_utils
from Uboreshaji_Modeli.common import data
from . import base
from . import decoders


class WhisperTransform(data.PicklableProcessorMixin):
  """Picklable transform callable for Whisper data loader workers.

  Uses ``__getstate__``/``__setstate__`` to re-load the processor from
  disk in each DataLoader worker process, since the processor's native
  backends do not survive Python pickling.
  """

  def __init__(
      self,
      processor,
      cfg,
      model_id = None,
  ):
    self.processor = processor
    self.cfg = cfg
    self._model_id = model_id or getattr(processor, "name_or_path", None)

  def _load_processor(self, model_id):

    return transformers.WhisperProcessor.from_pretrained(model_id)

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

    valid_audio_arrays = []
    valid_indices = []
    for i, x in enumerate(audio_inputs):
      arr = audio_utils.get_audio_array(x, target_sr=sampling_rate)
      if arr is None:
        logging.warning("Skipping bad audio record at index %d", i)
        continue
      valid_audio_arrays.append(arr)
      valid_indices.append(i)

    if not valid_audio_arrays:
      return {k: [] for k in batch_dict.keys()}

    if len(valid_indices) < len(audio_inputs):
      for k, v in batch_dict.items():
        if isinstance(v, list) and len(v) == len(audio_inputs):
          batch_dict[k] = [v[i] for i in valid_indices]

    inputs = self.processor(
        valid_audio_arrays, sampling_rate=sampling_rate, return_tensors="pt"
    )

    if isinstance(audio_column, dict):
      batch_dict["input_features"] = inputs.input_features[0]
    else:
      batch_dict["input_features"] = inputs.input_features

    if "text" in batch_dict:
      text_key = "text"
    elif "transcription" in batch_dict:
      text_key = "transcription"
    else:
      text_key = None
    if text_key is not None:
      text_column = batch_dict[text_key]
      if isinstance(text_column, str):
        labels = self.processor.tokenizer([text_column]).input_ids
        batch_dict["labels"] = labels[0]
      else:
        labels = self.processor.tokenizer(text_column).input_ids
        batch_dict["labels"] = labels

    return batch_dict


class WhisperPreprocessor(base.DataPreprocessor):
  """Preprocessor for Whisper speech recognition."""

  def get_transform_fn(
      self,
      processor,
      cfg = None,
      *,
      is_train = False,
      **kwargs,
  ):
    """Returns a transform function converting examples to Whisper SFT inputs."""
    del self  # Unused.

    temp_dir = data.stage_processor_locally(
        processor, prefix="local_whisper_processor_"
    )

    return WhisperTransform(
        processor=processor,
        cfg=cfg,
        model_id=temp_dir,
    )

  def get_collate_fn(
      self,
      cfg = None,
      **kwargs,
  ):
    """Returns a collation function that dynamically pads Whisper inputs.

    Args:
      cfg: Optional configuration dictionary.
      **kwargs: Config options containing the model's processor.

    Returns:
      A collation callable that dynamically pads speech inputs.
    """
    del self  # Unused.
    processor = kwargs.get("processor")
    if processor is None:
      raise ValueError(
          "processor is required in kwargs for WhisperPreprocessor collate_fn."
      )

    device_type = cfg.training.get("device", "cuda") if cfg else "cuda"
    if device_type == "tpu":
      padding = "max_length"
      max_length = cfg.get("max_seq_length", 256) if cfg else 256
    else:
      padding = True
      max_length = None

    return data.DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        padding=padding,
        max_length=max_length,
    )

  def get_sft_config_overrides(
      self, cfg
  ):
    """Returns SFTConfig overrides for the custom Whisper preprocessor."""
    del cfg  # Unused.
    return {
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }


class WhisperEngine(base.ModelEngine):
  """Coordinated speech loader settings and attention layers for the Whisper pipeline."""

  def __init__(self):
    """Initializes the Whisper engine instance with preprocessor and prediction decoder."""
    super().__init__(
        preprocessor=WhisperPreprocessor(),
        loss_handler=None,
        decoder=decoders.TextDecoder(),
    )

  def load_model_and_processor(
      self,
      model_id,
      device,
      **kwargs,
  ):
    """Loads WhisperConditionalGeneration checkpoints and feature processors.

    Args:
      model_id: Pretrained repository ID or local checkpoint directory.
      device: Target hardware device mapping.
      **kwargs: Additional keyword settings.

    Returns:
      A tuple (model, processor), where model is the initialized
      WhisperForConditionalGeneration model on the device, and processor is the
      WhisperProcessor instance.
    """
    del self, kwargs  # Unused.
    processor = transformers.WhisperProcessor.from_pretrained(model_id)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_id
    )
    model.to(device)
    return model, processor
