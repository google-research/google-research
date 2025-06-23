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

from contextlib import nullcontext
import os
import tempfile
from typing import Optional, Union
import urllib.parse
import warnings

from accelerate import Accelerator
from act.config.base_config import BaseConfig, BaseInitializationConfig
from act.config.model.hf_model_config import HFModelConfig
from act.models.base_model import BaseModel
from act.utils.storage_utils import load_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def get_kbit_device_map():
  """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
  return (
      {"": Accelerator().local_process_index}
      if torch.cuda.is_available()
      else None
  )


def get_quantization_config(
    model_args,
):
  if model_args.load_in_4bit:
    compute_dtype = torch.float16
    if model_args.torch_dtype not in {"auto", None}:
      compute_dtype = getattr(torch, model_args.torch_dtype)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        bnb_4bit_quant_storage=model_args.bnb_4bit_quant_storage,
    )
  elif model_args.load_in_8bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
  else:
    quantization_config = None

  return quantization_config


class HFModel(BaseModel):

  def __init__(
      self,
      config,
      tokenizer = None,
      model = None,
      model_config = None,
  ):
    super().__init__(config)
    if model_config is None:
      self.model_config = self.config.policy_model_config
    else:
      self.model_config = model_config
    if not model:
      if isinstance(self.model_config.model_id, str):
        warnings.warn(
            "You passed a model_id to the DPOTrainer. This will automatically"
            " create an `AutoModelForCausalLM` or a `PeftModel` (if you passed"
            " a `peft_config`) for you."
        )
        self.tokenizer, self.model = HFModel.construct_hf_model(
            config, self.model_config
        )
    elif isinstance(model, PreTrainedModel):
      self.model = model
      assert tokenizer, "You must pass a tokenizer with a model."
      assert isinstance(
          tokenizer, PreTrainedTokenizerBase
      ), "Tokenizer must be a PreTrainedTokenizerBase."
      self.tokenizer = tokenizer
    else:
      raise ValueError("model must be a PreTrainedModel or a model_id.")
    self.generate_context_manager = (
        nullcontext
        if not self.config.training_config.mixed_precision
        else torch.cuda.amp.autocast
    )

  @staticmethod
  def get_torch_dtype(model_config):
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    return torch_dtype

  @staticmethod
  def construct_hf_model(config, model_config):
    """Load a model from Hugging Face or Google Storage."""
    torch_dtype = HFModel.get_torch_dtype(model_config)
    quantization_config = get_quantization_config(model_config)
    policy_model_kwargs = dict(
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        use_flash_attention_2=model_config.use_flash_attention_2,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        token=model_config.token,
        use_cache=False
        if config.training_config.gradient_checkpointing
        else True,
        device_map=get_kbit_device_map()
        if quantization_config is not None
        else None,
        quantization_config=quantization_config,
    )

    if model_config.model_path is None:
      model = AutoModelForCausalLM.from_pretrained(
          model_config.model_id,
          **policy_model_kwargs,
      )
      tokenizer = None
    else:
      model, tokenizer = load_model(
          model_config.model_path,
          AutoModelForCausalLM,
          AutoTokenizer,
          **policy_model_kwargs,
      )

    tokenizer = HFModel.get_tokenizer(config, tokenizer, model_config)
    return tokenizer, model

  @property
  def device(self):
    return self.model.device

  @device.setter
  def device(self, device):
    self.model.to(device)

  def generate(self, inputs, **generation_kwargs):
    if isinstance(inputs, str):
      input_ids = self.tokenizer(inputs)
    else:
      input_ids = inputs
    # Should we assume that inputs will always be on the same device as the
    # model? Or should we move them over?
    with self.generate_context_manager():
      response = self.model.generate(input_ids=input_ids, **generation_kwargs)
    return response

  def decode(self, inputs, **decode_kwargs):
    response = self.tokenizer.decode(inputs, **decode_kwargs)
    return response

  def encode(self, inputs, **encode_kwargs):
    response = self.tokenizer.encode(inputs, **encode_kwargs)
    return response

  @staticmethod
  def get_tokenizer(
      config,
      tokenizer = None,
      model_config = None,
  ):
    if model_config is None:
      model_config = config.policy_model_config
    assert isinstance(model_config, HFModelConfig)
    if not tokenizer:
      tokenizer = AutoTokenizer.from_pretrained(
          model_config.model_id
          if model_config.tokenizer_name_or_path is None
          else model_config.tokenizer_name_or_path,
          token=model_config.token,
          revision=model_config.revision,
          trust_remote_code=model_config.trust_remote_code,
      )
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    if tokenizer.pad_token_id is None:
      tokenizer.pad_token_id = tokenizer.eos_token_id

    if config.data_config.truncation_side is not None:
      tokenizer.truncation_side = config.data_config.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
      if config.training_config.max_length is not None:
        tokenizer.model_max_length = config.training_config.max_length
      else:
        tokenizer.model_max_length = 2048

    return tokenizer
