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

"""Gemma vision engine for composed VLM fine-tuning."""

import collections
from collections.abc import Callable, Mapping, Sequence
import os
import threading
from typing import Any

from absl import logging
import ml_collections
from PIL import Image
import torch
from torch import nn
import transformers

from Uboreshaji_Modeli.common import config as common_config
from Uboreshaji_Modeli.common import data
from Uboreshaji_Modeli.common import gemma_box_utils

from . import base

_PROCESSOR_LOCK = threading.Lock()


class GemmaVisionTransform(data.PicklableProcessorMixin):
  """Picklable transform callable for Gemma Vision data loader workers.

  Uses ``__getstate__``/``__setstate__`` to re-load the processor from
  disk in each DataLoader worker process, since the processor's native
  backends do not survive Python pickling.

  On reload, custom ``<loc>`` tokens are re-added to the tokenizer
  when ``detection_format == 'loc'``, matching the setup in
  ``GemmaVisionEngine.load_model_and_processor``.

  `transform_example` handles one example (the Bagz path, inside the Grain
  map); calling the transform handles a batch (the HF `datasets` path).
  """

  def __init__(
      self,
      processor,
      class_names,
      prompt_text,
      detection_format,
      do_pan_and_scan,
      cfg,
      model_id = None,
  ):
    self.processor = processor
    self.class_names = class_names
    self.prompt_text = prompt_text
    self.detection_format = detection_format
    self.do_pan_and_scan = do_pan_and_scan
    self.cfg = cfg
    self._model_id = model_id or getattr(processor, "name_or_path", None)
    self._marker_ids: list[int] | None = None

  def _post_load_processor(self):
    if self.detection_format == "loc":
      loc_tokens = [f"<loc{i:04d}>" for i in range(1024)]
      num_added = self.processor.tokenizer.add_tokens(loc_tokens)
      logging.debug("Added %d new loc tokens to vocabulary.", num_added)
    self.processor.tokenizer.padding_side = "right"
    self._marker_ids = None

  def transform_example(
      self, image, objects
  ):
    """Turns one image and its boxes into Gemma VLM inputs."""
    image = image.convert("RGB")
    w, h = image.size
    detection_string = gemma_box_utils.format_objects_to_detection_string(
        objects,
        self.class_names,
        image_width=w,
        image_height=h,
        detection_format=self.detection_format,
    )
    image_size = (
        self.cfg.get("dataset", {}).get("image_size") if self.cfg else 0
    )
    if image_size:
      image = image.resize((image_size, image_size))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": detection_string}],
        },
    ]
    model_flavor = self.cfg.get("model_flavor") if self.cfg else None
    marker_text = (
        "<|turn>model\n"
        if model_flavor == common_config.ModelFlavor.GEMMA_4
        else "<start_of_turn>model\n"
    )
    with _PROCESSOR_LOCK:
      text = self.processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=False
      )
      inputs = self.processor(
          text=text,
          images=image,
          return_tensors="pt",
          padding=False,
          do_pan_and_scan=self.do_pan_and_scan,
      )
      if self._marker_ids is None:
        self._marker_ids = list(
            self.processor.tokenizer.encode(
                marker_text, add_special_tokens=False
            )
        )
      marker = self._marker_ids

    input_ids = inputs["input_ids"].squeeze(0)
    max_seq_length = self.cfg.get("max_seq_length") if self.cfg else None
    if max_seq_length and len(input_ids) > max_seq_length:
      raise ValueError(
          f"Example has {len(input_ids)} tokens >"
          f" max_seq_length={max_seq_length}; records are not truncated."
      )
    attention_mask = inputs["attention_mask"].squeeze(0)
    pixel_values = inputs["pixel_values"]
    if pixel_values.ndim == 5:
      pixel_values = pixel_values.squeeze(0)

    ids_list = input_ids.tolist()
    ends = [
        i + len(marker)
        for i in range(len(ids_list) - len(marker) + 1)
        if marker and ids_list[i : i + len(marker)] == marker
    ]
    if not ends:
      raise ValueError(f"Assistant turn marker {marker_text!r} not found.")
    labels = input_ids.clone()
    labels[: ends[-1]] = -100

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
    }
    for key in ("token_type_ids", "image_position_ids"):
      if key in inputs:
        result[key] = inputs[key].squeeze(0)
    return result

  def __call__(
      self,
      examples,
  ):
    """Transforms a batch of examples into Gemma VLM inputs."""
    results = [
        self.transform_example(image, objects)
        for image, objects in zip(examples["image"], examples["objects"])
    ]
    keys = ["input_ids", "labels", "pixel_values", "attention_mask"]
    for key in ("token_type_ids", "image_position_ids"):
      if any(key in r for r in results):
        keys.append(key)
    return {key: [r[key] for r in results if key in r] for key in keys}


class GemmaVisionCollate:
  """Picklable collate callable for Gemma Vision data loader workers."""

  def __init__(
      self,
      is_tpu = False,
      static_max_len = None,
      pad_token_id = 0,
  ):
    self.is_tpu = is_tpu
    self.static_max_len = static_max_len
    self.pad_token_id = pad_token_id

  def __call__(self, batch):
    input_ids = [torch.as_tensor(item["input_ids"]) for item in batch]
    labels = [torch.as_tensor(item["labels"]) for item in batch]
    attention_masks = [
        torch.as_tensor(item["attention_mask"]) for item in batch
    ]
    pixel_values = torch.cat(
        [torch.as_tensor(item["pixel_values"]) for item in batch], dim=0
    )

    if self.is_tpu and self.static_max_len:
      max_len = self.static_max_len
    else:
      max_len = max(ids.shape[0] for ids in input_ids)
    padded_input_ids = torch.full(
        (len(batch), max_len), self.pad_token_id, dtype=input_ids[0].dtype
    )
    padded_labels = torch.full(
        (len(batch), max_len), -100, dtype=labels[0].dtype
    )
    padded_attention = torch.zeros(
        (len(batch), max_len), dtype=attention_masks[0].dtype
    )

    for i, (ids, lab, att) in enumerate(
        zip(input_ids, labels, attention_masks)
    ):
      seq_len = min(ids.shape[0], max_len)
      padded_input_ids[i, :seq_len] = ids[:seq_len]
      padded_labels[i, :seq_len] = lab[:seq_len]
      padded_attention[i, :seq_len] = att[:seq_len]

    result = {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "pixel_values": pixel_values.float(),
        "attention_mask": padded_attention,
    }
    if "token_type_ids" in batch[0]:
      token_types = [torch.as_tensor(item["token_type_ids"]) for item in batch]
      padded_types = torch.zeros(
          (len(batch), max_len), dtype=token_types[0].dtype
      )
      for i, tt in enumerate(token_types):
        seq_len = min(tt.shape[0], max_len)
        padded_types[i, :seq_len] = tt[:seq_len]
      result["token_type_ids"] = padded_types
    if "image_position_ids" in batch[0]:
      result["image_position_ids"] = torch.stack(
          [torch.as_tensor(item["image_position_ids"]) for item in batch],
          dim=0,
      )
    return result


def _make_transform(
    processor,
    class_names,
    cfg,
    model_id = None,
):
  c = cfg or {}
  return GemmaVisionTransform(
      processor=processor,
      class_names=class_names,
      prompt_text=c.get("prompt") or f"detect {' ; '.join(class_names)}",
      detection_format=c.get("detection_format", "loc"),
      do_pan_and_scan=c.get("do_pan_and_scan", False),
      cfg=cfg,
      model_id=model_id,
  )


class GemmaVisionPreprocessor(base.DataPreprocessor):
  """Preprocessor for Gemma vision-language detection."""

  def get_transform_fn(  # pyrefly: ignore[bad-override]
      self,
      processor,
      cfg = None,
      *,
      is_train = False,
      dataset_features,
      **kwargs,
  ):
    """Returns a transform function converting examples to Gemma VLM inputs."""
    del self  # Unused.
    class_names = dataset_features["objects"]["category"].feature.names

    # Save processor locally so DataLoader workers can load without
    # remote access.
    temp_dir = data.stage_processor_locally(
        processor, prefix="local_vision_processor_"
    )
    return _make_transform(processor, class_names, cfg, model_id=temp_dir)

  def get_preprocess_fn(
      self,
      processor,
      cfg = None,
      *,
      is_train = False,
      class_names = None,
      **kwargs,
  ):
    """Returns the per-example `{image, objects}` preprocessor for Gemma VLM."""
    del self, is_train, kwargs
    names = class_names or (cfg.dataset.get("class_names") if cfg else None)
    if not names:
      raise ValueError("Gemma vision requires class_names.")
    transform = _make_transform(processor, list(names), cfg)
    return lambda ex: transform.transform_example(ex["image"], ex["objects"])

  def get_collate_fn(
      self,
      cfg = None,
      *,
      pad_token_id = 0,
      **kwargs,
  ):
    """Returns a collation function to pad batch components in PyTorch loaders."""
    del self  # Unused.
    device_type = cfg.training.get("device", "cuda") if cfg else "cuda"
    is_tpu = device_type == "tpu"
    static_max_len = cfg.get("max_seq_length") if cfg else None

    return GemmaVisionCollate(
        is_tpu=is_tpu,
        static_max_len=static_max_len,
        pad_token_id=pad_token_id,
    )

  def get_sft_config_overrides(
      self, cfg
  ):
    """Returns SFTConfig overrides specifically for custom Gemma preprocessors.

    Args:
      cfg: Configuration dictionary containing max_seq_length and training
        flags.

    Returns:
      A dictionary containing dataset and gradient checkpointing overrides.
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


def _unwrap_gemma4_clippable(model):
  """Replaces Gemma4ClippableLinear layers with nn.Linear.

  This is necessary because PEFT modules like LoRA are not compatible
  with custom linear layers like Gemma4ClippableLinear. Unwrapping them
  to standard nn.Linear allows PEFT to be applied correctly.

  NOTE: The actual dispatch logic in //third_party/py/peft/tuners/lora/
  layer.py (dispatch_default) only checks for nn.Linear, not
  Gemma4ClippableLinear. This workaround remains necessary until PEFT's
  dispatcher is updated.

  Args:
    model: The PyTorch model to modify.

  Returns:
    The modified model with Gemma4ClippableLinear layers replaced.
  """
  try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear  # pylint: disable=g-import-not-at-top
  except (ImportError, AttributeError, ModuleNotFoundError):
    logging.warning(
        "Could not import Gemma4ClippableLinear. Skipping unwrapping."
    )
    return model

  count = 0
  for _, module in list(model.named_modules()):
    for attr_name, child in list(module.named_children()):
      if isinstance(child, Gemma4ClippableLinear):
        setattr(module, attr_name, child.linear)
        count += 1
  if count > 0:
    logging.info(
        "Unwrapped %d Gemma4ClippableLinear -> nn.Linear for PEFT.", count
    )
  return model


class GemmaVisionEngine(base.ModelEngine):
  """Coordinated training and evaluation setup for the Gemma VLM.

  This manages model loading configurations, customized dataset transforms, and
  SFT overrides.
  """

  def __init__(self):
    """Initializes the instance."""
    super().__init__(
        preprocessor=GemmaVisionPreprocessor(),
        loss_handler=None,  # SFT computes loss internally in model forward().
        decoder=None,  # Pluggable decoders are passed config-wise in
        # orchestration.
    )

  def load_model_and_processor(
      self,
      model_id,
      device,
      *,
      use_torchao = False,
      precision = "bf16",
      **kwargs,
  ):
    """Loads the pretrained Gemma visual language model and its processor.

    Args:
      model_id: Pretrained repository ID or checkpoint path.
      device: PyTorch target mapping device.
      use_torchao: Whether to enable TorchAO activation/weight quantization.
      precision: Model float loading datatype precision (e.g., "bf16", "fp32").
      **kwargs: Additional config options.

    Returns:
      A tuple (model, processor), where model is the initialized
      AutoModelForImageTextToText model on the device, and processor is the
      AutoProcessor instance.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Safetensor wrapper slice index error fallback monkey-patch
    try:
      import transformers.core_model_loading as _cml  # pylint: disable=g-import-not-at-top

      original_materialize = _cml._materialize_copy  # pylint: disable=protected-access

      def patched_materialize(tensor, device=None, dtype=None):
        try:
          return original_materialize(tensor, device, dtype)
        except Exception as e:
          if "SafetensorError" in type(e).__name__:
            res = tensor[Ellipsis]
            if dtype is not None or device is not None:
              res = res.to(device=device, dtype=dtype)
            return res
          raise

      _cml._materialize_copy = patched_materialize  # pylint: disable=protected-access
    except (ImportError, AttributeError):
      pass

    processor = transformers.AutoProcessor.from_pretrained(model_id)
    base_vocab_size = len(processor.tokenizer)

    cfg = kwargs.get("cfg")
    detection_format = cfg.get("detection_format", "loc") if cfg else "loc"
    if detection_format == "loc":
      loc_tokens = [f"<loc{i:04d}>" for i in range(1024)]
      num_added = processor.tokenizer.add_tokens(loc_tokens)
      logging.info("Added %s new <loc> tokens to the vocabulary.", num_added)

    dtype_by_precision = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_dtype = dtype_by_precision.get(precision, torch.float32)

    load_kwargs = {
        "torch_dtype": model_dtype,
        "attn_implementation": (
            "sdpa" if torch.cuda.is_available() else "eager"
        ),
        **(
            {"quantization_config": GemmaVisionEngine._get_torchao_config()}
            if use_torchao
            else {}
        ),
    }
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_fsdp = bool(cfg and cfg.training.get("fsdp") and world_size > 1)
    if torch.cuda.is_available() and not use_fsdp:
      load_kwargs["device_map"] = {"": local_rank}

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id, **load_kwargs
    )

    if detection_format != "json":
      model.resize_token_embeddings(
          len(processor.tokenizer),
          pad_to_multiple_of=64,
          mean_resizing=False,
      )

    model.config.use_cache = False
    processor.tokenizer.padding_side = "right"

    if device.type in ["tpu", "xla"]:
      logging.info("Migrating Gemma Vision model to TPU...")
      model = model.to(device)

    if cfg and cfg.training.get("use_lora", True):
      import peft  # pylint: disable=g-import-not-at-top

      new_token_ids = list(range(base_vocab_size, len(processor.tokenizer)))
      trainable_tokens = None
      if new_token_ids:
        trainable_tokens = (
            new_token_ids
            if getattr(model.config, "tie_word_embeddings", False)
            else {"embed_tokens": new_token_ids, "lm_head": new_token_ids}
        )
      peft_config = peft.LoraConfig(
          r=cfg.lora.r,
          lora_alpha=cfg.lora.alpha,
          target_modules=list(cfg.lora.target_modules),
          exclude_modules=(
              ".*vision_tower.*"
              if cfg.training.get("freeze_vision_tower", False)
              else None
          ),
          lora_dropout=cfg.lora.dropout,
          bias="none",
          task_type="CAUSAL_LM",
          trainable_token_indices=trainable_tokens,
      )
      model = _unwrap_gemma4_clippable(model)
      peft_model = peft.get_peft_model(model, peft_config)  # pyrefly: ignore[bad-argument-type]
      lora_params = sum(
          p.numel()
          for n, p in peft_model.named_parameters()
          if p.requires_grad and "lora_" in n
      )
      if not lora_params:
        raise ValueError(
            "No module matched cfg.lora.target_modules="
            f"{list(cfg.lora.target_modules)}."
        )
      trainable, total = peft_model.get_nb_trainable_parameters()
      logging.info(
          "LoRA: %d/%d trainable (%d in adapters).",
          trainable,
          total,
          lora_params,
      )
      return peft_model, processor

    return model, processor

  @classmethod
  def _get_torchao_config(cls):
    """Returns the TorchAO quantization config for Gemma."""
    import torchao  # pylint: disable=g-import-not-at-top

    attn_re = (
        r"re:model\.language_model\.layers\.\d+"
        r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    )
    mlp_re = (
        r"re:model\.language_model\.layers\.\d+"
        r"\.mlp\.(gate_proj|up_proj|down_proj)"
    )
    mapping = collections.OrderedDict([
        (attn_re, torchao.quantization.Int4WeightOnlyConfig()),
        (mlp_re, torchao.quantization.Int8WeightOnlyConfig()),
    ])
    return transformers.TorchAoConfig(torchao.quantization.FqnToConfig(mapping))  # pyrefly: ignore[bad-argument-type]

  def freeze_vision_tower(self, model):
    """Freezes all parameters in the vision tower."""
    del self  # Unused.
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
      model = model.base_model.model

    vision_tower = None
    if hasattr(model, "vision_tower"):
      vision_tower = model.vision_tower
    elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
      vision_tower = model.model.vision_tower

    if vision_tower is not None:
      for param in vision_tower.parameters():
        param.requires_grad = False
    else:
      logging.warning("Could not find vision tower to freeze.")

  def get_transform_fn(
      self,
      processor,
      text_inputs = (),
      dataset_id2label = (),
      model_label2id = ml_collections.FrozenConfigDict({}),  # pyrefly: ignore[bad-function-definition]
      cfg = None,
      is_train = False,
      **kwargs,
  ):
    """Delegates to composed preprocessor.

    Args:
      processor: The model's AutoProcessor instance to process texts & images.
      text_inputs: Unused legacy positional arguments.
      dataset_id2label: Unused legacy positional arguments.
      model_label2id: Unused legacy positional arguments.
      cfg: Optional configuration dictionary containing hyper-parameters.
      is_train: Whether the returned function will be used for training mode.
      **kwargs: Additional keyword arguments passed to preprocessor.

    Returns:
      A transform callable function.
    """
    if self.preprocessor is None:
      raise ValueError("GemmaVisionEngine preprocessor is not initialized.")
    dataset_features = kwargs.pop("dataset_features", None)
    return self.preprocessor.get_transform_fn(
        processor,
        cfg,
        is_train=is_train,
        dataset_features=dataset_features,
        **kwargs,
    )
