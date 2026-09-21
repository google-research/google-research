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
from typing import Any

from absl import logging
import ml_collections
import torch
from torch import nn
import transformers

from Uboreshaji_Modeli.common import config as common_config
from Uboreshaji_Modeli.common import data
from Uboreshaji_Modeli.common import gemma_box_utils
from . import base


class GemmaVisionTransform(data.PicklableProcessorMixin):
  """Picklable transform callable for Gemma Vision data loader workers.

  Uses ``__getstate__``/``__setstate__`` to re-load the processor from
  disk in each DataLoader worker process, since the processor's native
  backends do not survive Python pickling.

  On reload, custom ``<loc>`` tokens are re-added to the tokenizer
  when ``detection_format == 'loc'``, matching the setup in
  ``GemmaVisionEngine.load_model_and_processor``.
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

  def _post_load_processor(self):
    if self.detection_format == "loc":
      loc_tokens = [f"<loc{i:04d}>" for i in range(1024)]
      num_added = self.processor.tokenizer.add_tokens(loc_tokens)
      logging.debug("Added %d new loc tokens to vocabulary.", num_added)
    self.processor.tokenizer.padding_side = "right"

  def __call__(
      self,
      examples,
  ):
    """Transforms a batch of examples into Gemma VLM inputs."""
    logging.info(
        "GemmaVisionTransform called with %d examples", len(examples["image"])
    )
    all_input_ids = []
    all_labels = []
    all_pixel_values = []
    all_attention_masks = []
    all_image_position_ids = []

    for i, (image, objects) in enumerate(
        zip(examples["image"], examples["objects"])
    ):
      logging.info("Processing example %d...", i)
      image = image.convert("RGB")
      w, h = image.size

      logging.info("Formatting objects to detection string...")
      detection_string = gemma_box_utils.format_objects_to_detection_string(
          objects,
          self.class_names,
          image_width=w,
          image_height=h,
          detection_format=self.detection_format,
      )
      logging.info("Detection string: %s", detection_string)

      if (
          self.cfg is not None
          and "dataset" in self.cfg
          and "image_size" in self.cfg.dataset
          and self.cfg.dataset.image_size
      ):
        logging.info("Resizing image to %d...", self.cfg.dataset.image_size)
        image = image.resize(
            (self.cfg.dataset.image_size, self.cfg.dataset.image_size)
        )

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
              "content": [
                  {"type": "text", "text": detection_string},
              ],
          },
      ]

      logging.info("Applying chat template...")
      text = self.processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=False
      )
      logging.info("Chat template applied.")

      logging.info(
          "Calling processor with do_pan_and_scan=%s...", self.do_pan_and_scan
      )
      inputs = self.processor(
          text=text,
          images=image,
          return_tensors="pt",
          padding=False,
          do_pan_and_scan=self.do_pan_and_scan,
      )
      logging.info("Processor finished.")

      input_ids = inputs["input_ids"].squeeze(0)
      attention_mask = inputs["attention_mask"].squeeze(0)
      pixel_values = inputs["pixel_values"]
      if pixel_values.ndim == 5:
        pixel_values = pixel_values.squeeze(0)

      labels = input_ids.clone()
      model_flavor = self.cfg.get("model_flavor") if self.cfg else None
      if model_flavor == common_config.ModelFlavor.GEMMA_4:
        assistant_token = self.processor.tokenizer.encode(
            "<|turn>model\n", add_special_tokens=False
        )
      else:
        assistant_token = self.processor.tokenizer.encode(
            "\nmodel\n", add_special_tokens=False
        )
      logging.info("Searching for assistant token...")
      for j in range(len(input_ids) - len(assistant_token) + 1):
        if (
            input_ids[j : j + len(assistant_token)].tolist()
            == assistant_token
        ):
          prompt_len = j + len(assistant_token)
          break
      else:
        prompt_len = 0
      logging.info("Searched for assistant token. prompt_len: %d", prompt_len)

      if prompt_len > 0:
        labels[:prompt_len] = -100

      all_input_ids.append(input_ids)
      all_labels.append(labels)
      all_pixel_values.append(pixel_values)
      all_attention_masks.append(attention_mask)
      if "image_position_ids" in inputs:
        all_image_position_ids.append(inputs["image_position_ids"].squeeze(0))

    logging.info("GemmaVisionTransform finished processing all examples.")
    result = {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "pixel_values": all_pixel_values,
        "attention_mask": all_attention_masks,
    }
    if all_image_position_ids:
      result["image_position_ids"] = all_image_position_ids
    return result


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

    image_position_ids = None
    has_image_position_ids = "image_position_ids" in batch[0]
    if has_image_position_ids:
      image_position_ids = torch.stack(
          [torch.as_tensor(item["image_position_ids"]) for item in batch],
          dim=0,
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
    if has_image_position_ids:
      result["image_position_ids"] = image_position_ids
    return result


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
    prompt_text = (
        cfg.get("prompt")
        if cfg is not None and cfg.get("prompt")
        else f"detect {' ; '.join(class_names)}"
    )
    detection_format = cfg.get("detection_format", "loc") if cfg else "loc"
    do_pan_and_scan = cfg.get("do_pan_and_scan", False) if cfg else False

    # Save processor locally so DataLoader workers can load without
    # remote access.
    temp_dir = data.stage_processor_locally(
        processor, prefix="local_vision_processor_"
    )

    return GemmaVisionTransform(
        processor=processor,
        class_names=class_names,
        prompt_text=prompt_text,
        detection_format=detection_format,
        do_pan_and_scan=do_pan_and_scan,
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

    # Determine detection format from config.
    cfg = kwargs.get("cfg")
    detection_format = cfg.get("detection_format", "loc") if cfg else "loc"

    # Add custom tokens for location — only needed for loc token format.
    if detection_format == "loc":
      loc_tokens = [f"<loc{i:04d}>" for i in range(1024)]
      processor.tokenizer.add_tokens(loc_tokens)

    dtype_by_precision = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_dtype = dtype_by_precision.get(precision, torch.float32)

    load_kwargs = {
        "torch_dtype": model_dtype,
        "attn_implementation": "eager",
        **(
            {"quantization_config": GemmaVisionEngine._get_torchao_config()}
            if use_torchao
            else {}
        ),
    }
    if torch.cuda.is_available():
      load_kwargs["device_map"] = {"": local_rank}

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id, **load_kwargs
    )

    # Resize embeddings to match new tokenizer — only for loc token format.
    # JSON format uses the model's native vocabulary, no resize needed.
    if detection_format != "json":
      # Disable mean_resizing to avoid covariance alloc OOMs on nearly-full
      # GPUs. Random init is fine — these <loc> tokens are fully fine-tuned.
      model.resize_token_embeddings(
          len(processor.tokenizer), mean_resizing=False
      )

    model.config.use_cache = False
    processor.tokenizer.padding_side = "right"

    if device.type in ["tpu", "xla"]:
      logging.info("Migrating Gemma Vision model to TPU...")
      model = model.to(device)

    if cfg and cfg.training.get("use_lora", True):
      import peft  # pylint: disable=g-import-not-at-top

      peft_config = peft.LoraConfig(
          r=cfg.lora.r,
          lora_alpha=cfg.lora.alpha,
          target_modules=list(cfg.lora.target_modules),
          lora_dropout=cfg.lora.dropout,
          bias="none",
          task_type="CAUSAL_LM",
          modules_to_save=["embed_tokens", "lm_head"],
      )
      model = _unwrap_gemma4_clippable(model)
      peft_model = peft.get_peft_model(model, peft_config)  # pyrefly: ignore[bad-argument-type]
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
