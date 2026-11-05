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

"""Custom SFTTrainer wrapper for multimodal and PEFT fine-tuning."""

import trl


class CustomSFTTrainer(trl.SFTTrainer):
  """SFT trainer that delegates ``hf_quantizer`` from PEFT base models.

  When a quantized model is wrapped by ``PeftModel``, the ``hf_quantizer``
  attribute stays on the base model rather than being proxied directly on the
  wrapper, preventing ``SFTTrainer`` from seeing it.

  Delegating ``hf_quantizer`` ensures k-bit training configurations (such as
  gradient checkpointing setup for quantized models) are applied properly.
  """

  def _patch_hf_quantizer(self, model):
    """Delegates hf_quantizer from the base model if needed."""
    if model is not None:
      # Check for get_base_model which is characteristic of PeftModel wrappers.
      if not hasattr(model, "hf_quantizer") and hasattr(
          model, "get_base_model"
      ):
        base_model = model.get_base_model()
        if hasattr(base_model, "hf_quantizer"):
          model.hf_quantizer = base_model.hf_quantizer

  def __init__(self, *args, **kwargs):
    # If a model instance is passed, patch it before calling super().__init__
    # so that the parent class can properly initialize based on the model's
    # properties, such as setting up gradient checkpointing for quantized
    # models.
    model_arg = kwargs.get("model") or (args[0] if args else None)
    if model_arg is not None:
      self._patch_hf_quantizer(model_arg)

    super().__init__(*args, **kwargs)


    # Patch self.model again after super().__init__ in case the model
    # was loaded from a string path by the parent class, or if the initial
    # patching was somehow insufficient.
    self._patch_hf_quantizer(getattr(self, "model", None))

  def create_model_card(self, *args, **kwargs):
    """No-op: skips TRL HuggingFace Hub model card template generation.

    Args:
      *args: Positional arguments passed to the parent class.
      **kwargs: Keyword arguments passed to the parent class.

    Returns:
      None.
    """
