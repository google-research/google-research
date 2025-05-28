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

"""Extension of the huggingface T5forConditionalGeneration which also supports our special interpretability functions and efficient sampling.

Currently does not support beam sampling or nucleus sampling; however, there
are no technical limitations to applying the same tecniques of speculative
decoding to these other common methods of top-K and top-P generation.
"""

import copy
import inspect
import time
from typing import Callable, List, Optional, Union
import warnings

import numpy as np
import torch
import torch.distributed as dist
from transformers import LogitsProcessorList
from transformers import StoppingCriteriaList
from transformers import T5ForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
from transformers.generation.utils import GreedySearchEncoderDecoderOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ExplicitEnum
from transformers.utils import logging


logger = logging.get_logger(__name__)


# pylint: disable=function-redefined
class GenerationMode(ExplicitEnum):
  """Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method."""

  # Non-beam methods
  CONTRASTIVE_SEARCH = "contrastive_search"
  GREEDY_SEARCH = "greedy_search"
  SAMPLE = "sample"
  ASSISTED_GENERATION = "assisted_generation"

  # new interpretability methods
  TREE_ASSISTED_GEN = "tree_assisted_generation"
  FID_INTERPRETABILITY_GEN = "fid_encoding_interpretability_generation"

  # Beam methods
  BEAM_SEARCH = "beam_search"
  BEAM_SAMPLE = "beam_sample"
  CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
  GROUP_BEAM_SEARCH = "group_beam_search"

  # TODO(enounen): might need to reattach this function to the
  # 'T5ForInterpretableGeneration' class since it might be that the old code is
  # inherited from 'T5ForConditionalGeneration' class
  def _get_generation_mode(
      self,
      generation_config,
      assistant_model,
      speculative_generation_tree=None,
      number_of_out_decodings=None,
  ):
    """Returns the generation mode triggered by a [`GenerationConfig`] instance."""
    if (
        generation_config.constraints is not None
        or generation_config.force_words_ids is not None
    ):
      generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
    elif generation_config.num_beams == 1:
      if not generation_config.do_sample:
        if (
            generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
        ):
          generation_mode = GenerationMode.CONTRASTIVE_SEARCH
        else:
          generation_mode = GenerationMode.GREEDY_SEARCH
      else:
        generation_mode = GenerationMode.SAMPLE
    else:
      if generation_config.num_beam_groups > 1:
        generation_mode = GenerationMode.GROUP_BEAM_SEARCH
      elif generation_config.do_sample:
        generation_mode = GenerationMode.BEAM_SAMPLE
      else:
        generation_mode = GenerationMode.BEAM_SEARCH

    # Assisted generation may extend some generation modes
    if assistant_model is not None:
      if generation_mode in ("greedy_search", "sample"):
        generation_mode = GenerationMode.ASSISTED_GENERATION
      else:
        raise ValueError(
            "You've set `assistant_model`, which triggers assisted generate."
            " Currently, assisted generate is only supported with Greedy Search"
            " and Sample."
        )
    if speculative_generation_tree is not None:
      generation_mode = GenerationMode.TREE_ASSISTED_GEN
    if number_of_out_decodings is not None:
      generation_mode = GenerationMode.FID_INTERPRETABILITY_GEN

    return generation_mode


GreedySearchOutput = Union[
    GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
]


class T5ForInterpretableGeneration(T5ForConditionalGeneration):
  """Extension of T5ForConditionalGeneration with support for special interpretability functions and efficient sampling."""

  # replaces first layer's needs_decoder_pos so that it will generate the
  # necessary position biases
  def update_decoder_position_biases(self, needs_decoder_pos, pos_biases=None):
    i = 0
    self.decoder.block[i].layer[
        0
    ].SelfAttention.needs_decoder_positions = needs_decoder_pos
    self.decoder.block[i].layer[
        0
    ].SelfAttention.decoder_position_bias_indices = pos_biases

  @torch.no_grad()
  def jam_speculative_generate(
      self,
      inputs = None,
      generation_config = None,
      logits_processor = None,
      stopping_criteria = None,
      prefix_allowed_tokens_fn = None,
      synced_gpus = None,
      assistant_model = None,
      streamer = None,
      negative_prompt_ids = None,
      negative_prompt_attention_mask = None,
      speculative_generation_tree=None,
      **kwargs,
  ):

    if synced_gpus is None:
      if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
        synced_gpus = True
      else:
        synced_gpus = False

    # 1. Handle `generation_config` and kwargs that might update it,
    # and validate the `.generate()` call
    self._validate_model_class()

    # priority: `generation_config` argument > `model.generation_config`
    # (the default generation config)
    if generation_config is None:
      # legacy: users may modify the model configuration to control
      # generation -- update the generation config model attribute accordingly,
      # if it was created from the model config
      # pylint: disable=protected-access
      if self.generation_config._from_model_config:
        new_generation_config = GenerationConfig.from_model_config(self.config)
        if new_generation_config != self.generation_config:
          warnings.warn(
              "You have modified the pretrained model configuration to control"
              " generation. This is a"
              " deprecated strategy to control generation and will be removed"
              " soon, in a future version."
              " Please use a generation configuration file (see"
              " https://huggingface.co/docs/transformers/main_classes/text_generation )"
          )
          self.generation_config = new_generation_config
      generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(
        **kwargs
    )  # All unused kwargs must be model kwargs
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = (
        logits_processor
        if logits_processor is not None
        else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria
        if stopping_criteria is not None
        else StoppingCriteriaList()
    )

    if (
        generation_config.pad_token_id is None
        and generation_config.eos_token_id is not None
    ):
      if model_kwargs.get("attention_mask", None) is None:
        logger.warning(
            "The attention mask and the pad token id were not set. As a"
            " consequence, you may observe unexpected behavior. Please pass"
            " your input's `attention_mask` to obtain reliable results."
        )
      eos_token_id = generation_config.eos_token_id
      if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
      logger.warning(
          f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for"
          " open-end generation."
      )
      generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = (
        generation_config.output_hidden_states
    )
    # decoder-only models with inputs_embeds forwarding must use caching
    # (otherwise we can't detect whether we are generating the first new token
    # or not, and we only want to use the embeddings for the first new token)
    if (
        not self.config.is_encoder_decoder
        and model_input_name == "inputs_embeds"
    ):
      model_kwargs["use_cache"] = True
    else:
      model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(
        inspect.signature(self.forward).parameters.keys()
    )
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if (
        model_kwargs.get("attention_mask", None) is None
        and requires_attention_mask
        and accepts_attention_mask
    ):
      model_kwargs["attention_mask"] = (
          self._prepare_attention_mask_for_generation(
              inputs_tensor,
              generation_config.pad_token_id,
              generation_config.eos_token_id,
          )
      )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
      # If `input_ids` was given, check if the last id in any sequence is
      # `pad_token_id`. Note: If using, `inputs_embeds` this check does not
      # work, because we want to be more hands-off.
      if (
          generation_config.pad_token_id is not None
          and len(inputs_tensor.shape) == 2
          and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id)
          > 0
      ):
        logger.warning(
            "A decoder-only architecture is being used, but right-padding was"
            " detected! For correct generation results, please set"
            " `padding_side='left'` when initializing the tokenizer."
        )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
      # if model is encoder decoder encoder_outputs are created
      # and added to `model_kwargs`
      model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
          inputs_tensor, model_kwargs, model_input_name
      )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    # JAM: this is what I need to change in order to alter 'decoder_input_ids'
    if self.config.is_encoder_decoder:
      input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
          batch_size=batch_size,
          model_input_name=model_input_name,
          model_kwargs=model_kwargs,
          decoder_start_token_id=generation_config.decoder_start_token_id,
          bos_token_id=generation_config.bos_token_id,
          device=inputs_tensor.device,
      )
    else:
      input_ids = (
          inputs_tensor
          if model_input_name == "input_ids"
          else model_kwargs.pop("input_ids")
      )
    # 'input_ids' = 'decoder_input_ids' right now
    # and 'decoder_attention_mask' is inside of the 'model_kwargs'

    if speculative_generation_tree is not None:
      original_batch_size = input_ids.shape[0]
      input_ids = torch.repeat_interleave(
          torch.LongTensor(speculative_generation_tree["tok_list"])[None],
          repeats=original_batch_size,
          dim=0,
      ).to(inputs_tensor.device)
      decoder_attention_mask = torch.repeat_interleave(
          speculative_generation_tree["attn_mask"][None],
          repeats=original_batch_size,
          dim=0,
      ).to(inputs_tensor.device)
      pos_bias_indices = speculative_generation_tree["pos_bias_inds"].to(
          inputs_tensor.device
      )

      model_kwargs["decoder_attention_mask"] = decoder_attention_mask
      self.update_decoder_position_biases(True, pos_bias_indices)

    if streamer is not None:
      streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = (
        kwargs.get("max_length") is None
        and generation_config.max_length is not None
    )
    if generation_config.max_new_tokens is not None:
      if not has_default_max_length:
        logger.warning(
            f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and"
            " `max_length`(="
            f"{generation_config.max_length}) seem to have been set."
            " `max_new_tokens` will take precedence. "
            "Please refer to the documentation for more information. "
            "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
        )
      generation_config.max_length = (
          generation_config.max_new_tokens + input_ids_length
      )
    self._validate_generated_length(
        generation_config, input_ids_length, has_default_max_length
    )

    # 7. determine generation mode
    generation_mode = self._get_generation_mode(
        generation_config, assistant_model, speculative_generation_tree
    )

    if streamer is not None and (generation_config.num_beams > 1):
      raise ValueError(
          "`streamer` cannot be used with beam search (yet!). Make sure that"
          " `num_beams` is set to 1."
      )

    if self.device.type != input_ids.device.type:
      warnings.warn(
          "You are calling .generate() with the `input_ids` being on a device"
          " type different than your model's device. `input_ids` is on"
          f" {input_ids.device.type}, whereas the model is on"
          f" {self.device.type}. You may experience unexpected behaviors or"
          " slower generation. Please make sure that you have put `input_ids`"
          " to the correct device by calling for example input_ids ="
          f" input_ids.to('{self.device.type}') before running `.generate()`.",
          UserWarning,
      )

    # 8. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
      if generation_config.num_return_sequences > 1:
        raise ValueError(
            "num_return_sequences has to be 1 when doing assisted generate, "
            f"but is {generation_config.num_return_sequences}."
        )
      if batch_size > 1:
        raise ValueError(
            "assisted generate is only supported for batch_size = 1"
        )
      if not model_kwargs["use_cache"]:
        raise ValueError("assisted generate requires `use_cache=True`")

      # 11. If the assistant model is an encoder-decoder, prepare its
      # encoder outputs
      if assistant_model.config.is_encoder_decoder:
        assistant_model_kwargs = copy.deepcopy(model_kwargs)
        inputs_tensor, model_input_name, assistant_model_kwargs = (
            # pylint: disable=protected-access
            assistant_model._prepare_model_inputs(
                inputs_tensor,
                assistant_model.generation_config.bos_token_id,
                assistant_model_kwargs,
            )
        )
        assistant_model_kwargs = (
            # pylint: disable=protected-access
            assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_model_kwargs, model_input_name
            )
        )
        model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs[
            "encoder_outputs"
        ]

      # 12. run assisted generate
      return self.assisted_decoding(
          input_ids,
          assistant_model=assistant_model,
          do_sample=generation_config.do_sample,
          logits_processor=logits_processor,
          logits_warper=self._get_logits_warper(generation_config)
          if generation_config.do_sample
          else None,
          stopping_criteria=stopping_criteria,
          pad_token_id=generation_config.pad_token_id,
          eos_token_id=generation_config.eos_token_id,
          output_scores=generation_config.output_scores,
          return_dict_in_generate=generation_config.return_dict_in_generate,
          synced_gpus=synced_gpus,
          streamer=streamer,
          **model_kwargs,
      )
    if generation_mode == GenerationMode.TREE_ASSISTED_GEN:
      # 11. run greedy search
      tree_outputs = self.speculation_tree_assisted_greedy(
          input_ids,
          logits_processor=logits_processor,
          stopping_criteria=stopping_criteria,
          pad_token_id=generation_config.pad_token_id,
          eos_token_id=generation_config.eos_token_id,
          output_scores=generation_config.output_scores,
          return_dict_in_generate=generation_config.return_dict_in_generate,
          synced_gpus=synced_gpus,
          streamer=streamer,
          speculative_generation_tree=speculative_generation_tree,
          **model_kwargs,
      )
      return tree_outputs

    if generation_mode == GenerationMode.GREEDY_SEARCH:
      # 11. run greedy search
      return self.greedy_search(
          input_ids,
          logits_processor=logits_processor,
          stopping_criteria=stopping_criteria,
          pad_token_id=generation_config.pad_token_id,
          eos_token_id=generation_config.eos_token_id,
          output_scores=generation_config.output_scores,
          return_dict_in_generate=generation_config.return_dict_in_generate,
          synced_gpus=synced_gpus,
          streamer=streamer,
          **model_kwargs,
      )

  @torch.no_grad()
  def jam_FiD_enc_interpretation(
      self,
      inputs = None,
      generation_config = None,
      logits_processor = None,
      stopping_criteria = None,
      prefix_allowed_tokens_fn = None,
      synced_gpus = None,
      assistant_model = None,
      streamer = None,
      negative_prompt_ids = None,
      negative_prompt_attention_mask = None,
      speculative_generation_tree=None,
      number_of_out_decodings=None,
      tokenizer=None,
      interpretability_mode=None,
      **kwargs,
  ):

    if synced_gpus is None:
      if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
        synced_gpus = True
      else:
        synced_gpus = False

    # 1. Handle `generation_config` and kwargs that might update it, and
    # validate the `.generate()` call
    self._validate_model_class()

    # priority: `generation_config` argument > `model.generation_config` (the
    # default generation config)
    if generation_config is None:
      # legacy: users may modify the model configuration to control generation
      # -- update the generation config model attribute accordingly, if it was
      # created from the model config
      # pylint: disable=protected-access
      if self.generation_config._from_model_config:
        # pylint: disable=protected-access
        new_generation_config = GenerationConfig.from_model_config(self.config)
        if new_generation_config != self.generation_config:
          warnings.warn(
              "You have modified the pretrained model configuration to control"
              " generation. This is a"
              " deprecated strategy to control generation and will be removed"
              " soon, in a future version."
              " Please use a generation configuration file (see"
              " https://huggingface.co/docs/transformers/main_classes/text_generation )"
          )
          self.generation_config = new_generation_config
      generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(
        **kwargs
    )  # All unused kwargs must be model kwargs
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = (
        logits_processor
        if logits_processor is not None
        else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria
        if stopping_criteria is not None
        else StoppingCriteriaList()
    )

    if (
        generation_config.pad_token_id is None
        and generation_config.eos_token_id is not None
    ):
      if model_kwargs.get("attention_mask", None) is None:
        logger.warning(
            "The attention mask and the pad token id were not set. As a"
            " consequence, you may observe unexpected behavior. Please pass"
            " your input's `attention_mask` to obtain reliable results."
        )
      eos_token_id = generation_config.eos_token_id
      if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
      logger.warning(
          f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for"
          " open-end generation."
      )
      generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = (
        generation_config.output_hidden_states
    )
    # decoder-only models with inputs_embeds forwarding must use caching
    # (otherwise we can't detect whether we are generating the first new token
    # or not, and we only want to use the embeddings for the first new token)
    if (
        not self.config.is_encoder_decoder
        and model_input_name == "inputs_embeds"
    ):
      model_kwargs["use_cache"] = True
    else:
      model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(
        inspect.signature(self.forward).parameters.keys()
    )
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if (
        model_kwargs.get("attention_mask", None) is None
        and requires_attention_mask
        and accepts_attention_mask
    ):
      model_kwargs["attention_mask"] = (
          self._prepare_attention_mask_for_generation(
              inputs_tensor,
              generation_config.pad_token_id,
              generation_config.eos_token_id,
          )
      )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
      # If `input_ids` was given, check if the last id in any sequence is
      # `pad_token_id`. Note: If using, `inputs_embeds` this check does not
      # work, because we want to be more hands-off.
      if (
          generation_config.pad_token_id is not None
          and len(inputs_tensor.shape) == 2
          and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id)
          > 0
      ):
        logger.warning(
            "A decoder-only architecture is being used, but right-padding was"
            " detected! For correct generation results, please set"
            " `padding_side='left'` when initializing the tokenizer."
        )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
      # if model is encoder decoder encoder_outputs are created
      # and added to `model_kwargs`
      model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
          inputs_tensor, model_kwargs, model_input_name
      )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    # JAM: this is what I need to change in order to alter 'decoder_input_ids'
    if self.config.is_encoder_decoder:
      input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
          batch_size=batch_size,
          model_input_name=model_input_name,
          model_kwargs=model_kwargs,
          decoder_start_token_id=generation_config.decoder_start_token_id,
          bos_token_id=generation_config.bos_token_id,
          device=inputs_tensor.device,
      )
    else:
      input_ids = (
          inputs_tensor
          if model_input_name == "input_ids"
          else model_kwargs.pop("input_ids")
      )
    # 'input_ids' = 'decoder_input_ids' right now
    # and 'decoder_attention_mask' is inside of the 'model_kwargs'

    if speculative_generation_tree is not None:
      input_ids = torch.LongTensor(speculative_generation_tree["tok_list"])[
          None
      ].to(inputs_tensor.device)
      decoder_attention_mask = speculative_generation_tree["attn_mask"][
          None
      ].to(inputs_tensor.device)
      pos_bias_indices = speculative_generation_tree["pos_bias_inds"].to(
          inputs_tensor.device
      )

      model_kwargs["decoder_attention_mask"] = decoder_attention_mask
      self.update_decoder_position_biases(True, pos_bias_indices)

    if number_of_out_decodings is not None:
      input_ids = input_ids.repeat_interleave(number_of_out_decodings, axis=0)

    if streamer is not None:
      streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = (
        kwargs.get("max_length") is None
        and generation_config.max_length is not None
    )
    if generation_config.max_new_tokens is not None:
      if not has_default_max_length:
        logger.warning(
            f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and"
            " `max_length`(="
            f"{generation_config.max_length}) seem to have been set."
            " `max_new_tokens` will take precedence. "
            "Please refer to the documentation for more information. "
            "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
        )
      generation_config.max_length = (
          generation_config.max_new_tokens + input_ids_length
      )
    self._validate_generated_length(
        generation_config, input_ids_length, has_default_max_length
    )

    # 7. determine generation mode
    generation_mode = self._get_generation_mode(
        generation_config,
        assistant_model,
        speculative_generation_tree,
        number_of_out_decodings,
    )

    if streamer is not None and (generation_config.num_beams > 1):
      raise ValueError(
          "`streamer` cannot be used with beam search (yet!). Make sure that"
          " `num_beams` is set to 1."
      )

    if self.device.type != input_ids.device.type:
      warnings.warn(
          "You are calling .generate() with the `input_ids` being on a device"
          " type different than your model's device. `input_ids` is on"
          f" {input_ids.device.type}, whereas the model is on"
          f" {self.device.type}. You may experience unexpected behaviors or"
          " slower generation. Please make sure that you have put `input_ids`"
          " to the correct device by calling for example input_ids ="
          f" input_ids.to('{self.device.type}') before running `.generate()`.",
          UserWarning,
      )

    # 8. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    if generation_mode == GenerationMode.FID_INTERPRETABILITY_GEN:
      # here is where i can put the actual interpretability sampling structure
      # (e.g. perm sampling)
      interp_outputs = self.jam_FiD_interp_greedy_v2(
          input_ids,
          logits_processor=logits_processor,
          stopping_criteria=stopping_criteria,
          pad_token_id=generation_config.pad_token_id,
          eos_token_id=generation_config.eos_token_id,
          output_scores=generation_config.output_scores,
          return_dict_in_generate=generation_config.return_dict_in_generate,
          synced_gpus=synced_gpus,
          streamer=streamer,
          number_of_out_decodings=number_of_out_decodings,
          tokenizer=tokenizer,
          interpretability_mode=interpretability_mode,
          **model_kwargs,
      )
      return interp_outputs

    if generation_mode == GenerationMode.TREE_ASSISTED_GEN:
      # 11. run greedy search
      tree_outputs = self.speculation_tree_assisted_greedy(
          input_ids,
          logits_processor=logits_processor,
          stopping_criteria=stopping_criteria,
          pad_token_id=generation_config.pad_token_id,
          eos_token_id=generation_config.eos_token_id,
          output_scores=generation_config.output_scores,
          return_dict_in_generate=generation_config.return_dict_in_generate,
          synced_gpus=synced_gpus,
          streamer=streamer,
          speculative_generation_tree=speculative_generation_tree,
          **model_kwargs,
      )

      return tree_outputs

    if generation_mode == GenerationMode.GREEDY_SEARCH:
      self.update_decoder_position_biases(False)
      # 11. run greedy search
      return self.greedy_search(
          input_ids,
          logits_processor=logits_processor,
          stopping_criteria=stopping_criteria,
          pad_token_id=generation_config.pad_token_id,
          eos_token_id=generation_config.eos_token_id,
          output_scores=generation_config.output_scores,
          return_dict_in_generate=generation_config.return_dict_in_generate,
          synced_gpus=synced_gpus,
          streamer=streamer,
          **model_kwargs,
      )

  def speculation_tree_assisted_greedy(
      self,
      input_ids,
      logits_processor = None,
      stopping_criteria = None,
      max_length = None,
      pad_token_id = None,
      eos_token_id = None,
      output_attentions = None,
      output_hidden_states = None,
      output_scores = None,
      return_dict_in_generate = None,
      synced_gpus = False,
      streamer = None,
      speculative_generation_tree=None,
      **model_kwargs,
  ):

    # init values
    if max_length is not None:
      warnings.warn(
          "`max_length` is deprecated in this function, use"
          " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])`"
          " instead.",
          UserWarning,
      )
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = (
        () if (return_dict_in_generate and output_attentions) else None
    )
    cross_attentions = (
        () if (return_dict_in_generate and output_attentions) else None
    )
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and
    # hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
      encoder_attentions = (
          model_kwargs["encoder_outputs"].get("attentions")
          if output_attentions
          else None
      )
      encoder_hidden_states = (
          model_kwargs["encoder_outputs"].get("hidden_states")
          if output_hidden_states
          else None
      )

    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # forward pass to get next token
    outputs = self(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    next_token_logits = outputs.logits[:, :, :]

    # pre-process distribution
    next_tokens_scores = next_token_logits

    # Store scores, attentions and hidden_states when required
    if return_dict_in_generate:
      if output_scores:
        scores += (next_tokens_scores,)
      if output_attentions:
        decoder_attentions += (
            (outputs.decoder_attentions,)
            if self.config.is_encoder_decoder
            else (outputs.attentions,)
        )
        if self.config.is_encoder_decoder:
          cross_attentions += (outputs.cross_attentions,)

      if output_hidden_states:
        decoder_hidden_states += (
            (outputs.decoder_hidden_states,)
            if self.config.is_encoder_decoder
            else (outputs.hidden_states,)
        )

    if streamer is not None:
      streamer.end()

    if return_dict_in_generate:
      if self.config.is_encoder_decoder:
        return GreedySearchEncoderDecoderOutput(
            sequences=input_ids,
            scores=scores,
            encoder_attentions=encoder_attentions,  # pylint: disable=undefined-variable
            encoder_hidden_states=encoder_hidden_states,  # pylint: disable=undefined-variable
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            decoder_hidden_states=decoder_hidden_states,
        )
      else:
        return GreedySearchDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
        )
    else:
      return input_ids

  # this one is for interpretabiltiy, think that moving the encoding matrix too
  # much is what is leading to the slowdown
  # pylint: disable=too-complex
  def jam_FiD_interp_greedy_v2(
      self,
      input_ids,
      logits_processor = None,
      stopping_criteria = None,
      max_length = None,
      pad_token_id = None,
      eos_token_id = None,
      output_attentions = None,
      output_hidden_states = None,
      output_scores = None,
      return_dict_in_generate = None,
      synced_gpus = False,
      streamer = None,
      number_of_out_decodings=None,
      tokenizer=None,
      interpretability_mode=None,
      **model_kwargs,
  ):

    # init values
    logits_processor = (
        logits_processor
        if logits_processor is not None
        else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria
        if stopping_criteria is not None
        else StoppingCriteriaList()
    )
    if max_length is not None:
      warnings.warn(
          "`max_length` is deprecated in this function, use"
          " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])`"
          " instead.",
          UserWarning,
      )
      stopping_criteria = validate_stopping_criteria(
          stopping_criteria, max_length
      )
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
      eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )

    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = (
        () if (return_dict_in_generate and output_attentions) else None
    )
    cross_attentions = (
        () if (return_dict_in_generate and output_attentions) else None
    )
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and
    # hidden states
    # if return_dict_in_generate and self.config.is_encoder_decoder:
    #   encoder_attentions = (
    #       model_kwargs["encoder_outputs"].get("attentions")
    #       if output_attentions
    #       else None
    #   )
    #   encoder_hidden_states = (
    #       model_kwargs["encoder_outputs"].get("hidden_states")
    #       if output_hidden_states
    #       else None
    #   )

    # TODO(enounen) JAM: need to make sure the decoded sequences are size
    # BS x sp2 & yet we only encode one time (but still decode all multiple
    # times)

    DBS = interpretability_mode["DBS"]  #  pylint: disable=invalid-name
    D = interpretability_mode["D"]  #  pylint: disable=invalid-name
    prefix = 0
    suffix = 0
    mb = 2  # mini-block

    big_dict = {"size": D}  # needed for some existing code
    lil_dict = {}
    for d in range(D):
      lil_dict[d] = {}
      big_dict[d] = []
    list_of_perms = []

    prep_time_taken = 0.0

    i = 0
    sparsity2 = self.decoder.block[i].layer[1].EncDecAttention.sparsity_matrix
    x = 0
    x_pds = []

    sparsity2 = torch.ones((DBS, 1, D * mb), device=sparsity2.device)
    sparsity2[:, :, mb * prefix : sparsity2.shape[2] - mb * suffix] = (
        0  # initialize properly for first batch
    )

    for i in range(24):
      self.decoder.block[i].layer[1].EncDecAttention.sparsity_matrix = sparsity2

    if interpretability_mode["mode"] == "shapley":
      P = interpretability_mode["P"]  #  pylint: disable=invalid-name
      for p in range(P):
        perm = np.random.permutation(D)
        list_of_perms.append(perm)

        p_ds = []
        prev_p_d = -1
        for d in range(D):
          p_d = perm[d]
          p_ds.extend([p_d * mb + mmbb for mmbb in range(mb)])
          x_pds.append((prev_p_d, p_d))
          sparsity2[x, :, p_ds] = 1
          prev_p_d = p_d
          x += 1

          if x == DBS or (p == P - 1 and d == D - 1):
            if x < DBS:
              sparsity2[x:, :, :] = 1

            input_ids = torch.zeros(
                (DBS, 1), dtype=torch.long, device=input_ids.device
            )
            model_kwargs["past_key_values"] = None

            # keep track of which sequences are already finished
            unfinished_sequences = torch.ones(
                input_ids.shape[0], dtype=torch.long, device=input_ids.device
            )

            this_peer_finished = False  # used by synced_gpus only
            while True:
              prep_start_time = time.time()
              model_inputs = self.prepare_inputs_for_generation(
                  input_ids, **model_kwargs
              )
              prep_time_taken += time.time() - prep_start_time

              outputs = self(
                  **model_inputs,
                  return_dict=True,
                  output_attentions=output_attentions,
                  output_hidden_states=output_hidden_states,
              )

              if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

              next_token_logits = outputs.logits[:, -1, :]

              # pre-process distribution
              next_tokens_scores = logits_processor(
                  input_ids, next_token_logits
              )

              # Store scores, attentions and hidden_states when required
              if return_dict_in_generate:
                if output_scores:
                  scores += (next_tokens_scores,)
                if output_attentions:
                  decoder_attentions += (
                      (outputs.decoder_attentions,)
                      if self.config.is_encoder_decoder
                      else (outputs.attentions,)
                  )
                  if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                  decoder_hidden_states += (
                      (outputs.decoder_hidden_states,)
                      if self.config.is_encoder_decoder
                      else (outputs.hidden_states,)
                  )

              # argmax
              next_tokens = torch.argmax(next_tokens_scores, dim=-1)

              # finished sentences should have their next token be a padding
              # token
              if eos_token_id is not None:
                if pad_token_id is None:
                  raise ValueError(
                      "If `eos_token_id` is defined, make sure that"
                      " `pad_token_id` is defined."
                  )
                next_tokens = (
                    next_tokens * unfinished_sequences
                    + pad_token_id * (1 - unfinished_sequences)
                )

              # update generated ids, model inputs, and length for next step
              input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
              if streamer is not None:
                streamer.put(next_tokens.cpu())
              model_kwargs = self._update_model_kwargs_for_generation(
                  outputs,
                  model_kwargs,
                  is_encoder_decoder=self.config.is_encoder_decoder,
              )

              # if eos_token was found in one sentence, set sentence to finished
              if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                  this_peer_finished = True

              # stop if we exceed the maximum length
              if stopping_criteria(input_ids, scores):
                this_peer_finished = True

              if this_peer_finished and not synced_gpus:
                break

            tokens = input_ids
            xd = tokenizer.batch_decode(
                tokens, skip_special_tokens=True
            )  # dont necessarily want to pass the tokenizer
            for x in range(DBS):
              if x < len(x_pds):
                curr_text = xd[x]
                prev_p_d, p_d = x_pds[x]
                if prev_p_d == -1:
                  past_text = ""
                # need to ignore 'past_text' warning bc of DBS
                # being an odd multiple of the sampling (leading
                # to past_text being carried from last batch)
                big_dict[p_d].append((past_text, curr_text))  # pylint: disable=undefined-variable
                if past_text != curr_text:  # pylint: disable=undefined-variable
                  if curr_text in lil_dict[p_d]:
                    lil_dict[p_d][curr_text] += 1
                  else:
                    lil_dict[p_d][curr_text] = 1
                past_text = curr_text
            # reset here at the end
            x = 0
            x_pds = []
            sparsity2[:, :, mb * prefix : sparsity2.shape[2] - mb * suffix] = 0
    elif interpretability_mode["mode"] == "banzhaf":
      S = interpretability_mode["S"]  #  pylint: disable=invalid-name
      for s in range(S):
        conf = np.random.rand(D) > 0.5
        list_of_perms.append(conf)

        s_ds = []
        for d in range(D):
          if conf[d]:
            s_ds.extend([d * mb + mmbb for mmbb in range(mb)])
        prev_p_d = -1
        for d in range(-1, D):
          s_d = -1
          if d >= 0:
            s_d = conf[d]
          x_pds.append((d, s_d))
          sparsity2[x, :, s_ds] = 1
          if d >= 0:
            sparsity2[x, :, d * mb : d * mb + mb] = 1 - int(s_d)
          x += 1
          if x == DBS or (s == S - 1 and d == D - 1):
            if x < DBS:
              # cant just slice because I only have pointer; cant
              # set to zero, else default behavior pads to maxlen
              sparsity2[x:, :, :] = 1

            input_ids = torch.zeros(
                (DBS, 1), dtype=torch.long, device=input_ids.device
            )
            model_kwargs["past_key_values"] = None

            unfinished_sequences = torch.ones(
                input_ids.shape[0], dtype=torch.long, device=input_ids.device
            )

            this_peer_finished = False  # used by synced_gpus only
            while True:
              prep_start_time = time.time()
              model_inputs = self.prepare_inputs_for_generation(
                  input_ids, **model_kwargs
              )
              prep_time_taken += time.time() - prep_start_time

              # forward pass to get next token
              outputs = self(
                  **model_inputs,
                  return_dict=True,
                  output_attentions=output_attentions,
                  output_hidden_states=output_hidden_states,
              )

              if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

              next_token_logits = outputs.logits[:, -1, :]

              # pre-process distribution
              next_tokens_scores = logits_processor(
                  input_ids, next_token_logits
              )

              # Store scores, attentions and hidden_states when required
              if return_dict_in_generate:
                if output_scores:
                  scores += (next_tokens_scores,)
                if output_attentions:
                  decoder_attentions += (
                      (outputs.decoder_attentions,)
                      if self.config.is_encoder_decoder
                      else (outputs.attentions,)
                  )
                  if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                  decoder_hidden_states += (
                      (outputs.decoder_hidden_states,)
                      if self.config.is_encoder_decoder
                      else (outputs.hidden_states,)
                  )

              # argmax
              next_tokens = torch.argmax(next_tokens_scores, dim=-1)

              # finished sentences should have their next token be a padding
              # token
              if eos_token_id is not None:
                if pad_token_id is None:
                  raise ValueError(
                      "If `eos_token_id` is defined, make sure that"
                      " `pad_token_id` is defined."
                  )
                next_tokens = (
                    next_tokens * unfinished_sequences
                    + pad_token_id * (1 - unfinished_sequences)
                )

              # update generated ids, model inputs, and length for next step
              input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
              if streamer is not None:
                streamer.put(next_tokens.cpu())
              model_kwargs = self._update_model_kwargs_for_generation(
                  outputs,
                  model_kwargs,
                  is_encoder_decoder=self.config.is_encoder_decoder,
              )

              # if eos_token was found in one sentence, set sentence to finished
              if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                  this_peer_finished = True

              # stop if we exceed the maximum length
              if stopping_criteria(input_ids, scores):
                this_peer_finished = True

              if this_peer_finished and not synced_gpus:
                break

            tokens = input_ids
            xd = tokenizer.batch_decode(
                tokens, skip_special_tokens=True
            )  # dont necessarily want to pass the tokenizer
            for x in range(DBS):
              if x < len(x_pds):
                curr_text = xd[x]
                d, s_d = x_pds[x]
                # pylint: disable=undefined-variable,
                # past_text always defined
                # cant reinit here because need to keep the same across batches
                if d == -1:
                  past_text = curr_text
                if d >= 0:
                  if not s_d:
                    big_dict[d].append((past_text, curr_text))
                  else:
                    big_dict[d].append((curr_text, past_text))
                if past_text != curr_text and d >= 0:
                  if s_d:
                    # need to update baseline instead of perturb (because
                    # baseline subset already included the current
                    # dimension/feature being considered)
                    curr_text = past_text

                  if curr_text in lil_dict[d]:
                    lil_dict[d][curr_text] += 1
                  else:
                    lil_dict[d][curr_text] = 1

            # reset
            x = 0
            x_pds = []
            sparsity2[:, :, mb * prefix : sparsity2.shape[2] - mb * suffix] = 0

    elif interpretability_mode["mode"] == "banzhaf10":
      S = interpretability_mode["S"]  #  pylint: disable=invalid-name
      prob = interpretability_mode["prob"]
      for s in range(S):
        conf = np.random.rand(D) > (1 - prob)
        list_of_perms.append(conf)

        s_ds = []
        for d in range(D):
          if conf[d]:
            s_ds.extend([d * mb + mmbb for mmbb in range(mb)])
        prev_p_d = -1
        for d in range(-1, D):
          s_d = -1
          if d >= 0:
            s_d = conf[d]
          x_pds.append((d, s_d))
          sparsity2[x, :, s_ds] = 1
          if d >= 0:
            sparsity2[x, :, d * mb : d * mb + mb] = 1 - int(s_d)
          x += 1
          pass
          if x == DBS or (s == S - 1 and d == D - 1):
            if x < DBS:
              # maybe this will not do <pad><pad><pad> (cant just slice
              # because I only have pointer)
              sparsity2[x:, :, :] = 1

            input_ids = torch.zeros(
                (DBS, 1), dtype=torch.long, device=input_ids.device
            )
            model_kwargs["past_key_values"] = None

            unfinished_sequences = torch.ones(
                input_ids.shape[0], dtype=torch.long, device=input_ids.device
            )

            this_peer_finished = False  # used by synced_gpus only
            while True:
              prep_start_time = time.time()
              model_inputs = self.prepare_inputs_for_generation(
                  input_ids, **model_kwargs
              )
              prep_time_taken += time.time() - prep_start_time

              outputs = self(
                  **model_inputs,
                  return_dict=True,
                  output_attentions=output_attentions,
                  output_hidden_states=output_hidden_states,
              )

              next_token_logits = outputs.logits[:, -1, :]

              # pre-process distribution
              next_tokens_scores = logits_processor(
                  input_ids, next_token_logits
              )

              # Store scores, attentions and hidden_states when required
              if return_dict_in_generate:
                if output_scores:
                  scores += (next_tokens_scores,)
                if output_attentions:
                  decoder_attentions += (
                      (outputs.decoder_attentions,)
                      if self.config.is_encoder_decoder
                      else (outputs.attentions,)
                  )
                  if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                  decoder_hidden_states += (
                      (outputs.decoder_hidden_states,)
                      if self.config.is_encoder_decoder
                      else (outputs.hidden_states,)
                  )

              # argmax
              next_tokens = torch.argmax(next_tokens_scores, dim=-1)

              # finished sentences should have their next token be a padding
              # token
              if eos_token_id is not None:
                if pad_token_id is None:
                  raise ValueError(
                      "If `eos_token_id` is defined, make sure that"
                      " `pad_token_id` is defined."
                  )
                next_tokens = (
                    next_tokens * unfinished_sequences
                    + pad_token_id * (1 - unfinished_sequences)
                )

              # update generated ids, model inputs, and length for next step
              input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
              if streamer is not None:
                streamer.put(next_tokens.cpu())
              model_kwargs = self._update_model_kwargs_for_generation(
                  outputs,
                  model_kwargs,
                  is_encoder_decoder=self.config.is_encoder_decoder,
              )

              # if eos_token was found in one sentence, set sentence to finished
              if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                  this_peer_finished = True

              # stop if we exceed the maximum length
              if stopping_criteria(input_ids, scores):
                this_peer_finished = True

              if this_peer_finished and not synced_gpus:
                break

            tokens = input_ids
            xd = tokenizer.batch_decode(
                tokens, skip_special_tokens=True
            )  # dont necessarily want to pass the tokenizer

            for x in range(DBS):
              if x < len(x_pds):
                curr_text = xd[x]
                d, s_d = x_pds[x]
                if d == -1:
                  past_text = curr_text
                if d >= 0:
                  if not s_d:
                    big_dict[d].append((past_text, curr_text))
                  else:
                    big_dict[d].append((curr_text, past_text))
                if past_text != curr_text and d >= 0:
                  if s_d:
                    # need to update baseline instead of perturb (because
                    # baseline subset already included the current
                    # dimension/feature being considered)
                    curr_text = past_text

                  if curr_text in lil_dict[d]:
                    lil_dict[d][curr_text] += 1
                  else:
                    lil_dict[d][curr_text] = 1

            # reset here
            x = 0
            x_pds = []
            sparsity2[:, :, mb * prefix : sparsity2.shape[2] - mb * suffix] = 0

    if streamer is not None:
      streamer.end()

    return lil_dict, big_dict, list_of_perms
