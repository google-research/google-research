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

"""Forked from transformers/src/utils.py."""

import copy
import inspect
import warnings
from torch import Tensor
from torch.nn import functional as F
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import math

from ..cache_utils import (
    Cache,
    DynamicCache,
    HQQQuantizedCache,
    HybridCache,
    QuantizedCacheConfig,
    QuantoQuantizedCache,
    SlidingWindowCache,
    StaticCache,
)
from ..integrations.deepspeed import is_deepspeed_zero3_enabled
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from ..models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from ..tokenization_utils import ExtensionsTrie
from ..utils import (
    ModelOutput,
    is_accelerate_available,
    is_hqq_available,
    is_quanto_available,
    is_torchdynamo_compiling,
    logging,
)
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
    AssistedCandidateGenerator,
    CandidateGenerator,
    PromptLookupCandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from .configuration_utils import GenerationConfig, GenerationMode
from .logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
    WatermarkLogitsProcessor,
)
from .stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..tokenization_utils_base import PreTrainedTokenizerBase
    from .streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

NEED_SETUP_CACHE_CLASSES_MAPPING = {"static": StaticCache, "sliding_window": SlidingWindowCache, "hybrid": HybridCache}
QUANT_BACKEND_CLASSES_MAPPING = {"quanto": QuantoQuantizedCache, "HQQ": HQQQuantizedCache}


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    correction_rate: float = None

@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateBeamEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


# Equivalent classes (kept for retrocompatibility purposes)
GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput

ContrastiveSearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput

BeamSearchDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
BeamSampleDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput

BeamSearchEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
BeamSampleEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]

# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]


class GenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in [`PreTrainedModel`].

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *contrastive search* if `penalty_alpha>0` and `top_k>1`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
        - *diverse beam-search decoding* if `num_beams>1` and `num_beam_groups>1`
        - *constrained beam-search decoding* if `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`."
        )

    def _prepare_model_inputs(
        self,
        inputs = None,
        bos_token_id = None,
        model_kwargs = None,
    ):
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. "
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs = None,
        bos_token_id = None,
        model_kwargs = None,
    ):
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs.last_hidden_state.size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break

        if "inputs_embeds" in model_kwargs:
            return torch.ones((batch_size, 0), dtype=torch.long, device=self.device)

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_attention_mask_for_generation(
        self,
        inputs,
        pad_token_id,
        eos_token_id,
    ):
        # No information for attention mask inference -> return default attention mask
        default_attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
        if pad_token_id is None:
            return default_attention_mask

        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        if not is_input_ids:
            return default_attention_mask

        # Otherwise we have may have information -> try to infer the attention mask
        if inputs.device.type == "mps":
            # mps does not support torch.isin (https://github.com/pytorch/pytorch/issues/77764)
            raise ValueError(
                "Can't infer missing attention mask on `mps` device. Please provide an `attention_mask` or use a different device."
            )

        is_pad_token_in_inputs = (pad_token_id is not None) and (
            torch.isin(elements=inputs, test_elements=pad_token_id).any()
        )
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or ~(
            torch.isin(elements=eos_token_id, test_elements=pad_token_id).any()
        )
        can_infer_attention_mask = is_pad_token_in_inputs * is_pad_token_not_equal_to_eos_token_id
        attention_mask_from_padding = inputs.ne(pad_token_id).long()

        attention_mask = (
            attention_mask_from_padding * can_infer_attention_mask + default_attention_mask * ~can_infer_attention_mask
        )
        return attention_mask

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor,
        model_kwargs,
        model_input_name,
        generation_config,
    ):
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs and generation config.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size,
        model_input_name,
        model_kwargs,
        decoder_start_token_id,
        device = None,
    ):
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. `decoder_start_token_id` must have shape (batch_size, 1)
        if device is None:
            device = self.device
        if decoder_start_token_id.ndim == 1:
            if decoder_start_token_id.shape[0] != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expected to have length {batch_size} but got {decoder_start_token_id.shape[0]}"
                )
            decoder_start_token_id = decoder_start_token_id.view(-1, 1)
        else:
            decoder_start_token_id = (
                torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )

        # 3. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_start_token_id
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token. Note that the
        # original checkpoints can't be detected through `self.__class__.__name__.lower()`, needing custom logic.
        # See: https://github.com/huggingface/transformers/pull/31470
        elif "donut" in self.__class__.__name__.lower() or (
            self.config.model_type == "vision-encoder-decoder" and "donut" in self.config.encoder.model_type.lower()
        ):
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item():
            decoder_input_ids = torch.cat([decoder_start_token_id, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size = 1,
        is_encoder_decoder = False,
        input_ids = None,
        **model_kwargs,
    ):
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _extract_past_from_model_output(self, outputs, standardize_cache_format = False):
        past_key_values = None
        cache_name = "past_key_values"
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        elif "cache_params" in outputs:
            past_key_values = outputs.cache_params
            cache_name = "cache_params"

        # Bloom fix: standardizes the cache format when requested
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return cache_name, past_key_values

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder = False,
        standardize_cache_format = False,
        num_new_tokens = 1,
    ):
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError(
            f"Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to"
            f" enable beam search for {self.__class__}"
        )

    def _get_candidate_generator(
        self,
        generation_config,
        input_ids,
        inputs_tensor,
        assistant_model,
        logits_processor,
        model_kwargs,
    ):
        """
        Returns the candidate generator to be used in `assisted_generation`
        """
        if generation_config.prompt_lookup_num_tokens is not None:
            candidate_generator = PromptLookupCandidateGenerator(
                num_output_tokens=generation_config.prompt_lookup_num_tokens,
                max_matching_ngram_size=generation_config.max_matching_ngram_size,
                max_length=generation_config.max_length,
            )
        else:
            candidate_generator = AssistedCandidateGenerator(
                input_ids=input_ids,
                assistant_model=assistant_model,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
                logits_processor=logits_processor,
            )
        return candidate_generator

    def _get_logits_warper(
        self,
        generation_config,
        device,
    ):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config.eos_token_id)) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config.eos_token_id, list):
                min_tokens_to_keep = len(generation_config.eos_token_id) + 1
            elif isinstance(generation_config.eos_token_id, torch.Tensor):
                min_tokens_to_keep = generation_config.eos_token_id.shape[0] + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.min_p is not None:
            # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
            warpers.append(MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            warpers.append(
                TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            warpers.append(
                EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            warpers.append(
                EtaLogitsWarper(
                    epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep, device=device
                )
            )
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

    def _get_logits_processor(
        self,
        generation_config,
        input_ids_seq_length,
        encoder_input_ids,
        prefix_allowed_tokens_fn,
        logits_processor,
        device = None,
        model_kwargs = None,
        negative_prompt_ids = None,
        negative_prompt_attention_mask = None,
    ):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
            processors.append(
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    generation_config.guidance_scale,
                    self,
                    unconditional_ids=negative_prompt_ids,
                    unconditional_attention_mask=negative_prompt_attention_mask,
                    use_cache=model_kwargs["use_cache"],
                )
            )
        if generation_config.sequence_bias is not None:
            processors.append(SequenceBiasLogitsProcessor(sequence_bias=generation_config.sequence_bias))

        if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=generation_config.diversity_penalty,
                    num_beams=generation_config.num_beams,
                    num_beam_groups=generation_config.num_beam_groups,
                )
            )
        if (
            generation_config.encoder_repetition_penalty is not None
            and generation_config.encoder_repetition_penalty != 1.0
        ):
            processors.append(
                EncoderRepetitionPenaltyLogitsProcessor(
                    penalty=generation_config.encoder_repetition_penalty,
                    encoder_input_ids=encoder_input_ids,
                )
            )
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        if (
            generation_config.encoder_no_repeat_ngram_size is not None
            and generation_config.encoder_no_repeat_ngram_size > 0
        ):
            processors.append(
                EncoderNoRepeatNGramLogitsProcessor(
                    generation_config.encoder_no_repeat_ngram_size,
                    encoder_input_ids,
                )
            )
        if generation_config.bad_words_ids is not None:
            processors.append(
                NoBadWordsLogitsProcessor(
                    generation_config.bad_words_ids,
                    generation_config.eos_token_id,
                )
            )
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
        ):
            processors.append(
                MinLengthLogitsProcessor(
                    generation_config.min_length,
                    generation_config.eos_token_id,
                    device=device,
                )
            )
        if (
            generation_config.min_new_tokens is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_new_tokens > 0
        ):
            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length,
                    generation_config.min_new_tokens,
                    generation_config.eos_token_id,
                    device=device,
                )
            )
        if prefix_allowed_tokens_fn is not None:
            processors.append(
                PrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn,
                    generation_config.num_beams // generation_config.num_beam_groups,
                )
            )
        if generation_config.forced_bos_token_id is not None:
            processors.append(
                ForcedBOSTokenLogitsProcessor(
                    generation_config.forced_bos_token_id,
                )
            )
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(
                    generation_config.max_length,
                    generation_config.forced_eos_token_id,
                    device=device,
                )
            )
        if generation_config.remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if generation_config.exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(
                    generation_config.exponential_decay_length_penalty,
                    generation_config.eos_token_id,
                    input_ids_seq_length,
                )
            )
        if generation_config.suppress_tokens is not None:
            processors.append(
                SuppressTokensLogitsProcessor(
                    generation_config.suppress_tokens,
                    device=device,
                )
            )
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None:
                # generation starts after the last token that is forced
                begin_index += generation_config.forced_decoder_ids[-1][0]
            processors.append(
                SuppressTokensAtBeginLogitsProcessor(
                    generation_config.begin_suppress_tokens,
                    begin_index,
                    device=device,
                )
            )
        if generation_config.forced_decoder_ids is not None:
            # TODO(Sanchit): deprecate in v4.40 by removing this logic
            warnings.warn(
                "You have explicitly specified `forced_decoder_ids`. This functionality has been deprecated and will throw an error in v4.40. Please remove the `forced_decoder_ids` argument in favour of `input_ids` or `decoder_input_ids` respectively.",
                FutureWarning,
            )
            processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids, _has_warned=True))
        if generation_config.watermarking_config is not None:
            processors.append(
                WatermarkLogitsProcessor(
                    vocab_size=self.config.vocab_size,
                    device=device,
                    greenlist_ratio=generation_config.watermarking_config.greenlist_ratio,
                    bias=generation_config.watermarking_config.bias,
                    hashing_key=generation_config.watermarking_config.hashing_key,
                    seeding_scheme=generation_config.watermarking_config.seeding_scheme,
                    context_width=generation_config.watermarking_config.context_width,
                )
            )
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

    def _get_stopping_criteria(
        self,
        generation_config,
        stopping_criteria,
        tokenizer = None,
        **kwargs,
    ):
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        if generation_config.stop_strings is not None:
            if tokenizer is None:
                raise ValueError(
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
            criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))
        if generation_config.eos_token_id is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _merge_criteria_processor_list(
        self,
        default_list,
        custom_list,
    ):
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `.generate()` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def compute_transition_scores(
        self,
        sequences,
        scores,
        beam_indices = None,
        normalize_logits = False,
    ):
        """
        Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was
        used). This is a convenient method to quicky obtain the scores of the selected tokens at generation time.

        Parameters:
            sequences (`torch.LongTensor`):
                The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or
                shorter if all batches finished early due to the `eos_token_id`.
            scores (`tuple(torch.FloatTensor)`):
                Transition scores for each vocabulary token at each generation step. Beam transition scores consisting
                of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
                Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
                with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
            beam_indices (`torch.LongTensor`, *optional*):
                Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
                `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
                generate-time.
            normalize_logits (`bool`, *optional*, defaults to `False`):
                Whether to normalize the logits (which, for legacy reasons, may be unnormalized).

        Return:
            `torch.Tensor`: A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)` containing
                the transition scores (logits)

        Examples:

        ```python
        >>> from transformers import GPT2Tokenizer, AutoModelForCausalLM
        >>> import numpy as np

        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer.pad_token_id = tokenizer.eos_token_id
        >>> inputs = tokenizer(["Today is"], return_tensors="pt")

        >>> # Example 1: Print the scores for each token generated with Greedy Search
        >>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
        >>> transition_scores = model.compute_transition_scores(
        ...     outputs.sequences, outputs.scores, normalize_logits=True
        ... )
        >>> # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
        >>> # encoder-decoder models, like BART or T5.
        >>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        >>> generated_tokens = outputs.sequences[:, input_length:]
        >>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
        ...     # | token | token string | log probability | probability
        ...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
        |   262 |  the     | -1.414 | 24.33%
        |  1110 |  day     | -2.609 | 7.36%
        |   618 |  when    | -2.010 | 13.40%
        |   356 |  we      | -1.859 | 15.58%
        |   460 |  can     | -2.508 | 8.14%

        >>> # Example 2: Reconstruct the sequence scores from Beam Search
        >>> outputs = model.generate(
        ...     **inputs,
        ...     max_new_tokens=5,
        ...     num_beams=4,
        ...     num_return_sequences=4,
        ...     return_dict_in_generate=True,
        ...     output_scores=True,
        ... )
        >>> transition_scores = model.compute_transition_scores(
        ...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        ... )
        >>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
        >>> # Tip 1: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
        >>> # use case, you might want to recompute it with `normalize_logits=True`.
        >>> # Tip 2: the output length does NOT include the input length
        >>> output_length = np.sum(transition_scores.numpy() < 0, axis=1)
        >>> length_penalty = model.generation_config.length_penalty
        >>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
        >>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
        True
        ```"""
        # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected
        if beam_indices is None:
            beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
            beam_indices = beam_indices.expand(-1, len(scores))

        # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
        # seq_len - input_length
        scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

        # 3. Optionally normalize the logits (across the vocab dimension)
        if normalize_logits:
            scores = scores.reshape(-1, self.config.vocab_size, scores.shape[-1])
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            scores = scores.reshape(-1, scores.shape[-1])

        # 4. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
        beam_indices[beam_indices_mask] = 0

        # 6. multiply beam_indices with vocab size to gather correctly from scores
        beam_sequence_indices = beam_indices * self.config.vocab_size

        # 7. Define which indices contributed to scores
        cut_idx = sequences.shape[-1] - max_beam_length
        indices = sequences[:, cut_idx:] + beam_sequence_indices

        # 8. Compute scores
        transition_scores = scores.gather(0, indices)

        # 9. Mask out transition_scores of beams that stopped early
        transition_scores[beam_indices_mask] = 0

        return transition_scores

    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        if not self.can_generate():
            generate_compatible_mappings = [
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
                MODEL_FOR_VISION_2_SEQ_MAPPING,
                MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            generate_compatible_classes = set()
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            raise TypeError(exception_message)

    def _validate_assistant(self, assistant_model):
        if assistant_model is None:
            return

        if self.config.is_encoder_decoder and not assistant_model.config.is_encoder_decoder:
            attributes_to_check = ["encoder_attention_heads", "encoder_ffn_dim", "encoder_layers"]
            attributes_to_check = [attr for attr in dir(assistant_model.config) if attr in attributes_to_check]
            are_equal = all(
                getattr(self.config, attr) == getattr(assistant_model.config, attr) for attr in attributes_to_check
            )
            if not are_equal:
                raise ValueError(
                    "The main model and the assistant don't have compatible encoder-dependent input shapes. "
                    "Ensure you load the assistant with the correct encoder-decoder class, e.g. `AutoModelForSpeechSeq2Seq` for Whisper."
                )

        if not self.config.vocab_size == assistant_model.config.vocab_size:
            raise ValueError("Make sure the main and assistant model use the same tokenizer")

    def _validate_model_kwargs(self, model_kwargs):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # If a `Cache` instance is passed, checks whether the model is compatible with it
        if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
            raise ValueError(
                f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
                "check the model documentation for supported cache formats."
            )

        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)

        # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
        if self.config.is_encoder_decoder:
            base_model = getattr(self, self.base_model_prefix, None)

            # allow encoder kwargs
            encoder = getattr(self, "encoder", None)
            # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
            # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
            # TODO: A better way to handle this.
            if encoder is None and base_model is not None:
                encoder = getattr(base_model, "encoder", None)

            if encoder is not None:
                encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                model_args |= encoder_model_args

            # allow decoder kwargs
            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

            # allow assistant_encoder_outputs to be passed if we're doing assisted generating
            if "assistant_encoder_outputs" in model_kwargs:
                model_args |= {"assistant_encoder_outputs"}

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

        # 2. Min length warnings due to unfeasible parameter combinations
        min_length_error_suffix = (
            " Generation will stop at the defined maximum length. You should decrease the minimum length and/or "
            "increase the maximum length."
        )
        if has_default_max_length:
            min_length_error_suffix += (
                f" Note that `max_length` is set to {generation_config.max_length}, its default value."
            )
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            warnings.warn(
                f"Unfeasible length constraints: `min_length` ({generation_config.min_length}) is larger than"
                f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                UserWarning,
            )
        if generation_config.min_new_tokens is not None:
            min_length = generation_config.min_new_tokens + input_ids_length
            if min_length > generation_config.max_length:
                warnings.warn(
                    f"Unfeasible length constraints: `min_new_tokens` ({generation_config.min_new_tokens}), when "
                    f"added to the prompt length ({input_ids_length}), is larger than"
                    f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                    UserWarning,
                )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
    ):
        """Prepared max and min length in generaion configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        # if both `inputs_embeds` and `input_ids` are passed, we do not correct the length
        # otherwise we need total length [inputs-embeds-len + new-tokens-len] to not go beyond indicated `max_length``
        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.max_length -= inputs_tensor.shape[1]

        # same for min length
        if generation_config.min_new_tokens is not None:
            if not has_default_min_length:
                logger.warning(
                    f"Both `min_new_tokens` (={generation_config.min_new_tokens}) and `min_length`(="
                    f"{generation_config.min_length}) seem to have been set. `min_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.min_length = generation_config.min_new_tokens + input_ids_length

        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)

        return generation_config

    def _prepare_generation_config(
        self, generation_config, **kwargs
    ):
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs.
        """
        # TODO joao: when we can detect `fullgraph=True` in `torch.compile` (https://github.com/pytorch/pytorch/pull/120400)
        # replace `is_torchdynamo_compiling` by the corresponding check. As it is, we are being too restrictive with
        # the parameterization in `fullgraph=False` so as to enable `fullgraph=True`.

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # three conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same);
            # 3) the user must have set generation parameters in the model config.
            # NOTE: `torch.compile` can't compile `hash`, this legacy support is disabled with compilation.
            if (
                not is_torchdynamo_compiling()
                and self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and self.config._has_non_default_generation_parameters()
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled.
        if is_torchdynamo_compiling():
            model_kwargs = kwargs
            generate_attributes_in_kwargs = [
                key for key, value in kwargs.items() if getattr(generation_config, key, None) != value
            ]
            if len(generate_attributes_in_kwargs) > 0:
                raise ValueError(
                    "`torch.compile` exception: all generation configuration attributes must be passed within a "
                    f"`generation_config` instance passed to `generate` (found: {generate_attributes_in_kwargs})."
                )
        else:
            generation_config = copy.deepcopy(generation_config)
            model_kwargs = generation_config.update(**kwargs)

        return generation_config, model_kwargs

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        if not model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = None
            return model_kwargs

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        else:
            cur_len = input_ids.shape[-1]
        model_kwargs["cache_position"] = torch.arange(past_length, cur_len, device=input_ids.device)
        return model_kwargs

    def _get_cache(self, cache_implementation, max_batch_size, max_cache_len):
        """
        Sets a cache for `generate`, that will persist across calls. A new cache will only be initialized a
        new `generate` call requires a larger cache.

        Returns the resulting cache object.
        """
        cache_cls: Cache = NEED_SETUP_CACHE_CLASSES_MAPPING[cache_implementation]
        if cache_implementation == "sliding_window":
            max_cache_len = min(self.config.sliding_window, max_cache_len)

        need_new_cache = (
            not hasattr(self, "_cache")
            or (not isinstance(self._cache, cache_cls))
            or self._cache.max_batch_size != max_batch_size
            or self._cache.max_cache_len < max_cache_len
        )

        if need_new_cache:
            if hasattr(self.config, "_pre_quantization_dtype"):
                cache_dtype = self.config._pre_quantization_dtype
            else:
                cache_dtype = self.dtype
            self._cache = cache_cls(
                config=self.config,
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                device=self.device,
                dtype=cache_dtype,
            )
        else:
            self._cache.reset()
        return self._cache

    def _supports_default_dynamic_cache(self):
        """
        Return `True` if current model can use a `DynamicCache` instance when initializing the `past_key_values`.
        This is mostly the same as `_supports_cache_class` attribute, but add exception for `Jamba` model which
        uses its own `HybridMambaAttentionDynamicCache` and do not need to initialize the Cache in advance in
        order to save memory (because no back and forth `to_legacy_cache` and `from_legacy_cache` will be performed
        for `HybridMambaAttentionDynamicCache`).
        """
        return self._supports_cache_class and "jamba" not in self.__class__.__name__.lower()

    def _prepare_special_tokens(
        self,
        generation_config,
        kwargs_has_attention_mask = None,
        device = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors (if they exist either in kwargs or in self.config)
        def _tensor_or_none(token_kwargs, token_self, device=None):
            if device is None:
                device = self.device

            token = token_kwargs if token_kwargs is not None else token_self
            if token is None or isinstance(token, torch.Tensor):
                return token
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_id = _tensor_or_none(
            generation_config.bos_token_id, self.generation_config.bos_token_id, device=device
        )
        eos_token_id = _tensor_or_none(
            generation_config.eos_token_id, self.generation_config.eos_token_id, device=device
        )
        pad_token_id = _tensor_or_none(
            generation_config.pad_token_id, self.generation_config.pad_token_id, device=device
        )
        decoder_start_token_id = _tensor_or_none(
            generation_config.decoder_start_token_id, self.generation_config.decoder_start_token_id, device=device
        )

        # for BC we also try to get `decoder_start_token_id` or `bos_token_id` (#30892)
        if self.config.is_encoder_decoder:
            decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else bos_token_id

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_id is not None and eos_token_id.ndim == 0:
            eos_token_id = eos_token_id.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_id is None and eos_token_id is not None:
            if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            pad_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_id} for open-end generation.")

        # we can't infer attn mask if pad token is set to be eos token in model's generation config
        if eos_token_id is not None and torch.isin(elements=eos_token_id, test_elements=pad_token_id).any():
            if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                logger.warning_once(
                    "The attention mask is not set and cannot be inferred from input because pad token is same as eos token."
                    "As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` "
                    "to obtain reliable results."
                )

        # Sanity checks/warnings
        if self.config.is_encoder_decoder and decoder_start_token_id is None:
            raise ValueError(
                "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
            )
        if eos_token_id is not None and (torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any()):
            logger.warning(
                f"`eos_token_id` should consist of positive integers, but is {eos_token_id}. Your generation will not "
                "stop until the maximum length is reached. Depending on other flags, it may even crash."
            )

        # Update generation config with the updated special tokens tensors
        generation_config.bos_token_id = bos_token_id
        generation_config.eos_token_id = eos_token_id
        generation_config.pad_token_id = pad_token_id
        generation_config.decoder_start_token_id = decoder_start_token_id

    @torch.no_grad()
    def generate(
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
        teacher_k = 0,
        teacher_p = 0,
        teacher_k_lower = 0,
        expected_seq_len = 0,
        **kwargs,
    ):
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        use_dynamic_cache_by_default = False
        if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs["past_key_values"] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    generation_config.max_length,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                if cache_config.backend == "quanto" and not is_quanto_available():
                    raise ImportError(
                        "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                        "Please install it via  with `pip install quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )

                model_kwargs["past_key_values"] = cache_class(cache_config)
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get("past_key_values", None)
            if past is None:
                model_kwargs["past_key_values"] = DynamicCache()
                use_dynamic_cache_by_default = True
            elif isinstance(past, tuple):
                model_kwargs["past_key_values"] = DynamicCache.from_legacy_cache(past)
                use_dynamic_cache_by_default = True

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # 10. go into different generation modes
        # print(generation_mode)
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            # if batch_size > 1:
            #     raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation == "static":
                raise ValueError("assisted generate is not supported with `static_cache`")
            if self._is_stateful:
                # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                )

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
            )

            # 12. prepare logits warper (if `do_sample` is `True`)
            prepared_logits_warper = (
                self._get_logits_warper(
                    generation_config,
                    device=input_ids.device,
                )
                if generation_config.do_sample
                else None
            )

            # 13. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                teacher_k=teacher_k,
                teacher_p=teacher_p,
                expected_seq_len=expected_seq_len,
                teacher_k_lower=teacher_k_lower,
                tokenizer=tokenizer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(
                    f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
                )

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache if needed
        if use_dynamic_cache_by_default and generation_config.return_legacy_cache:
            if isinstance(result, ModelOutput) and hasattr(result, "past_key_values"):
                if isinstance(result.past_key_values, DynamicCache):
                    result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def _has_unfinished_sequences(self, this_peer_finished, synced_gpus, device):
        """
        Returns whether there are still unfinished sequences in the device. The existence of unfinished sequences is
        fed through `this_peer_finished`. ZeRO stage 3-friendly.
        """
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                return False
        elif this_peer_finished:
            return False
        return True

    def heal_tokens(
        self, input_ids, tokenizer = None
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head.
        Parameters:
            input_ids (`torch.LongTensor`): The sequence used as a prompt for the generation.
            tokenizer (`PreTrainedTokenizerBase`, *optional*): The tokenizer used to decode the input ids.
        Return:
            `torch.LongTensor` where each sequence has its tail token replaced with its appropriate extension.
        """
        if tokenizer is None:
            raise ValueError(
                " When generating with token healing, you must pass the model's tokenizer to the `tokenizer` "
                "argument of `generate`."
            )

        bos_token_id, pad_token_id = tokenizer.bos_token_id, tokenizer.pad_token_id
        vocab_trie = ExtensionsTrie(tokenizer.get_vocab())
        generation_config = GenerationConfig(max_new_tokens=1, pad_token_id=pad_token_id)

        # assumption: leading/trailing whitespace is not meaningful, so the prompts are
        # stripped before re-tokenizing to desensitize generation to whitespace artefacts
        prompts = [p.strip() for p in tokenizer.batch_decode(input_ids, skip_special_tokens=True)]
        input_ids = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).input_ids.to(input_ids.device)

        # replace bos with pad to not condition healing on it
        input_ids = torch.where(input_ids == bos_token_id, pad_token_id, input_ids)

        tail_ids = input_ids[:, -1].tolist()
        space_tok = tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(" "))[0]
        # tail tokens are used for a prefix search, thus, whitespaces are replaced with
        # their tokenization (e.g. 'Ġ') to enable search for tokens prefixed with a whitespace
        tail_toks = (tokenizer.decode(t).replace(" ", space_tok) for t in tail_ids)

        for batch_idx, (tail_id, tail_tok) in enumerate(zip(tail_ids, tail_toks)):
            batch_ids = input_ids[batch_idx]
            if torch.all(batch_ids == pad_token_id).item():
                continue  # skip empty sequences (all pad ids)

            # apply bias for alternatives (extensions) to the tail token
            seq_bias = {(alt_tok,): 10.0 for alt_tok in vocab_trie.values(prefix=tail_tok)}
            if len(seq_bias) == 1:
                continue  # skip if there are no token alternatives to heal with

            # slightly favor original token to limit aggressive healing e.g. 'http' -> 'https'
            seq_bias[(tail_id,)] += 1.0
            generation_config.update(sequence_bias=seq_bias)

            trimmed_ids = batch_ids[:-1]
            # if the prompt is a single (non-pad) token, regenerate from bos
            if len(batch_ids[batch_ids != pad_token_id]) == 1:
                trimmed_ids[-1] = bos_token_id

            input_ids[batch_idx] = self.generate(trimmed_ids.unsqueeze(0), generation_config=generation_config)

        return input_ids

    def contrastive_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `contrastive_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._contrastive_search(*args, **kwargs)

    @torch.no_grad()
    def _contrastive_search(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **contrastive search** and can
        be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        top_k = generation_config.top_k
        penalty_alpha = generation_config.penalty_alpha
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        this_peer_finished = False

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # if the first step in the loop, encode all the prefix and obtain: (1) past_key_values;
            # (2) last_hidden_states; (3) logit_for_next_step; (4) update model kwargs for the next step
            if model_kwargs.get("past_key_values") is None or (
                isinstance(model_kwargs["past_key_values"], Cache)
                and model_kwargs["past_key_values"].get_seq_length() == 0
            ):
                # prepare inputs
                model_kwargs["use_cache"] = True
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # encode the given prefix and prepare model inputs; encoder-decoder model process the prefix and save
                # the `encoder_outputs`
                outputs = self(
                    **model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions
                )

                # last decoder hidden states will be used to compute the degeneration penalty (cosine similarity with
                # previous tokens)
                if self.config.is_encoder_decoder:
                    last_hidden_states = outputs.decoder_hidden_states[-1]
                else:
                    last_hidden_states = outputs.hidden_states[-1]

                # next logit for contrastive search to select top-k candidate tokens
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for this first iteration
                # (the clone itself is always small)
                logit_for_next_step = outputs.logits[:, -1, :].clone()

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    standardize_cache_format=True,
                )

                if not sequential:
                    # Expands model inputs top_k times, for batched forward passes (akin to beam search).
                    _, model_kwargs = self._expand_inputs_for_generation(
                        expand_size=top_k, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
                    )

                past_key_values = model_kwargs.get("past_key_values")
                if past_key_values is None:
                    raise ValueError(
                        f"{self.__class__.__name__} does not support caching and therefore **can't** be used "
                        "for contrastive search."
                    )
                elif (
                    not isinstance(past_key_values[0], (tuple, torch.Tensor))
                    or past_key_values[0][0].shape[0] != batch_size
                ):
                    raise ValueError(
                        f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be "
                        "used for contrastive search without further modifications."
                    )

            # contrastive_search main logic start:
            # contrastive search decoding consists of two steps: (1) candidate tokens recall; (2) candidate re-rank by
            # degeneration penalty
            processed_logit_for_next_step = logits_processor(input_ids, logit_for_next_step)
            next_probs = nn.functional.softmax(processed_logit_for_next_step, dim=-1)

            top_k_probs, top_k_ids = torch.topk(next_probs, dim=-1, k=top_k)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (logit_for_next_step,)
                if output_scores:
                    scores += (processed_logit_for_next_step,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # This is needed to properly delete outputs.logits which may be very large for this first iteration
            # Otherwise a reference to outputs.logits is kept all along until after the next call to self.forward()
            del outputs

            if not sequential:
                # Replicates the new past_key_values to match the `top_k` candidates
                past = model_kwargs["past_key_values"]
                # If it is a static cache, modify it in-place layer after layer to save memory
                if isinstance(past, DynamicCache):
                    past.batch_repeat_interleave(top_k)
                else:
                    new_key_values = []
                    for layer in past:
                        items = []
                        # item is either the key or the value matrix
                        for item in layer:
                            items.append(item.repeat_interleave(top_k, dim=0))
                        new_key_values.append(tuple(items))

                    past = tuple(new_key_values)

                model_kwargs["past_key_values"] = past

            if sequential:
                all_outputs = []
                for i in range(top_k):
                    # compute the candidate tokens by the language model and collect their hidden_states
                    next_model_inputs = self.prepare_inputs_for_generation(top_k_ids[:, i].view(-1, 1), **model_kwargs)

                    outputs = self(
                        **next_model_inputs,
                        return_dict=True,
                        output_hidden_states=True,
                        output_attentions=output_attentions,
                    )
                    if isinstance(outputs["past_key_values"], DynamicCache):
                        # Remove past K-V from output since we don't need to stack later
                        outputs["past_key_values"] = None
                        # Remove last token from past K-V since we don't want to append it at this point
                        model_kwargs["past_key_values"].crop(-1)

                    all_outputs.append(outputs)
                outputs = stack_model_outputs(all_outputs)

            else:
                # compute the candidate tokens by the language model and collect their hidden_states
                # assembles top_k_ids into batch of size k
                next_model_inputs = self.prepare_inputs_for_generation(top_k_ids.view(-1, 1), **model_kwargs)

                outputs = self(
                    **next_model_inputs,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )

            # This is essential to avoid having a last reference to the big past K-V and double the necesary memory
            # in the next loop
            del next_model_inputs

            # name is different for encoder-decoder and decoder-only models
            if self.config.is_encoder_decoder:
                next_hidden = outputs.decoder_hidden_states[-1]
                full_hidden_states = outputs.decoder_hidden_states
            else:
                next_hidden = outputs.hidden_states[-1]
                full_hidden_states = outputs.hidden_states

            logits = outputs.logits[:, -1, :]
            context_hidden = last_hidden_states.repeat_interleave(top_k, dim=0)

            # compute the degeneration penalty and re-rank the candidates based on the degeneration penalty and the
            # model confidence. Keeping `selected_idx` on CPU enables multi-device contrastive search and doesn't
            # introduce (noticeable) slowdowns on single-device runs.
            selected_idx = _ranking_fast(context_hidden, next_hidden, top_k_probs, penalty_alpha, top_k)
            selected_idx = selected_idx.to("cpu")

            # This will be used instead of the previous inneficient torch.stack(torch.split())
            augmented_idx = torch.tensor([x + i * top_k for i, x in enumerate(selected_idx)])

            # prepare for the next step: (1) next token_id; (2) past_key_values; (3) last_hidden_states for computing
            # the degeneration penalty; (4) logits for selecting next top-k candidates; (5) selected tokens scores
            # (model confidence minus degeneration penalty); (6) decoder hidden_states
            next_tokens = top_k_ids[range(len(top_k_ids)), selected_idx]
            next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), top_k))
            next_hidden = next_hidden[range(batch_size), selected_idx, :]
            last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)

            next_decoder_hidden_states = ()
            for layer in full_hidden_states:
                layer = torch.stack(torch.split(layer, top_k))[range(batch_size), selected_idx, :]
                next_decoder_hidden_states += (layer,)

            # generate past_key_values cache of only the selected token
            if sequential:
                next_model_input = self.prepare_inputs_for_generation(
                    top_k_ids[:, selected_idx].view(-1, 1), **model_kwargs
                )

                selected_outputs = self(
                    **next_model_input,
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                next_past_key_values = selected_outputs["past_key_values"]

            else:
                _, next_past_key_values = self._extract_past_from_model_output(outputs, standardize_cache_format=True)
                # Do it in-place layer per layer to save memory
                if isinstance(next_past_key_values, DynamicCache):
                    next_past_key_values.batch_select_indices(augmented_idx)
                else:
                    new_key_values = []
                    for layer in next_past_key_values:
                        items = []
                        # item is either the key or the value matrix
                        for item in layer:
                            items.append(item[augmented_idx, Ellipsis])
                        new_key_values.append(tuple(items))

                    next_past_key_values = tuple(new_key_values)

            logit_for_next_step = torch.stack(torch.split(logits, top_k))[range(batch_size), selected_idx, :]

            # Rebuilds the relevant parts of the model output for the selected token, for use in the next iteration
            if self.config.is_encoder_decoder:
                next_step_cross_attentions = ()
                next_step_decoder_attentions = ()
                if output_attentions:
                    for layer in outputs.cross_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, Ellipsis]
                        next_step_cross_attentions += (layer,)
                    for layer in outputs.decoder_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, Ellipsis]
                        next_step_decoder_attentions += (layer,)
                outputs = Seq2SeqLMOutput(
                    past_key_values=next_past_key_values,
                    decoder_hidden_states=next_decoder_hidden_states,
                    decoder_attentions=next_step_decoder_attentions or None,
                    cross_attentions=next_step_cross_attentions or None,
                )
            else:
                next_step_attentions = ()
                if output_attentions:
                    for layer in outputs.attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, Ellipsis]
                        next_step_attentions += (layer,)
                outputs = CausalLMOutputWithPast(
                    past_key_values=next_past_key_values,
                    hidden_states=next_decoder_hidden_states,
                    attentions=next_step_attentions or None,
                )
            # contrastive_search main logic end

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # stop when each sentence is finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            # Contrastive search works by forward looking at the next token, so we need to exclude it from
            # `past_key_values` to be consistent with the other decoding methods
            if model_kwargs.get("past_key_values") is not None:
                if isinstance(model_kwargs["past_key_values"], DynamicCache):
                    model_kwargs["past_key_values"].crop(-1)
                else:
                    past_key_values = []
                    for layer in model_kwargs["past_key_values"]:
                        layer_past_key_values = []
                        for item in layer:
                            layer_past_key_values.append(item[Ellipsis, :-1, :])
                        past_key_values.append(tuple(layer_past_key_values))
                    model_kwargs["past_key_values"] = tuple(past_key_values)

            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _greedy_search(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        **model_kwargs,
    ):
        r"""
        Deprecated. Use `._sample()` instead, passing the same arguments.
        """

        logger.warning_once(
            "Calling `._greedy_search()` directly is deprecated and will be removed in v4.42. Use `._sample()` "
            "instead, passing the same arguments."
        )
        return self._sample(
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    def _sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        logits_warper = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        """
        Temporary function to handle the different types of cache reordering processes while we roll out `Cache`.

        TODO: standardize cache formats and make all models compatible with `Cache`. It would remove the need
        for this function, with `Cache.reorder_cache` being the sole remaining code path
        """
        model_class = self.__class__.__name__.lower()
        # Exception 1: code path for models using the legacy cache format
        if isinstance(past_key_values, (tuple, list)):
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
        # Exception 2: models with different cache formats. These are limited to `DynamicCache` until their
        # cache format is standardized, to avoid adding complexity to the codebase.
        elif "bloom" in model_class or "gptbigcode" in model_class:
            if not isinstance(past_key_values, DynamicCache):
                raise ValueError(
                    f"Using an unsupported cache format with {model_class}. Currently, it only supports the "
                    "legacy tuple format or `DynamicCache`"
                )
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # Standard code path: use the `Cache.reorder_cache`
        else:
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    # TODO (joao, v4.42): remove default for `logits_warper`
    def _beam_search(
        self,
        input_ids,
        beam_scorer,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        logits_warper = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "bloom",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(
                        **inputs_per_sub_batch,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)

            else:  # Unchanged original behavior
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            if do_sample:
                next_token_scores_processed = logits_warper(input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def _beam_sample(
        self,
        input_ids,
        beam_scorer,
        logits_processor,
        stopping_criteria,
        logits_warper,
        generation_config,
        synced_gpus,
        **model_kwargs,
    ):
        r"""
        Deprecated. Use `._beam_search()` instead, passing the same arguments.
        """

        logger.warning_once(
            "Calling `._beam_sample()` directly is deprecated and will be removed in v4.42. Use `._beam_search()` "
            "instead, passing the same arguments."
        )
        return self._beam_search(
            input_ids=input_ids,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    def _group_beam_search(
        self,
        input_ids,
        beam_scorer,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **diverse beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        device = input_ids.device

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if return_dict_in_generate and output_scores:
            beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]
        else:
            beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam of each group with 0 and the rest with -1e9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            if output_scores:
                processed_score = torch.zeros_like(outputs.logits[:, -1, :])
            if output_logits:
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                raw_logit_score = outputs.logits[:, -1, :].clone()

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of current group only
                # No need to clone() the logits here as they will not retain outputs.logits at the end of the loop
                next_token_logits = outputs.logits[batch_group_indices, -1, :]

                next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * group_size, vocab_size)
                vocab_size = next_token_scores.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                if output_scores:
                    processed_score[batch_group_indices] = next_token_scores_processed

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
                n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, max(2, 1 + n_eos_tokens) * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                    decoder_prompt_len=decoder_prompt_len,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                if return_dict_in_generate and output_scores:
                    beam_indices[beam_group_idx] = tuple(
                        beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0]))
                    )

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor")
                    + group_start_idx
                    + (beam_idx % group_size)
                )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (processed_score,)
                if output_logits:
                    raw_logits += (raw_logit_score,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], reordering_indices
                )

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def _constrained_beam_search(
        self,
        input_ids,
        constrained_beam_scorer,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **constrained beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation, while satisfying a list of positive constraints. For more information, the
                documentation of [`ConstrainedBeamSearchScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        batch_size = len(constrained_beam_scorer._beam_hyps)
        num_beams = constrained_beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            scores_for_all_vocab = next_token_scores.clone()

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = constrained_beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                scores_for_all_vocab,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if constrained_beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = constrained_beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def _assisted_decoding(
        self,
        input_ids,
        candidate_generator,
        logits_processor,
        logits_warper,
        stopping_criteria,
        generation_config,
        synced_gpus,
        teacher_k,
        teacher_p,
        expected_seq_len,
        teacher_k_lower,
        streamer,
        tokenizer,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
        **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
        candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
        models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            candidate_generator (`CandidateGenerator`):
                A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
                more information, the documentation of [`CandidateGenerator`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only used if sampling is active.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        do_sample = logits_warper is not None
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # This is needed if return_dict_in_generate is True
        start_from_empty_dynamic_cache = False
        if isinstance(model_kwargs.get("past_key_values", None), DynamicCache):
            if len(model_kwargs["past_key_values"]) == 0:
                start_from_empty_dynamic_cache = True

        this_peer_finished = False

        max_len = stopping_criteria[0].max_length
        # 2 * max_len to give us room to potentially left cut
        position_ids = torch.arange(2 * max_len, device=input_ids.device, dtype=torch.long)[None, :].broadcast_to(batch_size, 2 * max_len) if batch_size > 1 else None
        attention_mask = torch.ones_like(position_ids) if position_ids is not None else  None
        n_matches = None
        eos_tokens_mask = None

        # calculate number of tokens to perform decay
        if expected_seq_len>0:
            num_token_decay = max(int(expected_seq_len/teacher_k), 1)
            num_token_per_decay = max(math.ceil(teacher_k/expected_seq_len), 1)
            cur_token_index = 0
            # cur_teacher_k = teacher_k

        cor_count = 0
        tot_count = 0

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # tot_count += 1
            # Rotate everything for bsz > 1
            if n_matches is not None and position_ids is not None:
                # compute by how much everything can be rotated
                shift = unfinished_sequences * (n_matches.max() - n_matches) + (1 - unfinished_sequences) * (eos_tokens_mask.sum(-1) - 1)

                for i in range(batch_size):
                    if shift[i] > 0:
                        input_ids[i][shift[i]:] = input_ids[i][:-shift[i]].clone()
                        input_ids[i][:shift[i]] = self.config.pad_token_id

                position_ids = position_ids.add(-shift[:, None]).clamp(min=0)
                attention_mask[:, :-1] = position_ids[:, 1:] > 0

                left_cut = (1 - attention_mask).sum(-1).min()

                if left_cut > 0:
                    position_ids = position_ids[:, left_cut:]
                    attention_mask = attention_mask[:, left_cut:]
                    input_ids = input_ids[:, left_cut:]

                    model_kwargs["past_key_values"] = _crop_past_key_values(self, model_kwargs["past_key_values"], left_cut=left_cut)
                    model_kwargs["assistant_past_key_values"] = _crop_past_key_values(
                        candidate_generator.assistant_model, model_kwargs["assistant_past_key_values"], left_cut=left_cut
                    )  # the assistant does not have the token after the last match, hence the -1
            cur_len = input_ids.shape[-1]

            #  1. Fetch candidate sequences from a `CandidateGenerator`
            candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids, tokenizer)
            candidate_input_ids = candidate_input_ids.to(self.device)
            if candidate_logits is not None:
                candidate_logits = candidate_logits.to(self.device)

            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            is_done_candidate = stopping_criteria(candidate_input_ids, None)

            # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
            # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
            # we use this forward pass to also pick the subsequent logits in the original model.

            # 2.1. Prepare the model inputs
            candidate_kwargs = copy.copy(model_kwargs)
            candidate_kwargs = _prepare_attention_mask(
                candidate_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
            )
            candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
            if "cache_position" in candidate_kwargs:
                candidate_kwargs["cache_position"] = torch.cat(
                    (
                        candidate_kwargs["cache_position"],
                        torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                    ),
                    dim=0,
                )

            model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
            if "num_logits_to_keep" in model_inputs:
                model_inputs["num_logits_to_keep"] = candidate_length + 1

            # 2.2. Run a forward pass on the candidate sequence
            outputs = self(
                **model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # 2.3. Process the new logits
            new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
            next_token_logits = new_logits.clone()

            if len(logits_processor) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
            if do_sample and len(logits_warper) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

            # 3. Select the accepted tokens. There are two possible cases:
            # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
            # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
            if do_sample and candidate_logits is not None:
                valid_tokens, n_matches, plus_enable, trigger_enable = _speculative_kd_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_length,
                    new_logits,
                    is_done_candidate,
                    teacher_k,
                    teacher_p,
                )

                if trigger_enable:
                    cor_count+=1
                # print("Num of matches: ", n_matches)
                tot_count+=n_matches.item()

                if expected_seq_len>0:
                    # if cur_token_index < int(expected_seq_len/2):
                    #     cur_teacher_k = teacher_k
                    # else:
                    #     cur_teacher_k = teacher_k_lower
                    if cur_token_index % num_token_decay == 0:
                        teacher_k -= num_token_per_decay
                        teacher_k = max(teacher_k, 1)
                    cur_token_index+=1

            # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
            # original model logits with the candidate tokens. We can keep the candidate tokens until the first
            # mismatch, or until the max length is reached.
            else:
                # In this case, we do not allow candidate generation since we reach the maximum length
                n_matches = 0
                plus_enable = False

            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.

            # 4.1. Get the valid continuation, after the matching tokens
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            if streamer is not None:
                streamer.put(valid_tokens.cpu())
            new_cur_len = input_ids.shape[-1]

            # 4.2. Discard past key values relative to unused assistant tokens
            new_cache_size = new_cur_len - 1
            outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

            # 5. Update the candidate generation strategy if needed
            candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Store scores, attentions and hidden_states when required
            # Assistant: modified to append one tuple element per token, as in the other generation methods.
            if return_dict_in_generate:
                if output_scores:
                    if plus_enable:
                        scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))
                    else:
                        scores += tuple(new_logits[:, i, :] for i in range(n_matches))
                if output_logits:
                    if plus_enable:
                        raw_logits += (next_token_logits[:, :n_matches+1, :],)
                    else:
                        raw_logits += (next_token_logits[:, :n_matches, :],)

                if "past_key_values" not in model_kwargs or start_from_empty_dynamic_cache:
                    added_len = new_cur_len
                    # set it to false for other iterations
                    start_from_empty_dynamic_cache = False
                else:
                    if plus_enable:
                        added_len = n_matches + 1
                    else:
                        added_len = n_matches

                if output_attentions:
                    if self.config.is_encoder_decoder:
                        cross_attentions = _split_model_outputs(
                            cross_attentions, outputs.cross_attentions, cur_len, added_len
                        )
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.decoder_attentions,
                            cur_len,
                            added_len,
                            is_decoder_attention=True,
                        )
                    else:
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.attentions,
                            cur_len,
                            added_len,
                            is_decoder_attention=True,
                        )
                if output_hidden_states:
                    if self.config.is_encoder_decoder:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len
                        )
                    else:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                        )

            if plus_enable:
                plus_added_len=1
            else:
                plus_added_len=0

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                num_new_tokens=n_matches+plus_added_len,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if (
            hasattr(candidate_generator, "assistant_model")
            and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
        ):
            candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
                candidate_generator.num_assistant_tokens
            )

        # print(cor_count/tot_count)
        # print("total: ", tot_count)

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    correction_rate=cor_count/tot_count,
                )
        else:
            return input_ids

def top_k_top_p_filtering(
    logits,
    top_k = 0,
    top_p = 1.0,
    filter_value = -float("Inf"),
    min_tokens_to_keep = 1,
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][Ellipsis, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[Ellipsis, :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[Ellipsis, 1:] = sorted_indices_to_remove[Ellipsis, :-1].clone()
        sorted_indices_to_remove[Ellipsis, 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def _speculative_kd_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    is_done_candidate,
    teacher_k,
    teacher_p,
):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    # use ancestral sampling to sample one student token
    q_selected_tokens = torch.multinomial(q[0, :, :], num_samples=1)
    # next_token_logscores = top_k_top_p_filtering(candidate_logits[0, :, :], 0, 0.5)
    # probs = F.softmax(next_token_logscores, dim=-1)
    # q_selected_tokens = torch.multinomial(probs, num_samples=1)
    # print("q_selected_tokens: ", q_selected_tokens.size())
    p = new_logits.softmax(dim=-1)
    # check if student token is within top k at teacher's tokens
    if teacher_k > 0:
        # last token at teacher should be removed
        p_selected_tokens_tuple = torch.topk(p[0, :-1, :], teacher_k)
        p_selected_tokens = p_selected_tokens_tuple.indices
    elif teacher_p > 0:
        # last token at teacher should be removed
        sorted_logits, p_selected_tokens = torch.sort(p[0, :-1, :], descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
        # remove all tokens which are above proability mass
        sorted_indices_to_remove = cumulative_probs >= teacher_p
        # we need to include first token which is above probability mass
        sorted_indices_to_remove[:, 0] = False
        p_selected_tokens[sorted_indices_to_remove] = -1
    else:
        print("We do not support other truncation for teacher other than top-k or top-p!")
        exit(1)

    is_accepted = torch.sum(p_selected_tokens==q_selected_tokens.repeat(1, p_selected_tokens.size(1)), dim=1)>0
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1
    # print("Num of matches: ", n_matches)

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if is_done_candidate and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
        return valid_tokens, n_matches, True, False
    else:
        gamma = candidate_logits.shape[1]
        # Resample from teacher if student's sample is unlikely to generated (use ancestral sampling to sample teacher token)
        p_prime = p[:, n_matches, :]
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
        # p_logits_prime = new_logits[:, n_matches, :]
        # p_next_token_logscores = top_k_top_p_filtering(p_logits_prime, 0, 0.5)
        # p_probs = F.softmax(p_next_token_logscores, dim=-1)
        # t = torch.multinomial(p_probs, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0 and n_matches == gamma:
            valid_tokens = new_candidate_input_ids[:, :n_matches]
            return valid_tokens, n_matches, False, False
        if n_matches > 0 and n_matches < gamma:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
            return valid_tokens, n_matches, True, True
        else:
            valid_tokens = t
            return valid_tokens, n_matches, True, True

def _split_model_outputs(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    """
    Given the (decoder/cross attentions)/(decoder hidden states) for multiple generated tokens, splits it into a tuple
    where each member corresponds to a single generated token.
    """
    # Retrocompatibility: in our generation functions, the first iteration includes the attention/hidden states for the
    # prompt.
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[Ellipsis, :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        # The first iteration contains the prompt + 1 generated token, let's update the length variables accordingly
        cur_len += 1
        added_len -= cur_len

    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[Ellipsis, i : i + 1, :last_dim_size],)
        outputs += (new_tuple,)
    return outputs


def _ranking_fast(
    context_hidden,
    next_hidden,
    next_top_k_probs,
    alpha,
    beam_width,
):
    """
    Reranks the top_k candidates based on a degeneration penalty (cosine similarity with previous tokens), as described
    in the paper "A Contrastive Framework for Neural Text Generation". Returns the index of the best candidate for each
    row in the batch.
    """
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)  # [B*K, S]
    degeneration_penalty, _ = torch.max(cosine_matrix, dim=-1)  # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]
    contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty
    contrastive_score = torch.stack(torch.split(contrastive_score, beam_width))  # [B, K]
    _, selected_idx = contrastive_score.max(dim=-1)  # [B]
    return selected_idx


def _split(data, full_batch_size, split_size = None):
    """
    Takes care of three cases:
    1. data is a tensor: e.g. last_hidden_state, pooler_output etc. split them on the batch_size dim
    2. data is a tuple: e.g. hidden_states, attentions etc. Keep the tuple as it is and split each tensor in it and
       return a list of tuples
    3. data is a tuple of tuples, e.g. past_key_values. Keep the tuple as it is and split each tuple in it and
       return a list of tuples of tuples
    (see documentation of ModelOutput)
    """
    if data is None:
        return [None] * (full_batch_size // split_size)
    if isinstance(data, torch.Tensor):
        return [data[i : i + split_size] for i in range(0, full_batch_size, split_size)]
    # New cache format
    elif isinstance(data, DynamicCache):
        return data.batch_split(full_batch_size, split_size)
    elif isinstance(data, tuple):
        # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
        if isinstance(data[0], tuple):
            return [
                tuple(tuple(tensor[i : i + split_size] for tensor in inner_tuple) for inner_tuple in data)
                for i in range(0, full_batch_size, split_size)
            ]

        else:
            return [
                tuple(sub_tensor[i : i + split_size] for sub_tensor in data)
                for i in range(0, full_batch_size, split_size)
            ]
    else:
        raise ValueError(f"Unexpected attribute type: {type(data)}")


def _split_model_inputs(
    model_input, split_size, full_batch_size
):
    """
    Split a ModelOutput object (or its subclasses) or Dict into a list of same-class objects based on a specified split
    size. The input object is dict when it was prepared for forward pass and ModelOutput when it was returned from
    previous forward pass.
    """
    # Edge case: if model_input is None, return a list of Nones
    # this happens with Whisper where encoder_outputs is None
    if model_input is None:
        return [model_input] * (full_batch_size // split_size)
    # Infer the class from the object
    model_output_cls = type(model_input)
    if (full_batch_size % split_size) != 0:
        raise ValueError("`full_batch_size` must be divisible by `split_size`")

    if split_size > full_batch_size:
        raise ValueError("`split_size` must be smaller or equal to `full_batch_size`")

    # Helper function to split tensors or tuples of tensors

    # Find all the dataclass fields (e.g., last_hidden_state, pooler_output etc.) and split them
    keys = (
        model_input.__dataclass_fields__.keys() if hasattr(model_input, "__dataclass_fields__") else model_input.keys()
    )
    # We only keep keys that are in the model_input
    keys = [k for k in keys if k in model_input]
    # Here we can have four types of values: tensors, tuples of tensors and booleans, and encoder_outputs which is a
    # ModelOutput object.
    # bool should not be split but replicated for each split
    bool_keys = [k for k in keys if isinstance(model_input[k], bool) or k == "cache_position"]
    keys_to_ignore = ["cache_position", "encoder_outputs", "num_logits_to_keep"]
    non_bool_keys = [k for k in keys if not isinstance(model_input[k], bool) and k not in keys_to_ignore]

    # we split the tensors and tuples of tensors
    data_split_list = [
        {k: _split(model_input[k], full_batch_size, split_size)[i] for k in non_bool_keys}
        for i in range(full_batch_size // split_size)
    ]
    # bool values are the same and replicated for each split
    bool_data = {k: model_input[k] for k in bool_keys}
    # encoder_outputs is a ModelOutput object and should be split by its own
    if "encoder_outputs" in model_input:
        encoder_outputs_split = _split_model_inputs(model_input["encoder_outputs"], split_size, full_batch_size)
        data_split_list = [
            {**data_split, "encoder_outputs": encoder_outputs_split[i]} for i, data_split in enumerate(data_split_list)
        ]
    # num_logits_to_keep should be replicated for each split, similar to bool values
    if "num_logits_to_keep" in model_input:
        data_split_list = [
            {**data_split, "num_logits_to_keep": model_input["num_logits_to_keep"]} for data_split in data_split_list
        ]

    # Convert each dictionary in the list to an object of the inferred class
    split_model_inputs: List[Union[ModelOutput, Dict]] = [
        model_output_cls(**data_split, **bool_data) for data_split in data_split_list
    ]

    return split_model_inputs


def stack_model_outputs(model_outputs):
    """
    Stack a list of ModelOutput objects (or its subclasses) along the batch_size dimension. The function infers the
    specific ModelOutput subclass from the list provided.
    """
    if not model_outputs:
        raise ValueError("Input list is empty.")

    # Infer the class from the first object in the list
    model_output_cls = type(model_outputs[0])

    # Ensure all objects are of the same type
    if not all(isinstance(obj, model_output_cls) for obj in model_outputs):
        raise ValueError("All elements in the list should be of the same type.")

    # Helper function to concat tensors or tuples of tensors
    def _concat(data):
        """
        Reverse of `_split` function above.
        """
        if any(data is None for data in data):
            return None
        if isinstance(data[0], torch.Tensor):
            return torch.cat(data, dim=0)
        # New cache format
        elif isinstance(data[0], DynamicCache):
            return DynamicCache.from_batch_splits(data)
        elif isinstance(data[0], tuple):
            # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
            if isinstance(data[0][0], tuple):
                return tuple(
                    tuple(torch.cat([attr[i][j] for attr in data], dim=0) for j in range(len(data[0][0])))
                    for i in range(len(data[0]))
                )
            else:
                return tuple(torch.cat([attr[i] for attr in data], dim=0) for i in range(len(data[0])))
        elif isinstance(data[0], (int, float)):
            # If the elements are integers or floats, return a tensor
            return torch.tensor(data)
        else:
            raise ValueError(f"Unexpected attribute type: {type(data[0])}")

    # Use a dictionary comprehension to gather attributes from all objects and concatenate them
    concatenated_data = {
        k: _concat([getattr(model_output, k) for model_output in model_outputs])
        for k in model_output_cls.__dataclass_fields__.keys()
    }

    # Return a new object of the inferred class with the concatenated attributes
    return model_output_cls(**concatenated_data)
