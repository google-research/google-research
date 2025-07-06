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

"""Run hugging face models."""

import collections
from collections.abc import Sequence
import gc
import re
import threading
from typing import Any, Callable
import peft
import torch
import transformers
from cisc.src.runners import runner as runner_lib

AutoModelForCausalLM = transformers.AutoModelForCausalLM
AutoTokenizer = transformers.AutoTokenizer
# BitsAndBytesConfig = transformers.BitsAndBytesConfig
GenerationConfig = transformers.GenerationConfig

# Holds history of the amount of GPU memory that was reserved.
_DEBUG_RESERVED_MEMORY_GB = collections.Counter()
_LOCK = threading.Lock()  # Lock for updating _DEBUG_RESERVED_MEMORY_GB.


# Load each model only once to save expensive memory.
_hf_cache = {}


def get_completion_likelihood_multi_token(
    prefix, completions, model, tokenizer, device="cuda"
):
  """Returns a sequence of log-probs for each completion.

  This is similar to `get_completion_likelihoods` but support multiple tokens.
  It is more expansive, because it runs the model again for each of the
  completions.

  Args:
    prefix: The prefix to the completion.
    completions: The completions to score.
    model: The model to use.
    tokenizer: The tokenizer to use.
    device: The device to use.

  Returns:
    A sequence of log-probs for each completion.
  """
  assert tokenizer.padding_side == "left"
  assert prefix, "prefix should include at least the start seq token."

  full_prompts = [prefix + c for c in completions]
  with _LOCK:  # Use lock to avoid github.com/huggingface/tokenizers/issues/537
    full_inputs = tokenizer(
        full_prompts,
        max_length=1024,
        truncation=True,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
  input_ids = full_inputs.input_ids
  attention_mask = full_inputs.attention_mask
  if torch.cuda.is_available() and device == "cuda":
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
  with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

  # Logits has shape [batch_size, prefix + completion, vocab].
  assert outputs is not None and outputs.logits is not None
  log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

  # Advance by 1. That is, the logit of the first token should model the
  # probability to get the second token.
  input_ids = input_ids[:, 1:].unsqueeze(-1)

  # Instead of having the log-prob for all the vocab at each position, get
  # only the log_probs that correspond to the input sequence.
  chosen_log_probs = (
      torch.gather(log_probs, dim=-1, index=input_ids).squeeze(-1).cpu()
  )

  # Keep only the tokens from `completions` not including the `prefix`.
  with _LOCK:  # Use lock to avoid github.com/huggingface/tokenizers/issues/537
    completions_ids = tokenizer(
        completions,
        max_length=1024,
        add_special_tokens=False,
        padding=False,
    ).input_ids
  chosen_log_probs = [
      chosen_log_probs[i, -1 * len(c) :] for i, c in enumerate(completions_ids)
  ]

  del input_ids
  del attention_mask
  del outputs
  del log_probs
  gc.collect()
  torch.cuda.empty_cache()

  return chosen_log_probs


def cache_or_init(key, init_func):
  """Returns the model from the cache or initializes it."""
  if key not in _hf_cache:
    print(f"loading {key}")
    _hf_cache[key] = init_func()
  return _hf_cache[key]


class Runner(runner_lib.Runner):
  """Interface for running a model."""

  def __init__(
      self,
      model_dir,
      torch_dtype = "auto",
      return_scores = False,
      return_embeddings = False,
      # Optional. If not None will also try to load the lora adapters.
      peft_model_dir = None,
  ):
    self.tokenizer = cache_or_init(
        key=f"{model_dir}_tokenizer",
        init_func=lambda: AutoTokenizer.from_pretrained(
            model_dir,
            device_map="auto",
            padding_side="left",
            padding=True,
            truncation=True,
            truncation_side="left",
            trust_remote_code=True,
        ),
    )
    self.model = cache_or_init(
        key=f"{model_dir}_model",
        init_func=lambda: AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            trust_remote_code=True,
        ),
    )
    if peft_model_dir:
      print(f"Loading LoRA (PEFT) model from {peft_model_dir}")
      self.model = peft.PeftModel.from_pretrained(self.model, peft_model_dir)

    self.model.config.pad_token_id = self.tokenizer.pad_token_id = (
        self.tokenizer.eos_token_id
    )
    self._return_scores = return_scores
    self._return_embeddings = return_embeddings

  def generate(
      self,
      prompts,
      max_new_tokens,
      temperature,
      enable_formatting,
      return_scores = None,
      use_cache = True,
  ):
    """Generates a single response for each prompt."""
    if return_scores is None:
      return_scores = self._return_scores

    if enable_formatting:
      prompts = self.tokenizer.apply_chat_template(
          [[{"role": "user", "content": p}] for p in prompts],
          return_tensors="pt",
          add_generation_prompt=True,
          tokenize=False,
      )
    # If enable formatting is false, we don't need special tokens. If it is
    # true, than apply_chat_template already adds the special tokens.
    # Use lock to avoid github.com/huggingface/tokenizers/issues/537
    with _LOCK:
      inputs = self.tokenizer(
          prompts, return_tensors="pt", padding=True, add_special_tokens=False
      )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    if torch.cuda.is_available():
      input_ids = input_ids.cuda()
      attention_mask = attention_mask.cuda()

    # Decode the formatted prompts. This information is returned to the caller,
    # so that it is possible to reproduce the exact same generation.
    # TODO(amirt): Consider removing only the padding tokens.
    processed_prompts = self.tokenizer.batch_decode(
        input_ids, skip_special_tokens=False
    )

    try:
      return_dict_in_generate = return_scores or self._return_embeddings
      generation_output = self.model.generate(
          input_ids=input_ids,
          attention_mask=attention_mask,
          generation_config=GenerationConfig(
              do_sample=temperature != 0,
              temperature=temperature,
              top_p=0.95,  # Nucleus sampling
              num_beams=1,
              use_cache=use_cache,  # True for speedup, False to save memory..
              max_new_tokens=max_new_tokens,
              pad_token_id=self.tokenizer.pad_token_id,
              eos_token_id=self.tokenizer.eos_token_id,
          ),
          return_dict_in_generate=return_dict_in_generate,
          output_scores=return_scores,
          output_hidden_states=self._return_embeddings,
          max_new_tokens=max_new_tokens,
          # Avoid the cases where the model returns <eos> immediately. This can
          # happen when the last turn was not properly formatted.
          min_new_tokens=min(max_new_tokens, 10),
          pad_token_id=self.tokenizer.pad_token_id,
      )
      # Now, break all fields of generation_output by the batch size dimension.

      # `output_token` has shape of [batch_size, input_len + ouput_len]. Remove
      # the input tokens, and decode only the generated tokens.
      output_token = (
          generation_output["sequences"].cpu()
          if return_dict_in_generate
          else generation_output.cpu()
      )
      generated_tokens = output_token[:, input_ids.shape[1] :]
      generated_texts = self.tokenizer.batch_decode(
          generated_tokens, skip_special_tokens=False
      )
      # Remove consecutive padding tokens from the end of the generated text.
      generated_texts = [
          re.sub(
              f"({self.tokenizer.pad_token})\\1+$",
              self.tokenizer.pad_token,
              text,
          )
          for text in generated_texts
      ]

      generated_scores = [None] * len(generated_texts)
      if return_scores:
        # generation_output["scores"] is a tuple of length [ouput_len] where
        # each element is a vector of shape [batch_size, vocab_size].
        generated_scores = list(
            torch.stack(generation_output["scores"]).permute(1, 0, 2).cpu()
        )
      generated_embeddings = [None] * len(generated_texts)
      if self._return_embeddings:
        # generation_output["hidden_states"] has shape
        # [output_len, model_layers, batch_size, X, embedding_size]
        # - output_len, model_layers are tuples.
        # - X size is [input_len] for the first element and [1] for the rest.
        generated_embeddings = [
            last_layer.cpu()
            for last_layer in generation_output["hidden_states"][0][-1]
        ]
      del generation_output  # Release expensive GPU memory.
    finally:
      # Note that in case of a crash this still won't clear all the memory.
      # But we do it anyway as best effort.
      if torch.cuda.is_available():
        with _LOCK:
          _DEBUG_RESERVED_MEMORY_GB.update(
              [round(torch.cuda.memory_reserved(0) / 1e9)]
          )
      # Release expensive GPU memory.
      del input_ids
      del attention_mask
      gc.collect()
      torch.cuda.empty_cache()

    return [
        runner_lib.GenerationOutput(
            prompt=prompt,
            response=text,
            exception="",
            scores=scores,
            embeddings=embs,
        )
        for prompt, text, scores, embs in zip(
            processed_prompts,
            generated_texts,
            generated_scores,
            generated_embeddings,
        )
    ]

  def get_normalized_probability_for_sequence(
      self,
      prefix,
      completion,
  ):
    sequence_probs = get_completion_likelihood_multi_token(
        prefix, [completion], self.model, self.tokenizer
    )
    assert len(sequence_probs) == 1  # Only a single completion.
    sequence_probs = sequence_probs[0]
    # Normalized sequence probablity, like in the self consistency paper.
    return torch.exp(sum(sequence_probs) / len(sequence_probs)).item()

  def get_completion_likelihoods(
      self,
      prefixes,
      completions,
      enable_formatting,
  ):
    # Only support completions of a single token.
    completion_ids = []
    for completion in completions:
      completion_id = self.tokenizer.convert_tokens_to_ids(completion)
      assert isinstance(completion_id, int)
      assert (
          completion_id != self.tokenizer.unk_token_id
      ), f"completion {completion} must be a signle valid token."
      completion_ids.append(completion_id)

    # Generate one token, just to get the scores.
    generate_results = self.generate(
        prompts=list(prefixes),
        max_new_tokens=1,
        temperature=0,
        return_scores=True,
        enable_formatting=enable_formatting,
        use_cache=False,
    )

    # For a specific result, extract the score of each completion.
    def get_score(result):
      if result.exception:
        return None
      assert (
          result.scores is not None and len(result.scores) == 1
      ), "Expected a single score for the single token generation."
      scores = result.scores.squeeze()
      return [scores[id] for id in completion_ids]

    return [get_score(result) for result in generate_results]
