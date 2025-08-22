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

"""Top-level code for the LitM experiments using external libraries."""

from abc import ABC
from abc import abstractmethod
import copy
import json
import os
import time
import uuid

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration


device = "cuda:0"

SUPPORTED_MODEL_DICT = {
    "gemma7b": "google/gemma-7b",
    "gemma2b": "google/gemma-2b",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "t5-large": "google-t5/t5-large",
}


class InferencePlatform(ABC):
  """An abstract class for the LLM inference platform we use."""

  @abstractmethod
  def predict(self, prompt):
    pass


class HuggingFace(InferencePlatform):
  """An implementation for using HuggingFace as the platform for LLM inference."""

  def __init__(self):
    self._tokenizer = None
    self._model = None

  def authenticate(self, huggingface_token):
    os.environ["HF_TOKEN"] = huggingface_token

  def setup_model(self, model_name):
    if model_name not in SUPPORTED_MODEL_DICT:
      raise ValueError(f"Unsupported model: {model_name}")
    self.model_name = model_name
    if model_name in ["gemma2b", "gemma7b", "llama3"]:
      hf_path = SUPPORTED_MODEL_DICT[model_name]

      self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
      self.model = AutoModelForCausalLM.from_pretrained(
          hf_path,
          device_map="auto",
          torch_dtype=torch.float16,
          attn_implementation="flash_attention_2",
      )

    else:
      hf_path = SUPPORTED_MODEL_DICT[model_name]
      self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
      self.model = T5ForConditionalGeneration.from_pretrained(hf_path)

  def predict(self, prompt):
    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
    generate_ids = self.model.generate(inputs.input_ids)
    return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

  def predict_from_tokens(self, input_tokens):
    generate_ids = self.model.generate(input_tokens)
    return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)


class VertexAI(InferencePlatform):
  """An implementation for using Google Cloud's Vertex AI as the platform for LLM inference."""

  def __init__(self):
    pass

  def predict(self, prompt):
    pass


def sliced_process_documents_from_index_with_hierarchy(result, sparse_vec, t):
  """Processes documents from index."""
  ctx = result["ctxs"]
  question = result["question"]

  search_results_list = []
  dd = 0
  doc_remap = {}
  for doc_ind, on_or_off in enumerate(sparse_vec):
    if on_or_off:
      doc_remap[dd] = doc_ind
      doc = ctx[doc_ind]
      search_results_list.append(
          f"Document [{str(dd+1)}] (Title: {doc['title']}) {doc['text']}\n"
      )
      dd += 1
  search_results = "".join(search_results_list)
  nop = dd

  prompt = (
      "Write a high-quality answer for the given question using only the"
      " provided search results (some of which might be"
      f" irrelevant).\n\n{search_results}\nQuestion: {question}\nAnswer:"
  )
  tokens = t.encode(prompt)
  token_strings = [t.decode(thing) for thing in tokens]

  doc_id = t.encode("Document")[1]
  doc_indices = np.argwhere(np.array(tokens) == doc_id)[:, 0]
  doc_indices = list(doc_indices)

  q_id = t.encode("Question")[1]
  q_ind = np.argwhere(np.array(tokens) == q_id)[:, 0]
  doc_indices.extend(list(q_ind))

  first_layer = {}
  for dd in range(nop):
    doc_span_tup_dd = (
        doc_indices[dd],
        doc_indices[dd + 1],
    )

    doc_dd_splits = []
    doc_dd = token_strings[doc_span_tup_dd[0] : doc_span_tup_dd[1]]
    doc_dd_toks = tokens[doc_span_tup_dd[0] : doc_span_tup_dd[1]]

    doc = ctx[doc_remap[dd]]
    prompt_end_ind = (
        len(
            t.encode(
                "Document [" + str(dd + 1) + "] (Title: " + doc["title"] + ")"
            )
        )
        - 1
    )

    doc_dd_splits.append(0)
    doc_dd_splits.append(prompt_end_ind)

    things_to_split = [
        ".",
        ",",
    ]

    for xx in range(prompt_end_ind, len(doc_dd)):
      if doc_dd[xx] in things_to_split:
        doc_dd_splits.append(xx + 1)
    last = len(doc_dd)
    if last not in doc_dd_splits:
      doc_dd_splits.append(last)

    queue = [7904]  # arbitrary token used for splitting
    toktups = []
    for xx in range(len(doc_dd_toks)):
      queue.append(doc_dd_toks[xx])
      if len(t.decode(queue).split(" ")) > 2:
        toktups.append(queue[1:-1])
        while len(queue) > 2:
          queue.pop(0)
    toktups.append(queue[1:])

    second_layer = {}
    for dd_ph in range(len(doc_dd_splits) - 1):
      phrase_span_tup_ph = (
          doc_dd_splits[dd_ph],
          doc_dd_splits[dd_ph + 1],
      )
      third_layer = {}
      word_pos = 0
      xx = 0
      for _, toktup in enumerate(toktups):
        if (
            word_pos >= phrase_span_tup_ph[0]
            and word_pos + len(toktup) <= phrase_span_tup_ph[1]
        ):
          word_tup = (
              word_pos - phrase_span_tup_ph[0],
              word_pos + len(toktup) - phrase_span_tup_ph[0],
          )
          third_layer[xx] = (None, word_tup)
          xx += 1
        word_pos += len(toktup)
      second_layer[dd_ph] = (third_layer, phrase_span_tup_ph)
    first_layer[dd] = (second_layer, doc_span_tup_dd)
  return first_layer, tokens


def document_level_interpretability_steps(
    result,
    num_documents=10,
    num_samples=10,
    doc_masking_type=None,
    doc_interp_style="DOC_BANZ_50",
):
  t = inference_platform.tokenizer
  max_new_tokens = 10
  extra_verbose = False

  list_of_perms = []
  start_time = time.time()
  new_gen_list_list_doc = []
  new_conf_list_doc = []

  if doc_masking_type == "DOC_PAD":
    first_layer, tokens = sliced_process_documents_from_index_with_hierarchy(
        result, torch.ones(num_documents), t
    )
    input_tokens = torch.Tensor(tokens).long().to(device)
    len_tokens = len(tokens)

  for s in range(num_samples):
    print("s", s)
    if doc_interp_style == "DOC_BANZ_50":
      conf = np.random.rand(num_documents) > 0.5
      list_of_perms.append(conf)
    elif doc_interp_style == "DOC_REMOVAL_100":
      conf = np.ones(num_documents, dtype=bool)
    elif doc_interp_style == "DOC_INCLUSION_0":
      conf = np.zeros(num_documents, dtype=bool)
    elif doc_interp_style == "DOC_SHAP":
      unif = np.random.rand()
      conf = np.random.rand(num_documents) > unif
    else:
      raise ValueError("doc interp method not available")
    if extra_verbose:
      print(conf)

    if doc_masking_type == "DOC_SLICE":
      sparsity_grid = torch.zeros((num_documents + 1, num_documents), dtype=int)
      for d in range(num_documents):
        if conf[d]:
          sparsity_grid[:, d] = 1
      sparsity_grid[np.arange(num_documents) + 1, np.arange(num_documents)] = (
          1
          - sparsity_grid[
              np.arange(num_documents) + 1, np.arange(num_documents)
          ]
      )

    if doc_masking_type == "DOC_PAD":
      sparsity2 = torch.zeros((num_documents + 1, len_tokens), dtype=int).to(
          device
      )
      sparsity2[:, : first_layer[0][1][0]] = 1  # prefix
      sparsity2[:, first_layer[num_documents - 1][1][1] :] = 1  # suffix
      for d in range(num_documents):
        doc_span_tup_dd = first_layer[d][1]
        if conf[d]:
          sparsity2[:, doc_span_tup_dd[0] : doc_span_tup_dd[1]] = 1
      for d in range(num_documents):
        doc_span_tup_dd = first_layer[d][1]
        sparsity2[d + 1, doc_span_tup_dd[0] : doc_span_tup_dd[1]] = (
            1 - sparsity2[d + 1, doc_span_tup_dd[0] : doc_span_tup_dd[1]]
        )

      sparsity3 = sparsity2.clone()
      prefix = first_layer[0][1][0]
      suffix = first_layer[num_documents - 1][1][1]
      sparsity3[:, prefix:suffix] = 1 - sparsity3[:, prefix:suffix]

    conf_tup_1 = (list(conf),)
    conf_tup_2 = (list(~conf),)
    new_conf_list_doc.append(conf_tup_1)
    new_conf_list_doc.append(conf_tup_2)

    new_gen_list = []
    for ss in range(num_documents + 1):
      if extra_verbose:
        print(ss)
      if doc_masking_type == "DOC_SLICE":
        tokens = sliced_process_documents_from_index(
            result, sparsity_grid[ss], t
        )
        input_ss = torch.Tensor(tokens).long().to(device)[None]
      elif doc_masking_type == "DOC_PAD":
        input_ss = input_tokens[None] * sparsity2[ss][None]
      else:
        raise ValueError("doc masking method not available")

      generate_ids = inference_platform.model.generate(
          input_ss, max_new_tokens=max_new_tokens
      )
      new_generate_ids = generate_ids[:, input_ss.shape[1] :]
      new_gen_list.append(new_generate_ids.cpu())
      if extra_verbose:
        print(
            inference_platform.tokenizer.batch_decode(
                new_generate_ids, skip_special_tokens=True
            )
        )
        print(time.time() - start_time)

      if doc_masking_type == "DOC_SLICE":
        tokens = sliced_process_documents_from_index(
            result, 1 - sparsity_grid[ss], t
        )
        input_ss = torch.Tensor(tokens).long().to(device)[None]
      elif doc_masking_type == "DOC_PAD":
        input_ss = input_tokens[None] * sparsity3[ss][None]
      else:
        raise ValueError("doc masking method not available")

      generate_ids = inference_platform.model.generate(
          input_ss, max_new_tokens=max_new_tokens
      )
      new_generate_ids = generate_ids[:, input_ss.shape[1] :]
      new_gen_list.append(new_generate_ids.cpu())
      if extra_verbose:
        print(
            inference_platform.tokenizer.batch_decode(
                new_generate_ids, skip_special_tokens=True
            )
        )
        print(time.time() - start_time)
        print()
    new_gen_list_list_doc.append(new_gen_list)
  print(time.time() - start_time)
  pass
  return new_gen_list_list_doc, new_conf_list_doc


def sentence_level_interpretability_steps(
    result,
    num_documents=10,
    num_samples=5,
    d_to_check=0,
    doc_masking_type="DOC_SLICE",
    doc_interp_style=None,
    sen_masking_type=None,
    sen_interp_style="SEN_BANZ_50",
):
  max_new_tokens = 10
  extra_verbose = False

  unchanging_context_interp_styles = ["DOC_REMOVAL_100"]

  if doc_interp_style in unchanging_context_interp_styles:
    first_layer_ss, tokens = sliced_process_documents_from_index_with_hierarchy(
        result, torch.ones(num_documents), t
    )
    len_tokens = len(tokens)
    input_tokens = torch.Tensor(tokens).long().to(device)

  list_of_perms = []

  start_time = time.time()
  new_gen_list_list_sen = []
  new_conf_list_sen = []

  for s in range(num_samples):
    print("s", s)
    if doc_interp_style == "DOC_BANZ_50":
      conf = np.random.rand(num_documents) > 0.5
      list_of_perms.append(conf)
    elif doc_interp_style == "DOC_REMOVAL_100":
      conf = np.ones(num_documents, dtype=bool)
    elif doc_interp_style == "DOC_SHAP":
      unif = np.random.rand()
      conf = np.random.rand(num_documents) > unif
    else:
      raise ValueError("doc interp method not available")
    if extra_verbose:
      print(conf)

    sparsity_grid = torch.zeros((1, num_documents), dtype=int)
    s_to_check = -1
    s_d = 0
    for d in range(num_documents):
      if d == d_to_check:
        sparsity_grid[:, d] = 1
        s_to_check = s_d
      if conf[d]:
        sparsity_grid[:, d] = 1
        s_d += 1
    num_documents_to_check = int(np.array(torch.sum(sparsity_grid)))

    if doc_interp_style not in unchanging_context_interp_styles:
      first_layer_ss, tokens = (
          sliced_process_documents_from_index_with_hierarchy(
              result, sparsity_grid[0], t
          )
      )
      len_tokens = len(tokens)
      input_tokens = torch.Tensor(tokens).long().to(device)

    second_layer = first_layer_ss[s_to_check][0]
    local_dim = len(second_layer.keys())

    sparsity2 = torch.zeros((local_dim + 1, len_tokens), dtype=int).to(device)
    sparsity2[:, : first_layer_ss[0][1][0]] = 1  # prefix
    sparsity2[:, first_layer_ss[num_documents_to_check - 1][1][1] :] = (
        1 # suffix
    )
    if doc_masking_type == "DOC_SLICE":
      for d in range(num_documents_to_check):
        doc_span_tup_dd = first_layer_ss[d][1]
        if (
            d != s_to_check
        ):  # include everything but the doc to check (already sliced)
          sparsity2[:, doc_span_tup_dd[0] : doc_span_tup_dd[1]] = 1
    elif doc_masking_type == "DOC_PAD":
      for d in range(num_documents_to_check):
        doc_span_tup_dd = first_layer_ss[d][1]
        if (
            conf[d] and d != s_to_check
        ):
          # only include certain docs (because padding),
          # make sure not to include doc to check
          sparsity2[:, doc_span_tup_dd[0] : doc_span_tup_dd[1]] = 1
    else:
      raise ValueError("doc masking method not available")

    if sen_interp_style == "SEN_BANZ_50":
      conf_ll = np.random.rand(local_dim) > 0.5
    elif sen_interp_style == "SEN_SHAP":
      unif_ll = np.random.rand()
      conf_ll = np.random.rand(local_dim) > unif_ll
    for l in range(local_dim):
      doc_span_tup_dd = first_layer_ss[s_to_check][1]
      doc_span_tup_dd_ll = second_layer[l][1]
      if conf_ll[l]:
        sparsity2[
            :,
            doc_span_tup_dd[0]
            + doc_span_tup_dd_ll[0] : doc_span_tup_dd[0]
            + doc_span_tup_dd_ll[1],
        ] = 1

    for l in range(local_dim):
      doc_span_tup_dd = first_layer_ss[s_to_check][1]
      doc_span_tup_dd_ll = second_layer[l][1]
      sparsity2[
          l + 1,
          doc_span_tup_dd[0]
          + doc_span_tup_dd_ll[0] : doc_span_tup_dd[0]
          + doc_span_tup_dd_ll[1],
      ] = (
          1
          - sparsity2[
              l + 1,
              doc_span_tup_dd[0]
              + doc_span_tup_dd_ll[0] : doc_span_tup_dd[0]
              + doc_span_tup_dd_ll[1],
          ]
      )

    sparsity3 = sparsity2.clone()
    prefix = first_layer_ss[s_to_check][1][0]
    suffix = first_layer_ss[s_to_check][1][1]
    sparsity3[:, prefix:suffix] = 1 - sparsity3[:, prefix:suffix]

    conf_tup_1 = (list(conf), list(conf_ll))
    conf_tup_2 = (list(conf), list(conf_ll))
    new_conf_list_sen.append(conf_tup_1)
    new_conf_list_sen.append(conf_tup_2)

    new_gen_list = []
    for ss in range(local_dim + 1):
      if extra_verbose:
        print(ss)

      if sen_masking_type == "SEN_SLICE":  # v5,v4
        input_ss = input_tokens[sparsity2[ss] == 1][None]
      elif sen_masking_type == "SEN_PAD":  # v3
        input_ss = input_tokens[None] * sparsity2[ss][None]
      else:
        raise ValueError("sen masking method not available")

      generate_ids = inference_platform.model.generate(
          input_ss, max_new_tokens=max_new_tokens
      )
      new_generate_ids = generate_ids[:, input_ss.shape[1] :]
      new_gen_list.append(new_generate_ids.cpu())
      if extra_verbose:
        print(
            inference_platform.tokenizer.batch_decode(
                new_generate_ids, skip_special_tokens=True
            )
        )
        print(time.time() - start_time)

      if sen_masking_type == "SEN_SLICE":  # v5,v4
        input_ss = input_tokens[sparsity3[ss] == 1][None]
      elif sen_masking_type == "SEN_PAD":  # v3
        input_ss = input_tokens[None] * sparsity3[ss][None]
      else:
        raise ValueError("sen masking method not available")

      generate_ids = inference_platform.model.generate(
          input_ss, max_new_tokens=max_new_tokens
      )
      new_generate_ids = generate_ids[:, input_ss.shape[1] :]
      new_gen_list.append(new_generate_ids.cpu())
      if extra_verbose:
        print(
            inference_platform.tokenizer.batch_decode(
                new_generate_ids, skip_special_tokens=True
            )
        )
        print(time.time() - start_time)
        print()
    new_gen_list_list_sen.append(new_gen_list)
  print(time.time() - start_time)
  pass
  return new_gen_list_list_sen, new_conf_list_sen


def word_level_interpretability_steps(
    result,
    num_documents=10,
    num_samples=5,
    d_to_check=0,
    sen_to_check=0,
    doc_masking_type="DOC_SLICE",
    doc_interp_style=None,
    sen_masking_type=None,
    sen_interp_style="SEN_BANZ_50"
):
  max_new_tokens = 10
  extra_verbose = False

  unchanging_context_interp_styles = ["DOC_REMOVAL_100"]

  if doc_interp_style in unchanging_context_interp_styles:
    first_layer_ss, tokens = sliced_process_documents_from_index_with_hierarchy(
        result, torch.ones(num_documents), t
    )
    len_tokens = len(tokens)
    input_tokens = torch.Tensor(tokens).long().to(device)

  list_of_perms = []

  start_time = time.time()
  new_gen_list_list_wor = []
  new_conf_list_wor = []

  for s in range(num_samples):
    print("s", s)
    if doc_interp_style == "DOC_BANZ_50":
      conf = np.random.rand(num_documents) > 0.5
      list_of_perms.append(conf)
    elif doc_interp_style == "DOC_REMOVAL_100":
      conf = np.ones(num_documents, dtype=bool)
    elif doc_interp_style == "DOC_SHAP":
      unif = np.random.rand()
      conf = np.random.rand(num_documents) > unif
    else:
      raise ValueError("doc interp method not available")
    if extra_verbose:
      print(conf)

    sparsity_grid = torch.zeros((1, num_documents), dtype=int)
    s_to_check = -1
    s_d = 0
    for d in range(num_documents):
      if d == d_to_check:
        sparsity_grid[:, d] = 1
        s_to_check = s_d
      if conf[d]:
        sparsity_grid[:, d] = 1
        s_d += 1
    num_documents_to_check = int(np.array(torch.sum(sparsity_grid)))

    if doc_interp_style not in unchanging_context_interp_styles:
      first_layer_ss, tokens = (
          sliced_process_documents_from_index_with_hierarchy(
              result, sparsity_grid[0], t
          )
      )
      len_tokens = len(tokens)
      input_tokens = torch.Tensor(tokens).long().to(device)

    second_layer_ss = first_layer_ss[s_to_check][0]
    local_dim = len(second_layer_ss.keys())
    third_layer_ss = second_layer_ss[sen_to_check][0]
    word_local_dim = len(third_layer_ss.keys())
    if extra_verbose:
      print("third_layer_ss", third_layer_ss)
      print("local_dim", local_dim)
      print("word_local_dim", word_local_dim)

    sparsity2 = torch.zeros((word_local_dim + 1, len_tokens), dtype=int).to(device)
    sparsity2[:, : first_layer_ss[0][1][0]] = 1  # prefix
    sparsity2[:, first_layer_ss[num_documents_to_check - 1][1][1] :] = 1  # suffix
    if doc_masking_type == "DOC_SLICE":
      for d in range(num_documents_to_check):
        doc_span_tup_dd = first_layer_ss[d][1]
        if (
            d != s_to_check
        ):  # include everything but the doc to check (already sliced)
          sparsity2[:, doc_span_tup_dd[0] : doc_span_tup_dd[1]] = 1
    elif doc_masking_type == "DOC_PAD":
      for d in range(num_documents_to_check):
        doc_span_tup_dd = first_layer_ss[d][1]
        if (
            conf[d] and d != s_to_check
        ):
          # only include certain docs (because padding),
          # make sure not to include doc to check
          sparsity2[:, doc_span_tup_dd[0] : doc_span_tup_dd[1]] = 1
    else:
      raise ValueError("doc masking method not available")

    if sen_interp_style == "SEN_BANZ_50":
      conf_ll = np.random.rand(local_dim) > 0.5
    elif sen_interp_style == "SEN_SHAP":
      unif_ll = np.random.rand()
      conf_ll = np.random.rand(local_dim) > unif_ll
    for l in range(local_dim):
      doc_span_tup_dd = first_layer_ss[s_to_check][1]
      doc_span_tup_dd_ll = second_layer_ss[l][1]
      if conf_ll[l] and l != sen_to_check:
        sparsity2[
            :,
            doc_span_tup_dd[0]
            + doc_span_tup_dd_ll[0] : doc_span_tup_dd[0]
            + doc_span_tup_dd_ll[1],
        ] = 1

    if sen_interp_style == "SEN_BANZ_50":
      conf_ww = np.random.rand(word_local_dim) > 0.5
    elif sen_interp_style == "SEN_SHAP":
      conf_ww = np.random.rand(word_local_dim) > unif_ll
    for w in range(word_local_dim):
      third_layer_ss = second_layer_ss[sen_to_check][0]
      doc_span_tup_dd = first_layer_ss[s_to_check][1]
      doc_span_tup_dd_ll = second_layer_ss[sen_to_check][1]
      doc_span_tup_dd_ll_ww = third_layer_ss[w][1]
      if conf_ww[w]:
        # why did I do it like this...
        sparsity2[
            :,
            doc_span_tup_dd[0]
            + doc_span_tup_dd_ll[0]
            + doc_span_tup_dd_ll_ww[0] : doc_span_tup_dd[0]
            + doc_span_tup_dd_ll[0]
            + doc_span_tup_dd_ll_ww[1],
        ] = 1

    for w in range(word_local_dim):
      third_layer_ss = second_layer_ss[sen_to_check][0]
      doc_span_tup_dd = first_layer_ss[s_to_check][1]
      doc_span_tup_dd_ll = second_layer_ss[sen_to_check][1]
      doc_span_tup_dd_ll_ww = third_layer_ss[w][1]
      sparsity2[
          w + 1,
          doc_span_tup_dd[0]
          + doc_span_tup_dd_ll[0]
          + doc_span_tup_dd_ll_ww[0] : doc_span_tup_dd[0]
          + doc_span_tup_dd_ll[0]
          + doc_span_tup_dd_ll_ww[1],
      ] = (
          1
          - sparsity2[
              w + 1,
              doc_span_tup_dd[0]
              + doc_span_tup_dd_ll[0]
              + doc_span_tup_dd_ll_ww[0] : doc_span_tup_dd[0]
              + doc_span_tup_dd_ll[0]
              + doc_span_tup_dd_ll_ww[1],
          ]
      )

    sparsity3 = sparsity2.clone()
    prefix = (
        first_layer_ss[s_to_check][1][0] + second_layer_ss[sen_to_check][1][0]
    )
    suffix = (
        first_layer_ss[s_to_check][1][0] + second_layer_ss[sen_to_check][1][1]
    )
    sparsity3[:, prefix:suffix] = 1 - sparsity3[:, prefix:suffix]

    conf_tup_1 = (list(conf), list(conf_ll), list(conf_ww))
    conf_tup_2 = (list(conf), list(conf_ll), list(~conf_ww))
    new_conf_list_wor.append(conf_tup_1)
    new_conf_list_wor.append(conf_tup_2)

    new_gen_list = []
    for ss in range(word_local_dim + 1):
      if extra_verbose:
        print(ss)

      if sen_masking_type == "SEN_SLICE":  # v5,v4
        input_ss = input_tokens[sparsity2[ss] == 1][None]
      elif sen_masking_type == "SEN_PAD":  # v3
        input_ss = input_tokens[None] * sparsity2[ss][None]
      else:
        raise ValueError("sen masking method not available")

      generate_ids = inference_platform.model.generate(
          input_ss, max_new_tokens=max_new_tokens
      )
      new_generate_ids = generate_ids[:, input_ss.shape[1] :]
      new_gen_list.append(new_generate_ids.cpu())
      if extra_verbose:
        print(
            inference_platform.tokenizer.batch_decode(
                new_generate_ids, skip_special_tokens=True
            )
        )
        print(time.time() - start_time)

      if sen_masking_type == "SEN_SLICE":
        input_ss = input_tokens[sparsity3[ss] == 1][None]
      elif sen_masking_type == "SEN_PAD":
        input_ss = input_tokens[None] * sparsity3[ss][None]
      else:
        raise ValueError("sen masking method not available")

      generate_ids = inference_platform.model.generate(
          input_ss, max_new_tokens=max_new_tokens
      )
      new_generate_ids = generate_ids[:, input_ss.shape[1] :]
      new_gen_list.append(new_generate_ids.cpu())
      if extra_verbose:
        print(
            inference_platform.tokenizer.batch_decode(
                new_generate_ids, skip_special_tokens=True
            )
        )
        print(time.time() - start_time)
        print()
    new_gen_list_list_wor.append(new_gen_list)
  print(time.time() - start_time)
  pass
  return new_gen_list_list_wor, new_conf_list_wor


def convert_conjugate_to_sunflower(
    new_gen_list_list, new_conf_list=None, root=None
):
  num_samples = len(new_gen_list_list)
  d_loc = len(new_gen_list_list[0]) // 2 - 1

  if new_conf_list is None:
    new_conf_list = [None] * num_samples

  sunflower_list = []
  for ss in range(num_samples):
    pass

    new_gen_list = new_gen_list_list[ss]

    for parity in range(2):  # both conjugate pairs
      sunflower = {}
      sunflower["size"] = d_loc
      sunflower["conferences"] = new_conf_list[ss]
      for d in range(-1, d_loc):
        batch_ind = 0
        out_tok_list = list(
            new_gen_list[d * 2 + 2 + parity].cpu().numpy()[batch_ind]
        )
        sunflower[d] = out_tok_list
      sunflower_list.append(sunflower)

  sunflower_object = {
      "root": root,
      "d_loc": d_loc,
      "s_loc": 2 * num_samples,
      "sunflower_list": sunflower_list,
  }
  return sunflower_object


def scores_from_sunflower(sunflower_obj):
  d_loc = sunflower_obj["d_loc"]
  s_loc = sunflower_obj["s_loc"]

  scores_arr = np.zeros((d_loc, 4), dtype=int)

  for ss in range(s_loc):
    sunflower = sunflower_obj["sunflower_list"][ss]
    base = sunflower[-1]
    for dl in range(d_loc):
      new = sunflower[dl]

      if base != new:  # match full list
        scores_arr[dl, 0] += 1

      if base[0] != new[0]:  # match first token
        scores_arr[dl, 1] += 1

  return scores_arr


def accumulate_potential_outputs(sunflower_collection, root_list=None):
  if root_list is None:
    root_list = list(sunflower_collection.keys())

  list_of_tok_lists = []
  dict_of_tok_lists = {}
  for root in root_list:
    sunflower = sunflower_collection[root]

    for sample in sunflower["sunflower_list"]:
      d_loc = sample["size"]
      for d in range(-1, d_loc):
        tok_list = sample[d]
        if tok_list not in list_of_tok_lists:
          list_of_tok_lists.append(tok_list)
          dict_of_tok_lists[tuple(tok_list)] = 1
        else:
          dict_of_tok_lists[tuple(tok_list)] += 1

  return list_of_tok_lists, dict_of_tok_lists


def accumulate_scores_at_specified_root(
    sunflower_collection, root, list_of_tok_lists
):
  sunflower_obj = sunflower_collection[root]
  len_token_lists = len(list_of_tok_lists)
  d_loc = sunflower_obj["d_loc"]
  s_loc = sunflower_obj["s_loc"]
  print("d_loc", d_loc)
  print("s_loc", s_loc)

  scores_tens = np.zeros((d_loc, len_token_lists, 4), dtype=int)
  for ss in range(s_loc):
    sunflower = sunflower_obj["sunflower_list"][ss]
    conf = sunflower["conferences"][-1]

    base = copy.copy(sunflower[-1])
    for d in range(d_loc):
      new = copy.copy(sunflower[d])
      if base != new:  # match full list
        new_ind = list_of_tok_lists.index(new)
        if conf[d]:
          new_ind = list_of_tok_lists.index(base)
        scores_tens[d, new_ind, 0] += 1

      if base[0] != new[0]:  # match first token
        new_ind = list_of_tok_lists.index(new)
        if conf[d]:
          new_ind = list_of_tok_lists.index(base)
        scores_tens[d, new_ind, 1] += 1

  return scores_tens.astype(float) / s_loc


def get_first_token_representative(list_of_tok_lists):
  repn_list_of_tok_lists_first = []
  repn_list_of_tok_lists = []
  remap_dict = {}

  for tt, tok_list in enumerate(list_of_tok_lists):
    tok = tok_list[0]
    if tok in repn_list_of_tok_lists_first:
      remap_dict[tt] = repn_list_of_tok_lists_first.index(tok)
    else:
      repn_list_of_tok_lists_first.append(tok)
      repn_list_of_tok_lists.append(tok_list)
      remap_dict[tt] = repn_list_of_tok_lists_first.index(tok)

  len_token_lists = len(list_of_tok_lists)
  len_repn_token_lists = len(repn_list_of_tok_lists)
  remap_array = np.zeros((len_token_lists, len_repn_token_lists), dtype=int)
  for tt in range(len_token_lists):
    remap_array[tt, remap_dict[tt]] = 1

  return (
      repn_list_of_tok_lists_first,
      repn_list_of_tok_lists,
      remap_dict,
      remap_array,
  )


def convert_sunflower_collection_to_visualizable_json(
    sunflower_collection, first_layer, question, score_index=0
):
  list_of_tok_lists, dict_of_tok_lists = accumulate_potential_outputs(
      sunflower_collection
  )
  inds = np.argsort(-np.array(list(dict_of_tok_lists.values())))
  sorted_list_of_out_strings = [t.decode(list_of_tok_lists[i]) for i in inds]
  sorted_list_of_tok_lists = [(list_of_tok_lists[i]) for i in inds]
  print(
      "sorted_list_of_out_strings",
  )
  print(sorted_list_of_out_strings)

  (
      _,
      repn_list_of_tok_lists,
      _,
      remap_array,
  ) = get_first_token_representative(sorted_list_of_tok_lists)
  repn_sorted_list_of_out_strings = [
      t.decode(thing) for thing in repn_list_of_tok_lists
  ]
  print(
      "repn_sorted_list_of_out_strings",
  )
  print(repn_sorted_list_of_out_strings)

  full_json = {
      "children": [],
      "question": question,
  }
  if USING_REPN_STRINGS:
    full_json["text_outputs"] = repn_sorted_list_of_out_strings
  else:
    full_json["text_outputs"] = sorted_list_of_out_strings

  len_first_layer = len(first_layer.keys())

  scores_tens_doc = accumulate_scores_at_specified_root(
      sunflower_collection, (), sorted_list_of_tok_lists
  )
  if USING_REPN_STRINGS:
    scores_tens_doc = np.tensordot(scores_tens_doc, remap_array, axes=(1, 0))

  for d_0 in range(len_first_layer):
    doc_span_tup_dd = first_layer[d_0][1]
    text_d0 = t.decode(tokens[doc_span_tup_dd[0] : doc_span_tup_dd[1]])

    scores_d0 = scores_tens_doc[d_0, score_index, :]

    doc_dict = {
        "index1": d_0,
        "text": text_d0,
        "children": [],
        "layer1_tspans": [],
        "scores": str(list(scores_d0)),
    }
    full_json["children"].append(doc_dict)

    root_0 = (d_0,)
    if root_0 in sunflower_collection:
      second_layer = first_layer[d_0][0]
      pass

      scores_tens_sen = accumulate_scores_at_specified_root(
          sunflower_collection, root_0, sorted_list_of_tok_lists
      )
      if USING_REPN_STRINGS:
        scores_tens_sen = np.tensordot(
            scores_tens_sen, remap_array, axes=(1, 0)
        )
      len_second_layer = len(second_layer.keys())
      for d_1 in range(len_second_layer):
        # print("\t\t",d_1)
        doc_span_tup_dd_ll = second_layer[d_1][1]
        text_d1 = t.decode(
            tokens[
                doc_span_tup_dd[0]
                + doc_span_tup_dd_ll[0] : doc_span_tup_dd[0]
                + doc_span_tup_dd_ll[1]
            ]
        )

        scores_d1 = scores_tens_sen[d_1, score_index, :]

        sen_dict = {
            "index1": d_0,
            "index2": d_1,
            "text": text_d1,
            "children": [],
            "scores": str(list(scores_d1)),
        }
        full_json["children"][d_0]["children"].append(sen_dict)

        root_1 = (d_0, d_1)
        if root_1 in sunflower_collection:
          third_layer = second_layer[d_1][0]
          pass

          scores_tens_wor = accumulate_scores_at_specified_root(
              sunflower_collection, root_1, sorted_list_of_tok_lists
          )
          if USING_REPN_STRINGS:
            scores_tens_wor = np.tensordot(
                scores_tens_wor, remap_array, axes=(1, 0)
            )
          D_2 = len(third_layer.keys())
          for d_2 in range(D_2):
            doc_span_tup_dd_ll_ww = third_layer[d_2][1]
            text_d2 = t.decode(
                tokens[
                    doc_span_tup_dd[0]
                    + doc_span_tup_dd_ll[0]
                    + doc_span_tup_dd_ll_ww[0] : doc_span_tup_dd[0]
                    + doc_span_tup_dd_ll[0]
                    + doc_span_tup_dd_ll_ww[1]
                ]
            )

            scores_d2 = scores_tens_wor[d_2, score_index, :]

            wor_dict = {
                "index1": d_0,
                "index2": d_1,
                "index3": d_2,
                "text": text_d2,
                "children": [],
                "scores": str(list(scores_d2)),
            }
            full_json["children"][d_0]["children"][d_1]["children"].append(
                wor_dict
            )

  pass
  return full_json


def sliced_process_documents_from_index(result, sparse_vec, t):
  """Processes documents from index."""
  ctx = result["ctxs"]
  question = result["question"]

  search_results_list = []
  dd = 0
  for doc_ind, on_or_off in enumerate(sparse_vec):
    if on_or_off:
      doc = ctx[doc_ind]
      search_results_list.append(
          f"Document [{str(dd+1)}] (Title: {doc['title']}) {doc['text']}\n"
      )
      dd += 1
  search_results = "".join(search_results_list)

  prompt = (
      "Write a high-quality answer for the given question using only the"
      " provided search results (some of which might be"
      f" irrelevant).\n\n{search_results}\nQuestion: {question}\nAnswer:"
  )
  tokens = t.encode(prompt)
  return tokens


HF_TOKEN = os.environ["HF_TOKEN"]
inference_platform = HuggingFace()
inference_platform.authenticate(HF_TOKEN)
inference_platform.setup_model("gemma2b")

path_for_query_and_documents = "/app/lost-in-the-middle/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl"

with open(path_for_query_and_documents, "r") as json_file:
  json_list = list(json_file)

interp_session_id = str(uuid.uuid4()).replace("-", "_")
print(interp_session_id)
print()

query_id = 0  # index of query within the file
query_id = 18
json_str = json_list[query_id]
result = json.loads(json_str)

DOC_MASK = "DOC_SLICE"  # "DOC_SLICE" or "DOC_PAD"
SEN_MASK = "SEN_SLICE"  # "SEN_SLICE" or "SEN_PAD"

DOC_INTERP = "DOC_BANZ_50"
DOC_INTERP = "DOC_SHAP"
SEN_INTERP = "SEN_BANZ_50"
SEN_INTERP = "SEN_SHAP"

# DOC_SAMPLES = 10; SEN_SAMPLES = 5; WOR_SAMPLES = 5;
DOC_SAMPLES = 10
SEN_SAMPLES = 10
WOR_SAMPLES = 10
TOP_SENT = 3
TOP_WORD = 2

USING_REPN_STRINGS = True

ACCUMULATE_DIFF = "different_full_generation"  # different anywhere

ctx = result["ctxs"]
question = result["question"]
answers = result["answers"]
print("question", question)
print("answers", answers)

num_documents = 10
t = inference_platform.tokenizer
first_layer, tokens = sliced_process_documents_from_index_with_hierarchy(
    result, torch.ones(num_documents), t
)
print(t.decode(tokens))
JSON_PATH = (
    "/app/" + "textgenshap-data-json/" + interp_session_id + "_data.json"
)

score_index = 0
if ACCUMULATE_DIFF == "different_first_token":
  score_index = 1


def save_json_constructor():
  def save_json_now(sunflower_collection):

    full_json = convert_sunflower_collection_to_visualizable_json(
        sunflower_collection, first_layer, question, score_index
    )

    print(full_json["text_outputs"])
    print("full_json")
    print(full_json)
    with open(
        JSON_PATH, "w+", encoding="utf-8"
    ) as f:
      json.dump(full_json, f, indent=4)

  return save_json_now


save_json_now = save_json_constructor()


sunflower_collection = {}

new_gen_list_list_doc_1, conf_list_doc = document_level_interpretability_steps(
    result,
    num_samples=DOC_SAMPLES,
    doc_masking_type=DOC_MASK,
    doc_interp_style=DOC_INTERP,
)
root = ()
sunflower_obj_doc = convert_conjugate_to_sunflower(
    new_gen_list_list_doc_1, conf_list_doc, root=root
)
sunflower_collection[root] = sunflower_obj_doc
save_json_now(sunflower_collection)

# START THE DFS
scores_arr = scores_from_sunflower(sunflower_obj_doc)
num_documents = 10
for d in range(num_documents):
  score_allToks = scores_arr[d, 0]
  score_firstTok = scores_arr[d, 1]
  print("d", d, "\t", score_firstTok, "  \t", score_allToks)
print()
print()

list_of_tok_lists, dict_of_tok_lists = accumulate_potential_outputs(
    sunflower_collection
)
inds = np.argsort(-np.array(list(dict_of_tok_lists.values())))
sorted_list_of_out_strings = [t.decode(list_of_tok_lists[i]) for i in inds]
sorted_list_of_tok_lists = [(list_of_tok_lists[i]) for i in inds]
scores_tens_doc = accumulate_scores_at_specified_root(
    sunflower_collection, (), sorted_list_of_tok_lists
)
scores_arr = np.sum(scores_tens_doc, axis=1)
for d in range(num_documents):
  score_allToks = scores_arr[d, 0]
  score_firstTok = scores_arr[d, 1]
  print("d", d, "\t", score_firstTok, "  \t", score_allToks)
print()
print()

queue = np.argsort(-scores_arr[:, 1])[:TOP_SENT]
print("queue", queue)
queue = list(queue)

while len(queue) > 0:
  next_doc = queue.pop(0)
  print("NEXT_DOC", next_doc)

  new_gen_list_list_sen, new_conf_list_sen = (
      sentence_level_interpretability_steps(
          result,
          num_samples=SEN_SAMPLES,
          d_to_check=next_doc,
          doc_masking_type=DOC_MASK,
          doc_interp_style=DOC_INTERP,
          sen_masking_type=SEN_MASK,
          sen_interp_style=SEN_INTERP,
      )
  )

  root = (next_doc,)
  sunflower_obj_sen = convert_conjugate_to_sunflower(
      new_gen_list_list_sen, new_conf_list_sen, root=root
  )
  sunflower_collection[root] = sunflower_obj_sen  # (or extend existing list)
  save_json_now(sunflower_collection)

  scores_arr_sen = scores_from_sunflower(sunflower_obj_sen)
  sen_queue = np.argsort(-scores_arr_sen[:, 1])[:TOP_WORD]
  print("sen_queue", sen_queue)
  sen_queue = list(sen_queue)

  while len(sen_queue) > 0:
    next_sen = sen_queue.pop(0)
    print("NEXT_SEN", next_sen)

    new_gen_list_list_wor, new_conf_list_wor = (
        word_level_interpretability_steps(
            result,
            num_samples=WOR_SAMPLES,
            d_to_check=next_doc,
            sen_to_check=next_sen,
            doc_masking_type=DOC_MASK,
            doc_interp_style=DOC_INTERP,
            sen_masking_type=SEN_MASK,
            sen_interp_style=SEN_INTERP,
        )
    )

    pass
    root = (next_doc, next_sen)
    sunflower_obj_wor = convert_conjugate_to_sunflower(
        new_gen_list_list_wor, new_conf_list_wor, root=root
    )
    sunflower_collection[root] = sunflower_obj_wor
    save_json_now(sunflower_collection)

for root in sunflower_collection:
  print("root", root)
  print(sunflower_collection[root])
  print()
