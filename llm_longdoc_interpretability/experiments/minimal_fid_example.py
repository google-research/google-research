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

"""Runs a minimal FiD interpretability experiment."""

from collections.abc import Sequence
import json
import time
from absl import app
from datasets_and_prompts import get_prompt_list_from_doc
from datasets_and_prompts import return_shifts_from_prompt_list_v2
from interp_and_visualization import plot_big_dict
from t5_with_flash.modeling_t5_with_flash import fid_adjust_existing_model
from t5_with_flash.modeling_t5_with_flash import return_pretrained_model
import torch


def fid_prompt_preparation(
    prompt_style, ctx, question, model4, t, device, number_of_passages
):
  """Prepares the input_ids for the FiD prompt given the queries and context.

  Args:
    prompt_style: FiD_style prompt is required
    ctx: context documents
    question: query string
    model4: FiD T5 model to be preprepared (with sparsity matrix)
    t: tokenizer
    device: device of model4
    number_of_passages: maximum number of passages considered by FiD model

  Returns:
    the input ids to be passed to generation function
  """
  nop = number_of_passages
  _, prompt_list = get_prompt_list_from_doc(prompt_style, ctx, question)
  inp_ids = t(
      prompt_list,
      return_tensors="pt",
      padding=True,
      return_attention_mask=True,
  ).input_ids

  max_in_size = 256
  inp_ids = torch.nn.functional.pad(
      inp_ids, (0, max_in_size - inp_ids.shape[-1])
  )
  inp_ids = inp_ids[:, :max_in_size]
  inp_ids = inp_ids.to(device)
  inp_ids = inp_ids.reshape(1, -1)
  # assuming max_in = 256 and block = 128
  sparsity = torch.zeros((2 * nop, 2 * nop))
  for pp in range(nop):  # block diagonal sparsity
    sparsity[2 * pp : 2 * pp + 2, 2 * pp : 2 * pp + 2] = 1
  sparsity = sparsity.to(device)
  bias_shift_np, _, local_lengths_np = return_shifts_from_prompt_list_v2(
      prompt_list, 128
  )
  fid_adjust_existing_model(
      model4, device, sparsity, bias_shift_np, local_lengths_np, nop
  )

  inp_ids = inp_ids[:, : nop * max_in_size]
  return inp_ids


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  model_name = "FiD-t5-large"
  # number of passages to check "dimension" or "documents"
  number_of_passages = 10
  interpretability_mode = {
      "mode": "shapley",
      "D": number_of_passages,
      "P": 100,
      "DBS": 100,
  }
  max_output_length = 20

  path_for_query_and_documents = (
      "contriever_msmarco_nq/nq-open-oracle-top1000.jsonl"
  )
  query_id = 2  # index of query within the file

  with open(path_for_query_and_documents, "r") as json_file:
    json_list = list(json_file)
  json_str = json_list[query_id]

  device = torch.device("cuda:0")
  model4, t = return_pretrained_model(model_name)
  start_load = time.time()
  model4 = model4.to(device)
  print("model loaded onto GPU in ", time.time() - start_load, "seconds")

  # LOAD QUERY AND DOCS FROM JSON STRING
  result = json.loads(json_str)
  ctx = result["ctxs"]
  question = result["question"]
  answers = result["answers"]
  print("question", question)
  print("answers", answers)

  # PREPARATION OF THE FID-STYLE PROMPT
  prompt_style = "FiD_sparse"
  inp_ids = fid_prompt_preparation(
      prompt_style, ctx, question, model4, t, device, number_of_passages
  )

  # GENERATE THE INTERPRETATION, by calling the 'FiD_generate_interpret' fn
  interp_start_time = time.time()
  print("generating interpretations")
  outputs = model4.jam_FiD_enc_interpretation(
      inp_ids,
      max_length=max_output_length,
      output_scores=True,
      return_dict_in_generate=True,
      number_of_out_decodings=number_of_passages,
      tokenizer=t,
      interpretability_mode=interpretability_mode,
  )
  lil_dict, big_dict, list_of_perms = outputs
  print(
      "interp_time_taken",
      time.time() - interp_start_time,
      "  for",
      len(list_of_perms),
      "samples",
  )

  # PLOT THE VISUALIZATION, in matplotlib and print the top given answers
  plot_big_dict(big_dict, lil_dict, verbose=True, plotting=True)


if __name__ == "__main__":
  app.run(main)
