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

"""Utility methods used to extract prompts from the supported datasets."""

import numpy as np


def get_prompt_list_from_doc(
    prompt_style, ctx, question, max_num_of_passages=None
):
  """Gets list of prompts based on the prompt style provided."""
  prompt = ""
  prompt_list = []

  if prompt_style in ["default", "FiD_sparse"]:
    prompt_list = []
    for _, doc in enumerate(ctx):
      prompt_dd = (
          f"question: {question} title: {doc['title']} context: {doc['text']}"
      )
      prompt_list.append(prompt_dd)

  if prompt_style == "LitM_prompt_connected_distilled":
    prompt_list = []
    prompt = ""
    prompt_header = (
        "Write a high-quality answer for the given question using only the"
        " provided search results (some of which might be irrelevant).\n"
    )
    prompt += prompt_header
    prompt_list.append(prompt_header)

    for dd, doc in enumerate(ctx):
      if dd < max_num_of_passages:
        prompt_dd = (
            f"Document [{str(dd+1)}] (Title: {doc['title']}) {doc['text']}\n"
        )
        prompt += prompt_dd
        prompt_list.append(prompt_dd)

    prompt_footer = f"Question: {question}\nAnswer:"
    prompt += prompt_footer
    prompt_list.append(prompt_footer)

  if prompt_style == "LitM_prompt_connected_distilled_singleDoc":
    prompt_list = []
    prompt = ""
    prompt_header = (
        "Write a high-quality answer for the given question using only the"
        " provided search results (some of which might be irrelevant).\n"
    )
    prompt += prompt_header
    prompt_list.append(prompt_header)
    dd = max_num_of_passages - 1
    doc = ctx[dd]
    prompt_dd = f"Document [1] (Title: {doc['title']}) {doc['text']}\n"
    prompt += prompt_dd
    prompt_list.append(prompt_dd)

    prompt_footer = f"Question: {question}\nAnswer:"
    prompt += prompt_footer
    prompt_list.append(prompt_footer)

  return prompt, prompt_list


# note: this code is only designed right now for hard-coded block size
#      combinations of 256 tokens per passage, and 128 per flash attn.
# Here, we are reformatting the FiD prompts to better fit into the Flash
# attention function.  For instance, an FiD prompt may only be 170 tokens
# long, but we still want to format it into two 128x128 blocks for efficient
# computation using Flash Attention.  Accordingly, we have to pad the other
# 256 - 170 = 86 tokens with "white space".  The consequences of this are
# that we need to do a lot of work with carrying around the position biases
# at the total 'real lengths' at each current flash attention block.
# e.g. if the first two prompts are sizes [170, 210], then we need to have
# 86 tokens and 46 tokens of white space.  The block lengths at those points
# will be 256 and 512; however, the real lengths will be 170 and 380.
def return_shifts_from_prompt_list_v2(prompt_list, t, block_size=128):
  """Reformat the FiD prompts to fit in Flash Attention."""
  prompt_toks = t(prompt_list)["input_ids"]

  # prompt part lengths
  ppls = np.array([len(prompt_tok) for prompt_tok in prompt_toks])

  l = len(prompt_list)
  split_cumsum_np = np.zeros(2 * l + 1, dtype=int)
  local_lengths_np = np.zeros(2 * l, dtype=int)

  ppls[ppls > 2 * block_size] = 2 * block_size

  split_cumsum_np[1::2] = ppls
  split_cumsum_np[2::2] = 0
  split_cumsum_np[1::2][ppls > block_size] = block_size
  split_cumsum_np[2::2][ppls > block_size] = (
      ppls[ppls > block_size] - block_size
  )
  split_cumsum_np = np.cumsum(split_cumsum_np)[:-1]

  local_lengths_np[::2] = ppls
  local_lengths_np[1::2] = ppls
  local_lengths_np[::2][ppls > block_size] = block_size
  local_lengths_np[1::2][ppls > block_size] = ppls[ppls > block_size]
  local_lengths_np += np.repeat(np.arange(0, l) * block_size * 2, 2)

  block_part_lengths_v2 = (
      np.arange(2 * l) * block_size
  )  # works for now (256,128)

  split_difference = block_part_lengths_v2 - split_cumsum_np
  split_difference = np.array(split_difference)
  split_diff_matrix = split_difference[:, None] - split_difference[None, :]

  return split_diff_matrix, None, local_lengths_np
