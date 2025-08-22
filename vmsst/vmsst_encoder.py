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

"""VMSST encoder."""

import torch
import tqdm
import transformers


class VMSSTEncoder:
  """VMSST encoder."""

  def __init__(
      self, device="cuda", max_batch_size=32, max_length=512, cache_dir=None
  ):
    self.max_batch_size = max_batch_size
    self.max_length = max_length
    if device == "cuda":
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = "cpu"

    self.tokenizer = transformers.T5Tokenizer.from_pretrained(
        "google/mt5-large", cache_dir=cache_dir
    )
    self.model = transformers.AutoModel.from_pretrained(
        "jwieting/vmsst", trust_remote_code=True
    )

    self.model.to(self.device)
    self.model.eval()

  def encode(self, inputs, verbose=False, return_input_ids=False):
    """Function to encode VMSST inputs."""
    all_embeddings = []
    all_input_ids = []
    for i in tqdm.tqdm(
        range(0, len(inputs), self.max_batch_size),
        total=(len(inputs) // self.max_batch_size) + 1,
        disable=not verbose,
        desc="Encoding inputs:",
    ):
      tokenized_inputs = self.tokenizer(
          inputs[i : i + self.max_batch_size], return_tensors="pt", padding=True
      )

      for k, v in tokenized_inputs.items():
        tokenized_inputs[k] = v[:, :self.max_length]
      tokenized_inputs = tokenized_inputs.to(self.device)

      with torch.inference_mode():
        batch_embeddings = self.model(**tokenized_inputs)
      all_embeddings.append(batch_embeddings)

      if return_input_ids:
        all_input_ids.extend(tokenized_inputs.input_ids.cpu().tolist())

    return {
        "embeddings": torch.cat(all_embeddings, dim=0).detach().cpu().numpy(),
        "input_ids": all_input_ids,
    }
