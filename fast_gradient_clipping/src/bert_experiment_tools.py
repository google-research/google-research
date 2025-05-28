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

"""BERT model training experiment functions."""

import collections
import sys
from fast_gradient_clipping.src import model_generators
from fast_gradient_clipping.src import profiling_tools


def dp_bert_model_runner(params, runtimes, peak_memories, repeats=1):
  """Runs two small BERT models and gets the compute profiles."""
  batch_size, vocab_size, query_size, num_epochs, num_steps = params
  print(
      f"vocab_size = {vocab_size}, batch_size = {batch_size}, "
      f"query_size = {query_size}, num_epochs = {num_epochs}, "
      f"num_steps = {num_steps}"
  )
  sys.stdout.flush()
  # Naive DP model.
  print("Running naive DP...")
  naive_dp_model = model_generators.make_dp_bert_model(vocab_size, False)
  cur_time, cur_mem = profiling_tools.get_train_bert_model_compute_profile(
      naive_dp_model,
      batch_size,
      vocab_size,
      query_size,
      num_epochs,
      num_steps,
      repeats,
  )
  runtimes["naive_dp_model"].append(cur_time)
  peak_memories["naive_dp_model"].append(cur_mem)
  # Fast DP model.
  print("Running fast DP...")
  fast_dp_model = model_generators.make_dp_bert_model(vocab_size, True)
  cur_time, cur_mem = profiling_tools.get_train_bert_model_compute_profile(
      fast_dp_model,
      batch_size,
      vocab_size,
      query_size,
      num_epochs,
      num_steps,
      repeats,
  )
  runtimes["fast_dp_model"].append(cur_time)
  peak_memories["fast_dp_model"].append(cur_mem)


def get_dp_bert_model_compute_profile(params, repeats=1):
  runtimes = collections.defaultdict(list)
  peak_memories = collections.defaultdict(list)
  for p in params:
    dp_bert_model_runner(p, runtimes, peak_memories, repeats)
  return runtimes, peak_memories
