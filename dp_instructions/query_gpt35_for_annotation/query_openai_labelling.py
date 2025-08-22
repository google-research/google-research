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

"""Query openai for annotation."""

import argparse
import functools
import multiprocessing
import os
import pickle
import signal
import time

import numpy as np
import openai
from openai import OpenAI

client = OpenAI(api_key='<Use Your OWN Key>')


def timeout(sec):
  """Timeout decorator that raises TimeoutError after given seconds."""

  def decorator(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):  # pylint: disable=redefined-outer-name

      def _handle_timeout(signum, frame):  # pylint: disable=unused-argument
        err_msg = f'Function {func.__name__} timed out after {sec} seconds'
        raise TimeoutError(err_msg)

      signal.signal(signal.SIGALRM, _handle_timeout)
      signal.alarm(sec)
      try:
        result = func(*args, **kwargs)
      finally:
        signal.alarm(0)
      return result

    return wrapped_func

  return decorator


@timeout(60)
def call_primitive_function(query, model):
  completion = client.chat.completions.create(
      model=model,
      messages=query,
      max_tokens=MAX_TOKENS,
      temperature=temp,
      top_p=topp,
  )
  return completion


def process_result(completion):
  answer = completion.choices[0].message.content
  input_tokens = completion.usage.prompt_tokens
  output_tokens = completion.usage.completion_tokens
  result = [answer, input_tokens, output_tokens]
  return result


def base_query_function(args):  # pylint: disable=redefined-outer-name
  """Base query function."""
  # quries: list of messages
  # query_indexes: list of query indexes, the index in the original dataset
  # model: openai model name
  queries, query_indexes, model, process_id = args  # pylint: disable=redefined-outer-name

  random_sec = np.random.randint(0, 30)
  # adovid bunching up requests
  time.sleep(random_sec)

  total_queries = len(queries)

  exception_count = 0
  start_time = time.time()

  results = {}

  num_prompt_tokens = 0
  num_answer_tokens = 0

  for i, q in enumerate(queries):
    results[query_indexes[i]] = None
    for retry in range(RETRY):
      try:
        if results[query_indexes[i]] is None:
          completion = call_primitive_function(q, model)
          results[query_indexes[i]] = process_result(completion)
          num_prompt_tokens += results[query_indexes[i]][1]
          num_answer_tokens += results[query_indexes[i]][2]
      except Exception as e:  # pylint: disable=broad-exception-caught
        exception_count += 1
        if isinstance(e, openai.RateLimitError):
          print(
              f'process {process_id} rate limit error, will retry after'
              ' sleep 60s'
          )
          time.sleep(60)
          retry -= 1
        else:
          if retry == RETRY - 1:
            print(f'process {process_id} error {e}, skip this query')
          else:
            print(f'process {process_id} error {e}, retrying...')
          # time.sleep(10)
    if (i + 1) % 10 == 0:
      print(
          f'process {process_id} of {NUM_PROCESS}:'
          f' {i+1}/{total_queries} queries',
          f'time elapsed: {time.time() - start_time:.2f}s',
          f'total prompt tokens {num_prompt_tokens/1000:.2f}K, answer tokens'
          f' {num_answer_tokens/1000:.2f}K',
          f'# failures: {exception_count}',
      )
  return process_id, results


parser = argparse.ArgumentParser()
parser.add_argument('--samples_to_label', type=int, required=True)
parser.add_argument('--instruction_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)


args = parser.parse_args()


openai.api_key = ''

if not openai.api_key:
  raise ValueError('please set openai.api_key')

NUM_PROCESS = 20
MAX_TOKENS = 1024
RETRY = 3
temp = 0.7
topp = 0.95


SAMPLES_TO_LABEL = args.samples_to_label

assert args.output_file.endswith('.pkl')

# check whether the output file exists
if os.path.exists(args.output_file):
  outputs = pickle.load(open(args.output_file, 'rb'))
  print('output file exists, loading...')
else:
  print('output file does not exist, creating...')
  outputs = {}

# check whether the instruction file exists
if os.path.exists(args.instruction_file):
  instructions = pickle.load(open(args.instruction_file, 'rb'))
else:
  raise ValueError('instruction file does not exist')

num_samples = len(instructions)
num_done = len(outputs)
num_remainning = num_samples - num_done

if num_done == num_samples:
  print('all samples are labelled, exiting...')
  exit(0)
elif num_remainning < 1000 or SAMPLES_TO_LABEL < 1000:
  NUM_PROCESS = min(NUM_PROCESS, SAMPLES_TO_LABEL // 20, num_remainning // 20)
  NUM_PROCESS = max(NUM_PROCESS, 1)

print(
    f'number of samples: {num_samples}, number of labelled samples: {num_done}'
)

message_list = []
index_list = []
# find the first SAMPLES_TO_LABEL samples that are not labelled
for j in range(num_samples):
  if j not in outputs:
    message = [{'role': 'user', 'content': instructions[j]}]
    message_list.append(message)
    index_list.append(j)
    if len(message_list) == SAMPLES_TO_LABEL:
      break

num_queries = len(message_list)
num_queries_per_process = num_queries // NUM_PROCESS
query_indexes = index_list

args_list = []
# split job loads
for j in range(NUM_PROCESS):
  if j == NUM_PROCESS - 1:
    _queries = message_list[j * num_queries_per_process :]
    _query_indexes = query_indexes[j * num_queries_per_process :]
  else:
    _queries = message_list[
        j * num_queries_per_process : (j + 1) * num_queries_per_process
    ]
    _query_indexes = query_indexes[
        j * num_queries_per_process : (j + 1) * num_queries_per_process
    ]

  print(f'process {j} has {len(_queries)} queries')
  args_list.append([_queries, _query_indexes, 'gpt-3.5-turbo', j])

all_results = {}

with multiprocessing.Pool(NUM_PROCESS) as p:
  tmp_results = p.map(base_query_function, args_list)
  for tmp_result in tmp_results:
    process_id, result_dict = tmp_result
    for key in result_dict:
      all_results[key] = result_dict[key]

success_count = 0
for key in all_results:
  if all_results[key] is not None:
    outputs[key] = all_results[key]
    success_count += 1

print(f'success count: {success_count}, all queries: {num_queries}')
# dump the results to args.output_file
pickle.dump(outputs, open(args.output_file, 'wb'))
