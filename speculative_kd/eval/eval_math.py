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

"""Evaluate math problems using a model."""
import math

import click
from datasets import load_dataset
from grader import math_equal
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams


def _last_boxed_only_string(string):
  """Return the last boxed string from a string."""
  idx = string.rfind('\\boxed')
  if idx < 0:
    idx = string.rfind('\\fbox')
    if idx < 0:
      return None

  i = idx
  left_brace_idx = None
  right_brace_idx = None
  num_left_braces_open = 0
  while i < len(string):
    if string[i] == '{':
      num_left_braces_open += 1
      if left_brace_idx is None:
        left_brace_idx = i
    elif string[i] == '}':
      num_left_braces_open -= 1
      if num_left_braces_open == 0:
        right_brace_idx = i
        break

    i += 1

  if left_brace_idx is None or right_brace_idx is None:
    return None

  return string[left_brace_idx + 1: right_brace_idx].strip()


def match_answer(response):
  """Match the answer from a response."""
  is_matched = False
  ans_marker = 'answer:\n'
  ans_idx = response.lower().rfind(ans_marker)
  if ans_idx != -1:
    is_matched = True
    response = response[ans_idx + len(ans_marker):].strip()
    if response.endswith('\n'):
      response = response[:-2]

  ans_marker = 'answer: '
  ans_idx = response.lower().rfind(ans_marker)
  if ans_idx != -1:
    is_matched = True
    response = response[ans_idx + len(ans_marker):].strip()
    if response.endswith('\n'):
      response = response[:-2]

  # Find boxed
  ans_boxed = _last_boxed_only_string(response)
  if ans_boxed:
    is_matched = True
    response = ans_boxed

  # Grade
  return is_matched, response


@click.command()
@click.option('-inp_length', type=int, default=1024)
@click.option('-max_tokens', type=int, default=1024)
@click.option('-ckpt', type=str, default='math_seed_20_on-policy_kl_1260')
def main(max_tokens, ckpt, inp_length):
  # load in validation set
  vali_dataset = load_dataset(
      'json',
      data_files='data/Math_CoT_vali.json',
      field='instances',
      split='train',
  )
  tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-it')
  vali_data = [
      tokenizer.apply_chat_template(
          [{'role': 'user', 'content': ele}],
          tokenize=False,
          add_generation_prompt=True,
          max_length=inp_length,
      )
      for ele in vali_dataset['instruction']
  ]

  sampling_params = SamplingParams(
      max_tokens=max_tokens,
      temperature=0,
      stop=['\nSolve the following math problem step-by-step.', '\n\n\n\n'],
  )
  llm = LLM(model=ckpt)
  gen_outputs = llm.generate(vali_data, sampling_params)

  count, total = 0, 0
  for _, ele, ref in zip(vali_data, gen_outputs, vali_dataset['response']):
    if '\\boxed{' in ele.outputs[0].text and '}' in ele.outputs[0].text:
      out = ele.outputs[0].text.split('\\boxed{')[1].split('}')[0].strip()
      ans = ref.split('\\boxed{')[1].split('}')[0].strip()
      try:
        if r'\pi' in out or r'\pi' in ans:
          equivs = []
          for pi in [math.pi, 3.14]:
            equivs.append(math_equal(out, ans, timeout=True, pi=pi))
          equiv = any(equivs)
        else:
          equiv = math_equal(out, ans, timeout=True)
      except (ValueError, TypeError):
        equiv = False

      if equiv:
        count += 1

    total += 1
  print('Acc: ', count / total)


if __name__ == '__main__':
  main()
