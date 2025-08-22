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

"""Evaluate a model on the GSM8K dataset."""
import math

import click
from datasets import load_dataset
from grader import math_equal
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams


@click.command()
@click.option('-max_tokens', type=int, default=512)
@click.option('-visualize_text', type=bool, default=True)
@click.option('-ckpt', type=str, default='gemma-7b-it-gsm-1k')
def main(max_tokens, visualize_text, ckpt):
  # load in validation set
  vali_dataset = load_dataset('openai/gsm8k', 'main')['test']
  tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-it')
  vali_data_inp = [
      tokenizer.apply_chat_template(
          [{'role': 'user', 'content': f'Q: {ele} \n\nA:'}],
          tokenize=False,
          add_generation_prompt=True,
      )
      for ele in vali_dataset['question']
  ]

  sampling_params = SamplingParams(
      max_tokens=max_tokens, temperature=0, top_p=1, stop=['\n\n']
  )
  llm = LLM(model=ckpt)
  gen_outputs = llm.generate(vali_data_inp, sampling_params)
  count = 0
  total = 0
  for _, ele, ref in zip(vali_data_inp, gen_outputs, vali_dataset['answer']):
    if '####' in ele.outputs[0].text:
      ans = ele.outputs[0].text.split('####')[1].split('\n')[0].strip()
      gt = ref.split('####')[1].strip()
      try:
        if r'\pi' in ans or r'\pi' in gt:
          equivs = []
          for pi in [math.pi, 3.14]:
            equivs.append(math_equal(ans, gt, timeout=True, pi=pi))
          equiv = any(equivs)
        else:
          equiv = math_equal(ans, gt, timeout=True)
      except (ValueError, TypeError) as error:
        equiv = False
        print(error)
      if equiv:
        count += 1
      else:
        if visualize_text:
          print(ele.outputs[0].text)
          print('-' * 50)
          print(ref)
          print('>' * 50)
    total += 1
  print('Acc: ', count / total)


if __name__ == '__main__':
  main()
