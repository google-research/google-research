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

"""Evaluate the model on test set."""

import argparse
import json
import os

import transformers
from utils import evaluate
from utils import get_alphabet_choice
from utils import get_yes_no
from utils import parse_math_boxed
from utils import parse_number
from vllm import LLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--n", default=0, type=int)
  parser.add_argument("--task", default="SQA", type=str)
  parser.add_argument("--model", default="mistral-7b", type=str)
  parser.add_argument("--model_dir", default="", type=str)
  parser.add_argument("--lora_rank", default=32, type=int)
  args = parser.parse_args()

  if args.model == "mistral-7b":
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
  elif args.model == "gemma-2b":
    base_model = "google/gemma-2b-it"
  elif args.model == "gemma-7b":
    base_model = "google/gemma-7b-it"
  else:
    raise ValueError(f"Unsupported model: {args.model}")

  if args.task == "BoolQ":
    adapter_path = f"./checkpoints/{args.model}_SQA_{args.n}"
  elif args.task == "OBQA":
    adapter_path = f"./checkpoints/{args.model}_ARC_{args.n}"
  elif args.task == "ESNLI":
    adapter_path = f"./checkpoints/{args.model}_ANLI_{args.n}"
  elif args.task == "GSM8K-Rev":
    adapter_path = f"./checkpoints/{args.model}_GSM8K_{args.n}"
  else:
    adapter_path = f"./checkpoints/{args.model}_{args.task}_{args.n}"

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      base_model,
      trust_remote_code=True,
      padding_side="right"
  )
  tokenizer.pad_token_id = tokenizer.eos_token_id
  tokenizer.add_bos_token = False
  tokenizer.add_eos_token = False

  if "mistral" in args.model:
    fr_template = """<s>[INST] Answer the following question:\n### Question:
    {question} [/INST] ### Answer: {answer}"""
  elif "gemma" in args.model:
    fr_template = """<bos><start_of_turn>user\nAnswer the following question:\n
    ### Question: {question}<end_of_turn>\n<start_of_turn>model\n
    ### Answer: {answer}"""
  else:
    raise ValueError(f"Unsupported model: {args.model}")

  with open(f"./test_data/{args.task}_test.json", "r") as f:
    test_samples = json.load(f)

  prompts = [fr_template.format(question=i["question"], answer="")
             for i in test_samples]

  sampling_params = SamplingParams(n=1,
                                   temperature=0,
                                   max_tokens=1024,
                                   stop_token_ids=[tokenizer.eos_token_id])

  llm = LLM(model=base_model,
            enable_lora=True,
            max_lora_rank=32,
            download_dir=args.model_dir,
            tensor_parallel_size=1)

  if adapter_path:
    lora_request = LoRARequest("finetined_adapter", 1, adapter_path)
  else:
    lora_request = None

  outputs = llm.generate(prompts,
                         sampling_params,
                         lora_request=lora_request
                        )

  is_math = False
  if args.task in ["SQA", "BoolQ"]:
    answer_extraction = get_yes_no
  elif args.task in ["ANLI", "ARC", "Date", "CSQA", "OBQA", "ESNLI"]:
    answer_extraction = get_alphabet_choice
  elif args.task in ["GSM8K", "GSM8K-Rev"]:
    answer_extraction = parse_number
    is_math = True
  elif args.task in ["TabMWP", "MATH"]:
    answer_extraction = parse_math_boxed
    is_math = True
  else:
    raise ValueError(f"Unsupported task: {args.task}")

  for e, output in enumerate(outputs):
    test_samples[e]["reasoning"] = output.outputs[0].text
    test_samples[e]["pred"] = answer_extraction(output.outputs[0].text)

  acc = evaluate(test_samples, is_math=is_math, pred_key="pred")
  print(f"accuracy on {args.task}: {acc}")

  os.makedirs("./results/", exist_ok=True)
  save_path = f"./results/{args.model}_{args.task}_{args.n}_{acc}.json"
  with open(save_path, "w") as f:
    json.dump(test_samples, f)
