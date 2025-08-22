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

"""Run inference with instruction."""

import argparse
import os
import sys
import time
from accelerate import Accelerator
import peft
from peft import prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from utils.data_utils import DataCollatorForSupervisedDataset
from utils.data_utils import TokenizedSupervisedInstructDataset


def generate_batch_with_input_ids(
    model,
    input_ids,
    attention_mask,
    accelerator,
    tokenizer,
    sample=False,
    output_fname="generations",
    min_length=0,
    max_length=100,
    min_new_tokens=0,
    top_k=50,
    top_p=1,
    temperature=1,
    repetition_penalty=1,
):
  """Generate batch with input ids."""
  with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=sample,
        min_length=min_length,
        max_length=max_length,
        min_new_tokens=min_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    generated = tokenizer.batch_decode(output, skip_special_tokens=True)

  generated = [g.strip() for g in generated]

  local_rank = accelerator.process_index
  output_str = "Generated:\n" + "\n\n".join(generated)
  output_str = output_str.encode("utf-8").strip()
  with open(f"{output_fname}_gpu{local_rank}.txt", "ab") as f:
    f.write(output_str)
    f.write("\n".encode("utf-8"))


def parse_args():
  """Parse args."""
  parser = argparse.ArgumentParser(
      description="Run inference on a model for generation."
  )
  parser.add_argument(
      "--model_name_or_path",
      type=str,
      help=(
          "Path to pretrained model or model identifier from"
          " huggingface.co/models"
      ),
  )
  parser.add_argument(
      "--lora_weights_path", default=None, type=str, help="Path to lora weights"
  )
  parser.add_argument(
      "--instruction_file", type=str, default=None, help="Dataset name."
  )
  parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
  parser.add_argument(
      "--max_length",
      type=int,
      default=256,
      help="Max length of the generated sequence.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="inference_generation",
      help="Output directory.",
  )
  parser.add_argument("--qbits", type=int, default=8, help="Quantization bits.")
  parser.add_argument(
      "--exp_path", type=str, default="./", help="Path to working dir."
  )
  # generation config
  parser.add_argument(
      "--prompt_style",
      type=str,
      default="vicuna",
      help="Prompt style. vicuna or self_instruct",
  )
  parser.add_argument(
      "--sample",
      action="store_true",
      help="use sampling or not. When false greedy decoding is used.",
  )
  parser.add_argument(
      "--top_k", type=int, default=0, help="top k words for generation"
  )
  parser.add_argument(
      "--top_p", type=float, default=1.0, help="top p probability"
  )
  parser.add_argument(
      "--temperature", type=float, default=1.0, help="sampling temperature"
  )
  parser.add_argument(
      "--repetition_penalty",
      type=float,
      default=1.0,
      help="The parameter for repetition penalty. 1.0 means no penalty.",
  )
  parser.add_argument(
      "--access_token", type=str, default=None, help="Huggingface access token"
  )
  parser.add_argument(
      "--enforce_min_new_tokens",
      action="store_true",
      help="enforce min new tokens",
  )
  args = parser.parse_args()

  return args


def main():
  args = parse_args()

  torch.manual_seed(int(time.time() * 1000))

  assert (
      args.instruction_file is not None
  ), "Please specify the instruction file."
  assert (
      args.batch_size == 1
  ), "Currently only support generation with batchsize = 1."

  # huggingface access token
  access_token = args.access_token
  if access_token is not None:
    os.system("huggingface-cli login --token " + access_token)

  compute_dtype = (
      torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  )

  if args.qbits == 4 or args.qbits == 8:
    raise NotImplementedError
  elif args.qbits == 16:
    if args.exp_path == "./":
      model = AutoModelForCausalLM.from_pretrained(
          args.model_name_or_path,
          torch_dtype=compute_dtype,
          low_cpu_mem_usage=True,
      )
      tokenizer = AutoTokenizer.from_pretrained(
          args.model_name_or_path, use_fast=False
      )
    else:
      model = AutoModelForCausalLM.from_pretrained(
          args.exp_path + args.model_name_or_path.split("/")[-1],
          torch_dtype=compute_dtype,
          low_cpu_mem_usage=True,
      )
      tokenizer = AutoTokenizer.from_pretrained(
          args.exp_path + args.model_name_or_path.split("/")[-1], use_fast=False
      )

  # if there is no pad token id, add it
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id
  if args.lora_weights_path is not None and args.lora_weights_path != "None":
    model = peft.PeftModel.from_pretrained(
        model, args.lora_weights_path
    )  # all requires_grad=False by default

  if args.qbits == 4 or args.qbits == 8:
    model = prepare_model_for_kbit_training(model)

  model.eval()
  for p in model.parameters():
    p.requires_grad = False

  accelerator = Accelerator()
  model = accelerator.prepare(model)

  # build a text dataset from the instruction file
  with open(args.instruction_file, "r") as f:
    lines = f.readlines()
    instructions = []
    inputs = []

    num_lines = len(lines)
    current_index = 0
    while current_index < num_lines:
      if "### BOI ###" in lines[current_index]:
        # begin of instruction
        current_instruction = ""
        repeat = int(lines[current_index].split("###")[-1])
        while True:
          current_index += 1
          if "### EOI ###" in lines[current_index]:
            to_extend = [current_instruction.strip()] * repeat
            input_to_extend = [""] * repeat  # dummy
            instructions.extend(to_extend)
            inputs.extend(input_to_extend)
            break
          current_instruction += lines[current_index]
      else:
        current_index += 1
        continue

    # [print(i) for i in instructions]

    if len(instructions) < 1:
      raise ValueError("empty instruction file")
    answers = [""] * len(instructions)

    json_dataset = {}
    for i in range(len(instructions)):
      json_dataset[i] = {
          "instruction": instructions[i],
          "input": inputs[i],
          "output": answers[i],
      }

  dataset = TokenizedSupervisedInstructDataset(
      json_dataset,
      tokenizer=tokenizer,
      max_length=args.max_length,
      truncation=True,
      num_proc=1,
      tokenize_type="eval",
      prmopt_template=args.prompt_style,
  )
  data_collator = DataCollatorForSupervisedDataset(
      tokenizer,
      padding="longest",
      return_tensors="pt",
      device=accelerator.device,
      padding_side="left",
  )
  # dataLoaders creation:
  eval_data_loader = DataLoader(
      dataset,
      shuffle=False,
      collate_fn=data_collator,
      batch_size=args.batch_size,
  )
  eval_data_loader = accelerator.prepare(eval_data_loader)

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

  for batch in eval_data_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    if args.enforce_min_new_tokens:
      raise NotImplementedError
    else:
      min_new_tokens = 0
      generation_fname = "/generations"
    generate_batch_with_input_ids(
        model,
        input_ids,
        attention_mask,
        accelerator,
        tokenizer,
        sample=args.sample,
        max_length=args.max_length,
        min_new_tokens=min_new_tokens,
        output_fname=args.output_dir + generation_fname,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    sys.stdout.flush()


if __name__ == "__main__":
  main()
