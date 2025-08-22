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

"""Use KL to evaluate validation set."""

import click
from datasets import load_dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer


AutoModelForCausalLM = transformers.AutoModelForCausalLM

# CUDA_VISIBLE_DEVICES=3 python3 eval/vali_loss_compute.py \
# -checkpoint gemma-7b-sft-math_1k \
# -assistant_checkpoint math_1k_seed_20_skd_kl_50_100_google-gemma-2b-it \
# -loss_type reverse_kl -task_type math -enable_chat True \
# -inp_length 1024 -max_length 1024


class JSD(nn.Module):

  def __init__(
      self, beta=0.5, reduction="batchmean", epsilon=1e-12, log_target=True
  ):
    super(JSD, self).__init__()
    self.kl = nn.KLDivLoss(reduction=reduction, log_target=log_target)
    self.beta = beta
    self.epsilon = epsilon

  def forward(self, p, q):
    p = nn.Softmax(p, dim=-1)
    q = nn.Softmax(q, dim=-1)
    m = self.beta * p + (1 - self.beta) * q
    m = (m + self.epsilon).log()
    return self.beta * self.kl(p, m) + (1 - self.beta) * self.kl(q, m)


class ReverseKL(nn.Module):

  def __init__(self, reduction="batchmean", log_target=True):
    super(ReverseKL, self).__init__()
    self.kl = nn.KLDivLoss(reduction=reduction, log_target=log_target)

  def forward(self, q, p):
    return self.kl(p, q)


@click.command()
@click.option("-checkpoint", type=str)
@click.option("-assistant_checkpoint", type=str)
@click.option("-loss_type", type=str)
@click.option("-task_type", type=str)
@click.option("-enable_chat", type=bool)
@click.option("-inp_length", type=int)
@click.option("-max_length", type=int)
def main(
    checkpoint,
    assistant_checkpoint,
    loss_type,
    task_type,
    enable_chat,
    inp_length,
    max_length,
):
  if loss_type == "kl":
    loss_funct = nn.KLDivLoss(reduction="batchmean", log_target=True)
  elif loss_type == "ce":
    loss_funct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
  elif loss_type == "reverse_kl":
    loss_funct = ReverseKL(reduction="batchmean", log_target=True)
  elif loss_type == "jsd":
    loss_funct = JSD(beta=0.5, reduction="batchmean", log_target=True)
  else:
    print("Loss type is not supported!")
    exit(1)

  model = AutoModelForCausalLM.from_pretrained(
      checkpoint,
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
      device_map="auto",
  )
  model.eval()
  assistant_model = AutoModelForCausalLM.from_pretrained(
      assistant_checkpoint,
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
      device_map="auto",
  )
  assistant_model.eval()

  if task_type == "gsm":
    data = load_dataset(
        "json",
        data_files="data/gsm8k_vali.json",
        field="instances",
        split="train",
    )
    vali_data_src = [f"Q: {ele} \n\nA:" for ele in data["question"]]
    vali_data_ref = [ele.strip() + "\n\n" for ele in data["answer"]]
  elif task_type == "math":
    vali_data = load_dataset(
        "json",
        data_files="data/Math_CoT_vali.json",
        field="instances",
        split="train",
    )
    vali_data_src = [ele for ele in vali_data["instruction"]]
    vali_data_ref = [ele for ele in vali_data["response"]]
  else:
    print("We don't support other task types!")
    exit(1)

  tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")

  total_loss, n = 0, 0
  with torch.no_grad():
    for src, ref in tqdm(
        zip(vali_data_src, vali_data_ref), total=len(vali_data_ref)
    ):
      if enable_chat:
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": src},
            ],
            return_dict=True,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=inp_length,
            add_generation_prompt=True,
        ).to("cuda")
        final_outputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": src},
                {"role": "assistant", "content": ref},
            ],
            return_dict=True,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_generation_prompt=False,
        ).to("cuda")
      else:
        inputs = tokenizer(
            [src],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=inp_length,
        ).to("cuda")
        final_outputs = tokenizer(
            [src + ref],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to("cuda")
      student_outputs = assistant_model(**final_outputs)
      teacher_outputs = model(**final_outputs)

      if loss_type == "ce":
        # Shift so that tokens < n predict n
        shift_logits = student_outputs.logits[Ellipsis, :-1, :].contiguous()
        # padding input labels
        labels = final_outputs["input_ids"].clone()
        labels[:, : inputs.input_ids.shape[1]] = -100
        shift_labels = labels[Ellipsis, 1:].contiguous()
        # flatten logits and labels
        shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = loss_funct(shift_logits, shift_labels)
      else:
        shift_student_logps = (
            student_outputs.logits[0, inputs.input_ids.shape[1] :]
            .log_softmax(dim=-1)
            .contiguous()
        )
        shift_teacher_logps = (
            teacher_outputs.logits[0, inputs.input_ids.shape[1] :]
            .log_softmax(dim=-1)
            .contiguous()
        )
        loss = loss_funct(shift_student_logps, shift_teacher_logps)

      total_loss += loss
      n += 1
      # print(shift_student_logps)
      # print(shift_teacher_logps)
      # print(total_loss/n)

      # if torch.isnan(total_loss/n):
      #     break
      # print('>'*50)

  print(f"Validation {loss_type} loss: ", total_loss / n)


if __name__ == "__main__":
  main()
