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

import math
import os
import random
from typing import Iterable, List, TypeVar

from accelerate import Accelerator
import click
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import get_scheduler
from transformers import set_seed
import wandb


# load in accelerator
accelerator = Accelerator()

T = TypeVar("T")


def prompt_batchify(data, batch_size):
  """Batchify a list of data into batches of a given size."""
  assert batch_size > 0

  batch = []
  for item in data:
    # Yield next batch
    if len(batch) == batch_size:
      yield batch
      batch = []

    batch.append(item)

  # Yield last un-filled batch
  if len(batch) != 0:
    yield batch


class JSD(nn.Module):
  def __init__(self):
    super(JSD, self).__init__()

  def forward(self, log_p, log_q):
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    m = 0.5 * (p + q + 1e-12)
    return 0.5 * (
        torch.nn.functional.kl_div(
            log_p, torch.log(m), reduction="batchmean", log_target=True
        )
        + torch.nn.functional.kl_div(
            log_q, torch.log(m), reduction="batchmean", log_target=True
        )
    )


class reverse_kl(nn.Module):

  def __init__(self):
    super(reverse_kl, self).__init__()

  # q and p are logprobs
  def forward(self, log_p, log_q):
    return torch.nn.functional.kl_div(
        log_q, log_p, reduction="batchmean", log_target=True
    )


def disable_dropout_in_model(model):
  for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
      module.p = 0


def compute_vali_loss(vali_data_src, vali_data_ref, assistant_model, tokenizer,
                      loss_funct, model, input_length, max_length, loss_type):
  """Compute validation loss."""
  total_loss, n, index = 0, 0, 0
  assistant_model.eval()
  print("Run Evaluation...")
  with torch.no_grad():
    for src, ref in tqdm(zip(vali_data_src, vali_data_ref),
                         total=len(vali_data_ref)):
      inputs = tokenizer.apply_chat_template(
          [{"role": "user", "content": src},],
          return_dict=True, tokenize=True, return_tensors="pt", padding=True,
          truncation=True, max_length=input_length, add_generation_prompt=True
      )
      # query + response already covers generation token in the prompt
      final_outputs = tokenizer.apply_chat_template(
          [{"role": "user", "content": src},
           {"role": "assistant", "content": ref}],
          return_dict=True, tokenize=True, return_tensors="pt",
          padding=True, truncation=True, max_length=max_length,
          add_generation_prompt=False
      )
      student_outputs = assistant_model(
          **final_outputs.to(assistant_model.device)
      )
      if loss_type == "ce":
        # Shift so that tokens < n predict n
        shift_logits = student_outputs.logits[Ellipsis, :-1, :].contiguous()
        # padding input labels
        labels = final_outputs["input_ids"].clone()
        labels[:, :inputs.input_ids.shape[1]] = -100
        shift_labels = labels[Ellipsis, 1:].contiguous()
        # flatten logits and labels
        shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = loss_funct(shift_logits, shift_labels)
      else:
        shift_student_logps = student_outputs.logits[
            0, inputs.input_ids.shape[1]:].log_softmax(dim=-1).contiguous()
        teacher_outputs = model(**final_outputs.to(model.device))
        shift_teacher_logps = teacher_outputs.logits[
            0, inputs.input_ids.shape[1]:].log_softmax(dim=-1).contiguous()
        loss = loss_funct(shift_student_logps,
                          shift_teacher_logps.to(assistant_model.device))
      total_loss += loss
      n += 1
      index += 1
  assistant_model.train()
  return total_loss/n


def parse_output(gen_output, checkpoint, prompt, end_of_string_ls, max_length,
                 tokenizer):
  """convert model output into new tokens, and apply chat tempplate again for logit computation."""

  if "gemma" in checkpoint.lower():
    split_token = "\nmodel\n"
  elif 'qwen' in checkpoint.lower():
    split_token = "assistant\n"
  else:
    print("Model not supported!")
    exit(1)

  if split_token in gen_output[0]:
    new_str = "".join(gen_output[0].split(split_token)[1])
    if end_of_string_ls != None:
      for stop_ele in end_of_string_ls:
        new_str = new_str.split(stop_ele)[0]

    final_outputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": new_str}],
        return_dict=True, tokenize=True, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length, add_generation_prompt=False,
    )
  # if split token not in output, input must be too long.
  # just use input in this case. Loss is zero?
  else:
    new_str = "<Input is too long>"
    final_outputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_dict=True, tokenize=True, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length, add_generation_prompt=False
    )

  return final_outputs, new_str


@click.command()
@click.option("-top_k", type=int, default=0)
@click.option("-top_p", type=int, default=0)
@click.option("-task_type", type=str, default=0)
@click.option("-grad_acc_size", type=int)
@click.option("-n_epoch", type=int)
@click.option("-eval_step", type=int)
@click.option("-seed", type=int, default=None)
@click.option("-max_new_tokens", type=int)
@click.option("-kd_type", type=str)
@click.option("-distance_metric", type=str)
@click.option("-debug_enable", type=bool, default=False)
@click.option("-prefix", type=str)
@click.option("-lr", type=float, default=1e-5)
@click.option("-checkpoint", type=str)
@click.option("-assistant_checkpoint", type=str)
@click.option("-inp_length", type=int, default=256)
@click.option("-max_length", type=int, default=768)
@click.option("-wandb_key", type=str)
@click.option("-wandb_proj", type=str, help="speculative_kd")
@click.option("-teacher_temperature", type=float, help="0.5")
@click.option("-teacher_top_p", type=float, help="0.5")
@click.option("-student_temperature", type=float, help="0.5")
@click.option("-student_top_p", type=float, help="0.5")
@click.option("-jsd_beta", type=float, help="0.5 or 0.1")
@click.option("-enable_stop_token", type=bool)
@click.option("-tokenizer_name", type=str)
@click.option("-ckpt_prefix", type=str)
@click.option("-early_stop_epoch", type=int, default=3)
@click.option("-mixed_ratio", type=float, default=0.5)
@click.option("-expected_seq_len", type=int, default=0)
def main(top_k, top_p, task_type, grad_acc_size, kd_type, distance_metric,
         debug_enable, n_epoch, seed, eval_step, prefix, lr, checkpoint,
         assistant_checkpoint, inp_length, max_length, wandb_key, wandb_proj,
         max_new_tokens, teacher_temperature, teacher_top_p,
         student_temperature, student_top_p, jsd_beta, enable_stop_token,
         tokenizer_name, ckpt_prefix, early_stop_epoch,
         mixed_ratio, expected_seq_len):

  if seed:
    set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.use_deterministic_algorithms(True)

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
  if enable_stop_token:
    if task_type == "gsm_1k" or task_type == "gsm_100":
      end_of_string_ls = ["\n\n"]
    elif task_type == "math" or task_type == "math_1k":
      end_of_string_ls = [
          "\n###\nProblem: ",
          "\nSolve the following math problem step-by-step.",
          "\nuser",
          "\n\n\n\n",
          "Solve the following math problem step-by-step.",
      ]
    elif task_type == "summ_1k" or task_type == "summ_100":
      end_of_string_ls = ["\nuser"]
    elif task_type == "mt_1k" or task_type == "mt_100":
      end_of_string_ls = ["\n"]
    else:
      print("We currently support math, mt_1k, summ, QA and code!")
      exit(1)
  else:
    end_of_string_ls = None

  # assistant model is assumed to be greedy decoding.
  # Need to modify in the source code if needed
  assistant_model = AutoModelForCausalLM.from_pretrained(
      assistant_checkpoint,
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
  )
  assistant_model = accelerator.prepare_model(
      assistant_model, device_placement=True
  )

  if kd_type != "seq_kd":
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = accelerator.prepare_model(model, device_placement=True)
    model.eval()
    for _, param in model.named_parameters():
      param.requires_grad = False

  if kd_type == "skd" or kd_type == "mixed_skd":
    assistant_model.module.generation_config.num_assistant_tokens = 5
    assistant_model.tokenizer = tokenizer
    assistant_model.num_beams = 1  # disable top-k
    assistant_model.do_sample = True  # enable sampling
    # perform ancestral sampling/top-p samping
    assistant_model.top_p = student_top_p
    assistant_model.temperature = student_temperature
    assistant_model.module.generation_config.num_assistant_tokens_schedule = (
        "constant"
    )

  max_grad_norm = 1
  optimizer = AdamW(assistant_model.parameters(), lr=lr)
  optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)

  if kd_type == "seq_kd":
    loss_funct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
    loss_type = "ce"
  else:
    loss_type = "non-ce"
    if distance_metric == "kl":
      loss_funct = nn.KLDivLoss(reduction="batchmean", log_target=True)
    elif distance_metric == "reverse_kl":
      loss_funct = reverse_kl()
    elif distance_metric == "jsd":
      loss_funct = JSD()
    else:
      print("We do not support other loss functions at the moment!")
      exit(1)

  student_save_name = "-".join(assistant_checkpoint.split("/")[-2:])

  if not debug_enable and accelerator.is_main_process:
    # make sure login id is right
    wandb.login(key=wandb_key, relogin=True)
    wandb.init(project=wandb_proj,
               name=f"{task_type}_seed_{seed}_{kd_type}_{distance_metric}_"
               f"topk_{top_k}_top_p_{top_p}_{student_save_name}_"
               f"jsd_beta_{jsd_beta}_{ckpt_prefix}", config={
                   "lr": lr,
                   "max_grad_norm": max_grad_norm,
                   "top_k": top_k,
                   "top_p": top_p,
                   "task_type": task_type,
                   "grad_acc_size": grad_acc_size,
                   "kd_type": kd_type,
                   "distance_metric": distance_metric,
                   "seed": seed
               })

  if task_type == "gsm_1k" or task_type == "gsm_100":
    data = load_dataset(
        "json", data_files=f'data/{task_type}_train.json',
        field="instances", split="train",
    )
    if (kd_type == "supervised_kd" or kd_type == "mixed" or
        kd_type == "mixed_skd" or kd_type == "mixed_train"):
      train_src_ref_dict = {
          src: ref for src, ref in zip(data["instruction"], data["response"])
      }

    train_data_src = [ele for ele in data["instruction"]]
    vali_data = load_dataset(
        "json", data_files="data/gsm_1k_vali.json",
        field="instances", split="train"
    )
    vali_data_src = [ele for ele in vali_data["instruction"]]
    vali_data_ref = [ele for ele in vali_data["response"]]
  elif task_type == "math":
    data = load_dataset(
        "json", data_files="data/Math_CoT_train.json", field="instances",
        split="train"
    )
    if (kd_type == "supervised_kd" or kd_type == "mixed" or
        kd_type == "mixed_skd"):
      train_src_ref_dict = {
          src: ref for src, ref in zip(data["instruction"], data["response"])
      }

    train_data_src = [ele for ele in data["instruction"]]
    vali_data = load_dataset(
        "json", data_files="data/Math_CoT_vali.json",
        field="instances", split="train"
    )
    vali_data_src = [ele for ele in vali_data["instruction"]]
    vali_data_ref = [ele for ele in vali_data["response"]]

  elif task_type == "math_1k":
    data = load_dataset(
        "json", data_files="data/math_train_1k.json", field="instances",
        split="train"
    )
    if (kd_type == "supervised_kd" or kd_type == "mixed" or
        kd_type == "mixed_skd" or kd_type == "mixed_train"):
      train_src_ref_dict = {
          src: ref for src, ref in zip(data["instruction"], data["response"])
      }

    train_data_src = [ele for ele in data["instruction"]]
    vali_data = load_dataset(
        "json", data_files="data/math_vali_1k.json", field="instances",
        split="train"
    )
    vali_data_src = [ele for ele in vali_data["instruction"]]
    vali_data_ref = [ele for ele in vali_data["response"]]
  elif task_type == "summ_1k" or task_type == "summ_100":
    train_dataset = load_dataset(
        "json", data_files=f'data/{task_type}_train.json',
        field="instances", split="train"
    )
    if (kd_type == "supervised_kd" or kd_type == "mixed" or
        kd_type == "mixed_skd" or kd_type == "mixed_train"):
      train_src_ref_dict = {
          src: ref for src, ref in zip(train_dataset["dialogue"],
                                       train_dataset["summary"])
      }

    train_data_src = [ele for ele in train_dataset["dialogue"]]
    vali_data = load_dataset(
        "json", data_files="data/summ_1k_vali.json", field="instances",
        split="train"
    )
    vali_data_src = [ele for ele in vali_data["dialogue"]]
    vali_data_ref = [ele for ele in vali_data["summary"]]
  elif task_type == "mt_1k" or task_type == "mt_100":
    data = load_dataset(
        "json", data_files=f'data/{task_type}_train.json',
        field="instances", split="train"
    )
    if (kd_type == "supervised_kd" or kd_type == "mixed" or
        kd_type == "mixed_skd" or kd_type == "mixed_train"):
      train_src_ref_dict = {
          src: ref for src, ref in zip(data["instruction"], data["response"])
      }

    train_data_src = [ele for ele in data["instruction"]]
    vali_data = load_dataset(
        "json", data_files="data/mt_1k_vali.json", field="instances",
        split="train"
    )
    vali_data_src = [ele for ele in vali_data["instruction"]]
    vali_data_ref = [ele for ele in vali_data["response"]]
  else:
    print("We currently support math, mt_1k, summ, QA and code!")
    exit(1)

  # decide max step dynamically based on epoch, train size and grad acc size
  max_step = math.ceil(len(train_data_src) * n_epoch / grad_acc_size)
  lr_scheduler = get_scheduler(
      "linear", optimizer, num_warmup_steps=0, num_training_steps=max_step
  )
  lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

  assistant_model.train()
  all_steps = 0

  save_inp_data_file = open(
      f"{task_type}_seed_{seed}_{kd_type}_{distance_metric}_{top_k}_"
      f"{student_save_name}_jsd_beta_{jsd_beta}_{ckpt_prefix}.input", "w"
  )
  save_out_data_file = open(
      f"{task_type}_seed_{seed}_{kd_type}_{distance_metric}_{top_k}_"
      f"{student_save_name}_jsd_beta_{jsd_beta}_{ckpt_prefix}.output", "w"
  )

  with accelerator.split_between_processes(
      train_data_src) as train_data_src_tot:
    with tqdm(total=len(list(train_data_src_tot))*n_epoch) as pbar:
      for cur_epoch_index in range(n_epoch):
        if cur_epoch_index == early_stop_epoch:
          break
        assistant_model.zero_grad()
        global_step = 0
        total_loss = 0
        # shuffle for each process at given GPU
        random.shuffle(train_data_src_tot)

        for prompts in prompt_batchify(train_data_src_tot, grad_acc_size):
          # sampling from teacher and studnet, no grad propogation
          temp_report_dict = {}
          batch_loss = 0

          # only save weights when it is not at debugging mode.
          # save_step = eval_step (main processor handles all)
          if (accelerator.is_main_process and all_steps % eval_step == 0 and
              not debug_enable):
            # compute validation loss
            if kd_type == "seq_kd":
              vali_loss = compute_vali_loss(
                  vali_data_src, vali_data_ref, assistant_model.module,
                  tokenizer, loss_funct, None, inp_length, max_length,
                  loss_type
              )
            else:
              vali_loss = compute_vali_loss(
                  vali_data_src, vali_data_ref, assistant_model.module,
                  tokenizer, loss_funct, model.module, inp_length,
                  max_length, loss_type
              )
            temp_report_dict['vali_loss'] = vali_loss

            if kd_type == "skd" or kd_type == "mixed_skd":
              save_name = f'{prefix}/{task_type}_seed_{seed}_{kd_type}_{distance_metric}_{top_k}_{all_steps}_{student_save_name}_jsd_beta_{jsd_beta}_{ckpt_prefix}'
            else:
              save_name = f'{prefix}/{task_type}_seed_{seed}_{kd_type}_{distance_metric}_{all_steps}_{student_save_name}_jsd_beta_{jsd_beta}_{ckpt_prefix}'

            tokenizer.save_pretrained(save_name)
            unwrapped_model = accelerator.unwrap_model(assistant_model)
            unwrapped_model.save_pretrained(
                save_name,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            print("Model and tokenizer are saved!")

          torch.cuda.empty_cache()
          for prompt in prompts:
            with torch.no_grad():
              if kd_type == "on-policy":
                inputs = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                    ],
                    return_dict=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=inp_length,
                    add_generation_prompt=True,
                ).to(assistant_model.device)
                final_outputs = assistant_model.module.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    num_beams=1,
                    stop_strings=end_of_string_ls,
                    tokenizer=tokenizer,
                    temperature=student_temperature,
                    top_p=student_top_p,
                )
              elif kd_type == "skd":
                inputs = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                    ],
                    return_dict=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=inp_length,
                    add_generation_prompt=True,
                ).to("cuda")
                final_outputs = model.module.generate(
                    **inputs,
                    assistant_model=assistant_model.module,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    expected_seq_len=expected_seq_len,
                    num_beams=1,
                    teacher_k=top_k,
                    teacher_p=top_p,
                    stop_strings=end_of_string_ls,
                    tokenizer=tokenizer,
                    temperature=teacher_temperature,
                    top_p=teacher_top_p,
                    return_dict_in_generate=True,
                )
                # print("Outside cor rate: ", final_outputs.correction_rate)
              elif kd_type == "supervised_kd" or kd_type == "seq_kd":
                inputs = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                    ],
                    return_dict=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=inp_length,
                    add_generation_prompt=True,
                ).to(assistant_model.device)
                final_outputs = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": train_src_ref_dict[prompt],
                        },
                    ],
                    return_dict=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    add_generation_prompt=False,
                )
                new_str = train_src_ref_dict[prompt]
              elif kd_type == "mixed":
                inputs = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                    ],
                    return_dict=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_new_tokens=max_new_tokens,
                    add_generation_prompt=True,
                ).to(assistant_model.device)
                rand_index = random.choices(
                    [0, 1], k=1, weights=[mixed_ratio, 1 - mixed_ratio]
                )[0]
                if rand_index == 0:
                  final_outputs = assistant_model.module.generate(
                      **inputs,
                      max_new_tokens=max_new_tokens,
                      do_sample=True,
                      num_beams=1,
                      stop_strings=end_of_string_ls,
                      tokenizer=tokenizer,
                  )
                elif rand_index == 1:
                  final_outputs = tokenizer.apply_chat_template(
                      [
                          {"role": "user", "content": prompt},
                          {
                              "role": "assistant",
                              "content": train_src_ref_dict[prompt],
                          },
                      ],
                      return_dict=True,
                      tokenize=True,
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=max_length,
                      add_generation_prompt=False,
                  )
                  new_str = train_src_ref_dict[prompt]
                else:
                  print("Bugs in rand index!")
                  exit(1)
              elif kd_type == "mixed_train":
                inputs = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                    ],
                    return_dict=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_new_tokens=max_new_tokens,
                    add_generation_prompt=True,
                ).to(assistant_model.device)
                if all_steps < int(len(train_data_src_tot) * n_epoch / 2):
                  final_outputs = tokenizer.apply_chat_template(
                      [
                          {"role": "user", "content": prompt},
                          {
                              "role": "assistant",
                              "content": train_src_ref_dict[prompt],
                          },
                      ],
                      return_dict=True,
                      tokenize=True,
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=max_length,
                      add_generation_prompt=False,
                  )
                  new_str = train_src_ref_dict[prompt]
                else:
                  final_outputs = assistant_model.module.generate(
                      **inputs,
                      max_new_tokens=max_new_tokens,
                      do_sample=True,
                      num_beams=1,
                      stop_strings=end_of_string_ls,
                      tokenizer=tokenizer,
                  )
              elif kd_type == "mixed_skd":
                inputs = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                    ],
                    return_dict=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=inp_length,
                    add_generation_prompt=True,
                ).to("cuda")
                rand_index = random.choices(
                    [0, 1], k=1, weights=[mixed_ratio, 1 - mixed_ratio]
                )[0]
                if rand_index == 0:
                  final_outputs = model.module.generate(
                      **inputs,
                      assistant_model=assistant_model.module,
                      max_new_tokens=max_new_tokens,
                      do_sample=True,
                      expected_seq_len=expected_seq_len,
                      num_beams=1,
                      teacher_k=top_k,
                      teacher_p=top_p,
                      stop_strings=end_of_string_ls,
                      tokenizer=tokenizer,
                      temperature=teacher_temperature,
                      top_p=teacher_top_p,
                  )
                elif rand_index == 1:
                  final_outputs = tokenizer.apply_chat_template(
                      [
                          {"role": "user", "content": prompt},
                          {
                              "role": "assistant",
                              "content": train_src_ref_dict[prompt],
                          },
                      ],
                      return_dict=True,
                      tokenize=True,
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=max_length,
                      add_generation_prompt=False,
                  )
                  new_str = train_src_ref_dict[prompt]
                else:
                  print("Bugs in rand index!")
                  exit(1)
              else:
                print(
                    "We only support on-policy KD and speculative KD for"
                    " distributed inference and training"
                )
                exit(1)

              if (
                  kd_type == "on-policy"
                  or kd_type == "skd"
                  or (kd_type == "mixed" and rand_index == 0)
                  or (kd_type == "mixed_skd" and rand_index == 0)
                  or (
                      kd_type == "mixed_train"
                      and all_steps
                      >= int(len(train_data_src_tot) * n_epoch / 2)
                  )
              ):
                gen_output = tokenizer.batch_decode(
                    final_outputs.sequences, skip_special_tokens=True
                )
                final_outputs, new_str = parse_output(
                    gen_output,
                    assistant_checkpoint,
                    prompt,
                    end_of_string_ls,
                    max_length,
                    tokenizer,
                )

              if accelerator.is_main_process and all_steps % eval_step == 0:
                if (
                    kd_type == "supervised_kd"
                    or kd_type == "seq_kd"
                    or (kd_type == "mixed" and rand_index == 1)
                    or (kd_type == "mixed_skd" and rand_index == 1)
                    or (
                        kd_type == "mixed_train"
                        and all_steps
                        < int(len(train_data_src_tot) * n_epoch / 2)
                    )
                ):
                  print("Current prompt: ")
                  print(prompt)
                  print(">" * 20)
                  print("Current generation: ")
                  print(train_src_ref_dict[prompt])
                  print("-" * 50)
                else:
                  # print out the generated data at eval steps
                  print("Current prompt: ")
                  print(prompt)
                  print(">" * 20)
                  print("Before parsing (input+output): ")
                  print(gen_output[0])
                  print(">" * 20)
                  print("Current generation: ")
                  print(new_str)
                  print("-" * 50)

              save_inp_data_file.write(prompt + "[SEP_WENDA]\n")
              save_out_data_file.write(new_str + "[SEP_WENDA]\n")

              student_outputs = assistant_model(**final_outputs)

              if kd_type == "seq_kd":
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
                # print(shift_student_logps.size())
                # print('-'*25)
                final_outputs = final_outputs.to(model.device)
                teacher_outputs = model(**final_outputs)
                shift_teacher_logps = (
                    teacher_outputs.logits[0, inputs.input_ids.shape[1] :]
                    .log_softmax(dim=-1)
                    .contiguous()
                )
                # compute distance metric
                if (
                    distance_metric == "kl"
                    or distance_metric == "reverse_kl"
                    or distance_metric == "jsd"
                ):
                  # input, target for kl
                  loss = loss_funct(shift_student_logps, shift_teacher_logps)
                else:
                  print("Other distance metrics are not supported!")
                  exit(1)

            loss = loss / len(prompts)
            batch_loss += loss
            accelerator.backward(loss)
            torch.cuda.empty_cache()
            pbar.update(1)

          # only pass the gradient, learning rate, clip gradients at
          # accumulation step
          total_loss += batch_loss
          grad_norm = torch.nn.utils.clip_grad_norm_(
              assistant_model.parameters(),
              max_grad_norm,
          )
          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          torch.cuda.empty_cache()
          # store all wandb data
          temp_report_dict['train_batch_loss'] = batch_loss
          temp_report_dict['grad_norm'] = grad_norm
          global_step += 1
          all_steps += 1
          temp_report_dict['train_loss'] = total_loss/global_step

          if not debug_enable and accelerator.is_main_process:
            wandb.log(temp_report_dict)

        print("Total loss: ", total_loss/global_step)

if __name__ == "__main__":
    main()