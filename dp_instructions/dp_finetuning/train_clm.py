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

"""Train a LoRA model with DP-SGD."""

import argparse
import json
import logging
import math
import os
# packages for handling mixed precision training and distributed training
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import datasets
# packages for handling parameter-efficient fine-tuning
import peft
from peft import get_peft_model
from peft import LoraConfig
from peft import prepare_model_for_kbit_training
import torch
# packages for handling pre-trianed model and data loaders
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import get_scheduler
from transformers import SchedulerType
# code for dp training, dataset creation/tokenization, and helper functions
from utils import data_utils
from utils import dp_utils
from utils import general_utils
from utils import train_utils


logger = get_logger(__name__)


def parse_args():
  """Parse arguments."""
  parser = argparse.ArgumentParser(
      description="Finetune LLMs with (QLoRA+) DP-SGD."
  )
  # general
  parser.add_argument(
      "--dataset_name",
      type=str,
      default=None,
      help="The name of the training dataset",
  )
  parser.add_argument(
      "--prompt_style",
      type=str,
      default=None,
      help=(
          "Prompt style. Decide whether train a condintional generator or"
          " unconditional generator.  Will atuomatically set to uncondintional"
          " if None"
      ),
  )
  parser.add_argument(
      "--model_name_or_path",
      type=str,
      required=True,
      help=(
          "Path to pretrained model or model identifier from"
          " huggingface.co/models."
      ),
  )
  parser.add_argument(
      "--per_device_train_batch_size",
      type=int,
      default=8,
      help="Batch size (per device) for the training dataloader.",
  )
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=1e-5,
      help="Initial learning rate (after the potential warmup period) to use.",
  )
  parser.add_argument(
      "--weight_decay", type=float, default=0.0, help="Weight decay."
  )
  parser.add_argument(
      "--num_train_epochs",
      type=int,
      default=3,
      required=True,
      help="Total number of training epochs to perform.",
  )
  parser.add_argument(
      "--gradient_accumulation_steps",
      type=int,
      default=1,
      help=(
          "Number of updates steps to accumulate before performing a"
          " backward/update pass."
      ),
  )
  parser.add_argument(
      "--lr_scheduler_type",
      type=SchedulerType,
      default="cosine",
      help="The scheduler type to use.",
      choices=[
          "linear",
          "cosine",
          "cosine_with_restarts",
          "polynomial",
          "constant",
          "constant_with_warmup",
      ],
  )
  parser.add_argument(
      "--num_warmup_steps",
      type=int,
      default=0,
      help="Number of steps for the warmup in the lr scheduler.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default=None,
      required=True,
      help="Where to store the model and log.",
  )
  parser.add_argument("--seed", type=int, default=0, help="Random seed.")
  parser.add_argument(
      "--block_size",
      type=int,
      default=1024,
      help=(
          "Max sequence length after tokenization. Sequences longer than this"
          " will be truncated."
      ),
  )
  parser.add_argument(
      "--log_freq", type=int, default=100, help="Freq of loss logging."
  )
  parser.add_argument(
      "--access_token", type=str, default=None, help="Huggingface access token"
  )
  parser.add_argument(
      "--gradient_ckpt",
      action="store_true",
      help=(
          "Use gradient checkpointing or not. If true, will drop some forward"
          " activations and re-compute them during backward. This will save"
          " memory but slow down training."
      ),
  )
  parser.add_argument(
      "--no_eval_at_start",
      action="store_true",
      help="Do not do evaluation before training.",
  )
  parser.add_argument(
      "--eval_only", action="store_true", help="Only do evaluation and exit."
  )
  parser.add_argument(
      "--exp_path", type=str, default="./", help="Experiment path."
  )
  # qlora hyperparameters
  parser.add_argument(
      "--lora_r", type=int, default=8, help="Rank of LoRA fine-tuning."
  )
  parser.add_argument(
      "--lora_alpha", type=float, default=16, help="Value of alpha for lora."
  )
  parser.add_argument(
      "--qbits",
      type=int,
      default=4,
      help="Number of bits for quantization. Choices are 4, 8, 16.",
  )
  # dp hyperparameters
  parser.add_argument(
      "--delta", type=float, default=1e-5, help="Privacy parameter delta."
  )
  parser.add_argument(
      "--clip_norm",
      type=float,
      default=-1,
      help=(
          "Clip norm for DP-SGD. If negative, there will be no per-example"
          " gradient computation and clipping."
      ),
  )
  parser.add_argument(
      "--noise_multiplier",
      type=float,
      default=-1,
      help="Noise multiplier for DP-SGD. If negative, no noise will be added.",
  )

  args = parser.parse_args()
  return args


def main():
  args = parse_args()
  assert (
      args.dataset_name in ["chatbot_arena_instructions_train180k"]
      or "labelled" in args.dataset_name
  ), "dataset/task not supported"
  assert args.qbits in [
      4,
      8,
      16,
  ], "only support 4-bit, 8-bit (int), 16-bit (bfloat) training"

  # if passed along, set the training seed now.
  if args.seed is not None:
    set_seed(args.seed)

  os.makedirs(args.output_dir, exist_ok=True)

  # Login to huggingface to get access to llama2
  access_token = args.access_token
  if not access_token:
    os.system("huggingface-cli login --token " + access_token)

  compute_dtype = (
      torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  )

  # load pretrained model and tokenizer
  # in distributed training, the .from_pretrained methods guarantee that only
  # one local process can concurrently download model & vocab.
  if args.qbits == 4 or args.qbits == 8:
    raise NotImplementedError(
        "Bits and Bytes quantization is not supported in this version"
    )
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
      # for running in offline mode
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
    # Note: this works for Llama family of models
    tokenizer.pad_token_id = tokenizer.unk_token_id

  if args.gradient_ckpt:
    # use gradient checkpointing to save memory
    # this will drop some forward activations and re-compute during backward
    model.gradient_checkpointing_enable()

  if args.qbits == 4 or args.qbits == 8:
    model = prepare_model_for_kbit_training(model)

  config = LoraConfig(
      r=args.lora_r,
      lora_alpha=args.lora_alpha,
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
  )

  # check if there is a checkpoint
  checkpoint_epoch = general_utils.find_newest_checkpoint_epoch(
      args.output_dir
  )  # -1 means no checkpoint found
  if checkpoint_epoch >= 0:
    model = peft.PeftModel.from_pretrained(
        model, args.output_dir + f"/peftmodel_epoch{checkpoint_epoch}"
    )
    # all requires_grad=False by default
    for p in model.named_parameters():
      if "lora_A" in p[0] or "lora_B" in p[0]:
        p[1].requires_grad = True
    print(
        "Continuing training from checkpoint, will start training at epoch"
        f" {checkpoint_epoch+1}"
    )
  else:
    model = get_peft_model(model, config)
    print("No checkpoint found. Start training from scratch.")

  general_utils.print_trainable_parameters(model)

  accelerator = Accelerator(args.output_dir)
  args.logical_batch_size = (
      accelerator.num_processes
      * args.per_device_train_batch_size
      * args.gradient_accumulation_steps
  )

  if args.clip_norm >= 0:
    # register hooks for per-example gradient computation
    dp_utils.make_lora_model_dp(model)
    if args.noise_multiplier > 0 and args.clip_norm > 0:
      print("Training with differential privacy.")
    else:
      print("Training WITHOUT differential privacy.")

  # leave mixed precision training and distributed training to accelerator
  model = accelerator.prepare(model)

  # make one log on every process with the configuration for debugging.
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
  )
  logger.info(accelerator.state, main_process_only=False)
  if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
  else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

  # handle the repository creation
  if accelerator.is_main_process and args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)
  accelerator.wait_for_everyone()

  # create tokenized training dataset
  # simply pass a string to TokenizedSupervisedInstructDataset().
  # We will build a Dataset object inside.
  train_dataset = data_utils.TokenizedSupervisedInstructDataset(
      args.dataset_name,
      tokenizer=tokenizer,
      split="train",
      max_length=args.block_size,
      truncation=True,
      num_proc=4,
      prmopt_template=args.prompt_style,
      exp_path=args.exp_path,
  )
  # create tokenized validation dataset
  eval_dataset = data_utils.TokenizedSupervisedInstructDataset(
      args.dataset_name,
      tokenizer=tokenizer,
      split="val",
      max_length=args.block_size,
      truncation=True,
      num_proc=4,
      prmopt_template=args.prompt_style,
      exp_path=args.exp_path,
  )

  # create data collator to collate batches of tokenized inputs.
  # this collator handles padding to the longest sequence in the batch
  data_collator = data_utils.DataCollatorForSupervisedDataset(
      tokenizer,
      padding="longest",
      return_tensors="pt",
      device=accelerator.device,
      max_length=args.block_size,
  )

  # DataLoaders creation:
  train_dataloader = DataLoader(
      train_dataset,
      shuffle=True,
      collate_fn=data_collator,
      batch_size=args.per_device_train_batch_size,
  )

  eval_batchsize = args.per_device_train_batch_size // 2
  if eval_batchsize == 0:
    eval_batchsize = 1

  if args.eval_only:
    eval_batchsize = 1
    assert (
        accelerator.num_processes == 1
    ), "eval_only mode only supports single process"

  eval_dataloader = DataLoader(
      eval_dataset,
      shuffle=False,
      collate_fn=data_collator,
      batch_size=eval_batchsize,
  )

  if args.eval_only:
    init_eval_loss = train_utils.eval_epoch(
        model,
        accelerator,
        eval_dataloader,
        -1,
        args,
        description="eval only",
        eval_only=args.eval_only,
    )
    exit(0)

  # only update lora parameters
  require_grad_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(
      require_grad_params, lr=args.learning_rate, weight_decay=args.weight_decay
  )

  # prepare optimizer, train_dataloader with accelerator. It will handle
  # distributed batch splitting and automated mixed precision scaling.
  optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
      optimizer, train_dataloader, eval_dataloader
  )

  # recalculate our total training steps as the size of the training dataloader
  # may have changed.
  num_update_steps_per_epoch = math.ceil(
      len(train_dataloader) / args.gradient_accumulation_steps
  )
  args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
  # afterwards we recalculate our number of training epochs
  args.num_train_epochs = math.ceil(
      args.max_train_steps / num_update_steps_per_epoch
  )

  # get lr scheduler
  lr_scheduler = get_scheduler(
      name=args.lr_scheduler_type,
      optimizer=optimizer,
      num_warmup_steps=args.num_warmup_steps,
      num_training_steps=args.max_train_steps,
  )

  # start training
  total_batch_size = (
      args.per_device_train_batch_size
      * accelerator.num_processes
      * args.gradient_accumulation_steps
  )
  logger.info(
      f"  Num examples = {len(train_dataset)},  Num Epochs ="
      f" {args.num_train_epochs}"
  )
  logger.info(
      f"  Learning rate = {args.learning_rate},  Noise multiplier ="
      f" {args.noise_multiplier}, Clip = {args.clip_norm}"
  )
  logger.info(
      "  Instantaneous batch size per device ="
      f" {args.per_device_train_batch_size},  Total train batch size (w."
      f" parallel, distributed & accumulation) = {total_batch_size}"
  )
  logger.info(
      f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}, "
      f" Total optimization steps = {args.max_train_steps}"
  )
  # only show the progress bar once on each machine.
  progress_bar = tqdm(
      range(args.max_train_steps), disable=not accelerator.is_local_main_process
  )
  completed_steps = 0
  start_epoch = checkpoint_epoch + 1 if checkpoint_epoch >= 0 else 0

  # check and load training states
  if checkpoint_epoch >= 0:
    completed_steps = general_utils.load_training_states(
        args.output_dir, optimizer, lr_scheduler, checkpoint_epoch
    )
    if accelerator.is_main_process:
      print(
          "\nOptimizer and lr_scheduler states loaded from checkpoint at epoch"
          f" {checkpoint_epoch}"
      )

  # update the progress_bar if load from checkpoint
  progress_bar.update(completed_steps)

  init_eval_loss = -1
  for epoch in range(start_epoch, args.num_train_epochs):
    # do one evaluation before training
    if epoch == 0 and not args.no_eval_at_start:
      init_eval_loss = train_utils.eval_epoch(
          model, accelerator, eval_dataloader, epoch, args, description="start"
      )

    epoch_loss, completed_steps = train_utils.train_epoch(
        model,
        tokenizer,
        accelerator,
        optimizer,
        lr_scheduler,
        train_dataloader,
        epoch,
        args,
        completed_steps,
        progress_bar,
    )
    eval_epoch_loss = train_utils.eval_epoch(
        model, accelerator, eval_dataloader, epoch, args, description="end"
    )

    if accelerator.is_main_process:
      # save the PEFT model and checkpoint
      general_utils.save_checkpoint(
          args.output_dir,
          model.module,
          optimizer,
          lr_scheduler,
          completed_steps,
          epoch,
      )
      with open(
          os.path.join(args.output_dir, f"all_results_epoch{epoch}.json"), "w"
      ) as f:
        json.dump(
            {
                "train_loss": epoch_loss,
                "eval_loss": eval_epoch_loss,
                "init_eval_loss": init_eval_loss,
                "epoch": epoch,
                "step": completed_steps,
            },
            f,
        )
      logger.info(f" \nEpoch {epoch} checkpoint saved at {args.output_dir}")


if __name__ == "__main__":
  main()
