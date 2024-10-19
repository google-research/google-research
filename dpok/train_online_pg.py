# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Finetune diffusion model via policy gradient method."""

import argparse
import copy
import dataclasses
import functools
import json
import logging
import os
import pickle
import random

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed  # pylint: disable=g-multiple-import
import datasets
from datasets import load_dataset
import diffusers
from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate
from diffusers.utils.import_utils import is_xformers_available
import ImageReward as imagereward
import numpy as np
from packaging import version
from pipeline_stable_diffusion_extended import StableDiffusionPipelineExtended
from reward_model import ValueMulti
from scheduling_ddim_extended import DDIMSchedulerExtended
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.utils.checkpoint
from tqdm.auto import tqdm
import transformers
from transformers import CLIPModel, CLIPProcessor  # pylint: disable=g-multiple-import
from transformers import CLIPTextModel, CLIPTokenizer  # pylint: disable=g-multiple-import
import utils

logger = get_logger(__name__, log_level="INFO")


def parse_args():
  """Parse command line flags."""

  parser = argparse.ArgumentParser(
      description="Simple example of a training script."
  )
  parser.add_argument(
      "--pretrained_model_name_or_path",
      type=str,
      default="runwayml/stable-diffusion-v1-5",
      # required=True,
      help=(
          "Path to pretrained model or model identifier from"
          " huggingface.co/models."
      ),
  )
  parser.add_argument(
      "--revision",
      type=str,
      default=None,
      required=False,
      help=(
          "Revision of pretrained model identifier from huggingface.co/models."
      ),
  )
  parser.add_argument(
      "--dataset_name",
      type=str,
      default=None,
      # default="lambdalabs/pokemon-blip-captions",
      help=(
          "The name of the Dataset (from the HuggingFace hub) to train on"
          " (could be your own, possibly private, dataset). It can also be a"
          " path pointing to a local copy of a dataset in your filesystem, or"
          " to a folder containing files that 🤗 Datasets can understand."
      ),
  )
  parser.add_argument(
      "--dataset_config_name",
      type=str,
      default=None,
      help=(
          "The config of the Dataset, leave as None if there's only one config."
      ),
  )
  parser.add_argument(
      "--train_data_dir",
      type=str,
      default=None,
      # previous: default="./dataset/align_test",
      help=(
          "A folder containing the training data. Folder contents must follow"
          " the structure described in"
          " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In"
          " particular, a `metadata.jsonl` file must exist to provide the"
          " captions for the images. Ignored if `dataset_name` is specified."
      ),
  )
  parser.add_argument(
      "--image_column",
      type=str,
      default="image",
      help="The column of the dataset containing an image.",
  )
  parser.add_argument(
      "--caption_column",
      type=str,
      default="text",
      help=(
          "The column of the dataset containing a caption or a list of"
          " captions."
      ),
  )
  parser.add_argument(
      "--max_train_samples",
      type=int,
      default=None,
      help=(
          "For debugging purposes or quicker training, truncate the number of"
          " training examples to this value if set."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="online_model",
      help=(
          "The output directory where the model predictions and checkpoints"
          " will be written."
      ),
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      default=None,
      help=(
          "The directory where the downloaded models and datasets will be"
          " stored."
      ),
  )
  parser.add_argument(
      "--seed", type=int, default=None, help="A seed for reproducible training."
  )
  parser.add_argument(
      "--resolution",
      type=int,
      default=512,
      help=(
          "The resolution for input images, all the images in the"
          " train/validation dataset will be resized to this resolution"
      ),
  )
  parser.add_argument(
      "--center_crop",
      default=True,
      # action="store_true",
      help=(
          "Whether to center crop the input images to the resolution. If not"
          " set, the images will be randomly cropped. The images will be"
          " resized to the resolution first before cropping."
      ),
  )
  parser.add_argument(
      "--random_flip",
      default=True,
      # action="store_true",
      help="whether to randomly flip images horizontally",
  )
  parser.add_argument(
      "--train_batch_size",
      type=int,
      default=8,
      help="Batch size (per device) for the training dataloader.",
  )
  parser.add_argument("--num_train_epochs", type=int, default=100)
  parser.add_argument(
      "--max_train_steps",
      type=int,
      default=10000,
      help=(
          "Total number of training steps to perform.  If provided, overrides"
          " num_train_epochs."
      ),
  )
  parser.add_argument(
      "--gradient_checkpointing",
      default=True,
      # action="store_true",
      help=(
          "Whether or not to use gradient checkpointing to save memory at the"
          " expense of slower backward pass."
      ),
  )
  parser.add_argument(
      "--scale_lr",
      action="store_true",
      default=False,
      help=(
          "Scale the learning rate by the number of GPUs, gradient accumulation"
          " steps, and batch size."
      ),
  )
  parser.add_argument(
      "--lr_scheduler",
      type=str,
      default="constant",
      help=(
          'The scheduler type to use. Choose between ["linear", "cosine",'
          ' "cosine_with_restarts", "polynomial", "constant",'
          ' "constant_with_warmup"]'
      ),
  )
  parser.add_argument(
      "--lr_warmup_steps",
      type=int,
      default=0,
      help="Number of steps for the warmup in the lr scheduler.",
  )
  parser.add_argument(
      "--use_8bit_adam",
      action="store_true",
      help="Whether or not to use 8-bit Adam from bitsandbytes.",
  )
  parser.add_argument(
      "--allow_tf32",
      action="store_true",
      help=(
          "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up"
          " training. For more information, see"
          " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
      ),
  )
  parser.add_argument(
      "--use_ema", default=False, help="Whether to use EMA model."
  )
  parser.add_argument(
      "--non_ema_revision",
      type=str,
      default=None,
      required=False,
      help=(
          "Revision of pretrained non-ema model identifier. Must be a branch,"
          " tag or git identifier of the local or remote repository specified"
          " with --pretrained_model_name_or_path."
      ),
  )
  parser.add_argument(
      "--dataloader_num_workers",
      type=int,
      default=0,
      help=(
          "Number of subprocesses to use for data loading. 0 means that the"
          " data will be loaded in the main process."
      ),
  )
  parser.add_argument(
      "--adam_beta1",
      type=float,
      default=0.9,
      help="The beta1 parameter for the Adam optimizer.",
  )
  parser.add_argument(
      "--adam_beta2",
      type=float,
      default=0.999,
      help="The beta2 parameter for the Adam optimizer.",
  )
  parser.add_argument(
      "--adam_weight_decay",
      type=float,
      default=0.0,
      help="Weight decay to use.",
  )
  parser.add_argument(
      "--adam_epsilon",
      type=float,
      default=1e-08,
      help="Epsilon value for the Adam optimizer",
  )
  parser.add_argument(
      "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
  )
  parser.add_argument(
      "--push_to_hub",
      action="store_true",
      help="Whether or not to push the model to the Hub.",
  )
  parser.add_argument(
      "--hub_token",
      type=str,
      default=None,
      help="The token to use to push to the Model Hub.",
  )
  parser.add_argument(
      "--hub_model_id",
      type=str,
      default=None,
      help=(
          "The name of the repository to keep in sync with the local"
          " `output_dir`."
      ),
  )
  parser.add_argument(
      "--logging_dir",
      type=str,
      default="logs",
      help=(
          "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
          " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
      ),
  )
  parser.add_argument(
      "--mixed_precision",
      type=str,
      default="fp16",
      choices=["no", "fp16", "bf16"],
      help=(
          "Whether to use mixed precision. Choose between fp16 and bf16"
          " (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU."
          "  Default to the value of accelerate config of the current system or"
          " the flag passed with the `accelerate.launch` command. Use this"
          " argument to override the accelerate config."
      ),
  )
  parser.add_argument(
      "--report_to",
      type=str,
      default="tensorboard",
      help=(
          "The integration to report the results and logs to. Supported"
          ' platforms are `"tensorboard"` (default), `"wandb"` and'
          ' `"comet_ml"`. Use `"all"` to report to all integrations.'
      ),
  )
  parser.add_argument(
      "--local_rank",
      type=int,
      default=-1,
      help="For distributed training: local_rank",
  )
  parser.add_argument(
      "--checkpointing_steps",
      type=int,
      default=2000,
      help=(
          "Save a checkpoint of the training state every X updates. These"
          " checkpoints are only suitable for resuming training using"
          " `--resume_from_checkpoint`."
      ),
  )
  parser.add_argument(
      "--checkpoints_total_limit",
      type=int,
      default=None,
      help=(
          "Max number of checkpoints to store. Passed as `total_limit` to the"
          " `Accelerator` `ProjectConfiguration`. See Accelerator::save_state"
          " https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
          " for more docs"
      ),
  )
  parser.add_argument(
      "--resume_from_checkpoint",
      type=str,
      default=None,
      help=(
          "Whether training should be resumed from a previous checkpoint. Use a"
          ' path saved by `--checkpointing_steps`, or `"latest"` to'
          " automatically select the last available checkpoint."
      ),
  )
  parser.add_argument(
      "--enable_xformers_memory_efficient_attention",
      action="store_true",
      help="Whether or not to use xformers.",
  )
  # Custom arguments
  parser.add_argument(
      "--p_step",
      type=int,
      default=5,
      help="The number of steps to update the policy per sampling step",
  )
  parser.add_argument(
      "--p_batch_size",
      type=int,
      default=2,
      help=(
          "batch size for policy update per gpu, before gradient accumulation;"
          " total batch size per gpu = gradient_accumulation_steps *"
          " p_batch_size"
      ),
  )
  parser.add_argument(
      "--v_flag",
      type=int,
      default=1,
  )
  parser.add_argument(
      "--g_step", type=int, default=1, help="The number of sampling steps"
  )
  parser.add_argument(
      "--g_batch_size",
      type=int,
      default=6,
      help="batch size of prompts for sampling per gpu",
  )
  parser.add_argument(
      "--sft_path",
      type=str,
      default="./checkpoints/models/finetune_b512_lr2e-05_max10000_w0.01",
      help="path to the pretrained supervised finetuned model",
  )
  parser.add_argument(
      "--reward_model_path",
      type=str,
      default="./checkpoints/reward/reward_model_5007.pkl",
      help="path to the pretrained reward model",
  )
  parser.add_argument(
      "--reward_weight", type=float, default=100, help="weight of reward loss"
  )
  parser.add_argument(
      "--reward_flag",
      type=int,
      default=0,
      help="0: ImageReward, 1: Custom reward model",
  )
  parser.add_argument(
      "--reward_filter",
      type=int,
      default=0,
      help="0: raw value, 1: took positive",
  )
  parser.add_argument(
      "--kl_weight", type=float, default=0.01, help="weight of kl loss"
  )
  parser.add_argument(
      "--kl_warmup", type=int, default=-1, help="warm up for kl weight"
  )
  parser.add_argument(
      "--buffer_size", type=int, default=1000, help="size of replay buffer"
  )
  parser.add_argument(
      "--v_batch_size", type=int, default=256, 
      help="batch size for value function update per gpu, no gradient accumulation"  # pylint: disable=line-too-long
  )
  parser.add_argument(
      "--v_lr", type=float, default=1e-4, help="learning rate for value fn"
  )
  parser.add_argument(
      "--v_step", type=int, default=5,
      help="The number of steps to update the value function per sampling step"
  )
  parser.add_argument(
      "--save_interval",
      type=int,
      default=100,
      help="save model every save_interval steps",
  )
  parser.add_argument(
      "--num_samples",
      type=int,
      default=1,
      help="number of samples to generate per prompt",
  )
  parser.add_argument(
      "--clip_norm", type=float, default=0.1, help="norm for gradient clipping"
  )
  parser.add_argument(
      "--gradient_accumulation_steps",
      type=int,
      default=12,
      help=(
          "Number of updates steps to accumulate before performing a"
          " backward/update pass for policy"
      ),
  )
  parser.add_argument("--lora_rank", type=int, default=4, help="rank for LoRA")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=1e-5,
      help="Learning rate for policy",
  )
  parser.add_argument(
      "--prompt_path",
      type=str,
      default="./dataset/drawbench/data_meta.json",
      help="path to the prompt dataset",
  )
  parser.add_argument(
      "--prompt_category",
      type=str,
      default="all",
      help="all or specific categories with comma [e.g., color,count]",
  )
  parser.add_argument(
      "--single_flag",
      type=int,
      default=1,
  )
  parser.add_argument(
      "--single_prompt",
      type=str,
      default="A green colored rabbit.",
  )
  parser.add_argument(
      "--sft_initialization",
      type=int,
      default=0,
  )
  parser.add_argument(
      "--num_validation_images",
      type=int,
      default=2,
  )
  parser.add_argument(
      "--multi_gpu",
      type=int,
      default=0,
  )
  parser.add_argument(
      "--ratio_clip",
      type=float,
      default=1e-4,
  )
  args = parser.parse_args()
  env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
  if env_local_rank != -1 and env_local_rank != args.local_rank:
    args.local_rank = env_local_rank

  # # Sanity checks
  # if args.dataset_name is None and args.train_data_dir is None:
  #     raise ValueError("Need either a dataset name or a training folder.")

  # default to using the same revision for the non-ema model if not specified
  if args.non_ema_revision is None:
    args.non_ema_revision = args.revision

  return args


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def get_random_indices(num_indices, sample_size):
  """Returns a random sample of indices from a larger list of indices.

  Args:
      num_indices (int): The total number of indices to choose from.
      sample_size (int): The number of indices to choose.

  Returns:
      A numpy array of `sample_size` randomly chosen indices.
  """
  return np.random.choice(num_indices, size=sample_size, replace=False)


def get_test_prompts(flag):
  """Gets test prompts."""

  if flag == "drawbench":
    test_batch = [
        "A pink colored giraffe.",
        (
            "An emoji of a baby panda wearing a red hat, green gloves, red"
            " shirt, and green pants."
        ),
        "A blue bird and a brown bear.",
        "A yellow book and a red vase.",
        "Three dogs on the street.",
        "Two cats and one dog sitting on the grass.",
        "A wine glass on top of a dog.",
        "A cube made of denim. A cube with the texture of denim.",
    ]
  elif flag == "partiprompt":
    test_batch = [
        "a panda bear with aviator glasses on its head",
        "Times Square during the day",
        "the skyline of New York City",
        "square red apples on a tree with circular green leaves",
        "a map of Italy",
        "a sketch of a horse",
        "the word 'START' on a blue t-shirt",
        "a dolphin in an astronaut suit on saturn",
    ]
  elif flag == "coco":
    test_batch = [
        "A Christmas tree with lights and teddy bear",
        "A group of planes near a large wall of windows.",
        "three men riding horses through a grassy field",
        "A man and a woman posing in front of a motorcycle.",
        "A man sitting on a motorcycle smoking a cigarette.",
        "A pear, orange, and two bananas in a wooden bowl.",
        "Some people posting in front of a camera for a picture.",
        "Some very big furry brown bears in a big grass field.",
    ]
  elif flag == "paintskill":
    test_batch = [
        "a photo of blue bear",
        "a photo of blue fire hydrant",
        "a photo of bike and skateboard; skateboard is left to bike",
        "a photo of bed and human; human is right to bed",
        "a photo of suitcase and bench; bench is left to suitcase",
        "a photo of bed and stop sign; stop sign is above bed",
        (
            "a photo of dining table and traffic light; traffic light is below"
            " dining table"
        ),
        "a photo of bear and bus; bus is above bear",
    ]
  else:
    test_batch = [
        "A dog and a cat.",
        "A cat and a dog.",
        "Two dogs in the park.",
        "Three dogs in the park.",
        "Four dogs in the park.",
        "A blue colored rabbit.",
        "A red colored rabbit.",
        "A green colored rabbit.",
    ]

  return test_batch


def _update_output_dir(args):
  """Modifies `args.output_dir` using configurations in `args`.

  Args:
      args: argparse.Namespace object.
  """
  if args.single_flag == 1:
    data_log = "single_prompt/" + args.single_prompt.replace(" ", "_") + "/"
  else:
    data_log = args.prompt_path.split("/")[-2] + "_"
    data_log += args.prompt_category + "/"
  learning_log = "p_lr" + str(args.learning_rate) + "_s" + str(args.p_step)
  learning_log += (
      "_b"
      + str(args.p_batch_size)
      + "_g"
      + str(args.gradient_accumulation_steps)
  )
  learning_log += "_l" + str(args.lora_rank)
  coeff_log = "_kl" + str(args.kl_weight) + "_re" + str(args.reward_weight)
  if args.kl_warmup > 0:
    coeff_log += "_klw" + str(args.kl_warmup)
  if args.sft_initialization == 0:
    start_log = "/pre_train/"
  else:
    start_log = "/sft/"
  if args.reward_flag == 0:
    args.output_dir += "/img_reward_{}/".format(args.reward_filter)
  else:
    args.output_dir += "/prev_reward_{}/".format(args.reward_filter)
  args.output_dir += start_log + data_log + "/" + learning_log + coeff_log
  if args.v_flag == 1:
    value_log = "_v_lr" + str(args.v_lr) + "_b" + str(args.v_batch_size)
    value_log += "_s" + str(args.v_step)
    args.output_dir += value_log


def _calculate_reward_ir(
    pipe,
    args,
    reward_tokenizer,
    tokenizer,
    weight_dtype,
    reward_clip_model,
    image_reward,
    imgs,
    prompts,
    test_flag=False,
):
  """Computes reward using ImageReward model."""
  if test_flag:
    image_pil = imgs
  else:
    image_pil = pipe.numpy_to_pil(imgs)[0]
  blip_reward, _ = utils.image_reward_get_reward(
      image_reward, image_pil, prompts, weight_dtype
  )
  if args.reward_filter == 1:
    blip_reward = torch.clamp(blip_reward, min=0)
  inputs = reward_tokenizer(
      prompts,
      max_length=tokenizer.model_max_length,
      padding="do_not_pad",
      truncation=True,
  )
  input_ids = inputs.input_ids
  padded_tokens = reward_tokenizer.pad(
      {"input_ids": input_ids}, padding=True, return_tensors="pt"
  )
  txt_emb = reward_clip_model.get_text_features(
      input_ids=padded_tokens.input_ids.to("cuda").unsqueeze(0)
  )
  return blip_reward.cpu().squeeze(0).squeeze(0), txt_emb.squeeze(0)


def _calculate_reward_custom(
    pipe,
    _,
    reward_tokenizer,
    tokenizer,
    weight_dtype,
    reward_clip_model,
    reward_processor,
    reward_model,
    imgs,
    prompts,
    test_flag=False,
):
  """Computes reward using custom reward model."""
  # img
  if test_flag:
    image_pil = imgs
  else:
    image_pil = pipe.numpy_to_pil(imgs)[0]
  pixels = (
      reward_processor(images=image_pil.convert("RGB"), return_tensors="pt")
      .pixel_values.to(weight_dtype)
      .to("cuda")
  )
  img_emb = reward_clip_model.get_image_features(pixels)
  # prompt
  inputs = reward_tokenizer(
      prompts,
      max_length=tokenizer.model_max_length,
      padding="do_not_pad",
      truncation=True,
  )
  input_ids = inputs.input_ids
  padded_tokens = reward_tokenizer.pad(
      {"input_ids": input_ids}, padding=True, return_tensors="pt"
  )
  txt_emb = reward_clip_model.get_text_features(
      input_ids=padded_tokens.input_ids.to("cuda").unsqueeze(0)
  )
  score = reward_model(txt_emb, img_emb)
  return score.to(weight_dtype).squeeze(0).squeeze(0), txt_emb.squeeze(0)


def _get_batch(data_iter_loader, data_iterator, prompt_list, args, accelerator):
  """Creates a batch."""
  batch = next(data_iter_loader, None)
  if batch is None:
    batch = next(
        iter(
            accelerator.prepare(
                data_iterator(prompt_list, batch_size=args.g_batch_size)
            )
        )
    )

  if args.single_flag == 1:
    for i in range(len(batch)):
      batch[i] = args.single_prompt

  batch_list = []
  for i in range(len(batch)):
    batch_list.extend([batch[i] for _ in range(args.num_samples)])
  batch = batch_list
  return batch


def _trim_buffer(buffer_size, state_dict):
  """Delete old samples from the bufffer."""
  if state_dict["state"].shape[0] > buffer_size:
    state_dict["prompt"] = state_dict["prompt"][-buffer_size:]
    state_dict["state"] = state_dict["state"][-buffer_size:]
    state_dict["next_state"] = state_dict["next_state"][-buffer_size:]
    state_dict["timestep"] = state_dict["timestep"][-buffer_size:]
    state_dict["final_reward"] = state_dict["final_reward"][-buffer_size:]
    state_dict["unconditional_prompt_embeds"] = state_dict[
        "unconditional_prompt_embeds"
    ][-buffer_size:]
    state_dict["guided_prompt_embeds"] = state_dict["guided_prompt_embeds"][
        -buffer_size:
    ]
    state_dict["txt_emb"] = state_dict["txt_emb"][-buffer_size:]
    state_dict["log_prob"] = state_dict["log_prob"][-buffer_size:]


def _save_model(args, count, is_ddp, accelerator, unet):
  """Saves UNET model."""
  save_path = os.path.join(args.output_dir, f"save_{count}")
  print(f"Saving model to {save_path}")
  if is_ddp:
    unet_to_save = copy.deepcopy(accelerator.unwrap_model(unet)).to(
        torch.float32
    )
    unet_to_save.save_attn_procs(save_path)
  else:
    unet_to_save = copy.deepcopy(unet).to(torch.float32)
    unet_to_save.save_attn_procs(save_path)


def _collect_rollout(args, pipe, is_ddp, batch, calculate_reward, state_dict):
  """Collects trajectories."""
  for _ in range(args.g_step):
    # samples for each prompt
    # collect the rollout data from the custom sampling function
    # (modified in pipeline_stable_diffusion.py and scheduling_ddim.py)
    with torch.no_grad():
      (
          image,
          latents_list,
          unconditional_prompt_embeds,
          guided_prompt_embeds,
          log_prob_list,
          _,
      ) = pipe.forward_collect_traj_ddim(prompt=batch, is_ddp=is_ddp)
      reward_list = []
      txt_emb_list = []
      for i in range(len(batch)):
        reward, txt_emb = calculate_reward(image[i], batch[i])
        reward_list.append(reward)
        txt_emb_list.append(txt_emb)
      reward_list = torch.stack(reward_list).detach().cpu()
      txt_emb_list = torch.stack(txt_emb_list).detach().cpu()
      # store the rollout data
      for i in range(len(latents_list) - 1):
        # deal with a batch of data in each step i
        state_dict["prompt"].extend(batch)
        state_dict["state"] = torch.cat((state_dict["state"], latents_list[i]))
        state_dict["next_state"] = torch.cat(
            (state_dict["next_state"], latents_list[i + 1])
        )
        state_dict["timestep"] = torch.cat(
            (state_dict["timestep"], torch.LongTensor([i] * len(batch)))
        )
        state_dict["final_reward"] = torch.cat(
            (state_dict["final_reward"], reward_list)
        )
        state_dict["unconditional_prompt_embeds"] = torch.cat((
            state_dict["unconditional_prompt_embeds"],
            unconditional_prompt_embeds,
        ))
        state_dict["guided_prompt_embeds"] = torch.cat(
            (state_dict["guided_prompt_embeds"], guided_prompt_embeds)
        )
        state_dict["txt_emb"] = torch.cat((state_dict["txt_emb"], txt_emb_list))
        state_dict["log_prob"] = torch.cat(
            (state_dict["log_prob"], log_prob_list[i])
        )
      del (
          image,
          latents_list,
          unconditional_prompt_embeds,
          guided_prompt_embeds,
          reward_list,
          txt_emb_list,
          log_prob_list,
          reward,
          txt_emb,
      )
      torch.cuda.empty_cache()


def _train_value_func(value_function, state_dict, accelerator, args):
  """Trains the value function."""
  indices = get_random_indices(state_dict["state"].shape[0], args.v_batch_size)
  # permutation = torch.randperm(state_dict['state'].shape[0])
  # indices = permutation[:v_batch_size]
  batch_state = state_dict["state"][indices]
  batch_timestep = state_dict["timestep"][indices]
  batch_final_reward = state_dict["final_reward"][indices]
  batch_txt_emb = state_dict["txt_emb"][indices]
  pred_value = value_function(
      batch_state.cuda().detach(),
      batch_txt_emb.cuda().detach(),
      batch_timestep.cuda().detach()
  )
  batch_final_reward = batch_final_reward.cuda().float()
  value_loss = F.mse_loss(
      pred_value.float().reshape([args.v_batch_size, 1]),
      batch_final_reward.cuda().detach().reshape([args.v_batch_size, 1]))
  accelerator.backward(value_loss/args.v_step)
  del pred_value
  del batch_state
  del batch_timestep
  del batch_final_reward
  del batch_txt_emb
  return (value_loss.item() / args.v_step)


@dataclasses.dataclass(frozen=False)
class TrainPolicyFuncData:
  tot_p_loss: float = 0
  tot_ratio: float = 0
  tot_kl: float = 0
  tot_grad_norm: float = 0


def _train_policy_func(
    args,
    state_dict,
    pipe,
    unet_copy,
    is_ddp,
    count,
    policy_steps,
    accelerator,
    tpfdata,
    value_function
):
  """Trains the policy function."""
  with torch.no_grad():
    indices = get_random_indices(
        state_dict["state"].shape[0], args.p_batch_size
    )
    batch_state = state_dict["state"][indices]
    batch_next_state = state_dict["next_state"][indices]
    batch_timestep = state_dict["timestep"][indices]
    batch_final_reward = state_dict["final_reward"][indices]
    batch_unconditional_prompt_embeds = state_dict[
        "unconditional_prompt_embeds"
    ][indices]
    batch_guided_prompt_embeds = state_dict["guided_prompt_embeds"][indices]
    batch_promt_embeds = torch.cat(
        [batch_unconditional_prompt_embeds, batch_guided_prompt_embeds]
    )
    batch_txt_emb = state_dict["txt_emb"][indices]
    batch_log_prob = state_dict["log_prob"][indices]
  # calculate loss from the custom function
  # (modified in pipeline_stable_diffusion.py and scheduling_ddim.py)
  log_prob, kl_regularizer = pipe.forward_calculate_logprob(
      prompt_embeds=batch_promt_embeds.cuda(),
      latents=batch_state.cuda(),
      next_latents=batch_next_state.cuda(),
      ts=batch_timestep.cuda(),
      unet_copy=unet_copy,
      is_ddp=is_ddp,
  )
  with torch.no_grad():
    if args.v_flag == 1:
      # pylint: disable=line-too-long
      adv = batch_final_reward.cuda().reshape([args.p_batch_size, 1]) - value_function(
          batch_state.cuda(),
          batch_txt_emb.cuda(),
          batch_timestep.cuda()).reshape([args.p_batch_size, 1])
    else:
      adv = batch_final_reward.cuda().reshape([args.p_batch_size, 1])
  ratio = torch.exp(log_prob - batch_log_prob.cuda())
  ratio = torch.clamp(ratio, 1.0 - args.ratio_clip, 1.0 + args.ratio_clip)
  loss = (
      -args.reward_weight
      * adv.detach().float()
      * ratio.float().reshape([args.p_batch_size, 1])
  ).mean()
  if count > args.kl_warmup:
    loss += args.kl_weight * kl_regularizer.mean()
  loss = loss / (args.gradient_accumulation_steps)
  accelerator.backward(loss)
  # logging
  tpfdata.tot_ratio += ratio.mean().item() / policy_steps
  tpfdata.tot_kl += kl_regularizer.mean().item() / policy_steps
  tpfdata.tot_p_loss += loss.item() / policy_steps


def main():
  args = parse_args()
  if args.non_ema_revision is not None:
    deprecate(
        "non_ema_revision!=None",
        "0.15.0",
        message=(
            "Downloading 'non_ema' weights from revision branches of the Hub is"
            " deprecated. Please make sure to use `--variant=non_ema` instead."
        ),
    )
  # Change log dir
  _update_output_dir(args)
  logging_dir = os.path.join(args.output_dir, args.logging_dir)

  accelerator_project_config = ProjectConfiguration(
      logging_dir=logging_dir, total_limit=args.checkpoints_total_limit
  )
  accelerator = Accelerator(
      mixed_precision=args.mixed_precision,
      log_with=args.report_to,
      project_config=accelerator_project_config,
  )

  # Make one log on every process with the configuration for debugging.
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
  )
  logger.info(accelerator.state, main_process_only=False)
  if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
  else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

  # If passed along, set the training seed now.
  if args.seed is not None:
    set_seed(args.seed)

  # Handle the repository creation
  if accelerator.is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)

  # Load scheduler, tokenizer and models.
  tokenizer = CLIPTokenizer.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="tokenizer",
      revision=args.revision,
  )
  text_encoder = CLIPTextModel.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="text_encoder",
      revision=args.revision,
  )

  weight_dtype = torch.float32
  if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
  elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

  # reward models
  reward_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
  reward_processor = CLIPProcessor.from_pretrained(
      "openai/clip-vit-large-patch14"
  )
  reward_tokenizer = CLIPTokenizer.from_pretrained(
      "openai/clip-vit-large-patch14"
  )
  if args.reward_flag == 0:
    image_reward = imagereward.load("ImageReward-v1.0")
    image_reward.requires_grad_(False)
    image_reward.to(accelerator.device, dtype=weight_dtype)
  else:
    reward_model = pickle.load(open(args.reward_model_path, "rb"))["reward"]
    reward_model.requires_grad_(False)
    reward_model.to(accelerator.device, dtype=weight_dtype)

  reward_clip_model.requires_grad_(False)

  if args.sft_initialization == 0:
    pipe = StableDiffusionPipelineExtended.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
    )
  else:
    pipe = StableDiffusionPipelineExtended.from_pretrained(
        args.sft_path, torch_dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.sft_path, subfolder="unet", revision=args.non_ema_revision
    )

  pipe.scheduler = DDIMSchedulerExtended.from_config(pipe.scheduler.config)
  vae = pipe.vae
  unet.requires_grad_(False)
  unet.eval()
  text_encoder = pipe.text_encoder

  # Freeze vae and text_encoder
  vae.requires_grad_(False)
  text_encoder.requires_grad_(False)
  # pretrain model to calculate kl
  unet_copy = UNet2DConditionModel.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="unet",
      revision=args.non_ema_revision,
  )
  # freeze unet copy
  unet_copy.requires_grad_(False)
  # Move text_encode and vae to gpu and cast to weight_dtype
  text_encoder.to(accelerator.device, dtype=weight_dtype)
  vae.to(accelerator.device, dtype=weight_dtype)
  unet.to(accelerator.device, dtype=weight_dtype)
  unet_copy.to(accelerator.device, dtype=weight_dtype)

  # Create EMA for the unet.
  if args.use_ema:
    ema_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    ema_unet = EMAModel(
        ema_unet.parameters(),
        model_cls=UNet2DConditionModel,
        model_config=ema_unet.config,
    )

  if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
      import xformers  # pylint: disable=g-import-not-at-top

      xformers_version = version.parse(xformers.__version__)
      if xformers_version == version.parse("0.0.16"):
        logger.warn(
            "xFormers 0.0.16 cannot be used for training in some GPUs. If you"
            " observe problems during training, please update xFormers to at"
            " least 0.0.17. See"
            " https://huggingface.co/docs/diffusers/main/en/optimization/xformers"
            " for more details."
        )
      unet.enable_xformers_memory_efficient_attention()
    else:
      raise ValueError(
          "xformers is not available. Make sure it is installed correctly"
      )

  # Define lora layers
  lora_attn_procs = {}
  for name in unet.attn_processors.keys():
    cross_attention_dim = (
        None
        if name.endswith("attn1.processor")
        else unet.config.cross_attention_dim
    )
    if name.startswith("mid_block"):
      hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
      block_id = int(name[len("up_blocks.")])
      hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
      block_id = int(name[len("down_blocks.")])
      hidden_size = unet.config.block_out_channels[block_id]

    lora_attn_procs[name] = LoRACrossAttnProcessor(
        hidden_size=hidden_size,
        cross_attention_dim=cross_attention_dim,
        rank=args.lora_rank,
    )

  unet.set_attn_processor(lora_attn_procs)
  lora_layers = AttnProcsLayers(unet.attn_processors)

  # Enable TF32 for faster training on Ampere GPUs,
  # cf https://pytorch.org/docs/stable/notes/cuda.
  # html#tensorfloat-32-tf32-on-ampere-devices
  if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()

  if args.scale_lr:
    args.learning_rate = (
        args.learning_rate
        * args.gradient_accumulation_steps
        * args.train_batch_size
        * accelerator.num_processes
    )

  # Initialize the optimizer
  if args.use_8bit_adam:
    try:
      import bitsandbytes as bnb  # pylint: disable=g-import-not-at-top
    except ImportError as exc:
      raise ImportError(
          "Please install bitsandbytes to use 8-bit Adam. You can do so by"
          " running `pip install bitsandbytes`"
      ) from exc

    optimizer_cls = bnb.optim.AdamW8bit
  else:
    optimizer_cls = torch.optim.AdamW

  optimizer = optimizer_cls(
      lora_layers.parameters(),
      lr=args.learning_rate,
      betas=(args.adam_beta1, args.adam_beta2),
      weight_decay=args.adam_weight_decay,
      eps=args.adam_epsilon,
  )

  # Get the datasets: you can either provide your own training and evaluation
  # files (see below) or specify a Dataset from the hub (the dataset will be
  # downloaded automatically from the datasets Hub).

  # In distributed training, the load_dataset function guarantees that only one
  # local process can concurrently download the dataset.
  if args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
    )
  else:
    with open(args.prompt_path) as json_file:
      prompt_dict = json.load(json_file)
    if args.prompt_category != "all":
      prompt_category = [e for e in args.prompt_category.split(",")]
    prompt_list = []
    for prompt in prompt_dict:
      category = prompt_dict[prompt]["category"]
      if args.prompt_category != "all":
        if category in prompt_category:
          prompt_list.append(prompt)
      else:
        prompt_list.append(prompt)

  # Data iterator for prompt dataset
  def _my_data_iterator(data, batch_size):
    # Shuffle the data randomly
    random.shuffle(data)

    for i in range(0, len(data), batch_size):
      batch = data[i : i + batch_size]
      yield batch

  data_iterator = _my_data_iterator(prompt_list, batch_size=args.g_batch_size)
  data_iterator = accelerator.prepare(data_iterator)

  lr_scheduler = get_scheduler(
      args.lr_scheduler,
      optimizer=optimizer,
      num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
      num_training_steps=args.max_train_steps
      * args.gradient_accumulation_steps,
  )
  value_function = ValueMulti(50, (4, 64, 64))
  value_optimizer = torch.optim.AdamW(value_function.parameters(), lr=args.v_lr)
  value_function, value_optimizer = accelerator.prepare(
      value_function, value_optimizer
  )

  # Prepare everything with our `accelerator`.
  if args.multi_gpu:
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )
  else:
    lora_layers, optimizer, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, lr_scheduler
    )
  if args.use_ema:
    ema_unet.to(accelerator.device)

  reward_clip_model.to(accelerator.device, dtype=weight_dtype)

  if accelerator.is_main_process:
    accelerator.init_trackers("text2image-fine-tune", config=vars(args))

  # Train!
  total_batch_size = (
      args.train_batch_size
      * accelerator.num_processes
      * args.gradient_accumulation_steps
  )
  global_step = 0
  if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest":
      path = os.path.basename(args.resume_from_checkpoint)
    else:
      # Get the most recent checkpoint
      dirs = os.listdir(args.output_dir)
      dirs = [d for d in dirs if d.startswith("checkpoint")]
      dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
      path = dirs[-1] if len(dirs) > 0 else None  # pylint: disable=g-explicit-length-test

    if path is None:
      accelerator.print(
          f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting"
          " a new training run."
      )
      args.resume_from_checkpoint = None
    else:
      accelerator.print(f"Resuming from checkpoint {path}")
      accelerator.load_state(os.path.join(args.output_dir, path))
      global_step = int(path.split("-")[1])

  logger.info("***** Running training *****")
  logger.info(f"  Num Epochs = {args.num_train_epochs}")
  logger.info(
      f"  Instantaneous batch size per device = {args.train_batch_size}"
  )
  logger.info(
      "  Total train batch size (w. parallel, distributed & accumulation) ="
      f" {total_batch_size}"
  )
  logger.info(
      f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
  )
  logger.info(
      f"  Total optimization steps = {args.max_train_steps // args.p_step}"
  )

  pipe.scheduler = DDIMSchedulerExtended.from_config(pipe.scheduler.config)
  pipe.to("cuda")

  # Only show the progress bar once on each machine.
  progress_bar = tqdm(
      range(global_step, args.max_train_steps),
      disable=not accelerator.is_local_main_process,
  )
  progress_bar.set_description("Steps")

  def _map_cpu(x):
    return x.cpu()

  state_dict = {}
  state_dict["prompt"] = []
  state_dict["state"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["next_state"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["timestep"] = _map_cpu(torch.LongTensor())
  state_dict["final_reward"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["unconditional_prompt_embeds"] = _map_cpu(
      torch.FloatTensor().to(weight_dtype)
  )
  state_dict["guided_prompt_embeds"] = _map_cpu(
      torch.FloatTensor().to(weight_dtype)
  )
  state_dict["txt_emb"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["log_prob"] = _map_cpu(torch.FloatTensor().to(weight_dtype))

  if args.reward_flag == 0:
    calculate_reward = functools.partial(
        _calculate_reward_ir,
        pipe,
        args,
        reward_tokenizer,
        tokenizer,
        weight_dtype,
        reward_clip_model,
        image_reward,
    )
  else:
    calculate_reward = functools.partial(
        _calculate_reward_custom,
        pipe,
        args,
        reward_tokenizer,
        tokenizer,
        weight_dtype,
        reward_clip_model,
        reward_processor,
        reward_model,
    )

  count = 0
  buffer_size = args.buffer_size
  policy_steps = args.gradient_accumulation_steps * args.p_step
  # test_batch = get_test_prompts(args.prompt_category)
  data_iter_loader = iter(data_iterator)
  is_ddp = isinstance(unet, DistributedDataParallel)
  pipe.unet = unet
  print("model is parallel:", is_ddp)

  for count in range(0, args.max_train_steps // args.p_step):
    # fix batchnorm
    unet.eval()

    batch = _get_batch(
        data_iter_loader, _my_data_iterator, prompt_list, args, accelerator
    )
    _collect_rollout(args, pipe, is_ddp, batch, calculate_reward, state_dict)
    _trim_buffer(buffer_size, state_dict)

    if args.v_flag == 1:
      tot_val_loss = 0
      value_optimizer.zero_grad()
      for v_step in range(args.v_step):
        if v_step < args.v_step-1:
          with accelerator.no_sync(value_function):
            tot_val_loss += _train_value_func(
                value_function, state_dict, accelerator, args
            )
        else:
          tot_val_loss += _train_value_func(
              value_function, state_dict, accelerator, args
          )
      value_optimizer.step()
      value_optimizer.zero_grad()
      if accelerator.is_main_process:
        print("value_loss", tot_val_loss)
        accelerator.log({"value_loss": tot_val_loss}, step=count)
      del tot_val_loss
      torch.cuda.empty_cache()

    # policy learning
    tpfdata = TrainPolicyFuncData()
    for _ in range(args.p_step):
      optimizer.zero_grad()
      for accum_step in range(int(args.gradient_accumulation_steps)):
        if accum_step < int(args.gradient_accumulation_steps) - 1:
          with accelerator.no_sync(unet):
            _train_policy_func(
                args,
                state_dict,
                pipe,
                unet_copy,
                is_ddp,
                count,
                policy_steps,
                accelerator,
                tpfdata,
                value_function,
            )
        else:
          _train_policy_func(
              args,
              state_dict,
              pipe,
              unet_copy,
              is_ddp,
              count,
              policy_steps,
              accelerator,
              tpfdata,
              value_function
          )
      if accelerator.sync_gradients:
        norm = accelerator.clip_grad_norm_(unet.parameters(), args.clip_norm)
      tpfdata.tot_grad_norm += norm.item() / args.p_step
      optimizer.step()
      lr_scheduler.step()
      if accelerator.is_main_process:
        print(f"count: [{count} / {args.max_train_steps // args.p_step}]")
        print("train_reward", torch.mean(state_dict["final_reward"]).item())
        accelerator.log(
            {"train_reward": torch.mean(state_dict["final_reward"]).item()},
            step=count,
        )
        print("grad norm", tpfdata.tot_grad_norm, "ratio", tpfdata.tot_ratio)
        print("kl", tpfdata.tot_kl, "p_loss", tpfdata.tot_p_loss)
        accelerator.log({"grad norm": tpfdata.tot_grad_norm}, step=count)
        accelerator.log({"ratio": tpfdata.tot_ratio}, step=count)
        accelerator.log({"kl": tpfdata.tot_kl}, step=count)
        accelerator.log({"p_loss": tpfdata.tot_p_loss}, step=count)
      torch.cuda.empty_cache()

    if accelerator.sync_gradients:
      global_step += 1
      if global_step % args.checkpointing_steps == 0:
        if accelerator.is_main_process:
          save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
          accelerator.save_state(output_dir=save_path)
          logger.info(f"Saved state to {save_path}")
      print("global_step", global_step)

    # Save model per interval
    if count % args.save_interval == 0:
      accelerator.wait_for_everyone()
      if accelerator.is_main_process:
        _save_model(args, count, is_ddp, accelerator, unet)

  # Create the pipeline using the trained modules and save it.
  accelerator.wait_for_everyone()
  if accelerator.is_main_process:
    _save_model(args, count, is_ddp, accelerator, unet)

  accelerator.end_training()


if __name__ == "__main__":
  main()
