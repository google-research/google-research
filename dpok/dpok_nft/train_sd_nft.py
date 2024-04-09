import torch
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import numpy as np
from tqdm import tqdm
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
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel
import torch.utils.checkpoint
from tqdm.auto import tqdm
import transformers
from transformers import CLIPModel, CLIPProcessor  # pylint: disable=g-multiple-import
from transformers import CLIPTextModel, CLIPTokenizer  # pylint: disable=g-multiple-import
import utils
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
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
        "--output_dir",
        type=str,
        default="online_model",
        help=(
              "The output directory where the model predictions and checkpoints"
            " will be written."
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
        "--sft_path",
        type=str,
        default="./checkpoints/models/finetune_b512_lr2e-05_max10000_w0.01",
        help="path to the pretrained supervised finetuned model",
    )
    
    
    
    
    return parser.parse_args()


args = parse_arguments()
device = "cuda" if torch.cuda.is_available() else "cpu"
diffusion_model_id = "CompVis/stable-diffusion-v1-4"
clip_model_id = "openai/clip-vit-base-patch32"
logging_dir = os.path.join(args.output_dir, args.logging_dir)
accelerator_project_config = ProjectConfiguration(
    logging_dir=logging_dir, total_limit=args.checkpoints_total_limit
)
num_epochs = args.num_epochs
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
)

def load_dataset(train_data_dir):
    dataset = None
    return dataset


weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)

pipe = StableDiffusionPipelineExtended.from_pretrained(
    args.sft_path, torch_dtype=weight_dtype
)
unet = UNet2DConditionModel.from_pretrained(
    args.sft_path, subfolder="unet", revision=args.non_ema_revision
)
image_reward = imagereward.load("ImageReward-v1.0")
image_reward.requires_grad_(False)
image_reward.to(accelerator.device, dtype=weight_dtype)
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
text_encoder.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)
unet.to(accelerator.device, dtype=weight_dtype)
unet_copy.to(accelerator.device, dtype=weight_dtype)
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


optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=args.learning_rate)
dataset = load_dataset(args.train_dir)
def calculate_image_reward(pipe, args, reward_tokenizer, tokenizer, weight_dtype, reward_clip_model, image_reward, images, prompts):
    #TODO add rarity score here analize how should it work ?
    image_pil = pipe.numpy_to_pil(images)[0]
    blip_reward, _ = utils.image_reward_get_reward(image_reward, image_pil, prompts, weight_dtype)
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

# Example training loop
for epoch in range(num_epochs):
    # This is a simplification. You should load your actual data here.
    text_prompts = ["A digital painting of a landscape", "A digital painting of a cat"]
    for text in tqdm(text_prompts, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        
        # Generate image from text
        with torch.no_grad():
            generated = diffusion_model([text])
            image = generated.images[0]

        # Calculate reward
        reward = calculate_image_reward(image, text)

        # Example: Define a simple loss based on the reward. This is for illustration; 
        # you'd want a more sophisticated approach.
        loss = 1.0 - torch.tensor(reward, requires_grad=True)
        loss.backward()
        optimizer.step()

        print(f"Processed text: {text}, Reward: {reward}")

# Save the fine-tuned model
torch.save(diffusion_model.unet.state_dict(), "fine_tuned_model.pth")
