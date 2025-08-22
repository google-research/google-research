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

# coding=utf-8

"""Runs the main codi training algorithm."""
# pylint: disable=g-importing-member,missing-function-docstring,missing-class-docstring,logging-fstring-interpolation, g-import-not-at-top, unused-argument, unused-variable

import logging
import math
import os
import random
import time
from typing import Any, Dict

from args import parse_args
from datasets import load_dataset
from datasets import load_from_disk
from diffusers import FlaxAutoencoderKL
from diffusers import FlaxControlNetModel
from diffusers import FlaxDDPMScheduler
from diffusers import FlaxStableDiffusionControlNetPipeline
from diffusers import FlaxUNet2DConditionModel
from diffusers.schedulers.scheduling_utils_flax import broadcast_to_shape_from_left
from diffusers.schedulers.scheduling_utils_flax import get_sqrt_alpha_prod
from diffusers.utils import check_min_version
from diffusers.utils import is_wandb_available
from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from flax.training import train_state
from flax.training.common_utils import shard
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
from PIL import PngImagePlugin
import torch
import torch.utils.checkpoint
from torch.utils.data import IterableDataset
from torchvision import transforms
from tqdm.auto import tqdm
import transformers
from transformers import CLIPTokenizer
from transformers import FlaxCLIPTextModel
from transformers import set_seed

# To prevent an error that occurs when there are abnormally
# large compressed data chunk in the png image
# see more https://github.com/python-pillow/Pillow/issues/5610
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

if is_wandb_available():
  import wandb

# Will error if the minimal version of diffusers is not installed.
# Remove at your own risks.
check_min_version("0.16.0.dev0")

logger = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
  ema_params: Dict[str, Any]


def scalings_for_boundary_conditions(
    timestep, sigma_data=0.5, timestep_scaling=1
):
  scaled_timestep = timestep * timestep_scaling

  c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
  c_out = (sigma_data * scaled_timestep) / (
      scaled_timestep**2 + sigma_data**2
  ) ** 0.5
  return c_skip, c_out


def image_grid(imgs, rows, cols):
  assert len(imgs) == rows * cols

  w, h = imgs[0].size
  grid = Image.new("RGB", size=(cols * w, rows * h))

  for i, img in enumerate(imgs):
    grid.paste(img, box=(i % cols * w, i // cols * h))
  return grid


def log_validation(
    pipeline,
    pipeline_params,
    controlnet_params,
    tokenizer,
    args,
    rng,
    weight_dtype,
):
  """log validation."""
  logger.info("Running validation...")

  pipeline_params = pipeline_params.copy()
  pipeline_params["controlnet"] = controlnet_params

  num_samples = jax.device_count()
  prng_seed = jax.random.split(rng, jax.device_count())

  if len(args.validation_image) == len(args.validation_prompt):
    validation_images = args.validation_image
    validation_prompts = args.validation_prompt
  elif len(args.validation_image) == 1:
    validation_images = args.validation_image * len(args.validation_prompt)
    validation_prompts = args.validation_prompt
  elif len(args.validation_prompt) == 1:
    validation_images = args.validation_image
    validation_prompts = args.validation_prompt * len(args.validation_image)
  else:
    raise ValueError(
        "number of `args.validation_image` and `args.validation_prompt` should"
        " be checked in `parse_args`"
    )

  image_logs = []

  for validation_prompt, validation_image in zip(
      validation_prompts, validation_images
  ):
    prompts = num_samples * [validation_prompt]
    prompt_ids = pipeline.prepare_text_inputs(prompts)
    prompt_ids = shard(prompt_ids)

    validation_image = Image.open(validation_image).convert("RGB")
    processed_image = pipeline.prepare_image_inputs(
        num_samples * [validation_image]
    )
    processed_image = shard(processed_image)
    images = pipeline(
        prompt_ids=prompt_ids,
        image=processed_image,
        params=pipeline_params,
        prng_seed=prng_seed,
        num_inference_steps=50,
        jit=True,
    ).images

    images = images.reshape(
        (images.shape[0] * images.shape[1],) + images.shape[-3:]
    )
    images = pipeline.numpy_to_pil(images)

    image_logs.append({
        "validation_image": validation_image,
        "images": images,
        "validation_prompt": validation_prompt,
    })

  if args.report_to == "wandb":
    formatted_images = []
    for log in image_logs:
      images = log["images"]
      validation_prompt = log["validation_prompt"]
      validation_image = log["validation_image"]

      formatted_images.append(
          wandb.Image(validation_image, caption="Controlnet conditioning")
      )
      for image in images:
        image = wandb.Image(image, caption=validation_prompt)
        formatted_images.append(image)

    wandb.log({"validation": formatted_images})
  else:
    logger.info(f"image logging not implemented for {args.report_to}")

  return image_logs


def make_train_dataset(args, tokenizer, batch_size=None):
  if args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )
  else:
    if args.train_data_dir is not None:
      if args.load_from_disk:
        dataset = load_from_disk(
            args.train_data_dir,
        )
      else:
        dataset = load_dataset(
            args.train_data_dir,
            cache_dir=args.cache_dir,
        )
    else:
      raise NotImplementedError("train_data_dir is not set")
    # See more about loading custom images at
    # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

  # Preprocessing the datasets.
  # We need to tokenize inputs and targets.
  if isinstance(dataset["train"], IterableDataset):
    column_names = next(iter(dataset["train"])).keys()
  else:
    column_names = dataset["train"].column_names

  # 6. Get the column names for input/target.
  if args.image_column is None:
    image_column = column_names[0]
    logger.info(f"image column defaulting to {image_column}")
  else:
    image_column = args.image_column
    if image_column not in column_names:
      raise ValueError(
          f"`--image_column` value '{args.image_column}' not found in dataset"
          f" columns. Dataset columns are: {', '.join(column_names)}"
      )

  if args.caption_column is None:
    caption_column = column_names[1]
    logger.info(f"caption column defaulting to {caption_column}")
  else:
    caption_column = args.caption_column
    if caption_column not in column_names:
      raise ValueError(
          f"`--caption_column` value '{args.caption_column}' not found in"
          f" dataset columns. Dataset columns are: {', '.join(column_names)}"
      )

  if args.conditioning_image_column is None:
    conditioning_image_column = column_names[2]
    logger.info(f"conditioning image column defaulting to {caption_column}")
  else:
    conditioning_image_column = args.conditioning_image_column
    if conditioning_image_column not in column_names:
      raise ValueError(
          "`--conditioning_image_column` value"
          f" '{args.conditioning_image_column}' not found in dataset columns."
          f" Dataset columns are: {', '.join(column_names)}"
      )

  def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
      if random.random() < args.proportion_empty_prompts:
        captions.append("")
      elif isinstance(caption, str):
        captions.append(caption)
      elif isinstance(caption, (list, np.ndarray)):
        # take a random caption if there are multiple
        captions.append(random.choice(caption) if is_train else caption[0])
      else:
        raise ValueError(
            f"Caption column `{caption_column}` should contain either strings"
            " or lists of strings."
        )
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

  image_transforms = transforms.Compose([
      transforms.Resize(
          args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
      ),
      transforms.CenterCrop(args.resolution),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]),
  ])

  conditioning_image_transforms = transforms.Compose([
      transforms.Resize(
          args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
      ),
      transforms.CenterCrop(args.resolution),
      transforms.ToTensor(),
  ])

  def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    images = [image_transforms(image) for image in images]

    conditioning_images = [
        image.convert("RGB") for image in examples[conditioning_image_column]
    ]
    conditioning_images = [
        conditioning_image_transforms(image) for image in conditioning_images
    ]

    examples["pixel_values"] = images
    examples["conditioning_pixel_values"] = conditioning_images
    examples["input_ids"] = tokenize_captions(examples)

    return examples

  if jax.process_index() == 0:
    if args.max_train_samples is not None:
      if args.streaming:
        dataset["train"] = (
            dataset["train"]
            .shuffle(seed=args.seed)
            .take(args.max_train_samples)
        )
      else:
        dataset["train"] = (
            dataset["train"]
            .shuffle(seed=args.seed)
            .select(range(args.max_train_samples))
        )
    # Set the training transforms
    if args.streaming:
      train_dataset = dataset["train"].map(
          preprocess_train,
          batched=True,
          batch_size=batch_size,
          remove_columns=list(dataset["train"].features.keys()),
      )
    else:
      train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
  """Collect distributed variables."""
  pixel_values = torch.stack([example["pixel_values"] for example in examples])
  pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

  conditioning_pixel_values = torch.stack(
      [example["conditioning_pixel_values"] for example in examples]
  )
  conditioning_pixel_values = conditioning_pixel_values.to(
      memory_format=torch.contiguous_format
  ).float()

  input_ids = torch.stack([example["input_ids"] for example in examples])

  batch = {
      "pixel_values": pixel_values,
      "conditioning_pixel_values": conditioning_pixel_values,
      "input_ids": input_ids,
  }
  batch = {k: v.numpy() for k, v in batch.items()}
  return batch


def get_params_to_save(params):
  return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main():
  """Main function."""
  args = parse_args()

  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
  )
  logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
  if jax.process_index() == 0:
    transformers.utils.logging.set_verbosity_info()
  else:
    transformers.utils.logging.set_verbosity_error()

  # wandb init
  if jax.process_index() == 0 and args.report_to == "wandb":
    wandb.init(
        entity=args.wandb_entity,
        project=args.tracker_project_name,
        job_type="train",
        config=args,
    )

  if args.seed is not None:
    set_seed(args.seed)

  rng = jax.random.PRNGKey(0)

  # Handle the repository creation
  if jax.process_index() == 0:
    if args.output_dir is not None:
      os.makedirs(args.output_dir, exist_ok=True)

  # Load the tokenizer and add the placeholder token as a additional token
  if args.tokenizer_name:
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
  elif args.pretrained_model_name_or_path:
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
  else:
    raise NotImplementedError("No tokenizer specified!")

  # Get the datasets: you can either provide your own training and evaluation
  total_train_batch_size = (
      args.train_batch_size
      * jax.local_device_count()
      * args.gradient_accumulation_steps
  )
  train_dataset = make_train_dataset(
      args, tokenizer, batch_size=total_train_batch_size
  )

  train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      shuffle=not args.streaming,
      collate_fn=collate_fn,
      batch_size=total_train_batch_size,
      num_workers=args.dataloader_num_workers,
      drop_last=True,
  )

  if args.mixed_precision == "fp16":
    weight_dtype = jnp.float16
  elif args.mixed_precision == "bf16":
    weight_dtype = jnp.bfloat16
  else:
    weight_dtype = jnp.float32

  # Load models and create wrapper for stable diffusion
  text_encoder = FlaxCLIPTextModel.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="text_encoder",
      dtype=weight_dtype,
      revision=args.revision,
      from_pt=args.from_pt,
  )
  vae, vae_params = FlaxAutoencoderKL.from_pretrained(
      args.pretrained_model_name_or_path,
      revision=args.revision,
      subfolder="vae",
      dtype=weight_dtype,
      from_pt=args.from_pt,
  )
  unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="unet",
      dtype=weight_dtype,
      revision=args.revision,
      from_pt=args.from_pt,
  )

  if args.controlnet_model_name_or_path:
    logger.info("Loading existing controlnet weights")
    controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path,
        revision=args.controlnet_revision,
        from_pt=args.controlnet_from_pt,
        dtype=jnp.float32,
    )
  else:
    logger.info("Initializing controlnet weights from unet")
    rng, rng_params = jax.random.split(rng)

    controlnet = FlaxControlNetModel(
        in_channels=unet.config.in_channels,
        down_block_types=unet.config.down_block_types,
        only_cross_attention=unet.config.only_cross_attention,
        block_out_channels=unet.config.block_out_channels,
        layers_per_block=unet.config.layers_per_block,
        attention_head_dim=unet.config.attention_head_dim,
        cross_attention_dim=unet.config.cross_attention_dim,
        use_linear_projection=unet.config.use_linear_projection,
        flip_sin_to_cos=unet.config.flip_sin_to_cos,
        freq_shift=unet.config.freq_shift,
    )
    controlnet_params = controlnet.init_weights(rng=rng_params)
    controlnet_params = unfreeze(controlnet_params)
    for key in [
        "conv_in",
        "time_embedding",
        "down_blocks_0",
        "down_blocks_1",
        "down_blocks_2",
        "down_blocks_3",
        "mid_block",
    ]:
      controlnet_params[key] = unet_params[key]

    ema_controlnet_params = jax.tree.map(jnp.array, controlnet_params)

    pipeline, pipeline_params = (
        FlaxStableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            tokenizer=tokenizer,
            controlnet=controlnet,
            safety_checker=None,
            dtype=weight_dtype,
            revision=args.revision,
            from_pt=args.from_pt,
        )
    )
    pipeline_params = jax_utils.replicate(pipeline_params)

    # Optimization
    if args.scale_lr:
      args.learning_rate = args.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    state = TrainState.create(
        apply_fn=controlnet.__call__,
        params=controlnet_params,
        ema_params=ema_controlnet_params,
        tx=optimizer,
    )

    noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Initialize our training
    validation_rng, train_rngs = jax.random.split(rng)
    if not args.debug:
      train_rngs = jax.random.split(train_rngs, jax.local_device_count())

    def compute_snr(timesteps):
      alphas_cumprod = noise_scheduler_state.common.alphas_cumprod
      sqrt_alphas_cumprod = alphas_cumprod**0.5
      sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

      alpha = sqrt_alphas_cumprod[timesteps]
      sigma = sqrt_one_minus_alphas_cumprod[timesteps]
      # Compute SNR.
      snr = (alpha / sigma) ** 2
      return snr

    def train_step(
        state, unet_params, text_encoder_params, vae_params, batch, train_rng
    ):
      # reshape batch, add grad_step_dim if gradient_accumulation_steps > 1
      if args.gradient_accumulation_steps > 1:
        grad_steps = args.gradient_accumulation_steps
        batch = jax.tree.map(
            lambda x: x.reshape(
                (grad_steps, x.shape[0] // grad_steps) + x.shape[1:]
            ),
            batch,
        )

      ema_params = state.ema_params

      def compute_loss(params, minibatch, sample_rng):
        # Convert images to latent space
        vae_outputs = vae.apply(
            {"params": vae_params},
            minibatch["pixel_values"],
            deterministic=True,
            method=vae.encode,
        )
        latents = vae_outputs.latent_dist.sample(sample_rng)
        # (NHWC) -> (NCHW)
        latents = jnp.transpose(latents, (0, 3, 1, 2))
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise_rng, timestep_rng = jax.random.split(sample_rng)
        noise = jax.random.normal(noise_rng, latents.shape)
        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            (bsz,),
            0,
            noise_scheduler.config.num_train_timesteps,
        )
        skipped_schedule = (
            noise_scheduler.config.num_train_timesteps
            // args.distill_learning_steps
        )
        next_timesteps = timesteps - skipped_schedule
        next_timesteps = jnp.where(next_timesteps < 0, 0, next_timesteps)

        noisy_latents = noise_scheduler.add_noise(
            noise_scheduler_state, latents, noise, timesteps
        )

        alpha_t, sigma_t = get_sqrt_alpha_prod(
            noise_scheduler_state.common,
            noisy_latents,  # only used for determining shape
            noise,  # unused code
            timesteps,
        )
        alpha_s, sigma_s = get_sqrt_alpha_prod(
            noise_scheduler_state.common,
            noisy_latents,  # only used for determining shape
            noise,  # unused code
            next_timesteps,
        )

        c_skip, c_out = scalings_for_boundary_conditions(
            timesteps, timestep_scaling=10
        )
        c_skip_next, c_out_next = scalings_for_boundary_conditions(
            next_timesteps, timestep_scaling=10
        )

        c_skip = broadcast_to_shape_from_left(c_skip, noisy_latents.shape)
        c_out = broadcast_to_shape_from_left(c_out, noisy_latents.shape)
        c_skip_next = broadcast_to_shape_from_left(
            c_skip_next, noisy_latents.shape
        )
        c_out_next = broadcast_to_shape_from_left(
            c_out_next, noisy_latents.shape
        )

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(
            minibatch["input_ids"],
            params=text_encoder_params,
            train=False,
        )[0]

        controlnet_cond = minibatch["conditioning_pixel_values"]

        # ======> step1 algorithm.12 of conditional distillation
        down_block_res_samples, mid_block_res_sample = None, None
        if args.onestepode == "control":
          if args.onestepode_control_params == "target":
            ode_control_params = ema_controlnet_params
          elif args.onestepode_control_params == "online":
            ode_control_params = controlnet_params
          else:
            raise NotImplementedError

          down_block_res_samples, mid_block_res_sample = controlnet.apply(
              {"params": ode_control_params},
              noisy_latents,
              timesteps,
              encoder_hidden_states,
              controlnet_cond,
              train=False,
              return_dict=False,
          )

        model_pred = unet.apply(
            {"params": unet_params},
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        if args.onestepode_sample_eps == "epsilon":
          sampler_eps = noise
        elif args.onestepode_sample_eps == "v_prediction":
          sampler_eps = alpha_t * model_pred + sigma_t * noisy_latents
        elif args.onestepode_sample_eps == "x_prediction":
          sampler_eps = (noisy_latents - alpha_t * model_pred) / sigma_t
        else:
          raise NotImplementedError

        # equation.7 of conditional distillation
        sampler_x = alpha_t * noisy_latents - sigma_t * model_pred
        hat_noisy_latents_s = alpha_s * sampler_x + sigma_s * sampler_eps

        # ======> step2 algorithm.11 of conditional distillation
        down_block_res_samples, mid_block_res_sample = controlnet.apply(
            {"params": ema_params},
            hat_noisy_latents_s,
            next_timesteps,
            encoder_hidden_states,
            controlnet_cond,
            train=False,
            return_dict=False,
        )

        model_pred = unet.apply(
            {"params": unet_params},
            hat_noisy_latents_s,
            next_timesteps,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # equation.7 of conditional distillation
        target_model_pred_x = (
            alpha_s * hat_noisy_latents_s - sigma_s * model_pred
        )
        target_model_pred_x = (
            c_skip_next * hat_noisy_latents_s + c_out_next * target_model_pred_x
        )
        target_model_pred_epsilon = (
            alpha_s * model_pred + sigma_s * hat_noisy_latents_s
        )
        target_model_pred_epsilon = (
            c_skip_next * hat_noisy_latents_s
            + c_out_next * target_model_pred_epsilon
        )

        # equation.8 of conditional distillation
        down_block_res_samples, mid_block_res_sample = controlnet.apply(
            {"params": params},
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            controlnet_cond,
            train=True,
            return_dict=False,
        )
        model_pred = unet.apply(
            {"params": unet_params},
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # equation.7 of conditional distillation
        online_model_pred_x = alpha_t * noisy_latents - sigma_t * model_pred
        online_model_pred_x = (
            c_skip * jax.lax.stop_gradient(noisy_latents)
            + c_out * online_model_pred_x
        )
        online_model_pred_epsilon = (
            alpha_t * model_pred + sigma_t * noisy_latents
        )
        online_model_pred_epsilon = (
            c_skip * jax.lax.stop_gradient(noisy_latents)
            + c_out * online_model_pred_epsilon
        )

        if args.distill_loss == "consistency_x":
          loss = (
              online_model_pred_x - jax.lax.stop_gradient(target_model_pred_x)
          ) ** 2
        elif args.distill_loss == "consistency_epsilon":
          loss = (
              online_model_pred_epsilon
              - jax.lax.stop_gradient(target_model_pred_epsilon)
          ) ** 2
        else:
          raise NotImplementedError

        beta_reg = (online_model_pred_x - jax.lax.stop_gradient(latents)) ** 2

        if args.snr_gamma is not None:
          snr = jnp.array(compute_snr(timesteps))
          snr_loss_weights = (
              jnp.where(
                  snr < args.snr_gamma, snr, jnp.ones_like(snr) * args.snr_gamma
              )
              / snr
          )
          loss = loss * snr_loss_weights

        return loss.mean() + beta_reg.mean()

      grad_fn = jax.value_and_grad(compute_loss)

      # get a minibatch (one gradient accumulation slice)
      def get_minibatch(batch, grad_idx):
        return jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
            batch,
        )

      def loss_and_grad(grad_idx, train_rng):
        # create minibatch for the grad step
        minibatch = (
            get_minibatch(batch, grad_idx) if grad_idx is not None else batch
        )
        sample_rng, train_rng = jax.random.split(train_rng, 2)
        loss, grad = grad_fn(state.params, minibatch, sample_rng)
        return loss, grad, train_rng

      if args.gradient_accumulation_steps == 1:
        loss, grad, new_train_rng = loss_and_grad(None, train_rng)
      else:
        init_loss_grad_rng = (
            0.0,  # initial value for cumul_loss
            jax.tree.map(
                jnp.zeros_like, state.params
            ),  # initial value for cumul_grad
            train_rng,  # initial value for train_rng
        )

        def cumul_grad_step(grad_idx, loss_grad_rng):
          cumul_loss, cumul_grad, train_rng = loss_grad_rng
          loss, grad, new_train_rng = loss_and_grad(grad_idx, train_rng)
          cumul_loss, cumul_grad = jax.tree.map(
              jnp.add, (cumul_loss, cumul_grad), (loss, grad)
          )
          return cumul_loss, cumul_grad, new_train_rng

        loss, grad, new_train_rng = jax.lax.fori_loop(
            0,
            args.gradient_accumulation_steps,
            cumul_grad_step,
            init_loss_grad_rng,
        )
        loss, grad = jax.tree.map(
            lambda x: x / args.gradient_accumulation_steps, (loss, grad)
        )

      if not args.debug:
        grad = jax.lax.pmean(grad, "batch")

      new_state = state.apply_gradients(grads=grad)

      new_ema_params = jax.tree.map(
          lambda ema, p: ema * args.ema_decay + (1 - args.ema_decay) * p,
          state.ema_params,
          new_state.params,
      )
      new_state = new_state.replace(ema_params=new_ema_params)

      metrics = {"loss": loss}
      if not args.debug:
        metrics = jax.lax.pmean(metrics, axis_name="batch")

      def l2(xs):
        return jnp.sqrt(
            sum([jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(xs)])
        )

      metrics["l2_grads"] = l2(jax.tree_util.tree_leaves(grad))

      return new_state, metrics, new_train_rng

    # Create parallel version of the train step
    if args.debug:
      p_train_step = train_step
    else:
      p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    text_encoder_params = text_encoder.params
    if not args.debug:
      # Replicate the train state on each device
      state = jax_utils.replicate(state)
      unet_params = jax_utils.replicate(unet_params)
      text_encoder_params = jax_utils.replicate(text_encoder_params)
      vae_params = jax_utils.replicate(vae_params)

    # Train!

    if args.streaming:
      dataset_length = args.max_train_samples
    else:
      dataset_length = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(
        dataset_length / args.gradient_accumulation_steps
    )

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
      args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    logger.info("***** Running training *****")

    if jax.process_index() == 0 and args.report_to == "wandb":
      wandb.define_metric("*", step_metric="train/step")
      wandb.define_metric("train/step", step_metric="walltime")
      wandb.config.update({
          "num_train_examples": (
              args.max_train_samples if args.streaming else len(train_dataset)
          ),
          "total_train_batch_size": total_train_batch_size,
          "total_optimization_step": (
              args.num_train_epochs * num_update_steps_per_epoch
          ),
          "num_devices": jax.device_count(),
          "controlnet_params": sum(
              np.prod(x.shape) for x in jax.tree_util.tree_leaves(state.params)
          ),
      })

    global_step = step0 = 0
    epochs = tqdm(
        range(args.num_train_epochs),
        desc="Epoch ... ",
        position=0,
        disable=jax.process_index() > 0,
    )
    if args.profile_memory:
      jax.profiler.save_device_memory_profile(
          os.path.join(args.output_dir, "memory_initial.prof")
      )
    t00 = t0 = time.monotonic()
    for epoch in epochs:
      # ======================== Training ================================

      train_metrics = []
      train_metric = None

      steps_per_epoch = (
          args.max_train_samples // total_train_batch_size
          if args.streaming or args.max_train_samples
          else len(train_dataset) // total_train_batch_size
      )
      train_step_progress_bar = tqdm(
          total=steps_per_epoch,
          desc="Training...",
          position=1,
          leave=False,
          disable=jax.process_index() > 0,
      )
      # train
      for batch in train_dataloader:
        if args.profile_steps and global_step == 1:
          train_metric["loss"].block_until_ready()
          jax.profiler.start_trace(args.output_dir)
        if args.profile_steps and global_step == 1 + args.profile_steps:
          train_metric["loss"].block_until_ready()
          jax.profiler.stop_trace()

        if not args.debug:
          batch = shard(batch)
        # with jax.profiler.StepTraceAnnotation("train", step_num=global_step):
        state, train_metric, train_rngs = p_train_step(
            state,
            unet_params,
            text_encoder_params,
            vae_params,
            batch,
            train_rngs,
        )
        train_metrics.append(train_metric)

        train_step_progress_bar.update(1)

        global_step += 1
        if global_step >= args.max_train_steps:
          break

        if (
            args.validation_prompt is not None
            and global_step % args.validation_steps == 0
            and jax.process_index() == 0
        ):
          _ = log_validation(
              pipeline,
              pipeline_params,
              state.ema_params,
              tokenizer,
              args,
              validation_rng,
              weight_dtype,
          )

        if global_step % args.logging_steps == 0 and jax.process_index() == 0:
          if args.report_to == "wandb":
            train_metrics = jax_utils.unreplicate(train_metrics)
            train_metrics = jax.tree_util.tree_map(
                lambda *m: jnp.array(m).mean(), *train_metrics
            )
            wandb.log({
                "walltime": time.monotonic() - t00,
                "train/step": global_step,
                "train/epoch": global_step / dataset_length,
                "train/steps_per_sec": (global_step - step0) / (
                    time.monotonic() - t0
                ),
                **{f"train/{k}": v for k, v in train_metrics.items()},
            })
          t0, step0 = time.monotonic(), global_step
          train_metrics = []
        if (
            global_step % args.checkpointing_steps == 0
            and jax.process_index() == 0
        ):
          controlnet.save_pretrained(
              f"{args.output_dir}/{global_step}",
              params=get_params_to_save(state.ema_params),
          )

      train_metric = jax_utils.unreplicate(train_metric)
      train_step_progress_bar.close()
      epochs.write(
          f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss:"
          f" {train_metric['loss']})"
      )

    # Final validation & store model.
    if jax.process_index() == 0:
      if args.validation_prompt is not None:
        if args.profile_validation:
          jax.profiler.start_trace(args.output_dir)
        image_logs = log_validation(
            pipeline,
            pipeline_params,
            state.params,
            tokenizer,
            args,
            validation_rng,
            weight_dtype,
        )
        if args.profile_validation:
          jax.profiler.stop_trace()
      else:
        image_logs = None

      controlnet.save_pretrained(
          args.output_dir,
          params=get_params_to_save(state.ema_params),
      )

    if args.profile_memory:
      jax.profiler.save_device_memory_profile(
          os.path.join(args.output_dir, "memory_final.prof")
      )
    logger.info("Finished training.")


if __name__ == "__main__":
  main()
