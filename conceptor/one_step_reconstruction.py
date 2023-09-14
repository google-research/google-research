# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Runs the main decomposition algorithm."""
# pylint: disable=g-multiple-import,g-importing-member,g-bad-import-order,missing-function-docstring,missing-class-docstring
import argparse
import glob
import logging
import math
import os
from pathlib import Path
import random

from accelerate import Accelerator
from accelerate.logging import get_logger
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from model_zoo import CLIPImageSimilarity
import numpy as np
from packaging import version
import PIL
from PIL import Image
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse(
    "9.1.0"
):
  PIL_INTERPOLATION = {
      "linear": PIL.Image.Resampling.BILINEAR,
      "bilinear": PIL.Image.Resampling.BILINEAR,
      "bicubic": PIL.Image.Resampling.BICUBIC,
      "lanczos": PIL.Image.Resampling.LANCZOS,
      "nearest": PIL.Image.Resampling.NEAREST,
  }
else:
  PIL_INTERPOLATION = {
      "linear": PIL.Image.LINEAR,
      "bilinear": PIL.Image.BILINEAR,
      "bicubic": PIL.Image.BICUBIC,
      "lanczos": PIL.Image.LANCZOS,
      "nearest": PIL.Image.NEAREST,
  }
# ------------------------------------------------------------------------------


logger = get_logger(__name__)


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description="Simple example of a training script."
  )
  parser.add_argument(
      "--pretrained_model_name_or_path",
      type=str,
      default=None,
      required=True,
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
      "--prompt", type=str, required=True, help="The prompt to be explained."
  )
  parser.add_argument(
      "--train_data_dir",
      type=str,
      default=None,
      required=True,
      help="A folder containing the training data.",
  )
  parser.add_argument(
      "--validation_data_dir",
      type=str,
      default=None,
      required=True,
      help="A folder containing the training data.",
  )
  parser.add_argument(
      "--placeholder_token",
      type=str,
      default=None,
      required=True,
      help="A token to use as a placeholder for the concept.",
  )
  parser.add_argument(
      "--concept",
      type=str,
      default=None,
      required=True,
      help="The concept to explain.",
  )
  parser.add_argument(
      "--repeats",
      type=int,
      default=100,
      help="How many times to repeat the training data.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="output",
      help=(
          "The output directory where the model predictions and checkpoints"
          " will be written."
      ),
  )
  parser.add_argument(
      "--seed", type=int, default=1024, help="A seed for train images."
  )
  parser.add_argument(
      "--validation_seed",
      type=int,
      default=2,
      help="A seed for validation images.",
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
      action="store_true",
      help="Whether to center crop images before resizing to resolution.",
  )
  parser.add_argument(
      "--remove_concept_tokens",
      action="store_true",
      default=False,
      help="Whether to remove the concept token from the dictionary.",
  )
  parser.add_argument(
      "--train_batch_size",
      type=int,
      default=6,
      help="Batch size (per device) for the training dataloader.",
  )
  parser.add_argument("--num_train_epochs", type=int, default=1000)
  parser.add_argument(
      "--max_train_steps",
      type=int,
      default=1000,
      help=(
          "Total number of training steps to perform.  If provided, overrides"
          " num_train_epochs."
      ),
  )
  parser.add_argument(
      "--dictionary_size",
      type=int,
      default=5000,
      help="Number of top tokens to consider as dictionary.",
  )
  parser.add_argument(
      "--num_explanation_tokens",
      type=int,
      default=50,
      help="Number of words to produce as explanation.",
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
      "--gradient_checkpointing",
      action="store_true",
      help=(
          "Whether or not to use gradient checkpointing to save memory at the"
          " expense of slower backward pass."
      ),
  )
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=1e-3,
      help="Initial learning rate (after the potential warmup period) to use.",
  )
  parser.add_argument(
      "--sparsity_coeff",
      type=float,
      default=0.001,
      help="Initial learning rate (after the potential warmup period) to use.",
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
      default=500,
      help="Number of steps for the warmup in the lr scheduler.",
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
      default=1e-2,
      help="Weight decay to use.",
  )
  parser.add_argument(
      "--adam_epsilon",
      type=float,
      default=1e-08,
      help="Epsilon value for the Adam optimizer",
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
      default="no",
      choices=["no", "fp16", "bf16"],
      help=(
          "Whether to use mixed precision. Choose"
          "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
          "and an Nvidia Ampere GPU."
      ),
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
      "--report_to",
      type=str,
      default="tensorboard",
      help=(
          "The integration to report the results and logs to. Supported"
          ' platforms are `"tensorboard"` (default)'
      ),
  )
  parser.add_argument(
      "--validation_prompt",
      type=str,
      default=None,
      help=(
          "A prompt that is used during validation to verify that the model is"
          " learning."
      ),
  )
  parser.add_argument(
      "--num_validation_images",
      type=int,
      default=20,
      help=(
          "Number of images that should be generated during validation with"
          " `validation_prompt`."
      ),
  )
  parser.add_argument(
      "--validation_steps",
      type=int,
      default=50,
      help=(
          "Run validation every X epochs. Validation consists of running the"
          " prompt `args.validation_prompt` multiple times:"
          " `args.num_validation_images` and logging the images."
      ),
  )
  parser.add_argument(
      "--path_to_encoder_embeddings",
      type=str,
      default="./clip_text_encoding.pt",
      help="Path to the saved embeddings matrix of the text encoder",
  )
  parser.add_argument(
      "--local_rank",
      type=int,
      default=-1,
      help="For distributed training: local_rank",
  )
  parser.add_argument(
      "--enable_xformers_memory_efficient_attention",
      action="store_true",
      help="Whether or not to use xformers.",
  )

  args = parser.parse_args()
  env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
  if env_local_rank != -1 and env_local_rank != args.local_rank:
    args.local_rank = env_local_rank

  if args.train_data_dir is None:
    raise ValueError("You must specify a train data directory.")

  return args


imagenet_templates_small = [
    "a photo of a {}",
]


def decode_latents(vae, latents):
  latents = 1 / 0.18215 * latents
  image = vae.decode(latents).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.permute(0, 2, 3, 1)
  return image


class ConceptDataset(Dataset):

  def __init__(
      self,
      data_root,
      tokenizer,
      size=512,
      repeats=100,
      interpolation="bicubic",
      flip_p=0.5,
      split="train",
      placeholder_token="*",
      center_crop=False,
  ):
    self.data_root = data_root
    self.tokenizer = tokenizer
    self.size = size
    self.placeholder_token = placeholder_token
    self.center_crop = center_crop
    self.flip_p = flip_p

    self.image_paths = [
        os.path.join(self.data_root, file_path)
        for file_path in os.listdir(self.data_root)
    ]

    self.num_images = len(self.image_paths)
    self._length = self.num_images

    if split == "train":
      self._length = self.num_images * repeats

    self.interpolation = {
        "linear": PIL_INTERPOLATION["linear"],
        "bilinear": PIL_INTERPOLATION["bilinear"],
        "bicubic": PIL_INTERPOLATION["bicubic"],
        "lanczos": PIL_INTERPOLATION["lanczos"],
    }[interpolation]

    self.templates = imagenet_templates_small
    self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

  def __len__(self):
    return self._length

  def __getitem__(self, i):
    example = {}
    image = Image.open(self.image_paths[i % self.num_images])

    if image.mode != "RGB":
      image = image.convert("RGB")

    placeholder_string = self.placeholder_token
    text = random.choice(self.templates).format(placeholder_string)

    example["input_ids"] = self.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=self.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]

    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)

    if self.center_crop:
      crop = min(img.shape[0], img.shape[1])
      (
          h,
          w,
      ) = (
          img.shape[0],
          img.shape[1],
      )
      img = img[
          (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
      ]

    image = Image.fromarray(img)
    image = image.resize((self.size, self.size), resample=self.interpolation)

    image = self.flip_transform(image)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)

    example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
    return example


def get_clip_encodings(data_root):
  clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
      "cuda"
  )
  clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

  image_paths = [
      f"{data_root}/{i}.png"
      for i in range(len(glob.glob(f"{data_root}/*.png")))
  ]
  images = []
  for image_p in image_paths:
    image = Image.open(image_p)

    if image.mode != "RGB":
      image = image.convert("RGB")
    images.append(image)

  images_processed = clip_processor(images=images, return_tensors="pt")[
      "pixel_values"
  ].cuda()
  target_image_encodings = clip_model.get_image_features(images_processed)
  target_image_encodings /= target_image_encodings.norm(dim=-1, keepdim=True)
  del clip_model
  torch.cuda.empty_cache()

  return target_image_encodings


def get_dictionary_indices(
    args, target_image_encodings, tokenizer, dictionary_size
):
  clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
      "cuda"
  )
  normalized_text_encodings = torch.load(args.path_to_encoder_embeddings)

  # calculate cosine similarities for the average image
  mean_target_image = target_image_encodings.mean(dim=0).reshape(1, -1)
  cosine_similarities = torch.cosine_similarity(
      mean_target_image, normalized_text_encodings
  ).reshape(1, -1)

  if args.remove_concept_tokens:
    # remove concept tokens
    clip_concept_inputs = tokenizer(
        [args.concept], padding=True, return_tensors="pt"
    ).to("cuda")
    clip_concept_features = clip_model.get_text_features(**clip_concept_inputs)

    concept_words_similarity = torch.cosine_similarity(
        clip_concept_features, normalized_text_encodings, axis=1
    )
    similar_words = (
        np.array(concept_words_similarity.detach().cpu()) > 0.9
    ).nonzero()[0]
    # Zero-out similar words
    for i in similar_words:
      print("removing similar word", tokenizer.decode(i))
      cosine_similarities[0, i] = 0

  # average similarities across the images
  mean_cosine = torch.mean(cosine_similarities, dim=0)
  _, sorted_indices = torch.sort(mean_cosine, descending=True)

  # return the indices of the words to consider in the dictionary
  return sorted_indices[:dictionary_size]


class Net(nn.Module):

  def __init__(self, num_tokens):
    super().__init__()
    self.num_tokens = num_tokens
    self.fc1 = nn.Linear(1024, 100)
    self.fc2 = nn.Linear(100, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x.flatten().abs()


def generate_sd_images(
    pipe, prompt, output_path, num_images_per_prompt, epochs, seed
):
  print(
      "************ using prompt: ",
      prompt,
      "epochs = ",
      epochs,
      "path = ",
      output_path,
  )

  output_path = Path(output_path)
  output_path.mkdir(parents=True)
  generator = torch.Generator("cuda").manual_seed(seed)

  for epoch in range(epochs):
    images = pipe(
        [prompt],
        guidance_scale=7.5,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
    ).images
    for i, image in enumerate(images):
      image.save(f"{output_path}/{epoch * 10 + i}.png")


def generate_images_if_needed(
    train_data_dir,
    validation_data_dir,
    pretrained_model_name_or_path,
    prompt,
    num_validation_images,
    seed,
    validation_seed,
):
  if os.path.exists(train_data_dir) and os.path.exists(validation_data_dir):
    return

  pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
  pipe.to("cuda")
  scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
  pipe.scheduler = scheduler
  pipe.text_encoder.text_model.embeddings.token_embedding.weight.requires_grad_(
      False
  )

  batch_size = 10
  if not os.path.exists(train_data_dir):
    generate_sd_images(
        pipe, prompt, train_data_dir, batch_size, 100 // batch_size, seed
    )

  if not os.path.exists(validation_data_dir):
    generate_sd_images(
        pipe,
        prompt,
        validation_data_dir,
        batch_size,
        num_validation_images // batch_size,
        validation_seed,
    )

  del pipe
  torch.cuda.empty_cache()


def main():
  args = parse_args()
  logging_dir = os.path.join(args.output_dir, args.logging_dir)

  accelerator = Accelerator(
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      mixed_precision=args.mixed_precision,
      log_with=args.report_to,
      logging_dir=logging_dir,
  )

  # Make one log on every process with the configuration for debugging.
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
  )
  logger.info(accelerator.state, main_process_only=False)
  if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
  else:
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

  with torch.no_grad():
    generate_images_if_needed(
        args.train_data_dir,
        args.validation_data_dir,
        args.pretrained_model_name_or_path,
        args.prompt,
        args.num_validation_images,
        args.seed,
        args.validation_seed,
    )

  # Handle the repository creation
  if accelerator.is_main_process:
    if args.output_dir is not None:
      os.makedirs(args.output_dir, exist_ok=True)

  # Load tokenizer
  tokenizer = CLIPTokenizer.from_pretrained(
      args.pretrained_model_name_or_path, subfolder="tokenizer"
  )

  # Load scheduler and models
  noise_scheduler = DDPMScheduler.from_pretrained(
      args.pretrained_model_name_or_path, subfolder="scheduler"
  )
  text_encoder = CLIPTextModel.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="text_encoder",
      revision=args.revision,
  )
  vae = AutoencoderKL.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="vae",
      revision=args.revision,
  )
  unet = UNet2DConditionModel.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="unet",
      revision=args.revision,
  )

  # Add the placeholder token in tokenizer
  num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
  if num_added_tokens == 0:
    raise ValueError(
        f"The tokenizer already contains the token {args.placeholder_token}."
        " Please pass a different `placeholder_token` that is not already in"
        " the tokenizer."
    )

  placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
  # Resize the token embeddings as we are adding new special tokens to the
  # tokenizer
  text_encoder.resize_token_embeddings(len(tokenizer))

  # Freeze vae and unet
  vae.requires_grad_(False)
  unet.requires_grad_(False)
  # Freeze all parameters except for the token embeddings in text encoder
  text_encoder.text_model.encoder.requires_grad_(False)
  text_encoder.text_model.final_layer_norm.requires_grad_(False)
  text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

  if args.gradient_checkpointing:
    # Keep unet in train mode if we are using gradient checkpointing to save
    # memory. The dropout cannot be != 0 so it doesn't matter if we are in eval
    # or train mode.
    unet.train()
    text_encoder.gradient_checkpointing_enable()
    unet.enable_gradient_checkpointing()

  if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
      unet.enable_xformers_memory_efficient_attention()
    else:
      raise ValueError(
          "xformers is not available. Make sure it is installed correctly"
      )

  if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

  if args.scale_lr:
    args.learning_rate = (
        args.learning_rate
        * args.gradient_accumulation_steps
        * args.train_batch_size
        * accelerator.num_processes
    )

  # initialize nn
  net = Net(args.dictionary_size)

  # Initialize the optimizer
  optimizer = torch.optim.AdamW(
      net.parameters(),  # only optimize the embeddings
      lr=args.learning_rate,
      betas=(args.adam_beta1, args.adam_beta2),
      weight_decay=args.adam_weight_decay,
      eps=args.adam_epsilon,
  )

  # Dataset and DataLoaders creation:
  train_dataset = ConceptDataset(
      data_root=args.train_data_dir,
      tokenizer=tokenizer,
      size=args.resolution,
      placeholder_token=args.placeholder_token,
      repeats=args.repeats,
      center_crop=args.center_crop,
      split="train",
  )
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.train_batch_size,
      shuffle=True,
      num_workers=args.dataloader_num_workers,
  )

  # Scheduler and math around the number of training steps.
  overrode_max_train_steps = False
  num_update_steps_per_epoch = math.ceil(
      len(train_dataloader) / args.gradient_accumulation_steps
  )
  if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

  lr_scheduler = get_scheduler(
      args.lr_scheduler,
      optimizer=optimizer,
      num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
      num_training_steps=args.max_train_steps
      * args.gradient_accumulation_steps,
  )

  # Prepare everything with our `accelerator`.
  text_encoder, optimizer, train_dataloader, lr_scheduler, net = (
      accelerator.prepare(
          text_encoder, optimizer, train_dataloader, lr_scheduler, net
      )
  )

  weight_dtype = torch.float32
  if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
  elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

  # Move vae and unet to device and cast to weight_dtype
  unet.to(accelerator.device, dtype=weight_dtype)
  vae.to(accelerator.device, dtype=weight_dtype)

  # We need to recalculate our total training steps as the size of the training
  # dataloader may have changed.
  num_update_steps_per_epoch = math.ceil(
      len(train_dataloader) / args.gradient_accumulation_steps
  )
  if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
  # Afterwards we recalculate our number of training epochs
  args.num_train_epochs = math.ceil(
      args.max_train_steps / num_update_steps_per_epoch
  )

  # We need to initialize the trackers we use, and also store our configuration.
  # The trackers initializes automatically on the main process.
  if accelerator.is_main_process:
    accelerator.init_trackers("Conceptor", config=vars(args))

  # Train!
  total_batch_size = (
      args.train_batch_size
      * accelerator.num_processes
      * args.gradient_accumulation_steps
  )

  logger.info("***** Running training *****")
  logger.info(f"  Num examples = {len(train_dataset)}")
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
  logger.info(f"  Total optimization steps = {args.max_train_steps}")
  global_step = 0
  first_epoch = 0

  # Only show the progress bar once on each machine.
  progress_bar = tqdm(
      range(global_step, args.max_train_steps),
      disable=not accelerator.is_local_main_process,
  )
  progress_bar.set_description("Steps")

  # keep original embeddings as reference
  orig_embeds_params = (
      accelerator.unwrap_model(text_encoder)
      .get_input_embeddings()
      .weight.data.clone()
  )

  norms = [i.norm().item() for i in orig_embeds_params]
  avg_norm = np.mean(norms)
  text_encoder.get_input_embeddings().weight.requires_grad_(False)

  # get dictionary
  num_tokens = args.dictionary_size
  target_image_encodings = get_clip_encodings(args.train_data_dir)
  validation_image_encodings = get_clip_encodings(args.validation_data_dir)
  dictionary_indices = get_dictionary_indices(
      args, target_image_encodings, tokenizer, num_tokens
  )

  print("saving dictionary")
  torch.save(dictionary_indices, f"{args.output_dir}/dictionary.pt")

  target_image_encodings.detach_().requires_grad_(False)
  validation_image_encodings.detach_().requires_grad_(False)
  dictionary = orig_embeds_params[dictionary_indices]

  # create pipeline (note: unet and vae are loaded again in float32)
  pipeline = DiffusionPipeline.from_pretrained(
      args.pretrained_model_name_or_path,
      text_encoder=accelerator.unwrap_model(text_encoder),
      tokenizer=tokenizer,
      unet=unet,
      vae=vae,
      revision=args.revision,
      torch_dtype=weight_dtype,
  )
  pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
      pipeline.scheduler.config
  )
  pipeline = pipeline.to(accelerator.device)
  pipeline.set_progress_bar_config(disable=True)

  best_validation_score = 0
  best_alphas = None
  best_epoch = None
  best_words = None
  validation_model = CLIPImageSimilarity()

  for epoch in range(first_epoch, args.num_train_epochs):
    net.train()
    for batch in train_dataloader:
      text_encoder.get_input_embeddings().weight.detach_().requires_grad_(False)

      # calculate current embeddings
      token_embeds = text_encoder.get_input_embeddings().weight
      alphas = net(dictionary)
      _, sorted_indices = torch.sort(alphas.abs(), descending=True)
      print_words = min(50, args.num_explanation_tokens)

      num_words = args.dictionary_size
      word_indices = sorted_indices[:num_words]
      embedding = torch.matmul(alphas[word_indices], dictionary[word_indices])
      embedding = torch.mul(embedding, 1 / embedding.norm())
      embedding = torch.mul(embedding, avg_norm)

      top_words = [
          tokenizer.decode(dictionary_indices[sorted_indices[i]])
          for i in range(print_words)
      ]
      print(
          "top words: ",
          top_words,
          "alphas: ",
          alphas[sorted_indices[:print_words]],
      )

      token_embeds[placeholder_token_id] = embedding
      text_encoder.get_input_embeddings().weight.requires_grad_(True)

      with accelerator.accumulate(net):
        # Convert images to latent space
        latents = (
            vae.encode(batch["pixel_values"].to(dtype=weight_dtype))
            .latent_dist.sample()
            .detach()
        )
        # latents = latents * vae.config.scaling_factor
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each
        # timestep (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(
            dtype=weight_dtype
        )

        # Predict the noise residual
        model_pred = unet(
            noisy_latents, timesteps, encoder_hidden_states
        ).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
          target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
          target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
          raise ValueError(
              "Unknown prediction type"
              f" {noise_scheduler.config.prediction_type}"
          )

        mse_loss = F.mse_loss(
            model_pred.float(), target.float(), reduction="mean"
        )

        top_indices = [
            sorted_indices[i].item() for i in range(args.num_explanation_tokens)
        ]
        top_embedding = torch.matmul(
            alphas[top_indices], dictionary[top_indices]
        )
        sparsity_loss = 1 - torch.cosine_similarity(
            top_embedding.reshape(1, -1), embedding.reshape(1, -1)
        )

        # calculate final loss
        loss = mse_loss + args.sparsity_coeff * sparsity_loss

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Let's make sure we don't update any embedding weights besides the
        # newly added token
        index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id
        with torch.no_grad():
          accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
              index_no_updates
          ] = orig_embeds_params[index_no_updates]

        # Checks if the accelerator has performed an optimization step behind
        # the scenes
        if accelerator.sync_gradients:
          progress_bar.update(1)
          global_step += 1
        logs = {
            "loss": mse_loss.detach().item(),
            "sparse": sparsity_loss.detach().item(),
            "norm": (
                text_encoder.text_model.embeddings.token_embedding.weight[
                    placeholder_token_id
                ]
                .norm()
                .detach()
                .item()
            ),
            "lr": lr_scheduler.get_last_lr()[0],
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

      if (
          args.validation_prompt is not None
          and global_step % args.validation_steps == 0
      ):
        token_embeds[placeholder_token_id] = top_embedding
        print(
            "Running validation... \n Generating"
            f" {args.num_validation_images} images with prompt:"
            f" {args.validation_prompt}."
        )
        pipeline.text_encoder = accelerator.unwrap_model(text_encoder)

        # run inference
        generator = torch.Generator(device=accelerator.device).manual_seed(
            args.validation_seed
        )
        with torch.no_grad():
          with torch.autocast("cuda"):
            images = []
            for _ in range(args.num_validation_images // 10):
              images += pipeline(
                  args.validation_prompt,
                  num_inference_steps=50,
                  generator=generator,
                  num_images_per_prompt=min(
                      10, args.num_validation_images - len(images)
                  ),
                  return_dict=False,
              )[0]

          probabilities = []
          for i, image in enumerate(images):
            validation_dir = f"{args.output_dir}/validation/{global_step}"

            validation_dir = Path(validation_dir)
            validation_dir.mkdir(exist_ok=True, parents=True)
            image.save(f"{validation_dir}/{i}.png")
            input_image = validation_model.transform(image)
            probability = validation_model.get_probability(
                input_image, target_images=validation_image_encodings[i : i + 1]
            )
            probabilities.append(probability.item())

          validation_probability = np.mean(probabilities)
          print("validation probability: ", validation_probability)

          if validation_probability > best_validation_score:
            print("replacing best alphas")
            best_validation_score = validation_probability
            best_alphas = alphas.detach()
            best_words = top_words
            best_epoch = global_step

        for tracker in accelerator.trackers:
          if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, epoch, dataformats="NHWC"
            )

        print("saving alphas from step: ", global_step)
        torch.save(alphas, f"{args.output_dir}/{global_step}_alphas.pt")

        torch.cuda.empty_cache()

      if global_step >= args.max_train_steps:
        break

  accelerator.end_training()
  print(
      f"saving best alphas from validation step {best_epoch}, words = ",
      best_words,
  )
  torch.save(best_alphas, f"{args.output_dir}/best_alphas.pt")


if __name__ == "__main__":
  main()
