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

"""A script for saving all the word embeddings."""
# pylint: disable=g-multiple-import,g-importing-member,g-bad-import-order,missing-function-docstring

import argparse

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
import torch
from transformers import CLIPModel, CLIPProcessor


def parse_args():
  parser = argparse.ArgumentParser(
      description="Simple example of a training script."
  )
  parser.add_argument(
      "--pretrained_model_name_or_path",
      type=str,
      default="stabilityai/stable-diffusion-2-1-base",
      help=(
          "Path to pretrained model or model identifier from"
          " huggingface.co/models."
      ),
  )
  parser.add_argument(
      "--clip_model",
      type=str,
      default="openai/clip-vit-base-patch32",
      help=(
          "The CLIP model to use for the calculation of the image-text"
          " matching."
      ),
  )
  parser.add_argument(
      "--path_to_encoder_embeddings",
      type=str,
      default="./clip_text_encoding.pt",
      help="Path to the saved embeddings matrix of the text encoder",
  )

  args = parser.parse_args()

  return args


def main():
  args = parse_args()
  model = CLIPModel.from_pretrained(args.clip_model).cuda()
  processor = CLIPProcessor.from_pretrained(args.clip_model)

  # initialize stable diffusion pipeline
  pipe = StableDiffusionPipeline.from_pretrained(
      args.pretrained_model_name_or_path
  )
  pipe.to("cuda")
  scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
  pipe.scheduler = scheduler
  orig_embeddings = (
      pipe.text_encoder.text_model.embeddings.token_embedding.weight.clone().detach()
  )

  imagenet_templates = [
      "a photo of a {}",
      "a rendering of a {}",
      "a cropped photo of the {}",
      "the photo of a {}",
      "a photo of a clean {}",
      "a photo of a dirty {}",
      "a dark photo of the {}",
      "a photo of my {}",
      "a photo of the cool {}",
      "a close-up photo of a {}",
      "a bright photo of the {}",
      "a cropped photo of a {}",
      "a photo of the {}",
      "a good photo of the {}",
      "a photo of one {}",
      "a close-up photo of the {}",
      "a rendition of the {}",
      "a photo of the clean {}",
      "a rendition of a {}",
      "a photo of a nice {}",
      "a good photo of a {}",
      "a photo of the nice {}",
      "a photo of the small {}",
      "a photo of the weird {}",
      "a photo of the large {}",
      "a photo of a cool {}",
      "a photo of a small {}",
  ]

  def get_embedding_for_prompt(prompt, templates):
    with torch.no_grad():
      texts = [
          template.format(prompt) for template in templates
      ]  # format with class
      text_preprocessed = processor(
          text=texts, return_tensors="pt", padding=True
      )
      text_encodings = model.get_text_features(
          input_ids=text_preprocessed["input_ids"].cuda(),
          attention_mask=text_preprocessed["attention_mask"].cuda(),
      )
      text_encodings /= text_encodings.norm(dim=-1, keepdim=True)
      text_encodings = text_encodings.mean(dim=0)
      text_encodings /= text_encodings.norm()
      return text_encodings.float()

  top_encodings_open_clip = [
      get_embedding_for_prompt(
          pipe.tokenizer.decoder[token], imagenet_templates
      )
      for token in range(orig_embeddings.shape[0])
  ]
  top_encodings_open_clip = torch.stack(top_encodings_open_clip, dim=0)

  torch.save(top_encodings_open_clip, args.path_to_encoder_embeddings)


if __name__ == "__main__":
  main()
