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

"""Generate an image given a prompt using a trained model."""

from argparse import ArgumentParser  # pylint: disable=g-importing-member
import random
from diffusers import DDIMScheduler  # pylint: disable=g-importing-member
from diffusers import StableDiffusionPipeline  # pylint: disable=g-importing-member
import numpy as np
import torch


def main(args):
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(args.seed)
  random.seed(args.seed)
  model_path = args.model_path
  pipe = StableDiffusionPipeline.from_pretrained(
      "runwayml/stable-diffusion-v1-5"
  )
  pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  pipe.unet.load_attn_procs(model_path)
  pipe.to("cuda")
  prompt_list = [args.prompt for i in range(10)]
  for i in range(10):
    images = pipe(prompt=prompt_list[i], eta=1.0).images
    image = images[0]

    image.save("image{}.png".format(i))


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--prompt", default="A green colored rabbit.")
  parser.add_argument("--model-path", default="./test")
  parser.add_argument("--seed", default=0, type=int)
  main(parser.parse_args())
