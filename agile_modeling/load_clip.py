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

"""Load CLIP model.

The code below is based on https://github.com/rom1504/clip-retrieval/
"""
# pylint: disable-all
from functools import lru_cache
import time
from typing import Any, Optional
import clip
from PIL import Image
import torch
from torch import autocast, nn


class OpenClipWrapper(nn.Module):
  """Wrap OpenClip for managing input types."""

  def __init__(self, inner_model: Any, device: str):
    super().__init__()
    self.inner_model = inner_model
    self.device = torch.device(device=device)
    if self.device.type == "cpu":
      self.dtype = torch.float32
    else:
      self.dtype = (
          torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
      )

  def encode_image(self, image):
    if self.device.type == "cpu":
      return self.inner_model.encode_image(image)
    with autocast(device_type=self.device.type, dtype=self.dtype):
      return self.inner_model.encode_image(image)

  def encode_text(self, text):
    if self.device.type == "cpu":
      return self.inner_model.encode_text(text)
    with autocast(device_type=self.device.type, dtype=self.dtype):
      return self.inner_model.encode_text(text)

  def forward(self, *args, **kwargs):
    return self.inner_model(*args, **kwargs)


def load_open_clip(
    clip_model: str,
    use_jit: bool = True,
    device: str = "cuda",
    clip_cache_path: Optional[str] = None,
) -> tuple[nn.Module, Any]:
  """Loads OpenClip model.

  Args:
    clip_model: The name of the clip model.
    use_jit: Whether to use jit.
    device: CPU or GPU.
    clip_cache_path: The clip cache path.

  Returns:
    A tuple containing the model and the preprocess function for it.
  """

  import open_clip  # pylint:disable=g-import-not-at-top

  torch.backends.cuda.matmul.allow_tf32 = True

  pretrained = dict(open_clip.list_pretrained())
  checkpoint = pretrained[clip_model]
  model, _, preprocess = open_clip.create_model_and_transforms(
      clip_model,
      pretrained=checkpoint,
      device=device,
      jit=use_jit,
      cache_dir=clip_cache_path,
  )
  model = OpenClipWrapper(inner_model=model, device=device)
  model.to(device=device)
  return model, preprocess


@lru_cache(maxsize=None)
def get_tokenizer(clip_model):
  """Load clip tokenizer."""
  if clip_model.startswith("open_clip:"):
    import open_clip  # pylint:disable=g-import-not-at-top

    clip_model = clip_model[len("open_clip:") :]
    return open_clip.get_tokenizer(clip_model)
  else:
    return lambda t: clip.tokenize(t, truncate=True)


@lru_cache(maxsize=None)
def load_clip_without_warmup(
    clip_model: str, use_jit: bool, device: str, clip_cache_path: str
) -> tuple[nn.Module, Any]:
  """Load clip without warmup.

  Args:
    clip_model: The name of the clip model.
    use_jit: Whether to use jit.
    device: CPU or GPU.
    clip_cache_path: The clip cache path.

  Returns:
    A tuple containing the model and the preprocess function for it.
  """
  if clip_model.startswith("open_clip:"):
    clip_model = clip_model[len("open_clip:") :]
    model, preprocess = load_open_clip(
        clip_model, use_jit, device, clip_cache_path
    )
  else:
    model, preprocess = clip.load(
        clip_model, device=device, jit=use_jit, download_root=clip_cache_path
    )
  return model, preprocess


@lru_cache(maxsize=None)
def load_clip(
    clip_model: str = "ViT-B/32",
    use_jit: bool = True,
    warmup_batch_size: int = 1,
    clip_cache_path: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[nn.Module, Any]:
  """Load clip then warmup.

  Args:
    clip_model: The name of the clip model.
    use_jit: Whether to use jit.
    warmup_batch_size: the warmup batch size.
    clip_cache_path: The clip cache path
    device: CPU or GPU.

  Returns:
    A tuple containing the model and the preprocess function for it.
  """
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = load_clip_without_warmup(
      clip_model, use_jit, device, clip_cache_path
  )

  start = time.time()
  print(
      f"warming up with batch size {warmup_batch_size} on {device}", flush=True
  )
  warmup(warmup_batch_size, device, preprocess, model)
  duration = time.time() - start
  print(f"done warming up in {duration}s", flush=True)
  return model, preprocess


def warmup(batch_size: int, device: str, preprocess: Any, model: nn.Module):
  """Warmup the CLIP model.

  Args:
    batch_size: batch size.
    device: CPU or GPU.
    preprocess: The preprocess function applied to the image.
    model: The CLIP model.
  """
  fake_img = Image.new("RGB", (224, 224), color="red")
  fake_text = ["fake"] * batch_size
  image_tensor = torch.cat(
      [torch.unsqueeze(preprocess(fake_img), 0)] * batch_size
  ).to(device)
  text_tokens = clip.tokenize(fake_text).to(device)
  for _ in range(2):
    with torch.no_grad():
      model.encode_image(image_tensor)
      model.encode_text(text_tokens)
