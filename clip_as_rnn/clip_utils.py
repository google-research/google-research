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

"""The utility functions for CLIP."""

from typing import Any, Callable, List, Optional, Union
from clip import simple_tokenizer
import torch
from torch import nn
import torch.nn.functional as F

from clip_as_rnn import clip_wrapper


def _forward_clip_single(
    model,
    image,
    texts,
    h,
    w,
):
  """Forward CLIP with a single text string or a list of string as inputs.

  Args:
      model: A nn.Module representing the CLIP model.
      image: A torch.Tensor representing the image tensor.
      texts: A list of string representing the text input.
      h: An integer representing the height of the image.
      w: An integer representing the width of the image.

  Returns:
      torch.Tensor: the logits.
  """
  if isinstance(texts, str):
    texts = [texts]
  text_tokens = tokenize(texts).to(image.device)
  text_prediction = model(image, text_tokens, h, w)
  return text_prediction.cpu().detach()


def forward_clip(
    model,
    image,
    texts,
    h,
    w,
):
  """Forward a list of text inputs.

  Args:
      model: A nn.Module representing the CLIP model.
      image: A torch.Tensor representing the image tensor.
      texts: A list of string or a list of list[string] representing the text
        input. When texts is a double list, each element in texts list is a list
        of class names extended by different prompt templates. If each element
        in texts is a string, then each class name is extened by only one prompt
        template.
      h: An integer representing the height of the image.
      w: An integer representing the width of the image.

  Returns:
      torch.Tensor: the logits.
  """
  if isinstance(texts[0], list):
    text_prediction = torch.stack(
        [_forward_clip_single(model, image, t, h, w) for t in texts], dim=0
    )
    text_prediction = torch.sum(text_prediction, dim=0)
    text_prediction = F.softmax(text_prediction.float(), dim=-1)
  else:
    text_prediction = _forward_clip_single(model, image, texts, h, w)
  return text_prediction.float()


def get_class_embeddings(
    class_names,
    model,
    templates = None,
    device = None
):
  """Get class embeddings.

  Args:
      class_names: A list of text representing the class names.
      model: A model of clip_wrapper.CLIPWrapper.
      templates: A list of text template for prompting.
      device: A str representing the device. Accepted values: ["cuda", "cpu"].

  Returns:
      class_embeddings: A torch.Tensor representing the weights for each
          class.
  """
  with torch.no_grad():
    class_embeddings = []
    for class_name in class_names:
      if templates is None:
        texts = [class_name]
      else:
        # format with class
        texts = [template.format(class_name) for template in templates]
      texts = tokenize(texts).to(device)  # tokenize
      single_embed = model.encode_text(texts)  # embed with text encoder
      single_embed /= single_embed.norm(dim=-1, keepdim=True)
      single_embed = single_embed.mean(dim=0)
      single_embed /= single_embed.norm()
      class_embeddings.append(single_embed)
    class_embeddings = torch.stack(class_embeddings, dim=1).to(device)
  return class_embeddings.t()


def tokenize(
    texts,
    tokenizer = None,
):
  """Tokenize a list of texts.

  Args:
    texts: A list of string representing the text input to be tokenized.
    tokenizer: The tokenizer to use.

  Returns:
    A torch.tensor representing the text tokens.

  """
  if tokenizer is None:
    tokenizer = simple_tokenizer.SimpleTokenizer()
  text_tokens = simple_tokenizer.tokenize(tokenizer, texts, context_length=10)
  text_tokens = torch.from_numpy(text_tokens)
  return text_tokens
