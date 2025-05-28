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

"""CLIP helpers."""

from typing import Iterable

import clip
import numpy as np
import torch
import torch.nn.functional as F


def load_clip(name, device):
  """Load CLIP models."""
  image_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device)
  image_mean = image_mean[None, :, None, None]
  image_std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device)
  image_std = image_std[None, :, None, None]

  model, _ = clip.load(
      name, device=device, jit=False, download_root="/workdir/clip_model_cache")

  def preprocess(images):
    images = F.interpolate(
        images, size=model.visual.input_resolution, mode="bicubic")
    images = (images - image_mean) / image_std
    return images

  return model, preprocess, model.visual.input_resolution


@torch.no_grad()
def embed_queries_clip(model, queries_list, device):
  """Embed captions."""
  # TODO(jainajay): This can be cached.
  queries_tok = clip.tokenize(queries_list).to(device)
  z_queries = model.encode_text(queries_tok).detach()
  z_queries = F.normalize(z_queries, dim=-1)
  return z_queries


@torch.inference_mode()
def compute_query_rank(model, preprocess, rendering, query,
                       queries_r, device):
  """Compute rank of `query` among `queries_r` according to CLIP.

  The score <CLIP_image(rendering), CLIP_text(query)> is used for ranking.

  Args:
    model: CLIP model.
    preprocess: Preprocessing function. Inputs to this fn are scaled to [0, 1].
    rendering: [batch, 3, height, width] tensor of the rendering.
    query: Caption used to generate the rendering.
    queries_r: List of negative captions (and optionally including `query`).
    device: PyTorch device for computation.

  Returns:
    rank (int), cosine_similarity (float)
  """
  if query not in queries_r:
    print(f"WARN: query \"{query}\" not in retrieval set. Adding it.")
    queries_r = queries_r + [query]
    query_idx = len(queries_r) - 1
  else:
    query_idx = queries_r.index(query)

  # Embed the retrieval set of captions.
  z_queries = embed_queries_clip(model, queries_r, device)

  # Embed render.
  assert rendering.ndim == 4
  assert rendering.shape[1] == 3
  x = preprocess(rendering)
  z_render = model.encode_image(x)
  z_render = F.normalize(z_render, dim=-1)

  ranks = []
  cosine_sims = []
  for zr in z_render:
    sim = torch.sum(zr.unsqueeze(0) * z_queries, dim=-1)
    rank = torch.argsort(sim, dim=0, descending=True)
    rank = torch.nonzero(rank == query_idx)[0].item()
    cosine_sim = sim[query_idx].item()
    ranks.append(int(rank))
    cosine_sims.append(float(cosine_sim))
  return np.array(ranks), np.array(cosine_sims)
