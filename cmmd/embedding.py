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

"""Embedding models used in the CMMD calculation."""

import jax
from scenic.projects.baselines.clip import model as clip

Array = jax.numpy.ndarray

_CLIP_MODEL_NAME = 'vit_l14_336px'


def _clip_preprocess(images, size):
  target_shape = images.shape[:-3] + (size, size, images.shape[-1])

  images = jax.image.resize(images, shape=target_shape, method='bicubic')

  # Apply CLIP-specific shifting/scaling.  The input to `normalize_image` is
  # expected to be in [0, 1].
  images = clip.normalize_image(images)

  return images


class ClipEmbeddingModel:
  """CLIP image embedding calculator."""

  def __init__(self):
    self._model = clip.MODELS[_CLIP_MODEL_NAME]()
    self._model_vars = clip.load_model_vars(_CLIP_MODEL_NAME)
    self.input_image_size = clip.IMAGE_RESOLUTION[_CLIP_MODEL_NAME]
    self.parallel_embed = jax.pmap(self.embed)

  def embed(self, images):
    """Computes CLIP embeddings for the given images.

    Args:
      images: An image array of shape (batch_size, height, width, 3). Values are
        in range [0, 1].

    Returns:
      Embedding array of shape (batch_size, embedding_width).
    """
    images = _clip_preprocess(images, self.input_image_size)
    image_embs, _ = self._model.apply(self._model_vars, images, None)
    return image_embs
