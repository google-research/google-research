# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Retrieval clients that retrieve images from the dataset."""

import base64
import concurrent.futures
import logging
import ssl
from typing import Any, Optional, Sequence
import urllib

import numpy as np
from sklearn.metrics import pairwise

from agile_deliberation.agile_deliberation_lib import image as image_py
from agile_deliberation.agile_deliberation_lib.nearest_neighbor import clip_knn_service


logger = logging.getLogger(__name__)


class RetrievalClient:
  """A client for image retrieval.

  Attributes:
    knn_service: The KnnService logic or instance for retrieval.
    first_index_name: The name of the first index used for search.
  """

  def __init__(self, indices_path = None):
    """Initializes the RetrievalClient.

    Args:
      indices_path: Path to the indices used for the KNN service.
    """
    self.knn_service = None
    self.first_index_name = None

    if indices_path:
      try:
        self.knn_service = clip_knn_service.create(
            indices_paths=indices_path,
            # Enable hdf5 caching for the metadata. This reduces metadata memory
            # use to near zero, but on the first run will take a bit of time to
            # create the memory-mapped hdf5 files.
            enable_hdf5=True,
            # Use an index with memory mapping, decreasing memory use to zero.
            enable_faiss_memory_mapping=True,
            enable_mclip_option=False,
        )
        if self.knn_service.clip_resources:
          self.first_index_name = next(
              iter(self.knn_service.clip_resources.keys())
          )
      except Exception as e:
        logger.error(
            'Failed to initialize KnnService with path %s: %s', indices_path, e
        )

    if not self.knn_service:
      logger.warning(
          'KnnService not initialized. Search functionality will '
          'fail unless indices_path is provided.'
      )

  def _get_text_embedding(
      self,
      text,
  ):
    """Get text embedding from KnnService.

    Args:
      text: The text to get an embedding for.

    Returns:
      A numpy array containing the text embedding.

    Raises:
      ValueError: If KnnService or the index is not initialized.
    """
    if not self.knn_service:
      raise ValueError('KnnService not initialized')

    if not self.first_index_name:
      raise ValueError('No index available in KnnService')

    clip_resource = self.knn_service.clip_resources[self.first_index_name]
    # Compute query embedding (1, D).
    embedding = self.knn_service.compute_query(
        clip_resource=clip_resource,
        text_input=text,
        image_input=None,
        image_url_input=None,
        embedding_input=None,
        use_mclip=False
    )
    return embedding.flatten()

  def query(
      self,
      query_text = None,
      query_image = None,
      num_neighbors = 10,
  ):
    """Search for nearest neighbors (backward compatibility wrapper).

    Args:
      query_text: Optional text query.
      query_image: Optional image query.
      num_neighbors: The number of neighbors to retrieve.

    Returns:
      A sequence of MyImage objects.
    """
    if query_text:
      return self.text_image_search(query_text, num_neighbors)
    elif query_image:
      return self.image_image_search(query_image, num_neighbors)
    else:
      return []

  def _download_image_bytes(self, url):
    """Downloads an image from a URL and returns the raw bytes.

    Args:
      url: The URL to download the image from.

    Returns:
      The image bytes, or None if the download fails.
    """
    try:
      req = urllib.request.Request(
          url,
          headers={'User-Agent': 'Mozilla/5.0'}
      )
      ctx = ssl.create_default_context()
      ctx.set_alpn_protocols(['http/1.1'])
      with urllib.request.urlopen(req, timeout=10, context=ctx) as r:
        return r.read()
    except Exception as e:
      logger.debug('Failed to download image from %s: %s', url, e)
      return None

  def rank_images_by_criteria(
      self,
      images,
      criteria,
  ):
    """Ranks a list of images based on their similarity to a text criteria.

    Args:
      images: A sequence of `image_py.MyImage` objects.
      criteria: The text string used to rank the images.

    Returns:
      A list of `image_py.MyImage` objects sorted by their cosine similarity
      to the embedding of the criteria, from most similar to least similar.
    """
    # Get the embedding of the criteria.
    try:
      embedding = self._get_text_embedding(criteria)
      embedding = embedding.reshape(1, -1)
    except Exception as e:
      logger.error(
          "Failed to get text embedding for criteria '%s': %s", criteria, e
      )
      return images  # Return unsorted if embedding fails

    valid_images = [img for img in images if img.image_features is not None]
    if not valid_images:
      return []

    # Ensure image features are numpy arrays of correct shape.
    image_features = []
    for img in valid_images:
      f = np.array(img.image_features)
      if len(f.shape) == 0:  # scalar? shouldn't happen
        continue
      image_features.append(f.flatten())

    if not image_features:
      return []

    image_features = np.array(image_features)
    similarities = pairwise.cosine_similarity(image_features, embedding)

    # Sort the images by the distance.
    # Flatten similarities to match zip.
    similarities = similarities.flatten()

    sorted_images = sorted(
        zip(similarities, valid_images), key=lambda x: x[0], reverse=True
    )
    # Return the images sorted by the distance.
    return [image for _, image in sorted_images]

  def _results_to_images(
      self, results,
      max_workers = 8,
  ):
    """Converts KNN query results to MyImage objects.

    Downloads images from URLs concurrently when local image data is not
    available in the metadata.

    Args:
      results: A list of result dictionaries from KnnService.query().
      max_workers: The maximum number of concurrent download threads.

    Returns:
      A list of MyImage objects with image data, embeddings, and metadata.
    """
    to_download = []
    for i, res in enumerate(results):
      if 'image' not in res and 'url' in res:
        to_download.append((i, res['url']))

    downloaded = {}
    if to_download:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=max_workers
      ) as executor:
        future_to_idx = {
            executor.submit(self._download_image_bytes, url): idx
            for idx, url in to_download
        }
        for future in concurrent.futures.as_completed(future_to_idx):
          idx = future_to_idx[future]
          try:
            downloaded[idx] = future.result()
          except Exception:
            downloaded[idx] = None

    images = []
    for i, res in enumerate(results):
      image_bytes = None
      if 'image' in res:
        image_bytes = base64.b64decode(res['image'])
      elif i in downloaded:
        image_bytes = downloaded[i]
      if image_bytes is None:
        continue
      img = image_py.MyImage(image_bytes=image_bytes)
      img.distance = res['similarity']
      img.image_features = np.array(res['embedding'])
      if 'url' in res:
        img.url = res['url']
      if 'caption' in res:
        img.image_caption = res['caption']
      images.append(img)
    return images

  def text_image_search(
      self, description, num_neighbors = 5
  ):
    """Search similar images for a text description.

    Args:
      description: The text description we want to search for images.
      num_neighbors: The number of neighbors we want to get.

    Returns:
      A list of images that matches the description.
    """
    if not self.knn_service:
      logger.warning(
          'KnnService not initialized, returning empty results for'
          ' text_image_search.'
      )
      return []
    results = self.knn_service.query(
        text_input=description,
        modality='image',
        num_images=num_neighbors,
        num_result_ids=num_neighbors * 3,
        deduplicate=True,
        indice_name=self.first_index_name,
    )
    images = self._results_to_images(results)
    for img in images:
      img.query = description
    return images

  def image_image_search(
      self, image, num_neighbors = 5
  ):
    """Search similar images for an image.

    Args:
      image: The query image.
      num_neighbors: The number of neighbors.

    Returns:
      A sequence of retrieved MyImage objects.
    """
    if not self.knn_service or image.image_features is None:
      logger.warning(
          'KnnService not initialized or image has no features, returning empty'
          ' results.'
      )
      return []

    embedding = np.array(image.image_features)

    results = self.knn_service.query(
        embedding_input=embedding,
        modality='image',
        num_images=num_neighbors,
        num_result_ids=num_neighbors * 3,
        deduplicate=True,
        indice_name=self.first_index_name,
    )
    return self._results_to_images(results)
