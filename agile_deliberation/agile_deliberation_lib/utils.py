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

"""Utility functions."""

import concurrent.futures as concurrent_futures
import hashlib
import logging
import math
import os
import pickle
import sqlite3
import time
from typing import Any, Callable, Optional, Tuple, TypeVar
import xml.etree.ElementTree as ET

import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
import tensorflow as tf

from agile_deliberation.agile_deliberation_lib import image as image_py


normalize = preprocessing.normalize
logger = logging.getLogger(__name__)

DictionaryLearning = decomposition.DictionaryLearning
MiniBatchDictionaryLearning = decomposition.MiniBatchDictionaryLearning


def async_execute(func):
  """Execute a function asynchronously.

  Args:
    func: The function to execute.

  Returns:
    A future representing the execution.
  """
  executor = concurrent_futures.ThreadPoolExecutor()
  future = executor.submit(func)
  return future


def run_in_batches(
    items, process_function, batch_size=5, **func_kwargs
):
  """Processes items in batches using a thread pool.

  Args:
    items: List of items to process.
    process_function: The function to apply to each item.
    batch_size: Number of items to process concurrently in each batch.
    **func_kwargs: Additional keyword arguments for process_function.

  Returns:
    A list of results in the same order as the input items.
  """

  num_items = len(items)
  if num_items == 0:
    return []

  all_results = [None] * len(items)
  for i in range(0, num_items, batch_size):
    batch = items[i : i + batch_size]
    batch_indices = range(i, i + len(batch))

    logger.info(
        'Processing batch: %d/%d',
        i // batch_size + 1,
        math.ceil(num_items / batch_size),
    )
    start_time = time.time()

    with concurrent_futures.ThreadPoolExecutor(
        max_workers=batch_size
    ) as executor:
      # Create futures, mapping them to their original index.
      future_to_index = {
          executor.submit(
              process_function, item, **func_kwargs
          ): idx
          for item, idx in zip(batch, batch_indices)
      }

      for future in concurrent_futures.as_completed(future_to_index):
        original_index = future_to_index[future]
        try:
          all_results[original_index] = future.result()
        except Exception as e:
          print(f'Error processing item {original_index}: {e}')
          all_results[original_index] = None
    logger.debug(
        'run_in_%d batches finished in %d seconds',
        i // batch_size + 1,
        time.time() - start_time,
    )
  return all_results


def generate_sharded_filenames(filename):
  """Generates a list of filenames for a sharded file.

  Args:
    filename: The filename to generate shards for.

  Returns:
    A list of filenames for the shards.
  """
  if '@' not in filename:
    return [filename]

  prefix, num_shards = filename.rsplit('@', 1)
  if not num_shards.isdigit():
    return [filename]

  n = int(num_shards)
  results = []
  for i in range(n):
    results.append(f'{prefix}-{i:05d}-of-{n:05d}')
  return results


def extract_a_portion_of_xml(xml_text, tag_name_to_extract):
  """Extracts a portion of a XML string.

  Args:
    xml_text: The XML string to extract from.
    tag_name_to_extract: The tag name to extract.

  Returns:
    The extracted XML string.
  """
  if '```xml' in xml_text:
    xml_str = xml_text.rpartition('```xml\n')[2].rpartition('```')[0]
  else:
    # If Gemini LLM returns xml without 'xml' in the string.
    xml_str = xml_text
  # Remove the text before the start of the tag.
  xml_str = (
      xml_str.rpartition(f'<{tag_name_to_extract}>')[1]
      + xml_str.rpartition(f'<{tag_name_to_extract}>')[2]
  )
  # Remove the text after the end of the tag.
  xml_str = (
      xml_str.rpartition(f'</{tag_name_to_extract}>')[0]
      + xml_str.rpartition(f'</{tag_name_to_extract}>')[1]
  )
  return xml_str


class SQLiteShardedAccessor:
  """Access a sharded SQLite database."""

  def __init__(self, paths, wrapper):
    """Initializes the SQLiteShardedAccessor.

    Args:
      paths: The paths to the SQLite databases.
      wrapper: The wrapper function to use.
    """
    self.paths = paths
    self.wrapper = wrapper

  def get(self, key):
    """Gets a value from the SQLite database.

    Args:
      key: The key to look up.

    Returns:
      The value from the SQLite database, or None if not found.
    """
    # Use md5 for stable sharding mostly compatible with typical hashing.
    # In a real scenario, this must match the writer's sharding.
    shard_idx = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % len(self.paths)
    path = self.paths[shard_idx]
    if not os.path.exists(path):
      return None
    try:
      with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM kv WHERE key=?", (key,))
        row = cursor.fetchone()
        if row:
          return self.wrapper(row[0])
    except sqlite3.Error as e:
      logger.error("SQLite error: %s", e)
    return None

def get_image_proto(
    url, sstable_name, feature_key
):
  """Get image bytes from a sqlite database.

  Args:
    url: The URL of the image.
    sstable_name: The name of the SQLite database.
    feature_key: The feature key to use.

  Returns:
    The image proto, or None if not found.
  """
  # We assume existing sstable paths are now sqlite paths.
  imgtable = SQLiteShardedAccessor(
      paths=generate_sharded_filenames(sstable_name),
      wrapper=tf.train.Example.FromString,
  )
  example = imgtable.get(url)
  if example is not None:
    image_bytes = example.features.feature['image/data'].bytes_list.value[0]
    image_features = get_image_embedding(
        example, feature_key
    )
    image = image_py.MyImage(image_bytes=image_bytes)
    image.image_features = image_features
    image.url = url
    return image
  return None


def get_image_proto_list(
    urls,
    sstable_name,
    feature_key,
    progress = False,
    filter_nones = True,
):
  """Parallel get image protos from an sstable.

  Args:
    urls: The URLs of the images.
    sstable_name: The name of the SQLite database.
    feature_key: The feature key to use.
    progress: Whether to show a progress bar.
    filter_nones: Whether to filter out None values.

  Returns:
    A list of image protos.
  """
  def get_proto_helper(kwargs):
    return get_image_proto(**kwargs)

  args_list = [
      {'url': url, 'sstable_name': sstable_name, 'feature_key': feature_key}
      for url in urls
  ]
  with concurrent_futures.ThreadPoolExecutor(max_workers=250) as executor:
    images = list(executor.map(get_proto_helper, args_list))
  if filter_nones:
    images = [example for example in images if example is not None]
  return images


def get_image_embedding(
    example,
    feature_name,
):
  """Get image embedding from a tf.train.Example proto.

  Args:
    example: The tf.train.Example proto.
    feature_name: The feature name to extract.

  Returns:
    The image embedding, or None if not found.
  """
  embedding = example.features.feature[feature_name].float_list.value
  return np.array(embedding)


def load_images(
    sstable_name,
    shard_to_load = None,
    feature_key = 'sbv5'
):
  """Load images from a sqlite database.

  Args:
    sstable_name: The name of the SQLite database.
    shard_to_load: The shard to load.
    feature_key: The feature key to use.

  Returns:
    A list of loaded images.
  """
  paths = generate_sharded_filenames(sstable_name)
  if shard_to_load is not None:
    paths = [paths[shard_to_load]]

  images = []
  for path in paths:
    if not os.path.exists(path):
      continue
    with sqlite3.connect(path) as conn:
      cursor = conn.cursor()
      # We might want to use a server-side cursor or fetchmany for large tables.
      # But basic iteration is fine for now.
      cursor.execute("SELECT key, value FROM kv")
      for key, value_bytes in cursor:
        try:
          example = tf.train.Example.FromString(value_bytes)
          image_bytes = example.features.feature['image/data'].bytes_list.value[0]
          image_features = get_image_embedding(
              example, feature_key
          )
          image = image_py.MyImage(image_bytes=image_bytes)
          image.image_features = image_features
          image.url = key
          images.append(image)
        except Exception as e:
          logger.error("Error processing row: %s", e)
  return images


def load_data(
    input_folder,
    shard_to_load,
):
  """Load data from an sstable.

  Args:
    input_folder: The input folder.
    shard_to_load: The shard to load.

  Returns:
    A tuple of keys and embeddings.
  """

  shard_path = os.path.join(input_folder, f'shard_{str(shard_to_load)}.pkl')

  with open(shard_path, 'rb') as f:
    data = pickle.load(f)

  keys = []
  embs = []
  for datum in data:
    key, embed = datum
    keys.append(key)
    embs.append(embed)
  return keys, embs


def get_element_text(element, tag):
  """Determine the text of an element if any.

  Args:
    element: The XML element.
    tag: The tag to look for.

  Returns:
    The text of the element, or None if not found.
  """
  tag_element = element.find(tag)
  if tag_element is not None:
    return tag_element.text
  return None


T = TypeVar('T')


def remove_ref_from_list(
    list_of_elements, element_to_remove):
  """Remove an element from a list.

  Args:
    list_of_elements: The list of elements.
    element_to_remove: The element to remove.

  Returns:
    The list of elements with the element removed.
  """
  indices_to_keep = []
  for i, element in enumerate(list_of_elements):
    if element is not element_to_remove:
      indices_to_keep.append(i)

  list_of_elements = [list_of_elements[i] for i in indices_to_keep]
  return list_of_elements


class PickleCache:
  """Class that caches URL --> Annotation."""

  def __init__(self, cache_path):
    """Initializes the PickleCache.

    Args:
      cache_path: The path to the cache file.
    """
    self.cache_path = cache_path
    if os.path.exists(cache_path):
      with open(cache_path, 'rb') as f:
        self.cache = pickle.load(f)
    else:
      self.cache = {}

  def __getitem__(self, key):
    return self.cache[key]

  def __setitem__(self, key, value):
    self.cache[key] = value

  def __contains__(self, key):
    return key in self.cache

  def save(self):
    """Saves the cache to the file."""
    with open(self.cache_path, 'wb') as f:
      pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)
      print('Saved at: ', self.cache_path)

def dictionary_learning(
    embeddings,
    n_components,
    alpha,
    mini_batch = True,
):
  """Dictionary learning.

  Args:
    embeddings: The embeddings to learn from.
    n_components: The number of components.
    alpha: The alpha parameter.
    mini_batch: Whether to use mini-batch dictionary learning.

  Returns:
    The trained dictionary learner object.
  """
  if mini_batch:
    dictionary_learner = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        transform_algorithm='lasso_cd',
        transform_max_iter=200,
        batch_size=256,
        max_iter=200,
        random_state=0,
    )
  else:
    dictionary_learner = DictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        transform_algorithm='lasso_cd',
        transform_max_iter=200,
        max_iter=200,
        random_state=0,
    )
  dictionary_learner.fit(embeddings)
  return dictionary_learner


def tune_dictionary_learning(
    embeddings,
    target_nonzeros=3.0,
):
  """Determine the best dictionary learning parameters for the given images.

  Args:
    embeddings: The embeddings to use.
    target_nonzeros: The target average number of nonzeros.

  Returns:
    A tuple of the learned dictionary components and the sparse representation
    matrix.
  """

  dictionary_sizes = [4, 5, 6, 8]
  alphas = (0.5, 0.8, 1.0, 1.5, 2.0)
  embeddings = normalize(embeddings.astype(np.float32))
  logger.debug(
      'Tune dictionary learning for %d embeddings; tried the following'
      ' parameters: %s, %s',
      len(embeddings),
      dictionary_sizes,
      alphas,
  )
  results = []

  for dict_size in dictionary_sizes:
    for alpha in alphas:
      dictionary_learner = dictionary_learning(
          embeddings,
          n_components=dict_size,
          alpha=alpha,
          mini_batch=True,
      )
      # (n_samples, dict_size)
      all_sparse_matrix = dictionary_learner.transform(embeddings)
      # (dict_size, n_dims)
      dictionary = dictionary_learner.components_

      # Metrics.
      recon = all_sparse_matrix.dot(dictionary)
      err = np.linalg.norm(embeddings - recon) / (
          np.linalg.norm(embeddings) + 1e-12
      )
      # Frac of images using each atom.
      nz = (np.abs(all_sparse_matrix) > 1e-6).sum(axis=1).mean()
      # Frac of images using each atom.
      util = (np.abs(all_sparse_matrix) > 1e-6).mean(axis=0)
      # Many dead atoms? bad.
      penalty_dead = (util < 0.05).mean()
      # Any monopoly atom? bad.
      penalty_over = max(0.0, util.max() - 0.60)
      # Simple composite score (lower is better).
      score = (
          err
          + 0.5 * (nz - target_nonzeros) ** 2
          + 0.5 * (penalty_dead + penalty_over)
      )

      info = dict(
          dict_size=dict_size,
          alpha=alpha,
          err=float(err),
          avg_nonzeros=float(nz),
          dead_frac=float(penalty_dead),
          top_util=float(util.max()),
          score=float(score),
      )
      results.append(info)

  results.sort(key=lambda r: r['score'])
  # Train the dictionary learner.
  logger.debug(
      'Best dictionary learning parameters: dict_size=%s, alpha=%s',
      results[0]['dict_size'],
      results[0]['alpha'],
  )
  logger.debug(
      'Descriptive stats: avg_nonzeros=%s, dead_frac=%s, top_util=%s',
      results[0]['avg_nonzeros'],
      results[0]['dead_frac'],
      results[0]['top_util'],
  )
  dictionary_learner = dictionary_learning(
      embeddings,
      n_components=results[0]['dict_size'],
      alpha=results[0]['alpha'],
      mini_batch=False,
  )
  dictionary_learner.fit(embeddings)
  dictionary_components = dictionary_learner.components_
  sparse_matrix = dictionary_learner.transform(embeddings)
  return dictionary_components, sparse_matrix
