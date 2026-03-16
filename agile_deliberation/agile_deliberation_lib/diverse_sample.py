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

import concurrent.futures as concurrent_futures
import logging
import os
import pickle
import time
from typing import Any, Optional

import numpy as np

from agile_deliberation_lib import definitions as definitions_py
from agile_deliberation_lib import image as image_py
from agile_deliberation_lib import utils as utils_py


Definition = definitions_py.Definition
MyImage = image_py.MyImage
logger = logging.getLogger(__name__)


class DiverseImageSampler:
  """Samples diverse images based on a definition, maintaining state.

  Attributes:
    reflection: The reflection unit or service.
    retrieval_client: The KnnService or retrieval client.
    definition: The Definition object.
    active_learning_batch: The number of images to reflect on at each round.
    reflecting_batch_size: The number of borderline images to surface for each
      round.
    weights: A dictionary of weights for calculating cluster scores.
    exploration_factor: The exploration factor for UCB strategy.
    definition_folder: The optional folder for storing logs and definitions.
    round_count: The number of rounds performed.
    image_pools: A list of MyImage objects.
    images_interactions: A dictionary containing interaction data for each
      image.
    images_explored_per_component: An array representing the number of images
      explored per dictionary component.
    all_explored_nums: The total number of explored images.
    dictionary_components: The dictionary components from dictionary learning.
    sparse_matrix: The sparse representations of images on dictionary
      components.
    images_for_each_component: A list of image indices associated with each
      component.
    components_for_each_image: A dictionary mapping image URLs to component
      indices.
  """

  def __init__(
      self,
      image_pools,
      reflection,
      retrieval_client,
      definition,
      active_learning_batch,
      reflecting_batch_size,
      definition_folder = None,
      log_path = None,
  ):
    """Initializes the DiverseImageSampler.

    Args:
      image_pools: The initial list of images.
      reflection: The reflection component.
      retrieval_client: The retrieval client.
      definition: The definition object to refine.
      active_learning_batch: Multiplier or size for active learning rounds.
      reflecting_batch_size: The batch size of reflecting candidates.
      definition_folder: Folder for logs.
      log_path: Optional path to load a saved session log.
    """
    self.reflection = reflection
    self.retrieval_client = retrieval_client
    self.definition = definition

    # How many images we want to reflect on at each round.
    self.active_learning_batch = active_learning_batch
    # How many images we will surface borderline images from for each round.
    self.reflecting_batch_size = reflecting_batch_size

    self.weights = {
        'mistake': 0.5,
        'feedback': 0.3,
        'exploration': 0.15,
        'diversity': 0.3,
    }
    self.exploration_factor = 2

    self.definition_folder = definition_folder
    if log_path is not None:
      self.load_logs(log_path)
    else:
      logger.info('Initializing image pools and clustering images.')
      self.round_count = 0
      self.image_pools = self.sample_valuable_images(image_pools)
      self.cluster_images()
      self.images_interactions = {}
      for image in self.image_pools:
        self.images_interactions[image.url] = {
            'visited_count': 0,
            'feedback_nums': 0,
            'whether_mistake': None,
            'user_rating': None,
        }
      self.images_explored_per_component = np.zeros(
          len(self.images_for_each_component)
      )
      self.all_explored_nums = 0

  def sample_valuable_images(self, image_pools):
    """Sample valuable images from the image pools.

    Args:
      image_pools: The total list of images available.

    Returns:
      The list of valuable image samples.
    """
    return image_pools

  def cluster_images(self):
    """Cluster images in dictionary learning manner."""
    self.image_pools = [
        img for img in self.image_pools if img.image_features is not None
    ]
    if not self.image_pools:
      logger.warning('No images with features available for clustering.')
      self.dictionary_components = np.array([])
      self.sparse_matrix = np.array([])
      self.images_for_each_component = []
      return
    image_features = np.array([img.image_features for img in self.image_pools])
    self.dictionary_components, self.sparse_matrix = (
        utils_py.tune_dictionary_learning(
            embeddings=image_features,
            target_nonzeros=3.0,
        )
    )
    dict_size = self.dictionary_components.shape[0]
    logger.info('Dictionary size: %d', dict_size)
    image_indices_for_each_component = [None for _ in range(dict_size)]
    for atom in range(dict_size):
      idx = np.where(self.sparse_matrix[:, atom] > 1e-6)[0]
      if idx.size:
        order = np.argsort(-self.sparse_matrix[idx, atom])
        idx = idx[order]
      image_indices_for_each_component[atom] = list(idx)
    self.images_for_each_component = image_indices_for_each_component

    components_for_each_image = {}
    for image_idx, image in enumerate(self.image_pools):
      idx = np.where(self.sparse_matrix[image_idx] > 1e-6)[0]
      if idx.size:
        order = np.argsort(-self.sparse_matrix[image_idx, idx])
        idx = idx[order]
      components_for_each_image[image.url] = list(idx)
    self.components_for_each_image = components_for_each_image

  def incorporate_feedbacks(
      self, feedbacks, definition
  ):
    """Incorporate feedbacks from image reflections.

    Args:
      feedbacks: A list of feedback dictionaries.
      definition: The updated Definition object.
    """
    for feedback in feedbacks:
      image_url = feedback['url']
      now_interaction = self.images_interactions[image_url]
      now_interaction['whether_mistake'] = (
          feedback['groundtruth'] != feedback['decision']
      )
      now_interaction['visited_count'] += 1
      if feedback['feedback']:
        now_interaction['feedback_nums'] = (
            now_interaction.get('feedback_nums', 0) + 1
        )
      now_interaction['user_rating'] = feedback['decision']
    for image in self.image_pools:
      if image.url in self.images_interactions:
        classifier_rating = image.retrieve_rating(definition)
        if classifier_rating is not None:
          self.images_interactions[image.url]['whether_mistake'] = (
              image.user_rating != classifier_rating['decision']
          )

  def _calculate_cluster_scores(self):
    """Calculates the score for each cluster based on the heuristics.

    Returns:
      An array of scores, one for each cluster.
    """
    assert self.images_for_each_component is not None
    assert self.images_explored_per_component is not None
    num_clusters = len(self.images_for_each_component)
    scores = np.zeros(num_clusters)
    for cluster_idx in range(num_clusters):
      explored_nums = 0
      feedback_nums = 0
      mistake_nums = 0
      ratings = []

      image_indices_in_cluster = self.images_for_each_component[cluster_idx]
      if not image_indices_in_cluster:
        scores[cluster_idx] = -1
        # Choose a bigger number to make sure this cluster will not be selected
        # again.
        self.images_explored_per_component[cluster_idx] = 100000
        continue

      for image_idx in image_indices_in_cluster:
        image_url = self.image_pools[image_idx].url
        now_interaction = self.images_interactions[image_url]
        if now_interaction:
          explored_nums += int(now_interaction['visited_count'] > 0)
          feedback_nums += int(now_interaction['feedback_nums'] > 0)
          if now_interaction['whether_mistake'] is not None:
            mistake_nums += int(now_interaction['whether_mistake'])
          if now_interaction['user_rating'] is not None:
            ratings.append(now_interaction['user_rating'])

      self.images_explored_per_component[cluster_idx] = explored_nums
      if explored_nums == 0:
        scores[cluster_idx] = 0.0
        continue

      # Heuristic 1: Mistake Rate (more valuable)
      mistake_score = mistake_nums / explored_nums

      # Heuristic 2: Feedback Rate (more valuable)
      feedback_score = feedback_nums / explored_nums

      # Heuristic 3: Exploration Rate (less valuable)
      # We favor clusters that are less explored.
      if image_indices_in_cluster:
        exploration_value = 1.0 - (
            explored_nums / len(image_indices_in_cluster)
        )
      else:
        exploration_value = 0.0

      # Heuristic 4: Rating Diversity (less valuable).
      # Higher standard deviation means more diverse ratings.
      if ratings:
        diversity_score = np.std(ratings) / 4.0
      else:
        diversity_score = 0.0

      # Combine scores with weights.
      weighted_score = (
          self.weights['mistake'] * mistake_score
          + self.weights['feedback'] * feedback_score
          + self.weights['exploration'] * exploration_value
          + self.weights['diversity'] * diversity_score
      )
      scores[cluster_idx] = weighted_score

    return scores

  def select_next_cluster(self):
    """Selects the next cluster to explore based on the cluster scores.

    Returns:
      The index of the selected cluster.
    """
    cluster_scores = self._calculate_cluster_scores()
    unpulled_clusters = np.where(cluster_scores == 0.0)[0]
    # If a cluster has never been pulled, prioritize it.
    if unpulled_clusters.size > 0:
      # Select the first one from the list
      chosen_cluster = unpulled_clusters[0]
      logger.info(
          'Selecting cluster %d because it has never been pulled',
          chosen_cluster,
      )
    else:
      exploration_bonus = self.exploration_factor * np.sqrt(
          np.log(self.round_count) / self.images_explored_per_component
      )
      ucb_scores = cluster_scores + exploration_bonus
      chosen_cluster = np.argmax(ucb_scores)
      logger.info(
          'Selecting cluster %d because it has the highest UCB score.',
          chosen_cluster,
      )
    return int(chosen_cluster)

  def select_next_images(self, component_idx):
    """Selects the next images to explore based on the cluster scores.

    Args:
      component_idx: The current component index to select images from.

    Returns:
      A list of selected MyImage objects.
    """
    image_indices_to_explore = []
    sampling_weights = []
    if not self.images_for_each_component[component_idx]:
      logger.error('No images found for component %d', component_idx)
      return []

    for image_idx in self.images_for_each_component[component_idx]:
      now_interaction = self.images_interactions[
          self.image_pools[image_idx].url
      ]
      if now_interaction['visited_count'] == 0:
        sampling_weights.append(1.0)
        image_indices_to_explore.append(image_idx)
      elif now_interaction['whether_mistake']:
        sampling_weights.append(2)
        image_indices_to_explore.append(image_idx)
    logger.debug(
        'We have %d images to explore in this round.',
        len(image_indices_to_explore),
    )
    if not image_indices_to_explore:
      return []
    # Normalize the weights.
    sampling_weights = np.array(sampling_weights, dtype=np.float64)
    sampling_weights /= np.sum(sampling_weights)

    # Sample images with weights.
    image_indices_to_explore = np.random.choice(
        image_indices_to_explore,
        size=min(self.reflecting_batch_size, len(image_indices_to_explore)),
        replace=False,
        p=sampling_weights,
    )
    return [self.image_pools[idx] for idx in image_indices_to_explore]

  def get_next_batch(self):
    """Gets the next batch of diverse images to be reviewed.

    This method selects an atom based on the current round count, samples images
    associated with that atom, and then uses reflection to surface borderline
    images from this sample.

    Returns:
      A list of `MyImage` objects representing the next batch for active
      learning.
    """
    start_time = time.time()

    # First pick the most important atom, and then get images for that atom.
    self.save_logs(self.round_count)
    component_idx_to_explore = self.select_next_cluster()
    now_images = self.select_next_images(component_idx_to_explore)
    cluster_future = concurrent_futures.ThreadPoolExecutor().submit(
        self.reflection.surface_borderline_images,
        definition=self.definition,
        images=now_images,
        min_cluster_size=self.active_learning_batch,
        max_eps=1,
    )
    final_images = cluster_future.result()

    logger.info(
        'found a new cluster of % d images in %d seconds',
        len(final_images),
        time.time() - start_time,
    )
    self.round_count += 1
    self.all_explored_nums += len(final_images)
    return final_images

  def load_logs(self, log_path = None):
    """Loads the state of the sampler from a log file.

    If a valid log path is provided and the file exists, the sampler's state
    including round count, image pools, interactions, and dictionary learning
    results are loaded. Otherwise, the state is initialized to defaults.

    Args:
      log_path: The file path from which to load the logs. If None or if the
        file does not exist, the sampler starts with an empty state.
    """
    if log_path is not None and os.path.exists(log_path):
      logger.info('Loading logs from: %s\n', log_path)
      with open(log_path, 'rb') as f:
        reflection_logs = pickle.load(f)
    else:
      reflection_logs = {}
    if 'definition' in reflection_logs:
      self.definition = reflection_logs['definition']
    self.round_count = reflection_logs.get('round_count', 0)
    self.image_pools = reflection_logs.get('image_pools', [])
    self.images_interactions = reflection_logs.get(
        'images_interactions', {}
    )
    self.images_for_each_component = reflection_logs.get(
        'images_for_each_component', None
    )
    self.components_for_each_image = reflection_logs.get(
        'components_for_each_image', None
    )
    self.all_explored_nums = reflection_logs.get('all_explored_nums', 0)
    self.images_explored_per_component = reflection_logs.get(
        'images_explored_per_component', None
    )
    if (
        self.images_explored_per_component is None
        and self.images_for_each_component is not None
    ):
      self.images_explored_per_component = np.zeros(
          len(self.images_for_each_component)
      )

  def save_logs(self, round_count):
    """Saves the current state of the sampler and the definition.

    The state includes image pools, interactions, round count, and the
    dictionary learning results. The definition is saved separately.

    Args:
      round_count: The current round number, used in the filename.
    """
    if self.definition_folder is None:
      return
    log_file_path = (
        f'{self.definition_folder}/logs_iteration_round_{round_count}.pkl'
    )
    logger.info('Saving logs for round %d to: %s', round_count, log_file_path)
    logs = {
        'definition': self.definition,
        'image_pools': self.image_pools,
        'images_interactions': self.images_interactions,
        'round_count': self.round_count,
        'images_for_each_component': self.images_for_each_component,
        'components_for_each_image': self.components_for_each_image,
        'all_explored_nums': self.all_explored_nums,
        'images_explored_per_component': self.images_explored_per_component,
    }
    with open(log_file_path, 'wb') as f:
      pickle.dump(logs, f)
