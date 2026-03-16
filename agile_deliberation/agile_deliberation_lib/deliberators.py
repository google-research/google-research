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

"""Concept deliberator that oversees the deliberation process."""

import logging
import os
import pickle
from typing import Any, Optional

from IPython.display import display
import ipywidgets as widgets

from agile_deliberation_lib import classifier as classifier_py
from agile_deliberation_lib import definitions as definitions_py
from agile_deliberation_lib import diverse_sample as diverse_sample_py
from agile_deliberation_lib import image as image_py
from agile_deliberation_lib import interaction as interaction_py
from agile_deliberation_lib import llm as llm_py
from agile_deliberation_lib import refine_definition as refine_definition_py
from agile_deliberation_lib import reflection as reflection_py
from agile_deliberation_lib import retrieval as retrieval_py
from agile_deliberation_lib import search_images as search_images_py


Definition = definitions_py.Definition
DiverseImageSampler = diverse_sample_py.DiverseImageSampler
MyImage = image_py.MyImage
logger = logging.getLogger(__name__)


def save_definition(definition, definition_filename):
  """Save the definition to a file.

  Args:
    definition: The definition to save.
    definition_filename: The filename to save to.
  """
  logger.info('Saving definition to: %s', definition_filename)
  with open(definition_filename, 'wb') as f:
    pickle.dump(definition, f)


class ConceptDeliberator:
  """Oversee the deliberation process."""

  def __init__(
      self,
      retrieval_client,
      model_client,
      definition_folder = None,
      keep_output = False,
  ):
    """Initializes the ConceptDeliberator.

    Args:
      retrieval_client: The retrieval client.
      model_client: The model client.
      definition_folder: The folder to store definitions.
      keep_output: Whether to keep output.
    """
    # How many images we want to reflect on at each round.
    self.active_learning_batch = 5
    # How many images we want to search for a given query.
    self.search_images_per_query = self.active_learning_batch * 2
    # How many images we want to have in the pool to surface borderline images
    # from for each round.
    self.min_images_to_reflect = self.active_learning_batch * 5
    # How far away we can tolerate for each image to be from the cluster center.
    self.max_image_clustering_distance = 0.8
    # How many rounds we want to run the deliberation process.
    self.iteration_rounds = 15
    # We want to save the definition at each round to a folder.

    self.definition_folder = definition_folder
    self.retrieval_client = retrieval_client
    self.image_curator = search_images_py.ImageCurator(
        retrieval_client, model_client, self.search_images_per_query
    )
    self.reflection = reflection_py.Reflection(
        retrieval_client, model_client, self.image_curator
    )
    self.classifier = classifier_py.ImageClassifier(model_client)
    self.refiner = refine_definition_py.DefinitionRefiner(
        model_client, self.classifier
    )
    self.interaction = interaction_py.DeliberationInteraction(
        self.reflection, self.classifier, self.refiner, keep_output=keep_output
    )
    self.image_sampler = None

  def update_model_client(self, model_client):
    """Updates the model client.

    Args:
      model_client: The new model client.
    """
    self.image_curator.model_client = model_client
    self.reflection.model_client = model_client
    self.classifier.model_client = model_client
    self.refiner.model_client = model_client

  def prepare_images_for_reflection(
      self, definition, total_images_num = 3000
  ):
    """Prepare images for reflection.

    Args:
      definition: The definition to use for reflection.
      total_images_num: The total number of images to prepare.

    Returns:
      A list of images prepared for reflection.
    """
    logger.info(
        'Preparing %d images for definition: %s',
        total_images_num,
        definition.concept,
    )
    all_images = []
    search_descriptions = [definition.description]
    previous_descriptions = []
    image_types = ['in-scope', 'ambiguous']
    round_count = 0
    search_images_num = 250
    while len(all_images) < total_images_num:
      while not search_descriptions:
        now_image_type = image_types[round_count % len(image_types)]
        now_descriptions = self.image_curator.generate_diverse_descriptions(
            definition, previous_descriptions, now_image_type
        )
        search_descriptions.extend(now_descriptions)
        search_images_num = 75
      now_description = search_descriptions.pop(0)
      previous_descriptions.append(now_description)
      new_images = self.retrieval_client.text_image_search(
          now_description, num_neighbors=search_images_num
      )
      logger.info(
          'Search %s images for description: %s', len(new_images),
          now_description,
      )
      all_images.extend(new_images)
      all_images = MyImage.deduplicate_images(all_images)
      round_count += 1
    logger.info('Total number of images: %d', len(all_images))
    return all_images

  def image_reflection(
      self,
      definition,
      reflect_images_path = None,
      log_path = None,
  ):
    """Reflect on images using a top-down and structured approach.

    Args:
      definition: The definition to use for reflection.
      reflect_images_path: The path to images to reflect on.
      log_path: The path to logs.
    """
    round_count = 0
    if reflect_images_path is not None:
      with open(reflect_images_path, 'rb') as f:
        images_to_reflect = pickle.load(f)
    elif log_path is not None:
      images_to_reflect = []
    else:
      images_to_reflect = self.prepare_images_for_reflection(
          definition, total_images_num=200
      )

    self.image_sampler = DiverseImageSampler(
        images_to_reflect,
        self.reflection,
        self.retrieval_client,
        definition,
        active_learning_batch=self.active_learning_batch,
        reflecting_batch_size=self.active_learning_batch * 6,
        definition_folder=self.definition_folder,
        log_path=log_path,
    )

    if log_path is not None:
      definition.update_content(self.image_sampler.definition)
      definition.groundtruth = self.image_sampler.definition.groundtruth

    def continue_reflection(
        new_definition = None,
        reflection_infos = None,
    ):
      """Continue reflection based on new definition and reflection infos.

      Args:
        new_definition: The new definition.
        reflection_infos: Actionable feedback points.
      """
      nonlocal round_count, definition
      if new_definition:
        definition.update_content(new_definition)

      if self.image_sampler is None:
        return

      if reflection_infos:
        self.image_sampler.incorporate_feedbacks(
            reflection_infos, new_definition
        )
      logger.info(
          '\nStart searching for a new cluster of images for reflection.'
      )

      images_to_reflect = self.image_sampler.get_next_batch()
      if not images_to_reflect:
        logger.info('No new images found for reflection.')
        return

      logger.info('round %d: %d images', round_count, len(images_to_reflect))
      self.interaction.image_reflections(
          definition, images_to_reflect, continue_reflection
      )

    # Give this stage its own fresh Output widget so it doesn't share state
    # with a previous stage (e.g. enrich_definitions) that also used _output.
    self.interaction._output = widgets.Output()
    display(self.interaction._output)

    continue_reflection(definition)

  def enrich_definitions(
      self, definition, log_path = None
  ):
    """Enrich definitions with sufficient signals.

    Args:
      definition: The definition to enrich.
      log_path: The path to enrichment logs.
    """

    if log_path is not None and os.path.exists(log_path):
      logger.info('Loading logs from: %s\n', log_path)
      with open(log_path, 'rb') as f:
        enrich_logs = pickle.load(f)
    else:
      enrich_logs = {}

    visited_logs = enrich_logs.get('visited_logs', {})
    now_definition = enrich_logs.get('now_definition', None)
    round_count = enrich_logs.get('round_count', 0)
    decompose_tried = enrich_logs.get('decompose_tried', False)

    def determine_next_definition():
      """Determine the next definition to enrich.

      Returns:
        The next definition to enrich, or None if there is none.
      """
      if not definition.necessary_signals:
        if definition.concept not in visited_logs:
          return definition
        elif visited_logs[definition.concept] < 2:
          return definition
        else:
          return None
      else:
        for signal in definition.necessary_signals:
          # If this signal has not been visited before.
          if signal.concept not in visited_logs:
            return signal
          if visited_logs[signal.concept] < 1:
            return signal
        return None

    def next_concept_fn():
      """Determine the next concept to enrich.

      Returns:
        The next concept string.
      """
      next_concept = determine_next_definition()
      if next_concept:
        return next_concept.concept
      else:
        return ''

    def save_logs():
      """Save the logs for the current round."""
      nonlocal round_count, definition, now_definition, visited_logs
      log_file_path = (
          f'{self.definition_folder}/logs_scoping_round_{round_count}.pkl'
      )
      logger.info('Saving logs for round %d to: %s', round_count, log_file_path)
      logs = {
          'definition': definition,
          'now_definition': now_definition,
          'visited_logs': visited_logs,
          'round_count': round_count,
          'decompose_tried': decompose_tried,
      }
      with open(log_file_path, 'wb') as f:
        pickle.dump(logs, f)

    def continue_enrichment(next_concept=False):
      """Continue enrichment with the next concept.

      Args:
        next_concept: Whether to proceed to the next concept.
      """
      nonlocal now_definition, round_count
      round_count += 1
      save_logs()
      if next_concept:
        now_definition = determine_next_definition()
      if now_definition:
        if now_definition.concept not in visited_logs:
          visited_logs[now_definition.concept] = 0
        visited_logs[now_definition.concept] += 1
        self.interaction.sufficient_signal_feedback_multiple(
            now_definition,
            continue_enrichment,
            next_concept_fn,
            signal_number=3
        )
      else:
        logger.info('All necessary signals have been enriched.')
        # Saving the definition here.
        save_definition(
            definition,
            f'{self.definition_folder}/definition_scoping.pkl',
        )

    # Give this stage its own fresh Output widget so it doesn't share state
    # with other stages.
    self.interaction._output = widgets.Output()
    display(self.interaction._output)

    if not definition.necessary_signals and not decompose_tried:
      self.interaction.decompose_concept(definition, continue_enrichment)
      decompose_tried = True
    else:
      continue_enrichment()
