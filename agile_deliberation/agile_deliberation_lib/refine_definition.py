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

"""Refine a definition based on user feedback."""

import logging
import math
import random
import time
from typing import Optional, Any

import numpy as np

from agile_deliberation_lib import classifier as classifier_py
from agile_deliberation_lib import definitions as definitions_py
from agile_deliberation_lib import image as image_py
from agile_deliberation_lib import llm as llm_py
from agile_deliberation_lib import utils


MyImage = image_py.MyImage
Definition = definitions_py.Definition
ImageClassifier = classifier_py.ImageClassifier
logger = logging.getLogger(__name__)


class DefinitionRefiner:
  """An agent that refines a definition according to user feedback.

  Attributes:
    image_classifier: The image classifier.
    bandit_exploration: The exploration factor for bandit algorithms.
    bandit_epochs: Optional number of epochs for bandit algorithm.
    bandit_batch_size: The batch size for bandit rounds.
    num_candidate_definitions: The number of candidate definitions to generate.
    model_client: The LLM client.
    number_of_few_shot_examples: The number of few-shot examples to add.
  """

  def __init__(
      self, model_client, image_classifier
  ):
    """Initializes the DefinitionRefiner.

    Args:
      model_client: The LLM model client.
      image_classifier: The image classifier instance.
    """
    self.image_classifier = image_classifier
    self.bandit_exploration = 0.1
    self.bandit_epochs = None
    self.bandit_batch_size = 5
    self.num_candidate_definitions = 5
    self.model_client = model_client
    self.number_of_few_shot_examples = 4

  def reflect_classified_image(
      self,
      item,
      definition,
  ):
    """Reflect on the classified image and propose improvements to the concept definition.

    Args:
      item: A tuple containing the MyImage object and its associated reflection
        info.
      definition: The current concept Definition.

    Returns:
      A dictionary representing the clarification extracted from the response, or
      None if parsing fails.
    """
    image, reflection_info = item
    reflect_classified_image_prompt = f"""
        <role>You are an expert linguist</role>.
        <input>
          You are given the definition of a visual concept and an image caption.
          Guided by this concept definition, human raters are asked to determine whether an image is within the scope of this concept, and write down their rationales and decisions.
          We also asked the concept owner to directly rate whether this image is in-scope or out-of-scope of this concept [ground-truth]
          The concept owner might also provide feedback regarding what the human raters should have noticed.
        </input>

        <task>
          The concept owner is still actively working on the definition of the target concept.
          The final goal is to enable human raters to interpret this definition to rate images exactly as the concept owner would do.
          Your task is to help articulate what this concept owner wants to clarify for this visual concept.
        </task>

        <define-a-concept>
        Before answering the question, it is a must for you to understand how we define a concept in a structured and iterative way.
        - If the concept is defined by a list of necessary conditions, then an image is in-scope if it satisfies all necessary conditions.
        - If the concept is defined by a list of positive and negative conditions, then an image is in-scope if it satisfies at least one positive condition and does not satisfy any negative condition.
        - A concept can be first defined by a list of necessary conditions, and then each necessary condition can be further defined by a list of positive and negative conditions.
        </define-a-concept>

        <step1>
        Within the context of this image, reason over what clarifications the concept owner might want to incorporate into the definition.
        You should try to refer to specific elements in the image to better ground your reasoning.
        Your reasoning should be based on the following questions:
        1) If the concept owner provides a clear feedback, what do you think the concept owner wants to clarify?
          Do not generalize too much beyond what the concept owner says.
        2) If the concept owner provides a different rating than the human raters, what is the possible reason for this disagreement?
          Do not generalize too much beyond this disagreement between ratings.
        3) When the concept owner provides no clear feedback and the human raters and the concept owner are in agreement,
          what does this agreement between the concept owner and the human raters confirm?
          Especially in this scenario, since there is no less clear information,
          you should be more conservative and specific, and try to avoid generalizing too much.
        </step1>
        <step2>
        Summarize your reasoning in the step 1 with a few sentences.
        Your summary will be used in downstream steps so make sure it covers all the necessary information.
        Please make sure that you ground your summary with specific elements or examples in the image.
        this will help others better understand your reasoning.
        </step2>

        Provide your answer in a valid XML format, adhering to the following structure:
          <reasoning>Describe your reasoning in the step 1</reasoning>
          <clarification>Provide your answer to the question in the step 2</clarification>

        <conceptDefinition>{definition.readable_string()}</conceptDefinition>
        <raterResponses>
          <decision>{ImageClassifier.rating_to_label(reflection_info['decision'])}</decision>
          <summary>{reflection_info['summary']}</summary>
        </raterResponses>
        <conceptOwner>
          <ground-truth>{ImageClassifier.rating_to_label(reflection_info['groundtruth'])}</ground-truth>
          <user-feedback>{reflection_info['feedback']}</user-feedback>
        </conceptOwner>
        <image_caption>{image.image_caption}</image_caption>
    """
    response = self.model_client.gemini_image_prompt(
        reflect_classified_image_prompt,
        [image],
        cheap=True,
    )
    # logger.info('reflect on a classified image\n%s', response)
    clarification = self.model_client.parse_xml(
        response, 'root', ['clarification']
    )
    clarification = clarification[0] if clarification else None
    if isinstance(clarification, dict):
      return clarification
    return None

  def improve_definition(
      self,
      definition,
      images,
      reflection_infos,
  ):
    """Improve the definition based on the reflections.

    Args:
      definition: The current Definition.
      images: A list of MyImage instances that were evaluated.
      reflection_infos: A list of dictionaries containing the reflection data
        for each image.

    Returns:
      A new improved Definition object, or None if improvement fails.
    """
    images_num = len(images)

    reflections_str = ''
    for reflection_info in reflection_infos:
      now_priority = reflection_info['priority']
      if now_priority == 2:
        reflections_str += f"""
          <clarification>{reflection_info['reflection']}</clarification>
        """
      # elif random.random() < 0.5 + now_priority / 4:
      #   reflections_str += f"""
      #     <clarification>{reflection_info['reflection']}</clarification>
      #   """
    if not reflections_str:
      # Randomly sample two reflections to improve the definition.
      random.shuffle(reflection_infos)
      reflections_str = ''
      for reflection_info in reflection_infos[:2]:
        reflections_str += f"""
          <clarification>{reflection_info['reflection']}</clarification>
        """

    improve_definition_prompt = f"""
        <role>You are an expert linguist</role>.
        <input>
          You are given the definition of a visual concept, which serves as the guideline for human raters to determine whether an image is within the scope of this concept.
          However, as the concept owner is still actively working on the definition of the target concept, human raters incorrectly rated an image regarding a particular focus concept within this definition.
          Therefore, the concept owner wants to clarify their points and improve the definition of this focus concept.
          They provide their clarifications for each of {images_num} images respectively.
          The final goal is to have a more accurate and comprehensive concept definition so that human raters can rate images exactly as the concept owner would do.
        </input>

        <task>
          Your task is to determine how to improve the definition of the focus concept in the most accurate and concise way.
        </task>

        <define-a-concept>
        Before answering the question, it is a must for you to understand how we define a concept in a structured and iterative way.
        - If the concept is defined by a list of necessary conditions, then an image is in-scope if it satisfies all necessary conditions.
        - If the concept is defined by a list of positive and negative conditions, then an image is in-scope if it satisfies at least one positive condition and does not satisfy any negative condition.
        - A concept can be first defined by a list of necessary conditions, and then each necessary condition can be further defined by a list of positive and negative conditions.
        </define-a-concept>

        <step1>
        Examine the clarifications for all these images and summarize the key points that the concept owner might want to incorporate into the definition.
        Some clarifications might be similar, so you should aggregate them;
        On the other hand, some clarifications might be very different, so you should consider listing them separately.
        </step1>
        <step2>
        Based on your answers in previous steps, reason how to incorporate the key points into the definition.
        You could either choose to add a new positive or negative signal to the definition, or modify an existing positive or negative signal.
        For an existing positive or negative signal, you should not add a child positive or negative signal to it.
        For an existing necessary condition, you should not add a sibling positive/negative/necessary signal to it.
        </step2>
        <step3>
        Based on your answers in previous steps, write down the changes you want to make to the definition.

        You should be careful about the language of your changes, in particular, there are several requirements.
        <description-requirements>
          1) Always make sure that your final description is CONCISE, COHERENT, and ACCURATE; an average person could easiy determine whether an image satisfies the signal based on the description.
          2) DO NOT write a complex sentence structure in a description of a signal.
          3) You should only make important changes to the description.

          If the original description misses a point, you are encouraged to use one of the following ways to incorporate the nuances the concept owner wants to convey:
          a) add new adjectives, b) use different verbs, or c) add a few constraint words.
          If the original description uses an ambiguous or misleading word or example, you are encouraged to a) refine the word or example, or b) simply remove them from the description.

          4) Be careful about your word choices of verbs, nouns, or adjectives, which might carry unexpected nuances.
              e.g., be careful about using 'depict' or 'mention', or 'show' as the previous two verbs introduce the slight emphasis on visual or textual aspects.
              e.g., be careful about using adjectives like 'clearly' or 'explicitly' as they might suggest a degree of visibility to the original concept.
          5) If your description consists of two independent conditions, you might consider use the format like "Images that 1) ... and 2) ..." to make it more clear.
        </description-requirements>

        </step3>

        Provide your answer in a valid XML format, adhering to the following structure:
          <keypoints>Describe your reasoning of the key points of these clarifications in the step1</keypoints>
          <reasoning>Describe your reasoning of how to incorporate the key points into the definition in the step2</reasoning>
          <improve-description>
            The improved description of the visual concept in the step3.
            You should only write down changes you proposed in the following format.
            1) If you want to edit an existing signal, the format is as follows:
            <concept>
              <old-name>The name of the signal you want to edit</old-name>
              <old-description>The original description of the signal</old-description>
              <new-description>The new description of the signal</new-description>
            </concept>
            2) If you want to add a new signal, the format is as follows:
            <concept>
              <parent-signal>The name of the parent signal</parent-signal>
              <type>The type of the new signal, either 'positive' or 'negative'</type>
              <new-name>The new name of the signal</new-name>
              <new-description>The new description of the signal</new-description>
            </concept>
            It might be possible that you need to make multiple changes, so you should write down all of them.
          </improve-description>

        <conceptDefinition>{definition.print_definition()}</conceptDefinition>
        <clarifications>{reflections_str}</clarifications>
      """
    retries = 3
    while True:
      try:
        response = self.model_client.gemini_prompt(
            improve_definition_prompt,
            cheap=True,
        )
        # logger.info('improve by changing description\n%s', response)
        proposed_improvements = self.model_client.parse_xml(
            response,
            'improve-description/concept',
            [
                'old-name',
                'new-name',
                'old-description',
                'new-description',
                'parent-signal',
                'type',
            ],
        )
        logger.debug('proposed_improvements: %s', proposed_improvements)
        new_definition = definition.copy()
        for improvement in proposed_improvements:
          if 'old-name' in improvement:
            focus_signal = new_definition.look_up_signal_by_concept(
                improvement['old-name']
            )
            if focus_signal:
              focus_signal.description = improvement['new-description']
          elif 'parent-signal' in improvement and 'type' in improvement:
            parent_signal = new_definition.look_up_signal_by_concept(
                improvement['parent-signal']
            )
            if parent_signal:
              new_signal = {
                  'name': improvement['new-name'],
                  'description': improvement['new-description'],
              }
              parent_signal.update_signals([new_signal], improvement['type'])
        return new_definition
      except Exception as e:
        retries = retries - 1
        if retries == 0:
          logger.error('Failed to improve definition: %s', e)
          return None

  def bandit_rank_candidates(
      self,
      candidates,
      images,
      groundtruths,
  ):
    """Rank the definition candidates based on the bandit algorithm.

    Args:
      candidates: A list of candidate Definitions to rank.
      images: A list of MyImage instances used for evaluation.
      groundtruths: A list of integer ground truth ratings corresponding to the
        images.

    Returns:
      A list of Definition candidates sorted by their performance scores.
    """
    logger.info(
        'Running bandit strategy with %d definition candidates'
        ' on a total of %d images.',
        len(candidates),
        len(images),
    )
    records = []
    counts = [0] * len(candidates)
    values = [0] * len(candidates)

    def select_best_arm(t):
      for arm in range(len(candidates)):
        # If any arm has not been pulled yet, choose it to explore.
        if counts[arm] == 0:
          return arm

        ucb_scores = []
        for arm in range(len(candidates)):
          average_reward = values[arm]
          bonus = self.bandit_exploration * ((np.log(t) / counts[arm])) ** 0.5
          ucb_scores.append(average_reward + bonus)
        best_arm = np.argmax(ucb_scores)
        return best_arm

    bandit_batch_size = self.bandit_batch_size
    if self.bandit_epochs:
      epochs = self.bandit_epochs
    else:
      epochs = math.ceil(
          math.ceil(len(images) / bandit_batch_size) * len(candidates) * 0.4
      )
    logger.debug(
        '\tWith %d rounds, batch size %d, and'
        ' exploration factor %s.',
        epochs,
        bandit_batch_size,
        self.bandit_exploration,
    )
    for t in range(epochs):
      sampled_indices = random.sample(range(len(images)), bandit_batch_size)
      sampled_images = [images[index] for index in sampled_indices]
      sampled_groundtruths = [
          groundtruths[index] for index in sampled_indices
      ]

      best_arm = select_best_arm(t)
      counts[best_arm] += 1
      best_candidate = candidates[best_arm]
      sampled_predictions = self.image_classifier.classify_images_in_parallel(
          sampled_images, best_candidate
      )
      sampled_predictions = [
          prediction['decision'] for prediction in sampled_predictions
      ]
      performance = self.image_classifier.evaluate_performance(
          sampled_predictions, sampled_groundtruths
      )
      values[best_arm] += performance['f1'] / counts[best_arm]

      actual_best_arm = np.argmax(values)
      records.append({
          'round': t,
          'actual_best_arm': actual_best_arm,
          'count': counts[actual_best_arm],
      })
    # Rank the candidates based on their values in the last round.
    # We pair each candidate with its value to sort them together.
    ranked_candidates = sorted(
        zip(candidates, values),
        key=lambda item: item[1],  # Sort by the 'value' score.
        reverse=True,  # Highest value first.
    )
    # Return topk candidates with the highest values.
    top_performers = [candidate for candidate, _ in ranked_candidates]
    return top_performers

  def overall_select_best_filters(
      self,
      candidates,
      images,
      groundtruths,
  ):
    """Select the best definition candidate based on the overall performance.

    Args:
      candidates: A list of candidate Definitions to filter.
      images: A list of MyImage instances for evaluation.
      groundtruths: A list of integer ground truth ratings corresponding to the
        images.

    Returns:
      A list of Definition candidates sorted by their F1 score.
    """
    logger.info(
        'Running overall strategy with %d definition candidates'
        ' on a total of %d images.',
        len(candidates),
        len(images),
    )
    performances = []
    for candidate in candidates:
      predictions = self.image_classifier.classify_images_in_parallel(
          images, candidate
      )
      predictions = [prediction['decision'] for prediction in predictions]
      performance = self.image_classifier.evaluate_performance(
          predictions, groundtruths
      )
      performances.append(performance)
    for index, performance in enumerate(performances):
      logger.info(
          'Candidate %d: f1 %.2f; precision %.2f; recall %f; accuracy %.2f',
          index,
          performance['f1'],
          performance['precision'],
          performance['recall'],
          performance['accuracy'],
      )

    # Rank candidates based on their f1 scores.
    # We pair each candidate with its performance dict to sort them together.
    ranked_candidates = sorted(
        zip(candidates, performances),
        key=lambda item: (item[1]['f1'], item[1]['accuracy']),
        # Sort by the 'f1' score.
        reverse=True  # Highest F1 score first.
    )
    # Rank candidates with the highest f1 scores.
    top_performers = [candidate for candidate, _ in ranked_candidates]
    return top_performers

  def filter_performant_candidates(
      self,
      definition_candidates,
      images,
      groundtruths,
      method = 'bandit',
      top_n = 1,
  ):
    """Filter the definition candidates based on the performance.

    Args:
      definition_candidates: A list of candidate definitions.
      images: A list of MyImage instances.
      groundtruths: A list of ground truth ratings.
      method: The filtering method to use, either 'bandit' or 'overall_best'.
      top_n: The maximum number of top candidates to return.

    Returns:
      A list of the top N filtered Definition candidates.

    Raises:
      ValueError: If the filtering method is unsupported.
    """
    if top_n >= len(definition_candidates):
      logger.warning(
          'The topN %s is larger than the number of definition candidates %s.'
          ' Return all candidates directly.',
          top_n,
          len(definition_candidates),
      )
      return definition_candidates

    if method not in ['bandit', 'overall_best']:
      raise ValueError('Unsupported method: {}'.format(method))

    ranked_candidates = definition_candidates
    if method == 'bandit':
      if len(images) <= 30:
        method = 'overall_best'
      else:
        ranked_candidates = self.bandit_rank_candidates(
            definition_candidates, images, groundtruths
        )

    if method == 'overall_best':
      ranked_candidates = self.overall_select_best_filters(
          definition_candidates, images, groundtruths
      )
    return ranked_candidates[:top_n]

  def evaluate_definition_candidates(
      self,
      definition,
      focus_images,
      definition_candidates,
  ):
    """Evaluate the definition candidates.

    Args:
      definition: The original baseline Definition.
      focus_images: A list of focused MyImage instances.
      definition_candidates: A list of new candidate Definitions.

    Returns:
      The most optimal Definition candidate.
    """
    if len(definition_candidates) == 1:
      return definition_candidates[0]

    # It should not at least be worse than the original definition.
    definition_candidates.append(definition)
    focus_groundtruths = [
        int(image.user_rating) if image.user_rating is not None else 0
        for image in focus_images
    ]

    # We first make sure we have a good performance on these focused images.
    logger.debug('Starting performance filtering on focus images...')
    start_time_focus = time.time()
    definition_candidates = self.filter_performant_candidates(
        definition_candidates,
        focus_images,
        focus_groundtruths,
        method='overall_best',
        top_n=3,
    )
    end_time_focus = time.time()
    elapsed_time_focus = end_time_focus - start_time_focus
    logger.info(
        'Focus image filtering took: %.2f seconds.', elapsed_time_focus
    )

    if len(definition_candidates) == 1:
      return definition_candidates[0]

    # Then we run a bandit algorithm to find the best definition
    # among the remaining candidates.
    logger.debug('Starting bandit algorithm to find the best definition...')
    start_time_bandit = time.time()

    test_instances = definition.groundtruth
    test_images = [instance['image'] for instance in test_instances]
    test_groundtruths = [instance['rating'] for instance in test_instances]
    definition_candidates = self.filter_performant_candidates(
        definition_candidates,
        test_images,
        test_groundtruths,
        method='overall_best',
        top_n=1,
    )

    end_time_bandit = time.time()
    elapsed_time_bandit = end_time_bandit - start_time_bandit
    logger.info(
        'Bandit algorithm filtering took: %.2f seconds.', elapsed_time_bandit
    )
    return definition_candidates[0]

  def refine_definition(
      self,
      images,
      definition,
      reflection_infos,
  ):
    """Reflect on the classified image and propose improvements to the concept definition.

    Args:
      images: A list of MyImage instances.
      definition: The Definition object to be refined.
      reflection_infos: A list of dictionaries containing initial reflection
        information.

    Returns:
      The optimal Definition after refining, generating candidates, and evaluating.
    """
    items_to_process = list(zip(images, reflection_infos))

    logger.debug('\nStarting reflecting on %d images...', len(images))
    start_time = time.time()
    reflections = utils.run_in_batches(
        items_to_process,
        self.reflect_classified_image,
        batch_size=10,
        definition=definition,
    )
    logger.info(
        'Reflecting on images took: %.2f seconds.', time.time() - start_time
    )
    for index, reflection_info in enumerate(reflection_infos):
      logger.debug('reflection %d: %s', index, reflections[index])
      reflection_info['reflection'] = reflections[index]
      if reflection_info['groundtruth'] != reflection_info['decision']:
        reflection_info['priority'] = 2
      elif not reflection_info['feedback']:
        reflection_info['priority'] = 2
      else:
        reflection_info['priority'] = 0

    ### Generate candidates for definition improvement. ###
    logger.debug(
        '\n\nStarting improving definition; %d candidates',
        self.num_candidate_definitions,
    )
    start_time = time.time()
    improved_candidates = utils.run_in_batches(
        [definition] * self.num_candidate_definitions,
        self.improve_definition,
        batch_size=10,
        images=images,
        reflection_infos=reflection_infos,
    )
    logger.info(
        'Improving definition took: %.2f seconds.', time.time() - start_time
    )
    # for index, candidate in enumerate(improved_candidates):
    #   logger.debug('candidate %d: %s', index, candidate.readable_string())

    ### Evaluate the candidates. ###
    logger.info('Starting evaluating definition candidates...')
    start_time = time.time()
    optimal_candidate = self.evaluate_definition_candidates(
        definition,
        images,
        improved_candidates,
    )
    logger.info(
        'Evaluating definition candidates took in total: %.2f seconds.',
        time.time() - start_time,
    )
    return optimal_candidate

  def add_few_shot_examples(self, definition):
    """Add few shot examples to the definition.

    Args:
      definition: The Definition to which few-shot examples will be added.

    Returns:
      The modified Definition with few-shot examples included.
    """
    logger.info('Adding few shot examples to the definition...')
    start_time = time.time()

    # This is not yet implemented.
    # TODO: Add few shot examples to the definition.

    logger.info(
        'Adding few shot examples took: %.2f seconds.', time.time() - start_time
    )
    return definition
