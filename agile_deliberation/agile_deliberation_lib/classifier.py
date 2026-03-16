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

"""Image Classifier that classifies images based on the structured definition."""

from collections import Counter
import concurrent.futures as concurrent_futures
import logging
from typing import Optional, Union, Any

import sklearn.metrics

from agile_deliberation_lib import definitions as definitions_py
from agile_deliberation_lib import image as image_py
from agile_deliberation_lib import llm as llm_py
from agile_deliberation_lib import utils as utils_py


MyImage = image_py.MyImage
Definition = definitions_py.Definition

logger = logging.getLogger(__name__)


class ImageClassifier:
  """Classify images in a modeling copilot way.

  Attributes:
    cheap_model: True if using a cheap model, False otherwise.
    model_client: The LLM model client.
  """

  def __init__(
      self, model_client, cheap_model = True
  ):
    """Initializes the ImageClassifier.

    Args:
      model_client: The language model client.
      cheap_model: Whether to use the cheaper model.
    """
    self.cheap_model = cheap_model
    self.model_client = model_client

  @classmethod
  def rating_to_label(cls, rating):
    """Convert a rating to a label.

    Args:
      rating: The numerical rating, either as string or int.

    Returns:
      'In-scope' if rating >= 3, else 'Out-of-scope'.
    """
    if isinstance(rating, str) and rating.isnumeric():
      rating = int(rating)
    if rating >= 3:
      return 'In-scope'
    else:
      return 'Out-of-scope'

  @classmethod
  def evaluate_performance(
      cls,
      predictions,
      groundtruths,
  ):
    """Evaluate performance of predictions against ground truth.

    Args:
      predictions: A list of predicted ratings.
      groundtruths: A list of ground truth ratings.

    Returns:
      A dictionary containing accuracy, precision, recall, and f1 scores.

    Raises:
      ValueError: If the length of predictions and groundtruths differ.
    """
    if len(predictions) != len(groundtruths):
      raise ValueError(
          'The length of predictions and groundtruths must be the same.'
      )
    y_pred = [pred >= 3 for pred in predictions]
    y_true = [groundtruth >= 3 for groundtruth in groundtruths]

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

    # Set zero_division=1.0 in case there are no positive or negative samples.
    # This avoids warnings and division-by-zero errors.
    precision = sklearn.metrics.precision_score(
        y_true, y_pred, zero_division=1.0
    )
    recall = sklearn.metrics.recall_score(y_true, y_pred, zero_division=1.0)
    f1_score = sklearn.metrics.f1_score(y_true, y_pred, zero_division=1.0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
    }

  @classmethod
  def determine_correct_rating(
      cls,
      decision,
      correct_rating,
  ):
    """Determine if the decision is correct relative to the true rating.

    Args:
      decision: The predicted rating.
      correct_rating: The ground truth rating.

    Returns:
      True if both are in-scope or both are out-of-scope, False otherwise.
    """
    if isinstance(decision, str) and decision.isnumeric():
      decision = int(decision)
    if isinstance(correct_rating, str) and correct_rating.isnumeric():
      correct_rating = int(correct_rating)
    if isinstance(decision, int) and isinstance(correct_rating, int):
      return (decision >= 3) == (correct_rating >= 3)
    else:
      logger.warning('The decision or the correct rating is not an integer.')
      return False

  def generate_image_captions(self, image):
    """Generate image captions.

    Args:
      image: The MyImage object to caption.

    Returns:
      The generated image caption as a string.
    """
    if image.image_caption is not None:
      return image.image_caption

    generate_image_caption_prompt = """
      <role>You are an expert image annotator.</role>
      <input>You will be provided with a single image.</input>
      <task>
      - Thoroughly examine the image.
      - Carefully consider all details of the image.
      - Provide an accurate description of the image.
      </task>

      <requirements>
        1. Your generated description should be detailed.
        2. You generated description should be accurate.
        3. Your generate should not include any elements that are not in the image.
        4. Avoid adding your own assumptions to your description; you should only add your interpretations if it would be generally agreed.
          If there are ambiguities in the image, provide all possible answers.
        5. Make sure that you pay attention to the following aspects of the image in your description if they are relevant.
        But you do not need to write your descriptions in the order of these aspects.
          - all the objects in the image (e.g., people, plants, animals, things, etc.)
          - the activities, appearances, location of these objects.
          - the setting or the context of the image.
          - the texts in the image.
          - the image angle or frame (e.g. a look-up angle, a zoomed-in view, etc.)
          - the image art styles (e.g., cartoon, photo, skethes, etc.)
          - the main focus of the image.
      </requirements>

      You are such an amazing annotator that I am 100% confident that you will do \
      this task without any mistake and hallucination. Thanks a ton in advance!
      Provide your output in a valid XML format, adhering to the following structure:
      Do not forget to add the final </description> tag.
        <description>Your description in the step2 here</description>
      """

    response = self.model_client.gemini_image_prompt(
        generate_image_caption_prompt,
        [image],
        cheap=True
    )
    logger.debug('response for image caption')
    image_description = self.model_client.parse_xml(
        response, 'root', ['description']
    )
    image.image_caption = image_description[0]['description']
    return image.image_caption

  def _evaluate_image_for_a_simple_concept(
      self,
      image,
      definition,
  ):
    """Evaluate image for a simple concept.

    Even though the definition might be further defined,
    we only use one prompt to evaluate it to save computation.

    Issues: when feeding images completely irrelevant to the given concept, the
    LLMs will run into a loop.

    Args:
      image: The image to be evaluated.
      definition: The definition of the concept.

    Returns:
      A dict containing the decision and summary, or None if parsing fails.
    """
    previous_rating = image.retrieve_rating(definition)
    if previous_rating is not None:
      return previous_rating

    if not image.image_caption:
      self.generate_image_captions(image)

    simple_concept_evaluate_image_prompt = f"""
      <role>You are an expert image annotator.</role>
      <input>You will be provided with a single image, an image caption and a concept definition.</input>

      <task>
      - Thoroughly examine the image.
      - Carefully consider all details of the concept
      - Determine if the image satisfies every aspect of the given concept.
      </task>

      <step1>
      Explicitly write down the specific requirements this concept demands. Make sure your decomposition is equivalent to the original definition.
      There are two special cases that demand your attention:
      - If the concept is defined by a list of necessary conditions, you should start by checking each condition explicitly.
        You should only give a high rating if all conditions are satisfied.
      - If the concept is defined by a list of positive and negative conditions, you should start by checking each condition explicitly.
        You should give a high rating if any of the positive conditions are satisfied and none of the negative conditions are satisfied.
      - A concept can be first defined by a list of necessary conditions, and then each necessary condition can be further defined by a list of positive and negative conditions.
        You should follow the same logic recursively to check if the image satisfies the condition.
      </step1>
      <step2>
      Based on responses in the first step, for each condition, you should determine whether the image satisfies the condition.
      </step2>
      <step3>
      You should then combine your responses for individual conditions to determine whether the image satisfies the overall concept or not.
      Your reasoning should be based solely on the definition, the image, and the image caption.
      Do not include any information that is not provided (no hallucinations).
      </step3>
      <step4>Based on your reasoning in the first step, determine whether the image completely fulfills every aspect of the concept.
      You should rate how in-scope the image is on a 1-5 Likert Scale where
        - Rate 5 if the image fully aligns with the concept
        - Rate 4 if the image mostly aligns with the concept; the small problem is a result of small ambiguities in the definition or small visual complexities in the image
        - Rate 3 if there are no strong evidence that indicate that the image violates the concept, but the supporting evidence is also not strong enough to support a rating of 4 or 5.
        - Rate 2 if the image violates some parts of the concept but there are some elements that are relevant to the concept.
        - Rate 1 if the image does not align with the concept description at all.
      </step4>
      <step5>
      Based on your answer in previous steps, provide a one-sentence summary why you give this answer.
      </step5>

      Provide your answer in the following XML structure:
        <requirements>Explicitly write down the specific requirements this concept demands.</requirements>
        <condition-eval>Write down your evaluation for each condition at step2 here.</condition-eval>
        <evaluation>You should differentiate between concept definitions that requires the satisfaction of all conditions and those that only require the satisfaction of one of the conditions. Describe your evaluation reasonings in the step4 here</evaluation>
        <decision>Rate on a 1-5 Likert Scale where 5 means the image is fully in-scope and 1 means the image is fully out-of-scope.</decision>
        <summary>Provide your summary in the step5 here</summary>
      The output will be later wrapped in a <root> tag, so do not wrap the content above in any tag such as xml, or root in your output.
    """

    classify_prompt = simple_concept_evaluate_image_prompt
    classify_prompt += f"""
        <input>
        Here is the criterion for you to work on:
          <criterion>{str(definition)}</criterion>

        Here is the high level image caption provided by an image annotator.
        As human raters might make mistakes when writing the caption, make sure you reference the provided image to verify them.
        Even though the caption provides key details about the image, the importance of each detail may vary visually.
        Always assess how clearly the image conveys each piece of information to an average viewer.
        For information that are emphasized, you should pay more attention to them; for information that is barely visible, you should pay less attention to them.
          <image_caption>{image.image_caption}</image_caption>
        </input>
    """
    response = self.model_client.gemini_image_prompt(
        classify_prompt,
        [image],
        cheap=self.cheap_model,
    )
    logger.debug('evaluate images for simple concepts')
    evaluation = self.model_client.parse_xml(
        response, 'root', ['decision', 'summary']
    )
    if len(evaluation) == 1 and isinstance(evaluation[0], dict):
      prediction: dict[str, Any] = evaluation[0]
      prediction['decision'] = int(prediction['decision'])
      return prediction
    else:
      return None

  def _stringify_queries(
      self, queries, signal_type
  ):
    """Stringify queries.

    Args:
      queries: A list of tuples containing the Definition and the answer dict.
      signal_type: The type of signal.

    Returns:
      A string representation of the queries.
    """
    query_string = ''
    for signal_definition, answer in queries:
      query_string += f"""
        <{signal_type}-query>
          <signal>{signal_definition.concept}:{signal_definition.description}</signal>
          <answer>{answer['decision']}</answer>
          <reasoning>{answer['summary']}</reasoning>
          <scenario>{answer.get('scenario', 'Not available')}</scenario>
        </{signal_type}-query>
      """
    return query_string

  def add_image_ocr_text(self, image):
    """Add image ocr text.

    Args:
      image: The MyImage object to add OCR text for.
    """
    if image.ocr_text is not None:
      return

    image_ocr_text_prompt = """
      Help translate any text in the image into English.
      Put your answer in a valid XML format, adhering to the following structure:
        <ocr_text>
          Put your answer here.
        </ocr_text>
      The output will be later wrapped in a <root> tag, so do not wrap the content above in any tag such as xml, or root in your output.
    """
    response = self.model_client.gemini_image_prompt(
        image_ocr_text_prompt,
        [image],
        cheap=True,
    )
    logger.debug('Add image ocr text')
    image_ocr_text = self.model_client.parse_xml(
        response, 'root', ['ocr_text']
    )
    if image_ocr_text and len(image_ocr_text) == 1:
      image.ocr_text = image_ocr_text[0]['ocr_text']

  def _evaluate_leaf_signals(
      self, image, signals
  ):
    """Evaluate leaf signals for an image.

    Args:
      image: The image to evaluate.
      signals: A list of leaf signals.

    Returns:
      A list of tuples containing the Definition and the associated answer.

    Raises:
      ValueError: If evaluation fails for any signal.
    """
    queries = []
    with concurrent_futures.ThreadPoolExecutor(max_workers=3) as executor:
      for index in range(0, len(signals), 3):
        batch_of_signals = signals[index : index + 3]
        future_to_signal = {
            executor.submit(
                self.classify_image, image, signal
            ): signal for signal in batch_of_signals
        }
        for future in concurrent_futures.as_completed(future_to_signal):
          signal = future_to_signal[future]
          answer = future.result()
          if answer:
            queries.append((signal, answer))
          else:
            raise ValueError(
                f'Failed to evaluate image for signal {signal}'
            )
    return queries

  def _evaluate_signals(
      self, image, signals
  ):
    """Generate queries for a list of signals.

    Args:
      image: The image to evaluate.
      signals: A list of signals to evaluate.

    Returns:
      A list of tuples containing the Definition and the associated answer.

    Raises:
      ValueError: If evaluation fails for any signal.
    """
    queries = []
    for signal_definition in signals:
      answer = self.classify_image(
          image, signal_definition
      )
      if answer:
        queries.append((signal_definition, answer))
      else:
        raise ValueError(
            f'Failed to evaluate image for signal {signal_definition}'
        )
    return queries

  def _evaluate_image_for_a_complex_concept(
      self, image, definition
  ):
    """Evaluate image for a complex concept.

    Args:
      image: The underlying image to evaluate.
      definition: The complex concept definition.

    Returns:
      A dictionary containing the evaluation result or None if parsing fails.
    """
    if not image.image_caption:
      self.generate_image_captions(image)

    leaf_signals = definition.collect_leaf_signals()
    leaf_signal_queries = self._evaluate_leaf_signals(image, leaf_signals)
    leaf_signal_query_string = self._stringify_queries(
        leaf_signal_queries, 'leaf-signal'
    )
    logger.debug(
        'leaf_signal_query_string\n%s', leaf_signal_query_string
    )

    evaluate_image_for_complex_concept_prompt = f"""
      <role>You are an expert image annotator.</role>
      <input>
        You will be provided with a single image, an image caption and a concept definition.
        Crowdworkers have also examined the image and answered a series of questions about the image.
        You should use the answers to help you evaluate the image for the concept, but keep in mind that they might be wrong.
      </input>

      <task>
      - Thoroughly examine the image.
      - Carefully consider all details of the concept
      - Determine if the image satisfies every aspect of the given concept.
      </task>

      <step1>
      Repeat the concept definition here.
      </step1>
      <step2>
      Explicitly write down the specific requirements this concept demands. Make sure your decomposition is equivalent to the original definition.
      There are two special cases that demand your attention:
      - If the concept definitions consists of an "and" relationship, you should start by checking each condition explicitly. You should only give a high rating if all conditions are satisfied.
      - If the concept definitions consists of an "or" relationship, you should start by checking each condition explicitly. You should give a high rating if any of the conditions are satisfied.
      <step2>
      <step3>
      Based on responses in the first three steps and crowdworker's responses,
      for each condition, you should determine whether the image satisfies the condition.
      </step3>
      <step4>
      You should then combine your responses for individual conditions to determine whether the image satisfies the overall concept or not.
      Your reasoning should be based solely on the definition, the image, the image caption, and crowdworker's responses.
      Do not include any information that is not provided (no hallucinations).
      </step4>
      <step5>Based on your reasoning in the first step, determine whether the image completely fulfills every aspect of the concept.
      You should rate how in-scope the image is on a 1-5 Likert Scale where
        - Rate 5 if the image fully aligns with the concept
        - Rate 4 if the image mostly aligns with the concept; the small problem is a result of small ambiguities in the definition or small visual complexities in the image
        - Rate 3 if there are no strong evidence that indicate that the image violates the concept, but the supporting evidence is also not strong enough to support a rating of 4 or 5.
        - Rate 2 if the image violates some parts of the concept but there are some elements that are relevant to the concept.
        - Rate 1 if the image does not align with the concept description at all.
      </step5>
      <step6>
      Based on your answer in step2 and step3, provide a one-sentence summary why you give this answer.
      </step6>

      Provide your answer in the following XML structure:
        <description>Repeat the concept definition here</description>
        <requirements>Explicitly write down the specific requirements this concept demands.</requirements>
        <condition-eval>Write down your evaluation for each condition at step3 here.</condition-eval>
        <evaluation>You should differentiate between concept definitions that requires the satisfaction of all conditions and those that only require the satisfaction of one of the conditions. Describe your evaluation reasonings in the step4 here</evaluation>
        <decision>Rate on a 1-5 Likert Scale where 5 means the image is fully in-scope and 1 means the image is fully out-of-scope.</decision>
        <summary>Provide your summary in the step6 here</summary>
      The output will be later wrapped in a <root> tag, so do not wrap the content above in any tag such as xml, or root in your output.

      <examples>
        <example>
          <criterion>Kitchen with a Baking Activity and a Child</criterion>
          <image_caption>The image shows a busy kitchen with a countertop covered in baking ingredients like flour, sugar, and eggs. A person is mixing ingredients in a bowl, and a young child is standing on a stool next to them, watching intently. The kitchen is well-lit and organized, and the child is actively engaged in the baking activity.</image_caption>
          <description>The concept of a kitchen with a baking activity and a child</description>
          <requirements>The image must satisfy two conditions: presence of a baking activity and presence of a child.</requirements>
          <condition-eval>
          In-scope images must satisfy both conditions:
          - Presence of a Baking Activity: The person is mixing ingredients in a bowl, indicating an ongoing baking process. This condition is satisfied.
          - Presence of a Child: A young child is visibly present and engaged in the activity. This condition is satisfied.
          </condition-eval>
          <evaluation>
          As the image clearly meets both conditions of the criterion, this image is a strong match for the concept of a kitchen scene with both a baking activity and a child.
          </evaluation>
          <decision>5</decision>
          <summary>The image fully aligns with the criterion by showing both a baking activity and a child in the kitchen.</summary>
        </example>
        <example>
          <criterion>Food</criterion>
          <image_caption>The image shows a hamburger made out of cigarettes. The top bun is made of a stack of cigarettes, the bottom bun is made of a stack of cigarettes, and the patty is made of a stack of red and green matchsticks. </image_caption>
          <description>Food</description>
          <requirements>The image only has one requirement: the image must show food.</requirements>
          <condition-eval>
          In-scope images must satisfy the only condition:
          - the image must show food: The image depicts a composition of cigarettes and matchsticks arranged to resemble a hamburger, but it does not represent actual food.
          The concept of "Food" refers to items that are edible and used for nourishment, which this image does not satisfy. This condition is not satisfied at all.
          </condition-eval>
          <evaluation>
          As the image does not satisfy the only condition of the concept at all, it is not in-scope for the concept of food.
          </evaluation>
          <decision>1</decision>
          <summary>The image is composed of non-edible items arranged to look like a hamburger, so it does not meet the food concept at all.</summary>`
        </example>
        <example>
          <criterion>Images that show two boys are rowing a boat</criterion>
          <image_caption>The image depicts two animated children, a boy and a girl, rowing a small yellow boat on a river. The boy, wearing a purple shirt and white shorts, sits at the front of the boat, while the girl, with red hair and wearing a green shirt, sits behind him.  </image_caption>
          <description>Images that show two boys are rowing a boat</description>
          <requirements>The image must satisfy two conditions: presence of two boys and rowing a boat.</requirements>
          <condition-eval>
          In-scope images must satisfy both conditions:
          - Presence of two boys: The image shows two children, a boy and a girl, but the girl is not a boy. This condition is not satisfied despite being relevant.
          - Rowing a boat: The two children are rowing a boat. This condition is fully satisfied.
          </condition-eval>
          <evaluation>
            The image clearly meets one of the conditions but not the other:
            Therefore, the image violates part of the concept despite stilling being relevant somehow to the concept. Therefore, it could be rated as 2.
          </evaluation>
          <decision>2</decision>
          <summary>The image depicts a boy and a girl rowing a boat, so it does not fully align with the criterion of showing two boys rowing a boat.</summary>
        </example>
        <example>
          <criterion>Gardening Tools</criterion>
          <image_caption>The image shows a backyard garden with various plants growing. There are several gardening tools visible, including a trowel, pruning shears, and a watering can. A person is seen using the watering can, and a garden fork is lying on the ground. The image is bright and shows the tools in use, but the focus is more on the plants and overall garden layout.</image_caption>
          <description>Gardening Tools</description>
          <requirements>The image has only one condition: presence of gardening tools</requirements>
          <condition-eval>
          In-scope images must satisfy the only condition:
          - Presence of gardening tools: The image shows several gardening tools, including a trowel, pruning shears, and a watering can.
            While the primary focus of the image is on the garden and the plants, the tools are well-represented and their presence aligns with the concept of gardening tools.
            The slight emphasis on the garden layout rather than exclusively on the tools introduces a minor issue, but it does not greatly impact the overall alignment.
            Therefore, this condition is mostly satisfied.
          </condition-eval>
          <evaluation>
            The image mostly satisfies the only condition of this concept, despite some visual complexities available that prevents a full alignment.
          </evaluation>
          <decision>4</decision>
          <summary>The image mostly aligns with the concept of gardening tools, as it shows several tools in use, though the focus is somewhat broader on the garden layout.</summary>
        </example>
        <example>
          <criterion>Outdoor Sports</criterion>
          <image_caption>The image shows a group of people in a park. Some are playing frisbee, while others are sitting on the grass and having a picnic. A few people in the background are jogging.</image_caption>
          <description>Outdoor Sports</description>
          <requirements>The image has only one condition: presence of outdoor sports activities</requirements>
          <condition-eval>
          In-scope images must satisfy the only condition:
          - Presence of outdoor sports activities: The image shows several people engaged in various outdoor sports activities, including playing frisbee, jogging, and having a picnic.
            However, the presence of people simply sitting on the grass and having a picnic makes it unclear if the entire scene is focused on outdoor sports. There are strong elements of both outdoor sports and leisure activities,
            There are strong elements of both outdoor sports and leisure activities, so the image is not fully in-scope or out-of-scope for the concept.
          </condition-eval>
          <evaluation>
            The image shows mixed evidence regarding whether its only condition is satisfied or not, leading to a neutral rating.
          </evaluation>
          <decision>3</decision>
          <summary>The image features both outdoor sports activities and leisure activities, making it unclear whether the image should be considered fully in-scope or out-of-scope for the criterion.</summary>
        </example>
      </examples>

      <Input>
      Here is the criterion for you to work on:
        <criterion>{str(definition)}</criterion>

      Here is the high level image caption provided by an image annotator.
      As human raters might make mistakes, make sure you reference the provided image to verify them.
      Even though the caption provides key details about the image, the importance of each detail may vary visually.
      Always assess how clearly the image conveys each piece of information to an average viewer.
      For information that are emphasized, you should pay more attention to them; for information that is barely visible, you should pay less attention to them.
        <image_caption>{image.image_caption}</image_caption>

      Here is how crowdworkers answered the questions about the image.
        <query>
        {leaf_signal_query_string}
        </query>
      </Input>
    """
    response = self.model_client.gemini_image_prompt(
        evaluate_image_for_complex_concept_prompt,
        [image],
        cheap=self.cheap_model,
    )
    logger.debug(
        'response for a complex concept:\n%s',
        response,
    )
    evaluation = self.model_client.parse_xml(
        response, 'root', ['decision', 'summary']
    )
    if len(evaluation) == 1 and isinstance(evaluation[0], dict):
      return evaluation[0]
    else:
      return None

  def classify_images_in_parallel(
      self,
      images,
      definition,
      batch_size = 10,
      **kwargs,
  ):
    """The endpoint for classifying a batch of images.

    Args:
      images: A list of images to classify.
      definition: The concept definition.
      batch_size: The batch size for processing.
      **kwargs: Additional keyword arguments.

    Returns:
      A list of evaluation dictionaries or None.
    """
    logger.debug('classifying images in parallel: %d images', len(images))
    predictions = utils_py.run_in_batches(
        images,
        self.classify_image,
        batch_size=batch_size,
        definition=definition,
        **kwargs,
    )
    return predictions

  def classify_image(
      self,
      image,
      definition,
      majority_voting = 1,
      complex_concept = False,
  ):
    """The endpoint for classifying an image.

    Args:
      image: The MyImage to classify.
      definition: The Definition to classify against.
      majority_voting: The number of times to vote for majority decision.
      complex_concept: True if treating the concept as a complex concept.

    Returns:
      A dictionary containing the decision and summary.
    """

    previous_rating = image.retrieve_rating(definition)
    if previous_rating is not None:
      return previous_rating

    if not image.image_caption:
      self.generate_image_captions(image)

    majority_results = []
    for _ in range(majority_voting):
      # We allow retrying in case of failure.
      retries = 0
      result = None
      while retries < 3:
        try:
          if not complex_concept:
            result = self._evaluate_image_for_a_simple_concept(
                image, definition
            )
          else:
            result = self._evaluate_image_for_a_complex_concept(
                image, definition
            )
          break
        except Exception as e:
          logger.error('Failed to classify the image: %s', e)
          retries = retries + 1
          continue
      if result is None:
        continue
      result['decision'] = int(result['decision'])
      majority_results.append(result)

    if majority_results:
      final_result = None
      scores = [int(result['decision']) >= 3 for result in majority_results]
      if len(set(scores)) == 1:
        # We have reached a consensus, simply return the first one.
        final_result: dict[str, Any] = majority_results[0]
        final_result['consistency'] = 1.0
      else:
        # We have not reached a consensus, we need to aggregate the results.
        majority_decision = Counter(scores).most_common(1)[0][0]
        # Find the corresponding summary.
        for result in majority_results:
          if (int(result['decision']) >= 3) == majority_decision:
            final_result = result
            # How much percentage of the raters agree on this decision.
            final_result['consistency'] = (
                len(
                    [
                        now_score
                        for now_score in scores
                        if now_score == majority_decision
                    ]
                )
                / len(scores)
            )
            break

      if final_result is not None:
        image.cache_rating(definition, final_result)
        return final_result
    return {
        'decision': '1',
        'summary': 'Failed to classify the image.',
        'consistency': None,
    }
