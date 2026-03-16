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

"""Reflection on concept definitions."""

import logging
from typing import Optional

import numpy as np
from sklearn import cluster

from agile_deliberation_lib import definitions as definitions_py
from agile_deliberation_lib import image as image_py
from agile_deliberation_lib import llm as llm_py
from agile_deliberation_lib import retrieval as retrieval_py
from agile_deliberation_lib import search_images as search_images_py
from agile_deliberation_lib import utils as utils_py


DBSCAN = cluster.DBSCAN
MyImage = image_py.MyImage
Definition = definitions_py.Definition
logger = logging.getLogger(__name__)


class Reflection:
  """Reflect on concept definitions or classification results."""

  def __init__(
      self,
      retrieval_client,
      model_client,
      image_curator,
  ):
    """Initializes the Reflection object.

    Args:
      retrieval_client: The retrieval client.
      model_client: The LLM model client.
      image_curator: The image curator.
    """
    self.retrieval_client = retrieval_client
    self.image_curator = image_curator
    self.model_client = model_client
    self.cached_concepts = {}

  def brainstorm_golden_category(
      self,
      definition,
      context = '',
  ):
    """Brainstorm golden category for a categorical concept.

    Args:
      definition: The concept definition.
      context: The context of the concept.

    Returns:
      The generated golden category Definition, or None.
    """
    cached_golden_categories = None
    if definition.parent and definition.parent.concept in self.cached_concepts:
      now_cached_concept = self.cached_concepts[definition.parent.concept]
      for cached_signal in now_cached_concept:
        if cached_signal['concept'] == definition.concept:
          cached_golden_categories = cached_signal.get(
              'golden_categories', []
          )
    elif definition.concept in self.cached_concepts:
      now_cached_concept = self.cached_concepts[definition.concept][0]
      cached_golden_categories = now_cached_concept.get('golden_categories', [])

    if cached_golden_categories:
      sufficient_signal = cached_golden_categories.pop(0)
      sufficient_signal = Definition(
          sufficient_signal['concept'],
          sufficient_signal['description']
      )
      definition.add_previous_signals(sufficient_signal)
      return sufficient_signal

    previous_signal_str = ''
    for previous_signal in definition.previous_signals:
      previous_signal_str += previous_signal.print_definition(level=0)
    if definition.parent and definition.signal_type == 'necessary':
      context = definition.parent.print_definition(level=1)
    brainstorm_golden_category_prompt = f"""
        <role>You are an expert linguist recognized internationally.</role>
        <input>You will be provided an overall concept definition and a focus concept.</input>
        <task>
        The concept owner wants to catch all images that are in-scope for the focus concept.
        To make sure that his decisions about image classifications are consistent, he wants to explicitly clarify the decision-making boundaries for this focus concept.
        However, as he starts with a few images, it is possible that he either starts with a narrower concept or a broader concept than he actually wants.
        Your task is to infer what are the golden subconcepts that the concept owner wants to include within the scope of this focus concept.
        </task>

        <step1>
        Reason what is the primary concept that the concept owner wants to explicitly define.
        There are a few requirements.
        1) This description might mention several concepts but you should only focus on the primary concept.
        2) If the context indicates that the focus concept is part of the necessary signals of a larger concept, then the primary concepts of these necessary signals should focus on different subconcepts of this larger concept.
        In other words, your primary concept should have a different focus than those of the other necessary signals.
        3) The primary concept should be more categorical (concepts where you could think about specific instances) rather than descriptive (where you could only describe different aspects of the concept).
        Examples of categorical concepts are "fruit", 'electronic devices', 'physical affection', 'outdoor activities',
        whereas examples of descriptive concepts are "sleeping person", 'romantic relationship', 'sexual suggestive content'.
        </step1>

        <step2>
        List out categories of subconcepts that have been explored before.
        Here the categories mean a way to categorize specific subconcepts, for instance, 'earphones' as a subconcept can belong to the category 'electronic devices' and 'accessories' at the same time.
        This includes the following two cases:
        1) categories that are already included in the concept definition as positive or negative signals.
        2) categories that have beed explored in the previous rounds of brainstorming.
        </step2>

        <step3>
        Based on your answer in step2 and step3, reason and propose a category of subconcepts that you think is the most coherent and widely recognized.
        While you can include previous explored subconcepts, your category should not significantly overlap with previously explored categories that you listed in step3
        This is because a subconcept can belong to multiple categories at the same time.
        In particular, we have the following requirements:
        <requirements>
        1. You should ensure that this category itself is a well-defined and well-known concept so that average people can easily tell whether an image satisfies this category or not.
        2. Your category should not be too narrow that it only covers one or two instances.
        3. In cases where there are many potential categories of subconcepts, you should prioritize the one that most people would agree to be in-scope for the concept.
        4. You do not aim for proposing a category that includes the most subconcepts; Instead, you should prioritize proposing a category that is coherent, and well-defined.
        </requirements>

        <examples>
        - For the primary concept "fruits", "fruits with red internal flesh" is not a well-known concept, whereas "citrus fruits" is.
        - For the primary concept "flowers", "Flowers with five petals" is not a well-known concept and also difficult to recognize in an image; "Rose varieties" is instantly recognizable.
        - For the primary concept "birds", "birds of prey" is a well-known concept, visually distinct, and specific without being too narrow.
        - For the primary concept "buildings," while "buildings with green roofs" may not be a widely recognized concept,  "religious buildings" is a well-known concept that is often visually distinctive.
        - For the primary concept "health care products," while "Vitamin C supplements" might seem like a well-defined category, it is too narrow.
          "Dietary supplements" is a more suitable category encompassing various products like vitamins, minerals, herbs, and fish oil, providing a broader yet well-defined and easily recognizable concept.
        </examples>
        </step3>

        <step4>
        For the category in step4, you should write a one-sentence description and a shortname.
        Your description should be concise, self explanatory, and easy for an average person to determine if an image satisfies this subconcept or not.

        <description-requirements>
        You should be careful about your language, in particular, there are several requirements.
        1) The recommended format for the description would be "Images show [a general term for the subconcept], such as [at most three specific examples from step3]".
        These examples shoulld be representative of the subconcept and should be as specific as possible so that human image annotators can easily know whether an image includes this example or not.
        These examples should form a coherent category, and should be specific.
        Your short name for the description should be exactly the term that covers this set of examples.

        2) Avoid concept descriptions with too many specific and unnecessary details.
            e.g., for the concept 'beverages', your subconcept description should just be 'Images showing various types of tea drinks such as green tea, black tea, and herbal tea'
            rather than 'Images that show people drinking various types of tea drinks with different colors and flavors such as green tea, black tea, and herbal tea'
          Similarly, avoid adding too many unnecessary details to the examples you provide.
          For instance, "eagles" is a good example of the category "birds of prey", but "bald eagles" or "eagles in the sky" are not.

        3) Be careful about your word choices of verbs, nouns, or adjectives, which might carry unexpected nuances.
            e.g., be careful about using 'depict' or 'mention', or 'show' as the previous two verbs introduce the slight emphasis on visual or textual aspects.
            e.g., be careful about using adjectives like 'clearly' or 'explicitly' as they might suggest a degree of visibility to the original concept.
        </description-requirements>
        </step4>

        Write your output strictly in this valid xml format:
          <repeat-focus-concept>Repeat the focus concept here.</repeat-focus-concept>
          <primary-concept>List out the primary concept this categorical concept focuses on here.</primary-concept>
          <explored-subconcepts>
          List out the questions that have been asked for this concept before if any in a bullet point list at step2.
          </explored-subconcepts>
          <category-reasoning>
          List out your reasonings at step3 here.
          </category-reasoning>
          <subconcept>
            <description></description>
            <name></name>
          </subconcept>

        <examples>
        <example>
          <conceptDefinition>
            <name>Electronic Devices</name>
            <description>Images that show electronic devices.</description>
            <positive-signals>
              <description>Images show consumer electronics, such as smartphones, laptops, and tablets.</description>
              <name>Consumer Electronics</name>
            </positive-signals>
          </conceptDefinition>
          <primary-concept>Electronic devices</primary-concept>
          <explored-subconcepts>
            - Consumer Electronics: Images show consumer electronics, such as smartphones, laptops, and tablets.
          </explored-subconcepts>
          <category-reasoning>
            The user has provided a broad concept, "Electronic Devices." They have already introduced "Consumer Electronics" as a subconcept. To avoid overlap, we should focus on another distinct category within "Electronic Devices." A suitable category could be "Home Appliances," as they are also electronic devices but differ significantly from personal consumer electronics.
          </category-reasoning>
          <subconcept>
            <description>Images show home appliances, such as refrigerators, washing machines, and ovens.</description>
            <name>Home Appliances</name>
          </subconcept>
        </example>
        <example>
          <conceptDefinition>
            <name>Fruits</name>
            <description>Images that show fruits on the table.</description>
            <negative-signals>
              <description>Images show fruits commonly used in savory preparations, such as cucumbers, tomatoes, and avocados.</description>
              <name>Culinary Fruits</name>
            </negative-signals>
          </conceptDefinition>
          <primary-concept>Fruits</primary-concept>
          <explored-subconcepts>
            - Culinary Fruits: Images show fruits commonly used in savory preparations, such as cucumbers, tomatoes, and avocados.
          </explored-subconcepts>
          <category-reasoning>
            We're aiming for a visually distinct subcategory of "Fruits" that is also relatively well-known. Citrus fruits, with their vibrant colors and often bumpy skin, provide a good balance of visual distinctiveness and common knowledge.
          </category-reasoning>
          <subconcept>
            <description>Images show various types of citrus fruits, such as oranges, lemons, and grapefruits.</description>
            <name>Citrus Fruits</name>
          </subconcept>

        </examples>

        <conceptDefinition>
          {str(definition)}
        </conceptDefinition>
        <previous-signals>
          {previous_signal_str}
        </previous-signals>s
        <context>{context}</context>
      """

    xml_response = self.model_client.gemini_prompt(
        brainstorm_golden_category_prompt, cheap=False
    )
    logger.debug('brainstorm golden signals: %s', xml_response)
    sufficient_signals = self.model_client.parse_xml(
        xml_response, 'subconcept', ['name', 'description']
    )
    sufficient_signal = sufficient_signals[0] if sufficient_signals else None
    sufficient_signal = Definition(
        sufficient_signal['name'],
        sufficient_signal['description']
    )
    definition.add_previous_signals(sufficient_signal)
    return sufficient_signal

  def brainstorm_borderline_category(
      self, definition, context = '',
  ):
    """Brainstorm borderline categories for a categorical concept.

    Args:
      definition: The concept definition.
      context: The context of the concept.

    Returns:
      The generated borderline category Definition, or None.
    """
    cached_borderline_categories = None
    if definition.parent and definition.parent.concept in self.cached_concepts:
      now_cached_concept = self.cached_concepts[definition.parent.concept]
      for cached_signal in now_cached_concept:
        if cached_signal['concept'] == definition.concept:
          cached_borderline_categories = cached_signal.get(
              'borderline_categories', []
          )
    elif definition.concept in self.cached_concepts:
      now_cached_concept = self.cached_concepts[definition.concept][0]
      cached_borderline_categories = now_cached_concept.get('borderline_categories', [])

    if cached_borderline_categories:
      signal_dict = cached_borderline_categories.pop(0)
      sufficient_signal = Definition(
          signal_dict['concept'],
          signal_dict['description']
      )
      definition.add_previous_signals(sufficient_signal)
      return sufficient_signal

    previous_signal_str = ''
    for index in range(len(definition.previous_signals)):
      # We only include previously explored borderline signals here.
      # As we alternate between golden and borderline categories every 3 steps,
      # indices 3N + 2 are the borderline categories.
      if index % 3 == 2:
        previous_signal = definition.previous_signals[index]
        previous_signal_str += previous_signal.print_definition(level=0)
    if definition.parent and definition.signal_type == 'necessary':
      context = definition.parent.print_definition(level=1)
    brainstorm_borderline_category_prompt = f"""
        <role>You are an expert linguist recognized internationally.</role>
        <input>You will be provided a concept definition and an optional context for this concept.</input>
        <task>
        The concept owner wants to catch all images that are in-scope for this concept.
        To make sure that his decisions about image classifications are consistent, he wants to explicitly clarify the decision-making boundaries for this concept.
        However, as he starts with a few images, it is possible that he either starts with a narrower concept.
        Your task is to suggest a new borderline subconcept that the concept owner might want to include.
        </task>

        <step1>
        Reason what is the primary concept that the concept owner wants to explicitly define.
        There are a few requirements.
        1) This description might mention several concepts but you should only focus on the primary concept.
        2) If the context indicates that the focus concept is part of the necessary signals of a larger concept, then the primary concepts of these necessary signals should focus on different subconcepts of this larger concept.
        In other words, your primary concept should have a different focus than those of the other necessary signals.
        3) The primary concept should be more categorical (concepts where you could think about specific instances) rather than descriptive (where you could only describe different aspects of the concept).
        Examples of categorical concepts are "fruit", 'electronic devices', 'physical affection', 'outdoor activities',
        whereas examples of descriptive concepts are "sleeping person", 'romantic relationship', 'sexual suggestive content'.
        </step1>

        <step2>
        If the user starts with a narrower concept, reason what will be the broader concept that the user might want to define.
        You should examine the given context and summarize the broader concept the user might intend to say in replacement of the primary concept.
        This broader concept should fit nicely with other parts of the context.

        <example>For the concept 'health supplements' within the context of "images that show health supplements to promote wellness", the broader concept might be 'wellness products'</example>
        <example>For the concept 'fake websites' within the context of "images that show fake websites as part of online fraud", the broader concept might be 'online fraudulent schemes'</example>
        <example>For the concept 'electronic devices' within the context of "images that show electronic devices in a library", the broader concept might be 'things we can find in a library'</example>
        </step2>

        <step3>
        List out categories of edgecase categories that have been explored before from the previous signals input.
        </step3>

        <step4>
        Based on your answer in step1 and step2, what other subconcepts might be in-scope for this broader concept in step2 but obviously not part of the primary concept in step1.
        In other words, you should not focus on detailing specific edgecase categories of this primary concept. Instead you should think about what other subconcepts (replacing the primary concept) frequently appear in the same context.

        There are a few requirements.
        <requirements>
        1) You should NOT focus on detailing specific edgecase categories of this primary concept.
        2) Your category should NOT significantly overlap with the subconcepts that have been explored at step3.
        3) Your category should not refer to examples that significantly overlap with the examples that have been explored before in step2.
        4) We will later define the other necessary signals for this concept, so your category should NOT try to define other necessary signals.
        </requirements>

        <example>For the concept 'health supplements' within the context of "images that show health supplements to promote wellness", 'fresh fruits', 'yoga mats', or 'spa treatments' might also be interesting because they can also visually symbolize natural and holistic approaches to health and well-being.</example>
        <example>For the concept 'fake websites' within the context of "images that show fake websites as part of online fraud", "counterfeit product pages", "fake social media accounts", or "fraudulent payment forms" might also be interesting because they can also be depicted as part of online fraud schemes.</example>
        <example>For the concept 'electronic devices' within the context of "images that show electronic devices in a library", 'board games', 'books', or 'musical instruments' might also be interesting because they might also appear in a library despite not electronic.</example>
        </step4>

        <step4>
        Based on your reasoning in step4, write a one-sentence description and a shortname for your borderline subconcepts.

        <description-requirements>
        You should be careful about your language, in particular, there are several requirements.
        1) The recommended format for the description would be "Images show [a general term for the subconcept], such as [at most three specific examples from step3]".
        These examples shoulld be representative of the subconcept and should be as specific as possible so that human image annotators can easily know whether an image includes this example or not.
        These examples should form a coherent category, and should be specific.
        Your short name for the description should be exactly the term that covers this set of examples.

        2) Avoid concept descriptions with too many specific and unnecessary details.
            e.g., for the concept 'beverages', your subconcept description should just be 'Images showing various types of tea drinks such as green tea, black tea, and herbal tea'
            rather than 'Images that show people drinking various types of tea drinks with different colors and flavors such as green tea, black tea, and herbal tea'
          Similarly, avoid adding too many unnecessary details to the examples you provide.
          For instance, "eagles" is a good example of the category "birds of prey", but "bald eagles" or "eagles in the sky" are not.

        3) Be careful about your word choices of verbs, nouns, or adjectives, which might carry unexpected nuances.
            e.g., be careful about using 'depict' or 'mention', or 'show' as the previous two verbs introduce the slight emphasis on visual or textual aspects.
            e.g., be careful about using adjectives like 'clearly' or 'explicitly' as they might suggest a degree of visibility to the original concept.
        </description-requirements>
        </step4>

        Write your output strictly in this valid xml format:
          <primary-concept>List out the primary concept this categorical concept focuses on here.</primary-concept>
          <broader-concept>List out the broader concept that the user might want to define here.</broader-concept>
          <previous-signals>List out the previous signals explored before here at step3 in a bullet point list.</previous-signals>
          <reasoning>
          List out your reasonings at step4 here.
          </reasoning>
          <subconcept>
            <description></description>
            <name></name>
          </subconcept>

        <conceptDefinition>
          {str(definition)}
        </conceptDefinition>
        <previous-signals>
        {previous_signal_str}
        </previous-signals>
        <context>{context}</context>
      """

    xml_response = self.model_client.gemini_prompt(
        brainstorm_borderline_category_prompt, cheap=False
    )
    logger.debug('brainstorm sufficient signals response: %s', xml_response)
    sufficient_signals = self.model_client.parse_xml(
        xml_response, 'subconcept', ['name', 'description']
    )

    final_signal = None
    if len(sufficient_signals) == 1:
      signal_dict = sufficient_signals[0]
      final_signal = Definition(
          signal_dict['name'],
          signal_dict['description']
      )
      definition.add_previous_signals(final_signal)

    return final_signal

  def determine_whether_composite(
      self, definition,
      decomposed_signals = None,
  ):
    """Determine whether a concept is composite and thus needs to be decomposed.

    Args:
      definition: The concept definition.
      decomposed_signals: Optional list of previously decomposed signals.

    Returns:
      True if the concept is composite, False otherwise.
    """
    if not decomposed_signals:
      decomposed_signals = self.decompose_concept(definition)
    decomposed_signal_str = ''
    for signal in decomposed_signals:
      decomposed_signal_str += f'{str(signal)}\n'
    determine_concept_type_prompt = f"""
      <role>You are an expert linguist</role>
      <input>
      You will be provided a visual concept name and its description.
      We also provide a preliminary decomposition of this concept into a list of subconcepts.
      </input>
      <task>
      The concept owner wants to catch all images that are in-scope for the focus concept.
      To make sure that his decisions about image classifications are consistent, he wants to explicitly clarify the decision-making boundaries for this focus concept.
      Your task is to evaluate whether this decomposition actually makes it easier for human raters to determine whether an image is in-scope or out-of-scope according to the definition.
      </task>

      <guideline>
        <composite-concepts>
        Each subconcept also carries their own focus and is not redundant; each subconcept should be visually grounded and specific.
        Examples of composite concepts are 'children playing basketball', 'people in a party', 'a woman in a bra', 'gourmet food in a high-end restaurant'.
        </composite-concepts>
        <non-composite-concepts>
        The concept refers to a diverse categories of images.
        While each category of images is in-scope of this concept, we cannot decisively determine a subconcept that is shared by all categories.
        Examples of non-composite concepts often relate to intuitive concepts, such as 'delicious food', 'cute animals', 'beautiful landscapes', or 'adult content'.
        Even though these examples seem to consist of several subconcepts, they are so closely connected that it is difficult to describe one subconcept without the other.
        Non-composite concepts can also relate to concepts that are too diverse, such as 'online fraud', 'hateful speech'.
        </non-composite-concepts>
      </guideline>

      <step1>
      Examine the prelimary decomposition of the concept, reason whether this process actually makes it easier for human raters to determine whether an image is in-scope or out-of-scope according to the definition.
      In particular, you should examine each subconcept from the following perspectives.
      1) Whether this subconcept is too vague for image rating and is very difficult to be further defined alone.
      Examples of such subconcepts are 'images that are beautiful', 'images that are cute', 'images that evoke a sense of disgust'.
      2) Whether this subconcept is actually not a necessary condition for the concept.
      For instance, 'images show spring reasons' is actually not necessary for 'images show spring flowers' because 'spring flowers' is a well-defined concept.
      3) Whether this subconcept is redundant given other subconcepts.
      For instance, "images that show food" is redundant given "images where the food is delicious" because "delicious" is less meaningful without "food" and thus determining delicious often assumes the existence of food.
      Similarly, "images show a restaurant" is redundant given "images show high-end restaurant" because "high-end" is less meaningful without "restaurant".
      In contrast, "Images that show woman" is not redundant given "images that show woman in a bra" because "bra" and "woman" is more distinct from each other.
      </step1>

      <step2>
      If we do not decompose the concept, reason how would you further define the concept by brainstorming many different categories of in-scope images.
      </step2>

      <step3>
      Compare the solution in step 1 and step 2, determine which one might be easier for human raters to determine whether an image is in-scope or out-of-scope according to the definition.
      If you think the decomposition is better, then answer 'Yes'.
      If you think the non-decomposition is better, then answer 'No'.
      Note that the preliminary decomposition might not be perfect, so if you think the direction of the decomposition is better than a descriptive approach, you should still answer 'Yes'.
      The concept owners can still improve this decomposition later.
      </step3>

      Write your output strictly in this valid xml format:
        <decomposition>Write down your reasoning in step 1 here</decomposition>
        <descriptive>Write down your reasoning in step 2 here</descriptive>
        <reasoning>Write down your reasoning in step 3 here</reasoning>
        <decision>'Yes' or 'No' from step 3</decision>

      <conceptDefinition>
      {str(definition)}
      </conceptDefinition>
      <decomposed-signals>
      {decomposed_signal_str}
      </decomposed-signals>
    """
    xml_response = self.model_client.gemini_prompt(
        determine_concept_type_prompt, cheap=False
    )
    logger.debug('Determine concept type: %s', xml_response)
    decision = self.model_client.parse_xml(
        xml_response, 'root', ['decision']
    )[0]
    return decision['decision'] == 'Yes'

  def decompose_concept(
      self, definition, want_decomposition = False,
  ):
    """Decompose the concept into necessary conditions.

    Args:
      definition: The concept definition.
      want_decomposition: Whether decomposition is explicitly requested.

    Returns:
      A list of decomposed necessary conditions (Definitions), or None.
    """
    if definition.concept in self.cached_concepts:
      decomposed_signals = self.cached_concepts[definition.concept]
      decomposed_signals = [
          Definition(
              concept=signal['concept'],
              description=signal['description'],
              signal_type='necessary',
              parent=definition,
          )
          for signal in decomposed_signals
      ]
      return decomposed_signals

    decompose_concept_prompt = f"""<role>You are an expert linguist recognized internationally.</role>
      <input>You will receive a visual concept name and a description.</input>
      <task>There are human image annotators who need to determine if an image is in scope or out of scope of this visual concept.
      Your job is to break down this visual concept into at most two necessary conditions: the conjunction of these necessary conditions will be logically equivalent to the given visual concept.
      In other words, images that satisfy all these conditions will be exactly images that satisfy the visual concept.
      We call this process as decomposition.
      The final goal is to make it easier and more accurate for human raters to determine if an image is in-scope or out-of-scope for the given visual concept.
      </task>


      <step1>
      Rewrite the description by removing words that are redundant, too specific, or actually not necessary for this overall concept.
      Other than removing words, you should keep most parts of the description intact.
      This is because users might provide a draft description in the beginning without knowing what are in-scope images look like in the wild.
      As a result, they might introduce too many details that are not actually not necessary for the concept.
      </step1>
      <step2>
      Based on your response in step1, reason how would you decompose the visual concept into several conditions.
      Each necessary condition should be concise, self explanatory, and easily understood by human annotators.
      Each necessary condition should only focus on one concept, and you should not generate a still complex necessary condition.
      For each condition, you should provide a description of the condition, and a concept name that summarizes the aspect this condition focuses on.
      Explicitly write down a description and a name for your decomposed conditions here.
      Remember not all concepts can be further decomposed.

      </step2>
      <step3>
      Examine the following aspect for your decomposition.
      (1) Each condition must not significantly overlap with other conditions in their focused concepts.
      Examples
      - the condition that "a family is gathering together" and "a family is having a meal together" are too similar to each other despite they have different focus.
      - the condition that "a vase is broken into pieces" and "someone used some tools to break a vase into pieces" are too similar to each other despite slightly different wording and focus.."

      (2) Each condition carries meaningful information, meaning it should not hold true for every image.
        For instance, the condition “the image describes an object” is too broad and would be true for all images.


      </step3>
      <step4>
      Write down your decomposed conditions.
      In cases when you find it hard to decompose the concept, you can just write down a improved concept description. of the original concept here.

      You should follow these guidance in your description.
      (1) Avoid using verbs, adjectives, or adverbs that carry nuances unless they are an important part of the concept.
      Examples
      - if the original concept only uses the phrase "show a group of children", then avoid adding using the phrase "depict a group of children" as "depicts" introduces the slight emphasis on visual aspects and ignore the possibility of textual information in the image.
      - if the original concept only mentions the phrase "electronic devices", then avoiding using the phrase "such as a phone or a laptop" as the new listed examples suggest a focus on these specific examples.
      - if the original concept only uses the phrase "show a beautiful park", then avoid using the phrase "clearly show a beautiful park" because "clearly" implies a degree of visibility to the original concept.

      (2) Avoid further defining complex, abstract, or subjective concepts in the original concept definitions.
      We only want to break down a composite concept into more unit concepts, and we will define them more clearly in the next round.
      Examples:
      - if the original concept is "show a beautiful painting", you should just decompose it into "show a painting" and "the painting is beautiful"; you do not need to explicate what make a painting beautiful.
      - if the original concept is "show people are gathered happily", you should just decompose it into "show people" and "people are gathered happily"; you do not need to explain visual elements of being happy.

      (3) You must not include information beyond the provided information, as new information would effectively change the intended meaning of the concept.
      Examples
      - if the original concept "woman in a bra" does not mention the context "at the beach", then do not add this context in your necessary conditions.

      </step4>

      Provide your answer in the following XML structure:
        <new-description>Your refined description</new-description>
        <reasoning>Add your reasoning here at step2</reasoning>
        <examination>Add your examination here at step3</examination>
        <conditions>
          <condition>
            <description>Add a condition here</description>
            <name>a short name that summarizes the description</name>
          </condition>
          <condition>
            <description></description>
            <name></name>
          </condition>
          <!-- Add more necessary conditions here if needed. -->
        </conditions>

      <examples>
      <example>
      <visualConceptName>A woman in her nice underwear is standing there and talking with a man besides</visualConceptName>
      <new-description>A woman in her underwear is talking with a man</new-description>
      <reasoning>
        This concept involves three main components: the presence of a woman and a man, the woman wearing underwear, and the interaction of talking between them.
        Therefore, images that fit this concept must first show both a woman and a man.
        Next, the woman should be wearing underwear, and finally, the interaction between the woman and the man should be visible as talking.
        Images should first show a woman and a man. The woman is then required to wear an underwear and then talking with the man.
      </reasoning>
      <examination>
        1) Each condition is simpler to assess individually compared to the overall concept.
          Even though the final condition 'the woman is talking with the man' looks similar to the provided concept,
          it emphasizes the 'talking' aspect of the visual concept as other conditions already cover the 'woman and man' and 'underwear' aspects.
          They break down the complex concept into clear, discrete elements.
        2) Each condition directly relates to information provided in the concept description: the presence of both individuals, the specific clothing of the woman, and the nature of their interaction.
        3) Each condition targets a distinct aspect of the visual concept: identifying the individuals, the specific clothing, and the type of interaction, ensuring different elements are evaluated separately.
        4) The conditions are meaningful and specific: while "The image contains a woman and a man" is not universally true for all images,
          and "The woman in the image is wearing underwear" narrows down the scope, "The woman and the man in the image are talking" ensures the interaction is as described.
      </examination>
      <conditions>
        <condition>
          <description>The image contains a woman and a man.</description>
          <name>Woman and Man</name>
        </condition>
        <condition>
          <description> The woman in the image is wearing an underwear.</description>
          <name>Underwear</name>
        </condition>
        <condition>
          <description>The woman and the man in the image are talking.</description>
          <name>Talking</name>
        </condition>
      </conditions>
      </example>
      <example>
      <visualConceptName>Beautiful images for a viewer/visualConceptName>
      <new-description>Images that are beautiful</new-description>
      <decomposition>The only way to decompose would be 'there is an image' and 'the image is beautiful'</decomposition>
      <examination>
        'There is an image' conveys little information as human image annotators will find that this condition just holds true for every image.
        On the other hand, 'the image is beautiful' here would be of the similar complexity to the provided concept.
      </examination>
      <conditions>
        <condition>
          <description>The image is beautiful</description>
          <name>Beautiful images</name>
        </condition>
      </conditions>
      </example>
      </examples>
      <visualConceptName>{definition.concept}</visualConceptName>
      <visualConceptDescription>{definition.description}</visualConceptDescription>"""
    xml_response = self.model_client.gemini_prompt(
        decompose_concept_prompt, cheap=False,
    )
    logger.debug('decompose concept response: %s', xml_response)

    decomposed_signals = self.model_client.parse_xml(
        xml_response, 'conditions/condition', ['name', 'description']
    )
    decomposed_signals = [
        Definition(
            concept=signal['name'],
            description=signal['description'],
            signal_type='necessary',
            parent=definition,
        )
        for signal in decomposed_signals
    ]
    if not want_decomposition:
      whether_needed = self.determine_whether_composite(
          definition, decomposed_signals
      )
      if not whether_needed:
        return [definition]
    return decomposed_signals

  def summarize_definition(self, definition):
    """Summarize the definition.

    Args:
      definition: The concept definition.

    Returns:
      A summarized string of the definition.
    """
    summarize_definition_prompt = f"""
      <role>You are an expert linguist.</role>
      <input>
      You are given a visual concept definition.
      </input>
      <task>
      Your task is to summarize the concept definition in a concise and clear way.
      You should not miss any important information in the definition, nor include information that is not in the definition.
      </task>
      <define-a-concept>
        Before answering the question, it is a must for you to understand how we define a concept in a structured and iterative way.
        There are three types of signals in a concept definition:
        1) Necessary signals: a concept could be defined by a list of necessary signals.
          As long as an image satisfies all necessary signals, it is in-scope of the concept.

        2) Positive or negative signals: a concept could be defined by a list of positive or negative signals.
          If an image strongly satisfies at least one negative signal, it is out-of-scope of the concept, regardless of how many positive signals it satisfies.
          If an image does not satisfy any negative signal and satisfies at least one positive signal, it is in-scope of the concept.

        A concept could be defined iteratively, and therefore each signal of a concept can also be viewed as a concept.
        The most common case is that a concept is first defined by a list of necessary signals, and then each necessary signal is further defined by a list of positive or negative signals.
        In this tree structure, the overall concept is the root concept, and its signals is the first-level concepts.
        We call signals that are at the bottom level of the definition as "leaf concepts", i.e., they are not further defined by other signals.
      </define-a-concept>

      Provide your answer in the following XML structure:
        <summary>Your summary</summary>

      Here is the input concept definition:
        <visualConceptDescription>{definition.print_definition(prettify=True)}</visualConceptDescription>
    """
    xml_response = self.model_client.gemini_prompt(
        summarize_definition_prompt, cheap=False
    )
    logger.debug('decompose concept response: %s', xml_response)

    summary = self.model_client.parse_xml(xml_response, 'summary', [])
    return str(summary[0])

  def whether_borderline_images(
      self,
      image,
      definition,
  ):
    """Whether the image is borderline for the definition.

    Args:
      image: The image object.
      definition: The concept definition.

    Returns:
      A string containing the summary of the most important ambiguity, or an
      empty string.
    """
    whether_borderline_prompt = f"""
      <role>You are an expert linguist who is good at brainstorming creatively.</role>
      <input>
      You are given the definition of a visual concept,
      which serves as the guideline for human raters to determine whether an image is within the scope of this concept.
      You will also be given an image.
      </input>
      <task>
      As the concept owner is still actively working on the definition of the target concept,
      there are still some ambiguities in this concept definition.
      You will help examine whether an image might highlight important ambiguities in the concept definition and thus should be reviewed by the concept owner,
      so that the concept owner could further improve the definition.
      </task>

      <step1>
      Examine the image against the definition and determines whether the image should be classified as in-scope or out-of-scope.
      </step1>
      <step2>
      Now assume that the concept owner actually gives a different classification result for this image.
      Examine the image against the definition and reason what might be the important ambiguities that the current definition fails to capture.
      As the current definition is mostly correct, you should not completely ignore the current definition,
      but instead you should focus on identifying subtle but important ambiguities.
      </step2>
      <step3>
      Examine your reasoning in step2, and determine how likely these ambiguities actually make sense.
      This means that the concept owner is likely to be unclear about his definition at this point, or this point is likely to cause confusion to human raters.
      Some images are actually clear-cut in-scope or out-of-scope examples--in these cases, your ambiguities might not make much sense.
      </step3>
      <step4>
      If you believe that the ambiguities are important in step2, pick the most important ambiguity from step2 and summarize it in one sentence less than 30 words.
      Your summary should directly point out the elements that might cause the ambiguity in the image and the specific requirements in the definition.
      for instance,
      - "The image shows two people use sign language to communicate with each other, but it is unclear whether sign language is considered as "chatting"",
      - "The image shows a set of cartoon dogs, but it is unclear whether cartoon is considered as "dog" or not."
      But if you believe that the ambiguities are not important, then your summary should be an empty string.
      </step4>

      Provide your answer in the following XML structure:
        <classification>The classification result of the image and your reasoning.</classification>
        <counter-reasoning>Your reasoning about why the image might have been misclassified at step2</counter-reasoning>
        <examination>Your reasoning about whether the ambiguities are important at step3</examination>
        <summary>Your summary of the most important ambiguity if your answer is "yes" at step3; otherwise, leave it empty.</summary>

      Here is the definition you should work on:
        <visualConceptDescription>{definition.readable_string()}</visualConceptDescription>
    """
    response = self.model_client.gemini_image_prompt(
        whether_borderline_prompt,
        [image],
        cheap=True
    )
    # logger.debug('whether borderline image response: %s', response)
    summary = self.model_client.parse_xml(response, 'summary', [])
    if not summary:
      return ''
    return str(summary[0])

  def process_image_ambiguity(
      self,
      image,
      definition,
  ):
    """Process the image ambiguity.

    Args:
      image: The image object.
      definition: The concept definition.

    Returns:
      The determined ambiguity string for the image.
    """
    retries = 3
    while not image.ambiguity and retries > 0:
      try:
        image.ambiguity = self.whether_borderline_images(image, definition)
        # We want to distinguish between none and empty string.
        if image.ambiguity.strip() == '':
          break
      except Exception as e:  # pylint: disable=broad-except
        logger.error('Failed to process image ambiguity: %s', e)
        retries -= 1
    if image.ambiguity is None:
      logger.error(
          'Failed to process image ambiguity after %d retries.', retries
      )
    return image.ambiguity

  def surface_borderline_images(
      self,
      definition,
      images,
      min_cluster_size = 5,
      max_eps = 1,
  ):
    """Surface borderline images for the definition.

    Args:
      definition: The concept definition.
      images: The list of images.
      min_cluster_size: Minimum number of images to form a cluster.
      max_eps: Maximum epsilon for DBSCAN clustering.

    Returns:
      A list of borderline images representing the largest valid cluster,
      or None if no cluster was found.
    """
    # definition_summary = self.summarize_definition(definition)
    logger.info('Reflect on ambiguities for %d images\n', len(images))
    utils_py.run_in_batches(
        images, self.process_image_ambiguity, definition=definition
    )
    remaining_images = []
    remaining_ambiguities = []
    for image in images:
      if image.ambiguity:
        remaining_images.append(image)
        remaining_ambiguities.append(image.ambiguity)
    embeddings = utils_py.run_in_batches(
        remaining_ambiguities,
        self.image_curator.retrieval_client._get_text_embedding,
        batch_size=5,
    )

    # We choose an adaptive eps logic to find the best eps value.
    # Set a maximum eps value to 0.8 to avoid overclustering.
    start_eps = 0.2
    eps_step = 0.01

    final_clusters = None
    clusters = {}
    for eps_candidate in np.arange(start_eps, max_eps, eps_step):
      dbscan = DBSCAN(
          eps=eps_candidate, min_samples=min_cluster_size, metric='euclidean'
      )
      labels = dbscan.fit_predict(embeddings)

      # Group points into clusters, ignoring noise (label -1).
      clusters = {}
      for image_index, label in enumerate(labels):
        if label == -1:
          continue
        clusters.setdefault(int(label), []).append(image_index)

      # Check if we found at least one valid cluster.
      if clusters:
        logger.info(
            'Found %d cluster(s) of at least %d images with eps = %.2f.',
            len(clusters),
            min_cluster_size,
            eps_candidate,
        )
        final_clusters = clusters
        break

    if final_clusters is None:
      logger.warning(
          'Could not find any clusters of size >= %d even after trying eps up'
          ' to %.2f.',
          min_cluster_size,
          max_eps,
      )
      # If we cannot find a good cluster, we will use the last one.
      final_clusters = clusters

    final_clusters = list(final_clusters.values())
    # Report the stats of the final clusters found.
    if not final_clusters:
      logger.warning('No clusters found.')
      return None
    return [remaining_images[idx] for idx in final_clusters[0]]
