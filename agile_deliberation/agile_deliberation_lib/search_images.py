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

"""Image Curator that searches for images for a concept definition."""

import logging
from typing import Iterator, Optional, List, Any
from agile_deliberation_lib import definitions as definitions_py
from agile_deliberation_lib import image as image_py
from agile_deliberation_lib import llm as llm_py
from agile_deliberation_lib import retrieval as retrieval_py


MyImage = image_py.MyImage
Definition = definitions_py.Definition
logger = logging.getLogger(__name__)


class ImageCurator:
  """Curates a list of images according to a definition."""

  def __init__(
      self,
      retrieval_client,
      model_client,
      images_per_query = 10,
  ):
    """Initializes the ImageCurator.

    Args:
      retrieval_client: The retrieval client.
      model_client: The LLM model client.
      images_per_query: The number of images to retrieve per query.
    """
    self.retrieval_client = retrieval_client
    self.images_per_query = images_per_query
    self.queries_per_description = 5
    self.model_client = model_client

  def image_search_sufficient_signal(
      self,
      definition,
      focus_signal,
      description_num = 3,
      images_num = 10,
  ):
    """Search images for a new focus signal that might be added to the definition.

    According to the structure of the concept definition,
    we call different functions to search for images.

    Args:
      definition: The parent of the focus signal.
      focus_signal: The descriptive signal to reflect on.
      description_num: The number of descriptions to generate.
      images_num: The number of images we want to get in each turn.

    Returns:
      A list of images, or None if the definition type is not supported.
    """

    # Here the groundtruth stores all images that
    # the user has reflected on before.
    if definition.signal_type == 'necessary' and definition.parent is not None:
      images = self.image_search_sufficient_signal_of_a_necessary_concept(
          definition, focus_signal, description_num,
          images_num
      )
    elif definition.parent is None:
      images = self.image_search_sufficient_signal_of_a_descriptive_concept(
          definition, focus_signal, description_num,
          images_num
      )
    else:
      return None

    return images

  def image_search_sufficient_signal_of_a_descriptive_concept(
      self,
      definition,
      focus_signal,
      description_num,
      images_num,
  ):
    """Search images for a new focus signal as part of a descriptive concept.

    Args:
      definition: The overall concept definition.
      focus_signal: The signal that we want to search for images.
      description_num: The number of descriptions to generate.
      images_num: The number of images we want to get in each turn.

    Returns:
      A list of images that satisfy the focus signal but none of the negative
      signals, or None if the given concept is a necessary concept.
    """

    # Sanity check.
    if definition.necessary_signals:
      logger.warning(
          'This concept is a necessary concept rather than a descriptive'
          ' concept.'
      )
      return None

    if focus_signal.parent is not definition:
      logger.debug(
          'The focus signal is not a positive or negative signal of the target'
          ' concept. So we copy the definition and add the focus signal.'
      )
      definition = definition.copy()
      definition.update_signals([focus_signal], 'positive')
    image_search_descriptive_concept_prompt = f"""
      <role>You are an expert linguist who is good at brainstorming creatively and using image search engines.</role>
      <input>
      You are given a focused concept name and an optional context for this focused concept.
      </input>
      <task>
      Your task is to generate image descriptions that fully align with the focus concept.
      These image descriptions will later be used to find images that satisfy the focus concept through a search engine.
      </task>

      Now take a deep breath and follow my guidance step by step.
      <step1>
      Repeat the name and the description of the focus concept.
      </step1>

      <step2>
      Write down {description_num} keyword(s) that satisfy the focus concept and its description.
      The first keyword should just be the keyword of the whole description, oftentimes its concept name.
      For the other keywords, you could use the specific examples mentioned in the concept description if there are any;
      otherwise, you should come up with a more specific instance of the focus concept or a different way of saying the same thing.
      You should aim for distinct and diverse keywords so that we could find more diverse in-scope images.

      <examples>
      - for the focus concept 'Images show that these two peole are in physical affection', we could start with "physical affection", "hugging", "holding hands".
      - for the focus concept 'Images show that these dogs are living in a big city', we could start with "city dog walking", "dog walking skyscrapers", "dog walking New York".
      - for the condition "Images show that these children playing summer activities", we could start with "children summer activities", "children swimming", "children beach vacation".
      - for the condition 'Images show that the person is exercising at the gym', good keywords are: '"exercise equipment gym', 'gym lifting weights', 'gym workout'
      </examples>
      </step2>

      <step3>
      For each keyword you generated in step 1 for the focus concept, follow these steps.
      On the basis of this keyword, write a less than 10 words description of an image that features this keyword and also satisfies other parts of the focus concept.

      Here are more requirements for this description.
      - This description must include and feature this keyword.
      - Avoid adding less important details that only enrich the description but are actually not necessary for an image to be in-scope.
      - You should try your best to describe a category of images that are very likely to be in-scope for this focus concept.
        In cases when you find it impossible to do so, you should make sure that your description is very visually similar to the overall concept.
      - Your description should be not be self-contradictory.
      - You should aim for less than 10 words for each description.
      </step3>


      Write your answer in a valid XML format, adhering to the following structure:
        <condition-keywords>
          Write your three keywords for the focus concept in step 1 here.
        </condition-keywords>
        <descriptions>
          <!-- there should be {description_num} descriptions in total -->
          <description></description>
          <description></description>
          ....
        </descriptions>

      Here is the concept definition you should work on:
      <focus-concept>{str(focus_signal)}</focus-concept>
      <context>{definition.description}</context>
    """
    response = self.model_client.gemini_prompt(
        image_search_descriptive_concept_prompt, cheap=True
    )
    logger.debug('searching images for descriptive concept\n%s', response)
    descriptions = self.model_client.parse_xml(
        response, 'descriptions/description', []
    )
    return self.rank_images_from_description(descriptions, images_num)

  def determine_which_signals(
      self, definition, focus_signal
  ):
    """Determine which signals of other necessary signals should be included.

    For the brainstorming of a new focus signal, determine which signals of
    other necessary signals should be included for image searching.

    Sometimes the edge case signal might only apply to a specific signal.
    For instance, an overlaid before and after image is only possible for
    skin treatments but not other body treatments.

    In some cases, the edge case signal conflicts with every other signals.
    For instance, a before and after image about house renovation is not
    possible for any other body modifications.

    We need to have an agent that determines which signals to include for
    image searching.

    Args:
      definition: The parent of the focus signal, also a necessary concept.
      focus_signal: The signal that we want to search for images.

    Returns:
      The definition that we decided to search for images with.
    """
    overall_definition = definition.parent
    if definition.signal_type != 'necessary' or overall_definition is None:
      logger.warning(
          'This function is only for concepts that are decomposed into'
          ' necessary signals rather than descriptive signals.'
      )
      return None

    overall_definition = overall_definition.copy()
    new_necessary_signals = [focus_signal.copy()]
    for signal in overall_definition.necessary_signals:
      if signal.concept == definition.concept:
        continue
      else:
        new_necessary_signals.append(signal)
    overall_definition.necessary_signals = []
    overall_definition.update_signals(new_necessary_signals, 'necessary')
    determine_which_signals_prompt = f"""
      <role>You are an expert linguist who is good at brainstorming creatively and using image search engines.</role>
      <input>
      You are given a concept definition and a focused concept as part of the definition.
      </input>
      <task>
      We want to search for images that must satisfy the focused concept, and if possible, satisfy other necessary signals of the overall concept.
      Your task is to identify whether and which descriptive signal of each other necessary signal goes well with the focused concept.
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

      Now take a deep breath and follow my guidance step by step.
      <step1>
      Repeat the description of the focus concept.
      </step1>

      <step2>
      For each other necessary signal of the overall concept, reason the following two questions.
      <step2.1>Determine whether this necessary signal conflicts with the focus concept.
      You should only continue to step 2.2 if your answer is 'No'. Otherwise, you should skip the step 2.2.
      </step2.1>
      <step2.2>
      Among the positive signals of this necessary signal, determine which positive signal is mostly like to appear together with the focus concept.
      For each positive signal, you should imagine what it looks like if an image satisfies the focus concept and this positive signal.
      You should then select the positive signal that is most likely to appear together with the focus concept.
      If it does not have any positive signal, you should choose this necessary signal instead.
      </step2.2>
      </step2>

      <step3>
      Write down the name of positive signals you selected in step 2.2; you should also write down the corresponding name of the necessary signal.
      For necessary signals that you have answered 'No' in step 2.1, you should leave the positive signal field empty.
      You do not need to include the focus concept here.
      </step3>

      Write your answer in a valid XML format, adhering to the following structure:
        <repeat-focus-concept>
          Repeat the focus concept here.
        </repeat-focus-concept>
        <selection-reasoning>
        Write down your reasoning for each necessary signal (except the focus concept) in step 2.1 and 2.2 here.
        </selection-reasoning>
        <selections>
          <signal>
            <necessary-signal>Write the name of the necessary signal here.</necessary-signal>
            <positive-signal>Write the name of the positive signal you selected in step 2.2 here.</positive-signal>
          </signal>
        </selections>

      Here is the concept definition you should work on:
      <conceptDefinition>{overall_definition.print_definition()}</conceptDefinition>
      <focus-concept>{str(focus_signal)}</focus-concept>
    """
    response = self.model_client.gemini_prompt(
        determine_which_signals_prompt, cheap=True
    )
    logger.debug('image descriptions for necessary concept\n%s', response)
    selections = self.model_client.parse_xml(
        response, 'selections/signal', ['necessary-signal', 'positive-signal']
    )
    selections = {
        signal['necessary-signal']:
            signal.get('positive-signal', signal['necessary-signal'])  # pytype: disable=attribute-error
        for signal in selections
    }

    new_overall_definition = Definition(
        overall_definition.concept, overall_definition.description
    )
    new_necessary_signals = [focus_signal.copy()]
    for signal in definition.parent.necessary_signals:  # pytype: disable=attribute-error
      if signal.concept == definition.concept:
        continue
      elif signal.concept in selections:
        selected_name = selections[signal.concept]
        if signal.positive_signals:
          selected_signal = signal[selected_name]
        else:
          selected_signal = signal
        new_necessary_signals.append(
            Definition(
                selected_signal.concept,
                selected_signal.description,
                signal_type='necessary',
            )
        )
    new_overall_definition.update_signals(new_necessary_signals, 'necessary')
    logger.debug(
        'the overall definition for image searching is\n%s',
        new_overall_definition.print_definition(prettify=True)
    )
    logger.debug('the final prompt is\n%s', determine_which_signals_prompt)
    return new_overall_definition

  def image_search_sufficient_signal_of_a_necessary_concept(
      self,
      definition,
      focus_signal,
      description_num,
      images_num,
  ):
    """Search images for a new focus signal as part of a descriptive concept.

    The focus signal itself is a necessary condition of the overall concept.

    Args:
      definition: The necessary concept.
      focus_signal: The signal that we want to search for images.
      description_num: The number of descriptions to generate.
      images_num: The number of images we want to get finally.

    Returns:
      A list of images that satisfy the focus signal and the other necessary
      conditions, or None.
    """
    overall_definition = definition.parent
    if definition.signal_type != 'necessary' or overall_definition is None:
      logger.warning(
          'This function is only for concepts that are decomposed into'
          ' necessary signals rather than descriptive signals.'
      )
      return None

    overall_definition = self.determine_which_signals(definition, focus_signal)
    if overall_definition is None:
      return None
    reflect_on_a_necessary_concept_prompt = f"""
      <role>You are an expert linguist who is good at brainstorming creatively and using image search engines.</role>
      <input>
      You are given a concept definition and a focused concept name.
      </input>
      <task>
      Your task is to generate image descriptions that fully align with the focus concept.
      These image descriptions will later be used to find images that satisfy the focus concept through a search engine.
      In particular, we want images that must highlight the focus concept.
      </task>

      <define-a-concept>
      We define our concepts in a structured way.
      The concept is defined by a list of necessary conditions, the conjunction of which will be logically equivalent to the visual concept.
      In other words, images that satisfy these conditions will be exactly images that satisfy the visual concept.
      </define-a-concept>

      Now take a deep breath and follow my guidance step by step.

      <step1>
      Write down {description_num} key phrases that satisfy the focus concept of the description.
      You could generate these key phrases in the following ways:
      - The first phrase should just be the key phrase of the whole description, oftentimes its concept name.
      - You could use the specific examples mentioned in the concept description if there are any; otherwise, you should come up with a more specific instance of the focus concept.
      - You should not introduce unnecessary details that are not necessary for the focus concept when you come up with the key phrases.
      You should aim for distinct and diverse phrases so that we could find more diverse in-scope images.
      If the focus concept has a requirement about the visibility of the focus concept, or the modality of the focus concept (e.g., visual or textual), make sure you include them in the key phrases.

      <examples>
      - for the focus concept 'Images show that these two peole are in physical affection', we could start with "physical affection", "hugging", "holding hands".
      - for the focus concept 'Images show that these dogs are living in a big city', we could start with "city dog walking", "dog walking skyscrapers", "dog walking New York".
      - for the condition "Images show that these children playing summer activities", we could start with "children summer activities", "children swimming", "children beach vacation".
      - for the condition 'Images show that the person is exercising at the gym', good key phrases are: '"exercise equipment gym', 'gym lifting weights', 'gym workout'
      </examples>
      At the end of this step, you should have {description_num} key phrases.
      </step1>

      <step2>
      For each key phrase you generated in step 1 for the focus concept, follow these steps.
      On the basis of this key phrase, write a less than 10 words description of an image that features this key phrase and also satisfies each other necessary condition.
      At the end of this step, you should have {description_num} descriptions.
      Here are more requirements for this description.
      - This description must include and feature this key phrase.
      - Avoid adding less important details that only enrich the description but are actually not necessary for an image to be in-scope.
      - You should try your best to describe a category of images that are very likely to be in-scope for the overall concept, i.e., satisfying all necessary conditions.
        However, if the key phrase contradicts with some other necessary conditions, you should prioritize the key phrase over the other necessary conditions.
        In cases when you find it impossible to do so, you should make sure that your description is very visually similar to the overall concept.
      - Your description should be not be self-contradictory.
      - You should aim for less than 10 words for each description.

      <examples>
      - for the key phrase 'romantic kiss' with the other necessary condition as 'Images show a travel destination', the description could be 'a couple is having a romantic kiss in a travel destination'.
      - for the key phrase 'friends' with the other necessary condition as 'Images show a romantic relationship', the description could be 'two intimate friends' (since friends conflict with romantic relationship, we pick intimacy as a close substitute)
      - for the key phrase 'gym workout' with the other necessary condition as 'Images show weight loss', the description could be 'a person is showing their fit body after gym workout'.
      - for the key phrase 'family picnic' with the other necessary condition as 'Images show a rural setting', the description could be 'a family is enjoying a picnic in a rural setting'.
      </examples>
      </step2>

      Write your answer in a valid XML format, adhering to the following structure:
        <condition-keyphrases>
          Write your three key phrases for the focus concept in step 1 here.
        </condition-keyphrases>
        <descriptions>
          <!-- there should be {description_num} descriptions in total -->
          <description></description>
          ....
        </descriptions>

      Here is the concept definition you should work on:
      <conceptDefinition>{overall_definition.print_definition(level=1)}</conceptDefinition>
      <focus-concept>{str(focus_signal)}</focus-concept>
    """
    response = self.model_client.gemini_prompt(
        reflect_on_a_necessary_concept_prompt, cheap=True
    )
    logger.debug('image descriptions for necessary concept\n%s', response)
    descriptions = self.model_client.parse_xml(
        response, 'descriptions/description', []
    )
    return self.rank_images_from_description(descriptions, images_num)

  def generate_queries_from_description(
      self,
      description,
      previous_queries = None,
  ):
    """Generate search queries from a description.

    Args:
      description: The description to generate queries for.
      previous_queries: Options list of previous queries.

    Returns:
      A list of search query strings.
    """
    if previous_queries is None:
      previous_queries = []
    previous_queries_str = '\n--'.join(previous_queries)
    generate_query_prompt = f"""
      <role>You are an expert linguist who is good at brainstorming creatively and using image search engines.</role>
      <input>
        You are a description of images we want to find.
        In some cases, there are queries that are previously explored before.
      </input>
      <task>
        Your task is to generate search queries for finding images that fully align with the description.
        These search queries will be given to a image search engine to find relevant images.
      </task>

      <step1>
        According to our guidance, generate {self.queries_per_description} search queries that help us find images that satisfy the image description.
        These search queries will later be used to retrieve images from an image search engine.
        Make sure that your generated queries are distinct from each other and can retrieve a different set of images from the search engine.
        You should also avoid repeating previously explored queries if there are any.
        There should be {self.queries_per_description} search queries in total.

        <guidance>
          Here are a few rule of thumbs about a good search query.
          1. Search engines assign equal weights to each non-stop word in the query,
            so you should not add too many less important words in the query which undermines the importance of the actual keywords.
          2. Words that are more concrete, visually concrete, specific are preferred.
            For instance, 'countryside' > 'rural setting', 'smile' > 'happy', 'coffee' > 'drinks', 'kissing' > 'affection'.
          3. Search engines are not good at capturing negative visual elements in a query, such as 'hidden laptop', 'unnoticeable person', 'there is no man'.
          Instead, mentioning these negative elements in the query will on the contrary make the search engine pay attention to the object that you wish to exclude.
          As we will filters images to make sure they are indeed in-scope, you should pay less attention to negative elements in the description.
          Or you could think about what positive elements might exclude the presence of the negative elements.
        </guidance>
      </step1>

      Write your answer in a valid XML format, adhering to the following structure:
        <search-queries>
          <!-- there should be {self.queries_per_description} queries in total -->
          <query>search query</query>
          <query>search query</query>
          <query>search query</query>
          ....
        </search-queries>

      Here is the concept description you should work on:
      <description>{description}</description>
      <previous-queries>{previous_queries_str}</previous-queries>
    """
    response = self.model_client.gemini_prompt(
        generate_query_prompt, cheap=True
    )
    logger.debug(
        'Generated search queries for description: %s\n%s',
        description,
        response,
    )
    queries = self.model_client.parse_xml(response, 'search-queries/query', [])
    return queries or []

  def generate_diverse_descriptions(
      self,
      definition,
      previous_descriptions,
      image_type,
  ):
    """Generate diverse descriptions for a concept definition.

    Args:
      definition: The concept definition.
      previous_descriptions: List of previously generated descriptions.
      image_type: The type of image to generate descriptions for.

    Returns:
      A list of new and diverse description strings.
    """
    previous_descriptions_str = '\n'.join(previous_descriptions)
    num_descriptions = 3
    generate_descriptions_prompt = f"""
      <role>You are an expert linguist who is good at brainstorming creatively.</role>
      <input>
        You are given a structured concept definition and a list of previously generated descriptions.
      </input>
      <task>
        Your task is to generate {num_descriptions} more description that covers a possibly {image_type} category of images.
        This category of images should be different from the categories covered by previous descriptions.
        Since users still explore and improve their concept definition, you should also consider similar categories
        that might be fully covered by the current definition but are relevant.
        This generated description will later be used to find images that satisfy the concept through a search engine.
      </task>

      <define-a-concept>
        Before answering the question, it is a must for you to understand how we define a concept in a structured and iterative way.
        - If the concept is defined by a list of necessary conditions, then an image is in-scope if it satisfies all necessary conditions.
        - If the concept is defined by a list of positive and negative conditions, then an image is in-scope if it satisfies at least one positive condition and does not satisfy any negative condition.
        - A concept can be first defined by a list of necessary conditions, and then each necessary condition can be further defined by a list of positive and negative conditions.
      </define-a-concept>
      <image-type>
        - in-scope: images that are likely to be in-scope for the concept.
        - ambiguous: images that are borderline in-scope for the concept.
        There are some important ambiguities that the concept definition does not articulate clearly.
        - out-of-scope: images that are likely to be out-of-scope for the concept.
      </image-type>


      <step1>
      Examine the concept definition and previous descriptions, propose {num_descriptions} new categoriesof images are in-scope but are different from the previous descriptions.
      These categories should cover significantly different categories of images from the previous descriptions.
      </step1>

      <step2>
      Based on your reasoning in step1, write down {num_descriptions} new descriptions for the {image_type} category of images.
      The description should be concise, specific, and clear. You should aim for less than 20 words for this description.
      </step2>

      Write your answer in a valid XML format, adhering to the following structure:
        <reasoning>
          Write your reasoning for new categories of images in step 1 here.
        </reasoning>
        <descriptions>
          Write your description for the in-scope category in step 2 here.
          <!-- there should be {num_descriptions} descriptions in total -->
          <description></description>
          <description></description>
          <description></description>
        </descriptions>

      Here is the concept definition you should work on:
      <definition>{definition.readable_string()}</definition>
      <previous-descriptions>
        {previous_descriptions_str}
      </previous-descriptions>
    """

    response = self.model_client.gemini_prompt(
        generate_descriptions_prompt, cheap=True
    )
    descriptions = self.model_client.parse_xml(
        response, 'descriptions/description', []
    )
    logger.debug(
        'Generate %s category of images for definition: %s\n',
        image_type,
        descriptions
    )
    return descriptions or []

  def rank_images_from_description(
      self,
      descriptions,
      images_num = 40,
  ):
    """Search for images from a description and rank them.

    Args:
      descriptions: Description strings to generate queries for.
      images_num: The maximum number of images.

    Returns:
      A list of deduplicated images.
    """
    images = []
    for description in descriptions:
      images.extend(
          self.retrieval_client.text_image_search(description, images_num)
      )

    # Rank the images according to their distances to the queries.
    deduplicated_images = MyImage.deduplicate_images(images)
    logger.debug(
        'Deduplicated images from %d to %d images',
        len(images),
        len(deduplicated_images),
    )
    return deduplicated_images
