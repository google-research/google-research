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

"""A class that represents a structured definition."""

import logging
from typing import Optional, Union, Any
import xml.etree.ElementTree as ET
from agile_deliberation.agile_deliberation_lib import image as image_py
from agile_deliberation.agile_deliberation_lib import utils

MyImage = image_py.MyImage
logger = logging.getLogger(__name__)


class Definition:
  """The definition of a concept.

  Attributes:
    id: The id of the definition.
    concept: str. The concept name.
    description: str. The concept description.

    necessary_signals: list[Definition]
      The necessary signals of this concept if any.
    positive_signals: list[Definition]
      The positive signals of this concept if any.
    negative_signals: list[Definition]
      The negative signals of this concept if any.
    signals: list[Definition]
      A list of all signals of this concept.
    parent: Optional[Definition]
      The parent definition.
    signal_type: Optional[str]
      The signal type, can be 'necessary', 'positive', 'negative', or None.
      If it is None, then it is a root definition;
      otherwise it is a subconcept and is one of the 'signal_type' signals
      of its parent concept.

    previous_signals: list[Definition]
      The previous signals that have been explored in the enrich stage.
    previous_borderline_descriptions: list[str]
      The previous borderline descriptions that have been explored
      in the reflection stage.

    conditions: list[str]
      The list of all conditions that we should reflect on for surfacing
      conditional ambiguities in the reflection stage.
    condition_to_examine: Optional[int]
      The index of the next condition we should examine later.
      We will use this to retrieve the condition from the list of conditions.

    groundtruth: list[dict[str, Union[MyImage, str]]]
      The annotated images of this concept that we collect
      during the image reflection stage.

    dataset: list[MyImage]
      The dataset of images that we collect during the image enrichment stage.
      As these images have not been reflected yet,
      we want to reuse them in the reflection stage.
  """
  _next_id = 0

  def __init__(
      self,
      concept,
      description,
      signal_type = None,
      parent = None,
  ):
    """Initialize a definition.

    Args:
      concept: The concept name.
      description: The concept description.
      signal_type: The signal type, can be 'necessary', 'positive', or
        'negative'.
      parent: The parent definition.
    """
    # TODO(leijiew): We need to make sure the concept name in a definition tree
    # is unique. An assumption in our function look_up_signal_by_concept.
    self.concept = concept.strip()
    self.description = description.strip()

    self.necessary_signals = []
    self.positive_signals = []
    self.negative_signals = []
    self.signals = []

    self.parent = parent
    self.signal_type = signal_type

    self.id = Definition._next_id
    Definition._next_id += 1

    self.previous_signals = []
    self.previous_borderline_descriptions = []
    self.conditions = []
    self.condition_to_examine = None

    # images where users have provided feedback.
    self.groundtruth = []
    # store the dataset of images that we collect during the scoping stage.
    self.dataset = []

  def __getitem__(self, key):
    """Retrieve subconcepts of a definition.

    Args:
      key: The subconcept name.

    Returns:
      The subconcept Definition if found, else None.
    """
    for signal in self.signals:
      if signal.concept == key:
        return signal
    return None

  def look_up_signal_by_concept(
      self, concept
  ):
    """Look up a signal by concept name.

    Args:
      concept: The concept name to search for.

    Returns:
      The Definition object if found, else None.
    """
    if not concept:
      return None
    if self.concept == concept:
      return self
    for signal in self.signals:
      lookup_result = signal.look_up_signal_by_concept(concept)
      if lookup_result:
        return lookup_result
    return None

  def update_content(self, new_definition):
    """Update the definition with the new information.

    Args:
      new_definition: The new definition to update with.
    """
    new_definition_xml = new_definition.print_definition()
    self.parse_xml_string(new_definition_xml)


  def copy(self):
    """Copy a definition.

    Returns:
      A copied Definition instance.
    """
    new_def = Definition(self.concept, self.description, self.signal_type)
    for signal in self.signals:
      new_signal = signal.copy()
      new_signal.parent = new_def
      new_def.signals.append(new_signal)
      if signal.signal_type == 'necessary':
        new_def.necessary_signals.append(new_signal)
      elif signal.signal_type == 'positive':
        new_def.positive_signals.append(new_signal)
      elif signal.signal_type == 'negative':
        new_def.negative_signals.append(new_signal)
    new_def.groundtruth = self.groundtruth
    new_def.dataset = self.dataset
    new_def.previous_signals = self.previous_signals
    new_def.previous_borderline_descriptions = (
        self.previous_borderline_descriptions
    )
    return new_def

  def delete(self):
    """Delete a signal."""
    if self.parent:
      self.parent.signals = utils.remove_ref_from_list(
          self.parent.signals, self
      )
      if self.signal_type == 'necessary':
        self.parent.necessary_signals = utils.remove_ref_from_list(
            self.parent.necessary_signals, self
        )
      elif self.signal_type == 'positive':
        self.parent.positive_signals = utils.remove_ref_from_list(
            self.parent.positive_signals, self
        )
      elif self.signal_type == 'negative':
        self.parent.negative_signals = utils.remove_ref_from_list(
            self.parent.negative_signals, self
        )
    for signal in self.signals:
      del signal
    self.parent = None
    del self

  def update_signals(
      self, signals, signal_type
  ):
    """Update the signals with the new information.

    Args:
      signals: List of signals to update with.
      signal_type: The type of signals to update.

    Returns:
      The updated list of signals.
    """
    if not signals:
      return
    if isinstance(signals[0], dict):
      # Filter out empty signals.
      signals = [
          signal
          for signal in signals
          if (signal['name'] or signal['description'])
      ]
      signals = [
          Definition(
              signal['name'], signal['description'], signal_type, self,
          )
          for signal in signals
      ]
    else:
      for signal in signals:
        signal.parent = self
        signal.signal_type = signal_type
    if signal_type == 'necessary':
      self.necessary_signals.extend(signals)
    elif signal_type == 'positive':
      self.positive_signals.extend(signals)
    elif signal_type == 'negative':
      self.negative_signals.extend(signals)
    self.signals.extend(signals)
    return signals

  def collect_leaf_signals(self):
    """Collect all leaf signals.

    Returns:
      A list of leaf Definition instances.
    """
    leaf_signals = []
    if self.signals:
      for signal in self.signals:
        leaf_signals.extend(signal.collect_leaf_signals())
    else:
      leaf_signals.append(self)
    return leaf_signals

  def add_annotated_image(
      self, new_image,
      new_rating = None,
  ):
    """Add an annotated image to the groundtruth.

    Args:
      new_image: The image to be added.
      new_rating: The rating of the image.
    """

    # Make sure the new rating is a number.
    if new_rating is not None:
      try:
        new_rating = int(new_rating)
      except ValueError:
        logging.debug('The new rating is not a number: %s', new_rating)
        return
    else:
      new_rating = new_image.retrieve_rating(self) or {}
      if new_rating:
        new_rating = new_rating['decision']
      else:
        logger.debug(
            'The image has not been annotated for the concept %s.', self.concept
        )
        return

    # Check if this image has already been annotated.
    for annotation in self.groundtruth:
      if annotation['image'] == new_image:
        if annotation['rating'] != new_rating:
          logger.debug(
              'It seems that you want to update the rating of an image from %s'
              ' to %s.',
              annotation['rating'],
              new_rating,
          )
          annotation['rating'] = new_rating
          return
        else:
          logger.debug('The image has already been annotated.')
          return

    # Add the new image to the groundtruth.
    self.groundtruth.append({
        'image': new_image, 'rating': new_rating
    })

  def add_interesting_images(self, images):
    """Add images that have been searched but not reflected.

    Args:
      images: A list of MyImage objects.
    """
    self.dataset.extend(images)

  def add_previous_signals(self, previous_signal):
    """Add signals that have been explored in the enrich stage.

    Args:
      previous_signal: A previous signal definition.
    """
    self.previous_signals.append(previous_signal.copy())

  def add_previous_borderline_descriptions(
      self, descriptions
  ):
    """Add descriptions that have been explored in the reflection stage.

    Args:
      descriptions: List of descriptions.
    """
    self.previous_borderline_descriptions.extend(descriptions)

  def _valid_def_attrs(
      self,
      concept,
      description,
      signal_type,
  ):
    """Check if the definition attributes are valid.

    Args:
      concept: The concept string.
      description: The description string.
      signal_type: The signal type string.

    Returns:
      True if valid, False otherwise.
    """
    if concept is None or description is None or signal_type is None:
      return False
    if not (concept.strip() or description.strip()):
      # If both of them are empty strings.
      return False
    signal_type = signal_type.strip().lower()
    if signal_type not in ['necessary', 'positive', 'negative']:
      return False
    return True

  def _add_signal_section(
      self,
      signals,
      signal_type,
      level,
      prettify,
      editable,
      changed_signal_ids,
      image,
  ):
    """Stringify a signal section.

    Args:
      signals: List of signals to stringify.
      signal_type: Type of the signals.
      level: The level in the hierarchy.
      prettify: Whether to format for readability.
      editable: Whether the format is editable.
      changed_signal_ids: List of signal IDs that were changed.
      image: An optional MyImage instance.

    Returns:
      A list of stringified signal parts.
    """

    signals_list_str = []
    signals_list_str.append(f'<{signal_type}-signals>')
    for signal in signals:
      signal_list_str = signal.print_definition(
          level, prettify, editable, changed_signal_ids, image
      ).split('\n')
      # Indent two spaces for each signal line.
      signal_list_str = ['  ' + s for s in signal_list_str]
      signals_list_str.extend(signal_list_str)
      # This will leave a blank line after each concept.
      signals_list_str.append('')
    signals_list_str.append(f'</{signal_type}-signals>')
    if prettify:
      signals_list_str = ['  ' + s for s in signals_list_str]
    return signals_list_str

  def print_definition(
      self,
      level = None,
      prettify = False,
      editable = False,
      changed_signal_ids = None,
      image = None,
  ):
    """The ultimate print function.

    Args:
      level: The level of the definition that we should print out. None
        represents no level limit while 0 represents the top level.
      prettify: Whether to prettify the output for human readability.
      editable: Whether to print out the definition in an editable format, which
        means inserting some templates for the user to fill in. When it is set
        to True, then the prettify argument will be True as well.
      changed_signal_ids: Signals that we want to highlight.
      image: Where we want to retrieve the human rating.

    Returns:
      A string of the definition.
    """
    list_str = []
    signals_list_str = []
    new_level = level - 1 if level else None
    if self.necessary_signals and ((level is None) or (level > 0)):
      signals_list_str.extend(
          self._add_signal_section(
              self.necessary_signals,
              'necessary',
              new_level,
              prettify,
              editable,
              changed_signal_ids,
              image,
          )
      )

    if self.positive_signals and ((level is None) or (level > 0)):
      signals_list_str.extend(
          self._add_signal_section(
              self.positive_signals,
              'positive',
              new_level,
              prettify,
              editable,
              changed_signal_ids,
              image,
          )
      )

    if self.negative_signals and ((level is None) or (level > 0)):
      signals_list_str.extend(
          self._add_signal_section(
              self.negative_signals,
              'negative',
              new_level,
              prettify,
              editable,
              changed_signal_ids,
              image,
          )
      )

    list_str.append('<concept>')
    if prettify:
      if changed_signal_ids and (self.id in changed_signal_ids):
        list_str.append('  <!-- Start of the changed part-->')
      list_str.extend([
          f'  <name>{self.concept}</name>',
          f'  <description>{self.description}</description>',
      ])
      if image:
        human_rating = image.retrieve_rating(self)
        if human_rating:
          list_str.extend([
              '  <humanRating>',
              f'    <decision>{human_rating["decision"]}</decision>',
              f'    <summary>{human_rating["summary"]}</summary>',
              '  </humanRating>'
          ])
        else:
          list_str.append('  <humanRating>Not available</humanRating>')
      if changed_signal_ids and (self.id in changed_signal_ids):
        list_str.append('  <!-- End of the changed part -->')
    else:
      # Putting all information in one line to save space.
      concept_str = f'<name>{self.concept}</name><description>{self.description}</description>'
      if image:
        human_rating = image.retrieve_rating(self)
        if human_rating:
          concept_str += (
              f'<humanRating>{human_rating["decision"]}/humanRating>'
          )
        else:
          concept_str += '<humanRating>Not available</humanRating>'
      list_str.append(concept_str)
    list_str.extend(signals_list_str)
    list_str.append('</concept>')
    return '\n'.join(list_str)

  def __str__(self):
    return self.print_definition()

  def parse_xml_element(
      self, element
  ):
    """Update or create definition instances from xml elements.

    Args:
      element: The XML element to parse.
    """
    for signal_type in ['necessary', 'positive', 'negative']:
      new_signals = []
      for signal_element in element.findall(f'{signal_type}-signals/concept'):
        concept = utils.get_element_text(signal_element, 'name')
        description = utils.get_element_text(signal_element, 'description')
        if self._valid_def_attrs(concept, description, signal_type):
          new_signal = Definition(concept, description, signal_type, self)
          new_signal.parse_xml_element(signal_element)
          new_signals.append(new_signal)
      if signal_type == 'necessary':
        self.necessary_signals = new_signals
      elif signal_type == 'positive':
        self.positive_signals = new_signals
      elif signal_type == 'negative':
        self.negative_signals = new_signals
      self.signals.extend(new_signals)

  def parse_xml_string(self, xml_string):
    """Parse XML string into the definition.

    Args:
      xml_string: The XML string to parse.
    """
    root = ET.fromstring(xml_string)
    self.concept = utils.get_element_text(root, 'name')
    self.description = utils.get_element_text(root, 'description')
    self.signals = []
    self.parse_xml_element(root)

  def readable_string(self):
    """Convert the Definition instance to a readable string.

    Returns:
      A readable representation of the definition.
    """
    definition_str = self.concept + ': ' + self.description + '\n\n'
    if self.necessary_signals:
      definition_str += (
          'In-scope images must meet two necessary conditions.\n\n'
      )
      for signal in self.necessary_signals:
        definition_str += (
            f'<condition>\n{signal.readable_string()}\n'
            f'</condition>\n\n'
        )

    if self.positive_signals:
      definition_str += 'This includes any of the following visual elements:\n'
      for signal in self.positive_signals:
        definition_str += f'- {signal.concept}: {signal.description}\n'
    if self.negative_signals:
      definition_str += 'However, the following visual elements are excluded:\n'
      for signal in self.negative_signals:
        definition_str += f'- {signal.concept}: {signal.description}\n'
    return definition_str

  def serialize(self):
    """Convert the Definition instance to a serializable dictionary.

    Returns:
      A dictionary representation of the definition.
    """
    serialized_dataset = []
    for image in self.dataset:
      serialized_dataset.append(image.serialize())
    serialized_groundtruth = []
    for annotation in self.groundtruth:
      serialized_annotation = {
          key: value.serialize() if isinstance(value, MyImage) else value
          for key, value in annotation.items()
      }
      serialized_groundtruth.append(serialized_annotation)
    return {
        'concept': self.concept,
        'description': self.description,
        'signal_type': self.signal_type,
        'signals': [signal.serialize() for signal in self.signals],
        'previous_signals': [
            signal.serialize() for signal in self.previous_signals
        ],
        'previous_borderline_descriptions': (
            self.previous_borderline_descriptions
        ),
        'groundtruth': serialized_groundtruth,
        'dataset': serialized_dataset
    }

  @classmethod
  def deserialize(cls, data):
    """Reconstruct the Definition instance from a serialized dictionary.

    Args:
      data: The serialized dictionary.

    Returns:
      A reconstructed Definition instance.
    """
    instance = cls(
        concept=data['concept'],
        description=data['description'],
        signal_type=data['signal_type']
    )
    instance.previous_signals = [
        cls.deserialize(signal) for signal in data['previous_signals']
    ]
    instance.previous_borderline_descriptions = data[
        'previous_borderline_descriptions'
    ]
    signals = [
        cls.deserialize(signal) for signal in data['signals']
    ]
    for signal in signals:
      instance.update_signals([signal], signal.signal_type)

    groundtruth = []
    for item in data['groundtruth']:
      annotation = {}
      for key, value in item.items():
        if isinstance(value, dict) and 'image_bytes' in value:
          annotation[key] = MyImage.deserialize(value)
        else:
          annotation[key] = value
      groundtruth.append(annotation)
    instance.groundtruth = groundtruth

    dataset = []
    for item in data['dataset']:
      dataset.append(MyImage.deserialize(item))
    instance.dataset = dataset
    return instance
