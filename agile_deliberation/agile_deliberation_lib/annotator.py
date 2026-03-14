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

"""A port of pigeon with keyboard shortcuts."""

import functools
import logging
import time
from typing import Optional

from google.colab import output
from IPython.display import clear_output
from IPython.display import display
from IPython.display import Javascript
import ipywidgets as widgets
import numpy as np

from agile_deliberation.agile_deliberation_lib import components as components_py
from agile_deliberation.agile_deliberation_lib import image as image_py
from agile_deliberation.agile_deliberation_lib import utils


logger = logging.getLogger(__name__)


class Annotator:
  """Annotator class.

  Attributes:
    cache: A PickleCache object if cache_file_name was provided.
    images: The list of MyImage objects.
    mode: 'complete' or 'adaptive'.
    expected_num: The expected number of positive and negative labels.
    image_urls: List of image URLs.
    images_to_annotate: List of (index, url) tuples for images to annotate.
    annotations: Dict storing annotations for each image URL.
    timestamps: List tracking the timing for each image interaction.
    current_item: The current (index, url) being annotated.
    counts: Dict storing counts for 'positive', 'negative', and 'no image'.
    out: The output widget for displaying the image.
    options: List of annotation labels.
    count_label: HTML widget displaying annotation progress.
    url_label: HTML widget displaying current URL.
    buttons: List of Button widgets for annotation.
  """

  def __init__(
      self,
      images,
      mode = 'complete',
      options = None,
      cache_file_name = None,
      expected_num = 100,
  ):
    """Initializes the Annotator.

    Args:
      images: List of images to annotate.
      mode: Annotation mode, 'complete' or 'adaptive'.
      options: List of annotation options.
      cache_file_name: Optional filename for caching.
      expected_num: Expected number of images to annotate in 'adaptive' mode.
    """
    if not options:
      options = ['positive', 'negative', 'no image']
    self.cache = None
    if cache_file_name:
      self.cache = utils.PickleCache(cache_file_name)

    self.images = images
    self.mode = mode  # 'complete' or 'adaptive'
    self.expected_num = expected_num
    self.image_urls = [image.url for image in images]

    if self.cache:
      # Represents the indices of the images that wait to be annotated.
      self.images_to_annotate = [
          (index, url) for index, url in enumerate(self.image_urls)
          if url not in self.cache
      ]
      # Represent the annotations of all images.
      self.annotations = {}
      for url in self.image_urls:
        self.annotations[url] = self.cache[url] if url in self.cache else None
    else:
      self.images_to_annotate = [
          (index, url) for index, url in enumerate(self.image_urls)
      ]
      self.annotations = {url: None for url in self.image_urls}
    self.timestamps = []
    self.current_item = None

    # Keep counts of each label. If some urls were already in the cache,
    # count these.
    self.counts = {}
    for rating in ('positive', 'negative', 'no image'):
      self.counts[rating] = sum(
          1 if label == rating else 0 for label in self.annotations)

    self.out = widgets.Output()
    self.options = options
    self.count_label = widgets.HTML()
    self.url_label = widgets.HTML()
    self.buttons: list[widgets.Button] = []

  def add_annotation(self, annotation):
    """Add annotation to the current item.

    Args:
      annotation: The annotation value to add.
    """
    if self.current_item is not None:
      url = self.current_item[1]

      # Keep track of the time spent on each image.
      if self.annotations[url] is None:
        # Only append the timestamp if the current item is not annotated.
        self.timestamps.append((self.current_item, time.time()))

      # Update annotation counts, making sure that if we replace an annotation,
      # we decrease the count for the former annotation.
      if self.annotations[url] is not None:
        self.counts[self.annotations[url]] -= 1
      self.annotations[url] = annotation
      self.counts[annotation] += 1
    self.show_next()

  def set_button_color(self):
    """Set the color of the buttons according to the annotation."""
    url = self.current_item[1]
    if self.annotations[url] == 'positive':
      self.buttons[0].button_style = 'success'
      self.buttons[1].button_style = ''
      self.buttons[2].button_style = ''
    elif self.annotations[url] == 'negative':
      self.buttons[0].button_style = ''
      self.buttons[1].button_style = 'danger'
      self.buttons[2].button_style = ''
    elif self.annotations[url] == 'no image':
      self.buttons[0].button_style = ''
      self.buttons[1].button_style = ''
      self.buttons[2].button_style = 'warning'
    else:
      self.buttons[0].button_style = ''
      self.buttons[1].button_style = ''
      self.buttons[2].button_style = ''

  def determine_annotation_order(self):
    """Determine the annotation order of the current item.

    Determine whether the current item is already annotated, and if so,
    find the order of the current item.

    Returns:
      The order index of the current item.
    """
    for i in range(len(self.timestamps) - 1, -1, -1):
      if self.timestamps[i][0] == self.current_item:
        return i
    return len(self.timestamps)

  def show_prev(self):
    """Show the previous image."""

    current_item_order = self.determine_annotation_order()
    # If there are images before the current item, go back to the previous item.
    if current_item_order > 0:
      if (
          self.current_item is not None
          and self.annotations[self.current_item[1]] is None
      ):
        # Add this item back to the queue.
        self.images_to_annotate.append(self.current_item)
        # Sort the queue by the index.
        self.images_to_annotate.sort(key=lambda x: x[0])
      self.current_item = self.timestamps[current_item_order - 1][0]
    # Otherwise, we do not update the current item.

    self.set_label_text()
    self.set_button_color()
    with self.out:
      clear_output(wait=True)
      image = self.images[self.current_item[0]]
      display(components_py.set_image(image))

  def show_next(self):
    """Show the next image."""
    # If the user has not annotated the current image, do not show the next.
    if (
        self.current_item is not None
        and self.annotations[self.current_item[1]] is None
    ):
      return
    self.set_label_text()
    if self.determine_end():
      timestamps = [item[1] for item in self.timestamps]
      timestamp_diffs = np.diff(timestamps)
      print(
          f'Average time spent per image: {np.mean(timestamp_diffs)} +/-'
          f' {np.std(timestamp_diffs)}'
      )
      # Filter out outliers.
      if len(timestamps) > 5:
        timestamps = sorted(np.diff(timestamps))
        start_idx = int(len(timestamps) / 4)
        end_idx = len(timestamps) - start_idx
        timestamps = timestamps[start_idx:end_idx]
        print(
            'Average time spent per image (filtered):'
            f' {np.mean(timestamps)} +/- {np.std(timestamps)}'
        )
      print('Annotation done.')
      for btn in self.buttons:
        btn.disabled = True
      if self.cache:
        print('Saving annotations into cache.')
        for url, annotation in self.annotations.items():
          self.cache[url] = annotation
        self.cache.save()
      return

    self.current_item = self.determine_next_item()
    self.set_label_text()
    if self.current_item is None:
      return
    self.set_button_color()
    with self.out:
      clear_output(wait=True)
      image = self.images[self.current_item[0]]
      display(components_py.set_image(image, height=400))

  def set_label_text(self):
    """Show the progress of the annotation."""
    if self.mode == 'complete':
      num_annotated = sum(
          [label is not None for label in self.annotations.values()]
      )
      current_item_order = self.determine_annotation_order()
      self.count_label.value = 'On image {} | {} annotated | {} left'.format(
          current_item_order + 1,
          num_annotated,
          len(self.image_urls) - num_annotated
      )
    elif self.mode == 'adaptive':
      self.count_label.value = (
          '{}/{} positives annotated | {}/{} negatives annotated'.format(
              self.counts['positive'],
              self.expected_num,
              self.counts['negative'],
              self.expected_num,
          )
      )
    if self.current_item is not None:
      self.url_label.value = (
          f'{self.current_item[1]}'
      )

  def determine_end(self):
    """Determine whether the annotation process is complete.

    Returns:
      True if complete, False otherwise.
    """
    if self.mode == 'complete':
      # If there is no image to annotate, we are done.
      return not self.images_to_annotate
    elif self.mode == 'adaptive':
      # If there are more than expected_num positive and negative images, we
      # are done.
      return (
          self.counts['positive'] >= self.expected_num
          and self.counts['negative'] >= self.expected_num
      ) or (not self.images_to_annotate)

  def determine_next_item(self):
    """Determine the next index to annotate.

    Returns:
      A tuple of (index, url) or None.
    """
    current_item_order = self.determine_annotation_order()
    # If we are in the previous images.
    if current_item_order < len(self.timestamps) - 1:
      return self.timestamps[current_item_order + 1][0]

    if self.mode == 'complete':
      next_item = self.images_to_annotate.pop(0)
      return next_item
    elif self.mode == 'adaptive':
      # If there are any images with indices less than 2 * expected_num,
      # we will annotate them first.
      for index, url in self.images_to_annotate:
        if index < 2 * self.expected_num and self.annotations[url] is None:
          self.images_to_annotate.remove((index, url))
          return (index, url)

      if self.counts['positive'] < self.expected_num:
        # Iterate through the indices from the end to the beginning.
        for index, url in reversed(self.images_to_annotate):
          # Because we rank the images by the probability of being positive,
          # the images at the end of the list are more likely to be positive.
          if self.annotations[url] is None:
            self.images_to_annotate.remove((index, url))
            return (index, url)
      elif self.counts['negative'] < self.expected_num:
        # Iterate through the indices from the beginning to the end.
        for index, url in self.images_to_annotate:
          if self.annotations[url] is None:
            self.images_to_annotate.remove((index, url))
            return (index, url)

      return None

  def annotate(self):
    """Annotate the images.

    Returns:
      A dictionary of the annotations.
    """
    if self.determine_end():
      print('We have enough annotations for our {} task'.format(self.mode))
      return self.annotations
    self.set_label_text()
    display(self.count_label)
    display(self.url_label)

    self.buttons = []

    descriptions_dict = {
        'positive': 'In Scope',
        'negative': 'Out of Scope',
        'no image': 'No Image',
    }
    for label in self.options:
      btn = widgets.Button(description=descriptions_dict[label])

      def on_click(label, _):

        self.add_annotation(label)

      btn.on_click(functools.partial(on_click, label))
      self.buttons.append(btn)

    box = widgets.HBox(self.buttons)

    display(box)

    self.out = widgets.Output()
    display(self.out)

    self.show_next()
    self.add_keyboard_callback()
    return self.annotations

  def add_keyboard_callback(self):
    """Allow users to annotate images with keyboard shortcuts."""

    def key_handler(key):
      # TODO(kenjihata): I don't know why the arrows have to come before the
      # number keys.
      if key == 'ArrowLeft':
        self.show_prev()
        return
      elif key == 'ArrowRight':
        self.show_next()
        return
      for i, label in enumerate(self.options):
        if int(key) == i + 1:
          self.add_annotation(label)

    output.register_callback('keydown', key_handler)

    display(
        Javascript("""
    document.body.addEventListener('keydown', (e) => {
      google.colab.kernel.invokeFunction('keydown', [e.key], {});
    });
    """))

  def get_counts(self):
    """Get the counts of positive, negative, and no image annotations.

    Returns:
      A tuple of (positive_count, negative_count, no_image_count).
    """
    return (self.counts['positive'], self.counts['negative'],
            self.counts['no image'])

