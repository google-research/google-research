# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""An image annotator with keyboard shortcuts.

This file is based on https://github.com/agermanidis/pigeon.
"""
# pylint: disable-all
import functools
import time
from typing import Optional, Sequence

from google.colab import output
from IPython.display import Javascript, clear_output, display
from ipywidgets import Button, HBox, HTML, Output
import numpy as np
import pickle_cache
from PIL import Image
import utils


def im_display_fn(im):
  """Displays the image in the Colab cell.

  Args:
    im: The image to display.

  Returns:
    Whether the image was displayed.
  """
  if not im:
    display('The image cannot be displayed.')
    return False
  display(im)
  return True


class Annotator:
  """An image annotator with keyboard shortcuts that can be shown in Colab.

  Attributes:
    annotations: The annotations.
    buttons: The buttons in the UI.
    cache: A cache storing the previous ratings a user has done.
    count_label: A label we show in the UI keeping track of rating stats.
    counts: A counter that keeps track of how many positive, negative, etc.
    current_index: The index of the current image.
    image_urls: The urls of the images.
    images: The PIL Image objects of each image. Corresponds to image_urls.
    indices: The indices of the images that were downloaded properly.
    timestamps: The timestamps to compute how fast people are rating.
    options: Options for the UI.
    out: The output UI.
    url_label: A label in the output UI showing the current URL.
  """

  # pylint:disable=dangerous-default-value
  def __init__(
      self,
      image_urls,
      options=['positive', 'negative', 'no image'],
      cache_file_name = None,
      automatically_label_no_images=True,
  ):
    self.cache = None
    if cache_file_name:
      self.cache = pickle_cache.PickleCache(cache_file_name)

    self.current_index = -1
    self.image_urls = list(image_urls)
    self.images = utils.download_images_parallel(self.image_urls)
    if self.cache:
      if automatically_label_no_images:
        self.indices = [
            i
            for i, (url, img) in enumerate(zip(self.image_urls, self.images))
            if url not in self.cache and img is not None
        ]
      else:
        self.indices = [
            i for i, url in enumerate(self.image_urls) if url not in self.cache
        ]
      self.annotations = [
          self.cache[url] if url in self.cache else None
          for url in self.image_urls
      ]
    else:
      if automatically_label_no_images:
        self.indices = [
            i for i, image in enumerate(self.images) if image is not None
        ]
      else:
        self.indices = list(range(len(self.image_urls)))
      self.annotations = [None] * len(self.image_urls)
    if automatically_label_no_images:
      for i, image in enumerate(self.images):
        if image is None:
          self.annotations[i] = 'no image'
    self.timestamps = [None] * len(self.indices)

    # Keep counts of each label. If some urls were already in the cache,
    # count these.
    self.counts = {}
    for rating in ('positive', 'negative', 'no image'):
      self.counts[rating] = sum(
          1 if label == rating else 0 for label in self.annotations
      )

    self.out = Output()
    self.options = options
    self.count_label = HTML()
    self.url_label = HTML()

  def add_annotation(self, annotation):
    """Once the user rates an image, adds the annotation.

    Args:
      annotation: the annotation
    """
    if self.current_index < len(self.indices):
      idx = self.indices[self.current_index]
      # Update annotation counts, making sure that if we replace an annotation,
      # we decrease the count for the former annotation.
      if self.annotations[idx] is not None:
        self.counts[self.annotations[idx]] -= 1
      self.annotations[idx] = annotation
      self.counts[annotation] += 1
      self.timestamps[self.current_index] = time.time()
    self.show_next()

  def set_button_color(self):
    """Sets the color of the buttons."""
    idx = self.indices[self.current_index]
    if self.annotations[idx] == 'positive':
      self.buttons[0].button_style = 'success'
      self.buttons[1].button_style = ''
      self.buttons[2].button_style = ''
    elif self.annotations[idx] == 'negative':
      self.buttons[0].button_style = ''
      self.buttons[1].button_style = 'danger'
      self.buttons[2].button_style = ''
    elif self.annotations[idx] == 'no image':
      self.buttons[0].button_style = ''
      self.buttons[1].button_style = ''
      self.buttons[2].button_style = 'warning'
    else:
      self.buttons[0].button_style = ''
      self.buttons[1].button_style = ''
      self.buttons[2].button_style = ''

  def show_prev(self):
    self.current_index -= 1
    if self.current_index < 0:
      self.current_index = 0
      return
    self.set_label_text()
    self.set_button_color()
    with self.out:
      clear_output(wait=True)
      im_display_fn(self.images[self.indices[self.current_index]])

  def show_next(self):
    """Moves to the next image, if it exists."""
    non_cached_index = self.indices[self.current_index]
    if self.current_index >= 0 and self.annotations[non_cached_index] is None:
      return
    self.current_index += 1
    if self.current_index >= len(self.indices):
      timestamp_diffs = np.diff(self.timestamps)
      print(f'Timestamps(debug): {self.timestamps}')
      print(
          f'Average time spent per image: {np.mean(timestamp_diffs)} +/-'
          f' {np.std(timestamp_diffs)}'
      )
      # Filter out outliers.
      if len(self.timestamps) > 5:
        timestamps = sorted(np.diff(self.timestamps))
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
        for url, annotation in zip(self.image_urls, self.annotations):
          self.cache[url] = annotation
        self.cache.save()
      return
    self.set_label_text()
    self.set_button_color()
    with self.out:
      clear_output(wait=True)
      im_display_fn(self.images[self.indices[self.current_index]])

  def set_label_text(self):
    """Updates the label text."""
    noncached = [self.annotations[i] for i in self.indices]
    num_annotated = sum([x is not None for x in noncached])
    self.count_label.value = 'On image {} | {} annotated | {} left'.format(
        self.current_index, num_annotated, len(self.indices) - num_annotated
    )
    self.url_label.value = (
        f'{self.image_urls[self.indices[self.current_index]]}'
    )

  def annotate(self):
    """Annotates the images."""
    if len(self.indices) == 0:
      print('All images already annotated before.')
      return self.annotations
    self.set_label_text()
    display(self.count_label)
    display(self.url_label)

    self.buttons = []

    for i, label in enumerate(self.options):
      btn = Button(description=label)

      def on_click(label, btn):
        self.add_annotation(label)

      btn.on_click(functools.partial(on_click, label))
      self.buttons.append(btn)

    box = HBox(self.buttons)

    display(box)

    self.out = Output()
    display(self.out)

    self.show_next()
    self.add_keyboard_callback()
    return self.annotations

  def add_keyboard_callback(self):
    """Adds the proper callbacks so users can rate with the keyboard."""

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

    display(Javascript("""
    document.body.addEventListener('keydown', (e) => {
      google.colab.kernel.invokeFunction('keydown', [e.key], {});
    });
    """))

  def get_counts(self):
    return (
        self.counts['positive'],
        self.counts['negative'],
        self.counts['no image'],
    )
