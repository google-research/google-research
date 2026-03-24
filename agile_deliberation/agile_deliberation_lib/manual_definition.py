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

"""Functions for the manual definition conditions."""

from concurrent import futures
from typing import Any, Callable

import ipywidgets as widgets

from agile_deliberation_lib import components as components_py
from agile_deliberation_lib import image as image_py


ThreadPoolExecutor = futures.ThreadPoolExecutor
MyImage = image_py.MyImage


class ManualDefinition:
  """Class for the manual definition conditions."""

  def __init__(self, concept, description):
    """Initializes the ManualDefinition.

    Args:
      concept: The concept name.
      description: The concept description.
    """
    self.concept = concept
    self.description = description

  def serialize(self):
    """Serialize the manual definition to a string.

    Returns:
      A dictionary representation of the definition.
    """
    return {
        'concept': self.concept,
        'description': self.description,
    }

  @classmethod
  def deserialize(cls, data):
    """Deserialize the manual definition from a string.

    Args:
      data: The serialized dictionary data.

    Returns:
      A new ManualDefinition instance.
    """
    return cls(concept=data['concept'], description=data['description'])

  def readable_string(self):
    """Return a readable string for the manual definition.

    Returns:
      A formatted HTML string of the concept and description.
    """
    concept_str = f"""
        <name>{self.concept}</name>
        <description>{self.description}</description>
    """
    return concept_str

  def __str__(self):
    """Return a readable string representation.

    Returns:
      The readable string.
    """
    return self.readable_string()


def create_description_editor(definition):
  """Creates a UI with a non-editable concept and an editable description.

  Args:
    definition: An instance of the ManualDefinition class.

  Returns:
    A VBox widget containing the editor.
  """
  # 1. Display the concept as non-editable, styled HTML text.
  concept_display = components_py.set_instruction(
      f"""<b style='font-size: 16px;'>Concept:&nbsp;&nbsp;{definition.concept}</b>"""
  )

  # 2. Create the editable Textarea for the description.
  description_label = widgets.HTML(
      value='<b style="font-size: 14px;">Description:</b>',
      layout=widgets.Layout(width='90px', margin='0 8px 0 0'),
  )
  description_input = widgets.Textarea(
      value=definition.description,
      placeholder='Enter the full description...',
      layout=widgets.Layout(
          width='80%',
          height='500px',
          border='2px solid #4285F4',
          margin='0 0 2px 0',
          padding='0px'
      )
  )

  # 3. Set up the auto-save functionality for the description.
  def on_text_change(change):
    definition.description = change['new']

  description_input.observe(on_text_change, names='value')

  description_box = widgets.VBox(
      [description_label, description_input],
      layout=widgets.Layout(align_items='stretch', width='100%'),
  )

  # 4. Combine all parts into the final editor widget.

  return widgets.VBox(
      [concept_display, description_box],
      layout=widgets.Layout(
          align_items='flex-start',
          width='100%',
          padding='10px',  # Add some overall padding.
      ),
    )


def set_image_with_actions(
    image,
    classify_callback,
    search_callback,
    save_callback,
    status,
):
  """Set an image with Search and Predict buttons underneath.

  Args:
    image: The image object.
    classify_callback: The callback for classification.
    search_callback: The callback for searching.
    save_callback: The callback for saving.
    status: The status, 'default' or 'searched'.

  Returns:
    A container widget with the image and actions.
  """
  if status == 'default':
    border = '2px solid #848891'
  else:
    border = '4px solid #f27755'
  image_widget = components_py.set_image(
      image, height=250, border=border
  )

  image_actions = []

  # Output area for prediction results.
  info_output_container = components_py.vbox([])
  if classify_callback is not None:
    # --- Predict Button ---
    predict_button = widgets.Button(
        description='Predict',
        icon='lightbulb-o',
        tooltip='Run classification model on this image',
        layout=widgets.Layout(width='auto'),
    )
    predict_button.style.button_color = '#b5d2f7'
    predict_button.style.text_color = '#000'
    classifier_future = ThreadPoolExecutor(max_workers=1).submit(
        classify_callback, image
    )
    @components_py.with_loading(predict_button)
    def _predict_action(b):
      info_output_container.children = tuple([
          components_py.set_instruction('Loading...')
      ])
      classify_result = classifier_future.result()
      classifier_label = (
          'In Scope' if classify_result['decision'] >= 3 else 'Out of Scope'
      )
      classify_instruction = components_py.set_instruction(
          f"""
          <div style='padding: 5px 0 0 0; font-family: sans-serif; max-width: 300px;'>
              <div style='font-size: 13px; margin-top: 3px;'>
                  <b>Classifier Decision:</b> {classifier_label}
              </div>
              <div style='font-size: 13px; margin-top: 6px;'>
                  <i><b>Classifier Rationale:</b> {classify_result['summary']}</i>
              </div>
          </div>
          """
      )
      info_output_container.children = tuple([classify_instruction])

    predict_button.on_click(_predict_action)
    image_actions.append(predict_button)

  if search_callback is not None:
    # --- Search Button ---
    search_button = widgets.Button(
        description='Search',
        icon='search',
        tooltip='Find and add similar images to the gallery',
        layout=widgets.Layout(width='auto'),
    )
    search_button.style.button_color = '#e0edda'
    search_button.style.text_color = '#000'

    @components_py.with_loading(search_button)
    def search_func_with_loading(image):
      search_callback(image)
    search_button.on_click(lambda _: search_func_with_loading(image))
    image_actions.append(search_button)

  if save_callback is not None:
    # --- Save Button ---
    save_button = widgets.Button(
        description='Save',
        icon='save',
        tooltip='Save the image for future testing',
        layout=widgets.Layout(width='auto'),
    )
    save_button.style.button_color = '#ffebe8'
    save_button.style.text_color = '#000'

    def _save_action(_):
      save_callback(image)
      # Disable the button after saving.
      save_button.disabled = True
      save_button.description = 'Saved'

    save_button.on_click(_save_action)
    image_actions.append(save_button)

  # Arrange buttons side-by-side.
  buttons_hbox = components_py.hbox(
      image_actions, layout=widgets.Layout(gap='10px'),
  )

  # Combine image, buttons, and prediction output vertically.
  container = components_py.vbox(
      [image_widget, buttons_hbox, info_output_container],
      layout=widgets.Layout(
          align_items='flex-start',
          width='auto',
          padding='5px',
          border='1px solid transparent',
      ),
  )
  container.add_class('image-widget')
  return container


def display_image_gallery(
    images,
    classify_func,
    search_func,
    save_func,
    search_images_by_query_func,
):
  """Displays a gallery of PIL Images in a horizontal row.

  Args:
    images: A list of PIL Image objects.
    classify_func: A function to classify an image.
    search_func: A function to search for more images.
    save_func: A function to save an image.
    search_images_by_query_func: A function to search images by query.

  Returns:
    A widgets.HBox object containing the image widgets.
  """

  gallery_widget = widgets.HBox(
      [],
      layout=widgets.Layout(
          overflow='auto hidden',
          display='flex',
          flex_flow='row',
          align_items='center',
          gap='10px',
          width='100%',
          padding='0 0 0 32px',
      ),
  )

  if search_func is not None:
    def append_images(image):
      new_images = search_func(image)
      new_image_widgets = [
          set_image_with_actions(
              image, classify_func, append_images, save_func, 'searched'
          )
          for image in new_images
      ]
      gallery_widget.children = (
          tuple(new_image_widgets) + gallery_widget.children
      )

  else:
    append_images = None

  if search_images_by_query_func is not None:
    search_images_button = widgets.Button(
        description='Search',
        icon='search',
        tooltip='Find and add similar images to the gallery',
        layout=widgets.Layout(width='auto'),
        button_style='primary'
    )

    @components_py.with_loading(search_images_button)
    def search_images(_):
      images = search_images_by_query_func()
      new_image_widgets = [
          set_image_with_actions(
              image, classify_func, append_images, save_func, 'default'
          )
          for image in images
      ]
      gallery_widget.children = tuple(new_image_widgets)

    search_images_button.on_click(search_images)
    return components_py.vbox(
        [search_images_button, gallery_widget], width='100%'
    )
  else:
    new_image_widgets = [
        set_image_with_actions(
            image, classify_func, append_images, save_func, 'default'
        )
        for image in images
    ]
    gallery_widget.children = tuple(new_image_widgets)
    return gallery_widget
