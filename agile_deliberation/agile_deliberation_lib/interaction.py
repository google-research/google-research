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

"""Deliberation interaction that supports each step of the deliberation process in Colab."""

import concurrent.futures as concurrent_futures
import logging
from typing import Any, Callable, Optional

from IPython.display import clear_output
from IPython.display import display
from IPython.display import HTML
import ipywidgets as widgets

from agile_deliberation_lib import classifier as classifier_py
from agile_deliberation_lib import components as components_py
from agile_deliberation_lib import definitions as definitions_py
from agile_deliberation_lib import image as image_py
from agile_deliberation_lib import refine_definition as refine_definition_py
from agile_deliberation_lib import reflection as reflection_py
from agile_deliberation_lib import utils


Definition = definitions_py.Definition
MyImage = image_py.MyImage
ImageClassifier = classifier_py.ImageClassifier
DefinitionRefiner = refine_definition_py.DefinitionRefiner
display_with_styles = components_py.display_with_styles
logger = logging.getLogger(__name__)


class DeliberationInteraction:
  """Display and interact with a concept definition.

  Attributes:
    reflection: The reflection agent that reason about conceptual ambiguity.
    image_curator: The image curator that curates images for a concept.
    definition_refiner: The definition refiner that refines the definition.
    classifier: The classifier agent that classifies images according to the
      concept definition.
    test_signal_times: The number of times to test a signal to make sure
      it actually works on an image.
    keep_output: Whether to keep the output in each turn.
  """

  def __init__(
      self,
      reflection,
      classifier,
      definition_refiner,
      keep_output = False,
  ):
    """Initializes the DeliberationInteraction object.

    Args:
      reflection: The reflection agent.
      classifier: The classifier agent.
      definition_refiner: The definition refiner agent.
      keep_output: Whether to keep the output in each turn.
    """
    self.reflection = reflection
    self.image_curator = reflection.image_curator
    self.definition_refiner = definition_refiner
    self.classifier = classifier
    self.keep_output = keep_output
    self.test_signal_times = 3
    # Persistent Output widget so display() calls from button callbacks are
    # routed to the correct cell output rather than a hidden widget output.
    self._output = widgets.Output()

  def decompose_concept(
      self,
      now_definition,
      continue_fn,
      want_decomposition = False,
  ):
    """Interaction components for decomposing a concept.

    Args:
      now_definition: The current concept definition.
      continue_fn: Callback to continue to the next step.
      want_decomposition: Whether we want to enforce decomposition.

    Returns:
      None.
    """

    children_definitions = self.reflection.decompose_concept(
        now_definition, want_decomposition
    )
    if len(children_definitions) == 1:
      # This means that this concept should not be further decomposed.
      logger.info(
          'The concept %s is not further decomposed.', now_definition.concept)
      continue_fn(True)
      return

    introduce_text = components_py.set_instruction(f"""
        <div style="color: black; font-size: 16px;">
          For the concept <span style="color: #144a74; font-style: italic; font-weight: bold;">{now_definition.concept}</span>,
          the agent decomposes it into necessary signals as follows:
        </div>""")

    # Texts for asking users to take some action.
    action_text = components_py.set_instruction("""
      <div style="color: black; font-size: 16px;">
        Feel free to edit these signals or retry the decomposition. If you are satisfied with the decomposition, please click the confirm button.
      </div>""")

    # Red button.
    retry_button = components_py.set_button('Retry', color='red')
    # Green button.
    confirm_button = components_py.set_button('Confirm', color='green')

    action_widget = components_py.hbox(
        [action_text, retry_button, confirm_button]
    )

    children_definition_widgets = []
    # Create TextArea widgets for each condition.
    for _, definition in enumerate(children_definitions):
      definition_widget = components_py.edit_signal(definition)
      children_definition_widgets.append(definition_widget)

    children_definition_widgets = components_py.vbox(
        children_definition_widgets,
        width='100%',
        gap='10px',
    )

    all_widgets = components_py.vbox(
        [introduce_text, children_definition_widgets, action_widget],
        width='100%',
    )

    def redefine_concept():
      components_py.clear_container(all_widgets)

      for definition in children_definitions:
        definition.delete()

      self.decompose_concept(now_definition, continue_fn, want_decomposition=True)

    retry_button.on_click(lambda _: redefine_concept())
    @components_py.with_loading(confirm_button)
    def confirm_decomposition():
      # As deleted signals are still part of the list of children_definitions,
      # we need to filter them out.
      remaining_children = [
          child
          for child in children_definitions
          if child.parent == now_definition
      ]
      now_definition.update_signals(remaining_children, 'necessary')
      continue_fn(True)

    confirm_button.on_click(lambda _: confirm_decomposition())
    with self._output:
      clear_output(wait=True)
      display_with_styles(all_widgets)

  def sufficient_signal_feedback(
      self,
      now_definition,
      display_continue_button,
  ):
    """The shared interaction components for sufficient signal feedback.

    Args:
      now_definition: The current concept definition.
      display_continue_button: Callback to display the continue button.

    Returns:
      The generated interaction widget.
    """
    all_images = []
    gallery_widget = components_py.vbox(
        [], padding='0 0 0 32px',
        width='100%', overflow='hidden hidden',
        align_items='stretch',
    )
    image_gallery = None

    if len(now_definition.previous_signals) % 3 != 2:
      # We alternate between golden and borderline categories.
      sufficient_signal = self.reflection.brainstorm_golden_category(
          now_definition
      )
    else:
      sufficient_signal = self.reflection.brainstorm_borderline_category(
          now_definition
      )

    if sufficient_signal is None:
      logger.warning('The agent could not brainstorm a sufficient signal.')
      return

    signal_widget = components_py.set_instruction(
        f"""<div style="color: #c45a65; font-size: 18px;">
              {sufficient_signal.description}
            </div>
        """
    )
    show_example_button = components_py.set_button('Review Relevant Images')
    future_images = utils.async_execute(
        lambda: self.image_curator.image_search_sufficient_signal(
            now_definition,
            sufficient_signal.copy(),
            description_num=3,
            images_num=50,
        )
    )
    @components_py.with_loading(show_example_button)
    def show_image_examples():
      """Show example images for a sufficient signal."""
      nonlocal gallery_widget, image_gallery, future_images

      display_needed = False
      if not gallery_widget.children:
        display_needed = True

      if future_images:
        try:
          image_examples = future_images.result(timeout=120)
        except concurrent_futures.TimeoutError:
          logger.warning('Image search timed out for this signal.')
          image_examples = []
        except Exception:
          logger.warning('Image search failed for this signal.', exc_info=True)
          image_examples = []
      else:
        image_examples = []
      if image_examples:
        image_gallery = components_py.display_image_gallery(
            image_examples, image_gallery)
        all_images.extend(image_examples)
        if display_needed:
          gallery_widget.children = [image_gallery]
      else:
        if not all_images:
          gallery_widget.children = [
              components_py.set_instruction(
                  """
                  <div style="color: gray; font-size: 16px;">
                    The agent could not find any images for this signal from the dataset, which might suggest that this signal is irrelevant.
                  </div>
                  """
              )
          ]
      show_example_button.disabled = True
      show_example_button.style.button_color = '#d9d9d9'

    show_example_button.on_click(lambda _: show_image_examples())

    signal_widget = components_py.hbox(
        [signal_widget, show_example_button],
        width='100%',
        gap='10px',
    )

    action_instruction = components_py.set_instruction(
        """
            <div style="color: black; font-size: 16px;">Do you want to incorporate this category as part of your definition?</div>
        """
    )

    positive_button = components_py.set_button(
        'In-scope signals', color='green'
    )  # green
    negative_button = components_py.set_button(
        'Out-of-scope signals', color='red'
    )  # red
    irrelevant_button = components_py.set_button(
        'Clearly out-of-scope signals', color='gray'
    )  # black
    feedback_widgets = components_py.hbox(
        # [positive_button, negative_button, irrelevant_button],
        [positive_button, negative_button],
    )
    action_widget = components_py.hbox(
        [action_instruction, feedback_widgets],
        padding='0 0 0 32px',
    )

    edit_signal_widget = components_py.vbox(
        [], align_items='stretch',
        max_width='1500px',
        padding='0 0 0 32px',
    )
    all_widgets = components_py.vbox(
        [
            signal_widget,
            gallery_widget,
            action_widget,
            edit_signal_widget,
        ],
        overflow='hidden hidden',
        align_items='stretch',
    )
    def add_signal(signal, signal_type):
      """Incorporate a signal into the definition based on the user feedback."""
      nonlocal edit_signal_widget
      edit_signal_widget.children = []
      if signal_type == 'positive':
        # Remove the other two buttons.
        feedback_widgets.children = (positive_button,)
      elif signal_type == 'negative':
        feedback_widgets.children = (negative_button,)
      else:
        feedback_widgets.children = (irrelevant_button,)

      if signal_type in ['positive', 'negative']:
        signal = now_definition.update_signals([signal], signal_type)[0]
        action_instruction = components_py.set_instruction("""
          <div style="color: black; font-size: 16px;">Now help edit the language of the signal however you see fit.</div>
        """)
        definition_widget = components_py.edit_signal(signal)
        signal.add_interesting_images(all_images)
        edit_signal_widget.children += tuple(
            [action_instruction, definition_widget]
        )
      display_continue_button()

    positive_button.on_click(
        lambda _: add_signal(sufficient_signal, 'positive')
    )
    negative_button.on_click(
        lambda _: add_signal(sufficient_signal, 'negative')
    )
    irrelevant_button.on_click(
        lambda _: add_signal(sufficient_signal, 'irrelevant')
    )
    return all_widgets

  def sufficient_signal_feedback_multiple(
      self,
      now_definition,
      continue_signal_fn,
      next_concept_fn,
      signal_number = 1,
  ):
    """Brainstorm several sufficient signals for a concept in one turn.

    Args:
      now_definition: The current concept definition.
      continue_signal_fn: Callback for continuing the signal feedback.
      next_concept_fn: Callback to get the next concept name.
      signal_number: Number of signals to explore.
    """
    title_widget = components_py.set_instruction(
        f"""
          <div style="color: black; font-size: 16px;">
            For the concept <span style="color: #144a74; font-style: italic; font-weight: bold;">{now_definition.concept}</span>,
            the agent wants to hear your thoughts on the following {signal_number} categories of images:
          </div>
        """)

    # Placeholder Output widget so the continue button appears inline in the
    # layout rather than being sent to a hidden widget-callback output.
    continue_output = widgets.Output()

    feedback_collected = [False for _ in range(signal_number)]
    def display_continue_button(index):
      """Display the continue button after the user feedback."""
      feedback_collected[index] = True
      if not all(feedback_collected):
        # We only show the continue button when all signals have been explored.
        return
      next_widget_items = []
      next_concept_name = next_concept_fn()
      if next_concept_name == now_definition.concept:
        continue_button = components_py.set_button(
            f'Explore more signals for the "{now_definition.concept}"'
        )
        @components_py.with_loading(continue_button)
        def continue_event():
          continue_signal_fn(False)
        continue_button.on_click(lambda _: continue_event())

        next_widget_items.append(continue_button)

      if next_concept_name != now_definition.concept:
        if not next_concept_name:
          exit_button_label = 'Done'
        else:
          exit_button_label = f'Explore "{next_concept_name}"'
        exit_button = components_py.set_button(
            exit_button_label,
            color='gray'
        )

        @components_py.with_loading(exit_button)
        def exit_event():
          # Explore the next concept.
          continue_signal_fn(True)

        exit_button.on_click(lambda _: exit_event())
        next_widget_items.append(exit_button)

      next_widget = components_py.hbox(
          next_widget_items,
          width='75%'
      )

      with continue_output:
        display_with_styles(next_widget)

    signal_widgets = []
    for index in range(signal_number):
      signal_widget = self.sufficient_signal_feedback(
          now_definition, lambda index=index: display_continue_button(index)
      )
      border_widget = components_py.set_instruction("""
        <div style="color: black; font-size: 16px;">
          <hr style="border: 1px solid #ccc; margin: 16px 0 16px 0;">
        </div>
      """)
      signal_widgets.extend([signal_widget, border_widget])

    all_widgets = components_py.vbox(
        [
            title_widget,
            components_py.vbox(
                signal_widgets,
                align_items='stretch',
            ),
            continue_output,
        ],
        overflow='hidden hidden',
    )

    with self._output:
      if not self.keep_output:
        clear_output(wait=True)
      display_with_styles(all_widgets)

  def examine_improvements(
      self,
      new_definition,
      old_definition,
      continue_fn
  ):
    """Allow users to examine the iterations of the definition.

    Args:
      new_definition: The improved definition.
      old_definition: The original definition.
      continue_fn: Callback to continue with the accepted definition.
    """
    title_widget = components_py.set_instruction(
        f"""
          <div style="color: black; font-size: 16px;">
            Based on your feedback, our system proposes the following definition for the concept
            <span style="color: #144a74; font-style: italic; font-weight: bold;">{new_definition.concept}</span>:
          </div>
        """)

    continue_button = components_py.set_button('Accept', color='green')

    @components_py.with_loading(continue_button)
    def continue_event(_):
      continue_fn(new_definition)

    continue_button.on_click(continue_event)

    toggle_view_button = components_py.set_button('Edit', color='linen')
    output_area = widgets.Output()
    state = {'current_view': 'diff'}  # 'diff' or 'interactive'

    def show_diff_view():
      """Renders the HTML diff between old and new."""
      diff_html = components_py.generate_improved_diff_html(
          old_definition.readable_string(), new_definition.readable_string()
      )
      with output_area:
        output_area.clear_output(wait=True)
        display(HTML(f"""
        <div style="border: 1px solid #ccc; padding: 12px; border-radius: 8px; line-height: 1.4; font-family: sans-serif; white-space: pre-wrap;">
          {diff_html}
        </div>
        """))

    def show_interactive_view():
      """Shows the interactive definition widget."""
      with output_area:
        output_area.clear_output(wait=True)
        components_py.load_css_styles()
        display(components_py.interactive_definition(new_definition))

    def on_toggle_clicked(b):
      if state['current_view'] == 'diff':
        state['current_view'] = 'interactive'
        b.description = 'View Diff'
        show_interactive_view()
      else:
        state['current_view'] = 'diff'
        b.description = 'Edit Definition'
        show_diff_view()

    toggle_view_button.on_click(on_toggle_clicked)

    button_group = components_py.hbox([toggle_view_button, continue_button])
    top_bar = components_py.hbox(
        [title_widget, button_group],
        layout=widgets.Layout(
            width='100%',
            justify_content='space-between',
            align_items='center',
        )
    )

    # Initial display
    show_diff_view()  # Start with the diff view

    with self._output:
      clear_output(wait=True)
      display_with_styles(
          components_py.vbox(
              [top_bar, output_area],
              align_items='stretch',
          )
      )

  def evaluate_image(
      self, image, definition
  ):
    """A wrapper function to call the classifier to evaluate an image.

    Args:
      image: The image object.
      definition: The concept definition.

    Returns:
      The classification result containing decision and summary, or None.
    """
    return self.classifier.classify_image(image, definition)

  def prepare_reflection_widget(
      self,
      image,
      definition,
      monitor_label_fn,
  ):
    """Prepare the reflection widget for an image and prompt reflection.

    Args:
      image: The image object.
      definition: The concept definition.
      monitor_label_fn: Function to monitor the label changes.

    Returns:
      A dictionary of generated widgets and the classifier future.
    """

    executor = concurrent_futures.ThreadPoolExecutor()
    future = executor.submit(self.evaluate_image, image, definition)

    ambiguity_widget = components_py.set_instruction(
        f"""
          <div style="color: #0085ff; font-size: 14px; font-weight: bold; font-style: italic;">
            Ambiguity: {image.ambiguity}
          </div>
        """
    )

    default_rating = None
    if image.user_rating is not None:
      default_rating = 'In Scope' if image.user_rating == 5 else 'Out of Scope'
    rating_button = widgets.RadioButtons(
        options=['In Scope', 'Out of Scope'],
        description='Your Decision:',
        value=default_rating,
        # Allow description to take space.
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='auto')
    )
    rating_button.observe(monitor_label_fn, names='value')

    image_widget = components_py.set_image(image, height=250)
    image_vbox = components_py.vbox([image_widget], width='600px')

    comparison_widget = widgets.HTML()
    feedback_prompt = components_py.set_instruction(
        "<div style='font-size: 14px; font-weight: bold;'>Tell u why you make"
        ' such a decision? Does this ambiguity provoke any thinking?</div>'
    )
    feedback_box = components_py.set_textarea(
        label='',
        value=image.feedback if image.feedback else '',
        placeholder=(
            'e.g., "The image contains text which should be out of scope."'
        ),
        height='120px',
    )
    return {
        'image_widget': image_vbox,
        'rating_buttons': rating_button,
        'ambiguity_widget': ambiguity_widget,
        'comparison_widget': comparison_widget,
        'feedback_prompt': feedback_prompt,
        'feedback_box': feedback_box,
        'classifier_future': future,
    }

  def prepare_comparison_summary(
      self,
      change,
      reflection_widget,
      reflection_info,
  ):
    """Observer function to update comparison and feedback prompt.

    Args:
      change: A dictionary containing the new label value.
      reflection_widget: The reflection corresponding to the image.
      reflection_info: The classification info corresponding to the image.
    """
    user_label = change['new']
    if user_label is None: return
    user_rating = 5 if user_label == 'In Scope' else 1

    whether_aligned = ImageClassifier.determine_correct_rating(
        reflection_info['decision'], user_rating)

    if whether_aligned:
      color = 'green'
      match_text = 'MATCH ✅'
    else:
      color = 'red'
      match_text = 'MISMATCH ❌'

    classifier_label = (
        'In Scope' if reflection_info['decision'] >= 3 else 'Out of Scope'
    )
    html_value = f"""
        <div style='color:{color}; padding: 5px 0 0 0; font-family: sans-serif;'>
            <div style='font-size: 14px; font-weight: bold; margin-bottom: 5px;'>
                {match_text}
            </div>
            <div style='font-size: 13px; margin-top: 3px;'>
                <b>Classifier Decision:</b> {classifier_label}
            </div>
            <div style='font-size: 13px; margin-top: 6px;'>
                <i><b>Classifier Rationale:</b> {reflection_info['summary']}</i>
            </div>
        </div>
    """
    reflection_widget['comparison_widget'].value = html_value

  def show_reflection_ui(
      self,
      reflection_widget,
      reflection_info,
  ):
    """Sets up the reflection UI for a single image after initial labeling.

    Args:
      reflection_widget: A dictionary containing UI components for reflection.
      reflection_info: The dictionary tracking classification and feedback info.

    Returns:
      A UI widget organizing the components.
    """
    try:
      classifier_result = reflection_widget['classifier_future'].result()
      reflection_info['decision'] = int(classifier_result['decision'])
      reflection_info['summary'] = classifier_result['summary']

      # Create the observer function.
      def update_fn(change):
        self.prepare_comparison_summary(
            change, reflection_widget, reflection_info
        )

      # Initial call to set content.
      rating_button = reflection_widget['rating_buttons']
      if rating_button.value:
        update_fn({'new': rating_button.value})
      rating_button.observe(update_fn, names='value')

      feedback_widget = components_py.vbox(
          [
              reflection_widget['feedback_prompt'],
              reflection_widget['ambiguity_widget'],
              reflection_widget['feedback_box'],
          ],
          margin='0 0 0 5px',
          justify_content='flex-start',
          width='35%',
          height='100%',
      )
      comparison_widget = components_py.vbox(
          [
              reflection_widget['rating_buttons'],
              reflection_widget['comparison_widget'],
          ],
          align_items='stretch',
          justify_content='flex-start',
          width='30%',
      )
      reflection_widget['image_widget'].layout.width = '30%'
      return components_py.hbox(
          [
              reflection_widget['image_widget'],
              comparison_widget,
              feedback_widget,
          ],
          align_items='stretch',
          justify_content='space-between',
          margin='0 0 0 5px'
      )
    except Exception as e:
      error_widget = widgets.HTML(
          value=(
              "<div style='color:red; padding-left:10px;'><hr>Error"
              f' processing: {e}</div>'
          )
      )
      return components_py.hbox(
          [
              reflection_widget['image_widget'],
              error_widget,
          ],
          align_items='stretch',
      )

  def image_reflections(
      self,
      definition,
      images_to_reflect,
      continue_fn,
  ):
    """Interaction components for image reflection.

    Args:
      definition: The definition we want to improve.
      images_to_reflect: The images we want to reflect on.
      continue_fn: The function to continue to the next cluster of images.
    """
    # We want to first show an image gallery for users to label.
    # At the same time, we want to start classifying the images in the backend.
    # Once we get the user labels and the classier outputs, we want to highlight
    # misclassified images and ask users for any feedback for each image.
    # This would then help us improve the classifier in a batch.
    label_instruction = components_py.set_instruction(f"""
      <div style="color: black; font-size: 16px;">
        Tell us if you think each following image is in scope or out of scope for the concept
        <br/>
      </div>""")

    finish_label_button = components_py.set_button(
        'Finish Labeling', disabled=True
    )

    submit_feedback_button = components_py.set_button(
        'Submit Feedback', color='blue', visibility='hidden'
    )

    reflection_widgets = []
    reflection_infos = [{} for _ in range(len(images_to_reflect))]
    output_area = widgets.Output()

    def check_all_ratings(_):
      """Monitor any RadioButton changes. Enables button if all are rated."""
      # Checks if every button has a value other than its initial `Skip` state.
      all_rated = all(
          reflection_widget['rating_buttons'].value is not None
          for reflection_widget in reflection_widgets
      )
      logger.debug('check_all_ratings: %s', all_rated)
      finish_label_button.disabled = not all_rated

    utils.run_in_batches(images_to_reflect, self.classifier.add_image_ocr_text)
    for image in images_to_reflect:
      image_widget_dict = self.prepare_reflection_widget(
          image, definition, check_all_ratings
      )
      reflection_widgets.append(image_widget_dict)
    # In case all images have a default valid rating.
    check_all_ratings(None)

    @components_py.with_loading(finish_label_button)
    def on_finish_click(_):
      finish_label_button.disabled = True
      finish_label_button.layout.display = 'none'
      submit_feedback_button.layout.visibility = 'visible'

      feedback_widgets = []
      for index, reflection_widget in enumerate(reflection_widgets):
        reflection_info = reflection_infos[index]
        feedback_widgets.append(
            self.show_reflection_ui(reflection_widget, reflection_info)
        )
      with output_area:
        output_area.clear_output(wait=True)
        # components_py.load_css_styles()
        display(components_py.vbox(
            feedback_widgets,
            align_items='stretch'
        ))

    finish_label_button.on_click(on_finish_click)

    @components_py.with_loading(submit_feedback_button)
    def on_submit_feedback_click(_):
      # Disable the button to prevent multiple clicks.
      submit_feedback_button.disabled = True
      for index, reflection_widget in enumerate(reflection_widgets):
        feedback = reflection_widget['feedback_box'].value.strip()
        image = images_to_reflect[index]
        reflection_info = reflection_infos[index]
        reflection_info['url'] = image.url

        reflection_info['feedback'] = feedback
        image.feedback = feedback

        current_label = reflection_widget['rating_buttons'].value
        user_rating = 5 if current_label == 'In Scope' else 1
        image.user_rating = user_rating
        reflection_info['groundtruth'] = user_rating

        definition.add_annotated_image(image, user_rating)

      # If all classifications are correct,
      # we will skip the refinement step.
      if any(
          not ImageClassifier.determine_correct_rating(
              reflection_info['decision'], reflection_info['groundtruth']
          )
          for reflection_info in reflection_infos
      ):
        not_empty_feedbacks = [
            index
            for index, reflection_info in enumerate(reflection_infos)
            if reflection_info['feedback']
        ]
        mistake_images = [
            index
            for index, reflection_info in enumerate(reflection_infos)
            if not ImageClassifier.determine_correct_rating(
                reflection_info['decision'], reflection_info['groundtruth']
            )
        ]
        if not_empty_feedbacks:
          logger.debug(
              'These images have not empty feedbacks: %s', not_empty_feedbacks
          )
        if mistake_images:
          logger.info('These images have mistakes: %s', mistake_images)
        improved_definition = self.definition_refiner.refine_definition(
            images_to_reflect, definition, reflection_infos
        )
        new_continue_fn = lambda new_definition: continue_fn(
            new_definition, reflection_infos
        )
        self.examine_improvements(
            improved_definition, definition, new_continue_fn
        )
      else:
        continue_fn(definition, reflection_infos)

    finish_label_button.on_click(on_finish_click)
    submit_feedback_button.on_click(on_submit_feedback_click)

    with output_area:
      # Show annotation UIs.
      logger.debug('show annotation UIs')
      output_area.clear_output(wait=True)

      annotation_widgets = []
      for reflection_widget in reflection_widgets:
        radio_buttons_widget = components_py.vbox(
            [
                reflection_widget['ambiguity_widget'],
                reflection_widget['rating_buttons'],
            ],
            align_items='stretch',
        )
        annotation_widget = components_py.hbox(
            [
                reflection_widget['image_widget'],
                radio_buttons_widget,
            ],
            align_items='stretch',
        )
        annotation_widgets.append(annotation_widget)

      display(
          components_py.vbox([
              label_instruction,
              components_py.vbox(annotation_widgets, align_items='center'),
          ])
      )

    with self._output:
      if not self.keep_output:
        clear_output(wait=True)
      components_py.load_css_styles()
      display(components_py.vbox(
          [
              output_area,
              components_py.hbox([finish_label_button, submit_feedback_button])
          ],
          align_items='stretch',
          width='100%',
      ))
