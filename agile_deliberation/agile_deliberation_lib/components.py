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

"""Basic components for interactions in colab."""

import base64
import difflib
import html
import logging
import re
from typing import Any, Callable, Optional, Union

from google.colab import files as colab_files
from IPython.display import display
from IPython.display import HTML
import ipywidgets as widgets

from agile_deliberation.agile_deliberation_lib import definitions as definitions_py
from agile_deliberation.agile_deliberation_lib import image as image_py


logger = logging.getLogger(__name__)

MyImage = image_py.MyImage
Definition = definitions_py.Definition
colors = {
    'linen': 'linen',
    'red': '#f0ced5',
    'green': '#c6dfc8',
    'gray': '#d9d9d9',
    'blue': '#b4d6f0'
}

signal_text_colors = {
    'Positive': '#63d66c',
    'Negative': '#f25c7a',
    'Necessary': '#f2b407',
}

signal_background_colors = {
    'Positive': '#dfede0',
    'Negative': '#f2e4e7',
    'Necessary': '#f7f2e1',
}


def load_css_styles():
  """Load the custom CSS styles for the widgets, especially useful for
  displaying the interactive definition."""
  display(widgets.HTML(f"""
    <style>
      .widget-textarea textarea {{
          background-color: white !important;
          border: none !important;  /* Remove all borders */
          font-size: 14px !important;
      }}
      .feedback-textarea textarea {{
          border: 1px solid gray !important;
      }}
      .widget-text input {{
          background-color: white !important;
          border: none !important;  /* Remove all borders */
          font-weight: bold;
          font-size: 16px !important;
      }}
      .overall-concept,
      .overall-concept .widget-text input {{
          background-color: #cbdaf7!important;
      }}
      .concept-widget {{
          border-bottom: 1px solid gray !important;
      }}
      .necessary-concept,
      .necessary-concept .widget-text input{{
          background-color: {signal_background_colors['Necessary']}!important;
      }}
      .positive-concept,
      .positive-concept .widget-text input {{
          background-color: {signal_background_colors['Positive']}!important;
      }}
      .negative-concept,
      .negative-concept .widget-text input {{
          background-color: {signal_background_colors['Negative']}!important;
      }}
      .light-button {{
          background-color: transparent !important;  /* Remove background color */
          border: none !important;  /* Remove all borders */
          padding: 0 8px 0 8px !important;  /* Remove padding */
      }}
      .light-icon {{
          background-color: transparent !important;  /* Remove background color */
          border: none !important;  /* Remove all borders */
          padding: 0 1px 0 1px !important;  /* Remove padding */
      }}
      .images-gallery-box {{
          overflow: auto !important;
          width: 95vw !important;
      }}
      .image-widget {{
          overflow: visible !important;
      }}
      .selected-button {{
          border: 2px solid #6fbaf9!important;
      }}
      .widget-radio .widget-label {{
          font-size: 15px !important;
          font-weight: bold !important;
          color: #333;
          margin-right: 10px; /* Adjust spacing */
      }}
      .widget-radio-box label {{
          font-size: 15px !important;
          font-weight: bold !important;
          color: #333;
          margin-right: 10px; /* Adjust spacing */
      }}
    </style>
    """))


def display_with_styles(*args, **kwargs):
  """Custom display function that first loads CSS styles.

  Args:
    *args: Arguments to pass to the display function.
    **kwargs: Keyword arguments to pass to the display function.
  """
  # Load custom CSS styles.
  load_css_styles()

  # Call the original display function to show widgets.
  display(*args, **kwargs)


def with_loading(
    button
):
  """Decorator to add loading state to a button.

  Args:
    button: The button widget to apply loading state to.

  Returns:
    The decorator function.
  """
  def decorator(func):
    def wrapper(*args, **kwargs):
      original_label = button.description
      button.description = 'Loading...'
      button.disabled = True

      # Call the original function.
      func(*args, **kwargs)

      # Restore original button label and re-enable it.
      button.description = original_label
      button.disabled = False
    return wrapper
  return decorator


def set_button(label, **kwargs):
  """Create a button widget with custom styles in a consistent way.

  Args:
    label: The text label for the button.
    **kwargs: Additional keyword arguments for styling.

  Returns:
    The created Button widget.
  """
  disabled = kwargs.get('disabled', False)
  button = widgets.Button(description=label, disabled=disabled)
  color_name = kwargs.get('color', 'linen')
  color_code = colors.get(color_name, color_name)
  button.style.button_color = color_code
  button.layout.width = kwargs.get('width', 'auto')
  button.layout.height = kwargs.get('height', '30px')
  button.layout.border = kwargs.get('border', 'none')
  button.layout.visibility = kwargs.get('visibility', 'visible')
  return button


def hbox(items, **kwargs):
  """Create a horizontal box widget with custom styles in a consistent way.

  Args:
    items: A list of widgets to include in the HBox.
    **kwargs: Additional keyword arguments for layout.

  Returns:
    The created HBox widget.
  """
  layout_kwargs = {
      'align_items': 'center',
      'justify_content': 'flex-start',
      'padding': '4px 0 4px 0',
  }
  layout_kwargs.update(kwargs)
  return widgets.HBox(items, layout=widgets.Layout(**layout_kwargs))


def vbox(items, **kwargs):
  """Create a vertical box widget with custom styles in a consistent way.

  Args:
    items: A list of widgets to include in the VBox.
    **kwargs: Additional keyword arguments for layout.

  Returns:
    The created VBox widget.
  """
  layout_kwargs = {
      'align_items': 'flex-start',
      'justify_content': 'flex-start',
      'padding': '4px 0 4px 0',
  }
  layout_kwargs.update(kwargs)
  return widgets.VBox(items, layout=widgets.Layout(**layout_kwargs))


def format_bullet_points(content):
  """Format a string as a bullet point list.

  We use this to format the classifier summary.

  Args:
    content: The string content to format.

  Returns:
    The HTML string representing the bullet point list.
  """
  lines = content.split('\n')
  lines = [line.strip() for line in lines]
  lines = [line for line in lines if line]
  # Wrap each line with <li> to create a bullet point list.
  bullet_points = ''.join(f'<li>{line.strip()}</li>' for line in lines)
  # Wrap the bullet points in an unordered list.
  return f'<ul style="margin: 0; padding-left: 30px;">{bullet_points}</ul>'


def set_instruction(instruction):
  """Display the instruction with custom styles.

  Args:
    instruction: The HTML instruction string.

  Returns:
    The created HTML widget.
  """
  instruct_widget = widgets.HTML(value=f"""
      <div style="
          display: flex;
          flex-direction: column;
          align-items: start;
          justify-content: start;
          text-align: left;
          gap: 0px;
          line-height: 1.5;
      ">
          {instruction}
      </div>""")
  return instruct_widget


def set_image(image, **kwargs):
  """Display an image with custom styles.

  Args:
    image: The MyImage object to display.
    **kwargs: Additional keyword arguments.

  Returns:
    The created HTML widget for the image.
  """
  height = kwargs.get('height', 250)
  border = kwargs.get('border', '')
  encoded_image = base64.b64encode(image.get_image_bytes()).decode('utf-8')
  img_tag = (
      f'<img src="data:image/jpeg;base64,{encoded_image}" height="{height}"'
      f' style="width:auto; border: {border}; border-radius: 8px; display: block;">'
  )
  image_widget = widgets.HTML(img_tag)
  children = [image_widget]
  if hasattr(image, 'ocr_text') and image.ocr_text:
    escaped_ocr = html.escape(image.ocr_text)
    ocr_html_content = f"""
    <div style="max-width: 100%; width: 100%; overflow-wrap: break-word; text-align: left; font-size: 12px; color: #333; background-color: #f9f9f9; padding: 8px; border: 1px solid #eee; margin-top: 5px; border-radius: 4px; box-sizing: border-box;">
        <strong style="display: block; margin-bottom: 4px; color: #007bff;">OCR Text:</strong>
        <pre style="white-space: pre-wrap; margin: 0; font-family: monospace;">{escaped_ocr}</pre>
    </div>
    """
    ocr_widget = widgets.HTML(ocr_html_content)
    ocr_widget.layout.display = 'none'
    ocr_widget.layout.width = '80%'

    # Toggle button
    ocr_button = widgets.Button(
        description='Show/Hide OCR',
        layout=widgets.Layout(width='auto', margin='5px 0 0 0'),
        button_style='',
        tooltip='Toggle OCR text visibility'
    )

    def on_ocr_button_clicked(b):
      if ocr_widget.layout.display == 'none':
        ocr_widget.layout.display = 'block'
      else:
        ocr_widget.layout.display = 'none'

    ocr_button.on_click(on_ocr_button_clicked)

    children.extend([ocr_button, ocr_widget])

  final_widget = vbox(children, min_width='300px')
  final_widget.add_class('image-widget')
  return final_widget


def set_dropdown(options, **kwargs):
  """Return a dropdown widget.

  Args:
    options: The list of options for the dropdown.
    **kwargs: Additional keyword arguments.

  Returns:
    The created Dropdown widget.
  """
  value = kwargs.get('value', None)
  if value is None:
    value = options[0][1]
  label = kwargs.get('label', '')
  width = kwargs.get('width', 'auto')
  layout = widgets.Layout(width=width)
  dropdown_widget = widgets.Dropdown(
      options=options,
      value=value,
      description=label,
      disabled=False,
      layout=layout
  )
  return dropdown_widget


def set_textarea(label, **kwargs):
  """Display a textarea with custom styles.

  Args:
    label: The descriptive label for the textarea.
    **kwargs: Additional keyword arguments.

  Returns:
    The created Textarea widget.
  """
  placeholder = kwargs.get('placeholder', '')
  value = kwargs.get('value', '')
  width = kwargs.get('width', '100%')
  height = kwargs.get('height', '100%')
  input_widget = widgets.Textarea(
      value=value,
      placeholder=placeholder,
      description=label,
      disabled=False,
      layout=widgets.Layout(width=width, height=height)
  )
  return input_widget


def clear_container(containers):
  """Clear the children of a container.

  Args:
    containers: The container or list of containers to clear.
  """
  if isinstance(containers, list):
    for container in containers:
      container.children = []
  else:
    containers.children = []


def upload_image():
  """Upload images from the local device.

  Returns:
    A list of upregulated MyImage objects.
  """
  image_map = colab_files.upload()
  images = []
  for _, image_bytes in image_map.items():
    images.append(MyImage(image_bytes))
  return images


def insert_widgets(
    parent_widget,
    widget_to_insert,
    index,
):
  """Insert a child widget to a parent widget at a specific index.

  Args:
    parent_widget: The parent widget.
    widget_to_insert: The child widget to insert.
    index: The index where the child widget should be inserted.
  """
  if hasattr(parent_widget, 'children'):
    children = list(parent_widget.children)
    children.insert(index, widget_to_insert)
    parent_widget.children = tuple(children)


def display_image_gallery(
    images,
    gallery_widget = None,
    **kwargs,
):
  """Displays a gallery of PIL Images in a horizontal row.

  Args:
      images: A list of PIL Image objects.
      gallery_widget: A widgets.HBox object containing the image widgets.
      **kwargs: Additional arguments to pass to the set_image function.

  Returns:
      A widgets.HBox object containing the image widgets.
  """
  widgets_images = []
  for image in images:
    # Create an Image widget.
    image_widget = set_image(image, **kwargs)
    widgets_images.append(image_widget)
  if gallery_widget is None:
    # Display images in a horizontal box.
    gallery_widget = widgets.HBox(
        widgets_images,
        Layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            align_items='center',
            gap='10px',
            padding='0 0 0 32px',
        ),
    )
    gallery_widget.add_class('images-gallery-box')
  else:
    gallery_widget.children += tuple(widgets_images)
  return gallery_widget


def generate_concepts_list(definition):
  """Generate a list of concepts.

  Args:
    definition: The Definition object.

  Returns:
    A list of tuples containing concept names.
  """
  concepts = [(definition.concept, definition.concept)]
  children_concepts = []
  if definition.necessary_signals:
    for signal in definition.necessary_signals:
      new_concepts = generate_concepts_list(signal)
      new_concepts[0] = (f'🔒{new_concepts[0][0]}', new_concepts[0][1])
      children_concepts.extend(new_concepts)
  if definition.positive_signals:
    for signal in definition.positive_signals:
      new_concepts = generate_concepts_list(signal)
      children_concepts.extend(
          [(f'✅ {label}', value) for label, value in new_concepts]
      )
  if definition.negative_signals:
    for signal in definition.negative_signals:
      new_concepts = generate_concepts_list(signal)
      children_concepts.extend(
          [(f'❌{label}', value) for label, value in new_concepts]
      )
  children_concepts = [
      (f'   {label}', value) for label, value in children_concepts
  ]
  concepts.extend(children_concepts)
  return concepts


def generate_actions_list(
    concept):
  """Generate a list of actions.

  Args:
    concept: The optional Definition object.

  Returns:
    A list of tuples containing action names.
  """
  actions = [('🤖 Let the agent figure out', 'auto')]
  if concept is None:
    return actions
  if concept.positive_signals or concept.negative_signals:
    actions.append(('✅/❌ Add a new signal', 'add-signal'))
  if not concept.signals:
    actions.append(('📝 Change the description', 'change-description'))

  return actions


def edit_signal(
    definition, old_definition = None
):
  """Interaction components for editing a signal.

  Args:
    definition: The current Definition object.
    old_definition: The original Definition object.

  Returns:
    The generated HBox widget.
  """

  def create_update_function(definition, field):
    def update_function(change):
      setattr(definition, field, change['new'].strip())

    return update_function

  # For the concept name.
  concept_widgegt = widgets.Text(
      description='',
      value=definition.concept,
      disabled=False,
      layout=widgets.Layout(width='18%', flex='0 0 auto')
  )
  concept_widgegt.observe(
      create_update_function(definition, 'concept'), names='value'
  )
  # For the concept description.
  desc_widget = widgets.Textarea(
      value=definition.description,
      layout=widgets.Layout(width='77%', flex='1 1 auto'),
  )
  desc_widget.observe(
      create_update_function(definition, 'description'), names='value'
  )
  # For the deletion button.
  delete_button = set_button('✖', flex='0 0 auto', width='5%')
  delete_button.add_class('light-button')

  if definition.signal_type == 'necessary':
    vertical_margin = '8px'
  else:
    vertical_margin = '4px'
  definition_widget = hbox([concept_widgegt, desc_widget], width='100%')
  definition_widget.add_class('concept-widget')
  if definition.signal_type:
    definition_widget.add_class(f'{definition.signal_type}-concept')
  else:
    definition_widget.add_class('overall-concept')

  definition_wrapper_widget = hbox(
      [definition_widget, delete_button],
      width='90%',
      margin=f'{vertical_margin} 0 {vertical_margin} 0',
  )

  def delete_signal():
    clear_container(definition_wrapper_widget)
    logger.info('Delete signal: %s', definition.concept)
    if definition.parent:
      logger.info('actually delete the signal: %s', definition.concept)
      # We do not allow deleting the root concept.
      definition.delete()
  delete_button.on_click(lambda _: delete_signal())
  return definition_wrapper_widget


def create_signal_section(
    signal_type,
    signals,
    add_new_signal_func,
    old_definition = None,
):
  """Create a VBox section with a label and collapsible signals.

  Args:
    signal_type: The type of signal.
    signals: The list of Definition objects.
    add_new_signal_func: Callback function to add a new signal.
    old_definition: Optional original Definition object.

  Returns:
    The generated VBox widget.
  """

  title = signal_type.capitalize()
  section_label = widgets.HTML(f"""
      <b style='color: {signal_text_colors[title]}; font-style: italic; font-weight: bold; font-size: 16px;'>{title} Signals</b>
  """)

  child_widgets = [
      interactive_definition(signal, old_definition) for signal in signals
  ]

  children_box = vbox(
      child_widgets,
      margin='0px 0 0 32px',
      align_items='stretch'
  )

  if signal_type != 'necessary':
    add_signal_button = set_button(
        'Add New signal',
        flex='0 0 auto',
    )

    def add_new_signal():
      new_signal = add_new_signal_func(signal_type)
      new_signal_widget = interactive_definition(new_signal)
      children_box.children += (new_signal_widget,)

    add_signal_button.on_click(lambda _: add_new_signal())
    section_header = hbox(
        [section_label, add_signal_button], width='100%', gap='8px'
    )
  else:
    section_header = section_label

  return vbox(
      [section_header, children_box],
      align_items='stretch',
      padding='8px 0 8px 0',
  )


def interactive_definition(
    definition, old_definition = None
):
  """Function to create a collapsible structure for concepts.

  Args:
    definition: The current Definition object.
    old_definition: The optional original Definition object.

  Returns:
    The generated VBox widget representing the interactive definition.
  """
  # Create parent HTML widget.
  parent_widget = edit_signal(definition, old_definition)

  def add_new_signal(signal_type):
    new_signal = Definition('Name', 'Description')
    definition.update_signals([new_signal], signal_type)
    return new_signal

  if definition.signals:
    children_widgets = []
    if definition.necessary_signals:
      necessary_section = create_signal_section(
          'necessary',
          definition.necessary_signals,
          add_new_signal,
          old_definition,
      )
      children_widgets.append(necessary_section)
    if definition.positive_signals:
      positive_section = create_signal_section(
          'positive',
          definition.positive_signals,
          add_new_signal,
          old_definition,
      )
      children_widgets.append(positive_section)
    if definition.negative_signals:
      negative_section = create_signal_section(
          'negative',
          definition.negative_signals,
          add_new_signal,
          old_definition,
      )
      children_widgets.append(negative_section)
    children_widgets = vbox(children_widgets, align_items='stretch')

    toggle_button = set_button('Expand')

    def toggle_children(_):
      if children_widgets.layout.display == 'none':
        # Show children.
        children_widgets.layout.display = 'block'
        toggle_button.description = 'Collapse'
      else:
        children_widgets.layout.display = 'none'
        toggle_button.description = 'Expand'

    toggle_button.on_click(toggle_children)
    children_widgets.layout.display = 'none'

    return vbox(
        [
            hbox(
                [parent_widget, toggle_button],
                justify_content='space-between',
                flex='1 1 auto',
            ),
            children_widgets,
        ],
        align_items='stretch',
        max_width='1500px',
    )
  else:
    return vbox([parent_widget], width='auto')


def generate_word_diff_html(
    old_str, new_str, inline = True
):
  """Original word-level diff, slightly modified.

  Args:
    old_str: The old string.
    new_str: The new string.
    inline: Whether to display inline.

  Returns:
    The generated HTML diff string.
  """
  if old_str == new_str:
    return html.escape(new_str)

  old_str_safe = html.escape(old_str)
  new_str_safe = html.escape(new_str)
  old_tokens = re.split(r'(\s+)', old_str_safe)
  new_tokens = re.split(r'(\s+)', new_str_safe)

  # Filter out empty strings that re.split might create
  old_tokens = [t for t in old_tokens if t]
  new_tokens = [t for t in new_tokens if t]

  matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens)
  html_output = []
  for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    old_chunk = ''.join(old_tokens[i1:i2])
    new_chunk = ''.join(new_tokens[j1:j2])

    if tag == 'equal':
      html_output.append(new_chunk)
    elif tag == 'insert':
      style = 'background-color: #d4edda; border-radius: 4px; padding: 0 2px;'
      html_output.append(f'<span style="{style}">{new_chunk}</span>')
    elif tag == 'delete':
      style = (
          'background-color: #f8d7da; text-decoration: line-through;'
          ' border-radius: 4px; padding: 0 2px;'
      )
      html_output.append(f'<span style="{style}">{old_chunk}</span>')
    elif tag == 'replace':
      del_style = (
          'background-color: #f8d7da; text-decoration: line-through;'
          ' border-radius: 4px; padding: 0 2px;'
      )
      ins_style = (
          'background-color: #d4edda; border-radius: 4px; padding: 0 2px;'
      )
      html_output.append(f'<span style="{del_style}">{old_chunk}</span>')
      html_output.append(f'<span style="{ins_style}">{new_chunk}</span>')
  return ''.join(html_output)


def generate_improved_diff_html(old_str, new_str):
  """Compares two strings line by line, then word by word within changed lines.

  Args:
    old_str: The old string.
    new_str: The new string.

  Returns:
    The generated HTML diff string.
  """
  old_lines = old_str.splitlines()
  new_lines = new_str.splitlines()
  matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)

  html_output = []

  for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    old_chunk_lines = old_lines[i1:i2]
    new_chunk_lines = new_lines[j1:j2]

    if tag == 'equal':
      for line in new_chunk_lines:
        html_output.append(html.escape(line))
    elif tag == 'insert':
      for line in new_chunk_lines:
        escaped_line = html.escape(line)
        html_output.append(
            f'<div style="background-color: #d4edda;"> {escaped_line}</div>'
        )
    elif tag == 'delete':
      for line in old_chunk_lines:
        escaped_line = html.escape(line)
        html_output.append(
            '<div style="background-color: #f8d7da; text-decoration:'
            f' line-through;"> {escaped_line}</div>'
        )
    elif tag == 'replace':
      # For blocks of lines that are replaced, diff them line by line
      # And for lines that are 1-to-1 replaced, do word diff
      len_old = len(old_chunk_lines)
      len_new = len(new_chunk_lines)
      max_len = max(len_old, len_new)

      for i in range(max_len):
        old_line = old_chunk_lines[i] if i < len_old else None
        new_line = new_chunk_lines[i] if i < len_new else None

        if old_line is not None and new_line is not None:
          # Line exists in both old and new, perform word-level diff
          diffed_line = generate_word_diff_html(old_line, new_line)
          html_output.append(f' {diffed_line}')
        elif new_line is not None:
          # Line only in new (insertion within the replace block)
          escaped_line = html.escape(new_line)
          html_output.append(
              f'<div style="background-color: #d4edda;"> {escaped_line}</div>'
          )
        elif old_line is not None:
          # Line only in old (deletion within the replace block)
          escaped_line = html.escape(old_line)
          html_output.append(
              '<div style="background-color: #f8d7da; text-decoration:'
              f' line-through;"> {escaped_line}</div>'
          )

  # Join with <br> initially to separate inline elements
  almost_final_html = '<br>'.join(html_output)

  # Post-processing to remove <br> immediately following a </div>
  final_html = re.sub(r'(</div>)<br>', r'\1', almost_final_html)

  return final_html


def display_interactive_diff(
    old_definition, new_definition
):
  """Displays the new definition with differences from the old one highlighted.

  Includes a button to switch to a placeholder edit mode.

  Args:
    old_definition: The original definition.
    new_definition: The new definition.
  """
  # State for toggling view.
  # A simple dictionary is used to be mutable in the inner function.
  state = {'edit_mode': False}

  # Create widgets
  toggle_button = widgets.Button(
      description='Edit', button_style='primary', icon='pencil'
  )
  output_area = widgets.Output()

  def show_diff_view():
    """Renders the HTML diff."""
    # diff_html = generate_diff_html(
    #     old_definition.readable_string(), new_definition.readable_string()
    # )
    diff_html = generate_improved_diff_html(
        old_definition.readable_string(), new_definition.readable_string()
    )
    with output_area:
      output_area.clear_output(wait=True)
      # A wrapper div for better styling
      display(HTML(f"""
      <div style="border: 1px solid #ccc; padding: 12px; border-radius: 8px; line-height: 1.3; font-family: sans-serif;">
        {diff_html}
      </div>
      """))

  def show_edit_view():
    """Shows a placeholder for the editing UI."""
    with output_area:
      output_area.clear_output(wait=True)
      load_css_styles()
      display(interactive_definition(new_definition))

  def on_button_clicked(b):
    """Toggle between diff view and edit view."""
    state['edit_mode'] = not state['edit_mode']
    if state['edit_mode']:
      b.description = 'View Diff'
      b.button_style = 'info'
      show_edit_view()
    else:
      b.description = 'Edit'
      b.button_style = 'primary'
      show_diff_view()

  toggle_button.on_click(on_button_clicked)

  # Initial display
  show_diff_view()

  # Display the final composite widget
  display(widgets.VBox([output_area, toggle_button]))

