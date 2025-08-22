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

"""Utilities for webdriver."""

import collections
import enum
import re
import time
import io
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from PIL import Image
from PIL import ImageDraw

WIDTH = 160
HEIGHT = 210

# Map to special key, with some fuzzy mapping
SPECIAL_KEY_MAP = {
    'BACKSPACE': Keys.BACKSPACE,
    'BACK SPACE': Keys.BACKSPACE,
    'RIGHTARROW': Keys.ARROW_RIGHT,
    'ARROWRIGHT': Keys.ARROW_RIGHT,
    'LEFTARROW': Keys.ARROW_LEFT,
    'ARROWLEFT': Keys.ARROW_LEFT,
    'DOWNARROW': Keys.ARROW_DOWN,
    'ARROWDOWN': Keys.ARROW_DOWN,
    'UPARROW': Keys.ARROW_UP,
    'ARROWUP': Keys.ARROW_UP,
    'DOWN': Keys.DOWN,
    'UP': Keys.UP,
    'TAB': Keys.TAB,
    'SPACE': Keys.SPACE,
    'ENTER': Keys.ENTER,
}


class STATUS(enum.Enum):
  IN_PROGRESS = 0
  CORRECT = 1
  CYCLE = 2  # cycles to a prev screen
  NO_CHANGE = 3  # no response
  INCOMPLETE = 4  # no further action but task unfinished
  EXCEPTION = 5   # webdriver exception
  FAILED = -1


def get_3x3_pos(width, height, xmin, xmax, ymin, ymax):
  """Converts bbox to an absolute position on the screen (3x3 blocks)."""
  clamp = lambda p: max(0, min(2, p))
  grid_pos = {
      # keyed by (y grid index, x grid index)
      (0, 0): 'top left',
      (0, 1): 'top',
      (0, 2): 'top right',
      (1, 0): 'left',
      (1, 1): 'center',
      (1, 2): 'right',
      (2, 0): 'bottom left',
      (2, 1): 'bottom',
      (2, 2): 'bottom right',
  }
  xcenter = (xmin + xmax) / 2
  ycenter = (ymin + ymax) / 2
  x_denom = width / 3
  y_denom = height / 3
  grid_idx = (clamp(int(ycenter / y_denom)), clamp(int(xcenter / x_denom)))
  return grid_pos[grid_idx]


def flat_html(dom_elements, width=WIDTH, height=HEIGHT):
  """Converts flat DOMElements to HTML."""
  refs = [p.ref for p in dom_elements]
  tags = [p.tag for p in dom_elements]
  classes = [
      p.classes if p.classes and p.classes != 'NO_CLASS' else p.id
      for p in dom_elements
  ]
  placeholders = [p.placeholder for p in dom_elements]
  values = [p.value for p in dom_elements]
  texts = [p.text for p in dom_elements]
  is_leafs = [False if p.children else True for p in dom_elements]
  bboxes = [
      (int(p.left), int(p.top), int(p.right), int(p.bottom))
      for p in dom_elements
  ]
  xmins = [p[0] for p in bboxes]
  ymins = [p[1] for p in bboxes]
  xmaxs = [p[2] for p in bboxes]
  ymaxs = [p[3] for p in bboxes]
  slots = []
  html_attrs = []
  bboxes = []

  for _, (
      ref,
      tag,
      clas,
      placeholder,
      value,
      text,
      is_leaf,
      xmin,
      xmax,
      ymin,
      ymax,
  ) in enumerate(
      zip(
          refs,
          tags,
          classes,
          placeholders,
          values,
          texts,
          is_leafs,
          xmins,
          xmaxs,
          ymins,
          ymaxs,
      )
  ):
    # in some rare cases, ref can be not initialized.
    # assert ref
    if not is_leaf:
      continue
    pos_str = get_3x3_pos(width, height, xmin, xmax, ymin, ymax)
    # print(width, height, xmin, xmax, ymin, ymax, pos_str)
    tag = tag if tag else 'div'
    text = text if text else ''
    clas = clas if clas and clas != 'NO_CLASS' else ''

    slots.append([ref, tag, clas, placeholder, value, is_leaf, pos_str])
    id_str = f'id={ref}' if ref else ''
    class_str = f'class="{clas}"' if clas else ''
    placeholder_str = f'placeholder="{placeholder}"' if placeholder else ''
    value_str = f'value="{value}"' if value else ''
    pos_str = f'pos="{pos_str}"'
    html_attrs.append(
        [id_str, tag, class_str, placeholder_str, value_str, pos_str, text]
    )
    bboxes.append((xmin, xmax, ymin, ymax))

  compact_htmls = []
  compact_ref_slots = []
  compact_bboxes = []
  for _, (slot, attrs, bbox) in enumerate(zip(slots, html_attrs, bboxes)):
    id_str, type_str, text_str = attrs[0], attrs[1], attrs[-1]
    id_str = f'{id_str} ' if id_str else ''
    content = [p for p in attrs[2:-1] if p]
    html = (
        f'<{type_str} {id_str}{" ".join(content)}>{text_str}</{type_str}>'
    )
    compact_htmls.append(html)
    compact_ref_slots.append(slot)
    compact_bboxes.append(bbox)

  htmls = compact_htmls
  ref_slots = compact_ref_slots
  bboxes = compact_bboxes

  return '\n'.join(htmls), ref_slots, bboxes


def disable_clicks_in_html(html, id_set):
  for idx in id_set:
    html = html.replace(f' id={idx}', '')
  return html


def initialize_instance(instance):
  instance.create_driver()
  driver = instance.driver
  driver.execute_script('core.EPISODE_MAX_TIME=3600000;')


def _get_cursor_type(driver, cursor_state):
  """Returns cursor type for focused element, e.g. "pointer", "default", etc."""
  command = (
      'return'
      f' document.elementFromPoint({cursor_state.ptr_x},{cursor_state.ptr_y});'
  )
  element = driver.execute_script(command)
  cursor_type = element.value_of_css_property('cursor')
  return cursor_type


def _click(driver, cursor_state, x, y):
  chain = ActionChains(driver)
  chain.w3c_actions.pointer_action.move_to_location(x, y)
  chain.w3c_actions.pointer_action.click()
  cursor_state.ptr_x = x
  cursor_state.ptr_y = y
  chain.w3c_actions.perform()


def click_by_pos(instance, cursor_state, x, y):
  """Click a screen position."""
  x, y = int(x), int(y)
  _click(instance.driver, cursor_state, x, y)
  # for some unknown reason, ptr can be float sometimes.
  cursor_state.cursor = _get_cursor_type(instance.driver, cursor_state)
  cursor_state.ptr_x = int(cursor_state.ptr_x)
  cursor_state.ptr_y = int(cursor_state.ptr_y)


def click_by_ref_xpath(instance, cursor_state, ref):
  """Click element by xpath-based operation."""
  # TODO: for now just not move cursor (as it's only for debug view).
  del cursor_state
  ele = instance.driver.find_element(By.XPATH, f'//*[@data-wob_ref="{ref}"]')
  # For some reason, ele.click once sometimes is unreliable.
  # ele.click()
  chain = ActionChains(instance.driver)
  chain.move_to_element(ele).click().perform()


def click_by_ref(instance, cursor_state, ref):
  """Click element by ref/id."""
  state = instance.get_state()
  element = next(p for p in state.dom_elements if p.ref == ref)
  xcenter = int(element.left) + element.width // 2
  ycenter = int(element.top) + element.height // 2
  _click(instance.driver, cursor_state, xcenter, ycenter)
  # for some unknown reason, ptr can be float sometimes.
  cursor_state.cursor = _get_cursor_type(instance.driver, cursor_state)
  cursor_state.ptr_x = int(cursor_state.ptr_x)
  cursor_state.ptr_y = int(cursor_state.ptr_y)


def click_by_ref_scroll(instance, cursor_state, ref):
  """Click element by ref/id with implicit scroll if obj is out of bound."""
  element = next(p for p in instance.get_state().dom_elements if p.ref == ref)
  center = (
      (int(element.left) + int(element.right)) // 2,
      (int(element.top) + int(element.bottom)) // 2,
  )
  if center[1] <= 0 or center[1] >= HEIGHT:
    scroll_y(instance, center[1] - HEIGHT // 2)
    pause(instance, duration=0.5)

  click_by_ref(instance, cursor_state, ref)


def enter_by_ref(instance, cursor_state, ref, text):
  del cursor_state
  ele = instance.driver.find_element(By.XPATH, f'//*[@data-wob_ref="{ref}"]')
  ele.click()
  ele.clear()
  pause(instance, duration=0.5)
  keys(instance, text)


def key_press(instance, key):
  """Execute key press."""
  # try to match special key
  key = SPECIAL_KEY_MAP.get(key.upper().replace(' ', ''), key)
  chain = ActionChains(instance.driver)
  chain.send_keys(key).perform()


def keys(instance, text):
  key_press(instance, text)


def key_press_ntimes(instance, key, n):
  """Execute key press for n times."""
  for _ in range(n):
    key_press(instance, key)


def scroll_y(instance, delta):
  get_scrollable_element_fn = 'document.getElementById("area")'
  command = '%s.scrollBy(0,%d);' % (get_scrollable_element_fn, delta)
  instance.driver.execute_script(command)


def do_js(instance, js):
  instance.driver.execute_script(js)


def hold_key_toggle(instance, action):
  """Hold/release special key."""
  action = action.lower()
  toggle = ''
  special_key = ''
  if toggle_match := re.search(r'(hold|release)', action):
    toggle = toggle_match.group()
  if key_match := re.search(r'(ctrl|shift)', action):
    special_key = key_match.group()
  if toggle == 'hold':
    hold_key(instance, special_key)
  elif toggle == 'release':
    release_key(instance, special_key)
  else:
    raise ValueError('unrecognized toggle action in', action)


def hold_key(instance, key):
  key = key.lower()
  key_hold_map = {
      'ctrl': Keys.CONTROL,
      'shift': Keys.SHIFT,
  }
  key_hold = key_hold_map[key]
  chain = ActionChains(instance.driver)
  chain.key_down(key_hold).perform()


def release_key(instance, key):
  key = key.lower()
  key_hold_map = {
      'ctrl': Keys.CONTROL,
      'shift': Keys.SHIFT,
  }
  key_hold = key_hold_map[key]
  chain = ActionChains(instance.driver)
  chain.key_up(key_hold).perform()


def pause(instance, duration=1):
  """pause the environment for duration to wait for responses to flush."""
  del instance
  time.sleep(duration)


def categorize_action(action):
  """Categorize action type."""
  action = action.lower()
  if any(action.startswith(p) for p in ['click', 'tap', 'move']):
    return 'click'
  if any(action.startswith(p) for p in ['press']):
    return 'press'
  if any(action.startswith(p) for p in ['type', 'enter']):
    return 'type'
  if any(action.startswith(p) for p in ['scroll', 'flip']):
    return 'scroll'
  if any(action.startswith(p) for p in ['hold', 'release']):
    return 'hold'
  return ''


def maj_vote_text(text_ls):
  counter = collections.Counter(text_ls)
  maj = sorted([(cnt, text) for text, cnt in counter.items()])[-1][1]
  return maj


def extract_text_to_enter(action):
  """Extracts text to enter from the action."""
  matches = re.findall(r'"(.*?)"', action)
  if matches:
    return matches[0]
  raise ValueError('unrecognized type action', action)


def extract_id_to_enter(action):
  """Extracts element id/ref for enter action."""
  start = action.find('id=')
  if start == -1:
    return -1
  return parse_int(action[start+3:])


def extract_text_to_type(action):
  """Extracts text for enter command without referring to obj idx."""
  matches = re.findall(r'"(.*?)"', action)
  if matches:
    return matches[0]
  patterns = ['enter', 'Enter', 'ENTER', 'type', 'Type', 'TYPE']
  for p in patterns:
    if action.startswith(p):
      return action.split(p)[1].strip()
  raise ValueError('unrecognized type action', action)


def extract_id_to_click(action):
  # assuming this action is a click action
  patterns = ['click', 'Click', 'tap', 'Tap']
  for p in patterns:
    if action.startswith(p):
      return parse_int(action)
  # fallback to bruteforcely get an integer.
  return parse_int(action)


def extract_press_key(action):
  """Extract special key press from action."""
  end_patterns = [',', '.', ';']
  while any(action.endswith(p) for p in end_patterns):
    action = action[:-1]

  beg_patterns = ['press', 'Press', 'PRESS']
  n = 1   # by default, press once.
  for p in beg_patterns:
    if action.startswith(p):
      key = action.split(p)[1].strip()
      # if the action is like "press ARROWDOWN x 4"
      if ' x ' in key:
        key, n = key.split(' x ')[0], key.split(' x ')[1]
        n = int(n)
      return key, n
  raise ValueError('unrecognized press action', action)


def execute_action(instance, cursor_state, action):
  """Execute action."""
  action_type = categorize_action(action)
  if action_type == 'click':
    obj_idx = int(extract_id_to_click(action))
    click_by_ref_xpath(instance, cursor_state, obj_idx)
  elif action_type == 'press':
    key_to_press, n = extract_press_key(action)
    key_press_ntimes(instance, key_to_press, n)
  elif action_type == 'type':
    obj_idx = int(extract_id_to_enter(action))
    if obj_idx != -1:
      text = extract_text_to_enter(action)
      if text is None:
        print('skip type action', action)
        return
      enter_by_ref(instance, cursor_state, obj_idx, text)
    else:
      text = extract_text_to_type(action)
      if text is None:
        print('skip type action', action)
        return
      keys(instance, text)
  elif action_type == 'scroll':
    scroll_y(instance, int(HEIGHT * 0.75))
  elif action_type == 'hold':
    hold_key_toggle(instance, action)


def parse_int(pred):
  # get the first number in whatever kinda string
  matched = re.search(r'\d+', pred)
  if matched:
    return matched.group()
  return 'N/A'


def parse_reflection_action(text):
  matches = re.findall(r'index=(\d+)', text)
  idx = (int(matches[0]) - 1) if matches else -1
  body = ''
  if ', you should ' in text:
    body = text.split(', you should ')[1].strip()
  return idx, body


def take_longest(texts):
  return sorted((len(text), text) for text in texts)[-1][1]


CURSOR_POINTER = "./left_ptr.png"
CURSOR_CROSSHAIR = "./cross.png"

CURSOR_CACHE = {}

def add_cursor(screenshot, cursor_state):
  """Renders a cursor on top of the screenshot."""
  global CURSOR_CACHE

  if cursor_state.cursor == "crosshair":
    file = CURSOR_CROSSHAIR
    x_offset = 10
    y_offset = 9
  else:
    file = CURSOR_POINTER
    # Small offset for cursor graphic.
    x_offset = 3
    y_offset = 4

  if file not in CURSOR_CACHE:
    print("Adding cursor to cache.")
    CURSOR_CACHE[file] = Image.open(open(file, "rb"))
  cursor = CURSOR_CACHE[file]

  screenshot.paste(
      cursor,
      (cursor_state.ptr_x - x_offset, cursor_state.ptr_y - y_offset),
      cursor,
  )
  return screenshot


def png_to_image(image_png):
  return Image.open(io.BytesIO(image_png))


def image_to_png(image):
  bytes_buffer = io.BytesIO()
  image.save(bytes_buffer, "png")
  return bytes_buffer.getvalue()


def crop(screenshot, width, height):
  screenshot = screenshot.crop((0, 0, width, height))
  return screenshot


def augment_screenshot(image, render_marker):
  """Augments the screenshot to display additional information.

  For now, we just optionally render a small red rectangle in the upper right.

  Arguments:
    image: Should be an Image.Image.
    render_marker: Whether to render red square.

  Returns:
    Bytes corresponding to screenshot png with button press indicator.
  """
  if render_marker:
    width, _ = image.size
    draw = ImageDraw.Draw(image)
    # Render a small red rectangle in the upper right corner.
    xy = [(width - 10, 5), (width - 5, 10)]
    # RGB and transparency values from 0 to 255.
    fill_color = (255, 0, 0, 255)
    draw.rectangle(xy=xy, fill=fill_color)
    return image
  else:
    return image

def get_screenshot(driver, cursor_state, width=WIDTH, height=HEIGHT):
  """Returns bytes corresponding to png of screenshot."""
  screenshot = png_to_image(driver.get_screenshot_as_png())
  screenshot = crop(screenshot, width, height)
  screenshot = add_cursor(screenshot, cursor_state)
  screenshot = augment_screenshot(
      screenshot, cursor_state.ptr_down
  )
  return screenshot
