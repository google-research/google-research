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

# pylint: disable=line-too-long

"""MiniWoB fields."""

import collections
import functools
import json
import os
import re
import sys


# Mapping from task_name to field extractor
# Each extractor is a function that takes the utterance string and
# returns a key-value dict
FIELD_EXTRACTORS = {}


def default_field_extractor(task_name):
  def extractor(utterance):
    try:
      fields = json.loads(utterance)
      return Fields({str(x): str(y) for (x, y) in fields.items()})
    except Exception as exc:
      print(utterance)
      raise ValueError(
          '{} does not have a field extractor.'.format(task_name)
      ) from exc

  return extractor


def get_field_extractor(task_name):
  try:
    return FIELD_EXTRACTORS[task_name]
  except KeyError:
    return default_field_extractor(task_name)


def _add(task_name, regex, keys):
  def extractor(utterance):
    match = re.search(regex, utterance)
    return Fields(dict(list(zip(keys, match.groups()))))

  FIELD_EXTRACTORS[task_name] = extractor


class Fields(object):
  """Wrapper around a dict to make it immutable."""

  def __init__(self, dictionary):
    """Constructor.

    Args:
      dictionary (dict): the wrapped dict.
    """
    self._dictionary = collections.OrderedDict(sorted(dictionary.items()))
    if not self._dictionary:
      # Ensure at least one key to prevent the type prob. from being ignored
      self._dictionary['dummy'] = 'dummy'

  def __getitem__(self, key):
    return self._dictionary[key]

  def __len__(self):
    return len(self._dictionary)

  @property
  def keys(self):
    return list(self._dictionary.keys())

  @property
  def values(self):
    return list(self._dictionary.values())

  def __repr__(self):
    return '\n'.join(
        '{}: {}'.format(k, repr(v)) for k, v in self._dictionary.items()
    )

  __str__ = __repr__


_add(
    'bisect-angle',
    r'Create a line that bisects the angle evenly in two, then press submit\.',
    [],
)

# Book the shortest one-way flight from: Prudhoe Bay/Deadhorse, AK to: EUE on 11/08/2016.
# Book the cheapest one-way flight from: Teller Mission, AK to: MKL on 12/28/2016.
# Book the cheapest one-way flight from: HCR to: SBY on 11/22/2016.
# Book the cheapest one-way flight from: La Crosse, WI to: Austin, TX on 10/21/2016.
# Book the shortest one-way flight from: RBH to: Ponce, Puerto Rico on 12/02/2016.
# Book the cheapest one-way flight from: Iliamna, AK to: CEZ on 12/24/2016.
# Book the cheapest one-way flight from: IMT to: SXP on 12/01/2016.
# Book the cheapest one-way flight from: Shungnak, AK to: Utica, NY on 11/25/2016.
# Book the shortest one-way flight from: Sheldon Point, AK to: SRQ on 11/10/2016.
# Book the shortest one-way flight from: KLW to: FOD on 10/14/2016.
_add(
    'book-flight',
    r'Book the (.*) one-way flight from: (.*) to: (.*) on (.*)\.',
    ['criterion', 'from', 'to', 'date'],
)
FIELD_EXTRACTORS['book-flight-nodelay'] = FIELD_EXTRACTORS['book-flight']

_add(
    'chase-circle',
    r'Keep your mouse inside the circle as it moves around\.',
    [],
)

# Select 12/07/2016 as the date and hit submit.
# Select 12/10/2016 as the date and hit submit.
# Select 03/27/2016 as the date and hit submit.
# Select 05/27/2016 as the date and hit submit.
# Select 01/18/2016 as the date and hit submit.
# Select 08/11/2016 as the date and hit submit.
# Select 12/10/2016 as the date and hit submit.
# Select 04/10/2016 as the date and hit submit.
# Select 08/26/2016 as the date and hit submit.
# Select 12/10/2016 as the date and hit submit.
_add(
    'choose-date',
    r'Select 0*(\d*)/0*(\d*)/0*(\d*) as the date and hit submit\.',
    ['month', 'day', 'year'],
)
FIELD_EXTRACTORS['choose-date-nodelay'] = FIELD_EXTRACTORS[
    'choose-date-easy'
] = FIELD_EXTRACTORS['choose-date-medium'] = FIELD_EXTRACTORS['choose-date']

# Select Qatar from the list and click Submit.
# Select Konstanze from the list and click Submit.
# Select Hedy from the list and click Submit.
# Select Clio from the list and click Submit.
# Select Libya from the list and click Submit.
# Select Darrelle from the list and click Submit.
# Select Togo from the list and click Submit.
# Select Poland from the list and click Submit.
# Select Botswana from the list and click Submit.
# Select Robyn from the list and click Submit.
_add('choose-list', r'Select (.*) from the list and click Submit\.', ['target'])
_add(
    'choose-list-internal',
    r'Select (.*) from the list and click Submit\.',
    ['target'],
)

_add(
    'circle-center',
    r'Find and click on the center of the circle, then press submit\.',
    [],
)

# Click on the "Cancel" button.
# Click on the "yes" button.
# Click on the "Submit" button.
# Click on the "Next" button.
# Click on the "Previous" button.
# Click on the "Cancel" button.
# Click on the "cancel" button.
# Click on the "Ok" button.
# Click on the "ok" button.
# Click on the "no" button.
_add('click-button', r'Click on the "(.*)" button\.', ['target'])
_add('click-button-sequence', r'Click button ONE, then click button TWO\.', [])


# Select nothing and click Submit.
# Select delivering,walked and click Submit.
# Select bono and click Submit.
# Select sunglasses,sumitomo,raja and click Submit.
# Select attended,relieve,published and click Submit.
# Select moore,resign and click Submit.
# Select 1937 and click Submit.
# Select governments and click Submit.
# Select aquarium,output,batsmen,hour and click Submit.
# Select resemble,padres,brooklyn,miller and click Submit.
def extract_click_checkboxes(utterance):
  targets = re.search(r'Select (.*) and click Submit\.', utterance).group(1)
  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['button'] = 'submit'
  return Fields(fields)


FIELD_EXTRACTORS['click-checkboxes'] = extract_click_checkboxes
FIELD_EXTRACTORS['click-checkboxes-large'] = extract_click_checkboxes
FIELD_EXTRACTORS['click-checkboxes-transfer'] = extract_click_checkboxes


# Select words similar to humorous, rabbit, home, slice and click Submit.
# Select words similar to furious, petite and click Submit.
# Select words similar to genuine, chubby and click Submit.
# Select words similar to archaic, enormous and click Submit.
# Select words similar to huge, hate, old, stupid, cut and click Submit.
# Select words similar to finish and click Submit.
# Select words similar to irritated, swine, TVs and click Submit.
# Select words similar to mild, response and click Submit.
# Select words similar to pig and click Submit.
# Select words similar to cheerful, adore and click Submit.
# Select words similar to fires and click Submit.
def extract_click_checkboxes_soft(utterance):
  targets = re.search(
      r'Select words similar to (.*) and click Submit\.', utterance
  ).group(1)
  targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['button'] = 'submit'
  return Fields(fields)


FIELD_EXTRACTORS['click-checkboxes-soft'] = extract_click_checkboxes_soft

_add('click-collapsible', r'Expand the section below and click submit\.', [])
FIELD_EXTRACTORS['click-collapsible-nodelay'] = FIELD_EXTRACTORS[
    'click-collapsible'
]

# Expand the sections below, to find and click on the link "confirming.".
# Expand the sections below, to find and click on the link "opening".
# Expand the sections below, to find and click on the link "Add".
# Expand the sections below, to find and click on the link "nevada".
# Expand the sections below, to find and click on the link "42".
# Expand the sections below, to find and click on the link "ongoing".
# Expand the sections below, to find and click on the link "explanation".
# Expand the sections below, to find and click on the link "Shire.".
# Expand the sections below, to find and click on the link "proliferation".
# Expand the sections below, to find and click on the link "beauty".
_add(
    'click-collapsible-2',
    r'Expand the sections below, to find and click on the link "(.*)"\.',
    ['target'],
)
FIELD_EXTRACTORS['click-collapsible-2-nodelay'] = FIELD_EXTRACTORS[
    'click-collapsible-2'
]

# Click on the colored box.
# Click on the yellow colored box.
# Click on the pink colored box.
# Click on the colored box.
# Click on the colored box.
# Click on the colored box.
# Click on the colored box.
# Click on the colored box.
# Click on the blue colored box.
# Click on the colored box.
_add('click-color', r'Click on the (.*) colored box\.', ['target'])

_add('click-dialog', r'Close the dialog box by clicking the "x"\.', [])

# Click the button in the dialog box labeled "Cancel".
# Click the button in the dialog box labeled "Cancel".
# Click the button in the dialog box labeled "OK".
# Click the button in the dialog box labeled "Cancel".
# Click the button in the dialog box labeled "x".
# Click the button in the dialog box labeled "Cancel".
# Click the button in the dialog box labeled "OK".
# Click the button in the dialog box labeled "Cancel".
# Click the button in the dialog box labeled "Cancel".
# Click the button in the dialog box labeled "Cancel".
_add(
    'click-dialog-2',
    r'Click the button in the dialog box labeled "(.*)"\.',
    ['target'],
)

# Click on the link "nba".
# Click on the link "Domestic".
# Click on the link "tariff".
# Click on the link "salam".
# Click on the link "anti-".
# Click on the link "operation.".
# Click on the link "staffer".
# Click on the link "rao".
# Click on the link "class".
# Click on the link "plummer".
_add('click-link', r'Click on the link "(.*)"\.', ['target'])

# Select Kelli
# Select Vanya
# Select Valma>Kelila>Mercedes
# Select Cathlene
# Select Alisha
# Select Yetta
# Select Wilmette>Lynsey
# Select Juana
# Select Kalila>Bird
# Select Peria>Kata>Caitrin
_add('click-menu', r'Select (.*)', ['target'])

# Click the "Menu" button, and then find and click on the item with the icon.
# Click the "Menu" button, and then find and click on the item with the icon.
# Click the "Menu" button, and then find and click on the item with the icon.
# Click the "Menu" button, and then find and click on the item with the icon.
# Click the "Menu" button, and then find and click on the item with the icon.
# Click the "Menu" button, and then find and click on the item labeled "Prev".
# Click the "Menu" button, and then find and click on the item with the icon.
# Click the "Menu" button, and then find and click on the item labeled "Zoom Out".
# Click the "Menu" button, and then find and click on the item with the icon.
# Click the "Menu" button, and then find and click on the item with the icon.
_add(
    'click-menu-2',
    r'Click the "Menu" button, and then find and click on the item (.*)\.',
    ['target'],
)

# Select D8 and click Submit.
# Select 5qAbzn and click Submit.
# Select g3x09N and click Submit.
# Select qbfXGf and click Submit.
# Select XDyepg and click Submit.
# Select wUGvHai and click Submit.
# Select mL and click Submit.
# Select wtuEd4 and click Submit.
# Select oagd and click Submit.
# Select qGWE and click Submit.
_add('click-option', r'Select (.*) and click Submit\.', ['target'])

# Expand the pie menu below and click on the item labeled "o".
# Expand the pie menu below and click on the item labeled "h".
# Expand the pie menu below and click on the item labeled "h".
# Expand the pie menu below and click on the item labeled "Q".
# Expand the pie menu below and click on the item labeled "U".
# Expand the pie menu below and click on the item labeled "N".
# Expand the pie menu below and click on the item labeled "8".
# Expand the pie menu below and click on the item labeled "3".
# Expand the pie menu below and click on the item labeled "q".
# Expand the pie menu below and click on the item labeled "R".
_add(
    'click-pie',
    r'Expand the pie menu below and click on the item labeled "(.*)"\.',
    ['target'],
)
FIELD_EXTRACTORS['click-pie-nodelay'] = FIELD_EXTRACTORS['click-pie']

# Select Norway, Luxembourg from the scroll list and click Submit.
# Select Bosnia and Herzegovina, Zambia from the scroll list and click Submit.
# Select Ainslie, Daffi from the scroll list and click Submit.
# Select Loree, Ophelie from the scroll list and click Submit.
# Select Flori from the scroll list and click Submit.
# Select Belgium from the scroll list and click Submit.
# Select Botswana from the scroll list and click Submit.
# Select Eadie, Marjy from the scroll list and click Submit.
# Select Latvia from the scroll list and click Submit.
# Select Cocos Islands from the scroll list and click Submit.
_add(
    'click-scroll-list',
    r'Select (.*) from the scroll list and click Submit\.',
    ['target'],
)

# Select all the shades of blue and press Submit.
# Select all the shades of red and press Submit.
# Select all the shades of red and press Submit.
# Select all the shades of blue and press Submit.
# Select all the shades of blue and press Submit.
# Select all the shades of red and press Submit.
# Select all the shades of red and press Submit.
# Select all the shades of green and press Submit.
# Select all the shades of red and press Submit.
# Select all the shades of green and press Submit.
_add(
    'click-shades',
    r'Select all the shades of (.*) and press Submit\.',
    ['target'],
)


# Click on a 0
# Click on a large green digit
# Click on a small blue item
# Click on a large item
# Click on a black x
# Click on a small green letter
# Click on a small item
# Click on a letter
# Click on a circle
# Click on a small red p
def parse_shape_desc(words):
  fields = {}
  for word in words:
    if word in ('large', 'small'):
      fields['size'] = word
    elif word in ('red', 'green', 'blue', 'aqua', 'black', 'magenta', 'yellow'):
      fields['color'] = word
    elif word in ('shape', 'digit', 'letter', 'item'):
      fields['type'] = word
    else:
      fields['target'] = word
  return fields


def extract_click_shape(utterance):
  words = re.search(r'Click on a (.*)', utterance).group(1).split()
  return Fields(parse_shape_desc(words))


FIELD_EXTRACTORS['click-shape'] = extract_click_shape

# Click on Tab #2.
# Click on Tab #2.
# Click on Tab #3.
# Click on Tab #1.
# Click on Tab #2.
# Click on Tab #2.
# Click on Tab #3.
# Click on Tab #3.
# Click on Tab #2.
# Click on Tab #2.
_add('click-tab', r'Click on Tab #(.*)\.', ['target'])

# Switch between the tabs to find and click on the link "retreated".
# Switch between the tabs to find and click on the link "culminating".
# Switch between the tabs to find and click on the link "Spokesperson.".
# Switch between the tabs to find and click on the link "3-1".
# Switch between the tabs to find and click on the link "karachi".
# Switch between the tabs to find and click on the link "Memorable".
# Switch between the tabs to find and click on the link "collegiate.".
# Switch between the tabs to find and click on the link "sections".
# Switch between the tabs to find and click on the link "cahill.".
# Switch between the tabs to find and click on the link "fauna".
_add(
    'click-tab-2',
    r'Switch between the tabs to find and click on the link "(.*)"\.',
    ['target'],
)
FIELD_EXTRACTORS['click-tab-2-easy'] = FIELD_EXTRACTORS[
    'click-tab-2-medium'
] = FIELD_EXTRACTORS['click-tab-2-hard'] = FIELD_EXTRACTORS['click-tab-2']

_add('click-test', r'Click the button\.', [])

_add('click-test-2', r'Click button (.*)\.', ['target'])

_add('click-test-transfer', r'Click button (.*)\.', ['target'])

# Click on a "textarea" widget.
# Click on a "checkbox" widget.
# Click on a "text" widget.
# Click on a "button" widget.
# Click on a "button" widget.
# Click on a "button" widget.
# Click on a "button" widget.
# Click on a "radio" widget.
# Click on a "checkbox" widget.
# Click on a "textarea" widget.
_add('click-widget', r'Click on a "(.*)" widget\.', ['target'])

_add(
    'copy-paste',
    r'Copy the text in the textarea below, paste it into the textbox and press'
    r' Submit\.',
    [],
)

# Copy the text from the 1st text area below and paste it into the text input, then press Submit.
# Copy the text from the 2nd text area below and paste it into the text input, then press Submit.
# Copy the text from the 3rd text area below and paste it into the text input, then press Submit.
# Copy the text from the 1st text area below and paste it into the text input, then press Submit.
# Copy the text from the 3rd text area below and paste it into the text input, then press Submit.
# Copy the text from the 3rd text area below and paste it into the text input, then press Submit.
# Copy the text from the 3rd text area below and paste it into the text input, then press Submit.
# Copy the text from the 1st text area below and paste it into the text input, then press Submit.
# Copy the text from the 2nd text area below and paste it into the text input, then press Submit.
# Copy the text from the 2nd text area below and paste it into the text input, then press Submit.
_add(
    'copy-paste-2',
    r'Copy the text from the (\d+).. text area below and paste it into the text'
    r' input, then press Submit\.',
    ['target'],
)


# How many small aqua items are there?
# How many letters are there?
# How many large items are there?
# How many large 6s are there?
# How many small items are there?
# How many small green items are there?
# How many small rectangles are there?
# How many yellow items are there?
# How many red triangles are there?
# How many small yellow items are there?
def extract_count_shape(utterance):
  words = re.search(r'How many (.*)s are there\?', utterance).group(1).split()
  return Fields(parse_shape_desc(words))


FIELD_EXTRACTORS['count-shape'] = extract_count_shape

_add(
    'count-sides',
    r'Press the button that correctly denotes how many sides the shape has\.',
    [],
)

_add(
    'drag-box',
    r'Drag the smaller box so that it is completely inside the larger box\.',
    [],
)

# Move the cube around so that "2" is the active side facing the user.
# Move the cube around so that "5" is the active side facing the user.
# Move the cube around so that "4" is the active side facing the user.
# Move the cube around so that "1" is the active side facing the user.
# Move the cube around so that "1" is the active side facing the user.
# Move the cube around so that "2" is the active side facing the user.
# Move the cube around so that "3" is the active side facing the user.
# Move the cube around so that "2" is the active side facing the user.
# Move the cube around so that "2" is the active side facing the user.
# Move the cube around so that "3" is the active side facing the user.
_add(
    'drag-cube',
    r'Move the cube around so that "(.*)" is the active side facing the user\.',
    ['target'],
)

# Drag the circle up then press Submit.
# Drag the circle left then press Submit.
# Drag the circle up then press Submit.
# Drag the circle up then press Submit.
# Drag the circle right then press Submit.
# Drag the circle right then press Submit.
# Drag the circle left then press Submit.
# Drag the circle left then press Submit.
# Drag the circle down then press Submit.
# Drag the circle up then press Submit.
_add('drag-item', r'Drag the circle (.*) then press Submit\.', ['target'])

# Drag Lanna to the 5th position.
# Drag Blythe up by one position.
# Drag Tootsie to the top.
# Drag Patrice to the 2nd position.
# Drag Deeann to the 5th position.
# Drag Diane to the 4th position.
# Drag Christiane to the bottom.
# Drag Audra down by one position.
# Drag Bari to the 5th position.
# Drag Anestassia to the bottom.
_add('drag-items', r'Drag (.*)\.', ['target'])

# Drag Evvie to the top right.
# Drag Shell to the bottom center.
# Drag Davita to the bottom left.
# Drag Doroteya left by one.
# Drag Eddi to the bottom left.
# Drag Elnora to the bottom left.
# Drag Francisca left by one.
# Drag Hollie to the top left.
# Drag Kaila up by one.
# Drag Cherry to the top center.
_add('drag-items-grid', r'Drag (.*)\.', ['target'])

# Drag all triangles into the black box.
# Drag all rectangles into the black box.
# Drag all rectangles into the black box.
# Drag all rectangles into the black box.
# Drag all circles into the black box.
# Drag all rectangles into the black box.
# Drag all circles into the black box.
# Drag all circles into the black box.
# Drag all triangles into the black box.
# Drag all triangles into the black box.
_add('drag-shapes', r'Drag all (.*) into the black box\.', ['target'])

_add(
    'drag-sort-numbers',
    r'Sort the numbers in increasing order, starting with the lowest number at'
    r' the top of the list\.',
    [],
)

# Find the email by Cosette and forward that email to Elwira.
# Find the email by Sheba and reply to them with the text "Dar. Twain.".
# Find the email by Olimpia and forward that email to Hendrika.
# Find the email by Milka and click the star icon to mark it as important.
# Find the email by Shaylynn and click the star icon to mark it as important.
# Find the email by Stefa and click the star icon to mark it as important.
# Find the email by Jacklin and click the star icon to mark it as important.
# Find the email by Germaine and reply to them with the text "Highly cruise reproduce agree.".
# Find the email by Leonore and reply to them with the text "Ancient defending.".
# Find the email by Caterina and click the trash icon to delete it.
EMAIL_INBOX_PATTERNS = [
    (
        'delete',
        r'Find the email by (.*) and click the trash icon to (.*) it\.',
        ['by', 'task'],
    ),
    (
        'forward',
        r'Find the email by (.*) and (.*) that email to (.*)\.',
        ['by', 'task', 'to'],
    ),
    (
        'important',
        (
            r'Find the email by (.*) and click the (.*) icon to mark it as'
            r' important\.'
        ),
        ['by', 'task'],
    ),
    (
        'reply',
        r'Find the email by (.*) and (.*) to them with the text "(.*)"\.',
        ['by', 'task', 'message'],
    ),
]


def extract_email_inbox(utterance):
  for _, email_regex, email_keys in EMAIL_INBOX_PATTERNS:
    match = re.search(email_regex, utterance)
    if match:
      return Fields(dict(list(zip(email_keys, match.groups()))))
  raise ValueError('Bad email-inbox utterance: {}'.format(utterance))


for task, e_regex, e_keys in EMAIL_INBOX_PATTERNS:
  _add('email-inbox-' + task, e_regex, e_keys)
FIELD_EXTRACTORS['email-inbox-star-reply'] = FIELD_EXTRACTORS['email-inbox'] = (
    FIELD_EXTRACTORS['email-inbox-reduced']
) = FIELD_EXTRACTORS['email-inbox-noscroll'] = extract_email_inbox


# Natural language version: no fields at test time
def extract_email_inbox_nl(utterance):
  del utterance
  return Fields({})


FIELD_EXTRACTORS['email-inbox-forward-nl'] = FIELD_EXTRACTORS[
    'email-inbox-forward-nl-turk'
] = FIELD_EXTRACTORS['email-inbox-forward-nl-turk-reduced'] = FIELD_EXTRACTORS[
    'email-inbox-nl-turk'
] = FIELD_EXTRACTORS[
    'email-inbox-nl-turk-reduced'
] = extract_email_inbox_nl

# Enter 01/02/2014 as the date and hit submit.
# Enter 05/01/2011 as the date and hit submit.
# Enter 06/20/2016 as the date and hit submit.
# Enter 06/17/2010 as the date and hit submit.
# Enter 07/09/2017 as the date and hit submit.
# Enter 08/22/2010 as the date and hit submit.
# Enter 05/01/2016 as the date and hit submit.
# Enter 01/26/2018 as the date and hit submit.
# Enter 03/15/2018 as the date and hit submit.
# Enter 09/11/2017 as the date and hit submit.
_add('enter-date', r'Enter (.*) as the date and hit submit\.', ['target'])

# Enter the password "KA6" into both text fields and press submit.
# Enter the password "d1u" into both text fields and press submit.
# Enter the password "rT" into both text fields and press submit.
# Enter the password "jsB" into both text fields and press submit.
# Enter the password "u6Rzw" into both text fields and press submit.
# Enter the password "3gu" into both text fields and press submit.
# Enter the password "Q7" into both text fields and press submit.
# Enter the password "Qvx" into both text fields and press submit.
# Enter the password "md" into both text fields and press submit.
# Enter the password "2f" into both text fields and press submit.
_add(
    'enter-password',
    r'Enter the password "(.*)" into both text fields and press submit\.',
    ['target'],
)

# Enter "Donovan" into the text field and press Submit.
# Enter "Rex" into the text field and press Submit.
# Enter "Lyda" into the text field and press Submit.
# Enter "Nathalie" into the text field and press Submit.
# Enter "Macie" into the text field and press Submit.
# Enter "Kasie" into the text field and press Submit.
# Enter "Enola" into the text field and press Submit.
# Enter "Michel" into the text field and press Submit.
# Enter "Emile" into the text field and press Submit.
# Enter "Deneen" into the text field and press Submit.
_add(
    'enter-text',
    r'Enter "(.*)" into the text field and press Submit\.',
    ['target'],
)

# Type "KENETH" in all lower case letters in the text input and press Submit.
# Type "CHAS" in all lower case letters in the text input and press Submit.
# Type "THADDEUS" in all lower case letters in the text input and press Submit.
# Type "CHEREE" in all lower case letters in the text input and press Submit.
# Type "KARRIE" in all lower case letters in the text input and press Submit.
# Type "JOYE" in all lower case letters in the text input and press Submit.
# Type "ANNIS" in all lower case letters in the text input and press Submit.
# Type "KASIE" in all lower case letters in the text input and press Submit.
# Type "enola" in all upper case letters in the text input and press Submit.
# Type "JOYE" in all lower case letters in the text input and press Submit.
_add(
    'enter-text-2',
    r'Type "(.*)" in all (.*) case letters in the text input and press'
    r' Submit\.',
    ['text', 'case'],
)

# Enter "LQosL" into the text field and press Submit.
# Enter "83" into the text field and press Submit.
# Enter "HPzD" into the text field and press Submit.
# Enter "qRBcu" into the text field and press Submit.
# Enter "M46t" into the text field and press Submit.
# Enter "s8Y" into the text field and press Submit.
# Enter "3em" into the text field and press Submit.
# Enter "JJSF" into the text field and press Submit.
# Enter "jHbNS" into the text field and press Submit.
# Enter "8" into the text field and press Submit.
_add(
    'enter-text-dynamic',
    r'Enter "(.*)" into the text field and press Submit\.',
    ['target'],
)


# Enter 3:57 AM as the time and press submit.
# Enter 8:16 PM as the time and press submit.
# Enter 9:01 AM as the time and press submit.
# Enter 8:18 AM as the time and press submit.
# Enter 5:15 AM as the time and press submit.
# Enter 9:21 AM as the time and press submit.
# Enter 8:47 AM as the time and press submit.
# Enter 10:05 PM as the time and press submit.
# Enter 9:26 AM as the time and press submit.
# Enter 12:25 AM as the time and press submit.
def extract_enter_time(utterance):
  target = re.search(
      r'Enter (.*) as the time and press submit\.', utterance
  ).group(1)
  return Fields({'target': target.replace(' ', '')})


FIELD_EXTRACTORS['enter-time'] = extract_enter_time
FIELD_EXTRACTORS['enter-time-internal'] = extract_enter_time

_add(
    'find-midpoint',
    r'Find and click on the shortest mid-point between the two points, then'
    r' press submit\.',
    [],
)

# Find the 7th word in the paragraph, type that into the textbox and press "Submit".
# Find the 8th word in the paragraph, type that into the textbox and press "Submit".
# Find the 4th word in the paragraph, type that into the textbox and press "Submit".
# Find the 8th word in the paragraph, type that into the textbox and press "Submit".
# Find the 4th word in the paragraph, type that into the textbox and press "Submit".
# Find the 5th word in the paragraph, type that into the textbox and press "Submit".
# Find the 5th word in the paragraph, type that into the textbox and press "Submit".
# Find the 10th word in the paragraph, type that into the textbox and press "Submit".
# Find the 7th word in the paragraph, type that into the textbox and press "Submit".
# Find the 9th word in the paragraph, type that into the textbox and press "Submit".
_add(
    'find-word',
    r'Find the (\d+).. word in the paragraph, type that into the textbox and'
    r' press "Submit"\.',
    ['target'],
)

_add('focus-text', r'Focus into the textbox\.', [])

# Focus into the 1st input textbox.
# Focus into the 3rd input textbox.
# Focus into the 1st input textbox.
# Focus into the 3rd input textbox.
# Focus into the 2nd input textbox.
# Focus into the 1st input textbox.
# Focus into the 1st input textbox.
# Focus into the 3rd input textbox.
# Focus into the 2nd input textbox.
# Focus into the 1st input textbox.
_add('focus-text-2', r'Focus into the (\d+).. input textbox\.', ['target'])

# Click on the grid coordinate (-1,-1).
# Click on the grid coordinate (-1,0).
# Click on the grid coordinate (-2,2).
# Click on the grid coordinate (1,2).
# Click on the grid coordinate (2,1).
# Click on the grid coordinate (1,2).
# Click on the grid coordinate (0,2).
# Click on the grid coordinate (1,0).
# Click on the grid coordinate (-1,1).
# Click on the grid coordinate (-1,2).
_add(
    'grid-coordinate',
    r'Click on the grid coordinate \((.*),(.*)\)\.',
    ['x', 'y'],
)

_add(
    'guess-number',
    r'Guess the number between 0-9 and press Submit\. Use the feedback below to'
    r' find the right number\.',
    [],
)

_add(
    'highlight-text',
    r'Highlight the text in the paragraph below and click submit\.',
    [],
)

# Highlight the text in the 2nd paragraph and click submit.
# Highlight the text in the 3rd paragraph and click submit.
# Highlight the text in the 2nd paragraph and click submit.
# Highlight the text in the 1st paragraph and click submit.
# Highlight the text in the 3rd paragraph and click submit.
# Highlight the text in the 1st paragraph and click submit.
# Highlight the text in the 2nd paragraph and click submit.
# Highlight the text in the 2nd paragraph and click submit.
# Highlight the text in the 3rd paragraph and click submit.
# Highlight the text in the 3rd paragraph and click submit.
_add(
    'highlight-text-2',
    r'Highlight the text in the (\d+).. paragraph and click submit\.',
    ['target'],
)

_add(
    'identify-shape',
    r'Click the button that best describes the figure below\.',
    [],
)

# Enter the username "kanesha" and the password "DRbGP" into the text fields and press login.
# Enter the username "jess" and the password "S2" into the text fields and press login.
# Enter the username "joye" and the password "A48ui" into the text fields and press login.
# Enter the username "tula" and the password "ET" into the text fields and press login.
# Enter the username "beaulah" and the password "RJ3" into the text fields and press login.
# Enter the username "vanda" and the password "iygs" into the text fields and press login.
# Enter the username "keli" and the password "W6PX" into the text fields and press login.
# Enter the username "nathalie" and the password "gbyS0" into the text fields and press login.
# Enter the username "beaulah" and the password "OV5" into the text fields and press login.
# Enter the username "nathalie" and the password "MN" into the text fields and press login.
_add(
    'login-user',
    r'Enter the username "(.*)" and the password "(.*)" into the text fields'
    r' and press login\.',
    ['username', 'password'],
)
FIELD_EXTRACTORS['login-user-popup'] = FIELD_EXTRACTORS['login-user']

_add('moving-items', r'Click as many moving circles as possible\.', [])

# Search for paranoid movies directed by David from year 1982.
# Search for political movies directed by Mcneil from year 1997.
# Search for political movies directed by Schroeder from year 1996.
# Search for political movies directed by Harrell from year 1975.
# Search for action movies directed by Manning from year 1998.
# Search for thriller movies directed by Spence from year 2009.
# Search for saga movies directed by Rosario from year 2006.
# Search for paranoid movies directed by Mitchell from year 1975.
# Search for fantasy movies directed by Short from year 1993.
# Search for thriller movies directed by Stein from year 1991.
_add(
    'multi-layouts',
    r'Search for (.*) movies directed by (.*) from year (.*)\.',
    ['genre', 'director', 'year'],
)
FIELD_EXTRACTORS['multi-orderings'] = FIELD_EXTRACTORS['multi-layouts']

# Navigate through the file tree. Find and click on the folder or file named "Nieves".
# Navigate through the file tree. Find and click on the folder or file named "Truman".
# Navigate through the file tree. Find and click on the folder or file named "Alan".
# Navigate through the file tree. Find and click on the folder or file named "Rex".
# Navigate through the file tree. Find and click on the folder or file named "Ignacio".
# Navigate through the file tree. Find and click on the folder or file named "Jerald".
# Navigate through the file tree. Find and click on the folder or file named "Kanesha".
# Navigate through the file tree. Find and click on the folder or file named "Cierra".
# Navigate through the file tree. Find and click on the folder or file named "Ashlea".
# Navigate through the file tree. Find and click on the folder or file named "Kasie".
_add(
    'navigate-tree',
    r'Navigate through the file tree\. Find and click on the folder or file'
    r' named "(.*)"\.',
    ['target'],
)

# Draw the number "9" in the checkboxes using the example on the right and press Submit when finished.
# Draw the number "0" in the checkboxes using the example on the right and press Submit when finished.
# Draw the number "8" in the checkboxes using the example on the right and press Submit when finished.
# Draw the number "2" in the checkboxes using the example on the right and press Submit when finished.
# Draw the number "3" in the checkboxes using the example on the right and press Submit when finished.
# Draw the number "7" in the checkboxes using the example on the right and press Submit when finished.
# Draw the number "9" in the checkboxes using the example on the right and press Submit when finished.
# Draw the number "5" in the checkboxes using the example on the right and press Submit when finished.
# Draw the number "6" in the checkboxes using the example on the right and press Submit when finished.
# Draw the number "9" in the checkboxes using the example on the right and press Submit when finished.
_add(
    'number-checkboxes',
    r'Draw the number "(.*)" in the checkboxes using the example on the right'
    r' and press Submit when finished\.',
    ['target'],
)

# Enter the value of Religion into the text field and press Submit.
# Enter the value of Country into the text field and press Submit.
# Enter the value of Gender into the text field and press Submit.
# Enter the value of First name into the text field and press Submit.
# Enter the value of Language into the text field and press Submit.
# Enter the value of Color into the text field and press Submit.
# Enter the value of Country into the text field and press Submit.
# Enter the value of Gender into the text field and press Submit.
# Enter the value of Country into the text field and press Submit.
# Enter the value of Religion into the text field and press Submit.
_add(
    'read-table',
    r'Enter the value of (.*) into the text field and press Submit\.',
    ['target'],
)

_add(
    'read-table-2',
    r'Enter the value that corresponds with each label into the form and submit'
    r' when done\.',
    [],
)

# Resize the textarea so that the height is larger than its initial size then press Submit.
# Resize the textarea so that the width is smaller than its initial size then press Submit.
# Resize the textarea so that the height is larger than its initial size then press Submit.
# Resize the textarea so that the width is smaller than its initial size then press Submit.
# Resize the textarea so that the height is smaller than its initial size then press Submit.
# Resize the textarea so that the height is smaller than its initial size then press Submit.
# Resize the textarea so that the height is smaller than its initial size then press Submit.
# Resize the textarea so that the width is larger than its initial size then press Submit.
# Resize the textarea so that the width is smaller than its initial size then press Submit.
# Resize the textarea so that the height is larger than its initial size then press Submit.
_add(
    'resize-textarea',
    r'Resize the textarea so that the (.*) than its initial size then press'
    r' Submit\.',
    ['target'],
)

_add(
    'right-angle',
    r'Add a third point to create a right angle, then press submit\.',
    [],
)

_add(
    'scroll-text',
    r'Find the last word in the text area, enter it into the text field and hit'
    r' Submit\.',
    [],
)

# Scroll the textarea to the top of the text hit submit.
# Scroll the textarea to the top of the text hit submit.
# Scroll the textarea to the bottom of the text hit submit.
# Scroll the textarea to the bottom of the text hit submit.
# Scroll the textarea to the bottom of the text hit submit.
# Scroll the textarea to the top of the text hit submit.
# Scroll the textarea to the top of the text hit submit.
# Scroll the textarea to the top of the text hit submit.
# Scroll the textarea to the top of the text hit submit.
# Scroll the textarea to the bottom of the text hit submit.
_add(
    'scroll-text-2',
    r'Scroll the textarea to the (.*) of the text hit submit\.',
    ['target'],
)

# Use the textbox to enter "Tora" and press "Search", then find and click the 9th search result.
# Use the textbox to enter "Ignacio" and press "Search", then find and click the 8th search result.
# Use the textbox to enter "Jess" and press "Search", then find and click the 2nd search result.
# Use the textbox to enter "Marcella" and press "Search", then find and click the 8th search result.
# Use the textbox to enter "Livia" and press "Search", then find and click the 6th search result.
# Use the textbox to enter "Michel" and press "Search", then find and click the 7th search result.
# Use the textbox to enter "Michel" and press "Search", then find and click the 1st search result.
# Use the textbox to enter "Lyda" and press "Search", then find and click the 6th search result.
# Use the textbox to enter "Annis" and press "Search", then find and click the 5th search result.
# Use the textbox to enter "Sergio" and press "Search", then find and click the 4th search result.
_add(
    'search-engine',
    r'Use the textbox to enter "(.*)" and press "Search", then find and click'
    r' the (\d+).. search result\.',
    ['query', 'rank'],
)

_add('simon-says', r'Push the buttons in the order displayed\.', [])

_add(
    'simple-algebra',
    r'Solve for x and type your answer into the textbox\. Press Submit when'
    r' done\.',
    [],
)

_add(
    'simple-arithmetic',
    r'Solve the math problem and type your answer into the textbox\. Press'
    r' submit when done\.',
    [],
)

# For the user @jess, click on the "Block" button.
# For the user @qaeda, click on the "Reply" button.
# For the user @canadian, click on the "Like" button.
# For the user @keneth, click on the "Share via DM" button.
# For the user @arbitration, click on the "Reply" button.
# For the user @fighter, click on the "Reply" button.
# For the user @castles, click on the "Share via DM" button.
# For the user @lies, click on the "Share via DM" button.
# For the user @renda, click on the "Reply" button.
# For the user @equity, click on the "Block" button.
_add(
    'social-media',
    r'For the user (.*), click on the "(.*)" button\.',
    ['user', 'button'],
)
_add(
    'social-media-reduced',
    r'For the user (.*), click on the "(.*)" button\.',
    ['user', 'button'],
)

# Click the "Like" button on all posts by @nieves and then click Submit.
# Click the "Retweet" button on all posts by @cierra and then click Submit.
# Click the "Retweet" button on all posts by @a and then click Submit.
# Click the "Retweet" button on all posts by @keli and then click Submit.
# Click the "Like" button on all posts by @natoque and then click Submit.
# Click the "Share" button on all posts by @mi and then click Submit.
# Click the "Like" button on all posts by @cristin and then click Submit.
# Click the "Reply" button on all posts by @kanesha and then click Submit.
# Click the "Like" button on all posts by @karrie and then click Submit.
# Click the "Reply" button on all posts by @ultricies and then click Submit.
_add(
    'social-media-all',
    r'Click the "(.*)" button on all posts by (.*) and then click (Submit)\.',
    ['button', 'user', 'submit'],
)
_add(
    'social-media-all-reduced',
    r'Click the "(.*)" button on all posts by (.*) and then click (Submit)\.',
    ['button', 'user', 'submit'],
)

# Click the "Retweet" button on 3 posts by @etiam and then click Submit.
# Click the "Retweet" button on 1 post by @leonie and then click Submit.
# Click the "Retweet" button on 2 posts by @michel and then click Submit.
# Click the "Like" button on 3 posts by @kasie and then click Submit.
# Click the "Retweet" button on 2 posts by @karrie and then click Submit.
# Click the "Like" button on 1 post by @renda and then click Submit.
# Click the "Reply" button on 4 posts by @jess and then click Submit.
# Click the "Share" button on 1 post by @nullam and then click Submit.
# Click the "Retweet" button on 1 post by @karrie and then click Submit.
# Click the "Share" button on 2 posts by @vina and then click Submit.
_add(
    'social-media-some',
    r'Click the "(.*)" button on (.*) posts? by (.*) and then click (Submit)\.',
    ['button', 'amount', 'user', 'submit'],
)
_add(
    'social-media-some-reduced',
    r'Click the "(.*)" button on (.*) posts? by (.*) and then click (Submit)\.',
    ['button', 'amount', 'user', 'submit'],
)

# Use the terminal below to delete a file ending with the extension .gif
# Use the terminal below to delete a file ending with the extension .zip
# Use the terminal below to delete a file ending with the extension .tar.gz
# Use the terminal below to delete a file ending with the extension .gif
# Use the terminal below to delete a file ending with the extension .gif
# Use the terminal below to delete a file that has no file extension.
# Use the terminal below to delete a file ending with the extension .zip
# Use the terminal below to delete a file ending with the extension .gif
# Use the terminal below to delete a file ending with the extension .gpg
# Use the terminal below to delete a file ending with the extension .py
_add('terminal', r'Use the terminal below to delete a file (.*)', ['target'])

# Using the text editor, make everything the color red and press Submit.
# Using the text editor, give everything the style underlined and press Submit.
# Using the text editor, give everything the style italics and press Submit.
# Using the text editor, give everything the style underlined and press Submit.
# Using the text editor, give everything the style italics and press Submit.
# Using the text editor, give the text remarkably. the style italics and press Submit.
# Using the text editor, give the text Swaziland. the style italics and press Submit.
# Using the text editor, give everything the style underlined and press Submit.
# Using the text editor, give the text Sniper. the color yellow.
# Using the text editor, give the text 41 the color orange.
_add('text-editor', r'Using the text editor, (.*)\.', ['target'])

_add(
    'text-transform',
    r'Type the text below into the text field and press Submit\.',
    [],
)

_add('tic-tac-toe', r"Playing as 'X', win a game of tic-tac-toe\.", [])

_add('unicode-test', r'Click on the "(.*)" button\.', ['target'])


# Enter an item that starts with "Mart" and ends with "ique".
# Enter an item that starts with "Ni" and ends with "er".
# Enter an item that starts with "Bel" and ends with "ize".
# Enter an item that starts with "Tr" and ends with "bago".
# Enter an item that starts with "Bur" and ends with "Faso".
# Enter an item that starts with "Hong" and ends with "Kong".
# Enter an item that starts with "Sur".
# Enter an item that starts with "Para".
# Enter an item that starts with "Bahr" and ends with "rain".
# Enter an item that starts with "Tanz" and ends with "ia".
def extract_use_autocomplete(utterance):
  match = re.search(
      r'Enter an item that starts with "([^"]*)" and ends with "([^"]*)"\.',
      utterance,
  )
  if match:
    return Fields({'start': match.group(1), 'end': match.group(2)})
  else:
    match = re.search(r'Enter an item that starts with "([^"]*)"', utterance)
    return Fields({'start': match.group(1)})


FIELD_EXTRACTORS['use-autocomplete'] = extract_use_autocomplete
FIELD_EXTRACTORS['use-autocomplete-nodelay'] = extract_use_autocomplete

# Select gray with the color picker and hit Submit.
# Select white with the color picker and hit Submit.
# Select lime with the color picker and hit Submit.
# Select gray with the color picker and hit Submit.
# Select red with the color picker and hit Submit.
# Select magenta with the color picker and hit Submit.
# Select green with the color picker and hit Submit.
# Select white with the color picker and hit Submit.
# Select navy with the color picker and hit Submit.
# Select white with the color picker and hit Submit.
_add(
    'use-colorwheel',
    r'Select (.*) with the color picker and hit Submit\.',
    ['target'],
)

_add(
    'use-colorwheel-2',
    r'Select the following color with the color picker and hit Submit\.',
    [],
)

# Select 9 with the slider and hit Submit.
# Select 11 with the slider and hit Submit.
# Select 102 with the slider and hit Submit.
# Select 109 with the slider and hit Submit.
# Select -37 with the slider and hit Submit.
# Select -64 with the slider and hit Submit.
# Select 105 with the slider and hit Submit.
# Select 35 with the slider and hit Submit.
# Select 20 with the slider and hit Submit.
# Select 7 with the slider and hit Submit.
_add('use-slider', r'Select (.*) with the slider and hit Submit\.', ['target'])

# Set the sliders to the combination [0,14,0] and submit.
# Set the sliders to the combination [14,3,2] and submit.
# Set the sliders to the combination [20,14,0] and submit.
# Set the sliders to the combination [15,18,3] and submit.
# Set the sliders to the combination [18,7,18] and submit.
# Set the sliders to the combination [14,15,3] and submit.
# Set the sliders to the combination [0,20,0] and submit.
# Set the sliders to the combination [11,5,5] and submit.
# Set the sliders to the combination [8,3,16] and submit.
# Set the sliders to the combination [18,20,15] and submit.
_add(
    'use-slider-2',
    r'Set the sliders to the combination \[(.*),(.*),(.*)\] and submit\.',
    ['n1', 'n2', 'n3'],
)

# Select 5 with the spinner and hit Submit.
# Select -10 with the spinner and hit Submit.
# Select -2 with the spinner and hit Submit.
# Select 5 with the spinner and hit Submit.
# Select 7 with the spinner and hit Submit.
# Select -7 with the spinner and hit Submit.
# Select 8 with the spinner and hit Submit.
# Select -3 with the spinner and hit Submit.
# Select 5 with the spinner and hit Submit.
# Select -4 with the spinner and hit Submit.
_add(
    'use-spinner', r'Select (.*) with the spinner and hit Submit\.', ['target']
)

_add(
    'visual-addition',
    r'Type the total number of blocks into the textbox and press Submit\.',
    [],
)


def extract_flight_subtasks(utterance):
  fields = json.loads(utterance)
  return Fields({str(x): str(y) for (x, y) in fields.items()})


FIELD_EXTRACTORS['flight.AA'] = extract_flight_subtasks
FIELD_EXTRACTORS['flight.Alaska'] = extract_flight_subtasks
FIELD_EXTRACTORS['flight.Alaska-auto-medium'] = extract_flight_subtasks
FIELD_EXTRACTORS['flight.Alaska-auto'] = extract_flight_subtasks
FIELD_EXTRACTORS['flight.Delta'] = extract_flight_subtasks
FIELD_EXTRACTORS['flight.JetBlue'] = extract_flight_subtasks
FIELD_EXTRACTORS['flight.United'] = extract_flight_subtasks


# for compositional tasks
def extract_click_button_click_checkboxes(utterance, order=0, reverse=False):
  """Extract fields from instruction."""
  if reverse:
    if order == 0:
      rematch = re.search(
          r'Select (.*) and click Submit, after clicking on the "(.*)"'
          r' button\.',
          utterance,
      )
      button_text = rematch.group(2)
      targets = rematch.group(1)
    else:
      rematch = re.search(
          r'Click on the "(.*)" button, after selecting (.*) and clicking'
          r' Submit\.',
          utterance,
      )
      button_text = rematch.group(1)
      targets = rematch.group(2)
  else:
    if order == 0:
      rematch = re.search(
          r'Click on the "(.*)" button, and then select (.*) and click'
          r' Submit\.',
          utterance,
      )
      button_text = rematch.group(1)
      targets = rematch.group(2)
    else:
      rematch = re.search(
          r'Select (.*) and click Submit, and then click on the "(.*)"'
          r' button\.',
          utterance,
      )
      button_text = rematch.group(2)
      targets = rematch.group(1)

  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['button'] = 'submit'
  fields['target'] = button_text
  return Fields(fields)


def extract_click_button_sequence_click_checkboxes(utterance, reverse=False):
  """Extract fields from instruction."""
  if reverse:
    targets = re.search(
        r'Select (.*) and click Submit, after clicking button ONE, then'
        r' clicking button TWO.',
        utterance,
    ).group(1)
  else:
    targets = re.search(
        r'Click button ONE, then click button TWO, and then select (.*) and'
        r' click Submit\.',
        utterance,
    ).group(1)
  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['button'] = 'submit'
  return Fields(fields)


def extract_click_link_click_button_click_checkboxes_click_dialog(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Select (.*) and click Submit, and close the dialog box by clicking'
        r' the "x", after clicking on the link "(.*)", and clicking on the'
        r' "(.*)" button.',
        utterance,
    )
    link_text = rematch.group(2)
    button_text = rematch.group(3)
    targets_check = rematch.group(1)
  else:
    rematch = re.search(
        r'Click on the link "(.*)", click on the "(.*)" button, select (.*) and'
        r' click Submit, and then close the dialog box by clicking the "x".',
        utterance,
    )
    link_text = rematch.group(1)
    button_text = rematch.group(2)
    targets_check = rematch.group(3)

  if targets_check == 'nothing':
    targets_check = []
  else:
    targets_check = re.split(', ?', targets_check)
  fields = dict(
      list(
          zip(
              ['target {}'.format(i) for i in range(len(targets_check))],
              targets_check,
          )
      )
  )
  fields['button'] = 'submit'
  fields['target'] = button_text
  fields[f'target {len(targets_check)+1}'] = link_text
  return Fields(fields)


# click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog
def extract_click_link_click_button_click_checkboxes_click_option_click_dialog(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Select (.*), select (.*) and click Submit, and close the dialog box'
        r' by clicking the "x", after clicking on the link "(.*)", and clicking'
        r' on the "(.*)" button.',
        utterance,
    )
    link_text = rematch.group(3)
    button_text = rematch.group(4)
    targets_check = rematch.group(1)
    option_text = rematch.group(2)
  else:
    rematch = re.search(
        r'Click on the link "(.*)", click on the "(.*)" button, select (.*),'
        r' select (.*) and click Submit, and then close the dialog box by'
        r' clicking the "x".',
        utterance,
    )
    link_text = rematch.group(1)
    button_text = rematch.group(2)
    targets_check = rematch.group(3)
    option_text = rematch.group(4)

  if targets_check == 'nothing':
    targets_check = []
  else:
    targets_check = re.split(', ?', targets_check)
  fields = dict(
      list(
          zip(
              ['target {}'.format(i) for i in range(len(targets_check))],
              targets_check,
          )
      )
  )
  fields['button'] = 'submit'
  fields['target'] = button_text
  fields[f'target {len(targets_check)+1}'] = link_text
  fields[f'target {len(targets_check)+2}'] = option_text
  return Fields(fields)


def extract_click_widget_click_link_click_button_click_checkboxes_click_option_click_dialog(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Select (.*), select (.*) and click Submit, and close the dialog box'
        r' by clicking the "x", after clicking on a "(.*)" widget, clicking on'
        r' the link "(.*)", and clicking on the "(.*)" button.',
        utterance,
    )
    widget_text = rematch.group(3)
    link_text = rematch.group(4)
    button_text = rematch.group(5)
    targets_check = rematch.group(1)
    option_text = rematch.group(2)
  else:
    rematch = re.search(
        r'Click on a "(.*)" widget, click on the link "(.*)", click on the'
        r' "(.*)" button, select (.*), select (.*) and click Submit, and then'
        r' close the dialog box by clicking the "x".',
        utterance,
    )
    widget_text = rematch.group(1)
    link_text = rematch.group(2)
    button_text = rematch.group(3)
    targets_check = rematch.group(4)
    option_text = rematch.group(5)

  if targets_check == 'nothing':
    targets_check = []
  else:
    targets_check = re.split(', ?', targets_check)
  fields = dict(
      list(
          zip(
              ['target {}'.format(i) for i in range(len(targets_check))],
              targets_check,
          )
      )
  )
  fields['button'] = 'submit'
  fields['target'] = button_text
  fields[f'target {len(targets_check)+1}'] = link_text
  fields[f'target {len(targets_check)+2}'] = option_text
  fields[f'target {len(targets_check)+3}'] = widget_text
  return Fields(fields)


def extract_click_button_sequence_click_widget_click_link_click_button_click_checkboxes_click_option_click_dialog(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Select (.*), select (.*) and click Submit, and close the dialog box'
        r' by clicking the "x", after clicking button ONE, then clicking button'
        r' TWO, clicking on a "(.*)" widget, clicking on the link "(.*)", and'
        r' clicking on the "(.*)" button.',
        utterance,
    )
    widget_text = rematch.group(3)
    link_text = rematch.group(4)
    button_text = rematch.group(5)
    targets_check = rematch.group(1)
    option_text = rematch.group(2)
  else:
    rematch = re.search(
        r'Click button ONE, then click button TWO, click on a "(.*)" widget,'
        r' click on the link "(.*)", click on the "(.*)" button, select (.*),'
        r' select (.*) and click Submit, and then close the dialog box by'
        r' clicking the "x".',
        utterance,
    )
    widget_text = rematch.group(1)
    link_text = rematch.group(2)
    button_text = rematch.group(3)
    targets_check = rematch.group(4)
    option_text = rematch.group(5)

  if targets_check == 'nothing':
    targets_check = []
  else:
    targets_check = re.split(', ?', targets_check)
  fields = dict(
      list(
          zip(
              ['target {}'.format(i) for i in range(len(targets_check))],
              targets_check,
          )
      )
  )
  fields['button'] = 'submit'
  fields['target'] = button_text
  fields[f'target {len(targets_check)+1}'] = link_text
  fields[f'target {len(targets_check)+2}'] = option_text
  fields[f'target {len(targets_check)+3}'] = widget_text
  return Fields(fields)


def extract_click_button_sequence_click_widget_click_link_click_button_click_checkboxes_click_option_click_dialog_login_user(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Select (.*), select (.*) and click Submit, close the dialog box by'
        r' clicking the "x", and enter the username "(.*)" and the password'
        r' "(.*)" into the text fields and press login, after clicking button'
        r' ONE, then clicking button TWO, clicking on a "(.*)" widget, clicking'
        r' on the link "(.*)", and clicking on the "(.*)" button.',
        utterance,
    )
    widget_text = rematch.group(3)
    link_text = rematch.group(4)
    button_text = rematch.group(5)
    targets_check = rematch.group(1)
    option_text = rematch.group(2)
  else:
    rematch = re.search(
        r'Click button ONE, then click button TWO, click on a "(.*)" widget,'
        r' click on the link "(.*)", click on the "(.*)" button, select (.*),'
        r' select (.*) and click Submit, close the dialog box by clicking the'
        r' "x", and then enter the username "(.*)" and the password "(.*)" into'
        r' the text fields and press login.',
        utterance,
    )
    widget_text = rematch.group(1)
    link_text = rematch.group(2)
    button_text = rematch.group(3)
    targets_check = rematch.group(4)
    option_text = rematch.group(5)

  if targets_check == 'nothing':
    targets_check = []
  else:
    targets_check = re.split(', ?', targets_check)
  fields = dict(
      list(
          zip(
              ['target {}'.format(i) for i in range(len(targets_check))],
              targets_check,
          )
      )
  )
  fields['button'] = 'submit'
  fields['target'] = button_text
  fields[f'target {len(targets_check)+1}'] = link_text
  fields[f'target {len(targets_check)+2}'] = option_text
  fields[f'target {len(targets_check)+3}'] = widget_text
  return Fields(fields)


def extract_click_button_enter_text(utterance, order=0):
  """Extract fields from instruction."""
  if order == 0:
    rematch = re.search(
        r'Click on the "(.*)" button, and then enter "(.*)" into the text field'
        r' and press Submit.',
        utterance,
    )
    button_text = rematch.group(1)
    targets = rematch.group(2)
  else:
    rematch = re.search(
        r'Enter "(.*)" into the text field and press Submit, and then click on'
        r' the "(.*)" button.',
        utterance,
    )
    button_text = rematch.group(2)
    targets = rematch.group(1)

  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['button'] = 'submit'
  fields['target'] = button_text
  return Fields(fields)


def extract_click_link_enter_text(utterance, order=0, reverse=False):
  """Extract fields from instruction."""
  if reverse:
    if order == 0:
      rematch = re.search(
          r'Enter "(.*)" into the text field and press Submit, after clicking'
          r' on the link "(.*)"',
          utterance,
      )
      link_text = rematch.group(2)
      targets = rematch.group(1)
    else:
      rematch = re.search(
          r'Click on the link "(.*)", after entering "(.*)" into the text field'
          r' and pressing Submit',
          utterance,
      )
      link_text = rematch.group(1)
      targets = rematch.group(2)
  else:
    if order == 0:
      rematch = re.search(
          r'Click on the link "(.*)", and then enter "(.*)" into the text field'
          r' and press Submit.',
          utterance,
      )
      link_text = rematch.group(1)
      targets = rematch.group(2)
    else:
      rematch = re.search(
          r'Enter "(.*)" into the text field and press Submit, and then click'
          r' on the link "(.*)".',
          utterance,
      )
      link_text = rematch.group(2)
      targets = rematch.group(1)

  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['button'] = 'submit'
  fields['target'] = link_text
  return Fields(fields)


def extract_click_checkboxes_multi_layouts(utterance, reverse=False):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'.*selecting (.*), and searching for (.*) movies directed by (.*) from'
        r' year (.*)',
        utterance,
    )
    targets = rematch.group(1)
    genre = rematch.group(2)
    director = rematch.group(3)
    year = rematch.group(4)
  else:
    rematch = re.search(
        r'Select (.*), search for (.*) movies directed by (.*) from year (.*)',
        utterance,
    )
    targets = rematch.group(1)
    genre = rematch.group(2)
    director = rematch.group(3)
    year = rematch.group(4)

  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['genre'] = genre
  fields['director'] = director
  fields['year'] = year
  return Fields(fields)


def extract_enter_password_click_checkboxes_login_user_popup(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Select (.*), and enter the username "(.*)" and the password "(.*)"'
        r' into the text fields and press login, after entering the password'
        r' "(.*)" into both text fields and pressing submit',
        utterance,
    )
    passoword1 = rematch.group(4)
    targets = rematch.group(1)
    username = rematch.group(2)
    passoword2 = rematch.group(3)
  else:
    rematch = re.search(
        r'Enter the password "(.*)" into both text fields and press submit,'
        r' select (.*), and then enter the username "(.*)" and the password'
        r' "(.*)" into the text fields and press login',
        utterance,
    )
    passoword1 = rematch.group(1)
    targets = rematch.group(2)
    username = rematch.group(3)
    passoword2 = rematch.group(4)

  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['passoword1'] = passoword1
  fields['username'] = username
  fields['passoword2'] = passoword2
  return Fields(fields)


def extract_click_checkboxes_transfer_click_button_sequence_enter_password(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Click button ONE, then click button TWO, and enter the password'
        r' "(.*)" into both text fields and press login, after selecting (.*)'
        r' and clicking Submit',
        utterance,
    )
    targets = rematch.group(2)
    passoword = rematch.group(1)
  else:
    rematch = re.search(
        r'Select (.*) and click Submit, click button ONE, then click button'
        r' TWO, and then enter the password "(.*)" into both text fields and'
        r' press login',
        utterance,
    )
    targets = rematch.group(1)
    passoword = rematch.group(2)

  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['passoword'] = passoword
  return Fields(fields)


# click-checkboxes_click-widget_click-button-sequence
def extract_click_checkboxes_click_widget_click_button_sequence(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Click on a "(.*)" widget, and click button ONE, then click button'
        r' TWO, after selecting (.*) and clicking Submit',
        utterance,
    )
    targets = rematch.group(2)
    widget = rematch.group(1)
  else:
    rematch = re.search(
        r'Select (.*) and click Submit, click on a "(.*)" widget', utterance
    )
    targets = rematch.group(1)
    widget = rematch.group(2)

  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['widget'] = widget
  return Fields(fields)


# click-dialog_click-checkboxes-transfer_click-widget
def extract_click_dialog_click_checkboxes_transfer_click_widget(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Select (.*), and click on a "(.*)" widget, after closing the dialog'
        r' box by clicking the "x"',
        utterance,
    )
  else:
    rematch = re.search(
        r'Close the dialog box by clicking the "x", select (.*) and click'
        r' Submit, and then click on a "(.*)" widget',
        utterance,
    )
  targets = rematch.group(1)
  widget = rematch.group(2)

  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['widget'] = widget
  return Fields(fields)


# click-checkboxes-transfer_enter-password_click-dialog
def extract_click_checkboxes_transfer_enter_password_click_dialog(
    utterance, reverse=False
):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Enter the password "(.*)" into both text fields and press login, and'
        r' close the dialog box by clicking the "x", after selecting (.*)',
        utterance,
    )
  else:
    rematch = re.search(
        r'Select (.*), enter the password "(.*)" into both text fields and'
        r' press login, and then close the dialog box by clicking the "x"',
        utterance,
    )
  targets = rematch.group(1)
  password = rematch.group(2)

  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)
  fields = dict(
      list(zip(['target {}'.format(i) for i in range(len(targets))], targets))
  )
  fields['password'] = password
  return Fields(fields)


# click-checkboxes-soft_enter-password
def extract_click_checkboxes_soft_enter_password(utterance, reverse=False):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Enter the password "(.*)" into both text fields and press submit,'
        r' after selecting words similar to (.*)',
        utterance,
    )
    targets = rematch.group(2)
    password = rematch.group(1)
  else:
    rematch = re.search(
        r'Select words similar to (.*), and then enter the password "(.*)" into'
        r' both text fields and press submit',
        utterance,
    )
    targets = rematch.group(1)
    password = rematch.group(2)
  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)

  fields = dict(
      zip(['target {}'.format(i) for i in range(len(targets))], targets)
  )
  fields['password'] = password
  return Fields(fields)


# click-checkboxes-soft_multi-layouts
def extract_click_checkboxes_soft_multi_layouts(utterance, reverse=False):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Search for (.*) movies directed by (.*) from year (.*), after'
        r' selecting words similar to (.*)',
        utterance,
    )
    genre = rematch.group(1)
    director = rematch.group(2)
    year = rematch.group(3)
    targets = rematch.group(4)
  else:
    rematch = re.search(
        r'Select words similar to (.*), and then search for (.*) movies'
        r' directed by (.*) from year (.*)',
        utterance,
    )
    genre = rematch.group(2)
    director = rematch.group(3)
    year = rematch.group(4)
    targets = rematch.group(1)
  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)

  fields = dict(
      zip(['target {}'.format(i) for i in range(len(targets))], targets)
  )
  fields['genre'] = genre
  fields['director'] = director
  fields['year'] = year
  return Fields(fields)


# click-widget_click-checkboxes-soft
def extract_click_widget_click_checkboxes_soft(utterance, reverse=False):
  """Extract fields from instruction."""
  if reverse:
    rematch = re.search(
        r'Select words similar to (.*) and click Submit, after clicking on a'
        r' "(.*)" widget',
        utterance,
    )
    targets = rematch.group(1)
    widget = rematch.group(2)
  else:
    rematch = re.search(
        r'Click on a "(.*)" widget, and then select words similar to (.*) and'
        r' click Submit',
        utterance,
    )
    targets = rematch.group(2)
    widget = rematch.group(1)
  if targets == 'nothing':
    targets = []
  else:
    targets = re.split(', ?', targets)

  fields = dict(
      zip(['target {}'.format(i) for i in range(len(targets))], targets)
  )
  fields['target'] = widget
  return Fields(fields)


# click-button-sequence_use-autocomplete
def extract_click_button_sequence_use_autocomplete(utterance, reverse=False):
  """Extract fields from instruction."""
  if reverse:
    match = re.search(
        r'Enter an item that starts with "([^"]*)" and ends with "([^"]*)",'
        r' after clicking button ONE, then clicking button TWO',
        utterance,
    )
    if match:
      return Fields({'start': match.group(1), 'end': match.group(2)})
    else:
      match = re.search(
          r'Enter an item that starts with "([^"]*)", after clicking button'
          r' ONE, then clicking button TWO',
          utterance,
      )
      return Fields({'start': match.group(1)})
  else:
    match = re.search(
        r'Click button ONE, then click button TWO, and then enter an item that'
        r' starts with "([^"]*)" and ends with "([^"]*)"',
        utterance,
    )
    if match:
      return Fields({'start': match.group(1), 'end': match.group(2)})
    else:
      match = re.search(
          r'Click button ONE, then click button TWO, and then enter an item'
          r' that starts with "([^"]*)"',
          utterance,
      )
      return Fields({'start': match.group(1)})


# use-autocomplete_click-dialog
def extract_use_autocomplete_click_dialog(utterance, reverse=False):
  """Extract fields from instruction."""
  if reverse:
    match = re.search(
        r'Close the dialog box by clicking the "x", after entering an item that'
        r' starts with "([^"]*)" and ends with "([^"]*)"',
        utterance,
    )
    if match:
      return Fields({'start': match.group(1), 'end': match.group(2)})
    else:
      match = re.search(
          r'Close the dialog box by clicking the "x", after entering an item'
          r' that starts with "([^"]*)"',
          utterance,
      )
      return Fields({'start': match.group(1)})
  else:
    match = re.search(
        r'Enter an item that starts with "([^"]*)" and ends with "([^"]*)", and'
        r' then close the dialog box by clicking the "x"',
        utterance,
    )
    if match:
      return Fields({'start': match.group(1), 'end': match.group(2)})
    else:
      match = re.search(
          r'Enter an item that starts with "([^"]*)", and then close the dialog'
          r' box by clicking the "x"',
          utterance,
      )
      return Fields({'start': match.group(1)})


# dummy operations ('target' will be overridden in each other)
_add(
    'click-button_click-dialog',
    r'Click on the "(.*)" button, and then close the dialog box by clicking the'
    r' "x".',
    ['target'],
)
_add(
    'click-button_click-link',
    r'Click on the "(.*)" button, and then click on the link "(.*)".',
    ['target 0', 'target 1'],
)
_add(
    'click-link_click-button',
    r'Click on the link "(.*)", and then click on the "(.*)" button.',
    ['target 0', 'target 1'],
)
_add(
    'click-link_click-button_click-dialog',
    r'Click on the link "(.*)", click on the "(.*)" button, and then close the'
    r' dialog box by clicking the "x".',
    ['target 0', 'target 1'],
)
_add(
    'click-link_click-dialog',
    r'Click on the link "(.*)", and then close the dialog box by clicking the'
    r' "x".',
    ['target'],
)
_add(
    'click-button_click-option',
    r'Click on the "(.*)" button, and then select (.*) and click Submit\.',
    ['target 0', 'target 1'],
)
_add(
    'click-button-sequence_click-option',
    r'Click button ONE, then click button TWO, and then select (.*) and click'
    r' Submit\.',
    ['target'],
)
_add(
    'click-option_enter-text',
    r'Select (.*), and then enter "(.*)" into the text field and press Submit.',
    ['target 0', 'target 1'],
)
_add(
    'click-option_login-user',
    r'Select (.*), and then enter the username "(.*)" and the password "(.*)"'
    r' into the text fields and press login.',
    ['target', 'username', 'password'],
)
_add(
    'enter-password_click-option',
    r'Enter the password "(.*)" into both text fields, and then select (.*) and'
    r' click Submit',
    ['target 0', 'target 1'],
)
_add(
    'multi-layouts_login-user',
    r'Search for (.*) movies directed by (.*) from year (.*), and then enter'
    r' the username "(.*)" and the password "(.*)" into the text fields and'
    r' press login\.',
    ['genre', 'director', 'year', 'username', 'password'],
)
_add(
    'click-link_click-widget',
    r'Click on the link "(.*)", and then click on a "(.*)" widget\.',
    ['target 0', 'target 1'],
)
_add(
    'click-option_navigate-tree',
    r'Select (.*) and click Submit, and then navigate through the file tree\.'
    r' Find and click on the folder or file named "(.*)"\.',
    ['target 0', 'target 1'],
)
_add(
    'login-user_navigate-tree',
    r'Enter the username "(.*)" and the password "(.*)" into the text fields'
    r' and press login, and then navigate through the file tree\. Find and'
    r' click on the folder or file named "(.*)"\.',
    ['username', 'password', 'target'],
)
_add(
    'click-button-sequence_login-user-popup',
    r'Click button ONE, then click button TWO, and then enter the username'
    r' "(.*)" and the password "(.*)" into the text fields and press login\.',
    ['username', 'password'],
)
_add(
    'click-widget_enter-password',
    r'Click on a "(.*)" widget, and then enter the password "(.*)" into both'
    r' text fields and press submit\.',
    ['target 0', 'target 1'],
)
_add(
    'click-widget_multi-layouts',
    r'Click on a "(.*)" widget, and then search for (.*) movies directed by'
    r' (.*) from year (.*)\.',
    ['target', 'genre', 'director', 'year'],
)
_add(
    'login-user-popup_email-inbox-forward-nl-turk-transition',
    r'Enter the username "(.*)" and the password "(.*)" into the text fields'
    r' and press login',
    ['username', 'password'],
)
_add(
    'click-option_multi-layouts_click-widget_login-user-popup-transition',
    r'Select (.*), search for (.*) movies directed by (.*) from year (.*),'
    r' click on a "(.*)" widget, and then enter the username "(.*)" and the'
    r' password "(.*)" into the text fields and press login',
    [
        'target 0',
        'genre',
        'director',
        'year',
        'target 1',
        'username',
        'password',
    ],
)
_add(
    'click-widget_click-option_click-dialog',
    r'Click on a "(.*)" widget, select (.*) and click Submit, and then close'
    r' the dialog box by clicking the "x"',
    ['target 0', 'target 1'],
)
_add(
    'click-button-sequence_click-option_login-user',
    r'Click button ONE, then click button TWO, select (.*), and then enter the'
    r' username "(.*)" and the password "(.*)" into the text fields and press'
    r' login',
    ['target', 'username', 'password'],
)
_add(
    'click-dialog_click-button-sequence_enter-password',
    r'Close the dialog box by clicking the "x", click button ONE, then click'
    r' button TWO, and then enter the password "(.*)" into both text fields and'
    r' press submit',
    ['password'],
)
_add(
    'click-button_click-option_login-user',
    r'Click on the "(.*)" button, select (.*), and then enter the username'
    r' "(.*)" and the password "(.*)" into the text fields and press login',
    ['target 0', 'target 1', 'username', 'password'],
)
_add(
    'click-button_click-tab-2-hard',
    r'Click on the "(.*)" button, and then switch between the tabs to find and'
    r' click on the link "(.*)"',
    ['target 0', 'target 1'],
)
_add(
    'click-dialog-2_login-user-popup',
    r'Click the button in the dialog box labeled "(.*)", and then enter the'
    r' username "(.*)" and the password "(.*)" into the text fields and press'
    r' login',
    ['target', 'username', 'password'],
)
_add(
    'click-dialog_search-engine',
    r'Close the dialog box by clicking the "x", and then use the textbox to'
    r' enter "(.*)" and press "Search", then find and click the (\d+).. search'
    r' result',
    ['query', 'rank'],
)
_add(
    'enter-date_login-user',
    r'Enter (.*) as the date, and then enter the username "(.*)" and the'
    r' password "(.*)" into the text fields and press login',
    ['target', 'password'],
)
_add(
    'click-dialog-2_click-widget',
    r'Click the button in the dialog box labeled "(.*)", and then click on a'
    r' "(.*)" widget',
    ['target 0', 'target 1'],
)


# reverse order instruction
_add(
    'click-button_click-dialog-reverse',
    r'Close the dialog box by clicking the "x", after clicking on the "(.*)"'
    r' button.',
    ['target'],
)
_add(
    'click-button_click-link-reverse',
    r'Click on the link "(.*)", after clicking on the "(.*)" button.',
    ['target 0', 'target 1'],
)
_add(
    'click-link_click-button-reverse',
    r'Click on the "(.*)" button, after clicking on the link "(.*)".',
    ['target 0', 'target 1'],
)
_add(
    'click-link_click-button_click-dialog-reverse',
    r'Close the dialog box by clicking the "x", after clicking on the link'
    r' "(.*)" and then clicking on the "(.*)" button.',
    ['target 0', 'target 1'],
)
_add(
    'click-link_click-dialog-reverse',
    r'Close the dialog box by clicking the "x", after clicking on the link'
    r' "(.*)".',
    ['target'],
)
_add(
    'click-button_click-option-reverse',
    r'Select (.*) and click Submit, after clicking on the "(.*)" button\.',
    ['target 0', 'target 1'],
)
_add(
    'click-button-sequence_click-option-reverse',
    r'Select (.*) and click Submit, after clicking button ONE, then clicking'
    r' button TWO\.',
    ['target'],
)
_add(
    'click-option_enter-text-reverse',
    r'Enter "(.*)" into the text field and press Submit, after selecting (.*).',
    ['target 0', 'target 1'],
)
_add(
    'click-option_login-user-reverse',
    r'Enter the username "(.*)" and the password "(.*)" into the text fields'
    r' and press login, after selecting (.*).',
    ['username', 'password', 'target'],
)
_add(
    'enter-password_click-option-reverse',
    r'Select (.*) and click Submit, after entering the password "(.*)" into'
    r' both text fields',
    ['target 0', 'target 1'],
)
_add(
    'multi-layouts_login-user-reverse',
    r'Enter the username "(.*)" and the password "(.*)" into the text fields'
    r' and press login, after searching for (.*) movies directed by (.*) from'
    r' year (.*)\.',
    ['username', 'password', 'genre', 'director', 'year'],
)
_add(
    'click-link_click-widget-reverse',
    r'Click on a "(.*)" widget, after clicking on the link "(.*)".',
    ['target 0', 'target 1'],
)
_add(
    'click-option_navigate-tree-reverse',
    r'Navigate through the file tree\. Find and click on the folder or file'
    r' named "(.*)", after selecting (.*) and clicking Submit',
    ['target 0', 'target 1'],
)
_add(
    'login-user_navigate-tree-reverse',
    r'Navigate through the file tree\. Find and click on the folder or file'
    r' named "(.*)", after entering the username "(.*)" and the password "(.*)"'
    r' into the text fields and pressing login',
    ['username', 'password', 'target'],
)
_add(
    'click-button-sequence_login-user-popup-reverse',
    r'Enter the username "(.*)" and the password "(.*)" into the text fields'
    r' and press login, after clicking button ONE, then clicking button TWO\.',
    ['username', 'password'],
)
_add(
    'click-widget_enter-password-reverse',
    r'Enter the password "(.*)" into both text fields and press submit, after'
    r' clicking on a "(.*)" widget\.',
    ['target 0', 'target 1'],
)
_add(
    'click-widget_multi-layouts-reverse',
    r'Search for (.*) movies directed by (.*) from year (.*), after clicking on'
    r' a "(.*)" widget\.',
    ['genre', 'director', 'year', 'target'],
)
_add(
    'login-user-popup_email-inbox-forward-nl-turk-transition-reverse',
    r'.*entering the username "(.*)" and the password "(.*)" into the text'
    r' fields and pressing login',
    ['username', 'password'],
)
_add(
    'click-option_multi-layouts_click-widget_login-user-popup-transition-reverse',
    r'Click on a "(.*)" widget, and enter the username "(.*)" and the password'
    r' "(.*)" into the text fields and press login, after selecting (.*), and'
    r' searching for (.*) movies directed by (.*) from year (.*)',
    [
        'target 1',
        'username',
        'password',
        'target 0',
        'genre',
        'director',
        'year',
    ],
)
_add(
    'click-widget_click-option_click-dialog-reverse',
    r'Select (.*) and click Submit, and close the dialog box by clicking the'
    r' "x", after clicking on a "(.*)" widget',
    ['target 0', 'target 1'],
)
_add(
    'click-button-sequence_click-option_login-user-reverse',
    r'Select (.*), and enter the username "(.*)" and the password "(.*)" into'
    r' the text fields and press login, after clicking button ONE, then'
    r' clicking button TWO',
    ['target', 'username', 'password'],
)
_add(
    'click-dialog_click-button-sequence_enter-password-reverse',
    r'Click button ONE, then click button TWO, and enter the password "(.*)"'
    r' into both text fields and press submit, after closing the dialog box by'
    r' clicking the "x"',
    ['password'],
)
_add(
    'click-button_click-option_login-user-reverse',
    r'Select (.*), and enter the username "(.*)" and the password "(.*)" into'
    r' the text fields and press login, after clicking on the "(.*)" button',
    ['target 1', 'username', 'password', 'target 0'],
)
_add(
    'click-button_click-tab-2-hard-reverse',
    r'Switch between the tabs to find and click on the link "(.*)", after'
    r' clicking on the "(.*)" button',
    ['target 0', 'target 1'],
)
_add(
    'click-dialog-2_login-user-popup-reverse',
    r'Enter the username "(.*)" and the password "(.*)" into the text fields'
    r' and press login, after clicking the button in the dialog box labeled'
    r' "(.*)"',
    ['username', 'password', 'target'],
)
_add(
    'click-dialog_search-engine-reverse',
    r'Use the textbox to enter "(.*)" and press "Search", then find and click'
    r' the (\d+).. search result, after closing the dialog box by clicking'
    r' the "x"',
    ['query', 'rank'],
)
_add(
    'enter-date_login-user-reverse',
    r'Enter the username "(.*)" and the password "(.*)" into the text fields'
    r' and press login, after entering (.*) as the date',
    ['target', 'password'],
)
_add(
    'click-dialog-2_click-widget-reverse',
    r'Click on a "(.*)" widget, after clicking the button in the dialog box'
    r' labeled "(.*)"',
    ['target 1', 'target 0'],
)


FIELD_EXTRACTORS['click-button_click-checkboxes'] = functools.partial(
    extract_click_button_click_checkboxes, order=0, reverse=False
)
FIELD_EXTRACTORS['click-link_enter-text'] = functools.partial(
    extract_click_link_enter_text, order=0, reverse=False
)
FIELD_EXTRACTORS['click-link_click-button_click-checkboxes_click-dialog'] = (
    functools.partial(
        extract_click_link_click_button_click_checkboxes_click_dialog,
        reverse=False,
    )
)
FIELD_EXTRACTORS[
    'click-link_click-button_click-checkboxes_click-option_click-dialog'
] = functools.partial(
    extract_click_link_click_button_click_checkboxes_click_option_click_dialog,
    reverse=False,
)
FIELD_EXTRACTORS['click-button_click-checkboxes-transfer'] = functools.partial(
    extract_click_button_click_checkboxes, order=0, reverse=False
)
FIELD_EXTRACTORS['click-button-sequence_click-checkboxes'] = functools.partial(
    extract_click_button_sequence_click_checkboxes, reverse=False
)
FIELD_EXTRACTORS[
    'click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog'
] = functools.partial(
    extract_click_widget_click_link_click_button_click_checkboxes_click_option_click_dialog,
    reverse=False,
)
FIELD_EXTRACTORS[
    'click-button-sequence_click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog'
] = functools.partial(
    extract_click_button_sequence_click_widget_click_link_click_button_click_checkboxes_click_option_click_dialog,
    reverse=False,
)
FIELD_EXTRACTORS[
    'click-button-sequence_click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog_login-user'
] = functools.partial(
    extract_click_button_sequence_click_widget_click_link_click_button_click_checkboxes_click_option_click_dialog_login_user,
    reverse=False,
)
FIELD_EXTRACTORS['login-user_navigate-tree-transition'] = FIELD_EXTRACTORS[
    'login-user_navigate-tree'
]
FIELD_EXTRACTORS['click-option_login-user-transition'] = FIELD_EXTRACTORS[
    'click-option_login-user'
]
FIELD_EXTRACTORS[
    'click-checkboxes-transfer_multi-layouts_email-inbox-forward-nl-transition'
] = functools.partial(extract_click_checkboxes_multi_layouts, reverse=False)
FIELD_EXTRACTORS['enter-password_click-checkboxes_login-user-popup'] = (
    functools.partial(
        extract_enter_password_click_checkboxes_login_user_popup, reverse=False
    )
)
FIELD_EXTRACTORS[
    'click-checkboxes-transfer_click-button-sequence_enter-password'
] = functools.partial(
    extract_click_checkboxes_transfer_click_button_sequence_enter_password,
    reverse=False,
)
FIELD_EXTRACTORS['click-checkboxes_click-widget_click-button-sequence'] = (
    functools.partial(
        extract_click_checkboxes_click_widget_click_button_sequence,
        reverse=False,
    )
)
FIELD_EXTRACTORS['click-dialog_click-checkboxes-transfer_click-widget'] = (
    functools.partial(
        extract_click_dialog_click_checkboxes_transfer_click_widget,
        reverse=False,
    )
)
FIELD_EXTRACTORS['click-checkboxes-transfer_enter-password_click-dialog'] = (
    functools.partial(
        extract_click_checkboxes_transfer_enter_password_click_dialog,
        reverse=False,
    )
)
FIELD_EXTRACTORS['click-checkboxes-soft_enter-password'] = functools.partial(
    extract_click_checkboxes_soft_enter_password, reverse=False
)
FIELD_EXTRACTORS['click-checkboxes-soft_multi-layouts'] = functools.partial(
    extract_click_checkboxes_soft_multi_layouts, reverse=False
)
FIELD_EXTRACTORS['click-widget_click-checkboxes-soft'] = functools.partial(
    extract_click_widget_click_checkboxes_soft, reverse=False
)
FIELD_EXTRACTORS['click-button-sequence_use-autocomplete'] = functools.partial(
    extract_click_button_sequence_use_autocomplete, reverse=False
)
FIELD_EXTRACTORS['use-autocomplete_click-dialog'] = functools.partial(
    extract_use_autocomplete_click_dialog, reverse=False
)


FIELD_EXTRACTORS['click-button_click-checkboxes-reverse'] = functools.partial(
    extract_click_button_click_checkboxes, order=0, reverse=True
)
FIELD_EXTRACTORS['click-link_enter-text-reverse'] = functools.partial(
    extract_click_link_enter_text, order=0, reverse=True
)
FIELD_EXTRACTORS[
    'click-link_click-button_click-checkboxes_click-dialog-reverse'
] = functools.partial(
    extract_click_link_click_button_click_checkboxes_click_dialog, reverse=True
)
FIELD_EXTRACTORS[
    'click-link_click-button_click-checkboxes_click-option_click-dialog-reverse'
] = functools.partial(
    extract_click_link_click_button_click_checkboxes_click_option_click_dialog,
    reverse=True,
)
FIELD_EXTRACTORS['click-button_click-checkboxes-transfer-reverse'] = (
    functools.partial(
        extract_click_button_click_checkboxes, order=0, reverse=True
    )
)
FIELD_EXTRACTORS['click-button-sequence_click-checkboxes-reverse'] = (
    functools.partial(
        extract_click_button_sequence_click_checkboxes, reverse=True
    )
)
FIELD_EXTRACTORS[
    'click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog-reverse'
] = functools.partial(
    extract_click_widget_click_link_click_button_click_checkboxes_click_option_click_dialog,
    reverse=True,
)
FIELD_EXTRACTORS[
    'click-button-sequence_click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog-reverse'
] = functools.partial(
    extract_click_button_sequence_click_widget_click_link_click_button_click_checkboxes_click_option_click_dialog,
    reverse=True,
)
FIELD_EXTRACTORS[
    'click-button-sequence_click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog_login-user-reverse'
] = functools.partial(
    extract_click_button_sequence_click_widget_click_link_click_button_click_checkboxes_click_option_click_dialog_login_user,
    reverse=True,
)
FIELD_EXTRACTORS['login-user_navigate-tree-transition-reverse'] = (
    FIELD_EXTRACTORS['login-user_navigate-tree-reverse']
)
FIELD_EXTRACTORS['click-option_login-user-transition-reverse'] = (
    FIELD_EXTRACTORS['click-option_login-user-reverse']
)
FIELD_EXTRACTORS[
    'click-checkboxes-transfer_multi-layouts_email-inbox-forward-nl-transition-reverse'
] = functools.partial(extract_click_checkboxes_multi_layouts, reverse=True)
FIELD_EXTRACTORS['enter-password_click-checkboxes_login-user-popup-reverse'] = (
    functools.partial(
        extract_enter_password_click_checkboxes_login_user_popup, reverse=True
    )
)
FIELD_EXTRACTORS[
    'click-checkboxes-transfer_click-button-sequence_enter-password-reverse'
] = functools.partial(
    extract_click_checkboxes_transfer_click_button_sequence_enter_password,
    reverse=True,
)
FIELD_EXTRACTORS[
    'click-checkboxes_click-widget_click-button-sequence-reverse'
] = functools.partial(
    extract_click_checkboxes_click_widget_click_button_sequence, reverse=True
)
FIELD_EXTRACTORS[
    'click-dialog_click-checkboxes-transfer_click-widget-reverse'
] = functools.partial(
    extract_click_dialog_click_checkboxes_transfer_click_widget, reverse=True
)
FIELD_EXTRACTORS[
    'click-checkboxes-transfer_enter-password_click-dialog-reverse'
] = functools.partial(
    extract_click_checkboxes_transfer_enter_password_click_dialog, reverse=True
)
FIELD_EXTRACTORS['click-checkboxes-soft_enter-password-reverse'] = (
    functools.partial(
        extract_click_checkboxes_soft_enter_password, reverse=True
    )
)
FIELD_EXTRACTORS['click-checkboxes-soft_multi-layouts-reverse'] = (
    functools.partial(extract_click_checkboxes_soft_multi_layouts, reverse=True)
)
FIELD_EXTRACTORS['click-widget_click-checkboxes-soft-reverse'] = (
    functools.partial(extract_click_widget_click_checkboxes_soft, reverse=True)
)
FIELD_EXTRACTORS['click-button-sequence_use-autocomplete-reverse'] = (
    functools.partial(
        extract_click_button_sequence_use_autocomplete, reverse=True
    )
)
FIELD_EXTRACTORS['use-autocomplete_click-dialog-reverse'] = functools.partial(
    extract_use_autocomplete_click_dialog, reverse=True
)


################################################################
# Script for printing out the utterances for a particular task


def extract_utterances():
  """Test extract utterances."""
  task_name = ''
  try:
    task_name = sys.argv[1]
  except Exception as e:  # pylint: disable=broad-except
    print('Usage: {} task_name'.format(sys.argv[0]), file=sys.stderr)
    print(e)
    exit(1)
  FIELD_EXTRACTORS[task_name] = lambda utt: Fields({})
  from miniwob_plusplus.environment import MiniWoBEnvironment  # pylint: disable=g-import-not-at-top,g-importing-member

  env = MiniWoBEnvironment(task_name)
  base_url = os.environ.get('MINIWOB_BASE_URL')
  env.configure(num_instances=4, seeds=list(range(4)), base_url=base_url)
  for i in range(25):
    states = env.reset()
    for state in states:
      print('UTT {}:\t{}'.format(i, state.utterance.replace('\n', ' ')))
  env.close()


if __name__ == '__main__':
  extract_utterances()
