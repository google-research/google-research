# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# -*- coding: utf-8 -*-
"""Copy and extended from WikiTableQuestions dataset release."""
u"""Official Evaluator for WikiTableQuestions Dataset

There are 3 value types
1. String (unicode)
2. Number (float)
3. Date (a struct with 3 fields: year, month, and date)
   Some fields (but not all) can be left unspecified. However, if only the year
   is specified, the date is automatically converted into a number.

Target denotation = a set of items
- Each item T is a raw unicode string from Mechanical Turk
- If T can be converted to a number or date (via Stanford CoreNLP), the
    converted value (number T_N or date T_D) is precomputed

Predicted denotation = a set of items
- Each item P is a string, a number, or a date
- If P is read from a text file, assume the following
  - A string that can be converted into a number (float) is converted into a
    number
  - A string of the form "yyyy-mm-dd" is converted into a date. Unspecified
    fields can be marked as "xx". For example, "xx-01-02" represents the date
    January 2nd of an unknown year.
  - Otherwise, it is kept as a string

The predicted denotation is correct if
1. The sizes of the target denotation and the predicted denotation are equal
2. Each item in the target denotation matches an item in the predicted
    denotation

A target item T matches a predicted item P if one of the following is true:
1. normalize(raw string of T) and normalize(string form of P) are identical.
   The normalize method performs the following normalizations on strings:
   - Remove diacritics (é → e)
   - Convert smart quotes (‘’´`“”) and dashes (‐‑‒–—−) into
   ASCII ones
   - Remove citations (trailing •♦†‡*#+ or [...])
   - Remove details in parenthesis (trailing (...))
   - Remove outermost quotation marks
   - Remove trailing period (.)
   - Convert to lowercase
   - Collapse multiple whitespaces and strip outermost whitespaces
2. T can be interpreted as a number T_N, P is a number, and P = T_N
3. T can be interpreted as a date T_D, P is a date, and P = T_D
   (exact match on all fields; e.g., xx-01-12 and 1990-01-12 do not match)
"""
__version__ = '1.0.2'

from abc import ABCMeta
from abc import abstractmethod
import argparse
import codecs
from math import isinf
from math import isnan
import os
import re
import sys
import unicodedata
from six import string_types
from tensorflow.compat.v1 import gfile

################ String Normalization ################


def normalize(x):
  """String Normalization."""
  if not isinstance(x, unicode):
    x = x.decode('utf8', errors='ignore')
  # Remove diacritics
  x = ''.join(
      c for c in unicodedata.normalize('NFKD', x)
      if unicodedata.category(c) != 'Mn')
  # Normalize quotes and dashes
  x = re.sub(ur'[‘’´`]', "'", x)
  x = re.sub(ur'[“”]', '"', x)
  x = re.sub(ur'[‐‑‒–—−]', '-', x)
  while True:
    old_x = x
    # Remove citations
    x = re.sub(ur'((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$', '',
               x.strip())
    # Remove details in parenthesis
    x = re.sub(ur'(?<!^)( \([^)]*\))*$', '', x.strip())
    # Remove outermost quotation mark
    x = re.sub(ur'^"([^"]*)"$', r'\1', x.strip())
    if x == old_x:
      break
  # Remove final '.'
  if x and x[-1] == '.':
    x = x[:-1]
  # Collapse whitespaces and convert to lower case
  x = re.sub(ur'\s+', ' ', x, flags=re.U).lower().strip()
  return x


################ Value Types ################


class Value(object):
  __metaclass__ = ABCMeta

  # Should be populated with the normalized string
  _normalized = None

  @abstractmethod
  def match(self, other):
    """Return True if the value matches the other value.

    Args: other (Value)

    Returns:
      Bool: a boolean
    """
    pass

  @property
  def normalized(self):
    return self._normalized


class StringValue(Value):

  def __init__(self, content):
    assert isinstance(content, string_types)
    self._normalized = normalize(content)
    self._hash = hash(self._normalized)

  def __eq__(self, other):
    return isinstance(other,
                      StringValue) and self.normalized == other.normalized

  def __hash__(self):
    return self._hash

  def __str__(self):
    return 'S' + str([self.normalized])

  __repr__ = __str__

  def match(self, other):
    assert isinstance(other, Value)
    return self.normalized == other.normalized


class NumberValue(Value):

  def __init__(self, amount, original_string=None):
    assert isinstance(amount, (int, long, float))
    if abs(amount - round(amount)) < 1e-6:
      self._amount = int(amount)
    else:
      self._amount = float(amount)
    if not original_string:
      self._normalized = unicode(self._amount)
    else:
      self._normalized = normalize(original_string)
    self._hash = hash(self._amount)

  @property
  def amount(self):
    return self._amount

  def __eq__(self, other):
    return isinstance(other, NumberValue) and self.amount == other.amount

  def __hash__(self):
    return self._hash

  def __str__(self):
    return ('N(%f)' % self.amount) + str([self.normalized])

  __repr__ = __str__

  def match(self, other):
    assert isinstance(other, Value)
    if self.normalized == other.normalized:
      return True
    if isinstance(other, NumberValue):
      return abs(self.amount - other.amount) < 1e-6
    return False

  @staticmethod
  def parse(text):
    """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
    try:
      return int(text)
    except:
      try:
        amount = float(text)
        assert not isnan(amount) and not isinf(amount)
        return amount
      except:
        return None


class DateValue(Value):

  def __init__(self, year, month, day, original_string=None):
    """Create a new DateValue. Placeholders are marked as -1."""
    assert isinstance(year, int)
    assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
    assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
    assert not (year == month == day == -1)
    self._year = year
    self._month = month
    self._day = day
    if not original_string:
      self._normalized = '{}-{}-{}'.format(year if year != -1 else 'xx',
                                           month if month != -1 else 'xx',
                                           day if day != '-1' else 'xx')
    else:
      self._normalized = normalize(original_string)
    self._hash = hash((self._year, self._month, self._day))

  @property
  def ymd(self):
    return (self._year, self._month, self._day)

  def __eq__(self, other):
    return isinstance(other, DateValue) and self.ymd == other.ymd

  def __hash__(self):
    return self._hash

  def __str__(self):
    return (('D(%d,%d,%d)' % (self._year, self._month, self._day)) + str(
        [self._normalized]))

  __repr__ = __str__

  def match(self, other):
    assert isinstance(other, Value)
    if self.normalized == other.normalized:
      return True
    if isinstance(other, DateValue):
      return self.ymd == other.ymd
    return False

  @staticmethod
  def parse(text):
    """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
    try:
      ymd = text.lower().split('-')
      assert len(ymd) == 3
      year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
      month = -1 if ymd[1] == 'xx' else int(ymd[1])
      day = -1 if ymd[2] == 'xx' else int(ymd[2])
      assert not (year == month == day == -1)
      assert month == -1 or 1 <= month <= 12
      assert day == -1 or 1 <= day <= 31
      return (year, month, day)
    except:
      return None


################ Value Instantiation ################


def to_value(original_string, corenlp_value=None):
  """Convert the string to Value object.

    Args:
        original_string (string_types): Original string
        corenlp_value (string_types): Optional value returned from CoreNLP

    Returns:
        Value
    """
  if isinstance(original_string, Value):
    # Already a Value
    return original_string
  if not corenlp_value:
    corenlp_value = original_string
  # Number?
  amount = NumberValue.parse(corenlp_value)
  if amount is not None:
    return NumberValue(amount, original_string)
  # Date?
  ymd = DateValue.parse(corenlp_value)
  if ymd is not None:
    if ymd[1] == ymd[2] == -1:
      return NumberValue(ymd[0], original_string)
    else:
      return DateValue(ymd[0], ymd[1], ymd[2], original_string)
  # String.
  return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None):
  """Convert a list of strings to a list of Values

  Args:
      original_strings: (list[string_types])
      corenlp_values: (list[string_types or None])

  Returns:
      list[Value]
  """
  assert isinstance(original_strings, (list, tuple, set))
  if corenlp_values is not None:
    assert isinstance(corenlp_values, (list, tuple, set))
    assert len(original_strings) == len(corenlp_values)
    return list(
        set(to_value(x, y) for (x, y) in zip(original_strings, corenlp_values)))
  else:
    return list(set(to_value(x) for x in original_strings))


################ Check the Predicted Denotations ################


def check_denotation(target_values, predicted_values):
  """Return True if the predicted denotation is correct.

  Args:
    target_values: (list[Value])
    predicted_values: (list[Value])

  Returns:
      bool
  """
  # Check size
  if len(target_values) != len(predicted_values):
    return False
  # Check items
  for target in target_values:
    if not any(target.match(pred) for pred in predicted_values):
      return False
  return True


################ Batch Mode ################


def tsv_unescape(x):
  """Unescape strings in the TSV file.

    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash

    Args: x (str or unicode)

    Returns:
        a unicode
    """
  return x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')


def tsv_unescape_list(x):
  """Unescape a list in the TSV file.

  List items are joined with vertical bars (0x5C)

  Args: x (str or unicode)

  Returns:
      a list of unicodes
  """
  return [tsv_unescape(y) for y in x.split('|')]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-t',
      '--tagged-dataset-path',
      default=os.path.join('.', 'tagged', 'data'),
      help='Directory containing CoreNLP-tagged dataset TSV file')
  parser.add_argument(
      'prediction_path',
      help='Path to the prediction file. Each line contains '
      'ex_id <tab> item1 <tab> item2 <tab> ...')
  args = parser.parse_args()

  # ID string --> list[Value]
  target_values_map = {}
  for filename in os.listdir(args.tagged_dataset_path):
    filename = os.path.join(args.tagged_dataset_path, filename)
    print >> sys.stderr, 'Reading dataset from', filename
    with codecs.getreader('utf-8')(gfile.GFile(filename, 'r')) as fin:
      header = fin.readline().rstrip('\n').split('\t')
      for line in fin:
        stuff = dict(zip(header, line.rstrip('\n').split('\t')))
        ex_id = stuff['id']
        original_strings = tsv_unescape_list(stuff['targetValue'])
        canon_strings = tsv_unescape_list(stuff['targetCanon'])
        target_values_map[ex_id] = to_value_list(original_strings,
                                                 canon_strings)
  print >> sys.stderr, 'Read', len(target_values_map), 'examples'

  print >> sys.stderr, 'Reading predictions from', args.prediction_path
  num_examples, num_correct = 0, 0
  with codecs.getreader('utf-8')(gfile.GFile(args.prediction_path, 'r')) as fin:
    for line in fin:
      line = line.rstrip('\n').split('\t')
      ex_id = line[0]
      if ex_id not in target_values_map:
        print 'WARNING: Example ID "%s" not found' % ex_id
      else:
        target_values = target_values_map[ex_id]
        predicted_values = to_value_list(line[1:])
        correct = check_denotation(target_values, predicted_values)
        print u'%s\t%s\t%s\t%s' % (ex_id, correct, target_values,
                                   predicted_values)
        num_examples += 1
        if correct:
          num_correct += 1
  print >> sys.stderr, 'Examples:', num_examples
  print >> sys.stderr, 'Correct:', num_correct
  print >> sys.stderr, 'Accuracy:', round(
      (num_correct + 1e-9) / (num_examples + 1e-9), 4)


# Added utility functions for computing preprocessing answers and computing
# rewards.
def target_values_map(target_value, target_cannon):
  original_strings = tsv_unescape_list(target_value)
  canon_strings = tsv_unescape_list(target_cannon)
  target_values = to_value_list(original_strings, canon_strings)
  return target_values


def check_prediction(ts_prediction_string, target_values):
  predicted_values = to_value_list(ts_prediction_string)
  correct = check_denotation(target_values, predicted_values)
  return correct


if __name__ == '__main__':
  main()
