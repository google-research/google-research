# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""A library for tokenizing text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six


# Pre-compile regexes that are use often
NON_ALPHANUM_PATTERN = r"[^a-z0-9]+"
NON_ALPHANUM_RE = re.compile(NON_ALPHANUM_PATTERN)
SPACES_PATTERN = r"\s+"
SPACES_RE = re.compile(SPACES_PATTERN)
VALID_TOKEN_PATTERN = r"^[a-z0-9]+$"
VALID_TOKEN_RE = re.compile(VALID_TOKEN_PATTERN)


def tokenize(text, stemmer):
  """Tokenize input text into a list of tokens.

  This approach aims to replicate the approach taken by Chin-Yew Lin in
  the original ROUGE implementation.

  Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.

  Returns:
    A list of string tokens extracted from input text.
  """

  # Convert everything to lowercase.
  text = text.lower()
  # Replace any non-alpha-numeric characters with spaces.
  text = NON_ALPHANUM_RE.sub(" ", six.ensure_str(text))

  tokens = SPACES_RE.split(text)
  if stemmer:
    # Only stem words more than 3 characters long.
    tokens = [six.ensure_str(stemmer.stem(x)) if len(x) > 3 else x
              for x in tokens]

  # One final check to drop any empty or invalid tokens.
  tokens = [x for x in tokens if VALID_TOKEN_RE.match(x)]

  return tokens
