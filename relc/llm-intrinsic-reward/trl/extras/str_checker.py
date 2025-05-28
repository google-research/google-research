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

r"""Check if string is gibberish."""
import re
import nltk
from nltk.corpus import words

# Download the set of words if you haven't already
# nltk.download('words')
word_list = set(words.words())


def is_empty_or_whitespace(s):
    return not s or s.isspace()

def is_repeating(s):
    for length in range(1, len(s) // 2 + 1):
        substring = s[:length]
        if substring * (len(s) // length) + substring[:len(s) % length] == s:
            return True
    return False

def has_coherent_words(s):
    # Tokenize the string into words
    tokens = re.findall(r'\b\w+\b', s.lower())
    return any(token in word_list for token in tokens)

def is_gibberish(s):
    if is_empty_or_whitespace(s) or is_repeating(s):
        return True
    return not has_coherent_words(s)