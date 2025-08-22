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

"""Vocabulary items and utility functions."""
import re
import gin

# initial vocabulary
VOCAB = set()
VOCAB.add("NULL")
VOCAB.add("OOV")


@gin.configurable
def tokenize(sentence, structured_input=False, lowercase=True,
             camelcase=False):
  """Tokenize a given sequence of words.

  Args:
    sentence: A sequence of characters to be tokenized.
    structured_input: If true, use simply split to tokenize.
    lowercase: If true, first lowercase the sentence.
    camelcase: If true, first tokenize w.r.t. camelcase text.

  Returns:
    A sequence of tokens that corresponds to the input sentence.
  """
  sentence = str(sentence)
  if not sentence or sentence == "None":
    return []
  sentence = sentence.replace(".", " ")
  # pylint: disable=line-too-long
  # See https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case/17328907
  if camelcase and not (sentence.lower() == sentence or
                        sentence.upper() == sentence):
    sentence = re.sub(r"(?<!^)(?=[A-Z])", " ", sentence)
  if lowercase:
    sentence = sentence.lower()
  if structured_input:
    tokenized = sentence.split()
  else:
    # Similar to nltk wordpuncttokenizer: split into alphabetic and
    # non-alphabetic tokens. Consecutive non-alphabetic tokens are splitted.
    tokenized = re.findall(
        r"\w+|[^\w\s]",
        sentence)
  return tokenized


def create_default_miniwob_vocabulary(path):
  """Create a default miniwob vocabulary from common miniwob tokens.

  Args:
    path: Path to the unprocessed vocabulary file. Each line will be processed
      and added to the vocabulary.
  Returns:
    A vocabulary of unique tokens.
  """
  lines = open(path).readlines()
  for line in lines:
    VOCAB.update(tokenize(line.strip()))
  return VOCAB
