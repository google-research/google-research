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

"""Language processing utilities."""

import spacy


def load_spacy_model(model='en_core_web_trf'):
  nlp = spacy.load(model)
  return nlp


def process_sentence(sentence, nlp):
  """Process a sentence."""
  doc = nlp(sentence)
  sentence_for_spacy = []

  for _, token in enumerate(doc):
    if token.text == ' ':
      continue
    sentence_for_spacy.append(token.text)

  sentence_for_spacy = ' '.join(sentence_for_spacy)
  noun_phrase, _, _ = extract_noun_phrase(
      sentence_for_spacy, nlp, need_index=True
  )
  return noun_phrase


def extract_noun_phrase(text, nlp, need_index=False):
  """Extract noun phrase from text. nlp is a spacy model.

  Args:
      text: str, text to be processed.
      nlp: spacy model.
      need_index: bool, whether to return the index of the noun phrase.

  Returns:
      noun_phrase: str, noun phrase of the text.
  """
  # text = text.lower()

  doc = nlp(text)

  chunks = {}
  chunks_index = {}
  for chunk in doc.noun_chunks:
    for i in range(chunk.start, chunk.end):
      chunks[i] = chunk
      chunks_index[i] = (chunk.start, chunk.end)

  for token in doc:
    if token.head.i == token.i:
      head = token.head

  if head.i not in chunks:
    children = list(head.children)
    if children and children[0].i in chunks:
      head = children[0]
    else:
      if need_index:
        return text, [], text
      else:
        return text

  head_noun = head.text
  head_index = chunks_index[head.i]
  head_index = [i for i in range(head_index[0], head_index[1])]

  sentence_index = [i for i in range(len(doc))]
  not_phrase_index = []
  for i in sentence_index:
    # not_phrase_index.append(i) if i not in head_index else None
    if i not in head_index:
      not_phrase_index.append(i)

  head = chunks[head.i]
  if need_index:
    return head.text, not_phrase_index, head_noun
  else:
    return head.text
