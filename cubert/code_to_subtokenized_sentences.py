# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""This modules demonstrates how to convert code to subtokenized sentences."""

import itertools
import json
from typing import List, Text

from absl import app
from absl import flags
from absl import logging
from tensor2tensor.data_generators import text_encoder

from cubert import cubert_tokenizer
from cubert import tokenizer_registry
from cubert import unified_tokenizer

FLAGS = flags.FLAGS

flags.DEFINE_string('vocabulary_filepath', None,
                    'Path to the subword vocabulary.')

flags.DEFINE_string('input_filepath', None,
                    'Path to the Python source code file.')

flags.DEFINE_string('output_filepath', None,
                    'Path to the output file of subtokenized source code.')

flags.DEFINE_enum_class(
    'tokenizer',
    default=tokenizer_registry.TokenizerEnum.PYTHON,
    enum_class=tokenizer_registry.TokenizerEnum,
    help='The tokenizer to use.')


def code_to_cubert_sentences(
    code,
    initial_tokenizer,
    subword_tokenizer,
):
  """Tokenizes code into a list of CuBERT sentences.

  Args:
    code: The source code to tokenize. This must be a parseable unit of code,
      meaning it represents an AST (or a complete subtree of an AST). For
      example, there should be no unmatched parentheses, and `if` and other
      blocks of code must have bodies.
    initial_tokenizer: The first tokenizer that creates sentences, probably a
      cubert_tokenizer.CuBertTokenizer.
    subword_tokenizer: A second tokenizer that splits tokens of the
      `initial_tokenizer` into subtokens.

  Returns:
    A list of sentences.
  """
  tokens = initial_tokenizer.tokenize(code)[:-1]  # type: List[Text]
  logging.vlog(5, 'Code >>>%s<<< is tokenized into >>>%s<<<.', code, tokens)

  # This will split the list into sublists of non-NEWLINE tokens (key is
  # False) and NEWLINE tokens (key is True).
  groups_by_endtoken = itertools.groupby(
      tokens, key=lambda x: x == unified_tokenizer.NEWLINE)
  # This will keep only the sublists that aren't just [NEWLINE]*, i.e., those
  # that have key False. We call these raw_sentences, because they're not
  # terminated.
  raw_sentences = [list(v) for k, v in groups_by_endtoken if not k
                  ]  # type: List[List[Text]]

  # Now we append a NEWLINE token after all sentences. Note that our tokenizer
  # drops any trailing \n's before tokenizing, but for the purpose of forming
  # properly terminated sentences, we always end sentences in a NEWLINE token.
  sentences = [s + [unified_tokenizer.NEWLINE] for s in raw_sentences
              ]  # type: List[List[Text]]
  logging.vlog(5, 'Tokens are split into sentences: >>>%s<<<.',
               sentences)

  # Now we have to encode tokens using the subword text encoder, expanding the
  # sentences.
  subtokenized_sentences = []  # type: List[List[Text]]
  for sentence in sentences:
    encoded_tokens = [subword_tokenizer.encode_without_tokenizing(t)
                      for t in sentence]  # type: List[List[int]]
    logging.vlog(5, 'Sentence encoded into >>>%s<<<.', encoded_tokens)
    flattened_encodings = sum(encoded_tokens, [])  # type: List[int]
    logging.vlog(5, 'Flattened into >>>%s<<<.', flattened_encodings)
    decoded_tokens = subword_tokenizer.decode_list(
        flattened_encodings)  # type: List[Text]
    logging.vlog(5, 'Sentence re-decoded into >>>%s<<<.', decoded_tokens)

    subtokenized_sentences.append(decoded_tokens)
  logging.vlog(5, 'Sentences are further subtokenized: >>>%s<<<.',
               subtokenized_sentences)

  return subtokenized_sentences


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # The value of the `TokenizerEnum` is a `CuBertTokenizer` subclass.
  tokenizer = FLAGS.tokenizer.value()
  subword_tokenizer = text_encoder.SubwordTextEncoder(FLAGS.vocabulary_filepath)

  with open(FLAGS.input_filepath, 'r') as input_file:
    code = input_file.read()
    print('#' * 80)
    print('Original Code')
    print('#' * 80)
    print(code)

  subtokenized_sentences = code_to_cubert_sentences(
      code=code,
      initial_tokenizer=tokenizer,
      subword_tokenizer=subword_tokenizer)
  print('#' * 80)
  print('CuBERT Sentences')
  print('#' * 80)
  print(subtokenized_sentences)

  with open(FLAGS.output_filepath, 'wt') as output_file:
    output_file.write(json.dumps(subtokenized_sentences, indent=2))

if __name__ == '__main__':
  flags.mark_flag_as_required('vocabulary_filepath')
  flags.mark_flag_as_required('input_filepath')
  flags.mark_flag_as_required('output_filepath')
  app.run(main)
