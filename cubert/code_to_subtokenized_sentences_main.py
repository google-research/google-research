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

"""This modules demonstrates how to convert code to subtokenized sentences."""
import json


from absl import app
from absl import flags
from tensor2tensor.data_generators import text_encoder


from cubert import code_to_subtokenized_sentences
from cubert import tokenizer_registry

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

  subtokenized_sentences = (
      code_to_subtokenized_sentences.code_to_cubert_sentences(
          code=code,
          initial_tokenizer=tokenizer,
          subword_tokenizer=subword_tokenizer))
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
