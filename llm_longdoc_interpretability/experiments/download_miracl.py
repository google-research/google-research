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

"""Downloads MirACL corpus and queries."""

from collections.abc import Sequence
import json
import os

from absl import app
import datasets


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)

  # Download passages from MirACL corpus
  print('DOWNLOADING PASSAGES')
  lang = 'en'
  miracl_corpus = datasets.load_dataset('miracl/miracl-corpus', lang)['train']

  corpus_path = 'miracl/corpus_' + lang + '/'
  if not os.path.exists(corpus_path):
    os.mkdir(corpus_path)

    pass

  print('SAVING PASSAGES')
  x = 0
  X = 100 * 1000
  with open(corpus_path + 'passages_' + lang + '.jsonl', 'w') as f2:
    for doc in miracl_corpus:
      docid = doc['docid']
      title = doc['title']
      text = doc['text']

      json_dict = {
          'id': str(x),
          'docid': docid,
          'title': title,
          'text': text,
      }

      f2.write(json.dumps(json_dict))
      f2.write('\n')

      x += 1
      if x % X == 0:
        pass
  pass

  # Download questions from MirACL
  print('DOWNLOADING QUERIES')
  miracl_questions = datasets.load_dataset(
      'miracl/miracl', lang, use_auth_token=True
  )

  print('SAVING QUERIES')
  split = 'train'  # or 'dev', 'testA', 'testB'

  x = 0
  with open(
      'miracl/queries_' + lang + '/queries_' + lang + '_' + split + '.jsonl',
      'w',
  ) as f2:
    for data in miracl_questions[split]:
      query_id = data['query_id']
      query = data['query']
      positive_passages = data['positive_passages']
      negative_passages = data['negative_passages']

      answers = [passage['text'] for passage in positive_passages]

      json_dict = {
          'question': query,
          'answers': answers,
          'query_id': query_id,
          'positive_passages': positive_passages,
          'negative_passages': negative_passages,
      }
      f2.write(json.dumps(json_dict))
      f2.write('\n')

    x += 1
