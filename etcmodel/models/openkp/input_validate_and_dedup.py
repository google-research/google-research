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

r"""Script to validate and dedup MS Marco OpenKP dataset.

Some urls appear multiple times in the MS Marco OpenKP dataset: 6x in the dev
and eval sets; ~50 urls appear ~20x in the train set. This script:
1. creates new json files with one example per url,
2. drops KeyPhrases that do not occur in text (no attempt to fix punctuation),
3. keeps at most 3 KeyPhrases.

OpenKPDev.jsonl: keeps 6610 examples out of 6616.
OpenKPTrain.jsonl: keeps 133724 examples out of 134894.
OpenKPEvalPublic.jsonl: keeps 6613 examples out of 6614.
"""

import json

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None,
    'Jsonl file with the MS Marco OpenKP dataset (train or dev).')

flags.DEFINE_string('output_file', None, 'Output jsonl file.')

flags.DEFINE_boolean(
    'is_eval', False,
    'Set True for eval files, and leave False for train files with KeyPhrases '
    'defined.')


def is_word_start(text, i):
  if i == 0:
    return True
  elif i > 0 and text[i - 1] == ' ':
    return True
  else:
    return False


def is_word_stop(text, i):
  if i == len(text):
    return True
  elif i > 0 and text[i] == ' ':
    return True
  else:
    return False


def is_keyphrase_whole_words(text, keyphrase_list):
  """Check if keyphrase is contained in text ans formed by whole words."""
  keyphrase = ' '.join(
      [item.strip() for item in keyphrase_list if item.strip()])
  keyphrase_lower = keyphrase.lower()
  n = len(keyphrase)
  # The keyphrase has to be a full word match.
  j = text.find(keyphrase_lower)
  while j >= 0:
    if is_word_start(text, j) and is_word_stop(text, j + n):
      return True, [keyphrase]
    j = text.find(keyphrase_lower, j + 1)
  return False, [keyphrase]


def main(unused_argv):
  lines = tf.gfile.Open(FLAGS.input_file).readlines()
  json_lines = [json.loads(line) for line in lines]

  with tf.gfile.Open(FLAGS.output_file, 'w') as fout:
    # Go through all the data line by line and check types of the data.
    counter_url = 0
    for line in json_lines:
      counter_url += 1
      url = line['url']
      assert isinstance(url, str)

      text = line['text']
      assert isinstance(text, str)
      text_split = text.split(' ')

      if not FLAGS.is_eval:
        keyphrases = line['KeyPhrases']
        assert isinstance(keyphrases, list)

      vdom_str = line['VDOM']
      assert isinstance(vdom_str, str)

      vdom = json.loads(vdom_str)
      assert isinstance(vdom, list)

      # Check that the vdom text actually occurs in the provided document text.
      # And that the VDOMS are contiguous and cover the whole document text.
      url_id = -1
      assert vdom
      assert vdom[0]['start_idx'] == 0  # First VDOM starts at position 0.
      assert vdom[-1]['end_idx'] == len(text_split)  # Last VDOM ends at end.
      for i in range(len(vdom)):
        el = vdom[i]
        el_id = el['Id']  # Seems to be example id, not element id.
        el_text = el['text']
        start_idx = el['start_idx']
        end_idx = el['end_idx']
        if i == 0:
          url_id = el_id
        else:
          assert el_id == url_id
        # Check that the VDOM text matches the correct position in the document
        # text.
        assert el_text == ' '.join(text_split[start_idx:end_idx])
        # Check that the VDOM elements are consecutively covering the document
        if i > 0:
          assert start_idx == vdom[i - 1]['end_idx']
    print('Processed %d json lines.' % counter_url)

    if FLAGS.is_eval:
      counter_url = 0
      url_set = set()
      for line in json_lines:
        url = line['url']
        if url in url_set:
          print('Duplicate url:', url)
          # There is only one duplicate in the eval set, and we keep its first
          # occurrence.
          continue
        url_set.add(url)
        counter_url += 1
        fout.write(json.dumps(line))
        fout.write('\n')
      print('Processed %d urls.' % counter_url)
    else:  # Train or dev file.
      # Count valid KeyPhrases and pick the examples with most valid KeyPhrases.
      counter_url = 0
      url2valid_line = {}  # Store tuple: url -> (valid_keyphrases, line number)
      num_valid_keyphrase = []
      for iline, line in enumerate(json_lines):
        counter_url += 1
        url = line['url']
        text = line['text'].lower()
        keyphrases = line['KeyPhrases']

        valid_keyphrases = []
        for keyphrase_list in keyphrases:
          is_valid, keyphrase = is_keyphrase_whole_words(text, keyphrase_list)
          if is_valid:
            valid_keyphrases.append(keyphrase)
          # else:
          #   print('invalid:', keyphrase, url)
        assert valid_keyphrases
        num_valid_keyphrase.append(len(valid_keyphrases))
        # Keep at most 3 keyphrases (the dev set has up to 5 for some examples).
        line['KeyPhrases'] = valid_keyphrases[:3]

        if url not in url2valid_line:
          url2valid_line[url] = []
        # Sorting this will take the example with most keyphrases, and lowest
        # line number.
        url2valid_line[url].append((-len(valid_keyphrases), iline))
      print('Processed %d json lines.' % counter_url)
      print('set(num_valid_keyphrase):', set(num_valid_keyphrase))

      counter_url = 0
      for url in url2valid_line:
        counter_url += 1
        pairs = url2valid_line[url]
        pairs.sort()
        if pairs[0][0] == 0:
          print('zero valid keyphrases:', url)
        else:
          if len(pairs) > 1:
            print('dropping some examples for:', url)
          # This is the line of the selected example.
          iline = pairs[0][1]
          # Dump the example
          fout.write(json.dumps(json_lines[iline]))
          fout.write('\n')
      print('Processed %d urls.' % counter_url)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_file')
  flags.mark_flag_as_required('output_file')
  app.run(main)
