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

"""Utilities for processing Bible corpora.

The corpus is a multilingual parallel corpus created from translations of
the Bible available from https://github.com/christos-c/bible-corpus:

  Christos Christodoulopoulos and Mark Steedman (2015): "A massively parallel
  corpus: the Bible in 100 languages", Proc. of Language Resources and
  Evaluation (LREC), pp. 375–395.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import xml.etree.ElementTree as et

from absl import flags
from absl import logging

import progress.bar as bar

flags.DEFINE_string(
    "input_xml_file", "",
    "Path to the XML file containing the text.")

flags.DEFINE_string(
    "test_set_ids_file", "",
    ("File containing newline separated list of book/chapter/verse IDs that "
     "should be used for test. Rest of the IDs will be used for training."))

flags.DEFINE_string(
    "output_text_file", "",
    "Parse the actual contents of XML and save as plain text.")

flags.DEFINE_string(
    "output_data_file", "",
    ("Output file containing the actual training/test data with "
     "orthography/pronunciation pairs."))

flags.DEFINE_boolean(
    "verbose", False,
    "Additional debugging output, if enabled.")

FLAGS = flags.FLAGS

# Default encoding.
_ENCODING = "utf-8"

# Advanced normalization.
_CLEANUP_RE = r'[-—,"\"“”;\.\?\!:0-9()»\']'


class BibleReader(object):
  """Simple Bible reading interface."""

  def __init__(self):
    if not FLAGS.input_xml_file:
      raise ValueError("Specify --input_xml_file!")
    if not FLAGS.test_set_ids_file:
      raise ValueError("Specify --test_set_ids_file!")

  def _read_test_ids(self):
    """Reads the IDs of the test verses into a set."""
    with open(FLAGS.test_set_ids_file, "r", encoding=_ENCODING) as f:
      return set(f.read().split("\n"))

  def _process_sentence(self, sentence):
    """Processes a single sentence."""
    raise ValueError("Please override this class!")

  def read(self):
    """Processes a single Bible."""
    output_text_file = None
    if FLAGS.output_text_file:
      logging.info("Saving original text to \"%s\".", FLAGS.output_text_file)
      output_text_file = open(FLAGS.output_text_file, "w", encoding=_ENCODING)

    output_data_file = None
    if FLAGS.output_data_file:
      logging.info("Saving the training/test data to \"%s\" ...",
                   FLAGS.output_data_file)
      output_data_file = open(FLAGS.output_data_file, "w", encoding=_ENCODING)

    test_ids = self._read_test_ids()
    logging.info("Read %d test set verse IDs.", len(test_ids))

    logging.info("Reading Bible from \"%s\" ...", FLAGS.input_xml_file)
    with open(FLAGS.input_xml_file, "r", encoding=_ENCODING) as f:
      root = et.fromstring(f.read())
    num_sentences = 0
    for n in root.iter("seg"):
      num_sentences += 1
    progress_bar = bar.IncrementalBar("Processing", max=num_sentences)
    for n in root.iter("seg"):
      if not n.text:
        continue
      sentence = n.text.strip()
      sent_id = n.attrib["id"]
      if FLAGS.verbose:
        logging.info("%s: %s", sent_id, sentence)

      # Simply save the original text.
      if output_text_file:
        output_text_file.write(n.text.strip() + "\n")

      # Process and save the training/test data.
      sentence = re.sub(_CLEANUP_RE, "", sentence)
      word_prons, bad_prons = self._process_sentence(sentence)
      if sent_id in test_ids:
        sent_id = "test_" + sent_id
      else:
        sent_id = "train_" + sent_id
      if bad_prons:
        sent_id += "_NULLPRON"
      if output_data_file:
        output_data_file.write("%s\t%s\n" % (sent_id, " ".join(word_prons)))

      progress_bar.next()

    # Cleanup.
    progress_bar.finish()
    if output_text_file:
      output_text_file.close()
