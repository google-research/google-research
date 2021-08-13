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

r"""Extracts data from XML Bible; produces char correlations a la Penn & Choma.

Penn, Gerald and Travis Choma. (2006). "Quantitative methods for classifying
writing systems." Proceedings of the North American Chapter of the Association
for Computational Linguistics, pages 117--120.

Example run:

python3 pennchoma/penn_choma.py \
  --bible=${HOME}/bible-corpus-master/bibles/Chinese.xml \
  --output=/var/tmp/penn_choma/chinese_ungrouped.txt
"""

import random
import sys
import xml.etree.ElementTree as ET

from absl import app
from absl import flags

import preprocessors
import scc

flags.DEFINE_string("bible", None, "Path to XML Bible")
flags.DEFINE_string("output", None, "Path to output file for correlations")
flags.DEFINE_integer("group_docs", -1, "Group this many chapters into a doc.")
flags.DEFINE_integer("random_subset", 500, "Random subset size.")
flags.DEFINE_bool("trigrams", False, "Use trigram preprocessor.")

FLAGS = flags.FLAGS


class Bible(object):
  """Produces documents for the given bible, where each document is a chapter.
  """

  def __init__(self, path, prepro=None):
    self._tree = ET.parse(path)
    self._root = self._tree.getroot()
    self._prepro = prepro

  @property
  def root(self):
    return self._root

  def get_documents(self, group_docs=-1):
    """Extracts documents.

    Args:
      group_docs: if > 1, then group that many chapters into a "document".
    Returns:
      a Corpus of texts
    """
    [text_body] = self.root.findall("text")[0].findall("body")
    books = text_body.findall("div")
    texts = []
    for book in books:
      assert book.attrib["type"] == "book"
      chapters = book.findall("div")
      for chapter in chapters:
        assert chapter.attrib["type"] == "chapter"
        verses = chapter.findall("seg")
        chapter_text = []
        for verse in verses:
          try:
            chapter_text.append(verse.text.strip())
          except AttributeError:
            pass
        texts.append(" ".join(chapter_text))
    if group_docs > 1:

      def pair_up(elts):
        new_elts = []
        for i in range(0, len(elts), group_docs):
          new_elts.append("".join(elts[i:i+group_docs]))
        return new_elts

      texts = pair_up(texts)
    texts = [scc.Document(t, self._prepro) for t in texts]
    return scc.Corpus(texts)


def bible_correlations(which_bible,
                       prepro=None,
                       stream=sys.stdout,
                       random_subset=0,
                       group_docs=-1):
  """Computes correlations for a given bible.

  Args:
    which_bible: path to an XML bible.
    prepro: preprocessor to use from preprocessors.py, or None.
    stream: a writeable stream.
    random_subset: if > 0, chose this many characters at random to test.
    group_docs: if > 0, group this many characters into a "document".
  """
  bible = Bible(which_bible, prepro)
  corpus = bible.get_documents(group_docs)
  sys.stderr.write("Corpus name:\t{}\n".format(which_bible))
  sys.stderr.write("Corpus size:\t{}\n".format(corpus.size))
  sys.stderr.write("Corpus ndocs:\t{}\n".format(corpus.ndocs))
  sys.stderr.write("Corpus nchars:\t{}\n".format(corpus.nchars))
  stream.write("Corpus name:\t{}\n".format(which_bible))
  stream.write("Corpus size:\t{}\n".format(corpus.size))
  stream.write("Corpus ndocs:\t{}\n".format(corpus.ndocs))
  stream.write("Corpus nchars:\t{}\n".format(corpus.nchars))
  characters = sorted(list(corpus.characters))
  random.shuffle(characters)
  if random_subset > 0:
    chosen_chars = set(characters[:random_subset])
  else:
    chosen_chars = set(characters)
  summed_absolute_value_of_correlation = 0
  for i in range(len(characters)):
    c1 = characters[i]
    for j in range(len(characters)):
      c2 = characters[j]
      in_index = 0
      if c1 in chosen_chars and c2 in chosen_chars:
        in_index = 1
        # Only do the calculation for chosen character pairs.
        corr = corpus.corr(c1, c2)
        stream.write(f"{in_index}:{c1},{c2}\t{corr:.6f}\n")
        summed_absolute_value_of_correlation += abs(corr)
  stream.write("Summed absolute value of correlation:\t" +
               f"{summed_absolute_value_of_correlation:.2f}\n")


def main(unused_argv):
  with open(FLAGS.output, "w") as stream:
    prepro = None
    if FLAGS.trigrams:
      prepro = preprocessors.trigram_letters
    bible_correlations(FLAGS.bible,
                       prepro=prepro,
                       stream=stream,
                       random_subset=FLAGS.random_subset,
                       group_docs=FLAGS.group_docs)


if __name__ == "__main__":
  app.run(main)
