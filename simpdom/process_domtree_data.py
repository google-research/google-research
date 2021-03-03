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

"""Builds vocab files for a vertical in word/char/tag level respectively."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import itertools
import json
import os
import sys
import tempfile

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from simpdom import constants

FLAGS = flags.FLAGS
flags.DEFINE_integer("dim_word_glove", 100,
                     "The dimensionality of the word embeddings.")
flags.DEFINE_integer("word_frequence_cutoff", 3,
                     "Ignore the words whose frequence is under this.")
flags.DEFINE_string(
    "domtree_path", "",
    "The path of the json file containing node and features from swde dataset.")
flags.DEFINE_string(
    "word_embedding_path", "",
    "The path of word embedding file, which should be in GloVe format.")


def get_leaf_type(xpath):
  """Gets the leaf type from the xpath."""
  # Example:
  # "/div[1]/span[2]/br[1]"-->"br"
  # "/div[1]/span[2]/tail"-->"span"
  # "/div[1]/span[2]/a"-->"a"
  html_tags = xpath.split("/")
  tag = html_tags[-1]
  if tag.startswith("tail"):
    tag = html_tags[-2]
  if tag.find("[") >= 0:
    return tag[:tag.find("[")]  # Clean the tag index.
  else:
    return tag


def split_xpath(xpath):
  """Gets a list of html tags from a xpath string."""
  # Example:
  # "/div[1]/span[2]/br[1]" --> ["div", "span", "br"]
  # "/div[1]/span[2]/tail" --> ["div", "span", "tail"]
  # "/div[1]/span[2]/a" --> ["div", "span", "a"]
  split_tags = []
  for tag in xpath.split("/"):
    if tag.find("[") >= 0:
      tag = tag[:tag.find("[")]
    if tag.strip():
      split_tags.append(tag.strip())
  return split_tags


def build_vocab(json_data, vertical_to_process):
  """Builds the vacabulary of a vertical's all pages."""
  counter_words = collections.Counter()
  vocab_labels = set()
  vocab_chars = set()
  vocab_leaf_html_tags = set()
  vocab_html_tags = set()

  for page in json_data["features"]:
    for node in page:
      path = node["html_path"]
      vertical = path.split("/")[0]
      if vertical == vertical_to_process:
        counter_words.update(node["text"])
        counter_words.update(
            list(itertools.chain.from_iterable(node["prev_text"])))
        vocab_labels.update([node["label"]])
        vocab_leaf_html_tags.update([get_leaf_type(node["xpath"])])
        vocab_html_tags.update(split_xpath(node["xpath"]))

  vocab_words = {
      w for w, c in counter_words.items() if c >= FLAGS.word_frequence_cutoff
  }
  for w in vocab_words:
    vocab_chars.update(w)
  return (vocab_words, vocab_labels, vocab_chars, vocab_leaf_html_tags,
          vocab_html_tags)


def get_emebeddings(vocab_words, embedding_file_lines, vertical_to_process):
  """Gets relevant glove vectors and saves to a file."""
  word_to_id = {
      word: index for index, word in enumerate(sorted(list(vocab_words)))
  }
  vocab_size = len(word_to_id)

  embeddings = np.zeros((vocab_size, FLAGS.dim_word_glove))
  found = 0
  print("Writing word embedding file (may take a while)...")
  glove_embedding_dict = {}

  for line in embedding_file_lines:
    line = line.strip().split()
    if len(line) != FLAGS.dim_word_glove + 1:
      continue
    word = line[0]
    embedding = line[1:]
    glove_embedding_dict[word] = embedding
  for vocab_word in vocab_words:
    if vocab_word in glove_embedding_dict:
      found += 1
      word_idx = word_to_id[vocab_word]
      embeddings[word_idx] = glove_embedding_dict[vocab_word]
    elif vocab_word.lower() in glove_embedding_dict:
      found += 1
      word_idx = word_to_id[vocab_word]
      embeddings[word_idx] = glove_embedding_dict[vocab_word.lower()]
  # Save np.array to file.
  with tempfile.TemporaryFile() as tmp, tf.gfile.Open(
      os.path.join(FLAGS.domtree_path, vertical_to_process + ".%d.emb.npz" %
                   (FLAGS.dim_word_glove)), "wb") as gfo:
    np.savez(tmp, embeddings=embeddings)
    tmp.seek(0)
    gfo.write(tmp.read())
    print("- done. Found {} vectors for {} words for {}".format(
        found, vocab_size, gfo.name))


def write_vocab(vocab_type, vocab, vertical_to_process):
  """Writes the vocabularies to files."""
  with tf.gfile.Open(
      os.path.join(FLAGS.domtree_path,
                   vertical_to_process + ".vocab.%s.txt" % (vocab_type)),
      "w") as vocab_file:
    for item in sorted(list(vocab)):
      vocab_file.write("{}\n".format(item))
    print("Saving done:", vocab_file.name, file=sys.stderr)


def main(_):
  verticals = constants.VERTICAL_WEBSITES.keys()

  with tf.gfile.Open(FLAGS.word_embedding_path, "r") as embedding_file:
    embedding_file_lines = embedding_file.read().split("\n")
  (all_vocab_words, all_vocab_labels, all_vocab_chars, all_vocab_leaf_html_tags,
   all_vocab_html_tags) = set(), set(), set(), set(), set()
  for vertical_to_process in verticals:
    print("Processing vertical:", vertical_to_process, file=sys.stderr)
    (vocab_words, vocab_labels, vocab_chars, vocab_leaf_html_tags,
     vocab_html_tags) = set(), set(), set(), set(), set()
    for json_data_path in tf.gfile.ListDirectory(FLAGS.domtree_path):
      if json_data_path.endswith(".json") and json_data_path.startswith(
          vertical_to_process) and len(json_data_path.split("-")) == 2:
        print("processing %s" % (json_data_path), file=sys.stderr)
        json_data = json.load(
            tf.gfile.Open(
                os.path.join(FLAGS.domtree_path, json_data_path), "r"))
        (current_words, current_tags, current_chars, current_leaf_types,
         current_xpath_units) = build_vocab(json_data, vertical_to_process)
        vocab_words.update(current_words)
        vocab_labels.update(current_tags)
        vocab_chars.update(current_chars)
        vocab_leaf_html_tags.update(current_leaf_types)
        vocab_html_tags.update(current_xpath_units)
    # Add the current vertrical's vocab to an over-all vocab.
    all_vocab_words.update(vocab_words)
    all_vocab_labels.update(vocab_labels)
    all_vocab_chars.update(vocab_chars)
    all_vocab_leaf_html_tags.update(vocab_leaf_html_tags)
    all_vocab_html_tags.update(vocab_html_tags)

    # Saving vocabs and word embeddings.
    write_vocab("words", vocab_words, vertical_to_process)
    write_vocab("tags", vocab_labels, vertical_to_process)
    write_vocab("chars", vocab_chars, vertical_to_process)
    get_emebeddings(vocab_words, embedding_file_lines, vertical_to_process)
    write_vocab("leaf_types", vocab_leaf_html_tags, vertical_to_process)
    write_vocab("xpath_units", vocab_html_tags, vertical_to_process)

  # Saving over-all vocabs and word embeddings.
  write_vocab("words", all_vocab_words, "all")
  write_vocab("tags", all_vocab_labels, "all")
  write_vocab("chars", all_vocab_chars, "all")
  get_emebeddings(all_vocab_words, embedding_file_lines, "all")
  write_vocab("leaf_types", all_vocab_leaf_html_tags, "all")
  write_vocab("xpath_units", all_vocab_html_tags, "all")


if __name__ == "__main__":
  app.run(main)
