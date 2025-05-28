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

"""Utility for generating training/test data for neural neighborhood model."""

import collections
import csv
import os
import random

import kruskal
import lingpy as lp
from lingvo import compat as tf

import pynini

tf.flags.DEFINE_string(
    "task_data_dir", None,
    "Directory containing the SIGTYP shared task data. The path should end "
    "with eigher `data` or `data-surprise` directory.")
tf.flags.DEFINE_float(
    "random_prop", 0.2,
    "Proportion of neighborhoods to randomly `prune`.")
tf.flags.DEFINE_float(
    "stop_prob", 0.75,
    "Probability of stopping when generating random pair.")
tf.flags.DEFINE_integer(
    "max_rand_len", 10,
    "Maximum length for randomly generated examples.")
tf.flags.DEFINE_string(
    "pairwise_algo", "basic",
    "Type of the pairwise alignment: `basic` (Kruskal) or `lingpy`.")
tf.flags.DEFINE_string(
    "random_target_algo", "basic",
    "Use randomly sampled target (`basic`) or the target "
    "generated from 2nd order Markov chain (`markov`).")
tf.flags.DEFINE_integer(
    "num_duplicates", 500,
    "Number of duplicates of data.")
tf.flags.DEFINE_integer(
    "num_random_per_real", 10,
    "Number of random examples to generate.")
tf.flags.DEFINE_string(
    "language_group", "hattorijaponic",
    "Which language group.")
tf.flags.DEFINE_string(
    "data_division", "0.10", "Which data division.")
tf.flags.DEFINE_string(
    "lang", "Kyoto",
    "Specific language to train. Set this to `All` to create the data for "
    "training the entire language group rather than a single language. The "
    "match is case-insensitive.")
tf.flags.DEFINE_string(
    "output_dir", "/tmp",
    "Output directory for TFRecord datasets and symbol table.")
tf.flags.DEFINE_bool(
    "has_test", True,
    "Does the set include the test data?")
tf.flags.DEFINE_boolean(
    "append_eos", True,
    "Append </s> symbol.")

FLAGS = tf.flags.FLAGS


# Minimal number of examples above which `lingpy` does not hang.
_MIN_EXAMPLES_FOR_BIGRAMS = 3


class SymbolTable:
  """Simple class to hold input and output symbols."""
  # Note that "<pad>" should get the zero label.
  _OBLIGATORY_SYMBOLS = ["<pad>", "<s>", "</s>", "<unk>"]

  def __init__(self):
    self._symbols = set()

  def add_symbols(self, inp, out):
    """Adds symbols from input and output.

    In this task, input is just the language name so can be split on
    code points, but the output is space-delimited. Note that "out" is
    already split.

    Args:
       inp: a string
       out: a list of symbols
    """
    for char in inp:
      if char.isspace():
        self._symbols.add("<spc>")
      else:
        self._symbols.add(char)
    for phon in filter(None, out):
      self._symbols.add(phon)

  def _write_symbols(self, strm, symbols):
    symbols = sorted(list(symbols))
    label = 0
    for symbol in self._OBLIGATORY_SYMBOLS + symbols:
      strm.write("{}\t{}\n".format(symbol, label))
      label += 1

  def write_symbols(self, strm):
    self._write_symbols(strm, self._symbols)

  def size(self):
    return len(self._symbols) + len(self._OBLIGATORY_SYMBOLS)


class BestMatches:
  """Mathers for alignments and random feature generation."""

  def __init__(self, input_lang, target_lang):
    # The neighbor language is specified by the `input_lang`, the target
    # language is the `main' language for which we are predicting the
    # pronunciation.
    self._alignments = collections.defaultdict(list)
    self._input_lang = input_lang
    self._target_lang = target_lang
    self._markov_chain = None

  def _pairwise_align(self, input_tokens, output_tokens):
    """Performs pairwise alignment between two lists."""
    if FLAGS.pairwise_algo == "basic":
      _, path = kruskal.best_match(input_tokens, output_tokens)
      return path
    elif FLAGS.pairwise_algo == "lingpy":
      out = lp.align.Pairwise(input_tokens, output_tokens)()
      assert out
      out = out[0]
      assert out[0]
      assert len(out[0]) == len(out[1])
      path = []
      for i in range(len(out[0])):
        inp_tok = None if out[0][i] == "-" else out[0][i]
        out_tok = None if out[1][i] == "-" else out[1][i]
        path.append((inp_tok, out_tok))
      return path
    else:
      raise ValueError(
          f"Invalid pairwise algorithm type: {FLAGS.pairwise_algo}")

  def add_match(self, inp, out):
    path = self._pairwise_align(inp.split(), out.split())
    for (i, o) in path:
      if i:
        self._alignments[i].append(o)

  def _generate_basic(self, max_len=10, stop=0.75):
    """Generates random feature (neighbor, target)."""
    keys = list(self._alignments.keys())
    result = []
    non_null_output = False
    for _ in range(max_len):
      inp = random.choice(keys)
      out = random.choice(self._alignments[inp])
      result.append((inp, out))
      if out:
        non_null_output = True
      if non_null_output and random.random() > stop:
        break
    inp = " ".join(c[0] for c in result if c[0])
    out = " ".join(c[1] for c in result if c[1])
    return inp, out

  def _generate_bigrams(self, max_len=10):
    """Random generation based on second-order Markov chains."""
    min_random_sequence_length = 2
    while True:
      # Note: If the following call hangs, run in unigram mode. This may
      # happen if there isn't enough training data.
      rand_input = self._markov_chain.get_string().split()
      if len(rand_input) >= min_random_sequence_length:
        break
    rand_input = rand_input[0:max_len]
    result = []
    for input_phone in rand_input:
      target_phones = self._alignments[input_phone]
      if target_phones:
        target_phone = random.choice(target_phones)
      else:
        target_phone = None
      result.append((input_phone, target_phone))
    inp = " ".join(c[0] for c in result if c[0])
    out = " ".join(c[1] for c in result if c[1])
    return inp, out

  def generate_neighborhood(self, max_len=10, stop=0.75):
    """Generates random neighborhood feature."""
    if FLAGS.random_target_algo == "basic":
      inp, out = self._generate_basic(max_len, stop)
    elif FLAGS.random_target_algo == "markov":
      # We will also use unigrams if there isn't enough data to train bigrams.
      if self._markov_chain:
        inp, out = self._generate_bigrams(max_len)
      else:
        inp, out = self._generate_basic(max_len, stop)
    else:
      raise ValueError(f"Invalid target randomization algo: "
                       f"{FLAGS.random_target_algo}")
    neighborhood = ["RAND", (self._target_lang, out),
                    (self._input_lang, inp)]
    return neighborhood

  def finalize_init(self, target_prons):
    """Completes the initialization."""
    if FLAGS.random_target_algo == "markov":
      # Train 2nd order Markov chain from target pronunciations.
      if len(target_prons) > _MIN_EXAMPLES_FOR_BIGRAMS:
        tf.logging.info("[%s] Training Markov chain from %d pronunciations ...",
                        self._input_lang, len(target_prons))
        self._markov_chain = lp.sequence.generate.MCPhon(
            [targ.split() for targ in target_prons], tokens=True)
      else:
        tf.logging.info("[%s] Falling back to unigrams ...", self._input_lang)


def is_all_languages():
  """Checks if we generate language-specific or family-specific data."""
  return True if FLAGS.lang.lower() == "all" else False


def best_basename():
  """Returns best basename for the generated file."""
  return FLAGS.language_group if is_all_languages() else FLAGS.lang


def get_language_offset(hdr, target_lang):
  offset = -1
  for i in range(1, len(hdr)):
    if hdr[i] == target_lang:
      offset = i
      break
  assert offset != -1
  return offset


def load_cognate_data():
  path_to_cognate_data = os.path.join(
      FLAGS.task_data_dir, FLAGS.language_group, "cognates.tsv")
  with open(path_to_cognate_data, encoding="utf8") as strm:
    data_reader = csv.reader(strm, delimiter="\t", quotechar='"')
    data = list(data_reader)
    return data[0], data[1:]


def load_training_data():
  """Reads training data."""
  path_to_training_data = os.path.join(
      FLAGS.task_data_dir, FLAGS.language_group,
      f"training-{FLAGS.data_division}.tsv")
  with open(path_to_training_data, encoding="utf8") as strm:
    data_reader = csv.reader(strm, delimiter="\t", quotechar='"')
    data = list(data_reader)
    return data[0], data[1:]


def load_test_data(target_lang):
  """Reads test data."""
  path_to_test_data = os.path.join(
      FLAGS.task_data_dir, FLAGS.language_group,
      f"test-{FLAGS.data_division}.tsv")
  with open(path_to_test_data, encoding="utf8") as strm:
    data_reader = csv.reader(strm, delimiter="\t", quotechar='"')
    data = list(data_reader)
    hdr = data[0]
    offset = get_language_offset(hdr, target_lang)
    rows = []
    for row in data[1:]:
      if row[offset] == "?":
        rows.append(row)
    return hdr, rows


_KEY = 0
_MAX_INPUT = 0
_MAX_OUTPUT = 0


def process_cognates(hdr, data, target_lang):
  offset = get_language_offset(hdr, target_lang)
  cognates = collections.defaultdict(str)
  for row in data:
    # Why in blazes they have "-" in some cases I have no idea.
    cognates[row[0].replace("-", "")] = row[offset]
  return cognates


def _string_to_labels(symbols, input_string, split_on_space=False):
  """Converts string to symbol labels."""
  labels = []
  if not split_on_space:
    for c in input_string:
      if c.isspace():
        idx = symbols.find("<spc>")
      else:
        idx = symbols.find(c)
        if idx < 0:
          idx = symbols.find("<unk>")
      if idx < 0:
        raise ValueError(f"Failed to index symbol {c}")
      labels.append(idx)
  else:
    for tok in filter(None, input_string.split()):
      idx = symbols.find(tok)
      if idx < 0:
        idx = symbols.find("<unk>")
      labels.append(idx)
  if FLAGS.append_eos:
    idx = symbols.find("</s>")
    if idx < 0:
      raise ValueError("End-of-sentence symbol not found!")
    labels.append(idx)
  return labels


def _encode_name(symbols, name):
  """Encodes name as an int list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(
      value=_string_to_labels(symbols, name)))


def _encode_pron(symbols, pron):
  """Encodes pron as an int list (for now)."""
  return tf.train.Feature(int64_list=tf.train.Int64List(
      value=_string_to_labels(symbols, pron,
                              split_on_space=True)))


def _encode_cognate_set(symbols, cognate_id, main_feature, neighbor_features):
  """Encodes cognate set as `tf.train.SequenceExample` proto."""
  # Encode main feature as context feature.
  main_feature = tf.train.Features(feature={
      "cognate_id": tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[str.encode(cognate_id)])),
      "main_name": _encode_name(symbols, main_feature[0]),
      "main_pron": _encode_pron(symbols, main_feature[1]),
  })

  # Encode neighbors as feature lists.
  neighbors_names = tf.train.FeatureList(
      feature=[
          _encode_name(symbols, name) for name, _ in neighbor_features])
  neighbors_prons = tf.train.FeatureList(
      feature=[
          _encode_pron(symbols, pron) for _, pron in neighbor_features])
  neighbors_dict = {
      "neighbor_names": neighbors_names,
      "neighbor_prons": neighbors_prons,
  }
  neighbor_features = tf.train.FeatureLists(feature_list=neighbors_dict)
  return tf.train.SequenceExample(context=main_feature,
                                  feature_lists=neighbor_features)

def next_key(advance=True):
  global _KEY
  if advance:
    _KEY = _KEY + 1
  return "{:016x}".format(_KEY)


def inspect_feature(inp, out):
  inp = " ".join(inp.split())
  splout = out.split()
  global _MAX_INPUT
  global _MAX_OUTPUT
  if len(inp) > _MAX_INPUT:
    _MAX_INPUT = len(inp)
  if len(splout) > _MAX_OUTPUT:
    _MAX_OUTPUT = len(splout)


def write_neighbors(neighborhoods, symbols, suffix):
  """Writes the augmented data in TFRecord format."""
  outfile = os.path.join(FLAGS.output_dir,
                         "%s_%s.tfrecords" % (best_basename(), suffix))
  options = tf.io.TFRecordOptions()  # Defaults, for now.
  tf.logging.info("%s: Writing records ...", suffix)
  num_records = 0
  with tf.io.TFRecordWriter(outfile, options) as writer:
    for neighbors in neighborhoods:
      cogset = neighbors[0]
      inp, out = neighbors[1]
      neighbors = neighbors[2:]
      main_spelling = inp
      main_pron = out
      inspect_feature(main_spelling, main_pron)
      for spelling, pron in neighbors:
        inspect_feature(spelling, pron)
      record_key = f"{next_key()}_{cogset}"
      if suffix == "test" and FLAGS.has_test:
        # Keep the COGID as is for the test data.
        record_key = cogset
      record = _encode_cognate_set(
          symbols, record_key, (main_spelling, main_pron),
          neighbors)
      writer.write(record.SerializeToString())
      num_records += 1
  tf.logging.info("%s: Wrote %d records to %s",
                  suffix, num_records, outfile)


def create_neighborhood_for_language(hdr, data, symbols, target_lang,
                                     is_training=True, cognates=None,
                                     num_duplicates=1):
  """Creates data for the supplied language possibly augmenting it."""
  tf.logging.info("%s: Generating data for %s ...",
                  "train" if is_training else "test", target_lang)
  offset = get_language_offset(hdr, target_lang)

  def randomly_subset_neighbors(neighborhood):
    if random.random() > FLAGS.random_prop:
      return neighborhood
    cogset = neighborhood[0]
    main_feat = neighborhood[1]
    rest = neighborhood[2:]
    random.shuffle(rest)
    rest = rest[:random.randint(1, len(rest) + 1)]
    neighborhood = [cogset, main_feat] + rest
    return neighborhood

  def expand_neighborhoods(neighborhoods, best_matchers):
    keys = list(best_matchers.keys())
    new_neighborhoods = []
    for n in neighborhoods:
      new_neighborhoods.append(n)
      for _ in range(FLAGS.num_random_per_real):
        matcher = best_matchers[random.choice(keys)]
        new_neighborhoods.append(
            matcher.generate_neighborhood(max_len=FLAGS.max_rand_len,
                                          stop=FLAGS.stop_prob))
    return new_neighborhoods

  if is_training:
    suffix = "train"
  else:
    suffix = "test"
  neighborhoods_ = []
  best_matchers = {}
  language_prons = collections.defaultdict(set)
  for row in data:
    cogset = row[0]
    # Remove leading "-" if any: Lord knows why some lines have
    # this...TODO(rws): This should now have been taken care of above.
    if cogset.startswith("-"):
      cogset = cogset[1:]
    orig_cogset = cogset
    cogset = cogset.split("-")[0]  # Remove language offset tag if present
    targ = row[offset].strip()
    if targ == "?" and cognates:
      targ = cognates[cogset]
    if FLAGS.has_test and not is_training:
      # For the test set, keep the full COGIDs that include language offset.
      cogset = orig_cogset
    if not targ:
      continue
    lang = f"{target_lang}"
    neighborhood = [cogset, (lang, targ)]
    for i in range(1, len(row)):
      if i == offset:
        continue
      cog = row[i].strip()
      if not cog:
        continue
      inplang = f"{hdr[i]}"
      neighborhood.append((inplang, cog))
      if is_training:
        if inplang not in best_matchers:
          best_matchers[inplang] = BestMatches(inplang, target_lang)
        best_matchers[inplang].add_match(cog, targ)
        language_prons[inplang].add(cog)
    neighborhoods_.append(neighborhood)

  if is_training:
    for lang in best_matchers:
      best_matchers[lang].finalize_init(list(language_prons[lang]))

  neighborhoods = neighborhoods_.copy()
  if is_training:
    neighborhoods = expand_neighborhoods(neighborhoods, best_matchers)
  for _ in range(num_duplicates - 1):
    random.shuffle(neighborhoods_)
    extensions = [randomly_subset_neighbors(n) for n in neighborhoods_]
    if is_training:
      extensions = expand_neighborhoods(extensions, best_matchers)
    neighborhoods.extend(extensions)

  return neighborhoods.copy()


def _load_symbols():
  """Loads combined pynini symbol table."""
  path = os.path.join(FLAGS.output_dir, "%s.syms" % best_basename())
  if not os.path.isfile(path):
    raise ValueError(f"Expected symbol table in {path}")
  tf.logging.info("Loading symbols from %s ...", path)
  return pynini.SymbolTable.read_text(path)


def main(unused_argv):
  try:
    os.mkdir(FLAGS.output_dir)
  except OSError:
    pass

  # Fill in symbol table.
  tf.logging.info("Building symbol table ...")
  symbol_table = SymbolTable()
  hdr, data = load_training_data()
  for row in data:
    for col_id, pron in enumerate(row[1:]):
      symbol_table.add_symbols(hdr[col_id + 1], pron.split())

  all_languages = is_all_languages()
  languages = hdr[1:] if all_languages else [FLAGS.lang]
  if all_languages:
    tf.logging.info("Languages: %s", languages)
  if FLAGS.has_test:
    for language in languages:
      hdr, data = load_test_data(language)
      for row in data:
        for col_id, pron in enumerate(row[1:]):
          symbol_table.add_symbols(hdr[col_id + 1], pron.split())
  syms_path = os.path.join(FLAGS.output_dir, "%s.syms" % best_basename())
  with open(syms_path, "w", encoding="utf8") as strm:
    symbol_table.write_symbols(strm)
  tf.logging.info("Wrote %d symbols to %s ...",
                  symbol_table.size(), syms_path)
  symbols = _load_symbols()

  # Generate training data.
  tf.logging.info("Generating training data ...")
  cog_hdr, cog_data = load_cognate_data()
  all_neighborhoods = []
  for language in languages:
    cognates = process_cognates(cog_hdr, cog_data, language)
    hdr, data = load_training_data()
    neighborhoods = create_neighborhood_for_language(
        hdr, data, symbols, language, is_training=True, cognates=None,
        num_duplicates=FLAGS.num_duplicates)
    all_neighborhoods.extend(neighborhoods)
  random.shuffle(all_neighborhoods)
  write_neighbors(all_neighborhoods, symbols, "train")

  # Generate test data.
  tf.logging.info("Generating test data ...")
  all_neighborhoods = []
  for language in languages:
    cognates = process_cognates(cog_hdr, cog_data, language)
    if FLAGS.has_test:
      hdr, data = load_test_data(language)
    else:  # In case we don't have a designated test set.
      hdr, data = load_training_data()
    neighborhoods = create_neighborhood_for_language(
        hdr, data, symbols, language,
        is_training=False, cognates=cognates, num_duplicates=1)
    all_neighborhoods.extend(neighborhoods)
  write_neighbors(all_neighborhoods, symbols, "test")

  tf.logging.info("Max spelling length is %d", _MAX_INPUT)
  tf.logging.info("Max pronunciation length is %d", _MAX_OUTPUT)
  tf.logging.info("Max neighbors: %d", len(cog_hdr) - 2)


if __name__ == "__main__":
  tf.app.run(main)
