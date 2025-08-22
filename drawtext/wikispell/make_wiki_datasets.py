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

"""Prepares the WikiSpell dataset."""

from collections.abc import Mapping, Sequence
import itertools
import random
import unicodedata

from absl import app
from absl import flags

WIKISPELL_ROOT = flags.DEFINE_string(
    "wikispell_root",
    help="Root directory containing the WikiSpell data.",
    required=True,
)
OUTPUT_ROOT = flags.DEFINE_string(
    "output_root",
    default="",
    help=(
        "Directory where the utterance data should be written. If empty, "
        "uses the `wikispell_root` flag value."
    ),
)

LANGUAGES = flags.DEFINE_list(
    "languages",
    default=["ar", "en", "fi", "ko", "ru", "th", "zh"],
    help="Languages to process, as a comma-separated list of language codes.",
)

TRAIN_SIZE = flags.DEFINE_integer(
    "train_size",
    default=10_000,
    help="The number of words to sample for each language's the training set.",
)
DEV_SIZE = flags.DEFINE_integer(
    "dev_size",
    default=1_000,
    help="The number of words to sample for each language's the dev set.",
)
TEST_SIZE = flags.DEFINE_integer(
    "test_size",
    default=1_000,
    help="The number of words to sample for each language's the test set.",
)

MAX_CHAR_LENGTH = flags.DEFINE_integer(
    "max_char_length",
    default=None,
    help="If provided, words exceeding this length will be filtered out.",
)


def _is_symbol(char):
  category = unicodedata.category(char)
  return category.startswith("P") or category.startswith("S")


UNICODE_SYMBOL = frozenset(
    c for c in (chr(i) for i in range(0x110000)) if _is_symbol(c)
)


def format_fraction(count, total):
  return f"{count}/{total} ({count / total * 100:.2f}%)"


def load_words_and_counts(lang):
  """Reads `lang`'s word-counts data file, and prepares its contents."""
  word_counts = {}
  num_read = 0
  num_filtered_long = 0
  num_filtered_punct = 0
  with open(f"{WIKISPELL_ROOT.value}/{lang}.word_counts.tsv") as f:
    for line in f:
      num_read += 1
      word, count = line.strip().split("\t")
      if (
          MAX_CHAR_LENGTH.value is not None
          and len(word) > MAX_CHAR_LENGTH.value
      ):
        num_filtered_long += 1
        continue
      if all(c in UNICODE_SYMBOL for c in word):
        num_filtered_punct += 1
        continue
      word_counts[word] = int(count)

  if MAX_CHAR_LENGTH.value is not None:
    print(
        f"  Filtered out entries with >{MAX_CHAR_LENGTH.value} chars: "
        f"{format_fraction(num_filtered_long, num_read)}"
    )

  if num_filtered_punct:
    print(
        "  Filtered out all-punctuation/symbol entries: "
        f"{format_fraction(num_filtered_punct, num_read)}"
    )

  num_zeros = sum(int(c == 0) for c in word_counts.values())
  print(
      "  Number of zero-count words: "
      f"{format_fraction(num_zeros, len(word_counts))}"
  )

  return word_counts


def write_split_file(lang, split_name, words):
  output_root = OUTPUT_ROOT.value or WIKISPELL_ROOT.value
  with open(f"{output_root}/{lang}.{split_name}.tsv", "w") as f:
    for word in words:
      f.write(f"{word}\t{' '.join(word)}\n")


def handle_language(lang):
  """Prepare the data for `lang`."""
  word_counts = load_words_and_counts(lang)
  num_words = len(word_counts)

  # Shuffle the order of the words (to randomize the order within groups of
  # words that have the same count), and then list the words from most
  # frequent to least.
  word_counts_shuffled = random.sample(list(word_counts.items()), k=num_words)
  word_counts_sorted = sorted(word_counts_shuffled, key=lambda x: -x[1])
  words_sorted = [w for w, _ in word_counts_sorted]

  def percentile(p):  # pylint: disable=cell-var-from-loop
    return int(num_words * p)  # pylint: disable=cell-var-from-loop

  pct1 = percentile(0.01)
  pct10 = percentile(0.1)
  pct20 = percentile(0.2)
  pct30 = percentile(0.3)
  pct50 = percentile(0.5)

  eval_set_pools = {
      "top_1_pct": words_sorted[:pct1],
      "1_to_10_pct": words_sorted[pct1:pct10],
      "10_to_20_pct": words_sorted[pct10:pct20],
      "20_to_30_pct": words_sorted[pct20:pct30],
      "bottom_50_pct": words_sorted[pct50:],
  }

  test_sets = {
      name: random.sample(pool, min(TEST_SIZE.value, len(pool)))
      for name, pool in eval_set_pools.items()
  }
  for name, words in test_sets.items():
    write_split_file(lang, f"test.{name}", words)

  dev_sets = {
      name: random.sample(pool, min(DEV_SIZE.value, len(pool)))
      for name, pool in eval_set_pools.items()
  }
  for name, words in dev_sets.items():
    write_split_file(lang, f"dev.{name}", words)

  all_eval_words = {
      *itertools.chain.from_iterable(test_sets.values()),
      *itertools.chain.from_iterable(dev_sets.values()),
  }

  # Sample half the training set uniformly. This half will be biased toward
  # rare words.
  train_set_rare = random.sample(
      list(set(word_counts) - all_eval_words), k=TRAIN_SIZE.value // 2
  )

  # Sample the other half of the training set such that each word's likelihood
  # is proportional to its frequency. This half will be biased toward frequent
  # words.
  excluded_words = {*all_eval_words, *train_set_rare}
  train_pool_words, train_pool_counts = zip(
      *((w, c) for w, c in word_counts.items() if w not in excluded_words)
  )
  train_set_frequent = random.sample(
      train_pool_words, k=TRAIN_SIZE.value // 2, counts=train_pool_counts
  )

  write_split_file(lang, "train", train_set_rare + train_set_frequent)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  random.seed(0)

  for lang in LANGUAGES.value:
    print(f"Language: {lang}")
    handle_language(lang)


if __name__ == "__main__":
  app.run(main)
