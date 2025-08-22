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

"""Converts a Wiktionary dump into language-specific word lists."""

import collections
from collections.abc import Sequence
import dataclasses
import itertools

from absl import app
from absl import flags

WIKTIONARY_DUMP_PATH = flags.DEFINE_string(
    "wiktionary_dump_path",
    default="/path/to/enwiktionary-20220820-pages-articles.xml",
    help="The path to the input Wiktionary dump file.",
)
OUTPUT_ROOT = flags.DEFINE_string(
    "output_root",
    help="The directory where all output files should be written.",
    required=True,
)

LANGUAGES = flags.DEFINE_list(
    "languages",
    default=["ar", "en", "fi", "ko", "ru", "th", "zh"],
    help="Languages to process, as a comma-separated list of language codes.",
)


LANG_NAMES_TO_CODES = {
    "Arabic": "ar",
    "English": "en",
    "Finnish": "fi",
    "Korean": "ko",
    "Russian": "ru",
    "Thai": "th",
    "Chinese": "zh",
}


@dataclasses.dataclass
class Data:
  line_num: int
  title: str = ""
  langs: set[str] = dataclasses.field(default_factory=set)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  wiktionary_dump_path = WIKTIONARY_DUMP_PATH.value
  output_root = OUTPUT_ROOT.value
  if LANGUAGES.value == ["ALL"]:
    languages = set(LANG_NAMES_TO_CODES.values())
  else:
    languages = set(LANGUAGES.value)

  word_langs_path = f"{output_root}/word_langs.txt"

  with open(word_langs_path, "w") as f_out:
    with open(wiktionary_dump_path) as f_in:
      buffer = None
      for i, line in itertools.islice(enumerate(f_in), 0, None):
        line = line.rstrip("\n")

        if line == "  <page>":
          if buffer is not None:
            raise ValueError(f"<page> not closed: title={buffer.title!r}")
          buffer = Data(line_num=i)

        if buffer is None:
          continue

        if line.startswith("    <title>"):
          buffer.title = line[len("    <title>") : -len("</title>")]
          if " " in buffer.title:
            buffer = None
          continue
        if line.startswith("    <ns>"):
          if line != "    <ns>0</ns>":
            buffer = None
          continue
        if line == "  </page>":
          if buffer.langs:
            f_out.write(f"{buffer.title}\t{','.join(sorted(buffer.langs))}\n")
          buffer = None
          continue

        for dubious in (
            "===Proverb===",
            "Pronunciation spelling",
            "(clitic)",
        ):
          if dubious in line:
            buffer = None

        # TODO(dhgarrette): Do a first pass in which we make a set of all
        #     potentially offensive words. Then reject all words in the list,
        #     but also follow links like "Plural of X" and "Alternative form
        #     of X", and reject words word that link to offensive words.
        #     Alternatively, reject words that are within a short edit
        #     distance from an offensive word?
        for banned in (
            "vulgar",
            "epithet",
            "profan",
            "offensiv",
            "derogator",
        ):
          if banned in line:
            buffer = None

        if buffer is not None:
          for lang_name, lang_code in LANG_NAMES_TO_CODES.items():
            if f"=={lang_name}==" in line:
              buffer.langs.add(lang_code)

  words_by_lang = collections.defaultdict(set)
  with open(word_langs_path) as f_in:
    for line in f_in:
      line = line.rstrip("\n")
      word, langs_str = line.split("\t")
      langs = set(langs_str.split(","))
      if "en" in langs:
        langs = {"en"}
      if "zh" in langs:
        langs.discard("ko")  # Remove Korean entries that are really Chinese.
      for lang in langs:
        words_by_lang[lang].add(word)

  for lang, words in sorted(words_by_lang.items()):
    print(f"{lang}  {len(words):7,d}")

    if lang not in languages:
      continue
    with open(f"{output_root}/{lang}.words.txt", "w") as f_out:
      for word in words:
        f_out.write(f"{word}\n")


if __name__ == "__main__":
  app.run(main)
