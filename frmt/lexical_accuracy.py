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

"""Library to compute simple region-based lexical accuracy metric."""

import collections
import csv
import dataclasses
import enum
import pathlib
import re
from typing import IO, Iterable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from immutabledict import immutabledict


# The following maps define the terms of interest, indexed on the original
# English seed term.

# Format: English: (Simp-CN, Simp-TW, Trad-TW, Trad-CN)
# Spaces are stripped from the Chinese corpus before matching these terms.
_CHINESE_TERMS = immutabledict({
    "Pineapple": ("菠萝", "凤梨", "鳳梨", "菠蘿"),
    "Computer mouse": ("鼠标", "滑鼠", "滑鼠", "鼠標"),
    # Original source had CN:牛油果, but translator used 鳄梨.
    "Avocado": ("鳄梨", "酪梨", "酪梨", "鱷梨"),
    "Band-Aid": ("创可贴", "OK绷", "OK繃", "創可貼"),
    "Blog": ("博客", "部落格", "部落格", "博客"),
    "New Zealand": ("新西兰", "纽西兰", "紐西蘭", "新西蘭"),
    "Printer (computing)": ("打印机", "印表机", "印表機", "打印機"),
    # Original source has TW:月臺, but translator used 月台.
    "Railway platform": ("站台", "月台", "月台", "站台"),
    "Roller coaster": ("过山车", "云霄飞车", "雲霄飛車", "過山車"),
    "Salmon": ("三文鱼", "鲑鱼", "鮭魚", "三文魚"),
    "Shampoo": ("洗发水", "洗发精", "洗髮精", "洗髮水"),
    # From Wikipedia page "Software testing"
    "Software": ("软件", "软体", "軟體", "軟件"),
    "Sydney": ("悉尼", "雪梨", "雪梨", "悉尼"),

    # The following two are excluded because they underpin the first 100
    # lexical exemplars used for priming the models.
    ## "Flip-flops": ("人字拖", "夹脚拖", "夾腳拖", "人字拖"),
    ## "Paper clip": ("回形针", "回纹针", "迴紋針", "回形針"),
})

# Portuguese terms.
# Format: English: (BR, PT)
# The Portuguese corpus is lowercased before matching these terms.
_PORTUGUESE_TERMS = immutabledict({
    "Bathroom": ("banheiro", "casa de banho"),
    # Original source had "pequeno almoço" but translator used "pequeno-almoço".
    "Breakfast": ("café da manhã", "pequeno-almoço"),
    "Bus": ("ônibus", "autocarro"),
    "Cup": ("xícara", "chávena"),
    "Computer mouse": ("mouse", "rato"),
    "Drivers license": ("carteira de motorista", "carta de condução"),
    # From Wikipedia page "Ice cream sandwich"
    "Ice cream": ("sorvete", "gelado"),
    "Juice": ("suco", "sumo"),
    "Mobile phone": ("celular", "telemóvel"),
    "Pedestrian": ("pedestre", "peão"),
    # From Wikipedia page "Pickpocketing"
    "Pickpocket": ("batedor de carteiras", "carteirista"),
    "Pineapple": ("abacaxi", "ananás"),
    "Refrigerator": ("geladeira", "frigorífico"),
    "Suit": ("terno", "fato"),
    "Train": ("trem", "comboio"),
    "Video game": ("videogame", "videojogos"),

    # Terms updated after original selection.

    # For BR, replaced "menina" (common in speech) with "garota" (common in
    # writing, matching the human translators.
    "Girl": ("garota", "rapariga"),

    # Replace original "Computer monitor": ("tela de computador", "ecrã") with
    # the observed use for just screen:
    "Screen": ("tela", "ecrã"),

    # Terms excluded.

    # The following three are excluded because they underpin the first 100
    # lexical exemplars used for priming the models.
    ## "Gym": ("academia", "ginásio"),
    ## "Stapler": ("grampeador", "agrafador"),
    ## "Nightgown": ("camisola", "camisa de noite"),

    # The following are excluded for other reasons:

    # BR translator primarily used 'comissário de bordo' and hardly ever
    # 'aeromoça'. PT translator used 'comissários/assistentes de bordo' or just
    # 'assistentes de bordo' Excluding the term as low-signal for now.
    ## "Flight attendant": ("aeromoça", "comissário ao bordo"),

    # Both regions' translators consistently used "presunto", so the term has
    # low signal.
    ## "Ham": ("presunto", "fiambre"),
})

_StrOrPurePath = Union[str, pathlib.PurePath]


def _open_file(path: _StrOrPurePath, mode: str = "r") -> IO[str]:
  return open(path, mode)  # pylint: disable=unreachable


@dataclasses.dataclass(frozen=True)
class TermCount:
  matched: int
  mismatched: int


TermCounts = Mapping[str, TermCount]


class ZhScript(enum.Enum):
  SIMPLIFIED = 1
  TRADITIONAL = 2


def _count_term_hits_with_regex(text: str, term: str) -> int:
  # Avoids overtriggering when term happens to be a substring in unrelated
  # words.
  pattern = r"\b" + term + r"\b"
  return len(re.findall(pattern, text))


def _score_terms(
    corpus: Sequence[str],
    matched_terms: Iterable[str],
    mismatched_terms: Iterable[str],
    per_example_cap: int = 1,
    use_regex: bool = True,
) -> TermCount:
  """Scores term by counting number of non-overlapping substring occurrences."""
  matched_total = 0
  mismatched_total = 0

  def _count(sentence: str, term: str) -> int:
    if use_regex:
      return _count_term_hits_with_regex(sentence, term)
    else:
      return sentence.count(term)

  for sentence in corpus:
    matched_term_counts = [
        _count(sentence, matched_term) for matched_term in matched_terms
    ]
    matched_count = min(sum(matched_term_counts), per_example_cap)
    matched_total += matched_count
    mismatched_term_counts = [
        _count(sentence, mismatched_term)
        for mismatched_term in mismatched_terms
    ]
    mismatched_count = min(sum(mismatched_term_counts), per_example_cap)
    mismatched_total += mismatched_count

    for matched_term, matched_term_count in zip(matched_terms,
                                                matched_term_counts):
      if matched_term_count > 0:
        logging.debug("Hit (match) '%s': %s", matched_term, sentence)
    for mismatched_term, mismatched_term_count in zip(mismatched_terms,
                                                      mismatched_term_counts):
      if mismatched_term_count > 0:
        logging.debug("Hit (mismatch) '%s': %s", mismatched_term, sentence)

  return TermCount(matched=matched_total, mismatched=mismatched_total)


def score_corpus(
    corpus: Iterable[str],
    terms: Mapping[str, Tuple[Iterable[str], Iterable[str]]],
    use_regex: bool = True,
) -> TermCounts:
  r"""Counts occurrences of matching and non-matching terms in corpus.

  Args:
    corpus: Text to evaluate, in the target language, as an iterable over e.g.
      sentences.
    terms: Map from a source language term to its target language translations
      (x, y) such that x is matched to the regional variant of the corpus, and y
      is mis-matched to the regional variant of the corpus. x and y are
      iterables that may contain alternative orthographic realizations of the
      term variant.
    use_regex: Whether to do term matching using a regex that requires word
      boundaries (\b) around the search term, otherwise uses str.count(). Set to
      False if the corpus is in a non-spaced language like Chinese.

  Returns:
    Map from each source language term to a TermCount, recording the number
    of occurrences of the matched and mismatched terms in the corpus.
  """
  corpus = list(corpus)
  return {
      source_word:
      _score_terms(corpus, matching, mismatching, use_regex=use_regex)
      for source_word, (matching, mismatching) in terms.items()
  }


def score_pt(corpus_br: Iterable[str],
             corpus_pt: Iterable[str]) -> Tuple[TermCounts, TermCounts]:
  """Calls score_corpus using the hardcoded list of Portuguese terms."""
  # _PORTUGUESE_TERMS is already organized as (match, mismatch)-pairs for pt-BR,
  # but needs to be converted from strings to lists of strings
  counts_br = score_corpus(
      corpus_br,
      terms={
          word: ([br], [pt]) for word, (br, pt) in _PORTUGUESE_TERMS.items()
      })

  # _PORTUGUESE_TERMS must be reorganized as (match, mismatch)-pairs for pt-PT.
  counts_pt = score_corpus(
      corpus_pt,
      terms={
          word: ([pt], [br]) for word, (br, pt) in _PORTUGUESE_TERMS.items()
      })

  return counts_br, counts_pt


def score_zh(corpus_cn: Iterable[str], corpus_tw: Iterable[str],
             script_cn: Optional[ZhScript],
             script_tw: Optional[ZhScript]) -> Tuple[TermCounts, TermCounts]:
  """Calls score_corpus using the hardcoded list of Chinese terms."""
  # Reformat the Chinese term dictionary into (match, mismatch)-pairs for each
  # corpus, based on the script of the corpus.
  terms_for_cn = {}
  for word, (simp_cn, simp_tw, trad_tw, trad_cn) in _CHINESE_TERMS.items():
    if script_cn == ZhScript.SIMPLIFIED:
      terms_for_cn[word] = ([simp_cn], [simp_tw])
    elif script_cn == ZhScript.TRADITIONAL:
      terms_for_cn[word] = ([trad_cn], [trad_tw])
    else:
      terms_for_cn[word] = ([simp_cn, trad_cn], [simp_tw, trad_tw])
  counts_cn = score_corpus(corpus_cn, terms=terms_for_cn, use_regex=False)

  terms_for_tw = {}
  for word, (simp_cn, simp_tw, trad_tw, trad_cn) in _CHINESE_TERMS.items():
    if script_tw == ZhScript.SIMPLIFIED:
      terms_for_tw[word] = ([simp_tw], [simp_cn])
    elif script_tw == ZhScript.TRADITIONAL:
      terms_for_tw[word] = ([trad_tw], [trad_cn])
    else:
      terms_for_tw[word] = ([simp_tw, trad_tw], [simp_cn, trad_cn])
  counts_tw = score_corpus(corpus_tw, terms=terms_for_tw, use_regex=False)

  return counts_cn, counts_tw


def compute_summary(results: Sequence[TermCounts]) -> float:
  """Returns the matched-fraction when summing over the results."""
  tally_matched = 0
  tally_mismatched = 0
  for corpus_counts in results:
    for term_pair in corpus_counts.values():
      tally_matched += term_pair.matched
      tally_mismatched += term_pair.mismatched

  # Set to zero if there were no hits.
  if tally_matched + tally_mismatched == 0:
    return 0.0

  return tally_matched / (tally_matched + tally_mismatched)


def _to_csv(corpus_results: Sequence[TermCounts], lang_codes: Sequence[str],
            path: str) -> None:
  """Outputs results to CSV, assuming parallel corpus_results & lang_codes."""
  assert len(corpus_results) == len(lang_codes), (corpus_results, lang_codes)
  fieldnames = ["source_word"]
  for lang_code in lang_codes:
    fieldnames.append(f"corpus_{lang_code}_matched")
    fieldnames.append(f"corpus_{lang_code}_mismatched")

  # Create a single row for each source language term.
  rows = collections.defaultdict(dict)
  for corpus_result, lang_code in zip(corpus_results, lang_codes):
    for source_word, term_pair in corpus_result.items():
      rows[source_word]["source_word"] = source_word
      rows[source_word][f"corpus_{lang_code}_matched"] = term_pair.matched
      rows[source_word][f"corpus_{lang_code}_mismatched"] = term_pair.mismatched

  with _open_file(path, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows.values())


def _maybe_read_lines(file_path: Optional[str]) -> Optional[Sequence[str]]:
  if file_path is None:
    return None
  with _open_file(file_path) as file:
    return file.readlines()


def _output(summary_metric: float, term_counts: Tuple[TermCounts, TermCounts],
            lang_codes: Tuple[str, str], output_path: Optional[str]) -> None:
  if output_path is not None:
    _to_csv(term_counts, lang_codes, f"{output_path}_terms.csv")
    with _open_file(f"{output_path}_lex_acc.txt", mode="wt") as out:
      print(summary_metric, file=out)


def run_pt_eval(
    corpus_br: Sequence[str],
    corpus_pt: Sequence[str],
) -> Tuple[float, Tuple[TermCounts, TermCounts]]:
  """Runs lexical accuracy evaluation on Portuguese.

  Includes lowercasing the input corpora.

  Args:
    corpus_br: List of BR-targeted translations, parallel to corpus_pt.
    corpus_pt: List of PT-targeted translations, parallel to corpus_br.

  Returns:
    - summary metric
    - TermCounts for BR-corpus and PT-corpus, in that order.
  """

  # Lowercase the Portuguese inputs.
  corpus_br = [line.strip().lower() for line in corpus_br]
  corpus_pt = [line.strip().lower() for line in corpus_pt]

  assert len(corpus_br) == len(corpus_pt), (
      f"{len(corpus_br)} != {len(corpus_pt)}")
  term_counts = score_pt(corpus_br=corpus_br, corpus_pt=corpus_pt)
  summary_metric = compute_summary(term_counts)
  return summary_metric, term_counts


def run_pt_eval_from_files(
    corpus_br_path: str,
    corpus_pt_path: str,
    output_path: Optional[str],
) -> Tuple[float, Tuple[TermCounts, TermCounts]]:
  """Runs lexical accuracy evaluation on Portuguese from files."""
  with _open_file(corpus_br_path) as file:
    corpus_br = file.readlines()
  with _open_file(corpus_pt_path) as file:
    corpus_pt = file.readlines()
  logging.info("Read %d BR entries from %s", len(corpus_br), corpus_br_path)
  logging.info("Read %d PT entries from %s", len(corpus_pt), corpus_pt_path)

  summary_metric, term_counts = run_pt_eval(
      corpus_br=corpus_br, corpus_pt=corpus_pt)

  # Literal language codes below follows order of the corpus pair in
  # term_counts.
  _output(summary_metric, term_counts, ("br", "pt"), output_path)
  return summary_metric, term_counts


def run_zh_eval(
    corpus_cn: Sequence[str],
    corpus_tw: Sequence[str],
    script_cn: Optional[ZhScript],
    script_tw: Optional[ZhScript],
) -> Tuple[float, Tuple[TermCounts, TermCounts]]:
  """Runs lexical accuracy evaluation on Chinese.

  Includes normalizing away spaces in the input corpora.

  Args:
    corpus_cn: List of CN-targeted translations, parallel to corpus_tw.
    corpus_tw: List of TW-targeted translations, parallel to corpus_cn.
    script_cn: The Chinese script to expect for corpus_cn, determining which
      script's term to use for matching. If None, matches against both scripts.
    script_tw: The Chinese script to expect for corpus_tw, determining which
      script's term to use for matching. If None, matches against both scripts.

  Returns:
    - summary metric
    - TermCounts for CN-corpus and TW-corpus, in that order.
  """
  # Normalize away all spaces, which translators sometimes include in
  # mixed-script words.
  corpus_cn = [line.strip().replace(" ", "") for line in corpus_cn]
  corpus_tw = [line.strip().replace(" ", "") for line in corpus_tw]

  assert len(corpus_cn) == len(corpus_tw), (
      f"{len(corpus_cn)} != {len(corpus_tw)}")
  term_counts = score_zh(
      corpus_cn=corpus_cn,
      corpus_tw=corpus_tw,
      script_cn=script_cn,
      script_tw=script_tw)
  summary_metric = compute_summary(term_counts)
  return summary_metric, term_counts


def run_zh_eval_from_files(
    corpus_cn_path: str,
    corpus_tw_path: str,
    script_cn: Optional[ZhScript],
    script_tw: Optional[ZhScript],
    output_path: Optional[str],
) -> Tuple[float, Tuple[TermCounts, TermCounts]]:
  """Runs lexical accuracy evaluation on Chinese using file paths."""
  with _open_file(corpus_cn_path) as file:
    corpus_cn = file.readlines()
  with _open_file(corpus_tw_path) as file:
    corpus_tw = file.readlines()
  logging.info("Read %d CN entries from %s", len(corpus_cn), corpus_cn_path)
  logging.info("Read %d TW entries from %s", len(corpus_tw), corpus_tw_path)

  summary_metric, term_counts = run_zh_eval(
      corpus_cn=corpus_cn,
      corpus_tw=corpus_tw,
      script_cn=script_cn,
      script_tw=script_tw)

  # Literal language codes below follows order of the corpus pair in
  # term_counts.
  _output(summary_metric, term_counts, ("cn", "tw"), output_path)
  return summary_metric, term_counts
