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

r"""Evaluate lexical choices in target language outputs relative to a term list.

Outputs lexical accuracy to the console and to '{output}_lex_acc.txt'. More
detailed term counts are written to '{output}_terms.csv'.

Example 1:
python -m frmt.lexical_accuracy_eval \
  --corpus_br=/tmp/br.txt \
  --corpus_pt=/tmp/pt.txt \
  --output=/tmp/lex_eval

Example 2:
python -m frmt.lexical_accuracy_eval \
  --corpus_cn=/tmp/cn.txt \
  --corpus_tw=/tmp/tw.txt \
  --script_cn=simplified \
  --script_tw=traditional \
  --output=/tmp/lex_eval

"""

from collections.abc import Sequence

from absl import app
from absl import flags
from frmt import lexical_accuracy

CORPUS_BR = flags.DEFINE_string(
    "corpus_br", None, "Path to the pt-BR corpus to evaluate, as a text file.")

CORPUS_PT = flags.DEFINE_string(
    "corpus_pt", None, "Path to the pt-PT corpus to evaluate, as a text file.")

CORPUS_CN = flags.DEFINE_string(
    "corpus_cn", None, "Path to the zh-CN corpus to evaluate, as a text file.")

CORPUS_TW = flags.DEFINE_string(
    "corpus_tw", None, "Path to the zh-TW corpus to evaluate, as a text file.")

SCRIPT_CN = flags.DEFINE_enum_class(
    "script_cn", None, lexical_accuracy.ZhScript,
    "The Chinese script of the file passed to --corpus_cn, restricting term "
    "matching to that script. Otherwise leave unset to match terms in either "
    "script.")

SCRIPT_TW = flags.DEFINE_enum_class(
    "script_tw", None, lexical_accuracy.ZhScript,
    "The Chinese script of the file passed to --corpus_tw, restricting term "
    "matching to that script. Otherwise leave unset to match terms in either "
    "script.")

OUTPUT = flags.DEFINE_string(
    "output", None,
    "Output file prefix, producing {output}_terms.csv. and {output}_lex_acc.txt"
)

MESSAGE = ("The flags --corpus_{br,pt} and --corpus_{cn_tw} are mutually "
           "exclusive, but one full pair is required.")


@flags.multi_flags_validator(
    ["corpus_br", "corpus_pt", "corpus_cn", "corpus_tw"], message=MESSAGE)
def _check_mutually_exclusive_inputs(flags_dict):
  input_pt = flags_dict["corpus_br"] and flags_dict["corpus_pt"]
  input_zh = flags_dict["corpus_cn"] and flags_dict["corpus_tw"]
  return input_pt or input_zh


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if CORPUS_BR.value and CORPUS_PT.value:
    score, _ = lexical_accuracy.run_pt_eval_from_files(
        corpus_br_path=CORPUS_BR.value,
        corpus_pt_path=CORPUS_PT.value,
        output_path=OUTPUT.value,
    )
    print("{:.4f}".format(score))
  elif CORPUS_CN.value and CORPUS_TW.value:

    score, _ = lexical_accuracy.run_zh_eval_from_files(
        corpus_cn_path=CORPUS_CN.value,
        corpus_tw_path=CORPUS_TW.value,
        script_cn=SCRIPT_CN.value,
        script_tw=SCRIPT_TW.value,
        output_path=OUTPUT.value,
    )
    print("{:.4f}".format(score))
  else:
    raise ValueError(MESSAGE)


if __name__ == "__main__":
  app.run(main)
