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

"""Tests for lexical_accuracy."""

from absl.testing import absltest

from frmt import lexical_accuracy


class LexicalAccuracyTest(absltest.TestCase):

  def test_score_terms(self):
    a = lexical_accuracy._score_terms(
        # Note repeated word 'celular' to test default clipping.
        ["Perdi meu celular celular.", "Ela alimentou o cachorro."],
        matched_terms=["celular"],
        mismatched_terms=["telemóvel"])
    b = lexical_accuracy._score_terms(
        ["Perdi meu celular.", "Ela alimentou o cachorro."],
        matched_terms=["celular"],
        mismatched_terms=["telemóvel"])
    exp = lexical_accuracy.TermCount(1, 0)
    self.assertEqual(a, exp)
    self.assertEqual(b, exp)

  def test_score_terms_regex(self):
    with self.subTest("use_regex=False"):
      # 'abc' is substring match
      self.assertEqual(
          lexical_accuracy._score_terms(["abcdef"],
                                        matched_terms=["abc"],
                                        mismatched_terms=["xyz"],
                                        use_regex=False),
          lexical_accuracy.TermCount(1, 0))
    with self.subTest("use_regex=True"):
      # 'abc' does not match at word boundaries.
      self.assertEqual(
          lexical_accuracy._score_terms(["abcdef"],
                                        matched_terms=["abc"],
                                        mismatched_terms=["xyz"],
                                        use_regex=True),
          lexical_accuracy.TermCount(0, 0))

  def test_score_pt(self):
    # For each corpus, the first segment is not lemmatized and the second is.
    results_br, results_pt = lexical_accuracy.score_pt(
        corpus_br=[
            "O café da manhã na África varia de região para região.",
            "paraíso de batedor de carteira"
        ],
        corpus_pt=[
            "O pequeno-almoço em África varia muito de região para região.",
            "zona de perigo ideal para o carteirista"
        ])
    # BR and PT expectations differ in this case.
    for word, scores in results_br.items():
      if word == "Breakfast":
        expected = lexical_accuracy.TermCount(matched=1, mismatched=0)
      elif word == "Pickpocket":
        # Not found "..de carteira" vs dictionary's "..de carteiras".
        expected = lexical_accuracy.TermCount(matched=0, mismatched=0)
      else:
        expected = lexical_accuracy.TermCount(matched=0, mismatched=0)
      self.assertEqual(
          scores, expected, msg=f"Unexpected pt-BR result for {word}.")
    for word, scores in results_pt.items():
      if word == "Breakfast":
        expected = lexical_accuracy.TermCount(matched=1, mismatched=0)
      elif word == "Pickpocket":
        # Found.
        expected = lexical_accuracy.TermCount(matched=1, mismatched=0)
      else:
        expected = lexical_accuracy.TermCount(matched=0, mismatched=0)
      self.assertEqual(
          scores, expected, msg=f"Unexpected pt-PT result for {word}.")

  def test_score_zh(self):
    results_cn, results_tw = lexical_accuracy.score_zh(
        corpus_cn=["我们在悉尼。"],
        corpus_tw=["我們在雪梨。"],
        script_cn=lexical_accuracy.ZhScript.SIMPLIFIED,
        script_tw=lexical_accuracy.ZhScript.TRADITIONAL,
    )
    for i, result in enumerate((results_cn, results_tw)):
      for word, scores in result.items():
        # print(word, scores)
        if word == "Sydney":
          expected = lexical_accuracy.TermCount(matched=1, mismatched=0)
        else:
          expected = lexical_accuracy.TermCount(matched=0, mismatched=0)
        self.assertEqual(scores, expected, msg=str(i))

  def test_compute_summary(self):
    score = lexical_accuracy.compute_summary([{
        "en_word1": lexical_accuracy.TermCount(1, 0),
        "en_word2": lexical_accuracy.TermCount(1, 1),
    }, {
        "en_word1": lexical_accuracy.TermCount(1, 0),
        "en_word2": lexical_accuracy.TermCount(1, 0),
    }])
    self.assertAlmostEqual(score, 4 / 5)

  def test_compute_summary_no_hits(self):
    score = lexical_accuracy.compute_summary([{
        "en_word1": lexical_accuracy.TermCount(0, 0),
    }, {
        "en_word1": lexical_accuracy.TermCount(0, 0),
    }])
    self.assertAlmostEqual(score, 0)

  def test_run_pt_eval(self):
    input_br = self.create_tempfile(
        content="Perdi meu celular .\nEla alimentou o cachorro .")
    input_pt = self.create_tempfile(
        content="Perdi meu telemóvel .\nEla alimentou o cachorro .")
    output = self.create_tempdir()
    score, _ = lexical_accuracy.run_pt_eval_from_files(
        input_br, input_pt, f"{output.full_path}/out")
    self.assertAlmostEqual(score, 1.0)

    with open(f"{output.full_path}/out_terms.csv") as f:
      csv = f.read()
    print(csv)

  def test_run_zh_eval(self):
    input_cn = self.create_tempfile(
        # "I am in Sydney." (using CN term)
        # "My dog is also in Sydney." (using TW term)
        content="我们在悉尼。\n我的狗也在雪梨。")
    input_tw = self.create_tempfile(
        # "I am in Sydney." (using TW term)
        # "My dog is also in Sydney." (using TW term)
        content="我們在雪梨。\n我的狗也在雪梨。")
    output = self.create_tempdir()

    # Run eval.
    score, _ = lexical_accuracy.run_zh_eval_from_files(
        input_cn,
        input_tw,
        script_cn=lexical_accuracy.ZhScript.SIMPLIFIED,
        script_tw=lexical_accuracy.ZhScript.TRADITIONAL,
        output_path=f"{output.full_path}/out")
    self.assertAlmostEqual(score, 3 / 4)

    with open(f"{output.full_path}/out_terms.csv") as f:
      csv = f.read()
    print(csv)


if __name__ == "__main__":
  absltest.main()
