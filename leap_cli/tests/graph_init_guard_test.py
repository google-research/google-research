# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for graph_init_guard."""

import _path  # noqa: F401  (puts leap/ on sys.path)
from absl.testing import absltest

import graph_init_guard


class CheckPreambleTest(absltest.TestCase):

  def test_accepts_a_real_file_header(self):
    self.assertEqual(
        graph_init_guard.check_preamble(
            "import Mathlib\n\nopen Finset\n\nabbrev f : ℕ := 0", "thm"
        ),
        "",
    )

  def test_accepts_a_bare_import(self):
    self.assertEqual(graph_init_guard.check_preamble("import Mathlib", "thm"), "")

  def test_rejects_an_empty_preamble(self):
    error = graph_init_guard.check_preamble("", "putnam_2025_a5")
    self.assertIn("is empty", error)
    self.assertIn("putnam_2025_a5", error)

  def test_rejects_a_whitespace_only_preamble(self):
    self.assertIn("is empty", graph_init_guard.check_preamble("  \n\t\n ", "thm"))

  def test_rejects_a_preamble_with_no_import(self):
    error = graph_init_guard.check_preamble("open Finset\n\ndef f : ℕ := 0", "thm")
    self.assertIn("no 'import' line", error)

  def test_a_commented_out_import_is_not_an_import(self):
    self.assertIn(
        "no 'import' line",
        graph_init_guard.check_preamble("-- import Mathlib", "thm"),
    )

  def test_every_rejection_says_how_to_fix_it(self):
    for preamble in ("", "open Finset"):
      self.assertIn(
          "ACTION REQUIRED", graph_init_guard.check_preamble(preamble, "thm")
      )

  def test_rejects_a_dangling_open_in(self):
    # The split leaves the modifier with nothing
    # to modify, so it binds to the first helper a prover adds instead of to
    # the root theorem.
    error = graph_init_guard.check_preamble(
        "import Mathlib\n\nopen Classical SimpleGraph in", "bollobas"
    )
    self.assertIn("dangling 'in'", error)
    self.assertIn("ACTION REQUIRED", error)

  def test_a_trailing_in_inside_a_comment_is_not_dangling(self):
    self.assertEqual(
        graph_init_guard.check_preamble(
            "import Mathlib\n\nopen Finset\n-- edges in", "thm"
        ),
        "",
    )

  def test_an_identifier_ending_in_in_is_not_dangling(self):
    self.assertEqual(
        graph_init_guard.check_preamble(
            "import Mathlib\n\nabbrev margin : ℕ := 0", "thm"
        ),
        "",
    )


class CheckRootGateTest(absltest.TestCase):

  def test_accepts_a_plain_signature(self):
    self.assertEqual(
        graph_init_guard.check_root_gate("theorem foo (n : ℕ) : n = n"), ""
    )

  def test_accepts_a_signature_carrying_comments(self):
    self.assertEqual(
        graph_init_guard.check_root_gate(
            "theorem zs (a b : ℕ) :\n    (a = 1) ∨  -- exception 1\n    (b = 2)"
        ),
        "",
    )

  def test_accepts_a_signature_carrying_a_let(self):
    self.assertEqual(
        graph_init_guard.check_root_gate(
            "theorem lt (n : ℕ) :\n    let k := n + 1\n    k > n"
        ),
        "",
    )

  def test_rejection_names_the_tool_not_the_statement(self):
    # A signature the gate cannot match: the scaffold renders `... :=` but the
    # trailing `:=` makes the expected token list unmatchable.
    error = graph_init_guard.check_root_gate("theorem foo : True :=")
    if error:
      self.assertIn("bug in the tool", error)


if __name__ == "__main__":
  absltest.main()
