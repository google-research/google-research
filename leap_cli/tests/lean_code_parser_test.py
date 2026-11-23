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

"""Tests for the Lean 4 parsing and signature-rendering utilities.

Pure logic, no network and no Lean verifier required. Run with a plain
`python3 tests/lean_code_parser_test.py`.
"""

import _path  # noqa: F401  (puts leap/ on sys.path)
import unittest

from absl.testing import absltest

import lean_code_parser as models  # pylint: disable=g-import-not-at-top


class DepthAwareScanningTest(absltest.TestCase):
  """The primitives that distinguish binders from propositions."""

  def test_find_depth0_colon_ignores_nested_colons(self):
    self.assertEqual(models.find_depth0_colon("(2:Nat) + 2 = 4"), -1)
    self.assertEqual(models.find_depth0_colon("(n : Nat) : n = n"), 10)
    self.assertEqual(models.find_depth0_colon("n + 0 = n"), -1)

  def test_find_depth0_colon_stops_at_assignment(self):
    self.assertEqual(models.find_depth0_colon("foo := by simp"), -1)

  def test_is_binder_only(self):
    self.assertTrue(models.is_binder_only("(n : Nat) "))
    self.assertTrue(models.is_binder_only("{a : Type} [inst : Foo a] "))
    self.assertTrue(models.is_binder_only(""))
    self.assertFalse(models.is_binder_only("forall x "))
    self.assertFalse(models.is_binder_only("(a + b) * c "))


class ToLeanTest(absltest.TestCase):
  """Regression cover for the dropped-colon bug (fix 1.5).

  A proposition that merely *starts* with a bracket used to be misread as a
  binder list, producing 'theorem h (2:Nat) + 2 = 4' — invalid Lean.
  """

  def _lean(self, statement, name="h"):
    return models.FormalSignature(name=name, statement=statement).to_lean()

  def test_proposition_opening_with_paren_keeps_colon(self):
    self.assertEqual(self._lean("(2:Nat) + 2 = 4"), "theorem h : (2:Nat) + 2 = 4")

  def test_proposition_opening_with_paren_arithmetic(self):
    self.assertEqual(
        self._lean("(a + b)^2 = a^2 + 2*a*b + b^2"),
        "theorem h : (a + b)^2 = a^2 + 2*a*b + b^2",
    )

  def test_binders_are_left_alone(self):
    self.assertEqual(
        self._lean("(n : Nat) : n + 0 = n"), "theorem h (n : Nat) : n + 0 = n"
    )

  def test_implicit_and_instance_binders_are_left_alone(self):
    stmt = "{x : Real} (hx : 0 < x) : Real.sqrt x * Real.sqrt x = x"
    self.assertEqual(self._lean(stmt), f"theorem h {stmt}")

  def test_quantified_proposition_gets_colon(self):
    # Has a top-level colon, but its prefix is not a binder group.
    self.assertEqual(
        self._lean("∀ x : Nat, x = x"), "theorem h : ∀ x : Nat, x = x"
    )

  def test_bare_proposition_gets_colon(self):
    self.assertEqual(self._lean("n + 0 = n"), "theorem h : n + 0 = n")

  def test_already_colon_led_is_not_doubled(self):
    self.assertEqual(self._lean(": True"), "theorem h : True")

  def test_full_declaration_passes_through(self):
    stmt = "theorem foo : True := trivial"
    self.assertEqual(self._lean(stmt), stmt)


class SubgoalToAxiomTest(absltest.TestCase):

  def test_axioms_are_syntactically_valid(self):
    src = (
        "lemma helper_three : (2:Nat) + 2 = 4 := sorry\n"
        "lemma helper_four (n : Nat) : n + 0 = n := sorry\n"
        "lemma helper_five : ∀ x : Nat, x = x := sorry\n"
    )
    out = models.convert_subgoals_to_axioms(src)
    self.assertEqual(
        out.splitlines(),
        [
            "axiom helper_three : (2:Nat) + 2 = 4",
            "axiom helper_four (n : Nat) : n + 0 = n",
            "axiom helper_five : ∀ x : Nat, x = x",
        ],
    )

  def test_non_subgoal_declarations_are_dropped(self):
    src = "def not_a_subgoal : Nat := 5\nlemma real_one : True := sorry\n"
    sigs = models.parse_subgoal_declarations(src)
    self.assertEqual([s.name for s in sigs], ["real_one"])


class SplitSignatureAndProofTest(absltest.TestCase):
  """':=' inside binders must not be mistaken for the proof separator."""

  def test_default_argument_binder(self):
    sig, _ = models.split_signature_and_proof(
        "theorem foo (n : Nat := 0) : n + 0 = n := by simp"
    )
    self.assertEqual(sig, "theorem foo (n : Nat := 0) : n + 0 = n")

  def test_let_in_statement(self):
    sig, _ = models.split_signature_and_proof(
        "theorem bar : (let x := 3; x) = 3 := by rfl"
    )
    self.assertEqual(sig, "theorem bar : (let x := 3; x) = 3")

  def test_instance_binder_with_default(self):
    sig, _ = models.split_signature_and_proof(
        "theorem baz {a : Type} [inst : Inhabited a := ⟨default⟩] : True := trivial"
    )
    self.assertEqual(
        sig, "theorem baz {a : Type} [inst : Inhabited a := ⟨default⟩] : True"
    )

  def test_unbracketed_let_with_a_term_proof(self):
    """The shape: a `let` at depth 0 is not the proof separator."""
    sig, _ = models.split_signature_and_proof(
        "theorem qux (n : ℕ) :\n    let k := n + 1\n    k > n := rfl"
    )
    self.assertEqual(sig, "theorem qux (n : ℕ) :\n    let k := n + 1\n    k > n")

  def test_several_unbracketed_lets(self):
    stmt = "theorem q (n : ℕ) :\n    let a := 1\n    let k := 2\n    a = k"
    sig, _ = models.split_signature_and_proof(f"{stmt} := rfl")
    self.assertEqual(sig, stmt)

  def test_an_already_split_statement_is_unchanged(self):
    """Idempotence. `signature_str` no longer re-splits, but nothing else may
    either: a second pass has no proof to find and must not invent one."""
    stmt = "theorem q (n : ℕ) :\n    let k := n + 1\n    k > n"
    self.assertEqual(models.split_signature_and_proof(stmt)[0], stmt)

  def test_over_counting_declines_to_split_rather_than_truncating(self):
    """A `let` inside a depth-0 string literal is not masked, so the counter
    over-counts. The safe response is to return the block whole -- the proof
    then survives into the signature and the scaffold fails to compile, which
    is loud. Silently cutting the statement short would not be."""
    block = 'theorem q : s = "let x" := rfl'
    self.assertEqual(models.split_signature_and_proof(block)[0], block)


class ExpectedSignatureTest(absltest.TestCase):

  EXPECTED = "theorem foo (n : Nat) : n + 0 = n"

  def _accepts(self, submitted):
    ok, _, _ = models.extract_proof_with_expected_signature(
        submitted, self.EXPECTED
    )
    return ok

  def test_exact_match(self):
    self.assertTrue(self._accepts(f"{self.EXPECTED} := by simp"))

  def test_tolerates_extra_whitespace_and_newlines(self):
    self.assertTrue(
        self._accepts("theorem foo  (n : Nat)  :\n  n + 0 = n := by simp")
    )

  def test_tolerates_leading_comment(self):
    self.assertTrue(self._accepts(f"-- my proof\n{self.EXPECTED} := by simp"))

  def test_expected_signature_may_itself_carry_comments(self):
    """The shape. Comments are stripped from the submission, so they
    must be stripped from the expected signature too -- otherwise the pattern
    contains tokens the text can never hold and the node is unsubmittable."""
    for sig in (
        "theorem zs (a b : ℕ) :\n    (a = 1) ∨  -- exception 1\n    (b = 2)",
        "theorem bo (n : ℕ) :\n    -- max edges\n    n = n",
        "theorem bc (n : ℕ) : /- note -/ n = n",
    ):
      ok, _, _ = models.extract_proof_with_expected_signature(
          models.strip_lean_comments(f"{sig} :=\nby\n  sorry").strip(), sig
      )
      self.assertTrue(ok, f"gate rejected its own scaffold for: {sig!r}")

  @unittest.expectedFailure
  def test_tolerates_omitted_space_before_colon(self):
    """'(n: Nat)' is identical to '(n : Nat)' in Lean but is rejected today.

    The gate joins whitespace-split tokens with r'\\s+', so it accepts *more*
    whitespace than expected but never less. Tracked as a usability item, not a
    soundness one.
    """
    self.assertTrue(self._accepts("theorem foo (n: Nat) : n + 0 = n := by simp"))


class CommentStrippingTest(absltest.TestCase):

  def test_simple_block_comment(self):
    self.assertEqual(
        models.strip_lean_comments("/- gone -/theorem t : True := trivial"),
        "theorem t : True := trivial",
    )

  def test_mask_preserves_indices(self):
    src = "/- xx -/theorem t : True := trivial"
    self.assertLen(models.mask_lean_comments(src), len(src))

  def test_nested_block_comments(self):
    """Lean 4 block comments nest, so a non-greedy regex stops one -/ too early."""
    self.assertEqual(
        models.strip_lean_comments("/- a /- b -/ c -/\ntheorem t : True").strip(),
        "theorem t : True",
    )

  def test_triply_nested_block_comments(self):
    self.assertEqual(
        models.strip_lean_comments(
            "/- a /- b /- c -/ d -/ e -/\ntheorem t : True"
        ).strip(),
        "theorem t : True",
    )

  def test_double_dash_inside_string_literal(self):
    """'--' inside a string literal is not a comment."""
    src = 'theorem t : String := "a -- b"'
    self.assertEqual(models.strip_lean_comments(src), src)

  def test_block_comment_marker_inside_string_literal(self):
    src = 'theorem t : String := "a /- b"'
    self.assertEqual(models.strip_lean_comments(src), src)

  def test_doc_comment_is_stripped(self):
    self.assertEqual(
        models.strip_lean_comments("/-- doc -/\ntheorem t : True").strip(),
        "theorem t : True",
    )

  def test_mask_preserves_indices_for_nested_comments(self):
    src = "/- a /- b -/ c -/\ntheorem t : True"
    masked = models.mask_lean_comments(src)
    self.assertLen(masked, len(src))
    self.assertIn("theorem t : True", masked)
    self.assertNotIn("a", masked.split("\n")[0])


class DefinitionsBlockClassificationTest(absltest.TestCase):
  """Every definitions block is hoisted above every proof, so reach is content."""

  def test_plain_helpers_are_inert(self):
    report = models.classify_definitions_block(
        "def suffix_flip (n : Nat) : Nat := n\n\n"
        "abbrev helper (n : Nat) : Nat := n + 0\n"
    )
    self.assertEmpty(report.banned)
    self.assertEmpty(report.escalating)
    self.assertEmpty(report.unnamed)
    self.assertEmpty(report.proofs)
    self.assertEqual(report.declared_names, ["suffix_flip", "helper"])
    self.assertFalse(report.is_rejected)
    self.assertFalse(report.needs_full_tree_check)

  def test_empty_block_is_inert(self):
    self.assertFalse(models.classify_definitions_block("").is_rejected)
    self.assertFalse(models.classify_definitions_block("   \n").is_rejected)

  def test_unicode_identifiers_are_named(self):
    """An ASCII-only class read 'δbound' as no name at all, so the helper was
    filed under 'unnamed' and the whole block rejected. These problems use 'δ'
    in their binders, so it was not hypothetical."""
    report = models.classify_definitions_block(
        "def δbound (n : ℕ) : ℕ := n\n\nlemma δ_pos : 0 < 1 := by norm_num\n"
    )
    self.assertEmpty(report.unnamed)
    self.assertEqual(report.declared_names, ["δbound", "δ_pos"])
    # The lemma is refused for being a proof, not for being unnameable, and the
    # refusal names it with its 'δ' intact -- an agent cannot act on "move the
    # declaration on line 3".
    self.assertEqual(report.proofs, ["lemma δ_pos"])

  def test_subscripted_names_stay_distinct(self):
    """'h₁' and 'h₂' both truncated to 'h', which is a spurious collision."""
    report = models.classify_definitions_block(
        "def h₁ : ℕ := 1\n\ndef h₂ : ℕ := 2\n"
    )
    self.assertEqual(report.declared_names, ["h₁", "h₂"])

  def test_apostrophe_names_still_work(self):
    """`'` is not a word character, so it needs to stay in the class."""
    self.assertEqual(
        models.extract_declaration_name("theorem Foo.bar' : True := trivial"),
        "Foo.bar'",
    )

  def test_modifiers_do_not_hide_a_plain_helper(self):
    """Only 'noncomputable' used to be accepted before the keyword.

    Everything else fell through to 'unnamed' and was rejected, so an agent
    writing 'private def' was told to delete a declaration form that is in fact
    supported -- and the name never reached the collision registry.
    """
    for block, name in (
        ("private def f : Nat := 1", "f"),
        ("protected abbrev t : Nat := 1", "t"),
        ("partial def g : Nat := 1", "g"),
        ("nonrec def h : Nat := 1", "h"),
        ("private noncomputable def m : Nat := 1", "m"),
    ):
      report = models.classify_definitions_block(block)
      self.assertFalse(report.is_rejected, block)
      self.assertEqual(report.declared_names, [name], block)

  def test_opaque_is_a_placeholder_not_a_definition(self):
    """fmf_v1_002: 'def X := sorry' was refused, so the worker wrote 'opaque X'.

    It compiles, adds no axiom, and declares a constant nothing can be proved
    about -- so it reached the shared header, made every child's goal
    unprovable, and ended the run.
    """
    report = models.classify_definitions_block(
        "opaque HeuleGraph : SimpleGraph (Fin 510)"
    )
    self.assertTrue(report.is_rejected)
    self.assertIn("opaque constant", report.placeholders[0])
    self.assertEmpty(report.banned)

  def test_modifiers_do_not_hide_an_opaque(self):
    for block in ("noncomputable opaque k : Nat", "private opaque k : Nat"):
      self.assertTrue(
          models.classify_definitions_block(block).placeholders, block
      )

  def test_opaque_still_yields_its_name(self):
    """The shared preamble may legitimately carry one, and the collision
    registry has to know the name is taken."""
    self.assertEqual(models.extract_declaration_name("opaque k : Nat"), "k")

  def test_an_indented_opaque_is_not_top_level(self):
    report = models.classify_definitions_block(
        "def s : Nat :=\n  let opaque_count := 1\n  opaque_count"
    )
    self.assertFalse(report.is_rejected)

  def test_a_modifier_does_not_downgrade_an_escalating_form(self):
    """Naming more forms must not let one skip the full-tree check."""
    for block in ("private structure S where", "protected instance i : C := x"):
      report = models.classify_definitions_block(block)
      self.assertTrue(report.needs_full_tree_check, block)

  def test_notation_is_banned(self):
    report = models.classify_definitions_block('notation "a ⊕ b" => Or a b')
    self.assertTrue(report.is_rejected)
    self.assertIn("notation", report.banned[0])

  def test_bare_open_is_banned_but_scoped_open_is_not(self):
    self.assertTrue(
        models.classify_definitions_block("open Finset").is_rejected
    )
    self.assertFalse(
        models.classify_definitions_block(
            "open Finset in\ndef s : Nat := 1"
        ).is_rejected
    )

  def test_bare_set_option_is_banned_but_scoped_is_not(self):
    self.assertTrue(
        models.classify_definitions_block("set_option maxHeartbeats 4000").is_rejected
    )
    self.assertFalse(
        models.classify_definitions_block(
            "set_option maxHeartbeats 4000 in\ndef s : Nat := 1"
        ).is_rejected
    )

  def test_dangling_scoped_in_is_banned(self):
    """A scoped form at the end of the block binds to the next block."""
    for block in (
        "set_option maxHeartbeats 4000 in",
        "def s : Nat := 1\nopen Finset in",
        "set_option maxHeartbeats 4000 in\n-- nothing follows",
    ):
      with self.subTest(block=block):
        self.assertTrue(models.classify_definitions_block(block).is_rejected)

  def test_scoped_in_followed_by_a_comment_then_a_decl_is_not_banned(self):
    """Comments do not break the binding, so they must not trip the check."""
    self.assertFalse(
        models.classify_definitions_block(
            "set_option maxHeartbeats 4000 in\n-- why\ndef s : Nat := 1"
        ).is_rejected
    )

  def test_attribute_command_is_banned(self):
    """Declares no name at all, so a uniqueness check alone would miss it."""
    report = models.classify_definitions_block("attribute [simp] Nat.add_zero")
    self.assertTrue(report.is_rejected)
    self.assertEmpty(report.declared_names)

  def test_variable_is_banned(self):
    self.assertTrue(
        models.classify_definitions_block("variable (n : Nat)").is_rejected
    )

  def test_simp_attribute_escalates_without_rejecting(self):
    report = models.classify_definitions_block(
        "@[simp] def bad (n : Nat) : Nat := n + 0"
    )
    self.assertFalse(report.is_rejected)
    self.assertTrue(report.needs_full_tree_check)
    self.assertEqual(report.declared_names, ["bad"])

  def test_instance_escalates_without_rejecting(self):
    report = models.classify_definitions_block(
        "instance : Inhabited Nat := ⟨0⟩"
    )
    self.assertFalse(report.is_rejected)
    self.assertTrue(report.needs_full_tree_check)

  def test_structure_escalates_without_rejecting(self):
    report = models.classify_definitions_block("structure P where\n  x : Nat")
    self.assertFalse(report.is_rejected)
    self.assertTrue(report.needs_full_tree_check)

  def test_a_proved_statement_is_refused(self):
    """fmf_v1_009/rollout_2 put 205 lines of one node's proof in the header.

    The header is append-only, so all of it survived the node that wrote it
    failing. `theorem` and `lemma` are the only two keywords `_DECL_KEYWORDS`
    matches that are not vocabulary, and neither was policed by any of the
    three form lists -- both passed straight through.
    """
    for block, entry in (
        ("theorem final_subtraction : True := trivial", "theorem final_subtraction"),
        ("lemma bound_x : 0 < 1 := by norm_num", "lemma bound_x"),
        ("private theorem hidden : True := trivial", "theorem hidden"),
    ):
      report = models.classify_definitions_block(block)
      self.assertEqual(report.proofs, [entry], block)
      self.assertTrue(report.is_rejected, block)

  def test_the_admitted_forms_are_not_refused_as_proofs(self):
    # Every keyword on the admit list, so the gate cannot quietly widen to
    # cover a form a sibling needs to state its own goal.
    for block in (
        "def f : Nat := 1",
        "abbrev g : Nat := 1",
        "structure P where\n  x : Nat",
        "inductive Q\n  | a",
        "class R (α : Type) where\n  y : α",
        "instance : Inhabited Nat := ⟨0⟩",
    ):
      self.assertEmpty(models.classify_definitions_block(block).proofs, block)

  def test_admit_list_excludes_forms_refused_elsewhere(self):
    # `variable` is a banned form and `opaque` is a placeholder. Admitting them
    # here would name them in the proofs rejection as an available remedy.
    self.assertNotIn("variable", models.HEADER_ADMITS)
    self.assertNotIn("opaque", models.HEADER_ADMITS)

  def test_a_def_alongside_a_lemma_still_refuses_the_lemma(self):
    # The realistic shape: one genuine shared definition and the proof about it.
    report = models.classify_definitions_block(
        "def rademacher_univ : Nat := 1\n\n"
        "lemma variance_rademacher : 0 < 1 := by norm_num\n"
    )
    self.assertEqual(report.proofs, ["lemma variance_rademacher"])
    self.assertEqual(
        report.declared_names, ["rademacher_univ", "variance_rademacher"]
    )

  def test_unknown_command_is_refused(self):
    report = models.classify_definitions_block("#eval 1 + 1")
    self.assertTrue(report.is_rejected)
    self.assertIn("#eval", report.unnamed[0])

  def test_comments_do_not_trigger_a_ban(self):
    report = models.classify_definitions_block(
        "-- we could add notation here, or an open, but we will not\n"
        "def s : Nat := 1"
    )
    self.assertFalse(report.is_rejected)
    self.assertEqual(report.declared_names, ["s"])

  def test_indented_forms_are_not_top_level(self):
    """'instance' inside a term is already scoped; only column 0 escapes."""
    report = models.classify_definitions_block(
        "def s : Nat :=\n  let instance_count := 1\n  instance_count"
    )
    self.assertFalse(report.is_rejected)
    self.assertFalse(report.needs_full_tree_check)

  def test_banned_form_wins_over_escalation(self):
    report = models.classify_definitions_block(
        "@[simp] theorem ok (n : Nat) : n = n := rfl\nopen Finset"
    )
    self.assertTrue(report.is_rejected)
    self.assertTrue(report.needs_full_tree_check)

  def test_reported_line_numbers_match_the_submission(self):
    report = models.classify_definitions_block(
        "def a : Nat := 1\ndef b : Nat := 2\nopen Finset"
    )
    self.assertIn("line 3", report.banned[0])


class ScaffoldShapeTest(absltest.TestCase):
  """One shape for proving and decomposing; a narrower one for disproof."""

  def _build(self, **kwargs):
    args = dict(
        immutable_context="import Mathlib",
        definitions="",
        subgoals="",
        proof="theorem t : True := by trivial",
    )
    args.update(kwargs)
    return models.build_lean_scaffold(**args)

  def test_prove_scaffold_offers_the_subgoals_block(self):
    # The whole point of the merge: an agent claimed to prove must have
    # somewhere to write a decomposition without being re-dispatched.
    text = self._build()
    self.assertIn("-- [START_SUBGOALS]", text)
    self.assertIn("-- [END_SUBGOALS]", text)

  def test_decompose_role_is_identical_to_prove(self):
    self.assertEqual(
        self._build(role="prove"), self._build(role="decompose")
    )

  def test_unknown_role_renders_the_proof_shape(self):
    self.assertEqual(self._build(role="prove"), self._build(role="banana"))

  def test_all_four_sections_present_for_prove(self):
    text = self._build()
    for marker in (
        "IMMUTABLE_CONTEXT",
        "DEFINITIONS",
        "SUBGOALS",
        "PROOF",
    ):
      self.assertIn(f"-- [START_{marker}]", text)
      self.assertIn(f"-- [END_{marker}]", text)

  def test_disproof_withholds_both_mutable_regions(self):
    # Gate D: a region that is never rendered cannot be filled in, so nothing
    # derived from the assumed-false target can reach the shared header.
    text = self._build(role=models.SCAFFOLD_DISPROVE)
    self.assertNotIn("DEFINITIONS", text)
    self.assertNotIn("SUBGOALS", text)

  def test_disproof_keeps_context_and_proof(self):
    text = self._build(role=models.SCAFFOLD_DISPROVE)
    self.assertIn("-- [START_IMMUTABLE_CONTEXT]", text)
    self.assertIn("import Mathlib", text)
    self.assertIn("-- [START_PROOF]", text)
    self.assertIn("theorem t : True := by trivial", text)

  def test_content_survives_verbatim(self):
    text = self._build(
        definitions="def f : Nat := 1",
        subgoals="theorem sub : True := sorry",
    )
    self.assertIn("def f : Nat := 1", text)
    self.assertIn("theorem sub : True := sorry", text)


class ScaffoldHintTest(absltest.TestCase):
  """Hints help an agent and lie to the kernel's diagnostics."""

  def _build(self, hints):
    return models.build_lean_scaffold(
        immutable_context="import Mathlib",
        definitions="",
        subgoals="",
        proof="theorem t : True := by trivial",
        hints=hints,
    )

  def test_hints_on_by_default(self):
    self.assertIn("Leave this block empty", self._build(hints=True))

  def test_assembly_emits_no_hint(self):
    # 'sorry_occurrences' counts literal text; a hint mentioning one would
    # report a placeholder in every clean proof.
    text = self._build(hints=False)
    self.assertNotIn("sorry", text)
    self.assertNotIn("Leave this block empty", text)

  def test_assembly_still_has_every_marker(self):
    text = self._build(hints=False)
    for marker in ("DEFINITIONS", "SUBGOALS"):
      self.assertIn(f"-- [START_{marker}]", text)

  def test_hint_is_not_mistaken_for_a_subgoal(self):
    # The trap the merge creates: the hint mentions ':= sorry', and emptiness
    # of this block is what distinguishes a proof from a decomposition. It is
    # read by parsing declarations, so a comment must yield nothing.
    text = models.build_lean_scaffold(
        immutable_context="", definitions="", subgoals="", proof="x"
    )
    block = text.split("-- [START_SUBGOALS]")[1].split("-- [END_SUBGOALS]")[0]
    self.assertIn("sorry", block, "precondition: the hint mentions sorry")
    self.assertEqual(models.parse_subgoal_declarations(block), [])

  def test_definitions_hint_is_not_mistaken_for_a_declaration(self):
    text = models.build_lean_scaffold(
        immutable_context="", definitions="", subgoals="", proof="x"
    )
    block = text.split("-- [START_DEFINITIONS]")[1].split(
        "-- [END_DEFINITIONS]"
    )[0]
    self.assertEqual(models.classify_definitions_block(block).declared_names,
                     [])


class ScaffoldFilenameTest(absltest.TestCase):

  def test_proving_and_decomposing_are_one_scaffold(self):
    # There is no separate `decompose` mode. One file covers both, and which
    # one it is is decided by whether its subgoal block is empty.
    self.assertEqual(
        models.scaffold_filename(models.SCAFFOLD_PROVE), "proof_scaffold.lean"
    )
    with self.assertRaises(KeyError):
      models.scaffold_filename("decompose")

  def test_default_is_the_proof_scaffold(self):
    self.assertEqual(models.scaffold_filename(), "proof_scaffold.lean")

  def test_disproof_gets_its_own_file(self):
    # It must never overwrite a proof of the same node.
    self.assertNotEqual(
        models.scaffold_filename(models.SCAFFOLD_DISPROVE),
        models.scaffold_filename(models.SCAFFOLD_PROVE),
    )

  def test_unknown_role_raises_rather_than_falling_back(self):
    # A silent fallback here hands a disproof agent a proof scaffold, so it
    # compiles locally and is rejected on submit. Fail at the typo instead.
    with self.assertRaises(KeyError):
      models.scaffold_filename("banana")


class DisproofHeaderTest(absltest.TestCase):
  """The header a disproof is compiled against. §2.8 Gates A and B live here."""

  SIG = "theorem foo (a : Nat) (h : a > 0) : a + 0 = a"

  def test_a_theorem_becomes_an_axiom(self):
    self.assertEqual(
        models.declaration_to_axiom(self.SIG),
        "axiom foo (a : Nat) (h : a > 0) : a + 0 = a",
    )

  def test_a_lemma_becomes_an_axiom(self):
    self.assertEqual(
        models.declaration_to_axiom("lemma foo : True"), "axiom foo : True"
    )

  def test_the_binders_are_left_exactly_as_written(self):
    # The point of a keyword rewrite: reproducing binder scoping in Python
    # fails silently into a different claim.
    self.assertIn("(a : Nat) (h : a > 0)", models.declaration_to_axiom(self.SIG))

  def test_only_the_leading_keyword_is_rewritten(self):
    self.assertEqual(
        models.declaration_to_axiom("theorem thm_about_lemma : True"),
        "axiom thm_about_lemma : True",
    )

  def test_the_target_is_assumed_never_negated(self):
    header = models.build_disproof_header("import Mathlib", self.SIG)
    self.assertIn("axiom foo (a : Nat) (h : a > 0) : a + 0 = a", header)
    self.assertNotIn("¬", header)
    self.assertNotIn("Not ", header)

  def test_the_shared_header_is_kept_above_the_assumption(self):
    header = models.build_disproof_header("import Mathlib", self.SIG)
    self.assertLess(header.index("import Mathlib"), header.index("axiom foo"))

  def test_the_assumption_is_labelled_as_under_refutation(self):
    header = models.build_disproof_header("import Mathlib", self.SIG)
    self.assertIn("Target under refutation", header)
    self.assertIn("Nothing proved from it may leave this file", header)

  def test_the_obligation_is_false_for_every_node(self):
    self.assertEqual(
        models.disproof_obligation("kid"), "theorem kid_disproof : False"
    )
    self.assertEqual(
        models.extract_declaration_name(models.disproof_obligation("kid")),
        "kid_disproof",
    )


class ResolvedNameTest(absltest.TestCase):
  """A declared name and a kernel-printed name have to be comparable.

  Both halves come from `fmf_v1_002`, whose immutable context opened
  `namespace SimpleGraph` and never closed it. Agents reached for `_root_.` to
  escape it; the kernel prints neither the `_root_.` nor, for anything declared
  inside, the bare name.
  """

  def test_root_prefix_is_stripped_from_the_declared_name(self):
    name, _ = models.parse_formal_statement(
        "theorem _root_.exists_five_chromatic (n : Nat) : n = n := by rfl"
    )
    self.assertEqual("exists_five_chromatic", name)

  def test_root_prefix_does_not_disturb_the_statement(self):
    """Stripping happens after the statement is cut, not before."""
    with_prefix = models.parse_formal_statement(
        "theorem _root_.foo (n : Nat) : n + 0 = n := by simp"
    )
    without = models.parse_formal_statement(
        "theorem foo (n : Nat) : n + 0 = n := by simp"
    )
    self.assertEqual(without, with_prefix)

  def test_a_name_merely_containing_root_is_untouched(self):
    name, _ = models.parse_formal_statement(
        "theorem _root_of_unity (n : Nat) : n = n := by rfl"
    )
    self.assertEqual("_root_of_unity", name)


class SubgoalModifierTest(absltest.TestCase):
  """`parse_subgoal_declarations` must accept every modifier the rest does.

  It accepted only `noncomputable` while `split_declarations` and
  `extract_declaration_name` accepted all of `_DECL_MODIFIERS`. A
  `private lemma` therefore parsed to nothing and the whole subgoals block was
  silently reclassified as a direct proof.
  """

  def _names(self, text):
    return [sig.name for sig in models.parse_subgoal_declarations(text)]

  def test_every_declaration_modifier_is_accepted(self):
    for modifier in ("private", "protected", "noncomputable", "partial",
                     "nonrec", "unsafe"):
      with self.subTest(modifier=modifier):
        self.assertEqual(
            ["h"], self._names(f"{modifier} theorem h : 1 = 1 := sorry")
        )

  def test_stacked_modifiers_are_accepted(self):
    self.assertEqual(
        ["h"],
        self._names("private noncomputable lemma h : 1 = 1 := sorry"),
    )

  def test_plain_and_attributed_declarations_still_parse(self):
    self.assertEqual(["h"], self._names("theorem h : 1 = 1 := sorry"))
    self.assertEqual(["h"], self._names("@[simp] lemma h : 1 = 1 := sorry"))

  def test_non_declarations_are_still_dropped(self):
    """The looser prefix must not start admitting `axiom` or `def`."""
    self.assertEmpty(self._names("axiom h : 1 = 1"))
    self.assertEmpty(self._names("def h : Nat := 1"))
    self.assertEmpty(self._names("private axiom h : 1 = 1"))

  def test_modifiers_are_kept_in_declaration_order(self):
    self.assertEqual(
        ["a", "b", "c"],
        self._names(
            "theorem a : 1 = 1 := sorry\n"
            "private lemma b : 2 = 2 := sorry\n"
            "protected theorem c : 3 = 3 := sorry"
        ),
    )


if __name__ == "__main__":
  absltest.main()
