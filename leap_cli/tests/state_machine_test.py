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

"""End-to-end walkthrough of the phase machine, one class per state.

Where the other suites test a unit, this one tests a *state*: for each phase a
node can be dispatched in, it checks the three things that can independently be
wrong.

  input       what the agent is handed -- the scaffold it must edit and the
              briefing sections it is shown
  submission  what the recorder accepts and refuses in that phase
  transition  where the node lands, and what the scheduler does next

Two verbs move a node. `--action=submit` hands over a Lean file and the tool
derives what it turned out to be; every other action is a declaration carrying
nothing but `--reason`. Both are exercised through `run_state_manager`, the same
entry point the shims call, so a flag the CLI refuses cannot pass here.

The fixtures are the real Putnam 2025 A5 run in `scratch/rollout_2`: its header,
its root statement, and its first two decompositions, copied verbatim. Real Lean
is worth the length here -- unicode identifiers, multi-line binders and
`:= sorry` subgoals spread over four lines are exactly the shapes the parser and
the signature gate are asked about in production, and a `theorem foo : True`
fixture exercises none of them.

The Lean verifier is stubbed, so this runs offline.
"""

import _path  # noqa: F401  (puts leap/ on sys.path)
import os
import tempfile
from unittest import mock

from absl.testing import absltest

import dag
import lean_code_parser as models
import pipeline
import review_trigger
import state_manager as sm


def _sigs(names):
  """Subgoal signatures for a hand-built AndNode.

  Nothing rejects an AndNode that names children without their statements any
  more, but a fixture that omits them is a legacy-format node and behaves like
  one, so build them here unless that is the point of the test.
  """
  return [
      models.FormalSignature(name=n, statement=f"Claim_{n}") for n in names
  ]


# --------------------------------------------------------------------------
# Fixtures, verbatim from scratch/rollout_2
# --------------------------------------------------------------------------

HEADER = """import Mathlib

open Finset

abbrev putnam_2025_a5_solution : (n : ℕ) → Set (Fin n → ℤˣ) :=
  fun n => {s | ∀ (i : Fin n), (h : i.val + 1 < n) → s ⟨i.val + 1, h⟩ = -s i}

abbrev f (n : ℕ) (s : Fin n → ℤˣ) : ℕ :=
  Finset.card {σ : Equiv.Perm (Fin (n + 1)) |
    ∀ i : Fin n, 0 < (s i : ℤ) * ((σ i.succ : ℤ) - (σ i.castSucc : ℤ))}"""

ROOT_ID = "putnam_2025_a5"

ROOT_STATEMENT = """theorem putnam_2025_a5
    (n : ℕ)
    (hn : 1 ≤ n)
    (s : Fin n → ℤˣ) :
    (∀ t : Fin n → ℤˣ, f n t ≤ f n s) ↔ s ∈ putnam_2025_a5_solution n"""

ROOT_SUBGOALS = """lemma exists_max_f (n : ℕ) (hn : 1 ≤ n) : ∃ (s : Fin n → ℤˣ), ∀ (t : Fin n → ℤˣ), f n t ≤ f n s := sorry

lemma not_sol_implies_lt_max (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℤˣ) (h : s ∉ putnam_2025_a5_solution n) :
    ∃ (t : Fin n → ℤˣ), f n s < f n t := sorry

lemma f_sol_eq (n : ℕ) (hn : 1 ≤ n) (s t : Fin n → ℤˣ) (hs : s ∈ putnam_2025_a5_solution n) (ht : t ∈ putnam_2025_a5_solution n) :
    f n s = f n t := sorry"""

ROOT_CHILDREN = ["exists_max_f", "not_sol_implies_lt_max", "f_sol_eq"]

ROOT_GLUE = ROOT_STATEMENT + """ := by
  constructor
  · intro hmax
    by_contra hnot
    rcases not_sol_implies_lt_max n hn s hnot with ⟨t, ht⟩
    have := hmax t
    omega
  · intro hs
    intro t
    rcases exists_max_f n hn with ⟨s_max, hmax_prop⟩
    have hs_max_in : s_max ∈ putnam_2025_a5_solution n := by
      by_contra hnot
      rcases not_sol_implies_lt_max n hn s_max hnot with ⟨t_max, ht_max⟩
      have := hmax_prop t_max
      omega
    have heq : f n t ≤ f n s_max := hmax_prop t
    have heq' : f n s = f n s_max := f_sol_eq n hn s s_max hs hs_max_in
    omega"""

# The second decomposition in the same run: a child that introduces a shared
# definition alongside its subgoals.
CHILD_ID = "not_sol_implies_lt_max"

CHILD_DEFINITIONS = """def valid_perms (n : ℕ) (s : Fin n → ℤˣ) : Finset (Equiv.Perm (Fin (n + 1))) :=
  {σ : Equiv.Perm (Fin (n + 1)) | ∀ i : Fin n, 0 < (s i : ℤ) * ((σ i.succ : ℤ) - (σ i.castSucc : ℤ))}"""

CHILD_SUBGOALS = """lemma f_eq_card_valid_perms (n : ℕ) (s : Fin n → ℤˣ) :
    f n s = (valid_perms n s).card := sorry

lemma not_sol_implies_exists_eq (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℤˣ) (h : s ∉ putnam_2025_a5_solution n) :
    ∃ (i : Fin n) (hi : i.val + 1 < n), s ⟨i.val + 1, hi⟩ = s i := sorry"""

CHILD_STATEMENT = """theorem not_sol_implies_lt_max (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℤˣ) (h : s ∉ putnam_2025_a5_solution n) :
    ∃ (t : Fin n → ℤˣ), f n s < f n t"""

CHILD_GLUE_BODY = """ := by
  obtain ⟨i, hi, heq⟩ := not_sol_implies_exists_eq n hn s h
  refine ⟨Function.update s ⟨i.val + 1, hi⟩ (-s i), ?_⟩
  rw [f_eq_card_valid_perms, f_eq_card_valid_perms]
  exact card_lt_card_of_flip n s i hi heq"""

# A real leaf proof from the same run, short enough to read.
LEAF_ID = "f_eq_card_valid_perms"

LEAF_PROOF = """theorem f_eq_card_valid_perms (n : ℕ) (s : Fin n → ℤˣ) :
    f n s = (valid_perms n s).card :=
by
  rfl"""

ROOT_PLAN = """1. [LEMMA] exists_max_f : a maximiser exists, by finiteness of the sequence space.
2. [LEMMA] [CRUX] not_sol_implies_lt_max : any sequence outside the claimed solution set is beaten by some other sequence.
3. [LEMMA] f_sol_eq : every sequence in the solution set has the same value, by the reversal symmetry.
4. [ELEM] Combine the three to get both directions of the iff."""


def _verifier(is_valid=True, used_axioms=("propext",), error=None):
  """A stand-in for run_lean_verifier with a fixed verdict."""

  def _run(input_data):
    del input_data
    return sm.lean_verifier_tool.LeanVerifierOutput(
        is_valid=is_valid,
        used_axioms=None if used_axioms is None else list(used_axioms),
        error_message=error,
    )

  return _run


class StateMachineTestCase(absltest.TestCase):
  """A one-node graph carrying the real header, plus submission helpers."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.filepath = os.path.join(self.root, "proof_graph.json")
    graph = dag.ProofGraph(
        environment=models.FormalEnvironment(shared_definitions=HEADER)
    )
    graph.add_node(
        dag.ProofGraphNode(
            id=ROOT_ID,
            informal_problem=models.InformalProblem(statement=ROOT_STATEMENT),
        )
    )
    self.make().initialize_graph(graph)

  def make(self):
    return sm.ProofGraphStateManager(
        filepath=self.filepath, root_dir=self.root, root_id=ROOT_ID
    )

  def load(self):
    return self.make().read_graph()

  def node(self, node_id=ROOT_ID):
    return self.load().nodes[node_id]

  def phase_of(self, node_id=ROOT_ID):
    # Not `search_metadata["phase"]`: a node that has never been dispatched has
    # no such key and is in the entry phase, which is not the same answer as a
    # node that left the machine and had `phase = None` written over it.
    return pipeline.current_phase_name(self.node(node_id))

  def briefing(self, claim):
    with open(claim["context_path"], encoding="utf-8") as f:
      return f.read()

  def scaffold(self, claim):
    with open(claim["scaffold_path"], encoding="utf-8") as f:
      return f.read()

  def put_in_phase(self, node_id, phase_name):
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      graph.nodes[node_id].search_metadata["phase"] = phase_name

  # -- The two verbs, both through the CLI entry point.

  def run_cli(self, action, **kwargs):
    return sm.run_state_manager(
        sm.StateManagerInput(action=action, root_dir=self.root, **kwargs)
    )

  def declare(self, node_id, action, reason="A full sentence saying why."):
    """`--action=<anything but submit>`: no file, `--reason` and nothing else."""
    out = self.run_cli(action, node_id=node_id, reason=reason)
    return out.success, (out.message or out.error_message or "")

  def submit(self, claim, proof, definitions="", subgoals="", verifier=None):
    """`--action=submit`: writes the scaffold, then hands over the path."""
    with open(claim["scaffold_path"], "w", encoding="utf-8") as f:
      f.write(
          models.build_lean_scaffold(
              immutable_context=HEADER,
              definitions=definitions,
              subgoals=subgoals,
              proof=proof,
              role=claim["scaffold_role"] or models.SCAFFOLD_PROVE,
          )
      )
    with mock.patch.object(
        sm.lean_verifier_tool,
        "run_lean_verifier",
        verifier or _verifier(),
    ):
      out = self.run_cli("submit", code_file=claim["scaffold_path"])
    return out.success, (out.message or out.error_message or "")

  # -- Stage helpers, each leaving the graph in the state its name describes.

  def reach_formal(self, node_id=ROOT_ID):
    """Plans the node and declares `planned`, landing it in `formal`."""
    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["node_id"], node_id)
    handled, msg = self.declare(
        node_id, "planned", "The blueprint is written and every step is closed."
    )
    self.assertTrue(handled, msg)
    return self.make().claim_next_subgoal()

  def reach_gate_review(self):
    """Commits the root's real decomposition, landing it in `review`."""
    claim = self.reach_formal()
    handled, msg = self.submit(claim, ROOT_GLUE, subgoals=ROOT_SUBGOALS)
    self.assertTrue(handled, msg)
    return self.make().claim_next_subgoal()

  def reach_blocked(self):
    """Approves the root's decomposition, releasing the children."""
    review = self.reach_gate_review()
    self.assertEqual(review["action_type"], "lemma_decomposition_reviewer")
    handled, msg = self.declare(
        ROOT_ID, "approved", "The three children close the target as sketched."
    )
    self.assertTrue(handled, msg)


# --------------------------------------------------------------------------
# informal -- the entry state
# --------------------------------------------------------------------------


class InformalStateTest(StateMachineTestCase):
  """Where every node starts: a plan, and a decision about what comes next."""

  def claim(self):
    return self.make().claim_next_subgoal()

  # -- input

  def test_a_fresh_node_is_dispatched_for_planning(self):
    claim = self.claim()
    self.assertEqual(claim["node_id"], ROOT_ID)
    self.assertEqual(claim["phase"], "informal")
    self.assertEqual(claim["action_type"], "lemma_informal_prover")

  def test_the_planner_is_handed_no_scaffold(self):
    # It writes prose, not Lean. A scaffold here is what used to brief the
    # planner as the prover.
    claim = self.claim()
    self.assertIsNone(claim["scaffold_role"])
    self.assertEqual(claim["scaffold_path"], "")

  def test_the_briefing_names_the_role_and_the_phase(self):
    context = self.briefing(self.claim())
    self.assertIn("You are the **lemma_informal_prover**", context)
    self.assertIn("phase **`informal`**", context)

  def test_the_briefing_carries_the_real_statement(self):
    context = self.briefing(self.claim())
    self.assertIn("s ∈ putnam_2025_a5_solution n", context)
    self.assertIn("Global Problem Specification", context)

  def test_the_briefing_withholds_the_formal_workspace_rules(self):
    # Section 4 is not on the informal row: nothing has been formalized yet, so
    # peer scaffolds and definition-block rules are noise.
    self.assertNotIn("Dynamic Peer Discovery", self.briefing(self.claim()))

  def test_the_briefing_tells_the_planner_where_to_write(self):
    context = self.briefing(self.claim())
    self.assertIn("## 8. Plan of Record", context)
    self.assertIn(f"Write your plan to `nodes/{ROOT_ID}/scratch/blueprint.md`", context)

  def test_the_briefing_offers_exactly_the_three_informal_actions(self):
    context = self.briefing(self.claim())
    for action in ("planned", "refute", "failed"):
      self.assertIn(f"**`--action={action}`**", context)
    self.assertNotIn("--action=approved", context)

  def test_the_offered_actions_are_the_informal_row(self):
    out = self.run_cli("claim_next")
    self.assertEqual(
        out.metadata["allowed_actions"], ["planned", "refute", "failed"]
    )

  # -- submission

  def test_a_declaration_without_a_reason_is_refused(self):
    self.claim()
    handled, msg = self.declare(ROOT_ID, "planned", "")
    self.assertFalse(handled)
    self.assertIn("--reason", msg)

  def test_a_one_word_reason_is_treated_as_no_reason(self):
    self.claim()
    handled, msg = self.declare(ROOT_ID, "planned", "done")
    self.assertFalse(handled)
    self.assertIn("not enough", msg)

  def test_a_refused_declaration_moves_nothing_and_keeps_the_claim(self):
    self.claim()
    self.declare(ROOT_ID, "planned", "done")
    self.assertEqual(self.phase_of(), "informal")
    self.assertTrue(self.node().search_metadata["processing"])

  def test_an_action_from_another_phase_is_refused(self):
    self.claim()
    handled, msg = self.declare(
        ROOT_ID, "approved", "The decomposition looks fine to me."
    )
    self.assertFalse(handled)
    self.assertIn("'informal'", msg)
    self.assertIn("--action=planned", msg)

  def test_submitting_a_file_from_a_phase_with_no_scaffold_is_refused(self):
    self.claim()
    path = os.path.join(self.root, "nodes", ROOT_ID, "proof_scaffold.lean")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      f.write("theorem x : True := by trivial")
    out = self.run_cli("submit", code_file=path)
    self.assertFalse(out.success)
    self.assertIn("writes no Lean", out.error_message)

  # -- transition

  def test_planned_moves_the_node_to_formalization_and_keeps_it_alive(self):
    self.claim()
    handled, msg = self.declare(ROOT_ID, "planned", "Blueprint written, four steps.")
    self.assertTrue(handled, msg)
    self.assertEqual(self.phase_of(), "formal")
    self.assertFalse(self.node().search_metadata.get("decomposition_failed"))

  def test_the_reason_is_left_for_the_next_agent(self):
    self.claim()
    self.declare(ROOT_ID, "planned", "Blueprint written, four steps.")
    self.assertEqual(
        self.node().search_metadata["last_error"], "Blueprint written, four steps."
    )

  def test_a_declaration_releases_the_claim(self):
    self.claim()
    self.declare(ROOT_ID, "planned", "Blueprint written, four steps.")
    self.assertFalse(self.node().search_metadata.get("processing"))

  def test_the_next_claim_is_the_formalization_of_the_same_node(self):
    self.claim()
    self.declare(ROOT_ID, "planned", "Blueprint written, four steps.")
    nxt = self.make().claim_next_subgoal()
    self.assertEqual(nxt["node_id"], ROOT_ID)
    self.assertEqual(nxt["phase"], "formal")
    self.assertEqual(nxt["action_type"], "lemma_proof_worker")

  def test_refute_diverts_the_node_to_the_disproof_route(self):
    self.claim()
    handled, msg = self.declare(
        ROOT_ID,
        "refute",
        "At n = 1 the solution set is everything, so the iff fails one way.",
    )
    self.assertTrue(handled, msg)
    self.assertEqual(self.phase_of(), "disprove")

  def test_failed_closes_a_route_and_returns_the_node_to_planning(self):
    self.claim()
    handled, msg = self.declare(
        ROOT_ID,
        "failed",
        "Three plans tried, all circular; nothing here is provable as stated.",
    )
    self.assertTrue(handled, msg)
    self.assertEqual(self.phase_of(), "informal")
    self.assertFalse(self.node().search_metadata.get("decomposition_failed"))
    self.assertNotIn("is dead", msg)

  def test_a_closed_route_leaves_the_node_claimable_again(self):
    """No termination budget: a problem too hard to plan is re-planned, with
    the previous planner's reason in the next one's briefing."""
    self.claim()
    self.declare(ROOT_ID, "failed", "Nothing here is provable as stated.")
    self.assertIsNotNone(self.make().claim_next_subgoal())


# --------------------------------------------------------------------------
# formal -- one shape, two submissions
# --------------------------------------------------------------------------


class FormalStateProvingTest(StateMachineTestCase):
  """The direct-proof reading of the formal submission."""

  def claim(self):
    return self.reach_formal()

  # -- input

  def test_the_prover_gets_the_proof_scaffold(self):
    claim = self.claim()
    self.assertEqual(claim["phase"], "formal")
    self.assertEqual(claim["action_type"], "lemma_proof_worker")
    self.assertEqual(claim["scaffold_role"], models.SCAFFOLD_PROVE)
    self.assertEqual(
        os.path.basename(claim["scaffold_path"]), "proof_scaffold.lean"
    )

  def test_the_scaffold_carries_the_header_and_the_obligation(self):
    text = self.scaffold(self.claim())
    self.assertIn("abbrev putnam_2025_a5_solution", text)
    self.assertIn("theorem putnam_2025_a5", text)
    self.assertIn("sorry", text)

  def test_the_subgoals_block_is_offered_even_to_a_prover(self):
    # One shape. Gating the block on intent forced a release-and-reclaim every
    # time a proof turned out to need a decomposition.
    text = self.scaffold(self.claim())
    self.assertIn("-- [START_SUBGOALS]", text)
    self.assertIn("-- [START_DEFINITIONS]", text)

  def test_the_briefing_points_at_the_blueprint_rather_than_copying_it(self):
    context = self.briefing(self.claim())
    self.assertIn("## 8. Plan of Record", context)
    self.assertIn(f"nodes/{ROOT_ID}/scratch/blueprint.md", context)

  def test_the_briefing_states_the_axiom_allowlist(self):
    context = self.briefing(self.claim())
    self.assertIn("Only these axioms are permitted", context)
    self.assertIn("`propext`", context)

  def test_the_briefing_gives_submit_an_absolute_path(self):
    # A relative --code_file resolves against the agent's shell cwd, and
    # nothing guarantees that is the problem root.
    claim = self.claim()
    self.assertIn(
        f"--code_file={claim['scaffold_path']}", self.briefing(claim)
    )

  def test_the_briefing_offers_submit_and_replan_and_nothing_else(self):
    context = self.briefing(self.claim())
    self.assertIn("**`--action=submit`**", context)
    self.assertIn("**`--action=replan`**", context)
    self.assertNotIn("--action=failed", context)

  # -- submission

  def test_a_clean_proof_is_accepted(self):
    claim = self.claim()
    handled, msg = self.submit(claim, ROOT_STATEMENT + " := by\n  sorry_free")
    self.assertTrue(handled, msg)

  def test_a_renamed_signature_is_refused(self):
    claim = self.claim()
    handled, msg = self.submit(
        claim, ROOT_STATEMENT.replace("putnam_2025_a5\n", "putnam_2025_a5_v2\n")
        + " := by\n  trivial"
    )
    self.assertFalse(handled)
    self.assertIn("theorem putnam_2025_a5", msg)

  def test_a_sorried_proof_is_refused(self):
    claim = self.claim()
    handled, msg = self.submit(claim, ROOT_STATEMENT + " := by\n  sorry")
    self.assertFalse(handled)
    self.assertIn("placeholder", msg)

  def test_a_proof_leaning_on_sorry_ax_is_refused_even_when_it_compiles(self):
    claim = self.claim()
    handled, msg = self.submit(
        claim,
        ROOT_STATEMENT + " := by\n  trivial",
        verifier=_verifier(used_axioms=("propext", "sorryAx")),
    )
    self.assertFalse(handled)
    self.assertIn("sorryAx", msg)

  def test_submit_without_a_code_file_is_refused_and_names_the_alternative(self):
    self.claim()
    out = self.run_cli("submit", node_id=ROOT_ID)
    self.assertFalse(out.success)
    self.assertIn("MISSING_CODE_FILE", out.error_message)
    self.assertIn("--action=replan", out.error_message)

  # -- transition

  def test_a_proved_root_leaves_the_machine(self):
    claim = self.claim()
    self.submit(claim, ROOT_STATEMENT + " := by\n  trivial")
    self.assertEqual(
        self.load().propagate_failures()[ROOT_ID],
        dag.NodeStatus.FORMAL_PROVED,
    )
    self.assertIsNone(self.phase_of())

  def test_a_proved_root_ends_the_search(self):
    claim = self.claim()
    self.submit(claim, ROOT_STATEMENT + " := by\n  trivial")
    self.assertIsNone(self.make().claim_next_subgoal())

  def test_replan_returns_the_node_to_planning_without_killing_it(self):
    self.claim()
    handled, msg = self.declare(
        ROOT_ID,
        "replan",
        "Three approaches tried; none closes the reverse direction.",
    )
    self.assertTrue(handled, msg)
    self.assertEqual(self.phase_of(), "informal")
    self.assertFalse(self.node().search_metadata.get("decomposition_failed"))

  def test_a_prover_cannot_kill_the_node(self):
    self.claim()
    handled, msg = self.declare(
        ROOT_ID, "failed", "I cannot see any way to prove this at all."
    )
    self.assertFalse(handled)
    self.assertIn("--action=replan", msg)


class FormalStateDecomposingTest(StateMachineTestCase):
  """The same submission read as a decomposition: `[START_SUBGOALS]` is filled."""

  # -- submission

  def test_the_real_decomposition_is_accepted(self):
    claim = self.reach_formal()
    handled, msg = self.submit(claim, ROOT_GLUE, subgoals=ROOT_SUBGOALS)
    self.assertTrue(handled, msg)

  def test_each_subgoal_becomes_its_own_node(self):
    claim = self.reach_formal()
    self.submit(claim, ROOT_GLUE, subgoals=ROOT_SUBGOALS)
    for child in ROOT_CHILDREN:
      self.assertIn(child, self.load().nodes)

  def test_the_children_are_recorded_as_one_branch_in_declared_order(self):
    claim = self.reach_formal()
    self.submit(claim, ROOT_GLUE, subgoals=ROOT_SUBGOALS)
    branches = self.node().and_nodes
    self.assertLen(branches, 1)
    self.assertEqual(branches[0].dependency_ids, ROOT_CHILDREN)

  def test_glue_left_sorried_is_refused(self):
    claim = self.reach_formal()
    handled, msg = self.submit(
        claim, ROOT_STATEMENT + " := by\n  sorry", subgoals=ROOT_SUBGOALS
    )
    self.assertFalse(handled)
    self.assertIn("placeholder", msg)

  def test_a_definition_block_is_hoisted_into_the_shared_header(self):
    self.reach_blocked()
    self.put_in_phase(CHILD_ID, "formal")
    # Claimed by subtree rather than off the top: the walk offers the children
    # in declared order, and this one is second.
    claim = self.make().claim_next_subgoal(root_id=CHILD_ID)
    self.assertEqual(claim["node_id"], CHILD_ID)
    handled, msg = self.submit(
        claim,
        CHILD_STATEMENT + CHILD_GLUE_BODY,
        definitions=CHILD_DEFINITIONS,
        subgoals=CHILD_SUBGOALS,
    )
    self.assertTrue(handled, msg)
    self.assertIn("def valid_perms", self.load().environment.to_lean_header())

  # -- transition

  def test_a_decomposition_moves_the_node_to_review_not_to_blocked(self):
    claim = self.reach_formal()
    self.submit(claim, ROOT_GLUE, subgoals=ROOT_SUBGOALS)
    self.assertEqual(self.phase_of(), "review")

  def test_the_children_are_not_dispatched_before_the_review(self):
    claim = self.reach_formal()
    self.submit(claim, ROOT_GLUE, subgoals=ROOT_SUBGOALS)
    self.assertEqual(self.make().claim_next_subgoal()["node_id"], ROOT_ID)

  def test_a_second_decomposition_while_one_is_live_is_refused(self):
    # The submitter is working from a stale snapshot: a node with a live
    # branch is waiting on its children and is not claimable for `formal`.
    claim = self.reach_formal()
    self.submit(claim, ROOT_GLUE, subgoals=ROOT_SUBGOALS)
    self.put_in_phase(ROOT_ID, "formal")
    handled, msg = self.submit(
        claim,
        ROOT_STATEMENT + " := by\n  exact other n hn s",
        subgoals="lemma other (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℤˣ) : True := sorry",
    )
    self.assertFalse(handled)
    self.assertIn(ROOT_ID, msg)


# --------------------------------------------------------------------------
# review -- the gate and the patrol
# --------------------------------------------------------------------------


class ReviewStateGateTest(StateMachineTestCase):
  """A committed decomposition, reviewed before its children are dispatched."""

  # -- input

  def test_the_reviewer_is_dispatched_on_the_parent(self):
    claim = self.reach_gate_review()
    self.assertEqual(claim["node_id"], ROOT_ID)
    self.assertEqual(claim["phase"], "review")
    self.assertEqual(claim["action_type"], "lemma_decomposition_reviewer")

  def test_the_reviewer_is_given_no_scaffold(self):
    self.assertIsNone(self.reach_gate_review()["scaffold_role"])

  def test_the_briefing_says_which_entrance_this_is(self):
    context = self.briefing(self.reach_gate_review())
    self.assertIn("a prover just submitted this decomposition", context)
    self.assertIn("be demanding", context)

  def test_the_briefing_lists_the_branch_under_review(self):
    context = self.briefing(self.reach_gate_review())
    self.assertIn("## 9. Subtree Ledger", context)
    for child in ROOT_CHILDREN:
      self.assertIn(f"`{child}`", context)

  def test_the_briefing_shows_the_glue_the_children_are_meant_to_close(self):
    self.assertIn(
        "Combined into the target by this sketch",
        self.briefing(self.reach_gate_review()),
    )

  def test_the_briefing_points_at_the_blueprint(self):
    self.assertIn("## 8. Plan of Record", self.briefing(self.reach_gate_review()))

  def test_the_offered_actions_are_the_review_row(self):
    self.reach_gate_review()
    self.make().release_subgoal(ROOT_ID)
    out = self.run_cli("claim_next")
    self.assertEqual(out.metadata["allowed_actions"], ["approved", "replan"])

  # -- submission

  def test_a_verdict_without_a_reason_is_refused(self):
    self.reach_gate_review()
    handled, msg = self.declare(ROOT_ID, "replan", "bad")
    self.assertFalse(handled)
    self.assertIn("not enough", msg)

  def test_an_action_from_another_phase_is_refused(self):
    self.reach_gate_review()
    handled, msg = self.declare(
        ROOT_ID, "planned", "The blueprint is written and closed."
    )
    self.assertFalse(handled)
    self.assertIn("'review'", msg)

  def test_a_verdict_on_a_node_with_no_live_branch_is_refused(self):
    self.put_in_phase(ROOT_ID, "review")
    handled, msg = self.declare(
        ROOT_ID, "replan", "There is nothing here worth keeping."
    )
    self.assertFalse(handled)
    self.assertIn("no live decomposition", msg)

  def test_a_reviewer_cannot_hand_over_a_file(self):
    claim = self.reach_gate_review()
    self.assertEqual(claim["scaffold_path"], "")
    path = os.path.join(claim["node_dir"], "proof_scaffold.lean")
    with open(path, "w", encoding="utf-8") as f:
      f.write("theorem x : True := by trivial")
    out = self.run_cli("submit", code_file=path)
    self.assertFalse(out.success)
    self.assertIn("writes no Lean", out.error_message)
    # And the branch it might have retired is still standing.
    self.assertFalse(self.node().and_nodes[0].failed)

  # -- transition

  def test_approved_returns_the_node_to_blocked_and_releases_the_children(self):
    self.reach_gate_review()
    handled, msg = self.declare(
        ROOT_ID, "approved", "The three children close the target as sketched."
    )
    self.assertTrue(handled, msg)
    self.assertEqual(self.phase_of(), "blocked")
    self.assertIn(self.make().claim_next_subgoal()["node_id"], ROOT_CHILDREN)

  def test_approval_marks_the_plan_rather_than_exempting_the_node(self):
    # A permanent exemption would let an approved branch grow without bound.
    # The mark is on the plan, so a later `replan` takes it out with the plan.
    self.reach_gate_review()
    self.declare(ROOT_ID, "approved", "The three children close the target.")
    self.assertTrue(review_trigger.has_been_reviewed(self.node()))
    # Nothing has run below it yet, so the fresh plan gets the full allowance.
    self.assertEqual(self.node().active_and_node.review_replan_rate, 0.0)
    self.assertEqual(
        review_trigger.review_threshold(self.node()),
        float(pipeline.SUBTREE_REVIEW_REPLANS_PER_NODE),
    )

  def test_replan_retires_the_branch_and_sends_the_node_back_to_planning(self):
    self.reach_gate_review()
    handled, msg = self.declare(
        ROOT_ID, "replan", "Step 2 is the whole problem under a new name."
    )
    self.assertTrue(handled, msg)
    self.assertEqual(self.phase_of(), "informal")
    self.assertTrue(self.node().and_nodes[0].failed)
    self.assertIn("RETIRED BY REVIEW", self.node().and_nodes[0].error_message)

  def test_a_retired_branch_does_not_kill_the_node(self):
    # A reviewer rules on plans, not on goals.
    self.reach_gate_review()
    self.declare(ROOT_ID, "replan", "Step 2 is the whole problem under a new name.")
    self.assertFalse(self.node().search_metadata.get("decomposition_failed"))
    self.assertEqual(
        self.load().compute_statuses()[ROOT_ID], dag.NodeStatus.FORMAL_STATEMENT
    )

  def test_the_node_replans_rather_than_dispatching_the_retired_children(self):
    self.reach_gate_review()
    self.declare(ROOT_ID, "replan", "Step 2 is the whole problem under a new name.")
    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["node_id"], ROOT_ID)
    self.assertEqual(claim["phase"], "informal")

  def test_the_replanned_node_gets_to_keep_the_plan_it_just_made(self):
    """`and_nodes` never forgets a retired branch, so "no live decomposition"
    stays true for the rest of the node's life. Applied on every walk instead
    of only out of `blocked`, it dragged the phase back to `informal` after
    each planning turn and the node was dispatched to a planner forever, with
    `--action=failed` as the only way out.
    """
    self.reach_gate_review()
    self.declare(ROOT_ID, "replan", "Step 2 is the whole problem under a new name.")
    self.make().claim_next_subgoal()
    handled, msg = self.declare(ROOT_ID, "planned", "A second blueprint, from scratch.")
    self.assertTrue(handled, msg)
    self.assertEqual(self.phase_of(), "formal")

    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["node_id"], ROOT_ID)
    self.assertEqual(claim["phase"], "formal")
    self.assertEqual(claim["action_type"], "lemma_proof_worker")

  def test_a_worker_on_a_retired_child_is_released_as_an_orphan(self):
    """The child stays in the graph; it is simply no longer reachable."""
    self.reach_blocked()
    child = self.make().claim_next_subgoal()["node_id"]
    self.put_in_phase(ROOT_ID, "review")
    self.declare(ROOT_ID, "replan", "The whole split was wrong from the start.")
    handled, msg = self.declare(
        child, "planned", "My blueprint for this child is finished."
    )
    self.assertTrue(handled, msg)
    self.assertIn(child, self.load().nodes)


class ReviewStatePatrolTest(StateMachineTestCase):
  """The other entrance: graph arithmetic, on a subtree already running.

  The trigger is a re-planning rate per live node, not a depth. Depth measured
  the wrong thing -- a node whose decompositions keep getting retired collapses
  its own live subtree back to one generation each time, so its ancestors'
  reading never moved and nothing ever woke them.

  The gate approval at the top of every fixture here marks the root's plan with
  a reading of 0.0, so the root starts back at the floor.
  """

  FLOOR = pipeline.SUBTREE_REVIEW_REPLANS_PER_NODE

  def stall(self, rate, root=ROOT_ID):
    """Makes every live node below `root` look re-planned `rate` times."""
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      for nid in graph.get_active_reachable_nodes(root) - {root}:
        graph.nodes[nid].search_metadata["informal_dispatches"] = rate

  def reach_patrol(self):
    """Stalled enough to clear the floor, quiet enough to clear it once.

    Exactly the floor: over the bar the gate approval left at the floor, and
    under the doubled bar the patrol approval will raise it to. A subtree that
    tripped both would re-fire immediately and the two transitions could not be
    told apart.
    """
    self.reach_blocked()
    self.stall(self.FLOOR)
    return self.make().claim_next_subgoal()

  # -- input

  def test_a_stalling_branch_pulls_its_parent_back_into_review(self):
    claim = self.reach_patrol()
    self.assertEqual(claim["node_id"], ROOT_ID)
    self.assertEqual(claim["phase"], "review")

  def test_a_branch_that_is_merely_working_does_not(self):
    self.reach_blocked()
    self.stall(self.FLOOR - 1)
    self.assertIn(self.make().claim_next_subgoal()["node_id"], ROOT_CHILDREN)

  def test_a_localized_stall_is_reviewed_at_the_node_above_it(self):
    """Dilution is what makes one predicate serve both scopes.

    A node reads the worst of its own children, and each child is read through
    its whole branch. So the stall's parent sees the two stalled nodes at the
    floor, while the root sees a mean over the branch that holds them plus the
    quiet node above them -- well under the floor. The top-down walk descends
    past the root and stops at the stall.
    """
    self.reach_blocked()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      for i in range(2):
        nid = f"{CHILD_ID}_step_{i}"
        graph.add_node(
            dag.ProofGraphNode(
                id=nid,
                informal_problem=models.InformalProblem(
                    statement=f"theorem {nid} : True"
                ),
            )
        )
        graph.nodes[nid].search_metadata["informal_dispatches"] = self.FLOOR
      kids = [f"{CHILD_ID}_step_{i}" for i in range(2)]
      graph.nodes[CHILD_ID].and_nodes.append(
          dag.AndNode(
              id=f"d_{CHILD_ID}",
              dependency_ids=kids,
              subgoal_signatures=_sigs(kids),
          )
      )
      graph.nodes[CHILD_ID].search_metadata["phase"] = "blocked"
      # The healthy siblings are out with workers. Without this the walk hands
      # out ordinary work first and never reaches the predicate at all, which
      # is correct scheduling but tests nothing about the trigger.
      for sibling in ROOT_CHILDREN:
        if sibling != CHILD_ID:
          graph.nodes[sibling].search_metadata["processing"] = True
    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["node_id"], CHILD_ID)
    self.assertEqual(claim["phase"], "review")

  def test_the_shallowest_over_threshold_node_is_the_one_reviewed(self):
    """A replan at the top makes every deeper review moot instead of racing it."""
    self.assertEqual(self.reach_patrol()["node_id"], ROOT_ID)

  def test_the_briefing_says_this_is_the_second_look(self):
    context = self.briefing(self.reach_patrol())
    self.assertIn("approved before", context)
    self.assertIn("default to `--action=approved`", context)

  def test_the_briefing_names_the_rate_that_triggered_it(self):
    self.assertIn(
        f"{float(self.FLOOR):.1f} planning attempts on its worst child branch",
        self.briefing(self.reach_patrol()),
    )

  def test_the_two_entrances_read_differently(self):
    gate = self.briefing(self.reach_gate_review())
    self.assertIn("be demanding", gate)
    self.assertNotIn("approved before", gate)

  # -- transition

  def test_the_subtree_is_held_shut_while_the_reviewer_reads(self):
    # Children the reviewer may be about to retire must not be dispatched.
    self.reach_patrol()
    self.assertIsNone(self.make().claim_next_subgoal())

  def test_approving_again_raises_the_bar_again(self):
    self.reach_patrol()
    self.declare(
        ROOT_ID, "approved", "Slow, but the split is still the right one."
    )
    self.assertEqual(
        self.node().active_and_node.review_replan_rate, float(self.FLOOR)
    )
    self.assertEqual(
        review_trigger.review_threshold(self.node()),
        float(self.FLOOR) * pipeline.SUBTREE_REVIEW_BACKOFF,
    )

  def test_approving_lets_the_subtree_run_on_without_re_firing(self):
    self.reach_patrol()
    self.declare(
        ROOT_ID, "approved", "Slow, but the split is still the right one."
    )
    self.assertEqual(self.phase_of(), "blocked")
    self.assertNotEqual(self.make().claim_next_subgoal()["node_id"], ROOT_ID)

  def test_more_stalling_past_the_raised_bar_fires_it_again(self):
    # Reviews recur while a branch stalls, and that is the intent: there is no
    # budget here, so the only thing that ends a stall is somebody ruling on it.
    self.reach_patrol()
    self.declare(
        ROOT_ID, "approved", "Slow, but the split is still the right one."
    )
    self.stall(int(self.FLOOR * pipeline.SUBTREE_REVIEW_BACKOFF))
    self.assertEqual(self.make().claim_next_subgoal()["node_id"], ROOT_ID)

  def test_retiring_the_stalled_branch_orphans_the_work_below_it(self):
    self.reach_patrol()
    self.declare(
        ROOT_ID, "replan", "Four re-plans a node and nothing closed; start over."
    )
    self.assertEqual(self.phase_of(), "informal")
    self.assertEqual(self.load().get_active_reachable_nodes(ROOT_ID), {ROOT_ID})

  def test_the_replacement_plan_starts_back_at_the_floor(self):
    """The mark dies with the plan it was a judgement about."""
    self.reach_patrol()
    self.declare(
        ROOT_ID, "replan", "Four re-plans a node and nothing closed; start over."
    )
    self.assertFalse(review_trigger.has_been_reviewed(self.node()))
    self.assertEqual(
        review_trigger.review_threshold(self.node()), float(self.FLOOR)
    )


# --------------------------------------------------------------------------
# blocked -- waiting, not idle
# --------------------------------------------------------------------------


class BlockedStateTest(StateMachineTestCase):
  """A decomposed parent. Its edges are graph arithmetic, not declarations."""

  def prove_children(self):
    """Proves every child directly, in whatever order the walk offers them."""
    for child in ROOT_CHILDREN:
      self.put_in_phase(child, "formal")
    for _ in ROOT_CHILDREN:
      claim = self.make().claim_next_subgoal()
      self.assertIn(claim["node_id"], ROOT_CHILDREN)
      node = self.node(claim["node_id"])
      handled, msg = self.submit(
          claim,
          f"theorem {claim['node_id']} "
          f"{node.informal_problem.statement} := by\n  trivial",
      )
      self.assertTrue(handled, msg)

  # -- input

  def test_a_blocked_parent_is_never_dispatched(self):
    self.reach_blocked()
    claim = self.make().claim_next_subgoal()
    self.assertIn(claim["node_id"], ROOT_CHILDREN)

  def test_the_blocked_phase_has_no_role_and_no_actions(self):
    self.reach_blocked()
    phase = pipeline.phase_for(self.node())
    self.assertEqual(phase.name, "blocked")
    self.assertIsNone(phase.role)
    self.assertEmpty(phase.actions)

  # -- transition

  def test_the_claim_walk_descends_into_the_children(self):
    self.reach_blocked()
    mgr = self.make()
    claimed = set()
    for _ in range(len(ROOT_CHILDREN)):
      claim = mgr.claim_next_subgoal()
      self.assertIsNotNone(claim)
      claimed.add(claim["node_id"])
    self.assertEqual(claimed, set(ROOT_CHILDREN))

  def test_the_children_are_offered_in_the_order_the_decomposition_declared(self):
    """Nothing reorders them: the prover chose the order and owns the choice."""
    self.reach_blocked()
    mgr = self.make()
    order = [mgr.claim_next_subgoal()["node_id"] for _ in ROOT_CHILDREN]
    self.assertEqual(order, ROOT_CHILDREN)

  def test_all_children_proved_resolves_the_parent(self):
    self.reach_blocked()
    self.prove_children()
    self.assertEqual(
        self.load().propagate_failures()[ROOT_ID],
        dag.NodeStatus.FORMAL_PROVED,
    )

  def test_a_resolved_parent_leaves_the_machine(self):
    self.reach_blocked()
    self.prove_children()
    self.assertIsNone(self.make().claim_next_subgoal())
    self.assertIsNone(self.phase_of())

  def test_a_dead_branch_returns_the_parent_to_planning(self):
    self.reach_blocked()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      graph.nodes[ROOT_ID].and_nodes[0].failed = True
    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["node_id"], ROOT_ID)
    self.assertEqual(claim["phase"], "informal")

  def test_a_dead_branch_does_not_kill_the_parent(self):
    """Only the planner concludes a goal is dead, and it has not been asked yet."""
    self.reach_blocked()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      graph.nodes[ROOT_ID].and_nodes[0].failed = True
    self.make().claim_next_subgoal()
    self.assertFalse(self.node().search_metadata.get("decomposition_failed"))

  def test_resolution_is_asked_before_the_patrol(self):
    """Order on the row: a dead subtree is answered by that fact, not audited."""
    self.reach_blocked()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      graph.nodes[ROOT_ID].and_nodes[0].failed = True
    self.assertEqual(
        self.make().claim_next_subgoal()["phase"], "informal"
    )


# --------------------------------------------------------------------------
# disprove -- the route out for a target that is false
# --------------------------------------------------------------------------


class DisproveStateTest(StateMachineTestCase):
  """Take the target as an axiom and derive `False`. Never negate it."""

  WITNESS = (
      "At n = 1 there are no indices i with i+1 < 1, so every s is in the"
      " solution set while f is constant: the forward direction is vacuous"
      " but the reverse fails."
  )

  DERIVATION = (
      "theorem putnam_2025_a5_disproof : False :=\nby\n"
      "  exact absurd (putnam_2025_a5 1 le_rfl ![1]) (by decide)"
  )

  def reach_disprove(self):
    self.make().claim_next_subgoal()
    handled, msg = self.declare(ROOT_ID, "refute", self.WITNESS)
    self.assertTrue(handled, msg)
    return self.make().claim_next_subgoal()

  # -- input

  def test_the_disproof_agent_gets_its_own_scaffold(self):
    claim = self.reach_disprove()
    self.assertEqual(claim["phase"], "disprove")
    self.assertEqual(claim["action_type"], "lemma_proof_worker")
    self.assertEqual(claim["scaffold_role"], models.SCAFFOLD_DISPROVE)
    self.assertEqual(
        os.path.basename(claim["scaffold_path"]), "disproof_scaffold.lean"
    )

  def test_the_target_is_assumed_as_an_axiom_with_its_binders_intact(self):
    text = self.scaffold(self.reach_disprove())
    self.assertIn("axiom putnam_2025_a5", text)
    self.assertIn("(s : Fin n → ℤˣ)", text)
    self.assertNotIn("¬", text)

  def test_the_obligation_is_deriving_false(self):
    self.assertIn(
        "theorem putnam_2025_a5_disproof : False",
        self.scaffold(self.reach_disprove()),
    )

  def test_neither_mutable_header_region_is_rendered(self):
    # A region never offered cannot be filled in, so nothing derived from a
    # false axiom has a route into the shared header.
    text = self.scaffold(self.reach_disprove())
    self.assertNotIn("[START_DEFINITIONS]", text)
    self.assertNotIn("[START_SUBGOALS]", text)

  def test_the_briefing_frames_the_job_as_refutation(self):
    context = self.briefing(self.reach_disprove())
    self.assertIn("You are refuting this target, not proving it", context)

  def test_the_witness_is_carried_into_the_briefing(self):
    self.assertIn(
        "At n = 1 there are no indices", self.briefing(self.reach_disprove())
    )

  # -- submission

  def test_a_verified_derivation_is_accepted(self):
    claim = self.reach_disprove()
    handled, msg = self.submit(
        claim,
        self.DERIVATION,
        verifier=_verifier(used_axioms=("propext", ROOT_ID)),
    )
    self.assertTrue(handled, msg)

  def test_restating_the_obligation_is_refused(self):
    claim = self.reach_disprove()
    handled, msg = self.submit(
        claim,
        "theorem putnam_2025_a5_disproof : 0 = 0 :=\nby\n  rfl",
        verifier=_verifier(used_axioms=("propext", ROOT_ID)),
    )
    self.assertFalse(handled)
    self.assertIn("putnam_2025_a5_disproof : False", msg)

  def test_a_derivation_that_never_used_the_target_is_an_alarm(self):
    claim = self.reach_disprove()
    handled, msg = self.submit(
        claim, self.DERIVATION, verifier=_verifier(used_axioms=("propext",))
    )
    self.assertFalse(handled)
    self.assertIn("ENVIRONMENT INCONSISTENT", msg)

  def test_a_sorried_derivation_is_refused(self):
    claim = self.reach_disprove()
    handled, msg = self.submit(
        claim,
        "theorem putnam_2025_a5_disproof : False :=\nby\n  sorry",
        verifier=_verifier(used_axioms=("propext", ROOT_ID)),
    )
    self.assertFalse(handled)

  # -- transition

  def test_a_disproved_node_is_marked_dead_with_the_reason(self):
    claim = self.reach_disprove()
    handled, msg = self.submit(
        claim,
        self.DERIVATION,
        verifier=_verifier(used_axioms=("propext", ROOT_ID)),
    )
    self.assertTrue(handled, msg)
    self.assertIsNone(self.phase_of())
    self.assertTrue(self.node().search_metadata["decomposition_failed"])
    self.assertIn("DISPROVED", self.node().search_metadata["last_error"])

  def test_nothing_from_the_disproof_reaches_the_shared_header(self):
    # Every line of it is sound only under a false axiom.
    before = self.load().environment.to_lean_header()
    claim = self.reach_disprove()
    self.submit(
        claim,
        self.DERIVATION,
        verifier=_verifier(used_axioms=("propext", ROOT_ID)),
    )
    self.assertEqual(self.load().environment.to_lean_header(), before)

  def test_a_failed_disproof_says_nothing_about_the_node(self):
    claim = self.reach_disprove()
    handled, _ = self.submit(
        claim,
        self.DERIVATION,
        verifier=_verifier(is_valid=False, error="unknown identifier 'foo'"),
    )
    self.assertFalse(handled)
    self.assertFalse(self.node().search_metadata.get("decomposition_failed"))
    self.assertEqual(self.phase_of(), "disprove")

  def test_abandoning_the_refutation_returns_the_node_to_planning(self):
    self.reach_disprove()
    handled, msg = self.declare(
        ROOT_ID,
        "replan",
        "The counterexample was wrong: at n = 1 the reverse direction holds"
        " vacuously too.",
    )
    self.assertTrue(handled, msg)
    self.assertEqual(self.phase_of(), "informal")
    self.assertFalse(self.node().search_metadata.get("decomposition_failed"))


# --------------------------------------------------------------------------
# Leaving the machine
# --------------------------------------------------------------------------


class TerminationTest(StateMachineTestCase):
  """The last transition: a tree with nothing left to claim becomes a proof.

  Every other class here ends at a phase. This one runs the root's real
  decomposition to the bottom and asks the question the orchestrator asks --
  *is it finished?* -- which is `--action=assemble`, because that is the only
  thing that can answer it: `assemble_current_proof` stubs each open node with
  `sorry` and produces a file either way, so the completeness guard in front of
  it is the whole check.
  """

  def prove(self, node_id):
    """Walks one node from `informal` to proved, the way the pool would."""
    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["node_id"], node_id, "claimed out of expected order")
    self.assertEqual(claim["phase"], "informal")
    handled, msg = self.declare(
        node_id, "planned", "Direct; no decomposition is needed here."
    )
    self.assertTrue(handled, msg)

    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["node_id"], node_id)
    self.assertEqual(claim["phase"], "formal")
    # The scaffold already carries the obligation, so discharging it is a
    # one-token edit -- which is the point: nothing about the shape of the
    # submission changes between a leaf and a decomposition.
    body = self.scaffold(claim).split("-- [START_PROOF]")[1]
    body = body.split("-- [END_PROOF]")[0].strip()
    handled, msg = self.submit(claim, body.replace("sorry", "trivial"))
    self.assertTrue(handled, msg)

  def finish(self):
    """Proves every child of the root's approved decomposition, in order."""
    self.reach_blocked()
    for node_id in ROOT_CHILDREN:
      self.prove(node_id)

  def assemble(self):
    return self.run_cli("assemble")

  def test_the_guard_refuses_while_the_children_are_open(self):
    self.reach_blocked()
    out = self.assemble()
    self.assertFalse(out.success)
    self.assertIn("ASSEMBLY INCOMPLETE", out.error_message)
    for node_id in [ROOT_ID] + ROOT_CHILDREN:
      self.assertIn(f"`{node_id}`", out.error_message)

  def test_the_guard_refuses_with_one_child_left(self):
    """The failure that motivated the guard: nearly done still reads as done."""
    self.reach_blocked()
    self.prove("exists_max_f")
    self.prove("not_sol_implies_lt_max")
    out = self.assemble()
    self.assertFalse(out.success)
    self.assertIn("`f_sol_eq`", out.error_message)

  def test_the_refusal_reads_as_a_worklist_not_an_error(self):
    self.reach_blocked()
    self.assertIn("Keep dispatching", self.assemble().error_message)

  def test_nothing_is_written_when_the_guard_refuses(self):
    self.reach_blocked()
    self.assertIsNone(self.assemble().assembled_code)

  def test_the_last_child_resolves_the_root(self):
    self.finish()
    self.assertEqual(
        self.load().compute_statuses()[ROOT_ID], dag.NodeStatus.FORMAL_PROVED
    )

  def test_a_finished_tree_has_nothing_left_to_claim(self):
    self.finish()
    self.assertIsNone(self.make().claim_next_subgoal())

  def test_a_finished_tree_passes_the_guard(self):
    self.finish()
    out = self.assemble()
    self.assertTrue(out.success, out.error_message)

  def test_the_assembled_proof_carries_every_lemma_and_no_sorry(self):
    self.finish()
    code = self.assemble().assembled_code
    for node_id in ROOT_CHILDREN:
      self.assertIn(node_id, code)
    self.assertNotIn("sorry", code)

  # -- The two ways the pool runs dry, and why they must read differently.

  def claim_via_cli(self):
    return self.run_cli("claim_next").error_message

  def test_a_dry_pool_with_work_running_says_wait(self):
    self.reach_blocked()
    while self.make().claim_next_subgoal():
      pass
    self.assertIn("STILL processing", self.claim_via_cli())

  def test_a_dry_pool_on_a_proved_tree_says_the_search_is_finished(self):
    self.finish()
    self.assertIn("SEARCH FINISHED", self.claim_via_cli())

  def test_the_pool_is_not_dry_after_a_closed_route(self):
    """It used to be: `failed` on the root emptied the pool and the run was
    over. The node now comes back, so there is something to hand out and the
    orchestrator is never told to report a defective problem."""
    self.make().claim_next_subgoal()
    self.declare(ROOT_ID, "failed", "Nothing here is provable as stated.")
    self.assertIsNotNone(self.make().claim_next_subgoal())

  def test_inspect_agrees_with_what_claim_next_said(self):
    self.finish()
    counts = self.run_cli("inspect").metadata["graph_summary"]
    self.assertEqual(counts["failed"], 0)
    self.assertEqual(counts["formal_proved"], counts["total_nodes"])

  def test_a_closed_route_does_not_read_as_a_failed_node(self):
    self.make().claim_next_subgoal()
    self.declare(ROOT_ID, "failed", "Nothing here is provable as stated.")
    counts = self.run_cli("inspect").metadata["graph_summary"]
    self.assertEqual(counts["failed"], 0)
    self.assertEqual(counts["formal_proved"], 0)

  def test_a_closed_route_still_refuses_to_assemble(self):
    self.make().claim_next_subgoal()
    self.declare(ROOT_ID, "failed", "Nothing here is provable as stated.")
    self.assertFalse(self.assemble().success)


# --------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------


class ReachabilityTest(StateMachineTestCase):
  """Every phase the table declares is exercised above, and nothing else is."""

  def test_the_states_walked_here_are_the_whole_table(self):
    walked = {"informal", "formal", "review", "blocked", "disprove"}
    self.assertEqual(walked, set(pipeline.PIPELINE))

  def test_a_claim_carries_everything_a_dispatch_prompt_needs(self):
    out = self.run_cli("claim_next")
    self.assertTrue(out.success)
    for field in ("node_id", "action_type", "node_dir", "context_path"):
      self.assertTrue(getattr(out, field), f"{field} is empty")
    self.assertCountEqual(
        out.metadata, ["phase", "role", "scaffold_role", "allowed_actions"]
    )

  def test_the_dispatch_role_and_the_phases_role_are_the_same_value(self):
    out = self.run_cli("claim_next")
    self.assertEqual(out.action_type, out.metadata["role"])
    self.assertEqual(
        out.action_type, pipeline.PIPELINE[out.metadata["phase"]].role
    )


if __name__ == "__main__":
  absltest.main()
