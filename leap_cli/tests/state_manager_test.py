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

"""Tests for ProofGraphStateManager: claim lifecycle, gates, and commit.

The Lean verifier is stubbed, so these run offline and deterministically.
"""

import _path  # noqa: F401  (puts leap/ on sys.path)
import os
import re
import shlex
import tempfile
import threading
import time
from unittest import mock

from absl.testing import absltest

import dag
import lean_code_parser as models
import pipeline
import review_trigger
import state_manager as sm
import workspace_manager as wm


def _sigs(names):
  """Subgoal signatures for a hand-built AndNode.

  Nothing rejects an AndNode that names children without their statements any
  more, but a fixture that omits them is a legacy-format node and behaves like
  one, so build them here unless that is the point of the test.
  """
  return [
      models.FormalSignature(name=n, statement=f"Claim_{n}") for n in names
  ]


def _fake_verifier(is_valid=True, used_axioms=("propext",), error=None):
  """Returns a stand-in for run_lean_verifier with a fixed verdict."""

  def _run(input_data):
    del input_data
    return sm.lean_verifier_tool.LeanVerifierOutput(
        is_valid=is_valid,
        used_axioms=None if used_axioms is None else list(used_axioms),
        error_message=error,
    )

  return _run


class ManagerTestCase(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.filepath = os.path.join(self.root, "proof_graph.json")

  def make(self, root_id="root"):
    return sm.ProofGraphStateManager(
        filepath=self.filepath, root_dir=self.root, root_id=root_id
    )

  def seed(self, statement="theorem root (a : Nat) : a + 0 = a"):
    graph = dag.ProofGraph()
    graph.add_node(
        dag.ProofGraphNode(
            id="root", informal_problem=models.InformalProblem(statement=statement)
        )
    )
    self.make().initialize_graph(graph)
    return graph

  def load(self):
    return self.make().read_graph()

  def write_scaffold(self, node_id, role, definitions="", subgoals="", proof=""):
    path = os.path.join(
        self.root, "nodes", node_id, models.scaffold_filename(role)
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      f.write(
          models.build_lean_scaffold(
              immutable_context="import Mathlib",
              definitions=definitions,
              subgoals=subgoals,
              proof=proof,
              role=role,
          )
      )
    return path

  def put_in_phase(self, node_id, phase_name):
    with self.make().lock_and_modify() as graph:
      graph.nodes[node_id].search_metadata["phase"] = phase_name

  def claim_formal(self, mgr=None, node_id="root"):
    """Claims `node_id` for `formal`, skipping the planning turn.

    A freshly seeded node is claimed for `informal`, which writes no Lean and
    hands out no scaffold, so anything about scaffolds or submissions has to
    move the node on first.
    """
    self.put_in_phase(node_id, "formal")
    claim = (mgr or self.make()).claim_next_subgoal()
    self.assertEqual(claim["node_id"], node_id)
    return claim


class ClaimLifecycleTest(ManagerTestCase):
  """Who holds a node, and how a hold ends.

  A claim never expires. The TTL that used to steal a node back from a slow
  worker was a guess about how long Lean takes, and it raced: two agents wrote
  the same scaffold and the second one's submission was rejected as stale.
  Liveness is the orchestrator's job now, and `release` is how it does it.
  """

  def test_a_claim_marks_the_node_and_blocks_a_second_claim(self):
    self.seed()
    mgr = self.make()
    first = mgr.claim_next_subgoal()
    self.assertEqual(first["node_id"], "root")
    self.assertIsNone(mgr.claim_next_subgoal(), "a held claim must not be reissued")

  def test_a_claim_does_not_expire_however_old_it_is(self):
    self.seed()
    mgr = self.make()
    mgr.claim_next_subgoal()
    with mgr.lock_and_modify() as graph:
      graph.nodes["root"].search_metadata["claim_timestamp"] = 0.0
    self.assertIsNone(
        mgr.claim_next_subgoal(), "a slow worker must not have its node stolen"
    )

  def test_the_claim_is_stamped_so_a_stuck_worker_can_be_found(self):
    # Nothing reads the stamp to make a decision; the orchestrator reads it to
    # decide whether to `release`.
    self.seed()
    before = time.time()
    self.make().claim_next_subgoal()
    self.assertGreaterEqual(
        self.load().nodes["root"].search_metadata["claim_timestamp"], before
    )

  def test_release_makes_the_node_claimable_again(self):
    self.seed()
    mgr = self.make()
    mgr.claim_next_subgoal()
    self.assertTrue(mgr.release_subgoal("root"))
    self.assertIsNotNone(mgr.claim_next_subgoal())

  def test_release_of_unknown_node_reports_failure(self):
    self.seed()
    self.assertFalse(self.make().release_subgoal("nope"))

  def test_release_does_not_check_who_is_asking(self):
    """The orchestrator is recovering a worker that is by definition not
    around to prove it owned the node."""
    self.seed()
    self.make().claim_next_subgoal()
    self.assertTrue(self.make().release_subgoal("root"))

  def test_a_failed_branch_returns_a_blocked_node_to_planning(self):
    """Otherwise the node sits in `blocked` forever with nothing left to wait on."""
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      node = graph.nodes["root"]
      node.search_metadata["phase"] = "blocked"
      node.formal_sketch = models.FormalProofSketch(
          signature=models.FormalSignature(name="root", statement="True"),
          proof=models.FormalProof(tactics="by trivial"),
      )
      node.and_nodes.append(
          dag.AndNode(id="d0", dependency_ids=["ghost"], subgoal_signatures=_sigs(["ghost"]), failed=True)
      )
    claim = mgr.claim_next_subgoal()
    self.assertEqual(claim["action_type"], "lemma_informal_prover")
    self.assertEqual(claim["phase"], "informal")
    self.assertEqual(
        self.load().nodes["root"].search_metadata["phase"], "informal"
    )


class GetContextTest(ManagerTestCase):
  """`--action=get_context` is a read. It must not touch the agent's files."""

  def test_it_returns_the_scaffold_as_the_agent_left_it(self):
    self.seed()
    mgr = self.make()
    claim = self.claim_formal(mgr)
    with open(claim["scaffold_path"], "w", encoding="utf-8") as f:
      f.write("AGENT WORK")
    ctx = mgr.get_node_context("root")
    self.assertIn("AGENT WORK", ctx["assembled_lean_scaffold"])

  def test_it_does_not_rewrite_the_scaffold(self):
    """`get_context` used to provision, which rotated the caller's own
    in-progress file out from under it mid-turn."""
    self.seed()
    mgr = self.make()
    claim = self.claim_formal(mgr)
    with open(claim["scaffold_path"], "w", encoding="utf-8") as f:
      f.write("AGENT WORK")
    mgr.get_node_context("root")
    with open(claim["scaffold_path"], encoding="utf-8") as f:
      self.assertEqual(f.read(), "AGENT WORK")

  def test_it_leaves_no_rotated_copy_behind(self):
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    before = sorted(os.listdir(os.path.join(self.root, "nodes", "root")))
    mgr.get_node_context("root")
    self.assertEqual(
        sorted(os.listdir(os.path.join(self.root, "nodes", "root"))), before
    )

  def test_an_unknown_node_reads_as_nothing(self):
    self.seed()
    self.assertIsNone(self.make().get_node_context("nope"))

  def test_the_phase_and_the_role_come_from_the_node(self):
    self.seed()
    ctx = self.make().get_node_context("root")
    self.assertEqual(ctx["phase"], "informal")
    self.assertEqual(ctx["action_type"], "lemma_informal_prover")

  def test_a_phase_with_no_scaffold_reports_no_scaffold_path(self):
    self.seed()
    self.assertEqual(self.make().get_node_context("root")["scaffold_path"], "")

  def test_re_provisioning_rotates_the_old_scaffold_aside(self):
    """A node is re-provisioned when its plan is redone. Overwriting in place
    silently discarded whatever the previous agent had written."""
    self.seed()
    mgr = self.make()
    claim = self.claim_formal(mgr)
    with open(claim["scaffold_path"], "w", encoding="utf-8") as f:
      f.write("AGENT WORK")
    mgr.release_subgoal("root")
    self.claim_formal(mgr)
    rotated = [
        f
        for f in os.listdir(os.path.join(self.root, "nodes", "root"))
        if f.startswith("proof_scaffold_old_")
    ]
    self.assertLen(rotated, 1)
    with open(
        os.path.join(self.root, "nodes", "root", rotated[0]), encoding="utf-8"
    ) as f:
      self.assertEqual(f.read(), "AGENT WORK")

  def test_re_provisioning_rotates_the_turn_journals_too(self):
    """The prover's log is one turn's, and appending across turns lies.

    `fmf_v1_004` shipped journals where a fresh agent appended to the last
    one's entries, and the next reader could not tell where one turn ended and
    the next began.
    """
    self.seed()
    mgr = self.make()
    claim = self.claim_formal(mgr)
    log = os.path.join(claim["node_dir"], "scratch", "prover_log.md")
    with open(log, "w", encoding="utf-8") as f:
      f.write("TURN ONE")
    mgr.release_subgoal("root")
    self.claim_formal(mgr)
    self.assertFalse(os.path.exists(log))
    rotated = [
        f
        for f in os.listdir(os.path.join(self.root, "nodes", "root", "scratch"))
        if f.startswith("prover_log_old_")
    ]
    self.assertLen(rotated, 1)

  def test_the_cross_turn_record_is_never_rotated(self):
    """`findings.md` and the blueprint are what survives to the next attempt.

    Rotating either would make a re-claimed node forget what it had already
    ruled out and derive it again, which is the loop the journal exists to
    break rather than cause.
    """
    self.seed()
    mgr = self.make()
    claim = self.claim_formal(mgr)
    keep = {
        os.path.join(claim["node_dir"], "findings.md"): "ESTABLISHED",
        os.path.join(claim["node_dir"], "scratch", "blueprint.md"): "THE PLAN",
    }
    for path, text in keep.items():
      with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    mgr.release_subgoal("root")
    self.claim_formal(mgr)
    for path, text in keep.items():
      with open(path, encoding="utf-8") as f:
        self.assertEqual(f.read(), text, path)

  def test_the_fresh_scaffold_is_the_one_the_agent_is_pointed_at(self):
    self.seed()
    mgr = self.make()
    claim = self.claim_formal(mgr)
    with open(claim["scaffold_path"], "w", encoding="utf-8") as f:
      f.write("AGENT WORK")
    mgr.release_subgoal("root")
    with open(self.claim_formal(mgr)["scaffold_path"], encoding="utf-8") as f:
      self.assertIn("[START_PROOF]", f.read())


class AxiomGateTest(ManagerTestCase):

  def _verify(self, used_axioms, allowed=None):
    graph = self.seed()
    if allowed is not None:
      graph.allowed_axioms = allowed
    with mock.patch.object(
        sm.lean_verifier_tool,
        "run_lean_verifier",
        _fake_verifier(used_axioms=used_axioms),
    ):
      return self.make()._verify_before_commit(
          graph=graph,
          node_id="root",
          definitions="",
          proof_text="theorem root (a : Nat) : a + 0 = a :=\n  by simp",
          is_decomposition=False,
      )

  def test_standard_axioms_pass(self):
    self.assertTrue(self._verify(["propext", "Classical.choice"]).ok)

  def test_no_axioms_passes(self):
    self.assertTrue(self._verify([]).ok)

  def test_sorry_ax_rejected(self):
    outcome = self._verify(["sorryAx"])
    self.assertFalse(outcome.ok)
    self.assertIn("sorryAx", outcome.error_message)

  def test_native_decide_rejected(self):
    outcome = self._verify(["Lean.ofReduceBool", "Lean.trustCompiler"])
    self.assertFalse(outcome.ok)
    self.assertIn("disallowed axioms", outcome.error_message)

  def test_unparseable_axiom_list_fails_closed(self):
    outcome = self._verify(None)
    self.assertFalse(outcome.ok)
    self.assertIn("AXIOM_CHECK_FAILED", outcome.error_message)

  def test_allowed_axioms_override_is_honoured(self):
    self.assertTrue(
        self._verify(
            ["Lean.ofReduceBool"],
            allowed=["propext", "Classical.choice", "Quot.sound",
                     "Lean.ofReduceBool"],
        ).ok
    )


class SyntacticGateTest(ManagerTestCase):

  def setUp(self):
    super().setUp()
    self.node = dag.ProofGraphNode(
        id="n", informal_problem=models.InformalProblem(statement="True")
    )
    self.subgoals = [models.FormalSignature(name="sub_one", statement="True")]

  def test_sorry_in_definitions_rejected_on_both_paths(self):
    defs = "lemma h : False := by sorry"
    self.assertIsNotNone(
        sm._validate_proof_syntactic_gates(self.node, "by simp", defs)
    )
    err, _ = sm._validate_decomp_syntactic_gates(
        self.node, "exact sub_one", self.subgoals, defs
    )
    self.assertIsNotNone(err)

  def test_axiom_in_definitions_rejected_on_decompose_path(self):
    err, _ = sm._validate_decomp_syntactic_gates(
        self.node, "exact sub_one", self.subgoals, "axiom h : False"
    )
    self.assertIsNotNone(err)

  def test_a_decomposition_declaring_nothing_is_refused(self):
    """Unreachable through `record_lean`, which reads an empty subgoals block
    as a proof. The gate stands because an AND-node with no dependencies
    resolves its parent to FORMAL_PROVED with nothing supporting it."""
    err, early = sm._validate_decomp_syntactic_gates(
        self.node, "exact trivial", []
    )
    self.assertIsNotNone(err)
    self.assertFalse(early)

  def _retired(self, names, retired_for):
    """A dead AND-node over `names`, as a reviewer verdict would leave it."""
    self.node.and_nodes.append(
        dag.AndNode(
            id=f"d{len(self.node.and_nodes)}",
            dependency_ids=list(names),
            subgoal_signatures=_sigs(names),
            failed=True,
            retired_for=retired_for,
        )
    )

  def _resubmit(self, names):
    return sm._validate_decomp_syntactic_gates(
        self.node, "exact " + names[0], _sigs(names)
    )

  def test_repeating_a_shape_retired_strategy_is_refused(self):
    self._retired(["a", "b"], "shape")
    err, early = self._resubmit(["a", "b"])
    self.assertIn("previously failed AND-node", err)
    self.assertFalse(early)

  def test_repeating_a_strategy_retired_before_this_field_existed_is_refused(self):
    """An empty `retired_for` is every AND-node written by an older run, and a
    prover that stalls out. Only an explicit `evidence` opens the gate."""
    self._retired(["a", "b"], "")
    err, _ = self._resubmit(["a", "b"])
    self.assertIsNotNone(err)

  def test_an_evidence_replan_may_be_answered_with_the_same_split(self):
    self._retired(["a", "b"], "evidence")
    err, _ = self._resubmit(["a", "b"])
    self.assertIsNone(err)

  def test_the_evidence_pass_is_spent_the_first_time_it_is_used(self):
    """Two dead attempts on one statement set means the revision already
    happened and died. There is no attempt budget to stop a third."""
    self._retired(["a", "b"], "evidence")
    self._retired(["a", "b"], "evidence")
    err, _ = self._resubmit(["a", "b"])
    self.assertIsNotNone(err)

  def test_a_different_split_is_unaffected_by_a_dead_evidence_attempt(self):
    self._retired(["a", "b"], "evidence")
    err, _ = self._resubmit(["a", "c"])
    self.assertIsNone(err)

  def test_active_collision_is_a_rejection_not_a_success(self):
    self.node.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["other"], subgoal_signatures=_sigs(["other"])))
    err, early = sm._validate_decomp_syntactic_gates(
        self.node, "exact sub_one", self.subgoals
    )
    self.assertIsNotNone(err)
    self.assertFalse(early, "a hard rejection must not report success")

  def test_failed_strategy_cannot_be_resubmitted_with_a_padding_lemma(self):
    """dependency_ids holds only attributed subgoals; the guard must use the declared set."""
    self.node.and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["sub_one"],
            subgoal_signatures=[
                models.FormalSignature(name="sub_one", statement="0 = 0"),
                models.FormalSignature(name="sub_two", statement="1 = 1"),
            ],
            failed=True,
        )
    )
    err, _ = sm._validate_decomp_syntactic_gates(
        self.node,
        "exact sub_one",
        [
            models.FormalSignature(name="sub_one", statement="0 = 0"),
            models.FormalSignature(name="sub_two", statement="1 = 1"),
        ],
    )
    self.assertIsNotNone(err)
    self.assertIn("previously failed", err)

  def test_a_failed_strategy_cannot_come_back_under_new_names(self):
    """The fmf_v1_004 rename spiral. The gate is keyed on statements."""
    self.node.and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["sub_one"],
            subgoal_signatures=[
                models.FormalSignature(name="sub_one", statement="0 = 0")
            ],
            failed=True,
        )
    )
    err, _ = sm._validate_decomp_syntactic_gates(
        self.node,
        "exact sub_one_new",
        [models.FormalSignature(name="sub_one_new", statement="0 = 0")],
    )
    self.assertIsNotNone(err)
    self.assertIn("previously failed", err)

  def test_whitespace_is_not_a_new_strategy(self):
    """Statements are compared normalized, so reflowing one changes nothing."""
    self.node.and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["sub_one"],
            subgoal_signatures=[
                models.FormalSignature(name="sub_one", statement="0 = 0")
            ],
            failed=True,
        )
    )
    err, _ = sm._validate_decomp_syntactic_gates(
        self.node,
        "exact sub_one",
        [models.FormalSignature(name="sub_one", statement="0   =\n  0")],
    )
    self.assertIsNotNone(err)

  def test_a_rename_over_the_live_branch_is_the_idempotent_resubmit(self):
    """Same mathematics, new names, branch still live: a duplicate, not a
    collision. Reported as early success so a retrying agent is not told its
    own committed work belongs to someone else."""
    self.node.and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["sub_one"],
            subgoal_signatures=[
                models.FormalSignature(name="sub_one", statement="0 = 0")
            ],
        )
    )
    err, early = sm._validate_decomp_syntactic_gates(
        self.node,
        "exact sub_one_new",
        [models.FormalSignature(name="sub_one_new", statement="0 = 0")],
    )
    self.assertIsNotNone(err)
    self.assertTrue(early)

  def test_different_mathematics_under_a_reused_name_still_passes(self):
    """The converse of the rename spiral: the name is not what is compared."""
    self.node.and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["sub_one"],
            subgoal_signatures=[
                models.FormalSignature(name="sub_one", statement="0 = 0")
            ],
            failed=True,
        )
    )
    err, _ = sm._validate_decomp_syntactic_gates(
        self.node,
        "exact sub_one",
        [models.FormalSignature(name="sub_one", statement="1 = 1")],
    )
    self.assertIsNone(err)

  def test_validation_never_mutates_the_prior_branch(self):
    self.node.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["other"], subgoal_signatures=_sigs(["other"])))
    self.node.search_metadata["decomposition_failed"] = True
    sm._validate_decomp_syntactic_gates(
        self.node, "sorry", self.subgoals, "axiom h : False"
    )
    self.assertFalse(
        self.node.and_nodes[0].failed,
        "a rejected retry must leave the prior decomposition viable",
    )


class DefinitionsBlockGateTest(ManagerTestCase):
  """Content policy on definitions blocks, the only lever on hoisted headers."""

  def _graph(self):
    graph = dag.ProofGraph()
    graph.add_node(
        dag.ProofGraphNode(
            id="root", informal_problem=models.InformalProblem(statement="True")
        )
    )
    graph.environment.dynamic_headers.append(
        models.SharedHeaderBlock(
            content="def suffix_flip (n : Nat) : Nat := n", node_id="other"
        )
    )
    return graph

  def test_plain_helper_is_accepted(self):
    self.assertIsNone(
        sm._validate_definitions_block(
            self._graph(), "root", "def fresh_helper : Nat := 1"
        )
    )

  def test_empty_block_is_accepted(self):
    self.assertIsNone(sm._validate_definitions_block(self._graph(), "root", ""))

  def test_banned_form_is_rejected_before_any_lean_call(self):
    err = sm._validate_definitions_block(
        self._graph(), "root", "attribute [simp] Nat.add_zero"
    )
    self.assertIsNotNone(err)
    self.assertIn("ACTION REQUIRED", err)

  def test_collision_with_another_nodes_header_is_named(self):
    err = sm._validate_definitions_block(
        self._graph(), "root", "def suffix_flip (n : Nat) : Nat := n + 1"
    )
    self.assertIsNotNone(err)
    self.assertIn("suffix_flip", err)
    self.assertIn("other", err, "the message must name the owning node")

  def test_a_node_may_resubmit_its_own_block(self):
    graph = self._graph()
    graph.environment.dynamic_headers[0].node_id = "root"
    self.assertIsNone(
        sm._validate_definitions_block(
            graph, "root", "def suffix_flip (n : Nat) : Nat := n"
        )
    )

  def test_collision_with_an_existing_subgoal_name_is_rejected(self):
    graph = self._graph()
    graph.add_node(
        dag.ProofGraphNode(
            id="sub", informal_problem=models.InformalProblem(statement="True")
        )
    )
    err = sm._validate_definitions_block(graph, "root", "def sub : Nat := 1")
    self.assertIsNotNone(err)
    self.assertIn("sub", err)

  def test_a_raw_lean_collision_error_is_framed_as_a_rename(self):
    """Mathlib collisions reach the agent only as raw compiler output."""
    msg = sm.RecordMessages.verification_dumped(
        "root", "error: 'add_comm' has already been declared", "logs/x.lean",
        is_decomp=False,
    )
    self.assertIn("NAME COLLISION", msg)
    self.assertIn("Rename", msg)

  def test_an_ordinary_compiler_error_gets_no_rename_advice(self):
    msg = sm.RecordMessages.verification_dumped(
        "root", "error: unsolved goals", "logs/x.lean", is_decomp=False
    )
    self.assertNotIn("NAME COLLISION", msg)

  def test_concurrent_blocks_with_the_same_name_cannot_both_commit(self):
    """The pre-verification gate cannot see a block committed while it ran.

    Each node verifies against a header that does not yet contain the other's
    block, so Lean is happy with both, and the revision check does not fire
    because neither node's own revision moved. Only the re-check under the
    commit lock stops two 'helper' declarations reaching to_lean_header -- and
    dynamic_headers is append-only, so nothing could remove one afterwards.
    """
    graph = dag.ProofGraph()
    for nid in ("root", "a", "b"):
      graph.add_node(
          dag.ProofGraphNode(
              id=nid,
              informal_problem=models.InformalProblem(
                  statement=f"theorem {nid} : True"
              ),
              search_metadata={"phase": "formal"},
          )
      )
    graph.nodes["root"].and_nodes.append(
        dag.AndNode(id="d0", dependency_ids=["a", "b"], subgoal_signatures=_sigs(["a", "b"]))
    )
    mgr = self.make()
    mgr.initialize_graph(graph)

    # Both workers snapshot before either has committed.
    snap_a, rev_a, _ = mgr._prepare_submission("a", is_decomp=False)
    snap_b, rev_b, _ = mgr._prepare_submission("b", is_decomp=False)
    self.assertIsNone(
        sm._validate_definitions_block(snap_a, "a", "def helper : Nat := 1")
    )
    self.assertIsNone(
        sm._validate_definitions_block(snap_b, "b", "def helper : Nat := 2")
    )

    def _verified():
      return sm.VerificationOutcome(ok=True, clean_body="by trivial")

    committed_a, _ = mgr._commit_submission(
        "a", _verified(), "def helper : Nat := 1", False, rev_a
    )
    self.assertTrue(committed_a)
    committed_b, msg_b = mgr._commit_submission(
        "b", _verified(), "def helper : Nat := 2", False, rev_b
    )
    self.assertFalse(committed_b, "the second commit must be rejected")
    self.assertIn("helper", msg_b)
    self.assertEqual(
        self.load().environment.to_lean_header().count("def helper"),
        1,
        "a redeclared name in the header breaks every later compile",
    )

  def _submit(self, definitions, subgoals=""):
    """One scaffold, submitted with and without a subgoals block."""
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    path = self.write_scaffold(
        "root",
        models.SCAFFOLD_PROVE,
        definitions=definitions,
        subgoals=subgoals,
        proof="theorem root (a : Nat) : a + 0 = a :=\n  by simp",
    )
    with mock.patch.object(
        sm.lean_verifier_tool, "run_lean_verifier", _fake_verifier()
    ):
      return mgr.record_lean(path)

  def test_the_gate_runs_on_the_proof_reading(self):
    ok, msg = self._submit("open Finset")
    self.assertFalse(ok)
    self.assertIn("START_DEFINITIONS", msg)

  def test_the_gate_runs_on_the_decomposition_reading_too(self):
    """One file, two readings. A gate on only one of them would let the same
    block through by adding a subgoal."""
    ok, msg = self._submit(
        "open Finset", subgoals="theorem sub_one : True := sorry"
    )
    self.assertFalse(ok)
    self.assertIn("START_DEFINITIONS", msg)

  def test_an_opaque_constant_is_refused_on_the_proof_reading(self):
    """The fmf_v1_002 escape: 'def X := sorry' is refused, 'opaque X' was not."""
    ok, msg = self._submit("opaque HeuleGraph : SimpleGraph (Fin 510)")
    self.assertFalse(ok)
    self.assertIn("opaque constant", msg)

  def test_an_opaque_constant_is_refused_on_the_decomposition_reading(self):
    """The reading the run that died actually used."""
    ok, msg = self._submit(
        "opaque HeuleGraph : SimpleGraph (Fin 510)",
        subgoals="theorem sub_one : True := sorry",
    )
    self.assertFalse(ok)
    self.assertIn("opaque constant", msg)

  def test_the_opaque_message_names_all_three_ways_out(self):
    """A worker that only hears 'no' rewrites it again, which is how this got
    in: it reached for `opaque` *because* the sorry gate said no and stopped."""
    _, msg = self._submit("opaque HeuleGraph : SimpleGraph (Fin 510)")
    self.assertIn("def C : T := <the actual construction>", msg)
    self.assertIn("START_SUBGOALS", msg)
    self.assertIn("your decomposition", msg)
    self.assertNotIn("change how *other* proofs elaborate", msg)


class GlobalCheckTieringTest(ManagerTestCase):
  """The full recompile is O(whole graph); it must run only when it can help."""

  def test_empty_definitions_skips_the_global_check(self):
    self.assertEqual(sm._global_check_mode(""), "skip")
    self.assertEqual(sm._global_check_mode("   \n"), "skip")

  def test_plain_definitions_use_the_cheap_smoke_test(self):
    self.assertEqual(sm._global_check_mode("def s : Nat := 1"), "smoke")

  def test_elaboration_affecting_definitions_force_a_full_recompile(self):
    self.assertEqual(
        sm._global_check_mode("@[simp] theorem t (n : Nat) : n = n := rfl"),
        "full",
    )
    self.assertEqual(
        sm._global_check_mode("instance : Inhabited Nat := ⟨0⟩"), "full"
    )

  def test_smoke_test_assembly_is_bounded_by_header_not_node_count(self):
    graph = dag.ProofGraph()
    graph.add_node(
        dag.ProofGraphNode(
            id="root",
            informal_problem=models.InformalProblem(statement="theorem root : True"),
        )
    )
    graph.environment.imports.append("Mathlib")
    smoke = graph.assemble_header_smoke_test("root")
    self.assertIn("import Mathlib", smoke)
    self.assertIn("sorry", smoke)
    self.assertIn("root", smoke)

  def test_no_global_call_when_there_are_no_definitions(self):
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    self.write_scaffold("root", "prove", proof="theorem root (a : Nat) : a + 0 = a :=\n  by simp")
    calls = []

    def _spy(input_data):
      calls.append(input_data.code)
      return sm.lean_verifier_tool.LeanVerifierOutput(
          is_valid=True, used_axioms=["propext"]
      )

    with mock.patch.object(sm.lean_verifier_tool, "run_lean_verifier", _spy):
      ok, _ = mgr.record_lean(
          os.path.join(self.root, "nodes", "root", "proof_scaffold.lean")
      )
    self.assertTrue(ok)
    self.assertLen(calls, 1, "only the local compile should have run")


class InfraFailureTest(ManagerTestCase):
  """A verifier that never answered is not evidence about the mathematics."""

  def _infra_verifier(self):
    def _run(input_data):
      del input_data
      return sm.lean_verifier_tool.LeanVerifierOutput(
          is_valid=False,
          is_infra_failure=True,
          error_message="The read operation timed out",
      )

    return _run

  def test_timeout_does_not_consume_a_proof_attempt(self):
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    self.write_scaffold("root", "prove", proof="theorem root (a : Nat) : a + 0 = a :=\n  by simp")
    with mock.patch.object(
        sm.lean_verifier_tool, "run_lean_verifier", self._infra_verifier()
    ):
      ok, msg = mgr.record_lean(
          os.path.join(self.root, "nodes", "root", "proof_scaffold.lean")
      )
    self.assertFalse(ok)
    self.assertIn("INFRA_FAILURE", msg)
    self.assertEqual(
        self.load().nodes["root"].search_metadata.get("proof_attempts", 0),
        0,
        "an infra failure must not count against the node",
    )

  def test_timeout_is_not_reported_as_a_naming_collision(self):
    """The old message told agents to rename lemmas in response to a timeout."""
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    self.write_scaffold("root", "prove", proof="theorem root (a : Nat) : a + 0 = a :=\n  by simp")
    with mock.patch.object(
        sm.lean_verifier_tool, "run_lean_verifier", self._infra_verifier()
    ):
      _, msg = mgr.record_lean(
          os.path.join(self.root, "nodes", "root", "proof_scaffold.lean")
      )
    self.assertNotIn("rename your helper", msg.lower())
    self.assertNotIn("naming collision", msg.lower())
    self.assertIn("submit the exact same content again", msg)

  def test_a_real_rejection_still_counts(self):
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    self.write_scaffold("root", "prove", proof="theorem root (a : Nat) : a + 0 = a :=\n  by simp")
    with mock.patch.object(
        sm.lean_verifier_tool,
        "run_lean_verifier",
        _fake_verifier(is_valid=False, error="unknown identifier 'foo'"),
    ):
      ok, _ = mgr.record_lean(
          os.path.join(self.root, "nodes", "root", "proof_scaffold.lean")
      )
    self.assertFalse(ok)
    self.assertEqual(
        self.load().nodes["root"].search_metadata.get("proof_attempts", 0), 1
    )


class InfraRetryBudgetTest(ManagerTestCase):
  """'Resubmit the same content' is only sound while the outage is brief.

  Unbounded, it is a spin against a dead service, and an agent that exhausts
  its own turn budget mid-loop strands the claim permanently -- nothing expires
  it, so the node is lost for the rest of the run.
  """

  def _submit(self, mgr, verifier):
    self.put_in_phase("root", "formal")
    self.write_scaffold(
        "root", "prove", proof="theorem root (a : Nat) : a + 0 = a :=\n  by simp"
    )
    with mock.patch.object(
        sm.lean_verifier_tool, "run_lean_verifier", verifier
    ):
      return mgr.record_lean(
          os.path.join(self.root, "nodes", "root", "proof_scaffold.lean")
      )

  def _infra(self):
    def _run(input_data):
      del input_data
      return sm.lean_verifier_tool.LeanVerifierOutput(
          is_valid=False,
          is_infra_failure=True,
          error_message="The read operation timed out",
      )

    return _run

  def test_claim_is_released_once_the_budget_runs_out(self):
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    for _ in range(sm._MAX_INFRA_RETRIES - 1):
      ok, msg = self._submit(mgr, self._infra())
      self.assertFalse(ok)
      self.assertIn("INFRA_FAILURE:", msg)
      self.assertTrue(
          self.load().nodes["root"].search_metadata.get("processing"),
          "the claim is held while retrying is still advised",
      )

    ok, msg = self._submit(mgr, self._infra())
    self.assertTrue(ok, "exhaustion concludes the turn; it is not a rejection")
    self.assertIn("INFRA_FAILURE_EXHAUSTED", msg)
    self.assertFalse(
        self.load().nodes["root"].search_metadata.get("processing"),
        "the node must return to the queue rather than stay claimed",
    )

  def test_exhaustion_never_counts_against_the_node(self):
    self.seed()
    mgr = self.make()
    for _ in range(sm._MAX_INFRA_RETRIES):
      self._submit(mgr, self._infra())
    meta = self.load().nodes["root"].search_metadata
    self.assertEqual(meta.get("proof_attempts", 0), 0)
    self.assertFalse(meta.get("decomposition_failed", False))
    self.assertEqual(
        meta.get("phase"), "formal",
        "submitting Lean is being in the formal phase, even when the service died",
    )
    self.assertIn(
        "not been counted against you", meta.get("last_error", ""),
        "the note the next agent inherits must say the service died, not that"
        " the proof was wrong",
    )

  def test_exhaustion_message_stops_the_agent_resubmitting(self):
    self.seed()
    mgr = self.make()
    for _ in range(sm._MAX_INFRA_RETRIES - 1):
      self._submit(mgr, self._infra())
    _, msg = self._submit(mgr, self._infra())
    self.assertIn("Conclude your turn", msg)
    self.assertNotIn("submit the exact same content again", msg)

  def test_the_budget_is_advertised_and_counts_down(self):
    self.seed()
    mgr = self.make()
    _, first = self._submit(mgr, self._infra())
    _, second = self._submit(mgr, self._infra())
    self.assertIn(f"{sm._MAX_INFRA_RETRIES - 1} more time(s)", first)
    self.assertIn(f"{sm._MAX_INFRA_RETRIES - 2} more time(s)", second)

  def test_a_verdict_resets_the_streak(self):
    """Two isolated blips hours apart must not add up to an exhaustion."""
    self.seed()
    mgr = self.make()
    self._submit(mgr, self._infra())
    self._submit(
        mgr, _fake_verifier(is_valid=False, error="unknown identifier 'foo'")
    )
    self.assertEqual(
        self.load().nodes["root"].search_metadata.get("infra_failures", 0), 0
    )


class RecordLeanDerivationTest(ManagerTestCase):
  """is_decomp is read off the submission, not off which method was called."""

  def _submit(self, subgoals="", proof="theorem root (a : Nat) : a + 0 = a :=\n  by simp",
              **kwargs):
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    path = self.write_scaffold(
        "root", models.SCAFFOLD_PROVE, subgoals=subgoals, proof=proof
    )
    with mock.patch.object(
        sm.lean_verifier_tool, "run_lean_verifier", _fake_verifier()
    ):
      ok, msg = mgr.record_lean(path, **kwargs)
    return ok, msg, self.load().nodes["root"]

  def test_empty_subgoals_is_a_proof(self):
    ok, msg, node = self._submit()
    self.assertTrue(ok, msg)
    self.assertFalse(node.and_nodes, "a proof must not create an AND-node")
    self.assertIsNotNone(node.formal_theorem)

  def test_filled_subgoals_is_a_decomposition(self):
    ok, msg, node = self._submit(
        subgoals="theorem step_one (a : Nat) : 0 + a = a := sorry",
        proof="theorem root (a : Nat) : a + 0 = a :=\n  by simpa using step_one a",
    )
    self.assertTrue(ok, msg)
    self.assertTrue(node.and_nodes, "a decomposition must create an AND-node")

  def test_untouched_hint_block_is_still_a_proof(self):
    # The scaffold ships the subgoals block pre-filled with a hint that
    # mentions ':= sorry'. Reading emptiness as raw text would classify every
    # direct proof as a decomposition.
    self.seed()
    mgr = self.make()
    claim = self.claim_formal(mgr)
    with open(claim["scaffold_path"], "r", encoding="utf-8") as f:
      pristine = f.read()
    self.assertIn("[START_SUBGOALS]", pristine)
    body = pristine.split("[START_SUBGOALS]")[1].split("[END_SUBGOALS]")[0]
    self.assertIn("sorry", body, "precondition: the hint mentions sorry")

    # Rewrite only the proof block, leaving the hint exactly as shipped.
    text = pristine.split("-- [START_PROOF]")[0]
    text += ("-- [START_PROOF]\ntheorem root (a : Nat) : a + 0 = a :=\n"
             "  by simp\n-- [END_PROOF]")
    with open(claim["scaffold_path"], "w", encoding="utf-8") as f:
      f.write(text)

    with mock.patch.object(
        sm.lean_verifier_tool, "run_lean_verifier", _fake_verifier()
    ):
      ok, msg = mgr.record_lean(claim["scaffold_path"])
    self.assertTrue(ok, msg)
    self.assertFalse(self.load().nodes["root"].and_nodes)

  def test_the_pre_merge_action_names_are_gone(self):
    """They forwarded here, so an agent on the old protocol succeeded and the
    violation never surfaced. One shape means one name for it."""
    for dead in ("record_proof", "record_decomposition"):
      self.assertNotIn(dead, sm._ACTION_DISPATCH)
      self.assertFalse(hasattr(sm.ProofGraphStateManager, dead))

  def test_the_two_readings_are_counted_separately(self):
    """The counter is briefing material, not a scheduling input -- nothing
    sorts on it. Which of the two moved tells the next agent whether the
    previous turn tried to close this goal or to split it."""
    _, _, node = self._submit()
    self.assertEqual(node.search_metadata["proof_attempts"], 1)
    self.assertEqual(node.search_metadata.get("decomp_attempts", 0), 0)

  def test_a_decomposition_counts_on_the_other_side(self):
    _, _, node = self._submit(
        subgoals="theorem step_one (a : Nat) : 0 + a = a := sorry",
        proof="theorem root (a : Nat) : a + 0 = a :=\n  by simpa using step_one a",
    )
    self.assertEqual(node.search_metadata["decomp_attempts"], 1)
    self.assertEqual(node.search_metadata.get("proof_attempts", 0), 0)

  def test_a_rejection_is_counted_too(self):
    """Both are submissions that reached a verdict; only an infra failure,
    which reached none, is exempt."""
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    path = self.write_scaffold(
        "root",
        models.SCAFFOLD_PROVE,
        proof="theorem root (a : Nat) : a + 0 = a :=\n  by simp",
    )
    with mock.patch.object(
        sm.lean_verifier_tool,
        "run_lean_verifier",
        _fake_verifier(is_valid=False, error="unsolved goals"),
    ):
      mgr.record_lean(path)
    self.assertEqual(
        self.load().nodes["root"].search_metadata["proof_attempts"], 1
    )


_DEFECT = "the hypothesis h : 0 < n is missing, so n = 0 is a counterexample"


class CommitTest(ManagerTestCase):

  def test_stale_revision_is_refused(self):
    self.seed()
    mgr = self.make()
    outcome = sm.VerificationOutcome(ok=True, clean_body="by simp")
    ok, msg = mgr._commit_submission(
        "root", outcome, "", False, baseline_revision=-1
    )
    self.assertFalse(ok)
    self.assertIn("changed while your submission", msg)

  def test_successful_commit_bumps_revision_and_clears_flags(self):
    self.seed()
    mgr = self.make()
    self.put_in_phase("root", "formal")
    with mgr.lock_and_modify() as graph:
      graph.nodes["root"].search_metadata["decomposition_failed"] = True
    outcome = sm.VerificationOutcome(ok=True, clean_body="by simp")
    ok, _ = mgr._commit_submission("root", outcome, "", False, baseline_revision=0)
    self.assertTrue(ok)
    node = self.load().nodes["root"]
    self.assertEqual(node.search_metadata["revision"], 1)
    self.assertFalse(node.search_metadata["decomposition_failed"])
    self.assertFalse(node.search_metadata["processing"])

  def test_headers_are_append_only(self):
    """Two nodes, because one node commits once: a proved node has left the
    machine and there is no phase left for it to submit from."""
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      graph.add_node(
          dag.ProofGraphNode(
              id="kid",
              informal_problem=models.InformalProblem(statement="theorem kid : True"),
              search_metadata={"phase": "formal"},
          )
      )
      graph.nodes["root"].and_nodes.append(
          dag.AndNode(id="d0", dependency_ids=["kid"], subgoal_signatures=_sigs(["kid"]))
      )
    self.put_in_phase("root", "formal")
    outcome = sm.VerificationOutcome(ok=True, clean_body="by simp")
    mgr._commit_submission(
        "root", outcome, "def marker : Nat := 1", False, baseline_revision=0
    )
    mgr._commit_submission(
        "kid", outcome, "def marker_two : Nat := 2", False, baseline_revision=0
    )
    headers = self.load().environment.dynamic_headers
    self.assertEqual(
        [h.content for h in headers],
        ["def marker : Nat := 1", "def marker_two : Nat := 2"],
    )


class RetractedHeaderCommitTest(ManagerTestCase):
  """The retirement race `actions._retire_header_blocks` cannot see.

  Verification runs unlocked. A worker compiles against a header, a reviewer
  retires one of its blocks -- correctly, because nothing in the graph referred
  to it yet -- and the worker then commits a proof citing a name that is no
  longer in any header. The node's own revision never moved, so the optimistic
  concurrency check does not fire, and a submission with no definitions block
  skips the regression compile entirely. Without this gate the proof is
  recorded as verified and the tree stops rebuilding.
  """

  def retire(self, content, node_id="other"):
    with self.make().lock_and_modify() as graph:
      graph.environment.dynamic_headers.append(
          models.SharedHeaderBlock(
              content=content, node_id=node_id, retired=True
          )
      )

  def commit(self, body, definitions=""):
    self.put_in_phase("root", "formal")
    return self.make()._commit_submission(  # pylint: disable=protected-access
        "root",
        sm.VerificationOutcome(ok=True, clean_body=body),
        definitions,
        False,
        baseline_revision=0,
    )

  def test_a_proof_citing_a_retracted_definition_is_refused(self):
    self.seed()
    self.retire("def orphan_helper : Nat := 1")
    ok, msg = self.commit("by simp [orphan_helper]")
    self.assertFalse(ok)
    self.assertIn("retracted", msg)
    self.assertIn("orphan_helper", msg)

  def test_the_refusal_does_not_blame_the_agent(self):
    self.seed()
    self.retire("def orphan_helper : Nat := 1")
    _, msg = self.commit("by simp [orphan_helper]")
    self.assertIn("nothing is wrong with your mathematics", msg)
    self.assertIn("ACTION REQUIRED", msg)

  def test_a_proof_that_does_not_cite_it_commits(self):
    self.seed()
    self.retire("def orphan_helper : Nat := 1")
    ok, _ = self.commit("by simp")
    self.assertTrue(ok)

  def test_a_name_a_live_block_took_back_is_not_refused(self):
    """Retirement releases the name, so a re-planned branch may legally hold
    it. Then it is in the header and there is nothing to refuse."""
    self.seed()
    self.retire("def orphan_helper : Nat := 1")
    with self.make().lock_and_modify() as graph:
      graph.environment.dynamic_headers.append(
          models.SharedHeaderBlock(
              content="def orphan_helper : Nat := 2", node_id="replanned"
          )
      )
    ok, _ = self.commit("by simp [orphan_helper]")
    self.assertTrue(ok)

  def test_a_substring_of_a_retracted_name_is_not_a_citation(self):
    self.seed()
    self.retire("def helper : Nat := 1")
    ok, _ = self.commit("by simp [helper_two]")
    self.assertTrue(ok)

  def test_a_subgoal_statement_counts_as_a_citation(self):
    self.seed()
    self.retire("def orphan_helper : Nat := 1")
    self.put_in_phase("root", "formal")
    outcome = sm.VerificationOutcome(
        ok=True,
        clean_body="by simp",
        actual_subgoals=[
            models.FormalSignature(
                name="kid", statement="lemma kid : orphan_helper = 1"
            )
        ],
    )
    ok, msg = self.make()._commit_submission(  # pylint: disable=protected-access
        "root", outcome, "", True, baseline_revision=0
    )
    self.assertFalse(ok)
    self.assertIn("orphan_helper", msg)


class AtomicityTest(ManagerTestCase):

  def test_state_file_survives_a_serialization_failure(self):
    self.seed()
    mgr = self.make()
    with self.assertRaises(RuntimeError):
      with mgr.lock_and_modify() as graph:
        graph.add_node(
            dag.ProofGraphNode(
                id="x", informal_problem=models.InformalProblem(statement="True")
            )
        )
        raise RuntimeError("crash mid-transaction")
    self.assertIn("root", self.load().nodes)
    self.assertNotIn("x", self.load().nodes)

  def test_no_temp_files_are_left_behind(self):
    self.seed()
    with self.make().lock_and_modify() as graph:
      graph.graph_metadata["touched"] = True
    leftovers = [f for f in os.listdir(self.root) if f.startswith(".proof_graph_")]
    self.assertEmpty(leftovers)

  def test_concurrent_threads_do_not_lose_updates(self):
    self.seed()

    def bump():
      mgr = self.make()
      for _ in range(20):
        with mgr.lock_and_modify() as graph:
          graph.graph_metadata["n"] = graph.graph_metadata.get("n", 0) + 1

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    self.assertEqual(self.load().graph_metadata["n"], 80)


class PruneOnCommitTest(ManagerTestCase):
  """Every write prunes. Nothing has to remember to ask for it."""

  def _seed_split(self):
    """root with one live plan naming `kid`."""
    graph = dag.ProofGraph()
    graph.add_node(
        dag.ProofGraphNode(
            id="root",
            informal_problem=models.InformalProblem(statement="theorem root"),
        )
    )
    graph.add_node(
        dag.ProofGraphNode(
            id="kid", informal_problem=models.InformalProblem(statement="k")
        )
    )
    graph.nodes["root"].and_nodes.append(
        dag.AndNode(
            id="d0", dependency_ids=["kid"], subgoal_signatures=_sigs(["kid"])
        )
    )
    graph.nodes["kid"].reverse_dependency_ids.append("root")
    self.make().initialize_graph(graph)
    return self.load()

  def test_a_node_nothing_reaches_never_reaches_disk(self):
    self.seed()
    with self.make().lock_and_modify() as graph:
      graph.add_node(
          dag.ProofGraphNode(
              id="ghost", informal_problem=models.InformalProblem(statement="g")
          )
      )
    self.assertNotIn("ghost", self.load().nodes)

  def test_retiring_a_plan_takes_the_children_it_stranded(self):
    self._seed_split()
    with self.make().lock_and_modify() as graph:
      graph.nodes["root"].and_nodes[0].failed = True
    self.assertNotIn("kid", self.load().nodes)

  def test_the_name_is_free_again_afterwards(self):
    """The point of pruning. A ghost holding a name caused the fmf_v1_004
    rename spiral; after the prune `name in nodes` means what it says."""
    self._seed_split()
    with self.make().lock_and_modify() as graph:
      graph.nodes["root"].and_nodes[0].failed = True
    with self.make().lock_and_modify() as graph:
      self.assertNotIn("kid", graph.nodes)
      graph.add_node(
          dag.ProofGraphNode(
              id="kid", informal_problem=models.InformalProblem(statement="k2")
          )
      )
      graph.nodes["root"].and_nodes.append(
          dag.AndNode(
              id="d1", dependency_ids=["kid"], subgoal_signatures=_sigs(["kid"])
          )
      )
      graph.nodes["kid"].reverse_dependency_ids.append("root")
    self.assertIn("kid", self.load().nodes)

  def test_a_stranded_child_that_was_worked_on_stays(self):
    """The interlock. Any search_metadata at all -- including the `processing`
    a live claim writes -- keeps the node, so nothing is deleted from under a
    worker or away from a directory on disk."""
    self._seed_split()
    with self.make().lock_and_modify() as graph:
      graph.nodes["kid"].search_metadata["processing"] = True
      graph.nodes["root"].and_nodes[0].failed = True
    self.assertIn("kid", self.load().nodes)

  def test_a_live_child_is_never_touched(self):
    self._seed_split()
    with self.make().lock_and_modify() as graph:
      del graph  # A commit that changes nothing still prunes.
    self.assertIn("kid", self.load().nodes)

  def test_the_prune_runs_on_actions_that_were_given_no_root_id(self):
    """`--root_id` is passed to `init_graph` and to nothing else, so the prune
    has to resolve the root itself or it would never fire on a real submit."""
    self._seed_split()
    mgr = sm.ProofGraphStateManager(filepath=self.filepath, root_dir=self.root)
    self.assertIsNone(mgr.root_id)
    with mgr.lock_and_modify() as graph:
      graph.nodes["root"].and_nodes[0].failed = True
    self.assertNotIn("kid", self.load().nodes)

  def test_the_root_survives_an_empty_graph(self):
    self.seed()
    with self.make().lock_and_modify() as graph:
      graph.nodes["root"].search_metadata.clear()
    self.assertIn("root", self.load().nodes)


class InitGraphRootTest(ManagerTestCase):
  """init_graph must root the graph at `theorem <root_id>` and nothing else.

  Lean is stubbed out: what is under test is which declaration gets picked.
  """

  def init(self, root_id, body):
    with mock.patch.object(
        sm.graph_init_guard, "check_root_compiles", return_value=""
    ):
      return sm.run_state_manager(
          sm.StateManagerInput(
              action="init_graph",
              filepath=self.filepath,
              root_dir=self.root,
              root_id=root_id,
              statement="import Mathlib\n\n" + body,
          )
      )

  def root_statement(self, root_id):
    return self.load().nodes[root_id].informal_problem.statement

  def test_a_longer_name_declared_first_is_not_the_root(self):
    """`\\b` used to stop inside each of these, rooting the graph at it."""
    for decoy in ("foo'", "foo!", "foo?", "foo₁", "fooₐ", "fooα", "foo℘",
                  "foo᾽", "foo.bar", "foo_bar", "foo2"):
      with self.subTest(decoy=decoy):
        if os.path.exists(self.filepath):
          os.remove(self.filepath)
        out = self.init(
            "foo",
            f"theorem {decoy} : 1 = 2 := sorry\n\n"
            "theorem foo : 1 = 1 := by\n  sorry\n",
        )
        self.assertTrue(out.success, out.error_message)
        self.assertEqual(self.root_statement("foo"), "theorem foo : 1 = 1")
        self.assertIn(decoy, self.load().environment.shared_definitions)

  def test_only_a_longer_name_means_not_found(self):
    out = self.init("foo", "theorem foo' : 1 = 2 := sorry\n")
    self.assertFalse(out.success)
    self.assertIn("was not found", out.error_message)
    self.assertFalse(os.path.exists(self.filepath))

  def test_universe_parameters_follow_the_name(self):
    out = self.init(
        "foo", "theorem foo.{u} (α : Sort u) (a : α) : a = a := by\n  sorry\n"
    )
    self.assertTrue(out.success, out.error_message)
    self.assertEqual(
        self.root_statement("foo"),
        "theorem foo.{u} (α : Sort u) (a : α) : a = a",
    )

  def test_a_commented_out_declaration_is_not_the_root(self):
    out = self.init(
        "foo",
        "/- old version\ntheorem foo : 1 = 2 := sorry\n-/\n\n"
        "/-- Docstring.\ntheorem foo : 2 = 3 -/\n"
        "theorem foo : 1 = 1 := by\n  sorry\n",
    )
    self.assertTrue(out.success, out.error_message)
    self.assertEqual(self.root_statement("foo"), "theorem foo : 1 = 1")

  def test_a_name_declared_twice_is_refused(self):
    """Two namespaces may each declare `foo`; neither is the obvious root."""
    out = self.init(
        "foo",
        "namespace A\ntheorem foo : 1 = 2 := sorry\nend A\n\n"
        "theorem foo : 1 = 1 := by\n  sorry\n",
    )
    self.assertFalse(out.success)
    self.assertIn("declared 2 times", out.error_message)
    self.assertFalse(os.path.exists(self.filepath))

  def test_a_root_name_with_a_prime_is_matched_whole(self):
    out = self.init(
        "foo'", "theorem foo'' : 1 = 2 := sorry\n\n"
        "theorem foo' : 1 = 1 := by\n  sorry\n",
    )
    self.assertTrue(out.success, out.error_message)
    self.assertEqual(self.root_statement("foo'"), "theorem foo' : 1 = 1")


class BriefingTest(ManagerTestCase):
  """context.md must state the constraints and only cite files that exist."""

  def _render(self, phase="formal", allowed=None, graph=None):
    graph = self.seed() if graph is None else graph
    if allowed is not None:
      graph.allowed_axioms = allowed
    return sm.workspace_manager.WorkspaceManager(self.root).render_context_md(
        "root", graph, pipeline.get_phase(phase)
    )

  def test_signature_immutability_is_stated_up_front(self):
    self.assertIn("signature is immutable", self._render())

  def test_a_phase_that_writes_no_lean_is_not_given_the_signature_rule(self):
    """The rule is about the scaffold, and a planner has not got one."""
    self.assertNotIn("signature is immutable", self._render("informal"))

  def test_axiom_policy_is_stated_and_derived_from_the_graph(self):
    """The briefing must track the enforced policy, not a hardcoded copy."""
    text = self._render(allowed=["propext", "Lean.ofReduceBool"])
    self.assertIn("`Lean.ofReduceBool`", text)
    self.assertIn("native_decide", text)

  def test_default_policy_is_rendered_when_unset(self):
    text = self._render()
    for axiom in dag.DEFAULT_ALLOWED_AXIOMS:
      self.assertIn(f"`{axiom}`", text)

  def test_the_plan_is_cited_rather_than_copied(self):
    """The graph does not hold the informal proof, so it cannot render one.

    Pasting a snapshot of a file the agents own means shipping a stale copy the
    moment either of them appends to it.
    """
    text = self._render()
    self.assertIn("nodes/root/scratch/blueprint.md", text)

  def test_the_planner_is_told_to_write_the_plan_the_others_read(self):
    self.assertIn("Write your plan to", self._render("informal"))

  def test_the_root_planner_is_warned_that_failed_ends_the_run(self):
    text = self._render("informal")
    self.assertIn("`--action=failed` here ends the entire run", text)
    self.assertIn("ill-posed or defective", text)

  def test_the_root_warning_names_the_excuses_that_are_not_reasons(self):
    """The 002 planner used every one of these to justify failing the root."""
    text = self._render("informal")
    for excuse in ("famous, hard", "no published proof", "solver, or tactic"):
      self.assertIn(excuse, text)

  def test_a_non_root_planner_is_not_warned(self):
    """`failed` on a child is cheap and correct -- the parent backtracks."""
    graph = self.seed()
    graph.add_node(
        dag.ProofGraphNode(
            id="kid",
            informal_problem=models.InformalProblem(statement="theorem kid : True"),
        )
    )
    # Attached, or `initialize_graph` prunes it and the render is "" -- which
    # passes the assertion below for the wrong reason.
    graph.nodes["root"].and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["kid"],
            subgoal_signatures=[
                models.FormalSignature(name="kid", statement="True")
            ],
        )
    )
    graph.nodes["kid"].reverse_dependency_ids.append("root")
    text = sm.workspace_manager.WorkspaceManager(self.root).render_context_md(
        "kid", graph, pipeline.get_phase("informal"), root_id="root"
    )
    self.assertIn("## 2. Active Target Goal", text)
    self.assertNotIn("ends the entire run", text)

  def test_phases_that_cannot_run_failed_are_not_warned(self):
    """The block is gated on the action, not the phase name."""
    for phase in ("formal", "review", "disprove"):
      with self.subTest(phase=phase):
        text = self._render(phase)
        self.assertIn("## 2. Active Target Goal", text)
        self.assertNotIn("ends the entire run", text)

  def test_backtracking_lists_the_full_declared_subgoal_set(self):
    """Padding a failed set with an unused lemma must not look like a new one."""
    graph = self.seed()
    self._write_failure_report("dead_lemma")
    graph.nodes["root"].and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["dead_lemma"],
            subgoal_signatures=[
                models.FormalSignature(name="dead_lemma", statement="0 = 0"),
                models.FormalSignature(name="padding_lemma", statement="1 = 1"),
            ],
            failed=True,
            error_message="Child subgoal 'dead_lemma' is an unprovable dead end.",
        )
    )
    text = self._render(graph=graph)
    self.assertIn("`dead_lemma`", text)
    self.assertIn("`padding_lemma`", text)
    self.assertEqual(
        text.count("Prior Decomposition History"),
        1,
        "the two backtracking sections were merged into one",
    )

  def test_no_history_section_when_nothing_has_been_tried(self):
    self.assertNotIn("Prior Decomposition History", self._render())

  def _write_failure_report(self, node_id, text="the kernel said no"):
    manager = sm.workspace_manager.WorkspaceManager(self.root)
    path = manager.get_failure_report_path(node_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      f.write(text)
    return path

  def _render_with_dead_child(self, error_message=""):
    graph = self.seed()
    graph.add_node(
        dag.ProofGraphNode(
            id="kid",
            informal_problem=models.InformalProblem(statement="kid holds"),
            reverse_dependency_ids=["root"],
        )
    )
    graph.nodes["kid"].search_metadata["decomposition_failed"] = True
    graph.nodes["root"].and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["kid"], subgoal_signatures=_sigs(["kid"]),
            failed=True,
            error_message=error_message,
        )
    )
    return self._render(graph=graph)

  def test_a_dead_child_with_a_report_is_pointed_at_it(self):
    self._write_failure_report("kid")
    text = self._render_with_dead_child()
    self.assertIn("nodes/kid/failure_report.md", text)
    self.assertIn("`kid` is a dead end", text)

  def test_a_child_with_no_report_is_not_sent_to_a_file_nobody_wrote(self):
    """A node given up on rather than rejected has no compiler output.

    Naming a file that was never written sends a weak model looking for it with
    `find`, which its own directive forbids -- so name what it did leave.
    """
    text = self._render_with_dead_child()
    self.assertNotIn("nodes/kid/failure_report.md", text)
    self.assertIn("nodes/kid/scratch/blueprint.md", text)
    self.assertIn("nodes/kid/findings.md", text)

  def test_a_child_that_was_never_created_is_not_called_a_dead_end(self):
    """The reviewer can retire a plan before its subgoals are dispatched, and
    then there is no node, no directory and no history -- but the test for
    "dead" was `not in graph.nodes or FAILED`, so the briefing branded it and
    cited files that were never written. In fmf_v1_009/rollout_2's final root
    briefing 5 of 11 "dead end" lines were this case and 8 of the cited paths
    did not exist, two of them naming trivially provable moment computations.
    The agent is forbidden `find` and `grep`, so it cannot check.
    """
    graph = self.seed()
    graph.nodes["root"].and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["never_born"],
            subgoal_signatures=_sigs(["never_born"]),
            failed=True,
            error_message="RETIRED BY REVIEW: the split leaks the hard case",
        )
    )
    text = self._render(graph=graph)
    self.assertIn("`never_born` was never attempted", text)
    self.assertNotIn("`never_born` is a dead end", text)
    self.assertNotIn("nodes/never_born/", text)

  def test_a_pruned_child_is_not_reported_as_evidence(self):
    """The consequence that costs: a planner that reads "dead end" will not
    re-propose it, and one of these was a moment computation it could close."""
    graph = self.seed()
    graph.nodes["root"].and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["never_born"],
            subgoal_signatures=_sigs(["never_born"]),
            failed=True,
        )
    )
    self.assertIn(
        "proposing it again is not a repeat", self._render(graph=graph)
    )

  def test_why_the_branch_died_is_carried_on_the_attempt_line(self):
    text = self._render_with_dead_child(
        error_message="RETIRED BY REVIEW: the split leaks the hard case"
    )
    self.assertIn("Blocker: RETIRED BY REVIEW: the split leaks", text)

  def test_a_surviving_sibling_is_offered_for_reuse(self):
    """Re-proving a lemma that is already proved is the expensive mistake."""
    graph = self.seed()
    for kid in ("dead_kid", "live_kid"):
      graph.add_node(
          dag.ProofGraphNode(
              id=kid,
              informal_problem=models.InformalProblem(statement=f"{kid} holds"),
              reverse_dependency_ids=["root"],
          )
      )
    graph.nodes["dead_kid"].search_metadata["decomposition_failed"] = True
    graph.nodes["live_kid"].formal_theorem = models.FormalTheorem(
        signature=models.FormalSignature(name="live_kid", statement="True"),
        proof=models.FormalProof(tactics="trivial"),
    )
    graph.nodes["root"].and_nodes.append(
        dag.AndNode(
            id="d0", dependency_ids=["dead_kid", "live_kid"], subgoal_signatures=_sigs(["dead_kid", "live_kid"]), failed=True
        )
    )
    text = self._render(graph=graph)
    self.assertIn("Still proved and reusable", text)
    self.assertIn("`live_kid`", text)

  def test_the_reason_a_node_came_back_is_not_called_a_rejection(self):
    """It may be a reviewer's verdict, and the agent has not attempted anything.

    Telling a fresh agent it failed at something it never tried is how a
    briefing gets ignored.
    """
    graph = self.seed()
    graph.nodes["root"].search_metadata["last_error"] = "the plan skips a case"
    text = self._render(graph=graph)
    self.assertIn("the plan skips a case", text)
    self.assertIn("Why this node is back on the queue", text)

  def test_a_failure_report_is_only_cited_once_it_exists(self):
    self.assertNotIn("**Failure Report**", self._render())
    self._write_failure_report("root")
    self.assertIn("nodes/root/failure_report.md", self._render())

  def test_the_submit_block_lists_exactly_this_phases_actions(self):
    text = self._render()
    self.assertIn("--action=submit", text)
    self.assertIn("--action=replan", text)
    self.assertNotIn("--action=approved", text)

  def test_the_submit_paths_are_absolute(self):
    """A relative path resolves against whatever directory the agent is in."""
    scaffold = os.path.join(self.root, "nodes", "root", "proof_scaffold.lean")
    self.assertIn(f"--code_file={scaffold}", self._render())


class FailureReportTest(ManagerTestCase):
  """The file context.md points agents at must actually be written."""

  def test_rejections_accumulate_in_the_report(self):
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      mgr._finalize_failure(graph.nodes["root"], "first failure", is_decomp=False)
    with mgr.lock_and_modify() as graph:
      mgr._finalize_failure(graph.nodes["root"], "second failure", is_decomp=False)

    path = sm.workspace_manager.WorkspaceManager(
        self.root
    ).get_failure_report_path("root")
    self.assertTrue(os.path.exists(path))
    with open(path, encoding="utf-8") as f:
      report = f.read()
    self.assertIn("first failure", report)
    self.assertIn("second failure", report)
    self.assertLess(
        report.index("first failure"),
        report.index("second failure"),
        "newest last",
    )

  def test_report_distinguishes_proof_from_decomposition(self):
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      mgr._finalize_failure(graph.nodes["root"], "boom", is_decomp=True)
    path = sm.workspace_manager.WorkspaceManager(
        self.root
    ).get_failure_report_path("root")
    with open(path, encoding="utf-8") as f:
      self.assertIn("Decomposition attempt", f.read())


class ReferenceMaterialTest(ManagerTestCase):
  """Material shipped with the problem has to be named to be found.

  Agents are forbidden `find`, `ls` and `grep`, so a folder no briefing
  mentions does not exist as far as they are concerned. It is also unverified
  human prose, and a planner that cites it as established has skipped the work
  the blueprint's evidence and falsification sections exist to force.
  """

  def _write_reference(self, *names):
    path = os.path.join(
        self.root, sm.workspace_manager.WorkspaceManager.REFERENCE_DIRNAME
    )
    os.makedirs(path, exist_ok=True)
    for name in names:
      with open(os.path.join(path, name), "w", encoding="utf-8") as f:
        f.write("an informal argument someone wrote down")

  def _render(self, phase="informal"):
    return sm.workspace_manager.WorkspaceManager(self.root).render_context_md(
        "root", self.seed(), pipeline.get_phase(phase), root_id="root"
    )

  def test_the_files_are_named_one_per_line(self):
    self._write_reference("informal_proof.md", "survey.pdf")
    text = self._render()
    self.assertIn("`reference/informal_proof.md`", text)
    self.assertIn("`reference/survey.pdf`", text)

  def test_the_listing_is_sorted(self):
    """Two briefings for the same node must not differ by directory order, or
    every re-provision archives a copy that says the same thing."""
    self._write_reference("zeta.md", "alpha.md")
    text = self._render()
    self.assertLess(text.index("alpha.md"), text.index("zeta.md"))

  def test_the_material_is_marked_unverified(self):
    """The risk is not discovery, it is a planner treating a human write-up as
    a checked result and skipping its own falsification work."""
    self._write_reference("informal_proof.md")
    self.assertIn("unverified", self._render())

  def test_contents_are_never_pasted_into_the_briefing(self):
    """Pointers, not contents -- the rule the rest of section 7 follows. A
    large reference must cost one line, not the whole page."""
    self._write_reference("informal_proof.md")
    self.assertNotIn("an informal argument someone wrote down", self._render())

  def test_a_run_with_no_reference_folder_is_unchanged(self):
    text = self._render()
    self.assertNotIn("reference/", text)
    self.assertIn("## 7. Established Findings", text)

  def test_an_empty_reference_folder_is_not_advertised(self):
    """An empty folder is not material; naming it sends agents to nothing."""
    os.makedirs(
        os.path.join(
            self.root, sm.workspace_manager.WorkspaceManager.REFERENCE_DIRNAME
        ),
        exist_ok=True,
    )
    self.assertNotIn("reference/", self._render())

  def test_subdirectories_and_dotfiles_are_not_listed(self):
    """`os.listdir` returns both, and neither is a document to read."""
    self._write_reference(".DS_Store")
    os.makedirs(
        os.path.join(
            self.root,
            sm.workspace_manager.WorkspaceManager.REFERENCE_DIRNAME,
            "images",
        ),
        exist_ok=True,
    )
    manager = sm.workspace_manager.WorkspaceManager(self.root)
    self.assertEmpty(manager.list_reference_files())

  def test_every_worked_phase_that_has_section_7_names_the_material(self):
    """The planner is the point, but a prover reaching for the same document
    should not have to be told it exists by a peer's findings file."""
    self._write_reference("informal_proof.md")
    for phase in ("informal", "formal", "disprove"):
      with self.subTest(phase=phase):
        self.assertIn(7, pipeline.PIPELINE[phase].sections)
        self.assertIn("`reference/informal_proof.md`", self._render(phase))

  def test_material_added_mid_run_reaches_the_next_briefing(self):
    """Briefings regenerate on every claim, so nothing needs restarting."""
    self.assertNotIn("reference/", self._render())
    self._write_reference("late_arrival.md")
    self.assertIn("`reference/late_arrival.md`", self._render())

  def test_the_layout_documents_the_folder_when_it_has_content(self):
    """An undocumented directory at the root reads as debris an agent may
    clean up, or a gap it invents its own conventions to fill."""
    manager = sm.workspace_manager.WorkspaceManager(self.root)
    self.assertNotIn("Supplied Reference Material", manager.describe_layout())
    self._write_reference("informal_proof.md")
    layout = manager.describe_layout()
    self.assertIn("Supplied Reference Material", layout)
    # Bare names here, not `reference/<name>`: the absolute directory is the
    # line above them. Section 7 of the briefing is the other way round, since
    # it has no such line and every path there is relative to the root.
    self.assertIn(f"{manager.REFERENCE_DIRNAME}/`", layout)
    self.assertIn("`informal_proof.md`", layout)


class BriefingRotationTest(ManagerTestCase):
  """Re-provisioning archives the old briefing only when it actually changed."""

  def _provision(self, phase="informal"):
    mgr = self.make()
    graph = mgr.read_graph()
    return sm.workspace_manager.WorkspaceManager(self.root).provision_node_workspace(
        "root", graph, pipeline.get_phase(phase), root_id="root"
    )

  def _archived(self):
    node_dir = os.path.join(self.root, "nodes", "root")
    if not os.path.isdir(node_dir):
      return []
    return [n for n in os.listdir(node_dir) if n.startswith("context_old_")]

  def test_an_unchanged_briefing_is_not_archived(self):
    """In fmf_v1_006 a node parked in `review` left 110 of these in an hour."""
    self.seed()
    for _ in range(5):
      self._provision()
    self.assertEmpty(self._archived())

  def test_the_briefing_is_still_written(self):
    self.seed()
    self._provision()
    written = self._provision()
    with open(written["context_path"], encoding="utf-8") as f:
      self.assertEqual(f.read(), written["context_content"])

  def test_a_changed_briefing_is_archived_with_its_old_contents(self):
    self.seed()
    first = self._provision("informal")
    self._provision("formal")
    archived = self._archived()
    self.assertLen(archived, 1)
    with open(
        os.path.join(self.root, "nodes", "root", archived[0]), encoding="utf-8"
    ) as f:
      self.assertEqual(f.read(), first["context_content"])

  def test_turn_artifacts_rotate_even_though_the_briefing_does_not(self):
    """Their new content is the agent's, so the tool cannot compare against it."""
    self.seed()
    self._provision()
    node_dir = os.path.join(self.root, "nodes", "root")
    artifact = pipeline.get_phase("informal").turn_artifacts[0]
    path = os.path.join(node_dir, artifact)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      f.write("an agent wrote this")
    self._provision()
    base, ext = os.path.splitext(os.path.basename(artifact))
    self.assertTrue(
        any(
            n.startswith(f"{base}_old_") and n.endswith(ext)
            for n in os.listdir(os.path.dirname(path))
        ),
        "the agent's own file must still be preserved on re-provision",
    )


class SectionSelectionTest(ManagerTestCase):
  """Which sections a briefing carries is the phase's call, not the renderer's."""

  #: The phases an agent is actually dispatched against.
  WORKED = ("informal", "formal", "disprove", "review")

  def _render(self, phase, graph=None):
    graph = graph if graph is not None else self.seed()
    return sm.workspace_manager.WorkspaceManager(self.root).render_context_md(
        "root", graph, pipeline.get_phase(phase), root_id="root"
    )

  def _headings(self, phase):
    """The numbered section headings the renderer actually emitted."""
    return re.findall(r"^## (\d+)\. (.+)$", self._render(phase), re.M)

  def test_no_phase_renders_a_section_it_did_not_declare(self):
    for name in self.WORKED:
      declared = set(pipeline.PIPELINE[name].sections)
      rendered = {int(n) for n, _ in self._headings(name)}
      self.assertContainsSubset(rendered, declared, name)

  def test_headings_are_titled_from_the_table(self):
    """A section is named in one place, so a prompt can cite it by name."""
    for name in self.WORKED:
      for number, title in self._headings(name):
        self.assertEqual(title.strip(), pipeline.SECTIONS[int(number)], name)

  def test_the_numbering_does_not_restart_per_phase(self):
    """A reviewer's briefing jumps 3 -> 8; renumbering would break a prompt
    that says "section 9"."""
    numbers = [int(n) for n, _ in self._headings("review")]
    self.assertEqual(numbers, sorted(numbers))
    self.assertNotIn(1, numbers)
    self.assertIn(9, numbers)

  def test_a_planner_is_not_shown_the_scaffold_section(self):
    # 4 is the scaffold and the peer set, which a planner has no use for: it
    # writes no Lean.
    text = self._render("informal")
    self.assertNotIn("## 4.", text)
    self.assertIn("## 3.", text)

  def test_a_planner_is_shown_whether_the_node_was_ever_attempted(self):
    """Section 5 is the compiler's own record. `informal` and `review` are the
    two roles that decide whether a node is worth continuing, and across both
    fmf_v1_009 rollouts all 18 nodes ever attempted directly in Lean ended
    proved while all 29 declared dead had never been attempted once -- a
    judgment made without the one fact that predicted its outcome."""
    self.assertIn("## 5.", self._render("informal"))
    self.assertIn("## 5.", self._render(pipeline.REVIEW_PHASE))

  def test_the_prover_sees_diagnostics_and_peers(self):
    text = self._render("formal")
    self.assertIn("## 4.", text)
    self.assertIn("## 5.", text)

  def test_a_roleless_phase_gets_no_briefing_at_all(self):
    """`blocked` dispatches nobody, so there is nothing to brief."""
    text = self._render("blocked")
    self.assertNotIn("##", text)
    self.assertNotIn("How To Submit", text)
    self.assertIn("blocked", text)

  def test_a_scaffold_name_is_not_a_phase_name(self):
    """`prove` and `disprove` name scaffolds; only one is also a phase.

    They used to share a parameter with the phase name, which let a caller ask
    for a briefing by scaffold and get a coherent-looking page for the wrong
    phase. `get_phase` resolves an unknown name to the entry phase.
    """
    self.assertEqual(self._render("prove"), self._render("informal"))

  def test_the_refutation_framing_replaces_the_proving_one(self):
    text = self._render("disprove")
    self.assertIn("refuting this target, not proving it", text)
    self.assertNotIn("Prove directly, or decompose", text)

  def test_every_action_the_phase_allows_is_spelled_out(self):
    for name in self.WORKED:
      text = self._render(name)
      for action in pipeline.allowed_actions(pipeline.PIPELINE[name]):
        self.assertIn(f"--action={action} --node_id=root", text, name)

  def test_a_submit_command_survives_a_quote_in_the_node_id(self):
    """`foo'bar` unquoted merged `--node_id` and `--code_file` into one word."""
    node_id = "foo'bar"
    graph = dag.ProofGraph()
    graph.add_node(
        dag.ProofGraphNode(
            id=node_id,
            informal_problem=models.InformalProblem(
                statement=f"theorem {node_id} : True"
            ),
        )
    )
    text = sm.workspace_manager.WorkspaceManager(self.root).render_context_md(
        node_id, graph, pipeline.get_phase("formal")
    )
    commands = [
        line for line in text.splitlines()
        if line.startswith("timeout 900s state_manager_bin ")
    ]
    self.assertNotEmpty(commands)
    for command in commands:
      args = shlex.split(command)
      self.assertIn(f"--node_id={node_id}", args, command)
      code_files = [a for a in args if a.startswith("--code_file=")]
      for code_file in code_files:
        self.assertIn(os.path.join("nodes", node_id), code_file)

  def test_predicate_names_never_reach_a_briefing(self):
    """Scheduler vocabulary an agent cannot cause must not be advertised."""
    for name in self.WORKED:
      text = self._render(name)
      for predicate in ("children_done", "children_failed", "subtree_stalled"):
        self.assertNotIn(predicate, text, name)

  def test_the_log_pointer_is_only_offered_once_a_log_exists(self):
    self.assertNotIn("Last rejection", self._render("formal"))
    self.seed()
    sm.workspace_manager.WorkspaceManager(self.root).write_attempt_log(
        "root", "unknown identifier 'foo'", "rejection"
    )
    self.assertIn("Last rejection", self._render("formal"))


class SubtreeLedgerTest(ManagerTestCase):
  """Section 9: the live decomposition, as statements plus an index."""

  def _seed_tree(self):
    graph = self.seed()
    for child in ("kid_a", "kid_b"):
      graph.add_node(
          dag.ProofGraphNode(
              id=child,
              informal_problem=models.InformalProblem(statement=f"{child} holds"),
              reverse_dependency_ids=["root"],
          )
      )
    graph.nodes["root"].and_nodes.append(
        dag.AndNode(id="d0", dependency_ids=["kid_a", "kid_b"], subgoal_signatures=_sigs(["kid_a", "kid_b"]))
    )
    return graph

  def _render(self, graph, phase="informal"):
    return sm.workspace_manager.WorkspaceManager(self.root).render_context_md(
        "root", graph, pipeline.get_phase(phase), root_id="root"
    )

  def test_one_row_per_live_child(self):
    text = self._render(self._seed_tree())
    self.assertIn("## 9. Subtree Ledger", text)
    self.assertIn("| `kid_a` |", text)
    self.assertIn("| `kid_b` |", text)

  def test_each_child_is_indexed_to_its_own_findings_file(self):
    """The findings are prose the child owns; the graph holds none of it."""
    text = self._render(self._seed_tree())
    self.assertIn("`nodes/kid_a/findings.md`", text)
    self.assertIn("`nodes/kid_b/findings.md`", text)

  def test_the_children_are_shown_as_statements_not_just_names(self):
    """A reviewer rules on whether these close the target, which a list of ids
    cannot answer."""
    text = self._render(self._seed_tree())
    self.assertIn("theorem kid_a : kid_a holds", text)

  def test_a_dependency_with_no_node_is_named_rather_than_left_blank(self):
    """An empty line inside the Lean block reads as a child with an empty
    goal, which is a different and much worse thing to show a reviewer."""
    graph = self._seed_tree()
    graph.nodes["root"].and_nodes[0].dependency_ids.append("ghost")
    self.assertIn("-- ghost: no formal statement yet", self._render(graph))

  def test_a_leaf_prover_is_not_shown_the_ledger(self):
    self.assertNotIn("## 9.", self._render(self._seed_tree(), phase="formal"))

  def test_a_retired_branch_leaves_no_live_decomposition(self):
    """Section 6 is where a dead split is read about; 9 is what is in flight.

    Listing retired children here as if they were live is how a planner ends up
    re-proposing the split a reviewer just threw out.
    """
    graph = self._seed_tree()
    graph.nodes["root"].and_nodes[0].failed = True
    text = self._render(graph)
    self.assertIn("*No live decomposition.*", text)
    self.assertNotIn("| `kid_a` |", text)


class ClaimVisibilityTest(ManagerTestCase):
  """Who is holding what, as seen from outside.

  A claim never expires, so there is no TTL to flag and no expiry sweep to
  test. What is left is that the orchestrator can see the hold at all: it is
  the only party that can free one, and a node held by an agent that died is
  otherwise indistinguishable from a node being worked on.
  """

  def test_a_held_node_is_counted_in_the_summary(self):
    self.seed()
    mgr = self.make()
    self.assertEqual(mgr.get_summary()["claimed"], 0)
    mgr.claim_next_subgoal()
    self.assertEqual(mgr.get_summary()["claimed"], 1)

  def test_releasing_uncounts_it(self):
    self.seed()
    mgr = self.make()
    mgr.claim_next_subgoal()
    mgr.release_subgoal("root")
    self.assertEqual(mgr.get_summary()["claimed"], 0)

  def test_inspect_carries_the_count_to_the_orchestrator(self):
    self.seed()
    self.make().claim_next_subgoal()
    out = sm.run_state_manager(
        sm.StateManagerInput(
            action="inspect", filepath=self.filepath, root_dir=self.root
        )
    )
    self.assertTrue(out.success)
    self.assertEqual(out.metadata["graph_summary"]["claimed"], 1)


class ClaimOrderTest(ManagerTestCase):
  """Which of several eligible siblings goes out first.

  Declared order, and nothing else. The crux-first / stuck-last sort this used
  to run guessed at which subgoal mattered from a plan the graph no longer
  stores, and its only observable effect was to make the walk unpredictable.
  """

  def seed_siblings(self, kids=("kid_a", "kid_b", "kid_c")):
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      for nid in kids:
        graph.add_node(
            dag.ProofGraphNode(
                id=nid,
                informal_problem=models.InformalProblem(
                    statement=f"{nid} holds"
                ),
                reverse_dependency_ids=["root"],
            )
        )
      graph.nodes["root"].and_nodes.append(
          dag.AndNode(id="d0", dependency_ids=list(kids), subgoal_signatures=_sigs(list(kids)))
      )
      graph.nodes["root"].search_metadata["phase"] = "blocked"
    return mgr

  def test_the_first_declared_child_goes_out_first(self):
    self.assertEqual(
        self.seed_siblings().claim_next_subgoal()["node_id"], "kid_a"
    )

  def test_the_walk_moves_past_a_held_child(self):
    mgr = self.seed_siblings()
    mgr.claim_next_subgoal()
    self.assertEqual(mgr.claim_next_subgoal()["node_id"], "kid_b")

  def test_the_order_is_the_same_every_time(self):
    """The graph is replayed from disk; a walk that reordered between runs
    would make the same state produce different searches."""
    mgr = self.seed_siblings()
    first = mgr.claim_next_subgoal()["node_id"]
    mgr.release_subgoal(first)
    self.assertEqual(mgr.claim_next_subgoal()["node_id"], first)

  def test_a_dangling_child_kills_the_split_rather_than_the_walk(self):
    """A dependency with no node can never be closed, so the split cannot be.

    Skipping it and handing out its siblings burns the pool on a branch that
    was already unclosable; the parent goes back to the planner instead, with
    the broken reference named.
    """
    mgr = self.seed_siblings()
    with mgr.lock_and_modify() as graph:
      graph.nodes["root"].and_nodes[0].dependency_ids.insert(0, "ghost")
    claim = mgr.claim_next_subgoal()
    self.assertEqual(claim["node_id"], "root")
    self.assertEqual(claim["phase"], "informal")
    and_node = self.load().nodes["root"].and_nodes[0]
    self.assertTrue(and_node.failed)
    self.assertIn("ghost", and_node.error_message)


_WITNESS = "Take a = 1: the statement then asserts 1 + 0 = 2, which decide refutes."


class DisproofRoutingTest(ManagerTestCase):
  """Getting *into* the disprove phase. `refute` is a redirection, not a death.

  The planner is the only role that can order one, and it does so with a
  declaration rather than a file: there is nothing to compile yet, and the
  witness it names is the whole of what the disproof agent inherits.
  """

  def claim_and_refute(self, witness=_WITNESS):
    mgr = self.make()
    claim = mgr.claim_next_subgoal()
    self.assertEqual(claim["phase"], "informal")
    return mgr, claim, sm.actions.declare(mgr, "root", "refute", witness)

  def test_a_suspicion_without_a_witness_is_refused(self):
    """`--reason` is the witness, so a one-word one leaves nothing to refute."""
    self.seed()
    _, _, (ok, msg) = self.claim_and_refute("looks false")
    self.assertFalse(ok, "an unsupported refutation must not be recorded")
    self.assertIn("--reason", msg)
    self.assertNotEqual(
        self.load().nodes["root"].search_metadata.get("phase"), "disprove"
    )

  def test_a_refusal_leaves_the_claim_held(self):
    self.seed()
    mgr, _, _ = self.claim_and_refute("nope")
    self.assertIsNone(
        mgr.claim_next_subgoal(),
        "a refused declaration is a correction, so the claim must survive",
    )

  def test_a_witness_routes_the_node_to_the_disprove_phase(self):
    self.seed()
    _, _, (ok, _) = self.claim_and_refute()
    self.assertTrue(ok)
    self.assertEqual(
        self.load().nodes["root"].search_metadata.get("phase"), "disprove"
    )

  def test_routing_releases_the_claim_without_failing_the_node(self):
    self.seed()
    self.claim_and_refute()
    node = self.load().nodes["root"]
    self.assertNotIn("decomposition_failed", node.search_metadata)
    self.assertFalse(node.search_metadata.get("processing"))

  def test_the_witness_is_kept_for_the_disproof_agent(self):
    """It is the only evidence there is; the graph stores no prose besides."""
    self.seed()
    self.claim_and_refute()
    self.assertEqual(
        self.load().nodes["root"].search_metadata.get("last_error"), _WITNESS
    )

  def test_a_redirection_is_not_counted_as_a_proof_attempt(self):
    """Nothing was compiled, so there is nothing to have attempted."""
    self.seed()
    self.claim_and_refute()
    meta = self.load().nodes["root"].search_metadata
    self.assertEqual(meta.get("proof_attempts", 0), 0)
    self.assertEqual(meta.get("decomp_attempts", 0), 0)

  def test_the_next_claim_hands_out_the_disproof_scaffold(self):
    self.seed()
    self.claim_and_refute()
    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["phase"], "disprove")
    self.assertEqual(
        os.path.basename(claim["scaffold_path"]), "disproof_scaffold.lean"
    )


class DisproofScaffoldTest(ManagerTestCase):
  """§2.8 Gate A and Gate D are enforced by the shape of the rendered file."""

  def render(self):
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      graph.nodes["root"].search_metadata["phase"] = "disprove"
    claim = self.make().claim_next_subgoal()
    with open(claim["scaffold_path"], encoding="utf-8") as f:
      return f.read()

  def test_the_target_is_assumed_as_an_axiom_never_negated(self):
    text = self.render()
    self.assertIn("axiom root (a : Nat) : a + 0 = a", text)
    self.assertNotIn("¬", text)

  def test_the_obligation_is_deriving_false(self):
    self.assertIn("theorem root_disproof : False", self.render())

  def test_the_mutable_header_regions_are_not_rendered(self):
    # Gate D. A region that is never offered cannot be filled in, so there is
    # nothing to validate and no way for a false-derived declaration to reach
    # the shared header.
    text = self.render()
    self.assertNotIn("[START_DEFINITIONS]", text)
    self.assertNotIn("[START_SUBGOALS]", text)

  def test_the_axiom_carries_the_warning_that_it_is_under_refutation(self):
    self.assertIn("Nothing proved from it may leave this file", self.render())


class DisproofBriefingTest(ManagerTestCase):
  """What a disproof agent is told. The prover's briefing is wrong here."""

  def brief(self, witness=""):
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      graph.nodes["root"].search_metadata["phase"] = "disprove"
      if witness:
        graph.nodes["root"].search_metadata["last_error"] = witness
    claim = self.make().claim_next_subgoal()
    with open(claim["context_path"], encoding="utf-8") as f:
      return f.read()

  def test_the_obligation_is_named_as_the_declaration_to_write(self):
    """It appears nowhere else on the page, and the scaffold is a file away."""
    text = self.brief()
    self.assertIn("theorem root_disproof : False", text)
    self.assertIn("Your obligation is `False`", text)

  def test_the_target_is_labelled_as_assumed_not_as_the_goal(self):
    text = self.brief()
    self.assertIn("Statement under refutation", text)
    self.assertNotIn("Target Lean Declaration", text)

  def test_the_provers_signature_rule_is_not_the_one_shown(self):
    """It says "begin with the exact declaration above", which here points at
    the axiom rather than at the obligation -- an instruction to re-state the
    thing being refuted."""
    text = self.brief()
    self.assertNotIn("MUST begin with the exact declaration above", text)

  def test_the_target_is_named_as_a_permitted_axiom(self):
    self.assertIn("plus the statement under refutation itself", self.brief())

  def test_not_using_the_target_is_advertised_as_a_rejection(self):
    """The gate rejects it, and no agent would guess that unprompted."""
    self.assertIn("MUST actually use that one", self.brief())

  def test_the_witness_is_carried_to_the_disproof_agent(self):
    witness = "Take a = 1: the goal reduces to 1 + 0 = 2, refuted by decide."
    self.assertIn(witness, self.brief(witness))

  def test_the_refutation_framing_is_in_the_role_section(self):
    self.assertIn("refuting this target, not proving it", self.brief())

  def test_the_retry_diagnostics_are_shown(self):
    # Deriving `False` rarely lands on the first compile.
    self.assertIn("Prior Attempt Diagnostics", self.brief())

  def test_the_two_ways_out_are_the_only_two_offered(self):
    text = self.brief()
    self.assertIn("--action=submit", text)
    self.assertIn("abandoning the refutation", text)
    self.assertNotIn("--action=failed", text)


class DisproofSubmissionTest(ManagerTestCase):
  """The disprove phase's own record path: one compile, two-sided axiom gate."""

  def seed_disproving(self, node_id="root", statement=None):
    """Seeds a graph with `node_id` claimed and in the disprove phase."""
    if node_id == "root":
      self.seed() if statement is None else self.seed(statement)
    else:
      self.seed()
      mgr = self.make()
      with mgr.lock_and_modify() as graph:
        graph.add_node(
            dag.ProofGraphNode(
                id=node_id,
                informal_problem=models.InformalProblem(statement=statement),
                reverse_dependency_ids=["root"],
            )
        )
        graph.nodes["root"].and_nodes.append(
            dag.AndNode(id="d0", dependency_ids=[node_id], subgoal_signatures=_sigs([node_id]))
        )
        graph.nodes["root"].search_metadata["phase"] = "blocked"
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      graph.nodes[node_id].search_metadata["phase"] = "disprove"
    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["node_id"], node_id)
    return claim["scaffold_path"]

  def submit(self, path, proof, verifier=None, target="root"):
    with open(path, "w", encoding="utf-8") as f:
      f.write(
          models.build_lean_scaffold(
              immutable_context=f"axiom {target} : ...",
              definitions="",
              subgoals="",
              proof=proof,
              role=models.SCAFFOLD_DISPROVE,
          )
      )
    # The default stand-in reports the target among the axioms, which is what
    # a real derivation of `False` from it would do.
    verifier = verifier or _fake_verifier(used_axioms=("propext", target))
    with mock.patch.object(sm.lean_verifier_tool, "run_lean_verifier", verifier):
      return self.make().record_lean(code_file=path)

  DERIVATION = (
      "theorem root_disproof : False :=\nby\n  exact absurd (root 1) (by decide)"
  )

  def disprove_kid(self):
    """Disproves a child of the root, asserting it actually landed."""
    path = self.seed_disproving("kid", "theorem kid : 1 = 2")
    handled, msg = self.submit(
        path,
        "theorem kid_disproof : False :=\nby\n  exact absurd kid (by decide)",
        target="kid",
    )
    self.assertTrue(handled, msg)
    return msg

  def test_a_verified_derivation_of_false_closes_the_node(self):
    path = self.seed_disproving()
    handled, msg = self.submit(path, self.DERIVATION)
    self.assertTrue(handled, msg)
    meta = self.load().nodes["root"].search_metadata
    self.assertTrue(meta["decomposition_failed"])
    self.assertTrue(meta["disproved"])

  def test_the_node_says_it_was_refuted_rather_than_merely_given_up_on(self):
    """The two deaths differ in what the next planner may propose, and the
    briefing renders this line."""
    path = self.seed_disproving()
    self.submit(path, self.DERIVATION)
    meta = self.load().nodes["root"].search_metadata
    self.assertIn("DISPROVED", meta["last_error"])
    self.assertIn("false as written", meta["last_error"])

  def test_the_compiled_derivation_is_kept_in_the_node_log(self):
    """The graph stores no certificate, so the only record of what was shown
    is the file that was compiled."""
    path = self.seed_disproving()
    self.submit(path, self.DERIVATION)
    logs = os.path.join(self.root, "nodes", "root", "logs")
    bodies = [
        open(os.path.join(logs, f), encoding="utf-8").read()
        for f in os.listdir(logs)
        if f.endswith(".log")
    ]
    self.assertTrue(
        any("absurd" in b and "root_disproof" in b for b in bodies), bodies
    )

  def test_the_node_reads_as_failed_to_the_rest_of_the_search(self):
    path = self.seed_disproving()
    self.submit(path, self.DERIVATION)
    statuses = self.load().propagate_failures()
    self.assertEqual(statuses["root"], dag.NodeStatus.FAILED)

  def test_nothing_reaches_the_shared_header(self):
    # Gate C. The derivation is sound only under a false axiom, so a single
    # promoted declaration would poison every other proof in the tree.
    path = self.seed_disproving()
    before = self.load().environment.to_lean_header()
    self.submit(path, self.DERIVATION)
    self.assertEqual(self.load().environment.to_lean_header(), before)

  def test_a_disproved_root_leaves_nothing_to_claim(self):
    """The root is the whole problem; refuting it ends the search."""
    path = self.seed_disproving()
    self.submit(path, self.DERIVATION)
    self.assertIsNone(self.make().claim_next_subgoal())

  def test_a_disproved_child_returns_its_parent_to_planning(self):
    self.disprove_kid()
    claim = self.make().claim_next_subgoal()
    self.assertEqual(claim["node_id"], "root")
    self.assertEqual(claim["phase"], "informal")

  def test_the_obligation_signature_is_immutable(self):
    # Gate A: restating the obligation is how an agent would claim to have
    # shown something weaker than `False`.
    path = self.seed_disproving()
    handled, msg = self.submit(
        path, "theorem root_disproof : 1 = 1 :=\nby\n  rfl"
    )
    self.assertFalse(handled)
    self.assertIn("theorem root_disproof : False", msg)

  def test_a_sorried_refutation_is_refused(self):
    """`sorry` proves anything, including `False`, so it proves nothing here."""
    path = self.seed_disproving()
    handled, msg = self.submit(
        path, "theorem root_disproof : False :=\nby\n  sorry"
    )
    self.assertFalse(handled)
    self.assertIn("sorry", msg)
    # Not the axiom message: the disproof scaffold has no subgoal block, so the
    # advice that goes with `unauthorized_axiom` points at a region that is not
    # there. It must offer `replan` instead.
    self.assertNotIn("forced axiom", msg)
    self.assertIn("--action=replan", msg)

  def test_a_forced_axiom_in_a_refutation_still_says_axiom(self):
    """The other half of the split: `axiom` keeps the axiom message."""
    path = self.seed_disproving()
    handled, msg = self.submit(
        path, "axiom cheat : False\n\ntheorem root_disproof : False :=\ncheat"
    )
    self.assertFalse(handled)
    self.assertIn("axiom", msg)

  def test_a_namespaced_target_axiom_is_recognised(self):
    """Fourth site of the resolved-name bug.

    `build_disproof_header` injects the target as an axiom inside whatever
    namespace the immutable context opened, so the kernel prints `Ns.<target>`.
    Exact-matching it rejected the submission as depending on a disallowed
    axiom -- and, but for statement order, would have reported the far worse
    ENVIRONMENT INCONSISTENT.
    """
    path = self.seed_disproving()
    handled, msg = self.submit(
        path,
        self.DERIVATION,
        verifier=_fake_verifier(used_axioms=("propext", "SimpleGraph.root")),
    )
    self.assertTrue(handled, msg)

  def test_a_namespaced_target_is_not_called_disallowed(self):
    path = self.seed_disproving()
    _, msg = self.submit(
        path,
        self.DERIVATION,
        verifier=_fake_verifier(used_axioms=("propext", "SimpleGraph.root")),
    )
    self.assertNotIn("disallowed", msg)

  def test_an_unrelated_namespaced_axiom_is_still_disallowed(self):
    """Suffix matching is for the target, not for anything carrying a dot."""
    path = self.seed_disproving()
    handled, msg = self.submit(
        path,
        self.DERIVATION,
        verifier=_fake_verifier(used_axioms=("SimpleGraph.something_else",)),
    )
    self.assertFalse(handled)
    self.assertIn("disallowed", msg)

  def test_a_refutation_that_never_touches_the_target_is_refused(self):
    """The gate that catches a proof of `False` from something else entirely."""
    path = self.seed_disproving()
    handled, _ = self.submit(
        path, self.DERIVATION, verifier=_fake_verifier(used_axioms=("propext",))
    )
    self.assertFalse(handled, "deriving False without the target proves nothing")

  def test_a_failed_compile_says_nothing_about_the_node(self):
    path = self.seed_disproving()
    handled, _ = self.submit(
        path,
        self.DERIVATION,
        verifier=_fake_verifier(is_valid=False, error="unknown identifier 'foo'"),
    )
    self.assertFalse(handled)
    meta = self.load().nodes["root"].search_metadata
    # Failing to refute a statement is not evidence that it is true.
    self.assertNotIn("decomposition_failed", meta)
    self.assertNotIn("disproof", meta)

  def test_a_failed_disproof_leaves_the_node_in_the_disprove_phase(self):
    path = self.seed_disproving()
    self.submit(
        path,
        self.DERIVATION,
        verifier=_fake_verifier(is_valid=False, error="unknown identifier 'foo'"),
    )
    self.assertEqual(
        self.load().nodes["root"].search_metadata.get("phase"), "disprove"
    )

  def test_a_derivation_that_never_used_the_target_is_an_alarm(self):
    # Gate B, second side: `False` without the target means the ambient
    # environment is already inconsistent, and every proof in the tree with it.
    path = self.seed_disproving()
    handled, msg = self.submit(
        path, self.DERIVATION, verifier=_fake_verifier(used_axioms=("propext",))
    )
    self.assertFalse(handled)
    self.assertIn("ENVIRONMENT INCONSISTENT", msg)
    self.assertNotIn("disproved", self.load().nodes["root"].search_metadata)

  def test_a_derivation_leaning_on_a_disallowed_axiom_is_refused(self):
    # Gate B, first side. The target is exempt; nothing else is.
    path = self.seed_disproving()
    handled, msg = self.submit(
        path,
        self.DERIVATION,
        verifier=_fake_verifier(used_axioms=("root", "sorryAx")),
    )
    self.assertFalse(handled)
    self.assertIn("sorryAx", msg)

  def test_the_gate_exempts_the_target_and_nothing_else(self):
    """One axiom is exempt, and only because assuming it is the whole method."""
    path = self.seed_disproving()
    handled, msg = self.submit(
        path,
        self.DERIVATION,
        verifier=_fake_verifier(used_axioms=("propext", "root", "otherThm")),
    )
    self.assertFalse(handled)
    self.assertIn("otherThm", msg)
    self.assertNotIn("`root`", msg)


class DisproofReuseTest(ManagerTestCase):
  """A refutation kills the proposition, so a rename must not recover it."""

  def graph_with_refuted_kid(self):
    graph = dag.ProofGraph()
    graph.add_node(
        dag.ProofGraphNode(
            id="root",
            informal_problem=models.InformalProblem(statement="theorem root : True"),
        )
    )
    graph.add_node(
        dag.ProofGraphNode(
            id="kid",
            informal_problem=models.InformalProblem(statement="theorem kid : 1 = 2"),
        )
    )
    graph.nodes["kid"].search_metadata["disproved"] = True
    graph.nodes["kid"].search_metadata["decomposition_failed"] = True
    return graph

  def propose(self, graph, name, statement):
    return graph.add_proposed_subgoals(
        "root", [models.FormalSignature(name=name, statement=statement)]
    )

  def test_the_same_statement_under_a_new_name_is_refused(self):
    graph = self.graph_with_refuted_kid()
    with self.assertRaises(ValueError) as ctx:
      self.propose(graph, "kid_v2", "theorem kid : 1 = 2")
    self.assertIn("REUSE OF DISPROVED LEMMA", str(ctx.exception))
    self.assertIn("renaming will not help", str(ctx.exception))

  def test_the_refusal_names_the_node_holding_the_certificate(self):
    graph = self.graph_with_refuted_kid()
    with self.assertRaises(ValueError) as ctx:
      self.propose(graph, "kid_v2", "theorem kid : 1 = 2")
    self.assertIn("'kid'", str(ctx.exception))

  def test_whitespace_does_not_defeat_the_match(self):
    graph = self.graph_with_refuted_kid()
    with self.assertRaises(ValueError):
      self.propose(graph, "kid_v3", "theorem   kid :   1 = 2  ")

  def test_a_different_statement_is_still_allowed(self):
    graph = self.graph_with_refuted_kid()
    self.propose(graph, "other", "theorem other : 2 = 2")
    self.assertIn("other", graph.nodes)

  def test_an_exhausted_node_still_recovers_by_rename(self):
    # The contrast that makes the disproof rule worth having: giving up kills
    # a name, and renaming is the documented recovery.
    graph = self.graph_with_refuted_kid()
    del graph.nodes["kid"].search_metadata["disproved"]
    self.propose(graph, "kid_v2", "theorem kid : 1 = 2")
    self.assertIn("kid_v2", graph.nodes)

  def test_re_citing_the_disproved_node_by_its_own_id_is_refused_too(self):
    """Not by the disproof rule -- by the dead-lemma rule, which is enough.

    The disproof guard is about a *rename*, so it does not fire on the node's
    own id. The status guard does, and refusing here is right either way: a
    decomposition that cites a node the planner has already given up on stalls
    on it forever.
    """
    graph = self.graph_with_refuted_kid()
    with self.assertRaisesRegex(ValueError, "REUSE OF FAILED LEMMA"):
      self.propose(graph, "kid", "theorem kid : 1 = 2")


class SummaryTest(ManagerTestCase):

  def test_broken_nodes_are_counted(self):
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      graph.nodes["root"].formal_sketch = models.FormalProofSketch(
          signature=models.FormalSignature(name="root", statement="True"),
          proof=models.FormalProof(tactics="by trivial"),
      )
      graph.nodes["root"].and_nodes.append(
          dag.AndNode(id="d0", dependency_ids=["does_not_exist"], subgoal_signatures=_sigs(["does_not_exist"]))
      )
    self.assertIn("broken", mgr.get_summary())


class AssemblyGateTest(ManagerTestCase):
  """What makes `assemble` a completion check instead of a snapshot.

  `assemble_current_proof` never fails: an unproved node is stubbed with
  `sorry` and the file is written anyway. So without a gate, "assembled
  successfully" is true of an empty graph, and an orchestrator that reads it as
  "the search is finished" hands a `sorry` to the final verifier. There is no
  flag to turn the gate off -- a partial file is exactly what nobody should be
  handed at the end of a run.
  """

  def tree(self, proved=()):
    """Roots a two-child decomposition, proving the named children."""
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      for nid in ("kid_a", "kid_b"):
        graph.add_node(
            dag.ProofGraphNode(
                id=nid,
                informal_problem=models.InformalProblem(
                    statement=f"{nid} holds"
                ),
                reverse_dependency_ids=["root"],
            )
        )
      graph.nodes["root"].and_nodes.append(
          dag.AndNode(id="d0", dependency_ids=["kid_a", "kid_b"], subgoal_signatures=_sigs(["kid_a", "kid_b"]))
      )
      for nid in proved:
        graph.nodes[nid].formal_theorem = models.FormalTheorem(
            signature=models.FormalSignature(name=nid, statement="True"),
            proof=models.FormalProof(tactics="by trivial"),
        )
    return mgr

  def prove_root(self, mgr):
    with mgr.lock_and_modify() as graph:
      graph.nodes["root"].formal_theorem = models.FormalTheorem(
          signature=models.FormalSignature(name="root", statement="True"),
          proof=models.FormalProof(tactics="by trivial"),
      )

  def assemble(self, **kwargs):
    return sm.run_state_manager(
        sm.StateManagerInput(action="assemble", root_dir=self.root, **kwargs)
    )

  def final_path(self):
    return os.path.join(self.root, "assembled", "final_proof.lean")

  def test_an_open_goal_is_refused(self):
    self.tree(proved=["kid_a"])
    out = self.assemble()
    self.assertFalse(out.success)
    self.assertIn("ASSEMBLY INCOMPLETE", out.error_message)

  def test_the_refusal_names_the_open_nodes(self):
    """A count is a status line; the names are a worklist."""
    self.tree(proved=["kid_a"])
    out = self.assemble()
    self.assertIn("`kid_b`", out.error_message)
    self.assertNotIn("`kid_a`", out.error_message)

  def test_the_refusal_says_what_to_do_about_a_node_nobody_is_offered(self):
    """The one state this message exists to make recoverable: a node held by a
    worker that died, which `claim_next` will never hand out again."""
    self.tree(proved=["kid_a"])
    self.assertIn("--action=release", self.assemble().error_message)

  def test_nothing_is_written_when_the_gate_refuses(self):
    """The point of gating before the write: no path to hand the verifier."""
    self.tree()
    self.assemble()
    self.assertFalse(os.path.exists(self.final_path()))

  def test_a_finished_tree_assembles(self):
    mgr = self.tree(proved=["kid_a", "kid_b"])
    self.prove_root(mgr)
    out = self.assemble()
    self.assertTrue(out.success, out.error_message)
    self.assertTrue(os.path.exists(self.final_path()))

  def test_there_is_no_way_to_ask_for_a_half_done_file(self):
    """`--allow_sorry` is gone rather than defaulted to on.

    A default is a flag someone can still pass; the input model not carrying
    the field at all is what makes the gate unconditional.
    """
    self.assertNotIn("allow_sorry", sm.StateManagerInput.model_fields)
    self.tree()
    self.assertFalse(self.assemble().success)

  def test_the_gate_follows_the_node_being_assembled(self):
    """Assembling a proved subtree succeeds even while the root is open."""
    self.tree(proved=["kid_a"])
    self.assertTrue(self.assemble(node_id="kid_a").success)
    self.assertFalse(self.assemble(node_id="kid_b").success)

  def test_a_retired_branch_is_not_owed_a_proof(self):
    """Only the active decomposition is walked; withdrawn work is not open."""
    mgr = self.tree()
    with mgr.lock_and_modify() as graph:
      graph.nodes["root"].and_nodes[-1].failed = True
    self.prove_root(mgr)
    self.assertTrue(self.assemble().success, self.assemble().error_message)

  def test_the_gate_finds_the_root_without_being_told_which_it_is(self):
    """The bug this replaced: `root_id` is unset on a discovered invocation,
    the target resolved to None, and the gate passed everything."""
    self.tree()
    out = sm.run_state_manager(
        sm.StateManagerInput(action="assemble", filepath=self.filepath)
    )
    self.assertFalse(out.success)
    self.assertIn("`root`", out.error_message)

  def test_an_empty_graph_has_nothing_to_assemble(self):
    self.make().initialize_graph(dag.ProofGraph())
    out = self.assemble()
    self.assertFalse(out.success)
    self.assertIn("no nodes", out.error_message)

  def test_a_long_list_is_truncated_but_still_counted(self):
    self.seed()
    mgr = self.make()
    with mgr.lock_and_modify() as graph:
      kids = [f"kid_{i:02d}" for i in range(60)]
      for nid in kids:
        graph.add_node(
            dag.ProofGraphNode(
                id=nid,
                informal_problem=models.InformalProblem(statement=f"{nid} ok"),
                reverse_dependency_ids=["root"],
            )
        )
      graph.nodes["root"].and_nodes.append(
          dag.AndNode(id="d0", dependency_ids=kids, subgoal_signatures=_sigs(kids))
      )
    out = self.assemble()
    self.assertIn("61 node(s)", out.error_message)
    self.assertIn("(+21 more)", out.error_message)


_LONG_CHILD = (
    "theorem card_lt_of_inj_not_surj {a : Type*} (s t : Finset a) (I : a -> a)"
    "\n    (h_map : forall x in s, I x in t) (h_inj : Set.InjOn I s)"
    "\n    (h_notsurj : exists y in t, forall x in s, I x != y)"
    " : s.card < t.card"
)


class ReviewBriefingTest(ManagerTestCase):
  """What the reviewer is handed must support the checks it is asked to make.

  Three of the six checks in `agents/lemma_decomposition_reviewer.md` were unperformable: the
  child signatures were truncated to a ledger width, the glue proof appeared
  in no section, and the ancestry section was not in the review phase's list.
  A reviewer that approves on a truncated signature has not reviewed anything,
  and the approval is what releases N children to the pool.
  """

  def seed_review(self, child_stmt=_LONG_CHILD, glue="by exact kid_a", depth=1):
    """A node in `review` over one child, `depth` levels below the root."""
    self.seed()
    mgr = self.make()
    chain = ["root"] + [f"mid{i}" for i in range(1, depth)]
    with mgr.lock_and_modify() as graph:
      for nid in chain[1:]:
        graph.add_node(
            dag.ProofGraphNode(
                id=nid,
                informal_problem=models.InformalProblem(
                    statement=f"theorem {nid} (a : Nat) : a * 1 = a"
                ),
            )
        )
      graph.add_node(
          dag.ProofGraphNode(
              id="kid_a", informal_problem=models.InformalProblem(
                  statement=child_stmt
              )
          )
      )
      for parent, child in zip(chain, chain[1:] + ["kid_a"]):
        graph.nodes[parent].and_nodes.append(
            dag.AndNode(id=f"d_{parent}", dependency_ids=[child], subgoal_signatures=_sigs([child]))
        )
        # Ancestry is walked upward through the reverse edges, not downward
        # through `and_nodes`. Omitting these leaves section 3 rendering
        # "*This is a top-level root goal.*" for every node in the chain.
        graph.nodes[child].reverse_dependency_ids.append(parent)
        graph.nodes[parent].formal_sketch = models.FormalProofSketch(
            signature=models.FormalSignature(
                name=parent, statement=f"{parent}_holds"
            ),
            proof=models.FormalProof(tactics=glue),
        )
        graph.nodes[parent].search_metadata["phase"] = "blocked"
        # Every ancestor of the node under review shares its subtree, so
        # without this the patrol fires on the root and the DFS hands out that
        # review instead: it returns the topmost node in `review` and never
        # descends past it. A high mark is what an earlier approval on each of
        # them would have left behind.
        graph.nodes[parent].and_nodes[-1].review_replan_rate = 99.0
      reviewed = chain[-1]
      # ...but not on the node under review: it is here because a prover just
      # submitted the split, and the entrance text says which of the two
      # entrances it came through.
      graph.nodes[reviewed].and_nodes[-1].review_replan_rate = None
      graph.nodes[reviewed].search_metadata["phase"] = "review"
      graph.nodes[reviewed].search_metadata["last_outcome"] = "decomposed"
    return mgr

  def brief(self, **kwargs):
    """Claims the review and returns the briefing text it points at."""
    claim = self.seed_review(**kwargs).claim_next_subgoal()
    self.assertEqual(claim["phase"], "review", claim)
    with open(claim["context_path"], encoding="utf-8") as f:
      return f.read()

  def test_the_child_signature_is_not_truncated(self):
    """The hypotheses sit past the 100-character ledger cut, and they are
    exactly what check 6 is about."""
    context = self.brief()
    self.assertIn("h_notsurj", context)
    self.assertIn("s.card < t.card", context)

  def test_the_children_are_rendered_as_lean_not_as_a_bullet_list(self):
    context = self.brief()
    self.assertIn("```lean\ntheorem card_lt_of_inj_not_surj", context)

  def test_the_glue_proof_is_shown(self):
    """Check 1 -- assume every child holds, does the parent follow? -- is a
    question about this text, and the reviewer has no scaffold to open."""
    context = self.brief(glue="by simpa using kid_a")
    self.assertIn("Combined into the target by this sketch", context)
    self.assertIn("by simpa using kid_a", context)

  def test_a_node_with_no_sketch_omits_the_glue_section(self):
    """A patrol can land on a node whose decomposition predates a sketch.
    Nothing to show is not the same as a section saying nothing."""
    context = self.brief(glue="   ")
    self.assertNotIn("Combined into the target", context)
    self.assertIn("theorem card_lt_of_inj_not_surj", context,
                  "the children still render")

  def test_the_reviewer_is_told_why_it_was_called(self):
    """A reviewer that thinks the branch is already built is reluctant to
    retire it; nothing has run yet, and the page has to say so."""
    self.assertIn("Nothing has been built on it", self.brief())

  def test_the_reviewer_is_shown_the_ancestor_chain(self):
    """Check 2's other half: a child that paraphrases its *grandparent* is the
    search going in circles, and nothing else on the page shows the chain."""
    context = self.brief(depth=3)
    self.assertIn("Upstream Ancestor Sketches", context)
    self.assertIn("**Parent Goal**: `theorem mid1", context)
    self.assertIn("theorem root : root_holds", context, "the root, two up")

  def test_ancestry_is_in_the_review_phase_section_list(self):
    self.assertIn(3, sm.pipeline.PIPELINE["review"].sections)

  def test_the_reviewer_still_gets_no_scaffold(self):
    """Widening the briefing must not turn the reviewer into a prover."""
    self.assertIsNone(sm.pipeline.PIPELINE["review"].scaffold)
    self.assertNotIn("[START_PROOF]", self.brief())


class ReviewerMemoryTest(ManagerTestCase):
  """Reviewers do not share state, so the page has to carry it.

  Each review is a fresh model with no record of the ones before it. Without
  the retired decompositions on the page the fourth reviewer of a node rules
  exactly like the first and is free to raise a fourth unrelated objection --
  a ratchet, not a gate. fmf_v1_002's root took four reviews, each finding
  something different, and never dispatched a child.
  """

  def _render(self, phase, failed=True):
    graph = self.seed()
    if failed:
      graph.nodes["root"].and_nodes.append(
          dag.AndNode(
              id="d0",
              dependency_ids=["dead_lemma"],
              subgoal_signatures=[
                  models.FormalSignature(name="dead_lemma", statement="0 = 0")
              ],
              failed=True,
              error_message="SHAPE: the split postpones the crux.",
          )
      )
    return sm.workspace_manager.WorkspaceManager(self.root).render_context_md(
        "root", graph, pipeline.get_phase(phase), root_id="root"
    )

  def test_the_reviewer_is_shown_what_was_retired_here_before(self):
    text = self._render(pipeline.REVIEW_PHASE)
    self.assertIn("Prior Decomposition History", text)
    self.assertIn("SHAPE: the split postpones the crux.", text)

  def test_the_history_is_in_the_review_phase_section_list(self):
    """The renderer only emits a section the phase declared, so the tuple is
    the whole of the fix."""
    self.assertIn(6, pipeline.PIPELINE[pipeline.REVIEW_PHASE].sections)

  def test_the_reviewer_is_not_told_to_write_a_different_decomposition(self):
    """The section's instruction is the planner's: it explains the commit-time
    refusal rules and orders a new split. A reviewer proposes nothing."""
    text = self._render(pipeline.REVIEW_PHASE)
    self.assertNotIn("A different decomposition is required", text)
    self.assertNotIn("Do NOT re-propose", text)
    self.assertNotIn("Formulate a genuinely different", text)
    self.assertIn("This is not the first plan for this node", text)

  def test_the_planner_still_gets_its_own_instruction(self):
    text = self._render("informal")
    self.assertIn("A different decomposition is required", text)
    self.assertIn("Formulate a genuinely different", text)
    self.assertNotIn("This is not the first plan for this node", text)

  def test_a_first_time_gate_carries_no_history(self):
    """Nothing has been retired here, so there is nothing to remember and the
    section must not appear at all."""
    self.assertNotIn(
        "Prior Decomposition History",
        self._render(pipeline.REVIEW_PHASE, failed=False),
    )


class PlotRenderingTest(ManagerTestCase):
  """The proof-graph SVG must survive the first failure in a run.

  `generate_plot` catches broadly and logs, so a crash in here is invisible:
  the picture just stops updating, and it stops exactly when a run gets
  interesting. These go through `_generate_dot` on purpose -- it is the half
  that raises, and testing the caught wrapper would pass either way.
  """

  def dead_root(self, cost=None):
    """Seeds a graph whose only node is dead, with `cost` already on disk."""
    self.seed()
    with self.make().lock_and_modify() as graph:
      meta = graph.nodes["root"].search_metadata
      meta["decomposition_failed"] = True
      if cost is not None:
        meta["live_heuristic_cost"] = cost
    return self.load()

  def test_a_failed_node_renders(self):
    dot = self.make()._generate_dot(self.dead_root())
    self.assertIn("h: ", dot)

  def test_a_dead_nodes_cost_survives_a_round_trip_to_disk(self):
    """The regression this class exists for.

    A dead node's cost was `float("inf")`, which pydantic writes to JSON as
    `null`; the graph reloaded with `None` there and the first `f"{cost:.1f}"`
    raised. `generate_plot` catches broadly, so the only symptom was that the
    picture quietly stopped updating -- at exactly the point a run got
    interesting. `dag.UNREACHABLE_COST` is a real float for this reason.
    """
    self.seed()
    with self.make().lock_and_modify() as graph:
      graph.nodes["root"].search_metadata["decomposition_failed"] = True
      graph.propagate_failures()
      graph.recalculate_empirical_metrics("root")
    reloaded = self.load().nodes["root"].search_metadata
    self.assertEqual(reloaded["live_heuristic_cost"], dag.UNREACHABLE_COST)
    self.assertIn(f"h: {dag.UNREACHABLE_COST:.1f}",
                  self.make()._generate_dot(self.load()))

  def test_the_whole_graph_renders_at_every_phase(self):
    """A phase with no colour is legible, but a phase that crashes the drawing
    is not, and the drawing runs on every mutation."""
    for phase in sm.pipeline.PIPELINE:
      with self.subTest(phase=phase):
        self.seed()
        with self.make().lock_and_modify() as graph:
          graph.nodes["root"].search_metadata["phase"] = phase
        self.assertIn("root", self.make()._generate_dot(self.load()))


class PlotStateVisibilityTest(ManagerTestCase):
  """The picture has to distinguish the states the pipeline actually has.

  It did not: phase was absent entirely, and a disproof -- a kernel-checked
  refutation and a successful end to the run -- was drawn in the same red as a
  branch that fell over, because `_record_disproof` sets `decomposition_failed`
  on its way out.
  """

  def styled(self, **meta):
    """Renders a one-node graph carrying exactly `meta`, returns its DOT line.

    The metadata is replaced rather than updated: `initialize_graph` merges
    into whatever is already on disk, so a second call in the same test was
    inheriting the first one's `disproved` and every terminal state came out
    purple.
    """
    self.seed()
    with self.make().lock_and_modify() as graph:
      graph.nodes["root"].search_metadata = dict(meta)
    dot = self.make()._generate_dot(self.load())
    return next(l for l in dot.splitlines() if l.strip().startswith('"root"'))

  def test_the_phase_is_on_the_node(self):
    self.assertIn("phase: formal", self.styled(phase="formal"))

  def test_two_phases_do_not_look_alike(self):
    """The whole point. Both are unclaimed and both were lightgray
    `formal_statement`, so a live run showed nothing about where work was."""
    self.assertNotEqual(
        self.styled(phase="informal"), self.styled(phase="review")
    )

  def test_every_phase_has_its_own_colour(self):
    """Five phases sharing three colours would leave the picture ambiguous in
    exactly the cases worth telling apart."""
    self.assertEqual(
        len(set(sm.plot_style.PHASE_STYLES.values())),
        len(sm.plot_style.PHASE_STYLES),
        sm.plot_style.PHASE_STYLES,
    )

  def test_no_phase_in_the_table_is_missing_a_colour(self):
    self.assertEqual((), sm.plot_style.missing_phase_styles())

  def test_a_node_out_of_the_machine_says_so(self):
    self.assertIn("left the machine", self.styled(phase=None))

  def test_a_disproof_is_not_drawn_as_a_failure(self):
    """`_record_disproof` sets `decomposition_failed` too -- nothing under a
    refuted proposition can be worked again -- so reading that marker first
    painted failure red over a successful end to the run."""
    disproved = self.styled(
        phase=None, decomposition_failed=True, disproved=True
    )
    self.assertIn("DISPROVED", disproved)
    self.assertNotIn(sm.plot_style.TERMINAL_STYLES["FAILED"][0], disproved)

  def test_an_ordinary_failure_is_still_red(self):
    self.assertIn(
        sm.plot_style.TERMINAL_STYLES["FAILED"][0],
        self.styled(phase=None, decomposition_failed=True),
    )

  def test_the_three_ends_of_a_run_are_three_colours(self):
    """Proved and refuted are both successes; only the third is bad news."""
    self.seed()
    with self.make().lock_and_modify() as graph:
      graph.nodes["root"].formal_theorem = models.FormalTheorem(
          signature=models.FormalSignature(name="root", statement="root_stmt"),
          proof=models.FormalProof(tactics="trivial"),
      )
    proved = next(
        l
        for l in self.make()._generate_dot(self.load()).splitlines()
        if l.strip().startswith('"root"')
    )
    kinds = [
        proved,
        self.styled(phase=None, decomposition_failed=True, disproved=True),
        self.styled(phase=None, decomposition_failed=True),
    ]
    fills = [re.search(r'fillcolor="([^"]+)"', k).group(1) for k in kinds]
    self.assertEqual(len(set(fills)), 3, fills)

  def test_a_live_claim_shows_its_age(self):
    """A claim never expires, so a node held by a dead worker and one held by
    a worker that is thinking look identical. The age is the only tell."""
    self.assertIn(
        "held 2m",
        self.styled(
            phase="formal", processing=True, claim_timestamp=time.time() - 120
        ),
    )

  def test_an_unclaimed_node_shows_no_age(self):
    self.assertNotIn("held", self.styled(phase="formal"))

  def test_a_claim_with_no_timestamp_still_says_it_is_claimed(self):
    self.assertIn("claimed", self.styled(phase="formal", processing=True))

  def test_a_retired_branch_keeps_its_edges(self):
    """Retiring a split used to detach its children from the drawing, orphaned
    proved descendants included -- the one event most worth seeing."""
    dot = self._two_node_graph(failed=True)
    self.assertIn('"root" -> "kid"', dot)
    self.assertIn("style=dashed", dot)

  def test_a_live_branch_is_not_dashed(self):
    self.assertIn(
        '"root" -> "kid" [color="black"];', self._two_node_graph(failed=False)
    )

  def test_a_retired_split_and_its_replacement_are_both_drawn(self):
    """The retired one is the history of how the live one got proposed, and
    whatever is proved under it is still proved."""
    self.seed()
    with self.make().lock_and_modify() as graph:
      for kid in ("kid", "kid2"):
        graph.add_node(
            dag.ProofGraphNode(
                id=kid,
                informal_problem=models.InformalProblem(
                    statement=f"theorem {kid} : True"
                ),
                reverse_dependency_ids=["root"],
            )
        )
      graph.nodes["root"].and_nodes = [
          dag.AndNode(id="d0", dependency_ids=["kid"], subgoal_signatures=_sigs(["kid"]), failed=True),
          dag.AndNode(id="d1", dependency_ids=["kid2"], subgoal_signatures=_sigs(["kid2"])),
      ]
    dot = self.make()._generate_dot(self.load())
    self.assertIn('"root" -> "kid" [style=dashed, color="black"];', dot)
    self.assertIn('"root" -> "kid2" [color="black"];', dot)

  def _two_node_graph(self, failed):
    """Renders `root -> kid` through one decomposition, retired or not."""
    self.seed()
    with self.make().lock_and_modify() as graph:
      graph.add_node(
          dag.ProofGraphNode(
              id="kid",
              informal_problem=models.InformalProblem(
                  statement="theorem kid : True"
              ),
              reverse_dependency_ids=["root"],
          )
      )
      graph.nodes["root"].and_nodes = [
          dag.AndNode(id="d0", dependency_ids=["kid"], subgoal_signatures=_sigs(["kid"]), failed=failed)
      ]
    return self.make()._generate_dot(self.load())


class NamespacedKernelNameTest(ManagerTestCase):
  """Kernel names are resolved; declared names are not.

  `fmf_v1_002` opened `namespace SimpleGraph` in its immutable context and never
  closed it, so a subgoal declared `foo` came back from the kernel as
  `SimpleGraph.foo`. Exact-matching the two rejected 13 sound decompositions as
  depending on a disallowed axiom.
  """

  SUBGOAL = "theorem step_one (a : Nat) : 0 + a = a := sorry"
  GLUE = "theorem root (a : Nat) : a + 0 = a :=\n  by simpa using step_one a"

  def _submit(self, used_axioms):
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    path = self.write_scaffold(
        "root", models.SCAFFOLD_PROVE, subgoals=self.SUBGOAL, proof=self.GLUE
    )
    with mock.patch.object(
        sm.lean_verifier_tool,
        "run_lean_verifier",
        _fake_verifier(used_axioms=used_axioms),
    ):
      ok, msg = mgr.record_lean(path)
    return ok, msg, self.load().nodes["root"]

  def test_a_namespaced_subgoal_is_recognised(self):
    ok, msg, node = self._submit(["propext", "SimpleGraph.step_one"])
    self.assertTrue(ok, msg)
    self.assertTrue(node.and_nodes, "the subgoal must still be attributed")
    self.assertEqual(["step_one"], node.and_nodes[0].dependency_ids)

  def test_a_deeply_namespaced_subgoal_is_recognised(self):
    ok, msg, _ = self._submit(["A.B.step_one"])
    self.assertTrue(ok, msg)

  def test_a_bare_subgoal_still_works(self):
    ok, msg, node = self._submit(["propext", "step_one"])
    self.assertTrue(ok, msg)
    self.assertEqual(["step_one"], node.and_nodes[0].dependency_ids)

  def test_an_unrelated_namespaced_axiom_is_still_disallowed(self):
    """Suffix matching is for declared subgoals, not for anything with a dot."""
    ok, msg, _ = self._submit(["SimpleGraph.something_else"])
    self.assertFalse(ok)
    self.assertIn("disallowed axioms", msg)

  def test_the_base_allowlist_is_never_suffix_matched(self):
    """The soundness half. `Evil.propext` is not `propext` and never may be."""
    ok, msg, _ = self._submit(["Evil.propext", "SimpleGraph.step_one"])
    self.assertFalse(ok, "a forged propext must not be waved through")
    self.assertIn("Evil.propext", msg)

  def test_a_name_merely_ending_in_the_subgoal_is_not_it(self):
    """Only a namespace boundary counts; `xstep_one` is a different lemma."""
    ok, msg, _ = self._submit(["xstep_one"])
    self.assertFalse(ok)
    self.assertIn("disallowed axioms", msg)


class UnclosedRegionTest(ManagerTestCase):
  """A start tag with no end tag made its whole region vanish in silence.

  The section regex spans start-to-end, so a deleted `-- [END_DEFINITIONS]`
  read as an empty definitions block and the helper lemmas were dropped without
  a word -- the agent then saw `unknown identifier` for a helper in its own
  file. `missing_defs_tag` was computed for this and never read.
  """

  PROOF = "theorem root (a : Nat) : a + 0 = a :=\n  by simp"

  def _submit_raw(self, text):
    self.seed()
    mgr = self.make()
    claim = self.claim_formal(mgr)
    with open(claim["scaffold_path"], "w", encoding="utf-8") as f:
      f.write(text)
    with mock.patch.object(
        sm.lean_verifier_tool, "run_lean_verifier", _fake_verifier()
    ):
      return mgr.record_lean(claim["scaffold_path"])

  def _scaffold(self, drop=None):
    text = models.build_lean_scaffold(
        immutable_context="import Mathlib",
        definitions="def helper : Nat := 0",
        subgoals="",
        proof=self.PROOF,
        role=models.SCAFFOLD_PROVE,
    )
    if drop is not None:
      end = f"-- [END_{drop}]"
      self.assertIn(end, text, "precondition: the tag is there to remove")
      text = text.replace(end, "")
    return text

  def test_a_complete_scaffold_is_accepted(self):
    ok, msg = self._submit_raw(self._scaffold())
    self.assertTrue(ok, msg)

  def _assert_named(self, region):
    ok, msg = self._submit_raw(self._scaffold(drop=region))
    self.assertFalse(ok, f"a dropped END_{region} must not pass silently")
    self.assertIn(f"[END_{region}]", msg)
    self.assertIn(f"[START_{region}]", msg)

  # One method per region rather than a subTest loop: a rejection moves the
  # node out of `formal`, so a second submission in the same test has nothing
  # left to claim.
  def test_a_missing_definitions_end_tag_is_named(self):
    self._assert_named("DEFINITIONS")

  def test_a_missing_subgoals_end_tag_is_named(self):
    self._assert_named("SUBGOALS")

  def test_a_missing_proof_end_tag_is_named(self):
    self._assert_named("PROOF")

  def test_a_missing_proof_end_tag_no_longer_reads_as_an_empty_block(self):
    """The old message blamed the agent for an empty block it had filled in."""
    _, msg = self._submit_raw(self._scaffold(drop="PROOF"))
    self.assertNotIn("block is empty", msg)

  def test_a_region_the_scaffold_never_renders_is_not_demanded(self):
    """The disproof scaffold has no definitions or subgoals region at all."""
    text = models.build_lean_scaffold(
        immutable_context="axiom root : ...",
        definitions="",
        subgoals="",
        proof="theorem root_disproof : False :=\nby\n  exact absurd (root 1) (by decide)",
        role=models.SCAFFOLD_DISPROVE,
    )
    self.assertNotIn("[START_DEFINITIONS]", text)
    self.assertNotIn("[START_SUBGOALS]", text)


class UnrecognizedSubgoalsTest(ManagerTestCase):
  """A subgoals block that parses to nothing was silently demoted to a proof.

  `record_lean` routes on the parsed list, so `axiom h1 : P` in the subgoals
  block -- the natural thing to write, since that is what the block becomes --
  yielded [], the submission was recorded as a direct proof, and the block was
  never mentioned again.
  """

  PROOF = "theorem root (a : Nat) : a + 0 = a :=\n  by simp"

  def _submit(self, subgoals):
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    path = self.write_scaffold(
        "root", models.SCAFFOLD_PROVE, subgoals=subgoals, proof=self.PROOF
    )
    with mock.patch.object(
        sm.lean_verifier_tool, "run_lean_verifier", _fake_verifier()
    ):
      return mgr.record_lean(path)

  def test_an_axiom_in_the_subgoals_block_is_refused(self):
    ok, msg = self._submit("axiom h1 : 1 = 1")
    self.assertFalse(ok, "this used to be recorded as a direct proof")
    self.assertIn("[START_SUBGOALS]", msg)

  def test_a_def_in_the_subgoals_block_is_refused(self):
    ok, _ = self._submit("def h1 : Nat := 1")
    self.assertFalse(ok)

  def test_an_empty_subgoals_block_is_still_a_direct_proof(self):
    ok, msg = self._submit("")
    self.assertTrue(ok, msg)
    self.assertFalse(self.load().nodes["root"].and_nodes)

  def test_the_untouched_hint_is_still_a_direct_proof(self):
    """The regression risk. The shipped hint is all comments, so it strips to
    empty -- but if it ever stopped doing so, every direct proof would be
    rejected by the check above."""
    self.seed()
    mgr = self.make()
    claim = self.claim_formal(mgr)
    with open(claim["scaffold_path"], "r", encoding="utf-8") as f:
      pristine = f.read()
    body = pristine.split("[START_SUBGOALS]")[1].split("[END_SUBGOALS]")[0]
    self.assertTrue(body.strip(), "precondition: the hint is not blank")
    text = pristine.split("-- [START_PROOF]")[0]
    text += f"-- [START_PROOF]\n{self.PROOF}\n-- [END_PROOF]"
    with open(claim["scaffold_path"], "w", encoding="utf-8") as f:
      f.write(text)
    with mock.patch.object(
        sm.lean_verifier_tool, "run_lean_verifier", _fake_verifier()
    ):
      ok, msg = mgr.record_lean(claim["scaffold_path"])
    self.assertTrue(ok, msg)

  def test_a_real_subgoal_still_decomposes(self):
    self.seed()
    mgr = self.make()
    self.claim_formal(mgr)
    path = self.write_scaffold(
        "root",
        models.SCAFFOLD_PROVE,
        subgoals="theorem step_one (a : Nat) : 0 + a = a := sorry",
        proof="theorem root (a : Nat) : a + 0 = a :=\n  by simpa using step_one a",
    )
    with mock.patch.object(
        sm.lean_verifier_tool,
        "run_lean_verifier",
        _fake_verifier(used_axioms=("propext", "step_one")),
    ):
      ok, msg = mgr.record_lean(path)
    self.assertTrue(ok, msg)
    self.assertTrue(self.load().nodes["root"].and_nodes)


class BackupDirTest(absltest.TestCase):
  """--backup_dir mirrors files atomically into a plain local directory."""

  def test_sync_file_copies_and_leaves_no_temp_files(self):
    with tempfile.TemporaryDirectory() as tmp:
      src = os.path.join(tmp, "proof_graph.json")
      with open(src, "w") as f:
        f.write('{"nodes": {}}')
      dest = os.path.join(tmp, "backup", "nested")
      sm.StorageClient().sync_file(src, dest)
      self.assertEqual(os.listdir(dest), ["proof_graph.json"])
      with open(os.path.join(dest, "proof_graph.json")) as f:
        self.assertEqual(f.read(), '{"nodes": {}}')

  def test_backup_dir_no_longer_refused_at_construction(self):
    with tempfile.TemporaryDirectory() as tmp:
      mgr = sm.ProofGraphStateManager(
          filepath=os.path.join(tmp, "proof_graph.json"),
          root_dir=tmp,
          backup_dir=os.path.join(tmp, "bk"),
      )
      self.assertEqual(mgr.backup_dir, os.path.join(tmp, "bk"))


if __name__ == "__main__":
  absltest.main()
