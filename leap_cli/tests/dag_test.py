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

"""Tests for ProofGraph status propagation and proof assembly.

Pure logic, no network and no Lean verifier required.
"""

import _path  # noqa: F401  (puts leap/ on sys.path)
import unittest

from absl.testing import absltest

import dag
import lean_code_parser as models


def _sigs(names):
  """Subgoal signatures for a hand-built AndNode.

  Nothing rejects an AndNode that names children without their statements any
  more, but a fixture that omits them is a legacy-format node and behaves like
  one, so build them here unless that is the point of the test.
  """
  return [
      models.FormalSignature(name=n, statement=f"Claim_{n}") for n in names
  ]


def _node(node_id, statement=None):
  return dag.ProofGraphNode(
      id=node_id,
      informal_problem=models.InformalProblem(
          statement=statement or f"theorem {node_id} : True"
      ),
  )


def _prove(node, tactics="by trivial"):
  node.formal_theorem = models.FormalTheorem(
      signature=models.FormalSignature(name=node.id, statement="True"),
      proof=models.FormalProof(tactics=tactics),
  )


def _sketch(node, tactics="by exact c"):
  node.formal_sketch = models.FormalProofSketch(
      signature=models.FormalSignature(name=node.id, statement="True"),
      proof=models.FormalProof(tactics=tactics),
  )


class SignatureStrTest(absltest.TestCase):
  """`signature_str` is what the scaffold renders and the submit gate demands,
  so anything it drops is silently dropped from the theorem being proved."""

  def test_preserves_an_unbracketed_let(self):
    """Test a problem that losts its `sSup`, its second `let` and its whole equation to a
    redundant second split. The tree then held a different, meaningless
    theorem and kept working on it."""
    stmt = (
        "theorem bollobas (n δ : ℕ) (hδ : 2 * δ < n) :\n"
        "    let maxEdges := sSup {c | c = 0}\n"
        "    let k := (n - 1) / 2\n"
        "    maxEdges = k"
    )
    self.assertEqual(_node("bollobas", stmt).signature_str, stmt)

  def test_preserves_inline_comments(self):
    stmt = "theorem zs (a b : ℕ) :\n    (a = 1) ∨  -- exception 1\n    (b = 2)"
    self.assertEqual(_node("zs", stmt).signature_str, stmt)

  def test_preserves_a_binder_default(self):
    stmt = "theorem bd (n : ℕ := 0) : n = n"
    self.assertEqual(_node("bd", stmt).signature_str, stmt)

  def test_wraps_a_bare_proposition(self):
    self.assertEqual(_node("bp", "2 + 2 = 4").signature_str, "theorem bp : 2 + 2 = 4")


class StatusPropagationTest(absltest.TestCase):

  def test_bare_node_is_formal_statement(self):
    g = dag.ProofGraph()
    g.add_node(_node("a"))
    self.assertEqual(
        g.propagate_failures()["a"], dag.NodeStatus.FORMAL_STATEMENT
    )

  def test_directly_proved_node(self):
    g = dag.ProofGraph()
    n = _node("a")
    _prove(n)
    g.add_node(n)
    self.assertEqual(
        g.propagate_failures()["a"], dag.NodeStatus.FORMAL_PROVED
    )

  def test_parent_resolves_when_all_children_proved(self):
    g = dag.ProofGraph()
    p, c = _node("p"), _node("c")
    _sketch(p)
    _prove(c)
    g.add_node(p)
    g.add_node(c)
    p.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["c"], subgoal_signatures=_sigs(["c"])))
    self.assertEqual(
        g.propagate_failures()["p"], dag.NodeStatus.FORMAL_PROVED
    )

  def test_parent_stays_sketched_while_child_unproved(self):
    g = dag.ProofGraph()
    p, c = _node("p"), _node("c")
    _sketch(p)
    g.add_node(p)
    g.add_node(c)
    p.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["c"], subgoal_signatures=_sigs(["c"])))
    self.assertEqual(
        g.propagate_failures()["p"], dag.NodeStatus.FORMAL_SKETCHED
    )

  def test_failed_child_marks_and_node_failed(self):
    g = dag.ProofGraph()
    p, c = _node("p"), _node("c")
    _sketch(p)
    c.search_metadata["decomposition_failed"] = True
    g.add_node(p)
    g.add_node(c)
    p.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["c"], subgoal_signatures=_sigs(["c"])))
    g.propagate_failures()
    self.assertTrue(p.and_nodes[0].failed)

  def test_and_node_with_no_dependencies_does_not_prove_parent(self):
    """A zero-dependency AND-node is vacuously satisfied and must not count.

    Creating one is blocked in add_proposed_subgoals; this covers a legacy or
    hand-edited graph that already contains one.
    """
    g = dag.ProofGraph()
    p = _node("p")
    _sketch(p)
    g.add_node(p)
    p.and_nodes.append(dag.AndNode(id="d0", dependency_ids=[]))
    self.assertNotEqual(
        g.propagate_failures()["p"], dag.NodeStatus.FORMAL_PROVED
    )

  def test_empty_decomposition_is_refused(self):
    g = dag.ProofGraph()
    g.add_node(_node("p"))
    with self.assertRaises(ValueError):
      g.add_proposed_subgoals("p", [])


class ActiveAndNodeConsistencyTest(absltest.TestCase):
  """Status and assembly must agree on which AND-node is active.

  Set-up: and_nodes = [live(->c), failed]. 'Last non-failed' selects the live
  one; 'and_nodes[-1]' selects the failed one.
  """

  def _graph(self):
    g = dag.ProofGraph()
    p, c = _node("p"), _node("c")
    _sketch(p)
    _prove(c)
    g.add_node(p)
    g.add_node(c)
    p.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["c"], subgoal_signatures=_sigs(["c"]), failed=False))
    p.and_nodes.append(dag.AndNode(id="d1", dependency_ids=["c"], subgoal_signatures=_sigs(["c"]), failed=True))
    return g

  def test_status_reports_proved(self):
    self.assertEqual(
        self._graph().propagate_failures()["p"], dag.NodeStatus.FORMAL_PROVED
    )

  def test_proved_node_does_not_assemble_to_sorry(self):
    """Status and assembly must agree; they disagreed before fix 3.2."""
    self.assertNotIn("sorry", self._graph().assemble_current_proof("p"))

  def test_proved_child_is_reachable(self):
    self.assertIn("c", self._graph().get_active_reachable_nodes("p"))

  def test_dependencies_follow_the_live_branch(self):
    deps = [n.id for n in self._graph().get_dependencies_recursive("p")]
    self.assertEqual(deps, ["c"])

  def test_active_and_node_picks_last_unfailed(self):
    g = self._graph()
    self.assertEqual(g.nodes["p"].active_and_node.id, "d0")

  def test_active_and_node_is_none_when_all_failed(self):
    g = self._graph()
    for an in g.nodes["p"].and_nodes:
      an.failed = True
    self.assertIsNone(g.nodes["p"].active_and_node)


class StatusPurityTest(absltest.TestCase):
  """compute_statuses must not touch the graph; propagate_failures must."""

  def _graph_with_dead_child(self):
    g = dag.ProofGraph()
    p, c = _node("p"), _node("c")
    _sketch(p)
    c.search_metadata["decomposition_failed"] = True
    g.add_node(p)
    g.add_node(c)
    p.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["c"], subgoal_signatures=_sigs(["c"])))
    return g

  def test_compute_statuses_does_not_mutate(self):
    g = self._graph_with_dead_child()
    before = g.model_dump_json()
    g.compute_statuses()
    self.assertEqual(g.model_dump_json(), before)

  def test_propagate_failures_retires_the_and_node(self):
    g = self._graph_with_dead_child()
    g.propagate_failures()
    self.assertTrue(g.nodes["p"].and_nodes[0].failed)
    self.assertIn("dead end", g.nodes["p"].and_nodes[0].error_message)

  def test_both_report_the_same_statuses(self):
    a, b = self._graph_with_dead_child(), self._graph_with_dead_child()
    self.assertEqual(a.compute_statuses(), b.propagate_failures())


class AssemblyTest(absltest.TestCase):

  def test_dependencies_precede_dependents(self):
    g = dag.ProofGraph()
    p, c = _node("p"), _node("c")
    _sketch(p)
    _prove(c)
    g.add_node(p)
    g.add_node(c)
    p.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["c"], subgoal_signatures=_sigs(["c"])))
    out = g.assemble_current_proof("p")
    self.assertLess(out.index("theorem c"), out.index("theorem p"))

  def test_unproved_leaf_becomes_sorry(self):
    g = dag.ProofGraph()
    g.add_node(_node("a"))
    self.assertIn("sorry", g.assemble_current_proof("a"))

  def test_cycle_is_rejected(self):
    g = dag.ProofGraph()
    a, b = _node("a"), _node("b")
    g.add_node(a)
    g.add_node(b)
    a.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["b"], subgoal_signatures=_sigs(["b"])))
    b.and_nodes.append(dag.AndNode(id="d1", dependency_ids=["a"], subgoal_signatures=_sigs(["a"])))
    with self.assertRaises(ValueError):
      g.get_dependencies_recursive("a")


class DynamicHeaderTest(absltest.TestCase):
  """Append-only header semantics (design decision D1)."""

  def test_order_is_preserved_and_exact_duplicates_collapse(self):
    env = models.FormalEnvironment(shared_definitions="def base : Nat := 1")
    env.dynamic_headers = [
        models.SharedHeaderBlock(content="def f : Nat := 2", node_id="A"),
        models.SharedHeaderBlock(content="def g : Nat := f", node_id="B"),
        models.SharedHeaderBlock(content="def f : Nat := 2", node_id="A"),
    ]
    self.assertEqual(
        env.to_lean_header().splitlines(),
        ["def base : Nat := 1", "def f : Nat := 2", "def g : Nat := f"],
    )


class HeaderSmokeTestAssemblyTest(absltest.TestCase):
  """The cheap tier of the global check: header + root statement only."""

  def _graph(self):
    g = dag.ProofGraph(
        environment=models.FormalEnvironment(shared_definitions="def base : Nat := 1")
    )
    p, c = _node("p"), _node("c")
    _sketch(p)
    _prove(c)
    g.add_node(p)
    g.add_node(c)
    p.and_nodes.append(dag.AndNode(id="d0", dependency_ids=["c"], subgoal_signatures=_sigs(["c"])))
    g.environment.dynamic_headers.append(
        models.SharedHeaderBlock(content="def helper : Nat := base", node_id="p")
    )
    return g

  def test_carries_the_whole_header(self):
    """Cost scales with the header, so the header is what it must cover."""
    out = self._graph().assemble_header_smoke_test("p")
    self.assertIn("def base : Nat := 1", out)
    self.assertIn("def helper : Nat := base", out)

  def test_omits_every_node_but_the_root(self):
    out = self._graph().assemble_header_smoke_test("p")
    self.assertIn("theorem p", out)
    self.assertNotIn("theorem c", out)

  def test_is_smaller_than_the_full_assembly(self):
    g = self._graph()
    self.assertLess(
        len(g.assemble_header_smoke_test("p").splitlines()),
        len(g.assemble_current_proof("p").splitlines()),
    )

  def test_root_body_is_sorry_so_is_valid_is_the_only_signal(self):
    """It proves nothing by construction; callers must ignore the axiom list."""
    self.assertIn("sorry", self._graph().assemble_header_smoke_test("p"))

  def test_unknown_root_yields_empty_rather_than_raising(self):
    self.assertEqual(self._graph().assemble_header_smoke_test("nope"), "")

  def test_empty_graph_yields_empty(self):
    self.assertEqual(dag.ProofGraph().assemble_header_smoke_test(), "")


class CycleDetectionTest(absltest.TestCase):

  def test_self_dependency_is_rejected(self):
    g = dag.ProofGraph()
    g.add_node(_node("a"))
    with self.assertRaises(ValueError):
      g.add_proposed_subgoals(
          "a", [models.FormalSignature(name="a", statement="True")]
      )

  def test_has_path_finds_transitive_route(self):
    g = dag.ProofGraph()
    for i in "abc":
      g.add_node(_node(i))
    g.nodes["a"].and_nodes.append(
        dag.AndNode(id="d0", dependency_ids=["b"], subgoal_signatures=_sigs(["b"]))
    )
    g.nodes["b"].and_nodes.append(
        dag.AndNode(id="d1", dependency_ids=["c"], subgoal_signatures=_sigs(["c"]))
    )
    self.assertTrue(g.has_path("a", "c"))
    self.assertFalse(g.has_path("c", "a"))


class DecompositionGuardTest(absltest.TestCase):
  """What `add_proposed_subgoals` refuses, and what it builds when it accepts.

  This is the only write path into the graph's shape, so every structural
  invariant the rest of the file assumes is established here.
  """

  def _graph(self):
    g = dag.ProofGraph()
    g.add_node(_node("parent"))
    return g

  def _propose(self, graph, *pairs):
    return graph.add_proposed_subgoals(
        "parent",
        [models.FormalSignature(name=n, statement=s) for n, s in pairs],
    )

  def test_an_empty_decomposition_is_refused(self):
    # An AND-node with no dependencies is vacuously satisfied, so committing
    # one would mark the parent proved with nothing under it.
    with self.assertRaisesRegex(ValueError, "empty decomposition"):
      self._propose(self._graph())

  def test_an_unknown_parent_is_refused(self):
    with self.assertRaisesRegex(ValueError, "not found"):
      dag.ProofGraph().add_proposed_subgoals(
          "ghost", [models.FormalSignature(name="k", statement="True")]
      )

  def test_a_new_name_becomes_a_node(self):
    g = self._graph()
    self._propose(g, ("kid", "kid holds"))
    self.assertEqual(g.nodes["kid"].informal_problem.statement, "kid holds")

  def test_the_child_learns_who_its_parent_is(self):
    g = self._graph()
    self._propose(g, ("kid", "kid holds"))
    self.assertEqual(g.nodes["kid"].reverse_dependency_ids, ["parent"])

  def test_the_branch_lists_its_children_in_the_order_proposed(self):
    g = self._graph()
    self._propose(g, ("b", "b holds"), ("a", "a holds"))
    self.assertEqual(
        g.nodes["parent"].active_and_node.dependency_ids, ["b", "a"]
    )

  def test_each_attempt_appends_a_branch_rather_than_replacing_one(self):
    # The old branch is the record of what was tried; a reviewer reads it.
    g = self._graph()
    self._propose(g, ("a", "a holds"))
    self._propose(g, ("b", "b holds"))
    self.assertEqual(
        [an.id for an in g.nodes["parent"].and_nodes],
        ["parent_decomp_0", "parent_decomp_1"],
    )

  def test_the_definitions_block_rides_along_with_the_branch(self):
    g = self._graph()
    g.add_proposed_subgoals(
        "parent",
        [models.FormalSignature(name="kid", statement="kid holds")],
        definitions_block="def helper : Nat := 1",
    )
    self.assertEqual(
        g.nodes["parent"].active_and_node.definitions_block,
        "def helper : Nat := 1",
    )

  def test_estimations_seed_the_search_metadata(self):
    g = self._graph()
    g.add_proposed_subgoals(
        "parent",
        [models.FormalSignature(name="kid", statement="kid holds")],
        estimations={"kid": {"estimated_cost": 9.0, "success_probability": 0.3}},
    )
    meta = g.nodes["kid"].search_metadata
    self.assertEqual(meta["heuristic_cost"], 9.0)
    self.assertEqual(meta["success_probability"], 0.3)

  def test_a_child_with_no_estimate_is_left_unseeded(self):
    g = self._graph()
    self._propose(g, ("kid", "kid holds"))
    self.assertNotIn("heuristic_cost", g.nodes["kid"].search_metadata)

  def test_reusing_a_name_for_a_different_statement_is_refused(self):
    """In Lean 4 the assembled file has one namespace, so names are unique."""
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    with self.assertRaisesRegex(ValueError, "SIGNATURE MISMATCH"):
      self._propose(g, ("kid", "something else entirely"))

  def test_the_mismatch_refusal_shows_both_statements(self):
    # Naming the collision without showing it makes the agent guess which of
    # the two is the one already in the graph.
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    with self.assertRaises(ValueError) as caught:
      self._propose(g, ("kid", "something else entirely"))
    message = str(caught.exception)
    self.assertIn("EXISTING: kid holds", message)
    self.assertIn("YOURS: something else entirely", message)

  def test_the_mismatch_refusal_forbids_a_decorated_name(self):
    # `kid_v2` is what the agent reaches for, and it is the worst answer: it
    # reads as the same lemma to everything downstream, which is how a refuted
    # statement comes back under a new spelling.
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    with self.assertRaises(ValueError) as caught:
      self._propose(g, ("kid", "something else entirely"))
    self.assertIn("Do NOT take this name and add a suffix", str(caught.exception))

  def test_whitespace_is_not_a_different_statement(self):
    # Statements are re-rendered on the way out, so a reflow would otherwise
    # read as a collision against the node's own claim.
    g = self._graph()
    g.add_node(_node("kid", "(a : Nat) : a + 0 = a"))
    self._propose(g, ("kid", "(a : Nat) :\n    a + 0 = a"))
    self.assertIn("kid", g.nodes["parent"].active_and_node.dependency_ids)

  def test_a_live_node_is_reused_rather_than_duplicated(self):
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    self._propose(g, ("kid", "kid holds"))
    self.assertEqual(g.nodes["parent"].active_and_node.dependency_ids, ["kid"])

  def test_reuse_adds_the_new_parent_without_dropping_the_old_one(self):
    g = self._graph()
    g.add_node(_node("other"))
    g.add_node(_node("kid", "kid holds"))
    g.nodes["kid"].reverse_dependency_ids.append("other")
    self._propose(g, ("kid", "kid holds"))
    self.assertEqual(g.nodes["kid"].reverse_dependency_ids, ["other", "parent"])

  def test_the_same_parent_is_not_listed_twice(self):
    g = self._graph()
    self._propose(g, ("kid", "kid holds"))
    self._propose(g, ("kid", "kid holds"))
    self.assertEqual(g.nodes["kid"].reverse_dependency_ids, ["parent"])

  def _kill(self, graph, node_id):
    """Kills `node_id` the way the planner does: `--action=failed`."""
    graph.nodes[node_id].search_metadata["decomposition_failed"] = True
    return graph

  def test_reusing_a_lemma_that_already_failed_is_refused(self):
    """Otherwise a re-plan cites the dead node and the search stalls on it."""
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    self._kill(g, "kid")
    with self.assertRaisesRegex(ValueError, "REUSE OF FAILED LEMMA"):
      self._propose(g, ("kid", "kid holds"))

  def test_a_lemma_whose_plan_died_is_still_citable(self):
    # Only the planner's own `failed` kills a node. A node whose every
    # decomposition has been retired resolves to FORMAL_STATEMENT -- it needs
    # a new plan, and refusing to cite it would strand work that is not dead.
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    g.nodes["kid"].and_nodes.append(
        dag.AndNode(
            id="k0",
            dependency_ids=["missing"],
            subgoal_signatures=_sigs(["missing"]),
        )
    )
    self._propose(g, ("kid", "kid holds"))
    self.assertIn("kid", g.nodes["parent"].active_and_node.dependency_ids)

  def test_the_guard_reads_the_status_and_not_a_probability(self):
    # `live_success_probability` was the old signal and nothing has ever
    # written it, so the guard silently never fired. A node carrying a zero
    # under that key but no failure is live and must still be citable.
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    g.nodes["kid"].search_metadata["live_success_probability"] = 0.0
    self._propose(g, ("kid", "kid holds"))
    self.assertIn("kid", g.nodes["parent"].active_and_node.dependency_ids)

  def test_a_live_node_is_citable(self):
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    self._propose(g, ("kid", "kid holds"))
    self.assertIn("kid", g.nodes["parent"].active_and_node.dependency_ids)

  def test_a_different_name_for_the_dead_statement_is_a_fresh_node(self):
    """The recovery the guard exists to force: re-plan it, do not re-cite it."""
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    self._kill(g, "kid")
    self._propose(g, ("kid_v2", "kid holds"))
    self.assertIn("kid_v2", g.nodes)

  def test_one_bad_subgoal_condemns_the_whole_submission(self):
    # Committing the honest siblings and dropping the bad one would leave a
    # decomposition the glue proof no longer closes.
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    self._kill(g, "kid")
    with self.assertRaises(ValueError):
      self._propose(g, ("honest", "honest holds"), ("kid", "kid holds"))
    self.assertNotIn("honest", g.nodes)

  def test_a_refused_submission_leaves_no_branch_behind(self):
    g = self._graph()
    g.add_node(_node("kid", "kid holds"))
    self._kill(g, "kid")
    with self.assertRaises(ValueError):
      self._propose(g, ("kid", "kid holds"))
    self.assertEmpty(g.nodes["parent"].and_nodes)


class PeerTableTest(absltest.TestCase):
  """`get_accessible_nodes_data`: what a prover may cite without a cycle."""

  def _graph(self, *specs):
    """Builds a root with a `reader` child plus one child per spec.

    Each spec is (name, statement, proved).
    """
    g = dag.ProofGraph()
    g.add_node(_node("root"))
    sigs = [models.FormalSignature(name="reader", statement="reader holds")]
    sigs += [models.FormalSignature(name=n, statement=s) for n, s, _ in specs]
    g.add_proposed_subgoals("root", sigs)
    for name, statement, proved in specs:
      if proved:
        g.nodes[name].formal_theorem = models.FormalTheorem(
            signature=models.FormalSignature(name=name, statement=statement),
            proof=models.FormalProof(tactics="by trivial"),
        )
    return g

  def _peers(self, *specs):
    return self._graph(*specs).get_accessible_nodes_data("reader")

  def test_a_peer_is_reported_with_everything_the_briefing_renders(self):
    row = self._peers(("a", "P holds", True))[0]
    self.assertEqual(row["name"], "a")
    self.assertEqual(row["status"], "formal_proved")
    self.assertTrue(row["is_proved"])
    self.assertTrue(row["is_sibling"])
    self.assertEqual(row["node_dir"], "nodes/a")

  def test_ancestors_are_never_offered(self):
    """Citing one would be a cycle, and the guard would refuse it anyway."""
    self.assertNotIn("root", [p["name"] for p in self._peers(("a", "P", True))])

  def test_a_node_is_not_its_own_peer(self):
    self.assertNotIn(
        "reader", [p["name"] for p in self._peers(("a", "P", True))]
    )

  def test_proved_peers_sort_above_unproved_ones(self):
    peers = self._peers(("unproved", "Q holds", False), ("proved", "P holds", True))
    self.assertEqual([p["is_proved"] for p in peers], [True, False])

  def test_distinct_statements_are_separate_rows(self):
    self.assertLen(self._peers(("a", "P holds", True), ("b", "Q holds", True)), 2)

  def test_two_names_for_one_statement_are_two_rows(self):
    """Reuse is by name, so a restated peer is a second node and shows as one.

    Collapsing them was tried and reverted: only a reader of the mathematics
    can say whether a rename was deliberate, and the table is not that reader.
    """
    self.assertLen(self._peers(("a", "P holds", True), ("b", "P holds", False)), 2)

  def test_a_dead_peer_is_withheld(self):
    g = self._graph(("a", "P holds", False))
    g.nodes["a"].search_metadata["decomposition_failed"] = True
    self.assertNotIn(
        "a", [p["name"] for p in g.get_accessible_nodes_data("reader")]
    )

  def test_a_peer_someone_else_holds_is_shown_as_processing(self):
    """Unproved and in flight: citing it now would be a forward reference."""
    g = self._graph(("a", "P holds", False))
    g.nodes["a"].search_metadata["processing"] = True
    row = g.get_accessible_nodes_data("reader")[0]
    self.assertEqual(row["status"], "processing")
    self.assertTrue(row["is_processing"])

  def test_being_held_outranks_every_status_but_dead(self):
    """`_resolve_statuses` short-circuits on `processing` before it looks at
    `formal_theorem`, so a held node reads as processing whatever it holds.

    Only reachable between the kernel accepting a proof and the claim being
    cleared, which happens under one lock, so no caller can observe it on a
    real run. Pinned because it makes the `st_val != "formal_proved"` guard in
    `get_accessible_nodes_data` unreachable, and that guard reads as live.
    """
    g = self._graph(("a", "P holds", True))
    g.nodes["a"].search_metadata["processing"] = True
    self.assertEqual(
        g.get_accessible_nodes_data("reader")[0]["status"], "processing"
    )

  def test_a_dead_peer_outranks_a_held_one(self):
    g = self._graph(("a", "P holds", False))
    g.nodes["a"].search_metadata["processing"] = True
    g.nodes["a"].search_metadata["decomposition_failed"] = True
    self.assertEmpty(g.get_accessible_nodes_data("reader"))

  def test_a_bare_statement_is_rendered_as_a_signature(self):
    self.assertStartsWith(
        self._peers(("a", "P holds", False))[0]["signature"], "theorem a"
    )

  def test_a_statement_that_is_already_a_theorem_is_left_alone(self):
    peers = self._peers(("a", "theorem a : P holds", False))
    self.assertEqual(peers[0]["signature"], "theorem a : P holds")

  def test_a_peers_own_children_are_listed(self):
    g = self._graph(("a", "P holds", False))
    g.add_proposed_subgoals(
        "a", [models.FormalSignature(name="a_kid", statement="kid holds")]
    )
    row = next(p for p in g.get_accessible_nodes_data("reader") if p["name"] == "a")
    self.assertEqual(row["dependencies"], ["a_kid"])

  def test_the_table_is_capped(self):
    """A briefing that lists the whole graph is one the agent stops reading."""
    specs = [(f"n{i}", f"P{i} holds", True) for i in range(40)]
    self.assertLen(self._peers(*specs), 25)

  def test_an_unknown_node_gets_an_empty_table_rather_than_a_crash(self):
    self.assertEmpty(dag.ProofGraph().get_accessible_nodes_data("ghost"))


class LiveMetricPersistenceTest(absltest.TestCase):
  """Every number `recalculate_empirical_metrics` writes must round trip.

  `float("inf")` was the honest in-memory value for a dead node's cost and a
  lie on disk: pydantic serializes it to JSON `null`, so the graph reloaded
  with `None` there and the first `f"{cost:.1f}"` downstream raised. The plot
  writer catches broadly, so what a run actually saw was the proof-graph SVG
  quietly ceasing to render from the first failure onward.
  """

  def _dead_graph(self):
    g = dag.ProofGraph()
    g.add_node(_node("root"))
    g.nodes["root"].search_metadata["decomposition_failed"] = True
    g.recalculate_empirical_metrics("root")
    return g

  def _round_trip(self, graph):
    return dag.ProofGraph.model_validate_json(graph.model_dump_json())

  def test_a_dead_node_gets_an_unreachable_cost(self):
    meta = self._dead_graph().nodes["root"].search_metadata
    self.assertEqual(meta["live_heuristic_cost"], dag.UNREACHABLE_COST)

  def test_a_dead_node_gets_a_zero_probability(self):
    meta = self._dead_graph().nodes["root"].search_metadata
    self.assertEqual(meta["live_success_probability"], 0.0)

  def test_that_cost_is_still_a_number_after_a_round_trip(self):
    reloaded = self._round_trip(self._dead_graph())
    self.assertIsInstance(
        reloaded.nodes["root"].search_metadata["live_heuristic_cost"], float
    )

  def test_the_cost_is_never_serialized_as_null(self):
    """The whole reason it is not `inf`. Guards the constant against a revert."""
    self.assertNotIn(
        '"live_heuristic_cost":null', self._dead_graph().model_dump_json()
    )

  def test_it_formats_which_is_all_the_plot_writer_asked_for(self):
    cost = self._round_trip(self._dead_graph()).nodes[
        "root"
    ].search_metadata["live_heuristic_cost"]
    self.assertEqual(f"{cost:.1f}", f"{dag.UNREACHABLE_COST:.1f}")

  def test_it_loses_every_comparison_a_scheduler_makes(self):
    self.assertGreater(dag.UNREACHABLE_COST, 1e6)

  def test_a_proved_node_keeps_its_zero_cost(self):
    g = dag.ProofGraph()
    n = _node("root")
    _prove(n)
    g.add_node(n)
    g.recalculate_empirical_metrics("root")
    reloaded = self._round_trip(g)
    self.assertEqual(
        reloaded.nodes["root"].search_metadata["live_heuristic_cost"], 0.0
    )
    self.assertEqual(
        reloaded.nodes["root"].search_metadata["live_success_probability"], 1.0
    )


class AndNodeHistoryTest(absltest.TestCase):
  """`subgoal_signatures` is what makes a retired attempt readable later.

  `dependency_ids` holds names, and after `prune_orphans` a name can be taken
  over by different mathematics. The duplicate-strategy guard compares what was
  tried, so it has to read statements the AND-node carries rather than looking
  the names up in a graph that has moved on.
  """

  def test_a_decomposition_with_no_children_needs_no_history(self):
    self.assertEmpty(dag.AndNode(id="d0").subgoal_signatures)

  def test_the_two_lists_cannot_disagree_in_a_graph_we_wrote(self):
    """Where the guarantee actually comes from, now that nothing asserts it.

    `add_proposed_subgoals` builds `dependency_ids` and `subgoal_signatures`
    from one `subgoals` list, so a decomposition this code records always
    carries the statements. A validator was tried and could not be kept -- see
    the comment on the field -- which makes this the only thing holding the
    invariant up.
    """
    g = dag.ProofGraph()
    g.add_node(_node("root"))
    g.add_proposed_subgoals("root", _sigs(["kid_a", "kid_b"]))
    an = g.nodes["root"].and_nodes[0]
    self.assertEqual(an.dependency_ids, [s.name for s in an.subgoal_signatures])

  def test_a_graph_written_before_the_field_existed_still_loads(self):
    """Every run in the archive predates `subgoal_signatures`.

    Refusing these would make 78 finished experiments unreadable, so they load
    with the history simply absent rather than not at all.
    """
    stale = (
        '{"nodes": {"p": {"id": "p", "and_nodes":'
        ' [{"id": "d0", "dependency_ids": ["kid"]}]}}}'
    )
    g = dag.ProofGraph.model_validate_json(stale)
    self.assertEqual(g.nodes["p"].and_nodes[0].dependency_ids, ["kid"])

  def test_a_legacy_and_node_declares_nothing(self):
    """And this is what that costs.

    The duplicate-strategy guard compares the set of declared statements. A
    legacy AND-node contributes the empty set, so a strategy that already
    failed in one of those runs can be re-proposed without being recognised.
    """
    stale = (
        '{"nodes": {"p": {"id": "p", "and_nodes":'
        ' [{"id": "d0", "dependency_ids": ["kid"], "failed": true}]}}}'
    )
    g = dag.ProofGraph.model_validate_json(stale)
    self.assertEmpty(g.nodes["p"].and_nodes[0].subgoal_signatures)

  def test_the_declared_set_can_be_wider_than_the_used_one(self):
    # A prover may declare a lemma and then not cite it in the glue. The guard
    # has to see it, or padding a dead set with an unused name reads as new.
    an = dag.AndNode(
        id="d0",
        dependency_ids=["used"],
        subgoal_signatures=_sigs(["used", "unused"]),
    )
    self.assertEqual([s.name for s in an.subgoal_signatures], ["used", "unused"])

  def test_history_survives_a_round_trip(self):
    an = dag.AndNode(
        id="d0", dependency_ids=["kid"], subgoal_signatures=_sigs(["kid"])
    )
    reloaded = dag.AndNode.model_validate_json(an.model_dump_json())
    self.assertEqual(reloaded.subgoal_signatures[0].statement, "Claim_kid")


class PruneOrphansTest(absltest.TestCase):
  """Retiring a plan strands its children; leaving them strands their names.

  Every uniqueness check in the system asks "is this name in nodes?", so a
  ghost refuses the name to whoever needs it next. `fmf_v1_004` ended with 15
  of them and lost a rename spiral to one.
  """

  def _graph(self):
    """root -> (kid_a, kid_b) via a live d0, plus a retired d_dead -> ghost."""
    g = dag.ProofGraph()
    for nid in ("root", "kid_a", "kid_b", "ghost"):
      g.add_node(_node(nid))
    g.nodes["root"].and_nodes.append(
        dag.AndNode(
            id="d_dead",
            dependency_ids=["ghost"],
            subgoal_signatures=_sigs(["ghost"]),
            failed=True,
        )
    )
    g.nodes["root"].and_nodes.append(
        dag.AndNode(
            id="d0",
            dependency_ids=["kid_a", "kid_b"],
            subgoal_signatures=_sigs(["kid_a", "kid_b"]),
        )
    )
    return g

  def test_the_ghost_goes(self):
    g = self._graph()
    self.assertEqual(g.prune_orphans("root"), ["ghost"])
    self.assertNotIn("ghost", g.nodes)

  def test_the_name_is_free_again_afterwards(self):
    """The whole point: the next prover may take the name and restate it."""
    g = self._graph()
    g.prune_orphans("root")
    g.add_proposed_subgoals(
        "kid_a", [models.FormalSignature(name="ghost", statement="something new")]
    )
    self.assertEqual(g.nodes["ghost"].informal_problem.statement, "something new")

  def test_the_retired_branch_still_remembers_what_it_proposed(self):
    # This is what makes the deletion safe. Without the signatures on the
    # AND-node, pruning would be losing history.
    g = self._graph()
    g.prune_orphans("root")
    dead = g.nodes["root"].and_nodes[0]
    self.assertEqual([s.name for s in dead.subgoal_signatures], ["ghost"])

  def test_the_live_children_stay(self):
    g = self._graph()
    g.prune_orphans("root")
    self.assertIn("kid_a", g.nodes)
    self.assertIn("kid_b", g.nodes)

  def test_the_root_is_never_pruned_even_alone(self):
    g = dag.ProofGraph()
    g.add_node(_node("root"))
    self.assertEmpty(g.prune_orphans("root"))
    self.assertIn("root", g.nodes)

  def test_a_child_shared_with_a_live_branch_stays(self):
    # A lemma proposed by two decompositions, one retired and one live, is
    # still reachable and still being worked on.
    g = self._graph()
    g.nodes["root"].and_nodes[0].dependency_ids.append("kid_a")
    g.nodes["root"].and_nodes[0].subgoal_signatures.extend(_sigs(["kid_a"]))
    g.prune_orphans("root")
    self.assertIn("kid_a", g.nodes)

  def test_an_orphan_that_did_work_stays(self):
    g = self._graph()
    _prove(g.nodes["ghost"])
    self.assertEmpty(g.prune_orphans("root"))
    self.assertIn("ghost", g.nodes)

  def test_an_orphan_that_planned_stays(self):
    g = self._graph()
    g.nodes["ghost"].informal_proof = models.InformalProof(text="an argument")
    self.assertEmpty(g.prune_orphans("root"))

  def test_an_orphan_that_decomposed_stays(self):
    g = self._graph()
    g.add_node(_node("grandchild"))
    g.nodes["ghost"].and_nodes.append(
        dag.AndNode(
            id="g0",
            dependency_ids=["grandchild"],
            subgoal_signatures=_sigs(["grandchild"]),
        )
    )
    self.assertNotIn("ghost", g.prune_orphans("root"))
    self.assertIn("ghost", g.nodes)

  def test_a_node_a_worker_is_holding_right_now_stays(self):
    """`search_metadata` is the broad exclusion and this is why it is broad."""
    g = self._graph()
    g.nodes["ghost"].search_metadata["processing"] = True
    self.assertEmpty(g.prune_orphans("root"))
    self.assertIn("ghost", g.nodes)

  def test_any_metadata_at_all_is_enough_to_keep_it(self):
    # A node that owns a directory on disk keeps the name it is filed under.
    g = self._graph()
    g.nodes["ghost"].search_metadata["dispatch_count"] = 1
    self.assertEmpty(g.prune_orphans("root"))

  def test_scoring_keys_do_not_count_as_work(self):
    """The one exclusion from the broad metadata test, and it is what makes
    the prune fire at all: every real subgoal is stamped with the planner's
    estimates at birth and with derived costs on every commit, so counting
    those as work leaves the method collecting nothing."""
    g = self._graph()
    for key in dag.SCORING_METADATA_KEYS:
      g.nodes["ghost"].search_metadata[key] = 0.7
    self.assertEqual(g.prune_orphans("root"), ["ghost"])

  def test_one_real_key_among_the_scoring_ones_still_keeps_it(self):
    g = self._graph()
    for key in dag.SCORING_METADATA_KEYS:
      g.nodes["ghost"].search_metadata[key] = 0.7
    g.nodes["ghost"].search_metadata["phase"] = "informal"
    self.assertEmpty(g.prune_orphans("root"))

  def test_it_is_idempotent(self):
    g = self._graph()
    g.prune_orphans("root")
    self.assertEmpty(g.prune_orphans("root"))

  def test_a_whole_stranded_generation_goes_in_one_pass(self):
    # Reachability is measured from the root, not walked down from the nodes
    # being deleted, so depth in the dead branch does not matter and nothing
    # here has to iterate to a fixed point.
    g = self._graph()
    g.add_node(_node("grandchild"))
    g.nodes["ghost"].and_nodes.append(
        dag.AndNode(
            id="g0",
            dependency_ids=["grandchild"],
            subgoal_signatures=_sigs(["grandchild"]),
        )
    )
    g.nodes["ghost"].and_nodes.clear()
    self.assertEqual(sorted(g.prune_orphans("root")), ["ghost", "grandchild"])


class RetiredBranchTest(absltest.TestCase):
  """Retiring a plan is `and_node.failed = True` and nothing else.

  That single write is what `--action=replan` does in `actions.py`, and the
  distinction it rests on is the one this class pins: a retired branch leaves
  the *goal* alive and re-plannable. Marking the node
  `decomposition_failed` instead would resolve it to FAILED and make its parent
  backtrack, which is a reviewer killing a node it has no standing to kill.
  """

  def _graph(self):
    g = dag.ProofGraph()
    for nid in ("parent", "kid_a", "kid_b"):
      g.add_node(_node(nid))
    g.nodes["parent"].and_nodes.append(
        dag.AndNode(id="d0", dependency_ids=["kid_a", "kid_b"], subgoal_signatures=_sigs(["kid_a", "kid_b"]))
    )
    g.nodes["kid_a"].reverse_dependency_ids.append("parent")
    g.nodes["kid_b"].reverse_dependency_ids.append("parent")
    return g

  def _retire(self, graph, reason="reviewer disagreed"):
    graph.nodes["parent"].active_and_node.failed = True
    graph.nodes["parent"].and_nodes[-1].error_message = f"RETIRED BY REVIEW: {reason}"

  def test_retiring_clears_the_active_branch(self):
    g = self._graph()
    self._retire(g)
    self.assertIsNone(g.nodes["parent"].active_and_node)

  def test_a_retired_parent_is_open_for_replanning_not_dead(self):
    g = self._graph()
    self._retire(g)
    self.assertEqual(
        g.propagate_failures().get("parent"), dag.NodeStatus.FORMAL_STATEMENT
    )

  def test_a_retired_branch_is_never_re_marked_by_propagation(self):
    """A reviewer's rethink is not evidence that the children were unprovable."""
    g = self._graph()
    self._retire(g)
    g.propagate_failures()
    self.assertNotIn("dead end", g.nodes["parent"].and_nodes[0].error_message)

  def test_the_reason_survives_for_the_next_planner(self):
    g = self._graph()
    self._retire(g, reason="step 2 is unsound")
    self.assertIn(
        "step 2 is unsound",
        g.assemble_backtracking_feedback("parent")[0],
    )

  def test_a_branch_whose_child_died_is_marked_failed_by_propagation(self):
    g = self._graph()
    g.nodes["kid_a"].search_metadata["decomposition_failed"] = True
    g.propagate_failures()
    self.assertTrue(g.nodes["parent"].and_nodes[0].failed)

  def test_propagation_does_not_overwrite_a_reviewers_note(self):
    g = self._graph()
    self._retire(g, reason="step 2 is unsound")
    g.nodes["kid_a"].search_metadata["decomposition_failed"] = True
    g.propagate_failures()
    self.assertIn(
        "step 2 is unsound", g.nodes["parent"].and_nodes[0].error_message
    )

  def test_the_children_leave_the_reachable_set(self):
    # This is what makes a worker still running on one of them an orphan.
    g = self._graph()
    self._retire(g)
    self.assertEmpty(g.get_active_reachable_nodes("parent") - {"parent"})

  def test_a_retired_branch_does_not_assemble(self):
    g = self._graph()
    _prove(g.nodes["kid_a"])
    self._retire(g)
    self.assertNotIn("theorem kid_a", g.assemble_current_proof("parent"))

  def test_proved_work_under_a_retired_branch_is_still_in_the_graph(self):
    """Nothing is deleted; a later plan can cite it again by name."""
    g = self._graph()
    _prove(g.nodes["kid_a"])
    self._retire(g)
    self.assertEqual(
        g.compute_statuses()["kid_a"], dag.NodeStatus.FORMAL_PROVED
    )

  def test_a_replacement_branch_becomes_the_active_one(self):
    g = self._graph()
    self._retire(g)
    g.add_proposed_subgoals(
        "parent", [models.FormalSignature(name="kid_c", statement="c holds")]
    )
    self.assertEqual(
        g.nodes["parent"].active_and_node.dependency_ids, ["kid_c"]
    )

  def test_a_node_with_every_branch_retired_has_no_active_one(self):
    g = self._graph()
    self._retire(g)
    self.assertIsNone(g.nodes["parent"].active_and_node)


if __name__ == "__main__":
  absltest.main()
