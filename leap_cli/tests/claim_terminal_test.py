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

"""Tests for `claim_terminal`: why `claim_next` had nothing to hand out.

Four situations used to share one sentence -- "Graph is fully dead-ended or
proved" -- and two of them call for opposite actions. These tests are mostly
about keeping those two apart, so several assert on the *absence* of the wrong
advice rather than only on the presence of the right advice.

`why_nothing_to_claim` is pure text over a graph, so the graphs here are built
by hand rather than driven through the state manager. `TerminalMessageEndToEnd`
covers the one thing that cannot be checked that way: that the real CLI path
actually reaches this module.
"""

import _path  # noqa: F401  (puts leap/ on sys.path)
import os
import tempfile

from absl.testing import absltest

import claim_terminal
import dag
import lean_code_parser as models
import state_manager as sm


def _prove(node):
  """Marks `node` proved the only way there is -- `status` is derived."""
  node.formal_theorem = models.FormalTheorem(
      signature=models.FormalSignature(name=node.id, statement="True"),
      proof=models.FormalProof(tactics="by trivial"),
  )
  return node


def _node(node_id, statement=None, **metadata):
  node = dag.ProofGraphNode(
      id=node_id,
      informal_problem=models.InformalProblem(
          statement=statement or f"theorem {node_id} : True"
      ),
  )
  node.search_metadata.update(metadata)
  return node


class WhyNothingToClaimTest(absltest.TestCase):
  """The classification itself."""

  def graph(self, *nodes):
    graph = dag.ProofGraph()
    for node in nodes:
      graph.add_node(node)
    return graph

  def test_an_empty_graph_says_to_initialize(self):
    self.assertIn(
        "init_graph", claim_terminal.why_nothing_to_claim(dag.ProofGraph(), None)
    )

  def test_a_held_node_says_to_wait(self):
    graph = self.graph(_node("root", processing=True))
    msg = claim_terminal.why_nothing_to_claim(graph, "root")
    # `state_machine_test` pins this substring too -- it is the string the
    # orchestrator directive tells the model to look for.
    self.assertIn("STILL processing", msg)
    self.assertIn("DO NOT assemble", msg)

  def test_a_held_node_is_named(self):
    """Naming it is the whole point: claims never expire, so the orchestrator
    has to be able to decide whether that specific worker is dead."""
    graph = self.graph(_node("root"), _node("kid", processing=True))
    self.assertIn("`kid`", claim_terminal.why_nothing_to_claim(graph, "root"))

  def test_processing_wins_over_everything_else(self):
    """A proved root with a worker still out is not a finished search."""
    root = _prove(_node("root", processing=True))
    msg = claim_terminal.why_nothing_to_claim(self.graph(root), "root")
    self.assertIn("STILL processing", msg)
    self.assertNotIn("SEARCH FINISHED", msg)

  def test_a_proved_root_says_the_search_is_finished(self):
    root = _prove(_node("root"))
    msg = claim_terminal.why_nothing_to_claim(self.graph(root), "root")
    self.assertIn("SEARCH FINISHED", msg)
    self.assertIn("assemble", msg)

  def test_a_proved_root_still_demands_a_verifier(self):
    """The gate reads the graph, and the graph can be wrong. A run that stops
    at "assembled successfully" is exactly the failure T5 stages."""
    root = _prove(_node("root"))
    self.assertIn(
        "verifier", claim_terminal.why_nothing_to_claim(self.graph(root), "root")
    )

  def test_a_disproved_root_is_a_result_not_a_stall(self):
    graph = self.graph(_node("root", disproved=True, decomposition_failed=True))
    msg = claim_terminal.why_nothing_to_claim(graph, "root")
    self.assertIn("ROOT DISPROVED", msg)
    self.assertIn("RESULT", msg)

  def test_a_disproved_root_is_not_read_as_a_dead_one(self):
    """`_record_disproof` sets `decomposition_failed` on its way out, so status
    alone cannot tell a refutation from a crash. The marker is checked first."""
    graph = self.graph(_node("root", disproved=True, decomposition_failed=True))
    msg = claim_terminal.why_nothing_to_claim(graph, "root")
    self.assertNotIn("STALLED", msg)
    self.assertIn("Do NOT assemble", msg)

  def test_a_planner_can_no_longer_produce_a_dead_root(self):
    """There used to be a fifth ending: ROOT DECLARED DEFECTIVE, reached when a
    planner ran `--action=failed`. That action no longer leaves the machine, so
    the only thing that still sets `decomposition_failed` is `_record_disproof`,
    and a root carrying it without the `disproved` marker is not a state the
    system can reach. Falling through to STALLED is right: an orchestrator that
    somehow sees it should keep dispatching, not report a result.
    """
    graph = self.graph(
        _node("root", decomposition_failed=True, last_error="No such n exists.")
    )
    msg = claim_terminal.why_nothing_to_claim(graph, "root")
    self.assertNotIn("DECLARED DEFECTIVE", msg)
    self.assertIn("STALLED", msg)

  def test_only_the_kernel_can_end_a_run_without_a_proof(self):
    """The invariant behind the change: a run ends on an assembled proof or a
    checked refutation, never on an agent's prose."""
    self.assertNotIn("DECLARED DEFECTIVE", claim_terminal.__doc__ or "")
    for reason in ("planner", "defective", "hopeless", "ill-posed"):
      self.assertNotIn(
          reason,
          claim_terminal.why_nothing_to_claim(
              self.graph(_node("root", decomposition_failed=True)), "root"
          ),
      )

  def test_a_blocked_subtree_under_a_live_root_is_a_stall(self):
    """Root is fine; something below it is dead and unclaimable. This is the
    only case that is genuinely neither a result nor progress."""
    graph = self.graph(_node("root"))
    graph.add_proposed_subgoals(
        "root", [models.FormalSignature(name="kid", statement="True")]
    )
    msg = claim_terminal.why_nothing_to_claim(graph, "root")
    self.assertIn("STALLED", msg)
    self.assertIn("NOT finished", msg)

  def test_a_stall_names_the_blocked_nodes(self):
    graph = self.graph(_node("root"))
    graph.add_proposed_subgoals(
        "root", [models.FormalSignature(name="kid", statement="True")]
    )
    graph.nodes["kid"].search_metadata["decomposition_failed"] = True
    msg = claim_terminal.why_nothing_to_claim(graph, "root")
    self.assertIn("`kid`", msg)
    self.assertIn("`root`", msg)

  def test_a_stall_says_there_is_no_give_up(self):
    graph = self.graph(_node("root"))
    graph.add_proposed_subgoals(
        "root", [models.FormalSignature(name="kid", statement="True")]
    )
    self.assertIn(
        "exhaustion is not an outcome",
        claim_terminal.why_nothing_to_claim(graph, "root").lower(),
    )

  def test_a_long_stall_list_is_capped_and_says_how_many_it_dropped(self):
    graph = self.graph(_node("root"))
    kids = [f"kid{i}" for i in range(claim_terminal.MAX_LISTED + 4)]
    graph.add_proposed_subgoals(
        "root",
        [models.FormalSignature(name=k, statement="True") for k in kids],
    )
    for k in kids:
      graph.nodes[k].search_metadata["decomposition_failed"] = True
    msg = claim_terminal.why_nothing_to_claim(graph, "root")
    self.assertIn("+5 more", msg)

  def test_an_unknown_root_id_falls_back_to_the_first_node(self):
    """`root_id` is optional throughout the state manager, so this path is
    reachable without a corrupt graph."""
    graph = self.graph(_node("only", decomposition_failed=True))
    self.assertIn(
        "`only`", claim_terminal.why_nothing_to_claim(graph, "nonexistent")
    )

  def test_a_dead_interior_node_does_not_make_the_root_a_defect_report(self):
    """`decomposition_failed` is read off the root only. A dead child means the
    parent has replanning to do, which is a stall, not a verdict."""
    graph = self.graph(_node("root"))
    graph.add_proposed_subgoals(
        "root", [models.FormalSignature(name="kid", statement="True")]
    )
    graph.nodes["kid"].search_metadata["decomposition_failed"] = True
    msg = claim_terminal.why_nothing_to_claim(graph, "root")
    self.assertIn("STALLED", msg)
    self.assertNotIn("DEFECTIVE", msg)


class TerminalMessageEndToEndTest(absltest.TestCase):
  """That `claim_next` actually routes through the module."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.mgr = sm.ProofGraphStateManager(
        filepath=os.path.join(self.root, "proof_graph.json"),
        root_dir=self.root,
        root_id="root",
    )
    graph = dag.ProofGraph()
    graph.add_node(_node("root", statement="theorem root (a : Nat) : a + 0 = a"))
    self.mgr.initialize_graph(graph)

  def claim(self):
    return sm.run_state_manager(
        sm.StateManagerInput(
            action="claim_next", root_dir=self.root, root_id="root"
        )
    )

  def test_a_refuted_root_reports_a_verdict_through_the_cli(self):
    with self.mgr.lock_and_modify() as graph:
      graph.nodes["root"].search_metadata["disproved"] = True
      graph.nodes["root"].search_metadata["decomposition_failed"] = True
    out = self.claim()
    self.assertFalse(out.success)
    self.assertIn("ROOT DISPROVED", out.error_message)

  def test_the_old_undifferentiated_sentence_is_gone(self):
    """It said "dead-ended or proved" and made the orchestrator guess."""
    with self.mgr.lock_and_modify() as graph:
      graph.nodes["root"].search_metadata["decomposition_failed"] = True
    self.assertNotIn("dead-ended or proved", self.claim().error_message)


if __name__ == "__main__":
  absltest.main()
