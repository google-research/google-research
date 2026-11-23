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

"""Tests for the rate-triggered patrol review.

Pure logic, no network, no Lean verifier and no model required.
"""

import _path  # noqa: F401  (puts leap/ on sys.path)
from absl.testing import absltest

import dag
import lean_code_parser as models
import pipeline
import review_trigger


FLOOR = float(pipeline.SUBTREE_REVIEW_REPLANS_PER_NODE)
BACKOFF = float(pipeline.SUBTREE_REVIEW_BACKOFF)


def _prove(node, tactics="by trivial"):
  """Gives `node` a direct proof, which the resolver reads as FORMAL_PROVED."""
  node.formal_theorem = models.FormalTheorem(
      signature=models.FormalSignature(name=node.id, statement="True"),
      proof=models.FormalProof(tactics=tactics),
  )


def _sketch(node, tactics="by exact c"):
  """Gives `node` a sketch but no theorem.

  This is the half of proved-ness `formal_theorem` cannot see: a node with a
  sketch whose decomposition closes resolves to FORMAL_PROVED, and the field
  stays None for the whole run.
  """
  node.formal_sketch = models.FormalProofSketch(
      signature=models.FormalSignature(name=node.id, statement="True"),
      proof=models.FormalProof(tactics=tactics),
  )


def _sigs(names):
  """Subgoal signatures for a hand-built AndNode.

  Nothing rejects an AndNode that names children without their statements any
  more, but a fixture that omits them is a legacy-format node and behaves like
  one, so build them here unless that is the point of the test.
  """
  return [
      models.FormalSignature(name=n, statement=f"Claim_{n}") for n in names
  ]


def _split(graph, parent, children, and_id=None):
  """Gives `parent` one more AND-node naming `children`. The live one."""
  graph.nodes[parent].and_nodes.append(
      dag.AndNode(
          id=and_id or f"a_{parent}_{len(graph.nodes[parent].and_nodes)}",
          dependency_ids=list(children),
          subgoal_signatures=_sigs(children),
      )
  )


def _graph(edges, dispatches=None):
  """A graph from `parent -> children`, one live AND-node per parent.

  Args:
    edges: Mapping of parent id to the ids of its children. Every id mentioned
      anywhere becomes a node.
    dispatches: Optional mapping of node id to `informal_dispatches`.

  Returns:
    A `dag.ProofGraph`.
  """
  graph = dag.ProofGraph()
  ids = set()
  for parent, children in edges.items():
    ids.add(parent)
    ids.update(children)
  for node_id in sorted(ids):
    graph.add_node(dag.ProofGraphNode(id=node_id))
  for parent, children in edges.items():
    _split(graph, parent, children)
  for node_id, count in (dispatches or {}).items():
    graph.nodes[node_id].search_metadata["informal_dispatches"] = count
  return graph


class ReplanRateTest(absltest.TestCase):

  def test_a_leaf_reads_zero(self):
    # No decomposition, so it has written no plan -- there is nothing for a
    # reviewer to rule on and nothing to hold it to.
    graph = _graph({}, {})
    graph.add_node(dag.ProofGraphNode(id="n0"))
    graph.nodes["n0"].search_metadata["informal_dispatches"] = 9
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 0.0)

  def test_the_node_itself_is_excluded(self):
    # A node's own re-plans are evidence about it, not about its plan.
    graph = _graph({"n0": ["n1"]}, {"n0": 10, "n1": 2})
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 2.0)

  def test_healthy_siblings_do_not_pay_for_a_hopeless_child(self):
    # This is the whole reason it is a maximum and not a mean. Averaged, these
    # four children read 2.0, under any threshold worth having, and the one
    # node that could retire `a` is never asked about it.
    graph = _graph({"n0": ["a", "b", "c", "d"]}, {"a": 8, "b": 0, "c": 0, "d": 0})
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 8.0)

  def test_a_child_is_read_through_its_whole_branch_not_just_itself(self):
    graph = _graph({"n0": ["n1"], "n1": ["n2"]}, {"n1": 3, "n2": 5})
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 4.0)

  def test_a_thrashing_child_reads_its_own_count_and_is_not_averaged_down(self):
    # A node that keeps being sent to a planner and keeps producing subgoals
    # that go nowhere would otherwise be divided by the children it proposed:
    # 20 over five nodes is 4.0, and the node doing the thrashing disappears
    # into its own output.
    graph = _graph(
        {"n0": ["hot"], "hot": ["k1", "k2", "k3", "k4"]}, {"hot": 20}
    )
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 20.0)

  def test_a_local_stall_dilutes_as_you_climb(self):
    # Dilution comes from inside a child's own branch, not from its siblings.
    # `mid` reads `hot`'s raw count because `hot` IS the stall; `root` reads a
    # mean over the five-node branch that contains it, so the top-down walk
    # stops at `mid`. If `mid` then approves `hot` anyway and `hot` keeps
    # climbing, `root`'s mean climbs behind it and the review escalates.
    graph = _graph(
        {"root": ["mid"], "mid": ["hot"], "hot": ["k1", "k2", "k3"]},
        {"hot": 10},
    )
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "mid"), 10.0)
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "root"), 2.0)

  def test_a_broad_stall_stays_high_at_the_root(self):
    graph = _graph(
        {"root": ["a", "b", "c", "d"]}, {"a": 10, "b": 10, "c": 10, "d": 10}
    )
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "root"), 10.0)

  def test_a_retired_branch_stops_counting_immediately(self):
    # The whole point of following `active_and_node`. If a retired epoch kept
    # counting, a `replan` could never relieve the pressure it created and the
    # ancestor would be pulled back forever.
    graph = _graph({"n0": ["n1"]}, {"n1": 20})
    graph.nodes["n0"].and_nodes[0].failed = True
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 0.0)

  def test_only_the_latest_live_and_node_counts(self):
    graph = _graph({"n0": ["dead_kid"]}, {"dead_kid": 20})
    graph.nodes["n0"].and_nodes[0].failed = True
    graph.add_node(dag.ProofGraphNode(id="fresh"))
    graph.nodes["fresh"].search_metadata["informal_dispatches"] = 1
    _split(graph, "n0", ["fresh"])
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 1.0)

  def test_a_node_with_no_dispatches_recorded_reads_zero_not_missing(self):
    # An agent that dies before recording anything still bumped the counter at
    # dispatch, so this only happens for a child nobody has claimed yet.
    graph = _graph({"n0": ["n1"]})
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 0.0)

  def test_a_missing_node_is_zero(self):
    self.assertEqual(
        review_trigger.worst_child_replan_rate(dag.ProofGraph(), "nope"), 0.0
    )

  def test_a_dangling_dependency_neither_raises_nor_counts(self):
    graph = dag.ProofGraph()
    for node_id in ("n0", "real"):
      graph.add_node(dag.ProofGraphNode(id=node_id))
    graph.nodes["real"].search_metadata["informal_dispatches"] = 6
    _split(graph, "n0", ["real", "ghost"])
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 6.0)

  def test_it_survives_a_cycle(self):
    # `add_proposed_subgoals` rejects edges that close a cycle, but this runs
    # against graphs loaded off disk and must not recurse forever on a bad file.
    graph = _graph({"n0": ["n1"]}, {"n0": 4, "n1": 4})
    _split(graph, "n1", ["n0"])
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 4.0)

  def test_a_diamond_is_not_double_counted_inside_one_branch(self):
    # Four distinct nodes in `mid`'s branch and nine re-plans, not five nodes
    # and twelve. `shared` hangs off both `left` and `right`.
    graph = _graph(
        {
            "root": ["mid"],
            "mid": ["left", "right"],
            "left": ["shared"],
            "right": ["shared"],
        },
        {"left": 3, "right": 3, "shared": 3},
    )
    self.assertAlmostEqual(
        review_trigger.worst_child_replan_rate(graph, "root"), 2.25
    )

  def test_the_memo_is_reusable_across_nodes(self):
    graph = _graph({"n0": ["n1"], "n1": ["n2"]}, {"n1": 3, "n2": 5})
    memo = {}
    self.assertEqual(
        review_trigger.worst_child_replan_rate(graph, "n1", memo), 5.0
    )
    self.assertEqual(
        review_trigger.worst_child_replan_rate(graph, "n0", memo), 4.0
    )


class ProvedChildTest(absltest.TestCase):
  """Proved-ness has to come from the resolver, not from `formal_theorem`."""

  def test_a_child_proved_directly_is_skipped(self):
    graph = _graph({"n0": ["done", "live"]}, {"done": 40, "live": 1})
    _prove(graph.nodes["done"])
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 1.0)

  def test_a_child_proved_by_its_own_decomposition_is_skipped(self):
    # `done` has no `formal_theorem` of its own -- it is closed by the child it
    # decomposed into. Testing the field alone would call it live and let its
    # 40 re-plans keep pulling `n0` back over finished work; on the run archive
    # that misread 301 of 1136 proved nodes.
    graph = _graph({"n0": ["done", "live"], "done": ["leaf"]},
                   {"done": 40, "live": 1})
    _sketch(graph.nodes["done"])
    _prove(graph.nodes["leaf"])
    self.assertIsNone(graph.nodes["done"].formal_theorem)
    self.assertEqual(
        graph.compute_statuses()["done"], dag.NodeStatus.FORMAL_PROVED
    )
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 1.0)

  def test_the_mean_inside_a_child_still_counts_proved_descendants(self):
    # They are the decay. Dropping them measures how hard the work that is LEFT
    # is, which reads higher the closer a branch gets to done -- and on the run
    # archive that lifted uninvolved ancestors above nodes that needed review.
    graph = _graph({"n0": ["live"], "live": ["a", "b", "c"]}, {"live": 4})
    for child in ("a", "b", "c"):
      _prove(graph.nodes[child])
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 4.0)

  def test_every_child_proved_reads_zero(self):
    graph = _graph({"n0": ["one", "two"]}, {"one": 40, "two": 40})
    _prove(graph.nodes["one"])
    _prove(graph.nodes["two"])
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 0.0)

  def test_the_callers_status_map_is_used_when_given(self):
    # `claim_next` resolves one already, and it is the map the surrounding walk
    # believes in -- `propagate_failures` retires AND-nodes with dead children
    # as a side effect, which an independent resolve here would undo.
    graph = _graph({"n0": ["done"]}, {"done": 40})
    _prove(graph.nodes["done"])
    self.assertEqual(review_trigger.worst_child_replan_rate(graph, "n0"), 0.0)
    self.assertEqual(
        review_trigger.worst_child_replan_rate(
            graph, "n0", None, {"done": dag.NodeStatus.FORMAL_SKETCHED}
        ),
        40.0,
    )


class ThresholdTest(absltest.TestCase):

  def test_a_plan_no_reviewer_has_ruled_on_sits_at_the_floor(self):
    graph = _graph({"n0": ["n1"]})
    self.assertEqual(review_trigger.review_threshold(graph.nodes["n0"]), FLOOR)

  def test_a_node_with_no_decomposition_at_all_sits_at_the_floor(self):
    self.assertEqual(
        review_trigger.review_threshold(dag.ProofGraphNode(id="n")), FLOOR
    )

  def test_the_quiet_an_approval_buys_scales_with_what_it_approved(self):
    # An increment would be swamped by a reading this large and the node would
    # be back in front of a reviewer after one more re-plan on one child.
    graph = _graph({"n0": ["n1"]})
    node = graph.nodes["n0"]
    node.active_and_node.review_replan_rate = 20.0
    self.assertEqual(review_trigger.review_threshold(node), 20.0 * BACKOFF)

  def test_it_never_drops_below_the_floor(self):
    graph = _graph({"n0": ["n1"]})
    node = graph.nodes["n0"]
    node.active_and_node.review_replan_rate = 0.5
    self.assertEqual(review_trigger.review_threshold(node), FLOOR)


class HasBeenReviewedTest(absltest.TestCase):
  """The briefing's commit-gate / patrol discriminator."""

  def test_an_unmarked_plan_reads_as_never_reviewed(self):
    graph = _graph({"n0": ["n1"]})
    self.assertFalse(review_trigger.has_been_reviewed(graph.nodes["n0"]))

  def test_a_zero_mark_still_reads_as_reviewed(self):
    # The gate approves before any child has run and so records 0.0. A
    # truthiness test here would call every second review a first one.
    graph = _graph({"n0": ["n1"]})
    node = graph.nodes["n0"]
    review_trigger.record_approval(graph, node)
    self.assertEqual(node.active_and_node.review_replan_rate, 0.0)
    self.assertTrue(review_trigger.has_been_reviewed(node))

  def test_a_replanned_node_is_at_a_commit_gate_again(self):
    # The mark is a judgement about a plan, not about a node. Once the plan it
    # judged is retired, the replacement gets briefed as the first look it is.
    graph = _graph({"n0": ["n1"]}, {"n1": 9})
    node = graph.nodes["n0"]
    review_trigger.record_approval(graph, node)
    node.and_nodes[0].failed = True
    graph.add_node(dag.ProofGraphNode(id="fresh"))
    _split(graph, "n0", ["fresh"])
    self.assertFalse(review_trigger.has_been_reviewed(node))

  def test_a_node_with_no_decomposition_has_not_been_reviewed(self):
    self.assertFalse(
        review_trigger.has_been_reviewed(dag.ProofGraphNode(id="n"))
    )


class RecordApprovalTest(absltest.TestCase):

  def test_it_marks_the_plan_with_the_reading_that_was_on_the_table(self):
    graph = _graph({"n0": ["n1"]}, {"n1": 7})
    node = graph.nodes["n0"]
    self.assertEqual(review_trigger.record_approval(graph, node), 7.0)
    self.assertEqual(node.active_and_node.review_replan_rate, 7.0)

  def test_it_marks_the_live_plan_and_not_a_retired_one(self):
    graph = _graph({"n0": ["dead"]}, {"dead": 7})
    node = graph.nodes["n0"]
    node.and_nodes[0].failed = True
    graph.add_node(dag.ProofGraphNode(id="fresh"))
    _split(graph, "n0", ["fresh"])
    review_trigger.record_approval(graph, node)
    self.assertIsNone(node.and_nodes[0].review_replan_rate)
    self.assertEqual(node.and_nodes[1].review_replan_rate, 0.0)

  def test_approving_a_node_with_no_plan_raises(self):
    # There is nowhere to put the mark, so the node would be pulled straight
    # back on the next scan. That is a livelock; refuse the action instead.
    graph = dag.ProofGraph()
    graph.add_node(dag.ProofGraphNode(id="n0"))
    with self.assertRaisesRegex(ValueError, "no live decomposition"):
      review_trigger.record_approval(graph, graph.nodes["n0"])

  def test_a_fresh_plan_still_gets_the_full_floor_allowance(self):
    graph = _graph({"n0": ["n1"]})
    node = graph.nodes["n0"]
    review_trigger.record_approval(graph, node)
    self.assertEqual(review_trigger.review_threshold(node), FLOOR)


class ShouldReviewTest(absltest.TestCase):

  def test_it_does_not_fire_below_the_floor(self):
    graph = _graph({"n0": ["n1"]}, {"n1": FLOOR - 1})
    self.assertFalse(review_trigger.should_review(graph, graph.nodes["n0"]))

  def test_it_fires_at_the_floor(self):
    graph = _graph({"n0": ["n1"]}, {"n1": FLOOR})
    self.assertTrue(review_trigger.should_review(graph, graph.nodes["n0"]))

  def test_an_approval_suppresses_it_until_the_reading_doubles(self):
    graph = _graph({"n0": ["n1"]}, {"n1": FLOOR})
    node = graph.nodes["n0"]
    review_trigger.record_approval(graph, node)
    self.assertFalse(review_trigger.should_review(graph, node))
    bar = FLOOR * BACKOFF
    graph.nodes["n1"].search_metadata["informal_dispatches"] = bar - 1
    self.assertFalse(review_trigger.should_review(graph, node))
    graph.nodes["n1"].search_metadata["informal_dispatches"] = bar
    self.assertTrue(review_trigger.should_review(graph, node))

  def test_retiring_the_branch_stops_it_firing(self):
    graph = _graph({"n0": ["n1"]}, {"n1": 50})
    graph.nodes["n0"].and_nodes[0].failed = True
    self.assertFalse(review_trigger.should_review(graph, graph.nodes["n0"]))

  def test_retiring_this_nodes_own_plan_puts_the_bar_back_at_the_floor(self):
    """The mark is a judgement about a plan and dies with it."""
    graph = _graph({"root": ["big"]}, {"big": 20})
    root = graph.nodes["root"]
    # A reviewer looks, approves, and the bar goes to 40.
    self.assertEqual(review_trigger.record_approval(graph, root), 20.0)
    self.assertEqual(review_trigger.review_threshold(root), 20.0 * BACKOFF)
    self.assertFalse(review_trigger.should_review(graph, root))

    # Later that plan is retired and the root writes a new one. The 20 re-plans
    # per node the mark was earned against are not in the live graph any more,
    # and neither is the mark.
    root.and_nodes[0].failed = True
    graph.add_node(dag.ProofGraphNode(id="fresh"))
    _split(graph, "root", ["fresh"])
    self.assertEqual(review_trigger.review_threshold(root), FLOOR)

    graph.nodes["fresh"].search_metadata["informal_dispatches"] = FLOOR
    self.assertTrue(review_trigger.should_review(graph, root))

  def test_the_bar_ratchets_down_when_the_subtree_shrinks_under_it(self):
    """The backoff. Without the ratchet this node is quiet for the whole run."""
    graph = _graph({"root": ["big"], "big": ["a", "b"]}, {"a": 30, "b": 30})
    root = graph.nodes["root"]
    # `big`'s branch is three nodes with 60 re-plans between them.
    self.assertEqual(review_trigger.record_approval(graph, root), 20.0)
    self.assertFalse(review_trigger.should_review(graph, root))

    # `big` retires its own split. The root's plan is untouched and still
    # marked, but what it is a judgement about has collapsed to one quiet node.
    graph.nodes["big"].and_nodes[0].failed = True
    self.assertEqual(
        review_trigger.worst_child_replan_rate(graph, "root"), 0.0
    )

    # Evaluating the predicate drags the mark down to what is true now, and
    # persists it -- comparing without storing would lower the number and
    # change nothing, since `rate * backoff` is a bar the rate can never reach.
    self.assertFalse(review_trigger.should_review(graph, root))
    self.assertEqual(root.active_and_node.review_replan_rate, 0.0)
    self.assertEqual(review_trigger.review_threshold(root), FLOOR)

    # So the root is answerable again after a normal amount of new evidence,
    # instead of needing the 40 the dead epoch had bought it.
    graph.nodes["big"].search_metadata["informal_dispatches"] = FLOOR
    self.assertTrue(review_trigger.should_review(graph, root))

  def test_the_ratchet_only_ever_lowers_the_mark(self):
    # It is called from a predicate on every scan, so an extra evaluation must
    # not be able to move the bar up -- otherwise the answer would depend on
    # how many times the scheduler happened to look.
    graph = _graph({"root": ["kid"]}, {"kid": 10})
    root = graph.nodes["root"]
    review_trigger.record_approval(graph, root)
    graph.nodes["kid"].search_metadata["informal_dispatches"] = 2
    review_trigger.should_review(graph, root)
    graph.nodes["kid"].search_metadata["informal_dispatches"] = 8
    review_trigger.should_review(graph, root)
    self.assertEqual(root.active_and_node.review_replan_rate, 2.0)

  def test_an_unreviewed_plan_is_not_marked_by_the_predicate(self):
    # `has_been_reviewed` is the briefing's discriminator; a predicate that
    # scans a node must not make it look like a reviewer has been here.
    graph = _graph({"root": ["kid"]}, {"kid": 1})
    root = graph.nodes["root"]
    review_trigger.should_review(graph, root)
    self.assertIsNone(root.active_and_node.review_replan_rate)
    self.assertFalse(review_trigger.has_been_reviewed(root))


class TableWiringTest(absltest.TestCase):

  def test_blocked_declares_the_predicate(self):
    blocked = pipeline.PIPELINE["blocked"]
    self.assertIn(("subtree_stalled", pipeline.REVIEW_PHASE), blocked.predicates)

  def test_resolution_predicates_are_checked_first(self):
    # A subtree that just finished must not be handed to a reviewer on its way
    # out, and the scheduler evaluates these in table order.
    names = [name for name, _ in pipeline.PIPELINE["blocked"].predicates]
    self.assertLess(names.index("children_done"), names.index("subtree_stalled"))
    self.assertLess(
        names.index("children_failed"), names.index("subtree_stalled")
    )

  def test_the_predicate_is_never_advertised_to_an_agent(self):
    for phase in pipeline.PIPELINE.values():
      self.assertNotIn("subtree_stalled", pipeline.allowed_actions(phase))


if __name__ == "__main__":
  absltest.main()
