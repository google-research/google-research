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

"""What `actions.declare` accepts, what it refuses, and what it writes.

`state_machine_test` walks the same function from the outside, one class per
phase, and asks where a node lands. This one asks the questions that are the
same in every phase and are therefore invisible there: is the refusal usable by
the agent that hit it, and -- the invariant the whole protocol rests on -- did a
refusal really write nothing?

  `success: true`   the claim was released and the turn is over
  `success: false`  nothing moved and the node is still held

An action that half-applies breaks both readings at once, so every refusal here
is checked against a snapshot of the node taken before the call.

The manager is real: `declare` reaches into it for the lock, the node log, the
reachability check and the claim, and a stub for those would be testing the
stub. Only the graph is a fixture, and it is the smallest one that has a
decomposition to retire.
"""

import _path  # noqa: F401  (puts leap/ on sys.path)
import os
import tempfile

from absl.testing import absltest

import actions
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


ROOT = "root_thm"
KID = "kid_lemma"
REASON = "The reverse direction needs a bound this plan never establishes."


class ActionsTestCase(absltest.TestCase):
  """A root with one child, plus a snapshot-and-compare for refusals."""

  def setUp(self):
    super().setUp()
    self.root_dir = tempfile.mkdtemp()
    graph = dag.ProofGraph()
    graph.add_node(
        dag.ProofGraphNode(
            id=ROOT,
            informal_problem=models.InformalProblem(
                statement="theorem root_thm (n : ℕ) : 0 ≤ n"
            ),
        )
    )
    graph.add_node(
        dag.ProofGraphNode(
            id=KID,
            informal_problem=models.InformalProblem(
                statement="lemma kid_lemma (n : ℕ) : 0 ≤ n"
            ),
        )
    )
    # KID hangs off ROOT from the start. A node nothing reaches and nothing has
    # touched is pruned the moment any action commits, so a fixture that leaves
    # it floating does not survive `initialize_graph`.
    graph.nodes[ROOT].and_nodes.append(
        dag.AndNode(
            id=f"{ROOT}_base",
            dependency_ids=[KID],
            subgoal_signatures=_sigs([KID]),
        )
    )
    graph.nodes[KID].reverse_dependency_ids.append(ROOT)
    self.make().initialize_graph(graph)

  def make(self):
    return sm.ProofGraphStateManager(
        filepath=os.path.join(self.root_dir, "proof_graph.json"),
        root_dir=self.root_dir,
        root_id=ROOT,
    )

  def load(self):
    return self.make().read_graph()

  def node(self, node_id=ROOT):
    return self.load().nodes[node_id]

  def meta(self, node_id=ROOT):
    return self.node(node_id).search_metadata

  def declare(self, action, reason=REASON, node_id=ROOT):
    return actions.declare(self.make(), node_id, action, reason)

  def set_phase(self, phase_name, node_id=ROOT):
    with self.make().lock_and_modify() as graph:
      graph.nodes[node_id].search_metadata["phase"] = phase_name

  def hold(self, node_id=ROOT):
    """Marks the node claimed, the way a dispatch would."""
    with self.make().lock_and_modify() as graph:
      graph.nodes[node_id].search_metadata["processing"] = True

  def decompose(self, parent=ROOT, children=(KID,)):
    """Hangs a live AND-node under `parent` without going through a submission."""
    with self.make().lock_and_modify() as graph:
      graph.nodes[parent].and_nodes.append(
          dag.AndNode(id=f"{parent}_d0", dependency_ids=list(children), subgoal_signatures=_sigs(list(children)))
      )

  def logs(self, node_id=ROOT):
    log_dir = self.make().workspace_mgr.get_logs_dir(node_id)
    return sorted(os.listdir(log_dir)) if os.path.isdir(log_dir) else []

  def snapshot(self, node_id=ROOT):
    """Everything a declaration could touch, as one comparable value."""
    node = self.node(node_id)
    return (
        dict(node.search_metadata),
        [(a.id, a.failed, a.error_message) for a in node.and_nodes],
        self.logs(node_id),
    )

  def assert_untouched(self, before, node_id=ROOT):
    self.assertEqual(self.snapshot(node_id), before, "a refusal wrote something")


class RefusalTest(ActionsTestCase):
  """Three ways a declaration is turned away, and the shared guarantee."""

  def test_an_action_the_phase_does_not_offer_is_refused(self):
    handled, msg = self.declare("approved")
    self.assertFalse(handled)
    self.assertIn("--action=approved", msg)

  def test_the_refusal_names_the_phase_the_node_is_actually_in(self):
    """The agent's mental model is wrong; the phase is the correction."""
    self.assertIn("'informal'", self.declare("approved")[1])

  def test_the_refusal_lists_what_the_phase_does_offer(self):
    msg = self.declare("approved")[1]
    for action in ("planned", "refute", "failed"):
      self.assertIn(f"--action={action}", msg)

  def test_the_refusal_says_the_claim_is_still_held(self):
    # Otherwise the agent concludes its turn on a node nobody will release.
    self.assertIn("claim is still held", self.declare("approved")[1])

  def test_an_action_from_no_phase_at_all_is_refused_the_same_way(self):
    handled, msg = self.declare("escalate")
    self.assertFalse(handled)
    self.assertIn("--action=escalate", msg)

  def test_submit_is_not_a_declaration(self):
    """It carries a file, and `record_lean` is the only thing that reads one."""
    self.set_phase("formal")
    handled, msg = self.declare(pipeline.ACTION_SUBMIT)
    self.assertFalse(handled)
    self.assertIn("--action=submit", msg)

  def test_an_empty_reason_is_refused(self):
    handled, msg = self.declare("planned", "")
    self.assertFalse(handled)
    self.assertIn("--reason", msg)

  def test_a_one_word_reason_is_refused_and_quoted_back(self):
    handled, msg = self.declare("planned", "failed")
    self.assertFalse(handled)
    self.assertIn("'failed' is not enough", msg)

  def test_the_boundary_is_the_word_count_and_nothing_else(self):
    self.assertFalse(self.declare("planned", "a b c")[0])
    self.assertTrue(self.declare("planned", "a b c d")[0])

  def test_whitespace_is_not_words(self):
    self.assertFalse(self.declare("planned", "   \n\t  ")[0])

  def test_the_action_is_checked_before_the_reason(self):
    """A good reason for an impossible action is still the wrong correction."""
    self.assertIn("is not something", self.declare("approved", "")[1])

  def test_a_refusal_writes_nothing(self):
    self.hold()
    before = self.snapshot()
    self.declare("approved")
    self.declare("planned", "no")
    self.assert_untouched(before)

  def test_a_refusal_leaves_the_node_where_it_was(self):
    self.declare("approved")
    self.assertEqual(pipeline.current_phase_name(self.node()), "informal")

  def test_a_refusal_leaves_the_claim_in_place(self):
    self.hold()
    self.declare("approved")
    self.assertTrue(self.meta()["processing"])

  def test_a_refusal_does_not_bump_the_revision(self):
    """A bump would invalidate an in-flight submission that is still good."""
    self.declare("planned", "no")
    self.assertEqual(self.meta().get("revision", 0), 0)


class RecordedTest(ActionsTestCase):
  """What a declaration writes when it is accepted."""

  def test_the_node_moves_along_the_edge_the_action_names(self):
    handled, _ = self.declare("planned")
    self.assertTrue(handled)
    self.assertEqual(pipeline.current_phase_name(self.node()), "formal")

  def test_the_reply_names_where_the_node_landed(self):
    # The agent picked an edge; it cannot see where the edge points.
    self.assertIn("phase 'formal'", self.declare("planned")[1])

  def test_the_reply_tells_the_agent_to_stop(self):
    self.assertIn("conclude your turn", self.declare("planned")[1])

  def test_the_claim_is_released(self):
    self.hold()
    self.declare("planned")
    self.assertFalse(self.meta().get("processing"))

  def test_the_reason_is_left_for_the_next_agent(self):
    self.declare("planned")
    self.assertEqual(self.meta()["last_error"], REASON)

  def test_the_reason_replaces_the_previous_one_rather_than_appending(self):
    """`last_error` briefs the *current* phase; a stale complaint re-briefs a
    fix that already happened."""
    self.declare("planned")
    self.declare("replan", "The formalization exposed a gap in step 2.")
    self.assertEqual(
        self.meta()["last_error"], "The formalization exposed a gap in step 2."
    )

  def test_the_revision_is_bumped(self):
    self.declare("planned")
    self.assertEqual(self.meta()["revision"], 1)

  def test_every_declaration_bumps_it_again(self):
    # The bump is what makes a submission that lands after an orchestrator
    # `release` get rejected as stale instead of applied to a moved node.
    self.declare("planned")
    self.declare("replan", "The formalization exposed a gap in step 2.")
    self.assertEqual(self.meta()["revision"], 2)

  def test_the_reason_is_written_to_the_node_log(self):
    self.declare("planned")
    logs = self.logs()
    self.assertLen(logs, 1)
    with open(
        os.path.join(self.make().workspace_mgr.get_logs_dir(ROOT), logs[0]),
        encoding="utf-8",
    ) as f:
      self.assertEqual(f.read(), REASON)

  def test_the_reason_is_filed_as_a_note_and_not_as_a_rejection(self):
    """11 of the 14 logs on `cyclotomic_val_gt_n` were the agent's own prose.

    The briefing hands the latest *rejection* to the next agent as "what the
    tool sent back on your last submission". Filing a planner's own account of
    what it did under that name makes the next agent try to fix it.
    """
    self.declare("planned")
    self.assertEqual([f.split("_")[-1] for f in self.logs()], ["note.log"])

  def test_it_does_not_become_the_last_rejection(self):
    self.declare("planned")
    self.assertEqual(
        self.make().workspace_mgr.get_latest_attempt_log_path(ROOT), ""
    )

  def test_a_real_rejection_is_still_found_past_a_later_note(self):
    # Notes are written after rejections in a turn, so filtering rather than
    # taking the newest file is what makes the briefing show the compiler.
    self.make().workspace_mgr.write_attempt_log(ROOT, "the kernel said no")
    self.declare("planned")
    path = self.make().workspace_mgr.get_latest_attempt_log_path(ROOT)
    with open(path, encoding="utf-8") as f:
      self.assertEqual(f.read(), "the kernel said no")

  def test_a_declaration_touches_only_the_node_it_names(self):
    self.declare("planned")
    # The live metrics are recalculated for the whole graph on every commit;
    # nothing the declaration itself writes may land on a sibling.
    for key in ("phase", "last_outcome", "revision", "processing"):
      self.assertNotIn(key, self.meta(KID))


class RouteClosedEdgeTest(ActionsTestCase):
  """`informal --action=failed`: a closed route, not a dead node.

  It used to be the one declaration that killed a node, and across both
  fmf_v1_009 rollouts every one of the 29 nodes it killed had never been
  attempted in Lean even once. The distinction agents were being asked to carry
  in prose -- "the route I planned is closed" versus "this target is
  unreachable" -- is not one an action set with a single exit can express, so
  they reported the first and the system recorded the second.
  """

  def test_the_node_is_not_marked_dead(self):
    handled, _ = self.declare("failed", "Every plan tried is circular.")
    self.assertTrue(handled)
    self.assertFalse(self.meta().get("decomposition_failed"))

  def test_it_lands_back_on_planning(self):
    self.declare("failed", "Every plan tried is circular.")
    self.assertEqual(pipeline.current_phase_name(self.node()), "informal")

  def test_the_reason_is_carried_to_the_next_planner(self):
    """The whole value of the edge: the next attempt starts from the finding."""
    self.declare("failed", "Every plan tried is circular.")
    self.assertEqual(self.meta()["last_error"], "Every plan tried is circular.")

  def test_the_reply_does_not_say_the_node_is_dead(self):
    msg = self.declare("failed", "Every plan tried is circular.")[1]
    self.assertNotIn("is dead", msg)
    self.assertIn("informal", msg)

  def test_the_node_stays_claimable(self):
    self.declare("failed", "Every plan tried is circular.")
    self.assertNotEqual(
        self.load().compute_statuses()[ROOT], dag.NodeStatus.FAILED
    )

  def test_no_other_phase_offers_it(self):
    for name, phase in pipeline.PIPELINE.items():
      if name == pipeline.ENTRY_PHASE:
        continue
      self.assertNotIn("failed", phase.actions, name)


class ReviewVerdictTest(ActionsTestCase):
  """The two actions with a side effect, both on the reviewer's row."""

  def setUp(self):
    super().setUp()
    self.set_phase(pipeline.REVIEW_PHASE)

  def branch(self):
    return self.node().and_nodes[0]

  # -- replan: retire the plan

  def test_replan_retires_the_live_branch(self):
    handled, _ = self.declare("replan")
    self.assertTrue(handled)
    self.assertTrue(self.branch().failed)

  def test_the_reason_is_kept_on_the_branch_where_a_re_planner_will_read_it(self):
    self.declare("replan")
    self.assertEqual(self.branch().error_message, f"RETIRED BY REVIEW: {REASON}")

  def test_the_node_goes_back_to_planning_and_stays_alive(self):
    self.declare("replan")
    self.assertEqual(pipeline.current_phase_name(self.node()), "informal")
    self.assertFalse(self.meta().get("decomposition_failed"))

  # -- replan: retire the header block the decomposition left behind

  def header(self, content, and_node_id=f"{ROOT}_base"):
    """Puts a block in the shared header the way a submission would."""
    with self.make().lock_and_modify() as graph:
      graph.environment.dynamic_headers.append(
          models.SharedHeaderBlock(
              content=content, node_id=ROOT, and_node_id=and_node_id
          )
      )

  def blocks(self):
    return self.load().environment.dynamic_headers

  def test_the_blocks_the_retired_plan_injected_are_retired_with_it(self):
    """A block lands in the header when its node submits, not when the plan is
    approved, so a retired plan leaves one behind in every node's environment
    for the rest of the run. fmf_v1_009/rollout_2 ended with 11 such lines and
    a duplicate instance nothing could see."""
    self.header("def orphan_helper : Nat := 1")
    self.declare("replan")
    self.assertTrue(self.blocks()[0].retired)
    self.assertNotIn(
        "orphan_helper", self.load().environment.to_lean_header()
    )

  def test_the_block_stays_in_the_json_for_forensics(self):
    self.header("def orphan_helper : Nat := 1")
    self.declare("replan")
    self.assertLen(self.blocks(), 1)
    self.assertIn("orphan_helper", self.blocks()[0].content)

  def test_a_block_something_else_still_references_is_left_alone(self):
    """The race the guard exists for. Review can fire hours after submission --
    patrol review fires on a decomposition that has been live all along -- so a
    sibling may already be verified against exactly these declarations."""
    self.header("def shared_vocab : Nat := 1")
    with self.make().lock_and_modify() as graph:
      graph.environment.dynamic_headers.append(
          models.SharedHeaderBlock(
              content="def built_on_it : Nat := shared_vocab", node_id=KID
          )
      )
    self.declare("replan")
    self.assertFalse(self.blocks()[0].retired)
    self.assertIn("shared_vocab", self.load().environment.to_lean_header())

  def test_the_plans_own_children_do_not_protect_its_block(self):
    """The case the whole mechanism exists for, and the one it used to miss.

    A definitions block is submitted so the plan's *own subgoals* can state
    their goals -- that is what the shared header is for -- so their statements
    name it essentially always. Counting them as references retired nothing,
    ever: `prune_orphans` runs at the end of the transaction, so at this point
    the children are still in `graph.nodes`. They are unreachable the moment
    `active.failed` is set, and unreachable is what decides it.
    """
    with self.make().lock_and_modify() as graph:
      graph.nodes[KID].informal_problem.statement = (
          "lemma kid_lemma : orphan_helper = 1"
      )
    self.header("def orphan_helper : Nat := 1")
    self.declare("replan")
    self.assertTrue(self.blocks()[0].retired)

  def test_an_instance_block_is_never_retired(self):
    """An instance is found by typeclass resolution, an '@[simp]' lemma by the
    simp set. Neither is ever written by name, so the name search cannot see
    the proof that depends on one and retiring it would break a node that never
    mentioned it."""
    self.header("instance : Inhabited Widget := ⟨default⟩")
    self.declare("replan")
    self.assertFalse(self.blocks()[0].retired)
    self.assertIn("Inhabited Widget", self.load().environment.to_lean_header())

  def test_an_attributed_definition_is_never_retired(self):
    self.header("@[simp]\ndef orphan_helper : Nat := 1")
    self.declare("replan")
    self.assertFalse(self.blocks()[0].retired)

  def test_an_unreachable_proved_bodys_reference_still_protects_a_block(self):
    """The asymmetry: a statement counts only while reachable, a proof body
    counts either way. `add_proposed_subgoals` re-adopts an existing node by
    name, so a later plan can pull this orphan back into the tree with its
    recorded proof intact -- citing a declaration that would be gone."""
    self.header("def shared_vocab : Nat := 1")
    with self.make().lock_and_modify() as graph:
      graph.nodes[KID].formal_theorem = models.FormalTheorem(
          signature=models.FormalSignature(name=KID, statement="True"),
          proof=models.FormalProof(tactics="by simp [shared_vocab]"),
      )
    self.declare("replan")
    self.assertFalse(self.load().nodes[KID].search_metadata.get("processing"))
    self.assertFalse(self.blocks()[0].retired)

  def test_a_block_a_proved_node_referenced_is_left_alone(self):
    self.header("def shared_vocab : Nat := 1")
    with self.make().lock_and_modify() as graph:
      graph.nodes[KID].formal_theorem = models.FormalTheorem(
          signature=models.FormalSignature(name=KID, statement="True"),
          proof=models.FormalProof(tactics="by simp [shared_vocab]"),
      )
    self.declare("replan")
    self.assertFalse(self.blocks()[0].retired)

  def test_another_decompositions_block_is_untouched(self):
    self.header("def mine : Nat := 1")
    self.header("def theirs : Nat := 2", and_node_id="some_other_and_node")
    self.declare("replan")
    self.assertTrue(self.blocks()[0].retired)
    self.assertFalse(self.blocks()[1].retired)

  def test_a_retired_block_releases_its_names(self):
    """Otherwise the re-planned branch collides with a declaration that is not
    in the header any more and cannot be read anywhere."""
    self.header("def orphan_helper : Nat := 1")
    self.declare("replan")
    registry = sm._existing_declared_names(self.load())  # pylint: disable=protected-access
    self.assertNotIn("orphan_helper", registry)

  def test_a_retired_plan_resolves_the_node_to_a_bare_statement(self):
    """Which is the re-plan state -- not FAILED, which would backtrack the
    parent over a plan the reviewer merely disliked."""
    self.declare("replan")
    self.assertEqual(
        self.load().compute_statuses()[ROOT], dag.NodeStatus.FORMAL_STATEMENT
    )

  def test_the_children_drop_out_of_the_reachable_set(self):
    self.declare("replan")
    self.assertEqual(self.load().get_active_reachable_nodes(ROOT), {ROOT})

  def test_a_verdict_with_no_live_branch_is_refused(self):
    with self.make().lock_and_modify() as graph:
      graph.nodes[ROOT].and_nodes[0].failed = True
    before = self.snapshot()
    handled, msg = self.declare("replan")
    self.assertFalse(handled)
    self.assertIn("no live decomposition", msg)
    self.assert_untouched(before)

  def test_that_refusal_is_addressed_to_the_orchestrator(self):
    """The reviewer cannot fix a node that should not have been dispatched."""
    with self.make().lock_and_modify() as graph:
      graph.nodes[ROOT].and_nodes[0].failed = True
    self.assertIn("report this to the orchestrator", self.declare("replan")[1])

  def test_only_the_newest_branch_is_retired(self):
    self.decompose(children=[KID])
    self.declare("replan")
    self.assertFalse(self.node().and_nodes[0].failed)
    self.assertTrue(self.node().and_nodes[1].failed)

  # -- approved: raise the bar

  def test_approval_raises_the_review_threshold(self):
    handled, _ = self.declare("approved", "The child closes the target.")
    self.assertTrue(handled)
    # The gate rules before any child has run, so the reading is 0.0 and the
    # fresh plan gets the full floor allowance rather than a raised bar.
    self.assertEqual(
        review_trigger.review_threshold(self.node()),
        float(pipeline.SUBTREE_REVIEW_REPLANS_PER_NODE),
    )

  def test_approval_marks_the_plan_it_approved(self):
    # A permanent exemption would let an approved branch grow without bound,
    # and an unmarked one would brief the next reviewer as a first look.
    self.declare("approved", "The child closes the target.")
    self.assertTrue(review_trigger.has_been_reviewed(self.node()))
    self.assertEqual(self.branch().review_replan_rate, 0.0)

  def test_approval_leaves_the_branch_alone(self):
    self.declare("approved", "The child closes the target.")
    self.assertFalse(self.branch().failed)

  def test_approval_parks_the_node_in_blocked(self):
    self.declare("approved", "The child closes the target.")
    self.assertEqual(pipeline.current_phase_name(self.node()), "blocked")

  def test_a_replan_marks_nothing(self):
    self.declare("replan")
    self.assertIsNone(self.node().and_nodes[0].review_replan_rate)
    self.assertFalse(review_trigger.has_been_reviewed(self.node()))

  def test_neither_verdict_is_offered_outside_review(self):
    self.set_phase("formal")
    self.assertFalse(self.declare("approved", "Looks fine to me here.")[0])


class OrphanTest(ActionsTestCase):
  """A node its parent abandoned while the agent was working on it."""

  def orphan(self):
    """Leaves KID dispatched and unreachable: the plan naming it was retired.

    The hold is what keeps it in the graph at all. An unreachable node nobody
    has touched is pruned; one a worker is holding carries `processing`, and
    that is the interlock that stops the prune deleting it mid-turn.
    """
    self.set_phase("informal", KID)
    self.hold(KID)
    with self.make().lock_and_modify() as graph:
      graph.nodes[ROOT].and_nodes[0].failed = True

  def test_the_claim_is_released_rather_than_the_action_refused(self):
    # `False` would send the agent back to retry against a node that no longer
    # matters, and it would retry forever.
    self.orphan()
    handled, msg = self.declare("planned", REASON, KID)
    self.assertTrue(handled)
    self.assertIn("no longer actively reachable", msg)

  def test_the_agent_is_told_its_work_is_obsolete(self):
    self.orphan()
    self.assertIn("obsolete", self.declare("planned", REASON, KID)[1])

  def test_the_action_is_not_applied(self):
    self.orphan()
    self.declare("planned", REASON, KID)
    self.assertEqual(pipeline.current_phase_name(self.node(KID)), "informal")

  def test_the_hold_is_dropped(self):
    self.orphan()
    self.declare("planned", REASON, KID)
    self.assertFalse(self.meta(KID).get("processing"))

  def test_the_node_stays_in_the_graph(self):
    """Unreachable is not deleted: a later decomposition may reuse it."""
    self.orphan()
    self.declare("planned", REASON, KID)
    self.assertIn(KID, self.load().nodes)

  def test_a_reachable_child_is_handled_normally(self):
    self.set_phase("informal", KID)
    self.declare("planned", REASON, KID)
    self.assertEqual(pipeline.current_phase_name(self.node(KID)), "formal")

  def test_reachability_is_checked_after_the_refusals(self):
    """An orphan reply ends the turn, so a malformed command on an orphan must
    not be answered with `success: true` -- the agent would never learn."""
    self.orphan()
    self.assertFalse(self.declare("approved", REASON, KID)[0])

  def test_the_root_is_never_an_orphan(self):
    self.assertTrue(self.declare("planned")[0])


class MissingNodeTest(ActionsTestCase):
  """An id that is not in the graph at all."""

  def test_the_turn_is_ended_rather_than_the_command_refused(self):
    handled, msg = self.declare("planned", REASON, "no_such_node")
    self.assertTrue(handled)
    self.assertIn("not in the proof graph", msg)

  def test_the_agent_is_told_to_report_it(self):
    # Nothing it can do differently would make the node exist.
    msg = self.declare("planned", REASON, "no_such_node")[1]
    self.assertIn("report this to the orchestrator", msg)

  def test_nothing_is_created(self):
    self.declare("planned", REASON, "no_such_node")
    self.assertNotIn("no_such_node", self.load().nodes)


class NodeReportTest(ActionsTestCase):
  """The read-only view: what the graph knows, and nothing that lives in a file."""

  def undecompose(self):
    """Retires the fixture's branch, leaving ROOT a bare statement again."""
    with self.make().lock_and_modify() as graph:
      graph.nodes[ROOT].and_nodes[0].failed = True

  def report(self, node_id=ROOT):
    ok, text = actions.node_report(self.make(), node_id)
    self.assertTrue(ok, text)
    return text

  def test_an_unknown_node_is_refused(self):
    ok, text = actions.node_report(self.make(), "no_such_node")
    self.assertFalse(ok)
    self.assertIn("not found", text)

  def test_the_report_names_the_node(self):
    self.assertIn(f"`{ROOT}`", self.report())

  def test_it_carries_the_status_and_the_phase(self):
    self.undecompose()
    text = self.report()
    self.assertIn("**Status**: formal_statement", text)
    self.assertIn("**Phase**: informal", text)

  def test_a_node_that_left_the_machine_says_so(self):
    # Reached by submitting a proof or a refutation, not by declaring anything.
    self.set_phase(None)
    self.assertIn("**Phase**: left the machine", self.report())

  def test_a_closed_route_still_reports_a_live_phase(self):
    self.declare("failed", "Every plan tried is circular.")
    self.assertIn("**Phase**: informal", self.report())

  def test_it_says_whether_a_worker_holds_the_node(self):
    self.assertIn("**Held by a worker**: False", self.report())
    self.hold()
    self.assertIn("**Held by a worker**: True", self.report())

  def test_an_undecomposed_node_reports_no_children(self):
    self.undecompose()
    self.assertIn("**Children**: none", self.report())

  def test_the_live_branchs_children_are_listed(self):
    self.assertIn(f"**Children**: `{KID}`", self.report())

  def test_a_retired_branchs_children_are_not(self):
    self.undecompose()
    self.assertIn("**Children**: none", self.report())

  def test_it_reports_how_many_times_the_node_has_been_dispatched(self):
    """Without this the orchestrator cannot see a node it is looping on.

    `cyclotomic_val_gt_n` was dispatched 14 times over 64 minutes and every
    message the orchestrator got back looked identical, so nothing in its own
    context distinguished progress from a livelock.
    """
    self.assertIn("**Dispatched / actions recorded**: 0 / 0", self.report())

  def test_the_counts_move_as_the_node_is_worked(self):
    self.undecompose()  # otherwise the scheduler descends to the child
    self.make().claim_next_subgoal()
    self.declare("planned")
    self.assertIn("**Dispatched / actions recorded**: 1 / 1", self.report())

  def test_the_statement_is_rendered_as_lean(self):
    text = self.report()
    self.assertIn("```lean", text)
    self.assertIn("theorem root_thm", text)

  def test_the_last_reason_is_shown(self):
    self.declare("planned")
    self.assertIn(REASON, self.report())

  def test_a_long_reason_is_truncated(self):
    self.declare("planned", "word " * 4000)
    self.assertLess(len(self.report()), 2500)

  def test_the_report_files_nothing_and_moves_nothing(self):
    self.hold()
    before = self.snapshot()
    self.report()
    self.assert_untouched(before)


if __name__ == "__main__":
  absltest.main()
