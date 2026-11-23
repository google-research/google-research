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

"""Tests for the per-node phase table.

Pure logic, no network, no Lean verifier and no model required.
"""

import _path  # noqa: F401  (puts leap/ on sys.path)
import copy
import dataclasses
import unittest

from absl.testing import absltest

import pipeline


class _Node:
  """Minimal stand-in for a ProofGraphNode: just the metadata bag."""

  def __init__(self, phase=None):
    self.search_metadata = {} if phase is None else {"phase": phase}


def _table(**overrides):
  """A copy of PIPELINE with named rows replaced, for validator tests."""
  table = copy.deepcopy(pipeline.PIPELINE)
  table.update(overrides)
  return table


def _row(name, **kwargs):
  """A Phase with the boring fields filled in."""
  fields = dict(name=name, role="lemma_proof_worker", scaffold=None, sections=(), actions={})
  fields.update(kwargs)
  return pipeline.Phase(**fields)


class ShippedTableTest(unittest.TestCase):
  """The table we actually ship must satisfy its own validator."""

  def test_validates(self):
    pipeline.validate_pipeline()

  def test_entry_phase_exists(self):
    self.assertIn(pipeline.ENTRY_PHASE, pipeline.PIPELINE)

  def test_every_phase_is_frozen(self):
    # Mutating a row at runtime would make a node's history unreplayable.
    for phase in pipeline.PIPELINE.values():
      with self.assertRaises(dataclasses.FrozenInstanceError):
        phase.name = "mutated"

  def test_a_scaffold_phase_declares_what_submitting_it_can_mean(self):
    for name, phase in pipeline.PIPELINE.items():
      self.assertEqual(
          bool(phase.scaffold), bool(phase.results), f"{name} is half-wired"
      )

  def test_submit_is_never_declared_as_an_action(self):
    # It is implied by `scaffold`. Declaring it as well would give it two
    # targets: one in `actions` and one per entry in `results`.
    for name, phase in pipeline.PIPELINE.items():
      self.assertNotIn(pipeline.ACTION_SUBMIT, phase.actions, name)

  def test_every_dispatchable_phase_has_a_role(self):
    for name, phase in pipeline.PIPELINE.items():
      if pipeline.allowed_actions(phase):
        self.assertIsNotNone(phase.role, f"{name} has actions but no role")

  def test_the_only_undispatchable_phase_is_blocked(self):
    idle = {n for n, p in pipeline.PIPELINE.items() if p.role is None}
    self.assertEqual(idle, {"blocked"})

  def test_no_declared_action_can_end_a_nodes_life(self):
    # The rule the whole design rests on: prose is never a verdict. A declared
    # action is an agent's account of its own progress with no artifact behind
    # it, so none of them may leave the machine. Only a `results` edge can, and
    # a `results` edge is reached by submitting a file the kernel has checked.
    killers = {
        name
        for name, phase in pipeline.PIPELINE.items()
        if any(target is None for target in phase.actions.values())
    }
    self.assertEqual(killers, set())

  def test_a_node_can_still_leave_the_machine_by_submitting(self):
    # The other half: with no terminal action, `results` is the only way out,
    # and a run that cannot end is as broken as one that ends too easily.
    exits = {
        name
        for name, phase in pipeline.PIPELINE.items()
        if any(target is None for target in phase.results.values())
    }
    self.assertEqual(exits, {"formal", "disprove"})

  def test_the_six_agent_actions_are_the_whole_vocabulary(self):
    vocabulary = set()
    for phase in pipeline.PIPELINE.values():
      vocabulary.update(pipeline.allowed_actions(phase))
    self.assertEqual(
        vocabulary,
        {"submit", "planned", "refute", "failed", "replan", "approved"},
    )

  def test_predicates_are_never_offered_to_an_agent(self):
    for phase in pipeline.PIPELINE.values():
      offered = set(pipeline.allowed_actions(phase))
      for name, _ in phase.predicates:
        self.assertNotIn(name, offered)

  def test_results_are_never_offered_to_an_agent(self):
    for phase in pipeline.PIPELINE.values():
      offered = set(pipeline.allowed_actions(phase))
      for result in phase.results:
        self.assertNotIn(result, offered)


class RowContentTest(unittest.TestCase):
  """The specific edges the design settled on."""

  def test_informal_offers_exactly_three_actions_in_menu_order(self):
    phase = pipeline.PIPELINE["informal"]
    self.assertEqual(
        pipeline.allowed_actions(phase), ("planned", "refute", "failed")
    )

  def test_failed_is_not_the_first_thing_a_planner_is_shown(self):
    # Sorted order put "kill this node" at the top of the menu.
    self.assertNotEqual(
        pipeline.allowed_actions(pipeline.PIPELINE["informal"])[0], "failed"
    )

  def test_formal_leads_with_submit(self):
    self.assertEqual(
        pipeline.allowed_actions(pipeline.PIPELINE["formal"]),
        ("submit", "replan"),
    )

  def test_disprove_leads_with_submit(self):
    self.assertEqual(
        pipeline.allowed_actions(pipeline.PIPELINE["disprove"]),
        ("submit", "replan"),
    )

  def test_review_offers_exactly_approved_and_replan(self):
    self.assertEqual(
        pipeline.allowed_actions(pipeline.PIPELINE["review"]),
        ("approved", "replan"),
    )

  def test_blocked_offers_nothing(self):
    self.assertEqual(pipeline.allowed_actions(pipeline.PIPELINE["blocked"]), ())

  def test_one_submission_means_two_things_in_formal(self):
    self.assertEqual(
        pipeline.PIPELINE["formal"].results,
        {"proved": None, "decomposed": pipeline.REVIEW_PHASE},
    )

  def test_a_disproof_ends_the_node(self):
    self.assertEqual(pipeline.PIPELINE["disprove"].results, {"disproved": None})

  def test_every_stuck_phase_lands_back_in_informal(self):
    for name in ("formal", "disprove", "review"):
      self.assertEqual(
          pipeline.PIPELINE[name].actions["replan"], "informal", name
      )

  def test_blocked_carries_all_three_predicates(self):
    self.assertEqual(
        pipeline.PIPELINE["blocked"].predicates,
        (
            ("children_done", None),
            ("children_failed", "informal"),
            ("subtree_stalled", pipeline.REVIEW_PHASE),
        ),
    )

  def test_a_resolved_subtree_is_checked_before_a_stalled_one(self):
    # Order matters: a branch that just finished must never be sent to review.
    names = [name for name, _ in pipeline.PIPELINE["blocked"].predicates]
    self.assertLess(names.index("children_done"), names.index("subtree_stalled"))

  def test_approval_only_ever_leads_to_blocked(self):
    self.assertEqual(pipeline.PIPELINE["review"].actions["approved"], "blocked")


class GuidanceTest(unittest.TestCase):
  """Every action an agent can be shown must have words attached."""

  def test_every_offered_action_is_described(self):
    for name, phase in pipeline.PIPELINE.items():
      for action in pipeline.allowed_actions(phase):
        self.assertTrue(
            pipeline.action_guidance(phase, action), f"{name}.{action}"
        )

  def test_a_qualified_key_beats_the_bare_one(self):
    formal = pipeline.action_guidance(pipeline.PIPELINE["formal"], "replan")
    review = pipeline.action_guidance(pipeline.PIPELINE["review"], "replan")
    self.assertNotEqual(formal, review)

  def test_submit_means_different_things_in_the_two_scaffold_phases(self):
    self.assertNotEqual(
        pipeline.action_guidance(pipeline.PIPELINE["formal"], "submit"),
        pipeline.action_guidance(pipeline.PIPELINE["disprove"], "submit"),
    )

  def test_an_undescribed_action_is_the_empty_string(self):
    self.assertEqual(
        pipeline.action_guidance(pipeline.PIPELINE["formal"], "nonsense"), ""
    )

  def test_guidance_never_uses_a_double_hyphen(self):
    # It renders into agent-facing markdown, where `--` reads as a flag prefix.
    for key, text in pipeline.ACTION_GUIDANCE.items():
      self.assertNotIn(" -- ", text, key)


class ValidatorTest(unittest.TestCase):
  """The validator has to catch the mistakes that are otherwise silent."""

  def test_a_missing_entry_phase_is_refused(self):
    table = copy.deepcopy(pipeline.PIPELINE)
    del table[pipeline.ENTRY_PHASE]
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(table)

  def test_a_row_keyed_under_the_wrong_name_is_refused(self):
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(_table(review=_row("nope")))

  def test_an_edge_to_a_phase_that_does_not_exist_is_refused(self):
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(
          _table(review=_row("review", actions={"approved": "typo"}))
      )

  def test_a_declared_action_that_ends_a_node_is_refused(self):
    # The guard on the change that stopped `informal --action=failed` killing
    # nodes. Without it the edge is one keystroke from coming back, and the way
    # it comes back is silently: a phase gains a terminal action, agents reach
    # for the cheapest exit, and nothing downstream notices that 29 nodes died
    # on prose.
    with self.assertRaises(pipeline.InvalidPipelineError) as caught:
      pipeline.validate_pipeline(
          _table(review=_row("review", actions={"approved": None}))
      )
    self.assertIn("approved", str(caught.exception))

  def test_a_predicate_edge_to_nowhere_is_refused_too(self):
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(
          _table(
              blocked=_row(
                  "blocked", role=None, predicates=(("children_done", "typo"),)
              )
          )
      )

  def test_a_scaffold_without_results_is_refused(self):
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(
          _table(
              formal=_row(
                  "formal",
                  scaffold=pipeline.SCAFFOLD_PROVE,
                  actions={"replan": "informal"},
              )
          )
      )

  def test_results_without_a_scaffold_are_refused(self):
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(
          _table(
              review=_row(
                  "review",
                  actions={"approved": "blocked", "replan": "informal"},
                  results={"proved": None},
              )
          )
      )

  def test_declaring_submit_as_an_action_is_refused(self):
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(
          _table(
              review=_row(
                  "review",
                  actions={
                      pipeline.ACTION_SUBMIT: "blocked",
                      "replan": "informal",
                  },
              )
          )
      )

  def test_a_dispatchable_phase_with_no_way_out_is_refused(self):
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(_table(review=_row("review", actions={})))

  def test_a_section_number_that_is_not_a_section_is_refused(self):
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(
          _table(
              review=_row(
                  "review", sections=(99,), actions={"replan": "informal"}
              )
          )
      )

  def test_an_action_with_no_guidance_is_refused(self):
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(
          _table(
              review=_row("review", actions={"undescribed": "informal"})
          )
      )

  def test_a_phase_nothing_reaches_is_refused(self):
    table = _table(orphan=_row("orphan", actions={"replan": "informal"}))
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(table)

  def test_a_machine_with_no_exit_is_refused(self):
    # Every edge lands somewhere, so no node ever finishes.
    table = {
        "informal": _row("informal", actions={"planned": "informal"}),
    }
    with self.assertRaises(pipeline.InvalidPipelineError):
      pipeline.validate_pipeline(table)

  def test_the_error_names_the_offending_phase(self):
    with self.assertRaises(pipeline.InvalidPipelineError) as caught:
      pipeline.validate_pipeline(
          _table(review=_row("review", actions={"approved": "typo"}))
      )
    self.assertIn("review", str(caught.exception))


class LookupTest(unittest.TestCase):
  """Reading a node's phase back off its metadata."""

  def test_a_fresh_node_starts_at_the_entry_phase(self):
    self.assertEqual(pipeline.phase_for(_Node()).name, pipeline.ENTRY_PHASE)

  def test_an_unknown_phase_name_resolves_to_the_entry_phase(self):
    # A node revived by its parent backtracking re-enters at planning.
    self.assertEqual(
        pipeline.phase_for(_Node("gibberish")).name, pipeline.ENTRY_PHASE
    )

  def test_current_phase_name_tells_unstarted_from_finished(self):
    self.assertEqual(pipeline.current_phase_name(_Node()), "informal")
    node = _Node()
    node.search_metadata["phase"] = None
    self.assertIsNone(pipeline.current_phase_name(node))

  def test_role_follows_the_phase(self):
    self.assertEqual(pipeline.role_for(_Node("formal")), "lemma_proof_worker")
    self.assertEqual(pipeline.role_for(_Node("disprove")), "lemma_proof_worker")
    self.assertEqual(pipeline.role_for(_Node("review")), "lemma_decomposition_reviewer")
    self.assertEqual(
        pipeline.role_for(_Node("informal")), "lemma_informal_prover"
    )

  def test_blocked_has_no_role_so_it_is_never_claimed(self):
    self.assertIsNone(pipeline.role_for(_Node("blocked")))

  def test_scaffold_mode_is_the_single_source_for_both_readers(self):
    self.assertEqual(
        pipeline.scaffold_mode(pipeline.PIPELINE["formal"]),
        pipeline.SCAFFOLD_PROVE,
    )
    self.assertEqual(
        pipeline.scaffold_mode(pipeline.PIPELINE["disprove"]),
        pipeline.SCAFFOLD_DISPROVE,
    )
    self.assertIsNone(pipeline.scaffold_mode(pipeline.PIPELINE["review"]))


class FlagRuleTest(unittest.TestCase):
  """`submit` takes --code_file; everything else takes --reason. Never both."""

  def test_submit_takes_the_code_file(self):
    for name in ("formal", "disprove"):
      self.assertTrue(
          pipeline.takes_code_file(pipeline.PIPELINE[name], "submit"), name
      )

  def test_nothing_else_ever_does(self):
    for name, phase in pipeline.PIPELINE.items():
      for action in pipeline.allowed_actions(phase):
        if action != pipeline.ACTION_SUBMIT:
          self.assertFalse(
              pipeline.takes_code_file(phase, action), f"{name}.{action}"
          )

  def test_submit_in_a_phase_with_no_scaffold_takes_no_file(self):
    # Belt and braces: the word alone must not be enough.
    self.assertFalse(
        pipeline.takes_code_file(pipeline.PIPELINE["review"], "submit")
    )

  def test_exactly_one_action_per_phase_reads_a_file(self):
    for name, phase in pipeline.PIPELINE.items():
      readers = [
          a
          for a in pipeline.allowed_actions(phase)
          if pipeline.takes_code_file(phase, a)
      ]
      self.assertLessEqual(len(readers), 1, name)


class ApplyActionTest(unittest.TestCase):
  """Moving a node along an edge the agent named."""

  def test_planned_goes_to_formalization(self):
    node = _Node("informal")
    self.assertEqual(pipeline.apply_action(node, "planned"), "formal")
    self.assertEqual(node.search_metadata["phase"], "formal")

  def test_refute_goes_to_disproof(self):
    node = _Node("informal")
    self.assertEqual(pipeline.apply_action(node, "refute"), "disprove")

  def test_failed_returns_to_planning(self):
    # `failed` closes a route, not a node. The planner that reports it is the
    # planner the node comes back to, with its own reason in the briefing.
    node = _Node("informal")
    self.assertEqual(pipeline.apply_action(node, "failed"), "informal")
    self.assertEqual(node.search_metadata["phase"], "informal")

  def test_approval_blocks_the_parent(self):
    node = _Node("review")
    self.assertEqual(pipeline.apply_action(node, "approved"), "blocked")

  def test_review_replan_returns_to_planning(self):
    node = _Node("review")
    self.assertEqual(pipeline.apply_action(node, "replan"), "informal")

  def test_the_action_is_recorded_alongside_the_move(self):
    node = _Node("formal")
    pipeline.apply_action(node, "replan")
    self.assertEqual(node.search_metadata["last_action"], "replan")

  def test_an_action_this_phase_does_not_offer_is_refused(self):
    node = _Node("informal")
    with self.assertRaises(pipeline.UnknownActionError):
      pipeline.apply_action(node, "approved")

  def test_the_refusal_lists_what_is_allowed(self):
    node = _Node("informal")
    with self.assertRaises(pipeline.UnknownActionError) as caught:
      pipeline.apply_action(node, "approved")
    self.assertIn("planned", str(caught.exception))

  def test_a_refused_action_moves_nothing(self):
    node = _Node("informal")
    with self.assertRaises(pipeline.UnknownActionError):
      pipeline.apply_action(node, "approved")
    self.assertEqual(node.search_metadata["phase"], "informal")

  def test_submit_is_not_an_action(self):
    # It is a result, applied by the recorder after it reads the file.
    node = _Node("formal")
    with self.assertRaises(pipeline.UnknownActionError):
      pipeline.apply_action(node, "submit")

  def test_a_predicate_name_is_not_an_action_either(self):
    node = _Node("blocked")
    with self.assertRaises(pipeline.UnknownActionError):
      pipeline.apply_action(node, "children_done")


class ApplyResultTest(unittest.TestCase):
  """Moving a node along what its submitted file turned out to be."""

  def test_an_empty_subgoal_block_ends_the_node_proved(self):
    node = _Node("formal")
    self.assertIsNone(pipeline.apply_result(node, "proved"))

  def test_a_filled_subgoal_block_goes_to_review(self):
    node = _Node("formal")
    self.assertEqual(
        pipeline.apply_result(node, "decomposed"), pipeline.REVIEW_PHASE
    )

  def test_a_disproof_ends_the_node(self):
    node = _Node("disprove")
    self.assertIsNone(pipeline.apply_result(node, "disproved"))

  def test_a_phase_cannot_produce_a_result_it_does_not_declare(self):
    node = _Node("disprove")
    with self.assertRaises(pipeline.UnknownActionError):
      pipeline.apply_result(node, "proved")

  def test_a_phase_with_no_scaffold_can_produce_nothing(self):
    node = _Node("review")
    with self.assertRaises(pipeline.UnknownActionError):
      pipeline.apply_result(node, "decomposed")

  def test_an_agent_action_is_not_a_result(self):
    node = _Node("formal")
    with self.assertRaises(pipeline.UnknownActionError):
      pipeline.apply_result(node, "replan")

  def test_the_result_is_recorded_alongside_the_move(self):
    node = _Node("formal")
    pipeline.apply_result(node, "decomposed")
    self.assertEqual(node.search_metadata["last_action"], "decomposed")


class ApplyPredicateEdgeTest(unittest.TestCase):
  """The scheduler's edges. Unvalidated on purpose."""

  def test_children_done_ends_the_node(self):
    node = _Node("blocked")
    self.assertIsNone(pipeline.apply_predicate_edge(node, "children_done", None))
    self.assertIsNone(node.search_metadata["phase"])

  def test_children_failed_returns_to_planning(self):
    node = _Node("blocked")
    self.assertEqual(
        pipeline.apply_predicate_edge(node, "children_failed", "informal"),
        "informal",
    )

  def test_subtree_stalled_pulls_the_node_back_for_review(self):
    node = _Node("blocked")
    self.assertEqual(
        pipeline.apply_predicate_edge(
            node, "subtree_stalled", pipeline.REVIEW_PHASE
        ),
        pipeline.REVIEW_PHASE,
    )

  def test_it_does_not_check_the_current_phase(self):
    # The caller named the edge in its own source; there is nothing to guard.
    node = _Node("formal")
    self.assertEqual(
        pipeline.apply_predicate_edge(node, "children_done", "informal"),
        "informal",
    )


class TurnArtifactTest(unittest.TestCase):
  """What gets rotated out from under an agent when its node is re-claimed."""

  def dispatchable(self):
    return [p for p in pipeline.PIPELINE.values() if p.role]

  def test_every_dispatchable_phase_names_its_own_turn_log(self):
    # A phase that names nothing leaves the last agent's log in place for the
    # next one to append to, which is how a journal ends up describing two
    # different turns as if they were one.
    for phase in self.dispatchable():
      self.assertTrue(
          phase.turn_artifacts, f"{phase.name} rotates nothing away"
      )

  def test_no_phase_rotates_the_cross_turn_record(self):
    # `findings.md` and the blueprint are the node's memory across turns. If
    # provisioning rotated either, a re-claimed node would forget what it
    # already ruled out and re-derive it.
    for phase in self.dispatchable():
      for rel in phase.turn_artifacts:
        self.assertNotIn("findings", rel, phase.name)
        self.assertNotIn("blueprint.md", rel, phase.name)

  def test_turn_artifacts_are_node_relative_paths(self):
    # They are joined onto the node directory, so an absolute path or a
    # traversal would rotate a file belonging to some other node.
    for phase in self.dispatchable():
      for rel in phase.turn_artifacts:
        self.assertFalse(rel.startswith("/"), rel)
        self.assertNotIn("..", rel)

  def test_a_phase_with_no_role_rotates_nothing(self):
    for phase in pipeline.PIPELINE.values():
      if not phase.role:
        self.assertEqual(phase.turn_artifacts, (), phase.name)


class ReviewTuningTest(unittest.TestCase):
  """The two knobs the rate-triggered patrol review is built on."""

  def test_the_floor_allows_a_few_replans_before_anyone_is_pulled_back(self):
    # Under 2 and a branch that re-planned twice per node -- normal for a
    # split that needed one correction -- would already be under review.
    self.assertGreaterEqual(pipeline.SUBTREE_REVIEW_REPLANS_PER_NODE, 2)

  def test_the_floor_stays_low_enough_for_a_review_to_escalate(self):
    # A high floor silences the healthy rollouts, but it also stops a
    # grandparent from ever being asked about a child its own child keeps
    # approving. On the run archive, 12 pushed the first grandparent review out
    # to 229 planning attempts on the stalled node, or never.
    self.assertLessEqual(pipeline.SUBTREE_REVIEW_REPLANS_PER_NODE, 6)

  def test_approval_raises_the_bar_rather_than_removing_it(self):
    self.assertGreater(pipeline.SUBTREE_REVIEW_BACKOFF, 1.0)

  def test_the_backoff_scales_with_the_reading_it_backs_off_from(self):
    # The readings are unbounded -- a node re-planned 78 times reads 78 -- so
    # the bar has to be a multiple of what the reviewer saw and not a fixed
    # increment, which stops being a backoff at all once readings are large.
    mark = 78.0
    self.assertGreaterEqual(mark * pipeline.SUBTREE_REVIEW_BACKOFF, mark * 1.5)

  def test_review_phase_names_a_real_row(self):
    self.assertIn(pipeline.REVIEW_PHASE, pipeline.PIPELINE)


if __name__ == "__main__":
  absltest.main()
