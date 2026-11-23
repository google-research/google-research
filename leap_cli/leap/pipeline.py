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

"""The per-node workflow, declared as data.

One node moves through a small state machine: it plans, it formalizes, it may be
reviewed, it may be refuted. Which phase follows which is declared here and
nowhere else, so adding a role or rewiring an edge is a change to a dict literal
rather than a change to `claim_next_subgoal`.

What an agent says is an *action*, and it is the only thing it says. There is no
second flag naming an outcome: `--action=replan` is the whole message. A phase
that hands out a Lean scaffold also offers `--action=submit`, and what the
submission turns out to be is read off the file.

Three kinds of edge, in three fields, because three different things cause them:

  * `actions` -- the agent said so. Rendered into `context.md` verbatim, so a
    prompt never hand-copies the vocabulary.
  * `results` -- the submitted file said so. `formal` submits one scaffold that
    is a proof when its subgoal block is empty and a decomposition when it is
    not, and the agent classifies neither. Never shown to an agent.
  * `predicates` -- graph state said so, with no agent involved. A `blocked`
    parent returns to planning because every decomposition of it died.

They cannot share a dict. `blocked` has no role, so nothing exists to *declare*
`children_done`; and `allowed_actions` -- which tells an agent what it may run
-- would otherwise advertise scheduler vocabulary to a model.

Only `informal` can end a node's life. Every other phase that gets stuck says
`replan`, which hands the node back to the planner with a reason; the planner is
the one role that sees the accumulated history, so it is the one role trusted to
call a node dead. There is no attempt budget anywhere: a hard problem is
supposed to keep running.

This module is a leaf: it imports nothing from `dag`, `state_manager` or
`workspace_manager`, so it is importable from anywhere without a cycle and
testable without a model.
"""

import dataclasses
from typing import Dict, Optional, Tuple


# --------------------------------------------------------------------------
# Tunable parameters
# --------------------------------------------------------------------------

#: How much re-planning the WORST child of a `blocked` node's live
#: decomposition may absorb before the node is pulled back for review. Each
#: child contributes the larger of its own `informal_dispatches` and the mean
#: over its live branch; the node's reading is the maximum of those, skipping
#: children that are already proved. Follows only `active_and_node`, so a
#: retired branch stops counting the moment it is retired.
#:
#: A decomposition is only ever reviewed once, immediately after it is proposed,
#: when the reviewer can judge the shape but has no evidence of how it plays
#: out. This is the second entrance: a plan with a child that has been re-planned
#: four times over without closing is a plan whose author was guessing.
#:
#: Two choices are load-bearing here.
#:
#: Re-planning turns, not total effort. A node proved on its third Lean attempt
#: was worked three times and re-planned once; counting total dispatches would
#: put a large, healthy, productive branch over the line for being large. A
#: healthy branch sits near 1 -- every node gets planned once -- so anything
#: much above that is rework.
#:
#: The worst child, not the mean over the whole subtree. A mean lets healthy
#: siblings pay for a hopeless one: four children, three finishing and one
#: never closing, averages to something no threshold catches, and the one node
#: that could retire the hopeless child is never asked. Within a child the mean
#: is still what dilutes with height, so the top-down walk lands near a local
#: stall rather than always at the root.
#:
#: Left at 4 deliberately. On the run archive, raising it to 12 bought silence
#: in the seven rollouts with nothing stuck -- at 4 they produce six reviews in
#: total, each of which self-corrects on the next scan -- but it also pushed the
#: first grandparent review out from 37/150/171 accumulated planning attempts on
#: the stalled node to 229/518/never. The escalation up the ancestor chain is
#: the mechanism, not the noise.
SUBTREE_REVIEW_REPLANS_PER_NODE = 4

#: What an approved decomposition's reading must be MULTIPLIED by before the
#: node is pulled back again.
#: Approval is not permanent -- it is a verdict on the tree as it stood -- so the
#: node goes quiet rather than deaf: approved at a reading of 20, it returns at
#: 40. This is also the escalation, since an approved node steps aside and the
#: next review lands somewhere else -- deeper if the stall is local, higher if
#: it has spread, and eventually at the ancestor that can stop needing the
#: branch at all.
#:
#: Multiplicative and not an increment. This was `+2`, calibrated when readings
#: ran 0-2 and an increment of 2 was a doubling. Readings are now unbounded --
#: a node sent to a planner 78 times reads 78 -- and against those an increment
#: of 2 is not a backoff at all: on the run archive it put one node in front of
#: a reviewer between 30 and 285 times in a row, the bar climbing 2 at a time
#: behind a reading it could never catch. Doubling ends the same sequence in 5
#: to 7 reviews, at every floor tested.
#:
#: Deliberately NOT a high-water mark. review_trigger stores the reading the
#: reviewer saw on the AND-node it approved, ratchets it down whenever the
#: subtree reads lower, and takes the bar as that times this factor -- so the
#: bar comes down when the subtree does, and a `replan` takes it out of the
#: graph entirely along with the plan it was a verdict on. A mark that only
#: ratcheted up would keep a number earned by a child that has since been
#: retired, and a branch that ran long enough would end up permanently unable to
#: reach its own threshold -- which would silently remove the only mechanism by
#: which an ancestor ever abandons it. There is no ceiling constant because the
#: bar cannot exceed this factor times what is true today.
#:
#: Graph state and not a timer, for the reason the old comment gave: the graph
#: is persisted JSON that gets resumed and replayed, and a wall-clock stamp
#: would make the same run replay differently. `informal_dispatches` is
#: persisted graph state and replays identically; a subtree nobody has
#: re-planned has nothing new for a reviewer to look at.
SUBTREE_REVIEW_BACKOFF = 2.0


# --------------------------------------------------------------------------
# context.md sections
# --------------------------------------------------------------------------

# Canonical numbering for the briefing. A phase declares which of these it sees,
# which is what makes adding a role a one-row change instead of another
# `if role == ...` branch in the renderer.
SECTIONS: Dict[int, str] = {
    1: "Global Problem Specification",
    2: "Active Target Goal",
    3: "Parent Plan & Upstream Ancestry",
    4: "Dynamic Peer Discovery & Role Guidelines",
    5: "Prior Attempt Diagnostics & History",
    6: "Prior Decomposition History & Backtracking Feedback",
    7: "Established Findings",
    8: "Plan of Record",
    9: "Subtree Ledger",
    10: "Reusable Proved Lemmas",
}


# --------------------------------------------------------------------------
# Scaffold modes
# --------------------------------------------------------------------------

# Which Lean scaffold a phase is handed. Distinct from the subagent role:
# `formal` and `disprove` are both worked by the prover but need different
# files, so the two must not be inferred from one string.
SCAFFOLD_PROVE = "prove"
SCAFFOLD_DISPROVE = "disprove"


# --------------------------------------------------------------------------
# Action vocabulary
# --------------------------------------------------------------------------

#: The one action that reads a file. Implied by `scaffold`, never listed in a
#: phase's `actions`: a phase hands out a scaffold or it does not, and offering
#: `submit` without one -- or a scaffold nobody can submit -- is a table bug.
ACTION_SUBMIT = "submit"

# What each action means, in the words an agent is shown. Keyed by the bare
# action name, or by "<phase>.<action>" where the same word means something
# materially different depending on who runs it.
#
# Written here rather than in the prompts so that rewiring an edge cannot leave
# a stale description behind: `validate_pipeline` refuses a table with an action
# nothing describes.
ACTION_GUIDANCE: Dict[str, str] = {
    "planned": (
        "You have written a complete, rigorous plan to `scratch/blueprint.md`"
        " — every case discharged, nothing hand-waved. The node moves to"
        " formalization and a prover works from that file."
    ),
    "refute": (
        "You have concrete evidence the target is FALSE — a specific witness,"
        " not a suspicion. Name the witness in `--reason`. The node moves to"
        " disproof, where the next agent must derive `False` from the target"
        " taken as an axiom."
    ),
    "failed": (
        "The route you planned is closed: you established that this particular"
        " approach cannot reach the target. This does NOT kill the node and it"
        " is NOT a verdict on the target — the target may well still be true,"
        " and saying so is the point. The node comes straight back to planning"
        " with your reason attached, so `--reason` must say what you ruled out"
        " and what it leaves open, in enough detail that the next planner does"
        " not walk the same route. Nothing you can do ends a node except"
        " `refute`, which needs a witness and a kernel-checked derivation of"
        " `False`."
    ),
    "formal.submit": (
        "Hand over `proof_scaffold.lean`. Leave `-- [START_SUBGOALS]` empty to"
        " prove the target outright; put subgoals there, each ending `:= sorry`,"
        " to split it instead. You do not say which of the two you did — the"
        " file does."
    ),
    "formal.replan": (
        "You cannot get there from the plan of record: it is wrong, or"
        " unprovable as written, or the statement itself is defective. The node"
        " goes back to planning with your reason attached, so name the"
        " offending step or the missing hypothesis. If you believe the target"
        " is actually FALSE, say that here too — the planner routes it to"
        " disproof."
    ),
    "disprove.submit": (
        "Hand over `disproof_scaffold.lean`, deriving `False` from the target"
        " taken as an axiom. This is a terminal verdict: the node is dead and"
        " its parent backtracks."
    ),
    "disprove.replan": (
        "You are abandoning the refutation, and which of the two cases this is"
        " decides what happens next. Say either that the counterexample is real"
        " but you could not formalize it — the node re-plans expecting it to be"
        " false — or that the counterexample was wrong, which is positive"
        " evidence *for* the target and stops the next agent re-walking your"
        " dead end."
    ),
    "approved": (
        "The decomposition is sound: the children close the target, none of them"
        " restates an ancestor, and each is a genuinely easier problem. They are"
        " dispatched."
    ),
    "review.replan": (
        "The decomposition is unsound, circular, or has drifted from the plan."
        " The branch is retired and the node re-plans from scratch. This throws"
        " away work its author believed in, so say exactly what is wrong."
    ),
}


def action_guidance(phase, action):
  """Returns the agent-facing description of an action in a given phase."""
  return ACTION_GUIDANCE.get(
      f"{phase.name}.{action}", ACTION_GUIDANCE.get(action, "")
  )


#: The phase a reviewer holds. Named because the scheduler must not walk past a
#: node under review into children the reviewer may be about to retire.
REVIEW_PHASE = "review"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class PipelineError(Exception):
  """Base class for pipeline faults."""


class InvalidPipelineError(PipelineError):
  """The table itself is malformed. Raised at import, never at runtime."""


class UnknownActionError(PipelineError):
  """An agent ran an action its current phase does not offer."""


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Phase:
  """One state of the per-node workflow.

  Attributes:
    name: Must equal this row's key in PIPELINE.
    role: Which subagent gets dispatched, by its registered name. None means
      the phase is not dispatchable -- the node sits here until a predicate
      moves it.
    scaffold: Which Lean scaffold to render, or None for phases that write no
      Lean. Non-None is exactly what makes `submit` available.
    sections: Which `SECTIONS` of `context.md` this phase is shown.
    actions: Agent-run action -> next phase. None as a target means the node
      leaves the machine; the verdict itself is set by the finalizer, not
      encoded here. Never contains `submit`.
    results: What a submitted scaffold can turn out to be -> next phase. Empty
      for a phase with no scaffold. Never shown to an agent.
    predicates: Graph-state edges, same target convention. Never shown to an
      agent.
    turn_artifacts: Files under the node directory that belong to one turn of
      this phase's role and are rotated out when the node is re-claimed. Not
      the cross-turn record -- `findings.md` and `scratch/blueprint.md` are
      deliberately absent and must stay absent.
  """

  name: str
  role: Optional[str]
  scaffold: Optional[str]
  sections: Tuple[int, Ellipsis]
  actions: Dict[str, Optional[str]]
  results: Dict[str, Optional[str]] = dataclasses.field(default_factory=dict)
  predicates: Tuple[Tuple[str, Optional[str]], Ellipsis] = ()
  turn_artifacts: Tuple[str, Ellipsis] = ()


#: Where a node with no recorded phase begins. Uniform -- there is no fast path
#: past planning, because seeing the ancestry is what stops a fourth spelling of
#: the same lemma from being proposed.
ENTRY_PHASE = "informal"

PIPELINE: Dict[str, Phase] = {
    "informal": Phase(
        name="informal",
        role="lemma_informal_prover",
        scaffold=None,
        # 5 is here because this is one of the two roles that decides whether a
        # node is worth continuing, and it is the only section that says whether
        # the node has ever been attempted in Lean at all. Across both
        # fmf_v1_009 rollouts, all 18 nodes ever attempted directly ended proved
        # and all 29 nodes declared dead had never been attempted once -- a
        # judgment made without the one fact that predicted the outcome.
        sections=(1, 2, 3, 5, 6, 7, 8, 9, 10),
        # No terminal action. `failed` used to end a node's life here; it now
        # lands back on `informal` with the reason attached, because the
        # distinction it was being asked to carry -- "my route is closed" versus
        # "this target is unreachable" -- is one the action set could not
        # express, so agents used it for the first and the system heard the
        # second. A target is only ever settled by `refute`, which needs a
        # witness and a kernel derivation of `False`. Nothing else is a verdict.
        actions={
            "planned": "formal",
            "refute": "disprove",
            "failed": "informal",
        },
        turn_artifacts=("scratch/informal_log.md",),
    ),
    "formal": Phase(
        name="formal",
        role="lemma_proof_worker",
        scaffold=SCAFFOLD_PROVE,
        # 6 is here as well as on `informal`: the collision gate rejects an
        # exact repeat of a dead subgoal set, and it fires on *this* phase's
        # submission. Withholding the list means rejecting an agent under a rule
        # it was never shown.
        sections=(1, 2, 3, 4, 5, 6, 7, 8, 10),
        actions={"replan": "informal"},
        turn_artifacts=(
            "scratch/prover_log.md",
            "scratch/prover_notes.md",
            "scratch/blueprint_fidelity_audit.md",
        ),
        # One file, two meanings, told apart by whether its subgoal block is
        # empty. Deriving rather than accepting a claim is what makes it
        # impossible to say "here is my proof" and "call it proved" in one
        # breath -- the bug that discarded a real proof with a success message.
        results={"proved": None, "decomposed": REVIEW_PHASE},
    ),
    "disprove": Phase(
        name="disprove",
        role="lemma_proof_worker",
        scaffold=SCAFFOLD_DISPROVE,
        # 4 carries the refutation framing -- "you are refuting this, not
        # proving it", plus the witness that routed the node here -- and the
        # workspace rules. Without it a disproof agent is briefed as if it were
        # proving the target it is meant to refute. 5 is here for the retry:
        # deriving `False` usually takes more than one compile, and without it
        # the agent cannot see why the last one was rejected.
        sections=(1, 2, 3, 4, 5, 7, 8, 10),
        actions={"replan": "informal"},
        turn_artifacts=(
            "scratch/prover_log.md",
            "scratch/prover_notes.md",
            "scratch/blueprint_fidelity_audit.md",
        ),
        results={"disproved": None},
    ),
    "review": Phase(
        name=REVIEW_PHASE,
        role="lemma_decomposition_reviewer",
        scaffold=None,
        # Section 3 is here for circularity. A child that paraphrases its own
        # parent is caught by comparing the two statements in section 2, but a
        # child that restates a *grandparent* is the search going in circles,
        # and nothing else on the page shows the chain. Reviewing without it
        # means approving the fourth spelling of a lemma three levels up.
        # 5 for the same reason `informal` gets it: `replan` retires a branch,
        # and whether the node under review has ever compiled is the cheapest
        # available evidence about whether its split is worth keeping. A
        # reviewer without it rules on shape alone.
        # 6 is what the *previous* reviewers of this node said. Reviewers do
        # not share state and each one is a fresh model, so without the
        # retired decompositions and their blockers on the page, the fourth
        # reviewer of a node rules exactly like the first and is free to raise
        # a fourth unrelated objection. That is a ratchet, not a gate: in
        # fmf_v1_002 the root took four reviews, each finding something
        # different, and never dispatched a child.
        sections=(2, 3, 5, 6, 8, 9, 10),
        actions={"approved": "blocked", "replan": "informal"},
        turn_artifacts=("scratch/reviewer_log.md",),
    ),
    # Decomposed, children in flight, nothing to do here. Not idle -- waiting.
    # It exists to give `review` an honest place to return to: an approved
    # parent is neither done nor ready to re-plan.
    "blocked": Phase(
        name="blocked",
        role=None,
        scaffold=None,
        sections=(),
        actions={},
        # All three are arithmetic on the graph, evaluated by the scheduler as
        # it walks: `children_done` when the subtree resolved the parent,
        # `children_failed` when every decomposition of it died, and
        # `subtree_stalled` when the branch is being re-planned more than
        # SUBTREE_REVIEW_REPLANS_PER_NODE times per node without closing. No
        # model call decides any of them, and none of them kills the node -- a
        # node with no live decomposition left simply goes back to the planner.
        #
        # Order matters: a resolved subtree is checked before a stalled one, so
        # a branch that just finished is never sent back to a reviewer.
        predicates=(
            ("children_done", None),
            ("children_failed", "informal"),
            ("subtree_stalled", REVIEW_PHASE),
        ),
    ),
}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _edge_targets(phase):
  """Returns every phase this one can move to, by any of the three edge kinds."""
  return (
      tuple(phase.actions.values())
      + tuple(phase.results.values())
      + tuple(t for _, t in phase.predicates)
  )


def validate_pipeline(table = None):
  """Checks the table's shape. Called at import, so a bad edit fails at startup.

  Only the mistakes that are otherwise *silent*. A typo'd scaffold mode already
  raises from `scaffold_filename`, so it is not checked here; a typo'd edge
  target, section number or action name produces no error at all -- a node
  `claim_next` skips forever, a briefing missing a section, or an action offered
  to an agent with no explanation of what it means.

  Args:
    table: The table to check. Defaults to the module-level PIPELINE.

  Raises:
    InvalidPipelineError: With a message naming the offending phase.
  """
  table = PIPELINE if table is None else table
  if ENTRY_PHASE not in table:
    raise InvalidPipelineError(f"entry phase {ENTRY_PHASE!r} is not in the table")

  for key, phase in table.items():
    if key != phase.name:
      raise InvalidPipelineError(f"phase keyed {key!r} but named {phase.name!r}")
    for target in _edge_targets(phase):
      if target is not None and target not in table:
        raise InvalidPipelineError(f"{key}: edge target {target!r} is not a phase")
    # Only a submitted artifact may end a node, so only `results` may point at
    # None. A declared action is a claim about the agent's own progress with
    # nothing behind it but prose, and the one time the table let such an action
    # be terminal (`informal --action=failed`) agents used it for "my route is
    # closed" and the system read "this target is unreachable" -- 29 nodes
    # killed across the fmf_v1_009 rollouts, none of them ever attempted.
    for action, target in phase.actions.items():
      if target is None:
        raise InvalidPipelineError(
            f"{key}: action {action!r} points at None. Declared actions may not"
            " end a node -- put a terminal edge in `results`, which is reached"
            " only by submitting an artifact the kernel has checked"
        )
    # `submit` is offered because a scaffold exists, so the two must agree.
    # Half of this pair alone is either an action an agent cannot perform or a
    # file nobody can hand in.
    if bool(phase.scaffold) != bool(phase.results):
      raise InvalidPipelineError(
          f"{key}: scaffold is {phase.scaffold!r} but results are"
          f" {sorted(phase.results)} -- a phase hands out a scaffold and"
          " declares what submitting it can mean, or neither"
      )
    if ACTION_SUBMIT in phase.actions:
      raise InvalidPipelineError(
          f"{key}: {ACTION_SUBMIT!r} is implied by `scaffold` and must not be"
          " listed in `actions`; its targets belong in `results`"
      )
    if phase.role is not None and not allowed_actions(phase):
      raise InvalidPipelineError(
          f"{key}: is dispatchable but offers no actions, so a node entering it"
          " can never leave"
      )
    # The renderer guards each block on `N in show`, so a number that is not a
    # section is not an error there -- the block simply never renders and the
    # agent is briefed with a hole it cannot know about.
    for section in phase.sections:
      if section not in SECTIONS:
        raise InvalidPipelineError(
            f"{key}: {section} is not a context section; known:"
            f" {sorted(SECTIONS)}"
        )
    # Without this, a new edge reaches `context.md` as a bare action name with
    # an empty description, which is how an agent is offered a command nobody
    # ever told it when to run.
    for action in allowed_actions(phase):
      if not action_guidance(phase, action):
        raise InvalidPipelineError(
            f"{key}: action {action!r} has no ACTION_GUIDANCE entry, so it"
            " would be advertised to an agent with no meaning"
        )

  reachable, frontier, terminates = {ENTRY_PHASE}, [ENTRY_PHASE], False
  while frontier:
    for target in _edge_targets(table[frontier.pop()]):
      if target is None:
        terminates = True
      elif target not in reachable:
        reachable.add(target)
        frontier.append(target)

  stranded = set(table) - reachable
  if stranded:
    raise InvalidPipelineError(f"{sorted(stranded)} unreachable from {ENTRY_PHASE!r}")
  if not terminates:
    raise InvalidPipelineError(
        f"no path from {ENTRY_PHASE!r} leaves the machine; every node would"
        " cycle forever"
    )


# --------------------------------------------------------------------------
# Runtime lookups
# --------------------------------------------------------------------------


def get_phase(name):
  """Resolves a phase name, falling back to the entry phase.

  An unrecognised or missing name is not an error: a node that has left the
  machine, or one revived by its parent backtracking, re-enters at `informal`.
  """
  if name and name in PIPELINE:
    return PIPELINE[name]
  return PIPELINE[ENTRY_PHASE]


def phase_for(node):
  """Returns the phase of a node.

  Args:
    node: Anything carrying a `search_metadata` mapping. Duck-typed so this
      module stays a leaf and does not import `dag`.
  """
  return get_phase(node.search_metadata.get("phase"))


def current_phase_name(node):
  """Returns the node's phase name, or None if it has left the machine.

  `phase_for` cannot answer this: it resolves a missing phase to the entry
  phase, which is right for a node that has not started and wrong for one that
  finished. The two are told apart by whether the key is present at all --
  terminal edges write `phase = None` rather than deleting it.
  """
  meta = node.search_metadata
  if "phase" in meta and meta["phase"] is None:
    return None
  return phase_for(node).name


def role_for(node):
  """Returns the subagent role for a node's current phase."""
  return phase_for(node).role


def scaffold_mode(phase):
  """Returns the scaffold a phase is handed, or None if it writes no Lean.

  The single source for both `get_scaffold_path` and `render_scaffold_lean`. If
  those two ever disagree the agent gets "compiled locally, rejected on submit".
  """
  return phase.scaffold


def allowed_actions(phase):
  """Returns every `--action` an agent in this phase may run, in menu order.

  The complete agent-facing vocabulary, and the only thing a briefing renders.
  `results` and `predicates` are excluded: an agent cannot cause `decomposed` or
  `children_done`, and should not learn that either string exists.

  Declaration order, not sorted. `submit` leads because doing the work is the
  expected outcome, and the rest follow the row above; sorting put `failed` --
  the one action that kills a node -- at the top of the planner's menu.
  """
  submit = (ACTION_SUBMIT,) if phase.scaffold else ()
  return submit + tuple(phase.actions)


def takes_code_file(phase, action):
  """Whether this action carries `--code_file` instead of `--reason`.

  Exactly one action does, in exactly the phases that hand out a scaffold. This
  is the whole flag rule an agent has to remember.
  """
  return action == ACTION_SUBMIT and bool(phase.scaffold)


def _move(node, name, target):
  """Writes the phase. The only place `search_metadata["phase"]` is assigned."""
  node.search_metadata["phase"] = target
  node.search_metadata["last_action"] = name
  return target


def apply_action(node, action):
  """Moves a node along an action the agent ran.

  Args:
    node: Anything carrying a mutable `search_metadata` mapping.
    action: The `--action` value, which must not be `submit` -- what a
      submission turned out to be is applied with `apply_result`.

  Returns:
    The name of the next phase, or None if the node leaves the machine. The
    terminal verdict itself -- proved, disproved, failed -- is the finalizer's
    job, not the table's.

  Raises:
    UnknownActionError: If the current phase does not offer this action.
  """
  phase = phase_for(node)
  if action not in phase.actions:
    raise UnknownActionError(
        f"phase {phase.name!r} does not offer action {action!r}; allowed:"
        f" {list(allowed_actions(phase))}"
    )
  return _move(node, action, phase.actions[action])


def apply_result(node, result):
  """Moves a node along what its submitted scaffold turned out to be.

  Separate from `apply_action` because no agent may name these: the caller is
  the recorder, and it has just read the file and verified it with the kernel.

  Raises:
    UnknownActionError: If the current phase cannot produce this result.
  """
  phase = phase_for(node)
  if result not in phase.results:
    raise UnknownActionError(
        f"phase {phase.name!r} cannot produce result {result!r}; known:"
        f" {sorted(phase.results)}"
    )
  return _move(node, result, phase.results[result])


def apply_predicate_edge(node, name, target):
  """Moves a node along a graph-state edge. Mirrors `apply_action`.

  Unvalidated on purpose: the caller is the scheduler, which named the edge in
  its own source rather than parsing it out of an agent's message, and
  `validate_pipeline` has already checked the target.
  """
  return _move(node, name, target)


validate_pipeline()
