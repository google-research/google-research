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

"""The actions an agent declares, and the read-only node report.

Two kinds of thing end an agent's turn. `submit` hands over a Lean file, and
what it turns out to be -- proved, decomposed, disproved -- is read off that
file and the kernel; that path lives in `state_manager`. Everything else is a
*declaration*: `planned`, `refute`, `failed`, `replan`, `approved`. There is no
artifact to read them off, so the agent says so, and the only thing it carries
is `--reason`.

One function handles all five. They differ in exactly two places -- retiring an
AND-node on `review --replan`, raising the review threshold on `approved` --
and both are three lines. Splitting them per role, which is what `channels.py`
did, meant an agent that never writes Lean had to learn which of two commands
to run and why.

Nothing here files a document. The proof graph holds certified truth and search
metadata; prose belongs to whoever wrote it. The informal prover writes its own
`scratch/blueprint.md` and `findings.md`, peers read those files where they
were written, and no action copies them into the graph -- that would make
`state_manager` the owner of a document it cannot check and give every reader
two places to look.

Phase validation is not repeated per action. An action is checked against
whatever phase the node is actually in, so `--action=approved` on a node in
`informal` is refused because `informal` does not offer it -- not because this
module knows `approved` belongs to the reviewer. One rule, in `pipeline`.

Kept out of `state_manager.py` on purpose: adding a role should cost a row in
`pipeline.PIPELINE`, not a diff to the proof-submission path every other role
depends on.
"""

from typing import Tuple

import lean_code_parser as models
import pipeline
import review_trigger


#: A reason shorter than this is not a reason. Deliberately crude: the point is
#: to refuse `--reason="failed"`, which passes an is-it-empty check and tells
#: the next agent nothing, not to grade prose.
MIN_REASON_WORDS = 4


def _unknown_action(action, phase):
  """Refusal for an action the node's current phase does not offer."""
  allowed = ", ".join(
      f"--action={a}" for a in pipeline.allowed_actions(phase)
  )
  return (
      f"Refused: `--action={action}` is not something a node in phase"
      f" '{phase.name}' can do.\nNothing was recorded and your claim is still"
      " held, so re-run the same command with a valid action.\nACTION"
      f" REQUIRED: use one of: {allowed}. The `## How To Submit` section of"
      " your `context.md` says what each one means for this node."
  )


def _needs_reason(action, reason):
  """Refusal for a declared action whose reason is missing or useless."""
  return (
      f"Refused: `--action={action}` submits no file, so `--reason` is the only"
      f" thing the next agent inherits, and '{reason}' is not enough. Without"
      " it the node is handed on with no evidence and the next attempt repeats"
      " yours.\nNothing was recorded and your claim is still held.\nACTION"
      " REQUIRED: re-run the same command with"
      ' `--reason="<one or two full sentences saying what you found and what'
      ' the next agent should do differently>"`.'
  )


def _nothing_to_retire(node_id):
  """Refusal for a review verdict on a node with no live decomposition."""
  return (
      f"Refused: '{node_id}' has no live decomposition to retire, so there is"
      " nothing for a review verdict to act on. Nothing was recorded and your"
      " claim is still held.\nACTION REQUIRED: report this to the"
      " orchestrator; the node should not have been dispatched for review."
  )


def _recorded(node_id, action, next_phase):
  """Confirms a declaration by naming where it left the node.

  The action names an edge and the agent cannot see where that edge points, so
  the reply says. There is no "and the node is dead" case: no declared action
  leaves the machine, and `validate_pipeline` enforces that. Only a submitted
  artifact -- a kernel-checked proof or a kernel-checked refutation -- ends a
  node, which is the whole point of the rule.
  """
  return (
      f"Recorded `{action}` for '{node_id}'. Your claim is released and the"
      f" node is now in phase '{next_phase}'.\nACTION REQUIRED: conclude your"
      " turn."
  )


def _retire_header_blocks(graph, and_node_id, root_id):
  """Marks the shared header blocks a retired decomposition left behind.

  A block reaches `dynamic_headers` when its node submits, which is before the
  reviewer has ruled. Retire the decomposition and the block is residue: it
  belongs to a plan that no longer exists, but it is compiled into every node's
  environment on every build for the rest of the run, and it holds its names
  against the re-planned branch that replaces it.

  The window is not short. Patrol review fires on a decomposition that has been
  live for hours, by which time siblings may have been proved against exactly
  these declarations -- so this refuses rather than retires whenever anything
  still refers to a name the block introduces. That is the whole safety
  argument: the worst case is today's behaviour, a block that stays.

  Two things the name search cannot see, handled separately below: what a
  reference looks like when it is not a name at all, and work that is in flight.
  The second one is not here -- verification runs unlocked, so a proof compiled
  against a header this call is about to change has not reached the graph yet.
  `state_manager._retired_header_references` catches that at commit.
  """
  if not and_node_id:
    return
  candidates = [
      b
      for b in graph.environment.dynamic_headers
      if b.and_node_id == and_node_id and not b.retired
  ]
  if not candidates:
    return

  # Everything that could hold a reference, minus the blocks being retired.
  # The other header blocks first -- a later block building on an earlier one is
  # exactly the case D1's ordering exists for.
  others = [b for b in graph.environment.dynamic_headers if b not in candidates]
  corpus = [b.content or "" for b in others if not b.retired]
  live_ids = (
      graph.get_active_reachable_nodes(root_id) if root_id else set(graph.nodes)
  )
  for node_id, node in graph.nodes.items():
    # A *statement* counts only while its node is reachable, and that
    # distinction is what makes this function do anything at all. The block
    # exists so the plan's own children could state their goals -- that is what
    # the header is for -- so those statements name these declarations
    # essentially always. Counting them retires nothing, ever. `active.failed`
    # is set before this call, so they are already out of `live_ids`, and
    # `prune_orphans` deletes whatever did no work at the end of this same
    # transaction. Sound because an unreachable node is never compiled:
    # `assemble_current_proof` walks the dependencies of the root.
    if node_id in live_ids:
      corpus.append(node.signature_str or "")
    # A *proof body* counts either way. An unreachable proved node is not
    # compiled today, but `add_proposed_subgoals` re-adopts an existing node by
    # name, so a later plan can pull it back into the tree with its recorded
    # proof intact -- and that proof would then cite a declaration that is gone.
    for artifact in (node.formal_theorem, node.formal_sketch):
      if artifact is not None:
        corpus.append(artifact.proof.tactics or "")
  haystack = "\n".join(corpus)

  for block in candidates:
    # An `instance` is found by typeclass resolution and an `@[simp]` lemma by
    # the simp set: neither is ever written by name, so the search below cannot
    # see the proof that depends on one, and retiring it would break a node
    # that never mentioned it. These are exactly the forms
    # `_global_check_mode` escalates to a full-tree recompile for, and the same
    # reach is what makes them unretirable. They stay, permanently.
    if models.classify_definitions_block(block.content or "").escalating:
      continue
    names = [
        n
        for n in (
            models.extract_declaration_name(d)
            for d in models.split_declarations(block.content or "")
        )
        if n
    ]
    # Substring rather than a token match: approximate on purpose, and
    # approximate in the safe direction. A false hit keeps a dead block, which
    # costs compile time; a false miss deletes a declaration something is
    # already verified against.
    if any(name in haystack for name in names):
      continue
    block.retired = True


def declare(mgr, node_id, action, reason):
  """Applies an action the agent declared, having submitted no file.

  Args:
    mgr: The `ProofGraphStateManager`, for the lock, the node log and the claim.
    node_id: The node the agent holds.
    action: One of the declared actions its current phase offers. Never
      `submit`, which reads a file and goes through `record_lean`.
    reason: Why. Required, and required to be a sentence.

  Returns:
    (ok, message). `ok` is True exactly when the claim was released and the
    turn is over; False always means nothing moved and the node is still held.
  """
  with mgr.lock_and_modify() as graph:
    node = graph.nodes.get(node_id)
    if not node:
      # True, not False: there is nothing to retry against, and telling the
      # agent to resubmit would spin it against a node that does not exist.
      return True, (
          f"Node '{node_id}' is not in the proof graph, so nothing was"
          " recorded. ACTION REQUIRED: conclude your turn and report this to"
          " the orchestrator."
      )

    phase = pipeline.phase_for(node)

    # --- Refusals first: nothing below this block may have written anything.
    if action not in phase.actions:
      return False, _unknown_action(action, phase)
    if len((reason or "").split()) < MIN_REASON_WORDS:
      return False, _needs_reason(action, reason)

    retiring = phase.name == pipeline.REVIEW_PHASE and action == "replan"
    active = node.active_and_node
    if retiring and active is None:
      return False, _nothing_to_retire(node_id)

    # A node its parent has already abandoned is not worth moving. The claim is
    # released and the agent is told to stop rather than being handed a
    # rejection it could act on.
    target_root = mgr._get_root_id(graph)  # pylint: disable=protected-access
    if target_root and not graph.is_node_reachable_from_root(
        node_id, target_root
    ):
      return mgr._finalize_orphaned_node(node)  # pylint: disable=protected-access

    # --- Side effects. Two actions have one each.
    if retiring:
      # Retire the *plan*, not the goal. `decomposition_failed` would mean this
      # node is dead and resolve it to FAILED so its parent backtracks; failing
      # the only live AND-node already resolves it to FORMAL_STATEMENT, which
      # is the re-plan state this edge points at. Children still in flight are
      # dropped from the reachable set by the same write, and the next thing
      # any worker on them submits is released as orphaned.
      active.failed = True
      active.error_message = f"RETIRED BY REVIEW: {reason}"
      # The graph half of the retraction is above. This is the other half: the
      # decomposition also injected a block into the shared header at submission
      # time, and a retraction that stops at the graph leaves that block
      # compiling into every node for the rest of the run.
      _retire_header_blocks(graph, active.id, target_root)
    elif action == "approved":
      review_trigger.record_approval(graph, node)

    pipeline.apply_action(node, action)

    # Bumped for the same reason a successful submission bumps it. The
    # orchestrator can `release` a node whose worker is still compiling, and
    # the freed node can be claimed and moved before that compile lands; the
    # bump is what makes the optimistic-concurrency check in `_commit_submission`
    # reject the stale write instead of applying it to a node that has moved on.
    node.search_metadata["revision"] = (
        node.search_metadata.get("revision", 0) + 1
    )

    # `last_error` is the *current* phase's note and the briefing renders it as
    # such, so it is overwritten rather than appended to: carrying a planner's
    # complaint forward past the fix briefs the next agent to fix it again.
    node.search_metadata["last_error"] = reason
    # `note`, not the default `rejection`. This is the agent's own account of
    # what it just did -- a planner's `planned`, a reviewer's `approved`.
    # Filing it as a rejection made 11 of the 14 logs on `cyclotomic_val_gt_n`
    # the agent's own prose, which the briefing then handed to the next agent
    # as "the compiler's own output".
    mgr._write_node_log(node_id, reason, "note")  # pylint: disable=protected-access
    mgr._clear_claim_metadata(node)  # pylint: disable=protected-access

    return True, _recorded(node_id, action, pipeline.current_phase_name(node))


def node_report(mgr, node_id):
  """Renders what only the proof graph knows about one node.

  Status, phase, who holds it, what it depends on. Nothing that lives in a file
  -- the blueprint and the findings are prose the agents own, and are read
  where they were written.
  """
  graph = mgr.read_graph()
  node = graph.nodes.get(node_id)
  if not node:
    return False, f"Node '{node_id}' not found in the proof graph."

  active = node.active_and_node
  meta = node.search_metadata
  status = graph.compute_statuses().get(node_id, node.mathematical_status)
  children = (
      ", ".join(f"`{c}`" for c in active.dependency_ids) if active else "none"
  )
  lines = [
      f"### Report for `{node_id}`",
      "",
      f"- **Status**: {getattr(status, 'value', status)}",
      f"- **Phase**: {pipeline.current_phase_name(node) or 'left the machine'}",
      f"- **Held by a worker**: {bool(meta.get('processing'))}",
      # Dispatches beside recorded actions. A node dispatched many times with
      # no new revision is being handed out and handed back without anything
      # being written down -- the livelock that froze the orchestrator in
      # rollout fmf005_sim, invisible at the time because nothing counted it.
      "- **Dispatched / actions recorded**:"
      f" {meta.get('dispatch_count', 0)} / {meta.get('revision', 0)}",
      f"- **Children**: {children}",
      "",
      "**Statement**",
      "```lean",
      node.signature_str or "(no formal statement yet)",
      "```",
  ]
  if meta.get("last_error"):
    lines += ["", "**Last error**", "```", str(meta["last_error"])[:1500], "```"]
  return True, "\n".join(lines)
