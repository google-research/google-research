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

"""Workspace Manager for LEMMA theorem proving architecture.

Manages filesystem layout, isolated per-node workspaces, Lean 4 scaffold
generation, and mathematical context markdown rendering.
"""

import json
import os
import shlex
import time
from typing import Any, Dict, List, Optional

import dag
import lean_code_parser as models
import pipeline
import review_trigger


class WorkspaceManager:
  """Manages filesystem layout, node workspaces, and context rendering."""

  #: Optional folder at the problem root holding material supplied with the
  #: problem -- an informal proof, a paper excerpt, prior notes. Nothing in the
  #: tool creates, reads or cleans it; it exists only to be named in briefings.
  REFERENCE_DIRNAME = "reference"

  def __init__(self, problem_root_dir):
    """Initializes the WorkspaceManager for a given problem root directory."""
    self.root_dir = os.path.abspath(problem_root_dir)
    self.nodes_dir = os.path.join(self.root_dir, "nodes")
    self.assembled_dir = os.path.join(self.root_dir, "assembled")

  def list_reference_files(self):
    """Returns the filenames in `<root>/reference/`, sorted; empty if absent.

    Agents are forbidden `find`, `ls` and `grep`, so a file no briefing names
    does not exist as far as they are concerned. Listing is re-read on every
    claim rather than cached, so material dropped in mid-run reaches the next
    agent without a restart.
    """
    path = os.path.join(self.root_dir, self.REFERENCE_DIRNAME)
    if not os.path.isdir(path):
      return []
    return sorted(
        name
        for name in os.listdir(path)
        if not name.startswith(".")
        and os.path.isfile(os.path.join(path, name))
    )

  def get_node_dir(self, node_id):
    """Returns the dedicated workspace directory for a node."""
    return os.path.join(self.nodes_dir, node_id)

  def get_logs_dir(self, node_id):
    """Returns the diagnostics logs directory for a node."""
    return os.path.join(self.get_node_dir(node_id), "logs")

  def get_scaffold_path(
      self, node_id, role = models.SCAFFOLD_PROVE
  ):
    """Returns the path to the ready-to-compile scaffold file for a role."""
    return os.path.join(
        self.get_node_dir(node_id), models.scaffold_filename(role)
    )

  def get_context_path(self, node_id):
    """Returns the path to the mathematical briefing context.md file."""
    return os.path.join(self.get_node_dir(node_id), "context.md")

  def describe_layout(self):
    """Returns human- and agent-readable documentation of the workspace layout.

    Every path an agent may meet is listed, with who writes it. Listing only
    the three obvious ones left the rest -- the rotated archives, the
    verification dumps, `failure_report.md` -- looking like debris an agent
    could clean up, or a gap it could invent its own conventions to fill.
    """
    # Same reason the rotated archives are listed: an undocumented directory at
    # the root reads as debris. Only appears when it holds something, so a run
    # with no supplied material is not told to go looking for one.
    references = self.list_reference_files()
    reference_section = ""
    if references:
      listing = "\n".join(f"    - `{name}`" for name in references)
      reference_section = f"""
## 4. Supplied Reference Material (read-only, not tool-owned)
- `{self.root_dir}/{self.REFERENCE_DIRNAME}/`
  - Shipped with the problem by whoever set it up. Nothing in this system
    wrote it, checked it, or verified it. Read it freely; treat every claim in
    it as a lead to confirm, never as an established fact. Do not edit it.
  - Holds:
{listing}
"""
    return f"""# LEMMA Filesystem Workspace Architecture

Problem Root: `{self.root_dir}`

Two owners, and the boundary is absolute. Anything under **Tool-owned** is
written by `state_manager` / `WorkspaceManager`: read it, never edit it, and
expect it to be replaced without warning. Anything under **Yours** is yours to
create, edit and delete.

## 1. Centralized State (tool-owned)
- `{self.root_dir}/proof_graph.json`
  - Dual-locked (thread + fcntl OS lock) centralized proof DAG state. Holds
    certified truth and search metadata only -- no prose, no plans, no findings.
  - Managed exclusively by `ProofGraphStateManager`. Never hand-edit it; a
    concurrent write will be lost or will corrupt the run.

## 2. Per-Node Workspaces: `{self.nodes_dir}/<node_id>/`

### Tool-owned
- `proof_scaffold.lean` / `disproof_scaffold.lean` — ready-to-compile Lean 4
  scaffold with the target as `:= by sorry`. Regenerated from a clean baseline
  every time the node is claimed. This is the file you fill in and the file
  `--action=submit` reads; do not rename it.
- `context.md` — your briefing. Regenerated on every claim.
- `*_old_<unix_ms>.*` — the previous turn's copy of everything the tool
  rotates: the scaffold, the briefing, and your role's own log, notes and
  audit under `scratch/`. Moved aside rather than overwritten when the node
  was re-claimed, so an earlier attempt's Lean and an earlier reviewer's
  verdicts are both still here. This is where the record of what has already
  been tried on this node lives. Read-only.
- `failure_report.md` — the full compiler output of every submission the
  **verifier rejected** on this node, newest last. Only exists if that has
  happened; a node given up on some other way has none.
- `logs/<unix_ts>_<type>.log` — one file per recorded attempt, holding the
  message the tool sent back. `<type>` is `rejection` (a submission refused),
  `note` (an agent's own stated reason for the action it took), `infra`,
  `disproof` or `unreachable_abort`.
- `logs/verification/<stamp>_{{local,global}}.lean` and the sibling `.json` —
  exactly what was sent to Lean and exactly what came back, for both compiles
  of every submission. Read these when a rejection looks like it might be the
  tool assembling something wrong rather than your proof being wrong.

### Yours
- `findings.md` — facts you established that outlive your turn. Peers and your
  parent read this at `nodes/<your_id>/findings.md`, so write it for them.
- `scratch/blueprint.md` — the plan of record for this node. Section 8 of every
  briefing under this node points here. The informal prover writes it; later
  phases follow it or send the node back with `--action=replan`.
- `scratch/` — everything else you need: draft proofs, probe scripts, verifier
  inputs. Name them what you like.
- **Never write outside your own node directory**, and never to `/tmp` or
  `/var/tmp`. You may *read* any peer's directory.

## 3. Final Assembled Output (tool-owned)
- `{self.assembled_dir}/final_proof.lean`
  - Unified zero-sorry Lean 4 proof file, generated deterministically by
    `--action=assemble` once the root is proved.
{reference_section}"""

  def inspect_workspace(self):
    """Inspects the live filesystem state of the problem workspace."""
    state_file = os.path.join(self.root_dir, "proof_graph.json")
    has_state_file = os.path.exists(state_file) and os.path.getsize(state_file) > 0
    assembled_file = os.path.join(self.assembled_dir, "final_proof.lean")
    has_assembled = os.path.exists(assembled_file) and os.path.getsize(assembled_file) > 0

    node_entries: Dict[str, Any] = {}
    if os.path.exists(self.nodes_dir):
      for node_name in sorted(os.listdir(self.nodes_dir)):
        node_path = os.path.join(self.nodes_dir, node_name)
        if not os.path.isdir(node_path):
          continue
        scaffold_path = os.path.join(
            node_path, models.scaffold_filename(models.SCAFFOLD_PROVE)
        )
        if not os.path.exists(scaffold_path):
          scaffold_path = os.path.join(
              node_path, models.scaffold_filename(models.SCAFFOLD_DISPROVE)
          )
        context_path = os.path.join(node_path, "context.md")
        logs_dir = os.path.join(node_path, "logs")

        log_count = 0
        if os.path.exists(logs_dir):
          log_count = len([f for f in os.listdir(logs_dir) if f.endswith(".log")])

        node_entries[node_name] = {
            "has_scaffold": os.path.exists(scaffold_path),
            "scaffold_size_bytes": os.path.getsize(scaffold_path) if os.path.exists(scaffold_path) else 0,
            "has_context": os.path.exists(context_path),
            "context_size_bytes": os.path.getsize(context_path) if os.path.exists(context_path) else 0,
            "log_files_count": log_count,
        }

    return {
        "root_dir": self.root_dir,
        "has_state_file": has_state_file,
        "has_assembled_proof": has_assembled,
        "total_nodes_on_disk": len(node_entries),
        "nodes": node_entries,
    }

  def provision_node_workspace(
      self,
      node_id,
      graph,
      phase,
      root_id = None,
  ):
    """Provisions {problem_root_dir}/nodes/{node_id}/ with scaffold (.lean) and context.md.

    Takes the phase, not a role string. The phase decides both what gets
    written and what the briefing says, and passing a role instead meant the
    two were derived separately -- a planner claimed for `informal` was handed
    a proof scaffold, which is an invitation to write Lean one phase early.
    """
    node_dir = self.get_node_dir(node_id)
    logs_dir = self.get_logs_dir(node_id)
    scratch_dir = os.path.join(node_dir, "scratch")
    os.makedirs(node_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    # Rotated, not left in place. These are one-turn files: the agent holding
    # the node now writes them from a template, and the agent that held it
    # last already did. Leaving them produced both failure modes at once in
    # fmf_v1_004 -- `prover_log.md` was reinitialized to a bare `[INIT]` on
    # every re-entry, destroying the three-attempt evidence the decompose gate
    # is defined over, while `blueprint_fidelity_audit.md` was *not* rewritten
    # and sat there for three attempts until a reviewer rejected the node over
    # its stale line numbers.
    #
    # `findings.md` and `scratch/blueprint.md` are deliberately not in
    # `turn_artifacts`: those two are the cross-turn record, peers and parents
    # read them, and the briefing says so.
    for rel in phase.turn_artifacts:
      self._rotate_existing(os.path.join(node_dir, rel))

    scaffold_path = ""
    scaffold_content = ""
    if phase.scaffold is not None:
      scaffold_content = self.render_scaffold_lean(
          node_id, graph, role=phase.scaffold
      )
      scaffold_path = self.get_scaffold_path(node_id, role=phase.scaffold)
      self._rotate_existing(scaffold_path)
      with open(scaffold_path, "w", encoding="utf-8") as f:
        f.write(scaffold_content)

      # Write the backup files .pristine_{scaffold_name}.backup
      scaffold_name = os.path.basename(scaffold_path)
      pristine_scaffold_path = os.path.join(
          node_dir, f".pristine_{scaffold_name}.backup"
      )
      with open(pristine_scaffold_path, "w", encoding="utf-8") as f:
        f.write(scaffold_content)

    context_content = self.render_context_md(
        node_id, graph, phase=phase, root_id=root_id
    )
    context_path = self.get_context_path(node_id)
    self._rotate_existing(context_path, context_content)
    with open(context_path, "w", encoding="utf-8") as f:
      f.write(context_content)

    return {
        "node_dir": node_dir,
        "logs_dir": logs_dir,
        "scratch_dir": scratch_dir,
        "scaffold_path": scaffold_path,
        "context_path": context_path,
        "scaffold_content": scaffold_content,
        "context_content": context_content,
    }

  def get_verification_dir(self, node_id):
    """Returns the directory holding verification traces for a node."""
    return os.path.join(self.get_logs_dir(node_id), "verification")

  def write_verification_trace(
      self,
      node_id,
      phase,
      code,
      result,
      attempt_stamp = None,
  ):
    """Dumps exactly what was sent to Lean and exactly what came back.

    Written for both verification phases of every submission:

      * 'local'  -- the reconstructed scaffold for this node, with subgoals
                    injected as axioms.
      * 'global' -- the whole proof tree assembled from the root, with this
                    submission's full prospective mutation already applied.

    Without this the two compiles are invisible, and a rejection blamed on the
    agent could equally be the tool assembling something wrong.

    Returns:
      Path to the .lean file; the sibling .json holds the verifier result.
    """
    directory = self.get_verification_dir(node_id)
    os.makedirs(directory, exist_ok=True)
    stamp = attempt_stamp or f"{time.time():.6f}".replace(".", "_")
    stem = os.path.join(directory, f"{stamp}_{phase}")
    lean_path = f"{stem}.lean"
    with open(lean_path, "w", encoding="utf-8") as f:
      f.write(code)
    with open(f"{stem}.json", "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2)
    return lean_path

  def get_failure_report_path(self, node_id):
    """Returns the path to the node's accumulated rejection history."""
    return os.path.join(self.get_node_dir(node_id), "failure_report.md")

  def has_failure_report(self, node_id):
    """Whether this node has a failure report on disk.

    Every pointer to `failure_report.md` in a briefing is gated on this. The
    file is written only when the verifier rejects a submission, so a node that
    died some other way -- the planner called it, a reviewer retired its
    decomposition -- has none, and the briefing used to send a backtracking
    parent to a path that does not exist. A dangling pointer at exactly the
    moment an agent needs evidence is worse than saying there is none.
    """
    path = self.get_failure_report_path(node_id)
    return os.path.exists(path) and os.path.getsize(path) > 0

  def append_failure_report(
      self, node_id, err_msg, attempt, is_decomp
  ):
    """Appends one rejection to the node's failure report.

    The briefing tells agents to read this file, and tells a backtracking parent
    to read its dead children's copies. Nothing wrote it before, so every one of
    those pointers dangled.
    """
    path = self.get_failure_report_path(node_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    kind = "Decomposition" if is_decomp else "Proof"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(path, "a", encoding="utf-8") as f:
      if is_new:
        f.write(f"# Failure Report: `{node_id}`\n\n")
        f.write(
            "Every rejected submission for this node, newest last. A parent"
            " backtracking off this subgoal should read this to understand what"
            " actually blocked it.\n"
        )
      f.write(f"\n## {kind} attempt {attempt} — {stamp}\n\n```\n")
      f.write(str(err_msg).strip())
      f.write("\n```\n")
    return path

  def _rotate_existing(
      self, path, new_content = None
  ):
    """Moves an existing file aside instead of overwriting it.

    A node is re-provisioned when a previously successful decomposition failed
    and has to be redone. Overwriting in place silently discarded whatever the
    prior agent had written, with no way to recover it.

    `new_content` is what is about to be written, for callers that know it up
    front. If it matches the file already there, nothing is archived: a
    re-provision producing identical bytes is not a new turn, and rotating it
    buries the one signal that the node is being handed out with nothing
    happening to it. In fmf_v1_006 a node parked in `review` was claimed 111
    times in an hour and left 110 archived briefings with five distinct
    contents between them, and the only place that showed up was the file
    count. Callers that cannot know the new content -- the turn artifacts the
    agent itself writes -- pass nothing and always rotate.
    """
    if not os.path.exists(path):
      return None
    if new_content is not None:
      with open(path, "r", encoding="utf-8") as f:
        if f.read() == new_content:
          return None
    base, ext = os.path.splitext(path)
    archived = f"{base}_old_{int(time.time() * 1000)}{ext}"
    os.replace(path, archived)
    return archived

  def render_node_files(
      self,
      node_id,
      graph,
      phase,
      root_id = None,
  ):
    """Returns the paths and rendered content for a node without writing.

    Read-only callers must use this rather than provision_node_workspace, which
    rotates the agent's in-progress files. Where a scaffold already exists on
    disk, its current contents are returned instead of a fresh baseline so the
    caller sees the agent's actual work.
    """
    scaffold_path = ""
    scaffold_content = ""
    if phase.scaffold is not None:
      scaffold_path = self.get_scaffold_path(node_id, role=phase.scaffold)
      if os.path.exists(scaffold_path):
        with open(scaffold_path, "r", encoding="utf-8") as f:
          scaffold_content = f.read()
      else:
        scaffold_content = self.render_scaffold_lean(
            node_id, graph, role=phase.scaffold
        )

    return {
        "node_dir": self.get_node_dir(node_id),
        "logs_dir": self.get_logs_dir(node_id),
        "scratch_dir": os.path.join(self.get_node_dir(node_id), "scratch"),
        "scaffold_path": scaffold_path,
        "context_path": self.get_context_path(node_id),
        "scaffold_content": scaffold_content,
        "context_content": self.render_context_md(
            node_id, graph, phase=phase, root_id=root_id
        ),
    }

  def render_scaffold_lean(
      self, node_id, graph, role = "prove"
  ):
    """Generates a clean, compilable Lean 4 scaffold for node_id with delimited sections."""
    if node_id not in graph.nodes:
      return ""

    target_node = graph.nodes[node_id]
    target_sig = target_node.signature_str
    header = graph.environment.to_lean_header()

    # A disproof is a different obligation, not the same one with a different
    # filename: the target moves into the header as an axiom and what the agent
    # must close is `False`. Rendering the target as the goal here would ask an
    # agent routed to `disprove` to prove the thing it was sent to refute.
    if role == models.SCAFFOLD_DISPROVE:
      header = models.build_disproof_header(header, target_sig)
      target_sig = models.disproof_obligation(node_id)

    return models.build_lean_scaffold(
        immutable_context=header,
        definitions="",
        subgoals="",
        proof=f"{target_sig} :=\nby\n  sorry",
        role=role,
    )

  def _render_review_entrance(self, node, graph):
    """Says which of the two entrances into `review` this node came through.

    The two ask opposite questions -- the gate rules on a split nothing has been
    built on, the patrol rules on one that has been running for generations --
    and the difference decides how demanding the reviewer should be. It is not
    stored anywhere, but it is derivable: a reviewer's approval marks the
    AND-node it approved, and the only route into `blocked` (and so into the
    patrol) is an approval, so an unmarked live decomposition means nobody has
    ruled on THIS one yet.

    Absence, not falsiness. The gate stores 0.0 -- a plan whose children have
    not run yet has a re-plan rate of zero -- so a truthiness test would call
    every second review a first one.
    """
    if not review_trigger.has_been_reviewed(node):
      return [
          "> [!IMPORTANT]",
          "> **Why you were called: a prover just submitted this"
          " decomposition.** Nothing has been built on it — the children are"
          " held and none of them has run. Retiring it costs one planning pass"
          " and nothing else, so be demanding.",
          "",
      ]
    rate = review_trigger.worst_child_replan_rate(graph, node.id)
    return [
        "> [!IMPORTANT]",
        "> **Why you were called: this decomposition was approved before, and"
        " the branch under it has been re-planning itself since without"
        f" closing anything — {rate:.1f} planning attempts on its worst child"
        " branch, where a healthy branch is 1.**"
        " Work already exists below it. Retiring it now orphans that work, so"
        " default to `--action=approved` unless the split is actually wrong."
        " Slowness is not wrongness; a hard problem taking a long time is the"
        " system working.",
        "",
    ]

  def render_context_md(
      self,
      node_id,
      graph,
      phase,
      root_id = None,
  ):
    """Generates a structured mathematical briefing context.md for node_id.

    Which sections appear is `phase.sections` and nothing else. The numbering
    is stable across phases -- a reviewer's briefing jumps from 3 to 8 -- so
    that a prompt can name "section 9" and mean the same thing everywhere,
    which renumbering per phase would quietly break.
    """
    if node_id not in graph.nodes:
      return ""
    if phase.role is None:
      # No role means nobody is dispatched here, so there is no briefing to
      # write. Rendering the page anyway produced a "How To Submit" block with
      # no commands under it, under a line reading "You are the **None**".
      return (
          f"# `{node_id}` is in phase `{phase.name}`\n\nNo agent works this"
          " phase. It resolves when the scheduler's own predicates fire, and"
          " there is nothing here for anyone to do.\n"
      )
    target_node = graph.nodes[node_id]
    show = set(phase.sections)
    role = phase.scaffold or phase.role or ""

    # 1. Root Problem
    root_stmt = ""
    target_root_id = root_id or (
        next(iter(graph.nodes.keys())) if graph.nodes else None
    )
    if target_root_id and target_root_id in graph.nodes:
      root_node = graph.nodes[target_root_id]
      if root_node.formal_theorem:
        root_stmt = root_node.formal_theorem.signature.to_lean()
      elif root_node.formal_sketch:
        root_stmt = root_node.formal_sketch.signature.to_lean()
      elif root_node.informal_problem:
        r_raw = root_node.informal_problem.statement.strip()
        if not r_raw.startswith("theorem ") and not r_raw.startswith("lemma "):
          root_stmt = models.FormalSignature(
              name=target_root_id, statement=r_raw
          ).to_lean()
        else:
          root_stmt = r_raw

    # 2. Target Goal
    target_stmt = ""
    if target_node.formal_theorem:
      target_stmt = target_node.formal_theorem.signature.to_lean()
    elif target_node.formal_sketch:
      target_stmt = target_node.formal_sketch.signature.to_lean()
    elif target_node.informal_problem:
      stmt = target_node.informal_problem.statement.strip()
      if not stmt.startswith("theorem ") and not stmt.startswith("lemma "):
        target_stmt = models.FormalSignature(
            name=node_id, statement=stmt
        ).to_lean()
      else:
        target_stmt = stmt

    # 3. Parent Ref & Upstream Ancestry
    parent_stmt = ""
    if (
        target_node.reverse_dependency_ids
        and target_node.reverse_dependency_ids[0] in graph.nodes
    ):
      parent_id = target_node.reverse_dependency_ids[0]
      p_node = graph.nodes[parent_id]
      if p_node.formal_theorem:
        parent_stmt = p_node.formal_theorem.signature.to_lean()
      elif p_node.formal_sketch:
        parent_stmt = p_node.formal_sketch.signature.to_lean()
      elif p_node.informal_problem:
        p_raw = p_node.informal_problem.statement.strip()
        if not p_raw.startswith("theorem ") and not p_raw.startswith("lemma "):
          parent_stmt = models.FormalSignature(
              name=parent_id, statement=p_raw
          ).to_lean()
        else:
          parent_stmt = p_raw

    ancestry_context = graph.assemble_for_tree_context(node_id)

    # 4. Diagnostics & Logs
    latest_log_path = self.get_latest_attempt_log_path(node_id)
    last_error = target_node.search_metadata.get("last_error", "")
    proof_attempts = target_node.search_metadata.get("proof_attempts", 0)
    decomp_attempts = target_node.search_metadata.get("decomp_attempts", 0)

    lines = [
        f"# Mathematical Context Briefing for Subgoal: `{node_id}`",
        "",
        f"You are the **{phase.role}** and this node is in phase"
        f" **`{phase.name}`**.",
        "",
    ]
    if 1 in show:
      lines.append("## 1. Global Problem Specification")
      if root_stmt:
        lines.append("```lean")
        lines.append(root_stmt)
        lines.append("```\n")
      else:
        lines.append("*Root problem specification active in proof graph.*\n")

    allowed_axioms = ", ".join(
        f"`{a}`" for a in sorted(graph.effective_allowed_axioms)
    )
    refuting = phase.scaffold == models.SCAFFOLD_DISPROVE
    if 2 in show:
      lines.extend([
          "## 2. Active Target Goal",
          f"- **Node ID**: `{node_id}`",
          (
              "- **Statement under refutation** (assumed for you as an axiom;"
              " it is NOT what you write):"
              if refuting
              else "- **Target Lean Declaration**:"
          ),
          "```lean",
          target_stmt,
          "```",
          "",
      ])
      if refuting:
        # The prover's two rules read as instructions to prove the target,
        # which is the opposite of this phase's job. Both have a disproof
        # counterpart, and both are rejections the agent cannot otherwise see
        # coming: the obligation is a declaration that appears nowhere above,
        # and the target axiom is the one axiom the gate insists on.
        lines.extend([
            "> [!CAUTION]",
            (
                "> **Two hard rejections to avoid — both cost a full"
                " verification round trip:**"
            ),
            (
                "> 1. **Your obligation is `False`, and its signature is"
                " immutable.** The `-- [START_PROOF]` block MUST begin with"
                f" `{models.disproof_obligation(node_id)}` character for"
                " character, followed by `:=`. The statement above is already"
                " assumed at the top of the scaffold; re-declaring it, or"
                " writing its negation, is not this."
            ),
            (
                f"> 2. **Only these axioms are permitted:** {allowed_axioms},"
                " plus the statement under refutation itself — and your"
                " derivation MUST actually use that one. A `False` derived"
                " without it means something already in scope is contradictory"
                " rather than that the target is false, and it is rejected as"
                " an alarm. `native_decide` and any `axiom` you declare"
                " yourself are rejected as usual, as is any leftover `sorry`."
            ),
            "",
        ])
      elif phase.scaffold is not None:
        lines.extend([
            "> [!CAUTION]",
            (
                "> **Two hard rejections to avoid — both cost a full"
                " verification round trip:**"
            ),
            (
                "> 1. **The signature is immutable.** Your `-- [START_PROOF]`"
                " block MUST begin with the exact declaration above, character"
                " for character, followed by `:=`. Do not rename it, reorder"
                " binders, or restyle the spacing (`(n : Nat)` and `(n: Nat)`"
                " are NOT interchangeable here)."
            ),
            (
                f"> 2. **Only these axioms are permitted:** {allowed_axioms}."
                " Tactics that discharge a goal by trusting the compiler"
                " instead of the kernel — `native_decide` above all —"
                " introduce other axioms and are rejected, even though they"
                " compile. So is any `axiom` you declare yourself, and any"
                " leftover `sorry`."
            ),
            "",
        ])
      if phase.name == pipeline.REVIEW_PHASE:
        lines.extend(self._render_review_entrance(target_node, graph))
      if node_id == target_root_id and "failed" in phase.actions:
        # Gated on the action being offered rather than on the phase name, so
        # the warning follows `failed` if it ever moves. Every other node can
        # afford `failed` -- the parent backtracks and the search continues.
        # The root cannot: `claim_terminal` reads it as ROOT DECLARED
        # DEFECTIVE and stops the run. Planners have spent that budget on
        # "this is famously hard and the published proof needs a solver we do
        # not have", which is a decomposition problem, not a defective goal.
        lines.extend([
            "> [!CAUTION]",
            (
                "> **This node is the root of the whole problem, and"
                " `--action=failed` here ends the entire run.**"
            ),
            ">",
            (
                "> On any other node `failed` is cheap: the parent backtracks,"
                " tries a different split, and your findings survive to inform"
                " it. The root has no parent. Running `failed` here marks the"
                " problem itself defective, the orchestrator stops, and no"
                " proof is produced. Nothing retries after it."
            ),
            ">",
            (
                "> `failed` on the root means one thing only: **the problem"
                " statement is ill-posed or defective** -- malformed, or"
                " contradicted by its own hypotheses. None of the following is"
                " that, and none of them is a reason to use it:"
            ),
            "> - The problem is famous, hard, or open-flavoured.",
            (
                "> - You searched and found no published proof, or none that a"
                " human can check by hand."
            ),
            (
                "> - The known proof needs a library, solver, or tactic this"
                " environment does not have."
            ),
            "> - You cannot see a full decomposition yet.",
            "> - Every previous attempt on this node failed.",
            ">",
            (
                "> **Not seeing the whole proof is the normal state at the"
                " root.** It is the hardest node in the graph by construction,"
                " and there is no attempt budget in this system and no clock --"
                " a hard root is *supposed* to be re-planned many times. So"
                " split it further instead of ruling it out: the parts you can"
                " state precisely become subgoals, and the part carrying the"
                " difficulty becomes the `[CRUX] [LEMMA]`, whose sketch is"
                " allowed to say the argument is not yet known. A crux"
                " strictly smaller than the goal is progress even while it is"
                " still hard. Write what you established and what you ruled"
                " out into `findings.md` -- every later revision of this node"
                " reads it, so a round that ends without a finished plan still"
                " moves the problem forward."
            ),
            "",
        ])

    if 3 in show:
      lines.append("## 3. Parent Plan & Upstream Ancestry")
    if 3 in show and parent_stmt:
      lines.append(f"- **Parent Goal**: `{parent_stmt}`")
      if target_node.reverse_dependency_ids:
        p_id = target_node.reverse_dependency_ids[0]
        lines.append(
            f"- **Parent Blueprint**: If `nodes/{p_id}/scratch/blueprint.md`"
            " exists, read it for the parent's informal strategy and"
            " walkthrough."
        )
    if 3 in show:
      if ancestry_context:
        lines.append("\n**Upstream Ancestor Sketches (Integration Weaving)**:")
        lines.append("```lean")
        lines.append(ancestry_context)
        lines.append("```\n")
      elif parent_stmt:
        lines.append("")
      else:
        lines.append("*This is a top-level root goal.*\n")

    # 4. Dynamic Peer Discovery & State Queries
    if 4 in show:
      lines.extend([
          "## 4. Dynamic Peer Discovery & Role Guidelines",
          "> [!IMPORTANT]",
      ])
      if role == models.SCAFFOLD_DISPROVE:
        lines.extend([
            (
                "> **You are refuting this target, not proving it.** The"
                " statement above is assumed as an axiom in your scaffold and"
                " your obligation is literally `False`. Deriving it refutes"
                " the target."
            ),
            (
                "> **Nothing you derive may escape.** Everything provable here"
                " follows from a proposition the tree believes is false, so"
                " the definitions and subgoals blocks are not rendered at all."
                " There is one block to fill."
            ),
        ])
      else:
        lines.extend([
            (
                "> **Follow the Plan of Record.** Section 8 points at the"
                " informal proof this node was planned around. Formalize"
                " that; if it is wrong, say so with `--action=replan` rather"
                " than quietly proving something else."
            ),
            (
                "> **Prove directly, or decompose in the same file.** Leave"
                " `-- [START_SUBGOALS]` empty to prove the target outright."
                " To decompose, declare intermediate subgoals there ending in"
                " `:= sorry` and link them in `-- [START_PROOF]` with no"
                " `sorry` in the glue. Before you do, write the split down in"
                " `scratch/blueprint.md` — peers and your parent read it"
                " there. Strictly avoid local helper lemma suites in `--"
                " [START_DEFINITIONS]`."
            ),
        ])
      lines.extend([
          (
              "> **DO NOT Blindly Copy Peer Implementations**: Sibling"
              " subgoals often have different mathematical properties. Do NOT"
              " copy peer tactic scripts or case structures."
          ),
          "",
          "**Filesystem Peeking & Workspace Isolation**:",
          f"- You own `nodes/{node_id}/` (write exclusively to your folder).",
          (
              f"- **Scratch Directory**: `nodes/{node_id}/scratch/` (Save all"
              " temporary tests, draft scripts, and verifier inputs here)."
          ),
          "> [!CAUTION]",
          (
              "> **STRICT LOCAL SCRATCH RULE**: NEVER write files to global"
              " `/tmp/` or `/var/tmp/`. All intermediate and trial scripts"
              " MUST stay in your assigned `scratch/` folder."
          ),
          "",
          "- You may inspect peer directories for status awareness"
          " (read-only):",
          "  - Peer blueprint: `nodes/<peer_id>/scratch/blueprint.md`",
          "  - Peer findings: `nodes/<peer_id>/findings.md`",
          "  - Peer failure report: `nodes/<peer_id>/failure_report.md`"
          " (only exists where the verifier rejected a submission)",
          "  - Peer compiler logs: `nodes/<peer_id>/logs/`",
          "",
      ])

    if 5 in show:
      lines.extend([
          "## 5. Prior Attempt Diagnostics & History",
          f"- **Proof Attempts**: {proof_attempts}",
          f"- **Decomposition Attempts**: {decomp_attempts}",
      ])
      # Only pointed at when it exists. It is written on verifier rejection and
      # nowhere else, so a node whose attempts all ended some other way has an
      # attempt count above and no file, and the pointer sent the agent to a
      # path that is not there.
      if self.has_failure_report(node_id):
        lines.append(
            f"- **Failure Report**: `nodes/{node_id}/failure_report.md`"
            " holds the full compiler output of every submission the verifier"
            " rejected on this node."
        )

      # Deliberately not called "your previous rejection". The note may be from
      # a rejected submission of yours, or it may be the reason another role
      # routed the node here -- a reviewer retiring a decomposition, a prover
      # sending a defective plan back. Naming it a rejection told a fresh agent
      # it had already failed at something it had not yet attempted.
      if last_error:
        lines.extend([
            "",
            "> [!WARNING]",
            "> **Why this node is back on the queue.** Read this before you do"
            " anything else:",
            "```",
            str(last_error).strip(),
            "```",
        ])

      if latest_log_path:
        rel_log_path = (
            os.path.relpath(latest_log_path, self.root_dir)
            if os.path.isabs(latest_log_path)
            else latest_log_path
        )
        lines.extend([
            "",
            "**Last rejection**:"
            f" `cat {rel_log_path}` for the full text. This is what the tool"
            " sent back the last time a submission on this node was refused:"
            " compiler output where Lean was what refused it, the gate's own"
            " words where it was not. Not a summary.",
        ])
      lines.append("")

    # Prior decomposition history. Previously split across two sections that
    # said the same thing, one of which described backtracking as a validation
    # rejection and cited a log block that is often not rendered.
    failed_and_nodes = [an for an in target_node.and_nodes if an.failed]
    if 6 in show and (
        failed_and_nodes
        or target_node.search_metadata.get("decomposition_failed")
    ):
      # The list below is the same for everyone; the instruction on top of it
      # is not. A planner is being told what it may not submit. A reviewer is
      # being told what its predecessors already objected to -- without it a
      # reviewer sees only "Decomposition Attempts: 4" and rules as though it
      # were the first, which is how four independent reviewers found four
      # different faults in fmf_v1_002 and the root never dispatched a child.
      if phase.name == pipeline.REVIEW_PHASE:
        header = (
            "> **This is not the first plan for this node.** Every"
            " decomposition below was retired by a reviewer before you, and"
            " the blocker line is the reason it gave. Read them before you"
            " rule. You are judging whether the split in §9 answers those"
            " objections -- not whether you can find a fresh one. If the"
            " objection that killed the last attempt has been fixed and what"
            " you have left is a smaller point, say so and approve: a node"
            " that gets a new complaint every round never dispatches"
            " anything, and the planner cannot converge on a target that"
            " moves."
        )
      else:
        header = (
            "> **A different decomposition is required.** Do NOT re-propose"
            " any subgoal set listed below; the system rejects an exact"
            " repeat, including one padded with extra unused lemmas."
            " However, it is perfectly fine and encouraged to selectively reuse specific,"
            " mathematically valid subgoals from previous attempts, as long as the overall"
            " blueprint structure is revised to bypass the dead ends."
        )
      lines.extend([
          "",
          "## 6. Prior Decomposition History & Backtracking Feedback",
          "> [!IMPORTANT]",
          header,
          "",
      ])
      # Only the children that actually died are worth reading about. Pointing
      # at every dependency sends the agent to files that do not exist for the
      # siblings that succeeded.
      dep_statuses = graph.compute_statuses()
      for idx, an in enumerate(failed_and_nodes, 1):
        declared = [sig.name for sig in an.subgoal_signatures]
        dep_str = ", ".join(f"`{d}`" for d in declared) or "*(none recorded)*"
        lines.append(f"- **Attempt {idx}**: subgoals [{dep_str}]")
        if an.error_message:
          lines.append(f"  - Blocker: {an.error_message}")
        # Three outcomes, not two, and conflating the first with the others is
        # how a briefing brands a child that was never tried. A child with no
        # node in the graph was *pruned* -- the reviewer retired this plan
        # before anything was dispatched -- so it has no history, no directory
        # and nothing to read. Reporting it as a dead end with paths to
        # `nodes/<id>/findings.md` sends the agent after files that were never
        # created, and it is forbidden `find` and `grep` so it cannot discover
        # that. In fmf_v1_009/rollout_2's final root briefing 5 of 11 "dead end"
        # lines were this case and 8 of the cited paths did not exist, two of
        # them naming moment computations that were trivially provable.
        for dep in an.dependency_ids:
          if dep not in graph.nodes:
            lines.append(
                f"  - `{dep}` was never attempted. This plan was retired before"
                " the subgoal was dispatched, so there is no node, no directory"
                " and nothing to read. It is not evidence about the"
                " mathematics, and proposing it again is not a repeat."
            )
          elif dep_statuses.get(dep) != dag.NodeStatus.FAILED:
            continue
          elif self.has_failure_report(dep):
            lines.append(
                f"  - `{dep}` is a dead end — read"
                f" `nodes/{dep}/failure_report.md` for what blocked it."
            )
          else:
            # No verifier rejection, so no report. The reason it died is
            # whatever the agent that killed it wrote, which is `last_error`
            # on that node and is already in the blocker line above.
            lines.append(
                f"  - `{dep}` is a dead end. It was given up on rather than"
                " rejected by the verifier, so there is no failure report;"
                f" `nodes/{dep}/scratch/blueprint.md` and"
                f" `nodes/{dep}/findings.md` are what it left behind."
            )
        proved = [
            d
            for d in an.dependency_ids
            if dep_statuses.get(d) == dag.NodeStatus.FORMAL_PROVED
        ]
        if proved:
          lines.append(
              "  - Still proved and reusable in a new decomposition: "
              + ", ".join(f"`{d}`" for d in proved)
          )
      if not failed_and_nodes:
        lines.append(
            "- This node was marked for re-decomposition, but no prior subgoal"
            " set was recorded. Check your own failure report above."
        )
      if phase.name == pipeline.REVIEW_PHASE:
        lines.append(
            "\n**Directive**: Rule on whether the split in §9 is a genuinely"
            " different strategy from the ones above and whether it clears"
            " their blockers. A subgoal reappearing from an attempt above is"
            " only recycling if the blocker that killed it still applies;"
            " otherwise it is reuse, which is allowed.\n"
        )
      else:
        lines.append(
            "\n**Directive**: Formulate a genuinely different mathematical"
            " strategy — different intermediate lemmas, not a rewording of the"
            " same ones.\n"
        )

    # 7-8. Pointers, not contents. Findings and the plan are prose the agents
    # write and own; the proof graph stores neither, so the briefing says where
    # they are rather than pasting a copy that goes stale the moment the file
    # is appended to.
    if 7 in show:
      lines.extend([
          "## 7. Established Findings",
          (
              f"- **Your own**: `nodes/{node_id}/findings.md` — every fact"
              " established or refuted on this node so far. Read it before you"
              " start; append to it as you learn things. It is the only thing"
              " that survives to the next attempt."
          ),
          (
              "- **Others'**: `nodes/<other_id>/findings.md`, same format. Read"
              " freely; write to no directory but your own."
          ),
      ])
      # Only listed when the folder has something in it. A run with no supplied
      # material renders exactly what it did before, and nothing is ever
      # summarized into the briefing: the files are named, not pasted, so a
      # 300-page PDF in there costs one line.
      references = self.list_reference_files()
      if references:
        lines.append(
            "- **Supplied reference material**: shipped with the problem, in"
            f" `{self.REFERENCE_DIRNAME}/`. Read whatever looks relevant before"
            " you plan — it is usually the fastest route to the intended"
            " strategy. **It is unverified and it is not yours to trust.** It"
            " is human prose that nothing in this system checked: it may be"
            " wrong, it may prove a *different* statement than the one in"
            " section 2, it may skip the step that is actually hard, and it may"
            " name lemmas Mathlib does not have. Mine it for strategy. Never"
            " record something as established because a reference asserts it,"
            " and never let one stand in for your own supporting evidence or"
            " your own falsification attempts — a reference is a claim to check,"
            " which is the opposite of a fact to cite."
        )
        lines.extend(
            f"  - `{self.REFERENCE_DIRNAME}/{name}`" for name in references
        )
      lines.append("")

    if 8 in show:
      blueprint = f"nodes/{node_id}/scratch/blueprint.md"
      lines.extend([
          "## 8. Plan of Record",
          (
              f"- `{blueprint}` holds the informal proof this node is being"
              " worked against."
              if phase.name != "informal"
              else f"- Write your plan to `{blueprint}`. Everything downstream"
              " of you reads it there, and nothing else you produce is kept."
          ),
          "",
      ])

    if 9 in show:
      lines.extend(["## 9. Subtree Ledger", ""])
      statuses = graph.compute_statuses()
      active = target_node.active_and_node
      rows = list(active.dependency_ids) if active else []
      if rows:
        # The statements, not just the ids. A reviewer rules on whether these
        # close the target and whether any of them restates an ancestor, and
        # neither question can be answered from a list of names.
        lines.append(
            "The live decomposition of this node: these statements were"
            " accepted as sufficient to close the target above."
        )
        lines.append("")
        lines.append("```lean")
        for child in rows:
          child_node = graph.nodes.get(child)
          lines.append(
              (child_node.signature_str if child_node else "")
              or f"-- {child}: no formal statement yet"
          )
        lines.append("```")
        lines.append("")
        sketch = target_node.formal_sketch
        if sketch and sketch.proof and sketch.proof.tactics.strip():
          lines.append("Combined into the target by this sketch:")
          lines.append("")
          lines.append("```lean")
          lines.append(sketch.proof.tactics.strip())
          lines.append("```")
          lines.append("")
        lines.append("| Child | Status | Findings |")
        lines.append("|---|---|---|")
        for child in rows:
          st = statuses.get(child)
          lines.append(
              f"| `{child}` | {getattr(st, 'value', st or 'unknown')} |"
              f" `nodes/{child}/findings.md` |"
          )
      else:
        lines.append("*No live decomposition.*")
      lines.append("")

    if 10 in show:
      lines.extend(["## 10. Reusable Proved Lemmas", ""])
      reusable = [
          n
          for n in graph.get_accessible_nodes_data(node_id)
          if n.get("status") == dag.NodeStatus.FORMAL_PROVED.value
      ]
      if reusable:
        lines.append(
            "Already proved and callable as black-box signatures. Use them"
            " rather than re-proving or re-deriving:"
        )
        lines.append("")
        lines.append("```lean")
        for n in reusable:
          lines.append(n.get("signature", ""))
        lines.append("```")
      else:
        lines.append("*Nothing proved elsewhere is in scope for this node yet.*")
      lines.append("")

    # How to submit. One command per action, in full, with the paths already
    # filled in -- the agent copies a line rather than assembling one. Rendered
    # from `pipeline` rather than written into the prompts so that rewiring an
    # edge cannot leave a briefing advertising a command the tool refuses.
    lines.extend([
        "## How To Submit",
        "",
        "Exactly one of these ends your turn. Every path below is absolute, so"
        " it does not matter what directory you run from. Anything else you do"
        " here is not recorded.",
        "",
    ])
    for action in pipeline.allowed_actions(phase):
      if pipeline.takes_code_file(phase, action):
        scaffold = models.scaffold_filename(phase.scaffold)
        # Absolute, not `nodes/<id>/<scaffold>`: a relative path resolves
        # against whatever directory the agent's shell happens to be in, and
        # nothing guarantees that is the problem root.
        argument = shlex.quote(
            f"--code_file={self.get_scaffold_path(node_id, phase.scaffold)}"
        )
        note = (
            f"Fill in `{scaffold}` first. You do not say what it turned out to"
            " be -- the file and the Lean kernel decide that."
        )
      else:
        argument = '--reason="<one or two full sentences>"'
        note = (
            "No file is read, so the reason is the only thing the next agent"
            " inherits. A one-word reason is treated as no reason and the"
            " command is refused."
        )
      # Quoted: a Lean name may contain `'` (`foo'bar`), and unquoted the
      # quotes pair up across arguments -- `--node_id` and `--code_file` merge
      # into one, and the shell says nothing. Plain names come out unchanged.
      lines.extend([
          f"**`--action={action}`** — {pipeline.action_guidance(phase, action)}",
          "",
          "```",
          "timeout 900s state_manager_bin "
          + shlex.quote(f"--root_dir={self.root_dir}")
          + f" --action={action} "
          + shlex.quote(f"--node_id={node_id}")
          + f" {argument}",
          "```",
          "",
          note,
          "",
      ])
    lines.extend([
        "There is no default and no other action: pick the one that describes"
        " what happened. You are done when a command answers `succeeded` --"
        " that, and only that, releases your claim. If it answers `failed`,"
        " nothing was recorded, you still hold the node, and the message says"
        " what to fix. **Do not end your turn while you still hold the node:**"
        " a claim never expires on its own, and only the orchestrator can free"
        " it.",
        "",
    ])

    return "\n".join(lines) + "\n"

  def get_latest_attempt_log_path(self, node_id):
    """Returns the path to the latest *rejection* log from logs/, if any.

    Deliberately not the latest `.log`. `logs/` also holds `note` -- the
    agent's own reason for the action it took -- plus `infra`, `disproof` and
    `unreachable_abort`, and the briefing presents this path as what the tool
    sent back on the last submission. Handing an agent its predecessor's own
    prose under that label is worse than handing it nothing.
    """
    logs_dir = self.get_logs_dir(node_id)
    if not os.path.exists(logs_dir):
      return ""
    log_files = sorted(
        [f for f in os.listdir(logs_dir) if f.endswith("_rejection.log")],
        reverse=True,
    )
    if not log_files:
      return ""
    return os.path.join(logs_dir, log_files[0])

  def write_attempt_log(self, node_id, content, log_type = "rejection"):
    """Writes a trace/submission log out of band for compilation/failure history."""
    logs_dir = self.get_logs_dir(node_id)
    if not os.path.exists(logs_dir):
      os.makedirs(logs_dir, exist_ok=True)
    ts = f"{time.time():.6f}".replace(".", "_")
    log_file = os.path.join(logs_dir, f"{ts}_{log_type}.log")
    with open(log_file, "w", encoding="utf-8") as f:
      f.write(content)

  def write_final_assembled_proof(self, full_lean_code):
    """Writes the final assembled zero-sorry proof to assembled/final_proof.lean."""
    os.makedirs(self.assembled_dir, exist_ok=True)
    out_path = os.path.join(self.assembled_dir, "final_proof.lean")
    with open(out_path, "w", encoding="utf-8") as f:
      f.write(full_lean_code)
    return out_path
