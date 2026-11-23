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

r"""State manager for centralized, thread-safe and process-safe ProofGraph synchronization.

This module coordinates centralized proof graph state when multiple worker subagents
(e.g., formal provers, decomposers, verifiers) work concurrently on different sub-theorems.
It eliminates racing risks by providing atomic single-step subgoal assignment via DFS
and dual-locking (in-memory reentrant thread locks + OS file locks: fcntl on POSIX,
msvcrt on Windows).

One flag varies. A subagent runs:

  state_manager_bin --root_dir=<problem_root> --action=<action> \
      --node_id=<node_id> ( --code_file=<path> | --reason="<one or two sentences>" )

`submit` takes `--code_file`; every other action takes `--reason`. Never both.
The action *is* the outcome -- there is no second flag naming one -- and which
actions a node accepts is decided by the phase it is in, declared in
`pipeline.PIPELINE`.

`state_manager_bin` below is whatever the install exposes this module as --
an entry-point script, a shell alias, or `python3 state_manager.py`.

Usage:
  # Orchestrator: claim the next available subgoal.
  state_manager_bin --root_dir=/path/to/problem --action=claim_next

  # Prover in phase `formal` or `disprove`: hand over the scaffold. Whether it
  # proved, decomposed or disproved is read off the file and the kernel.
  state_manager_bin --root_dir=/path/to/problem --action=submit \
      --node_id=two_plus_two_eq_four \
      --code_file=nodes/two_plus_two_eq_four/proof_scaffold.lean

  # Anyone, giving up on the current phase rather than submitting.
  state_manager_bin --root_dir=/path/to/problem --action=replan \
      --node_id=two_plus_two_eq_four \
      --reason="The plan's step 3 assumes n is positive, which the statement
                does not give."
"""

from contextlib import contextmanager  # pylint: disable=g-importing-member
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple  # pylint: disable=g-importing-member
import pydantic

# OS file locking: fcntl on POSIX, msvcrt on Windows (no shared locks there, so
# every lock is exclusive -- correct, only less concurrent for readers).
try:
  import fcntl  # pylint: disable=g-import-not-at-top
  msvcrt = None
except ImportError:  # Windows
  fcntl = None
  import msvcrt  # pylint: disable=g-import-not-at-top

import actions
import claim_terminal
import dag
import graph_init_guard
import lean_verifier_tool
import lean_code_parser as models
import pipeline
import plot_style
import review_trigger
import workspace_manager

#: What a subagent may run. Each one is a phase edge in `pipeline.PIPELINE`,
#: and the phase the node is in decides which of them it is allowed -- so this
#: tuple is the union, not a menu. `submit` carries `--code_file`; every other
#: one carries `--reason`. That is the whole flag rule.
#:
#: The action *is* the outcome. There is no second flag naming one, which is
#: what makes it impossible to hand over a proof and simultaneously declare
#: something else about it.
_AGENT_ACTIONS = (
    "submit",
    "planned",
    "refute",
    "failed",
    "replan",
    "approved",
)

#: What the orchestrator may run. Read-only reports, claim management and
#: assembly; none of them is a phase edge.
_ORCHESTRATOR_ACTIONS = (
    "claim_next",
    "release",
    "release_all",
    "assemble",
    "inspect",
    "describe_workspace",
    "sync",
    "get_context",
    "get_node_report",
    "get_accessible_nodes",
    "init_graph",
    "plot_graph",
)

#: Every action this tool accepts. One tuple, read by the absl enum, by
#: argparse and by the dispatch table's completeness check, so a new action
#: cannot be reachable under one entry point and unknown under the other.
_ACTION_NAMES = _AGENT_ACTIONS + _ORCHESTRATOR_ACTIONS

assert len(set(_ACTION_NAMES)) == len(_ACTION_NAMES), (
    "an agent action and an orchestrator action share a name:"
    f" {sorted(set(_AGENT_ACTIONS) & set(_ORCHESTRATOR_ACTIONS))}"
)

try:
  from absl import app
  from absl import flags
  from absl import logging

  def _def_flag(name, define_fn, *args, **kwargs):
    assert flags is not None
    if name not in flags.FLAGS:
      return define_fn(name, *args, **kwargs)
    return flags.FLAGS[name]

  _FILEPATH = _def_flag(
      "filepath", flags.DEFINE_string, "", "Path to local proof_graph.json state file."
  )
  _ROOT_DIR = _def_flag(
      "root_dir", flags.DEFINE_string, None, "Problem root directory for node workspaces and logs."
  )
  _BACKUP_DIR = _def_flag(
      "backup_dir", flags.DEFINE_string, None, "Optional directory to mirror state into after every commit."
  )
  _ROOT_ID = _def_flag(
      "root_id", flags.DEFINE_string, None, "Root node ID for DFS traversal and metrics."
  )
  _NODE_ID = _def_flag(
      "node_id", flags.DEFINE_string, None, "Assigned node ID for recording results or release."
  )
  _PARENT_ID = _def_flag(
      "parent_id", flags.DEFINE_string, None, "Parent node ID for subgoal decompositions."
  )
  _CODE_FILE = _def_flag(
      "code_file", flags.DEFINE_string, None, "Path to local Lean 4 scaffold file."
  )
  _STATEMENT = _def_flag(
      "statement", flags.DEFINE_string, None, "Statement string for init_graph."
  )
  _REASON = _def_flag(
      "reason",
      flags.DEFINE_string,
      "",
      "Why, in one or two sentences. Required by every agent action except"
      " `submit`, which carries --code_file instead.",
  )
  _ACTION = _def_flag(
      "action",
      flags.DEFINE_enum,
      "claim_next",
      list(_ACTION_NAMES),
      "Action to perform on centralized proof state.",
  )
  _JSON_OUTPUT = _def_flag(
      "json_output", flags.DEFINE_boolean, False, "Whether to dump full JSON output."
  )
  _HAS_ABSL = True
except (ImportError, ModuleNotFoundError):
  app = None
  flags = None
  logging = None
  _HAS_ABSL = False


def _node_id_from_code_file(code_file: Optional[str]) -> str:
  """Infers the node id from a scaffold path of the form nodes/<node_id>/<file>."""
  if not code_file:
    return ""
  return os.path.basename(os.path.dirname(os.path.abspath(code_file)))


def _matches_declared(kernel_name: str, declared_name: str) -> bool:
  """Whether a name the kernel printed is the subgoal the agent declared.

  Lean prints resolved names. A problem statement that opens a `namespace` and
  never closes it -- legal, and in the immutable context, so not ours to fix --
  puts the whole scaffold inside it, and a subgoal declared `foo` comes back as
  `Ns.foo`. Matching on the namespace suffix is exactly how Lean's own
  resolution works.

  Declared subgoals only, never the base allowlist. `propext` stays an exact
  match: suffix-matching it would wave through a hypothetical `Evil.propext` at
  the one gate that makes any of this sound.
  """
  return kernel_name == declared_name or kernel_name.endswith(
      "." + declared_name
  )


def _extract_scaffold_sections(code: str) -> Dict[str, Any]:
  """Extracts structured sections from a Lean 4 scaffold file."""
  defs_match = re.search(
      r"--\s*\[START_DEFINITIONS\]([\s\S]*?)--\s*\[END_DEFINITIONS\]", code
  )
  subgoals_match = re.search(
      r"--\s*\[START_SUBGOALS\]([\s\S]*?)--\s*\[END_SUBGOALS\]", code
  )
  proof_match = re.search(
      r"--\s*\[START_PROOF\]([\s\S]*?)--\s*\[END_PROOF\]", code
  )

  defs_raw = defs_match.group(1).strip() if defs_match else ""
  clean_defs = models.strip_lean_comments(defs_raw).strip()

  subgoals_raw = subgoals_match.group(1).strip() if subgoals_match else ""
  clean_subgoals = models.strip_lean_comments(subgoals_raw).strip()

  proof_raw = proof_match.group(1).strip() if proof_match else ""
  clean_proof = models.strip_lean_comments(proof_raw).strip()

  return {
      "definitions": clean_defs,
      "subgoals": clean_subgoals,
      "proof": clean_proof,
      "missing_proof_tag": proof_match is None,
      "missing_subgoals_tag": subgoals_match is None,
      "missing_defs_tag": defs_match is None,
  }


# In-memory path-mapped thread locks for multi-threading safety
_PATH_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_GLOBAL_LOCK = threading.Lock()


def _get_path_thread_lock(filepath: str) -> threading.RLock:
  """Returns a shared in-memory reentrant lock for a state file path."""
  abs_path = os.path.abspath(filepath)
  with _GLOBAL_LOCK:
    if abs_path not in _PATH_THREAD_LOCKS:
      _PATH_THREAD_LOCKS[abs_path] = threading.RLock()
    return _PATH_THREAD_LOCKS[abs_path]


class StateManagerInput(pydantic.BaseModel):
  """Input configuration for the ProofGraph State Manager."""

  action: str = pydantic.Field(
      default="claim_next",
      description="Action to execute on the ProofGraph.",
  )
  filepath: str = pydantic.Field(
      default="",
      description="Path to local centralized proof_graph.json state file.",
  )
  code_file: Optional[str] = pydantic.Field(
      default=None,
      description="Path to the local Lean 4 scaffold file, for `submit`.",
  )
  root_dir: Optional[str] = pydantic.Field(
      default=None,
      description="Problem root directory for workspaces and logs.",
  )
  backup_dir: Optional[str] = pydantic.Field(
      default=None,
      description="Directory to mirror state into after every commit.",
  )
  root_id: Optional[str] = pydantic.Field(
      default=None,
      description="Root node ID for DFS traversal or init_graph.",
  )
  node_id: Optional[str] = pydantic.Field(
      default=None,
      description="Assigned node ID for recording results or releasing.",
  )
  parent_id: Optional[str] = pydantic.Field(
      default=None,
      description="Parent node ID for subgoal decompositions.",
  )
  statement: Optional[str] = pydantic.Field(
      default=None,
      description="Problem statement string used solely for init_graph.",
  )
  reason: str = pydantic.Field(
      default="",
      description=(
          "Why the agent is taking this action. Carried by every agent action"
          " except `submit`."
      ),
  )
  json_output: bool = pydantic.Field(
      default=False,
      description="Whether to print structured JSON output to stdout.",
  )


class StateManagerOutput(pydantic.BaseModel):
  """Structured result from a ProofGraph State Manager execution."""

  success: bool = pydantic.Field(
      default=True, description="Whether the requested state action succeeded."
  )
  action: str = pydantic.Field(default="", description="Executed action.")
  message: str = pydantic.Field(
      default="", description="Human-readable status summary."
  )
  node_id: Optional[str] = pydantic.Field(
      default=None, description="Assigned or modified node ID."
  )
  action_type: Optional[str] = pydantic.Field(
      default=None,
      description=(
          "Which subagent to dispatch for a claimed node -- the phase's role,"
          " e.g. 'lemma_proof_worker' or 'lemma_decomposition_reviewer'."
      ),
  )
  node_dir: Optional[str] = pydantic.Field(
      default=None,
      description="Absolute path to the dedicated node workspace directory.",
  )
  scaffold_path: Optional[str] = pydantic.Field(
      default=None,
      description="Absolute path to the node's proof_scaffold.lean or decomposition_scaffold.lean file.",
  )
  context_path: Optional[str] = pydantic.Field(
      default=None,
      description="Absolute path to the node's context.md briefing file.",
  )
  statement: Optional[str] = pydantic.Field(
      default=None,
      description="Informal statement or signature of assigned node.",
  )
  assembled_code: Optional[str] = pydantic.Field(
      default=None,
      description="Assembled Lean source code if action is 'assemble'.",
  )
  metadata: Dict[str, Any] = pydantic.Field(
      default_factory=dict, description="Arbitrary metadata dictionary."
  )
  error_message: Optional[str] = pydantic.Field(
      default=None, description="Error details if action failed."
  )


class VerificationOutcome(pydantic.BaseModel):
  """Result of verifying a submission against the Lean kernel.

  Carries the subgoal attribution as well as the verdict, because attribution
  depends on the kernel's axiom list and the commit step needs both.
  """

  ok: bool = pydantic.Field(default=False)
  error_message: str = pydantic.Field(default="")
  used_axioms: List[str] = pydantic.Field(default_factory=list)
  assembled: str = pydantic.Field(
      default="", description="Reconstructed local scaffold, for the failure dump."
  )
  clean_body: str = pydantic.Field(
      default="", description="Proof body with the target signature stripped."
  )
  actual_subgoals: List[models.FormalSignature] = pydantic.Field(
      default_factory=list,
      description="Declared subgoals the proof actually depends on.",
  )
  unused_subgoals: List[models.FormalSignature] = pydantic.Field(
      default_factory=list,
      description="Declared subgoals the proof never referenced.",
  )
  is_infra_failure: bool = pydantic.Field(
      default=False,
      description=(
          "The verifier never returned a verdict. Distinct from ok=False, which"
          " means Lean rejected the submission. Callers must not count this"
          " against the node: it is not evidence about the mathematics."
      ),
  )


class StorageClient:
  """Mirrors state snapshots into a backup directory on the local filesystem.

  Engaged only when `--backup_dir` is passed. Point it at a synced folder
  (a network mount, Dropbox, a bucket mounted with gcsfuse/s3fs) to watch a run
  from another machine. To publish somewhere else, subclass and override
  `sync_file`, then pass an instance as
  `ProofGraphStateManager(storage_client=...)`.

    sync_file(src_path, dest_dir)
      Publish the local file `src_path` into `dest_dir` under its basename.
      Must be ATOMIC as observed by a concurrent reader -- a half-written
      `proof_graph.json` must never be visible under its final name -- and must
      create `dest_dir` if absent. Raising is fine; `sync_backup` reports the
      failure and the run continues on local state.

  Called after every committed mutation with, in order: the state JSON, the
  rendered SVG if one exists, and the assembled `_current_proof.lean`.
  """

  def sync_file(self, src_path: str, dest_dir: str) -> None:
    """Atomically copies a local file into `dest_dir`."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src_path))
    # The pid/tid suffix keeps two workers publishing at once from colliding.
    tmp = f"{dest}.tmp_{os.getpid()}_{threading.get_ident()}"
    shutil.copyfile(src_path, tmp)
    os.replace(tmp, dest)


class RecordMessages:
  """Centralized message formatting to instruct subagents on their next actions."""

  @staticmethod
  def missing_node(node_id: str) -> str:
    return (
        f"Node '{node_id}' not found in graph.\nACTION REQUIRED: The node_id is"
        " incorrect. Check the task description or graph state. If you cannot"
        " find the correct node_id you are working on, conclude your turn and"
        " report it; do not guess at another id."
    )

  @staticmethod
  def orphaned_node(node_id: str) -> str:
    return (
        f"Node '{node_id}' is no longer actively reachable from root. Claim"
        " safely released.\nACTION REQUIRED: Your task on this node is"
        " obsolete. Please conclude your turn."
    )

  @staticmethod
  def bad_filename(expected: str, actual: str) -> str:
    return (
        f"Invalid file target. Expected path ending strictly in '*{expected}'."
        f" Found: '{actual}'.\nACTION REQUIRED: Do not rename scaffold files."
        " Please re-run the command pointing to the correct scaffold file."
    )

  @staticmethod
  def empty_submission(is_decomp: bool) -> str:
    if is_decomp:
      return (
          "Rejected: The proof glue tactics block is empty.\nACTION REQUIRED:"
          " You must write valid Lean glue tactics connecting the subgoals to"
          " the parent theorem between the '-- [START_PROOF]' and '--"
          " [END_PROOF]' tags and try again."
      )
    return (
        "Rejected: The proof tactics block is empty.\nACTION REQUIRED: You must"
        " write valid Lean tactics between the '-- [START_PROOF]' and '--"
        " [END_PROOF]' tags and try again."
    )

  @staticmethod
  def contains_placeholder(is_decomp: bool) -> str:
    if is_decomp:
      return (
          "Rejected: The proof glue tactics contain 'sorry' or 'admit'"
          " placeholders.\nACTION REQUIRED: Placeholders are ONLY allowed"
          " inside '-- [START_SUBGOALS]'. Your glue tactics in '--"
          " [START_PROOF]' must be fully solved. If you cannot solve it"
          ' directly, re-run with `--action=replan --reason="<...>"` and no'
          " scaffold is read."
      )
    return (
        "Rejected: The proof contains 'sorry' or 'admit' placeholders.\nACTION"
        " REQUIRED: You must submit a fully solved formal theorem. If you"
        " cannot prove it outright, either split it -- put the pieces between"
        " '-- [START_SUBGOALS]', each ending ':= sorry' -- or re-run with"
        ' `--action=replan --reason="<...>"` and no scaffold is read.'
    )

  @staticmethod
  def axioms_unverified(target_name: str) -> str:
    return (
        "AXIOM_CHECK_FAILED: The Lean verifier compiled your submission, but"
        f" the axiom dependencies of '{target_name}' could not be read back."
        " Because 'sorry' compiles as a warning rather than an error, the"
        " submission cannot be accepted without that check.\nACTION REQUIRED:"
        " This is a verifier-side failure, not a mistake in your proof. Simply"
        " run the tool again. If it keeps happening, say so in"
        ' `--action=replan --reason="<...>"` so the next agent is not sent back'
        " into it."
    )

  @staticmethod
  def disallowed_axioms(found: list, allowed: list) -> str:
    if "sorryAx" in found:
      return (
          "Rejected: The kernel reports this proof still depends on 'sorryAx',"
          " so some goal is left unproved even though the file"
          " compiled.\nACTION REQUIRED: Remove every 'sorry'/'admit' —"
          " including any inside helper lemmas in '-- [START_DEFINITIONS]' —"
          " and submit a fully closed proof."
      )
    return (
        f"Rejected: The proof depends on disallowed axioms: {found}.\nAllowed"
        f" axioms are: {allowed}.\nACTION REQUIRED: Tactics such as"
        " 'native_decide' discharge goals by trusting the compiler instead of"
        " the Lean kernel and are not accepted here. Replace them with a"
        " kernel-checked proof (e.g. 'decide', 'norm_num', or an explicit"
        " argument)."
    )

  @staticmethod
  def disproof_target_unused(node_id: str, target: str) -> str:
    """The other side of the disproof axiom check, and the important one.

    A disproof that does not depend on the target derived `False` from the
    ambient context: the environment is inconsistent, every proof in the tree
    is worthless until it is fixed, and the symptom is otherwise
    indistinguishable from a routine rejection.
    """
    return (
        f"ENVIRONMENT INCONSISTENT. Your disproof of '{node_id}' compiled, but"
        f" the kernel does not list `{target}` among its axioms: you derived"
        " `False` without ever using the target. Something already in scope is"
        " contradictory — a bad definition, a bogus import, or a header block"
        " promoted from an earlier node.\nACTION REQUIRED: this is not a"
        " retry, and no action on this node can fix it. Stop work and report"
        " what you found to the orchestrator in your final message, quoting"
        " this paragraph."
    )

  @staticmethod
  def unauthorized_axiom() -> str:
    return (
        "Rejected: The proof contains unauthorized 'axiom'"
        " declarations.\nACTION REQUIRED: You must prove the theorem using"
        " existing Mathlib axioms only. Remove the forced axiom and try again."
    )

  @staticmethod
  def placeholder_in_definitions() -> str:
    return (
        "Rejected: The '-- [START_DEFINITIONS]' block contains 'sorry' or"
        " 'admit'.\nACTION REQUIRED: Definitions and helper declarations must"
        " be complete. Placeholders are ONLY allowed inside '--"
        " [START_SUBGOALS]', where they become tracked subgoals. Move the"
        " unproved claim into the subgoals section, or finish it."
    )

  @staticmethod
  def unclosed_region(region: str) -> str:
    return (
        f"Rejected: the scaffold has '-- [START_{region}]' but no matching"
        f" '-- [END_{region}]'.\nEverything after the opening tag is invisible"
        " to the tool, so that entire block was read as empty and"
        " discarded.\nACTION REQUIRED: put '-- [END_"
        f"{region}]' back on its own line, spelled exactly that way, and submit"
        " again. Never delete, rename, indent, or reorder any of the six"
        " boundary tags."
    )

  @staticmethod
  def subgoals_unrecognized() -> str:
    return (
        "Rejected: '-- [START_SUBGOALS]' has content, but none of it is a"
        " subgoal declaration, so the submission was about to be read as a"
        " direct proof and your block thrown away.\nEvery subgoal there must"
        " begin with `theorem` or `lemma` and end with `:= sorry`, like"
        " this:\n  theorem my_lemma (n : Nat) : n + 0 = n := sorry\nA"
        " `private`, `protected` or `noncomputable` prefix is fine. Do NOT"
        " write `axiom`, `def`, or `example` there -- the tool rewrites your"
        " `theorem` into an `axiom` itself.\nACTION REQUIRED: rewrite the block"
        " that way, or delete its contents entirely if you meant to prove the"
        " target outright, and submit the same file again."
    )

  @staticmethod
  def placeholder_in_disproof() -> str:
    return (
        "Rejected: the disproof contains 'sorry' or 'admit'.\nA refutation has"
        " to derive `False` outright. This scaffold has no subgoal block, so"
        " there is nothing to defer the gap to.\nACTION REQUIRED: close the"
        " argument, or -- if the statement looks true after all and you cannot"
        ' refute it -- re-run with `--action=replan --reason="<...>"` and no'
        " scaffold is read."
    )

  @staticmethod
  def decomp_no_subgoals_declared() -> str:
    return (
        "Rejected decomposition: No subgoal lemmas were found in '--"
        " [START_SUBGOALS]'.\nACTION REQUIRED: A decomposition must introduce"
        " at least one intermediate lemma ending in ':= sorry'. If you proved"
        " the target outright and need no subgoals, delete everything between"
        " '-- [START_SUBGOALS]' and '-- [END_SUBGOALS]' and submit the same"
        " file again. An empty subgoal block is what makes a submission a"
        " direct proof; there is no separate action for it."
    )

  @staticmethod
  def decomp_no_subgoals_used(declared: list) -> str:
    return (
        f"Rejected decomposition: You declared subgoals {declared}, but the"
        " glue proof in '-- [START_PROOF]' does not use any of them.\nACTION"
        " REQUIRED: Either cite the subgoals in your glue proof, or -- if the"
        " target is fully proved without them -- delete everything between"
        " '-- [START_SUBGOALS]' and '-- [END_SUBGOALS]' and submit the same"
        " file again. An empty subgoal block is what makes a submission a"
        " direct proof; there is no separate action for it."
    )

  @staticmethod
  def decomp_duplicate_failed_strategy(deps: list) -> str:
    return (
        "Rejected decomposition: The proposed subgoals perfectly match a"
        f" previously failed AND-node {deps}.\nACTION REQUIRED: You must devise"
        " a fundamentally different mathematical decomposition strategy with"
        " different subgoals in the '-- [START_SUBGOALS]' section."
    )

  @staticmethod
  def decomp_active_node_duplicate(parent_id: str) -> str:
    return (
        "Subgoals perfectly match the existing active decomposition for"
        f" '{parent_id}'. Claim yielded safely without errors."
    )

  @staticmethod
  def decomp_active_collision(parent_id: str) -> str:
    return (
        f"Rejected decomposition: Node '{parent_id}' already has an active"
        " living AND-node being solved.\nACTION REQUIRED: Terminate your"
        " execution."
    )

  @staticmethod
  def global_regression_failed(compiler_error: str) -> str:
    # Deliberately does not claim a naming collision. Those are caught earlier:
    # the local scaffold already elaborates new declarations against the full
    # header, and add_proposed_subgoals rejects node-level duplicates. What is
    # left at this point is elaboration perturbation -- a new attribute,
    # instance, or type declaration changing how other proofs elaborate.
    return (
        "GLOBAL_REGRESSION_FAILED: Your submission verified on its own, but"
        " recompiling the whole proof tree with it broke a proof that was"
        f" previously passing.\n\nERROR:\n{compiler_error}\n\nACTION REQUIRED:"
        " Something in '-- [START_DEFINITIONS]' changes how other proofs"
        " elaborate -- most often a new '@[simp]' lemma that rewrites where it"
        " should not, or a new 'instance' that outranks the one those proofs"
        " relied on. Read the error above to see which declaration broke, then"
        " either drop the attribute, narrow the declaration, or prove your"
        " helper without it."
    )

  @staticmethod
  def infra_failure(stage: str, detail: str, remaining: int) -> str:
    return (
        "INFRA_FAILURE: The Lean verifier service did not return a verdict"
        f" during the {stage} check, so nothing has been decided about your"
        f" proof.\n\nDETAIL:\n{detail}\n\nACTION REQUIRED: Nothing is wrong"
        " with your submission as far as we know. Do NOT rename anything, do"
        " NOT restructure the proof, and do NOT give up on this node. Wait a"
        " moment and submit the exact same content again. This attempt has not"
        " been counted against you.\n\nRETRY BUDGET: You may resubmit"
        f" {remaining} more time(s) for this node. After that the claim is"
        " released automatically and you must conclude your turn -- keeping the"
        " node held while the service is down helps nobody."
    )

  @staticmethod
  def infra_exhausted(node_id: str, attempts: int) -> str:
    # Deliberately does not quote the underlying failure: it is already framed
    # with its own 'resubmit the same content' instruction, and embedding it
    # here would hand the agent two contradictory ACTION REQUIRED blocks. The
    # full text is in the node's log and in search_metadata['last_error'].
    return (
        "INFRA_FAILURE_EXHAUSTED: The Lean verifier service failed to return a"
        f" verdict {attempts} times in a row for '{node_id}', so the claim has"
        " been released and the node returned to the queue. The last error is"
        " recorded in this node's logs.\n\nACTION REQUIRED: Conclude your turn"
        " now. Nothing is wrong with your proof and nothing has been counted"
        " against this node -- it will be handed out again once the service"
        " recovers. Do NOT resubmit, and do NOT report this as a mathematical"
        " failure."
    )

  @staticmethod
  def placeholder_declaration_forms(placeholders: List[str]) -> str:
    lines = "\n".join(f"  - {item}" for item in placeholders)
    return (
        "Rejected: '-- [START_DEFINITIONS]' declares a constant with no"
        f" definition.\n\nOFFENDING:\n{lines}\n\nWHY: 'opaque C : T' introduces"
        " a name with no body. Nothing can ever be proved about it, so every"
        " subgoal you state in terms of C is unprovable and the entire subtree"
        " under it will die. This is 'sorry' under a different spelling, and"
        " because it compiles cleanly and adds no axiom, no later check will"
        " catch it for you. It is also permanent: the definitions block is"
        " hoisted into the shared header for every node in the tree and is"
        " never removed, so an opaque constant committed here poisons the"
        " environment for the rest of the run.\n\nACTION REQUIRED: Pick exactly"
        " one.\n  (a) Give it a real body -- 'def C : T := <the actual"
        " construction>'. This is the only option that keeps it here.\n  (b) It"
        " is a PROPOSITION you cannot prove yet: delete it from"
        " '-- [START_DEFINITIONS]' and restate it inside"
        " '-- [START_SUBGOALS]' as 'lemma <name> : <statement> := sorry'. The"
        " tool turns that into a tracked subgoal and another agent is"
        " dispatched to prove it.\n  (c) It is DATA you cannot construct yet"
        " (a graph, an embedding, a sequence): there is no way to state"
        " subgoals about an object that does not exist, so your decomposition"
        " is wrong. Do not declare it away. Submit a different split, or record"
        " '--action=replan' with the reason."
    )

  @staticmethod
  def proved_declaration_forms(proofs: List[str], admitted) -> str:
    lines = "\n".join(f"  - {item}" for item in proofs)
    allowed = ", ".join(f"'{k}'" for k in sorted(admitted))
    return (
        "Rejected: '-- [START_DEFINITIONS]' contains proved statements, not"
        f" definitions.\n\nOFFENDING:\n{lines}\n\nWHY: the definitions block is"
        " hoisted into the shared header for every node in the tree, and the"
        " header is append-only -- it can never be edited, reordered or taken"
        " back. Anything you put there is compiled into every other node's"
        " environment for the rest of the run, including after this branch is"
        " abandoned. So it holds only the vocabulary a sibling cannot state its"
        f" own goal without: {allowed}.\n\nA 'theorem' or a 'lemma' is a proof"
        " step, and a proof step has an owner. Putting one here makes it"
        " nobody's: it is not reviewed, it is not attributed, it cannot be"
        " retracted when the branch that wrote it dies, and it re-elaborates on"
        " every build forever.\n\nACTION REQUIRED: Pick exactly one for each"
        " offending declaration.\n  (a) Only you need it -- prove it inline in"
        " your own proof body, where it costs nothing and disappears with"
        " you.\n  (b) A sibling needs it -- move it to '-- [START_SUBGOALS]' as"
        " '<name> : <statement> := sorry'. It becomes a tracked node with its"
        " own prover, and that is also the only form that can be retracted"
        " later.\n  Keep the definitions here and resubmit."
    )

  @staticmethod
  def header_retracted(names: List[str]) -> str:
    listed = ", ".join(f"'{n}'" for n in names)
    return (
        "Rejected: your submission cites shared definitions that were retracted"
        f" while it was being verified.\n\nRETRACTED: {listed}\n\nWHY: a"
        " reviewer retired the decomposition that introduced them, so they are"
        " no longer in any node's header. Your proof compiled against a header"
        " that no longer exists, and recording it would put a proof in the tree"
        " that cannot be rebuilt.\n\nACTION REQUIRED: nothing is wrong with your"
        " mathematics, and this is not something you did. The"
        " '-- [START_IMMUTABLE_CONTEXT]' region of your scaffold on disk is now"
        " out of date -- it still shows those declarations and they are gone."
        " Do not use them: either write what you need out longhand in your own"
        " proof, or declare it yourself in '-- [START_DEFINITIONS]' (the names"
        " are free again). Then verify and submit."
    )

  @staticmethod
  def banned_definition_forms(banned: List[str], unnamed: List[str]) -> str:
    lines = "\n".join(f"  - {item}" for item in banned + unnamed)
    return (
        "Rejected: '-- [START_DEFINITIONS]' contains declarations that change"
        " how *other* proofs elaborate.\n\nOFFENDING:\n"
        f"{lines}\n\nWHY: Every definitions block is hoisted above every proof"
        " in the assembled tree, so these forms silently reach proofs that were"
        " verified long before yours.\n\nACTION REQUIRED: Remove them. Helper"
        " 'def' and 'theorem'/'lemma' declarations are fully supported and can"
        " express anything these can. If you need a namespace or an option for"
        " one declaration only, use the scoped forms 'open Foo in' or"
        " 'set_option opt val in' immediately before it."
    )

  @staticmethod
  def definition_name_collision(collisions: List[str]) -> str:
    lines = "\n".join(f"  - {item}" for item in collisions)
    return (
        "Rejected: '-- [START_DEFINITIONS]' redeclares names that already exist"
        f" in the proof tree.\n\nCOLLISIONS:\n{lines}\n\nACTION REQUIRED:"
        " Either rename your helper, or delete it and use the existing"
        " declaration -- it is already in scope in your scaffold header."
    )

  @staticmethod
  def graph_rejected(err_msg: str) -> str:
    return (
        f"Rejected: The proof graph refused this decomposition.\n\n{err_msg}\n\n"
        "ACTION REQUIRED: Rename the offending lemma, or restructure the"
        " decomposition so it does not depend on a goal that already depends on"
        " this one."
    )

  @staticmethod
  def state_moved(node_id: str) -> str:
    return (
        f"Rejected: The graph state for '{node_id}' changed while your"
        " submission was being verified, so committing it could overwrite newer"
        " work.\nACTION REQUIRED: Nothing is wrong with your proof. Re-read your"
        " context and scaffold, then submit again."
    )

  @staticmethod
  def scaffold_assembly_failed(err_msg: str) -> str:
    return (
        "SCAFFOLD_ASSEMBLY_FAILED: Failed to structurally assemble the Lean"
        f" scaffold.\n\nERROR:\n{err_msg}\n\nACTION REQUIRED: Formatting error"
        " in your submission. Please ensure your structural boundary tags are"
        " correct and try again."
    )

  @staticmethod
  def signature_mismatch(expected: str) -> str:
    return (
        "SIGNATURE_MISMATCH: The theorem signature you proved does not"
        " strictly match the expected target theorem.\nExpected:"
        f" {expected}\n\nACTION REQUIRED: You MUST NOT alter the theorem"
        " signature. Copy the exact expected signature."
    )

  @staticmethod
  def verification_dumped(
      node_id: str, compiler_error: str, dump_path: str, is_decomp: bool
  ) -> str:
    target_name = "Sketch" if is_decomp else "Proof"
    # Collisions against Mathlib, and against names the graph does not track
    # (a subgoal named after another node's helper), reach the agent only as
    # raw compiler output. In rollout_1 unframed errors were what drove agents
    # to rewrite working proofs, so name the failure mode explicitly.
    hint = ""
    if "already been declared" in compiler_error:
      hint = (
          "\n\nNAME COLLISION: The error above means a name you introduced is"
          " already taken -- by Mathlib, by the shared preamble, or by another"
          " node's helper. Nothing is wrong with your mathematics. Rename that"
          " one declaration, or drop it and use the existing definition, which"
          " is already in scope. Do NOT alter the target theorem signature."
      )
    return (
        f"VERIFICATION_FAILED: Lean compiler rejected the {target_name}"
        f" submission for '{node_id}'.\nERROR:\n{compiler_error}\n\nACTION"
        " REQUIRED: Read the Lean compiler error above carefully. If needed,"
        f" inspect the exact reconstructed code dumped to '{dump_path}'. Fix"
        " the syntactic or logical error in your scaffold workspace file and"
        f" execute the tool again.{hint}"
    )

  @staticmethod
  def success(node_id: str, is_decomp: bool) -> str:
    if is_decomp:
      return (
          "Successfully recorded and verified formal decomposition sketch for"
          f" '{node_id}'."
      )
    return f"Successfully recorded and verified formal theorem for '{node_id}'."

  @staticmethod
  def no_scaffold(node_id: str, phase_name: Optional[str], allowed) -> str:
    """Refusal for `submit` on a phase that hands out nothing to submit."""
    return (
        f"Refused: '{node_id}' is in phase '{phase_name}', which writes no"
        " Lean, so there is no scaffold to read and `--code_file` means"
        " nothing here.\nNothing was recorded and your claim is still"
        " held.\nACTION REQUIRED: re-run with one of:"
        f" {', '.join('--action=' + a for a in allowed)}. The '## How To"
        " Submit' section of your `context.md` says what each one means."
    )

  @staticmethod
  def disproof_recorded(node_id: str) -> str:
    return (
        f"Refutation verified for '{node_id}': the kernel derived `False` from"
        " the target taken as an axiom, so the statement is false as written."
        f" '{node_id}' is now marked failed and its parent will backtrack."
        " Claim safely released.\nACTION REQUIRED: conclude your turn."
    )


def _parse_scaffold_file(
    code_file: str, expected_mode: str
) -> Tuple[bool, Dict[str, Any]]:
  if not code_file:
    return False, {
        "error": "MISSING_CODE_FILE: '--code_file' is strictly required."
    }
  if not os.path.exists(code_file):
    return False, {
        "error": f"FILE_NOT_FOUND: Code file '{code_file}' does not exist."
    }

  expected_filename = models.scaffold_filename(expected_mode)
  actual_filename = os.path.basename(code_file)
  if actual_filename != expected_filename:
    return False, {
        "error": RecordMessages.bad_filename(expected_filename, actual_filename)
    }

  try:
    with open(code_file, "r", encoding="utf-8") as f:
      file_content = f.read()
  except Exception as e:
    return False, {"error": f"Error reading code file: {e}"}

  sections = _extract_scaffold_sections(file_content)

  # A boundary tag the agent deleted or mistyped makes its whole region
  # invisible: the regex spans start-to-end, so it misses, the section reads as
  # empty, and the block is dropped without a word. Checked here rather than at
  # the point of use because every consumer downstream sees an empty string and
  # cannot tell an empty block from a missing end tag.
  for region, flag in (
      ("DEFINITIONS", "missing_defs_tag"),
      ("SUBGOALS", "missing_subgoals_tag"),
      ("PROOF", "missing_proof_tag"),
  ):
    if f"[START_{region}]" in file_content and sections[flag]:
      return False, {"error": RecordMessages.unclosed_region(region)}

  # Always parsed, never conditioned on an expected mode: a non-empty subgoal
  # block is what *makes* the submission a decomposition. Asking the caller
  # which one it is would put the question to the only party that cannot
  # already see the answer.
  subgoals = []
  if sections.get("subgoals"):
    try:
      subgoals = models.parse_subgoal_declarations(sections["subgoals"])
    except Exception as e:
      return False, {"error": f"Failed to parse subgoals: {e}"}
    # An unrecognised block parses to [], which routes the submission as a
    # direct proof and throws the block away in silence.
    if not subgoals:
      return False, {"error": RecordMessages.subgoals_unrecognized()}
  sections["parsed_subgoals_list"] = subgoals

  return True, sections


_MAX_INFRA_RETRIES = int(os.environ.get("LEMMA_MAX_INFRA_RETRIES", "3"))


def _infra_retries_remaining(node: dag.ProofGraphNode) -> int:
  """How many more times the agent may resubmit before the claim is released."""
  return max(
      0, _MAX_INFRA_RETRIES - node.search_metadata.get("infra_failures", 0) - 1
  )


def _existing_declared_names(graph: dag.ProofGraph) -> Dict[str, str]:
  """Maps every identifier already declared in the header to its provenance."""
  registry: Dict[str, str] = {}
  for decl in models.split_declarations(
      graph.environment.shared_definitions or ""
  ):
    name = models.extract_declaration_name(decl)
    if name:
      registry.setdefault(name, "the shared preamble")
  for block in graph.environment.dynamic_headers:
    # A retired block is not in the header any more, so its names are not taken.
    # Leaving them reserved would make a re-planned branch collide with a
    # declaration nothing can see.
    if block.retired:
      continue
    for decl in models.split_declarations(block.content or ""):
      name = models.extract_declaration_name(decl)
      if name:
        registry.setdefault(name, f"node '{block.node_id or '?'}'")
  return registry


def _retired_header_references(
    graph: dag.ProofGraph,
    outcome: "VerificationOutcome",
    definitions_block: str,
) -> Optional[str]:
  """Refuses a submission that cites a declaration retired while it compiled.

  The one race `actions._retire_header_blocks` structurally cannot cover.
  Verification runs unlocked, so a worker can compile against a header, have a
  reviewer retire one of its blocks -- correctly, nothing in the graph referred
  to it yet -- and then commit a proof that cites a name no longer in any
  header. The revision check above does not fire, because that node's own
  revision never moved, and the regression check is skipped outright for a
  submission with no definitions block. The result is a proof recorded as
  verified that no later build can reproduce, discovered at final assembly.

  Checked here because commit is the first moment both facts are visible under
  one lock: what was retired, and what this submission actually cites.
  """
  retired: set = set()
  for block in graph.environment.dynamic_headers:
    if not block.retired:
      continue
    for decl in models.split_declarations(block.content or ""):
      name = models.extract_declaration_name(decl)
      if name:
        retired.add(name)
  # A retired name is released back to the pool, so a later block may legally
  # have taken it. Then it is in the header and there is nothing to refuse.
  retired -= set(_existing_declared_names(graph))
  if not retired:
    return None

  cited = "\n".join(
      [outcome.clean_body or "", definitions_block or ""]
      + [
          sig.statement
          for sig in outcome.actual_subgoals + outcome.unused_subgoals
      ]
  )
  # Word boundaries, not the substring match retirement uses: there the safe
  # error is a false hit, here it is a false miss. A spurious rejection costs a
  # round trip on a proof that was fine.
  hits = sorted(
      name
      for name in retired
      if re.search(rf"\b{re.escape(name)}\b", cited)
  )
  if not hits:
    return None
  return RecordMessages.header_retracted(hits)


def _validate_definitions_block(
    graph: dag.ProofGraph, node_id: str, definitions_block: str
) -> Optional[str]:
  """Rejects definitions blocks that can reach beyond the submitting node.

  Runs before any Lean round trip, on both the prove and decompose paths. Lean
  would catch the collisions on its own -- the local scaffold elaborates new
  declarations against the full header -- but only after a compile, and only
  with a generic 'has already been declared'. The elaboration-affecting forms it
  would not catch at all without recompiling the entire tree.
  """
  if not definitions_block or not definitions_block.strip():
    return None

  report = models.classify_definitions_block(definitions_block)
  # Before the banned forms, because the two rejections share a report but not a
  # remedy: a banned form has to be deleted, a bodyless one has to be built,
  # proved as a subgoal, or planned around.
  if report.placeholders:
    return RecordMessages.placeholder_declaration_forms(report.placeholders)
  # Before the banned-forms branch: a block holding both a `lemma` and an `open`
  # should be told about the `lemma` first, because moving it is the larger edit
  # and the two remedies are independent.
  if report.proofs:
    return RecordMessages.proved_declaration_forms(
        report.proofs, models.HEADER_ADMITS
    )
  if report.is_rejected:
    return RecordMessages.banned_definition_forms(report.banned, report.unnamed)

  registry = _existing_declared_names(graph)
  collisions = []
  for name in report.declared_names:
    # A node re-recording its own block is a resubmission, not a collision, so
    # the gate lets it through and to_lean_header collapses the duplicate at
    # assembly. Note the local scaffold does NOT collapse it -- build_lean_scaffold
    # emits the header and the definitions block verbatim -- so an identical
    # resubmission is still rejected, by Lean, as a redeclaration.
    owner = registry.get(name)
    if owner and owner != f"node '{node_id}'":
      collisions.append(f"'{name}' is already declared by {owner}")
    elif name in graph.nodes and name != node_id:
      collisions.append(
          f"'{name}' is already the name of a subgoal in the tree"
      )
  if collisions:
    return RecordMessages.definition_name_collision(collisions)
  return None


def _global_check_mode(definitions_block: str) -> str:
  """Decides how much of the tree the pre-commit regression check must cover.

  The full recompile used to run on every submission, and on the 111-node tree
  in scratch/rollout_1 it dominated the run -- 108 of those 111 submissions
  carried no definitions block at all, so the majority of a 3.6h run went to a
  check that was structurally incapable of finding anything. (Absolute timings
  off the shared verifier swing widely between sessions -- the same full
  recompile has measured 76s, 167s and 295s -- so the ratios below are the
  stable numbers, not the seconds.)

  What the check can actually catch is narrower than it looks. A colliding name
  is already caught by the local scaffold, which elaborates the submission
  against the full header (step 1); a colliding subgoal is already caught by
  add_proposed_subgoals. The one failure mode left is a new declaration changing
  how *other* nodes' proofs elaborate, and only the forms in
  _ESCALATING_HEADER_FORMS can do that -- _validate_definitions_block rejects
  the rest outright.

  The smoke tier is deliberately thin, and worth being precise about: its
  assembly carries the same header lines, in the same order, as the local
  scaffold at step 1 -- the only difference is the trailing theorem, root's
  signature under a 'sorry' instead of the submitting node's real proof. So the
  one thing it adds is 'the root statement still elaborates'. That is not
  vacuous (a plain 'def' shadowing a short name the root statement uses will
  fail here and pass step 1), but it does not cover any other node's statement.
  Widening it to every node's signature was measured on rollout_1 at 19% of the
  full recompile against the smoke tier's 0.6%, which is no longer a cheap tier.

  Returns:
    'skip'  -- nothing is added to the header, so nothing can regress.
    'smoke' -- header + root statement only, ~0.6% of a full recompile.
    'full'  -- recompile every node; the block perturbs elaboration.
  """
  if not definitions_block or not definitions_block.strip():
    return "skip"
  report = models.classify_definitions_block(definitions_block)
  return "full" if report.needs_full_tree_check else "smoke"


def _validate_proof_syntactic_gates(
    node: dag.ProofGraphNode, proof_tactics: str, definitions_block: str
) -> Optional[str]:
  """Validates proof tactic submissions against syntactic rule constraints.

  Ensures proofs are non-empty, do not contain placeholders (sorry/admit/axiom),
  and rejects submissions on parents that already have active subgoals.
  """
  # Reject record_proof on a node that already has living AND-nodes
  if node.active_and_node is not None:
    return (
        "Rejected: Node is a decomposed parent with active living AND-nodes."
        " Parent theorems resolve automatically when child subgoals are proved."
    )
  if not proof_tactics:
    return RecordMessages.empty_submission(is_decomp=False)
  if re.search(r"\b(sorry|admit)\b", proof_tactics):
    return RecordMessages.contains_placeholder(is_decomp=False)
  # The definitions block was previously exempt from the placeholder check, so
  # a helper lemma proved 'by sorry' there slipped past to the kernel gate.
  if re.search(r"\b(sorry|admit)\b", definitions_block or ""):
    return RecordMessages.placeholder_in_definitions()
  if re.search(r"^\s*axiom\b", proof_tactics, flags=re.MULTILINE) or re.search(
      r"^\s*axiom\b", definitions_block or "", flags=re.MULTILINE
  ):
    return RecordMessages.unauthorized_axiom()
  return None


def _validate_decomp_syntactic_gates(
    node: dag.ProofGraphNode,
    sketch_code: str,
    used_subgoals: list,
    definitions_block: str = "",
) -> Tuple[Optional[str], bool]:
  """Validates proof tactic submissions against syntactic rule constraints.

  Ensures proofs are non-empty, do not contain placeholders (sorry/admit/axiom),
  and rejects submissions on parents that already have active subgoals.
  """
  if not sketch_code:
    return RecordMessages.empty_submission(is_decomp=True), False

  # Placeholders and axioms are legal only inside '-- [START_SUBGOALS]', which
  # the tool converts to axioms itself. This gate previously never saw the
  # definitions block at all, so an 'axiom' or 'sorry' there went unchecked.
  if re.search(r"\b(sorry|admit)\b", definitions_block or ""):
    return RecordMessages.placeholder_in_definitions(), False
  if re.search(r"^\s*axiom\b", sketch_code, flags=re.MULTILINE) or re.search(
      r"^\s*axiom\b", definitions_block or "", flags=re.MULTILINE
  ):
    return RecordMessages.unauthorized_axiom(), False

  # An AND-node with no dependencies resolves its parent to FORMAL_PROVED with
  # nothing supporting it, so an empty decomposition can never be recorded.
  if not used_subgoals:
    return RecordMessages.decomp_no_subgoals_declared(), False

  # Statements, not names. In fmf_v1_004 an agent walked straight past the
  # name-keyed version of this gate by appending `_new` to all five subgoals
  # of an already-retired decomposition and resubmitting the statements
  # byte-identical. It cost a full review cycle, and the only thing that
  # caught it was a reviewer reading the mathematics.
  proposed_deps = {
      models.normalize_string(sig.statement) for sig in used_subgoals
  }

  def _declared(and_node: dag.AndNode) -> set:
    """Every subgoal the submission declared, not just the attributed ones.

    dependency_ids holds only what the kernel actually used, so comparing
    against it let an agent re-propose a failed strategy simply by adding one
    unreferenced lemma.
    """
    return {
        models.normalize_string(sig.statement)
        for sig in and_node.subgoal_signatures
    }

  # Check for duplicate previously failed strategies.
  #
  # One exception, and it is narrow. An `EVIDENCE:` retirement means the
  # reviewer ruled the split sound and asked for it back with a gap in the
  # blueprint filled, so the identical set is the compliant answer and blocking
  # it deadlocks the node -- there is no attempt budget to end the loop.
  # The pass is spent the first time it is used: a second match means the set
  # has now died twice, the planner already had its one revision, and
  # resubmitting the same statements a third time is not a plan.
  matches = [
      prior for prior in node.and_nodes
      if prior.failed and _declared(prior) == proposed_deps
  ]
  if matches and not (
      len(matches) == 1 and matches[0].retired_for == "evidence"
  ):
    return (
        RecordMessages.decomp_duplicate_failed_strategy(
            matches[0].dependency_ids
        ),
        False,
    )

  # Check for active collisions
  active_and = node.active_and_node
  if active_and is not None:
    if _declared(active_and) == proposed_deps:
      return RecordMessages.decomp_active_node_duplicate(node.id), True
    # A live decomposition and a different one arriving is always a collision:
    # a node with a live AND-node is waiting on its children, not claimable for
    # `formal`, so the submitter is working from a stale snapshot. Re-planning
    # goes through `--action=replan` from `review`, which fails the live
    # AND-node first, and then there is no active one to collide with.
    return RecordMessages.decomp_active_collision(node.id), False

  if re.search(r"\b(sorry|admit)\b", sketch_code):
    return RecordMessages.contains_placeholder(is_decomp=True), False

  return None, False


class ProofGraphStateManager:
  """Manages centralized, dual-locked serialization of ProofGraph for multi-worker safety."""

  def __init__(
      self,
      filepath: str,
      root_dir: Optional[str] = None,
      backup_dir: Optional[str] = None,
      root_id: Optional[str] = None,
      storage_client: Optional[StorageClient] = None,
  ):
    """Initializes the ProofGraphStateManager.

    Args:
      filepath: Path to local proof_graph.json state file.
      root_dir: Optional problem root directory for workspaces.
      backup_dir: Optional directory to mirror state into after every commit.
      root_id: Optional root theorem node ID.
      storage_client: Optional dedicated storage client for filesystem sync.
    """
    self.filepath = filepath
    self.root_dir = (
        os.path.abspath(root_dir)
        if root_dir
        else os.path.dirname(os.path.abspath(filepath))
    )
    self.workspace_mgr = workspace_manager.WorkspaceManager(self.root_dir)
    self.nodes_dir: str = os.path.join(self.root_dir, "nodes") if self.root_dir else ""
    self.backup_dir = backup_dir
    self.root_id = root_id
    self.storage_client = storage_client or StorageClient()


  def _clear_claim_metadata(self, node: dag.ProofGraphNode) -> None:
    """Clears claim-related search metadata on a node."""
    node.search_metadata.pop("claim_timestamp", None)
    node.search_metadata["processing"] = False

  def _write_node_log(self, node_id: str, content: str, log_type: str = "rejection") -> None:
    """Logs rejection/failure/success reasons out of band using the WorkspaceManager."""
    self.workspace_mgr.write_attempt_log(node_id, content, log_type)

  def read_graph(self) -> dag.ProofGraph:
    """Reads the ProofGraph with shared thread lock and OS shared file lock.

    Returns:
      Deserialized ProofGraph.

    Raises:
      ValueError: If the state file does not exist or is empty.
    """
    if not os.path.exists(self.filepath):
      raise ValueError(f"State file {self.filepath} does not exist.")

    thread_lock = _get_path_thread_lock(self.filepath)
    with thread_lock, self._os_lock(exclusive=False):
      with open(self.filepath, "r", encoding="utf-8") as f:
        content = f.read()
      if not content.strip():
        raise ValueError(f"State file {self.filepath} is empty.")
      return dag.ProofGraph.model_validate_json(content)

  @contextmanager
  def _os_lock(self, exclusive: bool) -> Generator[None, None, None]:
    """Takes an OS lock on a dedicated lock file beside the state file.

    The lock deliberately does not live on the state file itself: commits
    replace that file rather than rewriting it, and a lock held on a replaced
    inode no longer excludes anyone.
    """
    lock_path = f"{self.filepath}.lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
      if fcntl is not None:
        fcntl.flock(
            lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        )
        try:
          yield
        finally:
          fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
      else:
        # msvcrt locks a byte range from the current position; lock byte 0
        # (beyond EOF is allowed). LK_NBLCK + retry gives a blocking lock
        # without LK_LOCK's 10-second give-up.
        fd = lock_file.fileno()
        while True:
          try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            break
          except OSError:
            time.sleep(0.05)
        try:
          yield
        finally:
          os.lseek(fd, 0, os.SEEK_SET)
          msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

  def _write_graph_atomic(self, graph: dag.ProofGraph) -> None:
    """Serializes the graph to a temp file, then renames it into place.

    Truncate-then-write leaves a window in which a crash empties the state file
    irrecoverably: every later action fails on 'State file is empty'.
    """
    directory = os.path.dirname(os.path.abspath(self.filepath))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=".proof_graph_", suffix=".tmp"
    )
    try:
      with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
        tmp_file.write(graph.model_dump_json(indent=2))
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
      os.replace(tmp_path, self.filepath)
    except BaseException:
      if os.path.exists(tmp_path):
        os.unlink(tmp_path)
      raise

  @contextmanager
  def lock_and_modify(
      self, initial_graph: Optional[dag.ProofGraph] = None
  ) -> Generator[dag.ProofGraph, None, None]:
    """Yields ProofGraph with exclusive thread lock and OS exclusive file lock.

    Serializes and flushes the mutated graph upon exiting the context.

    Args:
      initial_graph: Optional initial graph to write if file is not found.

    Yields:
      The mutable ProofGraph object.
    """
    thread_lock = _get_path_thread_lock(self.filepath)
    with thread_lock, self._os_lock(exclusive=True):
      content = ""
      if os.path.exists(self.filepath):
        with open(self.filepath, "r", encoding="utf-8") as f:
          content = f.read()

      if not content.strip():
        if not initial_graph:
          raise ValueError(
              f"State file {self.filepath} does not exist or is empty, and no"
              " initial graph was provided."
          )
        graph = initial_graph
      else:
        graph = dag.ProofGraph.model_validate_json(content)

      if not self.backup_dir and "backup_dir" in graph.graph_metadata:
        self.backup_dir = graph.graph_metadata["backup_dir"]
      elif self.backup_dir:
        graph.graph_metadata["backup_dir"] = self.backup_dir

      yield graph

      # Resolved, not read straight off the flag: `--root_id` is supplied on
      # `init_graph` and on no other action, so `self.root_id` is None for
      # every submit, declare and claim, and the prune has to run on all of
      # them.
      #
      # Here rather than at the moment a decomposition is retired. One call
      # site instead of four -- `declare` with a retiring review,
      # `propagate_failures`, the decomp-failed path and anything added later
      # all funnel through this transaction, so there is nowhere for a
      # retirement to leak past.
      #
      # Ordering matters: prune first, metrics after. The other order computes
      # metrics for nodes about to be deleted.
      #
      # Deliberately not inside the try below. The metrics call is wrapped
      # because a malformed graph should degrade to a logged warning rather
      # than fail the write. If the prune raises, something is wrong with
      # reachability itself and committing a graph in that state is worse than
      # failing the action.
      prune_root = self._get_root_id(graph)
      if prune_root:
        graph.prune_orphans(prune_root)

      if self.root_id and self.root_id in graph.nodes:
        try:
          graph.recalculate_empirical_metrics(self.root_id)
        except Exception as e:  # pylint: disable=broad-except
          if logging is not None:
            logging.error("Failed to recalculate empirical metrics: %s", e)
          else:
            print(f"Failed to recalculate empirical metrics: {e}")

      self._write_graph_atomic(graph)
      committed = graph

    # Plotting shells out to `dot` and the backup sync copies several files. Neither needs the lock, and holding it across them serialized
    # every other worker behind bookkeeping.
    try:
      self.generate_plot(committed)
    except Exception as e_plot:  # pylint: disable=broad-except
      if logging is not None:
        logging.warning("Failed to generate plot: %s", e_plot)

    if self.backup_dir:
      self.sync_backup(committed)

  def initialize_graph(self, initial_graph: dag.ProofGraph) -> bool:
    """Atomically initializes the state file with the initial graph."""
    with self.lock_and_modify(initial_graph=initial_graph):
      pass
    return True

  def claim_next_subgoal(
      self,
      root_id: Optional[str] = None,
  ) -> Optional[Dict[str, Any]]:
    """Atomically finds and claims the next available subgoal via DFS in one step.

    This eliminates multi-worker racing risks by selecting and locking the next
    available unproved leaf lemma within a single atomic exclusive lock section.

    There is no attempt budget. Nothing in this walk kills a node: a node dies
    only when the informal prover says `failed` or the kernel derives `False`
    from it. A subtree that keeps failing keeps being re-planned, because a hard
    problem is supposed to keep running.

    Args:
      root_id: Optional root theorem ID to anchor DFS traversal.

    Returns:
      Dictionary with assigned node details (id, statement, status, cost)
      if an eligible node was claimed, or None if no unclaimed subgoals remain.
    """
    with self.lock_and_modify() as graph:
      target_root = root_id or self.root_id
      if not target_root and graph.nodes:
        target_root = next(iter(graph.nodes.keys()))

      if not target_root or target_root not in graph.nodes:
        return None

      status_map = graph.propagate_failures()
      now = time.time()
      visited = set()
      # Shared for the whole walk: the graph's shape does not change during one
      # scan, and a chain of `blocked` parents would otherwise re-measure the
      # same subtree once per level.
      replan_memo: Dict[str, float] = {}

      def _dispatch(node: dag.ProofGraphNode, status: Any) -> Optional[Dict[str, Any]]:
        """Claims `node` for whatever its phase says, or declines.

        The phase is the only thing consulted, and it is a function of the
        graph. Nothing here decides a role: every caller that moves a node
        moves its phase, and the role, the scaffold and the briefing are all
        read back off that one value, so they cannot disagree.
        """
        # A claim never expires: liveness is the orchestrator's job, and it
        # recovers a dead worker's node with --action=release.
        if node.search_metadata.get("processing", False):
          return None
        phase = pipeline.phase_for(node)
        role = pipeline.role_for(node)
        if role is None:
          return None

        node.search_metadata["claim_timestamp"] = now
        node.search_metadata["processing"] = True
        # Counts dispatches. `revision` counts recorded actions, and the pair
        # is the only thing that distinguishes a node being worked from a node
        # being handed out over and over: a worker that dies, or returns
        # without a successful submission, leaves the graph byte-identical to
        # what it claimed. Four claim/release cycles were indistinguishable
        # from one, which is why the orchestrator in rollout fmf005_sim went
        # looking for the scheduler source code instead of noticing.
        node.search_metadata["dispatch_count"] = (
            node.search_metadata.get("dispatch_count", 0) + 1
        )
        # Planning turns, counted separately, because that is rework and the
        # total is not: a node proved on the third Lean attempt was worked
        # three times but re-planned once. review_trigger averages this over a
        # subtree to decide whether an ancestor should reconsider, and using
        # total dispatches there would read every productive branch as churn.
        if phase.name == pipeline.ENTRY_PHASE:
          node.search_metadata["informal_dispatches"] = (
              node.search_metadata.get("informal_dispatches", 0) + 1
          )
        return {
            "node_id": node.id,
            "action_type": role,
            "scaffold_role": pipeline.scaffold_mode(phase),
            "phase": phase.name,
            "statement": (
                node.informal_problem.statement
                if node.informal_problem
                else node.definitions or node.id
            ),
            "status": (
                status.value if isinstance(status, dag.NodeStatus) else str(status)
            ),
            "heuristic_cost": node.search_metadata.get(
                "live_heuristic_cost",
                node.search_metadata.get("heuristic_cost", 5.0),
            ),
            "dispatch_count": node.search_metadata["dispatch_count"],
            "revision": node.search_metadata.get("revision", 0),
        }

      def _dfs_find_claim(curr_id: str) -> Optional[Dict[str, Any]]:
        if curr_id in visited or curr_id not in graph.nodes:
          return None
        visited.add(curr_id)
        node = graph.nodes[curr_id]

        curr_status = status_map.get(curr_id, node.status)
        if curr_status in (
            dag.NodeStatus.FORMAL_PROVED,
            dag.NodeStatus.FAILED,
            dag.NodeStatus.BROKEN,
        ):
          # `children_done`. Two things track a node: its mathematical status,
          # derived from the kernel and its children, and its workflow phase, a
          # stored field. The status has just resolved; the phase has not, and
          # nothing else will move it -- every other phase is left by an agent
          # running an action, but `blocked` has no role, so no agent is ever
          # dispatched there and no action is ever run on it. This is the edge
          # that takes it out of the machine (`phase = None`).
          #
          # Without it the two contradict each other: a proved node still
          # reporting phase `blocked`, waiting on children that finished.
          # Placed above the `and_nodes` block so a subtree that has just
          # closed is never pulled back for review by `subtree_stalled`
          # instead.
          if pipeline.current_phase_name(node) == "blocked":
            pipeline.apply_predicate_edge(node, "children_done", None)
          return None

        # 1. If node is a decomposed parent, manage its active living AND-node
        if node.and_nodes:
          # Find the active non-failed living AND-node
          active_and = node.active_and_node

          if active_and is not None:
            # Review gates the descent. A freshly decomposed node sits in
            # `review` with a role of its own, and walking past it into the
            # children would dispatch subgoals that the reviewer may be about
            # to retire -- work thrown away, and a briefing written against a
            # plan that no longer exists. If a reviewer already holds it, the
            # subtree stays shut until the verdict lands.
            if pipeline.current_phase_name(node) == pipeline.REVIEW_PHASE:
              return _dispatch(node, curr_status)

            # `subtree_stalled`. The decomposition review happens once, when
            # the plan is proposed and there is no evidence of how it plays
            # out; this is the second look. A branch whose nodes are each being
            # re-planned several times over without anything closing is worth
            # a second opinion. It buys a review and nothing else -- only the
            # reviewer's `replan` retires a plan, and only the planner can call
            # its own goal dead.
            #
            # Returning here is what serializes the reviews: the walk stops at
            # the first node over threshold and the gate above then holds the
            # subtree shut until the verdict lands, so a `replan` up top makes
            # every deeper review moot instead of racing it. WHICH node that is
            # is decided in review_trigger, not here -- a node reads the worst
            # of its own children, so a localized stall reads sharp at its
            # parent and dilutes into a mean further up, and this top-down walk
            # lands near it, while a plan that is failing everywhere reads high
            # at the root and stops there.
            if pipeline.current_phase_name(node) == "blocked" and (
                review_trigger.should_review(
                    graph, node, replan_memo, status_map
                )
            ):
              pipeline.apply_predicate_edge(
                  node, "subtree_stalled", pipeline.REVIEW_PHASE
              )
              return _dispatch(node, curr_status)

            for dep_id in active_and.dependency_ids:
              dep_node = graph.nodes.get(dep_id)
              if not dep_node:
                continue
              dep_status = status_map.get(dep_id, dep_node.status)
              if dep_status != dag.NodeStatus.FORMAL_PROVED:
                found = _dfs_find_claim(dep_id)
                if found is not None:
                  return found

            # Decomposed parents with an active living AND-node are NEVER claimed for prove!
            # They automatically resolve to FORMAL_PROVED once all child subgoals finish.
            return None
          else:
            # Every decomposition has failed. This is the `children_failed`
            # predicate: nothing an agent declared moved the node, arithmetic on
            # the graph did, so the edge is applied here rather than waiting for
            # a submission that will never come.
            #
            # It goes back to the planner. A node with no live decomposition is
            # not a dead node -- it is a node whose every plan so far was
            # wrong, which is the planner's problem to solve, and the planner
            # is the only role allowed to conclude there is no plan.
            #
            # Guarded on `blocked`, exactly like `children_done` above and
            # `subtree_stalled` beside it, and the guard is load-bearing rather
            # than tidy. `and_nodes` keeps its retired entries forever, so this
            # branch is true for the rest of the node's life once a reviewer
            # retires its last plan. Unguarded it re-fires on every walk and
            # overwrites whatever phase the node had reached: the planner
            # declares `planned`, the phase moves to `formal`, the next
            # `claim_next` drags it back to `informal`, and the node is
            # dispatched to a planner forever. That livelock has exactly one
            # exit, `--action=failed`, so the bug can only be escaped by
            # killing a node whose mathematics was fine.
            if pipeline.current_phase_name(node) == "blocked":
              pipeline.apply_predicate_edge(node, "children_failed", "informal")
            return _dispatch(node, curr_status)

        # 2. Leaf node handling (no and_nodes)
        return _dispatch(node, curr_status)

      claimed_subgoal = _dfs_find_claim(target_root)
      if claimed_subgoal:
        ws_info = self.workspace_mgr.provision_node_workspace(
            node_id=claimed_subgoal["node_id"],
            graph=graph,
            phase=pipeline.get_phase(claimed_subgoal["phase"]),
            root_id=target_root,
        )
        claimed_subgoal["node_dir"] = ws_info["node_dir"]
        claimed_subgoal["scaffold_path"] = ws_info["scaffold_path"]
        claimed_subgoal["context_path"] = ws_info["context_path"]
      return claimed_subgoal

  def release_subgoal(self, node_id: str) -> bool:
    """Clears the processing flag on a node so it can be claimed again.

    This is an orchestrator recovery action, not an agent action: because a
    claim never expires, releasing is the only way a node held by a crashed
    worker becomes claimable. There is no ownership check -- any caller may
    release any node.

    Args:
      node_id: ID of the subgoal to release.

    Returns:
      True if the node exists, False otherwise.
    """
    with self.lock_and_modify() as graph:
      if node_id not in graph.nodes:
        return False

      node = graph.nodes[node_id]

      self._clear_claim_metadata(node)
      return True

  def release_all_subgoals(self) -> List[str]:
    """Clears the processing flag on all nodes in the graph so stranded nodes can be claimed again.

    This is an orchestrator recovery action when workers crash or disconnect.

    Returns:
      List of node IDs that were actively processing and got released.
    """
    released_nodes = []
    with self.lock_and_modify() as graph:
      for node_id, node in graph.nodes.items():
        if node.search_metadata and node.search_metadata.get("processing"):
          self._clear_claim_metadata(node)
          released_nodes.append(node_id)
    return released_nodes

  def _finalize_failure(
      self,
      node: dag.ProofGraphNode,
      err_msg: str,
      is_decomp: bool,
      log_type: str = "rejection",
      count_attempt: bool = True,
  ) -> Tuple[bool, str]:
    node.search_metadata["last_error"] = err_msg

    self._write_node_log(node.id, err_msg, log_type)

    # An infra failure is not evidence about the proof, so it must not consume a
    # budget or enter the fix history. In scratch/rollout_1 it did both: 66 read
    # timeouts burned attempts and were replayed back to agents as prior
    # mathematical rejections, which is what drove them to keep renaming lemmas.
    if not count_attempt:
      # Bounded, though: the retry advice is only sound while the outage is
      # brief. Left unbounded it becomes a spin against a dead service, and an
      # agent that exhausts its own turn budget mid-loop strands the claim --
      # nothing expires it, so the node is lost for the rest of the run.
      streak = node.search_metadata.get("infra_failures", 0) + 1
      node.search_metadata["infra_failures"] = streak
      if streak >= _MAX_INFRA_RETRIES:
        node.search_metadata["infra_failures"] = 0
        self._clear_claim_metadata(node)
        return True, RecordMessages.infra_exhausted(node.id, streak)
      return False, err_msg

    # Reaching here means the verifier answered, so any outage is over.
    node.search_metadata["infra_failures"] = 0

    # context.md points agents (and, on backtracking, their parents) at this
    # file. Appending here is what makes that reference real.
    attempt_no = (
        node.search_metadata.get(
            "decomp_attempts" if is_decomp else "proof_attempts", 0
        )
        + 1
    )
    self.workspace_mgr.append_failure_report(
        node.id, err_msg, attempt_no, is_decomp
    )

    attempt_key = "decomp_attempts" if is_decomp else "proof_attempts"
    node.search_metadata[attempt_key] = (
        node.search_metadata.get(attempt_key, 0) + 1
    )
    # DO NOT set decomposition_failed and DO NOT clear the claim, and do not
    # move the phase: a rejected submission leaves the node exactly where it
    # was, still held, so the agent can fix a typo and resubmit.
    return False, err_msg

  def _finalize_orphaned_node(
      self, node: dag.ProofGraphNode
  ) -> Tuple[bool, str]:
    self._write_node_log(
        node.id,
        "Node is completely unreachable from root; aborted.",
        "unreachable_abort",
    )
    self._clear_claim_metadata(node)
    return True, RecordMessages.orphaned_node(node.id)

  def _finalize_early_success(
      self, node: dag.ProofGraphNode, msg: str
  ) -> Tuple[bool, str]:
    self._clear_claim_metadata(node)
    return True, msg

  def _apply_submission(
      self,
      graph: dag.ProofGraph,
      node_id: str,
      outcome: "VerificationOutcome",
      definitions_block: str,
      is_decomp: bool,
  ) -> None:
    """Applies a verified submission to `graph`.

    Called twice per submission: once against a throwaway copy so the global
    regression check sees the real post-commit state, and once against the live
    graph on commit. Sharing one implementation is what makes those two agree;
    previously the test copy received only the new header, so the node under
    test was still rendered ':= by sorry' and the check proved nothing.

    Raises:
      ValueError: from add_proposed_subgoals on a cycle, signature mismatch, or
        reuse of a failed lemma. Callers run this against the test copy first so
        those surface as a framed rejection rather than an unhandled exception.
    """
    node = graph.nodes[node_id]
    stmt = node.informal_problem.statement if node.informal_problem else node_id

    if is_decomp:
      # A node with a live decomposition is waiting on its children and is not
      # claimable, so the cheap gates already refused this as a collision. If
      # one appeared since, another worker won the race and this submission is
      # stale: refuse rather than silently leaving the node with two live
      # decompositions, which would double-count against its parent.
      if node.active_and_node is not None:
        raise ValueError(
            f"'{node_id}' gained a live decomposition while yours was"
            " verifying. Yours was NOT recorded. Re-read the node's context"
            " before doing anything else."
        )

      graph.add_proposed_subgoals(
          node_id, outcome.actual_subgoals, definitions_block=definitions_block
      )
      if outcome.unused_subgoals and node.and_nodes:
        node.and_nodes[-1].subgoal_signatures.extend(outcome.unused_subgoals)

      node.formal_sketch = models.FormalProofSketch(
          signature=models.FormalSignature(name=node_id, statement=stmt),
          proof=models.FormalProof(tactics=outcome.clean_body),
      )
    else:
      node.formal_theorem = models.FormalTheorem(
          signature=models.FormalSignature(name=node_id, statement=stmt),
          proof=models.FormalProof(tactics=outcome.clean_body),
      )

    if definitions_block and definitions_block.strip():
      node.definitions = definitions_block
      # Append-only (design decision D1): never remove, reorder, or replace.
      graph.environment.dynamic_headers.append(
          models.SharedHeaderBlock(
              content=definitions_block.strip(),
              node_id=node_id,
              and_node_id=(
                  node.and_nodes[-1].id if (is_decomp and node.and_nodes) else ""
              ),
          )
      )

  def _finalize_success(
      self,
      graph: dag.ProofGraph,
      node: dag.ProofGraphNode,
      outcome: "VerificationOutcome",
      definitions_block: str,
      is_decomp: bool,
  ) -> Tuple[bool, str]:
    self._apply_submission(
        graph, node.id, outcome, definitions_block, is_decomp
    )

    # The result, not an action. Nobody declared it -- which of the two this is
    # was decided by whether the scaffold's subgoal block was empty, and it is
    # applied here, at the one point where that is settled and verified.
    pipeline.apply_result(node, "decomposed" if is_decomp else "proved")

    self._clear_claim_metadata(node)

    attempt_key = "decomp_attempts" if is_decomp else "proof_attempts"
    node.search_metadata[attempt_key] = (
        node.search_metadata.get(attempt_key, 0) + 1
    )

    # Cleared for both kinds: a node that just produced a verified artifact is
    # no longer a dead end, and leaving the flag set makes compute_statuses
    # report FAILED while assemble emits the proof.
    node.search_metadata["decomposition_failed"] = False
    node.search_metadata["last_error"] = ""
    node.search_metadata["infra_failures"] = 0
    # Bumped so a submission verified against an older snapshot of this node is
    # rejected at commit rather than overwriting newer work.
    node.search_metadata["revision"] = (
        node.search_metadata.get("revision", 0) + 1
    )
    return True, RecordMessages.success(node.id, is_decomp)

  def _reject(
      self,
      node_id: str,
      err_msg: str,
      is_decomp: bool,
      log_type: str = "rejection",
      count_attempt: bool = True,
  ) -> Tuple[bool, str]:
    """Records a rejection against the live graph under an exclusive lock."""
    with self.lock_and_modify() as graph:
      node = graph.nodes.get(node_id)
      if not node:
        return False, RecordMessages.missing_node(node_id)
      return self._finalize_failure(
          node, err_msg, is_decomp, log_type, count_attempt
      )

  def _get_root_id(self, graph: dag.ProofGraph) -> str:
    """Resolves the root theorem id for reachability and assembly.

    Prefers the explicitly configured root; falls back to insertion order only
    when none was supplied, since init_graph inserts the root first.
    """
    if self.root_id and self.root_id in graph.nodes:
      return self.root_id
    return next(iter(graph.nodes), "")

  def _trace_verification(
      self,
      node_id: str,
      phase: str,
      stamp: str,
      code: str,
      target_name: str,
      result: "lean_verifier_tool.LeanVerifierOutput",
      extra: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Records one Lean round trip to disk; never lets tracing break a proof."""
    try:
      summary = {
          "phase": phase,
          "node_id": node_id,
          "target_name": target_name,
          "is_valid": result.is_valid,
          # None here means the axiom line could not be read, which is itself a
          # rejection reason -- distinct from an empty list.
          "used_axioms": result.used_axioms,
          "error_message": result.error_message,
          "info_messages": result.info_messages,
          "code_line_count": len(code.splitlines()),
          "sorry_occurrences": code.count("sorry"),
      }
      if extra:
        summary.update(extra)
      self.workspace_mgr.write_verification_trace(
          node_id, phase, code, summary, attempt_stamp=stamp
      )
    except Exception as e:  # pylint: disable=broad-except
      if logging is not None:
        logging.warning("Failed to write verification trace: %s", e)

  def _verify_before_commit(
      self,
      graph: dag.ProofGraph,
      node_id: str,
      definitions: str,
      proof_text: str,
      is_decomposition: bool,
      subgoals_text: str = "",
  ) -> VerificationOutcome:
    """Verifies a submission without touching `graph`.

    Runs the local scaffold through the kernel, applies the axiom allowlist,
    attributes subgoals from the kernel's axiom list, then applies the full
    prospective mutation to a throwaway copy and checks it against the rest of
    the tree. How much of the tree that last step covers depends on what the
    submission adds to the shared header -- see _global_check_mode. A proof body
    cannot reach other nodes, so a submission with no definitions block skips it
    entirely.

    Performs no mutation and holds no lock, so the caller can run it outside the
    critical section.
    """
    target_node = graph.nodes[node_id]
    target_name = target_node.canonical_name
    # `parse_formal_statement` raises on a declaration with no depth-0 colon,
    # and this path -- unlike the pre-parse at _validate_subgoal_names -- did
    # not catch it, so a malformed subgoal reached the top-level handler and
    # came back as a bare `str(e)` instead of an actionable rejection.
    try:
      declared_subgoals = (
          models.parse_subgoal_declarations(subgoals_text)
          if is_decomposition
          else []
      )
    except ValueError as e:
      return VerificationOutcome(
          error_message=f"Could not parse the subgoals block: {e}"
      )

    # Step 1: the target signature is immutable.
    expected_sig = target_node.signature_str
    is_valid_sig, proof_body_or_err, clean_declaration = (
        models.extract_proof_with_expected_signature(proof_text, expected_sig)
    )
    if not is_valid_sig:
      return VerificationOutcome(
          error_message=RecordMessages.signature_mismatch(expected_sig)
      )

    axiom_subgoals = (
        models.convert_subgoals_to_axioms(subgoals_text)
        if is_decomposition
        else ""
    )
    try:
      local_assemble = models.build_lean_scaffold(
          immutable_context=graph.environment.to_lean_header(),
          definitions=definitions,
          subgoals=axiom_subgoals,
          proof=clean_declaration,
          role="decompose" if is_decomposition else "prove",
      )
    except Exception as e:  # pylint: disable=broad-except
      return VerificationOutcome(
          error_message=RecordMessages.scaffold_assembly_failed(str(e))
      )

    # Step 2: local compile.
    ver_res = lean_verifier_tool.run_lean_verifier(
        lean_verifier_tool.LeanVerifierInput(
            code=local_assemble, target_name=target_name
        )
    )
    trace_stamp = f"{time.time():.6f}".replace(".", "_")
    self._trace_verification(
        node_id, "local", trace_stamp, local_assemble, target_name, ver_res
    )
    if not ver_res.is_valid:
      if ver_res.is_infra_failure:
        return VerificationOutcome(
            error_message=RecordMessages.infra_failure(
                "local",
                ver_res.error_message or "",
                _infra_retries_remaining(target_node),
            ),
            is_infra_failure=True,
            assembled=local_assemble,
        )
      return VerificationOutcome(
          # Left unwrapped; the caller adds the dump path.
          error_message=ver_res.error_message or "",
          used_axioms=ver_res.used_axioms or [],
          assembled=local_assemble,
      )

    # Step 3: the axiom allowlist is the only real soundness gate, because
    # 'sorry' compiles as a warning and so is_valid alone proves nothing.
    if ver_res.used_axioms is None:
      return VerificationOutcome(
          error_message=RecordMessages.axioms_unverified(target_name),
          assembled=local_assemble,
      )
    used_axs = list(ver_res.used_axioms)

    base_allowed = graph.effective_allowed_axioms
    # Subgoals are deliberately injected as axioms for this check, so the
    # kernel is expected to depend on them. Suffix-matched, unlike base_allowed
    # just above -- see _matches_declared.
    declared = (
        [sig.name for sig in declared_subgoals] if is_decomposition else []
    )
    disallowed = sorted(
        ax
        for ax in set(used_axs)
        if ax not in base_allowed
        and not any(_matches_declared(ax, name) for name in declared)
    )
    if disallowed:
      return VerificationOutcome(
          error_message=RecordMessages.disallowed_axioms(
              disallowed, sorted(base_allowed)
          ),
          used_axioms=used_axs,
          assembled=local_assemble,
      )

    # Step 4: attribute subgoals from the kernel's axiom list.
    actual_subgoals: List[models.FormalSignature] = []
    unused_subgoals: List[models.FormalSignature] = []
    if is_decomposition:
      actual_subgoals = [
          sig
          for sig in declared_subgoals
          if any(_matches_declared(ax, sig.name) for ax in used_axs)
      ]
      if not actual_subgoals:
        # Textual fallback, for subgoals the elaborator may have inlined before
        # the kernel recorded a dependency.
        actual_subgoals = [
            sig
            for sig in declared_subgoals
            if re.search(
                rf"\b{re.escape(sig.name)}(?=[^a-zA-Z0-9_\']|$)", proof_text
            )
        ]
      if not actual_subgoals:
        return VerificationOutcome(
            error_message=RecordMessages.decomp_no_subgoals_used(
                [sig.name for sig in declared_subgoals]
            ),
            used_axioms=used_axs,
            assembled=local_assemble,
        )
      used_names = {sig.name for sig in actual_subgoals}
      unused_subgoals = [
          sig for sig in declared_subgoals if sig.name not in used_names
      ]

    outcome = VerificationOutcome(
        ok=True,
        used_axioms=used_axs,
        assembled=local_assemble,
        clean_body=proof_body_or_err,
        actual_subgoals=actual_subgoals,
        unused_subgoals=unused_subgoals,
    )

    # Step 5: apply the whole prospective mutation to a throwaway copy, then
    # check it against the rest of the tree -- see _global_check_mode for how
    # much of the tree that check covers.
    test_graph = copy.deepcopy(graph)
    try:
      self._apply_submission(
          test_graph, node_id, outcome, definitions, is_decomposition
      )
    except ValueError as e:
      # Cycle / signature mismatch / reuse of a failed lemma. Catching these
      # here keeps the live graph untouched and turns them into a framed
      # rejection instead of an unhandled exception.
      return VerificationOutcome(
          error_message=RecordMessages.graph_rejected(str(e)),
          used_axioms=used_axs,
          assembled=local_assemble,
      )

    target_root = self._get_root_id(test_graph)
    root_node = test_graph.nodes.get(target_root)
    root_target_name = target_root
    if root_node is not None:
      root_target_name = root_node.canonical_name

    mode = _global_check_mode(definitions)
    if mode == "skip":
      return outcome

    if mode == "full":
      global_assemble = test_graph.assemble_current_proof(target_root)
    else:
      global_assemble = test_graph.assemble_header_smoke_test(target_root)

    ver_global_res = lean_verifier_tool.run_lean_verifier(
        lean_verifier_tool.LeanVerifierInput(
            code=global_assemble, target_name=root_target_name
        )
    )
    self._trace_verification(
        node_id,
        "global",
        trace_stamp,
        global_assemble,
        root_target_name,
        ver_global_res,
        extra={"assembled_from_root": target_root, "global_check_mode": mode},
    )
    if ver_global_res.is_infra_failure:
      return VerificationOutcome(
          error_message=RecordMessages.infra_failure(
              "global regression",
              ver_global_res.error_message or "",
              _infra_retries_remaining(target_node),
          ),
          is_infra_failure=True,
          used_axioms=used_axs,
          assembled=global_assemble,
      )
    if not ver_global_res.is_valid:
      return VerificationOutcome(
          error_message=RecordMessages.global_regression_failed(
              ver_global_res.error_message or ""
          ),
          used_axioms=used_axs,
          assembled=global_assemble,
      )

    return outcome

  def _dump_failed_assembly(self, node_id: str, assembled: str) -> str:
    """Writes the reconstructed source for a failed verification to logs/."""
    if not self.nodes_dir or not assembled:
      return ""
    dump_path = os.path.join(
        self.nodes_dir, node_id, "logs", "failed_verification_assembled.lean"
    )
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
      f.write(assembled)
    if self.root_dir:
      return os.path.relpath(dump_path, self.root_dir)
    return dump_path

  def _prepare_submission(
      self, node_id: str, is_decomp: bool
  ) -> Tuple[Optional[dag.ProofGraph], int, Optional[Tuple[bool, str]]]:
    """Loads a snapshot and runs the checks that need no Lean compile.

    Returns (snapshot, baseline_revision, early_result). When early_result is
    set the caller must return it unchanged.
    """
    try:
      snapshot = self.read_graph()
    except ValueError as e:
      return None, 0, (False, str(e))

    node = snapshot.nodes.get(node_id)
    if not node:
      return None, 0, (False, RecordMessages.missing_node(node_id))

    target_root = self._get_root_id(snapshot)
    if target_root and not snapshot.is_node_reachable_from_root(
        node_id, target_root
    ):
      with self.lock_and_modify() as graph:
        live = graph.nodes.get(node_id)
        if not live:
          return None, 0, (False, RecordMessages.missing_node(node_id))
        return None, 0, self._finalize_orphaned_node(live)

    return snapshot, node.search_metadata.get("revision", 0), None

  def _commit_submission(
      self,
      node_id: str,
      outcome: VerificationOutcome,
      definitions_block: str,
      is_decomp: bool,
      baseline_revision: int,
      missing_tag_note: str = "",
  ) -> Tuple[bool, str]:
    """Re-validates against the live graph, then commits or rejects."""
    dump_rel = ""
    if not outcome.ok:
      dump_rel = self._dump_failed_assembly(node_id, outcome.assembled)

    with self.lock_and_modify() as graph:
      node = graph.nodes.get(node_id)
      if not node:
        return False, RecordMessages.missing_node(node_id)

      if not outcome.ok:
        err_msg = outcome.error_message
        # Bare compiler output; anything from RecordMessages is already framed.
        if "ACTION REQUIRED" not in err_msg:
          err_msg = RecordMessages.verification_dumped(
              node_id, err_msg, dump_rel, is_decomp=is_decomp
          )
          if missing_tag_note:
            err_msg += missing_tag_note
        return self._finalize_failure(
            node,
            err_msg,
            is_decomp,
            log_type="infra" if outcome.is_infra_failure else "rejection",
            count_attempt=not outcome.is_infra_failure,
        )

      # Optimistic concurrency: verification ran unlocked, so confirm this node
      # has not moved underneath us before writing.
      if node.search_metadata.get("revision", 0) != baseline_revision:
        return self._finalize_failure(
            node, RecordMessages.state_moved(node_id), is_decomp
        )

      target_root = self._get_root_id(graph)
      if target_root and not graph.is_node_reachable_from_root(
          node_id, target_root
      ):
        return self._finalize_orphaned_node(node)

      # The same gate ran in PHASE C, but against a snapshot taken before
      # verification, so it cannot see a block another worker committed since.
      # This is the only check that catches two nodes introducing the same name
      # concurrently: each verifies cleanly against a header that does not yet
      # contain the other's block, and the revision check above does not fire
      # because neither node's own revision moved. Both would commit, both
      # declarations would land in to_lean_header, and -- dynamic_headers being
      # append-only (D1) -- every later compile on every node would fail with no
      # recovery short of hand-editing proof_graph.json.
      defs_err = _validate_definitions_block(
          graph, node_id, definitions_block
      ) or _retired_header_references(graph, outcome, definitions_block)
      if defs_err:
        return self._finalize_failure(node, defs_err, is_decomp)

      try:
        return self._finalize_success(
            graph, node, outcome, definitions_block, is_decomp
        )
      except ValueError as e:
        # Should already have been caught against the test copy; treat a
        # late failure as a rejection rather than losing the transaction.
        return self._finalize_failure(
            node, RecordMessages.graph_rejected(str(e)), is_decomp
        )

  def record_lean(
      self, code_file: str, node_id: Optional[str] = None
  ) -> Tuple[bool, str]:
    """`--action=submit`: the one verb that reads a file.

    What the submission turns out to be is *derived*, never accepted: a
    non-empty subgoal block means `decomposed`, an empty one means `proved`,
    and a compiled `False` in the disproof scaffold means `disproved`. The agent
    hands over one file and says nothing about it.

    That is the point. When an agent could name its own outcome alongside a
    file, a prover passed `proved` on the abort path and had a real proof
    discarded under a success message. Here there is nothing to name: giving up
    is a different `--action`, so the two cannot be said in one call.
    """
    node_id = node_id or _node_id_from_code_file(code_file)

    graph = self.read_graph()
    node = graph.nodes.get(node_id)
    if not node:
      return True, RecordMessages.missing_node(node_id)

    phase = pipeline.phase_for(node)
    mode = pipeline.scaffold_mode(phase)
    if mode is None:
      return False, RecordMessages.no_scaffold(
          node_id, phase.name, pipeline.allowed_actions(phase)
      )

    is_valid, sections = _parse_scaffold_file(code_file, mode)
    if not is_valid:
      return self._reject(node_id, sections["error"], is_decomp=False)

    if mode == pipeline.SCAFFOLD_DISPROVE:
      return self._record_disproof(node_id, sections)
    if sections["parsed_subgoals_list"]:
      return self._record_decomposition(node_id, sections)
    return self._record_proof(node_id, sections)

  def _record_proof(
      self, node_id: str, sections: Dict[str, Any]
  ) -> Tuple[bool, str]:
    """Records a direct proof: empty subgoal block, target closed outright."""
    proof_tactics = sections["proof"]
    definitions_block = sections["definitions"]

    # --- PHASE C: Snapshot & Cheap Gates (no Lean compile) ---
    snapshot, baseline_revision, early = self._prepare_submission(
        node_id, is_decomp=False
    )
    if early is not None:
      return early
    assert snapshot is not None

    gate_err = _validate_proof_syntactic_gates(
        snapshot.nodes[node_id], proof_tactics, definitions_block
    ) or _validate_definitions_block(snapshot, node_id, definitions_block)
    if gate_err:
      return self._reject(node_id, gate_err, is_decomp=False)

    # --- PHASE D: Lean Kernel Verification (UNLOCKED) ---
    try:
      outcome = self._verify_before_commit(
          graph=snapshot,
          node_id=node_id,
          proof_text=proof_tactics,
          definitions=definitions_block,
          is_decomposition=False,
      )
    except Exception as e:  # pylint: disable=broad-except
      return self._reject(
          node_id, f"Verification exception: {e}", is_decomp=False
      )

    # --- PHASE E: Commit under exclusive lock ---
    note = ""
    if sections.get("missing_proof_tag"):
      note = "\n  [Formatting Note]: Missing '-- [START_PROOF]' tag in scaffold."
    return self._commit_submission(
        node_id, outcome, definitions_block, False, baseline_revision, note
    )

  def _record_decomposition(
      self, parent_id: str, sections: Dict[str, Any]
  ) -> Tuple[bool, str]:
    """Records a decomposition: non-empty subgoal block, target closed modulo it."""
    sketch_code = sections["proof"]
    definitions_block = sections["definitions"]
    used_subgoals = sections["parsed_subgoals_list"]

    # --- PHASE C: Snapshot & Cheap Gates (no Lean compile) ---
    snapshot, baseline_revision, early = self._prepare_submission(
        parent_id, is_decomp=True
    )
    if early is not None:
      return early
    assert snapshot is not None

    gate_err, early_success = _validate_decomp_syntactic_gates(
        snapshot.nodes[parent_id], sketch_code, used_subgoals, definitions_block
    )
    if early_success:
      with self.lock_and_modify() as graph:
        node = graph.nodes.get(parent_id)
        if not node:
          return False, RecordMessages.missing_node(parent_id)
        return self._finalize_early_success(node, gate_err or "")
    if gate_err:
      return self._reject(parent_id, gate_err, is_decomp=True)

    defs_err = _validate_definitions_block(
        snapshot, parent_id, definitions_block
    )
    if defs_err:
      return self._reject(parent_id, defs_err, is_decomp=True)

    # --- PHASE D: Lean Kernel Verification (UNLOCKED) ---
    try:
      outcome = self._verify_before_commit(
          graph=snapshot,
          node_id=parent_id,
          proof_text=sketch_code,
          definitions=definitions_block,
          subgoals_text=sections.get("subgoals", ""),
          is_decomposition=True,
      )
    except Exception as e:  # pylint: disable=broad-except
      return self._reject(
          parent_id, f"Sketch verification exception: {e}", is_decomp=True
      )

    # --- PHASE E: Commit under exclusive lock ---
    note = ""
    if sections.get("missing_subgoals_tag") or sections.get("missing_proof_tag"):
      note = (
          "\n  [Formatting Note]: Missing '-- [START_SUBGOALS]' or"
          " '-- [START_PROOF]' boundary tag."
      )
    return self._commit_submission(
        parent_id, outcome, definitions_block, True, baseline_revision, note
    )

  def _record_disproof(
      self, node_id: str, sections: Dict[str, Any]
  ) -> Tuple[bool, str]:
    """Records a refutation: `False` derived from the target taken as an axiom.

    Same five phases as a proof, and deliberately so -- but the gates differ at
    every one, which is why it is a sibling rather than a flag on
    `_record_proof`:

      * There is no definitions block and no subgoal block to validate. The
        disproof scaffold does not render either region, because everything a
        refutation derives is derived from a proposition the tree believes is
        false, and none of it may reach the shared header.
      * The obligation is `theorem <id>_disproof : False`, identical for every
        node, rather than the node's own signature.
      * The target's own name joins the axiom allowlist -- assuming it is the
        whole method -- where for a proof it would be the one axiom that makes
        the result worthless.
      * Nothing enters the header, so the global regression check that
        `_verify_before_commit` runs has nothing to check and is skipped.

    A success is terminal and negative: the node is marked dead with the
    witness, and its parent backtracks. That is the same treatment a
    decomposition whose children all died receives, because the consequence for
    the parent is the same.
    """
    proof_text = sections["proof"]
    if not proof_text:
      return self._reject(
          node_id, RecordMessages.empty_submission(is_decomp=False),
          is_decomp=False,
      )
    if re.search(r"\b(sorry|admit)\b", proof_text):
      return self._reject(
          node_id, RecordMessages.placeholder_in_disproof(), is_decomp=False
      )
    if re.search(r"^\s*axiom\b", proof_text, flags=re.MULTILINE):
      return self._reject(
          node_id, RecordMessages.unauthorized_axiom(), is_decomp=False
      )

    snapshot, baseline_revision, early = self._prepare_submission(
        node_id, is_decomp=False
    )
    if early is not None:
      return early
    assert snapshot is not None

    target = snapshot.nodes[node_id]
    obligation = models.disproof_obligation(node_id)
    is_valid_sig, _, clean_declaration = (
        models.extract_proof_with_expected_signature(proof_text, obligation)
    )
    if not is_valid_sig:
      return self._reject(
          node_id, RecordMessages.signature_mismatch(obligation),
          is_decomp=False,
      )

    assembled = models.build_lean_scaffold(
        immutable_context=models.build_disproof_header(
            snapshot.environment.to_lean_header(), target.signature_str
        ),
        definitions="",
        subgoals="",
        proof=clean_declaration,
        role=models.SCAFFOLD_DISPROVE,
        hints=False,
    )
    disproof_name = obligation.split()[1]
    ver_res = lean_verifier_tool.run_lean_verifier(
        lean_verifier_tool.LeanVerifierInput(
            code=assembled, target_name=disproof_name
        )
    )
    self._trace_verification(
        node_id,
        "disproof",
        f"{time.time():.6f}".replace(".", "_"),
        assembled,
        disproof_name,
        ver_res,
    )

    if not ver_res.is_valid:
      if ver_res.is_infra_failure:
        return self._reject(
            node_id,
            RecordMessages.infra_failure(
                "disproof",
                ver_res.error_message or "",
                _infra_retries_remaining(target),
            ),
            is_decomp=False,
            log_type="infra",
            count_attempt=False,
        )
      dump_rel = self._dump_failed_assembly(node_id, assembled)
      return self._reject(
          node_id,
          RecordMessages.verification_dumped(
              node_id, ver_res.error_message or "", dump_rel, is_decomp=False
          ),
          is_decomp=False,
      )

    if ver_res.used_axioms is None:
      return self._reject(
          node_id, RecordMessages.axioms_unverified(disproof_name),
          is_decomp=False,
      )
    base_allowed = snapshot.effective_allowed_axioms
    # Same resolved-name problem as the proof path: under an open namespace the
    # target axiom this header injects comes back as `Ns.<target>`.
    disallowed = sorted(
        ax
        for ax in set(ver_res.used_axioms)
        if ax not in base_allowed
        and not _matches_declared(ax, target.canonical_name)
    )
    if disallowed:
      return self._reject(
          node_id,
          RecordMessages.disallowed_axioms(disallowed, sorted(base_allowed)),
          is_decomp=False,
      )
    if not any(
        _matches_declared(ax, target.canonical_name)
        for ax in ver_res.used_axioms
    ):
      return self._reject(
          node_id,
          RecordMessages.disproof_target_unused(node_id, target.canonical_name),
          is_decomp=False,
      )

    with self.lock_and_modify() as graph:
      node = graph.nodes.get(node_id)
      if not node:
        return False, RecordMessages.missing_node(node_id)
      if node.search_metadata.get("revision", 0) != baseline_revision:
        return self._finalize_failure(
            node, RecordMessages.state_moved(node_id), is_decomp=False
        )
      pipeline.apply_result(node, "disproved")
      node.search_metadata["decomposition_failed"] = True
      # Distinct from an ordinary death, and the distinction is load-bearing:
      # an exhausted node is a name nobody could close and renaming it is the
      # documented recovery, while this one is a *proposition* the kernel has
      # shown to be false. `add_proposed_subgoals` reads this to refuse the
      # rename.
      node.search_metadata["disproved"] = True
      node.search_metadata["last_error"] = (
          f"DISPROVED: `False` was derived from `{target.canonical_name}`"
          " taken as an axiom, so the statement is false as written."
      )
      self._write_node_log(node_id, assembled, "disproof")
      self._clear_claim_metadata(node)
      return True, RecordMessages.disproof_recorded(node_id)

  def assemble_proof(
      self,
      target_node_id: Optional[str] = None,
  ) -> str:
    """Reads the state graph and generates a standalone compileable Lean file.

    Args:
      target_node_id: Optional target theorem ID (defaults to root_id).

    Returns:
      Compileable Lean 4 source string.
    """
    graph = self.read_graph()
    target = target_node_id or self.root_id
    if not target:
      if not graph.nodes:
        raise ValueError("Graph has no nodes to assemble.")
      target = next(iter(graph.nodes.keys()))

    return graph.assemble_current_proof(target)

  def get_summary(self) -> Dict[str, Any]:
    """Computes a summary of graph node statuses."""
    graph = self.read_graph()
    status_map = graph.compute_statuses()

    counts = {
        "total_nodes": len(graph.nodes),
        "formal_proved": 0,
        "formal_sketched": 0,
        "processing": 0,
        "formal_statement": 0,
        "failed": 0,
        "broken": 0,
        "claimed": 0,
    }

    now = time.time()
    for node_id, node in graph.nodes.items():
      status = status_map.get(node_id, node.status)
      status_val = (
          status.value
          if isinstance(status, dag.NodeStatus)
          else str(status)
      )
      if status_val in counts:
        counts[status_val] += 1

      if node.search_metadata.get("processing", False):
        counts["claimed"] += 1

    return counts

  def _generate_dot(self, graph: dag.ProofGraph) -> str:
    """Generates Graphviz dot content for the proof graph."""
    dot_lines = ["digraph ProofGraph {"]

    active_nodes = set()
    if self.root_id:
      active_nodes = graph.get_active_reachable_nodes(self.root_id)
    else:
      active_nodes = set(graph.nodes.keys())

    status_map = graph.compute_statuses()

    for node_id, node in graph.nodes.items():
      status_obj = status_map.get(node_id, node.status)
      status_str = (
          status_obj.value
          if isinstance(status_obj, dag.NodeStatus)
          else str(status_obj)
      )

      status_str, color_hex, fontcolor = plot_style.describe(node, status_str)

      if node_id not in active_nodes:
        color = plot_style.faded(color_hex)
        fontcolor = "#0000004D" if fontcolor == "black" else "#FFFFFF4D"
      else:
        color = color_hex

      introduced_defs = node.search_metadata.get("introduced_definitions", [])
      h_val = node.search_metadata.get(
          "live_heuristic_cost", node.search_metadata.get("heuristic_cost", 5.0)
      )
      p_val = node.search_metadata.get(
          "live_success_probability",
          node.search_metadata.get("success_probability", 0.7),
      )
      metrics_str = f"(h: {h_val:.1f}, P: {p_val:.2f})"
      label = f"{node_id}\n{status_str}\n{metrics_str}"

      dot_lines.append(
          f'  {json.dumps(node_id)} [label={json.dumps(label, ensure_ascii=False)}, style="filled", fillcolor="{color}",'
          f' fontcolor="{fontcolor}"];'
      )

      if introduced_defs:
        for def_name in introduced_defs:
          def_node_id = f"def_{def_name}"
          def_color = "#8B45134D" if node_id not in active_nodes else "#8B4513"
          def_fontcolor = "white" if node_id in active_nodes else "#FFFFFF4D"
          dot_lines.append(
              f"  {json.dumps(def_node_id)} [label={json.dumps(def_name)},"
              " style=filled,"
              f' fillcolor="{def_color}", fontcolor="{def_fontcolor}",'
              ' shape="box"];'
          )
          edge_color = "#8B45134D" if node_id not in active_nodes else "#8B4513"
          dot_lines.append(
              f"  {json.dumps(node_id)} -> {json.dumps(def_node_id)}"
              f' [style=dashed, color="{edge_color}"];'
          )

      # Every decomposition, not just the live one. Retiring a split used to
      # detach its children from the drawing, proved descendants included --
      # which is the one event most worth seeing, since those are the nodes
      # whose work is now orphaned. They are dashed instead.
      for and_node in node.and_nodes:
        edge_color = "#0000004D" if node_id not in active_nodes else "black"
        style = "style=dashed, " if and_node.failed else ""
        for dep_id in and_node.dependency_ids:
          dot_lines.append(
              f"  {json.dumps(node_id)} -> {json.dumps(dep_id)}"
              f' [{style}color="{edge_color}"];'
          )

    dot_lines.append("}")
    return "\n".join(dot_lines)

  def generate_plot(self, graph: dag.ProofGraph) -> Optional[str]:
    """Generates DOT and SVG plot files for the proof graph."""
    if not self.filepath:
      return None
    try:
      dot_content = self._generate_dot(graph)
      dot_path = self.filepath.replace(".json", ".dot")
      svg_path = self.filepath.replace(".json", ".svg")
      with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)

      subprocess.run(
          ["dot", "-Tsvg", dot_path, "-o", svg_path],
          check=False,
          capture_output=True,
      )
      if os.path.exists(svg_path):
        return svg_path
    except Exception as e:  # pylint: disable=broad-except
      if logging is not None:
        logging.warning("Failed to generate plot: %s", e)
    return None

  def sync_backup(self, graph: Optional[dag.ProofGraph] = None):
    """Mirrors the state file into the backup directory, if configured."""
    if not self.backup_dir:
      return
    try:
      self.storage_client.sync_file(self.filepath, self.backup_dir)

      svg_path = self.filepath.replace(".json", ".svg")
      if os.path.exists(svg_path):
        self.storage_client.sync_file(svg_path, self.backup_dir)

      target_graph = graph or self.read_graph()
      root = self.root_id or (
          next(iter(target_graph.nodes.keys()))
          if target_graph.nodes
          else None
      )
      if root and root in target_graph.nodes:
        try:
          final_content = target_graph.assemble_current_proof(root)
          lean_path = self.filepath.replace(".json", "_current_proof.lean")
          with open(lean_path, "w", encoding="utf-8") as f:
            f.write(final_content)
          self.storage_client.sync_file(lean_path, self.backup_dir)
        except Exception as e_lean:  # pylint: disable=broad-except
          if logging is not None:
            logging.warning("Failed to sync current_proof.lean: %s", e_lean)
    except Exception as e:  # pylint: disable=broad-except
      if logging is not None:
        logging.error("Failed to sync backup: %s", e)
      else:
        print(f"Failed to sync backup: {e}")

  def get_node_context(self, node_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full proof/decomposition context for an assigned subgoal."""
    graph = self.read_graph()
    if node_id not in graph.nodes:
      return None
    node = graph.nodes[node_id]
    parent_ref = ""
    if (
        node.reverse_dependency_ids
        and node.reverse_dependency_ids[0] in graph.nodes
    ):
      parent_node = graph.nodes[node.reverse_dependency_ids[0]]
      parent_ref = (
          parent_node.informal_problem.statement
          if parent_node.informal_problem
          else ""
      )
    phase = pipeline.phase_for(node)

    prior_diagnostics = {
        "proof_attempts": node.search_metadata.get("proof_attempts", 0),
        "decomp_attempts": node.search_metadata.get("decomp_attempts", 0),
        "last_error": node.search_metadata.get("last_error", ""),
    }

    # get_context is a read action: render only. Calling
    # provision_node_workspace here rotated the caller's own in-progress
    # scaffold out from under it.
    ws_info = self.workspace_mgr.render_node_files(
        node_id=node_id,
        graph=graph,
        phase=phase,
        root_id=self.root_id,
    )

    return {
        "node_id": node_id,
        "statement": (
            node.informal_problem.statement if node.informal_problem else ""
        ),
        "action_type": phase.role,
        "phase": phase.name,
        "status": (
            node.status.value
            if isinstance(node.status, dag.NodeStatus)
            else str(node.status)
        ),
        "node_dir": ws_info["node_dir"],
        "scaffold_path": ws_info["scaffold_path"],
        "context_path": ws_info["context_path"],
        "assembled_lean_scaffold": ws_info["scaffold_content"],
        "context_content": ws_info["context_content"],
        "prior_attempt_diagnostics": prior_diagnostics,
        "parent_ref": parent_ref,
        "ascendant_nodes_context": graph.assemble_for_tree_context(node_id),
        "available_lemmas_context": graph.assemble_available_lemmas_context(
            node_id
        ),
        "backtracking_feedback": graph.assemble_backtracking_feedback(node_id),
        "environment_header": graph.environment.to_lean_header(),
        "shared_definitions": graph.environment.shared_definitions,
        "allowed_axioms": graph.allowed_axioms,
    }


def _handle_claim_next(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Handles claiming the next available subgoal via DFS assignment."""
  action = input_data.action
  assigned = mgr.claim_next_subgoal(root_id=input_data.root_id)
  if assigned:
    nid = assigned["node_id"]
    phase = pipeline.PIPELINE[assigned["phase"]]
    return StateManagerOutput(
        success=True,
        action=action,
        node_id=nid,
        action_type=assigned.get("action_type"),
        statement=assigned.get("statement"),
        node_dir=assigned.get("node_dir"),
        scaffold_path=assigned.get("scaffold_path"),
        context_path=assigned.get("context_path"),
        # Everything the orchestrator needs to build a dispatch prompt without
        # deriving anything from the phase name itself. `allowed_actions` is
        # here so the prompt can name them: an agent that goes looking for its
        # own vocabulary is an agent that guesses one.
        metadata={
            "phase": phase.name,
            "role": assigned.get("action_type"),
            "scaffold_role": assigned.get("scaffold_role"),
            "allowed_actions": list(pipeline.allowed_actions(phase)),
        },
        message=(
            f"Assigned and claimed subgoal {nid}"
            f" (dispatch #{assigned.get('dispatch_count')},"
            f" {assigned.get('revision')} actions recorded so far)"
        ),
    )
  else:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message=claim_terminal.why_nothing_to_claim(
            mgr.read_graph(), mgr.root_id
        ),
    )


def _handle_release(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Handles releasing a claim so a stranded node becomes claimable again.

  Orchestrator recovery action: claims never expire, so this is the only way a
  node held by a crashed worker is freed.
  """

  action = input_data.action
  if not input_data.node_id:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message="node_id required for release.",
    )
  released = mgr.release_subgoal(node_id=input_data.node_id)
  if not released:
    return StateManagerOutput(
        success=False,
        action=action,
        node_id=input_data.node_id,
        error_message=RecordMessages.missing_node(input_data.node_id),
    )
  return StateManagerOutput(
      success=True,
      action=action,
      node_id=input_data.node_id,
      message=f"Released subgoal {input_data.node_id}",
  )


def _handle_release_all(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """`--action=release_all`: clears processing flag on all nodes."""
  released = mgr.release_all_subgoals()
  return StateManagerOutput(
      success=True,
      action=input_data.action,
      message=(
          f"Released {len(released)} processing subgoal(s): {', '.join(released)}"
          if released
          else "No processing subgoals to release."
      ),
  )


def _recorded(
    action: str, node_id: str, ok: bool, message: str
) -> StateManagerOutput:
  """Wraps a recorder's (ok, message) pair as tool output."""
  return StateManagerOutput(
      success=ok,
      action=action,
      node_id=node_id,
      message=message if ok else f"Failed to {action} for {node_id}: {message}",
      error_message=None if ok else message,
  )


def _handle_submit(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """`--action=submit`: hands over a scaffold from a phase that writes Lean."""
  node_id = input_data.node_id or (
      _node_id_from_code_file(input_data.code_file)
      if input_data.code_file
      else ""
  )
  if not node_id:
    return StateManagerOutput(
        success=False,
        action=input_data.action,
        error_message=(
            "MISSING_NODE: pass --node_id, or --code_file pointing at the"
            " scaffold inside the node's own directory."
        ),
    )
  if not input_data.code_file:
    return StateManagerOutput(
        success=False,
        action=input_data.action,
        node_id=node_id,
        error_message=(
            "MISSING_CODE_FILE: `--action=submit` hands over a file, so it"
            " needs `--code_file=<path to the scaffold in your node"
            " directory>`. Nothing was recorded and your claim is still held."
            " If you are giving up on this scaffold rather than submitting it,"
            ' the action is `--action=replan --reason="<...>"`.'
        ),
    )
  ok, msg = mgr.record_lean(
      code_file=input_data.code_file, node_id=node_id
  )
  return _recorded(input_data.action, node_id, ok, msg)


def _handle_declare(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Every agent action except `submit`. Carries `--reason` and no file.

  One handler for all five: which of them are legal is a property of the node's
  phase, not of the command, so there is nothing here to branch on.
  """
  node_id = input_data.node_id or ""
  if not node_id:
    return StateManagerOutput(
        success=False,
        action=input_data.action,
        error_message=(
            f"MISSING_NODE: `--action={input_data.action}` needs `--node_id`."
        ),
    )
  ok, msg = actions.declare(
      mgr, node_id, input_data.action, input_data.reason
  )
  return _recorded(input_data.action, node_id, ok, msg)


def _handle_get_node_report(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Reports what only the proof graph knows about one node."""
  ok, msg = actions.node_report(mgr, input_data.node_id or "")
  return StateManagerOutput(
      success=ok,
      action=input_data.action,
      node_id=input_data.node_id,
      message=msg if ok else "",
      error_message=None if ok else msg,
  )

def _handle_assemble(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Handles assembling a standalone Lean source file from the proof graph.

  Refuses unless the whole active subtree is proved. `assemble_current_proof`
  stubs every open leaf with `:= by sorry` and writes the file regardless, so
  without this guard "assembled successfully" is true of an empty graph and the
  orchestrator's only completion check means nothing. There is no flag to turn
  the guard off: the one caller that wanted a partial file was the old
  `--allow_sorry=true`, and a partial file is exactly what nobody should be
  handed at the end of a run.
  """
  action = input_data.action
  target = input_data.node_id or input_data.root_id
  graph = mgr.read_graph()
  target = target or mgr.root_id or (
      next(iter(graph.nodes)) if graph.nodes else None
  )
  if not target:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message="Graph has no nodes to assemble.",
    )
  statuses = graph.compute_statuses()
  open_nodes = sorted(
      nid
      for nid in graph.get_active_reachable_nodes(target)
      if statuses.get(nid) is not dag.NodeStatus.FORMAL_PROVED
  )
  if open_nodes:
    listed = ", ".join(f"`{n}`" for n in open_nodes[:40])
    more = "" if len(open_nodes) <= 40 else f" (+{len(open_nodes) - 40} more)"
    return StateManagerOutput(
        success=False,
        action=action,
        node_id=target,
        error_message=(
            f"ASSEMBLY INCOMPLETE: {len(open_nodes)} node(s) under `{target}`"
            f" are not proved, so nothing was written: {listed}{more}.\nThis is"
            " a worklist, not an error. Keep dispatching. A node here that"
            " `claim_next` never offers is either held by a crashed worker --"
            " free it with `--action=release --node_id=<id>` -- or dead under a"
            " decomposition its parent has not replaced yet."
        ),
    )
  code = mgr.assemble_proof(target_node_id=target)
  final_path = mgr.workspace_mgr.write_final_assembled_proof(code)
  return StateManagerOutput(
      success=True,
      action=action,
      assembled_code=code,
      message=f"Proof assembled successfully and written to {final_path}.",
      metadata={"assembled_file": final_path},
  )


def _handle_inspect(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Handles inspecting the proof graph and returning a status summary."""
  action = input_data.action
  graph = mgr.read_graph()
  summary = mgr.get_summary()
  return StateManagerOutput(
      success=True,
      action=action,
      message=f"ProofGraph loaded with {len(graph.nodes)} nodes.",
      metadata={"graph_summary": summary},
  )


def _handle_sync(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Handles mirroring the state file to the backup directory."""
  action = input_data.action
  graph = mgr.read_graph()
  mgr.sync_backup(graph)
  return StateManagerOutput(
      success=True,
      action=action,
      message=f"Synced state file to {input_data.backup_dir}",
  )


def _handle_get_context(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Handles retrieving full node context."""
  action = input_data.action
  node_id = input_data.node_id or _node_id_from_code_file(
      input_data.code_file
  ) or None
  if not node_id:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message="node_id or code_file required for get_context.",
    )
  ctx = mgr.get_node_context(node_id)
  if ctx is None:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message=f"Node {node_id} not found.",
    )
  return StateManagerOutput(
      success=True,
      action=action,
      node_id=node_id,
      node_dir=ctx.get("node_dir"),
      scaffold_path=ctx.get("scaffold_path"),
      context_path=ctx.get("context_path"),
      statement=ctx.get("statement", ""),
      message=f"Context retrieved for {node_id}",
      metadata={"node_context": ctx},
  )


# One character that continues a Lean identifier (`isIdRest`): `\w` plus the
# letter-like ranges it misses, such as `℘` and `᾽`. A superset is safe -- the
# extras cannot follow a declaration name in a file that parses.
_LEAN_ID_REST = (
    r"[\w'!?\u0370-\u03ff\u1f00-\u1ffe\u2100-\u214f"
    r"\U0001d49c-\U0001d59f]"
)


def _handle_init_graph(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Handles initializing a new proof graph."""
  action = input_data.action
  if os.path.exists(mgr.filepath) and os.path.getsize(mgr.filepath) > 0:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message=(
            f"State file {mgr.filepath} already exists and is not empty."
            " Refusing to initialize over existing graph."
        ),
    )
  root_id = input_data.root_id
  if not root_id:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message="root_id is required for init_graph action.",
    )
  statement = input_data.statement
  if not statement:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message="statement is required for init_graph action.",
    )
  init_graph = dag.ProofGraph()

  # Split benchmark file at root theorem declaration. The name must end where
  # Lean's identifier ends: a `\b` there also stops inside `foo'`, `foo₁`,
  # `foo!` and `foo.bar`, so a file declaring one of those before `foo` was
  # rooted at it -- and a disproof of it reported as foo's. `.{` is let through
  # because it opens universe parameters (`theorem foo.{u}`), not a name
  # component. Searched with comments masked, so a commented-out or docstring
  # `theorem foo` cannot be picked either.
  thm_matches = list(
      re.finditer(
          r"^(?:@\[[^\]]*\]\s+)?(?:noncomputable\s+)?(?:theorem|lemma)\s+"
          + re.escape(root_id)
          + rf"(?!{_LEAN_ID_REST}|\.(?!\{{))",
          models.mask_lean_comments(statement),
          re.MULTILINE,
      )
  )
  if len(thm_matches) > 1:
    # Legal Lean when the copies sit in different namespaces, but then nothing
    # here can tell which one the problem means.
    return StateManagerOutput(
        success=False,
        action=action,
        error_message=(
            f"Root theorem name '{root_id}' is declared"
            f" {len(thm_matches)} times in the statement content, so the"
            " root is ambiguous. Rename all but one and re-run init_graph."
        ),
    )
  thm_match = thm_matches[0] if thm_matches else None
  if not thm_match:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message=(
            f"Root theorem name '{root_id}' was not found in the statement content. "
            "Cannot split preamble or initialize graph successfully."
        ),
    )

  preamble = statement[: thm_match.start()].strip()
  preamble_error = graph_init_guard.check_preamble(preamble, root_id)
  if preamble_error:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message=preamble_error,
    )
  root_stmt_raw = statement[thm_match.start() :].strip()

  # Clean trailing assignment / tactics from root statement. A plain
  # re.sub(r":=.*$") cuts at the first ':=', truncating statements that carry
  # one inside a binder or a 'let' (e.g. '(n : Nat := 0)', 'let x := 3').
  root_stmt, _ = models.split_signature_and_proof(root_stmt_raw)
  root_stmt = root_stmt.strip()
  if not re.search(r"^(?:@\[[^\]]*\]\s+)?(?:noncomputable\s+)?(?:theorem|lemma)\b", root_stmt):
    root_stmt = f"theorem {root_id} : {root_stmt}"

  init_graph.environment.shared_definitions = preamble
  root_node = dag.ProofGraphNode(
      id=root_id,
      informal_problem=models.InformalProblem(statement=root_stmt),
  )

  init_graph.add_node(root_node)

  # The statement is the only thing in the tree that came from outside the
  # tool, and it is freeform. Everything downstream assumes it round-trips.
  # Checked after add_node -- render_scaffold_lean looks the node up -- and
  # before initialize_graph, so a rejected statement writes nothing to disk.
  for error in (
      graph_init_guard.check_root_gate(root_node.signature_str),
      graph_init_guard.check_root_compiles(
          mgr.workspace_mgr.render_scaffold_lean(root_id, init_graph)
      ),
  ):
    if error:
      return StateManagerOutput(
          success=False, action=action, error_message=error
      )

  mgr.initialize_graph(init_graph)
  return StateManagerOutput(
      success=True,
      action=action,
      node_id=root_id,
      statement=statement,
      message=f"Initialized proof graph with root {root_id}",
  )


def _handle_plot_graph(
    mgr: ProofGraphStateManager,
    input_data: StateManagerInput,
) -> StateManagerOutput:
  """Handles the 'plot_graph' action."""
  graph = mgr.read_graph()
  svg_path = mgr.generate_plot(graph)
  return StateManagerOutput(
      success=(svg_path is not None),
      action=input_data.action,
      metadata={"svg_path": svg_path} if svg_path else {},
      message=f"Generated plot at {svg_path}" if svg_path else "Plotting failed",
  )


def _handle_get_accessible_nodes(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Handles retrieving live accessible non-ancestor nodes for a target node."""
  action = input_data.action
  node_id = input_data.node_id or _node_id_from_code_file(
      input_data.code_file
  ) or None
  if not node_id:
    return StateManagerOutput(
        success=False,
        action=action,
        error_message="node_id or code_file required for get_accessible_nodes.",
    )
  graph = mgr.read_graph()
  nodes_data = graph.get_accessible_nodes_data(node_id)

  lines = [
      f"### Live Accessible Nodes for Subgoal `{node_id}`",
      "",
      "| Lemma Name | Status | Sibling? | Dependencies | Target Signature |",
      "| :--- | :--- | :--- | :--- | :--- |",
  ]
  if not nodes_data:
    lines.append("| *(None available)* | - | - | - | - |")
  else:
    for n in nodes_data:
      st = n["status"]
      status_badge = (
          "[PROVED]"
          if n["is_proved"]
          else (
              "[IN_PROGRESS]" if n["is_processing"] else f"[{st.upper()}]"
          )
      )
      sib_str = "Yes" if n["is_sibling"] else "No"
      deps_str = (
          ", ".join(f"`{d}`" for d in n["dependencies"])
          if n["dependencies"]
          else "[]"
      )
      lines.append(
          f"| `{n['name']}` | `{status_badge}` | {sib_str} | {deps_str} |"
          f" `{n['signature']}` |"
      )

  msg = "\n".join(lines)
  return StateManagerOutput(
      success=True,
      action=action,
      node_id=input_data.node_id,
      message=msg,
      metadata={"accessible_nodes": nodes_data},
  )


def _handle_describe_workspace(
    mgr: ProofGraphStateManager, input_data: StateManagerInput
) -> StateManagerOutput:
  """Handles describing the filesystem layout contract and inspecting live disk status."""
  action = input_data.action
  layout_doc = mgr.workspace_mgr.describe_layout()
  disk_inspection = mgr.workspace_mgr.inspect_workspace()
  return StateManagerOutput(
      success=True,
      action=action,
      message=layout_doc,
      metadata={"workspace_inspection": disk_inspection},
  )


_ACTION_DISPATCH: Dict[
    str,
    Callable[[ProofGraphStateManager, StateManagerInput], StateManagerOutput],
] = {
    # Agent actions. `submit` reads a file; the rest are one handler, because
    # which of them a node accepts is decided by its phase, not by the command.
    "submit": _handle_submit,
    "planned": _handle_declare,
    "refute": _handle_declare,
    "failed": _handle_declare,
    "replan": _handle_declare,
    "approved": _handle_declare,
    # Orchestrator actions.
    "claim_next": _handle_claim_next,
    "release": _handle_release,
    "release_all": _handle_release_all,
    "assemble": _handle_assemble,
    "inspect": _handle_inspect,
    "describe_workspace": _handle_describe_workspace,
    "sync": _handle_sync,
    "get_context": _handle_get_context,
    "get_node_report": _handle_get_node_report,
    "get_accessible_nodes": _handle_get_accessible_nodes,
    "init_graph": _handle_init_graph,
    "plot_graph": _handle_plot_graph,
}

# The flag's `choices` and this table are two lists of the same actions. If they
# drift, the loser is an agent told an action is legal that then comes back
# "Unknown action" -- so they disagree at import time instead.
assert set(_ACTION_DISPATCH) == set(_ACTION_NAMES), (
    "_ACTION_DISPATCH and _ACTION_NAMES disagree:"
    f" {set(_ACTION_DISPATCH) ^ set(_ACTION_NAMES)}"
)

# The agent's vocabulary is the union of what the phase table offers, and this
# is where the two are held together. Adding an edge to `pipeline.PIPELINE`
# without a flag value for it means an agent is briefed to run a command
# argparse rejects; adding a flag value with no edge means one it can never use.
_PIPELINE_ACTIONS = {
    action
    for phase in pipeline.PIPELINE.values()
    for action in pipeline.allowed_actions(phase)
}
assert _PIPELINE_ACTIONS == set(_AGENT_ACTIONS), (
    "_AGENT_ACTIONS and pipeline.PIPELINE disagree:"
    f" {_PIPELINE_ACTIONS ^ set(_AGENT_ACTIONS)}"
)


def find_problem_root(
    code_file: Optional[str] = None, start_dir: Optional[str] = None
) -> Optional[str]:
  """Discovers the problem root directory by walking up parent directories.

  Checks for 'proof_graph.json' or walking up from inside 'nodes'.
  """
  candidates = []
  if code_file:
    candidates.append(os.path.abspath(code_file))
  if start_dir:
    candidates.append(os.path.abspath(start_dir))
  if not candidates:
    candidates.append(os.getcwd())

  for candidate in candidates:
    cur = candidate if os.path.isdir(candidate) else os.path.dirname(candidate)
    while cur and cur != os.path.dirname(cur):
      if os.path.isfile(os.path.join(cur, "proof_graph.json")):
        return cur
      if os.path.basename(cur) == "nodes":
        return os.path.dirname(cur)
      cur = os.path.dirname(cur)
  return None


def run_state_manager(input_data: StateManagerInput) -> StateManagerOutput:
  """Executes a centralized proof state manager action.

  Args:
    input_data: StateManagerInput with target action and parameters.

  Returns:
    StateManagerOutput with results and execution status.
  """
  root_dir = input_data.root_dir
  if not root_dir:
    if input_data.filepath:
      root_dir = os.path.dirname(os.path.abspath(input_data.filepath))
    else:
      root_dir = find_problem_root(code_file=input_data.code_file)

  filepath = input_data.filepath
  if not filepath and root_dir:
    filepath = os.path.join(root_dir, "proof_graph.json")

  if not filepath:
    return StateManagerOutput(
        success=False,
        action=input_data.action,
        error_message=(
            "Could not discover problem root or proof_graph.json. "
            "Specify --root_dir or --filepath."
        ),
    )

  mgr = ProofGraphStateManager(
      filepath=filepath,
      root_dir=root_dir,
      backup_dir=input_data.backup_dir,
      root_id=input_data.root_id,
  )

  try:
    handler = _ACTION_DISPATCH.get(input_data.action)
    if handler is not None:
      return handler(mgr, input_data)
    else:
      return StateManagerOutput(
          success=False,
          action=input_data.action,
          error_message=f"Unknown action: {input_data.action}",
      )

  except Exception as e:  # pylint: disable=broad-except
    return StateManagerOutput(
        success=False,
        action=input_data.action,
        error_message=str(e),
    )


def main(argv: List[str]) -> None:
  """CLI entry point for ProofGraph State Manager."""
  if _HAS_ABSL and flags is not None:
    input_data = StateManagerInput(
        action=_ACTION.value,
        filepath=_FILEPATH.value,
        code_file=_CODE_FILE.value,
        root_dir=_ROOT_DIR.value,
        backup_dir=_BACKUP_DIR.value,
        root_id=_ROOT_ID.value,
        node_id=_NODE_ID.value,
        parent_id=_PARENT_ID.value,
        statement=_STATEMENT.value,
        reason=_REASON.value,
        json_output=bool(_JSON_OUTPUT.value) if _JSON_OUTPUT.value is not None else False,
    )
  else:
    import argparse  # pylint: disable=g-import-not-at-top

    parser = argparse.ArgumentParser(description="ProofGraph State Manager")
    parser.add_argument("--filepath", type=str, default="")
    parser.add_argument("--code_file", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--backup_dir", type=str, default=None)
    parser.add_argument("--root_id", type=str, default=None)
    parser.add_argument("--node_id", type=str, default=None)
    parser.add_argument("--parent_id", type=str, default=None)
    parser.add_argument("--statement", type=str, default=None)
    parser.add_argument("--reason", type=str, default="")
    parser.add_argument(
        "--json_output", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--action", type=str, default="claim_next", choices=list(_ACTION_NAMES)
    )
    args, _ = parser.parse_known_args(argv[1:])

    input_data = StateManagerInput(
        action=args.action,
        filepath=args.filepath,
        code_file=args.code_file,
        root_dir=args.root_dir,
        backup_dir=args.backup_dir,
        root_id=args.root_id,
        node_id=args.node_id,
        parent_id=args.parent_id,
        statement=args.statement,
        reason=args.reason,
        json_output=args.json_output,
    )

  result = run_state_manager(input_data)
  if input_data.json_output:
    dump = (
        result.model_dump(exclude_none=True)
        if hasattr(result, "model_dump")
        else result.dict()
    )
    if "metadata" in dump and not dump["metadata"]:
      del dump["metadata"]
    print(json.dumps(dump))
  else:
    if result.success:
      print(f"State manager {result.action} succeeded.")
      if result.message:
        print(result.message)
      if result.assembled_code:
        print("\n--- ASSEMBLED CODE ---")
        print(result.assembled_code)
    else:
      print(f"State manager {result.action} failed: {result.error_message}", file=sys.stderr)
      sys.exit(1)


if __name__ == "__main__":
  if _HAS_ABSL and app is not None:
    app.run(main)
  else:
    import sys
    main(sys.argv)