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

"""DAG core models and topological operations for theorem proving."""

from collections import deque
import enum
from typing import Any, Dict, List, Optional, Set
import pydantic

import lean_code_parser as models


# Lean's three standard axioms. Anything outside this set is a soundness
# escape: 'sorryAx' (an unfinished proof), 'Lean.ofReduceBool' /
# 'Lean.trustCompiler' (native_decide, which trusts the compiler rather than the
# kernel), or an axiom the agent declared itself.
#
# Lives here rather than in state_manager so the gate that enforces it and the
# briefing that explains it to agents read the same value and cannot drift.
DEFAULT_ALLOWED_AXIOMS = frozenset(
    {"propext", "Classical.choice", "Quot.sound"}
)

# What a dead node's `live_heuristic_cost` is set to. Not `float("inf")`, which
# is the honest in-memory value and a lie on disk: pydantic serializes it to
# JSON `null`, so the graph reloads with `None` there and the first
# `f"{cost:.1f}"` downstream raises. Large enough to lose every comparison.
UNREACHABLE_COST = 1e9

# `search_metadata` keys the search writes about a node rather than because of
# anything the node did. `add_proposed_subgoals` stamps the planner's estimates
# on a subgoal the moment it is created, and `recalculate_empirical_metrics`
# re-stamps derived costs on every node it walks -- so by the time any node can
# be stranded it already carries all six, and `prune_orphans` reading
# `search_metadata` as evidence of work would never collect anything.
#
# Keep this list current when adding a scoring key. Getting it wrong in this
# direction leaks a ghost; the opposite list, of keys that DO count as work,
# would delete a live node when someone forgot to extend it.
SCORING_METADATA_KEYS = frozenset({
    "initial_mapped_heuristic",
    "initial_mapped_probability",
    "heuristic_cost",
    "success_probability",
    "live_heuristic_cost",
    "live_success_probability",
})


class NodeStatus(str, enum.Enum):
  FORMAL_STATEMENT = "formal_statement"
  PROCESSING = "processing"
  FORMAL_SKETCHED = "formal_sketched"
  FORMAL_PROVED = "formal_proved"
  BROKEN = "broken"
  FAILED = "failed"


class AndNode(pydantic.BaseModel):
  """Represents an AND-edge (one decomposition attempt into required sub-lemmas)."""

  id: str = pydantic.Field(
      default="",
      description="Unique string identifier for decomposition attempt",
  )
  dependency_ids: List[str] = pydantic.Field(default_factory=list)
  definitions_block: str = pydantic.Field(
      default="",
      description="Opaque Lean 4 code block of definitions/structures introduced by this decomposition",
  )
  search_metadata: Dict[str, Any] = pydantic.Field(default_factory=dict)
  # Carries the statements and not just the names in `dependency_ids`, because
  # names are not stable: `prune_orphans` lets a retired branch's name be taken
  # over by different mathematics. Looking a prior attempt's statements up
  # through `nodes` after that returns the *new* lemma, so the
  # duplicate-strategy guard would be comparing against whatever the graph
  # holds now rather than against what was tried.
  #
  # Not enforced, deliberately. A model validator used to reject an AND-node
  # that named dependencies without recording their statements, but the field
  # postdates the entire run archive -- 5,619 of 5,731 AND-nodes across 78 runs
  # omit it -- so the check made every finished experiment unloadable and was
  # removed within the hour. Enforcement is not needed to keep the field
  # populated: `add_proposed_subgoals` derives `dependency_ids` and
  # `subgoal_signatures` from the same `subgoals` list, so the two cannot
  # disagree in a graph this code wrote. The cost of the permissiveness is that
  # legacy AND-nodes contribute an empty set to the duplicate-strategy guard
  # and so are invisible to it. If the invariant is ever worth asserting again,
  # assert it in `add_proposed_subgoals`, not in a validator that also runs on
  # load.
  subgoal_signatures: List[models.FormalSignature] = pydantic.Field(
      default_factory=list,
      description="Every subgoal this decomposition declared, used or not, with statements.",
  )
  review_replan_rate: Optional[float] = pydantic.Field(
      default=None,
      description=(
          "Subtree re-plan rate a reviewer last approved this decomposition"
          " at, ratcheted down if the subtree has since shrunk. None means no"
          " reviewer has ruled on THIS decomposition."
      ),
  )
  failed: bool = pydantic.Field(default=False)
  error_message: str = pydantic.Field(default="")
  retired_for: str = pydantic.Field(
      default="",
      description=(
          "Why a reviewer retired this decomposition: 'shape' if a different"
          " split is wanted, 'evidence' if the same split is wanted back with"
          " a gap in the blueprint filled. Empty means no reviewer retired it."
          " Only 'evidence' relaxes the duplicate-strategy guard, and only for"
          " the one resubmission it was asked for."
      ),
  )


class ProofGraphNode(pydantic.BaseModel):
  """Represents a single logical step (Lemma or Theorem) acting as an OR-node."""

  id: str = pydantic.Field(
      default="", description="Unique identifier for the node"
  )
  informal_problem: Optional[models.InformalProblem] = None
  informal_proof: Optional[models.InformalProof] = None
  formal_theorem: Optional[models.FormalTheorem] = None
  formal_sketch: Optional[models.FormalProofSketch] = None
  definitions: str = ""

  and_nodes: List[AndNode] = pydantic.Field(default_factory=list)
  reverse_dependency_ids: List[str] = pydantic.Field(default_factory=list)

  search_metadata: Dict[str, Any] = pydantic.Field(default_factory=dict)

  @property
  def signature_str(self):
    """Returns the canonical declaration signature for this node without sorry."""
    if self.formal_theorem is not None:
      sig = self.formal_theorem.signature.to_lean()
    elif self.formal_sketch is not None:
      sig = self.formal_sketch.signature.to_lean()
    elif self.informal_problem is not None:
      raw = self.informal_problem.statement.strip()
      sig = models.FormalSignature(name=self.id, statement=raw).to_lean()
    else:
      raise ValueError(
          f"Node '{self.id}' has no statement or signature defined in the"
          " graph."
      )
    # No second split. Every writer of `statement` -- `_handle_init_graph`,
    # `parse_formal_statement`, and so `parse_subgoal_declarations` -- has
    # already removed the proof, so re-splitting here can only find a `:=` that
    # is part of the statement. On a `let`-bound one it silently cut the
    # theorem in half and the tree proved the fragment. If a proof ever does
    # survive into `statement`, the scaffold is invalid Lean and the compiler
    # says so.
    return sig.strip()

  @property
  def mathematical_status(self):
    """Static AST mathematical status of this node."""
    if self.formal_theorem is not None:
      return NodeStatus.FORMAL_PROVED
    if self.formal_sketch is not None:
      return NodeStatus.FORMAL_SKETCHED
    return NodeStatus.FORMAL_STATEMENT

  @property
  def status(self):
    return self.mathematical_status

  @property
  def active_and_node(self):
    """The decomposition currently being pursued, if any.

    Defined as the last AND-node not marked failed. Every consumer must use
    this: reading `and_nodes[-1]` directly diverges whenever a failed
    decomposition sits after a live one, which made status reporting and proof
    assembly disagree about whether a node was proved.
    """
    return next((an for an in reversed(self.and_nodes) if not an.failed), None)

  @property
  def canonical_name(self):
    if self.formal_theorem:
      return self.formal_theorem.signature.name
    if self.formal_sketch:
      return self.formal_sketch.signature.name
    return self.id


class ProofGraph(pydantic.BaseModel):
  """The global registry managing proof graph nodes and dependencies."""

  nodes: Dict[str, ProofGraphNode] = pydantic.Field(default_factory=dict)
  environment: models.FormalEnvironment = pydantic.Field(
      default_factory=models.FormalEnvironment
  )
  allowed_axioms: List[str] = pydantic.Field(default_factory=list)
  global_informal_proof: Optional[str] = pydantic.Field(default=None)
  graph_metadata: Dict[str, Any] = pydantic.Field(default_factory=dict)

  def add_node(self, node):
    self.nodes[node.id] = node

  @property
  def effective_allowed_axioms(self):
    """The axiom allowlist actually in force for this problem.

    Single source of truth for both the verification gate and the agent
    briefing, so tightening the policy updates what agents are told.
    """
    return set(self.allowed_axioms) or set(DEFAULT_ALLOWED_AXIOMS)

  def has_path(self, start_id, end_id):
    """Returns True if there is a directed path from start_id to end_id."""
    visited = set()

    def _dfs(current_id):
      if current_id == end_id:
        return True
      if current_id in visited:
        return False
      visited.add(current_id)

      node = self.nodes.get(current_id)
      if not node:
        return False

      for and_node in node.and_nodes:
        for dep_id in and_node.dependency_ids:
          if _dfs(dep_id):
            return True
      return False

    return _dfs(start_id)

  def add_proposed_subgoals(
      self,
      parent_id,
      subgoals,
      estimations = None,
      definitions_block = "",
  ):
    """Unifies subgoal creation, dependency linking, and cycle validation."""
    parent_node = self.nodes.get(parent_id)
    if not parent_node:
      raise ValueError(f"Parent node {parent_id} not found.")

    if not subgoals:
      raise ValueError(
          f"Refusing to add an empty decomposition to '{parent_id}'. An"
          " AND-node with no dependencies is vacuously satisfied and would"
          " mark the parent FORMAL_PROVED with nothing supporting it."
      )

    for sig in subgoals:
      dep_id = sig.name
      if self.has_path(dep_id, parent_id):
        raise ValueError(
            f"Adding dependency to '{dep_id}' creates a cycle as it already"
            f" has a path to '{parent_id}'"
        )

    # Statements the kernel has refuted, by normalized text. A disproof kills
    # the proposition, not the name, so the rename that recovers an exhausted
    # lemma must not recover this one -- without the check the planner
    # re-proposes the same false statement under `_v2` and the branch loops
    # through a full disproof cycle every time.
    disproved = {
        models.normalize_string(node.informal_problem.statement): nid
        for nid, node in self.nodes.items()
        if node.search_metadata.get("disproved") and node.informal_problem
    }
    for sig in subgoals:
      prior = disproved.get(models.normalize_string(sig.statement))
      if prior is not None and prior != sig.name:
        raise ValueError(
            f"REUSE OF DISPROVED LEMMA: proposed subgoal '{sig.name}' states"
            f" exactly what '{prior}' was refuted for -- the kernel derived"
            " `False` from it. It is false, renaming will not help, and any"
            " decomposition that depends on it cannot close. Change the"
            " mathematics."
        )

    statuses = self.compute_statuses()

    for sig in subgoals:
      dep_id = sig.name
      if dep_id in self.nodes:
        existing_node = self.nodes[dep_id]
        if existing_node.informal_problem and models.normalize_string(
            existing_node.informal_problem.statement
        ) != models.normalize_string(sig.statement):
          raise ValueError(
              f"SIGNATURE MISMATCH: the name '{dep_id}' is taken in this proof"
              " and the lemma holding it states something else. The final"
              " proof is one Lean file, so one name cannot mean two"
              f" things.\nEXISTING: {existing_node.informal_problem.statement}"
              f"\nYOURS: {sig.statement}\nACTION REQUIRED: if this is the same"
              " lemma, state it exactly as the existing one does. If it is a"
              " different lemma, give it a name that says what *yours* claims."
              " Do NOT take this name and add a suffix: a name that differs"
              " only by decoration reads as the same lemma to every agent"
              " downstream, and that is how a refuted statement comes back."
          )

        # The node's status, not a probability. The old test read
        # `live_success_probability` / `success_probability`, and neither is
        # ever written -- the only writer is the `estimations` argument below,
        # which no caller passes -- so it resolved to the 1.0 default every
        # time and never fired.
        #
        # FAILED and nothing else. BROKEN is what `_resolve_statuses` returns
        # for an id that is not a node, and this branch only runs for ids that
        # are, so testing for it here would be unreachable. A node whose every
        # decomposition has died resolves to FORMAL_STATEMENT, not FAILED, and
        # that is right: it needs a new plan, it is not a dead end.
        if statuses.get(dep_id) == NodeStatus.FAILED:
          raise ValueError(
              f"REUSE OF FAILED LEMMA: Proposed subgoal '{dep_id}' has already"
              " FAILED in the graph."
          )

    for sig in subgoals:
      dep_id = sig.name
      if dep_id not in self.nodes:
        new_node = ProofGraphNode(
            id=dep_id,
            informal_problem=models.InformalProblem(statement=sig.statement),
            reverse_dependency_ids=[parent_id],
        )

        if estimations and dep_id in estimations:
          est = estimations[dep_id]
          new_node.search_metadata["initial_mapped_heuristic"] = est.get(
              "estimated_cost", 5.0
          )
          new_node.search_metadata["initial_mapped_probability"] = est.get(
              "success_probability", 0.7
          )
          new_node.search_metadata["heuristic_cost"] = est.get(
              "estimated_cost", 5.0
          )
          new_node.search_metadata["success_probability"] = est.get(
              "success_probability", 0.7
          )

        self.add_node(new_node)
      else:
        existing_node = self.nodes[dep_id]
        if parent_id not in existing_node.reverse_dependency_ids:
          existing_node.reverse_dependency_ids.append(parent_id)

    dep_ids = [sig.name for sig in subgoals]
    new_and_node = AndNode(
        id=f"{parent_id}_decomp_{len(parent_node.and_nodes)}",
        dependency_ids=dep_ids,
        subgoal_signatures=list(subgoals),
        definitions_block=definitions_block,
    )
    parent_node.and_nodes.append(new_and_node)

  def recalculate_empirical_metrics(self, root_id):
    """Bottom-up recalculation of live heuristic costs and joint probabilities."""
    visited = set()
    status_map = self.propagate_failures()
    unproved_subgraphs = {}

    def _traverse(node_id):
      if node_id in unproved_subgraphs:
        return unproved_subgraphs[node_id]
      if node_id in visited:
        return set()
      visited.add(node_id)

      node = self.nodes.get(node_id)
      if not node:
        return set()

      for and_node in node.and_nodes:
        for dep_id in and_node.dependency_ids:
          _traverse(dep_id)

      is_failed = node.search_metadata.get("decomposition_failed", False)
      status_str = status_map.get(node_id, node.mathematical_status)
      if isinstance(status_str, NodeStatus):
        status_str = status_str.value

      if status_str == "formal_proved" and node.formal_theorem:
        node.search_metadata["live_success_probability"] = 1.0
        node.search_metadata["live_heuristic_cost"] = 0.0
        unproved_subgraphs[node_id] = set()
      elif is_failed or status_str in ["broken", "failed"]:
        node.search_metadata["live_success_probability"] = 0.0
        node.search_metadata["live_heuristic_cost"] = UNREACHABLE_COST
        unproved_subgraphs[node_id] = {node_id}
      else:
        active_p = 1.0
        my_unproved_set = {node_id}
        has_active_and_branch = False

        active_and = node.active_and_node
        if active_and is not None:
          for dep_id in active_and.dependency_ids:
            dep_node = self.nodes.get(dep_id)
            if dep_node:
              has_active_and_branch = True
              child_set = _traverse(dep_id)
              my_unproved_set.update(child_set)
              child_p = dep_node.search_metadata.get(
                  "live_success_probability", 0.7
              )
              active_p = min(active_p, child_p)

          if has_active_and_branch:
            my_unproved_set.discard(node_id)

        active_h = 0.0
        if not has_active_and_branch:
          active_h = node.search_metadata.get("initial_mapped_heuristic", 5.0)
          active_p = node.search_metadata.get("initial_mapped_probability", 0.7)
          my_unproved_set = {node_id}
        else:
          for uid in my_unproved_set:
            unproved_node = self.nodes.get(uid)
            if unproved_node:
              active_h += unproved_node.search_metadata.get(
                  "initial_mapped_heuristic", 5.0
              )
            else:
              active_h += 5.0

        node.search_metadata["live_heuristic_cost"] = active_h
        node.search_metadata["live_success_probability"] = active_p
        unproved_subgraphs[node_id] = my_unproved_set

      return unproved_subgraphs[node_id]

    _traverse(root_id)

  def get_dependencies_recursive(self, node_id):
    """Fetches all dependencies topologically using a 3-Color DFS approach."""
    visited = set()
    path = set()
    deps = []

    def _dfs(current_id):
      if current_id in path:
        raise ValueError(
            f"Circular dependency detected! Node '{current_id}' relies on"
            " itself."
        )
      if current_id in visited:
        return

      path.add(current_id)
      node = self.nodes.get(current_id)
      if not node:
        raise ValueError(f"Node {current_id} not found in graph.")

      and_node = node.active_and_node
      if and_node is not None:
        for dep_id in and_node.dependency_ids:
          _dfs(dep_id)

      path.remove(current_id)
      visited.add(current_id)
      deps.append(node)

    target_node = self.nodes.get(node_id)
    if target_node is not None and target_node.active_and_node is not None:
      for target_dep_id in target_node.active_and_node.dependency_ids:
        _dfs(target_dep_id)

    return deps

  def assemble_current_proof(self, root_id = None):
    """Builds a flat, single Lean snapshot for the entire proof tree from root downward.

    Combines the global environment header (including dynamic headers), all
    topologically ordered dependency lemmas, and the root theorem. Active
    unproved leaves are safely resolved with ':= by sorry'.
    """
    target_root = root_id or (
        next(iter(self.nodes.keys())) if self.nodes else None
    )
    if not target_root or target_root not in self.nodes:
      return ""

    ordered_deps = self.get_dependencies_recursive(target_root)
    header_str = self.environment.to_lean_header().strip()
    lemmas_str = ""
    for dep_node in ordered_deps:
      if dep_node.formal_theorem and dep_node.formal_theorem.proof:
        sig = dep_node.formal_theorem.signature.to_lean()
        proof = dep_node.formal_theorem.proof.tactics
        lemmas_str += f"{sig} :=\n{proof}\n\n"
      elif dep_node.formal_sketch and not (
          dep_node.and_nodes and dep_node.active_and_node is None
      ):
        sig = dep_node.formal_sketch.signature.to_lean()
        proof = dep_node.formal_sketch.proof.tactics
        lemmas_str += f"{sig} :=\n{proof}\n\n"
      else:
        sig = dep_node.signature_str
        lemmas_str += f"{sig} :=\nby\n  sorry\n\n"

    root_node = self.nodes[target_root]
    if root_node.formal_theorem and root_node.formal_theorem.proof:
      target_sig = root_node.formal_theorem.signature.to_lean()
      target_proof = root_node.formal_theorem.proof.tactics
    elif root_node.formal_sketch and not (
        root_node.and_nodes and root_node.active_and_node is None
    ):
      target_sig = root_node.formal_sketch.signature.to_lean()
      target_proof = root_node.formal_sketch.proof.tactics
    else:
      target_sig = root_node.signature_str
      target_proof = "by\n  sorry"

    target_decl = f"{target_sig} :=\n{target_proof}"
    return f"{header_str}\n\n{lemmas_str}{target_decl}".strip() + "\n"

  def assemble_header_smoke_test(self, root_id = None):
    """Builds the header plus the root statement alone, with a 'sorry' body.

    A cheap standin for `assemble_current_proof` that answers one question: does
    the header still elaborate, and does the root statement still type-check
    against it? On the 111-node tree that cost 0.7s against 76s for the full
    assembly, because it is bounded by header size rather than node count.

    It deliberately proves nothing -- the body is 'sorry', so the caller must
    read `is_valid` and ignore the axiom list.
    """
    target_root = root_id or (
        next(iter(self.nodes.keys())) if self.nodes else None
    )
    if not target_root or target_root not in self.nodes:
      return ""
    header_str = self.environment.to_lean_header().strip()
    target_sig = self.nodes[target_root].signature_str
    return f"{header_str}\n\n{target_sig} :=\nby\n  sorry\n".strip() + "\n"

  def compute_statuses(self):
    """Returns a status report for every node without modifying the graph.

    Safe to call from read-only paths. Previously these shared one method with
    the write path, so a getter silently mutated `and_node.failed` — persisted
    when called under a lock, discarded otherwise, which let `inspect` report a
    status the next `claim_next` disagreed with.
    """
    return self._resolve_statuses(propagate=False)

  def propagate_failures(self):
    """Returns a status report and retires AND-nodes with dead children.

    Mutates the graph, so only call this from a write path.
    """
    return self._resolve_statuses(propagate=True)

  def _resolve_statuses(self, propagate):
    """Resolves statuses bottom-up, optionally recording failure propagation."""
    status_map = {}
    path = set()

    def _get_status(node_id):
      if node_id in status_map:
        return status_map[node_id]
      if node_id in path:
        return NodeStatus.BROKEN

      path.add(node_id)
      node = self.nodes.get(node_id)
      if not node:
        path.remove(node_id)
        return NodeStatus.BROKEN

      if node.search_metadata.get("decomposition_failed"):
        status_map[node_id] = NodeStatus.FAILED
        path.remove(node_id)
        return NodeStatus.FAILED

      if node.search_metadata.get("processing", False):
        status_map[node_id] = NodeStatus.PROCESSING
        path.remove(node_id)
        return NodeStatus.PROCESSING

      if (
          not node.formal_theorem
          and not node.formal_sketch
          and len(node.and_nodes) == 0
      ):
        status_map[node_id] = NodeStatus.FORMAL_STATEMENT
        path.remove(node_id)
        return NodeStatus.FORMAL_STATEMENT

      # Direct verified proof takes absolute immediate precedence
      if node.formal_theorem and node.formal_theorem.proof:
        status_map[node_id] = NodeStatus.FORMAL_PROVED
        path.remove(node_id)
        return NodeStatus.FORMAL_PROVED

      if len(node.and_nodes) > 0:
        any_proved = False
        all_failed = True
        for and_node in node.and_nodes:
          if and_node.failed:
            continue

          dead_child_id = None
          if and_node.dependency_ids:
            and_node_proved = True
            for dep_id in and_node.dependency_ids:
              dep_status = _get_status(dep_id)
              if dep_status in (NodeStatus.FAILED, NodeStatus.BROKEN):
                # BROKEN means a cycle or a dangling dependency id. Left alive
                # the AND-node is unprovable forever and claim_next reports the
                # graph as dead-ended with nothing actually marked failed.
                dead_child_id = dep_id
                and_node_proved = False
                break
              if dep_status != NodeStatus.FORMAL_PROVED:
                and_node_proved = False
          else:
            # A decomposition with no sub-lemmas proves nothing. Creating one
            # is blocked in add_proposed_subgoals; this keeps a legacy or
            # hand-edited graph from resolving its parent for free.
            and_node_proved = False

          if dead_child_id is not None:
            if propagate:
              and_node.failed = True
              if not and_node.error_message:
                and_node.error_message = (
                    f"Child subgoal '{dead_child_id}' is an unprovable dead"
                    " end."
                )
          else:
            all_failed = False

          if and_node_proved:
            any_proved = True
            break

        if any_proved and (node.formal_sketch is not None):
          final_status = NodeStatus.FORMAL_PROVED
        elif all_failed:
          final_status = NodeStatus.FORMAL_STATEMENT
        else:
          final_status = NodeStatus.FORMAL_SKETCHED
      else:
        final_status = node.mathematical_status

      status_map[node_id] = final_status
      path.remove(node_id)
      return final_status

    for node_id in self.nodes:
      _get_status(node_id)

    return status_map

  def get_active_reachable_nodes(self, root_id):
    """Collects the set of all nodes reachable from the root along active branches."""
    reachable = set()
    visited = set()

    def _dfs(curr_id):
      if curr_id in visited or curr_id not in self.nodes:
        return
      visited.add(curr_id)
      reachable.add(curr_id)

      node = self.nodes.get(curr_id)
      active_an = node.active_and_node if node else None
      if active_an is not None:
        for dep_id in active_an.dependency_ids:
          _dfs(dep_id)

    _dfs(root_id)
    return reachable

  def is_node_reachable_from_root(self, node_id, root_id):
    return node_id in self.get_active_reachable_nodes(root_id)

  def prune_orphans(self, root_id):
    """Deletes unreachable nodes that never did any work. Returns their ids.

    Retiring a decomposition leaves its children behind: unreachable from the
    root, never claimable again, and still occupying their names. `fmf_v1_004`
    ended with 15 of them. They are not harmless bookkeeping -- every
    uniqueness check in the system asks `is this name in nodes?`, so a ghost
    refuses the name to whoever needs it next. That cost the run a rename
    spiral: a reviewer refuted `two_pow_totient_gt` for n=4, the prover came
    back with the corrected `n != 4` statement, and the tool refused it because
    a ghost owned the name and then suggested `_alt`.

    Deleting rather than filtering at each check, because a node that exists
    but does not count is a thing every future reader has to be told about.
    After this, `name in self.nodes` means what it says.

    Three nodes are NOT pruned, and each exclusion is a safety interlock:

    - The root, always, even in a graph with nothing else left.
    - Anything reachable from the root. This covers the shared-child case: a
      lemma proposed by two decompositions, one retired and one live, is still
      reachable through the live one.
    - Anything that did work, meaning it has an AND-node, a proof, or any
      search_metadata other than the scoring keys in `SCORING_METADATA_KEYS`.
      The metadata test is deliberately broad -- a node that was merely claimed
      carries `processing` and `claim_timestamp`, so a node held by a worker
      right now cannot be pruned out from under it, and a node that owns a
      directory on disk keeps the name that directory is filed under. The
      scoring keys are the one exclusion, because nothing about them is
      evidence the node was touched: see `SCORING_METADATA_KEYS`.

    What makes this safe is that `AndNode` carries the statements of what it
    proposed (`subgoal_signatures`), so a retired attempt's history survives
    its children. Without that field this method loses history. With it, the
    only thing lost is a dict entry nothing reads.

    Args:
      root_id: The graph's root. Never pruned.

    Returns:
      The ids that were deleted, in iteration order.
    """
    reachable = self.get_active_reachable_nodes(root_id)
    pruned = []
    for node_id, node in list(self.nodes.items()):
      if node_id == root_id or node_id in reachable:
        continue
      if (
          node.and_nodes
          or node.formal_theorem
          or node.informal_proof
          or set(node.search_metadata) - SCORING_METADATA_KEYS
      ):
        continue
      del self.nodes[node_id]
      pruned.append(node_id)
    return pruned

  def assemble_for_tree_context(self, target_node_id):
    """Assembles ancestor signatures and proof sketches for context in prover/decomposer."""
    if target_node_id not in self.nodes:
      return ""
    ancestry = []
    visited_ancestors = set()
    curr = self.nodes[target_node_id]
    while curr.reverse_dependency_ids:
      parent_id = curr.reverse_dependency_ids[0]
      if parent_id in self.nodes and parent_id not in visited_ancestors:
        visited_ancestors.add(parent_id)
        ancestry.append(self.nodes[parent_id])
        curr = self.nodes[parent_id]
      else:
        break
    ancestry.reverse()
    blocks = []
    for node in ancestry:
      if node.formal_sketch:
        blocks.append(node.formal_sketch.assemble_for_verification())
      elif node.formal_theorem:
        blocks.append(node.formal_theorem.assemble_for_verification())
    return "\n\n".join(blocks).strip()

  def get_non_ancestor_nodes(
      self, target_node_id
  ):
    """Finds all nodes in the graph that do not depend on the target node."""
    dependents = set()

    def _dfs_down(current_id):
      if current_id in dependents:
        return
      dependents.add(current_id)
      node = self.nodes.get(current_id)
      if node:
        for rev_id in node.reverse_dependency_ids:
          _dfs_down(rev_id)

    _dfs_down(target_node_id)

    results = []
    for node_id, node in self.nodes.items():
      if node_id not in dependents and node_id != target_node_id:
        if not node.search_metadata.get("decomposition_failed"):
          results.append(node)

    return results

  def get_closest_non_ancestor_nodes(
      self, target_node_id, limit = 15
  ):
    """Finds non-ancestor nodes ranked by solved status and undirected graph distance."""
    non_ancestors = self.get_non_ancestor_nodes(target_node_id)
    non_ancestor_ids = {n.id for n in non_ancestors}

    if not non_ancestor_ids:
      return []

    distances = {}
    queue = deque([(target_node_id, 0)])
    visited = {target_node_id}

    while queue:
      curr_id, dist = queue.popleft()
      if curr_id in non_ancestor_ids:
        distances[curr_id] = dist

      curr_node = self.nodes.get(curr_id)
      if not curr_node:
        continue

      neighbors = set(curr_node.reverse_dependency_ids)
      for and_node in curr_node.and_nodes:
        neighbors.update(and_node.dependency_ids)

      for nbr_id in neighbors:
        if nbr_id not in visited:
          visited.add(nbr_id)
          queue.append((nbr_id, dist + 1))

    status_map = self.compute_statuses()

    def _sort_key(node):
      is_solved = (status_map.get(node.id) == NodeStatus.FORMAL_PROVED)
      dist = distances.get(node.id, float("inf"))
      return (not is_solved, dist, node.id)

    sorted_nodes = sorted(non_ancestors, key=_sort_key)
    return sorted_nodes[:limit]

  def assemble_available_lemmas_context(
      self, target_node_id, limit = 15
  ):
    """Formats available non-ancestor lemmas with instructions and status for LLM context."""
    nodes = self.get_closest_non_ancestor_nodes(target_node_id, limit=limit)
    if not nodes:
      return "No available lemmas in the graph."

    status_map = self.compute_statuses()
    lines = [
        "### Available Graph Nodes (Non-Ancestor Lemmas)",
        "The following lemmas already exist in the proof graph and CAN be used without creating circular dependencies.",
        "Prefer using lemmas that are marked as SOLVED (FORMAL_PROVED).",
        "",
        "INSTRUCTIONS FOR REUSE:",
        "To reuse an existing lemma in your proof or sketch, you MUST include its exact declaration in your Lean snippet header with a sorry placeholder:",
        "  theorem <lemma_name> <signature> := by sorry",
        "Do not rename the lemma or alter its statement. The system will automatically link to the existing graph node instead of creating a duplicate.",
        "",
        "```lean",
    ]

    for node in nodes:
      st = status_map.get(node.id, NodeStatus.PROCESSING)
      status_str = "SOLVED" if st == NodeStatus.FORMAL_PROVED else "UNSOLVED"
      stmt = ""
      if node.formal_theorem:
        stmt = node.formal_theorem.signature.to_lean()
      elif node.formal_sketch:
        stmt = node.formal_sketch.signature.to_lean()
      elif node.informal_problem:
        raw_s = node.informal_problem.statement.strip()
        if not raw_s.startswith("theorem ") and not raw_s.startswith("lemma "):
          stmt = models.FormalSignature(
              name=node.id, statement=raw_s
          ).to_lean()
        else:
          stmt = raw_s

      if stmt:
        lines.append(f"-- [{status_str}] Node: {node.id}")
        lines.append(f"{stmt} := by sorry\n")

    lines.append("```")
    return "\n".join(lines)

  def get_accessible_nodes_data(
      self, target_node_id, limit = 25
  ):
    """Gathers unambiguous status, signature, and dependency metadata for all non-ancestor nodes."""
    target_node = self.nodes.get(target_node_id)
    if not target_node:
      return []

    status_map = self.compute_statuses()
    non_ancestors = self.get_closest_non_ancestor_nodes(
        target_node_id, limit=limit
    )

    sibling_ids = set()
    for p_id in target_node.reverse_dependency_ids:
      p_node = self.nodes.get(p_id)
      p_active = p_node.active_and_node if p_node else None
      if p_active is not None:
        sibling_ids.update(p_active.dependency_ids)
    sibling_ids.discard(target_node_id)

    results = []
    for node in non_ancestors:
      st = status_map.get(node.id, node.mathematical_status)
      st_val = st.value if isinstance(st, NodeStatus) else str(st)

      is_processing = bool(node.search_metadata.get("processing", False))
      if is_processing and st_val != "formal_proved":
        st_val = "processing"

      stmt = ""
      if node.formal_theorem:
        stmt = node.formal_theorem.signature.to_lean()
      elif node.formal_sketch:
        stmt = node.formal_sketch.signature.to_lean()
      elif node.informal_problem:
        raw_s = node.informal_problem.statement.strip()
        if not raw_s.startswith("theorem ") and not raw_s.startswith("lemma "):
          stmt = models.FormalSignature(
              name=node.id, statement=raw_s
          ).to_lean()
        else:
          stmt = raw_s

      deps = []
      if node.active_and_node is not None:
        deps = list(node.active_and_node.dependency_ids)

      results.append({
          "name": node.id,
          "signature": stmt,
          "status": st_val,
          "is_proved": (st_val == "formal_proved"),
          "is_processing": is_processing,
          "is_sibling": (node.id in sibling_ids),
          "dependencies": deps,
          "node_dir": f"nodes/{node.id}",
      })
    return results

  def assemble_backtracking_feedback(
      self, target_node_id, limit = 15
  ):
    """Assembles error messages and correction history for a node."""
    if target_node_id not in self.nodes:
      return []
    node = self.nodes[target_node_id]
    feedback = []
    last_err = node.search_metadata.get("last_error")
    if last_err:
      feedback.append(str(last_err))
    for and_node in node.and_nodes:
      if and_node.failed and and_node.error_message:
        feedback.append(str(and_node.error_message))
    return feedback[-limit:]
