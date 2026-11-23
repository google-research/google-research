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

"""Why `claim_next` had nothing to hand out.

Split out of `state_manager._handle_claim_next`, which answered the question
with one sentence for four different situations:

    No available unclaimed subgoals found to claim.
    Graph is fully dead-ended or proved.

"Dead-ended or proved" is the one distinction the orchestrator most needs, and
that sentence refuses to make it. The two outcomes call for opposite actions --
assemble and report, versus diagnose a stall and keep dispatching -- and an
orchestrator that guesses wrong either declares a half-finished run done or
keeps grinding on a run that is already over. The graph knows the answer; it was
simply never asked.

`assemble` refuses on an unproved subtree, so a wrong guess in the optimistic
direction is caught one step later. A wrong guess in the *pessimistic*
direction is not caught at all -- there is nothing downstream of "keep
dispatching" to notice that the search finished twenty minutes ago.

The classification is text only. Nothing here mutates the graph, and no caller
branches on the result: the orchestrator is the one that has to read it, so the
distinction has to survive as prose.
"""

from typing import Optional

import dag


#: How many blocked node ids to name before summarising the rest. Matches the
#: cap `_handle_assemble` uses on its worklist, so the two refusals that list
#: the same nodes list them the same way.
MAX_LISTED = 40


def why_nothing_to_claim(graph, root_id):
  """Returns the `claim_next` refusal text for a graph with nothing claimable.

  Only call this once the scheduler has already come back empty -- this decides
  *why* it was empty, not whether it was.

  Args:
    graph: A `dag.ProofGraph`.
    root_id: The run's root. Falls back to the first node, which is what the
      rest of the state manager does when no root was configured; an empty
      graph has no root at all and gets its own message.

  Returns:
    One paragraph, written for an orchestrator that will act on it directly.
  """
  processing = sorted(
      nid
      for nid, node in graph.nodes.items()
      if node.search_metadata.get("processing")
  )
  if processing:
    # Deliberately first: workers still running means the question of whether
    # the search is over has not been decided yet, whatever the rest of the
    # graph currently looks like.
    listed = ", ".join(f"`{n}`" for n in processing[:MAX_LISTED])
    return (
        "No available unclaimed subgoals found, "
        f"BUT {len(processing)} nodes are STILL processing asynchronously! "
        "DO NOT assemble or terminate yet. Wait for workers to finish. "
        f"Held: {listed}."
    )

  root = root_id if root_id in graph.nodes else next(iter(graph.nodes), None)
  if root is None:
    return "The graph has no nodes. Run `--action=init_graph` first."

  # `_record_disproof` marks the node `decomposition_failed` on its way out, and
  # since `informal --action=failed` stopped being terminal that is now the only
  # thing that sets the flag -- so on the root it means exactly "refuted", and
  # this is the branch that says so.
  if graph.nodes[root].search_metadata.get("disproved"):
    return (
        f"ROOT DISPROVED: `{root}` carries a kernel-checked refutation. This is"
        " a RESULT and the search is finished. Do NOT assemble -- there is no"
        " proof to assemble. Report the disproof, citing the certificate under"
        f" `nodes/{root}/`."
    )

  statuses = graph.compute_statuses()
  open_nodes = sorted(
      nid
      for nid in graph.get_active_reachable_nodes(root)
      if statuses.get(nid) is not dag.NodeStatus.FORMAL_PROVED
  )
  if not open_nodes:
    return (
        f"SEARCH FINISHED: every node on the active branch under `{root}` is"
        " proved. Run `--action=assemble` and hand the assembled file to an"
        " independent verifier. A green assemble is a statement about the"
        " graph, not about Lean, so the verifier is not optional."
    )

  listed = ", ".join(f"`{n}`" for n in open_nodes[:MAX_LISTED])
  more = "" if len(open_nodes) <= MAX_LISTED else (
      f" (+{len(open_nodes) - MAX_LISTED} more)"
  )
  return (
      f"STALLED: {len(open_nodes)} node(s) under `{root}` are unproved and none"
      f" is claimable: {listed}{more}.\nThis is NOT a result and the search is"
      " NOT finished. Each of these is either dead under a decomposition its"
      " parent has not replaced yet, or blocked behind one that is. Run"
      " `--action=inspect`, find the failed node nearest the root, and get its"
      " parent replanned. There is no give-up condition here; exhaustion is not"
      " an outcome."
  )
