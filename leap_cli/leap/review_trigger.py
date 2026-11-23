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

"""When a `blocked` node gets pulled back for review.

A decomposition is normally reviewed exactly once, the moment it is proposed.
That review judges shape -- do the children close the target, is any of them a
restatement of an ancestor -- with no evidence of how the plan actually plays
out. This module is the second entrance: a branch whose nodes are each being
re-planned over and over without anything closing is a branch worth a second
opinion, and the cheapest place to get one is from the ancestor that could
stop needing it.

Firing the predicate reviews; it does not retire. Nothing here ends a node's
work. The two things that do are the planner calling its own goal dead and an
ancestor's reviewer returning `replan`, and this module exists to put that
ancestor in front of a reviewer while the question is still worth asking.

The trigger is a re-planning rate, not depth and not time. Depth measured the
wrong thing: a node stuck proposing decompositions that get retired collapses
its own live subtree back to one generation every time one dies, so its
ancestors' depth reading never moves and nothing ever wakes them. Time is not
usable at all -- the graph is persisted JSON that gets resumed and replayed, and
a wall-clock stamp would make the same run replay differently.
`informal_dispatches` is graph state, replays identically, and climbs even when
the agent below dies without recording anything.

Kept out of `dag.py` deliberately -- this is search policy, not graph structure,
and it reads nothing but public attributes.
"""

from typing import Any, Dict, Optional

import pipeline


def has_been_reviewed(node):
  """Whether a reviewer has already ruled on this node's LIVE decomposition.

  On the AND-node and not the node, because the question the briefing asks with
  it is "am I the commit gate or the patrol", and that is a question about the
  plan in front of the reviewer, not about the node's whole history. A node
  reviewed once, told to `replan`, and back with a fresh split is at a commit
  gate again -- a mark left on the node would have called that second first
  look a patrol and briefed the reviewer to protect work that does not exist.
  """
  and_node = node.active_and_node
  return and_node is not None and and_node.review_replan_rate is not None


def child_replan_rate(graph, child_id, memo):
  """Planning attempts attributable to one child branch.

  The larger of two numbers, because a branch can be going badly in two ways
  that do not share a scale.

  Its own `informal_dispatches`. A node sent to a planner 34 times that has
  never managed to produce a decomposition is the whole problem, and it has no
  subtree to average over -- folding it into a mean over the children it never
  proposed reports a small number for the worst case there is.

  The mean `informal_dispatches` over its live branch, itself included. This is
  the other shape: no single node stuck, but everything under it re-planning.
  The mean counts descendants that are already proved, and they are the decay --
  a branch that has closed forty lemmas should not be pulled back over the
  forty-first, and the arithmetic that says so is the forty sitting in the
  divisor. Counting only the unproved remainder instead measures how hard the
  work that is LEFT is, which reads higher the closer a branch gets to done. On
  the run archive that lifted uninvolved ancestors to 18.0 against nodes that
  genuinely needed review at 16.0, and no threshold separates those.

  Args:
    graph: A `dag.ProofGraph`.
    child_id: The child to measure. Must be present in `graph.nodes`.
    memo: Cache keyed by child id. Safe to reuse for a whole scan; see
      `should_review` for why the graph does not move underneath it.

  Returns:
    The larger of the child's own count and its branch mean.
  """
  if child_id in memo:
    return memo[child_id]
  # get_active_reachable_nodes carries its own visited set, so a cycle in a
  # corrupt graph terminates here rather than recursing forever. It includes
  # `child_id` itself, so the divisor is only ever zero for an id that is not
  # in the graph, which the caller has already excluded.
  branch = [
      graph.nodes[d]
      for d in graph.get_active_reachable_nodes(child_id)
      if d in graph.nodes
  ]
  total = sum(
      int(n.search_metadata.get("informal_dispatches", 0)) for n in branch
  )
  own = int(graph.nodes[child_id].search_metadata.get("informal_dispatches", 0))
  rate = max(float(own), total / len(branch) if branch else 0.0)
  memo[child_id] = rate
  return rate


def worst_child_replan_rate(
    graph,
    node_id,
    memo = None,
    statuses = None,
):
  """How badly the worst child of this node's live decomposition is going.

  The maximum over the children, not a mean over the whole subtree. The mean was
  the original design and it had one failure that mattered: a split into four
  children, one of them hopeless and three of them finishing normally, reads as
  a healthy branch. The three healthy children pay for the fourth, the reading
  never reaches the floor, and the one node that could retire the hopeless child
  is never asked. A maximum judges a plan by the worst thing in it, which is the
  thing a reviewer would actually be ruling on.

  It still falls off with height, which is what keeps the scheduler's top-down
  walk from stopping at the root every time. The parent of a stalled node reads
  that node's own count; its grandparent reads a mean over a branch containing
  the stall plus everything healthy beside it. So the walk lands near a local
  stall, and only a plan failing everywhere is still high at the root -- and
  when a parent is reviewed and keeps the child anyway, the child keeps
  climbing, the grandparent's mean climbs behind it, and the review escalates
  on its own.

  Children that are already proved are skipped. A child proved on its 25th
  attempt would otherwise keep reporting 25 forever and keep pulling its parent
  back over finished work. Proved-ness has to come from the resolver and not
  from `formal_theorem is not None`: that field is a DIRECT proof only, and a
  child closed by its own decomposition carries a sketch and no theorem. On the
  run archive, testing the field alone called 301 of 1136 proved nodes live.

  Follows only `active_and_node`, so a retired decomposition's children stop
  counting the moment it is retired -- which is what makes a `replan` actually
  relieve the pressure on the ancestors instead of leaving it banked.

  Args:
    graph: A `dag.ProofGraph`.
    node_id: The node to measure. An id with no node, and a node with no live
      decomposition, both read 0.0 -- a leaf has written no plan, so there is
      nothing to hold it to and nothing for a reviewer to rule on.
    memo: Optional cache of per-child rates, safe to reuse across a whole DFS
      scan.
    statuses: The caller's node-id to status map. `claim_next` already holds one
      from `propagate_failures()`, and passing it is not just cheaper than
      resolving once per node -- that call retires AND-nodes with dead children
      as a side effect, so a second, independent resolve would rule on a graph
      the surrounding walk does not believe in. Resolved here if omitted.

  Returns:
    The worst live child's rate, or 0.0 if there is no live decomposition or
    every child of it is proved.
  """
  memo = {} if memo is None else memo
  node = graph.nodes.get(node_id)
  and_node = node.active_and_node if node is not None else None
  if and_node is None:
    return 0.0
  if statuses is None:
    statuses = graph.compute_statuses()
  # `dag.NodeStatus` subclasses `str`, so comparing against the literal works
  # without importing dag and keeps this module clear of graph internals.
  return max(
      (
          child_replan_rate(graph, child_id, memo)
          for child_id in and_node.dependency_ids
          if child_id in graph.nodes
          and statuses.get(child_id) != "formal_proved"
      ),
      default=0.0,
  )


def review_threshold(node):
  """The rate this node's worst child must reach before it is pulled back.

  The floor, unless a reviewer has already approved the live decomposition, in
  which case the reading that approval saw, multiplied by the backoff. Two
  things keep that bar from running away, and between them there is no ceiling
  constant here and no need for one.

  Multiplied rather than incremented. A fixed increment only backs anything off
  if it is large next to the reading, and these readings are unbounded -- a node
  sent to a planner 78 times reads 78. On the run archive an increment of 2 put
  the same node in front of a reviewer between 30 and 285 times in a row, the
  bar crawling up 2 at a time behind a reading it could not catch; doubling ends
  the same sequence in 5 to 7 reviews. A multiplier also holds its meaning as
  the readings grow, which an increment calibrated against a 0-2 scale did not.

  It dies with the plan it approved. The mark is on the AND-node, so a
  reviewer's `replan` takes it out of the graph along with the decomposition
  it was a judgement about. The replacement plan starts at the floor: nothing
  is remembered about an epoch that has been excised.

  It ratchets down within the plan, in `should_review`. A node approved while a
  child was burning twenty re-plans is quiet until 40 -- but if that child is
  retired and the reading collapses to 1.5, the mark collapses with it and the
  bar is back at the floor. The bar is never more than the backoff times the
  worst this plan has looked, which is a reading, so it cannot outrun one.

  A mark of 0.0 -- what the commit-gate review stores, since no child has run
  yet -- multiplies to 0.0 and the floor governs, which is the intent: a freshly
  approved plan gets the full floor allowance.
  """
  and_node = node.active_and_node
  floor = float(pipeline.SUBTREE_REVIEW_REPLANS_PER_NODE)
  if and_node is None or and_node.review_replan_rate is None:
    return floor
  return max(
      floor, and_node.review_replan_rate * pipeline.SUBTREE_REVIEW_BACKOFF
  )


def should_review(
    graph,
    node,
    memo = None,
    statuses = None,
):
  """Whether the `subtree_stalled` predicate fires for this node.

  Ratchets the mark down before comparing, which is the half of the backoff
  that has to be written rather than derived: the bar is the backoff applied to
  the LOWEST the reading has been since the approval, not to where it was on the
  day. Persisting it is the point -- comparing against the live reading without
  storing it lowers the number and changes nothing, because a bar of `rate *
  backoff` is one the rate can never reach.

  Mutating from a predicate is deliberate and safe here: every caller runs
  inside `graph_transaction`, and the mark only ever decreases, so an extra
  evaluation cannot raise the bar and a replay lands on the same value.

  `memo` and `statuses` are shared across a whole scan, which is sound because
  nothing the scan does between two calls changes either of them.
  `propagate_failures` runs once before the walk starts; the phase edge this
  predicate's caller applies is not read by the resolver or by the rates; and
  the one thing that does move a rate -- `informal_dispatches` -- is incremented
  only on a path that immediately returns a claim and ends the walk.

  The caller is responsible for only asking about nodes in `blocked`, and for
  asking after the two resolution predicates: a subtree that just finished must
  not be sent to a reviewer on its way out.
  """
  rate = worst_child_replan_rate(graph, node.id, memo, statuses)
  and_node = node.active_and_node
  if and_node is not None and and_node.review_replan_rate is not None:
    and_node.review_replan_rate = min(and_node.review_replan_rate, rate)
  return rate >= review_threshold(node)


def record_approval(
    graph,
    node,
    memo = None,
    statuses = None,
):
  """Records what a reviewer approved. Returns the rate that was on the table.

  Stores the reading itself and not a computed next threshold, so the bar is
  always a comparison against the subtree rather than a number carried forward
  that nobody can talk back down.

  Applies to the decomposition review too. That one fires on submission, before
  any child has run, so it stores 0.0 and the fresh plan gets the full floor
  allowance. Its job there is to be PRESENT: the briefing tells the commit gate
  from the patrol by whether the live AND-node carries a mark, and 0.0 is a
  real reading, not an absent one.

  Raises:
    ValueError: if the node has no live decomposition. Both entrances into
      `review` guarantee one, so there is nowhere to put the mark and the node
      would be pulled back on the very next scan -- a livelock, not a
      degradation, and worth stopping the action for.
  """
  and_node = node.active_and_node
  if and_node is None:
    raise ValueError(
        f"Node '{node.id}' was approved by a reviewer but has no live"
        " decomposition to record the approval against."
    )
  rate = worst_child_replan_rate(graph, node.id, memo, statuses)
  and_node.review_replan_rate = rate
  return rate
