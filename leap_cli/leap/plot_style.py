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

"""How one proof-graph node is drawn.

Split out of `state_manager._generate_dot`, which had grown a fifteen-branch
if/elif over two colour dictionaries. The split is not cosmetic: what a node is
*doing* is now the phase, and the phase lives in `search_metadata`, so the
drawing has to read the same field the scheduler dispatches on. Keeping that
reading here means the picture and the pipeline table can be checked against
each other in one place.

Two things the old drawing got wrong, both fixed here:

  * **Phase was invisible.** A node waiting for an informal plan and a node
    waiting for a reviewer were both `formal_statement`, both lightgray, and
    both indistinguishable from each other in a live run -- so the picture
    could not answer the one question anyone opens it to ask, which is where
    the work is.

  * **A disproof was drawn as a crash.** `_record_disproof` sets
    `decomposition_failed` on its way out, because nothing under a refuted
    proposition can be worked again. The drawing read that marker and painted
    failure red over the single most valuable outcome a run can produce. The
    `disproved` marker written alongside it is checked first.
"""

import time
from typing import Any, Dict, Tuple

import pipeline


#: Fill and font for a node still inside the machine, keyed by phase. Every
#: phase in `pipeline.PIPELINE` needs an entry, and each entry has to be
#: distinct -- `PlotStateVisibilityTest` checks both.
PHASE_STYLES: Dict[str, Tuple[str, str]] = {
    "informal": ("#E0E0E0", "black"),
    "formal": ("#90CAF9", "black"),
    "disprove": ("#CE93D8", "black"),
    "review": ("#FFF59D", "black"),
    "blocked": ("#FFFFFF", "black"),
}

#: Fill and font for a node that has left the machine. The three ends a run can
#: reach are three colours on purpose -- proved green, refuted purple, dead red
#: -- because the first two are successes and only the third is bad news.
TERMINAL_STYLES: Dict[str, Tuple[str, str]] = {
    "PROVED": ("#4CAF50", "black"),
    "PROVED (via split)": ("#2E7D32", "white"),
    "DISPROVED": ("#6A1B9A", "white"),
    "FAILED": ("#C62828", "white"),
    "BROKEN": ("#8B0000", "white"),
}

#: Fill and font for a node in no state this module recognizes.
UNKNOWN_STYLE: Tuple[str, str] = ("#FFFFFF", "black")


def held_for(metadata):
  """How long the current claim has been out, as a short phrase.

  A claim never expires -- releasing one is a deliberate orchestrator action --
  so a node held by a worker that died looks exactly like a node held by a
  worker that is thinking hard. The age is the only thing that separates them,
  and the picture is where anyone would notice.
  """
  started = metadata.get("claim_timestamp")
  if not isinstance(started, (int, float)) or isinstance(started, bool):
    return "claimed"
  return f"held {max(0, int((time.time() - started) // 60))}m"


def describe(node, status_str):
  """Returns `(state, fillcolor, fontcolor)` for one node.

  The order of the checks is load-bearing, because the markers overlap by
  design: a disproved node also carries `decomposition_failed`, and a claimed
  node still has whatever phase it was claimed out of.

  Args:
    node: A `dag.ProofGraphNode`.
    status_str: The node's status from `graph.compute_statuses()`, as a plain
      string. Only consulted for the two things the metadata cannot say on its
      own: whether the subtree closed, and whether the graph is malformed here.

  Returns:
    The text that goes on the node's second label line, and its two colours.
  """
  metadata = node.search_metadata

  if metadata.get("disproved"):
    state = "DISPROVED"
  elif status_str == "broken":
    state = "BROKEN"
  elif metadata.get("decomposition_failed"):
    state = "FAILED"
  elif status_str == "formal_proved":
    state = "PROVED (via split)" if node.and_nodes else "PROVED"
  else:
    return _live(metadata)

  fill, font = TERMINAL_STYLES[state]
  return state, fill, font


def _live(metadata):
  """The `(state, fill, font)` of a node still inside the state machine."""
  phase = metadata.get("phase")
  if phase is None:
    # Not proved, not refuted, not dead, and out of the machine anyway. No
    # transition produces this, so it means the graph was edited by hand or a
    # write was lost, and a blank white node saying so beats one that blends in
    # with the phase colours.
    return "left the machine", *UNKNOWN_STYLE

  fill, font = PHASE_STYLES.get(phase, UNKNOWN_STYLE)
  state = f"phase: {phase}"
  if metadata.get("processing"):
    state = f"{state} ({held_for(metadata)})"
  return state, fill, font


def faded(color):
  """Returns `color` at 30% alpha, for a node outside the active subtree."""
  return f"{color}4D"


def missing_phase_styles():
  """Phases in the pipeline table that this module has no colour for.

  Adding a phase without a colour is not an error at runtime -- the node comes
  out white and readable -- but it is a gap, and the test asserts there is
  none.
  """
  return tuple(sorted(set(pipeline.PIPELINE) - set(PHASE_STYLES)))
