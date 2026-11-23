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

"""Guards on the content `init_graph` is asked to seed a proof graph from.

`init_graph` splits its `--statement` at the root theorem: everything before
becomes `environment.shared_definitions`, which is the immutable header every
scaffold in the tree renders and every submission compiles against.

The split itself cannot fail. An orchestrator that passes only the theorem
text -- rather than the whole benchmark file -- produces an empty preamble,
and every downstream consequence of that is silent: the header renders as an
empty `-- [START_IMMUTABLE_CONTEXT]` pair, the benchmark's own definitions are
absent so provers redeclare them in `-- [START_DEFINITIONS]` (where the
collision gate cannot see them, because there is nothing to collide with), and
the tree proves a statement that is not the one the benchmark asked for. The
first visible symptom is an unknown-identifier error hours later, in a node
several generations down.

So it is checked here instead, once, at the only moment the fix is cheap.
"""

import re

import lean_code_parser as models
import lean_verifier_tool

# A Lean file that declares a theorem over Mathlib opens with an import. A
# preamble without one did not come from a file.
_IMPORT_RE = re.compile(r"^\s*import\s+\S", re.MULTILINE)

# `open X in` / `attribute [...] in` scope to the *next declaration only*.
_DANGLING_IN_RE = re.compile(r"\bin\s*\Z")


def check_preamble(preamble, root_id):
  """Returns an error message if the preamble cannot be a real file header.

  Args:
    preamble: Everything in the statement before the root theorem declaration.
    root_id: The root theorem's name, for the diagnostic.

  Returns:
    The error message, or "" when the preamble is usable.
  """
  if not preamble.strip():
    detail = (
        f"The text before 'theorem {root_id}' is empty, so the proof tree "
        "would have no header at all."
    )
  elif not _IMPORT_RE.search(preamble):
    detail = (
        "The text before 'theorem "
        f"{root_id}' contains no 'import' line, so it is not a whole file."
    )
  elif _DANGLING_IN_RE.search(models.strip_lean_comments(preamble)):
    # Not a "not a whole file" complaint, so it gets its own message and an
    # early return rather than falling through to the shared one below.
    return (
        "Rejected: the immutable header ends with a dangling 'in'.\n\n"
        f"The line before 'theorem {root_id}' is an 'open ... in' (or "
        "'attribute ... in'), which scopes to the next declaration only. Here "
        "the split has left it with nothing to modify: it binds to whatever a "
        "prover writes into '-- [START_DEFINITIONS]' next, and the root "
        "theorem silently loses it.\n\n"
        "ACTION REQUIRED: drop the trailing 'in' so the open applies to the "
        "whole file, then re-run init_graph."
    )
  else:
    return ""

  return (
      "Rejected: --statement is not the whole problem file.\n\n"
      f"{detail}\n\n"
      "--statement must be the *entire* contents of the source .lean file -- "
      "its imports, its 'open' lines, and every 'abbrev'/'def' the theorem "
      "refers to -- not just the theorem. Everything before the root theorem "
      "becomes the immutable header that every node in the tree compiles "
      "against; if it is missing, provers silently redeclare the benchmark's "
      "own definitions and prove the wrong statement.\n\n"
      "ACTION REQUIRED: re-run init_graph passing the full file contents, "
      "e.g. --statement=\"$(cat <path to the .lean file>)\"."
  )


def check_root_gate(signature):
  """Returns an error if the root node could never accept its own scaffold.

  Every node is issued a scaffold whose proof block is `<signature> :=\\nby\\n
  sorry`, and every submission against it is checked by
  `extract_proof_with_expected_signature`. Those two are supposed to be
  inverses. When they are not -- comments surviving on one side but not the
  other was one way for them not to be -- the node is unsubmittable by
  construction: no agent can produce text the gate accepts, the rejection says
  to copy a signature that is already copied, and the run burns hours against a
  wall. One problem spent ~2h and 215 rejections there.

  So close the loop once, here, on the only statement in the tree that comes
  from outside the tool. This is not a Lean check and adds no parsing rules --
  it feeds the tool its own output and insists that it accepts it.

  Args:
    signature: The root node's `signature_str`.

  Returns:
    The error message, or "" when the gate accepts its own scaffold.
  """
  scaffold_proof = f"{signature} :=\nby\n  sorry"
  accepted, detail, _ = models.extract_proof_with_expected_signature(
      models.strip_lean_comments(scaffold_proof).strip(), signature
  )
  if accepted:
    return ""
  return (
      "Rejected: the root statement produces a scaffold this tool cannot "
      "accept back.\n\nA proof block containing exactly the issued signature "
      "was refused by the submission gate, so no agent could ever close this "
      f"node.\n\n{detail}\n\n"
      "This is a bug in the tool, not in the statement. Report it with the "
      "statement above rather than editing the problem to work around it."
  )


def check_root_compiles(scaffold):
  """Returns an error if the root scaffold does not compile.

  This is the one check here that is not an approximation. A scaffold whose
  proof is `by sorry` must compile -- if it does not, the problem cannot be
  worked on by anyone, and every hour the tree spends on it is wasted. Ten
  seconds here against hours there.

  It also subsumes the textual guards above, and the ones nobody has thought of
  yet, because the oracle is Lean rather than a regex. That is the point: the
  way to stop being fragile at parsing is to stop being the authority on what
  parses.

  No `target_name` is passed, so the verifier compiles and does not check
  axioms -- `sorryAx` is expected here and must not fail the check.

  Args:
    scaffold: The rendered root scaffold, header and all.

  Returns:
    The error message, or "" when it compiles (or the verifier is unreachable).
  """
  result = lean_verifier_tool.run_lean_verifier(
      lean_verifier_tool.LeanVerifierInput(code=scaffold)
  )
  if result.is_valid or result.is_infra_failure:
    # An unreachable verifier is not evidence against the statement.
    return ""
  return (
      "Rejected: the root scaffold does not compile.\n\n"
      f"{result.error_message or ''}\n\n"
      "The proof body is 'by sorry', so this is a problem with the header or "
      "the statement itself, not with any proof. Nothing in this tree can "
      "succeed until it is fixed.\n\n"
      "ACTION REQUIRED: fix the source .lean file and re-run init_graph."
  )
