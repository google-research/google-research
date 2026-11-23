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

"""Lean 4 code parser, formal context models, and proof assembly utilities."""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import pydantic



class SharedHeaderBlock(pydantic.BaseModel):
  """A definition/header block injected globally by an agent.

  The containing list is append-only and strictly monotonic: blocks are never
  removed, reordered, or replaced. That is what makes list order a valid
  dependency order, since it is exactly submission order.

  `node_id` and `and_node_id` are provenance for debugging and reference only.
  Nothing keys on them; filtering or de-duplicating by node is what previously
  reordered the list and broke definitions that referenced earlier ones.
  `and_node_id` has one consumer beyond debugging -- retirement, below -- and it
  still does not reorder anything.

  A block is appended when its node *submits*, not when its decomposition is
  approved, so between those two moments it is already in every other node's
  environment. If the reviewer then retires the decomposition, the block is
  residue: nothing references it, nothing can, and it compiles forever. That is
  what `retired` marks. It is the one mutation this structure allows, and it is
  on the right side of append-only -- nothing is removed, reordered or replaced,
  the block stays in the JSON for forensics, and what D1 protects is the header
  other nodes were *verified against*, which a retired decomposition's block was
  never part of. Retirement is refused outright if anything live turns out to
  reference the names, so that guarantee is checked rather than assumed.
  """
  content: str = pydantic.Field(default="")
  node_id: str = pydantic.Field(default="")
  and_node_id: str = pydantic.Field(
      default="",
      description="Decomposition that proposed this block, if any. Provenance only.",
  )
  retired: bool = pydantic.Field(
      default=False,
      description=(
          "The decomposition that proposed this block was retired by review"
          " before anything could depend on it. Skipped by to_lean_header and"
          " by the name registry; kept here for forensics, which does mean the"
          " raw list is no longer a picture of the live environment."
          " to_lean_header is."
      ),
  )

class FormalEnvironment(pydantic.BaseModel):
  """Represents the shared environment constraints required to compile Lean code."""

  imports: List[str] = pydantic.Field(
      default_factory=list,
      description=(
          "List of module paths to import, e.g. ['Mathlib.Data.Real.Basic']"
      ),
  )
  open_namespaces: List[str] = pydantic.Field(
      default_factory=list, description="Namespaces to open, e.g. ['Real']"
  )
  shared_definitions: str = pydantic.Field(
      default="", description="Any custom shared defs, structures or limits."
  )
  dynamic_headers: List[SharedHeaderBlock] = pydantic.Field(
      default_factory=list, description="Agent-proposed shared definitions."
  )

  def to_lean_header(self):
    """Compiles the environment into a valid Lean header block.

    Emits every dynamic header in submission order, minus the retired ones.
    There is deliberately no per-node filtering: verification and assembly must
    agree on one header set, and filtering by an "active" node set made a proof
    verify against one set and assemble against another. Retirement is not that
    -- it is a property of the block, identical for every caller, so the one
    header set stays one header set.
    """
    parts = []
    deduplicated_imports = list(dict.fromkeys(self.imports))
    for imp in deduplicated_imports:
      parts.append(f"import {imp}")

    deduplicated_opens = list(dict.fromkeys(self.open_namespaces))
    for ns in deduplicated_opens:
      parts.append(f"open {ns}")

    if self.shared_definitions.strip():
      parts.append(self.shared_definitions.strip())

    for block in self.dynamic_headers:
      if block.retired:
        continue
      clean_block = block.content.strip()
      # Exact-duplicate collapse: re-recording a node appends an identical
      # block, which must not become a redeclaration error.
      if clean_block and clean_block not in parts:
        parts.append(clean_block)

    return "\n".join(parts)


class InformalProblem(pydantic.BaseModel):
  """The natural language problem specification."""

  statement: str = pydantic.Field(
      description="The natural language description of the problem to solve."
  )


class InformalProof(pydantic.BaseModel):
  """The natural language mathematical proof."""

  text: str = pydantic.Field(description="The informal proof text.")


class FormalSignature(pydantic.BaseModel):
  """The proposition/claim part of a theorem without the proof."""

  name: str = pydantic.Field(description="The identifier name for the theorem")
  statement: str = pydantic.Field(
      description=(
          "The mathematical claim being made, e.g. 'Irrational (Real.sqrt 2)'"
      )
  )

  def to_lean(self):
    """Converts the signature to a Lean theorem declaration."""
    stmt = self.statement.strip()
    if re.search(
        r"^(?:@\[[^\]]*\]\s+)?(?:noncomputable\s+)?(?:theorem|lemma|def)\b",
        stmt,
    ):
      return stmt
    # A statement is already 'binders : prop' only if it has a top-level colon
    # whose prefix is nothing but binder groups. Testing the first character
    # alone misclassifies propositions that merely open with a bracket, e.g.
    # '(2:Nat) + 2 = 4', and drops the colon to emit invalid Lean.
    if not stmt.startswith(":"):
      colon_idx = find_depth0_colon(stmt)
      if colon_idx <= 0 or not is_binder_only(stmt[:colon_idx]):
        stmt = f": {stmt}"
    return f"theorem {self.name} {stmt}"


class FormalProof(pydantic.BaseModel):
  """The tactical justification block for a theorem."""

  tactics: str = pydantic.Field(
      description="The Lean tactical block, usually starting with 'by'"
  )


class FormalTheorem(pydantic.BaseModel):
  """A fully assembled theorem combining a signature, proof, and environment."""

  signature: FormalSignature
  proof: FormalProof
  environment: Optional[FormalEnvironment] = None

  def assemble_for_verification(self):
    """Builds the entire string representing a standalone compileable file."""
    header_str = self.environment.to_lean_header() if self.environment else ""
    return f"{header_str}\n\n{self.signature.to_lean()} :=\n{self.proof.tactics}".strip()


class FormalLemma(FormalTheorem):
  """A Lemma that is explicitly allowed/expected to contain 'sorry'."""

  pass


class FormalProofSketch(FormalTheorem):
  """A sketch with a signature, proof, relative lemmas, and environment."""

  lemmas: List[FormalLemma] = pydantic.Field(
      default_factory=list, description="List of lemmas used in the proof"
  )

  def assemble_for_verification(self):
    """Builds the entire string representing a standalone compileable file."""
    header_str = self.environment.to_lean_header() if self.environment else ""
    lemmas_str = ""
    for lemma in self.lemmas:
      lemmas_str += f"{lemma.signature.to_lean()} :=\n{lemma.proof.tactics}\n\n"
    theorem_str = f"{self.signature.to_lean()} :=\n{self.proof.tactics}"
    return f"{header_str}\n\n{lemmas_str}{theorem_str}".strip()


class ExtractedSubgoalsContext(pydantic.BaseModel):
  """Specifically for tracking multiple child theorems or subgoals."""

  subgoals: List[FormalTheorem] = pydantic.Field(
      description="Independent formal theorems representing subgoals"
  )


# ==============================================================================
# Parsing & Assembly Utilities
# ==============================================================================


def _scan_lean_comments(text):
  """Returns [start, end) spans of every comment in Lean source.

  Handles the two cases a regex cannot: Lean 4 block comments nest, and '--'
  inside a string literal is not a comment.
  """
  spans = []
  i = 0
  n = len(text)
  while i < n:
    char = text[i]
    if char == '"':
      # Skip a string literal wholesale so its contents are never scanned.
      j = i + 1
      while j < n:
        if text[j] == "\\":
          j += 2
          continue
        if text[j] == '"':
          break
        j += 1
      i = j + 1
    elif text.startswith("/-", i):
      depth = 1
      j = i + 2
      while j < n and depth:
        if text.startswith("/-", j):
          depth += 1
          j += 2
        elif text.startswith("-/", j):
          depth -= 1
          j += 2
        else:
          j += 1
      spans.append((i, j))
      i = j
    elif text.startswith("--", i):
      j = text.find("\n", i)
      j = n if j == -1 else j
      spans.append((i, j))
      i = j
    else:
      i += 1
  return spans


def strip_lean_comments(text):
  """Removes block and line comments from Lean code."""
  spans = _scan_lean_comments(text)
  if not spans:
    return text
  out = []
  prev = 0
  for start, end in spans:
    out.append(text[prev:start])
    prev = end
  out.append(text[prev:])
  return "".join(out)


def mask_lean_comments(text):
  """Replaces comments with spaces to preserve string indices."""
  if not text:
    return text
  chars = list(text)
  for start, end in _scan_lean_comments(text):
    for i in range(start, end):
      if chars[i] != "\n":
        chars[i] = " "
  return "".join(chars)


def strip_lean_attributes(text):
  # Potentially deprecated: no caller in dag.py / state_manager.py / workspace_manager.py.
  """Removes @[...] attribute blocks from Lean code, handling nested brackets."""
  result = []
  i = 0
  n = len(text)
  while i < n:
    if text[i : i + 2] == "@[":
      depth = 0
      j = i + 2
      while j < n:
        if text[j] == "[":
          depth += 1
        elif text[j] == "]":
          if depth == 0:
            i = j + 1
            break
          depth -= 1
        j += 1
      else:
        result.append(text[i:])
        break
    else:
      result.append(text[i])
      i += 1
  return "".join(result)


def normalize_string(s):
  """Normalizes a string by removing all whitespace."""
  return "".join(s.split())


_OPEN_BRACKETS = "([{⦃⟨"
_CLOSE_BRACKETS = ")]}⦄⟩"


def find_depth0_colon(text):
  """Returns the index of the first top-level ':' separator, or -1.

  Ignores colons nested inside brackets (e.g. the ':' in '(2:Nat)') and stops
  at a top-level ':=', which introduces a proof rather than a proposition.
  """
  depth = 0
  for i, char in enumerate(text):
    if char in _OPEN_BRACKETS:
      depth += 1
    elif char in _CLOSE_BRACKETS:
      depth -= 1
    elif char == ":" and depth == 0:
      if i + 1 < len(text) and text[i + 1] == "=":
        return -1
      return i
  return -1


def is_binder_only(text):
  """True if text is nothing but bracketed binder groups and whitespace.

  Distinguishes 'binders : prop' from a bare proposition that happens to start
  with a bracket. '(n : Nat) ' is binder-only; '∀ x ' is not.
  """
  depth = 0
  for char in text:
    if char in _OPEN_BRACKETS:
      depth += 1
    elif char in _CLOSE_BRACKETS:
      depth -= 1
    elif depth == 0 and not char.isspace():
      return False
  return True


def extract_proof_with_expected_signature(
    proof_text,
    expected_sig,
):
  """Strictly verifies that proof_text starts with expected_sig and extracts the proof body.

  Leading comments and docstrings are stripped before signature comparison.
  Directly matches the expected signature token pattern followed by ':='.

  Args:
    proof_text: Raw content from '-- [START_PROOF]'.
    expected_sig: Canonical expected signature (e.g. 'theorem foo (n : Nat) : n + 0 = n').

  Returns:
    (is_valid, proof_body_or_error_msg, clean_declaration)
  """
  clean_proof = strip_lean_comments(proof_text).strip()
  # Both sides, or the comparison is not a comparison. `clean_proof` has had its
  # comments removed; a signature carrying one -- which any statement copied out
  # of a benchmark file may -- then contributes a token the submission can never
  # contain, and the node becomes unsubmittable by anyone.
  tokens = strip_lean_comments(expected_sig).split()
  pattern = r"^\s*" + r"\s+".join(re.escape(t) for t in tokens) + r"\s*:="
  match = re.search(pattern, clean_proof)
  if not match:
    return (
        False,
        (
            f"TARGET THEOREM SIGNATURE MISMATCH:\n"
            f"Expected start of proof block:\n  {expected_sig} :=\n"
            f"The proof block must begin with the exact target theorem signature."
        ),
        "",
    )

  proof_body = clean_proof[match.end() :].strip()
  clean_declaration = f"{expected_sig} :=\n{proof_body}"
  return True, proof_body, clean_declaration


# The two keywords that bind a name with `:=` inside a *statement* body. Their
# `:=` sits at bracket depth 0 exactly like the proof's does, so "first depth-0
# `:=`" is not a proof detector. Deliberately not `set`/`suffices`: those are
# tactics, so they only appear after a `:= by` the primary regex already caught,
# and `suffices h : P from e` has no `:=` to claim -- counting it could only eat
# a `:=` belonging to something else.
_VALUE_BINDER_RE = re.compile(r"\b(?:let|have)\b")


def split_signature_and_proof(main_block):
  """Splits a Lean declaration block into signature and proof body.

  Not a substitute for `extract_proof_with_expected_signature` on a submission
  path: this reads whatever signature is there, where a submission has to be
  checked against the one the graph issued.
  """
  main_sig_str = main_block
  main_proof_str = "sorry"
  main_block_clean = mask_lean_comments(main_block)

  match = re.search(r":=\s*(by|sorry)", main_block_clean)
  if match:
    idx = match.start()
    main_sig_str = main_block[:idx]
    keyword = match.group(1)
    kw_start = match.start(1)
    proof_body = main_block[kw_start + len(keyword) :]

    if keyword == "by":
      main_proof_str = "by\n" + re.sub(r"^[ \t]*\n", "", proof_body)
    else:
      main_proof_str = "sorry" + proof_body.strip()
  else:
    # Fallback for a term-mode proof (`:= rfl`, `:= fun ...`), which has no
    # `by`/`sorry` to anchor on. Consume each value binder together with the
    # `:=` that belongs to it; the first `:=` left unclaimed is the proof.
    # Taking the first one outright split `theorem f : let k := n + 1; k > n
    # := rfl` at the `let`, recording the statement as `theorem f : let k`.
    #
    # Miscounting is one-sided by construction. Over-counting (a `let` inside a
    # depth-0 string literal, which mask_lean_comments does not hide) claims no
    # `:=`, so the block is returned whole, the proof survives into the
    # signature, and the scaffold fails to compile -- loud. Under-counting
    # would truncate silently, and needs a depth-0 `:=` bound by a keyword that
    # is neither `let` nor `have`; Lean's term grammar has no third one.
    idx = -1
    depth = 0
    pending = 0
    i = 0
    while i < len(main_block_clean):
      c = main_block_clean[i]
      if c in "([{⦃⟨":
        depth += 1
      elif c in ")]}⦄⟩":
        depth -= 1
      elif depth == 0:
        if c == ":" and main_block_clean[i + 1 : i + 2] == "=":
          if not pending:
            idx = i
            break
          pending -= 1
          i += 2
          continue
        binder = _VALUE_BINDER_RE.match(main_block_clean, i)
        if binder:
          pending += 1
          i = binder.end()
          continue
      i += 1

    if idx != -1:
      main_sig_str = main_block[:idx]
      main_proof_str = main_block[idx + 2 :].strip()

  return main_sig_str.strip(), main_proof_str.strip()


_END_PATTERN = re.compile(r"^end\s+[\w'.]+$")
_OPEN_PATTERN = re.compile(r"^open\s+[\w'.\s]+$")
_OPEN_SCOPED_PATTERN = re.compile(r"^open\s+scoped\s+[\w'.\s]+$")


def clean_declaration_block(block):
  """Strips any trailing namespace closures, opens, and comments from a block."""
  lines = block.rstrip().splitlines()
  while lines:
    last_line = lines[-1].strip()
    if not last_line:
      lines.pop()
    elif (
        _END_PATTERN.match(last_line)
        or _OPEN_PATTERN.match(last_line)
        or _OPEN_SCOPED_PATTERN.match(last_line)
        or last_line.startswith("--")
    ):
      lines.pop()
    elif last_line.startswith("/-") or last_line.endswith("-/"):
      lines.pop()
    else:
      break
  return "\n".join(lines).rstrip()


# Modifiers that may precede a declaration keyword without changing what is
# declared. Only 'noncomputable' used to be accepted, so 'private def foo' was
# not recognized as a declaration at all: the name never reached the collision
# registry, and classify_definitions_block filed the line under 'unnamed' and
# rejected a perfectly ordinary helper. Lean allows them in any order, hence the
# repeated group. 'local'/'scoped' are excluded here because they change where a
# declaration is visible; they are handled by the header-form patterns below.
_DECL_MODIFIERS = (
    r"(?:(?:private|protected|noncomputable|partial|nonrec|unsafe)\s+)*"
)
_DECL_KEYWORDS = (
    r"(?:theorem|lemma|def|structure|variable|class|instance|abbrev|inductive"
    r"|opaque)"
)
# Same modifier set plus the visibility ones, for matching a raw source line.
_HEADER_FORM_PREFIX = (
    r"^(?:(?:private|protected|noncomputable|partial|nonrec|unsafe|local"
    r"|scoped)\s+)*"
)


def extract_declaration_keyword(block):
  """Extracts the keyword ('def', 'lemma', ...) of a top-level declaration.

  The sibling of `extract_declaration_name`, matching the same prefix. Which
  keyword a declaration uses is what decides whether it may enter the shared
  header, so the two are read together.
  """
  match = re.match(
      rf"^(?:@\[[^\]]*\]\s+)?{_DECL_MODIFIERS}({_DECL_KEYWORDS})\s+", block
  )
  return match.group(1) if match else None


def extract_declaration_name(block):
  """Extracts the identifier name of a top-level Lean declaration."""
  name_match = re.match(
      # `\w` is Unicode by default in Python 3, so it covers the Greek letters
      # and subscripts Lean identifiers routinely use. An ASCII class read
      # `δbound` as no name at all -- so classify_definitions_block filed it
      # under `unnamed` and rejected the whole block -- and read both `h₁` and
      # `h₂` as `h`, which is a spurious collision. `'` is not a word
      # character, hence the explicit `['.]`.
      rf"^(?:@\[[^\]]*\]\s+)?{_DECL_MODIFIERS}{_DECL_KEYWORDS}\s+([\w'.]+)",
      block,
  )
  if name_match:
    return name_match.group(1)
  return None


def split_declarations(code):
  """Splits a block of Lean code into individual top-level declarations."""
  decl_pattern = re.compile(
      rf"^[ \t]*(?:@\[[^\]]*\]\s+)?{_DECL_MODIFIERS}{_DECL_KEYWORDS}\s+",
      re.MULTILINE,
  )
  clean_code = mask_lean_comments(code)
  matches = list(decl_pattern.finditer(clean_code))
  blocks = []
  if matches:
    if matches[0].start() > 0:
      initial_block = code[: matches[0].start()].strip()
      if initial_block:
        blocks.append(initial_block)
    for i, match in enumerate(matches):
      start = match.start()
      end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
      blocks.append(code[start:end].strip())
  elif code.strip():
    blocks.append(code.strip())
  return blocks


# Every dynamic header block is hoisted above every proof in the assembled file
# (see FormalEnvironment.to_lean_header and ProofGraph.assemble_current_proof),
# so a block committed late is elaborated before a proof verified hours earlier.
# Placement cannot be fixed -- shared vocabulary is referenced across subtrees,
# so no ownership-derived position keeps every consumer in scope -- which leaves
# content as the only lever.
#
# A declaration that merely adds a name is inert: nothing else elaborates
# differently because it exists. These forms are not. They change how
# *subsequent* text parses, resolves, or reduces, so they can break an
# already-verified proof body without touching any statement.
#
# Banned outright: no goal requires them to be stated or proved, and the
# 'open X in' / 'set_option ... in' scoped variants remain available for the
# cases that motivate them.
_BANNED_HEADER_FORMS: Tuple[Tuple[str, "re.Pattern[str]"], Ellipsis] = (
    (
        "notation",
        re.compile(r"^(?:local\s+|scoped\s+)*(?:notation3|notation|infixl|infixr|infix|prefix|postfix)\b"),
    ),
    (
        "syntax extension",
        re.compile(r"^(?:local\s+|scoped\s+)*(?:macro_rules|macro|syntax|elab_rules|elab|declare_syntax_cat)\b"),
    ),
    ("attribute", re.compile(r"^(?:local\s+|scoped\s+)*attribute\b")),
    ("open", re.compile(r"^(?:local\s+|scoped\s+)*open\b")),
    ("set_option", re.compile(r"^set_option\b")),
    ("variable", re.compile(r"^(?:local\s+|scoped\s+)*variables?\b")),
    ("reducibility", re.compile(r"^(?:unseal|seal)\b")),
)

# Bodyless declarations, kept apart from the banned forms above because the
# reason and the remedy are different: these perturb nothing's elaboration, they
# are 'sorry' under another spelling. 'opaque C : T' introduces a constant with
# no accessible body, so every goal stated in terms of C is unprovable -- and it
# compiles clean and adds no axiom, so neither the placeholder gate nor the
# '#print axioms' check can see it. In fmf_v1_002 a worker hit the placeholder
# gate on 'def HeuleGraph := sorry', rewrote it as 'opaque HeuleGraph', and the
# block was committed to the shared header; the child then correctly proved its
# own goal unprovable and the run ended with the root declared defective.
_PLACEHOLDER_HEADER_FORMS: Tuple[Tuple[str, "re.Pattern[str]"], Ellipsis] = (
    ("opaque constant", re.compile(rf"{_HEADER_FORM_PREFIX}opaque\b")),
)

# Allowed, because a goal can genuinely require them to be expressible, but they
# perturb elaboration all the same -- so a block containing one must be checked
# against the whole tree rather than the cheap header smoke test.
_ESCALATING_HEADER_FORMS: Tuple[Tuple[str, "re.Pattern[str]"], Ellipsis] = (
    ("attribute annotation", re.compile(r"^@\[")),
    ("instance", re.compile(rf"{_HEADER_FORM_PREFIX}instance\b")),
    (
        "type declaration",
        re.compile(rf"{_HEADER_FORM_PREFIX}(?:class|structure|inductive)\b"),
    ),
)

# 'open X in' and 'set_option o v in' scope to the next declaration only, so
# they cannot leak into anything else and stay legal.
_SCOPED_SUFFIX = re.compile(r"\bin\s*$")

# ... but only when something in this block follows it. A scoped form that ends
# the block has nothing left to scope to, so it binds to whatever the header
# emits next -- the following dynamic block, or the theorem -- which is a
# different declaration on every assembly. `graph_init_guard` makes the same
# check on the preamble, for the same reason.
_DANGLING_IN = re.compile(r"\bin\s*\Z")

# What the shared header is *for*: the vocabulary a sibling cannot state its own
# goal without. Everything else is a proof step, and a proof step belongs to the
# node that proved it, where it stays retractable.
#
# The forms above are policed for how they perturb elaboration. This one is
# policed for lifetime. `dynamic_headers` is append-only (D1), so a `lemma`
# admitted here outlives the branch that wrote it: it is re-elaborated on every
# local scaffold build and every regression check for the rest of the run, it
# permanently reserves its name, and it keeps briefing every future agent with a
# hint from a branch that may since have been abandoned. fmf_v1_009/rollout_2
# put 205 lines of one node's measure theory in the header -- 11 lemmas against
# 1 `def` -- and then failed that node. None of it could be taken back.
#
# `instance` is admitted even though an instance carries a proof, because a
# sibling genuinely cannot state a goal over a measure without the
# `IsProbabilityMeasure` instance in scope. Deliberate soft edge.
#
# `variable` and `opaque` are absent on purpose: both are in `_DECL_KEYWORDS`
# but both are already refused above, `variable` as a banned form and `opaque`
# as a placeholder. Listing them here would name them in a rejection message as
# an available remedy when they are not.
HEADER_ADMITS = frozenset(
    {"def", "abbrev", "structure", "inductive", "class", "instance"}
)


class DefinitionsBlockReport(pydantic.BaseModel):
  """How a '-- [START_DEFINITIONS]' block interacts with the rest of the tree."""

  banned: List[str] = pydantic.Field(
      default_factory=list,
      description="Rendered 'line N: form -- text' entries that must be removed.",
  )
  escalating: List[str] = pydantic.Field(
      default_factory=list,
      description="Legal forms that force a full-tree recompile before commit.",
  )
  placeholders: List[str] = pydantic.Field(
      default_factory=list,
      description="Bodyless declarations -- a 'sorry' the placeholder gate cannot see.",
  )
  unnamed: List[str] = pydantic.Field(
      default_factory=list,
      description="Text that is not a named declaration and not otherwise classified.",
  )
  proofs: List[str] = pydantic.Field(
      default_factory=list,
      description=(
          "Rendered '<keyword> <name>' entries: proved statements in a block"
          " that may only hold vocabulary. See HEADER_ADMITS."
      ),
  )
  declared_names: List[str] = pydantic.Field(
      default_factory=list,
      description="Top-level identifiers this block introduces.",
  )

  @property
  def is_rejected(self):
    return bool(self.placeholders or self.banned or self.unnamed or self.proofs)

  @property
  def needs_full_tree_check(self):
    return bool(self.escalating)


def classify_definitions_block(block):
  """Classifies a definitions block by how far its effects can reach.

  Comments are masked rather than stripped so reported line numbers match what
  the agent submitted. Only column-0 matches count: an 'instance' or 'open' that
  is indented is inside a term or tactic block, where it is already scoped.
  """
  report = DefinitionsBlockReport()
  if not block or not block.strip():
    return report

  masked = mask_lean_comments(block)
  flagged_lines = set()
  for idx, line in enumerate(masked.splitlines(), start=1):
    if not line.strip():
      continue
    placeholder = next(
        (form for form, pattern in _PLACEHOLDER_HEADER_FORMS if pattern.match(line)),
        None,
    )
    if placeholder:
      report.placeholders.append(f"line {idx}: {placeholder} -- {line.strip()}")
      flagged_lines.add(idx)
      continue
    for form, pattern in _BANNED_HEADER_FORMS:
      if pattern.match(line):
        # 'open Finset in' / 'set_option x y in' bind to the next declaration.
        if form in ("open", "set_option") and _SCOPED_SUFFIX.search(line):
          break
        report.banned.append(f"line {idx}: {form} -- {line.strip()}")
        flagged_lines.add(idx)
        break
    else:
      for form, pattern in _ESCALATING_HEADER_FORMS:
        if pattern.match(line):
          report.escalating.append(f"line {idx}: {form} -- {line.strip()}")
          flagged_lines.add(idx)
          break

  # The per-line loop above waves through every scoped 'in' form. One of them
  # may be the last line in the block, and that one scopes to nothing here.
  code_only = strip_lean_comments(block).strip()
  if _DANGLING_IN.search(code_only):
    last_code_line = max(
        (i for i, l in enumerate(masked.splitlines(), start=1) if l.strip()),
        default=len(masked.splitlines()),
    )
    report.banned.append(
        f"line {last_code_line}: dangling 'in' --"
        f" {code_only.splitlines()[-1].strip()}"
    )
    flagged_lines.add(last_code_line)

  # Anything that is neither a named declaration nor a form recognized above is
  # an unknown command ('#eval', 'run_cmd', 'example', 'deriving instance'). It
  # cannot be reasoned about, so it is refused rather than assumed harmless.
  for decl in split_declarations(block):
    name = extract_declaration_name(decl)
    if name:
      report.declared_names.append(name)
      keyword = extract_declaration_keyword(decl)
      if keyword and keyword not in HEADER_ADMITS:
        report.proofs.append(f"{keyword} {name}")
      continue
    residue = strip_lean_comments(decl).strip()
    if not residue:
      continue
    first = residue.splitlines()[0].strip()
    recognized = any(
        pattern.match(first)
        for _, pattern in _PLACEHOLDER_HEADER_FORMS
        + _BANNED_HEADER_FORMS
        + _ESCALATING_HEADER_FORMS
    )
    if not recognized:
      report.unnamed.append(first)

  return report


def parse_formal_statement(statement):
  """Parses a raw Lean statement string into a name and proposition."""
  clean_statement = strip_lean_comments(statement)
  clean_statement = strip_lean_attributes(clean_statement)
  clean_statement = clean_statement.strip()

  depth = 0
  colon_idx = -1

  for i, char in enumerate(clean_statement):
    if char in "([{⦃⟨":
      depth += 1
    elif char in ")]}⦄⟩":
      depth -= 1
    elif char == ":" and depth == 0:
      colon_idx = i
      break

  if colon_idx == -1:
    raise ValueError(
        "Could not find separator colon at depth 0 in statement:"
        f" {clean_statement}"
    )

  header = clean_statement[:colon_idx].strip()

  binders_start = -1
  for i, char in enumerate(header):
    if char in "([{⦃⟨":
      binders_start = i
      break

  if binders_start != -1:
    name_part = header[:binders_start].strip()
  else:
    name_part = header

  parts = name_part.split()
  keywords = ["theorem", "lemma", "def", "abbrev", "inductive"]
  keyword_idx = -1
  for i, part in enumerate(parts):
    if part in keywords:
      keyword_idx = i
      break

  if keyword_idx != -1 and keyword_idx + 1 < len(parts):
    name = parts[keyword_idx + 1]
  else:
    name = parts[-1] if parts else "unknown"

  clean_statement_no_proof, _ = split_signature_and_proof(clean_statement)
  if binders_start != -1:
    statement_part = clean_statement_no_proof[binders_start:].strip()
  else:
    statement_part = clean_statement_no_proof[colon_idx:].strip()

  if statement_part.startswith(":"):
    statement_part = statement_part[1:].strip()

  # `_root_.foo` and `foo` are the same declaration. Agents reach for `_root_.`
  # to escape a namespace the problem statement left open, and the kernel then
  # prints `foo`, so leaving the prefix on the signature means every downstream
  # comparison against a printed name misses.
  if name.startswith("_root_."):
    name = name[len("_root_.") :]

  return name, statement_part


def parse_subgoal_declarations(subgoals_text):
  """Parses intermediate subgoal declarations strictly starting with lemma/theorem.

  Drops non-subgoal declarations (e.g. def, structures) or unparsable blocks cleanly.
  """
  clean_text = strip_lean_comments(subgoals_text).strip()
  if not clean_text:
    return []

  subgoals = []
  decl_pattern = re.compile(
      rf"^[ \t]*(?:@\[[^\]]*\]\s+)?{_DECL_MODIFIERS}(theorem|lemma)\s+",
      re.MULTILINE,
  )
  clean_masked = mask_lean_comments(clean_text)
  matches = list(decl_pattern.finditer(clean_masked))
  for i, match in enumerate(matches):
    start = match.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
    block = clean_text[start:end].strip()
    block = clean_declaration_block(block)
    if block:
      name, stmt = parse_formal_statement(block)
      subgoals.append(FormalSignature(name=name, statement=stmt))
  return subgoals


def declaration_to_axiom(sig_lean):
  """Rewrites a `theorem`/`lemma` declaration keyword to `axiom`.

  A keyword rewrite, deliberately: Lean closes the binders itself. The
  alternative -- constructing a proposition -- means reproducing binder scoping
  in Python, which fails silently into a different claim.
  """
  sig_lean = sig_lean.strip()
  if sig_lean.startswith("theorem "):
    return "axiom " + sig_lean[8:]
  if sig_lean.startswith("lemma "):
    return "axiom " + sig_lean[6:]
  return re.sub(r"\b(theorem|lemma)\b", "axiom", sig_lean, count=1)


def convert_subgoals_to_axioms(subgoals_text):
  """Converts string block of logical subgoals (lemma x) safely to axioms (axiom x)."""
  return "\n".join(
      declaration_to_axiom(sg.to_lean())
      for sg in parse_subgoal_declarations(subgoals_text)
  )


def disproof_obligation(node_id):
  """Returns the one declaration a disproof of `node_id` may submit.

  The obligation is literally `False` for every node, because the target is
  assumed as an axiom rather than negated. Signature immutability then does the
  whole of Gate A for free: the agent fills tactics and cannot restate what it
  claims to have shown.
  """
  return f"theorem {node_id}_disproof : False"


def build_disproof_header(header, target_sig):
  """Returns the shared header with the target appended as an axiom.

  Called by the renderer that writes the scaffold and by the gate that verifies
  it. One helper rather than two agreeing call sites: if the two headers differ
  by so much as a line, the agent gets "compiled locally, rejected on submit",
  which is the worst debugging experience the system can produce.
  """
  return "\n".join([
      header.rstrip(),
      "",
      "-- Target under refutation. It is assumed here so that deriving `False`",
      "-- below refutes it. Nothing proved from it may leave this file.",
      declaration_to_axiom(target_sig),
  ])

SCAFFOLD_PROVE = "prove"
SCAFFOLD_DISPROVE = "disprove"

#: Two roles, two files. `decompose` is deliberately absent: it is the proof
#: scaffold, because an agent decomposes by filling in that file's subgoal
#: block, and a third filename would make the choice visible before the agent
#: has made it.
_SCAFFOLD_FILENAMES = {
    SCAFFOLD_PROVE: "proof_scaffold.lean",
    SCAFFOLD_DISPROVE: "disproof_scaffold.lean",
}


def scaffold_filename(role = SCAFFOLD_PROVE):
  """Maps a scaffold role to its filename.

  One definition, read by the renderer that writes the file and the gate that
  checks the submitted path. Two copies of this mapping would let the harness
  hand the agent one filename and reject it for using another.

  Raises:
    KeyError: on an unknown role. A default would silently hand a disproof the
      proof scaffold, which compiles a refutation against the wrong obligation.
  """
  return _SCAFFOLD_FILENAMES[role]

_SUBGOALS_HINT = (
    "-- Optional. Leave this block empty to prove the goal directly.\n"
    "-- To decompose instead, declare each subgoal here as a lemma ending in\n"
    "-- `:= sorry`. They become axioms you may cite in the proof below, and\n"
    "-- each one is dispatched as its own node."
)

_DEFINITIONS_HINT = (
    "-- Add any new helper definitions or lemmas here (no notation,"
    " attributes, 'open', or 'variable')."
)


def build_lean_scaffold(
    immutable_context,
    definitions,
    subgoals,
    proof,
    role = SCAFFOLD_PROVE,
    hints = True,
):
  """Renders the scaffold template natively.

  One shape for all proof work. The subgoals block is emitted whether or not
  the agent intends to use it: gating it on the role meant an agent claimed to
  prove had physically nowhere to write a decomposition, which forced a
  release-and-reclaim round trip every time a proof turned out to need one.

  Emptiness of that block is now the signal -- empty means proof, non-empty
  means decomposition -- so the block must always be offered for the signal to
  be expressible. Hint comments are safe here: subgoals are recognised by
  parsing declarations, and a comment parses to nothing.

  Args:
    immutable_context: The shared header. Never editable by the agent.
    definitions: Helper declarations hoisted into the tree-wide header.
    subgoals: Subgoal lemma declarations, or empty for a direct proof.
    proof: The declaration under proof, with its body.
    role: SCAFFOLD_PROVE or SCAFFOLD_DISPROVE. Disproof omits both mutable
      header regions -- see below.
    hints: Whether to fill empty blocks with instructional comments. True when
      rendering a file for an agent to edit; False when assembling code for
      the kernel, which needs no advice and whose diagnostics count literal
      'sorry' occurrences -- a hint mentioning one would report a placeholder
      in every clean proof.

  Returns:
    The rendered scaffold.
  """
  parts = [
      "-- [START_IMMUTABLE_CONTEXT]",
      immutable_context.strip(),
      "-- [END_IMMUTABLE_CONTEXT]\n",
  ]

  # A disproof assumes its own target as an axiom, so anything it exported
  # would be derived from a proposition the tree is trying to refute. Both
  # mutable regions are withheld rather than validated: a region that is never
  # rendered cannot be filled in, so there is no submission to reject and no
  # way for a false-derived declaration to reach the shared header.
  if role != SCAFFOLD_DISPROVE:
    parts.extend([
        "-- [START_DEFINITIONS]",
        # Deliberately does not offer 'notation': every block here is hoisted
        # above every proof in the tree, and notation reparses everything after
        # it. See _BANNED_HEADER_FORMS.
        definitions.strip() if definitions else (
            _DEFINITIONS_HINT if hints else ""
        ),
        "-- [END_DEFINITIONS]\n",
        "-- [START_SUBGOALS]",
        subgoals.strip() if subgoals else (_SUBGOALS_HINT if hints else ""),
        "-- [END_SUBGOALS]\n",
    ])

  parts.extend([
      "-- [START_PROOF]",
      proof.strip(),
      "-- [END_PROOF]",
  ])
  return "\n".join(parts)

