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

r"""Lean 4 Proof Verifier CLI Tool and Library for the LEMMA harness.

Provides a clean, lightweight CLI and Python interface for LEMMA agents
to verify Lean 4 proofs via kimina-lean-server or a local Lean install. Uses authoritative
Lean 4 '#print axioms' checking to guarantee mathematical soundness.

Two backends:

  local  -- the default: run `lake env lean --json` inside a local Lean project
            that depends on Mathlib: `LEAN_PROJECT_DIR`, else the project
            `leap lean` creates (`$LEAP_HOME/leap_lean`, `~/.leap/leap_lean`).
            No server needed.
  kimina -- POST to a kimina-lean-server (https://github.com/project-numina/
            kimina-lean-server) at `LEAN_VERIFIER_URL` / `--verifier_url`.
            Optional `LEAN_VERIFIER_API_KEY` is sent as `Authorization: Bearer`.

Local Lean is used unless the user set a verifier URL and that URL works: with
a URL set, it is tried first, and when it gives no verdict (unreachable, an HTTP
error, an empty reply) the same check runs on local Lean, and the output says
so (`backend`, `remote_error`). `LEAN_VERIFIER_BACKEND=local` or `--local`
(`local=True` in Python) never calls a URL. With no URL and no local project,
it raises `NotImplementedError` rather than silently reporting proofs as
unverifiable.

Usage (Python API):
  import lean_verifier_tool

  # Target theorem verification:
  is_valid = lean_verifier_tool.verify_proof(proof_code, target_name="my_thm")
  is_valid, error_msg = lean_verifier_tool.verify_proof(proof_code, target_name="my_thm", return_error_message=True)

Usage (CLI):
  python3 lean_verifier_tool.py --code_file=proof_scaffold.lean --target_name=my_thm
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, overload
import urllib.error
import urllib.request
import pydantic


try:
  # pylint: disable=g-import-not-at-top
  from absl import app
  from absl import flags

  def _def_flag(name, define_fn, *args, **kwargs):
    assert flags is not None
    if name not in flags.FLAGS:
      return define_fn(name, *args, **kwargs)
    return flags.FLAGS[name]

  _CODE = _def_flag("code", flags.DEFINE_string, "", "The Lean 4 code string to verify.")
  _CODE_FILE = _def_flag(
      "code_file", flags.DEFINE_string, "", "Optional path to a local Lean 4 code file to verify."
  )
  _TARGET_NAME = _def_flag(
      "target_name", flags.DEFINE_string, None, "Optional target theorem/lemma name."
  )
  _VERIFIER_URL = _def_flag(
      "verifier_url", flags.DEFINE_string, None, "Optional custom Lean verifier service base URL."
  )
  _TIMEOUT = _def_flag(
      "timeout", flags.DEFINE_integer, 700, "Verification timeout in seconds."
  )
  _LOCAL = _def_flag(
      "local", flags.DEFINE_boolean, False,
      "Use only local Lean (LEAN_PROJECT_DIR or ~/.leap/leap_lean); never call"
      " a verifier URL."
  )
  _JSON_OUTPUT = _def_flag(
      "json_output", flags.DEFINE_boolean, False, "Whether to print output in JSON format."
  )
  _HAS_ABSL = True
except (ImportError, ModuleNotFoundError):
  app = None
  flags = None
  _HAS_ABSL = False


#: Base URL of a kimina-lean-server (or any service speaking
#: `VERIFIER_WIRE_CONTRACT`). Empty by default: local Lean is the default, and
#: a URL is used only when set and answering.
DEFAULT_VERIFIER_URL = os.environ.get("LEAN_VERIFIER_URL", "")

#: Optional API key for the kimina server (its `LEAN_SERVER_API_KEY`). Sent as
#: `Authorization: Bearer <key>`. Unset means no Authorization header.
VERIFIER_API_KEY = os.environ.get("LEAN_VERIFIER_API_KEY", "")

#: Root of a local Lean 4 project with Mathlib as a dependency (the directory
#: holding `lakefile.lean`/`lakefile.toml`), for the `local` backend. Unset
#: means the project `leap lean` creates, if it exists (`_local_project`).
LEAN_PROJECT_DIR = os.environ.get("LEAN_PROJECT_DIR", "")

#: "local" never calls a URL; "kimina" or "" tries a set URL first, then local
#: Lean when the URL gives no verdict.
VERIFIER_BACKEND = os.environ.get("LEAN_VERIFIER_BACKEND", "").strip().lower()

#: The HTTP contract `run_lean_verifier` speaks, spelled out so an external
#: deployment can be stood up against it without reading the request code. It
#: is a string rather than a comment because the `NotImplementedError` raised
#: on an unconfigured install quotes it verbatim.
VERIFIER_WIRE_CONTRACT = """\
REQUEST -- POST to `{base}/api/check`, falling back to `{base}/check` on 404.
  Content-Type: application/json
  {
    "snippets": [{"id": "check_0", "code": "<full Lean 4 source>"}],
    "timeout":  <int, server-side Lean compile budget in seconds>,
    "debug":    false,
    "reuse":    true     // may reuse a warm Lean environment
  }
  Deliberately no "infotree" key: the full pretty-printed tactic tree is
  never read here and cost 576s / 24.6MB on a 4077-line assembly.

RESPONSE -- 200 with JSON:
  {
    "results": [
      {
        "error": "<string>",        // optional; set => nothing was decided.
                                    // Text matching /time-?out/i is treated
                                    // as an infra failure, not a rejection.
        "response": {
          "messages": [
            {
              "severity": "error" | "warning" | "information" | "info",
              "data":     "<message text>",
              "pos":      {"line": <1-based int>, "column": <int>}
            }
          ]
        }
      }
    ]
  }

SEMANTICS the caller depends on:
  * Any message with severity "error" means Lean rejected the code.
  * When `target_name` is given, the tool appends `#print axioms <name>` to
    the snippet and reads the axiom list out of the reply's message data,
    matching `'<ns.>?<name>' depends on axioms: [...]` or
    `'<ns.>?<name>' does not depend on any axioms`. A server that swallows
    `#print axioms` output breaks the soundness gate.
  * An empty `results` array, a transport failure, or a server-side timeout
    is reported as `is_infra_failure=True`, which callers treat as "no
    verdict" rather than "proof is wrong".
  * Retried on HTTP 408/425/429/500/502/503/504 and on any transport
    exception, up to `LEAN_VERIFIER_MAX_ATTEMPTS` times.
"""

# Slack between the server-side Lean compile budget and the client socket
# timeout, to cover response serialization and transfer.
_SOCKET_TIMEOUT_HEADROOM_SEC = 60

# Transport-level retry. A dropped connection or an overloaded server says
# nothing about the mathematics, but the caller previously could not tell that
# apart from a compiler rejection: in scratch/rollout_1, 66 read timeouts were
# reported to agents as regressions and eventually failed 10 AND-nodes as
# "unprovable dead ends". Retry the ones that can succeed on a second attempt,
# and flag the rest as infra failures so the caller can decline to penalize.
_MAX_TRANSPORT_ATTEMPTS = int(os.environ.get("LEAN_VERIFIER_MAX_ATTEMPTS", "3"))
_RETRY_BACKOFF_SEC = (5, 20)
_RETRYABLE_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})



class LeanVerifierInput(pydantic.BaseModel):
  """Input configuration for Lean 4 proof verification."""

  code: str = pydantic.Field(
      default="", description="The Lean 4 code string to verify."
  )
  target_name: Optional[str] = pydantic.Field(
      default=None, description="Optional target theorem/lemma name."
  )
  verifier_url: Optional[str] = pydantic.Field(
      default=None, description="Optional custom Lean verifier base URL."
  )
  timeout: int = pydantic.Field(
      default=700, description="Verification timeout in seconds."
  )
  local: bool = pydantic.Field(
      default=False,
      description=(
          "Use only local Lean (LEAN_PROJECT_DIR or ~/.leap/leap_lean); never"
          " call a verifier URL."
      ),
  )


class LeanVerifierOutput(pydantic.BaseModel):
  """Verification result for a Lean 4 proof attempt."""

  is_valid: bool = pydantic.Field(
      default=False, description="Whether the proof is valid."
  )
  error_message: Optional[str] = pydantic.Field(
      default=None, description="Error message if verification failed."
  )
  used_axioms: Optional[List[str]] = pydantic.Field(
      default=None, description="List of axioms the target theorem depends on."
  )
  info_messages: str = pydantic.Field(
      default="", description="Information and warning messages (e.g. exact? or #check outputs) from the compiler."
  )
  is_infra_failure: bool = pydantic.Field(
      default=False,
      description=(
          "True when no verdict was obtained at all (transport failure, empty"
          " result set, server-side timeout). Distinct from is_valid=False,"
          " which means Lean actually rejected the code."
      ),
  )
  backend: str = pydantic.Field(
      default="",
      description=(
          "Which backend gave the result: \"kimina\" (the verifier URL) or"
          " \"local\" (local Lean). Empty when none was reached."
      ),
  )
  remote_error: Optional[str] = pydantic.Field(
      default=None,
      description=(
          "Why the verifier URL gave no verdict, when one was set and the check"
          " fell back to local Lean."
      ),
  )


def _local_project() -> str:
  """The Lake project local Lean compiles in, or "" when there is none.

  `LEAN_PROJECT_DIR` when set; otherwise the project `leap lean` creates,
  `$LEAP_HOME/leap_lean` (`~/.leap/leap_lean`), if that directory exists. The
  path is spelled out here rather than read from `config`, so this module still
  runs on its own.
  """
  project = os.environ.get("LEAN_PROJECT_DIR", LEAN_PROJECT_DIR)
  if project:
    return os.path.expanduser(project)
  default = os.path.join(
      os.path.expanduser(os.environ.get("LEAP_HOME", "~/.leap")), "leap_lean"
  )
  return default if os.path.isdir(default) else ""


def _auth_headers() -> Dict[str, str]:
  """Authorization header for the kimina server, if an API key is set.

  kimina-lean-server strips a case-sensitive "Bearer " prefix before comparing
  the key, so the capitalisation here matters.
  """
  key = os.environ.get("LEAN_VERIFIER_API_KEY", VERIFIER_API_KEY)
  return {"Authorization": f"Bearer {key}"} if key else {}


def _resolve_backend(verifier_url: str) -> str:
  """The backend to try first: "kimina" when a URL is set, else "local".

  Local Lean is the default. A set URL is tried first unless
  `LEAN_VERIFIER_BACKEND=local`; "kimina" means the same as unset (a URL set
  is tried first, and local Lean takes over when it gives no verdict).
  """
  backend = os.environ.get("LEAN_VERIFIER_BACKEND", VERIFIER_BACKEND).strip().lower()
  if backend not in ("", "kimina", "local"):
    raise ValueError(
        f"LEAN_VERIFIER_BACKEND={backend!r} is not one of 'kimina', 'local'."
    )
  if backend != "local" and verifier_url:
    return "kimina"
  return "local"


def extract_info_messages(proof: str, messages: List[Dict[str, Any]]) -> str:
  """Extracts info and warning messages from Lean responses."""
  body_lines = proof.splitlines()
  info_lines = []
  for m in messages:
    sev = m.get("severity")
    if sev in ["information", "info", "warning"]:
      data = m.get("data", "").strip()
      pos = m.get("pos") or {}
      line_num = pos.get("line")
      if line_num is not None and 1 <= line_num <= len(body_lines):
        snippet = body_lines[line_num - 1].strip()
        info_lines.append(f"[{sev.upper()}] Line {line_num} ('{snippet}'): {data}")
      elif line_num is not None:
        info_lines.append(f"[{sev.upper()}] Line {line_num}: {data}")
      else:
        info_lines.append(f"[{sev.upper()}] {data}")
  return "\n".join(info_lines)

def extract_error_messages(proof: str, messages: List[Dict[str, Any]]) -> str:
  """Extracts and formats compiler error messages from Lean responses."""
  body_lines = proof.splitlines()
  error_lines = []
  for m in messages:
    if m.get("severity") == "error":
      data = m.get("data", "").strip()
      pos = m.get("pos") or {}
      line_num = pos.get("line")
      if line_num is not None and 1 <= line_num <= len(body_lines):
        snippet = body_lines[line_num - 1].strip()
        error_lines.append(f"Line {line_num} ('{snippet}'): {data}")
      elif line_num is not None:
        error_lines.append(f"Line {line_num}: {data}")
      else:
        error_lines.append(data)

  if not error_lines:
    return "Unknown Lean compilation error."
  return "\n".join(error_lines)


def _normalize_lean_imports(code: str) -> str:
  """Ensures all 'import' statements are hoisted to the absolute top of the Lean code.

  If no import statements are present, prepends 'import Mathlib'.
  Correctly tracks multi-line and nested block comments (/- ... -/) and single-line comments (--).
  Preserves non-import statements, comments, and structure after imports.
  """
  lines = code.splitlines()
  import_lines = []
  non_import_lines = []

  comment_depth = 0

  for line in lines:
    stripped = line.strip()
    if not stripped:
      non_import_lines.append(line)
      continue

    # Single-line comment when outside block comment: preserve directly
    if comment_depth == 0 and stripped.startswith("--"):
      non_import_lines.append(line)
      continue

    # If outside block comment and is a genuine import statement
    if comment_depth == 0 and re.match(r"^import\s+[A-Za-z0-9_.]", stripped):
      import_lines.append(stripped)
      # Check if import line opened a block comment that was not closed
      code_part = stripped
      if "--" in code_part:
        code_part = code_part[:code_part.find("--")]
      open_c = len(re.findall(r"/-", code_part))
      close_c = len(re.findall(r"-/", code_part))
      comment_depth = max(0, comment_depth + open_c - close_c)
      continue

    # Update comment depth for lines inside comments or other non-import lines
    code_part = stripped
    if comment_depth == 0 and "--" in code_part:
      code_part = code_part[:code_part.find("--")]
    open_c = len(re.findall(r"/-", code_part))
    close_c = len(re.findall(r"-/", code_part))
    comment_depth = max(0, comment_depth + open_c - close_c)

    non_import_lines.append(line)

  deduped_imports = list(dict.fromkeys(import_lines))
  if not deduped_imports:
    deduped_imports = ["import Mathlib"]

  header = "\n".join(deduped_imports)
  body = "\n".join(non_import_lines).strip()
  if body:
    return f"{header}\n\n{body}"
  return header


def _check_kimina(
    code: str, timeout: int, verifier_url: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
  """Sends one snippet to a kimina-lean-server.

  Returns `(result, None)` with the first entry of the reply's `results`, or
  `(None, reason)` when no verdict was obtained.
  """
  headers = {"Content-Type": "application/json", **_auth_headers()}

  # Deliberately no 'infotree': the server renders the full tactic tree with
  # every goal state pretty-printed, which nothing here reads -- only
  # 'response.messages' is consumed. On a 4077-line assembly it cost 576s /
  # 24.6MB versus 95s / 0.04MB without it, which is what drove the
  # 'read operation timed out' failures on large proof trees.
  payload = json.dumps({
      "snippets": [{"id": "check_0", "code": code}],
      "timeout": timeout,
      "debug": False,
      "reuse": True,
  }).encode("utf-8")

  endpoints = [f"{verifier_url}/api/check", f"{verifier_url}/check"]
  resp_data = None
  last_err = None

  # The payload 'timeout' bounds the Lean compile only; the server still has to
  # serialize and ship the result afterwards. Giving the socket the same budget
  # meant any job using most of its compile time could never return.
  socket_timeout = timeout + _SOCKET_TIMEOUT_HEADROOM_SEC

  for attempt in range(max(1, _MAX_TRANSPORT_ATTEMPTS)):
    retryable = False
    for endpoint in endpoints:
      req = urllib.request.Request(
          endpoint, data=payload, headers=headers, method="POST"
      )
      try:
        with urllib.request.urlopen(req, timeout=socket_timeout) as resp:
          resp_data = json.loads(resp.read().decode("utf-8"))
          break
      except urllib.error.HTTPError as e:
        last_err = f"HTTP {e.code}: {e.reason}"
        # 404 means this URL shape is wrong, not that the service is down; fall
        # through to the next endpoint rather than burning a retry.
        if e.code == 404:
          continue
        retryable = e.code in _RETRYABLE_HTTP_CODES
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Socket timeout, connection reset, DNS, truncated body. All transient.
        last_err = str(e)
        retryable = True
        break
    if resp_data is not None:
      break
    if not retryable or attempt >= max(1, _MAX_TRANSPORT_ATTEMPTS) - 1:
      break
    time.sleep(_RETRY_BACKOFF_SEC[min(attempt, len(_RETRY_BACKOFF_SEC) - 1)])

  if resp_data is None:
    return None, (
        f"request failed after {max(1, _MAX_TRANSPORT_ATTEMPTS)} attempt(s):"
        f" {last_err}"
    )
  results = resp_data.get("results", [])
  if not results:
    return None, "no verification results returned from verifier service"
  return results[0], None


def _check_local(
    code: str, timeout: int
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
  """Compiles one snippet with `lake env lean --json` in the local project.

  The project is `_local_project()`. Produces the same
  `{"response": {"messages": [...]}}` / `{"error": ...}` shape a kimina server
  returns, so everything downstream is shared. `lean --json` prints one message
  object per line with `severity`, `pos` and `data`.
  """
  project = _local_project()
  if not project or not os.path.isdir(project):
    return None, (
        f"no local Lean project ({project or 'LEAN_PROJECT_DIR unset'} is not"
        " a directory); run `leap lean`"
    )
  lake = shutil.which("lake")
  if lake is None:
    return None, "`lake` is not on PATH; install Lean via elan"

  fd, path = tempfile.mkstemp(prefix="leap_check_", suffix=".lean")
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      f.write(code)
    try:
      proc = subprocess.run(
          [lake, "env", "lean", "--json", path],
          cwd=project,
          capture_output=True,
          text=True,
          # Lean prints UTF-8; the locale default (cp1252 on Windows) would
          # mangle or fail on symbols such as ℝ.
          encoding="utf-8",
          errors="replace",
          timeout=timeout,
          check=False,
      )
    except subprocess.TimeoutExpired:
      return {"error": f"Lean compile timeout after {timeout}s"}, None
    except OSError as e:
      return None, str(e)
  finally:
    try:
      os.unlink(path)
    except OSError:
      pass

  messages = []
  for line in proc.stdout.splitlines():
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      msg = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(msg, dict) and "severity" in msg:
      messages.append(msg)

  if proc.returncode != 0 and not any(
      m.get("severity") == "error" for m in messages
  ):
    # Lean failed without reporting a source error: missing toolchain, a
    # broken project, an out-of-memory kill. Nothing was decided.
    detail = (proc.stderr or proc.stdout).strip()[-2000:]
    return None, f"lean exited {proc.returncode}: {detail}"
  return {"response": {"messages": messages}}, None


def run_lean_verifier(input_data: LeanVerifierInput) -> LeanVerifierOutput:
  """Verifies a Lean 4 proof: local Lean, or a set verifier URL while it works.

  Args:
    input_data: Code, optional target theorem, endpoint override and timeout.

  Returns:
    The verdict. `is_infra_failure=True` means no verdict was reached and says
    nothing about the mathematics. `backend` names who answered, and
    `remote_error` says why a set URL was passed over for local Lean.

  Raises:
    NotImplementedError: No URL is set and there is no local Lean project --
      reporting the absence as `is_valid=False` would look like Lean rejecting
      every proof.
  """
  code = input_data.code.strip()
  if not code:
    return LeanVerifierOutput(
        is_valid=False,
        error_message="Empty code provided.",
    )

  code = _normalize_lean_imports(code)


  # Inject #print axioms natively if we have a target
  if input_data.target_name:
    code += f"\n#print axioms {input_data.target_name}\n"


  verifier_url = (input_data.verifier_url
                  or os.environ.get("LEAN_VERIFIER_URL", DEFAULT_VERIFIER_URL)
                  ).rstrip("/")
  backend = "local" if input_data.local else _resolve_backend(verifier_url)
  res, last_err, remote_err = None, None, None
  if backend == "kimina":
    # A set URL is used only while it works: no verdict from it (unreachable,
    # an HTTP error, an empty reply) sends the same check to local Lean.
    res, remote_err = _check_kimina(code, input_data.timeout, verifier_url)
    if res is None:
      backend = "local"
  if backend == "local":
    if not _local_project():
      if remote_err is None:
        raise NotImplementedError(
            "No Lean verifier is configured. Local Lean is the default: run"
            " `leap lean` once (it creates ~/.leap/leap_lean), or set"
            " LEAN_PROJECT_DIR to a Lake project that depends on Mathlib. To use"
            " a kimina-lean-server instead, set LEAN_VERIFIER_URL to its base"
            " URL (plus LEAN_VERIFIER_API_KEY if it has one); it is used while"
            " it answers.\n\n"
            "Any other service must speak this contract:\n\n"
            + VERIFIER_WIRE_CONTRACT
        )
      return LeanVerifierOutput(
          is_valid=False,
          is_infra_failure=True,
          error_message=(
              f"Lean verifier at {verifier_url} failed: {remote_err}; no local"
              " Lean project to fall back to (run `leap lean`, or set"
              " LEAN_PROJECT_DIR)."
          ),
          remote_error=remote_err,
      )
    res, last_err = _check_local(code, input_data.timeout)

  if res is None:
    # No verdict was obtained, so this says nothing about the proof.
    error_message = f"Lean verifier ({backend}) failed: {last_err}"
    if remote_err is not None:
      error_message = (
          f"Lean verifier at {verifier_url} failed: {remote_err}; local Lean"
          f" failed too: {last_err}"
      )
    return LeanVerifierOutput(
        is_valid=False,
        is_infra_failure=True,
        error_message=error_message,
        backend=backend,
        remote_error=remote_err,
    )

  if res.get("error"):
    # A server-side timeout is an infra failure wearing a result's clothing: the
    # snippet never finished elaborating, so nothing was decided about it.
    err_text = str(res["error"])
    return LeanVerifierOutput(
        is_valid=False,
        is_infra_failure=bool(
            re.search(r"time(?:d)?\s*-?\s*out|timeout", err_text, re.IGNORECASE)
        ),
        error_message=f"Lean Verifier Error: {err_text}",
        backend=backend,
        remote_error=remote_err,
    )

  resp_obj = res.get("response", {})
  messages = resp_obj.get("messages", []) if isinstance(resp_obj, dict) else []
  severities = [m.get("severity") for m in messages]


  has_compiler_error = "error" in severities


  info_string = extract_info_messages(code, messages)
  # None means "could not determine"; [] means "confirmed to depend on nothing".
  # Collapsing the two lets an unparseable result silently pass the axiom gate.
  used_axioms = None
  if input_data.target_name:
    escaped_target = re.escape(input_data.target_name)
    # Lean answers `#print axioms foo` under the name it resolved, so an open
    # `namespace Ns` in the problem statement makes the reply say `Ns.foo`.
    # Anchored on the bare name the match misses, `used_axioms` stays None,
    # and the caller reports a verifier-side failure for a fine submission.
    target_pat = f"'(?:[^'\\s]+\\.)?{escaped_target}'"
    # Parse the raw message payloads rather than the flattened info_string:
    # Lean's pretty-printer wraps long axiom lists across multiple lines, which
    # a single-line regex over the rendered string cannot match.
    for message in messages:
      data = str(message.get("data", ""))
      ax_match = re.search(
          target_pat + " depends on axioms: \\[(.*?)\\]", data, re.DOTALL
      )
      if ax_match:
        used_axioms = [
            ax.strip() for ax in ax_match.group(1).split(",") if ax.strip()
        ]
        break
      if re.search(target_pat + " does not depend on any axioms", data):
        used_axioms = []
        break
  else:
    warning_notice = "[TOOL WARNING] Verification ran without --target_name argument. The tool only verified if the code compiles natively, it did NOT cryptographically check for sorryAx or other cheat axioms. To verify axiom safety, provide --target_name."
    info_string = warning_notice + ("\n" + info_string if info_string else "")
  if remote_err is not None:
    fallback_note = (
        f"[VERIFIER NOTE] The verifier at {verifier_url} gave no verdict"
        f" ({remote_err}); this result is from local Lean."
    )
    info_string = fallback_note + ("\n" + info_string if info_string else "")

  if has_compiler_error:
    error_msg = extract_error_messages(code, messages)
    return LeanVerifierOutput(
        is_valid=False,
        error_message=error_msg,
        used_axioms=used_axioms,
        info_messages=info_string,
        backend=backend,
        remote_error=remote_err,
    )

  return LeanVerifierOutput(
      is_valid=True,
      error_message=None,
      used_axioms=used_axioms,
      info_messages=info_string,
      backend=backend,
      remote_error=remote_err,
  )



@overload
def verify_proof(
    proof: str,
    timeout: int = 700,
    target_name: Optional[str] = None,
    return_error_message: Literal[False] = False,
    verifier_url: Optional[str] = None,
    local: bool = False,
) -> bool:
  ...


@overload
def verify_proof(
    proof: str,
    timeout: int = 700,
    target_name: Optional[str] = None,
    return_error_message: Literal[True] = True,
    verifier_url: Optional[str] = None,
    local: bool = False,
) -> Tuple[bool, Optional[str]]:
  ...


def verify_proof(
    proof: str,
    timeout: int = 700,
    target_name: Optional[str] = None,
    return_error_message: bool = False,
    verifier_url: Optional[str] = None,
    local: bool = False,
) -> Any:
  """Judge/verify an agent-generated Lean 4 proof attempt.

  Args:
    proof: Complete Lean 4 formal code string.
    timeout: Verification timeout in seconds.
    target_name: Target theorem name to ensure proof correctness for the goal.
    return_error_message: If True, returns a tuple (is_valid, error_message).
    verifier_url: Optional custom Lean verifier service base URL.
    local: Use only local Lean; never call a verifier URL.

  Returns:
    - If return_error_message=True: Tuple[bool, Optional[str]]
    - Otherwise: bool (True if proof is judged valid, False otherwise).
  """
  input_data = LeanVerifierInput(
      code=proof,
      target_name=target_name,
      verifier_url=verifier_url,
      timeout=timeout,
      local=local,
  )
  output = run_lean_verifier(input_data)
  if return_error_message:
    return output.is_valid, output.error_message
  return output.is_valid


@overload
async def verify_proof_async(
    proof: str,
    timeout: int = 700,
    target_name: Optional[str] = None,
    return_error_message: Literal[False] = False,
    verifier_url: Optional[str] = None,
    local: bool = False,
) -> bool:
  ...


@overload
async def verify_proof_async(
    proof: str,
    timeout: int = 700,
    target_name: Optional[str] = None,
    return_error_message: Literal[True] = True,
    verifier_url: Optional[str] = None,
    local: bool = False,
) -> Tuple[bool, Optional[str]]:
  ...


async def verify_proof_async(
    proof: str,
    timeout: int = 700,
    target_name: Optional[str] = None,
    return_error_message: bool = False,
    verifier_url: Optional[str] = None,
    local: bool = False,
) -> Any:
  """Asynchronous proof judge interface for concurrent agent proving loops."""
  input_data = LeanVerifierInput(
      code=proof,
      target_name=target_name,
      verifier_url=verifier_url,
      timeout=timeout,
      local=local,
  )
  output: LeanVerifierOutput = await asyncio.to_thread(
      run_lean_verifier, input_data
  )
  if return_error_message:
    return output.is_valid, output.error_message
  return output.is_valid


def main(argv: List[str]) -> None:
  if _HAS_ABSL and flags is not None:
    code = _CODE.value
    code_file = _CODE_FILE.value
    target_name = _TARGET_NAME.value
    verifier_url = _VERIFIER_URL.value
    timeout = _TIMEOUT.value
    local = _LOCAL.value
    json_output = _JSON_OUTPUT.value
  else:
    import argparse  # pylint: disable=g-import-not-at-top

    parser = argparse.ArgumentParser(
        description="Lean 4 Proof Verifier CLI Tool"
    )
    parser.add_argument(
        "--code", type=str, default="", help="Lean 4 code string"
    )
    parser.add_argument(
        "--code_file", type=str, default="", help="Path to Lean 4 code file"
    )
    parser.add_argument(
        "--target_name", type=str, default=None, help="Target theorem name"
    )
    parser.add_argument(
        "--verifier_url",
        type=str,
        default=None,
        help="Custom verifier service URL",
    )
    parser.add_argument(
        "--timeout", type=int, default=700, help="Timeout in seconds"
    )
    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only local Lean; never call a verifier URL",
    )
    parser.add_argument(
        "--json_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print JSON output",
    )
    args, _ = parser.parse_known_args(argv[1:])
    code = args.code
    code_file = args.code_file
    target_name = args.target_name
    verifier_url = args.verifier_url
    timeout = args.timeout
    local = args.local
    json_output = args.json_output

  if code_file and os.path.exists(code_file):
    with open(code_file, "r", encoding="utf-8") as f:
      code = f.read()

  if not code.strip():
    err_msg = "Error: No Lean 4 code provided."
    if json_output:
      print('{"is_valid": false, "error_message": "Error: No Lean 4 code provided."}')
    else:
      print(err_msg)
    sys.exit(1)

  input_data = LeanVerifierInput(
      code=code,
      target_name=target_name,
      verifier_url=verifier_url,
      timeout=timeout,
      local=local,
  )
  try:
    output = run_lean_verifier(input_data)
  except (NotImplementedError, ValueError) as e:
    # Unconfigured: say how to configure, without a traceback.
    if json_output:
      print(json.dumps({"is_valid": False, "is_infra_failure": True,
                        "error_message": str(e)}))
    else:
      print(f"CONFIG ERROR: {e}", file=sys.stderr)
    sys.exit(2)

  if json_output:
    print(output.model_dump_json(indent=2))
  else:
    if output.is_valid:
      print("VALID")
      if output.info_messages:
        print(output.info_messages)
    else:
      print("INVALID: " + str(output.error_message))

  if not output.is_valid:
    sys.exit(1)


if __name__ == "__main__":
  if _HAS_ABSL and app is not None:
    app.run(main)
  else:
    main(sys.argv)
