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

#!/usr/bin/env python3
"""Installs the LEAP agents and skills into a workspace for one agent CLI.

The prompts in `agents/` and `skills/` are host-neutral: their frontmatter
names *capabilities* (shell, read, write, subagents, ...) rather than one
CLI's tool names. This script renders them into the layout each host reads,
and appends a short "Host notes" section saying how dispatch, timeouts and
tool names work there.

  leap init claude        # (wraps install_claude, into the current directory)
  leap init codex
  leap init antigravity

Layouts written under --dest:

  claude       .claude/agents/<name>.md, .claude/skills/<name>/,
               .claude/settings.json (Bash allow-list + 15 min Bash timeout)
  codex        .codex/agents/<name>.toml, .agents/skills/<name>/,
               orchestrator as the skill .agents/skills/leap-prove/
  antigravity  .agents/agents/<name>.md, .agents/skills/<name>/

Re-running overwrites the files it owns and nothing else. Only the standard
library is used.
"""

import argparse
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(HERE, "agents")
SKILLS_DIR = os.path.join(HERE, "skills")

ORCHESTRATOR = "orchestrator_lemma"
SMOKE_TEST = "leap_smoke_test"
LEAP_BINS = ("state_manager_bin", "lean_verifier_tool_bin", "lean_search_tool_bin")

# Capability group -> host tool names. An empty list means the host has no
# separate tool for it (e.g. Codex agents always have shell and file editing).
TOOL_MAP: Dict[str, Dict[str, List[str]]] = {
    "claude": {
        "shell": ["Bash"],
        "read": ["Read"],
        "write": ["Write"],
        "edit": ["Edit"],
        "list": ["Glob", "Grep"],
        "web": ["WebSearch", "WebFetch"],
        "subagents": ["Agent"],
        "tasks": [],  # todo tooling is built in and its name varies by version
    },
    "antigravity": {
        "shell": ["run_command"],
        "read": ["view_file"],
        "write": ["write_to_file"],
        "edit": ["replace_file_content"],
        "list": ["list_dir"],
        "web": ["search_web", "read_url_content"],
        "subagents": ["invoke_subagent", "manage_subagents", "send_message"],
        "tasks": ["manage_task"],
    },
}


# --------------------------------------------------------------------------
# Host notes. Appended to the rendered prompt; they are the only host-specific
# prose. {role_name} is how this host spells a `metadata.role` value.
# --------------------------------------------------------------------------

_ORCH_NOTES = {
    "claude": """\
## Host notes: Claude Code

- **Dispatching.** Use the Agent tool with `subagent_type` set to
  `metadata.role` with every `_` replaced by `-` (`lemma_proof_worker` ->
  `lemma-proof-worker`). Run workers in the background so up to 8 run at once;
  you are notified as each one finishes, so yield instead of polling.
- A background subagent that has returned is already finished; there is
  nothing to kill. Its pool slot is free.
- **Timeouts.** Pass `timeout: 900000` on every Bash call to
  `state_manager_bin`. The default Bash timeout is 2 minutes.
""",
    "codex": """\
## Host notes: Codex

- **Dispatching.** Spawn one subagent per claim using the custom agent whose
  name is `metadata.role`, with the dispatch prompt from Section 2.2. Keep up
  to 8 open at once, wait for results, and close each thread once it reports.
- **Timeouts.** Lean compiles can take several minutes. If your shell tool
  takes a timeout, give `state_manager_bin` at least 15 minutes.
""",
    "antigravity": """\
## Host notes: Antigravity

- **Dispatching.** `invoke_subagent` with the agent named `metadata.role`.
  Terminate a finished worker with
  `manage_subagents(Action='kill', ConversationIds=[<id>])`.
""",
}

_WORKER_NOTES = {
    "claude": """\
## Host notes: Claude Code

- Shell commands run through the Bash tool. Pass `timeout: 600000` for
  `lean_verifier_tool_bin` / `lean_search_tool_bin` and `timeout: 900000` for
  `state_manager_bin` -- the default is 2 minutes, which kills a long compile.
- You cannot dispatch subagents. Your final message is your report to the
  orchestrator.
""",
    "codex": """\
## Host notes: Codex

- Run the LEAP binaries through your shell tool; give `state_manager_bin` and
  the Lean tools a generous timeout (15 / 10 minutes) when the tool takes one.
- Do not spawn subagents. Your final message is your report to the
  orchestrator.
""",
    "antigravity": """\
## Host notes: Antigravity

- Shell is `run_command`; files are `view_file`, `write_to_file`,
  `replace_file_content`, `list_dir`. Report to the orchestrator with
  `send_message` where the protocol asks for a final report.
""",
}


def parse_markdown(path):
  """Splits a Markdown file into a (frontmatter dict, body) pair.

  Handles the YAML subset the prompts use: `key: scalar`, `key: [a, b]`,
  `key: "quoted"` and block lists of `- item`.
  """
  with open(path, encoding="utf-8") as f:
    text = f.read()
  if not text.startswith("---\n"):
    return {}, text
  end = text.index("\n---", 4)
  meta: Dict[str, object] = {}
  key = None
  for line in text[4:end].splitlines():
    if re.match(r"^\s+- ", line) and key:
      meta.setdefault(key, [])
      meta[key].append(line.split("- ", 1)[1].strip())  # type: ignore[union-attr]
      continue
    m = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
    if not m:
      continue
    key, val = m.group(1), m.group(2).strip()
    if val.startswith("[") and val.endswith("]"):
      meta[key] = [v.strip() for v in val[1:-1].split(",") if v.strip()]
    elif val == "":
      meta[key] = []
    elif val in ("true", "false"):
      meta[key] = val == "true"
    else:
      meta[key] = val[1:-1] if val[:1] == val[-1:] == '"' else val
  return meta, text[end + 4:].lstrip("\n")


def _yaml_str(s):
  return json.dumps(s, ensure_ascii=False)


def _hyphen(name):
  return name.replace("_", "-")


def _write(path, content):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write(content)
  print(f"  wrote {path}")


def _copy_skill(src_dir, dest_dir, name):
  """Copies a skill folder, rewriting its frontmatter `name`."""
  if os.path.isdir(dest_dir):
    shutil.rmtree(dest_dir)
  shutil.copytree(src_dir, dest_dir)
  skill_md = os.path.join(dest_dir, "SKILL.md")
  with open(skill_md, encoding="utf-8") as f:
    text = f.read()
  text = re.sub(r"^name: .*$", f"name: {name}", text, count=1, flags=re.M)
  if "_" not in name:
    # Hyphenated host: `depends_on` and prose must name skills the same way.
    text = re.sub(r"lemma_mathlib_(search|tactics)",
                  lambda m: _hyphen(m.group(0)), text)
  with open(skill_md, "w", encoding="utf-8") as f:
    f.write(text)
  print(f"  wrote {dest_dir}/")


def _agents():
  for fname in sorted(os.listdir(AGENTS_DIR)):
    if fname.endswith(".md"):
      meta, body = parse_markdown(os.path.join(AGENTS_DIR, fname))
      yield meta, body


def _notes(host, name):
  table = _ORCH_NOTES if name in (ORCHESTRATOR, SMOKE_TEST) else _WORKER_NOTES
  return "\n\n" + table[host]


def install_claude(dest):
  for meta, body in _agents():
    name = str(meta["name"])
    tools: List[str] = []
    for group in meta.get("tools", []):  # type: ignore[union-attr]
      for t in TOOL_MAP["claude"][group]:
        if t not in tools:
          tools.append(t)
    lines = [
        "---",
        f"name: {_hyphen(name)}",
        f"description: {_yaml_str(str(meta['description']))}",
        f"tools: {', '.join(tools)}",
    ]
    skills = [_hyphen(s) for s in meta.get("skills", [])]  # type: ignore[union-attr]
    if skills:
      lines.append("skills:")
      lines += [f"  - {s}" for s in skills]
    lines.append("---")
    _write(os.path.join(dest, ".claude", "agents", f"{_hyphen(name)}.md"),
           "\n".join(lines) + "\n\n" + body.rstrip() + _notes("claude", name))

  for skill in sorted(os.listdir(SKILLS_DIR)):
    _copy_skill(os.path.join(SKILLS_DIR, skill),
                os.path.join(dest, ".claude", "skills", _hyphen(skill)),
                _hyphen(skill))

  # Let the LEAP binaries run without a prompt per call, and raise the Bash
  # timeout ceiling so a 15-minute state_manager_bin call is allowed at all.
  settings_path = os.path.join(dest, ".claude", "settings.json")
  settings = {}
  if os.path.exists(settings_path):
    with open(settings_path, encoding="utf-8") as f:
      settings = json.load(f)
  allow = settings.setdefault("permissions", {}).setdefault("allow", [])
  for b in LEAP_BINS:
    rule = f"Bash({b}:*)"
    if rule not in allow:
      allow.append(rule)
  env = settings.setdefault("env", {})
  env.setdefault("BASH_DEFAULT_TIMEOUT_MS", "900000")
  env.setdefault("BASH_MAX_TIMEOUT_MS", "900000")
  _write(settings_path, json.dumps(settings, indent=2) + "\n")


def _toml_multiline(s):
  # TOML literal multi-line strings take everything verbatim except `'''`.
  if "'''" not in s:
    return "'''\n" + s + "'''"
  return json.dumps(s, ensure_ascii=False)


def install_codex(dest):
  for meta, body in _agents():
    name = str(meta["name"])
    prompt = body.rstrip() + _notes("codex", name)
    if name in (ORCHESTRATOR, SMOKE_TEST):
      # The orchestrator is the main session, so it ships as a skill the user
      # invokes ($leap-prove / $leap-smoke-test), not as a spawnable agent.
      skill = "leap-prove" if name == ORCHESTRATOR else "leap-smoke-test"
      fm = (f"---\nname: {skill}\ndescription: "
            f"{_yaml_str(str(meta['description']))}\n---\n\n")
      _write(os.path.join(dest, ".agents", "skills", skill, "SKILL.md"),
             fm + prompt)
      continue
    toml = (
        f"name = {json.dumps(name)}\n"
        f"description = {json.dumps(str(meta['description']), ensure_ascii=False)}\n"
        f"developer_instructions = {_toml_multiline(prompt)}\n"
    )
    _write(os.path.join(dest, ".codex", "agents", f"{name}.toml"), toml)

  for skill in sorted(os.listdir(SKILLS_DIR)):
    _copy_skill(os.path.join(SKILLS_DIR, skill),
                os.path.join(dest, ".agents", "skills", skill), skill)


def install_antigravity(dest):
  for meta, body in _agents():
    name = str(meta["name"])
    tools: List[str] = []
    for group in meta.get("tools", []):  # type: ignore[union-attr]
      for t in TOOL_MAP["antigravity"][group]:
        if t not in tools:
          tools.append(t)
    lines = [
        "---",
        f"name: {name}",
        f"description: {_yaml_str(str(meta['description']))}",
        f"mainAgent: {str(meta.get('mainAgent', False)).lower()}",
        f"subagent: {str(meta.get('subagent', True)).lower()}",
        "tools:",
        *[f"  - {t}" for t in tools],
    ]
    skills = meta.get("skills", [])
    if skills:
      lines.append("skills:")
      lines += [f"  - skills/{s}" for s in skills]  # type: ignore[union-attr]
    lines.append("---")
    _write(os.path.join(dest, ".agents", "agents", f"{name}.md"),
           "\n".join(lines) + "\n\n" + body.rstrip() + _notes("antigravity", name))

  for skill in sorted(os.listdir(SKILLS_DIR)):
    _copy_skill(os.path.join(SKILLS_DIR, skill),
                os.path.join(dest, ".agents", "skills", skill), skill)


INSTALLERS = {
    "claude": install_claude,
    "codex": install_codex,
    "antigravity": install_antigravity,
}

def main(argv):
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--host", required=True, choices=sorted(INSTALLERS))
  parser.add_argument("--dest", default=".",
                      help="Workspace to install into (default: current dir).")
  args = parser.parse_args(argv[1:])
  INSTALLERS[args.host](os.path.abspath(os.path.expanduser(args.dest)))
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
