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

"""Tests for install.py: every host gets a complete, parseable layout."""

import _path  # noqa: F401  (puts leap/ on sys.path)
import contextlib
import io
import json
import os
import re
import tempfile
import unittest

try:
  import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
  tomllib = None

import install

_WORKERS = ("lemma_informal_prover", "lemma_proof_worker",
            "lemma_decomposition_reviewer", "lemma_proof_verifier")


def _install(host):
  tmp = tempfile.TemporaryDirectory()
  with contextlib.redirect_stdout(io.StringIO()):
    install.INSTALLERS[host](tmp.name)
  return tmp


def _frontmatter(path):
  meta, body = install.parse_markdown(path)
  return meta, body


class SourceTest(unittest.TestCase):

  def test_every_capability_has_a_mapping(self):
    for meta, _ in install._agents():  # pylint: disable=protected-access
      for group in meta["tools"]:
        for host in install.TOOL_MAP:
          self.assertIn(group, install.TOOL_MAP[host], (meta["name"], host))

  def test_no_host_specific_tool_names_in_sources(self):
    names = {t for m in install.TOOL_MAP.values() for ts in m.values() for t in ts}
    for fname in os.listdir(install.AGENTS_DIR):
      with open(os.path.join(install.AGENTS_DIR, fname)) as f:
        text = f.read()
      for n in names:
        if "_" in n:  # skip ordinary words like Read / Bash
          self.assertNotIn(f"`{n}`", text, (fname, n))

  def test_skills_referenced_by_agents_exist(self):
    skills = set(os.listdir(install.SKILLS_DIR))
    for meta, _ in install._agents():  # pylint: disable=protected-access
      for s in meta.get("skills", []):
        self.assertIn(s, skills, meta["name"])


class ClaudeTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = _install("claude")
    self.addCleanup(self.tmp.cleanup)
    self.root = os.path.join(self.tmp.name, ".claude")

  def test_names_are_hyphenated_and_workers_cannot_dispatch(self):
    for w in _WORKERS:
      meta, body = _frontmatter(
          os.path.join(self.root, "agents", w.replace("_", "-") + ".md"))
      self.assertEqual(meta["name"], w.replace("_", "-"))
      self.assertNotIn("Agent", meta["tools"].split(", "))
      self.assertIn("Host notes: Claude Code", body)

  def test_orchestrator_can_dispatch(self):
    meta, _ = _frontmatter(os.path.join(self.root, "agents", "orchestrator-lemma.md"))
    self.assertIn("Agent", meta["tools"].split(", "))

  def test_settings_allow_binaries_and_merge(self):
    path = os.path.join(self.root, "settings.json")
    with open(path) as f:
      s = json.load(f)
    self.assertIn("Bash(state_manager_bin:*)", s["permissions"]["allow"])
    s["permissions"]["allow"].append("Bash(ls:*)")
    with open(path, "w") as f:
      json.dump(s, f)
    with contextlib.redirect_stdout(io.StringIO()):
      install.install_claude(self.tmp.name)
    with open(path) as f:
      s2 = json.load(f)
    self.assertIn("Bash(ls:*)", s2["permissions"]["allow"])
    self.assertEqual(s2["permissions"]["allow"].count("Bash(state_manager_bin:*)"), 1)

  def test_skill_cross_references_are_hyphenated(self):
    with open(os.path.join(self.root, "skills", "lemma-mathlib-search", "SKILL.md")) as f:
      text = f.read()
    self.assertNotRegex(text, r"lemma_mathlib_")


class CodexTest(unittest.TestCase):

  @unittest.skipIf(tomllib is None, "needs Python 3.11+ for tomllib")
  def test_workers_are_valid_toml_and_orchestrator_is_a_skill(self):
    tmp = _install("codex")
    self.addCleanup(tmp.cleanup)
    for w in _WORKERS:
      with open(os.path.join(tmp.name, ".codex", "agents", w + ".toml"), "rb") as f:
        agent = tomllib.load(f)
      self.assertEqual(agent["name"], w)
      self.assertIn("Host notes: Codex", agent["developer_instructions"])
    meta, body = _frontmatter(
        os.path.join(tmp.name, ".agents", "skills", "leap-prove", "SKILL.md"))
    self.assertEqual(meta["name"], "leap-prove")
    self.assertIn("DISPATCH-ONLY", body)


class AntigravityTest(unittest.TestCase):

  def test_native_tool_names(self):
    tmp = _install("antigravity")
    self.addCleanup(tmp.cleanup)
    meta, _ = _frontmatter(
        os.path.join(tmp.name, ".agents", "agents", "orchestrator_lemma.md"))
    self.assertIn("invoke_subagent", meta["tools"])
    self.assertIn("run_command", meta["tools"])
    self.assertTrue(meta["mainAgent"])
    meta, _ = _frontmatter(
        os.path.join(tmp.name, ".agents", "agents", "lemma_proof_worker.md"))
    self.assertNotIn("invoke_subagent", meta["tools"])
    self.assertTrue(all(re.match(r"skills/\w+$", s) for s in meta["skills"]))


if __name__ == "__main__":
  unittest.main()
