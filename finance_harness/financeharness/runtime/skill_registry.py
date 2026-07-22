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

"""Skill registry — catalog + load-on-demand for workflow recipes.

A skill is `skills/<name>/SKILL.md`: YAML frontmatter (name == dir, description,
tags, requires_tools) + a markdown body the model follows as a recipe. Parallel
to the tool registry: catalog in the prompt, `load_skill` pulls the body and
auto-loads the skill's `requires_tools`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from financeharness.runtime import frontmatter as fm_lib

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]+$")
_MIN_DESC = 30
_ALLOWED_KEYS = frozenset({"name", "description", "tags", "requires_tools"})


@dataclass(frozen=True)
class SkillSpec:
  """A parsed skill recipe: catalog metadata, required tools, and markdown body."""

  name: str
  description: str
  body: str
  tags: tuple[str, Ellipsis] = field(default_factory=tuple)
  requires_tools: tuple[str, Ellipsis] = field(default_factory=tuple)
  source_path: str | None = None

  def __post_init__(self):
    if not _NAME_RE.match(self.name):
      raise fm_lib.FrontmatterError(
          f"skill name {self.name!r} must be kebab-case"
      )
    if len(self.description) < _MIN_DESC:
      raise fm_lib.FrontmatterError(
          f"skill {self.name} description < {_MIN_DESC} chars"
      )
    if not self.body.strip():
      raise fm_lib.FrontmatterError(f"skill {self.name} body is empty")

  def catalog_line(self):
    tag_suffix = f" Tags: {', '.join(self.tags)}." if self.tags else ""
    return f"- `{self.name}`: {self.description}{tag_suffix}"


def parse_skill_md(text, *, source_path = None):
  """Parse one SKILL.md document into a validated :class:`SkillSpec`."""

  fm, body = fm_lib.split_frontmatter(text, source=source_path)
  unknown = set(fm) - _ALLOWED_KEYS
  if unknown:
    raise fm_lib.FrontmatterError(
        f"skill frontmatter has unknown keys: {sorted(unknown)}"
    )
  name = fm.get("name")
  description = fm.get("description")
  if not isinstance(name, str) or not isinstance(description, str):
    raise fm_lib.FrontmatterError(
        "skill frontmatter needs string `name` and `description`"
    )
  return SkillSpec(
      name=name,
      description=description,
      body=body,
      tags=fm_lib.as_str_list(fm, "tags"),
      requires_tools=fm_lib.as_str_list(fm, "requires_tools"),
      source_path=source_path,
  )


def load_skill_dir(skill_dir):
  """Load and validate ``skill_dir/SKILL.md``."""

  md = skill_dir / "SKILL.md"
  if not md.is_file():
    raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")
  spec = parse_skill_md(md.read_text(encoding="utf-8"), source_path=str(md))
  if spec.name != skill_dir.name:
    raise fm_lib.FrontmatterError(
        f"skill name {spec.name!r} != directory {skill_dir.name!r}"
    )
  return spec


class SkillRegistry:
  """In-memory catalog of workflow skills available to a run."""

  def __init__(self):
    self._skills: dict[str, SkillSpec] = {}

  def register(self, spec, *, override = False):
    if spec.name in self._skills and not override:
      raise ValueError(f"Skill {spec.name} already registered")
    self._skills[spec.name] = spec

  def get(self, name):
    return self._skills.get(name)

  def names(self):
    return sorted(self._skills)

  def all(self):
    return list(self._skills.values())

  def catalog_text(self):
    return "\n".join(
        s.catalog_line()
        for s in sorted(self._skills.values(), key=lambda s: s.name)
    )


@dataclass
class SkillSessionState:
  """Per-run skill state: the recipes already loaded for this conversation."""

  loaded: set[str] = field(default_factory=set)


def load_skill_registry(skills_root):
  """Register every `skills/<name>/SKILL.md` under ``skills_root`` (strict — raises

  on a malformed skill; used for the bundled first-party skills).
  """
  reg = SkillRegistry()
  merge_skills(reg, skills_root, strict=True)
  return reg


def merge_skills(
    reg, skills_root, *, strict = False
):
  """Merge `<name>/SKILL.md` skills from ``skills_root`` into ``reg`` (override by

  name — later roots win). Returns the count merged. With ``strict=False`` a
  malformed user skill is skipped, not fatal — a dropped-in skill can't break a
  run.
  """
  if not skills_root.is_dir():
    return 0
  n = 0
  for d in sorted(p for p in skills_root.iterdir() if p.is_dir()):
    if not (d / "SKILL.md").is_file():
      continue
    try:
      reg.register(load_skill_dir(d), override=True)
      n += 1
    except Exception:  # noqa: BLE001
      if strict:
        raise
      continue  # best-effort for user/project skills
  return n
