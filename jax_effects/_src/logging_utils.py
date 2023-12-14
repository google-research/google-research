# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Printing and logging utilities."""

from collections.abc import Mapping
import re
from typing import Any

from jax import core as jax_core

import rich.console
import rich.highlighter
import rich.panel
import rich.pretty
import rich.text
import rich.theme


CONSOLE = rich.console.Console()

STEP_STYLE = 'bold blue'
ACTION_STYLE = 'bold purple4'

rprint = CONSOLE.print


class JaxprHighlighter(rich.highlighter.RegexHighlighter):
  """Apply style to Jaxpr text."""

  base_style = 'jaxpr.'
  highlights = [
      r'(?P<keyword> (lambda|in|let) )',
      r'(?P<type_annotation>:\w+\[[0-9,]*\])',
      r'(?P<handler>(?<= = )(handler|delimited_handler|handler_return)(?=\[))',
      r'(?P<handler_name>(?<=\s)name=[^]\s]+(?=\s*))',
  ]


JAXPR_HIGHLIGHTER = JaxprHighlighter()


def escape_ansi(line: str) -> str:
  ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
  return ansi_escape.sub('', line)


CONSOLE_JAXPR = rich.console.Console(
    highlighter=JAXPR_HIGHLIGHTER,
    theme=rich.theme.Theme({
        'jaxpr.keyword': 'bold blue',
        'jaxpr.type_annotation': 'magenta',
        'jaxpr.handler': 'bold on bright_yellow',
        'jaxpr.handler_name': 'bold on bright_green',
    }),
)


def highlight_jaxpr(obj: Any) -> rich.text.Text:
  return JAXPR_HIGHLIGHTER(escape_ansi(str(obj)))


def print_jaxpr(
    text: Any, title: str | None = None, panel: bool = True
) -> None:
  if panel:
    CONSOLE_JAXPR.print(rich.panel.Panel(highlight_jaxpr(text), title=title))
  else:
    CONSOLE_JAXPR.rule(title=title)
    CONSOLE_JAXPR.print(highlight_jaxpr(text))


def print_jaxpr_atom_mapping(
    mapping: Mapping[jax_core.Atom, jax_core.Atom],
    ctx: jax_core.JaxprPpContext,
    title: str | None = None,
    panel: bool = True,
) -> None:
  """Pretty-prints a mapping on Jaxpr values."""
  d = {
      jax_core.pp_var(key, ctx): jax_core.pp_var(value, ctx)
      for key, value in mapping.items()
  }
  if panel:
    CONSOLE_JAXPR.print(rich.panel.Panel(highlight_jaxpr(d), title=title))
  else:
    CONSOLE_JAXPR.rule(title=title)
    CONSOLE_JAXPR.print(highlight_jaxpr(d))


def print_panel(renderable: Any, title: str | None = None) -> None:
  CONSOLE_JAXPR.print(
      rich.panel.Panel(rich.pretty.Pretty(renderable), title=title)
  )
