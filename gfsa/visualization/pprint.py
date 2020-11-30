# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Pretty printer for dataclasses and ASTs."""
import ast
import sys
import dataclasses
import IPython
import IPython.lib.pretty

# pylint: disable=protected-access


class FancyPrettyPrinter(IPython.lib.pretty.RepresentationPrinter):
  """Pretty printer with fancy printing of dataclasses, ASTs, and dicts."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.type_pprinters.update({
        ast.AST: FancyPrettyPrinter.fields_pretty_,
        dict: FancyPrettyPrinter.dict_pretty_,
        object: FancyPrettyPrinter.fallback_pretty_,
    })

  @staticmethod
  def fields_pretty_(obj, p, cycle):
    """Pretty print an arbitrary object with _fields."""
    if hasattr(obj, "_fields"):
      fields = obj._fields
    elif dataclasses.is_dataclass(obj):
      fields = [field.name for field in dataclasses.fields(obj) if field.repr]
    else:
      raise ValueError(f"Can't find fields for {obj}")

    typename = type(obj).__name__
    if cycle:
      p.text(typename)
      p.text("(...)")
    else:
      with p.group(4, typename + "(", ")"):
        p.breakable("")
        for i, f in enumerate(fields):
          if i > 0:
            p.text(",")
            p.breakable()
          try:
            v = getattr(obj, f)
            with p.group(len(f) + 1, f + "=", ""):
              p.pretty(v)
          except AttributeError:
            p.text(f + "=<missing>")

  @staticmethod
  def dict_pretty_(obj, p, cycle):
    """Better pretty printer for dicts."""
    if cycle:
      return p.text("{...}")
    with p.group(1, "{", "}"):
      keys = obj.keys()
      for idx, key in p._enumerate(keys):
        if idx:
          p.text(",")
          p.breakable()

        if isinstance(key, str) and len(key) < 20:
          start = repr(key) + ": "
          with p.group(len(start), start, ""):
            p.pretty(obj[key])
        else:
          p.pretty(key)
          p.text(":")
          with p.group(2, "", ""):
            p.breakable()
            p.pretty(obj[key])

  @staticmethod
  def fallback_pretty_(obj, p, cycle):
    """Last-resort pretty printer, to check for dynamic things that aren't in MRO."""
    if dataclasses.is_dataclass(obj) or hasattr(obj, "_fields"):
      FancyPrettyPrinter.fields_pretty_(obj, p, cycle)
    else:
      IPython.lib.pretty._default_pprint(obj, p, cycle)

  def break_(self):
    """Improved break pretty printer.

    Any time we force a newline, first walk up the tree and break
    every containing object. This ensures that the newline has the right
    indentation level, and that we don't end up with weird outputs such as

      {
        1: MULTILINE
          MULTILINE
          MULTILINE, 2: MULTILINE
                        MULTILINE
                        MULTILINE, 3: 4
      }

    or worse

      {
        1: MULTILINE
        MULTILINE
        MULTILINE, 2: MULTILINE
        MULTILINE
        MULTILINE, 3: 4
      }

    which are hard to read.
    """
    # Force every containing group to break (based on _break_outer_groups)
    while True:
      group = self.group_queue.deq()
      if not group:
        break
      while group.breakables:
        x = self.buffer.popleft()
        self.output_width = x.output(self.output, self.output_width)
      while self.buffer and isinstance(self.buffer[0], IPython.lib.pretty.Text):
        x = self.buffer.popleft()
        self.output_width = x.output(self.output, self.output_width)
    # Insert the break as normal
    self.flush()
    self.output.write(self.newline)
    self.output.write(" " * self.indentation)
    self.output_width = self.indentation
    self.buffer_width = 0


def pprint(obj):
  printer = FancyPrettyPrinter(sys.stdout, max_width=128)
  printer.pretty(obj)
  printer.flush()
  sys.stdout.write("\n")
  sys.stdout.flush()
