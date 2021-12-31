# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""These templates define a program distribution full of runtime errors.

The Python programs may include the follow error types:
  NoError
  RuntimeError  # 1 second timeout
  ZeroDivisionError
  AssertionError
  ValueError  # math domain error
  TypeError
  IndexError
  NameError
  AttributeError
  RecursionError
  MemoryError

NoError indicates the program runs without error.
RuntimeError generally refers to a 1 second timeout, which to the best of my
knowledge always indicates an infinite loop (though in principle could be a
timeout of a program that would eventually terminate.)
ValueError in the current grammar generally indicates taking the square root of
a negative value.
RecursionErrors should not occur, and may indicate a bug in the generator if
they do.
"""

import dataclasses
import enum
from typing import FrozenSet, Optional, Text

from ipagnn.datasets.control_flow_programs.program_generators import constants
from ipagnn.datasets.control_flow_programs.program_generators import top_down_refinement
from ipagnn.datasets.control_flow_programs.program_generators.templates import base as base_templates

ConfigurableTemplate = base_templates.ConfigurableTemplate
TemplateData = base_templates.TemplateData

Program = top_down_refinement.ThingWithHoles

NAME_ERROR_WEIGHT = 0
TYPE_ERROR_WEIGHT = 1


class Hole(top_down_refinement.Hole):

  @property
  def indent(self):
    return self.metadata.indent_str


is_ivar = lambda v: v.startswith('v')
is_bvar = lambda v: v.startswith('b')
is_lvar = lambda v: v.startswith('L')


@dataclasses.dataclass(frozen=True)
class HoleMetadata:
  """Metadata passed to a hole guiding its construction."""
  indent: int = 0
  inside_loop: bool = False

  required_write: Optional[Text] = None  # MUST write this var. (Used by INIT.)
  allowed_reads: FrozenSet[Text] = frozenset()  # OK to read these vars.
  required_reads: FrozenSet[Text] = frozenset()  # MUST read these vars.

  def set(self, **kwargs):
    return dataclasses.replace(self, **kwargs)

  def split(self):
    """Splits HoleMetadata for use across two holes."""
    # Put required_reads and required_write on the later block.
    # Allowed reads are shared.
    # TODO(dbieber): Consider other splits as well.
    metadata1 = self.set(required_write=None, required_reads=frozenset())
    metadata2 = self
    return metadata1, metadata2

  @property
  def indent_str(self):
    return constants.INDENT_STRING * self.indent

  @property
  def unused_variables(self):
    return ((self.ivars_allowed | self.bvars_allowed | self.lvars_allowed)
            - self.allowed_reads)

  @property
  def unused_ivariables(self):
    return self.ivars_allowed - self.allowed_reads

  @property
  def unused_bvariables(self):
    return self.bvars_allowed - self.allowed_reads

  @property
  def unused_lvariables(self):
    return self.lvars_allowed - self.allowed_reads

  @property
  def ivars_allowed(self):
    # TODO(dbieber): Make these three properties constants or class attributes.
    return {'v0', 'v1', 'v2', 'v3', 'v4', 'v5'}

  @property
  def bvars_allowed(self):
    return {'b0', 'b1', 'b2'}

  @property
  def lvars_allowed(self):
    return {'L0', 'L1', 'L2'}

  @property
  def ivars_allowed_reads(self):
    return {v for v in self.allowed_reads if is_ivar(v)}

  @property
  def bvars_allowed_reads(self):
    return {v for v in self.allowed_reads if is_bvar(v)}

  @property
  def lvars_allowed_reads(self):
    return {v for v in self.allowed_reads if is_lvar(v)}

  @property
  def can_read(self):
    return bool(self.allowed_reads)

  @property
  def can_iread(self):
    return any(is_ivar(v) for v in self.allowed_reads)

  @property
  def can_bread(self):
    return any(is_bvar(v) for v in self.allowed_reads)

  @property
  def can_lread(self):
    return any(is_lvar(v) for v in self.allowed_reads)

  @property
  def can_modify(self):  # A "modify" is a read and write (e.g. AugAssign.)
    if self.required_write:
      return self.required_write in self.allowed_reads - self.required_reads
    return bool(self.allowed_reads - self.required_reads)

  @property
  def can_imodify(self):
    if self.required_write:
      return (self.required_write in (self.allowed_reads - self.required_reads)
              and is_ivar(self.required_write))
    # We remove the required reads because a required read must not be written
    # to, because this is the end of its scope.
    return any(is_ivar(v) for v in self.allowed_reads - self.required_reads)

  @property
  def can_bmodify(self):
    if self.required_write:
      return (self.required_write in self.allowed_reads - self.required_reads
              and is_bvar(self.required_write))
    return any(is_bvar(v) for v in self.allowed_reads - self.required_reads)

  @property
  def can_lmodify(self):
    if self.required_write:
      return (self.required_write in self.allowed_reads - self.required_reads
              and is_lvar(self.required_write))
    return any(is_lvar(v) for v in self.allowed_reads - self.required_reads)

  @property
  def must_read(self):
    return bool(self.required_reads)

  @property
  def must_iread(self):
    return self.required_reads and any(is_ivar(v) for v in self.required_reads)

  @property
  def num_required_ireads(self):
    return len([v for v in self.required_reads if is_ivar(v)])

  @property
  def must_bread(self):
    return self.required_reads and any(is_bvar(v) for v in self.required_reads)

  @property
  def num_required_breads(self):
    return len([v for v in self.required_reads if is_bvar(v)])

  @property
  def must_lread(self):
    return self.required_reads and any(is_lvar(v) for v in self.required_reads)

  @property
  def num_required_lreads(self):
    return len([v for v in self.required_reads if is_lvar(v)])

  @property
  def must_write(self):
    return bool(self.required_write)

  @property
  def must_iwrite(self):
    return self.required_write and is_ivar(self.required_write)

  @property
  def must_bwrite(self):
    return self.required_write and is_bvar(self.required_write)

  @property
  def must_lwrite(self):
    return self.required_write and is_lvar(self.required_write)


class HoleType(enum.Enum):
  """A type of hole for this task."""
  PROGRAM = 'PROGRAM'
  STMT = 'STMT'
  BLOCK = 'BLOCK'
  BLOCK_END = 'BLOCK_END'
  STMTS = 'STMTS'
  STMTS_NONEMPTY = 'STMTS_NONEMPTY'
  BVAR_READ = 'BVAR_READ'
  BVAR_MODIFY = 'BVAR_MODIFY'
  BAUG_OP = 'BAUG_OP'
  BCONSTANT = 'BCONSTANT'
  IVAR_READ = 'IVAR_READ'
  IVAR_MODIFY = 'IVAR_MODIFY'
  VAR_WRITE = 'VAR_WRITE'
  IAUG_OP = 'IAUG_OP'
  IBIN_OP = 'IBIN_OP'
  IBIN_OP_COMP = 'IBIN_OP_COMP'
  ICONSTANT = 'ICONSTANT'
  LVAR_READ = 'LVAR_READ'
  LVAR_MODIFY = 'LVAR_MODIFY'
  LCONSTANT = 'LCONSTANT'
  CONDITION = 'CONDITION'


class Accesses(enum.Enum):
  """A type of access (read/write/modify)."""
  BVAR_READ = 'BVAR_READ'
  BVAR_WRITE = 'BVAR_WRITE'
  BVAR_MODIFY = 'BVAR_MODIFY'
  IVAR_READ = 'IVAR_READ'
  IVAR_WRITE = 'IVAR_WRITE'
  IVAR_MODIFY = 'IVAR_MODIFY'
  LVAR_READ = 'LVAR_READ'
  LVAR_WRITE = 'LVAR_WRITE'
  LVAR_MODIFY = 'LVAR_MODIFY'
  # A compound statement cannot register all its accesses up front, so it
  # indicates that there can be additional accesses with 'MORE'.
  MORE = 'MORE'


HOLE_TYPE_WEIGHTS = {
    HoleType.PROGRAM: 10,
    HoleType.STMT: 150,
    HoleType.BLOCK: 100,
    HoleType.BLOCK_END: 10,
    HoleType.STMTS: 5,
    HoleType.STMTS_NONEMPTY: 50,
    HoleType.BVAR_READ: 10,
    HoleType.BVAR_MODIFY: 10,
    HoleType.BAUG_OP: 10,
    HoleType.BCONSTANT: 10,
    HoleType.IVAR_READ: 10,
    HoleType.IVAR_MODIFY: 10,
    HoleType.VAR_WRITE: 10,
    HoleType.IAUG_OP: 10,
    HoleType.IBIN_OP: 10,
    HoleType.IBIN_OP_COMP: 10,
    HoleType.ICONSTANT: 10,
    HoleType.LVAR_READ: 10,
    HoleType.LVAR_MODIFY: 10,
    HoleType.LCONSTANT: 10,
    HoleType.CONDITION: 10,
}


def const(x):
  return lambda *unused: x


def concat(*lines):
  return sum(lines, [])


def fmt(format_string):
  return lambda *args: [format_string.format(*args)]


def _max_value(config):
  if config.max_value is not None:
    return config.max_value
  return config.base ** config.num_digits - 1


REGISTERED_TEMPLATES = []


def register_template(template_cls):
  REGISTERED_TEMPLATES.append(template_cls)
  return template_cls


# pylint: disable=missing-class-docstring
@register_template
class ProgramTemplate(ConfigurableTemplate):
  fills_type = HoleType.PROGRAM
  required_cost = 2
  weight = 10

  def fill(self, hole, rng):
    init = Hole(HoleType.STMT,
                hole.metadata.set(required_write='v0'))
    body = Hole(HoleType.STMTS_NONEMPTY,
                hole.metadata.set(allowed_reads={'v0'},
                                  required_write='v0'))
    return Program(0, [init, body], concat)


@register_template
class StatementsNonEmptySingle(ConfigurableTemplate):
  fills_type = HoleType.STMTS_NONEMPTY
  required_cost = 1
  precedence = 0
  weight = 50

  def fill(self, hole, rng):
    stmt = Hole(HoleType.STMT, hole.metadata)
    return Program(0, [stmt], concat)

  def can_fill(self, hole, space, templates):
    return exists_stmt_satisfying(hole.metadata, space, templates)


@register_template
class StatementsNonEmptyMultiple(ConfigurableTemplate):
  fills_type = HoleType.STMTS_NONEMPTY
  required_cost = 2
  weight = 10

  def fill(self, hole, rng):
    metadata1, metadata2 = hole.metadata.split()
    stmt = Hole(HoleType.STMT, metadata1)
    stmts = Hole(HoleType.STMTS_NONEMPTY, metadata2)
    return Program(0, [stmt, stmts], concat)

  def can_fill(self, hole, space, templates):
    metadata1, metadata2 = hole.metadata.split()
    return (
        exists_stmt_satisfying(metadata1, 1, templates)
        and exists_stmt_satisfying(metadata2, 1, templates)
    )


@register_template
class StatementsNonEmptyMultipleFlip(ConfigurableTemplate):
  fills_type = HoleType.STMTS_NONEMPTY
  required_cost = 2
  weight = 100

  def fill(self, hole, rng):
    metadata1, metadata2 = hole.metadata.split()
    stmts = Hole(HoleType.STMTS_NONEMPTY, metadata1)
    stmt = Hole(HoleType.STMT, metadata2)
    return Program(0, [stmts, stmt], concat)

  def can_fill(self, hole, space, templates):
    metadata1, metadata2 = hole.metadata.split()
    return (
        exists_stmt_satisfying(metadata1, 1, templates)
        and exists_stmt_satisfying(metadata2, 1, templates))


@register_template
class StatementsNonEmptyMultipleGroups(ConfigurableTemplate):
  fills_type = HoleType.STMTS_NONEMPTY
  required_cost = 2
  weight = 10

  def fill(self, hole, rng):
    metadata1, metadata2 = hole.metadata.split()
    stmts1 = Hole(HoleType.STMTS_NONEMPTY, metadata1)
    stmts2 = Hole(HoleType.STMTS_NONEMPTY, metadata2)
    return Program(0, [stmts1, stmts2], concat)

  def can_fill(self, hole, space, templates):
    metadata1, metadata2 = hole.metadata.split()
    return (
        exists_stmt_satisfying(metadata1, 1, templates)
        and exists_stmt_satisfying(metadata2, 1, templates))


@register_template
class StatementsNonEmptyNewIVariable(ConfigurableTemplate):
  fills_type = HoleType.STMTS_NONEMPTY
  required_cost = 2
  weight = 30

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.unused_ivariables))
    init = Hole(HoleType.STMT,
                hole.metadata.set(required_write=var))
    stmts = Hole(HoleType.STMTS_NONEMPTY,
                 hole.metadata.set(
                     allowed_reads=hole.metadata.allowed_reads | {var},
                     required_reads=hole.metadata.required_reads | {var}))
    return Program(0, [init, stmts], concat)

  def can_fill(self, hole, space, templates):
    # if hole.metadata.required_write:
    #   return False
    if not hole.metadata.unused_ivariables:
      return False
    var = list(hole.metadata.unused_ivariables)[0]
    return exists_stmt_satisfying(
        hole.metadata.set(
            allowed_reads=hole.metadata.allowed_reads | {var},
            required_reads=hole.metadata.required_reads | {var}
        ),
        1, templates
    )


@register_template
class StatementsNonEmptyNewBVariable(ConfigurableTemplate):
  fills_type = HoleType.STMTS_NONEMPTY
  required_cost = 3
  weight = 20

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.unused_bvariables))
    init = Hole(HoleType.STMT,
                hole.metadata.set(required_write=var))
    stmts = Hole(HoleType.STMTS_NONEMPTY,
                 hole.metadata.set(
                     allowed_reads=hole.metadata.allowed_reads | {var},
                     required_reads=hole.metadata.required_reads | {var}))
    return Program(0, [init, stmts], concat)

  def can_fill(self, hole, space, templates):
    # if hole.metadata.required_write:
    #   return False
    if not hole.metadata.unused_bvariables:
      return False
    var = list(hole.metadata.unused_bvariables)[0]
    return exists_stmt_satisfying(
        hole.metadata.set(
            allowed_reads=hole.metadata.allowed_reads | {var},
            required_reads=hole.metadata.required_reads | {var}
        ),
        1, templates
    )


@register_template
class StatementsNonEmptyNewLVariable(ConfigurableTemplate):
  fills_type = HoleType.STMTS_NONEMPTY
  required_cost = 2
  weight = 20

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.unused_lvariables))
    init = Hole(HoleType.STMT,
                hole.metadata.set(required_write=var))
    stmts = Hole(HoleType.STMTS_NONEMPTY,
                 hole.metadata.set(
                     allowed_reads=hole.metadata.allowed_reads | {var},
                     required_reads=hole.metadata.required_reads | {var}))
    return Program(0, [init, stmts], concat)

  def can_fill(self, hole, space, templates):
    # if hole.metadata.required_write:
    #   return False
    if not hole.metadata.unused_lvariables:
      return False
    var = list(hole.metadata.unused_lvariables)[0]
    return exists_stmt_satisfying(
        hole.metadata.set(
            allowed_reads=hole.metadata.allowed_reads | {var},
            required_reads=hole.metadata.required_reads | {var}
        ),
        1, templates
    )


def accesses_can_fill(accesses, metadata, space):
  """Determines if a particular access pattern satisfies a Hole's metadata."""
  if Accesses.IVAR_WRITE in accesses and not metadata.must_iwrite:
    return False
  if Accesses.BVAR_WRITE in accesses and not metadata.must_bwrite:
    return False
  if Accesses.LVAR_WRITE in accesses and not metadata.must_lwrite:
    return False
  if Accesses.IVAR_READ in accesses and not metadata.can_iread:
    return False
  if Accesses.BVAR_READ in accesses and not metadata.can_bread:
    return False
  if Accesses.LVAR_READ in accesses and not metadata.can_lread:
    return False
  if Accesses.IVAR_MODIFY in accesses and not metadata.can_imodify:
    return False
  if Accesses.BVAR_MODIFY in accesses and not metadata.can_bmodify:
    return False
  if Accesses.LVAR_MODIFY in accesses and not metadata.can_lmodify:
    return False
  if metadata.must_iwrite and (Accesses.IVAR_WRITE not in accesses
                               and Accesses.IVAR_MODIFY not in accesses
                               and (Accesses.MORE not in accesses
                                    or space < 2)):
    return False
  if metadata.must_bwrite and (Accesses.BVAR_WRITE not in accesses
                               and Accesses.BVAR_MODIFY not in accesses
                               and (Accesses.MORE not in accesses
                                    or space < 2)):
    return False
  if metadata.must_lwrite and (Accesses.LVAR_WRITE not in accesses
                               and Accesses.LVAR_MODIFY not in accesses
                               and (Accesses.MORE not in accesses
                                    or space < 2)):
    return False
  num_possible_ireads = accesses.count(Accesses.IVAR_READ)
  num_possible_breads = accesses.count(Accesses.BVAR_READ)
  num_possible_lreads = accesses.count(Accesses.LVAR_READ)
  if Accesses.MORE in accesses and space > 2:
    num_possible_ireads += space - 2  # 1 for the condition, 1 for the write.
    num_possible_breads += space - 2
    num_possible_lreads += space - 2
  if num_possible_ireads < metadata.num_required_ireads:
    return False
  if num_possible_breads < metadata.num_required_breads:
    return False
  if num_possible_lreads < metadata.num_required_lreads:
    return False
  return True


class ConfigurableStatementTemplate(ConfigurableTemplate):
  accesses = []

  def can_fill(self, hole, space, templates):
    return accesses_can_fill(self.accesses, hole.metadata, space)


class ConfigurableConditionTemplate(ConfigurableTemplate):
  accesses = []

  def can_fill(self, hole, space, templates):
    return accesses_can_fill(self.accesses, hole.metadata, space)


@register_template
class IInitTemplate(ConfigurableStatementTemplate):
  """Assign statement, e.g. v0 = 3."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 400
  accesses = [Accesses.IVAR_WRITE]

  def fill(self, hole, rng):
    var = Hole(HoleType.VAR_WRITE, hole.metadata)
    value = Hole(HoleType.ICONSTANT, hole.metadata)
    return Program(1, [var, value], fmt(f'{hole.indent}{{0}} = {{1}}'))


@register_template
class BInitTemplate(ConfigurableStatementTemplate):
  """Assign statement, e.g. b0 = True."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 4
  accesses = [Accesses.BVAR_WRITE]

  def fill(self, hole, rng):
    var = Hole(HoleType.VAR_WRITE, hole.metadata)
    value = Hole(HoleType.BCONSTANT, hole.metadata)
    return Program(1, [var, value], fmt(f'{hole.indent}{{0}} = {{1}}'))


@register_template
class LInitTemplate(ConfigurableStatementTemplate):
  """Assign statement, e.g. L0 = []."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 4
  accesses = [Accesses.LVAR_WRITE]

  def fill(self, hole, rng):
    var = Hole(HoleType.VAR_WRITE, hole.metadata)
    value = Hole(HoleType.LCONSTANT, hole.metadata)
    return Program(1, [var, value], fmt(f'{hole.indent}{{0}} = {{1}}'))


@register_template
class LInitIVar(ConfigurableStatementTemplate):
  """Assign statement, e.g. L0 = [v1]."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 4
  accesses = [Accesses.LVAR_WRITE, Accesses.IVAR_READ]

  def fill(self, hole, rng):
    var = Hole(HoleType.VAR_WRITE, hole.metadata)
    value = Hole(HoleType.IVAR_READ, hole.metadata)
    return Program(1, [var, value], fmt(f'{hole.indent}{{0}} = [{{1}}]'))


@register_template
class IVarAugOpConstant(ConfigurableStatementTemplate):
  """AugAssign statement, e.g. v1 += 3."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 10
  accesses = [Accesses.IVAR_MODIFY]

  def fill(self, hole, rng):
    var = Hole(HoleType.IVAR_MODIFY, hole.metadata)
    aug_op = Hole(HoleType.IAUG_OP, hole.metadata)
    constant = Hole(HoleType.ICONSTANT, hole.metadata)
    return Program(1, [var, aug_op, constant],
                   fmt(f'{hole.indent}{{0}} {{1}} {{2}}'))


@register_template
class IVarSqrtIVar(ConfigurableStatementTemplate):
  """AugAssign statement, e.g. v1 = sqrt(v2)."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 5
  accesses = [Accesses.IVAR_WRITE, Accesses.IVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.VAR_WRITE, hole.metadata)
    var2 = Hole(HoleType.IVAR_READ, hole.metadata)
    return Program(1, [var1, var2],
                   fmt(f'{hole.indent}{{0}} = sqrt({{1}})'))


@register_template
class LVarAppendIVar(ConfigurableStatementTemplate):
  """AugAssign statement, e.g. L1.append(v2)."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 5
  accesses = [Accesses.LVAR_MODIFY, Accesses.IVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.LVAR_MODIFY, hole.metadata)
    var2 = Hole(HoleType.IVAR_READ, hole.metadata)
    return Program(1, [var1, var2],
                   fmt(f'{hole.indent}{{0}}.append({{1}})'))


@register_template
class IVarPopLVar(ConfigurableStatementTemplate):
  """Statement, e.g. v1 = L1.pop()."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 5
  accesses = [Accesses.IVAR_WRITE, Accesses.LVAR_MODIFY]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.VAR_WRITE, hole.metadata)
    var2 = Hole(HoleType.LVAR_MODIFY, hole.metadata)
    return Program(1, [var1, var2],
                   fmt(f'{hole.indent}{{0}} = {{1}}.pop()'))


@register_template
class LVarCopyLVar(ConfigurableStatementTemplate):
  """Statement, e.g. L1 = L2.copy()."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 5
  accesses = [Accesses.LVAR_WRITE, Accesses.LVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.VAR_WRITE, hole.metadata)
    var2 = Hole(HoleType.LVAR_READ, hole.metadata)
    return Program(1, [var1, var2],
                   fmt(f'{hole.indent}{{0}} = {{1}}.copy()'))


@register_template
class IVarIndexLVarByIVar(ConfigurableStatementTemplate):
  """Statement, e.g. v1 = L1[v2]."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 5
  accesses = [Accesses.IVAR_WRITE, Accesses.LVAR_READ, Accesses.IVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.VAR_WRITE, hole.metadata)
    var2 = Hole(HoleType.LVAR_READ, hole.metadata)
    var3 = Hole(HoleType.IVAR_READ, hole.metadata)
    return Program(1, [var1, var2, var3],
                   fmt(f'{hole.indent}{{0}} = {{1}}[{{2}}]'))


@register_template
class IVarLenLVar(ConfigurableStatementTemplate):
  """Statement, e.g. v1 = len(L1)."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 5
  accesses = [Accesses.IVAR_WRITE, Accesses.LVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.VAR_WRITE, hole.metadata)
    var2 = Hole(HoleType.LVAR_READ, hole.metadata)
    return Program(1, [var1, var2],
                   fmt(f'{hole.indent}{{0}} = len({{1}})'))


@register_template
class BVarAugOpConstant(ConfigurableStatementTemplate):
  """AugAssign statement, e.g. b1 &= True."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 5
  accesses = [Accesses.BVAR_MODIFY]

  def fill(self, hole, rng):
    var = Hole(HoleType.BVAR_MODIFY, hole.metadata)
    aug_op = Hole(HoleType.BAUG_OP, hole.metadata)
    constant = Hole(HoleType.BCONSTANT, hole.metadata)
    return Program(1, [var, aug_op, constant],
                   fmt(f'{hole.indent}{{0}} {{1}} {{2}}'))


@register_template
class IVarAugOpIVar(ConfigurableStatementTemplate):
  """AugAssign statement, e.g. v1 += v2."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 10
  accesses = [Accesses.IVAR_MODIFY, Accesses.IVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.IVAR_MODIFY, hole.metadata)
    aug_op = Hole(HoleType.IAUG_OP, hole.metadata)
    var2 = Hole(HoleType.IVAR_READ, hole.metadata)
    return Program(1, [var1, aug_op, var2],
                   fmt(f'{hole.indent}{{0}} {{1}} {{2}}'))


@register_template
class BVarAugOpBVar(ConfigurableStatementTemplate):
  """AugAssign statement, e.g. b1 |= b2."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 10
  accesses = [Accesses.BVAR_MODIFY, Accesses.BVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.BVAR_MODIFY, hole.metadata)
    aug_op = Hole(HoleType.BAUG_OP, hole.metadata)
    var2 = Hole(HoleType.BVAR_READ, hole.metadata)
    return Program(1, [var1, aug_op, var2],
                   fmt(f'{hole.indent}{{0}} {{1}} {{2}}'))


@register_template
class BVarIVarInLVar(ConfigurableStatementTemplate):
  """AugAssign statement, e.g. b0 = v1 in L2."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 10
  accesses = [Accesses.BVAR_WRITE, Accesses.IVAR_READ, Accesses.LVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.VAR_WRITE, hole.metadata)
    var2 = Hole(HoleType.IVAR_READ, hole.metadata)
    var3 = Hole(HoleType.LVAR_READ, hole.metadata)
    return Program(1, [var1, var2, var3],
                   fmt(f'{hole.indent}{{0}} = {{1}} in {{2}}'))


@register_template
class BVarBinOpIntVarZero(ConfigurableStatementTemplate):
  """AugAssign statement, e.g. b1 = v3 > 0."""
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 5
  accesses = [Accesses.BVAR_WRITE, Accesses.IVAR_READ]

  def fill(self, hole, rng):
    bvar = Hole(HoleType.VAR_WRITE, hole.metadata)
    ivar = Hole(HoleType.IVAR_READ, hole.metadata)
    ibin_op = Hole(HoleType.IBIN_OP_COMP, hole.metadata)
    return Program(1, [bvar, ivar, ibin_op],
                   fmt(f'{hole.indent}{{0}} = {{1}} {{2}} 0'))


@register_template
class AssertStatement(ConfigurableStatementTemplate):
  fills_type = HoleType.STMT
  required_cost = 1
  weight = 1

  def fill(self, hole, rng):
    cond = Hole(HoleType.CONDITION, hole.metadata)
    return Program(1, [cond], fmt(f'{hole.indent}assert {{0}}'))

  def can_fill(self, hole, space, templates):
    return exists_cond_satisfying(hole.metadata, 0, templates)


@register_template
class BVarCondition(ConfigurableConditionTemplate):
  fills_type = HoleType.CONDITION
  required_cost = 0
  weight = 10
  accesses = [Accesses.BVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.BVAR_READ, hole.metadata)
    return Program(0, [var1], '{0}'.format)


@register_template
class NotBVarCondition(ConfigurableConditionTemplate):
  fills_type = HoleType.CONDITION
  required_cost = 0
  weight = 10
  accesses = [Accesses.BVAR_READ]

  def fill(self, hole, rng):
    var1 = Hole(HoleType.BVAR_READ, hole.metadata)
    return Program(0, [var1], 'not {0}'.format)


@register_template
class ComparatorCondition(ConfigurableConditionTemplate):
  fills_type = HoleType.CONDITION
  required_cost = 0
  weight = 10
  accesses = [Accesses.IVAR_READ, Accesses.IVAR_READ]

  def fill(self, hole, rng):
    [var1, var2] = choose_reads(hole, rng, 2, is_ivar)  # pylint: disable=unbalanced-tuple-unpacking
    op = Hole(HoleType.IBIN_OP_COMP, hole.metadata)
    return Program(0, [op], f'{var1} {{0}} {var2}'.format)

  def can_fill(self, hole, space, templates):
    if len(hole.metadata.ivars_allowed_reads) < 2:
      return False
    return super(ComparatorCondition, self).can_fill(hole, space, templates)


def choose_reads(hole, rng, n, predicate):
  """Select reads from the hole metadata."""
  reads = []
  required_reads = set(v for v in hole.metadata.required_reads if predicate(v))
  allowed_reads = set(v for v in hole.metadata.allowed_reads if predicate(v))
  while len(reads) < n and (required_reads or allowed_reads):
    if required_reads:
      read = rng.choice(list(required_reads))
      reads.append(read)
      required_reads = required_reads - {read}
      allowed_reads = allowed_reads - {read}
    else:
      read = rng.choice(list(allowed_reads))
      reads.append(read)
      required_reads = required_reads - {read}
      allowed_reads = allowed_reads - {read}
  return reads


@register_template
class IVarRead(ConfigurableTemplate):
  fills_type = HoleType.IVAR_READ
  required_cost = 0
  weight = 40

  def fill(self, hole, rng):
    [var] = choose_reads(hole, rng, 1, is_ivar)  # pylint: disable=unbalanced-tuple-unpacking
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return hole.metadata.can_iread


@register_template
class IVarReadNameError(ConfigurableTemplate):
  fills_type = HoleType.IVAR_READ
  required_cost = 0
  weight = NAME_ERROR_WEIGHT

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.ivars_allowed
                          - hole.metadata.ivars_allowed_reads))
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return bool(hole.metadata.ivars_allowed
                - hole.metadata.ivars_allowed_reads)


@register_template
class IVarReadTypeError(ConfigurableTemplate):
  fills_type = HoleType.IVAR_READ
  required_cost = 0
  weight = TYPE_ERROR_WEIGHT

  def fill(self, hole, rng):
    # TODO(dbieber): This can produce NameErrors since the chosen bvar/lvar
    # may not have been assigned to. Don't allow this.
    # Same for BVarReadTypeError.
    var = rng.choice(list(hole.metadata.lvars_allowed
                          | hole.metadata.bvars_allowed))
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return bool(hole.metadata.lvars_allowed
                | hole.metadata.bvars_allowed)


@register_template
class BVarRead(ConfigurableTemplate):
  fills_type = HoleType.BVAR_READ
  required_cost = 0
  weight = 40

  def fill(self, hole, rng):
    [var] = choose_reads(hole, rng, 1, is_bvar)  # pylint: disable=unbalanced-tuple-unpacking
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return hole.metadata.can_bread


@register_template
class BVarReadNameError(ConfigurableTemplate):
  fills_type = HoleType.BVAR_READ
  required_cost = 0
  weight = NAME_ERROR_WEIGHT

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.bvars_allowed
                          - hole.metadata.bvars_allowed_reads))
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return bool(hole.metadata.bvars_allowed
                - hole.metadata.bvars_allowed_reads)


@register_template
class BVarReadTypeError(ConfigurableTemplate):
  fills_type = HoleType.BVAR_READ
  required_cost = 0
  weight = TYPE_ERROR_WEIGHT

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.lvars_allowed
                          | hole.metadata.ivars_allowed))
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return bool(hole.metadata.lvars_allowed
                | hole.metadata.ivars_allowed)


@register_template
class LVarRead(ConfigurableTemplate):
  fills_type = HoleType.LVAR_READ
  required_cost = 0
  weight = 15

  def fill(self, hole, rng):
    [var] = choose_reads(hole, rng, 1, is_lvar)  # pylint: disable=unbalanced-tuple-unpacking
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return hole.metadata.can_lread


@register_template
class LVarReadNameError(ConfigurableTemplate):
  fills_type = HoleType.LVAR_READ
  required_cost = 0
  weight = NAME_ERROR_WEIGHT

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.lvars_allowed
                          - hole.metadata.lvars_allowed_reads))
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return bool(hole.metadata.lvars_allowed
                - hole.metadata.lvars_allowed_reads)


@register_template
class LVarReadTypeError(ConfigurableTemplate):
  fills_type = HoleType.LVAR_READ
  required_cost = 0
  weight = TYPE_ERROR_WEIGHT

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.bvars_allowed
                          | hole.metadata.ivars_allowed))
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return bool(hole.metadata.bvars_allowed
                | hole.metadata.ivars_allowed)


@register_template
class VarWrite(ConfigurableTemplate):
  fills_type = HoleType.VAR_WRITE
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    # TODO(dbieber): Remove this template and just fill in the writes directly.
    var = hole.metadata.required_write
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return hole.metadata.required_write


@register_template
class IVarModify(ConfigurableTemplate):
  fills_type = HoleType.IVAR_MODIFY
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    # TODO(dbieber): Explain why we don't modify required_reads, or else
    # enable modifying required reads. Same below.
    var = rng.choice(list(hole.metadata.ivars_allowed_reads
                          - hole.metadata.required_reads))
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return hole.metadata.can_imodify


@register_template
class BVarModify(ConfigurableTemplate):
  fills_type = HoleType.BVAR_MODIFY
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.bvars_allowed_reads
                          - hole.metadata.required_reads))
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return hole.metadata.can_bmodify


@register_template
class LVarModify(ConfigurableTemplate):
  fills_type = HoleType.LVAR_MODIFY
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    var = rng.choice(list(hole.metadata.lvars_allowed_reads
                          - hole.metadata.required_reads))
    return Program(0, [], const(var))

  def can_fill(self, hole, space, templates):
    return hole.metadata.can_lmodify


@register_template
class IfStatement(ConfigurableStatementTemplate):
  fills_type = HoleType.STMT
  required_cost = 2
  weight = 10
  accesses = [Accesses.BVAR_READ, Accesses.MORE]

  def cond_metadata(self, hole):
    return hole.metadata.set(required_write=None)

  def block_metadata(self, hole):
    return hole.metadata.set(
        indent=hole.metadata.indent + 1,
    )

  def fill(self, hole, rng):
    def build(cond1, block1):
      return (
          [f'{hole.indent}if {cond1}:']
          + block1
      )
    cond_metadata = self.cond_metadata(hole)
    block_metadata = self.block_metadata(hole)
    cond = Hole(HoleType.CONDITION, cond_metadata)
    block = Hole(HoleType.BLOCK, block_metadata)
    return Program(1, [cond, block], build)

  def can_fill(self, hole, space, templates):
    cond_metadata = self.cond_metadata(hole)
    block_metadata = self.block_metadata(hole)
    return (
        exists_stmt_satisfying(block_metadata, 1, templates)
        and exists_cond_satisfying(cond_metadata, 0, templates)
    )


@register_template
class IfElseStatement(ConfigurableStatementTemplate):
  fills_type = HoleType.STMT
  required_cost = 4
  weight = 30
  accesses = [Accesses.BVAR_READ, Accesses.MORE]

  def cond_metadata(self, hole):
    return hole.metadata.set(required_write=None)

  def block_metadatas(self, hole):
    # TODO(dbieber): Currently required_writes is always only propagated to
    # the else block. Fix this.
    return hole.metadata.set(
        indent=hole.metadata.indent + 1,
    ).split()

  def fill(self, hole, rng):
    def build(cond1, block1, block2):
      return (
          [f'{hole.indent}if {cond1}:']
          + block1
          + [f'{hole.indent}else:']
          + block2
      )
    cond_metadata = self.cond_metadata(hole)
    metadata1, metadata2 = self.block_metadatas(hole)
    cond = Hole(HoleType.CONDITION, cond_metadata)
    block1 = Hole(HoleType.BLOCK, metadata1)
    block2 = Hole(HoleType.BLOCK, metadata2)
    return Program(2, [cond, block1, block2], build)

  def can_fill(self, hole, space, templates):
    cond_metadata = self.cond_metadata(hole)
    metadata1, metadata2 = self.block_metadatas(hole)
    return (
        exists_stmt_satisfying(metadata1, 1, templates)
        and exists_stmt_satisfying(metadata2, 1, templates)
        and exists_cond_satisfying(cond_metadata, 0, templates)
    )


@register_template
class WhileStatement(ConfigurableStatementTemplate):
  fills_type = HoleType.STMT
  required_cost = 2
  weight = 30
  accesses = [Accesses.BVAR_READ, Accesses.BVAR_MODIFY, Accesses.MORE]

  def cond_metadata(self, hole, var):
    return hole.metadata.set(
        required_write=None,
        required_reads={var},
    )

  def block_metadata(self, hole, var):
    return hole.metadata.set(
        indent=hole.metadata.indent + 1,
        inside_loop=True,
        required_write=var,  # TODO(dbieber): Support multiple required writes.
    )

  def fill(self, hole, rng):
    # Choosing a loop variable.
    [var] = choose_reads(hole, rng, 1, is_bvar)  # pylint: disable=unbalanced-tuple-unpacking
    # Should be read in the condition,
    # And written to in the body.
    # Do we want to require reading it in the body? No.
    def build(cond1, block1):
      return (
          [f'{hole.indent}while {cond1}:']
          + block1
      )
    cond_metadata = self.cond_metadata(hole, var)
    block_metadata = self.block_metadata(hole, var)
    cond = Hole(HoleType.CONDITION, cond_metadata)
    block = Hole(HoleType.BLOCK, block_metadata)
    return Program(1, [cond, block], build)

  def can_fill(self, hole, space, templates):
    if hole.metadata.required_write:
      return False
    var = next(iter(hole.metadata.unused_bvariables))
    return (
        exists_stmt_satisfying(self.block_metadata(hole, var), 1, templates)
        and exists_cond_satisfying(self.cond_metadata(hole, var), 0, templates)
    )


@register_template
class Block(ConfigurableTemplate):
  fills_type = HoleType.BLOCK
  required_cost = 1
  weight = 10

  def fill(self, hole, rng):
    stmts = Hole(HoleType.STMTS_NONEMPTY, hole.metadata)
    block_end = Hole(HoleType.BLOCK_END,
                     hole.metadata)
    return Program(0, [stmts, block_end], concat)


@register_template
class Continue(ConfigurableTemplate):
  fills_type = HoleType.BLOCK_END
  required_cost = 1
  weight = 5

  def can_fill(self, hole, space, templates):
    return hole.metadata.inside_loop

  def fill(self, hole, rng):
    return Program(1, [], const([f'{hole.indent}continue']))


@register_template
class Break(ConfigurableTemplate):
  fills_type = HoleType.BLOCK_END
  required_cost = 1
  weight = 5

  def can_fill(self, hole, space, templates):
    return hole.metadata.inside_loop

  def fill(self, hole, rng):
    return Program(1, [], const([f'{hole.indent}break']))


@register_template
class Fallthrough(ConfigurableTemplate):
  fills_type = HoleType.BLOCK_END
  required_cost = 0
  weight = 15

  def fill(self, hole, rng):
    return Program(0, [], const([]))


@register_template
class BAugOp(ConfigurableTemplate):
  fills_type = HoleType.BAUG_OP
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    op = rng.choice(('|=', '&=', '^='))
    return Program(0, [], const(op))


@register_template
class IBinOpComp(ConfigurableTemplate):
  fills_type = HoleType.IBIN_OP_COMP
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    ps = [1, 5, 5, 30, 20, 20]
    total = sum(ps)
    ps = [p/total for p in ps]
    op = rng.choice(('==', '>', '<', '!=', '>=', '<='), p=ps)
    return Program(0, [], const(op))


@register_template
class IAugOp(ConfigurableTemplate):
  fills_type = HoleType.IAUG_OP
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    op = rng.choice(('+=', '-=', '*=', '%=', '/='))
    return Program(0, [], const(op))


@register_template
class BConstant(ConfigurableTemplate):
  fills_type = HoleType.BCONSTANT
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    value = rng.choice((True, False))
    return Program(0, [], const(value))


@register_template
class IConstant(ConfigurableTemplate):
  fills_type = HoleType.ICONSTANT
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    max_value = _max_value(self.config)
    value = rng.randint(-max_value, max_value + 1)
    return Program(0, [], const(value))


@register_template
class LConstant(ConfigurableTemplate):
  fills_type = HoleType.LCONSTANT
  required_cost = 0
  weight = 10

  def fill(self, hole, rng):
    value = rng.choice(([], [0], [1]))
    return Program(0, [], const(value))


TEMPLATE_CLASSES = [
    ProgramTemplate,
    StatementsNonEmptySingle,
    StatementsNonEmptyMultiple,
    StatementsNonEmptyMultipleFlip,
    StatementsNonEmptyMultipleGroups,
    StatementsNonEmptyNewIVariable,
    StatementsNonEmptyNewBVariable,
    StatementsNonEmptyNewLVariable,
    IInitTemplate,
    BInitTemplate,
    LInitTemplate,
    LInitIVar,
    IVarAugOpConstant,
    IVarSqrtIVar,
    LVarAppendIVar,
    IVarPopLVar,
    LVarCopyLVar,
    IVarIndexLVarByIVar,
    IVarLenLVar,
    BVarAugOpConstant,
    IVarAugOpIVar,
    BVarAugOpBVar,
    BVarIVarInLVar,
    BVarBinOpIntVarZero,
    IVarRead,
    IVarReadNameError,
    IVarReadTypeError,
    BVarRead,
    BVarReadNameError,
    BVarReadTypeError,
    LVarRead,
    LVarReadNameError,
    LVarReadTypeError,
    IVarModify,
    BVarModify,
    LVarModify,
    VarWrite,
    AssertStatement,
    BVarCondition,
    NotBVarCondition,
    ComparatorCondition,
    IfStatement,
    IfElseStatement,
    WhileStatement,
    Block,
    Continue,
    Break,
    Fallthrough,
    BAugOp,
    IBinOpComp,
    IAugOp,
    BConstant,
    IConstant,
    LConstant,
]


# We use an explicit list of TEMPLATE_CLASSES to ensure pickling works.
# This assert verifies we haven't missed any classes from that explicit list.
assert set(TEMPLATE_CLASSES) == set(
    REGISTERED_TEMPLATES), set(TEMPLATE_CLASSES) ^ set(REGISTERED_TEMPLATES)


def exists_stmt_satisfying(metadata, space, templates):
  for weighted_template in templates:
    template = weighted_template.template
    if isinstance(template, ConfigurableStatementTemplate):
      if (template.required_cost <= space and
          template.can_fill(Hole(HoleType.STMT, metadata), space, templates)):
        return True


def exists_cond_satisfying(metadata, space, templates):
  for weighted_template in templates:
    template = weighted_template.template
    if isinstance(template, ConfigurableConditionTemplate):
      if (template.required_cost <= space and
          template.can_fill(
              Hole(HoleType.CONDITION, metadata), space, templates)):
        return True


def get_template_data(config):
  weighted_templates = []
  for template_cls in TEMPLATE_CLASSES:
    weighted_templates.append(top_down_refinement.WeightedTemplate(
        template=template_cls(config),
        weight=template_cls.weight,
        precedence=template_cls.precedence))

  return TemplateData(
      weighted_templates=weighted_templates,
      root_object=Program(
          0, [Hole(HoleType.PROGRAM, HoleMetadata())], lambda x: x),
      hole_type_weights=HOLE_TYPE_WEIGHTS,
      start_with_initialization=False,
  )
