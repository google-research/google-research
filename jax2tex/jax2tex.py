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

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""jax2tex prototype library code."""

import functools

import inspect
import operator as op
from typing import Callable, Any, Tuple, Dict, Union, List, TypeVar

import jax
from jax import core
from jax import lax
from jax import tree_util
from jax import util

from jax.abstract_arrays import ShapedArray
from jax.api import make_jaxpr
from jax.core import Primitive
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import xla

import jax.numpy as np

# pylint: disable=redefined-builtin
map = util.safe_map
zip = util.safe_zip
# pylint: enable=redefined-builtin

Array = np.ndarray
ArrayOrArrayTuple = Union[Array, Tuple[Array, ...]]
PyTree = Any
Params = Any
Shape = Tuple[int]

# API Functions.


def bind_names(fn: Callable[..., Array]) -> Callable[..., Array]:
  """Annotates a function, assigning names to function inputs and outputs."""
  fn_name = tuple(value for name, value in inspect.getmembers(fn) if
                  name == '__name__')[0]
  if fn_name == '<lambda>':
    fn_name = 'f'
  if len(fn_name) > 1:
    fn_name = '\\text{' + fn_name + '}'

  invar_names = [str(x) for x in inspect.signature(fn).parameters]
  @functools.wraps(fn)
  def named_fn(*args):
    assert len(args) == len(invar_names)

    named_args = ()
    idx = -1
    for a, name in zip(args, invar_names):
      if is_array_or_float(a):
        named_args += (tex_var(a, name, True),)
      else:
        def anonymous_var(a):
          nonlocal idx
          idx += 1
          return tex_var(a, f'\\theta^{idx}', True)
        named_args += (tree_util.tree_map(anonymous_var, a),)
    fn_out = fn(*named_args)
    return (tex_var(fn_out, fn_name, True) if is_array_or_float(fn_out)
            else fn_out)

  return named_fn


def jax2tex(fn: Callable[..., Array], *args) -> str:
  r"""Converts a function and example inputs to a LaTeX representation.

  `jax2tex` takes a jax function `fn` along with example arguments to `fn` and
  produces a string representation of latex describing the function. Here is a
  simple example of its use.

  ```
  jax2tex(lambda a, b: a + b / a, 1., 1.)
  ```

  Which will output "f &= a + {b \over a}".

  Args:
    fn: The function to be converted to latex. This function must be traceable
      by jax.
    *args: Example inputs to the fucntion. These example arguments will only be
      examined for their shape / dtype information, not their value.

  Returns:
    A string containing the latex output.
  """
  fn = bind_names(fn)
  jaxpr = make_jaxpr(fn)(*args)

  expressions = {}
  abstract = {}

  def read_shaped(v):
    if isinstance(v, core.Literal):
      if isinstance(v.val, float) or isinstance(v.val, int):
        return ShapedArray((), type(v.val))
      return ShapedArray(v.val.shape, v.val.dtype)
    else:
      return abstract[v]

  def write_shaped(v, val):
    if isinstance(v, core.Literal):
      if isinstance(v.val, float) or isinstance(v.val, int):
        val = ShapedArray((), type(v.val))
      val = ShapedArray(v.val.shape, v.val.dtype)
    abstract[v] = val

  def read_expr(v):
    if isinstance(v, core.Literal):
      return Variable(v, v.val)
    else:
      return expressions[v]

  def write_expr(v, expr):
    if isinstance(v, core.Literal):
      expr = str(expr)
    expressions[v] = expr

  args_flat, _ = tree_util.tree_flatten(args)

  invars = map(Variable, jaxpr.jaxpr.invars, args_flat)
  map(write_expr, jaxpr.jaxpr.invars, invars)
  map(write_shaped, jaxpr.jaxpr.invars, jaxpr.in_avals)

  constvars = canonicalize_consts(jaxpr.literals)
  map(write_expr,
      jaxpr.jaxpr.constvars,
      constvars)
  map(write_shaped,
      jaxpr.jaxpr.constvars,
      [ShapedArray(x.shape, x.dtype) if hasattr(x, 'shape') else
       ShapedArray((), type(x)) for x in jaxpr.literals])

  output_lines = []

  def jaxpr2tex(jaxpr):
    nonlocal output_lines
    for eqn in jaxpr.eqns:
      prim = eqn.primitive
      in_exprs = map(read_expr, eqn.invars)
      in_shaped = map(read_shaped, eqn.invars)

      if prim is xla.xla_call_p:
        call_jaxpr = eqn.params['call_jaxpr']
        map(write_expr, call_jaxpr.invars, map(read_expr, eqn.invars))
        map(write_shaped, call_jaxpr.invars, map(read_shaped, eqn.invars))
        jaxpr2tex(call_jaxpr)
        map(write_expr, eqn.outvars, map(read_expr, call_jaxpr.outvars))
        map(write_shaped, eqn.outvars, map(read_shaped, call_jaxpr.outvars))
      elif prim is jax.custom_derivatives.custom_jvp_call_jaxpr_p:
        # This is the same code (renamed) to above. Merge?
        fun_jaxpr = eqn.params['fun_jaxpr']
        assert not fun_jaxpr.literals
        fun_jaxpr = fun_jaxpr.jaxpr

        map(write_expr, fun_jaxpr.invars, map(read_expr, eqn.invars))
        map(write_shaped, fun_jaxpr.invars, map(read_shaped, eqn.invars))
        jaxpr2tex(fun_jaxpr)
        map(write_expr, eqn.outvars, map(read_expr, fun_jaxpr.outvars))
        map(write_shaped, eqn.outvars, map(read_shaped, fun_jaxpr.outvars))
      else:
        out_shaped = prim.abstract_eval(*in_shaped, **eqn.params)

        if prim.multiple_results:
          raise NotImplementedError()
        else:
          if prim is tex_var_p:
            name = eqn.params['name']
            depends_on = tuple(map(read_expr, eqn.invars[1:]))
            in_expr = read_expr(eqn.invars[0])
            potential_dependencies = get_dependencies(in_expr) + depends_on
            dependencies = ()
            for d in potential_dependencies:
              if d not in dependencies:
                dependencies += (d,)
            expr = Variable(name, out_shaped, dependencies)
            write_expr(eqn.outvars[0], expr)
            output_lines += [(eqn.outvars[0],
                              eqn.invars[0],
                              eqn.params['is_alias'])]
          else:
            write_expr(eqn.outvars[0],
                       TExpr(prim, out_shaped.shape, eqn.params,
                             eqn.invars, in_shaped, in_exprs))
          write_shaped(eqn.outvars[0], out_shaped)

  jaxpr2tex(jaxpr.jaxpr)

  def get_used_vars(var, expr):
    if isinstance(expr, Variable):
      if isinstance(var, core.Literal):
        return ()
      used_vars = (var,)
      in_var = [in_v for out_v, in_v, _ in output_lines if out_v is var]
      if in_var:
        assert len(in_var) == 1
        in_var = in_var[0]
        used_vars += get_used_vars(in_var, read_expr(in_var))
      return used_vars

    if isinstance(expr, TExpr):
      return functools.reduce(op.add, (get_used_vars(v, e) for v, e in
                                       zip(expr.in_vars, expr.in_ast_nodes)))
    raise ValueError()

  # Find variables that appear in the calculation of the outputs.
  used_vars = set(functools.reduce(op.add,
                                   tuple(get_used_vars(v, read_expr(v)) for
                                         v in jaxpr.jaxpr.outvars)))
  used_exprs = tuple(read_expr(v) for v in used_vars)

  # Go through the latex outputs and construct a string representation if they
  # have a data dependency on the output and are not supressed aliases.
  output = ''
  for out_var, in_var, alias in output_lines:

    in_expr = read_expr(in_var)
    out_expr = read_expr(out_var)

    if out_expr not in used_exprs:
      continue

    if isinstance(in_expr, Variable) and alias:
      continue

    shaped = read_shaped(out_var)
    indices = ''.join([chr(i + ord('i')) for i, s in enumerate(shaped.shape) if
                       s > 1])

    if output:
      output += '\\\\\n'
    output += f'{out_expr.bind(indices)} &= {in_expr.bind(indices)}'
  return output


def tex_var(x: Array,
            name: str,
            is_alias: bool = False,
            depends_on: ArrayOrArrayTuple = ()) -> Array:
  r"""Annotates a function with an intermediate variable in the tex expression.

  This function adds an intermediate variable to a latex expression. These
  intermediate variables transform properly under automatic differentiation or
  vectorization. Tangents are prefixed with a "d" and cotangents are prefixed
  by a \delta. For example,

  ```
    jax2tex(grad(lambda a, b: a + tex_var(b / a, 'z')), 1., 1.)
  ```

  Will return,

    z &= {b \over a}\\
    \delta z &= 1.0\\
    f &= 1.0 + -\delta z{a}^{-2}b

  Variables can have explicit dependence on other variables via the `depends_on`
  argument. It is also possible to alias variables without adding new lines to
  the final latex expression via the `is_alias` argument. `tex_var` acts as a
  no-op outside of the `jax2tex` function.

  Args:
    x: The jax variable that we are assigning to a new latex variable.
    name: A string with the latex name for the variable.
    is_alias: A boolean specifying whether or not this variable is an alias.
      Aliases do not show up in the final expression.
    depends_on: A tuple of jax variables that we the current variable depends
      on.

  Returns:
    `tex_var(x, ...)` acts like the identity and returns `x`.
  """
  if not isinstance(depends_on, tuple):
    # TODO(schsam): Better error checking here (what if depends_on is malformed
    # in some other way).
    depends_on = (depends_on,)
  return tex_var_p.bind(x, *depends_on, name=name, is_alias=is_alias)


# Implementation.


class BoundTExpr(object):
  """A Latex Expression node in the AST after index assignement."""

  def __init__(self, texpr: 'TExpr', indices: str, used_indices: str):
    """Creates a BoundTExpr from a TExpr with indices bound to characters.

    Args:
      texpr: The expression whose indices are to be bound to characters.
      indices: A string representing the character for each index.
      used_indices: A string containing characters that have already been
        assigned to indices in a parent node, but that do not appear in
        indices.
    """

    self.prim = texpr.prim
    self.shape = texpr.shape
    self.params = texpr.params
    self.in_vars = texpr.in_vars
    self.in_shaped = texpr.in_shaped
    self.indices = indices

    self.in_ast_nodes = ()

    if self.prim not in op2ind:
      raise NotImplementedError(f'No translation rule for {self.prim}.')
    in_indices = op2ind[self.prim](self.in_shaped,
                                   indices,
                                   used_indices,
                                   **self.params)
    in_used = [''.join([i for i in indices if i not in ind]) + used_indices
               for ind in in_indices]
    for ind, used, arg in zip(in_indices, in_used, texpr.in_ast_nodes):
      self.in_ast_nodes += ((arg.bind(ind, used),) if ind is not None else
                            (None,))

  def __str__(self) -> str:
    return str(op2tex[self.prim](*self.in_ast_nodes, **self.params))
  __repr__ = __str__


class TExpr(object):
  """A Latex Expression node in the AST before index assignement.

  Each TExpr node corresponds with an equation in the jaxpr. As such, each TExpr
  node wraps a jax primitive.

  Attributes:
    prim: The underlying jax primitive that this node corresponds to (e.g.
      add, mul, etc...).
    shape: The shape of the output from the primitive.
    params: The parameters passed to the primitive.
    in_vars: The input variables passed to the primitive.
    in_shaped: The shaped array inputs (containing shape and dtype) to the
      primitive.
    in_ast_nodes: The ASTNode corresponding to the input variables to the
      primitive.
  """

  def __init__(self,
               prim: core.Primitive,
               shape: Shape,
               params: Dict[str, Any],
               in_vars: List[Union[core.Literal, core.Var]],
               in_shaped: List[ShapedArray],
               in_ast_nodes: List['ASTNode']):
    self.prim = prim
    self.shape = shape
    self.params = params
    self.in_vars = in_vars
    self.in_shaped = in_shaped
    self.in_ast_nodes = in_ast_nodes

  def bind(self, out_indices: str, used_indices: str = '') -> BoundTExpr:
    """Recursively assign indices from the output to the input expression."""
    return BoundTExpr(self, out_indices, used_indices)


class BoundVariable(object):
  """A Variable node in the AST after index assignement."""

  def __init__(self,
               var: Union[str, core.Var, core.Literal],
               val: Union[float, int, Array, ShapedArray],
               depends_on: Tuple['Variable', ...],
               indices: str):
    self.var = var
    self.val = val
    self.shape = self.val.shape if hasattr(self.val, 'shape') else ()
    self.depends_on = depends_on
    self.indices = indices

  def __str__(self):
    suffix = ''

    if self.depends_on:
      suffix += '('
      for x in self.depends_on[:-1]:
        suffix += f'{x},'
      suffix += f'{self.depends_on[-1]})'

    if isinstance(self.val, float) or isinstance(self.val, int):
      return str(self.var) + suffix
    elif (isinstance(self.val, np.ndarray) or
          isinstance(self.val, ShapedArray)):
      if not self.val.shape:
        return str(self.var) + suffix
      return str(self.var) + '_{' + self.indices + '}' + suffix
    else:
      raise NotImplementedError()
  __repr__ = __str__


class Variable(object):
  """A Variable node in the AST before index assignement.

  Each Variable node corresponds to a jax variable.

  Attributes:
    var: The undarying jax variable or a string name for the variable.
    val: Either a float, int, or ndarray specifying the value of the variable or
      a ShapedArray.
    shape: The shape of `val` if it is in array or `()`.
    depends_on: A tuple of Variable nodes that this variable depends on.
  """

  def __init__(self,
               var: Union[str, core.Var, core.Literal],
               val: Union[float, int, Array, ShapedArray],
               depends_on: Tuple['Variable', ...] = ()):
    self.var = var
    self.val = val
    self.shape = self.val.shape if hasattr(self.val, 'shape') else ()
    self.depends_on = depends_on
    if var is core.Literal and abs(int(self.val) - self.val) < 1e-7:
      self.val = int(val)

  def bind(self, indices: str, _: str = '') -> BoundVariable:
    nontrivial_indices = ''
    if indices:
      skipped_indices = 0
      for i, s in enumerate(self.shape):
        if s > 1:
          nontrivial_indices += indices[i - skipped_indices]
        else:
          skipped_indices += 1
    return BoundVariable(
        self.var, self.val, self.depends_on, nontrivial_indices)

  def __str__(self):
    return str(self.var)
  __repr__ = __str__


T = TypeVar('T')
MaybeEmptyTuple = Union[Tuple[T, ...], Tuple[()]]

ASTNode = Union[Variable, TExpr]
BoundASTNode = Union[BoundVariable, BoundTExpr]


def is_array_or_float(x: Array) -> bool:
  return (isinstance(x, np.ndarray) or
          isinstance(x, float) or
          isinstance(x, int))


def canonicalize_consts(literals: List[Array]) -> MaybeEmptyTuple[Variable]:
  """Formats constants based on their shape, dtype, and value."""
  constvars = ()
  namedconstvars = 0
  for x in literals:
    square_size = 0

    # If constant is a scalar then the variable will just be the value of the
    # scalar.
    if isinstance(x, float) or isinstance(x, int):
      constvars += (Variable(x, x),)
      continue

    s_x = np.squeeze(x)

    for i in range(s_x.ndim - 1):
      if x.shape[i] == s_x.shape[i + 1] and s_x.shape[i] > square_size:
        square_size = s_x.shape[i]

    # If the constant is a matrix that is close to the identity replace it with
    # the identity symbol. Otherwise, we use capital letters starting at C.
    if square_size >= 1 and np.allclose(s_x, np.eye(square_size)):
      constvars += (Variable('\\mathbf I', x),)
    else:
      constvars += (Variable(chr(namedconstvars + ord('C')), x),)
      namedconstvars += 1
  return constvars


def get_dependencies(expr: ASTNode) -> MaybeEmptyTuple[Variable]:
  if isinstance(expr, Variable):
    return expr.depends_on

  dependencies = ()
  for e in expr.in_ast_nodes:
    dependencies += get_dependencies(e)
  return dependencies


# Translation Rules


op2tex = {}
op2ind = {}


def broadcast2ind(
    in_shaped: Tuple[ShapedArray, ...], out_indices: str, _: str) -> Tuple[str]:
  max_ndim = max(len([ss for ss in s.shape if ss > 1]) for s in in_shaped)
  assert max_ndim == len(out_indices)
  implicit_dims = tuple((max_ndim - len(s.shape)) for s in in_shaped)
  in_shaped = tuple((1,) * (max_ndim - len(s.shape)) + s.shape for
                    s in in_shaped)
  in_indices = tuple(''.join(out_indices[i] for i, ss in enumerate(s) if ss > 1)
                     for impl, s in zip(implicit_dims, in_shaped))
  return in_indices


def get_free_index(char: int, used: str) -> int:
  while chr(char) in used:
    char += 1
  return char


def add2tex(a: BoundASTNode, b: BoundASTNode) -> str:
  return f'{a} + {b}'
op2tex[lax.add_p] = add2tex
op2ind[lax.add_p] = broadcast2ind
op2tex[ad.add_jaxvals_p] = add2tex
op2ind[ad.add_jaxvals_p] = broadcast2ind


def sub2tex(a: BoundASTNode, b: BoundASTNode) -> str:
  return f'{a} - {b}'
op2tex[lax.sub_p] = sub2tex
op2ind[lax.sub_p] = broadcast2ind


def mul2tex(a: BoundASTNode, b: BoundASTNode) -> str:
  if hasattr(a, 'prim') and a.prim in (lax.add_p, lax.sub_p):
    a = f'\\left({a}\\right)'
  if hasattr(b, 'prim') and b.prim in (lax.add_p, lax.sub_p):
    b = f'\\left({b}\\right)'
  return f'{a}{b}'
op2tex[lax.mul_p] = mul2tex
op2ind[lax.mul_p] = broadcast2ind


def div2tex(a: BoundASTNode, b: BoundASTNode) -> str:
  return '{' + f'{a} \\over {b}' + '}'
op2tex[lax.div_p] = div2tex
op2ind[lax.div_p] = broadcast2ind


def dot_general2tex(a: BoundASTNode,
                    b: BoundASTNode,
                    dimension_numbers: lax.DotDimensionNumbers,
                    precision) -> str:
  """Converts dot_general op to latex."""
  del precision

  ((a_contract_dims, _),
   (a_broadcast_dims, _)) = dimension_numbers

  if a_broadcast_dims:
    raise NotImplementedError()

  cindices = ''
  skipped_indices = 0
  for i, s in enumerate(a.shape):
    if s <= 1:
      skipped_indices += 1
      continue

    if i in a_contract_dims:
      cindices += a.indices[i - skipped_indices]

  if hasattr(a, 'prim') and a.prim in (lax.add_p, lax.sub_p):
    a = f'\\left({a}\\right)'
  if hasattr(b, 'prim') and b.prim in (lax.add_p, lax.sub_p):
    b = f'\\left({b}\\right)'
  sum_str = '\\sum_{' + cindices +  '}' if cindices else ''
  return f'{sum_str}{a}{b}'
op2tex[lax.dot_general_p] = dot_general2tex


def dot_general2ind(in_shaped: Tuple[ShapedArray, ...],
                    out_indices: str,
                    out_used: str,
                    dimension_numbers: lax.DotDimensionNumbers,
                    precision) -> Tuple[str, ...]:
  """Computes indices of inputs given indices of outputs for dot_general."""
  del precision

  ((a_contract_dims, b_contract_dims), _) = dimension_numbers
  a, b = in_shaped

  a_indices = ''
  b_indices = ''

  cur_out_idx = 0
  offset = max(ord(x) + 1 for x in out_indices) if out_indices else ord('i')
  for i, s in enumerate(a.shape):
    if s <= 1:
      continue
    if i in a_contract_dims:
      offset = get_free_index(offset, out_used)
      a_indices += chr(offset)
      offset += 1
    else:
      a_indices += out_indices[cur_out_idx]
      cur_out_idx += 1

  offset = max(ord(x) + 1 for x in out_indices) if out_indices else ord('i')
  for i, s in enumerate(b.shape):
    if s <= 1:
      continue
    if i in b_contract_dims:
      offset = get_free_index(offset, out_used)
      b_indices += chr(offset)
      offset += 1
    else:
      b_indices += out_indices[cur_out_idx]
      cur_out_idx += 1

  return a_indices, b_indices
op2ind[lax.dot_general_p] = dot_general2ind


def reduce_sum2tex(x: BoundASTNode, axes: Tuple[int, ...]) -> str:
  cindices = ''
  for i, s in enumerate(x.indices):
    if i in axes:
      cindices += s

  if hasattr(x, 'prim') and x.prim in (lax.add_p, lax.sub_p):
    x = f'\\left({x}\\right)'
  sum_str = '\\sum_{' + cindices +  '}' if cindices else ''
  return f'{sum_str}{x}'
op2tex[lax.reduce_sum_p] = reduce_sum2tex


def reduce_sum2ind(in_shaped: Tuple[ShapedArray, ...],
                   out_indices: str,
                   out_used: str,
                   axes: Tuple[int, ...]) -> Tuple[str, ...]:
  """Computes indices of inputs given indices of outputs for reduce_sum."""
  x, = in_shaped

  x_indices = ''

  cur_out_idx = 0
  offset = ord(out_indices[-1]) + 1 if out_indices else ord('i')
  for i, s in enumerate(x.shape):
    if s <= 1:
      continue
    if i in axes:
      offset = get_free_index(offset, out_used)
      x_indices += chr(offset)
      offset += 1
    else:
      x_indices += out_indices[cur_out_idx]
      cur_out_idx += 1
  return x_indices,
op2ind[lax.reduce_sum_p] = reduce_sum2ind

noop2tex = lambda x, **kwargs: x
noop2ind = lambda in_shaped, out_indices, out_used, **kwargs: [out_indices]

op2tex[lax.convert_element_type_p] = noop2tex
op2ind[lax.convert_element_type_p] = noop2ind
op2tex[xla.device_put_p] = noop2tex
op2ind[xla.device_put_p] = noop2ind
op2tex[jax.ad_util.stop_gradient_p] = noop2tex
op2ind[jax.ad_util.stop_gradient_p] = noop2ind

if hasattr(lax.lax, 'tie_in_p'):
  tie_in2tex = lambda x, y: y
  tie_in2ind = lambda in_shaped, out_indices, out_used: (None, out_indices)
  op2tex[lax.lax.tie_in_p] = tie_in2tex
  op2ind[lax.lax.tie_in_p] = tie_in2ind


op2tex[lax.sqrt_p] = lambda x: '\\sqrt{' + str(x) + '}'
op2ind[lax.sqrt_p] = noop2ind

op2tex[lax.lt_p] = lambda a, b: f'{a}<{b}'
op2ind[lax.lt_p] = broadcast2ind

op2tex[lax.gt_p] = lambda a, b: f'{a}>{b}'
op2ind[lax.gt_p] = broadcast2ind

op2tex[lax.le_p] = lambda a, b: f'{a}\\leq{b}'
op2ind[lax.le_p] = broadcast2ind

op2tex[lax.ge_p] = lambda a, b: f'{a}\\geq{b}'
op2ind[lax.ge_p] = broadcast2ind

op2tex[lax.and_p] = lambda a, b: f'{a}\\wedge{b}'
op2ind[lax.and_p] = broadcast2ind

op2tex[lax.not_p] = lambda x: f'\\lnot{x}'
op2ind[lax.not_p] = noop2ind

op2tex[lax.tanh_p] = lambda x: f'\\tanh\\left({x}\\right)'
op2ind[lax.tanh_p] = noop2ind
op2tex[lax.cosh_p] = lambda x: f'\\cosh\\left({x}\\right)'
op2ind[lax.cosh_p] = noop2ind
op2tex[lax.sinh_p] = lambda x: f'\\sinh\\left({x}\\right)'
op2ind[lax.sinh_p] = noop2ind

op2tex[lax.cos_p] = lambda x: f'\\cos\\left({x}\\right)'
op2ind[lax.cos_p] = noop2ind
op2tex[lax.sin_p] = lambda x: f'\\sin\\left({x}\\right)'
op2ind[lax.sin_p] = noop2ind


def reduce_max2tex(x: BoundASTNode, axes: Tuple[int, ...]) -> str:
  mindices = ''
  skipped_indices = 0
  for i, s in enumerate(x.shape):
    if s <= 1:
      skipped_indices += 1
      continue
    if i in axes:
      mindices += x.indices[i - skipped_indices]
  return '\\max_{' + mindices + '}\\left\\{' + str(x) + '\\right\\}'
op2tex[lax.reduce_max_p] = reduce_max2tex


def reduce_max2ind(in_shaped: Tuple[ShapedArray, ...],
                   out_indices: str,
                   out_used: str,
                   axes: Tuple[int, ...]) -> Tuple[str, ...]:
  """Computes indices of inputs given indices of outputs for reduce_max."""
  x, = in_shaped

  skip_count = 0
  x_indices = ''
  offset = max(ord(a) + 1 for a in out_indices) if out_indices else ord('i')
  for i, s in enumerate(x.shape):
    if s >= 1:
      if i not in axes:
        x_indices += out_indices[i - skip_count]
      else:
        offset = get_free_index(offset, out_used)
        x_indices += chr(offset)
        skip_count += 1
        offset += 1
    else:
      skip_count += 1
  return x_indices,
op2ind[lax.reduce_max_p] = reduce_max2ind


def broadcast_in_dim2ind(in_shaped: Tuple[ShapedArray, ...],
                         out_indices: str,
                         out_used: str,
                         shape: Shape,
                         broadcast_dimensions) -> Tuple[str, ...]:
  """Computes indices of inputs given outputs for broadcast_in_dim."""
  del in_shaped, out_used

  skipped_index_count = 0
  skipped_indices = ()
  for s in shape:
    skipped_indices += (skipped_index_count,)
    if s == 1:
      skipped_index_count += 1

  in_indices = ''
  for i in broadcast_dimensions:
    if shape[i] > 1:
      in_indices += out_indices[i - skipped_indices[i]]
  return (in_indices,)
op2tex[lax.broadcast_in_dim_p] = noop2tex
op2ind[lax.broadcast_in_dim_p] = broadcast_in_dim2ind


def select2tex(pred: BoundASTNode, x: BoundASTNode, y: BoundASTNode) -> str:
  one = '\\mathbbm 1_{' + str(pred) +'}'
  return f'{one}{x} + \\left(1 - {one}\\right){y}'
op2tex[lax.select_p] = select2tex
op2ind[lax.select_p] = broadcast2ind


def neg2tex(a: BoundASTNode) -> str:
  if hasattr(a, 'prim') and a.prim in (lax.add_p, lax.sub_p):
    a = f'\\left({a}\\right)'
  return f'-{a}'
op2tex[lax.neg_p] = neg2tex
op2ind[lax.neg_p] = noop2ind


def abs2tex(a: BoundASTNode) -> str:
  return f'\\left|{a}\\right|'
op2tex[lax.abs_p] = abs2tex
op2ind[lax.abs_p] = noop2ind


def integer_pow2tex(a: BoundASTNode, y: int) -> str:
  if hasattr(a, 'prim'):
    a = f'\\left({a}\\right)'
  return '{' + str(a) + '}^{' + str(y) + '}'
op2tex[lax.integer_pow_p] = integer_pow2tex
op2ind[lax.integer_pow_p] = noop2ind


def pow2tex(a: BoundASTNode, b: BoundASTNode) -> str:
  if hasattr(a, 'prim'):
    a = f'\\left({a}\\right)'
  return '{' + str(a) + '}^{' + str(b) + '}'
op2tex[lax.pow_p] = pow2tex
op2ind[lax.pow_p] = broadcast2ind


def eq2tex(a: BoundASTNode, b: BoundASTNode) -> str:
  return f'{a}={b}'
op2tex[lax.eq_p] = eq2tex
op2ind[lax.eq_p] = broadcast2ind


def neq2tex(a: BoundASTNode, b: BoundASTNode) -> str:
  return f'{a}\\neq{b}'
op2tex[lax.ne_p] = neq2tex
op2ind[lax.ne_p] = broadcast2ind

op2tex[lax.exp_p] = lambda x: 'e^{' + str(x) + '}'
op2ind[lax.exp_p] = noop2ind

op2tex[lax.log_p] = lambda x: f'\\log\\left({x}\\right)'
op2ind[lax.log_p] = noop2ind


def is_finite2tex(x: BoundASTNode) -> str:
  return f'{x}<\\infty'
op2tex[lax.is_finite_p] = is_finite2tex
op2ind[lax.is_finite_p] = noop2ind


def max2tex(a: BoundASTNode, b: BoundASTNode) -> str:
  if isinstance(b, BoundVariable) and np.allclose(b.val, 0.):
    return '\\text{relu}' + f'({a})'
  return f'\\max({a},{b})'
op2tex[lax.max_p] = max2tex
op2ind[lax.max_p] = broadcast2ind


def min2tex(a: BoundASTNode, b: BoundASTNode) -> str:
  return f'\\min({a},{b})'
op2tex[lax.min_p] = min2tex
op2ind[lax.min_p] = broadcast2ind


def transpose2tex(a: BoundASTNode, permutation: Tuple[int, ...]) -> str:
  del permutation
  return str(a)
op2tex[lax.transpose_p] = transpose2tex


def transpose2ind(in_shaped: Tuple[ShapedArray],
                  out_indices: str,
                  out_used: str,
                  permutation: Tuple[int, ...]) -> Tuple[str, ...]:
  """Computes indices of inputs given indices of outputs for transpose."""
  del out_used

  x, = in_shaped
  indices = ''
  ignored_axes_so_far = 0
  ignored_axes = ()
  for s in x.shape:
    if s <= 1:
      ignored_axes_so_far += 1
    ignored_axes += (ignored_axes_so_far,)
  for p in permutation:
    if x.shape[p] > 1:
      indices += out_indices[p - ignored_axes[p]]
  return indices,
op2ind[lax.transpose_p] = transpose2ind


def reshape2tex(a: BoundASTNode, new_sizes, dimensions) -> str:
  del new_sizes, dimensions
  return str(a)
op2tex[lax.reshape_p] = reshape2tex


def reshape2ind(in_shaped: Tuple[ShapedArray, ...],
                out_indices: str,
                new_sizes,
                dimensions) -> Tuple[str, ...]:
  """Computes indices of inputs given indices of outputs for reshape."""
  x, = in_shaped
  assert not dimensions

  # Right now we only support trivial reshapes in the sense that it only either
  # adds or removes singleton dimensions.
  i_old = 0
  i_new = 0
  while i_old < len(x.shape) and i_new < len(new_sizes):
    if x.shape[i_old] <= 1 and new_sizes[i_new] > 1:
      i_old += 1
    elif x.shape[i_old] > 1 and new_sizes[i_new] <= 1:
      i_new += 1
    else:
      assert x.shape[i_old] == new_sizes[i_new]
      i_old += 1
      i_new += 1

  return out_indices,
op2ind[lax.reshape_p] = reshape2ind


# Custom Primitive


tex_var_p = Primitive('tex_var')
tex_var_p.def_impl(lambda x, *depends_on, **kwargs: x)
tex_var_p.def_abstract_eval(lambda x, *depends_on, **kwargs: x)


def tex_var_jvp(primals, tangents, name, is_alias):
  primal_out = tex_var(primals[0], name, is_alias, primals[1:])
  tangent_out = tex_var(tangents[0], 'd' + name, False, primals[1:])
  return primal_out, tangent_out
ad.primitive_jvps[tex_var_p] = tex_var_jvp


def tex_var_transpose(ct, x, *depends_on, **params):
  del x
  ct = tex_var(ct, '\\delta ' + params['name'][1:], False, depends_on)
  return (ct,) + (ad.Zero,) * len(depends_on)
ad.primitive_transposes[tex_var_p] = tex_var_transpose


def tex_var_batcher(x, batch_dims, name, is_alias):
  return tex_var(x[0], name, is_alias, x[1:]), batch_dims[0]
batching.primitive_batchers[tex_var_p] = tex_var_batcher
