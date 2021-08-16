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

"""Tests for jax2tex."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.api import grad
from jax.api import jvp
from jax.config import config
from jax.experimental import stax
import jax.numpy as np
from jax.tree_util import tree_map

from jax2tex import bind_names
from jax2tex import jax2tex
from jax2tex import tex_var

config.enable_omnistaging()


# pylint: disable=invalid-name
# pylint: disable=g-long-lambda


class Shape(tuple):
  pass
S = Shape
Scalar = S(())


class BoolShape(tuple):
  pass
BS = BoolShape

Jax2TexExample = collections.namedtuple('Jax2TexExample',
                                        ['fn', 'shape', 'expected'])

EXAMPLES = [
    # EX 0
    Jax2TexExample(lambda a, b: a + b, (Scalar, Scalar), 'f &= a + b'),
    # EX 1
    Jax2TexExample(lambda a, b: a + b / a,
                   (Scalar, Scalar),
                   'f &= a + {b \\over a}'),
    # EX 2
    Jax2TexExample(grad(lambda a, b: a + b / a),
                   (Scalar, Scalar),
                   'f &= 1.0 + -1.0{a}^{-2}b'),
    # EX 3
    Jax2TexExample(lambda a, b, c: a + a * (b + c) / a,
                   (S((3, 2)), Scalar, Scalar),
                   ('f_{ij} &= a_{ij} + {a_{ij}\\left(b + c\\right) '
                    '\\over a_{ij}}')),
    # EX 4
    Jax2TexExample(grad(lambda a, b, c: a + a * (b + c)),
                   (Scalar, Scalar, Scalar),
                   'f &= 1.0 + 1.0\\left(b + c\\right)'),
    # EX 5
    Jax2TexExample(grad(lambda a, b: a * (a - b) / (a + b) + b),
                   (Scalar, Scalar),
                   ('f &= -1.0{\\left(a + b\\right)}^{-2}a\\left(a - b\\right) '
                    '+ {1.0 \\over a + b}\\left(a - b\\right) + '
                    'a{1.0 \\over a + b}')),
    # EX 6
    Jax2TexExample(lambda W, x: W @ x,
                   (S((3, 2)), S((2,))),
                   'f_{i} &= \\sum_{j}W_{ij}x_{j}'),
    # EX 7
    Jax2TexExample(lambda W, x: W @ x,
                   (S((3, 2)), S((2, 3))),
                   'f_{ij} &= \\sum_{k}W_{ik}x_{kj}'),
    # EX 8
    Jax2TexExample(lambda W, x: (W + W) @ (x * x),
                   (S((3, 2)), S((2, 3))),
                   ('f_{ij} &= \\sum_{k}\\left(W_{ik} + W_{ik}\\right)'
                    'x_{kj}x_{kj}')),
    # EX 9
    Jax2TexExample(grad(lambda W, x: (W + W) @ (x * x)),
                   (S((2,)), S((2,))),
                   'f_{i} &= 1.0x_{i}x_{i} + 1.0x_{i}x_{i}'),
    # EX 10
    Jax2TexExample(lambda x: lax.broadcast_in_dim(x, (2, 3), (1,)),
                   (S((3,)),),
                   'f_{ij} &= x_{j}'),
    # EX 11
    # pylint: disable=unnecessary-lambda
    Jax2TexExample(lambda c, x, y: np.where(c, x, y),
                   (BS((3,)), S((3,)), S((3,))),
                   ('f_{i} &= \\mathbbm 1_{c_{i}}x_{i} + \\left(1 - '
                    '\\mathbbm 1_{c_{i}}\\right)y_{i}')),
    # EX 12
    Jax2TexExample(lambda c, x, y: np.where(c, x, y),
                   (BS(()), S((3,)), S((3,))),
                   ('f_{i} &= \\mathbbm 1_{c}x_{i} + \\left(1 - \\mathbbm '
                    '1_{c}\\right)y_{i}')),
    # EX 13
    Jax2TexExample(lambda x: np.transpose(x),
                   (S((3, 2)),),
                   ('f_{ij} &= x_{ji}')),
    # EX 14
    Jax2TexExample(grad(bind_names(lambda x: np.sin(x))), (S(),),
                   ('\\delta f &= 1.0\\\\\n\\delta x &= \\delta f'
                    '\\cos\\left(x\\right)')),
]


def get_fwd_vs_rev_fns():
  f_ = lambda a, b: a + tex_var(b / a, 'z')
  jvp_fn_ = lambda a, b: jvp(lambda a: f_(a, b), (a,), (1.,))[1]
  return f_, jvp_fn_
f, jvp_fn = get_fwd_vs_rev_fns()
EXAMPLES += [
    # EX 15
    Jax2TexExample(f,
                   (Scalar, Scalar),
                   'z &= {b \\over a}\\\\\nf &= a + z'),
    # EX 16
    Jax2TexExample(grad(f), (Scalar, Scalar),
                   ('\\delta z &= 1.0\\\\\nf &= 1.0 + '
                    '-\\delta z{a}^{-2}b')),
    # EX 17
    Jax2TexExample(jvp_fn,
                   (Scalar, Scalar),
                   ('dz &= \\left(-1.0\\right)b{a}^{-2}\\\\\n'
                    'f &= 1.0 + dz')),
]


def get_dep_fns():
  def q(x, y):
    def g(r):
      return tex_var(r ** 2, 'z', depends_on=r)
    return g(x) + g(y)
  return q
f = get_dep_fns()
EXAMPLES += [
    # EX 18
    Jax2TexExample(f,
                   (Scalar, Scalar),
                   ('z(x) &= {x}^{2}\\\\\nz(y) &= {y}^{2}\\\\\n'
                    'q(x,y) &= z(x) + z(y)'))
]

EXAMPLES += [
    # EX 19
    Jax2TexExample(lambda x_and_y: x_and_y[0] * x_and_y[1],
                   ((Scalar, Scalar),),
                   'f &= \\theta^0\\theta^1'),
    # EX 20
    Jax2TexExample(lambda x_and_y:
                   tex_var(x_and_y[0], 'x') * tex_var(x_and_y[1], 'y'),
                   ((Scalar, Scalar),),
                   'x &= \\theta^0\\\\\ny &= \\theta^1\\\\\nf &= xy'),
    # EX 21
    Jax2TexExample(
        lambda x_and_y:
        tex_var(x_and_y[0], 'x', True) * tex_var(x_and_y[1], 'y', True),
        ((Scalar, Scalar),),
        'f &= xy'),
]


def get_ex22_fn():
  def q(W, x):
    z = tex_var(W @ x, 'z')
    return z * z
  return q
EXAMPLES += [
    # EX 22
    Jax2TexExample(get_ex22_fn(), (S((4, 2)), S((2,))),
                   ('z_{i} &= \\sum_{j}W_{ij}x_{j}\\\\\n'
                    'q_{i} &= z_{i}z_{i}'))
]


def get_ex23_fn():
  def q(W, x):
    z1 = tex_var(W @ x, 'z^1')
    z2 = tex_var(W @ z1, 'z^2')
    return z2 @ z2
  return q
EXAMPLES += [
    # EX 23
    Jax2TexExample(grad(get_ex23_fn()), (S((2, 2)), S((2,))),
                   ('z^1_{i} &= \\sum_{j}W_{ij}x_{j}\\\\\nz^2_{i} &= '
                    '\\sum_{j}W_{ij}z^1_{j}\\\\\n\\delta z^2_{i} &= 1.0z^2_{i}'
                    ' + 1.0z^2_{i}\\\\\n\\delta z^1_{i} &= \\sum_{j}\\delta '
                    'z^2_{j}W_{ji}\\\\\nq_{ij} &= \\delta z^1_{i}x_{j}'
                    ' + \\delta z^2_{i}z^1_{j}'))
]


def get_ex24_fn():
  def q(W, x):
    z1 = tex_var(W @ x, 'z^1')
    z2 = tex_var(W @ z1, 'z^2')
    return np.sqrt(z2 @ z2)
  return q
EXAMPLES += [
    # EX 24
    Jax2TexExample(get_ex24_fn(), (S((2, 2)), S((2,))),
                   ('z^1_{i} &= \\sum_{j}W_{ij}x_{j}\\\\\nz^2_{i} &= '
                    '\\sum_{j}W_{ij}z^1_{j}\\\\\nq &= \\sqrt{'
                    '\\sum_{i}z^2_{i}z^2_{i}}'))
]


def get_ex25_fn():
  def f_(dr):
    idr = (tex_var(1, '\\sigma') / dr)
    idr6 = idr ** 6
    idr12 = idr ** 12
    return tex_var(4 * tex_var(1, '\\epsilon') * (idr12 - idr6), 'E')
  return f_
EXAMPLES += [
    # EX 25
    Jax2TexExample(get_ex25_fn(), (S((2, 2)),),
                   ('\\sigma &= 1\\\\\n\\epsilon &= 1\\\\\nE_{ij} &= \\epsilon'
                    '4\\left({\\left({\\sigma \\over dr_{ij}}\\right)}^{12} - '
                    '{\\left({\\sigma \\over dr_{ij}}\\right)}^{6}\\right)'))
]


# Begin Stax Tests


# Utility for the stax based test.


def TexVar(layer, name, param_names=(), explicit_depends=False):
  init, apply = layer
  def tex_apply(params, xs, **kwargs):
    if param_names:
      assert len(param_names) == len(params)
      params = tuple(tex_var(p, name, True) for
                     p, name in zip(params, param_names))
    return tex_var(apply(params, xs, **kwargs),
                   name,
                   depends_on=xs if explicit_depends else ())
  return init, tex_apply


def get_f_and_L():
  _, apply = stax.serial(TexVar(stax.Dense(256), 'z^1', ('W^1', 'b^1')),
                         TexVar(stax.Relu, 'y^1'),
                         TexVar(stax.Dense(3), 'z^2', ('W^2', 'b^2')))
  def f_(params, x):
    return apply(params, tex_var(x, 'x', True))

  def L_(params, x, y_hat):
    y_hat = tex_var(y_hat, '\\hat y', True)
    return tex_var(-np.sum(y_hat * jax.nn.log_softmax(f_(params, x))), 'L')
  return f_, L_
f, L = get_f_and_L()
shaped_params = ((S((5, 256)), S((256,))), (), (S((256, 3)), S((3,))))

EXAMPLES += [
    # EX 26
    Jax2TexExample(f, (shaped_params, S((3, 5))),
                   ('z^1_{ij} &= \\sum_{k}x_{ik}W^1_{kj} + b^1_{j}\\\\\n'
                    'y^1_{ij} &= \\text{relu}(z^1_{ij})\\\\\nz^2_{ij} &= '
                    '\\sum_{k}y^1_{ik}W^2_{kj} + b^2_{j}')),
    # EX 27
    Jax2TexExample(L, (shaped_params, S((3, 5)), S((3, 3))),
                   ('z^1_{ij} &= \\sum_{k}x_{ik}W^1_{kj} + b^1_{j}\\\\\n'
                    'y^1_{ij} &= \\text{relu}(z^1_{ij})\\\\\nz^2_{ij} &= '
                    '\\sum_{k}y^1_{ik}W^2_{kj} + b^2_{j}\\\\\nL &= -\\sum_{ij}'
                    '\\hat y_{ij}\\left(z^2_{ij} - \\max_{k}\\left\\{z^2_{ik}'
                    '\\right\\} - \\log\\left(\\sum_{k}e^{z^2_{ik} - \\max_{l}'
                    '\\left\\{z^2_{il}\\right\\}}\\right)\\right)')),
    # EX 28
    Jax2TexExample(
        grad(L), (shaped_params, S((3, 5)), S((3, 3))),
        ('z^1_{ij} &= \\sum_{k}x_{ik}W^1_{kj} + b^1_{j}\\\\\n'
         'y^1_{ij} &= \\text{relu}(z^1_{ij})\\\\\n'
         'z^2_{ij} &= \\sum_{k}y^1_{ik}W^2_{kj} + b^2_{j}\\\\\n'
         '\\delta L &= 1.0\\\\\n'
         '\\delta z^2_{ij} &= \\hat y_{ij}\\left(-\\delta L\\right)'
         ' + \\left({\\sum_{k}-\\hat y_{ik}\\left(-\\delta L\\right)'
         ' \\over \\sum_{k}e^{z^2_{ik} - '
         '\\max_{l}\\left\\{z^2_{il}\\right\\}}'
         '}\\right)e^{z^2_{ij} - \\max_{k}\\left\\{z^2_{ik}\\right'
         '\\}}\\\\\n'
         '\\delta b^2_{i} &= \\sum_{i}\\sum_{j}\\delta z^2_{ji}\\\\\n'
         '\\delta W^2_{ij} &= \\sum_{k}\\delta z^2_{kj}y^1_{ki}\\\\\n'
         '\\delta y^1_{ij} &= \\sum_{k}\\delta z^2_{ik}W^2_{jk}\\\\\n'
         '\\delta z^1_{ij} &= \\mathbbm 1_{z^1_{ij}>0.0}\\delta '
         'y^1_{ij} + '
         '\\left(1 - \\mathbbm 1_{z^1_{ij}>0.0}\\right)0.0\\\\\n'
         '\\delta b^1_{i} &= \\sum_{i}\\sum_{j}\\delta z^1_{ji}\\\\\n'
         '\\delta W^1_{ij} &= \\sum_{k}\\delta z^1_{kj}x_{ki}'))
]


def get_f_and_nngp():
  _, apply = stax.serial(
      TexVar(stax.Dense(256), 'z^1', ('W^1', 'b^1'), True),
      TexVar(stax.Relu, 'y^1'),
      TexVar(stax.Dense(3), 'z^2', ('W^2', 'b^2')))
  def f_(params, x):
    return apply(params, tex_var(x, 'x', True))
  def nngp_(params, x1, x2):
    x1 = tex_var(x1, 'x^1', True)
    x2 = tex_var(x2, 'x^2', True)
    return tex_var(apply(params, x1) @ apply(params, x2).T, '\\mathcal K')
  return f_, nngp_
f, nngp = get_f_and_nngp()


EXAMPLES += [
    # EX 29
    Jax2TexExample(f, (shaped_params, S((3, 5))),
                   ('z^1_{ij}(x) &= \\sum_{k}x_{ik}W^1_{kj} + b^1_{j}\\\\\n'
                    'y^1_{ij}(x) &= \\text{relu}(z^1_{ij}(x))\\\\\nz^2_{ij}(x) '
                    '&= \\sum_{k}y^1_{ik}(x)W^2_{kj} + b^2_{j}')),
    # EX 30
    Jax2TexExample(nngp, (shaped_params, S((3, 5)), S((3, 5))),
                   ('z^1_{ij}(x^1) &= \\sum_{k}x^1_{ik}W^1_{kj} + b^1_{j}\\\\\n'
                    'y^1_{ij}(x^1) &= \\text{relu}(z^1_{ij}(x^1))\\\\\n'
                    'z^2_{ij}(x^1) &= \\sum_{k}y^1_{ik}(x^1)W^2_{kj} + b^2_{j}'
                    '\\\\\nz^1_{ij}(x^2) &= \\sum_{k}x^2_{ik}W^1_{kj} + b^1_{j}'
                    '\\\\\ny^1_{ij}(x^2) &= \\text{relu}(z^1_{ij}(x^2))\\\\\n'
                    'z^2_{ij}(x^2) &= \\sum_{k}y^1_{ik}(x^2)W^2_{kj} + b^2_{j}'
                    '\\\\\n\\mathcal K_{ij}(x^1,x^2) &= \\sum_{k}z^2_{ik}(x^1)'
                    'z^2_{jk}(x^2)')),
]


# End Stax Tests


class Jax2TexTest(parameterized.TestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(
      {
          'testcase_name': f'_{i}',
          'fn': fn,
          'shape': shape,
          'expected': expected
      }
      for i, (fn, shape, expected) in enumerate(EXAMPLES)
  )
  def test_jax2tex(self, fn, shape, expected):
    ex_inputs = tree_map(lambda s: np.ones(s) if isinstance(s, Shape) else
                         np.ones(s, bool), shape)
    out = jax2tex(fn, *ex_inputs)
    msg = f'Incorrect output found.\nActual:\n{out}\nExpected:\n{expected}\n'
    self.assertEqual(out, expected, msg=msg)


if __name__ == '__main__':
  absltest.main()
