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

"""jax2tex examples."""

from absl import app
from absl import flags

import jax
from jax import grad
from jax import jvp
from jax import lax
from jax import random
from jax.config import config
from jax.experimental import stax

import jax.numpy as np

from jax2tex import bind_names
from jax2tex import jax2tex
from jax2tex import tex_var

config.enable_omnistaging()
FLAGS = flags.FLAGS

EXAMPLE_ID = 1


def print_ex(tex):
  global EXAMPLE_ID
  print(f'\033[32mEx {EXAMPLE_ID}:\033[00m')
  print(tex + '\n')
  EXAMPLE_ID += 1


def main(unused_argv):
  # EX 1
  print_ex(jax2tex(lambda a, b: a + b, 1, 2))

  # EX 2
  print_ex(jax2tex(lambda a, b: a + b / a, 1, 2))

  # EX 3
  f = lambda a, b: a + b / a
  print_ex(jax2tex(grad(f), 1., 2.))

  # EX 4
  def fn(a, b, c):
    return a + a * (b + c) / a
  print_ex(jax2tex(fn, np.array([[1, 2], [2, 4], [3, 7]]), 2, 3))

  # EX 5
  # pylint: disable=function-redefined
  # pylint: disable=invalid-name
  def fn(a, b, c):
    return a + a * (b + c)
  print_ex(jax2tex(grad(fn), 4., 2., 3.))

  # EX 6
  def fn(a, b):
    return a * (a - b) / (a + b) + b
  print_ex(jax2tex(grad(fn), 1., 1.))

  # EX 7
  print_ex(jax2tex(lambda W, x: W @ x, np.ones((3, 3)), np.ones((3,))))

  # EX 8
  print_ex(jax2tex(lambda W, x: W @ x, np.ones((3, 2)), np.ones((2, 3))))

  # EX 9
  def fn(W, x):
    return (W + W) @ (x * x)
  print_ex(jax2tex(fn, np.ones((3, 2)), np.ones((2, 3))))

  # EX 10
  def fn(W, x):
    return (W + W) @ (x * x)
  print_ex(jax2tex(grad(fn), np.ones((2,)), np.ones((2,))))

  # EX 11
  def fn(W, x):
    z = tex_var(W @ x, 'z')
    return z * z
  print_ex(jax2tex(fn, np.ones((4, 2,)), np.ones((2,))))

  # EX 12
  def fn(W, x):
    z1 = tex_var(W @ x, 'z^1')
    z2 = tex_var(W @ z1, 'z^2')
    return z2 @ z2
  print_ex(jax2tex(grad(fn), np.ones((2, 2,)), np.ones((2,))))

  # EX 13
  def fn(W, x):
    z1 = tex_var(W @ x, 'z^1')
    z2 = tex_var(W @ z1, 'z^2')
    return np.sqrt(z2 @ z2)
  print_ex(jax2tex(fn, np.ones((2, 2,)), np.ones((2,))))

  # EX 14
  def fn(x):
    return lax.broadcast_in_dim(x, (2, 3), (1,))
  print_ex(jax2tex(fn, np.ones((3,))))

  # EX 15
  def fn(c, x, y):
    return np.where(c, x, y)
  print_ex(jax2tex(fn, np.ones((3,), bool), np.ones((3,)), np.ones((3,))))

  # EX 16
  def fn(c, x, y):
    return np.where(c, x, y)
  print_ex(jax2tex(fn, True, np.ones((3,)), np.ones((3,))))

  # EX 17
  def fn(x):
    return np.transpose(x)
  print_ex(jax2tex(fn, np.ones((3, 2))))

  # EX 18
  def E(dr):
    idr = (tex_var(1, '\\sigma') / dr)
    idr6 = idr ** 6
    idr12 = idr ** 12
    return 4 * tex_var(1, '\\epsilon') * (idr12 - idr6)
  print_ex(jax2tex(E, np.ones((3, 3))))

  # Stax Examples
  def TexVar(layer, name, param_names=(), explicit_depends=False):
    init_fn, apply_fn = layer
    def tex_apply_fn(params, xs, **kwargs):
      if param_names:
        assert len(param_names) == len(params)
        params = tuple(tex_var(p, name, True) for p, name in
                       zip(params, param_names))
      return tex_var(apply_fn(params, xs, **kwargs),
                     name,
                     depends_on=xs if explicit_depends else ())
    return init_fn, tex_apply_fn
  init_fn, apply_fn = stax.serial(
      TexVar(stax.Dense(256), 'z^1', ('W^1', 'b^1')),
      TexVar(stax.Relu, 'y^1'),
      TexVar(stax.Dense(3), 'z^2', ('W^2', 'b^2')))

  # EX 19
  def f(params, x):
    return apply_fn(params, tex_var(x, 'x', True))
  _, params = init_fn(random.PRNGKey(0), (-1, 5))
  print_ex(jax2tex(f, params, np.ones((3, 5))))

  # pylint: disable=too-many-function-args
  def L(params, x, y_hat):
    y_hat = tex_var(y_hat, '\\hat y', True)
    return tex_var(-np.sum(y_hat * jax.nn.log_softmax(f(params, x))), 'L')
  # EX 20
  print_ex(jax2tex(L, params, np.ones((3, 5)), np.ones((3, 3))))
  # EX 21
  print_ex(jax2tex(grad(L), params, np.ones((3, 5)), np.ones((3, 3))))

  # EX 22
  init_fn, apply_fn = stax.serial(
      TexVar(stax.Dense(256), 'z^1', ('W^1', 'b^1'), True),
      TexVar(stax.Relu, 'y^1'),
      TexVar(stax.Dense(3), 'z^2', ('W^2', 'b^2')))
  def f(params, x):
    return apply_fn(params, tex_var(x, 'x', True))
  _, params = init_fn(random.PRNGKey(0), (-1, 5))
  print_ex(jax2tex(f, params, np.ones((3, 5))))

  # EX 23
  def nngp(params, x1, x2):
    x1 = tex_var(x1, 'x^1', True)
    x2 = tex_var(x2, 'x^2', True)
    return tex_var(apply_fn(params, x1) @ apply_fn(params, x2).T, '\\mathcal K')
  _, params = init_fn(random.PRNGKey(0), (-1, 5))
  print_ex(jax2tex(nngp, params, np.ones((3, 5)), np.ones((3, 5))))

  # Forward Mode vs Reverse Mode
  f = lambda a, b: a + tex_var(b / a, 'z')
  # EX 24
  print_ex(jax2tex(f, 1., 1.))
  # EX 25
  print_ex(jax2tex(grad(f), 1., 1.))
  # EX 26
  # pylint: disable=g-long-lambda
  print_ex(jax2tex(lambda a, b:
                   jvp(lambda a: f(a, b), (a,), (1.,))[1], 1., 1.))

  # EX 27
  def f(x, y):
    def g(r):
      return tex_var(r ** 2, 'z', depends_on=r)
    return g(x) + g(y)
  print_ex(jax2tex(f, 1., 1.))

  # EX 28
  def f(x_and_y):
    x, y = x_and_y
    return x * y
  print_ex(jax2tex(f, (1., 1.)))

  # EX 29
  def f(x_and_y):
    x, y = x_and_y
    return tex_var(x, 'x') * tex_var(y, 'y')
  print_ex(jax2tex(f, (1., 1.)))

  # EX 30
  def f(x_and_y):
    x, y = x_and_y
    return tex_var(x, 'x', True) * tex_var(y, 'y', True)
  print_ex(jax2tex(f, (1., 1.)))

  def f(x):
    return np.sin(x)
  # EX 31
  print_ex(jax2tex(grad(bind_names(f)), 1.))
  # EX 32
  print_ex(jax2tex(grad(f), 1.))


if __name__ == '__main__':
  app.run(main)
