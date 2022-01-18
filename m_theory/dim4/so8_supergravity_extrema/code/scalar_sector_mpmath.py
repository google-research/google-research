# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Potential and Stationarity computation: mpmath interface.

This module allows running the potential computation in very high accuracy.
As it depends on the 'opt_einsum' and 'mpmath' Python modules, it is optional.

"""

import os

import mpmath
import numpy
import opt_einsum

from dim4.so8_supergravity_extrema.code import scalar_sector


# Hack to bend `mpmath` so that `opt_einsum` can work with it.
mpmath.mpf.shape = ()
mpmath.mpc.shape = ()


# Observe that our E7-definitions have 'numerically exact' structure constants.
#
# So, it actually makes sense to take these as defined, and lift them to mpmath.
#
# >>> set(e7.t_a_ij_kl.reshape(-1))
# >>> {(-2+0j), (-1+0j), -2j, -1j, 0j, 1j, 2j, (1+0j), (2+0j)}


# `mpmath` does not work with numpy.einsum(), so for that reason alone,
# we use opt_einsum's generic-no-backend alternative implementation.
mpmath_scalar_manifold_evaluator = scalar_sector.get_scalar_manifold_evaluator(
    frac=lambda p, q: mpmath.mpf(p) / mpmath.mpf(q),
    to_scaled_constant=(
        lambda x, scale=1: numpy.array(
            x, dtype=mpmath.ctx_mp_python.mpc) * scale),
    # Wrapping up `expm` is somewhat tricky here, as it returns a mpmath
    # matrix-type that numpy does not understand.
    expm=lambda m: numpy.array(mpmath.expm(m).tolist()),
    einsum=lambda spec, *arrs: opt_einsum.contract(spec, *arrs),
    # trace can stay as-is.
    eye=lambda n: numpy.eye(n) + mpmath.mpc(0),
    complexify=lambda a: a + mpmath.mpc(0),
    # re/im are again tricky, since numpy.array(dtype=object)
    # does not forward .real / .imag to the contained objects.
    re=lambda a: numpy.array([z.real for z in a.reshape(-1)]).reshape(a.shape),
    im=lambda a: numpy.array([z.imag for z in a.reshape(-1)]).reshape(a.shape))
