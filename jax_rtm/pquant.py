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
# -*- coding:utf-8 -*-

"""Physical quantity dimension tracking domain-specific language for JAX-RTM."""

# pylint: disable=invalid-name

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp

# Enable float64 for JAX to ensure precision in tests
jax.config.update("jax_enable_x64", True)


# ==============================================================================
# Quantity PyTree Definition
# ==============================================================================
class Quantity(NamedTuple):
  """A PyTree representing a JAX array with physical dimensions.

  Dimensions are represented as a 4-tuple of exponents:
  (Mass, Length, Time, Temperature)
  """

  value: jax.Array
  dimensions: tuple[float, float, float, float]

  def __repr__(self):
    dim_str = ",".join(
        f"{d:.2f}".rstrip("0").rstrip(".") if isinstance(d, float) else str(d)
        for d in self.dimensions
    )
    return f"Quantity(val={self.value}, dims=[{dim_str}])"

  # --- Operator Overloading ---

  def __add__(self, other):
    if isinstance(other, Quantity):
      if self.dimensions != other.dimensions:
        raise ValueError(
            f"Dimension mismatch in addition: {self.dimensions} vs"
            f" {other.dimensions}"
        )
      return Quantity(self.value + other.value, self.dimensions)
    # Allow adding any scalar if self is dimensionless or temperature offset
    if self.dimensions == (0, 0, 0, 0) or self.dimensions == (0, 0, 0, 1):
      return Quantity(self.value + other, self.dimensions)
    # Allow adding 0.0 (identity) for any dimensions
    if other == 0.0 or other == 0:
      return Quantity(self.value + other, self.dimensions)
    raise ValueError(
        f"Cannot add non-zero scalar {other} to physical quantity with dims"
        f" {self.dimensions}"
    )

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    if isinstance(other, Quantity):
      if self.dimensions != other.dimensions:
        raise ValueError(
            f"Dimension mismatch in subtraction: {self.dimensions} vs"
            f" {other.dimensions}"
        )
      return Quantity(self.value - other.value, self.dimensions)
    if self.dimensions == (0, 0, 0, 0) or self.dimensions == (0, 0, 0, 1):
      return Quantity(self.value - other, self.dimensions)
    if other == 0.0 or other == 0:
      return Quantity(self.value - other, self.dimensions)
    raise ValueError(
        f"Cannot subtract non-zero scalar {other} from physical quantity with"
        f" dims {self.dimensions}"
    )

  def __rsub__(self, other):
    if isinstance(other, Quantity):
      if self.dimensions != other.dimensions:
        raise ValueError(
            f"Dimension mismatch in subtraction: {other.dimensions} vs"
            f" {self.dimensions}"
        )
      return Quantity(other.value - self.value, self.dimensions)
    if self.dimensions == (0, 0, 0, 0) or self.dimensions == (0, 0, 0, 1):
      return Quantity(other - self.value, self.dimensions)
    if other == 0.0 or other == 0:
      return Quantity(other - self.value, self.dimensions)
    raise ValueError(
        f"Cannot subtract physical quantity with dims {self.dimensions} from"
        f" non-zero scalar {other}"
    )

  def __mul__(self, other):
    if isinstance(other, Quantity):
      new_dims = tuple(a + b for a, b in zip(self.dimensions, other.dimensions))
      return Quantity(self.value * other.value, new_dims)
    return Quantity(self.value * other, self.dimensions)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __truediv__(self, other):
    if isinstance(other, Quantity):
      new_dims = tuple(a - b for a, b in zip(self.dimensions, other.dimensions))
      return Quantity(self.value / other.value, new_dims)
    return Quantity(self.value / other, self.dimensions)

  def __rtruediv__(self, other):
    if isinstance(other, Quantity):
      new_dims = tuple(a - b for a, b in zip(other.dimensions, self.dimensions))
      return Quantity(other.value / self.value, new_dims)
    # Scalar divided by physical quantity: invert dimensions
    inv_dims = tuple(-d for d in self.dimensions)
    return Quantity(other / self.value, inv_dims)

  def __pow__(self, power):
    if not isinstance(power, (int, float)):
      raise ValueError("Exponent must be a real number")
    new_dims = tuple(d * power for d in self.dimensions)
    return Quantity(self.value**power, new_dims)

  # --- Comparison Overloading ---

  def __lt__(self, other):
    if isinstance(other, Quantity):
      if self.dimensions != other.dimensions:
        raise ValueError(
            "Cannot compare quantities with different dimensions:"
            f" {self.dimensions} vs {other.dimensions}"
        )
      return self.value < other.value
    # Allow comparison with 0.0 for physical boundaries
    if other == 0.0 or other == 0:
      return self.value < other
    if self.dimensions == (0, 0, 0, 0):
      return self.value < other
    raise ValueError(
        f"Cannot compare physical quantity with dims {self.dimensions} to"
        f" non-zero scalar {other}"
    )

  def __le__(self, other):
    if isinstance(other, Quantity):
      if self.dimensions != other.dimensions:
        raise ValueError(
            "Cannot compare quantities with different dimensions:"
            f" {self.dimensions} vs {other.dimensions}"
        )
      return self.value <= other.value
    if other == 0.0 or other == 0:
      return self.value <= other
    if self.dimensions == (0, 0, 0, 0):
      return self.value <= other
    raise ValueError(
        f"Cannot compare physical quantity with dims {self.dimensions} to"
        f" non-zero scalar {other}"
    )

  def __gt__(self, other):
    if isinstance(other, Quantity):
      if self.dimensions != other.dimensions:
        raise ValueError(
            "Cannot compare quantities with different dimensions:"
            f" {self.dimensions} vs {other.dimensions}"
        )
      return self.value > other.value
    if other == 0.0 or other == 0:
      return self.value > other
    if self.dimensions == (0, 0, 0, 0):
      return self.value > other
    raise ValueError(
        f"Cannot compare physical quantity with dims {self.dimensions} to"
        f" non-zero scalar {other}"
    )

  def __ge__(self, other):
    if isinstance(other, Quantity):
      if self.dimensions != other.dimensions:
        raise ValueError(
            "Cannot compare quantities with different dimensions:"
            f" {self.dimensions} vs {other.dimensions}"
        )
      return self.value >= other.value
    if other == 0.0 or other == 0:
      return self.value >= other
    if self.dimensions == (0, 0, 0, 0):
      return self.value >= other
    raise ValueError(
        f"Cannot compare physical quantity with dims {self.dimensions} to"
        f" non-zero scalar {other}"
    )

  def __eq__(self, other):
    if isinstance(other, Quantity):
      if self.dimensions != other.dimensions:
        return False
      return self.value == other.value
    if other == 0.0 or other == 0:
      return self.value == other
    if self.dimensions == (0, 0, 0, 0):
      return self.value == other
    return False

  # --- Unary Operators & Math ---

  def __neg__(self):
    return Quantity(-self.value, self.dimensions)

  def __pos__(self):
    return self

  def __abs__(self):
    return Quantity(jnp.abs(self.value), self.dimensions)

  # --- JAX-safe Boolean coercions ---
  def __bool__(self):
    """Delegates boolean evaluation to the underlying value.

    This is critical because test frameworks (like absltest/unittest) and JAX
    assertions frequently call bool() on arrays or scalars to check truthiness.

    Returns:
      The boolean truthiness of the underlying value.
    """
    if isinstance(self.value, jax.Array):
      # JAX arrays with size > 1 raise TypeError on bool() inside JIT,
      # but for scalar shapes in tests, we can safely evaluate them.
      if self.value.size == 1:
        return bool(self.value)
      raise TypeError(
          "Boolean value of an empty JAX array or array with more than one"
          " element is ambiguous"
      )
    return bool(self.value)

  # --- JAX Array Interface Emulation ---

  @property
  def shape(self):
    return self.value.shape

  @property
  def dtype(self):
    return self.value.dtype

  @property
  def ndim(self):
    return self.value.ndim

  def __len__(self):
    return len(self.value)

  def __getitem__(self, item):
    return Quantity(self.value[item], self.dimensions)


# Register Quantity as a JAX PyTree
jax.tree_util.register_pytree_node(
    Quantity,
    lambda x: ((x.value,), x.dimensions),
    lambda aux, children: Quantity(children[0], aux),
)


# ==============================================================================
# Physical Quantity DSL Constructors (Factory Functions)
# ==============================================================================
def Dimensionless(val):
  return Quantity(jnp.asarray(val), (0, 0, 0, 0))


Dimensionless.dimensions = (0, 0, 0, 0)


def Length(val):
  return Quantity(jnp.asarray(val), (0, 1, 0, 0))


Length.dimensions = (0, 1, 0, 0)


def Area(val):
  return Quantity(jnp.asarray(val), (0, 2, 0, 0))


Area.dimensions = (0, 2, 0, 0)


def Volume(val):
  return Quantity(jnp.asarray(val), (0, 3, 0, 0))


Volume.dimensions = (0, 3, 0, 0)


def Mass(val):
  return Quantity(jnp.asarray(val), (1, 0, 0, 0))


Mass.dimensions = (1, 0, 0, 0)


def Time(val):
  return Quantity(jnp.asarray(val), (0, 0, 1, 0))


Time.dimensions = (0, 0, 1, 0)


def Temperature(val):
  return Quantity(jnp.asarray(val), (0, 0, 0, 1))


Temperature.dimensions = (0, 0, 0, 1)


def Density(val):
  """Density: Mass / Length^3 [kg/m^3]."""
  return Quantity(jnp.asarray(val), (1, -3, 0, 0))


Density.dimensions = (1, -3, 0, 0)


def NumberConcentration(val):
  """Number Concentration: 1 / Length^3 [1/m^3]."""
  return Quantity(jnp.asarray(val), (0, -3, 0, 0))


NumberConcentration.dimensions = (0, -3, 0, 0)


def MassExtinction(val):
  """Mass Extinction Cross Section: Area / Mass [m^2/kg]."""
  return Quantity(jnp.asarray(val), (-1, 2, 0, 0))


MassExtinction.dimensions = (-1, 2, 0, 0)


def IWP(val):
  """Ice Water Path: Mass / Area [kg/m^2]."""
  return Quantity(jnp.asarray(val), (1, -2, 0, 0))


IWP.dimensions = (1, -2, 0, 0)


def InverseIWP(val):
  """Inverse Ice Water Path: Area / Mass [m^2/kg]."""
  return Quantity(jnp.asarray(val), (-1, 2, 0, 0))


InverseIWP.dimensions = (-1, 2, 0, 0)


def TemperaturePower(val, power):
  """Temperature raised to an arbitrary power [K^power]."""
  return Quantity(jnp.asarray(val), (0, 0, 0, power))


# Note: TemperaturePower does not have a static .dimensions attribute because it
# depends on power.


def SpectralRadiance(val):
  """Spectral Radiance: Mass / (Length * Time^3) [W / (m^2 sr um)]."""
  return Quantity(jnp.asarray(val), (1, -1, -3, 0))


SpectralRadiance.dimensions = (1, -1, -3, 0)


# ==============================================================================
# Recursive Helpers for Nested Structures
# ==============================================================================
def contains_pq(x):
  """Recursively checks if any nested structure contains a Quantity."""
  if isinstance(x, Quantity):
    return True
  if isinstance(x, (list, tuple)):
    return any(contains_pq(item) for item in x)
  if isinstance(x, dict):
    return any(contains_pq(item) for item in x.values())
  return False


def unwrap_pq(x):
  """Recursively unwraps Quantity containers, leaving raw JAX arrays."""
  if isinstance(x, Quantity):
    return x.value
  if isinstance(x, list):
    return [unwrap_pq(item) for item in x]
  if isinstance(x, tuple):
    return tuple(unwrap_pq(item) for item in x)
  if isinstance(x, dict):
    return {k: unwrap_pq(v) for k, v in x.items()}
  return x


def get_pq_leaves(structure):
  """Flattens a structure, treating Quantity as a leaf node in the JAX PyTree."""
  return jax.tree_util.tree_leaves(
      structure, is_leaf=lambda x: isinstance(x, Quantity)
  )


# ==============================================================================
# Monkey-Patching JAX Interceptor Wrappers
# ==============================================================================
class JNPWrapper:
  """Wrapper for jax.numpy to intercept operations on physical quantities."""

  def __init__(self, real_jnp):
    self._real_jnp = real_jnp

  def __getattr__(self, name):
    attr = getattr(self._real_jnp, name)
    if callable(attr):

      def wrapper(*args, **kwargs):
        # 1. Fallback immediately if no physical quantities are present
        if not contains_pq(args) and not contains_pq(kwargs):
          return attr(*args, **kwargs)

        # 2. Extract and assert dimensions, returning the output dimension tuple
        out_dims = self._propagate_dimensions(name, *args, **kwargs)

        # 3. Unwrap all arguments to raw JAX arrays for real execution
        unwrapped_args = [unwrap_pq(x) for x in args]
        unwrapped_kwargs = {k: unwrap_pq(v) for k, v in kwargs.items()}

        res_val = attr(*unwrapped_args, **unwrapped_kwargs)

        # 4. Wrap output back in physical Quantity container
        if out_dims is None:
          return res_val
        return Quantity(res_val, out_dims)

      return wrapper
    return attr

  def _propagate_dimensions(self, name, *args, **kwargs):
    """Calculates output dimensions for a given JAX numpy function."""
    if name in ("add", "subtract", "maximum", "minimum", "clip"):
      # Dimensions must match across all physical quantity inputs
      pq_leaves = [
          x
          for x in get_pq_leaves(args) + get_pq_leaves(kwargs)
          if isinstance(x, Quantity)
      ]

      if pq_leaves:
        if len(pq_leaves) > 1:
          first_dims = pq_leaves[0].dimensions
          for x in pq_leaves[1:]:
            if x.dimensions != first_dims:
              raise ValueError(
                  f"Dimension mismatch in jnp.{name}: {x.dimensions} vs"
                  f" {first_dims}"
              )
        return pq_leaves[0].dimensions
      return (0, 0, 0, 0)

    elif name == "where":
      # Check condition and branches
      pq_branches = [
          x
          for x in get_pq_leaves(args[1:]) + get_pq_leaves(kwargs)
          if isinstance(x, Quantity)
      ]

      if pq_branches:
        first_dims = pq_branches[0].dimensions
        for x in pq_branches[1:]:
          if x.dimensions != first_dims:
            raise ValueError(
                f"Dimension mismatch in branches of jnp.where: {x.dimensions}"
                f" vs {first_dims}"
            )
        return first_dims
      return (0, 0, 0, 0)

    elif name in (
        "log",
        "log10",
        "exp",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "zeros",
        "ones",
        "arange",
        "linspace",
        "any",
        "all",
        "isnan",
        "isinf",
        "isfinite",
        "isneginf",
        "isposinf",
    ):
      return (0, 0, 0, 0)

    elif name == "interp":
      fp = args[2] if len(args) > 2 else kwargs.get("fp")
      if isinstance(fp, Quantity):
        return fp.dimensions
      pq_fp = [x for x in get_pq_leaves(fp) if isinstance(x, Quantity)]
      if pq_fp:
        return pq_fp[0].dimensions
      return (0, 0, 0, 0)

    elif name in (
        "array",
        "asarray",
        "concatenate",
        "stack",
        "vstack",
        "hstack",
        "dstack",
    ):
      inputs = args[0] if args else kwargs.get("arrays")
      pq_inputs = [x for x in get_pq_leaves(inputs) if isinstance(x, Quantity)]
      if pq_inputs:
        first_dims = pq_inputs[0].dimensions
        for x in pq_inputs[1:]:
          if x.dimensions != first_dims:
            raise ValueError(
                f"Dimension mismatch in {name}: {x.dimensions} vs {first_dims}"
            )
        return first_dims
      return (0, 0, 0, 0)

    elif name in (
        "sum",
        "mean",
        "std",
        "min",
        "max",
        "cumsum",
        "nansum",
        "nanmean",
        "nanstd",
        "nanmin",
        "nanmax",
    ):
      # Linear reductions: output has the exact same dimensions as the input
      pq_leaves = [
          x
          for x in get_pq_leaves(args) + get_pq_leaves(kwargs)
          if isinstance(x, Quantity)
      ]
      if pq_leaves:
        return pq_leaves[0].dimensions
      return (0, 0, 0, 0)

    elif name in ("var", "nanvar"):
      # Variance has squared dimensions
      pq_leaves = [
          x
          for x in get_pq_leaves(args) + get_pq_leaves(kwargs)
          if isinstance(x, Quantity)
      ]
      if pq_leaves:
        return tuple(d * 2 for d in pq_leaves[0].dimensions)
      return (0, 0, 0, 0)

    elif name in ("dot", "matmul"):
      dims_a = (0, 0, 0, 0)
      dims_b = (0, 0, 0, 0)
      pq_a = [x for x in get_pq_leaves(args[0]) if isinstance(x, Quantity)]
      pq_b = [x for x in get_pq_leaves(args[1]) if isinstance(x, Quantity)]
      if pq_a:
        dims_a = pq_a[0].dimensions
      if pq_b:
        dims_b = pq_b[0].dimensions
      return tuple(a + b for a, b in zip(dims_a, dims_b))

    raise NotImplementedError(
        f"Dimension propagation not implemented for jnp.{name}"
    )


class JaxWrapper:
  """Wrapper for jax module to intercept submodules (like jax.nn)."""

  def __init__(self, real_jax):
    self._real_jax = real_jax
    self.nn = NNWrapper(real_jax.nn)

  def __getattr__(self, name):
    return getattr(self._real_jax, name)


class NNWrapper:
  """Wrapper for jax.nn to intercept activation functions."""

  def __init__(self, real_nn):
    self._real_nn = real_nn

  def __getattr__(self, name):
    attr = getattr(self._real_nn, name)
    if callable(attr):

      def wrapper(*args, **kwargs):
        if not contains_pq(args) and not contains_pq(kwargs):
          return attr(*args, **kwargs)

        unwrapped_args = [unwrap_pq(x) for x in args]
        unwrapped_kwargs = {k: unwrap_pq(v) for k, v in kwargs.items()}
        res_val = attr(*unwrapped_args, **unwrapped_kwargs)

        return Quantity(res_val, (0, 0, 0, 0))

      return wrapper
    return attr


# ==============================================================================
# Module Instrumentation Helper
# ==============================================================================
def instrument_module(module):
  """Monkey-patches jnp and jax inside the target module for dimensional tracking."""
  if hasattr(module, "jnp"):
    module.jnp = JNPWrapper(module.jnp)
    print(f"Instrumented jnp in module: {module.__name__}")
  if hasattr(module, "jax"):
    module.jax = JaxWrapper(module.jax)
    print(f"Instrumented jax in module: {module.__name__}")
