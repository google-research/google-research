# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Memoize decorator specialized for tf computations.

Example usage:

import tf_memoize

# Memoize decorate `my_function`.

@tf_memoize.memoize
def my_function(x, y, z):
  return x + y + z

Create a memoize cache:

cache = tf_memoize.create_cache()

# Bind the cache

my_function_bound = tf_memoize.bind(my_function, cache)

my_function_bound(x0, y0, z0)
my_function_bound(x0, y0, z0)  # Here we will hit the cache.

assert my_function.get_total_cache_hits(cache) == 1

# If we don't want to memoize, we can call the function without binding it:

my_function(x0, y0, z0)
my_function(x0, y0, z0)  # No cache hit.

# Alternatively, we can bind it with a `None` cache:

my_function_nocache = tf_memoize.bind(my_function, cache=None)

my_function_nocache(x0, y0, z0)
my_function_nocache(x0, y0, z0)  # No cache hit.

"""
import collections
import functools
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import tensorflow as tf


def _to_ref(x):
  """Gets a tf .ref() reference to tf.Tensor, otherwise acts as identity."""
  if isinstance(x, tf.Tensor):
    return x.ref()
  else:
    return x


def _to_str(x):
  if isinstance(x, tf.Tensor):
    return f"<tf.Tensor: shape={x.shape}, dtype={repr(x.dtype)[3:]}, id={id(x)}>"
  else:
    return x


# Subclass `object` instead of `NamedTuple` to avoid recursive tracking in
# tf.Module, which breaks on the internal cache dicts.
class _Cache:

  def __init__(self, cache,
               hits):
    self.cache = cache
    self.hits = hits

  def __iter__(self):
    return iter([self.cache, self.hits])


Cache = Optional[_Cache]

ReturnType = TypeVar("ReturnType")
WithCache = Tuple[ReturnType, _Cache]


def create_cache():
  """Creates a memoize cache."""
  return _Cache(
      cache=collections.defaultdict(dict),
      hits=collections.defaultdict(lambda: collections.defaultdict(int)))


def _ensure_hashable(x):
  """Return a version of x that is hashable."""
  if isinstance(x, (list, tuple)):
    return tuple(_ensure_hashable(y) for y in x)

  if isinstance(x, dict):
    return tuple((_ensure_hashable(k), _ensure_hashable(v))
                 for k, v in x.items())

  if isinstance(x, tf.Tensor):
    return x.ref()

  assert hash(x) is not None
  return x


def memoize(f):
  """Memoize decorator, using a cache provided through `tf_memoize.bind`."""

  @functools.wraps(f)
  def wrapper(*args,
              _private_cache_kwarg = None,
              _expect_cache_hit = None,
              **kwargs):  # pylint: disable = invalid-name
    cache = _private_cache_kwarg
    if cache is None:
      assert _expect_cache_hit in [False, None]
      return f(*args, **kwargs)

    result_cache, hit_counter = cache

    # Retrieve this functions (local) cache from the global cache.
    result_cache = result_cache[f]
    hit_counter = hit_counter[f]

    # Construct the cache key from `args` and `kwargs`.

    # Sometimes we have nested memoize functions, and need to pass around
    # _Cache objects to bind the inner function. We explicitly ignore
    # such objects from the key.
    key_args = [x for x in args if not isinstance(x, _Cache)]
    key_kwargs = [(k, v)
                  for (k, v) in sorted(kwargs.items())
                  if not isinstance(v, _Cache)]
    key = (tuple(key_args), tuple(key_kwargs))

    # Tensors are not hashable, so we need to get their .ref() property.
    key = _ensure_hashable(key)

    if key in result_cache:
      assert _expect_cache_hit in [True, None]
      hit_counter[key] += 1
      result = result_cache[key]
    else:
      assert _expect_cache_hit in [False, None]
      result = f(*args, **kwargs)
      result_cache[key] = result
    return result

  # For inspection/testing only.
  wrapper.get_cache_hits = lambda cache, key: cache.hits[f][key]
  wrapper.get_total_cache_hits = lambda cache: sum(cache.hits[f].values())
  return wrapper


def bind(f, cache,
         expect_cache_hit = None):
  """Bind the cache `cache` to the function `f` (see module docstring).

  Args:
    f: A function which has been decorated with `tf_memoize.memoize`.
    cache: A memoize cache created by `tf_memoize.create_memoize_cache`, or
      `None` if no caching is desired.
    expect_cache_hit: To help debugging; None => no expectation.

  Returns:
    The memoized function, using `cache` as the memoize cache (or the
    non-memoized function if `cache` is None).
  """
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    return f(*args, _private_cache_kwarg=cache,
             _expect_cache_hit=expect_cache_hit, **kwargs)

  return wrapper
