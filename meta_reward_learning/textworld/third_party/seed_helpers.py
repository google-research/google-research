# coding=utf-8
"""Seed Helpers copied from OpenAI Gym.

https://github.com/openai/gym/blob/master/gym/utils/seeding.py
"""

import hashlib
import os
import struct
import numpy as np
from six import integer_types


def np_random(seed=None):
  if seed is not None and not (isinstance(seed, integer_types) and 0 <= seed):
    raise ValueError(
        'Seed must be a non-negative integer or omitted, not {}'.format(seed))

  seed = create_seed(seed)

  rng = np.random.RandomState()
  rng.seed(_int_list_from_bigint(hash_seed(seed)))
  return rng, seed


def hash_seed(seed=None, max_bytes=8):
  # pylint: disable=g-doc-return-or-yield
  """Function for hashing the seed.

    Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:

    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928

    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)

  Args:
    seed (Optional[int]): None seeds from an operating system specific
      randomness source.
    max_bytes: Maximum number of bytes to use in the hashed seed.
  """
  # pylint: enable=g-doc-return-or-yield
  if seed is None:
    seed = create_seed(max_bytes=max_bytes)
  hash_value = hashlib.sha512(str(seed).encode('utf8')).digest()
  return _bigint_from_bytes(hash_value[:max_bytes])


def create_seed(a=None, max_bytes=8):
  # pylint: disable=g-doc-return-or-yield
  """Create a strong random seed.

  Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.

  Args:
    a (Optional[int, str]): None seeds from an operating system specific
      randomness source.
    max_bytes: Maximum number of bytes to use in the seed.
  """
  # pylint: enable=g-doc-return-or-yield
  # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
  if a is None:
    a = _bigint_from_bytes(os.urandom(max_bytes))
  elif isinstance(a, str):
    a = a.encode('utf8')
    a += hashlib.sha512(a).digest()
    a = _bigint_from_bytes(a[:max_bytes])
  elif isinstance(a, integer_types):
    a = a % 2**(8 * max_bytes)
  else:
    raise ValueError('Invalid type for seed: {} ({})'.format(type(a), a))

  return a


# pylint: disable=redefined-builtin
def _bigint_from_bytes(bytes):
  sizeof_int = 4
  padding = sizeof_int - len(bytes) % sizeof_int
  bytes += b'\0' * padding
  int_count = int(len(bytes) / sizeof_int)
  unpacked = struct.unpack('{}I'.format(int_count), bytes)
  accum = 0
  for i, val in enumerate(unpacked):
    accum += 2**(sizeof_int * 8 * i) * val
  return accum


# pylint: enable=redefined-builtin


# pylint: disable=missing-docstring
def _int_list_from_bigint(bigint):
  if bigint < 0:  # Special case 0
    raise ValueError('Seed must be non-negative, not {}'.format(bigint))
  elif bigint == 0:
    return [0]

  ints = []
  while bigint > 0:
    bigint, mod = divmod(bigint, 2**32)
    ints.append(mod)
  return ints


# pylint: enable=missing-docstring
