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

# Lint as: python3
"""Memoize decorator."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
from absl import logging


class MemoizeCacheInputOutput(object):
  """Instance aware memoize which caches the input and output of functions.

  Note that this memoize does NOT support non-hashale arguments, i.e. they can't
  be the key of a dictionary. So in this case, the performance can be affected.
  The memoize decorator can be equipped to multiple functions of multiple
  classes, but they all share one memoize instance. So if different functions
  and different class instances need separate caches, they should be treated
  separately. One way to do that is to attach separate caches for each class
  instance by processing the `obj`, the instance, so as for the functions. See
  `__get__` for details. This memoize assumes the function is stateless, means
  the output only depends on the input value. Do NOT confuse with the
  `self.cache_handler`. When a function is called, the `obj.cache` is
  passed to `self.cache_handler` to do the calculation. So actually, the
  `self.cache_handler` does not store all instances' caches.
  """

  def __init__(self, function):
    self.function = function
    self.function_name = function.__name__
    # This `cache_handler` does not store all instances' caches, instead, they
    # are stored at `obj.cache`.
    self.cache_handler = {}

  def __call__(self, *args):
    if self.function not in self.cache_handler:
      self.cache_handler[self.function] = {}
    try:
      return self.cache_handler[self.function][args]
    # If it has both KeyError and TypeError, for example, `self.function` is not
    # in the cache yet and `args` is unhashable, it will go to the TypeError
    # first.
    except KeyError:
      value = self.function(*args)
      self.cache_handler[self.function][args] = value
      return value
    except TypeError:
      # Uncacheable argument. For instance, passing a list as an argument.
      # Better to not cache than to blow up entirely.
      logging.info(
          'The function "%s" gets uncacheable argument. Perhaps it is a list.',
          self.function_name)
      return self.function(*args)

  def __get__(self, obj, objtype):
    """Supports instance methods.

    If the memoize is instance aware, then this function uses separate caches
    for different instances. Using `self` alone is not able to handle different
    instances under the same class. The memoize is only associated with the
    class type, not each instance. When a function in a class is called, it
    first reaches the `__get__` then the `__call__`. `obj` and `objtype` give
    the information about the instance and the class type of that instance.

    Args:
      obj: The class instance calling the function.
      objtype: The instance's class type. This is required for __get__ to hold
          the position for the instance class type. It is not used in this
          function.

    Returns:
      function: The function being called.
    """
    function = functools.partial(self.__call__, obj)
    try:
      self.cache_handler = obj.cache
    # AttributeError: This corresponds to an instance being first time to run
    #     any function, in another word, the cache for all functions will be
    #     initilized.
    except AttributeError:
      obj.cache = {}
      self.cache_handler = obj.cache
    return function


class MemoizeClassFunctionOneRun(object):
  """Instance aware memoize which allows one call of each function.

  This memoize is used for some functions for class initialization, which
  prevents future changes. `first_run_args` is used to label each function if it
  has been run or not. Note that this memoize does NOT support non-hashable
  arguments if they are NOT hashable, i.e. they can't be the key of a
  dictionary. The memoize decorator can be equipped to multiple functions of
  multiple classes, but they all share one memoize instance. So if different
  functions and different class instances need separate caches, they should be
  treated separately. One way to do that is to attach separate caches for each
  class instance by processing the `obj`, so as for the functions. See `__get__`
  for details. Note that since this memoize only allows the function to run
  once, for later calls, no matter what the inputs are, it only returns the
  output from the first time run.  Do NOT confuse with the `self.cache_handler`.
  When a function is called, the `obj.cache` is passed to `self.cache_handler`
  to do the calculation. So actually, the `self.cache_handler` does not store
  all instances' caches.
  """

  def __init__(self, function):
    self.function = function
    self.function_name = function.__name__
    # This `cache_handler` does not store all instances' caches, instead, they
    # are stored at `obj.cache`.
    self.cache_handler = {}
    self.first_run_args = {}

  def __call__(self, *args):
    if self.function not in self.cache_handler:
      self.cache_handler[self.function] = {}
    try:
      # The args is in the tuple form (object address, input argument). So do
      # not worry about the empty input.
      if self.first_run_args[self.function]:
        logging.warning(
            'The memoized function "%s" has already been called.',
            self.function_name)
        # If the function has been executed once, then return the value from its
        # first time run. If it has multiple return values, then there is
        # something wrong.
        try:
          return self.cache_handler[self.function][
              self.first_run_args[self.function]]
        except TypeError:
          # Uncacheable. For instance, the function return a list, or a tuple.
          # Better to not cache than to blow up entirely. Return the result of
          # the firt run.
          return self.function(*self.first_run_args[self.function])
    # KeyError: Catch the case when not the first time of an instance to run
    #     any function, but the first time to run a specific function, i.e. the
    #     `cache_handler` has been initialized, but first time to face
    #     `function`, i.e. `self.function` is not in `self.first_run_args`.
    except KeyError:
      try:
        # The order here DOES matter. 1. Put args into first_run_args so that
        # it knows whether the function has been called or not. 2. Give it a
        # try on self.cache_handler[self.function][args] before calculating the
        # the function to check whether the input args is hashable or not. 3.
        # Then calculate the function output value. 2 is before 3 because if the
        # args is not hashable, the function will be called again in the except
        # if 3 is run before 2, then the function will be executed twice.
        self.first_run_args[self.function] = args
        self.cache_handler[self.function][args] = 0
        value = self.function(*args)
        self.cache_handler[self.function][args] = value
        return value
      except TypeError:
        # Uncacheable. For instance, the function return a list. Better to not
        # cache than to blow up entirely.
        logging.info(
            'The function "%s" gets uncacheable argument. Perhaps a list.',
            self.function_name)
        return self.function(*args)

  def __get__(self, obj, objtype):
    """Supports instance methods.

    If the memoize is instance aware, then this function uses separate caches
    for different instances. Using `self` alone is not able to handle different
    instances under the same class. When a function in a class is called, it
    first reaches the `__get__` then the `__call__`. `obj` and `objtype` give
    the information about the instance and the class type of that instance.

    Args:
      obj: The class instance calling the function.
      objtype: The instance's class type. This is required for __get__ to hold
          the position for the instance class type. It is not used in this
          function.

    Returns:
      function: The function is being called.
    """
    function = functools.partial(self.__call__, obj)
    try:
      self.cache_handler = obj.cache
      self.first_run_args = obj.first_run_args
    # AttributeError: This corresponds to an instance being first time to run
    #     any function, in another word, the cache for all functions will be
    #     initilized.
    except AttributeError:
      obj.cache = {}
      obj.first_run_args = {}
      self.cache_handler = obj.cache
      self.first_run_args = obj.first_run_args
    return function
