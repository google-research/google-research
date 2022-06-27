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

"""Base modules.
"""

# pylint: disable=g-importing-member, g-bad-import-order
from collections import OrderedDict
import tensorflow.compat.v1 as tf

from weak_disentangle.tensorsketch import utils as tsu


def build_with_name_scope(build_parameters):
  @tf.Module.with_name_scope
  def build_params_once_with_ns(self, *args):
    assert not self.built, "{}.built already True".format(self.name)
    build_parameters(self, *args)
    self.built = True
  return build_params_once_with_ns


class Repr(object):
  """Representation object.
  """

  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name


class Module(tf.Module):
  """Abstract module class.

  Module is a tree-structured class that can contain other Module objects.
  Traversal of the tree is supported via iterating through _child_modules. All
  models and layers should be subclasses of Module. This class provides support
  for several useful features: setting train/eval, tracking child modules,
  tracking tf.Variables, in/out hooks, mapping a function into the Module tree
  (the apply function), and printing the module as a string representation (for
  which we support printing at various levels of verbosity).
  """

  # Levels of read priority
  WITH_NAMES = 0
  WITH_EXTRA = 1
  WITH_VARS = 2
  WITH_DTYPE = 3
  WITH_NUMPY = 4

  def __init__(self, name=None):
    """Module initializer.

    Args:
      name: string for the name of the module used for tf name scoping.
    """
    # Special construction of _child_modules and _variables
    # to avoid triggering self.__setattr__
    self.__dict__["_blacklist"] = set()
    self.__dict__["_child_modules"] = OrderedDict()
    self.__dict__["_variables"] = OrderedDict()
    super().__init__(name=name)

    self.training = True
    self.built = False
    self.in_hooks = OrderedDict()
    self.out_hooks = OrderedDict()

  def __setattr__(self, name, value):
    # We catch non-blacklisted variables for the purposes of repr construction
    # only and do # not affect the computational graph.
    try:
      if name not in self._blacklist:
        if isinstance(value, Module):
          self._child_modules.update({name: value})
        else:
          if name in self._child_modules:
            del self._child_modules[name]

        if isinstance(value, tf.Variable):
          self._variables.update({name: value})
        else:
          if name in self._variables:
            del self._variables[name]
    except AttributeError as e:
      raise AttributeError(
          "Call super().__init__() before assigning variable to Module instance"
          ) from e

    # tf.Module makes modifications important to graph construction
    super().__setattr__(name, value)

  def __delattr__(self, name):
    if name not in self._blacklist:
      if name in self._child_modules:
        del self._child_modules[name]
      elif name in self._variables:
        del self._variables[name]

    super().__delattr__(name)

  def select_hooks_dict(self, in_hook):
    if in_hook:
      return self.in_hooks
    else:
      return self.out_hooks

  def train(self, mode=True):
    self.training = mode
    for m in self.submodules:
      m.train(mode)

  def apply(self, fn, filter_fn=None, targets=None):
    # Light wrapper to parse filter_fn and targets args
    if targets is not None:
      assert filter_fn is None, "Cannot use both filter_fn and targets"
      filter_fn = lambda m: isinstance(m, targets)

    elif filter_fn is None:
      filter_fn = lambda m: True

    self._apply(fn, filter_fn)

  def _apply(self, fn, filter_fn):
    # Apply fn to children first before applying to parent
    # This ensures that parent can override children's decisions
    # Run in chronological reverse order to get reverse topo+chrono apply
    for m in reversed(self._child_modules.values()):
      # pylint: disable=protected-access
      m._apply(fn, filter_fn)

    if filter_fn(self):
      fn(self)

  def eval(self):
    self.train(False)

  def build(self, *shapes, once=True):
    if once:
      assert not self.built, "{}.built already True".format(self.name)
    tensors = tsu.shapes_to_zeros(*shapes)
    self(*tensors)
    return self

  @build_with_name_scope
  def build_parameters(self, *inputs):
    pass  # By default, module is parameterless

  def reset_parameters(self):
    pass

  def forward(self, *inputs):
    return inputs

  @tf.Module.with_name_scope
  def __call__(self, *inputs):
    if not self.built:
      self.build_parameters(*inputs)

    for hook in self.in_hooks.values():
      response = hook(self, *inputs)
      if response is not None:
        inputs = tsu.pack(response)

    outputs = self.forward(*inputs)

    for hook in self.out_hooks.values():
      response = hook(self, *tsu.pack(outputs))
      if response is not None:
        outputs = response

    return outputs

  def __repr__(self):
    return self.to_string(verbose=0)

  def extra_repr(self):
    return ""

  def read(self, verbose=0, trainable=None):
    return Repr(self.to_string(verbose, trainable))

  def to_string(self, verbose=0, trainable=None):
    # Level 0: only names
    # Level 1: Level 0 + extra repr
    # Level 2: Level 1 + variable names and info
    # Level 3: Level 2 + dtype info
    # Level 4: Level 3 + actual variable info (shortened)
    main = self.name
    if verbose >= 1:
      main += self.extra_repr()

    if verbose >= 2:
      var_body = "\n"
      for (name, var) in self._variables.items():
        # Skip non-trainable variables if filtering by trainability
        if trainable and not var.trainable:
          continue

        # pylint: disable=protected-access
        var_body += "{}.{}: shape={}".format(self.name, name,
                                             var._shape_tuple())
        if verbose >= 3:
          var_body += ", dtype={}".format(repr(var.dtype))
        if var.trainable:
          var_body += " {train}"
        if verbose >= 4:
          var_body += "\n" + tsu.indent(tsu.shorten(str(var.numpy())))
        var_body += "\n"

      main += tsu.indent(var_body).rstrip()

    body = "\n"
    for module in self._child_modules.values():
      body += module.to_string(verbose, trainable) + "\n"
    main += tsu.indent(body).rstrip()

    # Wrap string as Repr object
    return main

  def flatten_modules(self, filter_fn=None, targets=None):
    # Returns a flattened version of tree in reverse topological order
    module_list = []
    def collect(m):
      module_list.append(m)
    self.apply(collect, filter_fn, targets)
    return list(reversed(module_list))


class ModuleList(Module):
  """Stores a list of Modules.
  """

  def __init__(self, *modules, name=None):
    """ModuleList initializer.

    Args:
      *modules: tuple of modules or a tuple of a single list of modules.
      name: name scope for this module.

    Raises:
      ValueError: input is not modules or a list of modules.
    """
    super().__init__(name=name)
    self.modules = list(self.disambiguate_modules(modules))
    self._child_modules.update(zip(range(len(self.modules)), self.modules))

  def disambiguate_modules(self, modules):
    # We support passing in either modules as arguments, or a single list
    # of modules. In other words, at this point the variable modules should
    # either be
    #   modules = (m, m, ...)
    # or
    #   modules = ((m, m, ...),) or ([m, m, ...],)
    # To disambiguate, check if elements of modules is Module.
    if tsu.elem_isinstance(modules, Module):
      # We leverage isinstance to properly handle edge-case where
      # modules=() here.
      return modules
    elif len(modules) == 1 and tsu.elem_isinstance(modules[0], Module):
      return modules[0]
    else:
      raise ValueError("Input must modules or a list of modules")

  def append(self, *modules):
    modules = self.disambiguate_modules(modules)
    for module in modules:
      self._child_modules.update({len(self.modules): module})
      self.modules.append(module)

  def __iter__(self):
    return iter(self.modules)

  def __getitem__(self, index):
    return self.modules[index]


class Sequential(ModuleList):
  """Stores a list of modules that can be daisy-chained in forward call.
  """

  def forward(self, *inputs):
    for module in self.modules:
      inputs = module(*tsu.pack(inputs))
    return inputs
