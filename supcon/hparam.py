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

"""A generic serializable hyperparameter container.

hparam is based on `attr` under the covers, but provides additional features,
such as serialization and deserialization in a format that is compatible with
(now defunct) tensorflow.HParams, runtime type checking, implicit casting where
safe to do so (e.g. int->float, scalar->list).

Unlike tensorflow.HParams, this supports hierarchical nesting of parameters for
better organization, aliasing parameters to short abbreviations for compact
serialization while maintaining code readability, and support for Enum values.

Example usage:
  @hparam.s
  class MyNestedHParams:
    learning_rate: float = hparam.field(abbrev='lr', default=0.1)
    layer_sizes: List[int] = hparam.field(abbrev='ls', default=[256, 64, 32])

  @hparam.s
  class MyHParams:
    nested_params: MyNestedHParams = hparam.nest(MyNestedHParams)
    non_nested_param: int = hparam.field(abbrev='nn', default=0)

  hparams = MyHParams(nested_params=MyNestedHParams(
                        learning_rate=0.02, layer_sizes=[100, 10]),
                      non_nested_param=5)
  hparams.nested_params.learning_rate = 0.002
  serialized = hparams.serialize()  # "lr=0.002,ls=[100,10],nn=5"
  hparams.nested_params.learning_rate = 0.003
  new_hparams = MyHParams(serialized)
  new_hparams.nested_params.learning_rate == 0.002 # True
"""

import collections
import copy
import csv
import enum
import inspect
import json
import numbers
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import attr
import six

_ABBREV_KEY = 'hparam.abbrev'
_SCALAR_TYPE_KEY = 'hparam.scalar_type'
_IS_LIST_KEY = 'hparam.is_list'
_PREFIX_KEY = 'hparam.prefix'
_SERIALIZED_ARG = '_hparam_serialized_arg'

# Define the regular expression for parsing a single clause of the input
# (delimited by commas).  A legal clause looks like:
#   <variable name> = <rhs>
# where <rhs> is either a single token or [] enclosed list of tokens.
# For example:  "var = a" or "x = [1,2,3]"
_PARAM_RE = re.compile(
    r"""
  (?P<name>[a-zA-Z][\w\.]*)      # variable name: 'var' or 'x'
  \s*=\s*
  ((?P<strval>".*")              # single quoted string value: '"a,b=c"' or None
   |
   (?P<val>[^,\[]*)              # single value: 'a' or None
   |
   \[(?P<vals>[^\]]*)\])         # list of values: None or '1,2,3'
                                 # (the regex removes the surrounding brackets)
  ($|,\s*)""", re.VERBOSE)

_ValidScalarInstanceType = Union[int, float, str, enum.Enum]
_ValidListInstanceType = Union[List[_ValidScalarInstanceType],
                               Tuple[_ValidScalarInstanceType]]
_ValidFieldInstanceType = Union[_ValidScalarInstanceType,
                                _ValidListInstanceType]
_ValidScalarType = Type[_ValidScalarInstanceType]
T = TypeVar('T')


@attr.s
class _FieldInfo:
  """Metadata for a single HParam field."""
  # The path to a field from the root hparam.s instance. This is to enable
  # finding fields that are in nested hparam.s classes. For example, a path
  # ['foo', 'bar'] means that the root class has a field called 'foo' created
  # with hparam.nest(), whose value is another hparam.s class, which contains a
  # field, 'bar' that was created using hparam.field(). All path elements other
  # than the last correspond to nested classes, while the last is a field.
  path: List[str] = attr.ib()
  # If the field value is a single (scalar) value, then this is the type of the
  # value. If it is a list then this is the type of the list elements, which
  # must all be the same.
  scalar_type: _ValidScalarType = attr.ib()
  # Whether this field is a list type.
  is_list: bool = attr.ib()
  # The default value for this field.
  default_value: _ValidFieldInstanceType = attr.ib()


_HParamsMapType = Dict[str, _FieldInfo]


def _get_type(instance):
  """Determines the type of a value.

  Both whether it is an iterable and what the scalar type is. For iterables,
  scalar type is the type of its elements, which are required to all be the same
  type.

  Valid types are int, float, string, Enum (where the value is int, float, or
  string), and lists or tuples of those types.

  Args:
    instance: The value whose type is to be determined.

  Returns:
    scalar_type: The type of `instance`'s elements, if `instance` is a list or
      tuple. Otherwise, the type of `instance`.
    is_list: Whether `instance` is a list or tuple.

  Raises:
    TypeError: If instance is not a valid type.
      * `instance` is an iterable that is not a list, tuple or string.
      * `instance` is a list or tuple with elements of different types.
      * `instance` is an empty list or tuple.
      * `instance` is a list or tuple whose elements are non-string iterables.
      * `instance` is None or a list or tuple of Nones.
      * `instance` is none of {int, float, str, Enum, list, tuple}.
      * `instance` is a list or tuple whose values are not one of the above
         types.
      * `instance` is an Enum type whose values are neither numbers nor strings.
  """

  is_list = False
  scalar_type = type(instance)
  if isinstance(instance, Iterable) and not issubclass(
      scalar_type, (six.string_types, six.binary_type)):
    is_list = True
    if not instance:
      raise TypeError('Empty iterables cannot be used as default values.')

    if not isinstance(instance, collections.abc.Sequence):
      # Most likely a Dictionary or Set.
      raise TypeError('Only numbers, strings, and lists are supported. Found '
                      f'{scalar_type}.')

    scalar_type = type(instance[0])
    if isinstance(instance[0], Iterable) and not issubclass(
        scalar_type, (six.string_types, six.binary_type)):
      raise ValueError('Nested iterables and dictionaries are not supported.')
    if not all([isinstance(i, scalar_type) for i in instance]):
      raise TypeError('Iterables of mixed type are not supported.')

  if issubclass(scalar_type, type(None)):
    raise TypeError('Fields cannot have a default value of None.')

  valid_field_types = (six.string_types, six.binary_type, numbers.Integral,
                       numbers.Number)
  if not issubclass(scalar_type, valid_field_types + (enum.Enum,)):
    raise TypeError(
        'Supported types include: number, string, Enum, and lists of those '
        f'types. {scalar_type} is not one of those.')

  if issubclass(scalar_type, enum.Enum):
    enum_value_type = type(list(scalar_type.__members__.values())[0].value)
    if not issubclass(enum_value_type, valid_field_types):
      raise TypeError(f'Enum type {scalar_type} has values of type '
                      f'{enum_value_type}, which is not allowed. Enum values '
                      'must be numbers or strings.')

  return (scalar_type, is_list)


def _make_converter(scalar_type,
                    is_list):
  """Produces a function that casts a value to the target type, if compatible.

  Args:
    scalar_type: The scalar type of the hparam.
    is_list: Whether the hparam is a list type.

  Returns:
    A function that casts its input to either `scalar_type` or
    `List[scalar_type]` depending on `is_list`.
  """

  def scalar_converter(value):
    """Converts a scalar value to the target type, if compatible.

    Args:
      value: The value to be converted.

    Returns:
      The converted value.

    Raises:
      TypeError: If the type of `value` is not compatible with scalar_type.
        * If `scalar_type` is a string type, but `value` is not.
        * If `scalar_type` is a boolean, but `value` is not, or vice versa.
        * If `scalar_type` is an integer type, but `value` is not.
        * If `scalar_type` is a float type, but `value` is not a numeric type.
    """
    if (isinstance(value, Iterable) and
        not issubclass(type(value), (six.string_types, six.binary_type))):
      raise TypeError('Nested iterables are not supported')

    # If `value` is already of type `scalar_type`, return it directly.
    # `isinstance` is too weak (e.g. isinstance(True, int) == True).
    if type(value) == scalar_type:  # pylint: disable=unidiomatic-typecheck
      return value

    # Some callers use None, for which we can't do any casting/checking. :(
    if issubclass(scalar_type, type(None)):
      return value

    # Avoid converting a non-string type to a string.
    if (issubclass(scalar_type, (six.string_types, six.binary_type)) and
        not isinstance(value, (six.string_types, six.binary_type))):
      raise TypeError(
          f'Expected a string value but found {value} with type {type(value)}.')

    # Avoid converting a number or string type to a boolean or vice versa.
    if issubclass(scalar_type, bool) != isinstance(value, bool):
      raise TypeError(
          f'Expected a bool value but found {value} with type {type(value)}.')

    # Avoid converting float to an integer (the reverse is fine).
    if (issubclass(scalar_type, numbers.Integral) and
        not isinstance(value, numbers.Integral)):
      raise TypeError(
          f'Expected an integer value, but found {value} with type '
          f'{type(value)}.'
      )

    # Avoid converting a non-numeric type to a numeric type.
    if (issubclass(scalar_type, numbers.Number) and
        not isinstance(value, numbers.Number)):
      raise TypeError(
          f'Expected a numeric type, but found {value} with type {type(value)}.'
      )

    return scalar_type(value)

  def converter(value):
    """Converts a value to the target type, if compatible.

    Args:
      value: The value to be converted.

    Returns:
      The converted value.

    Raises:
      TypeError: If the type of `value` is not compatible with `scalar_type` and
      `is_list`.
        * If `scalar_type` is a string type, but `value` is not.
        * If `scalar_type` is a boolean, but `value` is not, or vice versa.
        * If `scalar_type` is an integer type, but `value` is not.
        * If `scalar_type` is a float type, but `value` is not a numeric type.
        * If `is_list` is False, but value is a non-string iterable.
    """
    value_is_listlike = (
        isinstance(value, Iterable) and
        not issubclass(type(value), (six.string_types, six.binary_type)))
    if value_is_listlike:
      if is_list:
        return [scalar_converter(v) for v in value]
      else:
        raise TypeError('Assigning an iterable to a scalar field.')
    else:
      if is_list:
        return [scalar_converter(value)]
      else:
        return scalar_converter(value)

  return converter


def field(abbrev, default):
  """Create a new field on an HParams class.

  A field is a single hyperparameter with a value. Fields must have an
  abbreviation key, which by convention is a short string, which is used to
  produce a concise serialization. Fields must also have a default value that
  determines the hyperparameter's type, which cannot be dynamically changed.
  Valid types are integers, floats, strings, enums that have values that are
  those types, and lists of those types.

  An HParams class can have child HParams classes, but those should be added
  using `nest` instead of `field`.

  Example usage:
    @hparam.s
    class MyHparams:
      learning_rate: float = hparam.field(abbrev='lr', default=0.1)
      layer_sizes: List[int] = hparam.field(abbrev='ls', default=[256, 64, 32])
      optimizer: OptimizerEnum = hparam.field(abbrev='opt',
      default=OptimizerEnum.SGD)

  Args:
    abbrev: A short string that represents this hyperparameter in the serialized
      format.
    default: The default value of this hyperparameter. This is required. Valid
      types are integers, floats, strings, enums that have values that are those
      types, and lists of those types. Default values for list-typed fields must
      be non-empty lists. None is not an allowed default. List values will be
      copied into instances of the field, so modifications to a list provided as
      default will not be reflected in existing or subsequently created class
      instances.

  Returns:
    A field-descriptor which can be consumed by a class decorated by @hparam.s.

  Raises:
    TypeError if the default value is not one of the allowed types.
  """

  scalar_type, is_list = _get_type(default)
  kwargs = {
      'kw_only': True,
      'metadata': {
          _ABBREV_KEY: abbrev,
          _SCALAR_TYPE_KEY: scalar_type,
          _IS_LIST_KEY: is_list,
      },
      'converter': _make_converter(scalar_type, is_list),
  }
  if is_list:
    # Lists are mutable, so we generate a factory method to produce a copy of
    # the list to avoid different instances of the class mutating each other.
    kwargs['factory'] = lambda: copy.copy(default)
  else:
    kwargs['default'] = default
  return attr.ib(**kwargs)  # pytype: disable=duplicate-keyword-argument


def nest(nested_class,
         prefix = None):
  """Create a nested HParams class field on a parent HParams class.

  An HParams class (a class decorated with @hparam.s) can have a field that is
  another HParams class, to create a hierarchical structure. Use `nest` to
  create these fields.

  Example usage:
    @hparam.s
    class MyNestedHParams:
      learning_rate: float = hparam.field(abbrev='lr', default=0.1)
      layer_sizes: List[int] = hparam.field(abbrev='ls', default=[256, 64, 32])

    @hparam.s
    class MyHParams:
      nested_params: MyNestedHParams = hparam.nest(MyNestedHParams)
      non_nested_param: int = hparam.field(abbrev='nn', default=0)

  Args:
    nested_class: The class of the nested hyperparams. The class must be
      decorated with @hparam.s.
    prefix: An optional prefix to add to the abbrev field of all fields in the
      nested hyperparams. This enables nesting the same class multiple times, as
      long as the prefix is different.

  Returns:
    A field-descriptor which can be consumed by a class decorated by @hparam.s.

  Raises:
    TypeError if `nested_class` is not decorated with @hparam.s.
  """
  if not inspect.isclass(nested_class):
    raise TypeError('nest() must be passed a class, not an instance.')
  if not (attr.has(nested_class) and
          getattr(nested_class, '__hparams_class__', False)):
    raise TypeError('Nested hparams classes must use the @hparam.s decorator')
  return attr.ib(
      factory=nested_class, kw_only=True, metadata={_PREFIX_KEY: prefix})


def _serialize_value(value,
                     field_info):
  """Serializes a value to a string.

  Lists are serialized by recursively calling this function on each of their
  elements. Enums use the enum value. Bools are cast to int. Strings that
  contain any of {,=[]"} are surrounded by double quotes. Everything is then
  cast using str().

  Args:
    value: The value to be serialized.
    field_info: The field info corresponding to `value`.

  Returns:
    The serialized value.
  """
  if field_info.is_list:
    list_value = value  # type: _ValidListInstanceType  # pytype: disable=annotation-type-mismatch
    modified_field_info = copy.copy(field_info)
    modified_field_info.is_list = False
    # Manually string-ify the list, since default str(list) adds whitespace.
    return ('[' + ','.join(
        [str(_serialize_value(v, modified_field_info)) for v in list_value]) +
            ']')
  scalar_value = value  # type: _ValidScalarInstanceType  # pytype: disable=annotation-type-mismatch
  if issubclass(field_info.scalar_type, enum.Enum):
    enum_value = scalar_value  # type: enum.Enum
    return str(enum_value.value)
  elif field_info.scalar_type == bool:
    bool_value = scalar_value  # type: bool  # pytype: disable=annotation-type-mismatch
    # use 0/1 instead of True/False for more compact serialization.
    return str(int(bool_value))
  elif issubclass(field_info.scalar_type, six.string_types):
    str_value = scalar_value  # type: str
    if any(char in str_value for char in ',=[]"'):
      return f'"{str_value}"'
  return str(value)


def _parse_serialized(
    values,
    hparams_map):
  """Parses hyperparameter values from a string into a python map.

  `values` is a string containing comma-separated `name=value` pairs.
  For each pair, the value of the hyperparameter named `name` is set to
  `value`.

  If a hyperparameter name appears multiple times in `values`, a ValueError
  is raised (e.g. 'a=1,a=2').

  The `value` in `name=value` must follows the syntax according to the
  type of the parameter:

  *  Scalar integer: A Python-parsable integer point value.  E.g.: 1,
     100, -12.
  *  Scalar float: A Python-parsable floating point value.  E.g.: 1.0,
     -.54e89.
  *  Boolean: True, False, true, false, 1, or 0.
  *  Scalar string: A non-empty sequence of characters, possibly surrounded by
       double-quotes.  E.g.: foo, bar_1, "foo,bar".
  *  List: A comma separated list of scalar values of the parameter type
     enclosed in square brackets.  E.g.: [1,2,3], [1.0,1e-12], [high,low].

  Args:
    values: Comma separated list of `name=value` pairs where 'value' must follow
      the syntax described above.
    hparams_map: A mapping from abbreviation to field info, detailing the
      expected type information for each known field.

  Returns:
    A python map mapping each name to either:
    * A scalar value.
    * A list of scalar values.

  Raises:
    ValueError: If there is a problem with input.
    * If `values` cannot be parsed.
    * If the same hyperparameter is assigned to twice.
    * If an unknown hyperparameter is assigned to.
    * If a list is assigned to a scalar hyperparameter.
  """
  results_dictionary = {}
  pos = 0
  while pos < len(values):
    m = _PARAM_RE.match(values, pos)
    if not m:
      raise ValueError(f'Malformed hyperparameter value: {values[pos:]}')
    pos = m.end()
    # Parse the values.
    m_dict = m.groupdict()
    name = m_dict['name']
    if name not in hparams_map:
      raise ValueError(f'Unknown hyperparameter: {name}.')
    if name in results_dictionary:
      raise ValueError(f'Duplicate assignment to hyperparameter \'{name}\'')
    scalar_type = hparams_map[name].scalar_type
    is_list = hparams_map[name].is_list

    # Set up correct parsing function (depending on whether scalar_type is a
    # bool)
    def parse_bool(value):
      if value in ['true', 'True']:
        return True
      elif value in ['false', 'False']:
        return False
      else:
        try:
          return bool(int(value))
        except ValueError:
          raise ValueError(
              f'Could not parse {value} as a boolean for hyperparameter '
              f'{name}.')

    if scalar_type == bool:
      parse = parse_bool
    elif issubclass(scalar_type, enum.Enum):
      enum_type = scalar_type  # type: Type[enum.Enum]
      enum_value_type = type(list(enum_type.__members__.values())[0].value)
      enum_value_parser = (
          parse_bool if enum_value_type == bool else enum_value_type)
      parse = lambda x: enum_type(enum_value_parser(x))
    else:
      parse = scalar_type

    # If a single value is provided
    if m_dict['val'] is not None:
      results_dictionary[name] = parse(m_dict['val'])
      if is_list:
        results_dictionary[name] = [results_dictionary[name]]

    # A quoted string, so trim the quotes.
    elif m_dict['strval'] is not None:
      results_dictionary[name] = parse(m_dict['strval'][1:-1])
      if is_list:
        results_dictionary[name] = [results_dictionary[name]]

    # If the assigned value is a list:
    elif m_dict['vals'] is not None:
      if not is_list:
        raise ValueError(f'Expected single value for hyperparameter {name}, '
                         f'but found {m_dict["vals"]}')
      list_str = m_dict['vals']
      if list_str[0] == '[' and list_str[-1] == ']':
        list_str = list_str[1:-1]
      elements = list(csv.reader([list_str]))[0]
      results_dictionary[name] = [parse(e.strip()) for e in elements]

    else:  # Not assigned a list or value
      raise ValueError(f'Found empty value for hyperparameter {name}.')

  return results_dictionary


def _build_hparams_map(hparams_class):
  """Constructs a map representing the metadata of an hparams class.

  Contains the information needed to serialize, deserialize, and validate fields
  of the class.

  Includes information for fields in the class passed in, as well as any nested
  hparams class fields that are created using hparam.nest(), recursively.

  Args:
    hparams_class: A class that is decorated with @hparam.s.

  Returns:
    A mapping per field of abbreviation (used for serialization) to field
    metatdata.

  Raises:
    TypeError:
      * if `hparams_class` was not decorated with @hparam.s.
      * if a nested class was not decorated with @hparam.s.
      * if `hparams_class` has a field that was not created using @hparam.field
        or @hparam.nest.
    KeyError:
      * if two fields in `hparams_class` or any of its nested classes use the
        same abbreviation.
  """
  if not attr.has(hparams_class):
    raise TypeError(
        'Inputs to _build_hparams_map should be classes decorated with '
        '@hparam.s')

  hparams_map = {}
  for attribute in attr.fields(hparams_class.__class__):
    path = [attribute.name]
    default = attribute.default
    # pytype: disable=invalid-annotation
    factory_type = attr.Factory  # type: Type[attr.Factory]  # pytype: disable=annotation-type-mismatch
    # pytype: enable=invalid-annotation
    if isinstance(default, factory_type):
      default = default.factory()
    if attr.has(default):  # Nested.
      if '__hparams_map__' not in default.__dict__:
        raise TypeError('Nested hparams classes must also be decorated with '
                        '@hparam.s.')
      submap = default.__hparams_map__
      prefix = ''
      if _PREFIX_KEY in attribute.metadata:
        prefix = attribute.metadata[_PREFIX_KEY] or ''
      for key, value in submap.items():
        abbrev = prefix + key
        if abbrev in hparams_map:
          raise KeyError(f'Abbrev {abbrev} is duplicated.')
        updated = copy.copy(value)
        updated.path = path + value.path
        hparams_map[abbrev] = updated
    else:  # Leaf node.
      if attribute.name == _SERIALIZED_ARG:
        continue
      if _ABBREV_KEY not in attribute.metadata:
        raise AssertionError(
            f'Could not find hparam metadata for field {attribute.name}. Did '
            'you create a field without using hparam.field()?')
      abbrev = attribute.metadata[_ABBREV_KEY]
      if abbrev in hparams_map:
        raise KeyError(f'Abbrev {abbrev} is duplicated.')
      field_info = _FieldInfo(
          path=path,
          scalar_type=attribute.metadata[_SCALAR_TYPE_KEY],
          is_list=attribute.metadata[_IS_LIST_KEY],
          default_value=attribute.converter(default))
      hparams_map[abbrev] = field_info
  return hparams_map


def s(wrapped, *attrs_args,
      **attrs_kwargs):
  """A class decorator for creating an hparams class.

  The resulting class is based on `attr` under the covers, but this wrapper
  provides additional features, such as serialization and deserialization in a
  format that is compatible with (now defunct) tensorflow.HParams, runtime type
  checking, implicit casting where safe to do so (int->float, scalar->list).
  Unlike tensorflow.HParams, this supports hierarchical nesting of parameters
  for better organization, aliasing parameters to short abbreviations for
  compact serialization while maintaining code readability, and support for Enum
  values.

  Example usage:
    @hparam.s
    class MyNestedHParams:
      learning_rate: float = hparam.field(abbrev='lr', default=0.1)
      layer_sizes: List[int] = hparam.field(abbrev='ls', default=[256, 64, 32])

    @hparam.s
    class MyHParams:
      nested_params: MyNestedHParams = hparam.nest(MyNestedHParams)
      non_nested_param: int = hparam.field(abbrev='nn', default=0)

  Args:
    wrapped: The class being decorated. It should only contain fields created
      using `hparam.field()` and `hparam.nest()`.
    *attrs_args: Arguments passed on to `attr.s`.
    **attrs_kwargs: Keyword arguments passed on to `attr.s`.

  Returns:
    The class with the modifications needed to support the additional hparams
    features.
  """

  def attrs_post_init(self):
    self.__hparams_map__ = _build_hparams_map(self)
    serialized = getattr(self, _SERIALIZED_ARG, '')
    if serialized:
      self.parse(serialized)
      setattr(self, _SERIALIZED_ARG, '')

  def setattr_impl(self, name, value):
    ready = '__hparams_map__' in self.__dict__
    # Don't mess with setattrs that are called by the attrs framework or during
    # __init__.
    if ready:
      attribute = getattr(attr.fields(self.__class__), name)
      if attribute and attribute.converter:
        value = attribute.converter(value)
    super(wrapped, self).__setattr__(name, value)  # pytype: disable=wrong-arg-types

  def serialize(self, readable=False, omit_defaults=False):
    if readable:
      d = attr.asdict(self, filter=lambda a, _: a.name != _SERIALIZED_ARG)
      return json.dumps(d, default=str)
    else:
      serialized = ''
      for key, field_info in self.__hparams_map__.items():
        parent = self
        for childname in field_info.path:
          parent = getattr(parent, childname)
        if not omit_defaults or parent != field_info.default_value:
          value = _serialize_value(parent, field_info)
          serialized += f'{key}={value},'
      return serialized[:-1]  # Prune trailing comma.

  def parse(self, serialized):
    parsed_fields = _parse_serialized(serialized, self.__hparams_map__)
    for abbrev, value in parsed_fields.items():
      field_info = self.__hparams_map__[abbrev]
      parent = self
      for i, childname in enumerate(field_info.path):
        if i != len(field_info.path) - 1:
          parent = getattr(parent, childname)
        else:
          try:
            setattr(parent, childname, value)
          except:
            error_field = '.'.join(field_info.path)
            raise RuntimeError(f'Error trying to assign value {value} to field '
                               f'{error_field}.')

  wrapped.__hparams_class__ = True
  setattr(
      wrapped, _SERIALIZED_ARG,
      attr.ib(
          default='',
          type=str,
          kw_only=False,
          validator=attr.validators.instance_of(six.string_types),
          repr=False))
  wrapped.__attrs_post_init__ = attrs_post_init
  wrapped.__setattr__ = setattr_impl
  wrapped.serialize = serialize
  wrapped.parse = parse
  wrapped = attr.s(wrapped, *attrs_args, **attrs_kwargs)  # pytype: disable=wrong-arg-types  # attr-stubs
  return wrapped
