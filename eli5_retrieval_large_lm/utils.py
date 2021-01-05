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

"""Generic utilities.
"""
import contextlib
import itertools
import json
import logging
import operator
import os
import pathlib
import shutil
import tempfile
import textwrap
import time
from typing import Any, Callable, Container, Generator, Iterable, Optional
from typing import Type, Union

from absl import flags
import colorama
import dataclasses
import psutil

# pylint: enable=g-import-not-at-top
EXTERNAL = True
# pylint: disable=g-import-not-at-top
if EXTERNAL:
  from tensorflow.compat.v1 import gfile  # pytype: disable=import-error
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS
LOGGER = logging.getLogger(__name__)
PathType = Union[pathlib.Path, str]


@dataclasses.dataclass
class TimeStamp:
  """Simple dataclass to represent a timestamp."""
  hours: int
  minutes: int
  seconds: int
  milliseconds: int

  def format(self):
    return (f"{str(self.hours).zfill(2)}:{str(self.minutes).zfill(2)}:"
            f"{str(self.seconds).zfill(2)}.{str(self.milliseconds).zfill(3)}")

  @classmethod
  def from_seconds(cls, duration):
    hours = int(duration // 3600)
    duration %= 3600
    minutes = int(duration // 60)
    duration %= 60
    seconds = int(duration)
    duration %= 1
    milliseconds = int(1000 * (duration))
    return TimeStamp(hours, minutes, seconds, milliseconds)


@contextlib.contextmanager
def log_duration(logger, function_name, task_name,
                 level = logging.DEBUG):
  """With statement (context manager) to log the duration of a code block.

  Arguments:
    logger: The logger to use to do the logging.
    function_name: The function from which log_duration is called.
    task_name: A short description of the task being monitored.
    level: Logging level to use.
  Yields:
    None
  """
  # Do this at the entry inside of the with statement
  logger.log(level, "(%(function_name)s): Starting task  "
                    "`%(color)s%(task_name)s%(reset)s`.",
             dict(function_name=function_name, task_name=task_name,
                  color=colorama.Fore.CYAN, reset=colorama.Style.RESET_ALL))
  start = time.time()
  yield
  # Do this at the exit of the with statement
  duration = time.time() - start
  timestamp = TimeStamp.from_seconds(duration)
  logger.log(level,
             "(%(function_name)s): Done with task "
             "`%(cyan)s%(task_name)s%(style_reset)s`. "
             " Took %(color)s`%(ts)s`%(style_reset)s",
             dict(function_name=function_name, task_name=task_name,
                  color=colorama.Fore.GREEN, ts=timestamp.format(),
                  style_reset=colorama.Style.RESET_ALL,
                  cyan=colorama.Fore.CYAN))


def copy_to_tmp(in_file):
  """Copies a file to a tempfile.

  The point of this is to copy small files from CNS to tempdirs on
  the client when using code that's that hasn't been Google-ified yet.
  Examples of files are the vocab and config files of the Hugging Face
  tokenizer.

  Arguments:
    in_file: Path to the object to be copied, likely in CNS
  Returns:
    Path where the object ended up (inside of the tempdir).
  """
  # We just want to use Python's safe tempfile name generation algorithm
  with tempfile.NamedTemporaryFile(delete=False) as f_out:
    target_path = os.path.join(tempfile.gettempdir(), f_out.name)
  gfile.Copy(in_file, target_path, overwrite=True)
  return target_path


def check_equal(a, b):
  """Checks if two values are equal.

  Args:
    a: First value.
    b: Second value.

  Returns:
    Always returns `None`.

  Raises:
    RuntimeError: If the values aren't equal.
  """
  check_operator(operator.eq, a, b)


def check_contained(unit, container):
  check_operator(operator.contains, container, unit)


def check_operator(op, a, b):
  """Checks an operator with two arguments.

  Args:
    op: Comparison function.
    a: First value.
    b: Second value.

  Returns:
    Always returns `None`.

  Raises:
    RuntimeError: If the values aren't equal.
  """
  if not op(a, b):
    raise RuntimeError("Operator test failed.\n"
                       f"Operator:    {op}\n"
                       f"left arg:    {a}\n"
                       f"right arg:   {b}")


def check_isinstance(obj, type_):
  if not isinstance(obj, type_):
    raise RuntimeError("Failed isinstance check.\n"
                       f"\tExpected: {type_}\n"
                       f"\tGot:      {type(obj)}")


def check_exists(path):
  """Check if a directory or a path is at the received path.

  Arguments:
    path: The path to check.
  Returns:
    Nothing.
  Raises:
    RuntimeError: Raised if nothing exists at the received path.
  """
  if path is None:
    raise RuntimeError("Got None instead of a valid path.")

  if not gfile.Exists(path):
    raise RuntimeError(f"File path `{path}` doesn't exist.")


def check_glob_prefix(prefix):
  """Verifies that there is at least one match for a glob prefix.

  Args:
    prefix: Glob prefix to check.

  Returns:
    None

  Raises:
    RuntimeError: If there are no matches or the parent path doesn't exist.
  """
  if prefix is None:
    raise RuntimeError("Got None instead of a valid glob prefix.")

  path = pathlib.Path(prefix)
  # Check if the prefix path FLAGS.source_embeddings_prefix has at least one
  # match. This methods stays fast even if there are a trillion matches.
  # Definitely unnecessary. (len(list(matches)) > 0 felt ugly.)
  if not gfile.Exists(path.parent):
    raise RuntimeError(f"The parent of the glob prefix didn't exist:\n"
                       f" - Glob prefix: {path}\n"
                       f" - Glob parent: {path.parent}")
  matches = path.parent.glob(path.name + "*")
  at_least_one = len(list(itertools.islice(matches, 0, 1))) > 0  # pylint: disable=g-explicit-length-test
  if not at_least_one:
    raise RuntimeError("No matches to the globbing prefix:\n{prefix}")


def check_not_none(obj):
  if obj is None:
    raise RuntimeError("Object was None.")


def from_json_file(path):
  """Reads from a json file.

  Args:
    path: Path to read from.

  Returns:
    The object read from the json file.
  """
  with gfile.GFile(str(path)) as fin:
    return json.loads(fin.read())


def to_json_file(path, obj, indent = 4):
  """Saves to a json file.

  Args:
    path: Where to save.
    obj: The object to save

  Returns:
    None
  """
  with gfile.GFile(str(path), "w") as fout:
    fout.write(json.dumps(obj, indent=indent))


def log_module_args(
    logger, module_name,
    level = logging.DEBUG, sort = True
):
  """Logs the list of flags defined in a module, as well as their value.

  Args:
    logger: Instance of the logger to use for logging.
    module_name: Name of the module from which to print the args.
    level: Logging level to use.
    sort: Whether to sort the flags

  Returns:
    None
  """
  flags_ = FLAGS.flags_by_module_dict()[module_name]
  if sort:
    flags_.sort(key=lambda flag: flag.name)
  # `json.dumps` formats dicts in a nice way when indent is specified.
  content = "\n" + json.dumps({flag.name: flag.value for flag in flags_
                               }, indent=4)
  if logger is not None:
    logger.log(level, content)
  return content


def term_size(default_cols = 80):
  return shutil.get_terminal_size((default_cols, 20)).columns


def wrap_iterable(
    iterable, numbers = False, length = None
):
  """Takes a number of long lines, and wraps them to the terminal length.

  Adds dashes by default, numbers the lines if numbers=True. The length defaults
  to the length of the terminal at the moment the function is called. Defaults
  to 80 wide if not currently in a terminal.

  Args:
    iterable: The object with the text instances.
    numbers: Whether to use line numbers.

  Returns:

  """
  if length is None:
    # Can't set it as default as default value are evaluated at function
    # definition time.
    length = term_size(120)
  if numbers:
    wrapped = (textwrap.fill(str(line), length, initial_indent=f" {i} - ",
                             subsequent_indent=" " * len(f" {i} - "))
               for i, line in enumerate(iterable))
  else:
    wrapped = (textwrap.fill(str(line), length, initial_indent=" - ",
                             subsequent_indent="   ") for line in iterable)
  return "\n".join(wrapped)


class MovingAverage:
  """Creates a simple EMA (exponential moving average).
  """

  def __init__(self, constant, settable_average = False):
    """Creates the EMA object.

    Args:
      constant: update constant. The alpha in
        https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    """
    constant = float(constant)
    check_operator(operator.lt, constant, 1)
    self._constant = constant
    self._average = None
    self.settable_average = settable_average

  def update(self, value):
    value = float(value)
    if self._average is None:
      self._average = value
    else:
      self._average = (self._constant * self._average
                       + (1 - self._constant) * value)

  @property
  def average(self):
    return self._average

  @average.setter
  def average(self, value):
    if self.settable_average:
      self._average = float(value)
    else:
      raise RuntimeError("The value of average should not be set this way")

  def __repr__(self):
    return f"<MovingAverage: self.average={self._average}>"

  def __str__(self):
    return str(self._average)


class FlagChoices:
  """Adds a .choices function with the choices for the Flag.

  Example:
    >>> class DirectionChoices(FlagChoices):
    >>>     north = "north"
    >>>     south = "south"
    >>>     east = "east"
    >>>     west = "west"
    >>> # ...
    >>> flags.DEFINE_enum("direction", DirectionChoices.north,
    >>>                    DirectionChoices.choices(), "In which direciton do"
    >>>                                                " you want to go.")
    >>> # ...
    >>> # other case
    >>> if argument_value not in DirectionChoices:
    >>>     raise ValueError(f"Value {} not in DirectionChoices:"
    >>>                      f"{DirectionChoices.choices}")

  """

  @classmethod
  def choices(cls):
    if getattr(cls, "_choices", None) is None:
      cls._choices = frozenset([
          v for k, v in vars(cls).items()
          if k != "choices" and not k.startswith("_")
      ])
    return cls._choices


def print_mem(description, logger):
  """Prints the current memory use of the main process."""
  process = psutil.Process(os.getpid())
  logger.debug(
      "MEM USAGE:\n"
      " - Usage: %(mem)f GB\n"
      " - Description: %(yellow)s%(description)s%(reset)s",
      dict(mem=process.memory_info().rss / 1E9, description=description,
           yellow=colorama.Fore.YELLOW, reset=colorama.Style.RESET_ALL
           ))