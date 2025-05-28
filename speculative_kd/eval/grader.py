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

"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
"""

import contextlib
import math
from math import isclose
import re
import signal
from typing import Union

from sympy import N
from sympy import simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def is_digit(s):
  try:
    if "{,}" in str(s):
      num = float(str(s).replace("{,}", ""))
      return True, num

    num = float(str(s).replace(",", ""))
    return True, num
  except ValueError:
    return False, None


def normalize(answer, pi):
  """Normalize the answer string."""
  # checking if answer is $<number> and removing $ in that case to compare
  if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
    return answer[1:]

  # checking if answer is <number>% or <number>\\% and removing %
  if isinstance(answer, str) and (
      bool(re.match(r"^\d+(\.\d+)?%$", answer))
      or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
  ):
    return answer.replace("\\%", "").replace("%", "")

  # handle base
  answer = handle_base(answer)

  # handle pi
  answer = handle_pi(answer, pi)

  return answer


def handle_base(x):
  if isinstance(x, str) and "_" in x:
    # Due to base
    x = x.split("_")[0]
    x = float(x)
    return int(x)
  return x


def handle_pi(string, pi):
  """Handles pi in the string."""

  if isinstance(string, str) and "\\pi" in string:
    # Find the first occurrence of "\pi"
    idx = string.find("\\pi")

    # Iterate over the string and find all occurrences of "\pi" with a valid
    # previous character
    while idx != -1:

      if idx > 0 and string[idx-1].isdigit():
        # Replace "\pi" with "*math.pi" if the previous character is a digit
        string = string[:idx] + f"*{pi}" + string[idx+3:]
      else:
        # Replace "\pi" with "1*math.pi" if the previous character isn't a digit
        string = string[:idx] + f"1*{pi}" + string[idx+3:]

        # Find the next occurrence of "\pi"
        idx = string.find(r"\pi", idx + 1)

    # Evaluate the expression using eval() function
    try:
      string = eval(string)
    except:
      pass

  return string


def math_equal(
    prediction,
    reference,
    include_percentage = True,
    tolerance = 1e-4,
    timeout = 10.0,
    pi = math.pi,
):
  """Exact match of math if and only if: 1."""
  prediction = normalize(prediction, pi)
  reference = normalize(reference, pi)

  if (
      isinstance(prediction, str) and len(prediction) > 1000
  ):  # handling weird corner-cases
    prediction = prediction[:1000]

  # 0. string comparison
  if isinstance(prediction, str) and isinstance(reference, str):
    if prediction.strip().lower() == reference.strip().lower():
      return True
    if prediction.replace(" ", "") == reference.replace(" ", ""):
      return True

  try:  # 1. numerical equal
    if is_digit(prediction)[0] and is_digit(reference)[0]:
      prediction = is_digit(prediction)[1]
      reference = is_digit(reference)[1]
      # number questions
      if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
      else:
        gt_result = [reference]
      for item in gt_result:
        try:
          if isclose(item, prediction, rel_tol=tolerance):
            return True
        except TypeError:
          continue
      return False
  except (ValueError, TypeError):
    pass

  if not prediction and prediction not in [0, False]:
    return False

  # 2. symbolic equal
  reference = str(reference).strip()
  prediction = str(prediction).strip()

  ## deal with [], (), {}
  prediction = format_intervals(prediction)

  pred_str, ref_str = prediction, reference
  if (
      prediction.startswith("[")
      and prediction.endswith("]")
      and not reference.startswith("(")
  ) or (
      prediction.startswith("(")
      and prediction.endswith(")")
      and not reference.startswith("[")
  ):
    pred_str = pred_str.strip("[]()")
    ref_str = ref_str.strip("[]()")
  for s in ["{", "}", "(", ")"]:
    ref_str = ref_str.replace(s, "")
    pred_str = pred_str.replace(s, "")
  if pred_str == ref_str:
    return True

  ## [a, b] vs. [c, d], return a==c and b==d
  if (
      prediction
      and reference
      and prediction[0] in "(["
      and prediction[-1] in ")]"
      and prediction[0] == reference[0]
      and prediction[-1] == reference[-1]
  ):
    pred_parts = prediction[1:-1].split(",")
    ref_parts = reference[1:-1].split(",")
    if len(pred_parts) == len(ref_parts):
      if all([
          math_equal(pred_pt, ref_pt, include_percentage, tolerance)
          for pred_pt, ref_pt in zip(pred_parts, ref_parts)
      ]):
        return True

  if "," in prediction and "," in reference:
    pred_parts = [item.strip() for item in prediction.split(",")]
    ref_parts = [item.strip() for item in reference.split(",")]

    if len(pred_parts) == len(ref_parts):
      if all([
          math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance)
          for i in range(len(pred_parts))
      ]):
        return True
      else:
        return False

  # if we have point == tuple of values
  if (
      prediction.startswith("Point")
      and reference[0] == "("
      and reference[-1] == ")"
  ):
    pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
    ref_parts = reference[1:-1].split(",")
    if len(pred_parts) == len(ref_parts):
      if all([
          math_equal(pred_pt, ref_pt, include_percentage, tolerance)
          for pred_pt, ref_pt in zip(pred_parts, ref_parts)
      ]):
        return True

  # if reference is a matrix
  if "\begin{pmatrix}" in reference and prediction.startswith("Matrix"):
    try:
      pred_matrix = parse_expr(prediction)
      ref_matrix_items = reference.split()[1:-1:2]
      if len(pred_matrix) == len(ref_matrix_items):
        if all([
            math_equal(pred, ref, include_percentage, tolerance)
            for ref, pred in zip(ref_matrix_items, pred_matrix)
        ]):
          return True
    except (ValueError, TypeError):
      pass
  elif (
      "\begin{pmatrix}" in reference
      and prediction.startswith("[")
      and prediction.endswith("]")
  ):
    if isinstance(eval(prediction), list):
      try:
        pred_matrix = eval(prediction)
        # ref_matrix_items = reference.split()[1:-1:2]
        ref_matrix_items = (
            reference.lstrip("\\begin{pmatrix}")
            .lstrip("\begin{pmatrix}")
            .rstrip("\\end{pmatrix}")
            .rstrip("\end{pmatrix}")
        )
        ref_matrix_items = ref_matrix_items.split("\\")
        ref_matrix_items = [
            row.split("&") if "&" in row else row for row in ref_matrix_items
        ]
        if len(pred_matrix) == len(ref_matrix_items):
          if all([
              math_equal(pred, ref, include_percentage, tolerance)
              for ref, pred in zip(ref_matrix_items, pred_matrix)
          ]):
            return True
      except (ValueError, TypeError):
        pass

  return symbolic_equal(prediction, reference, tolerance, timeout)


def symbolic_equal(a, b, tolerance, timeout=10.0):
  """Symbolic comparison of two expressions."""
  def _parse(s):
    for f in [parse_expr, parse_latex]:
      try:
        with time_limit(timeout):
          return f(s)
      except (ValueError, TypeError):
        pass
    return s

  a = _parse(a)
  b = _parse(b)

  try:
    with time_limit(timeout):
      if simplify(a - b) == 0:
        return True
  except (ValueError, TypeError):
    pass

  try:
    with time_limit(timeout):
      if isclose(N(a), N(b), rel_tol=tolerance):
        return True
  except (ValueError, TypeError):
    pass
  return False


def extract_answer(string):
  r"""Extract Answer String from \\boxed expression."""
  idx = string.rfind("\\boxed")
  if idx < 0:
    idx = string.rfind("\\fbox")
    if idx < 0:
      return None

  i = idx
  right_brace_idx = None
  num_left_braces_open = 0
  while i < len(string):
    if string[i] == "{":
      num_left_braces_open += 1
    if string[i] == "}":
      num_left_braces_open -= 1
      if num_left_braces_open == 0:
        right_brace_idx = i
        break
    i += 1

  if right_brace_idx is None:
    retval = None
  else:
    retval = string[idx : right_brace_idx + 1]

  if retval:
    left = "\\boxed{"
    try:
      assert retval[: len(left)] == left
      assert retval[-1] == "}"
      return retval[len(left) : -1]
    except AssertionError:
      return None

  return None


class TimeoutException(Exception):
  pass


@contextlib.contextmanager
def time_limit(seconds):
  def signal_handler():
    raise TimeoutException("Timed out!")

  signal.setitimer(signal.ITIMER_REAL, seconds)
  signal.signal(signal.SIGALRM, signal_handler)
  try:
    yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)


def format_intervals(prediction):
  """Formats interval strings to a standard format."""
  patterns = {
      "Interval(": r"^Interval\((.*)\)$",
      "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
      "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
      "Interval.open(": r"^Interval\.open\((.*)\)$",
  }

  for key, pattern in patterns.items():
    match = re.match(pattern, prediction)
    if match:
      inner_content = match.group(1)

      if key == "Interval(":  # Intarval(a, b) == [a, b]
        return f"[{inner_content}]"
      elif key == "Interval.Ropen(":  # Intarval.Ropen(a, b) == [a, b)
        return f"[{inner_content})"
      elif key == "Interval.Lopen(":  # Intarval.Lopen(a, b) == (a, b]
        return f"({inner_content}]"
      elif key == "Interval.open(":  # Intarval.open(a, b) == (a, b)
        return f"({inner_content})"

  return prediction


def _test_math_equal_simple():
  """Tests math_equal function."""
  ref = "6,-2"
  pred = "6"
  print(math_equal(ref, pred))


def _test_math_equal_pi():
  """Tests math_equal function with pi."""
  pi = math.pi
  ref = r"900\pi"
  pred = 812.0
  print(math_equal(pred, ref, pi=pi))

  ref = r"25\pi"
  pred = 78.5
  print(math_equal(pred, ref, pi=pi))

  ref = r"90\pi"
  pred = 282.6
  print(math_equal(pred, ref, pi=pi))

  ref = r"24+4\pi"
  pred = 36.57142857142857
  print(math_equal(pred, ref, pi=pi))

  ref = r"9\pi"
  pred = 28.274309999999993
  print(math_equal(pred, ref, pi=pi))


def _test_math_equal_matrix():
  """Tests math_equal function with matrices."""
  ref = r"\begin{pmatrix}0&1\\1&0\end{pmatrix}"
  # ref=ref.split()[1:-1:2]
  pred = [[0, 1], [1, 0]]
  print(math_equal(pred, ref))


if __name__ == "__main__":
  _test_math_equal_simple()
  _test_math_equal_pi()
  _test_math_equal_matrix()
