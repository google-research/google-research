# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Tests for llm_utils.py."""

import textwrap

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from latent_programmer.spec_decomposition import llm_utils

DEEPCODER_DATASET = tf.data.Dataset.from_tensors({
    'inputs': tf.constant(['x0 = [ ] | x1 = [ 0 ]',
                           'x0 = [ 1 0 6 9 1 ] | x1 = [ 9 ]',
                           'x0 = [ 3 7 1 4 ] | x1 = [ -3 -1 ]']),
    'outputs': tf.constant(['[ ]', '[ 9 4 6 9 1 ]', '[ 4 7 1 4 ]']),
    'program': tf.constant('x0 = INPUT | x1 = INPUT | x2 = Sort x0 | '
                           'x3 = Reverse x2 | x4 = Map (/3) x3 | '
                           'x5 = Map (**2) x4 | x6 = ZipWith (max) x0 x5'),
})
DEEPCODER_EXAMPLE = llm_utils.DatasetElement(
    inputs={'x0': [[], [1, 0, 6, 9, 1], [3, 7, 1, 4]],
            'x1': [[0], [9], [-3, -1]]},
    outputs=[[], [9, 4, 6, 9, 1], [4, 7, 1, 4]],
    dsl_program=('x0 = INPUT | x1 = INPUT | x2 = Sort x0 | '
                 'x3 = Reverse x2 | x4 = Map (/3) x3 | '
                 'x5 = Map (**2) x4 | x6 = ZipWith (max) x0 x5'),
    python_program='''
def program(x0, x1):
  x2 = dsl.Sort(x0)
  x3 = dsl.Reverse(x2)
  x4 = dsl.Map(dsl.DIV_THREE, x3)
  x5 = dsl.Map(dsl.SQUARE, x4)
  x6 = dsl.ZipWith(dsl.MAX, x0, x5)
  return x6
'''.strip(),
)
DEEPCODER_EXAMPLE_V2 = llm_utils.DatasetElement(
    inputs={'x0': [[], [1, 0, 6, 9, 1], [3, 7, 1, 4]],
            'x1': [[0], [9], [-3, -1]]},
    outputs=[[], [9, 4, 6, 9, 1], [4, 7, 1, 4]],
    dsl_program=('x0 = INPUT | x1 = INPUT | x2 = Sort x0 | '
                 'x3 = Reverse x2 | x4 = Map (/3) x3 | '
                 'x5 = Map (**2) x4 | x6 = ZipWith (max) x0 x5'),
    python_program='''
def program(x0, x1):
  x2 = sorted(x0)
  x3 = list(reversed(x2))
  x4 = dsl.Map(dsl.DIV_THREE, x3)
  x5 = dsl.Map(dsl.SQUARE, x4)
  x6 = dsl.ZipWith(dsl.MAX, x0, x5)
  return x6
'''.strip(),
)
DEEPCODER_EXAMPLE_V3 = llm_utils.DatasetElement(
    inputs={'x0': [[], [1, 0, 6, 9, 1], [3, 7, 1, 4]],
            'x1': [[0], [9], [-3, -1]]},
    outputs=[[], [9, 4, 6, 9, 1], [4, 7, 1, 4]],
    dsl_program=('x0 = INPUT | x1 = INPUT | x2 = Sort x0 | '
                 'x3 = Reverse x2 | x4 = Map (/3) x3 | '
                 'x5 = Map (**2) x4 | x6 = ZipWith (max) x0 x5'),
    python_program='''
def program(x0, x1):
  x2 = sorted(x0)
  x3 = list(reversed(x2))
  x4 = [dsl.DIV_THREE(x) for x in x3]
  x5 = [dsl.SQUARE(x) for x in x4]
  x6 = [dsl.MAX(x, y) for (x, y) in zip(x0, x5)]
  return x6
'''.strip(),
)
DEEPCODER_EXAMPLE_V4 = llm_utils.DatasetElement(
    inputs={'x0': [[], [1, 0, 6, 9, 1], [3, 7, 1, 4]],
            'x1': [[0], [9], [-3, -1]]},
    outputs=[[], [9, 4, 6, 9, 1], [4, 7, 1, 4]],
    dsl_program=('x0 = INPUT | x1 = INPUT | x2 = Sort x0 | '
                 'x3 = Reverse x2 | x4 = Map (/3) x3 | '
                 'x5 = Map (**2) x4 | x6 = ZipWith (max) x0 x5'),
    python_program='''
def program(x0, x1):
  x2 = sorted(x0)
  x3 = list(reversed(x2))
  x4 = [x // 3 for x in x3]
  x5 = [x ** 2 for x in x4]
  x6 = [max(x, y) for (x, y) in zip(x0, x5)]
  return x6
'''.strip(),
)

DEEPCODER_TEST_PROBLEM = llm_utils.DatasetElement(
    inputs={'x0': [[1, 2, 3], [10, -10], [45]]},
    outputs=[6, 0, 45],
    dsl_program=None,  # Unused.
    python_program=None,  # Unused.
)

ROBUSTFILL_DATASET = tf.data.Dataset.from_tensors({
    'inputs': ['#My##:Gxbo[Ned[Er%', '#%$Ua.Qaeq?Opa%Kcr#',
               "%{Eos#(Mdjt#'Yi{Oclf", '%##Tq@Fh#Xza#?Fdlu'],
    'outputs': ['k[MY##:GXBO[NED[ER%8y##:Gxbo[Ned[',
                'kK%$UA.QAEQ?OPA%KCR#8aUa.Qaeq?Opa%',
                "kO{EOS#(MDJT#'YI{OCLF8osos#(Mdjt#'Yi",
                'kF##TQ@FH#XZA#?FDLU8qTq@Fh#Xza#?F'],
    'program': ('4 29|7 109 211|3 8 111 17 109 216|'
                '3 15 109 216 79 7 106 216|5 219 230'),
})
ROBUSTFILL_EXAMPLE = llm_utils.DatasetElement(
    inputs=['#My##:Gxbo[Ned[Er%', '#%$Ua.Qaeq?Opa%Kcr#',
            "%{Eos#(Mdjt#'Yi{Oclf", '%##Tq@Fh#Xza#?Fdlu'],
    outputs=['k[MY##:GXBO[NED[ER%8y##:Gxbo[Ned[',
             'kK%$UA.QAEQ?OPA%KCR#8aUa.Qaeq?Opa%',
             "kO{EOS#(MDJT#'YI{OCLF8osos#(Mdjt#'Yi",
             'kF##TQ@FH#XZA#?FDLU8qTq@Fh#Xza#?F'],
    dsl_program=('4 29|7 109 211|3 8 111 17 109 216|'
                 '3 15 109 216 79 7 106 216|5 219 230'),
    python_program='''
def program(x):
  parts = [
      dsl.Const('k'),
      dsl.GetToken(x, dsl.Type.CHAR, -4),
      dsl.ToCase(dsl.Remove(x, dsl.Type.CHAR, 1), dsl.Case.ALL_CAPS),
      dsl.Substitute(dsl.GetToken(x, dsl.Type.PROP_CASE, 1), dsl.Type.CHAR, 1, '8'),
      dsl.SubStr(x, 4, 15),
  ]
  return ''.join(parts)
'''.strip(),
)
ROBUSTFILL_TEST_PROBLEM = llm_utils.DatasetElement(
    inputs=['apple', 'banana', 'clementine', 'durian'],
    outputs=['Apple!', 'Banana!', 'Clementine!', 'Durian!'],
    dsl_program=None,  # Unused.
    python_program=None,  # Unused.
)


class LlmUtilsTest(parameterized.TestCase):

  def test_to_python_form(self):
    original = 'x1 = [ 1 2 ] | x2 = 3'
    python_form = 'x1 = [1, 2], x2 = 3'
    self.assertEqual(llm_utils.to_python_form(original), python_form)

  @parameterized.named_parameters(
      ('1', 1, DEEPCODER_EXAMPLE),
      ('2', 2, DEEPCODER_EXAMPLE_V2),
      ('3', 3, DEEPCODER_EXAMPLE_V3),
      ('4', 4, DEEPCODER_EXAMPLE_V4),
      ('5', 5, DEEPCODER_EXAMPLE_V3),  # Same program as v3.
  )
  def test_parse_dataset_deepcoder(self, version, expected_example):
    self.assertEqual(
        llm_utils.parse_dataset(DEEPCODER_DATASET, 'deepcoder',
                                version=version),
        [expected_example])

  @parameterized.named_parameters(
      ('1', DEEPCODER_EXAMPLE),
      ('2', DEEPCODER_EXAMPLE_V2),
      ('3', DEEPCODER_EXAMPLE_V3),
      ('4', DEEPCODER_EXAMPLE_V4),
  )
  def test_run_program_deepcoder(self, example):
    self.assertEqual(
        llm_utils.run_program(example.python_program, example.inputs,
                              'deepcoder'),
        example.outputs)

  def test_get_num_examples_deepcoder(self):
    self.assertEqual(
        llm_utils.get_num_examples(DEEPCODER_EXAMPLE.inputs, 'deepcoder'),
        3)

  def test_parse_dataset_robustfill(self):
    self.assertEqual(
        llm_utils.parse_dataset(ROBUSTFILL_DATASET, 'robustfill', version=1),
        [ROBUSTFILL_EXAMPLE])

  def test_run_program_robustfill(self):
    self.assertEqual(
        llm_utils.run_program(ROBUSTFILL_EXAMPLE.python_program,
                              ROBUSTFILL_EXAMPLE.inputs, 'robustfill'),
        ROBUSTFILL_EXAMPLE.outputs)

  def test_get_num_examples_robustfill(self):
    self.assertEqual(
        llm_utils.get_num_examples(ROBUSTFILL_EXAMPLE.inputs, 'robustfill'),
        4)

  def test_few_shot_prompt_deepcoder_version_1(self):
    prompt = llm_utils.few_shot_prompt(
        few_shot_examples=[DEEPCODER_EXAMPLE],
        test_problem=DEEPCODER_TEST_PROBLEM,
        dataset_type='deepcoder',
        version=1,
    )
    expected = '''
The `dsl` module is a custom library for manipulating lists of integers. It contains the following functions:

Head, Last, Take, Drop, Access, Minimum, Maximum, Reverse, Sort, Sum, Map, Filter, Count, ZipWith, Scanl1

Additionally, the module defines the following constants:

PLUS_ONE, MINUS_ONE, TIMES_TWO, DIV_TWO, NEGATE, SQUARE, TIMES_THREE, DIV_THREE, TIMES_FOUR, DIV_FOUR, IS_POSITIVE, IS_NEGATIVE, IS_EVEN, IS_ODD, ADD, SUBTRACT, MULTIPLY, MIN, MAX

Below are example programs using the `dsl` module, with input-output test cases illustrating their behavior.

Important: All programs begin with ```python and end with ``` alone.


Input-output test cases:
  * x0 = [], x1 = [0] --> []
  * x0 = [1, 0, 6, 9, 1], x1 = [9] --> [9, 4, 6, 9, 1]
  * x0 = [3, 7, 1, 4], x1 = [-3, -1] --> [4, 7, 1, 4]

Program:
```python
def program(x0, x1):
  x2 = dsl.Sort(x0)
  x3 = dsl.Reverse(x2)
  x4 = dsl.Map(dsl.DIV_THREE, x3)
  x5 = dsl.Map(dsl.SQUARE, x4)
  x6 = dsl.ZipWith(dsl.MAX, x0, x5)
  return x6
```


Input-output test cases:
  * x0 = [1, 2, 3] --> 6
  * x0 = [10, -10] --> 0
  * x0 = [45] --> 45

Program:
```python
'''.lstrip()
    self.assertEqual(prompt, expected)

  def test_few_shot_prompt_deepcoder_version_2(self):
    prompt = llm_utils.few_shot_prompt(
        few_shot_examples=[DEEPCODER_EXAMPLE_V2],
        test_problem=DEEPCODER_TEST_PROBLEM,
        dataset_type='deepcoder',
        version=2,
    )
    expected = '''
The `dsl` module is a custom library for manipulating lists of integers. It contains the following functions:

def Map(f, xs):
  return [f(x) for x in xs]

def Filter(f, xs):
  return [x for x in xs if f(x)]

def Count(f, xs):
  return len([x for x in xs if f(x)])

def ZipWith(f, xs, ys):
  return [f(x, y) for (x, y) in zip(xs, ys)]

def Scanl1(f, xs):
  ys = []
  for i, x in enumerate(xs):
    if i == 0:
      ys.append(x)
    else:
      ys.append(f(ys[-1], x))
  return ys

Additionally, the module defines the following constants:

PLUS_ONE = lambda x: x + 1
MINUS_ONE = lambda x: x - 1
TIMES_TWO = lambda x: x * 2
DIV_TWO = lambda x: x // 2
NEGATE = lambda x: -x
SQUARE = lambda x: x ** 2
TIMES_THREE = lambda x: x * 3
DIV_THREE = lambda x: x // 3
TIMES_FOUR = lambda x: x * 4
DIV_FOUR = lambda x: x // 4
IS_POSITIVE = lambda x: x > 0
IS_NEGATIVE = lambda x: x < 0
IS_EVEN = lambda x: x % 2 == 0
IS_ODD = lambda x: x % 2 == 1
ADD = lambda x, y: x + y
SUBTRACT = lambda x, y: x - y
MULTIPLY = lambda x, y: x * y
MIN = lambda x, y: min(x, y)
MAX = lambda x, y: max(x, y)

Below are example programs using the `dsl` module, with input-output test cases illustrating their behavior.

Important: All programs begin with ```python and end with ``` alone.


Input-output test cases:
  * x0 = [], x1 = [0] --> []
  * x0 = [1, 0, 6, 9, 1], x1 = [9] --> [9, 4, 6, 9, 1]
  * x0 = [3, 7, 1, 4], x1 = [-3, -1] --> [4, 7, 1, 4]

Program:
```python
def program(x0, x1):
  x2 = sorted(x0)
  x3 = list(reversed(x2))
  x4 = dsl.Map(dsl.DIV_THREE, x3)
  x5 = dsl.Map(dsl.SQUARE, x4)
  x6 = dsl.ZipWith(dsl.MAX, x0, x5)
  return x6
```


Input-output test cases:
  * x0 = [1, 2, 3] --> 6
  * x0 = [10, -10] --> 0
  * x0 = [45] --> 45

Program:
```python
'''.lstrip()
    self.assertEqual(prompt, expected)

  def test_few_shot_prompt_deepcoder_version_3(self):
    prompt = llm_utils.few_shot_prompt(
        few_shot_examples=[DEEPCODER_EXAMPLE_V3],
        test_problem=DEEPCODER_TEST_PROBLEM,
        dataset_type='deepcoder',
        version=3,
    )
    expected = '''
The `dsl` module is a custom library for manipulating lists of integers. It contains the following functions:

def Scanl1(f, xs):
  ys = []
  for i, x in enumerate(xs):
    if i == 0:
      ys.append(x)
    else:
      ys.append(f(ys[-1], x))
  return ys

Additionally, the module defines the following constants:

PLUS_ONE = lambda x: x + 1
MINUS_ONE = lambda x: x - 1
TIMES_TWO = lambda x: x * 2
DIV_TWO = lambda x: x // 2
NEGATE = lambda x: -x
SQUARE = lambda x: x ** 2
TIMES_THREE = lambda x: x * 3
DIV_THREE = lambda x: x // 3
TIMES_FOUR = lambda x: x * 4
DIV_FOUR = lambda x: x // 4
IS_POSITIVE = lambda x: x > 0
IS_NEGATIVE = lambda x: x < 0
IS_EVEN = lambda x: x % 2 == 0
IS_ODD = lambda x: x % 2 == 1
ADD = lambda x, y: x + y
SUBTRACT = lambda x, y: x - y
MULTIPLY = lambda x, y: x * y
MIN = lambda x, y: min(x, y)
MAX = lambda x, y: max(x, y)

Below are example programs using the `dsl` module, with input-output test cases illustrating their behavior.

Important: All programs begin with ```python and end with ``` alone.


Input-output test cases:
  * x0 = [], x1 = [0] --> []
  * x0 = [1, 0, 6, 9, 1], x1 = [9] --> [9, 4, 6, 9, 1]
  * x0 = [3, 7, 1, 4], x1 = [-3, -1] --> [4, 7, 1, 4]

Program:
```python
def program(x0, x1):
  x2 = sorted(x0)
  x3 = list(reversed(x2))
  x4 = [dsl.DIV_THREE(x) for x in x3]
  x5 = [dsl.SQUARE(x) for x in x4]
  x6 = [dsl.MAX(x, y) for (x, y) in zip(x0, x5)]
  return x6
```


Input-output test cases:
  * x0 = [1, 2, 3] --> 6
  * x0 = [10, -10] --> 0
  * x0 = [45] --> 45

Program:
```python
'''.lstrip()
    self.assertEqual(prompt, expected)

  def test_few_shot_prompt_deepcoder_version_4(self):
    prompt = llm_utils.few_shot_prompt(
        few_shot_examples=[DEEPCODER_EXAMPLE_V4],
        test_problem=DEEPCODER_TEST_PROBLEM,
        dataset_type='deepcoder',
        version=4,
    )
    expected = '''
The `dsl` module is a custom library for manipulating lists of integers. It contains the following functions:

def Scanl1(f, xs):
  ys = []
  for i, x in enumerate(xs):
    if i == 0:
      ys.append(x)
    else:
      ys.append(f(ys[-1], x))
  return ys

Below are example programs using the `dsl` module, with input-output test cases illustrating their behavior.

Important: All programs begin with ```python and end with ``` alone.


Input-output test cases:
  * x0 = [], x1 = [0] --> []
  * x0 = [1, 0, 6, 9, 1], x1 = [9] --> [9, 4, 6, 9, 1]
  * x0 = [3, 7, 1, 4], x1 = [-3, -1] --> [4, 7, 1, 4]

Program:
```python
def program(x0, x1):
  x2 = sorted(x0)
  x3 = list(reversed(x2))
  x4 = [x // 3 for x in x3]
  x5 = [x ** 2 for x in x4]
  x6 = [max(x, y) for (x, y) in zip(x0, x5)]
  return x6
```


Input-output test cases:
  * x0 = [1, 2, 3] --> 6
  * x0 = [10, -10] --> 0
  * x0 = [45] --> 45

Program:
```python
'''.lstrip()
    self.assertEqual(prompt, expected)

  def test_few_shot_prompt_deepcoder_version_5(self):
    few_shot_dataset = llm_utils.parse_dataset(
        llm_utils.get_handwritten_few_shot('deepcoder', 'NONE'),
        dataset_type='deepcoder',
        version=5)
    prompt = llm_utils.few_shot_prompt(
        few_shot_examples=few_shot_dataset[:2],
        test_problem=DEEPCODER_TEST_PROBLEM,
        dataset_type='deepcoder',
        version=5,
    )
    expected = '''
The `dsl` module is a custom library for manipulating lists of integers. It contains the following functions:

def Scanl1(f, xs):
  ys = []
  for i, x in enumerate(xs):
    if i == 0:
      ys.append(x)
    else:
      ys.append(f(ys[-1], x))
  return ys

Additionally, the module defines the following constants:

PLUS_ONE = lambda x: x + 1
MINUS_ONE = lambda x: x - 1
TIMES_TWO = lambda x: x * 2
DIV_TWO = lambda x: x // 2
NEGATE = lambda x: -x
SQUARE = lambda x: x ** 2
TIMES_THREE = lambda x: x * 3
DIV_THREE = lambda x: x // 3
TIMES_FOUR = lambda x: x * 4
DIV_FOUR = lambda x: x // 4
IS_POSITIVE = lambda x: x > 0
IS_NEGATIVE = lambda x: x < 0
IS_EVEN = lambda x: x % 2 == 0
IS_ODD = lambda x: x % 2 == 1
ADD = lambda x, y: x + y
SUBTRACT = lambda x, y: x - y
MULTIPLY = lambda x, y: x * y
MIN = lambda x, y: min(x, y)
MAX = lambda x, y: max(x, y)

Below are example programs using the `dsl` module, with input-output test cases illustrating their behavior.

Important: All programs begin with ```python and end with ``` alone.


Input-output test cases:
  * x0 = [4, 2, 7], x1 = 5 --> [7, 4, 2]
  * x0 = [-24, 15, 3, -8], x1 = 3 --> [15, 3, -24]
  * x0 = [18, 22, 36, 13, 29, 4, 15, 10, 7], x1 = 6 --> [36, 29, 22, 18, 13, 4]

Program:
```python
def program(x0, x1):
  x2 = x0[:x1]
  x3 = sorted(x2)
  x4 = list(reversed(x3))
  return x4
```


Input-output test cases:
  * x0 = [5, 2, 6, 7, 4] --> [2, 8, 12]
  * x0 = [19, 2, 12, 6, 11, 15, 7, 8] --> [2, 14, 20, 28]
  * x0 = [5, -4, 6, 7, -1, -2, 4, 1, -6] --> [-4, 2, 0, 4, -2]

Program:
```python
def program(x0):
  x1 = [x for x in x0 if dsl.IS_EVEN(x)]
  x2 = dsl.Scanl1(dsl.ADD, x1)
  return x2
```


Input-output test cases:
  * x0 = [1, 2, 3] --> 6
  * x0 = [10, -10] --> 0
  * x0 = [45] --> 45

Program:
```python
'''.lstrip()
    self.assertEqual(prompt, expected)

  def test_few_shot_prompt_robustfill(self):
    prompt = llm_utils.few_shot_prompt(
        few_shot_examples=[ROBUSTFILL_EXAMPLE],
        test_problem=ROBUSTFILL_TEST_PROBLEM,
        dataset_type='robustfill',
        version=1,
    )
    expected = '''
The `dsl` module is a custom library for manipulating strings. It contains the following functions:

Const, SubStr, GetSpan, GetToken, ToCase, Replace, Trim, GetUpto, GetFrom, GetFirst, GetAll, Substitute, SubstituteAll, Remove, RemoveAll

Additionally, the module defines the following constants:

dsl.Type.NUMBER, dsl.Type.WORD, dsl.Type.ALPHANUM, dsl.Type.ALL_CAPS, dsl.Type.PROP_CASE, dsl.Type.LOWER, dsl.Type.DIGIT, dsl.Type.CHAR, dsl.Case.PROPER, dsl.Case.ALL_CAPS, dsl.Case.LOWER, dsl.Boundary.START, dsl.Boundary.END

Below are example programs using the `dsl` module, with input-output test cases illustrating their behavior.

Important: All programs begin with ```python and end with ``` alone.


Input-output test cases:
  * "#My##:Gxbo[Ned[Er%" --> "k[MY##:GXBO[NED[ER%8y##:Gxbo[Ned["
  * "#%$Ua.Qaeq?Opa%Kcr#" --> "kK%$UA.QAEQ?OPA%KCR#8aUa.Qaeq?Opa%"
  * "%{Eos#(Mdjt#'Yi{Oclf" --> "kO{EOS#(MDJT#'YI{OCLF8osos#(Mdjt#'Yi"
  * "%##Tq@Fh#Xza#?Fdlu" --> "kF##TQ@FH#XZA#?FDLU8qTq@Fh#Xza#?F"

Program:
```python
def program(x):
  parts = [
      dsl.Const('k'),
      dsl.GetToken(x, dsl.Type.CHAR, -4),
      dsl.ToCase(dsl.Remove(x, dsl.Type.CHAR, 1), dsl.Case.ALL_CAPS),
      dsl.Substitute(dsl.GetToken(x, dsl.Type.PROP_CASE, 1), dsl.Type.CHAR, 1, '8'),
      dsl.SubStr(x, 4, 15),
  ]
  return ''.join(parts)
```


Input-output test cases:
  * "apple" --> "Apple!"
  * "banana" --> "Banana!"
  * "clementine" --> "Clementine!"
  * "durian" --> "Durian!"

Program:
```python
'''.lstrip()
    self.assertEqual(prompt, expected)

  @parameterized.named_parameters(
      ('code_only', 'def foo():\n  return 1\n'),
      ('prefix', 'Sure, here is code:\n```python\ndef foo():\n  return 1\n'),
      ('suffix_no_newline', 'def foo():\n  return 1\n```'),
      ('long_suffix', 'def foo():\n  return 1\n```\nMore text...'),
      ('both', 'Code:\n```python\ndef foo():\n  return 1\n```\nMore text...'),
      ('multiple', textwrap.dedent('''
          Here is a solution:
          ```python
          def foo():
            return 1
          ```
          Another solution:
          ```python
          def bar():
            return 2
          ```
          Hope that helps!''')),
  )
  def test_cut_program_from_sample(self, sample):
    self.assertEqual(llm_utils.cut_program_from_sample(sample),
                     'def foo():\n  return 1\n')


if __name__ == '__main__':
  absltest.main()
