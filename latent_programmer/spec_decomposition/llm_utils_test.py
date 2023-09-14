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
    'outputs': tf.constant(['[ ]', '[ 5 14 15 ]', '[ 4 5 9 ]']),
    'program': tf.constant('x0 = INPUT | x1 = INPUT | x2 = Scanl1 (-) x0 | '
                           'x3 = Map (*(-1)) x2 | x4 = Filter (>0) x3'),
})
DEEPCODER_EXAMPLE = llm_utils.DatasetElement(
    inputs={'x0': [[], [1, 0, 6, 9, 1], [3, 7, 1, 4]],
            'x1': [[0], [9], [-3, -1]]},
    outputs=[[], [5, 14, 15], [4, 5, 9]],
    dsl_program=('x0 = INPUT | x1 = INPUT | x2 = Scanl1 (-) x0 | '
                 'x3 = Map (*(-1)) x2 | x4 = Filter (>0) x3'),
    python_program='''
def program(x0, x1):
  x2 = dsl.Scanl1(dsl.SUBTRACT, x0)
  x3 = dsl.Map(dsl.NEGATE, x2)
  x4 = dsl.Filter(dsl.IS_POSITIVE, x3)
  return x4
'''.strip(),
)
DEEPCODER_TEST_PROBLEM = llm_utils.DatasetElement(
    inputs={'x3': [[1, 2, 3], [10, -10], [45]]},
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

  def test_parse_dataset_deepcoder(self):
    self.assertEqual(
        llm_utils.parse_dataset(DEEPCODER_DATASET, 'deepcoder'),
        [DEEPCODER_EXAMPLE])

  def test_run_program_deepcoder(self):
    self.assertEqual(
        llm_utils.run_program(DEEPCODER_EXAMPLE.python_program,
                              DEEPCODER_EXAMPLE.inputs, 'deepcoder'),
        DEEPCODER_EXAMPLE.outputs)

  def test_get_num_examples_deepcoder(self):
    self.assertEqual(
        llm_utils.get_num_examples(DEEPCODER_EXAMPLE.inputs, 'deepcoder'),
        3)

  def test_parse_dataset_robustfill(self):
    self.assertEqual(
        llm_utils.parse_dataset(ROBUSTFILL_DATASET, 'robustfill'),
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

  def test_few_shot_prompt_deepcoder(self):
    prompt = llm_utils.few_shot_prompt(
        few_shot_examples=[DEEPCODER_EXAMPLE],
        test_problem=DEEPCODER_TEST_PROBLEM,
        dataset_type='deepcoder',
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
  * x0 = [1, 0, 6, 9, 1], x1 = [9] --> [5, 14, 15]
  * x0 = [3, 7, 1, 4], x1 = [-3, -1] --> [4, 5, 9]

Program:
```python
def program(x0, x1):
  x2 = dsl.Scanl1(dsl.SUBTRACT, x0)
  x3 = dsl.Map(dsl.NEGATE, x2)
  x4 = dsl.Filter(dsl.IS_POSITIVE, x3)
  return x4
```


Input-output test cases:
  * x3 = [1, 2, 3] --> 6
  * x3 = [10, -10] --> 0
  * x3 = [45] --> 45

Program:
```python
'''.lstrip()
    self.assertEqual(prompt, expected)

  def test_few_shot_prompt_robustfill(self):
    prompt = llm_utils.few_shot_prompt(
        few_shot_examples=[ROBUSTFILL_EXAMPLE],
        test_problem=ROBUSTFILL_TEST_PROBLEM,
        dataset_type='robustfill',
    )
    expected = '''
The `dsl` module is a custom library for manipulating strings. It contains the following functions:

Const, SubStr, GetSpan, GetToken, ToCase, Replace, Trim, GetUpto, GetFrom, GetFirst, GetAll, Substitute, SubstituteAll, Remove, RemoveAll

Additionally, the module defines the following constants:

dsl.Type.NUMBER, dsl.Type.WORD, dsl.Type.ALPHANUM, dsl.Type.ALL_CAPS, dsl.Type.PROP_CASE, dsl.Type.LOWER, dsl.Type.DIGIT, dsl.Type.CHAR, dsl.Case.PROPER, dsl.Case.ALL_CAPS, dsl.Case.LOWER, dsl.Boundary.START, dsl.Boundary.END

Below are example programs using the `dsl` module, with input-output test cases illustrating their behavior.

Important: All programs begin with ```python and end with ``` alone.


Input-output test cases:
  * #My##:Gxbo[Ned[Er% --> k[MY##:GXBO[NED[ER%8y##:Gxbo[Ned[
  * #%$Ua.Qaeq?Opa%Kcr# --> kK%$UA.QAEQ?OPA%KCR#8aUa.Qaeq?Opa%
  * %{Eos#(Mdjt#'Yi{Oclf --> kO{EOS#(MDJT#'YI{OCLF8osos#(Mdjt#'Yi
  * %##Tq@Fh#Xza#?Fdlu --> kF##TQ@FH#XZA#?FDLU8qTq@Fh#Xza#?F

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
  * apple --> Apple!
  * banana --> Banana!
  * clementine --> Clementine!
  * durian --> Durian!

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
