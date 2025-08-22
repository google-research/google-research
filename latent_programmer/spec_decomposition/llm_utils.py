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

"""Utilities for measuring compositional generalization of LLMs."""

import ast
import collections
import copy
import math
import random
import re
import types
from typing import Any

import numpy as np
import tensorflow as tf

from latent_programmer.tasks.deepcoder import deepcoder_dsl
from latent_programmer.tasks.robust_fill import dsl as robustfill_dsl
from latent_programmer.tasks.robust_fill import tokens as robustfill_tokens


# `inputs` is a dict for DeepCoder (name to list of values for each example), or
# a list for RobustFill (values for each example).
DatasetElement = collections.namedtuple(
    'DatasetElement',
    ['inputs', 'outputs', 'dsl_program', 'python_program'])

# `states` is similar to the `inputs` of `DatasetElement`, and it records the
# states of the newly created variable after executing the partial python
# program, `targets` is the `remains` string for RobustFill, and is None for
# DeepCoder.
StepData = collections.namedtuple(
    'StepData', ['states', 'targets', 'python_program_step']
)
ExeDecTrajectory = list[StepData]

ROBUSTFILL_ID_TOKEN_TABLE, _ = robustfill_tokens.build_token_tables()
ROBUSTFILL_EOS_ID = 2
ROBUSTFILL_FUNCTIONS = [
    'Const', 'SubStr', 'GetSpan', 'GetToken', 'ToCase', 'Replace', 'Trim',
    'GetUpto', 'GetFrom', 'GetFirst', 'GetAll', 'Substitute', 'SubstituteAll',
    'Remove', 'RemoveAll',
]
ROBUSTFILL_ENUMS = [
    robustfill_dsl.Type, robustfill_dsl.Case, robustfill_dsl.Boundary,
]

# Enables using the exact same datasets for any settings *up to* these numbers
# of examples, for more consistent comparisons between experiments that use
# different settings. The datasets will change if these numbers are changed.
MAX_NUM_FEW_SHOT_EXAMPLES = 10
MAX_NUM_TEST_PROBLEMS = 200

DEEPCODER_MAX_LIST_LENGTH = 5


def to_python_form(io):
  """Convert Deepcoder's "x1 = [ 1 2 ] | x2 = 3" into "x1 = [1, 2], x2 = 3"."""
  io = io.replace(' | ', ', ').replace('[ ', '[').replace(' ]', ']')
  io = re.sub(r'(?<=\d) (?=-|\d)', ', ', io)
  return io


def parse_dataset(dataset,
                  dataset_type,
                  version):
  """Parses the tf.data.Dataset into a list of DatasetElement."""
  data = []

  for element in dataset:
    inputs = [x.decode() for x in element['inputs'].numpy().tolist()]
    outputs = [x.decode() for x in element['outputs'].numpy().tolist()]
    program = element['program'].numpy().decode()

    if dataset_type == 'deepcoder':
      input_names = re.findall(r'x\d', inputs[0])
      inputs_dict = {name: [] for name in input_names}
      for s in inputs:
        for name in input_names:
          value_str = re.search(name + r' = ([\[\] \-0-9]+)($| \|)', s).group(1)
          value = ast.literal_eval(to_python_form(value_str))
          inputs_dict[name].append(value)
      inputs = inputs_dict
      outputs = [ast.literal_eval(to_python_form(o)) for o in outputs]
      program_object = deepcoder_dsl.Program.from_str(program)
    elif dataset_type == 'robustfill':
      program_tokens = [int(t) for t in program.replace('|', ' ').split()]
      program_tokens.append(ROBUSTFILL_EOS_ID)
      program_object = robustfill_dsl.decode_program(
          encoding=program_tokens, id_token_table=ROBUSTFILL_ID_TOKEN_TABLE)
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')

    python_program = program_object.to_python_program(version=version)

    d = DatasetElement(inputs, outputs, program, python_program)
    if dataset_type == 'deepcoder':
      d = canonicalize_deepcoder_variables(d)
    actual_outputs = run_program(d.python_program, d.inputs, dataset_type)
    if d.outputs != actual_outputs:
      raise ValueError(
          f'Program:\n'
          f'{d.python_program}\n'
          f'Inputs: {d.inputs}\n'
          f'Expected outputs: {d.outputs}\n'
          f'Actual outputs: {actual_outputs}\n'
      )
    data.append(d)
  return data


def create_dataset(file_pattern, num_examples):
  """Loads a DeepCoder or RobustFill dataset of entire programs.

  Args:
    file_pattern: A file pattern for the TFRecord files to read.
    num_examples: The number of examples in an I/O specification.

  Returns:
    A tf.data.Dataset.
  """
  filenames = sorted(tf.io.gfile.glob(file_pattern))
  raw_dataset = tf.data.TFRecordDataset(filenames)

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    empty_default = [''] * num_examples
    feature_values = tf.io.parse_single_example(
        serialized=record,
        features={
            'inputs':
                tf.io.FixedLenFeature([num_examples], tf.string,
                                      default_value=empty_default),
            'outputs':
                tf.io.FixedLenFeature([num_examples], tf.string,
                                      default_value=empty_default),
            'program':
                tf.io.FixedLenFeature([], tf.string, default_value=''),
        })
    return {
        'inputs': feature_values['inputs'],
        'outputs': feature_values['outputs'],
        'program': feature_values['program'],
    }

  dataset = raw_dataset.map(_parse_fn)
  return dataset


def get_handwritten_few_shot(dataset_type,
                             generalization_task):
  """Gets a dataset of handwritten few-shot examples."""
  if dataset_type == 'robustfill':
    raise ValueError('Not implemented yet')
  if generalization_task != 'NONE':
    raise ValueError('Not implemented yet')

  problems = [
      {  # Take the first few elements and sort them in reverse.
          'inputs': ['x0 = [ 4 2 7 ] | x1 = 5',
                     'x0 = [ -24 15 3 -8 ] | x1 = 3',
                     'x0 = [ 18 22 36 13 29 4 15 10 7 ] | x1 = 6'],
          'outputs': ['[ 7 4 2 ]', '[ 15 3 -24 ]', '[ 36 29 22 18 13 4 ]'],
          'program': ('x0 = INPUT | x1 = INPUT | x2 = Take x1 x0 | '
                      'x3 = Sort x2 | x4 = Reverse x3'),
      },
      {  # Compute the running sum of the even elements.
          'inputs': ['x0 = [ 5 2 6 7 4 ]',
                     'x0 = [ 19 2 12 6 11 15 7 8 ]',
                     'x0 = [ 5 -4 6 7 -1 -2 4 1 -6 ]'],
          'outputs': ['[ 2 8 12 ]', '[ 2 14 20 28 ]', '[ -4 2 0 4 -2 ]'],
          'program': 'x0 = INPUT | x1 = Filter (%2==0) x0 | x2 = Scanl1 (+) x1',
      },
      {  # Count the number of negative elements.
          'inputs': ['x0 = [ -4 2 6 7 -1 4 0 -3 ]',
                     'x0 = [ 8 23 -14 32 -6 45 ]',
                     'x0 = [ -6 -8 -14 -23 -11 ]'],
          'outputs': ['3', '2', '5'],
          'program': 'x0 = INPUT | x1 = Count (<0) x0',
      },
      {  # Map by (x^2 - x) and drop the first few elements.
          'inputs': ['x0 = [ 1 4 5 2 ] | x1 = 1',
                     'x0 = [ -2 3 -1 0 6 ] | x1 = 0 ',
                     'x0 = [ -5 2 4 -3 1 7 5 ] | x1 = 3'],
          'outputs': ['[ 12 20 2 ]', '[ 6 6 2 0 30 ]', '[ 12 0 42 20 ]'],
          'program': ('x0 = INPUT | x1 = INPUT | x2 = Map (**2) x0 | '
                      'x3 = ZipWith (-) x2 x0 | x4 = Drop x1 x3'),
      },
  ]
  return tf.data.Dataset.from_tensor_slices({
      'inputs': tf.constant([p['inputs'] for p in problems]),
      'outputs': tf.constant([p['outputs'] for p in problems]),
      'program': tf.constant([p['program'] for p in problems]),
  })


def program_len(d, dataset_type):
  if dataset_type == 'deepcoder':
    return d.dsl_program.count('|') - d.dsl_program.count('INPUT') + 1
  elif dataset_type == 'robustfill':
    return d.dsl_program.count('|') + 1
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')


def distribute_lengths(
    dataset,
    target_size,
    max_length,
    dataset_type,
):
  """Selects a subset of elements with an even distribution of lengths."""
  data_by_length = collections.defaultdict(list)
  for element in dataset:
    data_by_length[program_len(element, dataset_type)].append(element)

  available_lengths = [length for length in data_by_length
                       if length <= max_length]
  num_per_length = math.ceil(target_size / len(available_lengths))
  selected = []
  for length in available_lengths:
    if len(data_by_length[length]) < num_per_length:
      raise ValueError(
          f'Not enough programs of length {length}: '
          f'need {num_per_length}, found {len(data_by_length[length])}')
    selected.extend(data_by_length[length][:num_per_length])
  random.shuffle(selected)
  selected = selected[:target_size]
  assert len(selected) == target_size
  return selected, data_by_length


def load_datasets(
    dataset_type,
    generalization_task,
    num_few_shot_examples,
    num_test_problems,
    cns_data_format,
    version,
):
  """Loads a few-shot dataset and a test dataset."""
  if num_few_shot_examples > MAX_NUM_FEW_SHOT_EXAMPLES:
    raise ValueError(f'Too many few shot examples: {num_few_shot_examples}')
  if num_test_problems > MAX_NUM_TEST_PROBLEMS:
    raise ValueError(f'Too many test problems: {num_test_problems}')

  # Set a seed for deterministic dataset shuffling. Set the seed here, not just
  # once elsewhere, so that the dataset shuffling is not dependent on the order
  # the datasets are constructed in.
  tf.random.set_seed(0)
  random.seed(0)
  np.random.seed(0)

  # Read data from CNS.
  if dataset_type == 'deepcoder':
    if generalization_task == 'LENGTH_GENERALIZATION':
      generalization_task = 'LENGTH_1_4_TO_5'
    cns_dataset_dir = 'deepcoder_hard_data_examples-3_length-5_max-50'
    num_examples = 3
  elif dataset_type == 'robustfill':
    if generalization_task == 'LENGTH_GENERALIZATION':
      generalization_task = 'LENGTH_1_6_TO_7_10'
    cns_dataset_dir = 'robustfill_data'
    num_examples = 4
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  train_data_path = cns_data_format.format(
      cns_dataset_dir=cns_dataset_dir, generalization_task=generalization_task,
      split='train')
  test_data_path = cns_data_format.format(
      cns_dataset_dir=cns_dataset_dir, generalization_task=generalization_task,
      split='test')

  target_few_shot_dataset_size = (
      MAX_NUM_TEST_PROBLEMS * MAX_NUM_FEW_SHOT_EXAMPLES)
  few_shot_tf_dataset = (
      create_dataset(train_data_path, num_examples)
      .shuffle(100000)
      .take(target_few_shot_dataset_size * 20))
  test_tf_dataset = (
      create_dataset(test_data_path, num_examples)
      .take(1000)  # Other experiments only use the first 1000 test problems.
      .shuffle(1000))  # Shuffle them all.

  if version == 5:
    few_shot_tf_dataset = get_handwritten_few_shot(
        dataset_type, generalization_task)

  # Parse each `tf.data.Dataset` into list[DatasetElement].
  few_shot_dataset = parse_dataset(
      few_shot_tf_dataset, dataset_type=dataset_type, version=version)
  test_dataset = parse_dataset(
      test_tf_dataset, dataset_type=dataset_type, version=version)

  # Select a good mix of lengths.
  max_length = 3
  if generalization_task.startswith('LENGTH_'):
    # Remember, generalization_task has been changed to 'LENGTH_1_4_TO_5' etc.
    few_shot_max_length = max_length - 1
  else:
    few_shot_max_length = max_length

  selected_few_shot, few_shot_by_length = distribute_lengths(
      few_shot_dataset,
      target_size=target_few_shot_dataset_size,
      max_length=few_shot_max_length,
      dataset_type=dataset_type)

  if generalization_task.startswith('LENGTH_'):
    # For length generalization, the test programs don't come from the actual
    # test dataset which has only programs of very long length. Instead, test on
    # programs of length `max_length` gathered from the training dataset, and
    # use programs of shorter length for few-shot examples.
    selected_test = few_shot_by_length[max_length][:MAX_NUM_TEST_PROBLEMS]
    if len(selected_test) != MAX_NUM_TEST_PROBLEMS:
      raise ValueError(f'Not enough test problems: need '
                       f'{MAX_NUM_TEST_PROBLEMS}, have {len(selected_test)}')
  else:
    selected_test, _ = distribute_lengths(
        test_dataset,
        target_size=MAX_NUM_TEST_PROBLEMS,
        max_length=max_length,
        dataset_type=dataset_type)

  selected_test = selected_test[:num_test_problems]
  assert len(selected_test) == num_test_problems

  return selected_few_shot, selected_test


def few_shot_examples_for_test_index(
    few_shot_dataset,
    test_index,
    num_few_shot_examples):
  """Returns the slice of few-shot examples for the test problem index."""
  if num_few_shot_examples > MAX_NUM_FEW_SHOT_EXAMPLES:
    raise ValueError(f'Too many few-shot examples: {num_few_shot_examples}')
  start_index = test_index * MAX_NUM_FEW_SHOT_EXAMPLES
  ans = few_shot_dataset[start_index : start_index + num_few_shot_examples]
  assert len(ans) == num_few_shot_examples
  return ans


def get_namespace(dataset_type):
  """Gets a namespace with the dsl loaded."""
  dsl_object = types.SimpleNamespace()
  if dataset_type == 'deepcoder':
    for lambda_ in deepcoder_dsl.LAMBDAS:
      setattr(dsl_object, lambda_.name, lambda_.func)
    for op in deepcoder_dsl.OPERATIONS:
      setattr(dsl_object, op.token, op.func)
  elif dataset_type == 'robustfill':
    for function_name in ROBUSTFILL_FUNCTIONS:
      if function_name == 'Const':
        op_class = robustfill_dsl.ConstStr
        wrapper = lambda c, op_class=op_class: op_class(c)(None)
      else:
        op_class = getattr(robustfill_dsl, function_name)
        wrapper = lambda x, *args, op_class=op_class: op_class(*args)(x)
      setattr(dsl_object, function_name, wrapper)
    for enum_class in ROBUSTFILL_ENUMS:
      setattr(dsl_object, enum_class.__name__, enum_class)
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return {'dsl': dsl_object}


def get_num_examples(inputs,
                     dataset_type):
  """Returns the number of examples in the inputs."""
  if dataset_type == 'deepcoder':
    assert isinstance(inputs, dict)
    inputs_dict = inputs
    num_examples = len(list(inputs_dict.values())[0])
    assert all(len(v) == num_examples for v in inputs_dict.values())
  elif dataset_type == 'robustfill':
    assert isinstance(inputs, list), f'RobustFill inputs: {inputs}'
    num_examples = len(inputs)
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return num_examples


def run_program(program_code,
                inputs,
                dataset_type,
                program_name = 'program'):
  """Runs a DeepCoder or RobustFill program."""
  # Set up code for calling the solution function with appropriate arguments.
  if dataset_type == 'deepcoder':
    assert isinstance(inputs, dict)
    call_code = f'{program_name}({", ".join(inputs.keys())})'
  elif dataset_type == 'robustfill':
    call_code = f'{program_name}(x)'
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  # Define the solution function.
  namespace = get_namespace(dataset_type)
  try:
    exec(program_code, namespace)  # pylint: disable=exec-used
  except:  # pylint: disable=bare-except
    return None

  # Run the solution function for each example.
  outputs = []
  for i in range(get_num_examples(inputs, dataset_type)):
    namespace_copy = namespace.copy()
    # Assign the argument values.
    if dataset_type == 'deepcoder':
      assert isinstance(inputs, dict)
      for input_name, input_values in inputs.items():
        namespace_copy[input_name] = input_values[i]
    elif dataset_type == 'robustfill':
      namespace_copy['x'] = inputs[i]
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
    # Call the solution function.
    try:
      output = eval(call_code, namespace_copy)  # pylint: disable=eval-used
    except:  # pylint: disable=bare-except
      output = None
    outputs.append(output)

  return outputs


_DEEPCODER_FUNCTION_IMPLS = [
    '''
def Map(f, xs):
  return [f(x) for x in xs]
''',
    '''
def Filter(f, xs):
  return [x for x in xs if f(x)]
''',
    '''
def Count(f, xs):
  return len([x for x in xs if f(x)])
''',
    '''
def ZipWith(f, xs, ys):
  return [f(x, y) for (x, y) in zip(xs, ys)]
''',
    '''
def Scanl1(f, xs):
  ys = []
  for i, x in enumerate(xs):
    if i == 0:
      ys.append(x)
    else:
      ys.append(f(ys[-1], x))
  return ys
''',
]

_DEEPCODER_LAMBDA_IMPLS = '''
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
'''.strip()


def dsl_description(dataset_type, version):
  """Gets a description of the DSL for prompting."""
  if dataset_type == 'deepcoder':
    dsl_purpose = 'manipulating lists of integers'
    if version == 1:
      function_details = ', '.join(
          [op.token for op in deepcoder_dsl.OPERATIONS])
      constant_details = ', '.join(
          [lambda_.name for lambda_ in deepcoder_dsl.LAMBDAS])
    elif version == 2:
      function_details = '\n\n'.join(
          [i.strip() for i in _DEEPCODER_FUNCTION_IMPLS])
      constant_details = _DEEPCODER_LAMBDA_IMPLS
    elif version == 3 or version == 5:
      function_details = _DEEPCODER_FUNCTION_IMPLS[-1].strip()
      constant_details = _DEEPCODER_LAMBDA_IMPLS
    elif version == 4:
      function_details = _DEEPCODER_FUNCTION_IMPLS[-1].strip()
      constant_details = None
    else:
      raise ValueError(f'Unhandled version: {version}')
  elif dataset_type == 'robustfill':
    if version == 1:
      dsl_purpose = 'manipulating strings'
      function_details = ', '.join(ROBUSTFILL_FUNCTIONS)
      constant_details = ', '.join(
          [robustfill_dsl.to_python(obj)  # pylint: disable=g-complex-comprehension
           for e in ROBUSTFILL_ENUMS for obj in e])
    else:
      raise ValueError(f'Unhandled version: {version}')
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return (
      f'The `dsl` module is a custom library for {dsl_purpose}. It contains '
      'the following functions:\n\n'
      f'{function_details}\n\n'
      + (
          'Additionally, the module defines the following constants:\n\n'
          f'{constant_details}\n\n'
          if constant_details else '') +
      'Below are example programming problems using the `dsl` module, with'
      ' input-output test cases illustrating their behavior.\n\nImportant:'
      ' All programs begin with ```python and end with ``` alone.\n\n'
  )


def get_prompt_prefix(dataset_element,
                      dataset_type):
  """Gets a prefix of the prompt describing one dataset element."""
  s = '[BEGIN PROBLEM]\n'
  s += 'Input-output test cases:\n'
  for i in range(get_num_examples(dataset_element.inputs, dataset_type)):
    s += f'  Case {i + 1}. '
    if dataset_type == 'deepcoder':
      sep = ''
      for name in dataset_element.inputs:
        s += f'{sep}{name} = {dataset_element.inputs[name][i]}'
        sep = ', '
      s += f' --> {dataset_element.outputs[i]}\n'
    elif dataset_type == 'robustfill':
      s += f'"{dataset_element.inputs[i]}" --> "{dataset_element.outputs[i]}"\n'
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
  s += '\nProgram:\n```python\n'
  return s


def get_prompt_suffix(dataset_element):
  return f'{dataset_element.python_program}\n```\n[END PROBLEM]\n\n'


def get_prompt(dataset_element, dataset_type):
  return (get_prompt_prefix(dataset_element, dataset_type)
          + get_prompt_suffix(dataset_element))


def few_shot_prompt(few_shot_examples,
                    test_problem,
                    dataset_type,
                    version):
  prompt_parts = [dsl_description(dataset_type, version=version)]
  prompt_parts.extend(get_prompt(d, dataset_type) for d in few_shot_examples)
  prompt_parts.append(get_prompt_prefix(test_problem, dataset_type))
  return '\n'.join(prompt_parts)


def check_deepcoder_object_valid(s):
  """Check if the object is a valid DeepCoder object."""
  # For DeepCoder, every object is either an int or a list of ints
  if not (isinstance(s, int) or isinstance(s, list)):
    raise ValueError(f'Invalid DeepCoder object: {s}, type: {type(s)}')
  if isinstance(s, list):
    # Every list has length <= DEEPCODER_MAX_LIST_LENGTH
    if not len(s) <= DEEPCODER_MAX_LIST_LENGTH:
      raise ValueError(f'Invalid DeepCoder object: {s}, length: {len(s)}')
    for x in s:
      if not isinstance(x, int):
        raise ValueError(f'Invalid DeepCoder object: {s}, type: {type(x)}')
      # Every int is in the range [-50, 50] inclusive
      if not (-50 <= x <= 50):
        raise ValueError(f'Invalid DeepCoder object: {s}, int: {x}')
  else:
    if not (-50 <= s <= 50):
      raise ValueError(f'Invalid DeepCoder object: {s}, int: {s}')
  return True


def check_robustfill_object_valid(s):
  if not isinstance(s, str):
    raise ValueError(f'Invalid RobustFill object: {s}, type: {type(s)}')
  if len(s) > 20:
    raise ValueError(f'Invalid RobustFill object: {s}, length: {len(s)}')
  return True


def get_exe_dec_trajectory(
    dataset_element, dataset_type
):
  """Decompose the dataset element into a ExeDec trajectory."""
  trajectory: ExeDecTrajectory = []
  program_steps = dataset_element.python_program.splitlines()
  # The initial step.
  states = copy.deepcopy(dataset_element.inputs)
  targets = (
      None
      if dataset_type == 'deepcoder'
      else copy.deepcopy(dataset_element.outputs)
  )
  trajectory.append(StepData(states, targets, program_steps[0]))
  # The middle steps before return
  for j in range(1, len(program_steps) - 1):
    python_program_step = program_steps[j]
    if dataset_type == 'deepcoder':
      new_var = python_program_step.strip().split('=', 1)[0].strip()

      compose_program = '\n'.join(
          [x.python_program_step for x in trajectory]
          + [python_program_step, f'  return {new_var}']
      )

      actual_states = run_program(
          compose_program, dataset_element.inputs, dataset_type
      )
      for s in actual_states:
        if not check_deepcoder_object_valid(s):
          raise ValueError(f'Invalid DeepCoder object: {s}')
      actual_states = {new_var: actual_states}
      new_targets = None
    elif dataset_type == 'robustfill':
      if python_program_step.strip() in ['parts = [', ']']:
        continue
      python_program_step = python_program_step.strip()
      if python_program_step.endswith(','):
        python_program_step = python_program_step[:-1]
      compose_program = (
          trajectory[0].python_program_step
          + '\n'
          + f'  return {python_program_step}'
      )
      actual_states = run_program(
          compose_program, dataset_element.inputs, dataset_type
      )
      previous_targets = trajectory[-1].targets
      new_targets = []
      num_examples = get_num_examples(dataset_element.inputs, dataset_type)

      for s in actual_states:
        if not check_robustfill_object_valid(s):
          raise ValueError(f'Invalid RobustFill object: {s}')
      for i in range(num_examples):
        if (not isinstance(actual_states[i], str)) or (
            not previous_targets[i].startswith(actual_states[i])
        ):
          raise ValueError(
              f'Case {i + 1}: {previous_targets[i]} does not match the prefix'
              f' of {actual_states[i]}'
          )
        new_targets.append(previous_targets[i][len(actual_states[i]) :])
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')

    trajectory.append(StepData(actual_states, new_targets, python_program_step))
  return trajectory


def get_exe_dec_prompt_prefix(
    dataset_element,
    dataset_type,
    ablation_style = False,
):
  """Gets a prefix of the ExeDec prompt describing one dataset element."""
  # TODO(yldeng): support the ablation-style prompts
  s = '[BEGIN PROBLEM]\n'
  s += 'Input-output test cases:\n'
  num_examples = get_num_examples(dataset_element.inputs, dataset_type)
  if dataset_type == 'deepcoder':
    for i in range(num_examples):
      s += f'  Case {i+1}. '
      sep = ''
      for name in dataset_element.inputs:
        s += f'{sep}{name} = {dataset_element.inputs[name][i]}'
        sep = ', '
      s += f' --> {dataset_element.outputs[i]}\n'
  elif dataset_type == 'robustfill':
    for i in range(num_examples):
      s += f'  Case {i+1}. x = '
      s += f'"{dataset_element.inputs[i]}" --> "{dataset_element.outputs[i]}"\n'
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')

  s += '\nWe solve this problem step-by-step.\n\n'
  if ablation_style:
    if dataset_element.python_program is None:
      s += 'Step 1 code:\n'
      return s
  else:
    if dataset_element.python_program is None:
      s += 'Step 1 computes:\n'
      return s
  trajectory = get_exe_dec_trajectory(dataset_element, dataset_type)
  for j in range(1, len(trajectory)):
    subgoals = f'Step {j} computes:\n'
    if dataset_type == 'deepcoder':
      for i in range(get_num_examples(dataset_element.inputs, dataset_type)):
        subgoals += f'  Case {i+1}. '
        sep = ''
        for name in trajectory[j].states:
          subgoals += f'{sep}{name} = {trajectory[j].states[name][i]}'
          sep = ', '
        subgoals += '\n'
    elif dataset_type == 'robustfill':
      for i in range(num_examples):
        subgoals += f'  Case {i+1}. '
        subgoals += (
            f'"{trajectory[j].states[i]}" so "{trajectory[j].targets[i]}"'
            ' remains\n'
        )
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
    step_code = f'Step {j} code:\n'
    step_code += (
        f'```python\n{trajectory[j].python_program_step.strip()}\n```\n'
    )
    if ablation_style:
      s += step_code + '\n' + subgoals + '\n'
    else:
      s += subgoals + '\n' + step_code + '\n'
  s += (
      'Putting the steps together, the problem is solved with the'
      ' program:\n```python\n'
  )
  return s


def get_exe_dec_prompt_suffix(dataset_element):
  return f'{dataset_element.python_program}\n```\n[END PROBLEM]\n\n'


def get_exe_dec_prompt(
    dataset_element,
    dataset_type,
    ablation_style = False,
):
  return get_exe_dec_prompt_prefix(
      dataset_element, dataset_type, ablation_style=ablation_style
  ) + get_exe_dec_prompt_suffix(dataset_element)


def few_shot_exe_dec_prompt(
    few_shot_examples,
    test_problem,
    dataset_type,
    version,
    ablation_style = False,
):
  """Generate the ExeDec few-shot prompt."""
  prompt_parts = [
      dsl_description(dataset_type, version=version).replace(
          'illustrating their behavior.',
          'illustrating the program behavior step-by-step.',
      )
  ]
  prompt_parts.extend(
      get_exe_dec_prompt(d, dataset_type, ablation_style=ablation_style)
      for d in few_shot_examples
  )
  prompt_parts.append(
      get_exe_dec_prompt_prefix(
          test_problem, dataset_type, ablation_style=ablation_style
      )
  )
  return '\n'.join(prompt_parts)


def canonicalize_deepcoder_variables(
    dataset_element,
):
  """Canonicalizes the variable names in deepcoder programs and inputs."""
  program: str = dataset_element.python_program
  input_mapping_dict = {}
  input_occurences = re.findall(r'x\d', program)
  x_index = 0
  for input_name in input_occurences:
    if input_name not in input_mapping_dict:
      input_mapping_dict[input_name] = f'x{x_index}'
      x_index += 1

  # canonicalize python program and dsl program
  # To avoid collision, (x7 --> x0, x0 ---> x2) is decomposed to 2 steps
  # (x7 --> y0, x0 --> y2), and (y0 --> x0, y2 --> x2).
  dsl_program = dataset_element.dsl_program
  for input_name, new_name in input_mapping_dict.items():
    program = program.replace(input_name, new_name.replace('x', 'y'))
    dsl_program = dsl_program.replace(input_name, new_name.replace('x', 'y'))
  for new_name in input_mapping_dict.values():
    program = program.replace(new_name.replace('x', 'y'), new_name)
    dsl_program = dsl_program.replace(new_name.replace('x', 'y'), new_name)

  # canonicalize inputs
  inputs = {}
  for input_name, value in dataset_element.inputs.items():
    inputs[input_mapping_dict[input_name]] = copy.deepcopy(value)

  return DatasetElement(inputs, dataset_element.outputs, dsl_program, program)


def cut_program_from_sample(sample):
  if '```python\n' in sample:
    sample = sample.partition('```python\n')[-1]
  if '```' in sample:
    sample = sample.partition('```')[0]
  return sample
