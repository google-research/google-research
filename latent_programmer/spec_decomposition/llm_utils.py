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

"""Utilities for measuring compositional generalization of LLMs."""

import ast
import collections
import re
import types
from typing import Any

import tensorflow as tf

from latent_programmer.tasks.deepcoder import deepcoder_dsl
from latent_programmer.tasks.robust_fill import dsl as robustfill_dsl
from latent_programmer.tasks.robust_fill import tokens as robustfill_tokens


# `inputs` is a dict for DeepCoder (name to list of values for each example), or
# a list for RobustFill (values for each example).
DatasetElement = collections.namedtuple(
    'DatasetElement',
    ['inputs', 'outputs', 'dsl_program', 'python_program'])


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


def to_python_form(io):
  """Convert Deepcoder's "x1 = [ 1 2 ] | x2 = 3" into "x1 = [1, 2], x2 = 3"."""
  io = io.replace(' | ', ', ').replace('[ ', '[').replace(' ]', ']')
  io = re.sub(r'(?<=\d) (?=-|\d)', ', ', io)
  return io


def parse_dataset(dataset,
                  dataset_type):
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

    python_program = program_object.to_python_program()

    d = DatasetElement(inputs, outputs, program, python_program)
    assert d.outputs == run_program(d.python_program, d.inputs, dataset_type)
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


def load_datasets(
    dataset_type,
    generalization_task,
    num_few_shot_examples,
    num_test_problems,
    cns_data_format,
):
  """Loads a few-shot dataset and a test dataset."""
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

  few_shot_dataset = parse_dataset(
      (create_dataset(train_data_path, num_examples)
       .shuffle(100000)
       .take(num_test_problems * num_few_shot_examples)),
      dataset_type=dataset_type)
  test_dataset = parse_dataset(
      (create_dataset(test_data_path, num_examples)
       .take(1000)  # Other experiments only use the first 1000 test problems.
       .shuffle(1000)  # Shuffle them all.
       .take(num_test_problems)),  # For LLMs, only use a random subset.
      dataset_type=dataset_type)

  return few_shot_dataset, test_dataset


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
    assert isinstance(inputs, list)
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


def dsl_description(dataset_type):
  """Gets a description of the DSL for prompting."""
  if dataset_type == 'deepcoder':
    dsl_purpose = 'manipulating lists of integers'
    functions = [op.token for op in deepcoder_dsl.OPERATIONS]
    constants = [lambda_.name for lambda_ in deepcoder_dsl.LAMBDAS]
  elif dataset_type == 'robustfill':
    dsl_purpose = 'manipulating strings'
    functions = ROBUSTFILL_FUNCTIONS
    constants = [robustfill_dsl.to_python(obj)  # pylint: disable=g-complex-comprehension
                 for e in ROBUSTFILL_ENUMS for obj in e]
  else:
    raise ValueError(f'Unhandled dataset type: {dataset_type}')
  return (
      f'The `dsl` module is a custom library for {dsl_purpose}. It contains '
      'the following functions:\n\n'
      f'{", ".join(functions)}\n\n'
      'Additionally, the module defines the following constants:\n\n'
      f'{", ".join(constants)}\n\n'
      'Below are example programs using the `dsl` module, with input-output '
      'test cases illustrating their behavior.\n\n'
      'Important: All programs begin with ```python and end with ``` alone.\n\n'
  )


def get_prompt_prefix(dataset_element,
                      dataset_type):
  """Gets a prefix of the prompt describing one dataset element."""
  s = 'Input-output test cases:\n'
  for i in range(get_num_examples(dataset_element.inputs, dataset_type)):
    s += '  * '
    if dataset_type == 'deepcoder':
      sep = ''
      for name in dataset_element.inputs:
        s += f'{sep}{name} = {dataset_element.inputs[name][i]}'
        sep = ', '
    elif dataset_type == 'robustfill':
      s += dataset_element.inputs[i]
    else:
      raise ValueError(f'Unhandled dataset type: {dataset_type}')
    s += f' --> {dataset_element.outputs[i]}\n'
  s += '\nProgram:\n```python\n'
  return s


def get_prompt_suffix(dataset_element):
  return f'{dataset_element.python_program}\n```\n\n'


def get_prompt(dataset_element, dataset_type):
  return (get_prompt_prefix(dataset_element, dataset_type)
          + get_prompt_suffix(dataset_element))


def few_shot_prompt(few_shot_examples,
                    test_problem,
                    dataset_type):
  prompt_parts = [dsl_description(dataset_type)]
  prompt_parts.extend(get_prompt(d, dataset_type) for d in few_shot_examples)
  prompt_parts.append(get_prompt_prefix(test_problem, dataset_type))
  return '\n'.join(prompt_parts)


def cut_program_from_sample(sample):
  if '```python\n' in sample:
    sample = sample.partition('```python\n')[-1]
  if '```' in sample:
    sample = sample.partition('```')[0]
  return sample
