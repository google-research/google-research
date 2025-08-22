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

"""Binary for splitting JSON Lines data into various eval sets."""
import copy
import enum
import json
import os
import random
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence, Tuple

from absl import app
from absl import flags
import apache_beam as beam


_INPUT_JSON_LINES_FILENAME = flags.DEFINE_string(
    'input_json_lines_filename',
    default=None,
    help='The input JSON Lines file name of the examples.',
    required=True)

_OUTPUT_JSON_LINES_DIR = flags.DEFINE_string(
    'output_json_lines_dir',
    default=None,
    help='The output JSON Lines file directory of the splitted sets.',
    required=True)

_RANDOM_SPLIT_RATIOS = flags.DEFINE_multi_integer(
    'random_split_ratios',
    default=(8, 1, 1),
    help='The ratios for randomly splitting train/eval/test data.')

_RANDOM_SPLIT_METHOD = flags.DEFINE_enum(
    'random_split_method',
    default='PRODUCT',
    enum_values=['PRODUCT', 'PRODUCT_ATTRIBUTE_KEY'],
    help='The method used for random split examples.')

_SELECTED_ATTRIBUTE_KEYS = flags.DEFINE_multi_string(
    'selected_attribute_keys',
    default=[
        'Type',
        'Style',
        'Material',
        'Size',
        'Capacity',
        'Black Tea Variety',
        'Staple Type',
        'Web Pattern',
        'Cabinet Configuration',
        'Power Consumption',
        'Front Camera Resolution',
    ],
    help='Holdout attribute keys.')

_HOLDOUT_ATTRIBUTE_KEYS = flags.DEFINE_multi_string(
    'holdout_attribute_keys',
    default=[
        'Device Type',
        'Special Occasion',
        'Boot Style',
        'Resolution',
        'Compatibility',
        'Winding Material',
        'Sensor',
        'Number of Sinks',
        'Food Processor Capacity',
    ],
    help='Holdout attribute keys.')

_Example = Any


@enum.unique
class RandomSplitMethod(enum.Enum):
  PRODUCT = 'PRODUCT'
  PRODUCT_ATTRIBUTE_KEY = 'PRODUCT_ATTRIBUTE_KEY'


class FlattenAttributesFn(beam.DoFn):
  """DoFn to flatten attributes."""

  def __init__(self, *unused_args, **unused_kwargs):
    self._num_input_examples = beam.metrics.Metrics.counter(
        self.__class__, 'num-input-examples')
    self._num_flattened_examples = beam.metrics.Metrics.counter(
        self.__class__, 'num-flattened-examples')

  def process(self, example, *args, **kwargs):
    self._num_input_examples.inc()
    for attribute in example['attributes']:
      self._num_flattened_examples.inc()
      yield {
          'id': example['id'],
          'category': example['category'],
          'paragraphs': copy.deepcopy(example['paragraphs']),
          'attributes': [copy.deepcopy(attribute)],
      }


class PseudoRandomSplitFn(beam.PartitionFn):
  """DnFn to perform pseudo random split on examples."""

  def __init__(self, random_split_ratios,
               random_split_method, *unused_args, **unused_kwargs):
    self._random_split_ratios = random_split_ratios
    self._random_split_method = random_split_method

  def partition_for(self, example, num_partitions, *args,
                    **kwargs):
    random_split_method = RandomSplitMethod(self._random_split_method)
    if random_split_method == RandomSplitMethod.PRODUCT:
      feature = json.dumps(example['paragraphs'])
    elif random_split_method == RandomSplitMethod.PRODUCT_ATTRIBUTE_KEY:
      feature = json.dumps(example)
    else:
      raise ValueError(
          f'Invalid random split method {self._random_split_method!r}')

    return self._pseudo_random_partition(feature)

  def _pseudo_random_partition(self, feature):
    bucket = sum(feature.encode('utf-8')) % sum(self._random_split_ratios)
    total = 0
    for i, part in enumerate(self._random_split_ratios):
      total += part
      if bucket < total:
        return i
    return len(self._random_split_ratios) - 1


class PseudoRandomSampleFn(beam.DoFn):
  """DnFn to perform pseudo random split on examples."""

  def __init__(self, num_sample, *unused_args, **unused_kwargs):
    self._num_sample = num_sample

  def process(self, element, *args,
              **kwargs):
    (category, attribute), examples = element
    # Creates a random seed based on category and attribute.
    seed = sum(category.encode('utf-8')) + sum(attribute.encode('utf-8'))
    # deterministically sample examples.
    example_strs = sorted(json.dumps(e) for e in examples)
    random.Random(seed).shuffle(example_strs)
    for index, example_str in enumerate(example_strs):
      example = json.loads(example_str)
      if index < self._num_sample:
        yield beam.pvalue.TaggedOutput('sample', example)
      else:
        yield example


class AttributeKeySplitFn(beam.DoFn):
  """DnFn to holdout head and tail attribute keys."""

  def __init__(self, holdout_attribute_keys, *unused_args,
               **unused_kwargs):
    self._holdout_attribute_keys = holdout_attribute_keys

  def process(self, example, *args, **kwargs):
    attribute_key = example['attributes'][0]['key']
    if attribute_key in self._holdout_attribute_keys:
      yield beam.pvalue.TaggedOutput(attribute_key, example)
    else:
      yield example


def split_by_attribute_keys(
    pipeline_prefix, examples,
    attribute_keys):
  """Returns attribute key splitted examples."""
  splited_examples = (
      examples
      | f'{pipeline_prefix}AttributeKeySelected' >> beam.ParDo(
          AttributeKeySplitFn(attribute_keys)).with_outputs())

  output_examples = {
      '00_All': examples,
      '01_Remain': splited_examples[None],
      **{
          f'{i:02d}_{"_".join(k.split())}': splited_examples[k]
          for i, k in enumerate(attribute_keys, start=2)
      },
  }
  return output_examples


def write_examples(pipeline_prefix, examples,
                   output_filename):
  """Writes examples to file."""
  _ = (
      examples
      | f'{pipeline_prefix}CountExamples' >> beam.combiners.Count.Globally()
      | f'{pipeline_prefix}CountExamplesJsonDumps' >>
      beam.Map(lambda x: json.dumps(x, indent=2))
      | f'{pipeline_prefix}CountExamplesWrite' >> beam.io.WriteToText(
          re.sub('(.jsonl)?$', '_counts', output_filename, count=1),
          shard_name_template='',  # To force unsharded output.
      ))

  _ = (
      examples
      | f'{pipeline_prefix}JsonDumps' >> beam.Map(json.dumps)
      | f'{pipeline_prefix}WriteExamples' >> beam.io.WriteToText(
          output_filename,
          shard_name_template='',  # To force unsharded output.
      ))


def _get_category_attribute(example):
  category = example['category']
  attribute = example['attributes'][0]['key']
  return category, attribute


def write_downsample_remain(pipeline_prefix, examples,
                            output_filename):
  """Writes downsampled remain examples."""
  prefix = f'{pipeline_prefix}_DownsampleHoldoutRemain'
  outputs = (
      examples
      | f'{prefix}_GroupByCategoryAttribute' >>
      beam.GroupBy(_get_category_attribute)
      | f'{prefix}_SamplePerCategoryAttribute' >> beam.ParDo(
          PseudoRandomSampleFn(10)).with_outputs())

  dirname = os.path.dirname(output_filename)
  basename = os.path.basename(output_filename)

  sample_output_path = os.path.join(dirname, 'sample', 'sppca010', basename)
  write_examples(f'{prefix}_sample', outputs['sample'], sample_output_path)
  remain_output_path = os.path.join(dirname, 'sample', 'remain', basename)
  write_examples(prefix, outputs[None], remain_output_path)


def write_fewshot_split(pipeline_prefix, examples,
                        output_filename):
  """Performs few shot split and writes examples."""

  def _get_key_fn(n):
    return lambda x: (f'{n}', '')

  prefix = f'{pipeline_prefix}_FewShotSplit'
  # Samples from the whole set, we assume the set is not too large.
  sample_100_outputs = (
      examples
      | f'{prefix}_100_Group' >> beam.GroupBy(_get_key_fn(100))
      | f'{prefix}_100_Sample' >> beam.ParDo(
          PseudoRandomSampleFn(100)).with_outputs())

  splits = {
      'remain': sample_100_outputs[None],
      'sp100': sample_100_outputs['sample']
  }
  for num_sample in [1, 2, 3, 5, 10, 50]:
    sample_n_outputs = (
        splits['sp100']
        | f'{prefix}_{num_sample}_Group' >>
        (beam.GroupBy(_get_key_fn(num_sample)))
        | f'{prefix}_{num_sample}_Sample' >> beam.ParDo(
            PseudoRandomSampleFn(num_sample)).with_outputs())
    splits[f'sp{num_sample:03d}'] = sample_n_outputs['sample']

  dirname = os.path.dirname(output_filename)
  basename = os.path.basename(output_filename)

  for split_name, split_examples in splits.items():
    output_path = os.path.join(dirname, 'sample', split_name, basename)
    write_examples(f'{prefix}_{split_name}', split_examples, output_path)


def pipeline(root):
  """Beam pipeline to run."""

  if len(_RANDOM_SPLIT_RATIOS.value) != 3:
    raise ValueError('Num of random split ratios incorrect: '
                     f'len({_RANDOM_SPLIT_RATIOS.value!r}) = '
                     f'{len(_RANDOM_SPLIT_RATIOS.value)} != 3')

  all_examples = (
      root
      | 'ReadExamples' >> beam.io.textio.ReadFromText(
          _INPUT_JSON_LINES_FILENAME.value)
      | 'JSONloads' >> beam.Map(json.loads)
      | 'FlattenAttributes' >> beam.ParDo(FlattenAttributesFn()))

  # Random splits examples.
  train_examples, eval_examples, test_examples = (
      all_examples
      | 'PseudoRandomSplit' >> beam.Partition(
          PseudoRandomSplitFn(_RANDOM_SPLIT_RATIOS.value,
                              _RANDOM_SPLIT_METHOD.value), 3))

  # Splits by selected attribute keys.
  for random_split_name, random_split_examples in zip(
      ['train', 'eval', 'test'],
      [train_examples, eval_examples, test_examples]):
    output_examples = split_by_attribute_keys(random_split_name,
                                              random_split_examples,
                                              _SELECTED_ATTRIBUTE_KEYS.value)

    for selected_split_name, selected_split_examples in output_examples.items():
      output_filename = os.path.join(
          _OUTPUT_JSON_LINES_DIR.value, _RANDOM_SPLIT_METHOD.value,
          random_split_name, selected_split_name,
          os.path.basename(_INPUT_JSON_LINES_FILENAME.value))

      write_examples(
          f'{random_split_name}_{selected_split_name}',
          selected_split_examples,
          output_filename,
      )

  # Splits by holdout attribute keys.
  train_eval_examples = ((train_examples, eval_examples)
                         | 'MergeTrainAndEval' >> beam.Flatten())
  output_examples = split_by_attribute_keys('holdout', train_eval_examples,
                                            _HOLDOUT_ATTRIBUTE_KEYS.value)
  for holdout_split_name, holdout_split_examples in output_examples.items():
    output_filename = os.path.join(
        _OUTPUT_JSON_LINES_DIR.value, _RANDOM_SPLIT_METHOD.value, 'holdout',
        holdout_split_name, os.path.basename(_INPUT_JSON_LINES_FILENAME.value))

    write_examples(
        f'holdout_{holdout_split_name}', holdout_split_examples, output_filename
    )

    if holdout_split_name.endswith('Remain'):
      write_downsample_remain(holdout_split_name, holdout_split_examples,
                              output_filename)
    elif not holdout_split_name.endswith('All'):
      write_fewshot_split(holdout_split_name, holdout_split_examples,
                          output_filename)


def main(unused_argv):
  # To enable distributed workflows, follow instructions at
  # https://beam.apache.org/documentation/programming-guide/
  # to set pipeline options.
  with beam.Pipeline() as p:
    pipeline(p)


if __name__ == '__main__':
  app.run(main)
