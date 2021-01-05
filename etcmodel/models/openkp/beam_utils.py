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

"""Apache Beam utilities."""

import collections
import json
import traceback
from typing import Any, Iterator, List, Mapping, Text

import apache_beam as beam

from etcmodel.models import tokenization
from etcmodel.models.openkp import eval_utils
from etcmodel.models.openkp import generate_examples_lib as lib

PCollection = beam.pvalue.PCollection


def singletons_to_dict(
    beam_label: Text = 'SingletonsToDict',
    **kwargs: PCollection[Any]) -> PCollection[Mapping[Text, Any]]:
  """Combines multiple singleton PCollections into a dictionary PCollection.

  Args:
    beam_label: A string label for this Beam transform. A unique label may need
      to be supplied instead of the default if this function is called multiple
      times in the same scope.
    **kwargs: Singleton PCollection arguments to combine. The argument names
      will become the keys in the resulting dictionary singleton PCollection.

  Returns:
    A singleton PCollection containing a dictionary of the combined results.

  Raises:
    ValueError: If `kwargs` is empty.
  """
  if not kwargs:
    raise ValueError('`kwargs` must not be empty.')
  first_arg = kwargs[next(iter(kwargs))]
  key_ordering = list(kwargs.keys())

  def make_dict(unused_element, key_ordering: List[Text], **kwargs):
    result = collections.OrderedDict()
    for key in key_ordering:
      result[key] = kwargs[key]
    return result

  singletons = {k: beam.pvalue.AsSingleton(v) for k, v in kwargs.items()}

  # `first_arg` is just used to have something to apply `beam.Map` to, but
  # it corresponds to `unused_element` in `make_dict()`.
  return (first_arg
          | beam_label >> beam.Map(make_dict, key_ordering, **singletons))


class ParseExampleFn(beam.DoFn):
  """DoFn for parsing OpenKP json string examples."""

  def __init__(self, config: lib.EtcFeaturizationConfig):
    self._config = config

  def setup(self):
    if self._config.spm_model_path:
      self._tokenizer = tokenization.FullTokenizer(
          None, do_lower_case=None, spm_model_file=self._config.spm_model_path)
    else:
      self._tokenizer = tokenization.FullTokenizer(
          self._config.bert_vocab_path,
          do_lower_case=self._config.do_lower_case)

  def process(self, json_str: Text) -> Iterator[Any]:
    try:
      example = lib.OpenKpExample.from_json(json_str)
      etc_features = example.to_etc_features(self._tokenizer, self._config)
      yield etc_features.to_tf_example()
    except Exception as e:  # pylint: disable=broad-except
      url = json.loads(json_str)['url']
      json_obj = dict(url=url, exception=f'{repr(e)}: {traceback.format_exc()}')
      yield beam.pvalue.TaggedOutput('parse_failures', json.dumps(json_obj))
      return

    text_example = eval_utils.OpenKpTextExample.from_openkp_example(example)
    yield beam.pvalue.TaggedOutput('text_examples',
                                   text_example.to_json_string())
