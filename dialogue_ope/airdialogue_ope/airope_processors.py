# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

""" Airdialogue OPE processors and helpers """

import logging
import os
import json
from enum import Enum
from typing import List, Optional, Union, Dict
from dataclasses import dataclass, field

from multiprocessing import Pool
from functools import partial

import tqdm

import copy

import transformers

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from transformers.data.processors.utils import DataProcessor, InputFeatures

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
  guid: str
  text_gen_agent: List[str]
  text_ref_agent: List[str]
  text_ref_customer: List[str]
  reward: Union[float, Dict[str, float]]

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(dataclasses.asdict(self), indent=2) + '\n'


@dataclass(frozen=True)
class InputFeatures:
  """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token
          indices.
            Mask values selected in ``[0, 1]``: Usually  ``1`` for tokens that
              are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and
          second portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for
          classification problems, float for regression problems.
  """

  input_ids: List[int]
  attention_mask: List[List[bool]]
  token_type_ids: List[int]
  position_ids: List[int]
  true_conv_end: int
  total_length: int
  ref_c_end_ids: List[int]
  ref_a_end_ids: List[int]
  gen_a_end_ids: List[int]
  reward: Union[float, Dict[str, float]]

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(dataclasses.asdict(self)) + '\n'


def convert2ids(example, tokenizer, max_length):
  """
    RoBERTa/BERT:
    tokens:     [CLS] C s1 [SEP] [SEP] A s1 [SEP] [SEP] A' s1 [SEP] [SEP] C s2
    [SEP] [SEP] A s2 [SEP] [SEP] A' s2 [SEP]
    type_ids:   0     0 0  0     1     1 1  1     1     1  1  1     1     0 0  0
    1     1 1  1     1     1  1  1
    """
  if tokenizer.__class__ not in (RobertaTokenizer, RobertaTokenizerFast):
    raise RuntimeError('This tokenizer is not supported: ' +
                       str(toeknizer.__class__))
  input_ids = []
  position_ids = []
  attention_mask = []
  token_type_ids = []
  true_conv_end = -1
  ref_c_end_ids = []
  ref_a_end_ids = []
  gen_a_end_ids = []

  gen_agent_ids = tokenizer.batch_encode_plus(
      example.text_gen_agent, add_special_tokens=False)['input_ids']
  ref_agent_ids = tokenizer.batch_encode_plus(
      example.text_ref_agent, add_special_tokens=False)['input_ids']
  ref_customer_ids = tokenizer.batch_encode_plus(
      example.text_ref_customer, add_special_tokens=False)['input_ids']

  #print(len(example.text_gen_agent),len(example.text_ref_agent),len(example.text_ref_customer))

  if tokenizer.__class__ in (RobertaTokenizer, RobertaTokenizerFast):
    # begining sep token
    gen_agent_ids = [[tokenizer.sep_token_id] + l + [tokenizer.sep_token_id]
                     for l in gen_agent_ids]
    ref_agent_ids = [[tokenizer.sep_token_id] + l + [tokenizer.sep_token_id]
                     for l in ref_agent_ids]
    ref_customer_ids = [[tokenizer.sep_token_id] + l + [tokenizer.sep_token_id]
                        for l in ref_customer_ids]
    # begining cls token
    ref_customer_ids[0][0] = tokenizer.cls_token_id
    # for first customer == ""
    if len(ref_customer_ids[0]) == 2:
      ref_customer_ids[0] = ref_customer_ids[0][:1]
      ref_agent_ids[0] = ref_agent_ids[0][1:]
      gen_agent_ids[0] = gen_agent_ids[0][1:]

  p_pointer = 0
  pre_att_mask = []
  for ref_c, ref_a, gen_a in zip(ref_customer_ids, ref_agent_ids,
                                 gen_agent_ids):
    pre_att_mask = pre_att_mask + [True] * len(ref_c)
    attention_mask += [copy.deepcopy(pre_att_mask) for _ in ref_c]
    pre_att_mask_refa = pre_att_mask + [True] * len(ref_a)
    attention_mask += [copy.deepcopy(pre_att_mask_refa) for _ in ref_a]
    pre_att_mask_gena = pre_att_mask + [False] * len(ref_a) + [True
                                                              ] * len(gen_a)
    attention_mask += [copy.deepcopy(pre_att_mask_gena) for _ in gen_a]
    pre_att_mask = pre_att_mask_refa + [False] * len(gen_a)

    ref_c_end_ids.append(len(input_ids) + len(ref_c) - 1)
    ref_a_end_ids.append(len(input_ids) + len(ref_c) + len(ref_a) - 1)
    gen_a_end_ids.append(
        len(input_ids) + len(ref_c) + len(ref_a) + len(gen_a) - 1)
    input_ids += ref_c + ref_a + gen_a
    token_type_ids += [0] * len(ref_c) + [1] * len(ref_a) + [1] * len(gen_a)
    position_ids += list(range(p_pointer, p_pointer+len(ref_c))) +\
        list(range(p_pointer+len(ref_c), p_pointer+len(ref_c)+len(ref_a))) +\
        list(range(p_pointer+len(ref_c), p_pointer+len(ref_c)+len(gen_a)))
    p_pointer += len(ref_c) + len(ref_a)

  # print(len(ref_a_end_ids),len(gen_a_end_ids))

  total_length = len(input_ids)
  true_conv_end = len(input_ids) - len(gen_agent_ids[-1]) - 1

  if not len(input_ids) <= max_length:
    # Skip these samples
    logger.info(
        'Skipping Example: input length {} > max length {}, max position {}'
        .format(len(input_ids), max_length, p_pointer))
    return None

  pad_length = max_length - len(input_ids)
  input_ids += [tokenizer.pad_token_id] * pad_length
  token_type_ids += [0] * pad_length
  position_ids += [0] * pad_length
  attention_mask = [
      attm + [False] * (max_length - len(attm)) for attm in attention_mask
  ] + [[False] * max_length] * pad_length

  return {
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'token_type_ids': token_type_ids,
      'position_ids': position_ids,
      'true_conv_end': true_conv_end,
      'total_length': total_length,
      'ref_c_end_ids': ref_c_end_ids,
      'ref_a_end_ids': ref_a_end_ids,
      'gen_a_end_ids': gen_a_end_ids,
      'reward': example.reward,
  }


def multiconvert2ids(examples, tokenizer, max_length):
  return [convert2ids(ex, tokenizer, max_length) for ex in examples]


def airope_convert_examples_to_features(
    examples,
    tokenizer,
    max_length = None,
    workers = 20,
):
  """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: Airdialogue OPE task

    Returns:
        A list of task-specific ``InputFeatures`` which can be fed to the model.

    """
  if max_length is None:
    max_length = tokenizer.max_len

  with Pool(processes=workers) as pool:
    chunksize = max(1, len(examples) // (10 * workers))
    chunkedexamples = [
        examples[i:i + chunksize] for i in range(0, len(examples), chunksize)
    ]
    inputsarray = list(
        tqdm.tqdm(
            pool.imap(
                partial(
                    multiconvert2ids,
                    tokenizer=tokenizer,
                    max_length=max_length), chunkedexamples),
            total=len(chunkedexamples),
            desc='creating features'))
  inputsarray = [ex for exs in inputsarray for ex in exs]
  features = []
  skip_num = 0
  for inputs in inputsarray:
    if inputs is None:
      skip_num += 1
      continue

    features.append(inputs)

  logger.info('Skiped {} examples'.format(skip_num))

  return features


class AirOPEProcessor(DataProcessor):
  """Processor for the data set for airdialogue OPE."""

  def get_train_examples(self, data_dir):
    """See base class."""
    with open(os.path.join(data_dir, 'data.json'), 'r') as json_file:
      json_data = [json.loads(l.rstrip()) for l in json_file]
    #print(data_dir, ":  ", len(json_data))
    #import ipdb; ipdb.set_trace()
    return self._create_examples(json_data, 'train')

  def get_dev_examples(self, data_dir):
    """See base class."""
    import inspect
    raise RuntimeError('should not access this function: ' +
                       inspect.currentframe().f_code.co_name)

  def get_test_examples(self, data_dir):
    """See base class."""
    import inspect
    raise RuntimeError('should not access this function: ' +
                       inspect.currentframe().f_code.co_name)

  def get_labels(self):
    """See base class."""
    import inspect
    raise RuntimeError('should not access this function: ' +
                       inspect.currentframe().f_code.co_name)

  def _create_examples(self, data, set_type):
    """Creates examples for the training, dev and test sets."""
    examples = []
    for (i, dic) in enumerate(data):
      guid = '%s-%s' % (set_type, str(i))
      text_gen_agent = dic['gen_agent_response']
      text_ref_agent = dic['ref_agent_response']
      text_ref_customer = dic['ref_customer_response']
      #print(len(text_gen_agent),len(text_ref_agent),len(text_ref_customer))
      reward = dic['reward']
      assert len(text_gen_agent) == len(text_ref_agent) == len(
          text_ref_customer)
      examples.append(InputExample(guid=guid, text_gen_agent=text_gen_agent, text_ref_agent=text_ref_agent, \
                                   text_ref_customer=text_ref_customer, reward=reward))
    return examples


class ParlAIOPEProcessor(AirOPEProcessor):
  reward_key = 'reward'

  def get_reward(self, dic):
    return dic['reward'][self.reward_key]

  def _create_examples(self, data, set_type):
    """Creates examples for the training, dev and test sets."""
    examples = []
    for (i, dic) in enumerate(data):
      guid = '%s-%s' % (set_type, str(i))

      text_ref_customer = [turn['text'] for turn in dic['dialog'][0::3]]
      text_ref_agent = [turn['text'] for turn in dic['dialog'][1::3]]
      text_gen_agent = [turn['text'] for turn in dic['dialog'][2::3]]
      #import ipdb; ipdb.set_trace()
      #print(len(text_gen_agent),len(text_ref_agent),len(text_ref_customer))
      reward = self.get_reward(dic)
      assert len(text_gen_agent) == len(text_ref_agent) == len(
          text_ref_customer)
      examples.append(InputExample(guid=guid, text_gen_agent=text_gen_agent, text_ref_agent=text_ref_agent, \
                                   text_ref_customer=text_ref_customer, reward=reward))
    return examples


class ParlAIOPEProcessorALL(ParlAIOPEProcessor):

  def get_reward(self, dic):
    return dic['reward']


class Convai2OPEProcessor(ParlAIOPEProcessor):
  pass


class Convai2OPEProcessorEnjoy(Convai2OPEProcessor):
  reward_key = 'enjoy'


class Convai2OPEProcessorRep(Convai2OPEProcessor):
  reward_key = 'avoid_rep'


class Convai2OPEProcessorAll(Convai2OPEProcessor):

  def get_reward(self, dic):
    return dic['reward']


airope_processors = {
    'syn_air_ope': AirOPEProcessor,
    'air_ope': ParlAIOPEProcessor,
    'air_ope_all': ParlAIOPEProcessorALL,
    'human_air_ope_all': ParlAIOPEProcessorALL,
    'selfplay_air_ope_all': ParlAIOPEProcessorALL,
    'convai2_ope': Convai2OPEProcessor,
    'convai2_ope_enjoy': Convai2OPEProcessorEnjoy,
    'convai2_ope_rep': Convai2OPEProcessorRep,
    'convai2_ope_all': ParlAIOPEProcessorALL,
}
