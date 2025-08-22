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

"""Data utils."""

import copy
import os

from datasets import Dataset
from datasets import DatasetDict
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import torch


def clm_tokenize_function(
    examples, tokenizer, max_length=64, truncation=True, ignore_index=-100
):
  """Tokenize the concatenated text."""
  # examples are from huggingface datasets that contains
  # (instruction, answer) pairs.
  # during training, the loss of instruction tokens will be ignored.
  instructions = examples['instruction']
  answers = examples['answer']

  # concatenate instructions and answers
  texts = [f'{instructions[i]}{answers[i]}' for i in range(len(instructions))]
  tokenized_texts = tokenizer(
      texts, max_length=max_length - 1, padding=False, truncation=truncation
  )
  # add <\s> token
  tokenized_texts['input_ids'] = [
      x + [tokenizer.eos_token_id] for x in tokenized_texts['input_ids']
  ]
  tokenized_texts['attention_mask'] = [
      x + [1] for x in tokenized_texts['attention_mask']
  ]
  tokenized_texts['labels'] = copy.deepcopy(tokenized_texts['input_ids'])

  # tokenize instructions to get the length of instruction tokens
  tokenized_instructions = tokenizer(
      instructions,
      max_length=max_length - 1,
      padding=False,
      truncation=truncation,
  )
  tokenized_instruction_lengths = [
      len(x) for x in tokenized_instructions['input_ids']
  ]

  # zero out the loss of instruction tokens
  for i in range(len(tokenized_texts['labels'])):
    np_labels = np.array(tokenized_texts['labels'][i])
    np_labels[0 : tokenized_instruction_lengths[i]] = ignore_index
    tokenized_texts['labels'][i] = np_labels.tolist()

  return tokenized_texts


# pylint: disable=unused-argument
def eval_tokenize_function(
    examples, tokenizer, max_length=64, truncation=True, ignore_index=-100
):
  """Tokenize the eval data."""
  # examples are from huggingface datasets that contains
  # (instruction, answer) pairs.
  # during training, the loss of instruction tokens will be ignored
  instructions = examples['instruction']
  answers = examples['answer']

  # input_ids contain instruction only
  texts = instructions
  tokenized_texts = tokenizer(
      texts, max_length=max_length - 1, padding=False, truncation=truncation
  )
  tokenized_texts['labels'] = tokenizer(
      answers, max_length=max_length - 1, padding=False, truncation=truncation
  )['input_ids']

  return tokenized_texts


def _preprocess_text_dataset_self_instruct(json_data, prompt_dict):
  """Preprocess instruct dataset."""
  num_examples = len(json_data)
  instructions = []
  answers = []
  for i in range(num_examples):
    if not json_data[i]['input']:
      instructions.append(
          prompt_dict['prompt_no_input'].format(
              instruction=json_data[i]['instruction']
          )
      )
    else:
      instructions.append(
          prompt_dict['prompt_input'].format(
              instruction=json_data[i]['instruction'],
              input=json_data[i]['input'],
          )
      )
    answers.append(json_data[i]['output'])

  text_dataset = Dataset.from_dict(
      {'instruction': instructions, 'answer': answers}
  )
  return text_dataset


def _preprocess_text_dataset_chatbot_arena_33k_one_round(
    hf_dataset, prompt_dict, split='train'
):
  raise ValueError('deprecated function')


def _preprocess_text_dataset_customized_instructions(json_file, prompt_dict):
  """Preprocess customized instructions."""
  num_examples = len(json_file)
  instructions = []
  answers = []
  if prompt_dict['type'] == 'self_instruct':
    for i in range(num_examples):
      if not json_file[i]['input']:
        instructions.append(
            prompt_dict['prompt_no_input'].format(
                instruction=json_file[i]['instruction']
            )
        )
      else:
        instructions.append(
            prompt_dict['prompt_input'].format(
                instruction=json_file[i]['instruction'],
                input=json_file[i]['input'],
            )
        )
      answers.append(json_file[i]['output'])
  elif (
      prompt_dict['type'] == 'vicuna'
      or prompt_dict['type'] == 'llama'
      or prompt_dict['type'] == 'mistral'
  ):
    for i in range(num_examples):
      instructions.append(
          prompt_dict['prompt'].format(instruction=json_file[i]['instruction'])
      )
      answers.append(json_file[i]['output'])
  elif prompt_dict['type'] == 'uncond_generation':
    num_samples = int(json_file[0]['instruction'])
    instructions.extend([''])
    answers.extend([''])
  elif prompt_dict['type'] == 'len_cond_generation':
    num_samples = int(json_file[0]['instruction'])
    instructions.extend([
        'An instruction with fewer than 100 words:',
        'An instruction with 100-200 words:',
        'An instruction with more than 200 words:',
    ])
    answers.extend(['', '', ''])

  text_dataset = Dataset.from_dict(
      {'instruction': instructions, 'answer': answers}
  )
  if prompt_dict['type'] == 'uncond_generation':
    text_dataset.repeats = [num_samples]
  elif prompt_dict['type'] == 'len_cond_generation':
    text_dataset.repeats = [
        int(num_samples * 0.85),
        int(num_samples * 0.075),
        int(num_samples * 0.075),
    ]
  return text_dataset


def _preprocess_text_dataset_generate_syn_arena180k_instructions(
    hf_dataset, prompt_dict, split='train'
):
  """Preprocess synthetic instructions."""
  # get TRANSFORMERS_OFFLINE variable
  transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE')
  if transformers_offline == '1':
    raise NotImplementedError
  else:
    # running on local machine
    nltk.download('punkt')

  prompt_tempalte = prompt_dict['prompt']
  ds = hf_dataset[split]
  # for generating synthetic instructions, we use unconditional generation.
  instructions = []
  # the target tokens here is to generate synthetic instructions.
  answers = ds['instruction']
  num_answers = len(answers)

  instruction_lengths = []
  for i in range(num_answers):
    instruction_lengths.append(len(word_tokenize(answers[i])))

  # quantiles = np.quantile(instruction_lengths, [0.3, 0.5, 0.7, 0.9])

  for _ in range(num_answers):
    if prompt_dict['type'] == 'uncond_generation':
      instructions.append(prompt_tempalte)
    else:
      raise ValueError('not implemented yet')

  text_dataset = Dataset.from_dict(
      {'instruction': instructions, 'answer': answers}
  )
  print(text_dataset)
  return text_dataset


def _preprocess_text_dataset_generate_syn_arena33k_instructions(
    hf_dataset, prompt_dict, split='train'
):
  raise ValueError('deprecated function')


def _preprocess_text_dataset_lablled_instructions(
    hf_dataset, prompt_dict, split
):
  """Preprocess labelled instructions."""
  hf_dataset = hf_dataset[split]
  instructions = hf_dataset['instructions']
  answers = hf_dataset['answers']
  formated_instructions = [
      prompt_dict['prompt'].format(instruction=instruction)
      for instruction in instructions
  ]

  text_dataset = Dataset.from_dict(
      {'instruction': formated_instructions, 'answer': answers}
  )
  print(text_dataset)
  return text_dataset


def preprocess_text_dataset(
    text_dataset, dataset_name, prmopt_template=None, split='train'
):
  """Preprocess text dataset."""
  if dataset_name == 'chatbot_arena_instructions_train180k':
    if prmopt_template is None:
      prmopt_template = 'uncond_generation'
    return _preprocess_text_dataset_generate_syn_arena180k_instructions(
        text_dataset, prompt_dict=get_prompt_dict(prmopt_template), split=split
    )
  elif 'labelled' in dataset_name:
    if prmopt_template is None:
      prmopt_template = 'vicuna'
    return _preprocess_text_dataset_lablled_instructions(
        text_dataset, prompt_dict=get_prompt_dict(prmopt_template), split=split
    )
  else:
    raise f'Dataset {dataset_name} is not supported.'


# pylint: disable=invalid-name
def get_prompt_dict(prmopt_template):
  """Get prompt dict."""
  if prmopt_template == 'self_instruct':
    PROMPT_DICT = {
        'type': 'self_instruct',
        'prompt_input': (
            'Below is an instruction that describes a task, paired with an'
            ' input that provides further context. Write a response that'
            ' appropriately completes the request.\n\n###'
            ' Instruction:\n{instruction}\n\n### Input:\n{input}\n\n###'
            ' Response:'
        ),
        'prompt_no_input': (
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n'
            '### Instruction:\n{instruction}\n\n### Response:'
        ),
    }
  elif prmopt_template == 'vicuna':
    PROMPT_DICT = {
        'type': 'vicuna',
        'prompt': (
            'A chat between a curious user and an artificial intelligence'
            ' assistant. The assistant gives helpful, detailed, and polite'
            " answers to the user's questions. USER: {instruction} ASSISTANT:"
        ),
    }
  elif prmopt_template == 'llama':
    PROMPT_DICT = {
        'type': 'llama',
        'prompt': (
            '[INST] <<SYS>>\nYou are a helpful, respectful and honest'
            ' assistant.\n<</SYS>>\n\n{instruction} [/INST]'
        ),
    }
  elif prmopt_template == 'uncond_generation':  # unconditional generation
    PROMPT_DICT = {'type': 'uncond_generation', 'prompt': ''}
  elif (
      prmopt_template == 'len_cond_generation'
  ):  # generation conditional on length
    PROMPT_DICT = {'type': 'len_cond_generation', 'prompt': '{length_type}'}
  elif (
      prmopt_template == 'mistral'
  ):  # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    PROMPT_DICT = {'type': 'mistral', 'prompt': '[INST] {instruction} [/INST]'}
  else:
    raise f'Prompt template {prmopt_template} is not supported.'
  return PROMPT_DICT


class TokenizedSupervisedInstructDataset(Dataset):
  """Tokenize the concatenated text from dataset of (insturction, answer) pairs."""

  def __init__(
      self,
      dataset_or_name,
      tokenizer,
      split='train',
      max_length=64,
      truncation=True,
      num_proc=4,
      tokenize_type='clm',
      prmopt_template=None,
      exp_path='./',
  ):

    # we shall build the text dataset from scratch
    # a processed text dataset contain two columns: instruction and answer
    if isinstance(dataset_or_name, str):
      dataset_name = dataset_or_name
      if dataset_name == 'self_instruct':
        raise NotImplementedError
      elif (
          dataset_name == 'chatbot_arena_instructions_train180k'
          or 'labelled' in dataset_name
      ):
        text_dataset = DatasetDict.load_from_disk('data/' + dataset_name)
      else:
        raise NotImplementedError

    # we alread have a text dataset, no need to build from scratch
    elif isinstance(dataset_or_name, dict):
      print('creating dataset from python dict')
      text_dataset = dataset_or_name
      dataset_name = 'customized_instructions'

    self.text_dataset = text_dataset
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.truncation = truncation
    self.num_proc = num_proc
    self.tokenize_type = tokenize_type
    self.dataset_name = dataset_name
    self.dataset_split = split
    self.prmopt_template = prmopt_template
    self.get_tokenized_dataset()

  def get_tokenized_dataset(self):
    processed_text_dataset = preprocess_text_dataset(
        self.text_dataset,
        self.dataset_name,
        split=self.dataset_split,
        prmopt_template=self.prmopt_template,
    )

    # tokenize the text dataset
    if self.tokenize_type == 'clm':
      # this option concatenates the instruction and answer, and tokenize
      # the concatenated text.
      # the loss on instruction is ignored
      tokenize_func = clm_tokenize_function
    else:
      # this option only tokenizes the instruction
      # usually used during inference
      tokenize_func = eval_tokenize_function

    self.tokenized_text_dataset = processed_text_dataset.map(
        lambda x: tokenize_func(
            x,
            self.tokenizer,
            max_length=self.max_length,
            truncation=self.truncation,
        ),
        batched=True,
        num_proc=self.num_proc,
    )

    if hasattr(processed_text_dataset, 'repeats'):
      new_input_ids = []
      new_attention_mask = []
      new_labels = []

      for i in range(len(self.tokenized_text_dataset)):
        new_input_ids.extend(
            [self.tokenized_text_dataset[i]['input_ids']]
            * processed_text_dataset.repeats[i]
        )
        new_attention_mask.extend(
            [self.tokenized_text_dataset[i]['attention_mask']]
            * processed_text_dataset.repeats[i]
        )
        new_labels.extend(
            [self.tokenized_text_dataset[i]['labels']]
            * processed_text_dataset.repeats[i]
        )

      # build a huggerface dataset with the repeated input_ids, attention_mask,
      # and labels. with the from_dict method.
      self.tokenized_text_dataset = Dataset.from_dict({
          'input_ids': new_input_ids,
          'attention_mask': new_attention_mask,
          'labels': new_labels,
      })
      # shuffle the dataset
      self.tokenized_text_dataset = self.tokenized_text_dataset.shuffle(seed=42)

  def __len__(self):
    return len(self.tokenized_text_dataset)

  def __getitem__(self, idx):
    # Return the tokenized text, attention mask, and labels
    if not isinstance(idx, list):
      idx = [idx]

    subset_dataset = self.tokenized_text_dataset[idx]
    input_ids = subset_dataset['input_ids']
    attention_mask = subset_dataset['attention_mask']
    labels = subset_dataset['labels']
    return {
        'index': idx,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
class DataCollatorForSupervisedDataset(object):
  """Data collator for supervised dataset."""

  IGNORE_INDEX = -100

  def __init__(
      self,
      tokenizer,
      padding='longest',
      return_tensors='pt',
      device='cuda',
      padding_side='right',
      max_length=64,
  ):
    self.tokenizer = tokenizer
    self.padding = padding
    self.return_tensors = return_tensors
    self.device = device
    self.padding_side = padding_side
    self.max_length = max_length

  def __call__(self, instances):
    if self.padding not in ['longest', 'max_length']:
      raise ValueError(f'Padding {self.padding} is not supported.')
    if self.return_tensors != 'pt':
      raise ValueError(
          f'return_tensors {self.return_tensors} is not supported.'
      )

    input_ids, labels = tuple(
        [instance[key] for instance in instances]
        for key in ('input_ids', 'labels')
    )
    if self.return_tensors == 'pt':
      input_ids = [torch.tensor(input_id).long() for input_id in input_ids]
      labels = [torch.tensor(label).long() for label in labels]

    if self.padding_side == 'left':
      # reverse each input_id in input_ids
      input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
      labels = [torch.flip(label, dims=[0]) for label in labels]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=self.IGNORE_INDEX
    )

    if self.padding_side == 'left':
      # reverse each input_id in input_ids
      input_ids = torch.flip(input_ids, dims=[1])
      labels = torch.flip(labels, dims=[1])

    input_ids = input_ids.to(self.device)
    labels = labels.to(self.device)

    if self.padding == 'max_length':
      pad_tensor = torch.zeros(
          (input_ids.shape[0], self.max_length - input_ids.shape[1]),
          dtype=torch.long,
      ).to(self.device)
      input_ids = torch.cat(
          [input_ids, pad_tensor + self.tokenizer.pad_token_id], dim=1
      )
      labels = torch.cat([labels, pad_tensor + self.IGNORE_INDEX], dim=1)

    index = [instance['index'] for instance in instances]
    return dict(
        index=index,
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
    )
