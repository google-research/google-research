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

"""Supervised fine-tuning script for decoder language models."""

import logging
import random
import sys

from alignment import apply_chat_template
from alignment import DataArguments
from alignment import get_checkpoint
from alignment import get_kbit_device_map
from alignment import get_peft_config
from alignment import get_quantization_config
from alignment import get_tokenizer
from alignment import H4ArgumentParser
from alignment import ModelArguments
from alignment import SFTConfig
import datasets
from datasets import load_dataset
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import set_seed
from trl import setup_chat_format
from trl import SFTTrainer


logger = logging.getLogger(__name__)


def create_messages(row):
  """convert original row in to chat input."""
  if 'instruction' in row:
    return {
        'messages': [
            {'content': row['instruction'], 'role': 'user'},
            {'content': row['response'], 'role': 'assistant'},
        ]
    }
  elif 'dialogue' in row:
    return {
        'messages': [
            {'content': row['dialogue'], 'role': 'user'},
            {'content': row['summary'], 'role': 'assistant'},
        ]
    }

  elif 'source' in row:
    return {
        'messages': [
            {
                'content': row['source'].split('## Conversation\n')[-1],
                'role': 'user',
            },
            {'content': row['target'], 'role': 'assistant'},
        ]
    }

  elif 'input_text' in row:
    return {
        'messages': [
            {'content': row['input_text'], 'role': 'user'},
            {'content': row['output_text'], 'role': 'assistant'},
        ]
    }

  else:
    print('Data field not supported')
    exit(1)


def prepare_datasets(args):
  """load and process original data."""

  # using local datasets
  if '..' in list(args.dataset_mixer.keys())[0]:
    train_file = f'{list(args.dataset_mixer.keys())[0]}_train.json'
    eval_file = f'{list(args.dataset_mixer.keys())[0]}_vali.json'

    train_dataset = load_dataset(
        'json', data_files=train_file, field='instances', split='train'
    )
    # train split is set by default
    eval_dataset = load_dataset(
        'json', data_files=eval_file, field='instances', split='train'
    )
  # using data from HF
  else:
    data_file = list(args.dataset_mixer.keys())[0]

    train_dataset = load_dataset(data_file, split='train')
    eval_dataset = load_dataset(data_file, split='validation')

  train_dataset = train_dataset.map(create_messages)
  eval_dataset = eval_dataset.map(create_messages)

  return train_dataset, eval_dataset


def process_data(args, data, tokenizer, remove_columns):
  """process data by applying chat format."""

  data = data.map(
      apply_chat_template,
      fn_kwargs={
          'tokenizer': tokenizer,
          'task': 'sft',
          'auto_insert_empty_system_msg': (
              False
          ),  # args.auto_insert_empty_system_msg,
      },
      num_proc=args.preprocessing_num_workers,
      remove_columns=remove_columns,
      desc='Applying chat template',
  )

  return data


def main():
  parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
  model_args, data_args, training_args = parser.parse()

  # Set seed for reproducibility
  set_seed(training_args.seed)

  ###############
  # Setup logging
  ###############
  logging.basicConfig(
      format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
      handlers=[logging.StreamHandler(sys.stdout)],
  )
  log_level = training_args.get_process_log_level()
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  # Log on each process a small summary
  logger.warning(
      'Process rank: %s, device: %s, n_gpu: %s distributed training: %s,'
      ' 16-bits training: %s',
      training_args.local_rank,
      training_args.device,
      training_args.n_gpu,
      bool(training_args.local_rank != -1),
      training_args.fp16,
  )
  logger.info('Model parameters %s', model_args)
  logger.info('Data parameters %s', data_args)
  logger.info('Training/evaluation parameters %s', training_args)

  # Check for last checkpoint
  last_checkpoint = get_checkpoint(training_args)
  if (
      last_checkpoint is not None
      and training_args.resume_from_checkpoint is None
  ):
    logger.info(
        'Checkpoint detected, resuming training at %s.', last_checkpoint
    )

  ################
  # Load tokenizer
  ################
  tokenizer = get_tokenizer(model_args, data_args)

  ###############
  # Load datasets
  ###############
  train_dataset, eval_dataset = prepare_datasets(data_args)

  #######################
  # Load pretrained model
  #######################
  logger.info('*** Load pretrained model ***')
  torch_dtype = (
      model_args.torch_dtype
      if model_args.torch_dtype in ['auto', None]
      else getattr(torch, model_args.torch_dtype)
  )
  quantization_config = get_quantization_config(model_args)

  model_kwargs = dict(
      revision=model_args.model_revision,
      trust_remote_code=model_args.trust_remote_code,
      attn_implementation=model_args.attn_implementation,
      torch_dtype=torch_dtype,
      use_cache=False if training_args.gradient_checkpointing else True,
      device_map=get_kbit_device_map()
      if quantization_config is not None
      else None,
      quantization_config=quantization_config,
  )

  model = model_args.model_name_or_path
  # For ChatML we need to add special tokens and resize the embedding layer
  if (
      '<|im_start|>' in tokenizer.chat_template
      and 'gemma-tokenizer-chatml' not in tokenizer.name_or_path
  ):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    model, tokenizer = setup_chat_format(model, tokenizer)
    model_kwargs = None

  column_names = list(train_dataset.features)
  train_dataset = process_data(
      data_args, train_dataset, tokenizer, column_names
  )
  eval_dataset = process_data(data_args, eval_dataset, tokenizer, column_names)

  logger.info(
      'train samples: %s; eval samples: %s',
      len(train_dataset),
      len(eval_dataset),
  )

  with training_args.main_process_first(
      desc='Log a few random samples from the processed training set'
  ):
    for index in random.sample(range(len(train_dataset)), 3):
      logger.info(
          'Sample %s of the processed training set:\n\n%s',
          index,
          train_dataset[index]['text'],
      )

  ########################
  # Initialize the Trainer
  ########################
  trainer = SFTTrainer(
      model=model,
      model_init_kwargs=model_kwargs,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      dataset_text_field='text',
      max_seq_length=training_args.max_seq_length,
      tokenizer=tokenizer,
      packing=True,
      peft_config=get_peft_config(model_args),
      dataset_kwargs=training_args.dataset_kwargs,
  )

  ###############
  # Training loop
  ###############
  logger.info('*** Train ***')
  checkpoint = None
  if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
  elif last_checkpoint is not None:
    checkpoint = last_checkpoint
  train_result = trainer.train(resume_from_checkpoint=checkpoint)
  metrics = train_result.metrics
  metrics['train_samples'] = len(train_dataset)
  trainer.log_metrics('train', metrics)
  trainer.save_metrics('train', metrics)
  trainer.save_state()

  ##################################
  # Save model and create model card
  ##################################
  logger.info('*** Save model ***')
  trainer.save_model(training_args.output_dir)
  logger.info('Model saved to %s', training_args.output_dir)

  # Save everything else on main process
  kwargs = {
      'finetuned_from': model_args.model_name_or_path,
      'dataset': list(data_args.dataset_mixer.keys()),
      'dataset_tags': list(data_args.dataset_mixer.keys()),
      'tags': ['alignment-handbook'],
  }
  if trainer.accelerator.is_main_process:
    trainer.create_model_card(**kwargs)
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)

  ##########
  # Evaluate
  ##########
  if training_args.do_eval:
    logger.info('*** Evaluate ***')
    metrics = trainer.evaluate()
    metrics['eval_samples'] = len(eval_dataset)
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)

  if training_args.push_to_hub:
    logger.info('Pushing to hub...')
    trainer.push_to_hub(**kwargs)

  logger.info('*** Training complete ***')


if __name__ == '__main__':
  main()
