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

"""Train the model on training set."""


import argparse
import json
import os

import peft
import torch
import transformers
from utils import CastOutputToFloat

Dataset = torch.utils.data.Dataset


class BackwardDataset(Dataset):
  def __init__(self, ex):
    self.data = ex

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    return {
        'FR': fr_template.format(question=item['question'],
                                 answer=item['forward_reasoning']),
        'BQ': bq_template.format(question=item['question'],
                                 ans=item['gold_answer'],
                                 backward_question=item['backward_question']),
        BR'': br_template.format(backward_question=item['backward_question'],
                                 backward_reasoning=item['backward_reasoning'])
    }


class BackwardDataCollator:
  """Collate the data for training."""

  def __init__(self,
               tokenizerr,
               label_pad_token_id=-100):
    self.tokenizerr = tokenizerr
    self.label_pad_token_id = label_pad_token_id

  def __call__(self, features):
    new_feat = {}
    for key in ['FR', 'BQ', 'BR']:
      new_feat[f'{key}'] = {}
      texts = [f[key] for f in features]
      inputs = self.tokenizerr(texts,
                               padding=True,
                               truncation=True,
                               return_tensors='pt')

      new_feat[f'{key}']['input_ids'] = inputs['input_ids']
      new_feat[f'{key}']['attention_mask'] = inputs['attention_mask']
      new_feat[f'{key}']['labels'] = inputs['input_ids']
    return new_feat


class BackwardTrainer(transformers.Trainer):
  """Collate the data for training."""

  def compute_loss(self, model_instance, inputs):
    loss1 = model_instance(**inputs['FR']).loss
    loss2 = model_instance(**inputs['BQ']).loss
    loss3 = model_instance(**inputs['BR']).loss
    return (loss1 + loss2 + loss3) / 3

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--n', default=0, type=int)
  parser.add_argument('--task', default='ANLI', type=str)
  parser.add_argument('--model', default='mistral', type=str)
  parser.add_argument('--model_dir', default='', type=str)

  args = parser.parse_args()

  if args.model == 'mistral-7b':
    base_model = 'mistralai/Mistral-7B-Instruct-v0.3'
  elif args.model == 'gemma-2b':
    base_model = 'google/gemma-2b-it'
  elif args.model == 'gemma-7b':
    base_model = 'google/gemma-7b-it'
  else:
    raise ValueError(f'Unsupported model: {args.model}')

  if 'mistral' in args.model:
    fr_template = """<s>[INST] Answer the following question:\n### Question: {question} [/INST] ### Answer: {answer}</s>"""
    bq_template = """<s>[INST] Generate the inverse question based on the following seed question and its answer:\n### Seed Question: {question} The correct answer is ({ans}). [/INST] ### Inverse Question: {backward_question}</s>"""
    br_template = """<s>[INST] Answer the backward question:\n### Question: {backward_question} [/INST] ### Answer: {backward_reasoning}</s>"""
  elif 'gemma' in args.model:
    fr_template = """<bos><start_of_turn>user\nAnswer the following question:\n### Question: {question}<end_of_turn>\n<start_of_turn>model\n### Answer: {answer}<eos>"""
    bq_template = """<bos><start_of_turn>user\nGenerate the inverse question based on the following seed question and its answer:\n### Seed Question: {question} The correct answer is ({ans}).<end_of_turn>\n<start_of_turn>model\n### Inverse Question: {backward_question}<end_of_turn><eos>"""
    br_template = """<bos><start_of_turn>user\nAnswer the backward question:\n### Question: {backward_question}<end_of_turn>\n<start_of_turn>model\n### Answer: {backward_reasoning}<eos>"""

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      base_model,
      model_max_length=1024,
      padding_side='right'
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.add_bos_token = False
  tokenizer.add_eos_token = False

  model = transformers.AutoModelForCausalLM.from_pretrained(
      base_model,
      device_map='auto',
      cache_dir=args.model_dir
  )

  model.config.use_cache = False
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  model.lm_head = CastOutputToFloat(model.lm_head)

  lora_config = peft.LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=['q_proj', 'v_proj'],
      lora_dropout=0.05,
      bias='none',
      task_type='CAUSAL_LM'
  )

  model = peft.get_peft_model(model, lora_config)
  teacher_data_file = f'./training_data/{args.task}.json'
  print(teacher_data_file)
  with open(teacher_data_file, 'r') as f:
    data = json.load(f)

  num_samples = 0
  training_data = []
  for sample in data:
    if sample['forward_pred'] == sample['gold_answer']:
      training_data.append(sample)
      num_samples += 1
    if args.n and (num_samples / len(data)) >= (args.n / 100):
      break

  print(f'Using {args.n}% of data. ({num_samples}/{len(data)})')
  print(len(training_data))
  dataset = BackwardDataset(training_data)
  data_collator = BackwardDataCollator(tokenizer)

  lr = 5e-6 if 'mistral' in args.model else 2e-4
  training_args = transformers.TrainingArguments(
      output_dir=f'./outputs/{args.model}_{args.task}_{args.n}',
      save_strategy='epoch',
      num_train_epochs=10,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      learning_rate=lr,
      weight_decay=0.001,
      logging_dir='./logs',
      logging_steps=100,
      remove_unused_columns=False,
      fp16=False,
      bf16=True,
      warmup_ratio=0.3,
      lr_scheduler_type='constant'
  )

  trainer = BackwardTrainer(
      model=model,
      args=training_args,
      train_dataset=dataset,
      data_collator=data_collator
  )

  trainer.train()
  save_path = f'./checkpoints/{args.model}_{args.task}_{args.n}'
  os.makedirs(save_path, exist_ok=True)
  trainer.model.save_pretrained(save_path)
