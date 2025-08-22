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

"""ACTTrainer is a subclass of DPOTrainer that implements the RL update loop of

the Action-Based Contrastive Self-Training algorithm.
ACTTrainer is a subclass of DPOTrainer which can be found at:
https://huggingface.co/docs/trl/trainer#trl.DPOTrainer
"""

from collections import defaultdict
import logging
import os
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from act.metrics.base_metrics import BaseMetrics
from act.models.action_classifier_model import ActionClassifierModel
from act.models.intent_model import UserIntentModel
from act.models.simulator_model import SimulatorModel
from act.simulation.simulator import Simulator
from datasets import Dataset
import torch
import torch.nn as nn
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import (
    DPOConfig,
    DPOTrainer,
)


class ACTTrainer(DPOTrainer):

  def __init__(
      self,
      model,
      ref_model,
      args,
      action_model,
      user_simulator,
      intent_summarization_model,
      train_dataset,
      eval_dataset,
      tokenizer,
      metrics,
      beta = 0.1,
      label_smoothing = 0,
      loss_type = 'sigmoid',
      label_pad_token_id = -100,
      padding_value = 0,
      truncation_mode = 'keep_end',
      data_collator = None,
      model_init = None,
      callbacks = None,
      optimizers = (None, None),
      preprocess_logits_for_metrics = None,
      peft_config = None,
      compute_metrics = None,
      special_stop_token = '\n',
      dialogue_act_classifier=None,
      sample_frequency = 1,
      hard_replacement_frequency = 1,
  ):

    super().__init__(
        model=model,
        ref_model=ref_model,
        beta=beta,
        label_smoothing=label_smoothing,
        loss_type=loss_type,
        args=args,
        data_collator=data_collator,
        label_pad_token_id=label_pad_token_id,
        padding_value=padding_value,
        truncation_mode=truncation_mode,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model_init=model_init,
        callbacks=callbacks,
        optimizers=optimizers,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
    )
    self.global_batches = 0
    self.hard_example_cache = []
    self.existing_rejected_responses = set()

    self.is_in_evaluate = False

    self.tokenizer = tokenizer

    self.special_stop_token = tokenizer.encode(special_stop_token)[-1]
    self.dialogue_act_classifier = dialogue_act_classifier

    self.sample_frequency = sample_frequency
    self.action_model = action_model
    self.user_simulator = user_simulator
    self.intent_summarization_model = intent_summarization_model
    self.metrics = metrics

  def evaluate(
      self,
      eval_dataset = None,
      ignore_keys = None,
      metric_key_prefix = 'eval',
  ):
    self.is_in_evaluate = True
    metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    self.is_in_evaluate = False
    return metrics

  def post_process_features(
      self, features, metadata
  ):
    if ' '.join(features['chosen'].lower().split()) == ' '.join(
        features['rejected'].lower().split()
    ):
      logging.info('Chosen is identical to rejected. Attempting to replace.')
      logging.info(
          'Replacing chosen {} -> {} and rejected {} -> {}'.format(
              features['chosen'],
              metadata['chosen'],
              features['rejected'],
              metadata['rejected'],
          )
      )
      features = {
          'prompt': features['prompt'],
          'chosen': metadata['chosen'],
          'rejected': metadata['rejected'],
      }
      if features['chosen'] == features['rejected']:
        rejected = 'Assistant: Could you clarify what you are asking for?'
        logging.info(
            'Still identical. rejected: %s -> %s',
            metadata['rejected'],
            rejected,
        )
        features['rejected'] = rejected
    elif features['chosen'].startswith('User:'):
      if 'Assistant:' in features['chosen']:
        features['chosen'] = (
            'Assistant: ' + features['chosen'].split('Assistant:')[-1].strip()
        )
      else:
        features['chosen'] = metadata['chosen']
      logging.info('Buggy rollout; New features: %s', features)
    return features

  def concatenated_forward(
      self, model, batch
  ):
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    self.global_batches += 1
    if not self.is_in_evaluate:
      if (
          self.sample_frequency > 0
          and self.global_batches % self.sample_frequency == 0
      ):
        new_batch = defaultdict(list)
        print('Batch chosen policies: {}'.format(batch['chosen_policy']))
        for i, input_ids in enumerate(batch['prompt_input_ids']):
          simulator = Simulator(
              model=model,
              tokenizer=self.tokenizer,
              user_intent_model=self.intent_summarization_model,
              user_simulator_model=self.user_simulator,
              action_model=self.action_model,
          )

          trajectory = simulator.generate_trajectory(
              inputs=input_ids,
              chosen_policy=batch['chosen_policy'][i],
              prompt=batch['prompt'][i],
              gold_trajectory=batch['gold_trajectory'][i],
              max_input_length=self.max_length,
          )

          if self.metrics.conditon_checker(
              prompt=batch['prompt'][i],
              gold_target=batch['gold_target'][i],
              final_answer=trajectory['final_answer'],
              gold_trajectory=batch['gold_trajectory'][i],
              response=trajectory['response'],
          ):
            rejected = trajectory['response']
            chosen = batch['chosen'][i]
          else:
            chosen = trajectory['response']
            rejected = batch['rejected'][i]
          features = {
              'prompt': batch['prompt'][i],
              'chosen': chosen,
              'rejected': rejected,
          }
          features = self.post_process_features(
              features,
              {
                  'chosen': batch['chosen'][i],
                  'rejected': batch['rejected'][i],
              },
          )
          for k in features.keys():
            new_batch[k].append(features[k])
          logging.info(
              'Chosen:\n%s\nRejected:\n%s\n----------',
              features['chosen'],
              features['rejected'],
          )

          del features

        new_batch = Dataset.from_dict(new_batch)
        new_batch = new_batch.map(self.tokenize_row)
        batch = new_batch
        del new_batch
        batch = self.data_collator(batch)

    concatenated_batch = self.concatenated_inputs(
        batch,
        is_encoder_decoder=self.is_encoder_decoder,
        label_pad_token_id=self.label_pad_token_id,
        padding_value=self.padding_value,
        device=self.accelerator.device,
    )
    len_chosen = batch['chosen_labels'].shape[0]

    model_kwargs = (
        {
            'labels': concatenated_batch['concatenated_labels'],
            'decoder_input_ids': concatenated_batch.pop(
                'concatenated_decoder_input_ids', None
            ),
        }
        if self.is_encoder_decoder
        else {}
    )
    all_logits = model(
        concatenated_batch['concatenated_input_ids'],
        attention_mask=concatenated_batch['concatenated_attention_mask'],
        **model_kwargs,
    ).logits

    all_logps, _ = self.get_batch_logps(
        all_logits,
        concatenated_batch['concatenated_labels'],
        is_encoder_decoder=self.is_encoder_decoder,
        label_pad_token_id=self.label_pad_token_id,
    )

    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, None)
