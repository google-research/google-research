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

import logging
import os
import time
import math
from dataclasses import dataclass, field
import collections
from collections.abc import Mapping
from enum import Enum
from typing import List, Optional, Union, Any, Callable, Dict, NewType, Tuple
import numpy as np

InputDataClass = NewType("InputDataClass", Any)

import torch
from filelock import FileLock
import msgpack
from torch.utils.data.dataset import Dataset
from torch.optim import SGD

import transformers
from transformers import Trainer
from transformers.optimization import *

from transformers.tokenization_utils import PreTrainedTokenizer
from airope_processors import airope_convert_examples_to_features, airope_processors, InputFeatures

logger = logging.getLogger(__name__)


def get_invsqrt_schedule_with_warmup(optimizer,
                                     num_warmup_steps,
                                     last_epoch=-1):

  def lr_lambda(current_step):
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, 1 / float(current_step - num_warmup_steps + 1)**0.5)

  return LambdaLR(optimizer, lr_lambda, last_epoch)


@dataclass
class AirOPETrainingArguments:
  """
    Arguments pertaining to what data we are going to input our model for
    training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

  task_name: str = field(
      metadata={
          "help":
              "The name of the task to train on: " +
              ", ".join(airope_processors.keys())
      })
  data_dir: str = field(
      metadata={
          "help":
              "The input data dir. Should contain the .json files (or other data files) for the task."
      })
  max_seq_length: int = field(
      default=512,
      metadata={
          "help":
              "The maximum total input sequence length after tokenization. Sequences longer "
              "than this will be truncated, sequences shorter will be padded."
      },
  )
  overwrite_cache: bool = field(
      default=False,
      metadata={"help": "Overwrite the cached training and evaluation sets"})
  workers: int = field(
      default=20,
      metadata={"help": "Number of workers for processing data"},
  )

  def __post_init__(self):
    self.task_name = self.task_name.lower()


class AirOPEDataset(Dataset):
  """
    This will be superseded by a framework-agnostic approach
    soon.
    """

  args: AirOPETrainingArguments
  output_mode: str
  features: List[InputFeatures]

  def __init__(
      self,
      args,
      tokenizer,
      cache_dir = None,
  ):
    self.args = args
    self.processor = airope_processors[args.task_name]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        cache_dir if cache_dir is not None else args.data_dir,
        "cached_{}_{}_{}".format(
            tokenizer.__class__.__name__,
            str(args.max_seq_length),
            args.task_name,
        ),
    )

    # Make sure only the first process in distributed training processes the dataset,
    # and the others will use the cache.
    lock_path = cached_features_file + ".lock"
    with FileLock(lock_path):

      if os.path.exists(cached_features_file) and not args.overwrite_cache:
        start = time.time()
        with open(cached_features_file, "rb") as read_file:
          feature_dicts = msgpack.unpackb(read_file.read(), raw=False)
        logger.info(
            f"Loading features from cached file {cached_features_file} [took %.3f s]",
            time.time() - start)
      else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")

        examples = self.processor.get_train_examples(args.data_dir)
        feature_dicts = airope_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            workers=args.workers,
        )
        start = time.time()
        with open(cached_features_file, "wb") as out_file:
          out_file.write(msgpack.packb(feature_dicts))

        # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
        logger.info("Saving features into cached file %s [took %.3f s]",
                    cached_features_file,
                    time.time() - start)
    self.features = [InputFeatures(**f) for f in feature_dicts]

  def __len__(self):
    return len(self.features)

  def __getitem__(self, i):
    return self.features[i]

  def get_labels(self):
    import inspect
    raise RuntimeError("should not access this function: " +
                       inspect.currentframe().f_code.co_name)


def airope_data_collator(
    features):
  batch = {}
  if isinstance(features[0].reward, float):
    batch["reward"] = torch.tensor([f.reward for f in features],
                                   dtype=torch.float)
  elif isinstance(features[0].reward, dict):
    batch["reward"] = {}
    keys = features[0].reward.keys()
    for k in keys:
      batch["reward"][k] = torch.tensor([f.reward[k] for f in features],
                                        dtype=torch.float)
  max_length = max(f.total_length for f in features)
  # hack for efficiency
  max_length = math.ceil(max_length / 8) * 8
  for key in ["input_ids", "token_type_ids", "position_ids"]:
    batch[key] = torch.tensor([getattr(f, key) for f in features],
                              dtype=torch.long)
    batch[key] = batch[key][:, :max_length]
  batch["attention_mask"] = torch.tensor([f.attention_mask for f in features],
                                         dtype=torch.bool)
  batch["attention_mask"] = batch[
      "attention_mask"][:, :max_length, :max_length]  # batch x seq x seq

  batch["true_conv_end"] = torch.tensor([f.true_conv_end for f in features],
                                        dtype=torch.long)
  for key in ["ref_c_end_ids", "ref_a_end_ids", "gen_a_end_ids"]:
    max_size = max([len(getattr(f, key)) for f in features])
    batch[key] = torch.tensor([
        getattr(f, key) + [-1] * (max_size - len(getattr(f, key)))
        for f in features
    ],
                              dtype=torch.long)
  return batch


class AvgMeter:

  def __init__(self):
    self.reset()

  def reset(self):
    self.count = 0
    self.sum = 0
    self.val = None

  def update(self, val, n=1):
    if val is not None:
      self.val = val
      self.sum += val * n
      self.count += n

  @property
  def avg(self):
    return self.sum / self.count if self.count > 0 else self.val


class LoggingMeter(Mapping):

  def __init__(self):
    self.meters = {}

  def reset(self):
    for k in self.meters:
      self.meters[k].reset()

  def __iter__(self):
    return iter(self.meters)

  def __len__(self):
    return len(self.meters)

  def __getitem__(self, key):
    return self.meters[key].avg

  def __setitem__(self, key, value):
    if key not in self.meters:
      self.meters[key] = AvgMeter()
    self.meters[key].reset()
    self._update(key, value)

  def items(self):
    for k, v in self.meters.items():
      yield k, v.avg

  def _update(self, k, v):
    if torch.is_tensor(v):
      self.meters[k].update(v.mean().item(), v.numel())
    elif isinstance(v, (float, int)):
      self.meters[k].update(v)
    else:
      raise NotImplementedError("Not implemented for this type logging")

  def update(self, loggings):
    for k, v in loggings.items():
      if k not in self.meters:
        self.meters[k] = AvgMeter()
      self._update(k, v)

  def getdict(self):
    r = {}
    for k in self.meters:
      r[k] = self.meters[k].avg
    return r


class AirOPETrainer(Trainer):

  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs)
    self.training_loggings = LoggingMeter()

  def evaluate(self, *args, **kargs):
    _log_info = self.training_loggings.getdict()
    keys = list(_log_info.keys())
    for k in keys:
      if k.startswith("est_reward_dual"):
        _log_info[k +
                  "_normalized"] = _log_info[k] / _log_info["c_ref_term_values"]
    self._log(_log_info)
    self.training_loggings.reset()

  def get_optimizers(
      self, num_training_steps
  ):
    """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's
        init,
        or override this method in a subclass.
        """
    if self.optimizers is not None:
      return self.optimizers
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": self.args.weight_decay,
        },
        {
            "params": [
                p for n, p in self.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if self.args.optimizer == "sgd":
      optimizer = SGD(optimizer_grouped_parameters, lr=self.args.learning_rate, momentum=self.args.sgd_momentum, \
                      weight_decay=self.args.weight_decay)
    elif self.args.optimizer == "adam":
      optimizer = AdamW(
          optimizer_grouped_parameters,
          lr=self.args.learning_rate,
          eps=self.args.adam_epsilon)
    if self.args.lr_schedule == "constant":
      scheduler = get_constant_schedule_with_warmup(
          optimizer, num_warmup_steps=self.args.warmup_steps)
    elif self.args.lr_schedule == "linear":
      scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.args.warmup_steps,
          num_training_steps=num_training_steps)
    elif self.args.lr_schedule == "invsqrt":
      scheduler = get_invsqrt_schedule_with_warmup(
          optimizer, num_warmup_steps=self.args.warmup_steps)
    return optimizer, scheduler

  def _training_step(self, model, inputs,
                     optimizer):
    model.train()
    for k, v in inputs.items():
      if torch.is_tensor(v):
        inputs[k] = v.to(self.args.device)
      elif isinstance(v, dict):
        for k2, v2 in v.items():
          inputs[k][k2] = v2.to(self.args.device)

    #print("--------------",inputs['input_ids'].shape)
    #if inputs['input_ids'].shape[0] < 64:
    #    import ipdb;ipdb.set_trace()
    outputs = model(**inputs)
    loss = outputs[
        0]  # model outputs are always tuple in transformers (see doc)
    loggings = outputs[1]
    self.training_loggings.update(loggings)
    if self.args.save_embedding:
      embeddings_ref = []
      cnet_outputs = outputs[-1][0].detach().cpu()
      for cnet_out, a_ids in zip(cnet_outputs, inputs["ref_a_end_ids"]):
        embeddings_ref.append([cnet_out[i, :].numpy() for i in a_ids if i > -1])
      embeddings_gen = []
      for cnet_out, a_ids in zip(cnet_outputs, inputs["gen_a_end_ids"]):
        embeddings_gen.append([cnet_out[i, :].numpy() for i in a_ids if i > -1])
      print(self.args.output_dir)
      np.save(
          os.path.join(self.args.output_dir, "embeddings_ref"), embeddings_ref)
      np.save(
          os.path.join(self.args.output_dir, "embeddings_gen"), embeddings_gen)
      np.save(
          os.path.join(self.args.output_dir, "reward"),
          inputs["reward"].cpu().numpy())
      import ipdb
      ipdb.set_trace()

    loss = loss.mean()
    if self.args.gradient_accumulation_steps > 1:
      loss = loss / self.args.gradient_accumulation_steps

    if self.args.fp16:
      with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    else:
      loss.backward()

    return loss.item()
