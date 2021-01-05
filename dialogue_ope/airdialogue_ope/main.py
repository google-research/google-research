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

# coding=utf-8
"""Finetuning the library models for offline policy evaluation of Airdialogue dataset"""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

import torch

from transformers import AutoConfig, AutoTokenizer, EvalPrediction
from modeling import AutoModelForAirOPE
from utils import AirOPEDataset, airope_data_collator, AirOPETrainer
from utils import AirOPETrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class MyTrainingArguments(TrainingArguments):
  save_embedding: bool = field(
      default=False, metadata={"help": "Whether to save feature embeddings."})
  sgd_momentum: Optional[float] = field(
      default=0.9, metadata={"help": "sgd momentum (default:0.9)"})
  lr_schedule: Optional[str] = field(
      default="linear",
      metadata={
          "help": "Learning rate schedule (default:linear)",
          "choices": ["linear", "constant", "invsqrt"]
      })
  optimizer: Optional[str] = field(
      default="adam",
      metadata={
          "help": "name of optimizer",
          "choices": ["adam", "sgd"]
      })


@dataclass
class ModelArguments:
  """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune from.
    """

  model_name_or_path: str = field(
      metadata={
          "help":
              "Path to pretrained model or model identifier from huggingface.co/models"
      })
  config_name: Optional[str] = field(
      default=None,
      metadata={
          "help": "Pretrained config name or path if not the same as model_name"
      })
  tokenizer_name: Optional[str] = field(
      default=None,
      metadata={
          "help":
              "Pretrained tokenizer name or path if not the same as model_name"
      })
  cache_dir: Optional[str] = field(
      default=None,
      metadata={
          "help":
              "Where do you want to store the pretrained models downloaded from s3"
      })

  # additional model arguments
  share_bert: bool = field(
      default=False, metadata={"help": "if cnet and qnet share bert"})
  fix_bert: bool = field(default=False, metadata={"help": "if finetune bert"})
  freeze_bert: bool = field(
      default=False,
      metadata={
          "help":
              "set bert in eval mode when training, i.e., turn off dropout etc."
      })

  # Dice Argument
  gamma: Optional[float] = field(
      default=1, metadata={"help": "discount factor (default:1)"})
  normalize_c: bool = field(
      default=True,
      metadata={
          "help":
              "Turn off/on normalizing the cnet output with addition penalty"
      })
  lambinit: Optional[float] = field(
      default=0, metadata={"help": "Initial value of lambda (default:0)"})
  finalact_q: Optional[str] = field(
      default="no",
      metadata={
          "help":
              "If or how to add positive constranit in qnet head (default: no). 'no' means do not add constraint",
          "choices": ["sigmoid", "no", "linear-no"]
      })
  finalact_aux: Optional[str] = field(
      default="sigmoid",
      metadata={
          "help":
              "If or how to add positive constranit in auxnet head (default: 'sigmoid'). 'no' means do not add constraint",
          "choices": ["sigmoid", "no", "linear-no"]
      })
  finalact_c: Optional[str] = field(
      default="square",
      metadata={
          "help":
              "If or how to add positive constranit in cnet head (default: square). 'no' means do not add constraint",
          "choices": [
              "square", "linear-square", "square_bias1", "no", "softplus"
          ]
      })
  alphaR: Optional[float] = field(
      default=1,
      metadata={
          "help": "alphaR rescales reward 0 or 1 (default:1)",
          "choices": [0, 1]
      })
  alphaQ: Optional[float] = field(
      default=0,
      metadata={
          "help":
              """alphaQ regularize q function output since q functions output is in (0,1),
                             which i think is not very necessary (default:0)"""
      })
  alphaC: Optional[float] = field(
      default=1, metadata={"help": "alphaC regularize c function (default:1)"})
  alphaL: Optional[float] = field(
      default=1, metadata={"help": "alphaL regularize lambda (default:1)"})
  alphaAux: Optional[float] = field(
      default=-1,
      metadata={
          "help":
              "alphaAux provides cotraining objective (default:-1 turned off)"
      })
  regfunQ: Optional[str] = field(
      default="square",
      metadata={"help": "regularization functions for Qnet (default: square )"})
  regfunC: Optional[str] = field(
      default="square",
      metadata={"help": "regularization functions for Cnet (default: square )"})
  regfunL: Optional[str] = field(
      default="square",
      metadata={
          "help": "regularization functions for lambda (default: square )"
      })
  minmax_training: Optional[str] = field(
      default="reverse_grad",
      metadata={
          "help":
              "How to implement minmax optimization (default: reverse_grad)",
          "choices": ["reverse_grad", "alter_update"]
      })
  max_turns: Optional[int] = field(
      default=15, metadata={"help": "max turns for paddings (default:15)"})
  normalize_obj_by_turns: bool = field(
      default=True,
      metadata={
          "help": "If normalize objective by # of turns, (default: true)"
      })

  # Learning rate scales
  lrscale_q: Optional[float] = field(
      default=1,
      metadata={"help": "learning rate scale for q values (default:1)"})
  lrscale_c: Optional[float] = field(
      default=1,
      metadata={"help": "learning rate scale for c values (default:1)"})
  lrscale_lamb: Optional[float] = field(
      default=10,
      metadata={"help": "learning rate scale for lambda (default:10)"})
  scale_lamb: Optional[float] = field(
      default=1,
      metadata={
          "help":
              "scale for lambda (default:1)"
              " since a too large lrscale_lamb will affect others dur to gradient norm"
      })
  lrscale_bert: Optional[float] = field(
      default=1,
      metadata={
          "help":
              "learning rate scale for BERT, the pre-trained feature extractor (default:1)"
      })


def main():
  # See all possible arguments in src/transformers/training_args.py
  # or by passing the --help flag to this script.
  # We now keep distinct sets of args, for a cleaner separation of concerns.

  parser = HfArgumentParser(
      (ModelArguments, DataTrainingArguments, MyTrainingArguments))

  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1]))
  else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  if (os.path.exists(training_args.output_dir) and
      os.listdir(training_args.output_dir) and training_args.do_train and
      not training_args.overwrite_output_dir):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

  # Setup logging
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO
      if training_args.local_rank in [-1, 0] else logging.WARN,
  )
  logger.warning(
      "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
      training_args.local_rank,
      training_args.device,
      training_args.n_gpu,
      bool(training_args.local_rank != -1),
      training_args.fp16,
  )
  logger.info("Training/evaluation parameters %s", training_args)
  logger.info("Model parameters %s", model_args)

  # Set seed
  set_seed(training_args.seed)

  # Load pretrained model and tokenizer
  #
  # Distributed training:
  # The .from_pretrained methods guarantee that only one local process can concurrently
  # download model & vocab.

  tokenizer = AutoTokenizer.from_pretrained(
      model_args.tokenizer_name
      if model_args.tokenizer_name else model_args.model_name_or_path,
      cache_dir=model_args.cache_dir,
  )
  model = AutoModelForAirOPE.from_pretrained(
      model_args.model_name_or_path,
      from_tf=bool(".ckpt" in model_args.model_name_or_path),
      cache_dir=model_args.cache_dir,
      args=model_args,
  )

  # Get datasets
  train_dataset = (AirOPEDataset(data_args, tokenizer=tokenizer))

  # Initialize our Trainer
  trainer = AirOPETrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      data_collator=airope_data_collator,
  )

  # Training
  if training_args.do_train:
    trainer.train(
        model_path=model_args.model_name_or_path if os.path
        .isdir(model_args.model_name_or_path) else None)
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_master():
      tokenizer.save_pretrained(training_args.output_dir)


def _mp_fn(index):
  # For xla_spawn (TPUs)
  main()


if __name__ == "__main__":
  main()
