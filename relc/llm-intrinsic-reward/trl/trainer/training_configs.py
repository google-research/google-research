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

# coding=utf-8
# coding=utf-8

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class RewardConfig(TrainingArguments):
    """
    RewardConfig collects all training arguments related to the [`RewardTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, *optional*, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        gradient_checkpointing (`bool`, *optional*, defaults to `True`):
                If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    """

    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator."
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
