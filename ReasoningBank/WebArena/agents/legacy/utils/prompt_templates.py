# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

from typing import List

from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from dataclasses import dataclass

"""
To use this class, you should have the ``openai`` python package installed, and the
environment variable ``OPENAI_API_KEY`` set with your API key.
"""


@dataclass
class PromptTemplate:
    """
    Base class for prompt templates.

    Defines a standard interface for prompt templates, ensuring that they contain
    the required fields for the CustomLLMChatbot.
    """

    system: str
    human: str
    ai: str
    prompt_end: str = ""

    def format_message(self, message):
        """
        Formats a given message based on its type.

        Args:
            message (BaseMessage): The message to be formatted.

        Returns:
            str: The formatted message.

        Raises:
            ValueError: If the message type is not supported.
        """
        if isinstance(message, SystemMessage):
            return self.system.format(input=message.content)
        elif isinstance(message, HumanMessage):
            return self.human.format(input=message.content)
        elif isinstance(message, AIMessage):
            return self.ai.format(input=message.content)
        else:
            raise ValueError(f"Message type {type(message)} not supported")

    def construct_prompt(self, messages):
        """
        Constructs a prompt from a list of messages.

        Args:
            messages (List[BaseMessage]): The list of messages to be formatted.

        Returns:
            str: The constructed prompt.
        """
        if not all(isinstance(m, BaseMessage) for m in messages):
            raise ValueError("All elements in the list must be of type BaseMessage")

        prompt = "".join([self.format_message(m) for m in messages])
        prompt += self.prompt_end
        return prompt


def get_prompt_template(model_name):
    for key, value in MODEL_PREFIX_TO_PROMPT_TEMPLATES.items():
        if key in model_name:
            return value
    raise NotImplementedError(f"Model {model_name} has no supported chat template")


## Prompt templates

STARCHAT_PROMPT_TEMPLATE = PromptTemplate(
    system="<|system|>\n{input}<|end|>\n",
    human="<|user|>\n{input}<|end|>\n",
    ai="<|assistant|>\n{input}<|end|>\n",
    prompt_end="<|assistant|>",
)


## Model prefix to prompt template mapping

MODEL_PREFIX_TO_PROMPT_TEMPLATES = {
    "starcoder": STARCHAT_PROMPT_TEMPLATE,
    "starchat": STARCHAT_PROMPT_TEMPLATE,
}
