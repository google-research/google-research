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

r"""Call LLLM for intrinsic rewards generation."""
from abc import ABC, abstractmethod
import multiprocessing
import time
from typing import Any, Dict, List, Tuple

import openai


class Generator(ABC):
    """Base class for all generators."""

    def __init__(
        self,
        name,
        model_name_or_path,
        top_p = 0.95,
        temperature = 0.9,
        num_return_sequences = 1,
    ):
        """
        Args:
            name: The name of the generator.
            model_name_or_path: The name or path of the model to use.
            top_p: Parameter for Nucleus Sampling.
            temperature: Temperature to use for sampling responses.
            num_return_sequences: The number of responses to generate for each prompt.
        """
        self.name = name
        self.model_name_or_path = model_name_or_path
        self.top_p = top_p
        self.temperature = temperature
        self.num_return_sequences = num_return_sequences

    @abstractmethod
    def __call__(self, prompts):
        """
        Args:
            prompts: A list of prompts to generate responses for.

        Returns:
            A list of lists of responses for each prompt.
        """
        pass


class OpenAIGenerator(Generator):
    """Generates responses from the OpenAI API.

    We currently only support the chat endpoint (e.g., GPT-3.5-Turbo and GPT-4).

    For more information on the API, see: https://platform.openai.com
    """

    def __init__(
        self,
        model_name_or_path,
        api_key,
        max_tokens = 2048,
        top_p = 0.95,
        temperature = 0.9,
        num_return_sequences = 1,
        wait_time = 10.0,
        num_workers = 1,
    ):
        """
        Args:
            model_name_or_path: The name or path of the model to use.
            api_key: The API key to use.
            max_tokens: The maximum number of tokens to generate.
            top_p: Parameter for Nucleus Sampling.
            temperature: Temperature to use for sampling responses.
            num_return_sequences: The number of responses to generate for each prompt.
            wait_time: The time to wait between failed requests to the API.
            num_workers: The number of workers to use for multiprocessing.
        """
        super().__init__(
            name="openai",
            model_name_or_path=model_name_or_path,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.wait_time = wait_time
        openai.api_key = api_key
        self.num_workers = num_workers

        if self.model_name_or_path not in ["gpt-3.5-turbo", "gpt-4"]:
            raise ValueError(f"{self.model_name_or_path} is not supported.")

    def __call__(self, prompts):
        # Make static inputs with class data for multiprocessing.
        # Functions submitted to multiprocessing.Pool.map() must be picklable.
        inputs = [
            (
                prompt,
                {
                    "api_key": self.api_key,
                    "wait_time": self.wait_time,
                    "model": self.model_name_or_path,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "num_return_sequences": self.num_return_sequences,
                },
            )
            for prompt in prompts
        ]

        pool = multiprocessing.Pool(processes=self.num_workers)
        responses = pool.map(make_request, inputs)

        return responses


def make_request(inputs):
    """
    Makes a request to the OpenAI API.

    Args:
        inputs: A tuple containing the prompt and metadata.

    Returns:
        List of responses for the prompt from the API.
    """
    prompt, metadata = inputs

    # Set the API key for the request.
    openai.api_key = metadata["api_key"]

    request = None
    retry_time = 0
    while request is None:
        try:
            retry_time += 1
            request = openai.ChatCompletion.create(
                model=metadata["model"],
                messages=[{"role": "user", "content": prompt}],
                top_p=metadata["top_p"],
                max_tokens=metadata["max_tokens"],
                temperature=metadata["temperature"],
                n=metadata["num_return_sequences"],
                request_timeout=10,
            )
        except Exception as e:
            print("OpenAI Error Message:", e)
            # Wait for a bit before trying again.
            time.sleep(metadata["wait_time"])

        if retry_time >= 3:
            break

    if request is None:
        return None
    responses = [choice["message"]["content"] for choice in request.choices]

    return responses