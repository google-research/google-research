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

import time
from typing import Optional

import tiktoken
import vertexai
from openai import OpenAI
from transformers import AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel, HarmCategory, HarmBlockThreshold


PROJECT_ID = "YOUR_GCP_PROJECT_ID"
LOCATION = "YOUR_GCP_LOCATION"


class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.provider = self.get_provider(model_name) # 'openai' or 'google' or 'vertex' or 'palm' or 'vllm'.
        self.context_limit = self.get_context_limit(model_name)
        # OpenAI models
        if self.provider == 'openai':
            self.client = OpenAI()
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        # Gemini models
        elif self.provider == 'google':
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            self.client = GenerativeModel(model_name)
        # vLLM models
        elif self.provider == 'vllm':
            self.client = OpenAI(
                base_url=f"http://localhost:8000/v1",
                api_key="token-abc123",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def get_provider(self, model_name):
        if 'gpt' in model_name:
            return 'openai'
        elif 'gemini' in model_name:
            return 'google'
        else:
            return 'vllm'

    def get_context_limit(self, model_name):
        if model_name == 'gpt-4-0125-preview' or model_name == 'gpt-4-turbo-2024-04-09' or model_name == 'gpt-4o-mini-2024-07-18':
            return 128000
        elif model_name == 'gpt-3.5-turbo-0125':
            return 16385
        elif model_name == 'gemini-pro' or model_name == 'gemini-ultra':
            return 32000
        elif model_name == 'gemini-1.5-pro-preview-0409' or model_name == 'gemini-1.5-flash':
            # return 1000000
            return 128000
        elif 'Mistral-Nemo' in model_name:
            return 128000
        else:
            raise ValueError(f'Unsupported model: {model_name}')

    def query(self, prompt, **kwargs):
        if not prompt:
            return 'Contents must not be empty.'
        if self.provider == 'openai':
            return self.query_openai(prompt, **kwargs)
        elif self.provider == "google":
            return self.query_gemini(prompt, **kwargs)
        elif self.provider == "vllm":
            return self.query_openai(prompt, **kwargs)
        else:
            raise ValueError(f'Unsupported provider: {self.provider}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def query_gemini_with_retry(self, prompt, generation_config):
        safety_config = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        response = self.client.generate_content(prompt, generation_config=generation_config, safety_settings=safety_config)
        try:
            response_text = response.text
        except Exception as e:
            response_text = str(e)
        return response_text

    def query_gemini(self, prompt, rate_limit_per_minute = None, **kwargs):
        generation_config = GenerationConfig(
            stop_sequences=kwargs.get('stop', []),
            temperature=kwargs.get('temperature'),
            top_p=kwargs.get('top_p'),
        )
        if rate_limit_per_minute:
            time.sleep(60 / rate_limit_per_minute)
        return self.query_gemini_with_retry(prompt, generation_config=generation_config)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def query_openai_with_retry(self, messages, **kwargs):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

    def query_openai(self,
                     prompt,
                     system = None,
                     rate_limit_per_minute = None, **kwargs):
        # Set default system message
        if system is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        response = self.query_openai_with_retry(messages, **kwargs)
        # Sleep to avoid rate limit if rate limit is set
        if rate_limit_per_minute:
            time.sleep(60 / rate_limit_per_minute)
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def query_gemini_token_count(self, prompt):
        return self.client.count_tokens(prompt).total_tokens

    def get_token_count(self, prompt):
        if not prompt:
            return 0
        if self.provider == 'openai':
            return len(self.tokenizer.encode(prompt))
        elif self.provider == "google":
            return self.query_gemini_token_count(prompt)
        elif self.provider == 'vllm':
            return len(self.tokenizer.encode(prompt))
        else:
            raise ValueError(f'Unsupported provider: {self.provider}')


if __name__ == '__main__':
    def test_model(model_name, prompt):
        print(f'Testing model: {model_name}')
        model = Model(model_name)
        print(f'Prompt: {prompt}')
        response = model.query(prompt)
        print(f'Response: {response}')
        num_tokens = model.get_token_count(prompt)
        print(f'Number of tokens: {num_tokens}')
    prompt = 'Hello, how are you?'
    for model in ['gpt-4o-mini-2024-07-18', 'gemini-1.5-flash']:
    # for model in ['mistralai/Mistral-Nemo-Instruct-2407']:
        test_model(model, prompt)
