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

import os
import base64
import openai
import numpy as np
from PIL import Image
from typing import Union, Optional
from google import genai
from functools import partial
from google.genai.types import HttpOptions, GenerateContentConfig
import time
import openai
from openai import OpenAI, ChatCompletion
# openai.api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
from anthropic import AnthropicVertex

# client = genai.Client(http_options=HttpOptions(api_version="v1"))

class LM_Client:
    def __init__(self, model_name = "gpt-3.5-turbo"):
        self.model_name = model_name

    def chat(self, messages, json_mode = False):
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        self.client = OpenAI()
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
            temperature=0,
        )
        response = chat_completion.choices[0].message.content
        return response, chat_completion

    def one_step_chat(
        self, text, system_msg = None, json_mode=False
    ):
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode)


class CLAUDE_Client:
    def __init__(self, model_name = "claude-3-7-sonnet@20250219"):
        self.model_name = model_name

    def chat(self, messages, sys_msg):
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        self.client = AnthropicVertex(region=os.environ.get("GOOGLE_CLOUD_LOCATION"), project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"))

        message = self.client.messages.create(
            model=self.model_name,
            system=sys_msg,
            messages=messages,
            max_tokens=500,
            temperature=0,
        )
        response = message.content[0].text
        return response, message

    def one_step_chat(
        self, text, system_msg = None
    ):
        messages = []
        # if system_msg is not None:
        #     messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, system_msg)


class GPT4V_Client:
    def __init__(self, model_name = "gpt-4o", max_tokens = 512):
        self.model_name = model_name
        self.max_tokens = max_tokens

    def encode_image(self, path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def one_step_chat(
        self, text, image,
        system_msg = None,
    ):
        jpg_base64_str = self.encode_image(image)
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages += [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}"},},
                ],
        }]
        self.client = OpenAI()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content, response


class GEMINI_Client:
    def __init__(self, model_name = "gemini-2.5-flash-lite"):
        self.model_name = model_name

    def chat(self, messages, json_mode = False):
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
        chat_completion = self.client.models.generate_content(
            model=self.model_name,
            contents=messages[1]['content'],
            config=GenerateContentConfig(
                temperature=0,
                system_instruction=messages[0]['content'],
            )
        )
        response = chat_completion.text
        return response, chat_completion

    def one_step_chat(
        self, text, system_msg = None, json_mode=False
    ):
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode)


CLIENT_DICT = {
    "gpt-3.5-turbo": LM_Client,
    "gpt-4": LM_Client,
    "gpt-4o": GPT4V_Client,
    "gemini-2.5-flash": GEMINI_Client,
    "gemini-2.5-pro": GEMINI_Client,
    "claude-3-7-sonnet@20250219": CLAUDE_Client,
}