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

import collections
import json
from pathlib import Path
import re
import time
from warnings import warn
import logging

from functools import cache
import numpy as np
import tiktoken
import yaml
# from langchain_openai import ChatOpenAI

from langchain.schema import SystemMessage, HumanMessage
from openai import BadRequestError
from joblib import Memory
from transformers import AutoModel
from transformers import AutoTokenizer
import io
import base64
from PIL import Image
from openai import RateLimitError


def _extract_wait_time(error_message, min_retry_wait_time=60):
    """Extract the wait time from an OpenAI RateLimitError message."""
    match = re.search(r"try again in (\d+(\.\d+)?)s", error_message)
    if match:
        return max(min_retry_wait_time, float(match.group(1)))
    return min_retry_wait_time


def retry(
    chat,
    messages,
    n_retry,
    parser,
    log=True,
    min_retry_wait_time=60,
    rate_limit_max_wait_time=60 * 30,
):
    """Retry querying the chat models with the response from the parser until it
    returns a valid value.

    If the answer is not valid, it will retry and append to the chat the  retry
    message.  It will stop after `n_retry`.

    Note, each retry has to resend the whole prompt to the API. This can be slow
    and expensive.

    Parameters:
    -----------
        chat (function) : a langchain ChatOpenAI taking a list of messages and
            returning a list of answers.
        messages (list) : the list of messages so far.
        n_retry (int) : the maximum number of sequential retries.
        parser (function): a function taking a message and returning a tuple
        with the following fields:
            value : the parsed value,
            valid : a boolean indicating if the value is valid,
            retry_message : a message to send to the chat if the value is not valid
        log (bool): whether to log the retry messages.
        min_retry_wait_time (float): the minimum wait time in seconds
            after RateLimtError. will try to parse the wait time from the error
            message.

    Returns:
    --------
        value: the parsed value
    """
    tries = 0
    rate_limit_total_delay = 0
    while tries < n_retry and rate_limit_total_delay < rate_limit_max_wait_time:
        try:
            answer = chat.invoke(messages)
        except RateLimitError as e:
            wait_time = _extract_wait_time(e.args[0], min_retry_wait_time)
            logging.warning(f"RateLimitError, waiting {wait_time}s before retrying.")
            time.sleep(wait_time)
            rate_limit_total_delay += wait_time
            if rate_limit_total_delay >= rate_limit_max_wait_time:
                logging.warning(
                    f"Total wait time for rate limit exceeded. Waited {rate_limit_total_delay}s > {rate_limit_max_wait_time}s."
                )
                raise
            continue

        messages.append(answer)

        value, valid, retry_message = parser(answer.content)
        if valid:
            return value

        tries += 1
        if log:
            msg = f"Query failed. Retrying {tries}/{n_retry}.\n[LLM]:\n{answer.content}\n[User]:\n{retry_message}"
            logging.info(msg)
        messages.append(HumanMessage(content=retry_message))

    raise ValueError(f"Could not parse a valid value after {n_retry} retries.")


def retry_parallel(chat, messages, n_retry, parser):
    """Retry querying the chat models with the response from the parser until it returns a valid value.

    It will stop after `n_retry`. It assuemes that chat will generate n_parallel answers for each message.
    The best answer is selected according to the score returned by the parser. If no answer is valid, the
    it will retry with the best answer so far and append to the chat the retry message. If there is a
    single parallel generation, it behaves like retry.

    This function is, in principle, more robust than retry. The speed and cost overhead is minimal with
    the prompt is large and the length of the generated message is small.

    Parameters:
    -----------
        chat (function) : a langchain ChatOpenAI taking a list of messages and returning a list of answers.
            The number of parallel generations is specified at the creation of the chat object.
        messages (list) : the list of messages so far.
        n_retry (int) : the maximum number of sequential retries.
        parser (function): a function taking a message and returning a tuple with the following fields:
            value : the parsed value,
            valid : a boolean indicating if the value is valid,
            retry_message : a message to send to the chat if the value is not valid,
            score : a score to select the best answer from the parallel generations

    Returns:
    --------
        value: the parsed value
    """

    for i in range(n_retry):
        try:
            answers = chat.generate([messages]).generations[0]  # chat.n parallel completions
        except BadRequestError as e:
            # most likely, the added messages triggered a message too long error
            # we thus retry without the last two messages
            if i == 0:
                raise e
            msg = f"BadRequestError, most likely the message is too long retrying with previous query."
            warn(msg)
            messages = messages[:-2]
            answers = chat.generate([messages]).generations[0]

        values, valids, retry_messages, scores = zip(
            *[parser(answer.message.content) for answer in answers]
        )
        idx = np.argmax(scores)
        value = values[idx]
        valid = valids[idx]
        retry_message = retry_messages[idx]
        answer = answers[idx].message

        if valid:
            return value

        msg = f"Query failed. Retrying {i+1}/{n_retry}.\n[LLM]:\n{answer.content}\n[User]:\n{retry_message}"
        warn(msg)
        messages.append(answer)  # already of type AIMessage
        messages.append(SystemMessage(content=retry_message))

    raise ValueError(f"Could not parse a valid value after {n_retry} retries.")


def truncate_tokens(text, max_tokens=8000, start=0, model_name="gpt-4"):
    """Use tiktoken to truncate a text to a maximum number of tokens."""
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    if len(tokens) - start > max_tokens:
        return enc.decode(tokens[start : (start + max_tokens)])
    else:
        return text


@cache
def get_tokenizer(model_name="openai/gpt-4"):
    if model_name.startswith("openai"):
        return tiktoken.encoding_for_model(model_name.split("/")[-1])
    else:
        return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text, model="openai/gpt-4"):
    if model.startswith('gemini'):
        from google import genai
        from google.genai.types import HttpOptions, GenerateContentConfig
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        return client.models.count_tokens(model=model, contents=[text]).total_tokens
    elif model.startswith("claude"):
        import requests
        import subprocess
        import os
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{os.environ.get('GOOGLE_CLOUD_PROJECT')}/locations/{os.environ.get('GOOGLE_CLOUD_LOCATION')}/publishers/anthropic/models/{model}:rawPredict"
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        payload = {
            "model": model,
            "messages": {
                "content": text,
                "role": "user",
            }
        }
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["input_tokens"]
    else:
        enc = get_tokenizer(model)
        return len(enc.encode(text))


def count_messages_token(messages, model="openai/gpt-4"):
    """Count the number of tokens in a list of messages.

    Args:
        messages (list): a list of messages, each message can be a string or a
            list of dicts or an object with a content attribute.
        model (str): the model to use for tokenization.

    Returns:
        int: the number of tokens.
    """
    token_count = 0
    for message in messages:
        if hasattr(message, "content"):
            message = message.content

        if isinstance(message, str):
            token_count += count_tokens(message, model)
        # handles messages with image content
        elif isinstance(message, (list, tuple)):
            for part in message:
                if not isinstance(part, dict):
                    raise ValueError(
                        f"The message is expected to be a list of dicts, but got list of {type(message)}"
                    )
                if part["type"] == "text":
                    token_count += count_tokens(part["text"], model)
        else:
            raise ValueError(
                f"The message is expected to be a string or a list of dicts, but got {type(message)}"
            )
    return token_count


def json_parser(message):
    """Parse a json message for the retry function."""

    try:
        value = json.loads(message)
        valid = True
        retry_message = ""
    except json.JSONDecodeError as e:
        warn(e)
        value = {}
        valid = False
        retry_message = "Your response is not a valid json. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def yaml_parser(message):
    """Parse a yaml message for the retry function."""

    # saves gpt-3.5 from some yaml parsing errors
    message = re.sub(r":\s*\n(?=\S|\n)", ": ", message)

    try:
        value = yaml.safe_load(message)
        valid = True
        retry_message = ""
    except yaml.YAMLError as e:
        warn(str(e))
        value = {}
        valid = False
        retry_message = "Your response is not a valid yaml. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def _compress_chunks(text, identifier, skip_list, split_regex="\n\n+"):
    """Compress a string by replacing redundant chunks by identifiers. Chunks are defined by the split_regex."""
    text_list = re.split(split_regex, text)
    text_list = [chunk.strip() for chunk in text_list]
    counter = collections.Counter(text_list)
    def_dict = {}
    id = 0

    # Store items that occur more than once in a dictionary
    for item, count in counter.items():
        if count > 1 and item not in skip_list and len(item) > 10:
            def_dict[f"{identifier}-{id}"] = item
            id += 1

    # Replace redundant items with their identifiers in the text
    compressed_text = "\n".join(text_list)
    for key, value in def_dict.items():
        compressed_text = compressed_text.replace(value, key)

    return def_dict, compressed_text


def compress_string(text):
    """Compress a string by replacing redundant paragraphs and lines with identifiers."""

    # Perform paragraph-level compression
    def_dict, compressed_text = _compress_chunks(
        text, identifier="§", skip_list=[], split_regex="\n\n+"
    )

    # Perform line-level compression, skipping any paragraph identifiers
    line_dict, compressed_text = _compress_chunks(
        compressed_text, "¶", list(def_dict.keys()), split_regex="\n+"
    )
    def_dict.update(line_dict)

    # Create a definitions section
    def_lines = ["<definitions>"]
    for key, value in def_dict.items():
        def_lines.append(f"{key}:\n{value}")
    def_lines.append("</definitions>")
    definitions = "\n".join(def_lines)

    return definitions + "\n" + compressed_text


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text` that match the key.

    Notes
    -----
    All text and keys will be converted to lowercase before matching.

    """
    content_dict = {}
    # text = text.lower()
    # keys = set([k.lower() for k in keys])
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


class ParseError(Exception):
    pass


def parse_html_tags_raise(text, keys=(), optional_keys=(), merge_multiple=False):
    """A version of parse_html_tags that raises an exception if the parsing is not successful."""
    content_dict, valid, retry_message = parse_html_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict


def parse_html_tags(text, keys=(), optional_keys=(), merge_multiple=False):
    """Satisfy the parse api, extracts 1 match per key and validates that all keys are present

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.
    optional_keys : list of str
        The HTML tags to extract the content from, but are optional.

    Returns
    -------
    dict
        A dictionary mapping each key to subset of `text` that match the key.
    bool
        Whether the parsing was successful.
    str
        A message to be displayed to the agent if the parsing was not successful.
    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_html_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if not key in content_dict:
            if not key in optional_keys:
                retry_messages.append(f"Missing the key <{key}> in the answer.")
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of the key {key}. You should have only one of them."
                    )
                else:
                    # merge the multiple instances
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message


class ChatCached:
    # I wish I could extend ChatOpenAI, but it is somehow locked, I don't know if it's pydantic soercey.

    def __init__(self, chat, memory=None):
        self.chat = chat
        self.memory = memory if memory else Memory(location=Path.home() / "llm-cache", verbose=10)
        self._call = self.memory.cache(self.chat.__call__, ignore=["self"])
        self._generate = self.memory.cache(self.chat.generate, ignore=["self"])

    def __call__(self, messages):
        return self._call(messages)

    def generate(self, messages):
        return self._generate(messages)


def download_and_save_model(model_name, save_dir = "."):
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print(f"Model downloaded and saved to {save_dir}")


def image_to_jpg_base64_url(image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"
