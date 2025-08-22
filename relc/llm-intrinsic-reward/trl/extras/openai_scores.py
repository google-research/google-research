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

r"""Create prompt & call OpenAI API"""
import re
import os
import ast
import torch

from trl.extras.score_generator import OpenAIGenerator


def tokenize_sent(sentence, tokenizer):
    ids = tokenizer.encode(sentence)
    return [tokenizer.decode(torch.tensor([i])) for i in ids]


def create_question(tokens):
    sentence = "".join(tokens)
    question = f"=== Task ===\nSentence: {sentence}"
    question += f"\n\nWords:\n"
    for i, w in enumerate(tokens):
        question += f"{i+1}. {w}\n"
    question += "\nRewards assigned:"
    return question


def extract_scores(input_string):
    if "the rewards assigned at each word" in input_string:
        match = re.search(r'\[.*?\]', input_string)
        if match:
            extracted_list_str = match.group(0)
            # Convert the string representation of list to actual list
            extracted_list = ast.literal_eval(extracted_list_str)
            return extracted_list
        else:
            raise ValueError(f"Unable to extract scores from: {input_string}")
    else:
        matches = re.findall(r': ([\-\d\.]+)', input_string)
        return [float(match) for match in matches]


# def query_one(tokens, model="gpt-3.5-turbo", max_tokens=600):
#     llm_query = prompt + "\n\n" + create_question(tokens)
#     try:
#         response = openai.ChatCompletion.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": llm_query},
#             ],
#             temperature=0.0,
#             max_tokens=max_tokens,
#             timeout=10,
#         )
#         scores = extract_scores(response['choices'][0]['message']['content'])
#     except (
#         openai.error.OpenAIError,
#         openai.error.RateLimitError,
#         openai.error.ServiceUnavailableError,
#         TimeoutError,
#         ValueError,
#     ) as error:
#         scores = None
#         print(type(error).__name__)

#     return scores


# def query_batch_multi_thread(batch_tokens, max_workers=2):
#     batch_llm_scores = []
#     with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         shared_kwargs = dict(
#             model="gpt-3.5-turbo",
#             max_tokens=600,
#         )
#         for score in executor.map(
#             lambda sample: query_one(sample, **shared_kwargs), batch_tokens
#         ):
#             batch_llm_scores.append(score)
#     return batch_llm_scores


def query_batch_new(prompt, batch_tokens, max_workers=2):
    llm_queries = [prompt + "\n\n" + create_question(ts) for ts in batch_tokens]
    generator = OpenAIGenerator(
        model_name_or_path="gpt-3.5-turbo",
        api_key=os.environ["OPENAI_API_KEY"],
        num_workers=max_workers,
        max_tokens=600,
        temperature=0.0,
    )
    responses = generator(llm_queries)

    batch_llm_scores = []
    for res in responses:
        try:
            batch_llm_scores.append(extract_scores(res[0]))
        except Exception:
            print("Format Error")
            batch_llm_scores.append(None)

    return batch_llm_scores


def query_batch_span(queries, max_workers=2, max_tokens=600, api_key=None):
    openai_api_key = os.environ["OPENAI_API_KEY"] if api_key is None else api_key
    generator = OpenAIGenerator(
        model_name_or_path="gpt-3.5-turbo",
        api_key=openai_api_key,
        num_workers=max_workers,
        max_tokens=max_tokens,
        temperature=0.0,
        wait_time=1.0,
    )
    responses = generator(queries)

    batch_llm_outputs = []
    for res in responses:
        try:
            batch_llm_outputs.append(res[0])
        except Exception:
            print(">>>>>>>>>>>>>>>>>>>>> TIMEOUT <<<<<<<<<<<<<<<<<<<<<")
            batch_llm_outputs.append(None)

    return batch_llm_outputs