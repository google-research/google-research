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

"""Utils for RevThink."""

import re

import backoff
import google.api_core.exceptions as google_exceptions
from math_utils import is_math_correct
import ratelimit
import torch
import vertexai
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel

# setup vertexai using your own credentials
project_id = ""  # fill here
location = ""   # fill here
vertexai.init(project=project_id, location=location)
pro_model = GenerativeModel(model_name="gemini-1.5-pro-001")
flash_model = GenerativeModel(model_name="gemini-1.5-flash-001")


@ratelimit.sleep_and_retry
@backoff.on_exception(backoff.expo,
                      google_exceptions.ResourceExhausted, max_tries=10)
@ratelimit.limits(calls=60, period=60)
def get_gemini_output(prompt, model="flash"):
  """Get Gemini output."""
  if model == "flash":
    gemini = flash_model
  elif model == "pro":
    gemini = pro_model
  else:
    raise ValueError(f"Unsupported model: {model}")
  output = gemini.generate_content(prompt,
                                   generation_config=GenerationConfig(
                                       temperature=0.8,
                                       max_output_tokens=1000,
                                   )).text
  return output


def get_token_counts(prompt):
  return flash_model.count_tokens(prompt).total_tokens


class CastOutputToFloat(torch.nn.Sequential):

  def forward(self, x):
    return super().forward(x).to(torch.float32)


def get_number_choice(text):
  if not text:
    return "N/A"
  match = re.findall(r"answer is \((\d)\)", text)
  if match:
    return match[-1]
  else:
    match = re.findall(r"\((\d)\)", text)
    return match[-1] if match else "N/A"


def get_alphabet_choice(text):
  if not text:
    return "N/A"
  match = re.findall(r"answer is \((A|B|C|D|E|F)\)", text)
  if match:
    return match[-1]
  else:
    match = re.findall(r"\((A|B|C|D|E|F)\)", text)
    return match[-1] if match else "N/A"


def get_true_false(text):
  if not text:
    return "N/A"
  match = re.findall(r"(true|false)", text, re.IGNORECASE)
  return match[-1].lower() if match else "N/A"


def get_yes_no(text):
  if not text:
    return "N/A"
  match = re.findall(r"(yes|no)", text, re.IGNORECASE)
  return match[-1].lower() if match else "N/A"


def remove_backward_answer(text):
  pattern = r"The correct answer is \([A-Z]\)\."
  modified = re.sub(pattern, "", text).strip()
  return modified


def last_boxed_only_string(string):
  """Returns the last boxed only string."""
  idx = string.rfind("\\boxed")
  if idx < 0:
    idx = string.rfind("\\fbox")
    if idx < 0:
      return string

  i = idx
  right_brace_idx = None
  num_left_braces_open = 0
  while i < len(string):
    if string[i] == "{":
      num_left_braces_open += 1
    if string[i] == "}":
      num_left_braces_open -= 1
      if num_left_braces_open == 0:
        right_brace_idx = i
        break
    i += 1

  if not right_brace_idx:
    retval = string
  else:
    retval = string[idx:right_brace_idx + 1]

  return retval


def floatify(num):
  try:
    num = float(num)
    return num
  except ValueError:
    return "N/A"


def evaluate(test_samples, pred_key="pred", is_math=False):
  num_correct = 0
  for i in test_samples:
    if is_math:
      if is_math_correct(i[pred_key], i["gold_answer"]):
        num_correct += 1
    else:
      if i[pred_key] == i["gold_answer"]:
        num_correct += 1
  acc = round(num_correct / len(test_samples), 4)
  return acc
