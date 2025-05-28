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

"""DePlot Prompts."""

import collections
from collections.abc import Callable, Iterable, Mapping, Sequence
import enum
import json
import random
import time
from typing import TypeVar

from absl import flags
import openai
from pix2struct import metrics as pix2struct_metrics
from t5.evaluation import metrics as t5_metrics
import tensorflow as tf



_OPENAI_CREDENTIALS = flags.DEFINE_list(
    'openai_credentials', None, 'Credentials to call OpenAI.', required=True)

T = TypeVar('T')
TFn = Callable[Ellipsis, T]


class Model(enum.Enum):
  GPT3 = 'gpt3'


def retry(
    try_count = 3,
    sleep_seconds = 2,  # pylint: disable=unused-argument
):
  """Retry decorator."""

  def decorator(fn):

    def newfn(*args, **kwargs):
      for idx in range(try_count):
        try:
          return fn(*args, **kwargs)
        except ValueError as e:
          time.sleep(sleep_seconds * (2**idx))
          if idx == try_count - 1:
            raise ValueError('No more retries') from e

    return newfn

  return decorator




@retry(try_count=3, sleep_seconds=1)
def _call_openai(
    prompt,
    engine,
    max_decode_steps,
    temperature,
    top_p = 1,
    frequency_penalty = 0,
    presence_penalty = 0,
    samples = 1,
    stop = ('Q:', 'A:', 'Summary:', '\n\n')):
  """Issues a completion request to the engine, while retrying on failure.

  Args:
    prompt: The prompt to send.
    engine: Model engine to use.
    max_decode_steps: The max_tokens parameter to send to the engine.
    temperature: Sampling temperature.
    top_p: Ratio of likelihood weighted token options to allow while sampling.
    frequency_penalty: Pentalty for the frequency of repeated tokens.
    presence_penalty: Penalty for the existence repeated tokens.
    samples: Number of outputs to generate.
    stop: Sequence of strings that elicit an end to decoding

  Returns:
    Text completion
  """
  openai.api_key = random.choice(_OPENAI_CREDENTIALS.value)

  try:
    reply = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_decode_steps,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=samples,
        stop=stop)
    return [choice['text'] for choice in reply['choices']] if reply else []

  except openai.error.RateLimitError as e:
    print('Sleeping 60 secs.')
    time.sleep(60)
    raise ValueError('RateLimitError') from e


def call_model(
    model,
    prompt,
    use_code,
    temperature,
    max_decode_steps,
    samples,
):
  """Calls model given a prompt."""
  results = []
  while len(results) < samples:
    if model == Model.GPT3:
      results.extend(
          _call_openai(
              prompt,
              engine='code-davinci-002' if use_code else 'text-davinci-003',
              temperature=temperature,
              max_decode_steps=max_decode_steps,
              samples=samples))
    else:
      raise ValueError(f'Unknown model_type={model}')
  return results[:samples]


def chunks(
    generator,
    chunk_size,
    filter_fn):
  """Splits generator into chunks."""
  chunk = []
  idx = 0
  skipped = 0

  for item in generator:
    if not filter_fn(item):
      skipped += 1
      continue
    if len(chunk) >= chunk_size:
      yield idx, chunk
      idx += 1
      chunk = [item]
    else:
      chunk.append(item)

  if chunk:
    yield idx, chunk
  print('Total skipped', skipped)


def _majority(predictions):
  """Finds most frequent result among the first N predictions for each N."""
  result = []
  counter = collections.Counter()
  for prediction in predictions:
    if prediction:
      counter[prediction] += 1
    if counter:
      result.append(counter.most_common(1)[0][0])
    else:
      result.append('')
  return result


def _exec(code):
  """Executed model output and returns the `ans` variable."""

  def execute(x):
    try:
      exec(x)  # pylint: disable=exec-used
      answer = locals().get('ans', '')
      if isinstance(answer, str):
        return answer
      elif isinstance(answer, bool):
        return 'Yes' if answer else 'No'
      elif isinstance(answer, collections.abc.Sequence):
        return ', '.join(str(a) for a in answer)
      return str(answer)
    except Exception:  # pylint: disable=broad-except
      return ''

  return execute(code)


def _extract_answer(prediction):
  output = prediction.split('\n\n')[0]
  if output.lower().startswith('#python'):
    return _exec(output)
  return output.split('answer is')[-1].strip().rstrip('.').strip()


def _extract_answers(predictions):
  return [_extract_answer(output) for output in predictions]


def compute_metrics(files, is_qa):
  """Computes the metrics given the list of prediction files."""
  targets = []
  predictions = []
  if is_qa:
    def metric_fn(targets, predictions):
      return dict(
          relaxed_accuracy=pix2struct_metrics.aggregate_metrics(
              targets=targets,
              predictions=predictions,
              metric_fn=pix2struct_metrics.relaxed_correctness,
          )
      )

  else:
    metric_fn = t5_metrics.bleu
  for predictions_file in files:
    with tf.io.gfile.GFile(predictions_file) as f:
      for line in f:
        prediction = json.loads(line)
        # Each prediction line contains a list of targets but only one is used.
        targets.append(prediction['target'][0])
        if is_qa:
          predictions.append(
              _majority(_extract_answers(prediction['predictions']))
          )
        else:
          predictions.append(
              [p.split('\n')[0] for p in prediction['predictions']]
          )
  metrics = {}
  for idx, sampled_predictions in enumerate(zip(*predictions)):
    metric = metric_fn(targets, list(sampled_predictions))
    for key, value in metric.items():
      metrics[f'{key}_maj{idx + 1}'] = value
  return metrics
