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

import argparse
import io
import json
import os
import time
from evaluation import evaluation
from google import genai
from PIL import Image
import soundfile as sf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def load_config(config_path):
  with open(config_path, 'r') as f:
    return json.load(f)


def main(config, sub_signal, api_key, prefix):
  client = genai.Client(api_key=api_key)

  with open(config['dataset'], 'r') as f:
    annotation = json.load(f)

  keys_list = list(annotation.keys())
  total_number = len(keys_list)
  print('Total number:', total_number)

  total_yes = 0
  total_no = 0
  correct_yes = 0
  correct_no = 0

  if os.path.exists(config['save_json_path']):
    with open(config['save_json_path'], 'r') as f:
      output = json.load(f)
  else:
    output = []

  print(
      config['save_json_path'],
      len(output),
      total_yes,
      total_no,
      correct_yes,
      correct_no,
  )

  for clip_id in tqdm(keys_list[len(output) :]):

    context = annotation[clip_id]['context']
    sequence = annotation[clip_id]['sequence']
    prompt = config['question']

    audio_path = os.path.join(config['audio_folder'], f'{clip_id}.wav')
    audio_file = client.files.upload(file=audio_path)

    # Image Processing
    images = []
    for i in range(config['frame_num']):
      image = Image.open(
          os.path.join(config['data_base_path'], 'frames', clip_id, f'{i}.jpg')
      )
      images.append(image)

    try:
      response = client.models.generate_content(
          model=config['model'], contents=[images, audio_file, prompt]
      )
    except:
      time.sleep(30)
      response = client.models.generate_content(
          model=config['model'], contents=[images, prompt]
      )

    response_text = response.text
    if response_text:
      response_text = response_text.lower()

    output.append({'key': clip_id, 'response': response_text})

    save_json_path = (
        config['save_json_path']
        + f'/{config['model']}_{sub_signal}_{config['frame_num']}f.json'
    )
    with open(save_json_path, 'w') as f:
      json.dump(output, f)

  res = evaluation(output, annotation)

  with open(
      f'../res/{config['model']}_{sub_signal}_{config['frame_num']}f.json', 'w'
  ) as f:
    json.dump(res, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Deepsocial model evaluation.')
  parser.add_argument(
      '--config',
      type=str,
      required=True,
      help='Path to the configuration file.',
  )
  parser.add_argument('--signal', type=str, required=True, help='s1-s8')
  parser.add_argument('--api', type=str, required=True, help='API key')
  parser.add_argument('--prefix', type=str, required=True, help='gemini1.5')
  args = parser.parse_args()

  # Load configuration from the provided config file path
  if os.path.exists(args.config):
    config = load_config(args.config)
  else:
    raise FileNotFoundError(f'Configuration file not found at {args.config}')

  main(config, args.signal, args.api, args.prefix)
