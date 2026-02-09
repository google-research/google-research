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

import argparse
import io
import json
import os
import time
from evaluation import evaluation
from google import genai
import numpy as np
from PIL import Image
import soundfile as sf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def load_config(config_path):
  with open(config_path, 'r') as f:
    return json.load(f)


def main(config, sub_signal, api_key, prefix):
  client = genai.Client(api_key=api_key)
  # keys = ['someone_talk', 'turn_talk', 'talk_to_me', 'i_talk', 'personal_space', 'look_at_me', 'i_look_at', 'i_focus', 'social_interaction']

  sub_signal_q_list = [
      '1. Is there anyone else talking?',
      (
          '2. Are there alternating speech turns? (or are there multiple people'
          ' talking?)'
      ),
      '3. Is there anybody talking to the wearer?',
      '4. Is the wearer talking?',
      (
          '5. Is there any person within person space to the wearer? (Person'
          ' space means within 1.2 meters.)'
      ),
      '6. Is there someone looking at the wearer?',
      '7. Is the wearer looking at someone?',
      '8. Is the wearer focusing?',
  ]

  if 'graph' in sub_signal:
    graph = {}
    subsignal_list = []
    file_list = [
        '_someone_else_talk.json',
        '_alternating_speech_turns.json',
        '_talk_to_me.json',
        '_I_am_talk.json',
        '_personal_space.json',
        '_looking_at_me.json',
        '_I_look_at_someone.json',
        '_I_focus.json',
    ]
    file_list = [prefix + file for file in file_list]
    for file in file_list:
      with open(
          os.path.join(config['data_base_path'], 'logs', file), 'r'
      ) as f:  # answer for each cue
        subsignal_list.append(json.load(f))
    for i, data in enumerate(subsignal_list[0]):
      key = data['key']
      graph[key] = []

      assert subsignal_list[0][i]['key'] == key
      assert subsignal_list[1][i]['key'] == key
      assert subsignal_list[2][i]['key'] == key
      assert subsignal_list[3][i]['key'] == key
      assert subsignal_list[4][i]['key'] == key
      assert subsignal_list[5][i]['key'] == key
      assert subsignal_list[6][i]['key'] == key
      assert subsignal_list[7][i]['key'] == key

      # for sub in subsignal_list:
      resp = subsignal_list[0][i]['response'].replace('interaction:', 'answer:')
      if 'answer: yes' in resp:
        graph[key].append('<someone, is, talking>')
      else:
        graph[key].append('<no one, is, talking>')

      resp = subsignal_list[1][i]['response'].replace('interaction:', 'answer:')
      if 'answer: yes' in resp:
        graph[key].append('<turn talk, is, happening>')
      else:
        graph[key].append('<turn talk, is not, happening>')

      resp = subsignal_list[2][i]['response'].replace('interaction:', 'answer:')
      if 'answer: yes' in resp:
        graph[key].append('<someone, talk to, the wearer>')
      else:
        graph[key].append('<no one, talk to, the wearer>')

      resp = subsignal_list[3][i]['response'].replace('interaction:', 'answer:')
      if 'answer: yes' in resp:
        graph[key].append('<the wearer, is, talking>')
      else:
        graph[key].append('<the wearer, is not, talking>')

      resp = subsignal_list[4][i]['response'].replace('interaction:', 'answer:')
      if 'answer: yes' in resp:
        graph[key].append('<someone, within, personal space>')
      else:
        graph[key].append('<no one, within, personal space>')

      resp = subsignal_list[5][i]['response'].replace('interaction:', 'answer:')
      if 'answer: yes' in resp:
        graph[key].append('<someone, look at, the wearer>')
      else:
        graph[key].append('<no one, look at, the wearer>')

      resp = subsignal_list[6][i]['response'].replace('interaction:', 'answer:')
      if 'answer: yes' in resp:
        graph[key].append('<the wearer, look at, someone>')
      else:
        graph[key].append('<the wearer, look at, no one>')

      resp = subsignal_list[7][i]['response'].replace('interaction:', 'answer:')
      if 'answer: yes' in resp:
        graph[key].append('<the wearer, is, focusing>')
      else:
        graph[key].append('<the wearer, is not, focusing>')

  sub_signal_q = ''
  if sub_signal == 's1':
    sub_signal_q = 'Is there anyone else talking?'
  elif sub_signal == 's2':
    sub_signal_q = (
        'Are there alternating speech turns? (or are there multiple people'
        ' talking?)'
    )
  elif sub_signal == 's3':
    sub_signal_q = 'Is there anybody talking to the wearer?'
  elif sub_signal == 's4':
    sub_signal_q = 'Is the wearer talking?'
  elif sub_signal == 's5':
    sub_signal_q = (
        'Is there any person within person space to the wearer? (Person space'
        ' means within 1.2 meters.)'
    )
  elif sub_signal == 's6':
    sub_signal_q = 'Is there someone looking at the wearer?'
  elif sub_signal == 's7':
    sub_signal_q = 'Is the wearer looking at someone?'
  elif sub_signal == 's8':
    sub_signal_q = 'Is the wearer focusing?'
  elif sub_signal == 'all_H':
    sub_signal_q = ' '.join(sub_signal_q_list)
  elif sub_signal == 'all_rea_H':
    sub_signal_q = ' '.join(sub_signal_q_list)

  with open(config['dataset'], 'r') as f:
    annotation = json.load(f)

  keys_list = list(annotation.keys())
  keys_list.sort()
  total_number = len(keys_list)
  print('Total number:', total_number)

  audio_text_path = config['audio_text_path']
  with open(audio_text_path, 'r') as f:
    audio_text_all = json.load(f)

  save_json_path = (
      config['save_json_path']
      + f'/{config['model']}_{sub_signal}_{config['frame_num']}f.json'
  )
  if os.path.exists(save_json_path):
    with open(save_json_path, 'r') as f:
      output = json.load(f)
  else:
    output = []

  print(save_json_path, len(output))

  for clip_id in tqdm(keys_list[len(output) :]):
    # print(clip_id)

    if sub_signal == 'graph100_audio':
      sub_signal_q = ' '.join(graph[clip_id][:4])
    elif sub_signal == 'graph100_video':
      sub_signal_q = ' '.join(graph[clip_id][4:])
    elif (
        sub_signal == 'graph100_H'
        or sub_signal == 'graph_H'
        or sub_signal == 'graph_rea_H'
    ):
      sub_signal_q = ' '.join(graph[clip_id])

    audio_text = audio_text_all[clip_id]
    context = annotation[clip_id]['context']
    sequence = annotation[clip_id]['sequence']
    assert '<audio_text>' in config['question']
    prompt = config['question'].replace('<audio_text>', audio_text)
    prompt = prompt.replace('<sub_signal>', sub_signal_q)

    audio_path = os.path.join(config['audio_folder'], f'{clip_id}.wav')
    audio_file = client.files.upload(file=audio_path)

    # Image Processing
    images = []
    for i in range(10):
      image = Image.open(
          os.path.join(config['data_base_path'], 'frames', clip_id, f'{i}.jpg')
      )
      images.append(image)

    if config['frame_num'] < 10:
      indices = np.linspace(0, len(images) - 1, config['frame_num'], dtype=int)
      images = [images[i] for i in indices]

    try:
      response = client.models.generate_content(
          model=config['model'], contents=[images, audio_file, prompt]
      )
    except:
      time.sleep(120)
      client = genai.Client(api_key=api_key)
      response = client.models.generate_content(
          model=config['model'], contents=[images, audio_file, prompt]
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

  if 'data_base_path' not in config:
    raise ValueError('The key "data_base_path" is missing in the config file.')

  main(config, args.signal, args.api, args.prefix)
