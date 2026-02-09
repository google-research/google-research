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

r"""Evaluation script for Phi-4 model with audio-to-text and graph reasoning.

This script extends the audio-to-text evaluation by incorporating a graph
representing relationships between different social cues to aid the model's
reasoning process.

To run this script, you need to have the following:
    - A Phi-4 model checkpoint.
    - A dataset with video frames, audio, and annotations.
    - A pre-computed audio-to-text transcription.
    - A graph representing relationships between social cues.

Example usage:
python phi4_video_audio_SI_baseline_audio2text_conv_all_graph.py \
    --signal graph_H --frame_num 5
"""

import argparse
import json
import os

from evaluation import evaluation
import numpy as np
from PIL import Image
import soundfile as sf
from tqdm import tqdm
import transformers

GenerationConfig = transformers.GenerationConfig
AutoProcessor = transformers.AutoProcessor
AutoModelForCausalLM = transformers.AutoModelForCausalLM


def main(sub_signal, frame_num):
  """Runs the evaluation for a given sub_signal and frame number."""

  # Define model path
  model_path = 'microsoft/Phi-4-multimodal-instruct'
  # frame_num = 10

  graph = {}
  subsignal_list = []
  file_list = [
      'video_someone_else_talk.json',
      'video_alternating_speech_turns.json',
      'video_talk_to_me.json',
      'video_I_am_talk.json',
      'video_within_personal_space.json',
      'video_looking_at_me.json',
      'video_looking_at_someone.json',
      'video_focus.json',
  ]

  # Load config
  with open('config.json', 'r') as f:
    config = json.load(f)
  data_base_path = config['data_base_path']

  log_base_path = os.path.join(data_base_path, 'logs')
  phi4_log_path = os.path.join(log_base_path, 'phi4')

  for file in file_list:
    with open(
        os.path.join(phi4_log_path, file),
        'r',
    ) as f:
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

  # Load model and processor
  processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(
      model_path,
      device_map='cuda',
      torch_dtype='auto',
      trust_remote_code=True,
      # if you do not use Ampere or later GPUs, change attention to "eager"
      _attn_implementation='flash_attention_2',
  ).cuda()

  # Load generation config
  generation_config = GenerationConfig.from_pretrained(model_path)

  # Define prompt structure
  user_prompt = '<|user|>'
  assistant_prompt = '<|assistant|>'
  prompt_suffix = '<|end|>'

  annotation_path = os.path.join(
      data_base_path, 'annotation_new_full_social_interaction_focus_v3.json'
  )

  with open(annotation_path, 'r') as f:
    annotation = json.load(f)

  audio_text_path = os.path.join(
      log_base_path, 'gemini2.5_aud2text_conv_dic.json'
  )
  with open(audio_text_path, 'r') as f:
    audio_text_all = json.load(f)

  keys_list = list(annotation.keys())
  keys_list.sort()
  total_number = len(keys_list)
  print('Total number:', total_number)
  total_yes = 0
  total_no = 0
  correct_yes = 0
  correct_no = 0

  os.makedirs(phi4_log_path, exist_ok=True)
  save_json_path = os.path.join(
      phi4_log_path,
      f'video_audio_SI_baseline_aud2text_conv_{sub_signal}_{frame_num}f.json',
  )
  if os.path.exists(save_json_path):
    with open(save_json_path, 'r') as f:
      output = json.load(f)
  else:
    output = []

  print(save_json_path, len(output))

  for clip_id in tqdm(keys_list[len(output) :]):
    # print(clip_id)
    audio_text = audio_text_all[clip_id]

    context = annotation[clip_id]['context']
    sequence = annotation[clip_id]['sequence']
    audio_path = os.path.join(data_base_path, 'audio', f'{clip_id}.wav')

    audio = sf.read(audio_path)

    # Part 1: Image Processing
    images_all = []
    images = []
    placeholder = ''
    frames_base_path = os.path.join(data_base_path, 'frames')
    if frame_num < 10:
      for i in range(10):
        image = Image.open(os.path.join(frames_base_path, clip_id, f'{i}.jpg'))
        images_all.append(image)
      indices = np.linspace(0, len(images_all) - 1, frame_num, dtype=int)
      for i in indices:
        images.append(images_all[i])
        placeholder += f'<|image_{i+1}|>'
    else:
      for i in range(10):
        image = Image.open(os.path.join(frames_base_path, clip_id, f'{i}.jpg'))
        images.append(image)
        placeholder += f'<|image_{i+1}|>'

    sub_signal_q = ', '.join(graph[clip_id])

    if sub_signal == 'graph_100_H':
      question = (
          'Task: You are analyzing a first-person (egocentric) video and audio'
          ' feed captured by AR glasses worn by an individual. Is the wearer'
          ' involved in a social interaction? \n Extra information: This is'
          ' the text converted from the video audio, please use it if it'
          f' helps. Text: {audio_text} \n Priori knowledge: There are 8'
          ' predicted factors graph related to social interaction in triplet'
          f' fromat: [{sub_signal_q}]. There is a hierarchical relation in'
          ' those 8 core factors, factors 1 and 5 are first level and if both'
          " false you don't need to analyze other core factors; factors 2,6,7"
          ' are midle levels, go further when anyone of them is true; factors'
          ' 3,4,8 are third level and are very important. Do not 100% rely on'
          ' this. Please refer one or some of them to help you answer the'
          ' question. \n Respond concisely in the following format: \n Answer:'
          ' [Yes/No]\n  Confidence: [High/Medium/Low] \n Reasoning: (Briefly'
          ' explain the reason.)'
      )
    elif sub_signal == 'graph_H':
      question = (
          'Task: You are analyzing a first-person (egocentric) video and audio'
          ' feed captured by AR glasses worn by an individual. Is the wearer'
          ' involved in a social interaction? \n Extra information: This is'
          ' the text converted from the video audio, please use it if it'
          f' helps. Text: {audio_text} \n Priori knowledge: There are 8'
          ' predicted factors graph related to social interaction in triplet'
          f' fromat: [{sub_signal_q}]. Do not 100% rely on this. There is a'
          ' hierarchical relation in those 8 core factors, factors 1 and 5 are'
          " first level and if both false you don't need to analyze other core"
          ' factors; factors 2,6,7 are midle levels, go further when anyone of'
          ' them is true; factors 3,4,8 are third level and are very'
          ' important. Do not 100% rely on this. Please refer one or some of'
          ' them to help you answer the question. \n Respond concisely in the'
          ' following format: \n Answer: [Yes/No]\n  Confidence:'
          ' [High/Medium/Low] \n Reasoning: (Briefly explain the reason.)'
      )
    elif sub_signal == 'graph_rea_H':
      question = (
          'Task: You are analyzing a first-person (egocentric) video and audio'
          ' feed captured by AR glasses worn by an individual. Is the wearer'
          ' involved in a social interaction? \n Extra information: This is'
          ' the text converted from the video audio, please use it if it'
          f' helps. Text: {audio_text} \n Priori knowledge: There are 8'
          ' predicted factors graph related to social interaction in triplet'
          f' fromat: [{sub_signal_q}]. Do not 100% rely on this. There is a'
          ' hierarchical relation in those 8 core factors, factors 1 and 5 are'
          " first level and if both false you don't need to analyze other core"
          ' factors; factors 2,6,7 are midle levels, go further when anyone of'
          ' them is true; factors 3,4,8 are third level and are very'
          ' important. Do not 100% rely on this. Please refer one or some of'
          ' them to help you answer the question. \n Respond concisely in the'
          ' following format: \n Answer: [Yes/No]\n  Confidence:'
          ' [High/Medium/Low] \n Reasoning: (List the core factor indexes you'
          ' used.)'
      )
    else:
      question = (
          'You are analyzing a first-person (egocentric) video and audio feed'
          ' captured by AR glasses worn by an individual. \n (This is the text'
          ' converted from the video audio, please use it if it helps. Text:'
          f' {audio_text}) Is the wearer involved in a social interaction?'
          f' Please refer to the core fact "{sub_signal_q}" to help you answer'
          ' the question. \n Respond concisely in the following format: \n'
          ' Answer: [Yes/No]\n  Confidence: [High/Medium/Low] \n Reasoning:'
          ' (Briefly explain the reason.)'
      )

    prompt = f'{user_prompt}{placeholder}<|audio_1|>{question}{prompt_suffix}{assistant_prompt}'
    inputs = processor(
        text=prompt, images=images, audios=[audio], return_tensors='pt'
    ).to('cuda:0')

    # Generate response
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    output.append({'key': clip_id, 'response': response.lower()})

    with open(save_json_path, 'w') as f:
      json.dump(output, f)

  res = evaluation(output, annotation)

  with open(f'./res/phi4_{sub_signal}_{frame_num}f.json', 'w') as f:
    json.dump(res, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Deepsocial model evaluation.')
  parser.add_argument('--signal', type=str, required=True, help='s1-s8')
  parser.add_argument(
      '--frame_num', type=int, required=True, help='frame umber'
  )
  args = parser.parse_args()

  main(args.signal, args.frame_num)
