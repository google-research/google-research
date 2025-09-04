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

"""Evaluation script for Phi-4 model with audio-to-text conversion.

This script evaluates the Phi-4 model on social interaction detection,
providing the model with transcribed audio text in addition to video frames.
It can test different sub-signals related to social interaction.
"""

import argparse
import io
import json
import os
from urllib.request import urlopen

from evaluation import evaluation
from PIL import Image
import requests
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import GenerationConfig


def main(sub_signal):
  """Runs the evaluation for a given sub_signal."""

  # Define model path
  model_path = 'microsoft/Phi-4-multimodal-instruct'
  frame_num = 10

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
  else:
    sub_signal_q = ' '.join(sub_signal_q_list)

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

  # Load config
  with open('config.json', 'r') as f:
    config = json.load(f)
  data_base_path = config['data_base_path']

  annotation_path = os.path.join(
      data_base_path, 'annotation_new_full_social_interaction_focus_v3.json'
  )

  with open(annotation_path, 'r') as f:
    annotation = json.load(f)

  log_base_path = os.path.join(data_base_path, 'logs')
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

  phi4_log_path = os.path.join(log_base_path, 'phi4')
  os.makedirs(phi4_log_path, exist_ok=True)
  save_json_path = os.path.join(
      phi4_log_path, f'video_audio_SI_baseline_aud2text_conv_{sub_signal}.json'
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
    images = []
    placeholder = ''
    frames_base_path = os.path.join(data_base_path, 'frames')
    for i in range(frame_num):
      image = Image.open(os.path.join(frames_base_path, clip_id, f'{i}.jpg'))
      images.append(image)
      placeholder += f'<|image_{i+1}|>'

    if sub_signal == 'all_H':
      question = (
          'Task: You are analyzing a first-person (egocentric) video and audio'
          ' feed captured by AR glasses worn by an individual. Is the wearer'
          ' involved in a social interaction? \n Extra information: This is'
          ' the text converted from the video audio, please use it if it'
          f' helps. Text: {audio_text} \n Priori knowledge: There are 8 core'
          f' factors related to social interaction: "{sub_signal_q}" There is a'
          ' hierarchical relation in those 8 core factors, qestion 1 and 5 are'
          " first level and if both false you don't need to analyze other"
          ' core factors; question 2,6,7 are midle levels, go further when'
          ' anyone of them is true; question 3,4,8 are third level and are'
          ' very important. Please refer one or some of them to help you'
          ' answer the question. \n Respond concisely in the following format:'
          ' \n Answer: [Yes/No]\n  Confidence: [High/Medium/Low] \n Reasoning:'
          ' (Briefly explain the reason.)'
      )
    elif sub_signal == 'all_rea_H':
      question = (
          'Task: You are analyzing a first-person (egocentric) video and audio'
          ' feed captured by AR glasses worn by an individual. Is the wearer'
          ' involved in a social interaction? \n Extra information: This is'
          ' the text converted from the video audio, please use it if it'
          f' helps. Text: {audio_text} \n Priori knowledge: There are 8 core'
          f' factors related to social interaction: "{sub_signal_q}" There is a'
          ' hierarchical relation in those 8 core factors, qestion 1 and 5 are'
          " first level and if both false you don't need to analyze other"
          ' core factors; question 2,6,7 are midle levels, go further when'
          ' anyone of them is true; question 3,4,8 are third level and are'
          ' very important. Please refer one or some of them to help you'
          ' answer the question. \n Respond concisely in the following format:'
          ' \n Answer: [Yes/No]\n  Confidence: [High/Medium/Low] \n Reasoning:'
          ' (List the core factor indexes you used.)'
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

  with open(f'./res/phi4_{sub_signal}.json', 'w') as f:
    json.dump(res, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Deepsocial model evaluation.')
  parser.add_argument('--signal', type=str, required=True, help='s1-s8')
  args = parser.parse_args()

  main(args.signal)
