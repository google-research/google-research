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

import io
import json
import os
from urllib.request import urlopen
from PIL import Image
import requests
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

"""Baseline evaluation script for the Phi-4 model on social interaction detection.

This script loads the Phi-4 model, processes video frames and audio,
and prompts the model to determine if a social interaction is taking place.
"""

# Define model path
model_path = 'microsoft/Phi-4-multimodal-instruct'
frame_num = 10

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

# Path to the annotation file
annotation_path = os.path.join(data_base_path, 'annotation.json')

with open(annotation_path, 'r') as f:
  annotation = json.load(f)

keys_list = list(annotation.keys())
total_number = len(keys_list)
print('Total number:', total_number)

# Variables for tracking evaluation metrics (currently not used in the loop)
total_yes = 0
total_no = 0
correct_yes = 0
correct_no = 0

log_base_path = os.path.join(data_base_path, 'logs/phi4')
os.makedirs(log_base_path, exist_ok=True)
save_json_path = os.path.join(log_base_path, 'video_audio_SI_baseline.json')

# Load existing results if the output file exists
if os.path.exists(save_json_path):
  with open(save_json_path, 'r') as f:
    output = json.load(f)
else:
  output = []


print(save_json_path, len(output), total_yes, total_no, correct_yes, correct_no)


for clip_id in tqdm(keys_list[len(output) :]):
  # print(clip_id)

  context = annotation[clip_id]['context']
  sequence = annotation[clip_id]['sequence']
  audio_path = os.path.join(data_base_path, 'audio', f'{clip_id}.wav')

  audio = sf.read(audio_path)

  # Part 1: Image Processing
  images = []
  placeholder = ''
  frame_base_path = os.path.join(data_base_path, 'frames')
  for i in range(frame_num):
    image = Image.open(os.path.join(frame_base_path, clip_id, f'{i}.jpg'))
    images.append(image)
    placeholder += f'<|image_{i+1}|>'

  question = (
      'You are analyzing a first-person (egocentric) video and audio feed'
      ' captured by AR glasses worn by an individual. Is the wearer involved in'
      ' a social interaction? \n Respond concisely in the following format: \n'
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
      generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
  )[0]

  output.append({'key': clip_id, 'response': response.lower()})

  with open(save_json_path, 'w') as f:
    json.dump(output, f)
