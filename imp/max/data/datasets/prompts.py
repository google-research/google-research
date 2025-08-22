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

"""Prompts for zero-shot classification.

Templates are based on CLIP, with some modifications to be able to generate more
general style prompts for images, video, and audio.

See also:
https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
"""

# Simple prompt templates.
NO_PROMPTS = (
    '{label}',
)
BASE_IMAGE_PROMPTS = (
    'This is a photo of {label}.',
    'This photo shows {label}.',
    'You can see {label} in this photo.',
    'There is a depiction of {label} in this photo.',
    'This image contains {label}.',
)
BASE_VIDEO_PROMPTS = (
    'This is a video of {label}.',
    'This video clip shows {label}.',
    'You can see {label} in this video.',
    'There is a depiction of {label} over time in this video.',
    'This video clip contains {label}.',
)
BASE_AUDIO_PROMPTS = (
    'This is an audio recording of {label}.',
    'This audio clip shows {label}.',
    'You hear the sound of {label} in this audio clip.',
    'A sound of {label} can be heard over time in this audio recording.',
    'This audio clip contains {label}.',
)

# More advanced prompt templates.
TRAIN_GENERIC_PROMPTS = (
    'a {instance} of {quantifier} {label}.',
    'a {instance} of {quantifier} interesting {label}.',
    'a {instance} of {quantifier} cool {label}.',
    'a {instance} of {quantifier} weird {label}.',
    'a interesting {instance} of {quantifier} {label}.',
    'this is a {instance} of {quantifier} {label}.',
    'this {instance} involves {quantifier} {label}.',
    'this {instance} consists of {quantifier} {label}.',
    'this {instance} has {quantifier} {label}.',
    'this {instance} contains {quantifier} {label}.',
    'you can find {quantifier} {label} in the {instance}.',
    'there is a depiction of {quantifier} {label} in {instance}.',
    'what is in this {instance}? there is {label}.',
    'can you describe this {instance}? it seems like {label}.',
    'i am pretty sure there is {label}.',
    '{quantifier} {label} in a {instance}.',
    '{instance} of {label}',
    '{quantifier} {label}',
    '{label}',
)

TRAIN_IMAGE_PROMPTS = TRAIN_GENERIC_PROMPTS + (
    'itap of {quantifier} {label}.',
    'i took a picture of {quantifier} {label} on my camera.',
)
TRAIN_VIDEO_PROMPTS = TRAIN_GENERIC_PROMPTS + (
    'a {instance} of a person {label}.',
    'a example of a person doing {label}.',
    'a {instance} of a person performing {label}.',
    'a demonstration of a person practicing {label}.',
    'a demonstration of {label} over time in this {instance}.',
    'there is motion in this {instance} with {label}.',
    'i took a video of {quantifier} {label} on my camera.',
)
TRAIN_AUDIO_PROMPTS = TRAIN_GENERIC_PROMPTS + (
    'a {label} with an interesting sound.',
    'you hear the sound of {label} in this {instance}.',
    'a sound of {label} can be heard over time in this {instance}.',
    'i recorded {label} with my microphone.',
)
EVAL_IMAGE_PROMPTS = (
    'an image of the {label}.',
    'itap of a {label}.',
    'a bad photo of the {label}.',
    'a origami {label}.',
    'a photo of the large {label}.',
    'a {label} in a video game.',
    'art of the {label}.',
    'a photo of the small {label}.',

    'a photo of many {label}.',
    'a sculpture of a {label}.',
    'a jpeg corrupted photo of the {label}.',
    'a photo of the cool {label}.',
    'the plastic {label}.',
    'this image has an object of type {label}.',
    'there is one object in this image. it is of {label}.',
    '{label}',
)
EVAL_VIDEO_PROMPTS = (
    'a video of the {label}.',
    'a clip of the large {label}.',
    'a {label} in a video game.',
    'a appearance of {label}.',
    'render of the {label}.',
    'a video of the small {label}.',
    'this movie shows {label}.',
    'there is a depiction of {label} in the video.',

    'a photo of a person doing {label}.',
    'a video of a person {label}.',
    'a example of a person doing {label}.',
    'a video of a person performing {label}.',
    'a demonstration of a person practicing {label}.',
    'a demonstration of {label}.',
    'a photo of a person using {label}.',
    'a photo of a person doing {label}.',
)
EVAL_AUDIO_PROMPTS = (
    'audio of the {label}.',
    'a bad recording of the {label}.',
    'a sound of {label} can be heard over time in this audio.',
    'a sound of the clear {label}.',
    'a {label} in a video game.',
    'sound of the {label}.',
    'a radio of this {label}.',
    'you can hear {label} in this recording.',

    'a sound of many {label}.',
    'a studio recording of a {label}.',
    'a corrupted waveform of the {label}.',
    'audio of the cool {label}.',
    'the object is {label}.',
    'this image has an object that makes the sound {label}.',
    'there is something making sound in this image. it is of {label}.',
    '{label}',
)
IMAGE_MODALITY_INSTANCES = (
    'image',
    'photo',
    'photograph',
    'picture',
    'jpeg',
    'crop',
    'frame',
    'rendition',
    'capture',
    'figure',
    'appearance',
    'photocopy',
    'still',
    'view',
)
VIDEO_MODALITY_INSTANCES = (
    'video',
    'movie',
    'clip',
    'motion picture',
    'film',
    'broadcast',
    'recording',
    'tape',
    'render',
    'rendering',
    'rendition',
    'cinematic',
    'cutscene',
    'cartoon',
    'figure',
    'appearance',
    'tv show',
    'capture',
    'mp4',
)
AUDIO_MODALITY_INSTANCES = (
    'audio',
    'clip',
    'audio clip',
    'recording',
    'sound',
    'broadcast',
    'tape',
    'radio',
    'music',
    'voice',
    'capture',
)
QUANTIFIER_INSTANCES = (
    'a',
    'an',
    'the',
    'this',
    'that',
    'my',
    'their',
    '',
)

# The original CLIP prompts.
CLIP_TRAIN_PROMPTS = (
    'a bad photo of a {label}.',
    'a photo of many {label}.',
    'a sculpture of a {label}.',
    'a photo of the hard to see {label}.',
    'a low resolution photo of the {label}.',
    'a rendering of a {label}.',
    'graffiti of a {label}.',
    'a bad photo of the {label}.',
    'a cropped photo of the {label}.',
    'a tattoo of a {label}.',
    'the embroidered {label}.',
    'a photo of a hard to see {label}.',
    'a bright photo of a {label}.',
    'a photo of a clean {label}.',
    'a photo of a dirty {label}.',
    'a dark photo of the {label}.',
    'a drawing of a {label}.',
    'a photo of my {label}.',
    'the plastic {label}.',
    'a photo of the cool {label}.',
    'a close-up photo of a {label}.',
    'a black and white photo of the {label}.',
    'a painting of the {label}.',
    'a painting of a {label}.',
    'a pixelated photo of the {label}.',
    'a sculpture of the {label}.',
    'a bright photo of the {label}.',
    'a cropped photo of a {label}.',
    'a plastic {label}.',
    'a photo of the dirty {label}.',
    'a jpeg corrupted photo of a {label}.',
    'a blurry photo of the {label}.',
    'a photo of the {label}.',
    'a good photo of the {label}.',
    'a rendering of the {label}.',
    'a {label} in a video game.',
    'a photo of one {label}.',
    'a doodle of a {label}.',
    'a close-up photo of the {label}.',
    'a photo of a {label}.',
    'the origami {label}.',
    'the {label} in a video game.',
    'a sketch of a {label}.',
    'a doodle of the {label}.',
    'a origami {label}.',
    'a low resolution photo of a {label}.',
    'the toy {label}.',
    'a rendition of the {label}.',
    'a photo of the clean {label}.',
    'a photo of a large {label}.',
    'a rendition of a {label}.',
    'a photo of a nice {label}.',
    'a photo of a weird {label}.',
    'a blurry photo of a {label}.',
    'a cartoon {label}.',
    'art of a {label}.',
    'a sketch of the {label}.',
    'a embroidered {label}.',
    'a pixelated photo of a {label}.',
    'itap of the {label}.',
    'a jpeg corrupted photo of the {label}.',
    'a good photo of a {label}.',
    'a plushie {label}.',
    'a photo of the nice {label}.',
    'a photo of the small {label}.',
    'a photo of the weird {label}.',
    'the cartoon {label}.',
    'art of the {label}.',
    'a drawing of the {label}.',
    'a photo of the large {label}.',
    'a black and white photo of a {label}.',
    'the plushie {label}.',
    'a dark photo of a {label}.',
    'itap of a {label}.',
    'graffiti of the {label}.',
    'a toy {label}.',
    'itap of my {label}.',
    'a photo of a cool {label}.',
    'a photo of a small {label}.',
    'a tattoo of the {label}.',
)
CLIP_EVAL_PROMPTS = (
    'itap of a {label}.',
    'a bad photo of the {label}.',
    'a origami {label}.',
    'a photo of the large {label}.',
    'a {label} in a video game.',
    'art of the {label}.',
    'a photo of the small {label}.',
)
