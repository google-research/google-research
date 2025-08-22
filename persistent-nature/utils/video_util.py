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

"""utils for writing video."""
import base64
import io
import re
import PIL
import skvideo.io


def get_writer(file_name, frame_rate=30, pix_fmt='yuv420p', vcodec='libx264'):
  rate = str(frame_rate)
  inputdict = {'-r': rate}
  outputdict = {
      '-pix_fmt': pix_fmt,
      '-r': rate,
      '-vcodec': vcodec,
  }
  writer = skvideo.io.FFmpegWriter(file_name, inputdict, outputdict)
  return writer


def save_image(url, path):
  data = re.sub('^data:image/.+;base64,', '', url)
  image = PIL.Image.open(io.BytesIO(base64.b64decode(data)))
  image.save(path)
