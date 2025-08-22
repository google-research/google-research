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

# pylint: skip-file
import os
import pickle as pkl
import tensorflow as tf


def save_checkpoint(filepath, step, content, keep_old=2):
  with tf.io.gfile.GFile(filepath + '-' + str(step), 'wb') as f:
    pkl.dump(content, f)
  if keep_old > 0:
    filelist = tf.io.gfile.glob(filepath + '-*')
    key_fn = lambda fn: int(os.path.basename(fn).split('-')[-1])
    filelist = sorted(filelist, key=key_fn)
    if len(filelist) > keep_old + 1:
      for fn in filelist[:-keep_old - 1]:
        tf.io.gfile.remove(fn)


def last_checkpoint(ckpt_dir):
  ckpt_list = tf.io.gfile.glob(os.path.join(ckpt_dir, 'ckpt-*'))
  if len(ckpt_list) > 0:
    key_fn = lambda fn: int(os.path.basename(fn).split('-')[-1])
    ckpt_list = sorted(ckpt_list, key=key_fn)
    ckpt_fn = ckpt_list[::-1]
    return ckpt_fn
  else:
    return None
