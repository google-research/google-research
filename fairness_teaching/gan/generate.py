# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import skimage.io as io
import data
import model
import util
# pylint: skip-file

#####################  parser  #####################

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', default='./data', help='Path of all image files')
parser.add_argument('--train_label', default='./annotations/train_label.txt', help='Train labels file')
parser.add_argument('--test_label', default='./annotations/test_label.txt', help='Test labels file')
parser.add_argument('--valid_label', default='./annotations/val_label.txt', help='Valid labels file')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--output_dir', default='output', help='Output dir [default: log]')
parser.add_argument('--n_epoch', type=int, default=60, help='Epoch to run [default: 60]')
parser.add_argument('--n_adv', type=int, default=20, help='Epoch to run adv_cls[default: 20]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size during training [default: 64]')
parser.add_argument('--sample_size', type=int, default=24, help='Sample size during validation [default: 24]')
parser.add_argument('--n_class', type=int, default=2, help='Number of class [default: 2]')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate [default: 0.1]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
FLAGS = parser.parse_args()

#####################  config  #####################

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
IMG_PATH = FLAGS.img_path
TRAIN_LABEL = FLAGS.train_label
TEST_LABEL = FLAGS.test_label
VALID_LABEL = FLAGS.valid_label
BATCH_SIZE = FLAGS.batch_size
N_SAMPLE = FLAGS.sample_size
N_EPOCH = FLAGS.n_epoch
N_ADV = FLAGS.n_adv
N_CLASS = FLAGS.n_class
LR = FLAGS.lr
MOMENTUM = FLAGS.momentum
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
LOG_PATH = os.path.join(ROOT_PATH, FLAGS.log_dir)
OUT_PATH = os.path.join(ROOT_PATH, FLAGS.output_dir)
if not os.path.exists(LOG_PATH): os.mkdir(LOG_PATH)
if not os.path.exists(OUT_PATH): os.mkdir(OUT_PATH)

(train_images, train_labels), train_iters = data.data_train(IMG_PATH, TRAIN_LABEL, BATCH_SIZE)
(valid_images, valid_labels), valid_iters = data.data_test(IMG_PATH, VALID_LABEL, N_SAMPLE)
Genc = model.Genc()
Gdec = model.Gdec()
D = model.D()
Adv = model.Adv_cls()

####################################################

def V_graph(sess, phv):
  real_labels = valid_labels*2 -1
  fake_labels = -real_labels

  u = Genc.build(valid_images, phv['is_training_v'])
  fake_images = Gdec.build(u, fake_labels, phv['is_training_v'])

  return fake_images

def train():
  is_training_v = tf.placeholder(tf.bool, shape=[])
  phv = {'is_training_v': is_training_v}
  with tf.Session() as sess:
    fa_images = V_graph(sess, phv)
    saver = tf.train.Saver(Genc.vars+Gdec.vars)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess,LOG_PATH+'/weight.ckpt')

    #####################  generate data  #####################
    v_dict = {phv['is_training_v']: True}
    cnt = 0
    for i in range(train_iters):
      sample = sess.run(fa_images, feed_dict = v_dict)
      sample = util.to_uint8(sample)
      for s in sample:
        cnt += 1
        io.imsave('%s/%06d.jpg' % (OUT_PATH, cnt), s, quality=95)


if __name__ == "__main__":
  train()
