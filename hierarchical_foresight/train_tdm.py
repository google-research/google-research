# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Training script for the temporal difference model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import h5py
from .models.tdm import TemporalModel
import numpy as np
import tensorflow as tf
from .utils import sample_batch_tdm_maze
from .utils import save_im

FLAGS = flags.FLAGS

flags.DEFINE_integer('batchsize', 256,
                     'Batch Size')
flags.DEFINE_integer('latentsize', 8,
                     'Latent Size')
flags.DEFINE_integer('trainsteps', 10000,
                     'Train Steps')
flags.DEFINE_string('datapath', '/tmp/test.hdf5',
                    'Path to the HDF5 dataset')
flags.DEFINE_string('savedir', '/tmp/mazetap/',
                    'Where to save the model')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  batchsize = FLAGS.batchsize
  savedir = FLAGS.savedir + str(batchsize) + 'TDM' '/'
  path = FLAGS.datapath

  if not os.path.exists(savedir):
    os.makedirs(savedir)

  f = h5py.File(path, 'r')

  eps = f['sim']['ims'][:].shape[0]
  trainims = f['sim']['ims'][:int(0.8*eps), :, :, :, :]
  testims = f['sim']['ims'][int(0.8*eps):, :, :, :, :]

  gt1 = tf.placeholder(tf.float32, shape=[None, 2])
  gt2 = tf.placeholder(tf.float32, shape=[None, 2])
  s1 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
  s2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
  s3 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
  lm = tf.placeholder(tf.float32, shape=[None, 1])
  it = TemporalModel()
  outall1 = it(s1, s2)
  outall2 = it(s1, s3)

  lmr = tf.reshape(lm, (-1, 1, 1, 1))
  smix = lmr * s2 + (1-lmr)*s3

  outallmix_pred = it(s1, smix)
  outallmix = lm * outall1 + (1-lm)*outall2

  loss1 = tf.losses.softmax_cross_entropy(gt1, outall1)
  loss2 = tf.losses.softmax_cross_entropy(gt2, outall2)
  loss3 = tf.reduce_mean(tf.reduce_sum(
      (outallmix_pred - outallmix)**2, 1))
  loss = loss1 + loss2 + loss3

  optim = tf.train.AdamOptimizer(0.001)
  optimizer_step = optim.minimize(loss)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for i in range(FLAGS.trainsteps):
      batch, labels = sample_batch_tdm_maze(batchsize, trainims)
      forward_feed = {
          s1: batch[:, 0] + np.random.uniform(-0.05, 0.05, batch[:, 0].shape),
          s2: batch[:, 1] + np.random.uniform(-0.05, 0.05, batch[:, 1].shape),
          s3: batch[:, 2] + np.random.uniform(-0.05, 0.05, batch[:, 2].shape),
          gt1: labels[:, :2],
          gt2: labels[:, 2:],
          lm: np.random.beta(1, 1, size=(batchsize, 1)),
      }

      l, _ = sess.run([loss, optimizer_step], forward_feed)
      if i % 10000 == 0:
        saver.save(sess, savedir + 'model', global_step=i)
        batch, labels = sample_batch_tdm_maze(batchsize, testims)
        forward_feed = {
            s1: batch[:, 0],
            s2: batch[:, 1],
            s3: batch[:, 2],
            gt1: labels[:, :2],
            gt2: labels[:, 2:],
            lm: np.random.beta(1, 1, size=(batchsize, 1)),
        }
        tl, to = sess.run([loss, tf.nn.softmax(outall1)], forward_feed)
        for j in range(batchsize)[:20]:
          save_im(255*batch[j, 0], savedir+str(i) + '_'+str(j) +
                  '_dist' + str(to[j, 0]) + '_1.jpg')
          save_im(255*batch[j, 1], savedir+str(i) + '_'+str(j) +
                  '_dist' + str(to[j, 0]) + '_2.jpg')
        print('ITER:' + str(i) + ',' + 'TRAIN:'+str(l) + ', ' + 'TEST:'+str(tl))

if __name__ == '__main__':
  app.run(main)
