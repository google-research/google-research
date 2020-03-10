# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Training script for VAE."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags
import h5py
from .models.vae import ImageTransformSC
import numpy as np
import tensorflow.compat.v1 as tf
from .utils import sample_batch_vae
from .utils import save_im

FLAGS = flags.FLAGS

flags.DEFINE_integer('batchsize', 256,
                     'Batch Size')
flags.DEFINE_integer('latentsize', 8,
                     'Latent Size')
flags.DEFINE_float('beta', 1,
                     'Beta')
flags.DEFINE_integer('trainsteps', 100000,
                     'Train Steps')
flags.DEFINE_string('datapath', '/tmp/test.hdf5',
                    'Path to the HDF5 dataset')
flags.DEFINE_string('savedir', '/tmp/mazevae/',
                    'Where to save the model')



def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  batchsize = FLAGS.batchsize
  latentsize = FLAGS.latentsize

  savedir = FLAGS.savedir + str(batchsize) + '_' + str(latentsize) + "_" + str(FLAGS.beta) + '/'
  path = FLAGS.datapath

  if not os.path.exists(savedir):
    os.makedirs(savedir)

  f = h5py.File(path, 'r')
  ims = f['sim']['ims'][:, :, :, :, :]

  it = ImageTransformSC(latentsize)
  outall = it(batchsize)
  out, _, mu, var = outall

  likelihood2 = tf.reduce_sum(it.s2 * tf.log(out) + (1-it.s2)*tf.log(1-out),
                              axis=[1, 2, 3])
  likelihood = likelihood2
  kl = 0.5 * tf.reduce_sum(-1 - tf.log(1e-5 +var) + tf.math.square(mu) + var,
                           axis=[1])
  loss = -1 * (tf.reduce_mean(likelihood) - FLAGS.beta * tf.reduce_mean(kl))

  optim = tf.train.AdamOptimizer(0.0001)
  optimizer_step = optim.minimize(loss)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for i in range(FLAGS.trainsteps):
      batch = sample_batch_vae(batchsize, ims, env='maze', epnum=ims.shape[0],
                               epsize=ims.shape[1])
      forward_feed = {
          it.s1: batch[:, 0],
          it.s2: batch[:, 1]
      }

      o, l, _ = sess.run([outall, loss, optimizer_step], forward_feed)
      delta, rc, _, _ = o
      if i % 10000 == 0:
        saver.save(sess, savedir + 'model', global_step=i)
        save_im(255*batch[0, 0], savedir+ 's1_'+str(i)+'.jpg')
        save_im(255*batch[0, 1], savedir+'s2_'+str(i)+'.jpg')
        save_im(255*(delta[0]), savedir+'s2pred_'+str(i)+'.jpg')
        save_im(255*(rc[0]), savedir+'s1pred_'+str(i)+'.jpg')

        sys.stdout.write(str(l) + ', ' +str(i) + '\n')

        forward_feed = {
            it.s1: np.repeat(np.expand_dims(batch[0, 0], 0), batchsize, 0),
            it.s2: np.repeat(np.expand_dims(batch[0, 1], 0), batchsize, 0),
            it.z: np.random.normal([0.]*latentsize, [1.]*latentsize,
                                   (batchsize, latentsize))
        }
        delta = sess.run(out, forward_feed)
        for j in range(batchsize)[:20]:
          save_im(255*(delta[j]), savedir + 'gen'+str(i)+'_'+str(j)+'.jpg')

if __name__ == '__main__':
  app.run(main)
