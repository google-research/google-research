# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Training script for Time Agnostic Prediction."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import h5py
from .models.tap import TAP
import tensorflow.compat.v1 as tf
from .utils import sample_batch_tap
from .utils import save_im
from tensorflow.contrib import checkpoint as contrib_checkpoint
tf.enable_eager_execution()


FLAGS = flags.FLAGS
flags.DEFINE_integer('batchsize', 256,
                     'Batch Size')
flags.DEFINE_integer('latentsize', 8,
                     'Latent Size')
flags.DEFINE_integer('trainsteps', 100000,
                     'Train Steps')
flags.DEFINE_string('datapath', '/tmp/test.hdf5',
                    'Path to the HDF5 dataset')
flags.DEFINE_string('savedir', '/tmp/mazetap/',
                    'Where to save the model')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  batchsize = FLAGS.batchsize
  latentsize = FLAGS.latentsize

  savedir = FLAGS.savedir + str(batchsize) + '_' + str(latentsize) + '/'
  path = FLAGS.datapath

  if not os.path.exists(savedir):
    os.makedirs(savedir)

  f = h5py.File(path, 'r')
  ims = f['sim']['ims'][:, :, :, :, :]

  it = TAP(latentsize, width=64)
  optim = tf.train.AdamOptimizer(0.001)
  batch, _ = sample_batch_tap(batchsize, ims, epnum=ims.shape[0],
                              epsize=ims.shape[1])
  start = tf.convert_to_tensor(batch[:, 0])
  end = tf.convert_to_tensor(batch[:, 1])
  it(batchsize, start, end)
  global_step = tf.train.get_or_create_global_step()
  ckpt = tf.train.Checkpoint(
      optimizer=optim, variables=contrib_checkpoint.List(it.variables))

  for i in range(FLAGS.trainsteps):
    def loss_fn():
      """Feed forward and compute loss."""
      batch, trajs = sample_batch_tap(batchsize, ims, epnum=ims.shape[0],
                                      epsize=ims.shape[1])
      start = tf.convert_to_tensor(batch[:, 0])
      end = tf.convert_to_tensor(batch[:, 1])
      outall = it(batchsize, start, end)
      rec = outall

      losses = []
      for b in range(batchsize):
        traj = trajs[b]
        tlen = traj.shape[0]
        bloss = []
        for t in range(tlen):
          frame = traj[t]
          framelikelihood = tf.reduce_sum(frame * tf.log(rec[b]) +
                                          (1-frame) * tf.log(1-rec[b]))
          fl = -1 * (tf.reduce_mean(framelikelihood))
          bloss.append(fl)
        bl = tf.reduce_min(bloss)
        losses.append(bl)

      loss = tf.reduce_mean(losses)
      if global_step.numpy() % 1000 == 0:
        # pylint: disable=cell-var-from-loop
        print(loss, global_step.numpy())
        save_im(255 * batch[0, 0], savedir + 's1_' + str(i) + '.jpg')
        save_im(255 * batch[0, 1], savedir +'s2_' + str(i) + '.jpg')
        save_im(255*rec[0].numpy(), savedir + 's1pred_' + str(i) + '.jpg')
        ckpt.save(savedir + 'model_' + str(global_step.numpy()))
      return loss
    optim.minimize(loss_fn, global_step=global_step)

if __name__ == '__main__':
  app.run(main)
