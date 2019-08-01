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

"""Subgoal Wrapper around Maze Environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from . import environment
from absl import flags
from ..models import tdm
from ..models import vae
import numpy as np
from tensor2tensor.bin.t2t_decoder import create_hparams
from tensor2tensor.utils import registry
import tensorflow as tf
from ..utils import save_im

FLAGS = flags.FLAGS


class SubGoalEnv(object):
  """Subgoal Wrapper around Maze Environment."""

  def __init__(self, difficulty, modeltype, cost, numsg=1, savedir='/tmp/',
               envtype='maze', phorizon=5, parallel=0,
               tdmdir='/tmp/mazetdm/', vaedir='/tmp/mazevae'):
    self.parallel = parallel
    self.envtype = envtype
    self.cost = cost
    self.modeltype = modeltype

    # Only use maze env
    if self.envtype == 'maze':
      self.env = environment.Environment(difficulty=difficulty)
      self.num_acts = 2
    else:
      raise NotImplementedError

    self.numsg = numsg
    self.eps = 0
    self.savedir = savedir
    self.planstep = 0
    self.phorizon = phorizon

    self.it_graph = tf.Graph()
    with self.it_graph.as_default():
      self.itsess = tf.Session()
      self.it = vae.ImageTransformSC(8)
      outall = self.it(bs=1)
      self.out, _, _, _ = outall

      itsaver = tf.train.Saver()
      vaedir = vaedir + '256_8/'
      # Restore variables from disk.
      itsaver.restore(self.itsess, vaedir + 'model-0')
      print('LOADED VAE!')

    # LOADING TDM
    self.tdm_graph = tf.Graph()
    with self.tdm_graph.as_default():
      self.tdmsess = tf.Session()
      self.tdm = tdm.TemporalModel()
      self.s1 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
      self.s2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
      self.tdout = tf.nn.softmax(self.tdm(self.s1, self.s2))

      tdmsaver = tf.train.Saver()
      tdmdir = tdmdir + '256TDM/'
      # Restore variables from disk.
      tdmsaver.restore(self.tdmsess, tdmdir + 'model-0')
      print('LOADED TDM!')

    # LOADING SV2P (Modify this to your path and problem)
    homedir = '/usr/local/google/home/nairsuraj'
    FLAGS.data_dir = homedir + '/data/maze3/'
    FLAGS.output_dir = homedir + '/models/rs=6.3/maze3/checkpoint'
    FLAGS.problem = 'video_bair_robot_pushing'
    FLAGS.hparams = 'video_num_input_frames=1,video_num_target_frames=10'
    FLAGS.hparams_set = 'next_frame_sv2p'
    FLAGS.model = 'next_frame_sv2p'
    # Create hparams
    hparams = create_hparams()
    hparams.video_num_input_frames = 1
    hparams.video_num_target_frames = self.phorizon

    # Params
    num_replicas = 200
    frame_shape = hparams.problem.frame_shape
    forward_graph = tf.Graph()
    with forward_graph.as_default():
      self.forward_sess = tf.Session()
      input_size = [num_replicas, hparams.video_num_input_frames]
      target_size = [num_replicas, hparams.video_num_target_frames]
      self.forward_placeholders = {
          'inputs':
              tf.placeholder(tf.float32, input_size + frame_shape),
          'input_action':
              tf.placeholder(tf.float32, input_size + [self.num_acts]),
          'targets':
              tf.placeholder(tf.float32, target_size + frame_shape),
          'target_action':
              tf.placeholder(tf.float32, target_size + [self.num_acts]),
      }
      # Creat model
      forward_model_cls = registry.model(FLAGS.model)
      forward_model = forward_model_cls(hparams, tf.estimator.ModeKeys.PREDICT)
      self.forward_prediction_ops, _ = forward_model(self.forward_placeholders)
      forward_saver = tf.train.Saver()
      forward_saver.restore(self.forward_sess,
                            homedir + '/models/rs=6.3/maze6/model.ckpt-0')
    print('LOADED SV2P!')

    _, self.state = self.env.get_observation()

  def reset(self):
    """Reset environment."""
    self.eps += 1  # Episode counter
    self.steps = 0
    self.env.reset()
    self.goalim = self.env.get_goal_im()
    _, self.state = self.env.get_observation()

    # Logdir for episode
    os.makedirs(self.savedir +  str(self.eps) + '/')
    # Save Current and Goal Image
    save_im(self.goalim, self.savedir + str(self.eps) + '/' + str(self.steps) +
            'goalim.jpg')
    save_im(self.state, self.savedir + str(self.eps) + '/' + str(self.steps) +
            'currstate.jpg')
    self._episode_ended = False
    return np.array(self.state, dtype=np.uint8)

  def latent_to_im(self, latent):
    """Takes subgoal latent and feeds it through decoder to generate image."""
    forward_feed = {
        self.it.s1:
            np.repeat(np.expand_dims(self.state/255., 0), self.numsg, 0),
        self.it.s2:
            np.repeat(np.expand_dims(self.state/255., 0), self.numsg, 0),
        self.it.z:
            latent.reshape((self.numsg, 8)),
        }
    delta = 255. * self.itsess.run(self.out, forward_feed)
    return delta

  def temporal_cost(self, ims, goal, tp=1):
    """Takes two images and returns temporal distance prediction."""
    if tp == 1:
      forward_feed = {
          self.s1:
              ims/255.0,
          self.s2:
              np.repeat(np.expand_dims(goal/255., 0), ims.shape[0], 0),
      }
      costs = self.tdmsess.run(self.tdout, forward_feed)[:, 0]
      return costs
    else:
      forward_feed = {
          self.s1: ims/255.0,
      }
      ims_e = self.tdmsess.run(self.tdout, forward_feed)
      forward_feed = {
          self.s1:
              np.repeat(np.expand_dims(goal/255., 0), ims.shape[0], 0),
      }
      g_e = self.tdmsess.run(self.tdout, forward_feed)
      return np.mean((g_e - ims_e)**2, 1)

  def plan(self, im1, im2):
    """Plans between two images."""
    action, cost = self.cem(self.forward_sess, self.forward_placeholders,
                            self.forward_prediction_ops, im1, im2)
    return action, cost

  def step(self, action):
    """Step one 'Subgoal' action in the environment."""
    self.steps += 1
    # Convert subgoals to image
    im = self.latent_to_im(action)
    costs = []
    trajs = []
    for i in range(self.numsg+1):
      if i == 0:
        acts, cost = self.plan(self.state, im[i])
      elif i == self.numsg:
        acts, cost = self.plan(im[i-1], self.goalim)
      else:
        acts, cost = self.plan(im[i-1], im[i])
      costs.append(cost)
      trajs.append(acts)

      if i == self.numsg:
        save_im(self.goalim, self.savedir + str(self.eps) +'/' + str(self.steps)
                +  '_C_' + str(cost) +  'subgoal' + str(i) + '.jpg')
      else:
        save_im(im[i], self.savedir + str(self.eps) + '/' + str(self.steps)
                +  '_C_' + str(cost) + 'subgoal'+ str(i) + '.jpg')

    return -np.max(costs), trajs
    # return -np.mean(costs), trajs

  def cem(self, forward_sess, forward_placeholders, forward_ops, im1, im2):
    """Runs Visual MPC between two images."""

    horizon = forward_placeholders['targets'].shape[1]
    mu1 = np.array([0]*(self.num_acts * horizon))
    sd1 = np.array([0.2]*(self.num_acts * horizon))

    t = 0
    sample_size = 200
    resample_size = 40

    if horizon == 15:
      hz = 5
    else:
      hz = horizon

    while np.max(sd1) > .001:
      if t == 0:
        acts1x = np.random.uniform(0, 3, (sample_size, hz, 1))
        acts1y = np.random.uniform(-3, 3, (sample_size, hz, 1))
        acts1 = np.concatenate([acts1x, acts1y], 2)
      else:
        acts1 = np.random.normal(mu1, sd1, (sample_size, self.num_acts *  hz))
      acts1 = acts1.reshape((sample_size, hz, self.num_acts))
      if horizon == 15:
        acts1 = np.repeat(acts1, 3, axis=1)
      acts0 = acts1[:, 0:1, :]

      forward_feed = {
          forward_placeholders['inputs']:
              np.repeat(np.expand_dims(np.expand_dims(im1, 0), 0),
                        sample_size, axis=0),
          forward_placeholders['input_action']:
              acts0,
          forward_placeholders['targets']:
              np.zeros(forward_placeholders['targets'].shape),
          forward_placeholders['target_action']:
              acts1
      }
      forward_predictions = forward_sess.run(forward_ops, forward_feed)

      losses = []
      if self.cost == 'temporal':
        losses = self.temporal_cost(forward_predictions.reshape(-1, 64, 64, 3),
                                    im2)
        losses = losses.reshape(sample_size, horizon)
      elif self.cost == 'pixel':
        goalim = np.repeat(np.expand_dims(
            np.repeat(np.expand_dims(im2, 0), horizon, 0), 0), sample_size, 0)
        losses = (goalim - forward_predictions[:, :, :, :, :, 0])**2
        losses = losses.mean(axis=(2, 3, 4))
      losses = losses[:, -1]

      for q in range(sample_size):
        head = self.savedir + str(self.eps) + '/' + str(self.planstep) + '/'
        tail = str(t) + '/' + str(losses[q]) + '_' + str(q) + '/'
        drr = head + tail
        os.makedirs(drr)
        with open(drr + 'acts.txt', 'a') as f:
          f.write(str(acts1[q]))
        save_im(im1, drr+'curr.jpg')
        save_im(im2, drr+'goal.jpg')
        for p in range(horizon):
          save_im(forward_predictions[q, p, :, :, :, 0], drr + str(p) + '.jpg')

      best_actions = np.array([x for _, x in sorted(
          zip(losses, acts1.tolist()), reverse=False)][:resample_size])
      best_costs = np.array([x for x, _ in sorted(
          zip(losses, acts1.tolist()), reverse=False)][:resample_size])

      if horizon == 15:
        best_actions = best_actions[:, ::3, :]
      else:
        best_actions = best_actions[:, :, :]
      best_actions1 = best_actions.reshape(resample_size, -1)
      mu1 = np.mean(best_actions1, axis=0)
      sd1 = np.std(best_actions1, axis=0)
      t += 1
      if t >= 5:
        break

    chosen = best_actions1[0]
    bestcost = best_costs[0]
    return chosen, bestcost
