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

"""Run HVF in the maze environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from absl import app
from absl import flags
from .env import subgoal_env
from .models.tap import TAP
import numpy as np
import tensorflow.compat.v1 as tf
from .utils import save_im
from .models import q_func


FLAGS = flags.FLAGS

flags.DEFINE_string('difficulty', 'm',
                    'difficulty')
flags.DEFINE_string('modeltype', 'sv2p',
                    'model')
flags.DEFINE_string('cost', 'pixel',
                    'cost')
flags.DEFINE_integer('numsg', 1,
                     'num subgoals')
flags.DEFINE_integer('horizon', 20,
                     'episode horizon')
flags.DEFINE_integer('parallel', 0,
                     'parallel')
flags.DEFINE_integer('num_parallel', 1,
                     'num_parallel')
flags.DEFINE_integer('gt_goals', 0,
                     'GT subgoals')
flags.DEFINE_integer('phorizon', 5,
                     'Planning Horiozn')
flags.DEFINE_string('envtype', 'franka',
                    'env type')
flags.DEFINE_string('savedir', '/tmp/meta_cem/',
                    'Where to save results')
flags.DEFINE_string('tapdir', '/tmp/mazetap/',
                    'Path to the TAP model')
flags.DEFINE_string('tdmdir', None,
                    'Path to the TDM model')
flags.DEFINE_string('vaedir', '/tmp/mazevae/',
                    'Path to the VAE model')
flags.DEFINE_integer('metacem_samples', 100, 
                     'Number of samples in CEM over subgoals')
flags.DEFINE_integer('cem_samples', 100, 
                     'Number of samples in CEM for planning')
flags.DEFINE_integer('metacem_iters', 3, 
                     'Number of iters in CEM over subgoals')
flags.DEFINE_integer('cem_iters', 3, 
                     'Number of iters in CEM for planning')


      
def meta_cem(env):
  """Runs the META-CEM procedure over the subgoals."""
  # Initialize subgoal distribution
  mu1 = np.array([0]*(8*FLAGS.numsg))
  sd1 = np.array([1.0]*(8*FLAGS.numsg))

  t = 0
  sample_size = FLAGS.metacem_samples
  resample_size = FLAGS.metacem_samples // 5

  while np.max(sd1) > .001:
    acts1 = np.random.normal(mu1, sd1, (sample_size, (8*FLAGS.numsg)))
    losses = []
    for i in range(sample_size):
      c, _ = env.step(acts1[i])
      losses.append(-c)
    losses = np.array(losses)

    best_actions = np.array([x for _, x in sorted(
        zip(losses, acts1.tolist()), reverse=False)][:resample_size])
    best_actions1 = best_actions.reshape(resample_size, -1)

    mu1 = np.mean(best_actions1, axis=0)
    sd1 = np.std(best_actions1, axis=0)

    t += 1
    if t >= FLAGS.metacem_iters:
      break
  chosen = best_actions1[0]
  return chosen


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  savedir = FLAGS.savedir
  savedir += 'DIFF' + str(FLAGS.difficulty)
  savedir += '_COST' + str(FLAGS.cost)
  savedir += '_MT' + str(FLAGS.modeltype)
  savedir += '_SG' + str(FLAGS.numsg)
  savedir += '_GTGOAL' + str(FLAGS.gt_goals)
  savedir += '_H' + str(FLAGS.horizon)
  savedir += '_PH' + str(FLAGS.phorizon)
  savedir += '/P' + str(FLAGS.parallel) + '/'

  if not os.path.exists(savedir):
    os.makedirs(savedir)

  # If using TAP as subgoals load TAP
  if FLAGS.gt_goals == 2:
    tapsess = tf.Session()
    tap = TAP(8, width=64)
    s1 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    s2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    out = tap(1, s1, s2)
    tapsaver = tf.train.Saver()
    tapdir = FLAGS.tapdir + '256_8/'
    # Restore variables from disk.
    tapsaver.restore(tapsess, tapdir + 'model_0-1')
    print('LOADED TAP!')

  env = subgoal_env.SubGoalEnv(difficulty=FLAGS.difficulty,
                               modeltype=FLAGS.modeltype,
                               cost=FLAGS.cost,
                               numsg=FLAGS.numsg,
                               savedir=savedir,
                               envtype=FLAGS.envtype,
                               phorizon=FLAGS.phorizon,
                               parallel=FLAGS.parallel,
                               tdmdir=FLAGS.tdmdir,
                               vaedir=FLAGS.vaedir,
                               cem_samples = FLAGS.cem_samples,
                               cem_iters = FLAGS.cem_iters)

  forward_episodes = 100 // int(FLAGS.num_parallel)
  forward_steps_per_ep = FLAGS.horizon

  if FLAGS.envtype == 'maze':
    maxsg = 10  # MAZE
  else:
    maxsg = 20  # FRANKA

  rl_successes = []
  for ep in range(forward_episodes):
    for step in range(forward_steps_per_ep):
      env.planstep = step
      if step == 0:
        env.reset()
        goalnum = 0
        sgsteps = 0

        if FLAGS.numsg == 0:
          goalim = np.expand_dims(env.goalim, 0)
          subgoals = goalim
        else:
          if FLAGS.gt_goals == 1:
            goalim = np.expand_dims(env.goalim, 0)
            subgoals = env.env.get_subgoal_ims(FLAGS.numsg)
            subgoals = np.concatenate([subgoals, goalim], 0)
          elif FLAGS.gt_goals == 2:
            if FLAGS.numsg == 1:
              _, cim = env.env.get_observation()
              goalim = np.expand_dims(env.goalim, 0)
              forward_feed = {
                  s1: np.expand_dims(cim/255.0, 0),
                  s2: goalim /255.0,
              }
              sm = tapsess.run(out, forward_feed)
              subgoals = np.concatenate([sm*255., goalim], 0)
            elif FLAGS.numsg == 2:
              _, cim = env.env.get_observation()
              goalim = np.expand_dims(env.goalim, 0)
              forward_feed = {
                  s1: np.expand_dims(cim/255.0, 0),
                  s2: goalim /255.0,
              }
              sm = tapsess.run(out, forward_feed)

              forward_feed = {
                  s1: np.expand_dims(cim/255.0, 0),
                  s2: sm,
              }
              sm2 = tapsess.run(out, forward_feed)

              subgoals = np.concatenate([sm2*255., sm*255., goalim], 0)
          else:
            goalim = np.expand_dims(env.goalim, 0)
            bestlatent = meta_cem(env)
            subgoals = env.latent_to_im(bestlatent)
            subgoals = np.concatenate([subgoals, goalim], 0)

        for i in range(FLAGS.numsg+1):
          save_im(subgoals[i], savedir + str(ep+1) + '/' + 'Bestsubgoal'+
                  str(i) + '.jpg')

      if FLAGS.envtype == 'real':
        im = env.state
      else:
        _, im = env.env.get_observation()
      if FLAGS.cost == 'temporal':
        cst = env.temporal_cost(np.expand_dims(im, 0), subgoals[goalnum])
      else:
        cst = [0] 
      pxcst = np.mean((im -subgoals[goalnum])**2)
      save_im(im, savedir + str(ep+1) + '/' + 'state'+ str(step) + '_pxcost' +
              str(pxcst) + '_tmcst' + str(cst[0])+ '.jpg')
      if FLAGS.cost == 'pixel':
        if (pxcst < 2) or (sgsteps >= maxsg):
          if goalnum < FLAGS.numsg:
            goalnum += 1
            sgsteps = 0
      elif FLAGS.cost == 'temporal':
        if (cst < 0.0) or (sgsteps >= maxsg):
          if goalnum < FLAGS.numsg:
            goalnum += 1
            sgsteps = 0

      acts, _ = env.plan(im, subgoals[goalnum])
      if FLAGS.envtype == 'real':
        break
      else:
        if FLAGS.envtype == 'maze':
          env.env.step(acts[:2])
        else:
          env.env.step(acts[:3])
        sgsteps += 1

      if step == forward_steps_per_ep - 1:
        break

      if env.env.is_goal():
        break
    if FLAGS.envtype == 'real':
      rl_successes.append(0)
    else:
      if env.env.is_goal():
        rl_successes.append(1)
      else:
        rl_successes.append(0)
    sys.stdout.write(str(np.sum(rl_successes)) + '_' + str(ep) + '\n')
  with open(savedir+'result.txt', 'w') as f:
    f.write(str(np.mean(rl_successes)) + '\n')
  sys.stdout.write('RL Success Rate: ' + str(np.mean(rl_successes)) + '\n')


if __name__ == '__main__':
  app.run(main)
