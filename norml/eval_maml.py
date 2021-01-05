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

r"""Create meta and fine-tuned rollouts from saved MAML RL policies.

The meta policy is rolled out using the first meta-policy network,
i.e. algo.train_policies[0]. After fine-tuning, weights of the fine-tuned policy
replaces that of meta policy, so that multiple fine-tune steps could be
performed.

How to use:
python -m norml.eval_maml \
--model_dir \
example_checkpoints/move_point_rotate_sparse/norml/all_weights.ckpt-991 \
--output_dir /usr/local/google/home/yxy
ang/temp \
--render=True \
--num_finetune_steps 1 \
--test_task_index 0 \
--eval_finetune=True

Some trained models for the NoRML paper are stored in example_checkpoints
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import cv2
from matplotlib.pylab import plt
import numpy as np
import tensorflow.compat.v1 as tf
from norml import maml_rl
from norml import tools
from norml.tools import utility

flags.DEFINE_integer('framerate', 25, 'Video framerate.')
flags.DEFINE_bool('render', True, 'Create video?')
flags.DEFINE_string('model_dir', None, 'Checkpoint path for saved model')
flags.DEFINE_string('output_dir', '/tmp', 'Where to store states, rewards...')
flags.DEFINE_integer('test_task_index', 0,
                     'Which task modifier to use for testing')
flags.DEFINE_bool('eval_meta', True, 'Whether to evaluate the meta policy')
flags.DEFINE_bool('eval_finetune', True,
                  'Whether to evaluate the finetune policy')
flags.DEFINE_integer('num_finetune_steps', 1,
                     'Number of finetune steps to perform')
plt.ion()


def _load_config():
  if '.ckpt-' in FLAGS.model_dir:
    config_path = os.path.dirname(FLAGS.model_dir)
  else:
    config_path = FLAGS.model_dir
  tf.logging.info('Loading config: %s' % config_path)
  config = utility.load_config(config_path)
  return config


def _save_result(renders, states, returns, actions, output_dir):
  """Saves the results of policy rollouts to file.

  Args:
    renders: video frames for the episode
    states: observations for the episode
    returns: rewards for the episode
    actions: actions taken for the episode
    output_dir: directory to write all result files
  """
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)
  if FLAGS.render:
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    height, width = renders[0].shape[0], renders[0].shape[1]
    vid_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'video.mp4'), fourcc, FLAGS.framerate,
        (width, height))
    for render in renders:
      vid_writer.write(render)
    vid_writer.release()
    print('Saved video to: %s' % os.path.join(output_dir, 'video.mp4'))
  with tf.gfile.GFile(os.path.join(output_dir, 'train_states.csv'), 'wb') as fh:
    np.savetxt(fh, np.array(states).astype(np.float32), delimiter=',')
  with tf.gfile.GFile(os.path.join(output_dir, 'train_rewards.csv'),
                      'wb') as fh:
    np.savetxt(fh, np.array(returns).astype(np.float32), delimiter=',')
  with tf.gfile.GFile(os.path.join(output_dir, 'train_actions.csv'),
                      'wb') as fh:
    np.savetxt(fh, np.array(actions).astype(np.float32), delimiter=',')


def _rollout_and_save(sess, task, policy, output_dir, max_rollout_len=2500):
  """Performs rollout on policy and save results.

  Args:
    sess: TF session
    task: task to perform rollout on
    policy: policy for the task
    output_dir: directory to output rollout files
    max_rollout_len: maximum length of the rollout

  Returns:
    sum of rewards for this current rollout
  """

  task = tools.wrappers.ClipAction(tools.wrappers.RangeNormalize(task))
  sample_op, state_var = policy.sample_op()
  state = task.reset().reshape((1, -1))

  renders = []
  done = False
  sample_vars = {}
  returns = []
  states = [state.ravel()]
  actions = []
  if FLAGS.render:
    renders.append(task.render(mode='rgb_array'))
  while (not done) and (len(states) < max_rollout_len):
    sample_vars[state_var] = state.reshape((1, -1))
    action = sess.run(sample_op, sample_vars).ravel()
    new_state, reward, done, _ = task.step(action)
    state = new_state
    if FLAGS.render:
      renders.append(task.render(mode='rgb_array'))
    returns.append(np.ravel(reward))
    states.append(new_state.ravel())
    actions.append(action.ravel())

  _save_result(renders, states, returns, actions, output_dir)
  return np.sum(returns)


def main(argv):
  del argv  # Unused
  config = _load_config()
  algo = maml_rl.MAMLReinforcementLearning(
      config, logdir=FLAGS.model_dir, save_config=False)

  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    algo.restore(sess, FLAGS.model_dir)
    task = config.task_generator()
    task_modifier = config.task_env_modifiers[FLAGS.test_task_index]
    for attr in task_modifier:
      task.__setattr__(attr, task_modifier[attr])

    if FLAGS.eval_meta:
      sum_reward = _rollout_and_save(
          sess,
          task,
          algo.train_policies[0],
          os.path.join(FLAGS.output_dir,
                       'task_{}'.format(FLAGS.test_task_index), 'meta'),
          max_rollout_len=config.max_rollout_len)
      print('Total reward for meta policy is: {}'.format(sum_reward))

    if FLAGS.eval_finetune:
      for step in range(FLAGS.num_finetune_steps):
        tf.logging.info('Finetune step: {}'.format(step))
        algo.finetune(sess, task_modifier)
        sum_reward = _rollout_and_save(
            sess,
            task,
            algo.train_policies[0],
            os.path.join(FLAGS.output_dir, 'task_{}'.format(
                FLAGS.test_task_index), 'finetune_{}'.format(step)),
            max_rollout_len=config.max_rollout_len)
        print('Total reward for fine-tuned policy is: {}'.format(sum_reward))


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  tf.app.run()
