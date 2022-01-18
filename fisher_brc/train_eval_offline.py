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

"""Run training loop for batch rl."""
import os

from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
import tqdm

from fisher_brc import behavioral_cloning
from fisher_brc import d4rl_utils
from fisher_brc import evaluation
from fisher_brc import fisher_brc

FLAGS = flags.FLAGS

flags.DEFINE_string('task_name', 'halfcheetah-expert-v0', 'Env name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('num_updates', 1_000_000, 'Num updates.')
flags.DEFINE_enum('algo_name', 'fbrc', ['bc', 'fbrc'], 'Algorithm.')
flags.DEFINE_integer('bc_pretraining_steps', 1_000_000, 'Num updates.')
flags.DEFINE_integer('num_eval_episodes', 10, 'Num eval episodes.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10_000, 'Evaluation interval.')
flags.DEFINE_string('save_dir', '/tmp/save/', 'Saving directory.')
flags.DEFINE_boolean('eager', False, 'Execute functions eagerly.')
flags.DEFINE_float('f_reg', 0.1, 'Fisher regularization.')
flags.DEFINE_float('reward_bonus', 5.0, 'CQL style reward bonus.')


def main(_):
  tf.config.experimental_run_functions_eagerly(FLAGS.eager)

  gym_env, dataset = d4rl_utils.create_d4rl_env_and_dataset(
      task_name=FLAGS.task_name, batch_size=FLAGS.batch_size)

  env = gym_wrapper.GymWrapper(gym_env)
  env = tf_py_environment.TFPyEnvironment(env)

  dataset_iter = iter(dataset)

  tf.random.set_seed(FLAGS.seed)

  hparam_str = f'{FLAGS.algo_name}_{FLAGS.task_name}_seed={FLAGS.seed}'

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb', hparam_str))
  result_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'results', hparam_str))

  if FLAGS.algo_name == 'bc':
    model = behavioral_cloning.BehavioralCloning(
        env.observation_spec(),
        env.action_spec())
  else:
    model = fisher_brc.FBRC(
        env.observation_spec(),
        env.action_spec(),
        target_entropy=-env.action_spec().shape[0],
        f_reg=FLAGS.f_reg,
        reward_bonus=FLAGS.reward_bonus)

    for i in tqdm.tqdm(range(FLAGS.bc_pretraining_steps)):
      info_dict = model.bc.update_step(dataset_iter)

      if i % FLAGS.log_interval == 0:
        with summary_writer.as_default():
          for k, v in info_dict.items():
            tf.summary.scalar(
                f'training/{k}', v, step=i - FLAGS.bc_pretraining_steps)

  for i in tqdm.tqdm(range(FLAGS.num_updates)):
    with summary_writer.as_default():
      info_dict = model.update_step(dataset_iter)

    if i % FLAGS.log_interval == 0:
      with summary_writer.as_default():
        for k, v in info_dict.items():
          tf.summary.scalar(f'training/{k}', v, step=i)

    if (i + 1) % FLAGS.eval_interval == 0:
      average_returns, average_length = evaluation.evaluate(env, model)
      average_returns = gym_env.get_normalized_score(average_returns) * 100.0

      with result_writer.as_default():
        tf.summary.scalar('evaluation/returns', average_returns, step=i+1)
        tf.summary.scalar('evaluation/length', average_length, step=i+1)

if __name__ == '__main__':
  app.run(main)
