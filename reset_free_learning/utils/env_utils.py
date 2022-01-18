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

"""Get the environment and related visualizations."""

import functools

import numpy as np
import tensorflow as tf
from tf_agents.environments.suite_gym import wrap_env

from reset_free_learning.envs import playpen
from reset_free_learning.envs import playpen_reduced
from reset_free_learning.envs.point_mass import PointMassEnv
from reset_free_learning.envs.point_mass_full_goal import PointMassEnv as PointMassFullGoalEnv
from reset_free_learning.envs.pusher2d_simple import PusherEnv

from reset_free_learning.utils.metrics import AnyStepGoalMetric
from reset_free_learning.utils.metrics import FailedEpisodes
from reset_free_learning.utils.metrics import StateVisitationHeatmap
from reset_free_learning.utils.metrics import ValueFunctionHeatmap


def get_env(name='sawyer_push',
            max_episode_steps=None,
            gym_env_wrappers=(),
            **env_kwargs):

  reset_state_shape = None
  reset_states = None

  eval_metrics = []
  train_metrics = []
  train_metrics_rev = []  # metrics for reverse policy
  value_metrics = []

  # if name == 'sawyer_push':
  #   env = SawyerObject(
  #       random_init=True,
  #       task_type='push',
  #       obs_type='with_goal',
  #       goal_low=(-0.1, 0.8, 0.05),
  #       goal_high=(0.1, 0.9, 0.3),
  #       liftThresh=0.04,
  #       sampleMode='equal',
  #       rewMode='orig',
  #       rotMode='fixed')
  #   env.set_camera_view(view='topview')
  #   env.set_max_path_length(int(1e8))
  #   eval_metrics += [
  #       FailedEpisodes(
  #           failure_function=functools.partial(
  #               sawyer_push_success, episodic=True),
  #           name='EvalSuccessfulEpisodes')
  #   ]
  #   train_metrics += [
  #       FailedEpisodes(
  #           failure_function=functools.partial(
  #               sawyer_push_success, episodic=False),
  #           name='TrainSuccessfulStates')
  #   ]

  #   if name == 'sawyer_door':
  #     env = SawyerDoor(random_init=True, obs_type='with_goal')
  #     env.set_camera_view(view='topview')
  #     env.set_max_path_length(int(1e8))
  #     env.set_reward_type(reward_type=env_kwargs.get('reward_type', 'dense'))
  #     eval_metrics += [
  #         FailedEpisodes(
  #             failure_function=functools.partial(
  #                 sawyer_door_success, episodic=True),
  #             name='EvalSuccessfulEpisodes')
  #     ]
  #     train_metrics += [
  #         FailedEpisodes(
  #             failure_function=functools.partial(
  #                 sawyer_door_success, episodic=False),
  #             name='TrainSuccessfulStates')
  #     ]
  #     # metrics for reverse policy
  #     train_metrics_rev += [
  #         FailedEpisodes(
  #             failure_function=functools.partial(
  #                 sawyer_door_success, episodic=False),
  #             name='TrainSuccessfulStatesRev')
  #     ]
  #     reset_state_shape = (6,)
  #     reset_states = np.array(
  #         [[-0.00356643, 0.4132358, 0.2534339, -0.21, 0.69, 0.15]])
  #     train_metrics += [
  #         StateVisitationHeatmap(
  #             trajectory_to_xypos=lambda x: x[:, :2],
  #             state_max=1.,
  #             num_bins=20,
  #             name='EndEffectorHeatmap',
  #         ),
  #         StateVisitationHeatmap(
  #             trajectory_to_xypos=lambda x: x[:, 3:5],
  #             state_max=None,
  #             x_range=(-0.25, 0.25),
  #             y_range=(0.4, 0.9),
  #             num_bins=20,
  #             name='DoorXYHeatmap',
  #         ),
  #         StateVisitationHeatmap(
  #             trajectory_to_xypos=lambda x: x[:, 6:8],
  #             state_max=None,
  #             x_range=(-0.25, 0.25),
  #             y_range=(0.4, 0.9),
  #             num_bins=20,
  #             name='EndEffectorGoalHeatmap',
  #         ),
  #         StateVisitationHeatmap(
  #             trajectory_to_xypos=lambda x: x[:, 9:11],
  #             state_max=None,
  #             x_range=(-0.25, 0.25),
  #             y_range=(0.4, 0.9),
  #             num_bins=20,
  #             name='DoorXYGoalHeatmap',
  #         ),
  #     ]

  #     # metrics to visualize the value function
  #     value_metrics += [
  #         ValueFunctionHeatmap(
  #             trajectory_to_xypos=lambda x: x[:, 3:5],
  #             state_max=None,
  #             x_range=(-0.25, 0.25),
  #             y_range=(0.4, 0.9),
  #             num_bins=20,
  #             name='DoorXYGoalValueHeatmap',
  #         ),
  #         # ValueFunctionHeatmap(
  #         #     trajectory_to_xypos=lambda x: x[:, 3:5],
  #         #     state_max=None,
  #         #     x_range=(-0.25, 0.25),
  #         #     y_range=(0.4, 0.9),
  #         #     num_bins=20,
  #         #     name='DoorXYCombinedHeatmap',
  #         # ),
  #     ]

  #     # metrics for reverse policy
  #     train_metrics_rev += [
  #         StateVisitationHeatmap(
  #             trajectory_to_xypos=lambda x: x[:, :2],
  #             state_max=1.,
  #             num_bins=20,
  #             name='EndEffectorHeatmapRev',
  #         ),
  #         StateVisitationHeatmap(
  #             trajectory_to_xypos=lambda x: x[:, 3:5],
  #             state_max=None,
  #             x_range=(-0.25, 0.25),
  #             y_range=(0.1, 0.7),
  #             num_bins=20,
  #             name='DoorXYHeatmapRev',
  #         ),
  #         StateVisitationHeatmap(
  #             trajectory_to_xypos=lambda x: x[:, 6:8],
  #             state_max=1.,
  #             num_bins=20,
  #             name='EndEffectorGoalHeatmapRev',
  #         ),
  #         StateVisitationHeatmap(
  #             trajectory_to_xypos=lambda x: x[:, 9:11],
  #             state_max=None,
  #             x_range=(-0.25, 0.25),
  #             y_range=(0.1, 0.7),
  #             num_bins=20,
  #             name='DoorXYGoalHeatmapRev',
  #         ),
  #     ]

  if name == 'pusher2d_simple':
    env = PusherEnv()
    eval_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(
                pusher2d_simple_success, episodic=True),
            name='EvalSuccessfulEpisodes')
    ]
    train_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(
                pusher2d_simple_success, episodic=False),
            name='TrainSuccessfulStates')
    ]

  if name == 'point_mass':
    env = PointMassEnv(**env_kwargs)
    eval_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(
                point_mass_success, episodic=True),
            name='EvalSuccessfulEpisodes')
    ]
    train_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(
                point_mass_success, episodic=False),
            name='TrainSuccessfulStates')
    ]

    # reverse metrics
    train_metrics_rev += [
        FailedEpisodes(
            failure_function=functools.partial(
                point_mass_success, episodic=False),
            name='TrainSuccessfulStatesRev')
    ]
    reset_state_shape = (2,)
    reset_state_by_env_type = {
        'default': np.array([
            0.0,
            0.0,
        ]),
        't': np.array([0.0, 0.0]),
        'y': np.array([0.0, 8.0]),
        'skewed_square': np.array([0.0, -8.0])
    }
    reset_states = np.expand_dims(
        reset_state_by_env_type[env_kwargs.get('env_type', 'default')], axis=0)

    train_metrics += [
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, :2],
            state_max=10.,
            num_bins=20,
            name='StateVisitationHeatmap',
        ),
        # distribution of goals: goals are always the last two dimensions
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, -2:],
            state_max=10.,
            num_bins=20,
            name='SelectedGoalHeatmap',
        )
    ]

    train_metrics_rev += [
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, :2],  # pylint: disable=invalid-sequence-index
            state_max=10.,
            num_bins=20,
            name='StateVisitationHeatmapRev',
        ),
        # distribution of goals: goals are always the last two dimensions
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, -2:],  # pylint: disable=invalid-sequence-index
            state_max=10.,
            num_bins=20,
            name='SelectedGoalHeatmapRev',
        )
    ]

  if name == 'point_mass_full_goal':
    env = PointMassFullGoalEnv(**env_kwargs)
    eval_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(
                point_mass_success, episodic=True),
            name='EvalSuccessfulEpisodes')
    ]
    train_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(
                point_mass_success, episodic=False),
            name='TrainSuccessfulStates')
    ]

    # reverse metrics
    train_metrics_rev += [
        FailedEpisodes(
            failure_function=functools.partial(
                point_mass_success, episodic=False),
            name='TrainSuccessfulStatesRev')
    ]
    reset_state_shape = (6,)
    reset_state_by_env_type = {
        'default': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        't': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'y': np.array([0.0, 8.0, 0.0, 0.0, 0.0, 0.0]),
        'skewed_square': np.array([0.0, -8.0, 0.0, 0.0, 0.0, 0.0])
    }
    reset_states = np.expand_dims(
        reset_state_by_env_type[env_kwargs.get('env_type', 'default')], axis=0)

    train_metrics += [
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, :2],
            state_max=10.,
            num_bins=20,
            name='StateVisitationHeatmap',
        ),
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, 6:8],
            state_max=10.,
            num_bins=20,
            name='SelectedGoalHeatmap',
        )
    ]

    train_metrics_rev += [
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, :2],
            state_max=10.,
            num_bins=20,
            name='StateVisitationHeatmapRev',
        ),
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, 6:8],
            state_max=10.,
            num_bins=20,
            name='SelectedGoalHeatmapRev',
        )
    ]

#   if name == 'kitchen':
#     env = kitchen.Kitchen(**env_kwargs)

#     eval_metrics += [
#         FailedEpisodes(
#             failure_function=functools.partial(
#                 kitchen_microwave_success, episodic=True),
#             name='EvalSuccessfulEpisodes')
#     ]
#     train_metrics += [
#         FailedEpisodes(
#             failure_function=functools.partial(
#                 kitchen_microwave_success, episodic=False),
#             name='TrainSuccessfulStates')
#     ]
#     reset_state_shape = kitchen.initial_state.shape[1:]
#     reset_states = kitchen.initial_state

  if name == 'playpen':
    env = playpen.ContinuousPlayPen(**env_kwargs)

    eval_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(playpen_success, episodic=True),
            name='EvalSuccessfulAtLastStep'),
        AnyStepGoalMetric(
            goal_success_fn=functools.partial(playpen_success, episodic=False),
            name='EvalSuccessfulAtAnyStep')
    ]
    train_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(playpen_success, episodic=False),
            name='TrainSuccessfulStates')
    ]
    reset_state_shape = playpen.initial_state.shape[1:]
    reset_states = playpen.initial_state.copy()

    # heatmap visualization
    task_list = env_kwargs.get('task_list', 'rc_o').split('-')
    interest_objects = []
    for task in task_list:
      subtask_list = task.split('__')
      interest_objects += [subtask[:2] for subtask in subtask_list]

    train_metrics += [
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, :2],
            state_max=3.,
            num_bins=20,
            name='GripperHeatmap',
        ),
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, 12:14],
            state_max=3.,
            num_bins=20,
            name='GripperGoalHeatmap',
        ),
    ]

    # metrics to visualize the value function
    value_metrics += [
        ValueFunctionHeatmap(
            trajectory_to_xypos=lambda x: x[:, :2],  # pylint: disable=invalid-sequence-index
            state_max=3.,
            num_bins=20,
            name='GripperGoalValueHeatmap',
        ),
    ]

    obj_to_name = {
        'rc': 'RedCube',
        'bc': 'BlueCube',
        'ks': 'BlackSphere',
        'yr': 'YellowCylinder'
    }
    obj_to_idx = {'rc': [2, 4], 'bc': [4, 6], 'ks': [6, 8], 'yr': [8, 10]}
    for obj_code in list(set(interest_objects)):
      state_ids = obj_to_idx[obj_code]
      heatmap_name = obj_to_name[obj_code]
      train_metrics += [
          StateVisitationHeatmap(
              trajectory_to_xypos=lambda x: x[:, state_ids[0]:state_ids[1]],  # pylint: disable=cell-var-from-loop
              state_max=3.,
              num_bins=20,
              name=heatmap_name + 'Heatmap',
          ),
          StateVisitationHeatmap(
              trajectory_to_xypos=lambda x: x[:, state_ids[0] + 12:state_ids[1]  # pylint: disable=cell-var-from-loop, g-long-lambda
                                              + 12],
              state_max=3.,
              num_bins=20,
              name=heatmap_name + 'GoalHeatmap',
          ),
      ]
      value_metrics += [
          ValueFunctionHeatmap(
              trajectory_to_xypos=lambda x: x[:, state_ids[0]:state_ids[1]],  # pylint: disable=cell-var-from-loop
              state_max=3.,
              num_bins=20,
              name=heatmap_name + 'GoalValueHeatmap',
          ),
      ]

  if name == 'playpen_reduced':
    env = playpen_reduced.ContinuousPlayPen(**env_kwargs)

    eval_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(
                playpen_reduced_success, episodic=True),
            name='EvalSuccessfulAtLastStep'),
        AnyStepGoalMetric(
            goal_success_fn=functools.partial(
                playpen_reduced_success, episodic=False),
            name='EvalSuccessfulAtAnyStep')
    ]
    train_metrics += [
        FailedEpisodes(
            failure_function=functools.partial(
                playpen_reduced_success, episodic=False),
            name='TrainSuccessfulStates')
    ]
    reset_state_shape = playpen_reduced.initial_state.shape[1:]
    reset_states = playpen_reduced.initial_state.copy()

    train_metrics += [
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, :2],
            state_max=3.,
            num_bins=20,
            name='GripperHeatmap',
        ),
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, 6:8],
            state_max=3.,
            num_bins=20,
            name='GripperGoalHeatmap',
        ),
    ]

    # metrics to visualize the value function
    value_metrics += [
        ValueFunctionHeatmap(
            trajectory_to_xypos=lambda x: x[:, :2],
            state_max=3.,
            num_bins=20,
            name='GripperGoalValueHeatmap',
        ),
    ]

    train_metrics += [
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, 2:4],
            state_max=3.,
            num_bins=20,
            name='RedCubeHeatmap',
        ),
        StateVisitationHeatmap(
            trajectory_to_xypos=lambda x: x[:, 8:10],
            state_max=3.,
            num_bins=20,
            name='RedCubeGoalHeatmap',
        ),
    ]
    value_metrics += [
        ValueFunctionHeatmap(
            trajectory_to_xypos=lambda x: x[:, 2:4],
            state_max=3.,
            num_bins=20,
            name='RedCubeGoalValueHeatmap',
        ),
    ]

  return wrap_env(
      env,
      max_episode_steps=max_episode_steps,
      gym_env_wrappers=gym_env_wrappers), train_metrics, eval_metrics, {
          'reset_state_shape': reset_state_shape,
          'reset_states': reset_states,
          'train_metrics_rev': train_metrics_rev,
          'value_fn_viz_metrics': value_metrics
      }


def playpen_success(trajectory, episodic=False, push_success_threshold=0.2):
  # works only when given goal observation
  assert trajectory.observation.shape[1] == 24
  obj_pos = trajectory.observation[:, :10]
  goal_pos = trajectory.observation[:, 12:-2]
  reached_goal = tf.less_equal(
      tf.norm(obj_pos - goal_pos, axis=1),
      tf.constant([push_success_threshold]))
  if episodic:
    return tf.logical_and(trajectory.is_last(), reached_goal)
  else:
    return reached_goal


def playpen_reduced_success(trajectory,
                            episodic=False,
                            push_success_threshold=0.2):
  # works only when given goal observation
  assert trajectory.observation.shape[1] == 12
  obj_pos = trajectory.observation[:, :4]
  goal_pos = trajectory.observation[:, 6:-2]
  reached_goal = tf.less_equal(
      tf.norm(obj_pos - goal_pos, axis=1),
      tf.constant([push_success_threshold]))
  if episodic:
    return tf.logical_and(trajectory.is_last(), reached_goal)
  else:
    return reached_goal


def kitchen_microwave_success(trajectory,
                              episodic=False,
                              push_success_threshold=0.05):
  # works only when given goal observation
  assert trajectory.observation.shape[1] == 24
  # currently hardcoded
  # microwave_pos = trajectory.observation[:, 22:23]
  microwave_pos = trajectory.observation[:, 13:14]
  # microwave_goal_pos = trajectory.observation[:, 37 + 22:37 + 23]
  microwave_goal_pos = -0.7
  reached_goal = tf.less_equal(
      tf.norm(microwave_pos - microwave_goal_pos, axis=1),
      tf.constant([push_success_threshold]))
  if episodic:
    return tf.logical_and(trajectory.is_last(), reached_goal)
  else:
    return reached_goal


def sawyer_push_success(trajectory,
                        episodic=False,
                        push_success_threshold=0.07):
  # works only when given goal observation
  assert trajectory.observation.shape[1] == 9
  # correction because we are providing deltas in state
  hand_xy = trajectory.observation[:, :2]
  obj_xy = trajectory.observation[:, 3:5] + hand_xy
  goal_xy = trajectory.observation[:, 6:8] + obj_xy
  reached_goal = tf.less_equal(
      tf.norm(obj_xy - goal_xy, axis=1), tf.constant([push_success_threshold]))
  if episodic:
    return tf.logical_and(trajectory.is_last(), reached_goal)
  else:
    return reached_goal


def sawyer_door_success(trajectory, episodic=False, push_success_threshold=0.2):
  # works only when given goal observation
  assert trajectory.observation.shape[1] == 12
  obj_xy = trajectory.observation[:, :6]
  goal_xy = trajectory.observation[:, 6:]
  reached_goal = tf.less_equal(
      tf.norm(obj_xy - goal_xy, axis=1), tf.constant([push_success_threshold]))
  if episodic:
    return tf.logical_and(trajectory.is_last(), reached_goal)
  else:
    return reached_goal


def pusher2d_simple_success(trajectory,
                            episodic=False,
                            push_success_threshold=0.07):
  obj_xy = trajectory.observation[:, 3:5]
  goal_xy = trajectory.observation[:, -2:]
  reached_goal = tf.less_equal(
      tf.norm(obj_xy - goal_xy, axis=1),
      tf.cast(tf.constant([push_success_threshold]), dtype=tf.float32))
  if episodic:
    return tf.logical_and(trajectory.is_last(), reached_goal)
  else:
    return reached_goal


def point_mass_success(trajectory, episodic=False, push_success_threshold=1.0):
  # correction because we are providing deltas in state
  obj_xy = trajectory.observation[:, :2]
  goal_xy = trajectory.observation[:, 6:8]
  reached_goal = tf.less_equal(
      tf.norm(obj_xy - goal_xy, axis=1),
      tf.cast(tf.constant([push_success_threshold]), dtype=tf.float32))
  if episodic:
    return tf.logical_and(trajectory.is_last(), reached_goal)
  else:
    return reached_goal
