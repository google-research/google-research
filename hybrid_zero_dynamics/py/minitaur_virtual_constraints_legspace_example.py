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

"""An example to run of the minitaur gym with virtual-constraints-based gaits.

The bezierdata_legspace file has all the gait coefficients data.

To run a single gait, use the hzd_policy() function. (set --env==0)
To run multiple gaits succesively, use gait_library_policy() function. (set
--env==1 from cmd line)

The main file runs hzd_policy() by default.

Extras:
                >> To disable logging and plotting, just don't provide a logging
                file path. (log_file = None).
"""

import argparse
import csv
import glob
import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from pybullet_envs.minitaur.envs import minitaur_gym_env
import scipy
import scipy.special
import tensorflow.compat.v1 as tf

# Define indices
MOTOR_DIRECTION = [-1, -1, -1, -1, 1, 1, 1, 1]
LEGSPACE_DIRECTION = [-1, -1, 1, 1, -1, -1, 1, 1]
NUM_MOTORS = 8
NUM_LEGS = 4
MOTOR_LABEL = ['FLL', 'FLR', 'BLL', 'BLR', 'FRL', 'FRR', 'BRL', 'BRR']
TIME_ID = 0
DESMOTORANG_ID_BEGIN = 1
DESMOTORANG_ID_END = NUM_MOTORS + DESMOTORANG_ID_BEGIN
DESMOTORVEL_ID_BEGIN = DESMOTORANG_ID_END
DESMOTORVEL_ID_END = NUM_MOTORS + DESMOTORVEL_ID_BEGIN
ACTMOTORANG_ID_BEGIN = DESMOTORVEL_ID_END
ACTMOTORANG_ID_END = NUM_MOTORS + ACTMOTORANG_ID_BEGIN
ACTMOTORVEL_ID_BEGIN = ACTMOTORANG_ID_END
ACTMOTORVEL_ID_END = NUM_MOTORS + ACTMOTORVEL_ID_BEGIN
ACTMOTORTORQ_ID_BEGIN = ACTMOTORVEL_ID_END
ACTMOTORTORQ_ID_END = NUM_MOTORS + ACTMOTORTORQ_ID_BEGIN


def motor_angles_to_leg_pose(motor_angles):
  leg_pose = np.zeros(NUM_MOTORS)
  for i in range(NUM_LEGS):
    leg_pose[i] = 0.5 * (-1)**(i // 2) * (
        motor_angles[2 * i + 1] - motor_angles[2 * i])
    leg_pose[NUM_LEGS + i] = 0.5 * (
        motor_angles[2 * i] + motor_angles[2 * i + 1])
  return leg_pose


def leg_pose_to_motor_angles(leg_pose):
  motor_pose = np.zeros(NUM_MOTORS)
  for i in range(NUM_LEGS):
    motor_pose[2 * i] = leg_pose[NUM_LEGS + i] + (-1)**(i // 2) * leg_pose[i]
    motor_pose[2 * i + 1] = (
        leg_pose[NUM_LEGS + i] - (-1)**(i // 2) * leg_pose[i])
  return motor_pose


def normalize(t_cur, t_max, t_min):
  """Normlize given variable w.r.t max and min values.

  Args:
    t_cur: variable of interest.
    t_max: max possible value.
    t_min: min possible value.

  Returns:
    The normalized time.
  """
  tau = (t_cur - t_min) / (t_max - t_min)
  return tau


def bezier_interpolation(control_points, tau):
  """Bezier Interpolation to determine desired motor position at time t.

  tau range must be in the range [0, 1].
  degree of the bezier curve is len(control_points) - 1

  Args:
    control_points: Bezier curve generations parameters.
    tau: normalized time value.

  Returns:
    The interpolated position and velocity at normalized time tau.

  """

  degree = len(control_points) - 1

  final_val = 0
  for idx in range(degree + 1):
    num_of_combs = scipy.special.comb(degree, idx)
    final_val = final_val + (
        num_of_combs * ((1 - tau)**(degree - idx)) *
        (tau**idx) * control_points[idx])

  final_velval = 0
  k = degree - 1
  for idx in range(k + 1):
    num_of_combs = scipy.special.comb(k, idx)
    final_velval = final_velval + (
        degree * num_of_combs * ((1 - tau)**(k - idx)) * (tau**idx) *
        (control_points[idx + 1] - control_points[idx]))

  return final_val, final_velval


def load_gait_txt(filename, num_phases=2):
  """Inputs MatLab generated gait data from a text file.

  Slices data matrix to bezier control points (vircons matrix) and phase
  durations array (phasevars)

  Args:
    filename: matlab generated gait data filename
    num_phases: number of hybrid phases used in the gait generation.

  Returns:
    The control point matrix and the phase duration array.
  """

  data = np.loadtxt(filename, delimiter=',', dtype='float32')

  data_shape = data.shape

  num_coeffs = data_shape[1]

  # m*n rows of bezier coefficients followed by m rows of phase duration info
  # m is num of phases
  # n is num of outputs
  num_outputs = NUM_MOTORS  #  one outputs for each motor dof.

  vircons = np.zeros([num_phases, num_outputs, num_coeffs])
  phasevars = np.zeros([num_phases, 1])

  for phase in range(num_phases):
    vircons[phase, :, :] = data[phase * num_outputs:(phase + 1) *
                                num_outputs, :num_coeffs]
    phasevars[phase] = data[num_outputs * num_phases + phase, 0]

  return vircons, phasevars


def write_csv(filename, actions_and_observations):
  """Write simulation data to file.

  Save actions and observed angles, angular velocities and torques for
  data
  analysis.

  Args:
    filename: The file to write. Can be locally or on CNS.
    actions_and_observations: the interested simulation quantities to save.
  """
  with tf.gfile.Open(filename, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    for row in actions_and_observations:
      csv_writer.writerow(row)


def read_csv(filename, print_data=False):
  """Read simulation data for plotting.

  Args:
    filename: The saved data filename.
    print_data: Flag to print the data (optional - for sanity check). Default is
      False.

  Returns:
    A numpy array of the simulation curves.
  """
  data = []
  with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      data.append(row)
      if print_data:
        logging.info(row)
  csvfile.close()
  return np.array(data)


def gait_library_policy(gaits,
                        log_file=None,
                        steps_per_gait=4,
                        feedforward=False):
  """An example of minitaur locomoting using hzd gai libraries.

  Args:
    gaits: A dictionary of fileids indexed by steplength value.
    log_file: The directory into which log files are written. If log_path is
      None, no data logging takes place.
    steps_per_gait: no of steps the robot will walk at a particular steplength
      before switching to the next one.
    feedforward: boolen to use augmented bezier coefficients. The original
      bezier coeffcients capture only the optimal leg trajectory information
      (without corresponding torque info). Augmented bezier coefficients are
      generated by combining the optimal torque information into the leg
      trajectory info. Using this improves tracking performance.
  """

  ## number of distinct phases in the gait cycle
  num_phases = 2
  for steplen, steplenfile in gaits.items():
    ## Load Bezier Parameters and Phase Timings Data
    vircons, phasevars = load_gait_txt(steplenfile, num_phases=num_phases)
    gaits[steplen] = [vircons, phasevars]
  steplens = list(gaits.keys())

  time_step = 0.01
  sum_reward = 0
  num_outputs = len(gaits['0.00'][0][0])

  init_motor_pos = []
  for output in range(num_outputs):
    coeffs = gaits['0.00'][0][0, output, :]
    output_val_at_0, _ = bezier_interpolation(coeffs, 0)
    init_motor_pos.append(float(output_val_at_0))

  if feedforward:
    init_motor_pos = list(np.multiply(MOTOR_DIRECTION, init_motor_pos))
  else:
    init_motor_pos = list(np.multiply(LEGSPACE_DIRECTION, init_motor_pos))
    init_motor_pos = leg_pose_to_motor_angles(init_motor_pos)

  init_motor_pos = np.array(init_motor_pos, dtype='float32')
  logging.info(init_motor_pos * (180 / math.pi))

  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.RAINBOW_DASH_V0_URDF_VERSION,
      render=True,
      motor_velocity_limit=np.inf,
      accurate_motor_model_enabled=True,
      remove_default_joint_damping=False,
      pd_control_enabled=True,
      leg_model_enabled=False,
      hard_reset=False,
      on_rack=False,
      motor_kp=1,
      motor_kd=0.005,
      control_latency=0.001,
      pd_latency=0.001,
      log_path=None,
      env_randomizer=None)

  environment.reset(initial_motor_angles=init_motor_pos, reset_duration=1.0)

  time.sleep(3.)

  cycle_times = [float(sum(gaits[steplen][1][:])) for steplen in gaits]

  cycle_timesum = np.cumsum(cycle_times)
  gait_id = [i for i in range(len(steplens))]

  steps = int((steps_per_gait * sum(cycle_times)) / time_step)

  actions_and_observations = []
  cycles = 0
  last_phase = 0
  for step_counter in range(steps):

    t = step_counter * time_step
    current_row = [t]

    cycle_time = cycle_times[cycles]

    t_in_cycle = (t -
                  cycle_timesum[cycles]) % (cycle_time) if cycles > 0 else t % (
                      cycle_time)

    if 0 <= t_in_cycle <= float(gaits[steplens[gait_id[cycles]]][1][0]):
      tau = normalize(t_in_cycle, float(gaits[steplens[gait_id[cycles]]][1][0]),
                      0)
      phase = 0
      if last_phase == 1:
        logging.info('Gait Cycle Time was %f', cycle_time)
        cycles += 1
        if cycles == len(steplens):
          break
      last_phase = phase
    elif float(
        gaits[steplens[gait_id[cycles]]][1][0]) <= t_in_cycle <= cycle_time:
      tau = normalize(t_in_cycle, cycle_time,
                      float(gaits[steplens[gait_id[cycles]]][1][0]))
      phase = 1
      last_phase = 1
    else:
      ValueError('Tau must lie between 0 and ' + str(cycle_time))

    action = []
    action_vel = []
    for output in range(num_outputs):
      coeffs = gaits[steplens[gait_id[cycles]]][0][phase, output, :]
      output_val_at_tau, output_vel_at_tau = bezier_interpolation(coeffs, tau)
      action.append(float(output_val_at_tau))
      action_vel.append(float(output_vel_at_tau))

    if feedforward:
      action = list(np.multiply(MOTOR_DIRECTION, action))
      action_vel = list(np.multiply(MOTOR_DIRECTION, action_vel))
    else:
      action = leg_pose_to_motor_angles(
          list(np.multiply(LEGSPACE_DIRECTION, action)))
      action_vel = leg_pose_to_motor_angles(
          list(np.multiply(LEGSPACE_DIRECTION, action_vel)))

    current_row.extend(action)
    current_row.extend(action_vel)

    observation, reward, done, _ = environment.step(action)
    observation[16:24] = [
        observation[16 + i] * MOTOR_DIRECTION[i] for i in range(NUM_MOTORS)
    ]
    current_row.extend(observation.tolist())
    base_angles = environment.minitaur.GetBaseRollPitchYaw()
    current_row.extend(base_angles.tolist())
    actions_and_observations.append(current_row)

    sum_reward += reward

    if done or (step_counter == (steps - 1)) or (cycles == steps_per_gait *
                                                 len(steplens)):
      logging.info('End of Sim')
      if log_file is not None:
        write_csv(log_file, actions_and_observations)


def hzd_policy(vircons=None,
               phasevars=None,
               log_file=None,
               sim_cycles=4,
               feedforward=False):
  """An example of minitaur locomoting with a virtual constraints.

  Args:
    vircons: A numpy array with bezier coefficients to generate optimal leg
      trajectories.
    phasevars: A numpy array of end time values for a stepping trajectory.
    log_file: The directory into which log files are written. If log_path is
      None, no data logging takes place.
    sim_cycles: number of trotting steps to simulate.
    feedforward: Boolean to use augmented bezier coefficients. The original
      bezier coeffcients capture only the optimal leg trajectory information
      (without corresponding torque info). Augmented bezier coefficients are
      generated by combining the optimal torque information into the leg
      trajectory info. Using this improves tracking performance.
  """

  time_step = 0.01
  sum_reward = 0
  num_outputs = len(vircons[0])

  init_motor_pos = []
  for output in range(num_outputs):
    coeffs = vircons[0, output, :]
    output_val_at_0, _ = bezier_interpolation(coeffs, 0)
    init_motor_pos.append(float(output_val_at_0))

  if feedforward:
    init_motor_pos = list(np.multiply(MOTOR_DIRECTION, init_motor_pos))
  else:
    init_motor_pos = list(np.multiply(LEGSPACE_DIRECTION, init_motor_pos))
    init_motor_pos = leg_pose_to_motor_angles(init_motor_pos)

  init_motor_pos = np.array(init_motor_pos, dtype='float32')

  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.RAINBOW_DASH_V0_URDF_VERSION,
      render=True,
      motor_velocity_limit=np.inf,
      accurate_motor_model_enabled=True,
      remove_default_joint_damping=False,
      pd_control_enabled=True,
      leg_model_enabled=False,
      hard_reset=False,
      on_rack=False,
      motor_kp=1.0,
      motor_kd=0.005,
      control_latency=0.001,
      pd_latency=0.001,
      log_path=None,
      env_randomizer=None)

  environment.reset(initial_motor_angles=init_motor_pos, reset_duration=1.0)

  time.sleep(3.)

  cycle_time = float(sum(phasevars))
  logging.info('Gait Cycle Time is %f', cycle_time)
  steps = int((sim_cycles * cycle_time) / time_step)
  actions_and_observations = []
  for step_counter in range(steps):

    t = step_counter * time_step
    current_row = [t]

    t_in_cycle = t % cycle_time
    if 0 <= t_in_cycle <= float(phasevars[0]):
      tau = normalize(t_in_cycle, float(phasevars[0]), 0)
      phase = 0
    elif float(phasevars[0]) <= t_in_cycle <= cycle_time:
      tau = normalize(t_in_cycle, cycle_time, float(phasevars[0]))
      phase = 1
    else:
      ValueError('Tau must lie between 0 and ' + str(cycle_time))

    action = []
    action_vel = []
    for output in range(num_outputs):
      coeffs = vircons[phase, output, :]
      output_val_at_tau, output_vel_at_tau = bezier_interpolation(coeffs, tau)
      action.append(float(output_val_at_tau))
      action_vel.append(float(output_vel_at_tau))

    if feedforward:
      action = list(np.multiply(MOTOR_DIRECTION, action))
      action_vel = list(np.multiply(MOTOR_DIRECTION, action_vel))
    else:
      action = leg_pose_to_motor_angles(
          list(np.multiply(LEGSPACE_DIRECTION, action)))
      action_vel = leg_pose_to_motor_angles(
          list(np.multiply(LEGSPACE_DIRECTION, action_vel)))

    current_row.extend(action)
    current_row.extend(action_vel)

    observation, reward, done, _ = environment.step(action)
    observation[16:24] = [
        observation[16 + i] * MOTOR_DIRECTION[i] for i in range(NUM_MOTORS)
    ]
    current_row.extend(observation.tolist())
    base_angles = environment.minitaur.GetBaseRollPitchYaw()
    current_row.extend(base_angles.tolist())
    actions_and_observations.append(current_row)

    sum_reward += reward

    if done or (step_counter == (steps - 1)):
      logging.info('End of Sim')
      if log_file is not None:
        write_csv(log_file, actions_and_observations)
      tf.logging.info('Return is {}'.format(sum_reward))
      environment.reset()


def plot_sim_logs(data_filename, savedata_filename=None):
  """Plot the simulation logs."""
  ## Format Data
  data = read_csv(data_filename)
  data = data.astype('float32')
  data = data.transpose()

  ## Assign filename for saving data
  if savedata_filename is not None:
    filepath, _ = os.path.split(data_filename)
    filename = savedata_filename
    fileformat = '.png'

    angleplots_filename = os.path.join(filepath,
                                       (filename + 'angles' + fileformat))
    velplots_filename = os.path.join(filepath, (filename + 'vels' + fileformat))
    torqueplots_filename = os.path.join(filepath,
                                        (filename + 'torques' + fileformat))

  ## Slicing Data
  unit = 'degree'

  sim_time = data[TIME_ID]
  if unit == 'degree':
    motor_angles_des = data[DESMOTORANG_ID_BEGIN:DESMOTORANG_ID_END] * (180 /
                                                                        math.pi)
    motor_angles_act = data[ACTMOTORANG_ID_BEGIN:ACTMOTORANG_ID_END] * (180 /
                                                                        math.pi)
  else:
    motor_angles_des = data[DESMOTORANG_ID_BEGIN:DESMOTORANG_ID_END]
    motor_angles_act = data[ACTMOTORANG_ID_BEGIN:ACTMOTORANG_ID_END]
  motor_vels_des = data[DESMOTORVEL_ID_BEGIN:DESMOTORVEL_ID_END]
  motor_vels_act = data[ACTMOTORVEL_ID_BEGIN:ACTMOTORVEL_ID_END]
  motor_torques_act = data[ACTMOTORTORQ_ID_BEGIN:ACTMOTORTORQ_ID_END]

  # Plotting Motor Angles
  plt.figure()
  fig, axs = plt.subplots(4, 2, figsize=(12, 12))
  for motor in range(NUM_MOTORS):

    if motor < (NUM_MOTORS / 2):
      column = 0
    else:
      column = 1

    row = int(motor % (NUM_MOTORS / 2))

    axs[row, column].plot(
        sim_time, motor_angles_des[motor], linewidth=1.5, label='desired')
    axs[row, column].plot(
        sim_time, motor_angles_act[motor], linewidth=1.5, label='actual')
    axs[row, column].set_title(
        MOTOR_LABEL[motor] + ' Motor Angle Plots', fontsize=12)
    axs[row, column].set_xlabel('Time (s)', fontsize=12)
    if unit == 'degree':
      axs[row, column].set_ylabel('Motor Angle (deg)', fontsize=12)
    else:
      axs[row, column].set_ylabel('Motor Angle (rad)', fontsize=12)
    axs[row, column].legend(fontsize=10,)
    axs[row, column].grid(True)

  fig.tight_layout()
  plt.show()
  if savedata_filename is not None:
    fig.savefig(angleplots_filename, dpi=600)

  # Plotting Motor Velocities
  plt.figure()
  fig, axs = plt.subplots(4, 2, figsize=(12, 12))
  for motor in range(NUM_MOTORS):

    if motor < (NUM_MOTORS / 2):
      column = 0
    else:
      column = 1

    row = int(motor % (NUM_MOTORS / 2))

    axs[row, column].plot(
        sim_time, motor_vels_des[motor], linewidth=1.5, label='desired')
    axs[row, column].plot(
        sim_time, motor_vels_act[motor], linewidth=1.5, label='actual')
    axs[row, column].set_title(
        MOTOR_LABEL[motor] + ' Motor Velocity Plot', fontsize=12)
    axs[row, column].set_xlabel('Time (s)', fontsize=12)
    axs[row, column].set_ylabel('Motor Vel (rad/s)', fontsize=12)
    axs[row, column].legend(fontsize=10,)
    axs[row, column].grid(True)

  fig.tight_layout()
  plt.show()
  if savedata_filename is not None:
    fig.savefig(velplots_filename, dpi=600)

  plt.figure()
  fig, axs = plt.subplots(4, 2, figsize=(12, 12))
  for motor in range(NUM_MOTORS):

    if motor < (NUM_MOTORS / 2):
      column = 0
    else:
      column = 1

    row = int(motor % (NUM_MOTORS / 2))

    axs[row, column].plot(
        sim_time, motor_torques_act[motor], linewidth=1.5, label='actual')
    axs[row, column].set_title(
        MOTOR_LABEL[motor] + ' Motor Torque Plot', fontsize=12)
    axs[row, column].set_xlabel('Time (s)', fontsize=12)
    axs[row, column].set_ylabel('Motor Torque (Nm)', fontsize=12)
    axs[row, column].grid(True)

  fig.tight_layout()
  plt.show()
  if savedata_filename is not None:
    fig.savefig(torqueplots_filename, dpi=600)


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--env',
      help='environment ID (0==single_gait, 1==gait_library)',
      type=int,
      default=0)
  args = parser.parse_args()
  logging.info('--env= %s', str(args.env))

  ## reading optimal gait data obtained from frost
  bezierdata_dir = 'bezierdata_legspace'

  # file_ids are not sorted by default.
  file_ids = glob.glob(os.path.join(bezierdata_dir, 'modbez*.txt'))
  steplens = ['0.00', '0.05', '0.10', '0.15', '0.20']
  gaits = dict(zip(steplens, file_ids))

  if args.env == 0:
    ## number of distinct phases in the gait cycle
    num_phases = 2
    ## Load Bezier Parameters and Phase Timings Data
    vircons, phasevars = load_gait_txt(gaits['0.05'], num_phases=num_phases)
    hzd_policy(
        vircons=vircons,
        phasevars=phasevars,
        sim_cycles=15,
        feedforward=os.path.split(file_ids[0])[1].startswith('mod'),
        log_file='test.csv')
    plot_sim_logs('test.csv')

  elif args.env == 1:
    gait_library_policy(
        gaits,
        steps_per_gait=5,
        feedforward=os.path.split(file_ids[0])[1].startswith('mod'),
        log_file='test.csv')
    time.sleep(3.)
    plot_sim_logs('test.csv')
  else:
    pass

if __name__ == '__main__':
  main()
