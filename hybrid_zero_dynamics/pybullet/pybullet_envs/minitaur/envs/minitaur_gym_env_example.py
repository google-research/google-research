"""An example to run of the minitaur gym environment with sine gaits.

"""

import csv
import math
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
print("parentdir=",parentdir)
os.sys.path.insert(0,parentdir)

import argparse
import numpy as np
import tensorflow as tf
from pybullet_envs.minitaur.envs import minitaur_gym_env
import time


#FLAGS = flags.FLAGS
#flags.DEFINE_enum(
#    "example_name", "sine", ["sine", "reset", "stand", "overheat"],
#    "The name of the example: sine, reset, stand, or overheat.")
#flags.DEFINE_string("output_filename", None, "The name of the output CSV file."
#                    "Each line in the CSV file will contain the action, the "
#                    "motor position, speed and torques.")
#flags.DEFINE_string("log_path", None, "The directory to write the log file.")


def WriteToCSV(filename, actions_and_observations):
  """Write simulation data to file.

  Save actions and observed angles, angular velocities and torques for data
  analysis.

  Args:
    filename: The file to write. Can be locally or on CNS.
    actions_and_observations: the interested simulation quantities to save.
  """
  with tf.gfile.Open(filename, "wb") as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=",")
    for row in actions_and_observations:
      csv_writer.writerow(row)


def ResetPoseExample(log_path=None):
  """An example that the minitaur stands still using the reset pose."""
  steps = 10000
  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
      render=True,
      leg_model_enabled=False,
      motor_velocity_limit=np.inf,
      pd_control_enabled=True,
      accurate_motor_model_enabled=True,
      motor_overheat_protection=True,
      hard_reset=False,
      log_path=log_path)
  action = [math.pi / 2] * 8
  for _ in range(steps):
    _, _, done, _ = environment.step(action)
    time.sleep(1./100.)
    if done:
      break


def MotorOverheatExample(log_path=None):
  """An example of minitaur motor overheat protection is triggered.

  The minitaur is leaning forward and the motors are getting obove threshold
  torques. The overheat protection will be triggered in ~1 sec.

  Args:
    log_path: The directory that the log files are written to. If log_path is
      None, no logs will be written.
  """

  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
      render=True,
      leg_model_enabled=False,
      motor_velocity_limit=np.inf,
      motor_overheat_protection=True,
      accurate_motor_model_enabled=True,
      motor_kp=1.20,
      motor_kd=0.00,
      on_rack=False,
      log_path=log_path)

  action = [2.0] * 8
  for i in range(8):
    action[i] = 2.0 - 0.5 * (-1 if i % 2 == 0 else 1) * (-1 if i < 4 else 1)

  steps = 500
  actions_and_observations = []
  for step_counter in range(steps):
    # Matches the internal timestep.
    time_step = 0.01
    t = step_counter * time_step
    current_row = [t]
    current_row.extend(action)

    observation, _, _, _ = environment.step(action)
    current_row.extend(observation.tolist())
    actions_and_observations.append(current_row)
    time.sleep(1./100.)

  if FLAGS.output_filename is not None:
    WriteToCSV(FLAGS.output_filename, actions_and_observations)


def SineStandExample(log_path=None):
  """An example of minitaur standing and squatting on the floor.

  To validate the accurate motor model we command the robot and sit and stand up
  periodically in both simulation and experiment. We compare the measured motor
  trajectories, torques and gains. The results are at:
    https://colab.corp.google.com/v2/notebook#fileId=0BxTIAnWh1hb_ZnkyYWtNQ1RYdkU&scrollTo=ZGFMl84kKqRx

  Args:
    log_path: The directory that the log files are written to. If log_path is
      None, no logs will be written.
  """
  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.RAINBOW_DASH_V0_URDF_VERSION,
      render=True,
      leg_model_enabled=False,
      motor_velocity_limit=np.inf,
      motor_overheat_protection=True,
      accurate_motor_model_enabled=True,
      motor_kp=1.20,
      motor_kd=0.02,
      on_rack=False,
      log_path=log_path)
  steps = 1000
  amplitude = 0.5
  speed = 3

  actions_and_observations = []

  for step_counter in range(steps):
    # Matches the internal timestep.
    time_step = 0.01
    t = step_counter * time_step
    current_row = [t]

    action = [math.sin(speed * t) * amplitude + math.pi / 2] * 8
    current_row.extend(action)

    observation, _, _, _ = environment.step(action)
    current_row.extend(observation.tolist())
    actions_and_observations.append(current_row)
    time.sleep(1./100.)

  if FLAGS.output_filename is not None:
    WriteToCSV(FLAGS.output_filename, actions_and_observations)


def SinePolicyExample(log_path=None):
  """An example of minitaur walking with a sine gait.

  Args:
    log_path: The directory that the log files are written to. If log_path is
      None, no logs will be written.
  """
  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
      render=True,
      motor_velocity_limit=np.inf,
      pd_control_enabled=True,
      hard_reset=False,
      on_rack=False,
      log_path=log_path)
  sum_reward = 0
  steps = 20000
  amplitude_1_bound = 0.5
  amplitude_2_bound = 0.5
  speed = 40

  for step_counter in range(steps):
    time_step = 0.01
    t = step_counter * time_step

    amplitude1 = amplitude_1_bound
    amplitude2 = amplitude_2_bound
    steering_amplitude = 0
    if t < 10:
      steering_amplitude = 0.5
    elif t < 20:
      steering_amplitude = -0.5
    else:
      steering_amplitude = 0

    # Applying asymmetrical sine gaits to different legs can steer the minitaur.
    a1 = math.sin(t * speed) * (amplitude1 + steering_amplitude)
    a2 = math.sin(t * speed + math.pi) * (amplitude1 - steering_amplitude)
    a3 = math.sin(t * speed) * amplitude2
    a4 = math.sin(t * speed + math.pi) * amplitude2
    action = [a1, a2, a2, a1, a3, a4, a4, a3]
    _, reward, done, _ = environment.step(action)
    time.sleep(1./100.)

    sum_reward += reward
    if done:
      tf.logging.info("Return is {}".format(sum_reward))
      environment.reset()




def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--env', help='environment ID (0==sine, 1==stand, 2=reset, 3=overheat)',type=int,  default=2)
  args = parser.parse_args()
  print("--env=" + str(args.env))
    
  if (args.env == 0):
    SinePolicyExample()
  if (args.env == 1):
    SineStandExample()
  if (args.env == 2):
    ResetPoseExample()
  if (args.env == 3):
    MotorOverheatExample()

if __name__ == '__main__':
    main()

