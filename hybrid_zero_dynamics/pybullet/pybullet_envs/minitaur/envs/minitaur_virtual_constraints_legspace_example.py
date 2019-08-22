# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:21:21 2019

@author: Avinash Siravuru
"""

"""An example to run of the minitaur gym environment with virtual-constraints-based gaits.
The bezierdata_legspace file has all the gait coefficients data.

To test a single gait, use the HZDPolicy() function.
To run test multiple gaits one after another, use GaitLibraryPolicy() function.

The main file runs GaitLibraryPolicy() by default.

"""
import pdb
import csv
import math
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
print("parentdir=",parentdir)
os.sys.path.insert(0,parentdir)
import glob
import argparse
import numpy as np
import scipy as sp
import tensorflow as tf
from pybullet_envs.minitaur.envs import minitaur_gym_env
from pybullet_envs.minitaur.envs.env_randomizers import minitaur_env_randomizer, minitaur_env_randomizer_from_config, minitaur_env_randomizer_config
import time

#import pickle
from hyperopt import fmin, tpe, hp

import matplotlib.pyplot as plt

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
  """
  tau = (t_cur - t_min) / (t_max - t_min)
  return tau    

    
def bezier_interpolation(control_points, tau):
  """Bezier Interpolation to determine desired motor position at time t.
  
  tau range must be in the range [0, 1].
  degree of the bezier curve is len(control_points) - 1
  
  Args:
    tau: normalized time value.
    control_points: Bezier curve generations parameters.
  """
  
  degree = len(control_points) - 1
  
  final_val = 0
  for idx in range(degree + 1):
    num_of_combs = sp.special.comb(degree, idx)
    final_val = final_val + ( num_of_combs * ( (1 - tau) ** (degree - idx) ) * ( tau ** idx) * control_points[idx] )
  
  final_velval = 0  
  k = degree - 1
  for idx in range(k + 1):
    num_of_combs = sp.special.comb(k, idx)
    final_velval = final_velval + ( degree * num_of_combs * ( (1 - tau) ** (k - idx) ) * ( tau ** idx) * (control_points[idx+1] - control_points[idx] ) )
    
  return final_val, final_velval


def load_gait_txt(filename, num_phases = 2):
  """ Inputs matab generated gait data from a text file.
  
  Slices data matrix to bezier control points (vircons matrix) and 
      phase durations array (phasevars)
      
  Args:
    filename: matlab generated gait data filename
    num_phases: number of hybrid phases used in the gait generation.
  """
    
  data = np.loadtxt(filename, delimiter = ',', dtype = 'float32')
 
  data_shape = data.shape
  
  num_coeffs = data_shape[1]
   
  ## m*n rows of bezier coefficients followed by m rows of phase duration info
  ## m is num of phases
  ## n is num of outputs
  num_outputs = 8#int( (data_shape[0] - num_phases ) / num_phases )
  
  vircons = np.zeros([num_phases, num_outputs, num_coeffs])
  phasevars = np.zeros([num_phases, 1])
  
  for phase in range(num_phases):
    vircons[phase, :, :] = data[phase*num_outputs:(phase+1)*num_outputs,
                                :num_coeffs]
    phasevars[phase] = data[num_outputs*num_phases + phase, 0]
  
  return vircons, phasevars


def write_csv(filename, actions_and_observations):
  """Write simulation data to file.

  Save actions and observed angles, angular velocities and torques for data
  analysis.

  Args:
    filename: The file to write. Can be locally or on CNS.
    actions_and_observations: the interested simulation quantities to save.
  """
  with tf.gfile.Open(filename, "w") as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=",")
    for row in actions_and_observations:
      csv_writer.writerow(row)
      
      
def read_csv(filename, print_data = False):
  """Read simulation data for plotting.
  
  Args:
    filename: The saved data filename.
    print_data: Flag to print the data (optional - for sanity check). Default is False.
  """
  data = []
  with open(filename, 'r') as csvFile:
      reader = csv.reader(csvFile)
      for row in reader:
          data.append(row)
          if print_data:
              print(row)
  csvFile.close()
  return np.array(data)      


def GaitLibraryPolicy(gaits, 
                      log_file = None,
                      steps_per_gait = 4,
                      randomize = False,
                      feedforward = False):
  """An example of minitaur locomoting using a 
     virtual constraints (a.k.a hybrid zero dynamics - based gait) gait library.

  Args:
    gaits: A dictionary of fileids indexed by steplength value.
    log_path: The directory into which log files are written. If log_path is None, no data logging takes place.
    steps_per_gait: no of steps the robot will walk at a particular steplength before switching to the next one. 
    randomize: boolean to invoke the RandomizerEnv() and randomizing the model parameters. Turn on for robustness testing.
    feedforward: boolen to use augmented bezier coefficients. 
                 The original bezier coeffcients capture only the optimal leg trajectory information (without corresponding torque info).
                 Augmented bezier coefficients are generated by combining the optimal torque information into the leg trajectory info.
                 Using this improves tracking performance. 
  """
  
  ## number of distinct phases in the gait cycle
  num_phases = 2
  for steplen, steplenfile in gaits.items():
    ## Load Bezier Parameters and Phase Timings Data  
    vircons, phasevars = load_gait_txt(steplenfile, num_phases = num_phases)
    gaits[steplen] = [vircons, phasevars]
  steplens = list(gaits.keys())

  time_step = 0.01
  sum_reward = 0
  num_outputs = len(gaits['0.00'][0][0])
  
  INIT_MOTOR_POS = []
  for output in range(num_outputs):
    coeffs = gaits['0.00'][0][0, output, :]
    output_val_at_0, _ = bezier_interpolation(coeffs, 0)
    INIT_MOTOR_POS.append(float(output_val_at_0))
  
  if feedforward:
    INIT_MOTOR_POS = list( np.multiply(MOTOR_DIRECTION, INIT_MOTOR_POS) )
  else:
    INIT_MOTOR_POS = list( np.multiply(LEGSPACE_DIRECTION, INIT_MOTOR_POS) )
    INIT_MOTOR_POS = leg_pose_to_motor_angles(INIT_MOTOR_POS)  
    
  
  INIT_MOTOR_POS = np.array(INIT_MOTOR_POS, dtype='float32')  
  print(INIT_MOTOR_POS*(180/math.pi))

  randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer()) if randomize else None


  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.RAINBOW_DASH_V0_URDF_VERSION,
      render= True,
      motor_velocity_limit= np.inf ,
      accurate_motor_model_enabled=True,
      remove_default_joint_damping=False,
      pd_control_enabled=True,
      leg_model_enabled=False,
      hard_reset=False,
      on_rack=False,
      motor_kp=1,
      motor_kd=0.005,
      motor_ki= 0,
      control_latency=0.001,
      pd_latency=0.001,
      log_path=None,
      env_randomizer=randomizer)
  
  #environment.minitaur.SetFootFriction(0.1)
#  environment.minitaur.SetFootRestitution(0.6)
  
  # if randomize:
  #   randomizer = minitaur_env_randomizer_from_config.MinitaurEnvRandomizerFromConfig()
  #   randomizer.randomize_env(environment)

  environment.reset(initial_motor_angles = INIT_MOTOR_POS, reset_duration = 1.0 )
 
  time.sleep(3.)
  
  cycle_times = [float(sum(gaits[steplen][1][:])) for steplen in gaits for step in range(steps_per_gait)] 

  cycle_timesum = np.cumsum(cycle_times)
  gait_id = [i for i in range(len(steplens)) for step in range(steps_per_gait)]
  
  steps = int( (steps_per_gait * sum(cycle_times) ) / time_step )  
  
  actions_and_observations = []  
  cycles = 0
  
  for step_counter in range(steps):
    
    t = step_counter * time_step
    current_row = [t]
    
    cycle_time = cycle_times[cycles]
        
    t_in_cycle = (t - cycle_timesum[cycles])%(cycle_time) if cycles > 0 else t % (cycle_time)

    if 0 <= t_in_cycle <= float(gaits[steplens[gait_id[cycles]]][1][0]):
      tau = normalize(t_in_cycle, float(gaits[steplens[gait_id[cycles]]][1][0]) , 0)
      phase = 0
#      print('Tau is {}s'.format(tau))
#      print(t, t_in_cycle, tau)
    elif float(gaits[steplens[gait_id[cycles]]][1][0]) <= t_in_cycle <= cycle_time:
      tau = normalize(t_in_cycle, cycle_time, float(gaits[steplens[gait_id[cycles]]][1][0]) )
      phase = 1
#      print('Tau is {}s'.format(tau))
#      print(t, t_in_cycle, tau)
      if math.isclose(tau, 1, rel_tol=5e-2):
        print('Gait Cycle Time was ' + str(cycle_time))
        cycles += 1
        if cycles == steps_per_gait * len(steplens):
          break
    else:
      ValueError('Tau must lie between 0 and ' + str(cycle_time))
#    pdb.set_trace()
    action = []
    action_vel = []
    for output in range(num_outputs):
      coeffs = gaits[steplens[gait_id[cycles]]][0][phase, output, :] 
      output_val_at_tau, output_vel_at_tau = bezier_interpolation(coeffs, tau)
      action.append(float(output_val_at_tau))
      action_vel.append(float(output_vel_at_tau))  
    
    if feedforward:
      action = list( np.multiply(MOTOR_DIRECTION, action) )
      action_vel = list( np.multiply(MOTOR_DIRECTION, action_vel) )
    else:
      action = leg_pose_to_motor_angles( list( np.multiply(LEGSPACE_DIRECTION, action) ) )
      action_vel = leg_pose_to_motor_angles( list( np.multiply(LEGSPACE_DIRECTION, action_vel) ) )
    
    current_row.extend(action)
    current_row.extend(action_vel)
#    print(action)    
    
    observation, reward, done, _ = environment.step(action)
    observation[16:24] = [observation[16+i]*environment.minitaur._motor_direction[i] for i in range(NUM_MOTORS)]
    current_row.extend(observation.tolist())
    base_angles = environment.minitaur.GetBaseRollPitchYaw()
    current_row.extend(base_angles.tolist())
    actions_and_observations.append(current_row)    
        
    time.sleep(1./50.)

    sum_reward += reward
    
  if done or (step_counter == (steps - 1) ) or (cycles == steps_per_gait * len(steplens) ):
    print('End of Sim')
    if log_file is not None:
      filename = os.path.join(currentdir, log_file)
      write_csv(filename, actions_and_observations)  
      
      data = np.array(actions_and_observations)
      data = data.transpose()
      motor_angles_des = data[DESMOTORANG_ID_BEGIN:DESMOTORANG_ID_END] 
      motor_angles_act = data[ACTMOTORANG_ID_BEGIN:ACTMOTORANG_ID_END]     
      motor_vels_des = data[DESMOTORVEL_ID_BEGIN:DESMOTORVEL_ID_END]
      motor_vels_act = data[ACTMOTORVEL_ID_BEGIN:ACTMOTORVEL_ID_END]
      ang_error = motor_angles_des - motor_angles_act
      vel_error = motor_vels_des - motor_vels_act
      base_angles_cum = data[-3:]
      diaglegs_sync_error = motor_angles_act[[0,1,2,3],:] - motor_angles_act[[7,6,5,4],:]
        #        error = ( 100*np.sqrt( np.sum(base_angles_cum[0]**2)/3 ) +
        #                         5*np.sqrt( ( np.sum(ang_error**2) + np.sum(vel_error**2) )/8 ) ) # / data.shape[1]
      error = ( 10*np.sqrt( np.sum(base_angles_cum[0]**2) ) +
         100*np.sqrt( np.sum(diaglegs_sync_error**2)/4 ) )
#      pdb.set_trace()
  

def HZDPolicy(vircons=None, 
                       phasevars=None,
                       log_file=None,
                       sim_cycles=4,
                       adapt_traj=False,
                       vircons_mod = None,
                       phasevars_mod = None):
  
  """An example of minitaur locomoting with a virtual constraints (a.k.a hybrid zero dynamics - based gait).

  Args:
    vircons: A numpy array with bezier coefficients to generate optimal leg trajectories.
    phasevars: A numpy array of end time values for a stepping trajectory. 
    sim_cycles: number of trotting steps to simulate.
    adapt_traj: boolean to modify the bezier coefficients with the optimal torque information. 
    vircons_mod: A numpy array of the same size as vircons that can be to used to augment the gait trajectories online, 
                 using Deep Reinforcement Learning (DRL) or Bayesian Optimization (BO).
    phasevars_mod: A numpy array of end times that can be used to augment the gait duration online using DRL or BO.
  """
  time_step = 0.01
  sum_reward = 0
  num_outputs = len(vircons[0])
  
  INIT_MOTOR_POS = []
  for output in range(num_outputs):
    coeffs = vircons[0, output, :]
    output_val_at_0, _ = bezier_interpolation(coeffs, 0)
    INIT_MOTOR_POS.append(float(output_val_at_0))
  INIT_MOTOR_POS = list( np.multiply(LEGSPACE_DIRECTION, INIT_MOTOR_POS) )
  INIT_MOTOR_POS = leg_pose_to_motor_angles(INIT_MOTOR_POS)  
    
  
  INIT_MOTOR_POS = np.array(INIT_MOTOR_POS, dtype='float32')  
  print(INIT_MOTOR_POS*(180/math.pi))

  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.RAINBOW_DASH_V0_URDF_VERSION,
      render= not adapt_traj,
      motor_velocity_limit= np.inf ,
      accurate_motor_model_enabled=True,
      remove_default_joint_damping=False,
      pd_control_enabled=True,
      leg_model_enabled=False,
      hard_reset=False,
      on_rack=False,
      motor_kp=2.0,
      motor_kd=0.02,
      motor_ki= 0.25, # 0, #
      control_latency=0.0,
      pd_latency=0.001,
      log_path=None)

  environment.reset(initial_motor_angles = INIT_MOTOR_POS, reset_duration = 1.0 )

  if not adapt_traj:
    time.sleep(10.)
  else:
    if phasevars_mod is not None:
      phasevars = phasevars + phasevars_mod*np.ones((2,1))
    if vircons_mod is not None:
      (l,b,h) = vircons.shape
      vircons_temp = np.zeros( (l, b, h) )
      vircons_temp[0,:,2] = vircons_mod
      vircons_temp[0,:,0:2] = np.zeros((b,2))
      vircons_temp[0,:,-3:-1] = np.zeros((b,2))
      vircons_temp[0,0:4] = -vircons_temp[0,0:4]
      vircons_temp[1,:] = -vircons_temp[0,[4,5,6,7,0,1,2,3]] 
      vircons = vircons + vircons_temp
    
  cycle_time = float(sum(phasevars)) 
  print('Gait Cycle Time is ' + str(cycle_time))
  steps = int( (sim_cycles * cycle_time) / time_step )  
  actions_and_observations = []  
  for step_counter in range(steps):

    t = step_counter * time_step
    current_row = [t]
    
    t_in_cycle = t % cycle_time
    if 0 <= t_in_cycle <= float(phasevars[0]):
      tau = normalize(t_in_cycle, float(phasevars[0]) , 0)
      phase = 0
    elif float(phasevars[0]) <= t_in_cycle <= cycle_time:
      tau = normalize(t_in_cycle, cycle_time, float(phasevars[0]) )
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
  
    action = leg_pose_to_motor_angles( list( np.multiply(LEGSPACE_DIRECTION, action) ) )
    action_vel = leg_pose_to_motor_angles( list( np.multiply(LEGSPACE_DIRECTION, action_vel) ) )
    current_row.extend(action)
    current_row.extend(action_vel)
#    print(action)    
    
    observation, reward, done, _ = environment.step(action)
    observation[16:24] = [observation[16+i]*environment.minitaur._motor_direction[i] for i in range(NUM_MOTORS)]
    current_row.extend(observation.tolist())
    base_angles = environment.minitaur.GetBaseRollPitchYaw()
    current_row.extend(base_angles.tolist())
    actions_and_observations.append(current_row)    
        
    #time.sleep(1./10.)

    sum_reward += reward
    
    if done or (step_counter == (steps - 1) ):
      print('End of Sim')
      if log_file is not None:
        filename = os.path.join(currentdir, log_file)
        write_csv(filename, actions_and_observations)  
      if adapt_traj:
        """ Define objective for BayesOpt: Minimizing L2 norm of angle and vel errors. """
        data = np.array(actions_and_observations)
        data = data.transpose()
        motor_angles_des = data[DESMOTORANG_ID_BEGIN:DESMOTORANG_ID_END] 
        motor_angles_act = data[ACTMOTORANG_ID_BEGIN:ACTMOTORANG_ID_END]     
        motor_vels_des = data[DESMOTORVEL_ID_BEGIN:DESMOTORVEL_ID_END]
        motor_vels_act = data[ACTMOTORVEL_ID_BEGIN:ACTMOTORVEL_ID_END]
        ang_error = motor_angles_des - motor_angles_act
        vel_error = motor_vels_des - motor_vels_act
        base_angles_cum = data[-3:]
        diaglegs_sync_error = motor_angles_act[[0,1,2,3],:] - motor_angles_act[[7,6,5,4],:]
#        pdb.set_trace()
#        error = ( 100*np.sqrt( np.sum(base_angles_cum[0]**2)/3 ) +
#                         5*np.sqrt( ( np.sum(ang_error**2) + np.sum(vel_error**2) )/8 ) ) # / data.shape[1]
        error = ( 10*np.sqrt( np.sum(base_angles_cum[0]**2) ) +
                         100*np.sqrt( np.sum(diaglegs_sync_error**2)/4 ) )  # / data.shape[1]
        return error
#      tf.logging.info("Return is {}".format(sum_reward))
#      environment.reset()


def logging_plots(data_filename, savedata_filename=None):
  ## Format Data
  data = read_csv(data_filename)
  data = data.astype('float32')
  data = data.transpose()
  
  ## Assign filename for saving data
  if savedata_filename is not None:
    filepath, _ = os.path.split(data_filename)
#    foldername = 'vircons_data'
#    filepath = os.path.join(current_dir, foldername)
    filename = savedata_filename #'inplace_trotting_wopd_onrack_'
    fileformat = '.png'
    
    angleplots_filename = os.path.join(filepath, (filename + 'angles' + fileformat)) 
    velplots_filename = os.path.join(filepath, (filename + 'vels' + fileformat) )
    torqueplots_filename = os.path.join(filepath, (filename + 'torques' + fileformat) )
      
  ## Slicing Data
  unit = 'degree'    
  
  time = data[TIME_ID]
  if unit == 'degree':
    motor_angles_des = data[DESMOTORANG_ID_BEGIN:DESMOTORANG_ID_END] * (180/math.pi)
    motor_angles_act = data[ACTMOTORANG_ID_BEGIN:ACTMOTORANG_ID_END] * (180/math.pi)
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
          
      row = int(motor % (NUM_MOTORS / 2) ) 
  
      axs[row, column].plot(time, motor_angles_des[motor], linewidth=1.5, label="desired")
      axs[row, column].plot(time, motor_angles_act[motor], linewidth=1.5, label="actual")
      axs[row, column].set_title( MOTOR_LABEL[motor] + ' Motor Angle Plots', fontsize=12)
      axs[row, column].set_xlabel('Time (s)', fontsize=12)
      if unit == 'degree':
        axs[row, column].set_ylabel('Motor Angle (deg)', fontsize=12)
      else:
        axs[row, column].set_ylabel('Motor Angle (rad)', fontsize=12)
      axs[row, column].legend(fontsize=10, )
      axs[row, column].grid(True)
  
#  fig.align_labels()
  fig.tight_layout()
  plt.show()
  #     if not os.path.exists(filename):
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
          
      row = int(motor % (NUM_MOTORS / 2) ) 
      
      axs[row, column].plot(time, motor_vels_des[motor], linewidth=1.5, label="desired")
      axs[row, column].plot(time, motor_vels_act[motor], linewidth=1.5, label="actual")
      axs[row, column].set_title( MOTOR_LABEL[motor] + ' Motor Velocity Plot', fontsize=12)
      axs[row, column].set_xlabel('Time (s)', fontsize=12)
      axs[row, column].set_ylabel('Motor Vel (rad/s)', fontsize=12)
      axs[row, column].legend(fontsize=10, )
      axs[row, column].grid(True)
  
#  fig.align_labels()
  fig.tight_layout()
  plt.show()
  #     if not os.path.exists(filename):
  if savedata_filename is not None:  
    fig.savefig(velplots_filename, dpi=600) 

  # Plotting Motor Torques
  plt.figure()
  fig, axs = plt.subplots(4, 2, figsize=(12, 12))
  for motor in range(NUM_MOTORS):
      
      if motor < (NUM_MOTORS / 2):
          column = 0
      else:
          column = 1
          
      row = int(motor % (NUM_MOTORS / 2) ) 
  
      axs[row, column].plot(time, motor_torques_act[motor], linewidth=1.5, label="actual")
      axs[row, column].set_title( MOTOR_LABEL[motor] + ' Motor Torque Plot', fontsize=12)
      axs[row, column].set_xlabel('Time (s)', fontsize=12)
      axs[row, column].set_ylabel('Motor Torque (Nm)', fontsize=12)
  #     axs[row, column].legend(fontsize=10, )
      axs[row, column].grid(True)
  
#  fig.align_labels()
  fig.tight_layout()
  plt.show()
  #     if not os.path.exists(filename):
  if savedata_filename is not None:
    fig.savefig(torqueplots_filename, dpi=600)



#def main():
if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--env', help='environment ID (0==sim, 1==plotdata, 2==gaitlib)',type=int,  default=2)
  args = parser.parse_args()
  print("--env=" + str(args.env))
  
  ## reading optimal gait data obtained from frost
  bezierdata_dir = 'bezierdata_legspace'  
  fileIDs = glob.glob(os.path.join(currentdir, bezierdata_dir,'modbez*.txt') ) 
  steplens = ['0.00','0.05','0.10','0.15','0.20']
  gaits = dict(zip(steplens, fileIDs))  
    
  ## number of distinct phases in the gait cycle
  num_phases = 2
  ## Load Bezier Parameters and Phase Timings Data  
  vircons, phasevars = load_gait_txt( gaits['0.20'] , num_phases = num_phases)
  
  if (args.env == 0):
#    phasevars[0] = 0.1
#    phasevars[1] = phasevars[0]
    HZDPolicy(vircons = vircons, 
                       phasevars = phasevars,
                       sim_cycles = 5,
                       log_file = 'test.csv')  
#    pdb.set_trace()
    logging_plots('test.csv')
  if (args.env == 1):     
    ## for saving all the bullet sim logged data  
    filename_woext = os.path.splitext(filename)[0]
    logdata_dir = 'vircons_data'
    logdata_filename = filename_woext + '_bulletsim_log.csv'
    logdata_fullfilename = os.path.join(currentdir,logdata_dir,logdata_filename)
    
    ## for saving all the plotting data
    savedata_filename = filename_woext + '_'
    
    # Run Sim and Log Data    
    HZDPolicy(vircons = vircons, 
                       phasevars = phasevars, 
                       log_file=logdata_fullfilename)  
    time.sleep(3.)
    # Plot logged Data
    logging_plots(logdata_fullfilename, savedata_filename = savedata_filename)
#    logging_plots(logdata_fullfilename, savedata_filename = None)
    
  if (args.env == 2):
    GaitLibraryPolicy(gaits, 
                      log_file = 'test.csv',
                      steps_per_gait = 5,
                      randomize = True,
                      feedforward = os.path.split(fileIDs[0])[1].startswith('mod'))
    time.sleep(3.)
    logging_plots('test.csv')
        
#if __name__ == '__main__':
#    main()

