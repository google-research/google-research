"""An example to run of the minitaur gym environment with virtual-constraints-based gaits.

"""
#import pdb
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
import scipy as sp
import tensorflow as tf
from pybullet_envs.minitaur.envs import minitaur_gym_env
import time

#import pickle
from hyperopt import fmin, tpe, hp

import matplotlib.pyplot as plt

# Define indices
NUM_MOTORS = 8
MOTOR_LABEL = ['FLL','FLR','BLL','BLR','FRL','FRR','BRL','BRR']
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


def Normalize(t_cur, t_max, t_min): 
  """Normlize given variable w.r.t max and min values.
  
  Args:
    t_cur: variable of interest.
    t_max: max possible value.
    t_min: min possible value.
  """
  tau = (t_cur - t_min) / (t_max - t_min)
  return tau    

    
def BezierCurveInterp(control_points, tau):
  """Bezier Interpolation to determine desired motor position at time t.
  
  tau range must be in the range [0, 1].
  degree of the bezier curve is len(control_points) - 1
  
  Args:
    tau: Normalized time value.
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


def LoadVirConsFromTXT(filename, num_phases = 2):
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
  num_outputs = int( (data_shape[0] - num_phases ) / num_phases )
  
  vircons = np.zeros([num_phases, num_outputs, num_coeffs])
  phasevars = np.zeros([num_phases, 1])
  
  for phase in range(num_phases):
    vircons[phase, :, :] = data[phase*num_outputs:(phase+1)*num_outputs,
                                :num_coeffs]
    phasevars[phase] = data[num_outputs*num_phases + phase, 0]
  
  return vircons, phasevars


def WriteToCSV(filename, actions_and_observations):
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
      
      
def ReadFromCSV(filename, print_data = False):
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


def MinitaurVirConsSim(vircons=None, 
                       phasevars=None,
                       log_file=None,
                       sim_cycles=2,
                       adapt_traj=False):
  """An example of minitaur locomoting with a virtual constraints (a.k.a hybrid zero dynamics - based gait).

  Args:
    log_path: The directory into which log files are written. 
    If log_path is None, no data logging takes place.
  """
  time_step = 0.001
  sum_reward = 0
#  steps = 5000
#  time.sleep(5.)
  
  

  num_outputs = len(vircons[0])

  cycle_time = float(sum(phasevars)) 
  #sim_cycles = 3
  print('Gait Cycle Time is ' + str(cycle_time))
  print('Estimated num of cycles in this Sim = ', str(sim_cycles))
  
  steps = int( (sim_cycles * cycle_time) / time_step )
  
  INIT_MOTOR_POS = []
  for output in range(num_outputs):
    coeffs = vircons[0, output, :]
    output_val_at_0, _ = BezierCurveInterp(coeffs, 0)
    if output < int(num_outputs/2):
      INIT_MOTOR_POS.append(-float(output_val_at_0))
    else:
      INIT_MOTOR_POS.append(float(output_val_at_0))
    
  INIT_MOTOR_POS = np.array(INIT_MOTOR_POS, dtype='float32')  
  print(INIT_MOTOR_POS)

  environment = minitaur_gym_env.MinitaurGymEnv(
      urdf_version=minitaur_gym_env.RAINBOW_DASH_V0_URDF_VERSION,
      render= not adapt_traj,
      motor_velocity_limit=np.inf,
      accurate_motor_model_enabled=True,
      remove_default_joint_damping=True,
      pd_control_enabled=True,
      leg_model_enabled=False,
      hard_reset=False,
      on_rack=False,
      motor_kp=1,
      motor_kd=0.015,
      control_latency=0.0,
      pd_latency=0.003,
      log_path=None)

  environment.reset(initial_motor_angles = INIT_MOTOR_POS, reset_duration = 3.0 )
#  pdb.set_trace()
#  tstart = environment.minitaur.GetTimeSinceReset()
  time.sleep(5.)
  actions_and_observations = []  
  for step_counter in range(steps):
    
#    t = environment.minitaur.GetTimeSinceReset() - tstart #step_counter * time_step
    t = step_counter * time_step
    current_row = [t]
    
#    print(t)
#    print(step_counter)
    
    cycles = t // cycle_time + 1
#    print(cycles)
    t_in_cycle = t % cycle_time
#    print('Time in Cycle is ' + str(t_in_cycle) )
    if 0 <= t_in_cycle <= float(phasevars[0]):
      tau = Normalize(t_in_cycle, float(phasevars[0]) , 0)
      phase = 0
#      print('Currently executing Phase ', str(phase+1))
#      print('Tau is ' + str(tau) )  
    elif float(phasevars[0]) <= t_in_cycle <= cycle_time:
      tau = Normalize(t_in_cycle, cycle_time, float(phasevars[0]) )
      phase = 1
#      print('Currently executing Phase ', str(phase+1))
#      print('Tau is ' + str(tau) )  
    else:
      ValueError('Tau must lie between 0 and ' + str(cycle_time))
      
    
    action = []
    action_vel = []
    for output in range(num_outputs):
      coeffs = vircons[phase, output, :]
      output_val_at_tau, output_vel_at_tau = BezierCurveInterp(coeffs, tau)
      if output < int(num_outputs/2):
        action.append(float(-output_val_at_tau))
        action_vel.append(float(-output_vel_at_tau))
      else:
        action.append(float(output_val_at_tau))
        action_vel.append(float(output_vel_at_tau))  
    #action = [math.pi / 2] * 8
    
    current_row.extend(action)
    current_row.extend(action_vel)
#    print(action)    
    
    observation, reward, done, _ = environment.step(action)
    current_row.extend(observation.tolist())
    actions_and_observations.append(current_row)
    
    
    time.sleep(1./100.)

    sum_reward += reward
    
    if done or (step_counter == (steps - 1) ):
      print('End of Sim')
      if log_file is not None:
        filename = os.path.join(currentdir, log_file)
        WriteToCSV(filename, actions_and_observations)
      return actions_and_observations
#      tf.logging.info("Return is {}".format(sum_reward))
#      environment.reset()
        
def MinitaurVirConsFun():
  pass

def LoggingDataPlots(data_filename, savedata_filename=None):
  ## Format Data
  data = ReadFromCSV(data_filename)
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



def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--env', help='environment ID (0==sim, 1==plotdata)',type=int,  default=0)
  args = parser.parse_args()
  print("--env=" + str(args.env))
  
  ## reading optimal gait data obtained from frost
  bezierdata_dir = 'bezierdata'  
  filename = 'bezcoeffs_20190328T065312_trot.txt'
  bezierdata_filename = os.path.join(currentdir,bezierdata_dir,filename)
  
  ## number of distinct phases in the gait cycle
  num_phases = 2
  ## Load Bezier Parameters and Phase Timings Data  
  vircons, phasevars = LoadVirConsFromTXT(bezierdata_filename, num_phases = num_phases)
  
  if (args.env == 0):
    MinitaurVirConsSim(vircons = vircons, 
                       phasevars = phasevars,
                       log_file = 'test.csv')  
    LoggingDataPlots('test.csv')
  if (args.env == 1):     
    ## for saving all the bullet sim logged data  
    filename_woext = os.path.splitext(filename)[0]
    logdata_dir = 'vircons_data'
    logdata_filename = filename_woext + '_bulletsim_log.csv'
    logdata_fullfilename = os.path.join(currentdir,logdata_dir,logdata_filename)
    
    ## for saving all the plotting data
    savedata_filename = filename_woext + '_'
    
    # Run Sim and Log Data    
    MinitaurVirConsSim(vircons = vircons, 
                       phasevars = phasevars, 
                       log_file=logdata_fullfilename)  
    time.sleep(10.)
    # Plot logged Data
    LoggingDataPlots(logdata_fullfilename, savedata_filename = savedata_filename)

        
if __name__ == '__main__':
    main()

