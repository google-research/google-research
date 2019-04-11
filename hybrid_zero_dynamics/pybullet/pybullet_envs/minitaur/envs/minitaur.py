"""This file implements the functionalities of a minitaur using pybullet.

"""

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import collections
import copy
import math
import re
import numpy as np
from pybullet_envs.minitaur.envs import motor

INIT_POSITION = [0, 0, .25]
INIT_RACK_POSITION = [0, 0, 1]
INIT_ORIENTATION = [0, 0, 0, 1]
KNEE_CONSTRAINT_POINT_RIGHT = [0, 0.005, 0.2]
KNEE_CONSTRAINT_POINT_LEFT = [0, 0.01, 0.2]
OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
LEG_POSITION = ["front_left", "back_left", "front_right", "back_right"]
MOTOR_NAMES = [
    "motor_front_leftL_joint", "motor_front_leftR_joint",
    "motor_back_leftL_joint", "motor_back_leftR_joint",
    "motor_front_rightL_joint", "motor_front_rightR_joint",
    "motor_back_rightL_joint", "motor_back_rightR_joint"
]
_CHASSIS_NAME_PATTERN = re.compile(r"chassis\D*center")
_MOTOR_NAME_PATTERN = re.compile(r"motor\D*joint")
_KNEE_NAME_PATTERN = re.compile(r"knee\D*")
SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.0, 0.0)
TWO_PI = 2 * math.pi


def MapToMinusPiToPi(angles):
  """Maps a list of angles to [-pi, pi].

  Args:
    angles: A list of angles in rad.
  Returns:
    A list of angle mapped to [-pi, pi].
  """
  mapped_angles = copy.deepcopy(angles)
  for i in range(len(angles)):
    mapped_angles[i] = math.fmod(angles[i], TWO_PI)
    if mapped_angles[i] >= math.pi:
      mapped_angles[i] -= TWO_PI
    elif mapped_angles[i] < -math.pi:
      mapped_angles[i] += TWO_PI
  return mapped_angles


class Minitaur(object):
  """The minitaur class that simulates a quadruped robot from Ghost Robotics.

  """

  def __init__(self,
               pybullet_client,
               urdf_root="",
               time_step=0.01,
               action_repeat=1,
               self_collision_enabled=False,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               accurate_motor_model_enabled=False,
               remove_default_joint_damping=False,
               motor_kp=1.0,
               motor_kd=0.02,
               pd_latency=0.0,
               control_latency=0.0,
               observation_noise_stdev=SENSOR_NOISE_STDDEV,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               on_rack=False):
    """Constructs a minitaur and reset it to the initial states.

    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
      urdf_root: The path to the urdf folder.
      time_step: The time step of the simulation.
      action_repeat: The number of ApplyAction() for each control step.
      self_collision_enabled: Whether to enable self collision.
      motor_velocity_limit: The upper limit of the motor velocity.
      pd_control_enabled: Whether to use PD control for the motors.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      remove_default_joint_damping: Whether to remove the default joint damping.
      motor_kp: proportional gain for the accurate motor model.
      motor_kd: derivative gain for the accurate motor model.
      pd_latency: The latency of the observations (in seconds) used to calculate
        PD control. On the real hardware, it is the latency between the
        microcontroller and the motor controller.
      control_latency: The latency of the observations (in second) used to
        calculate action. On the real hardware, it is the latency from the motor
        controller, the microcontroller to the host (Nvidia TX2).
      observation_noise_stdev: The standard deviation of a Gaussian noise model
        for the sensor. It should be an array for separate sensors in the
        following order [motor_angle, motor_velocity, motor_torque,
        base_roll_pitch_yaw, base_angular_velocity]
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
        details.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
    """
    self.num_motors = 8
    self.num_legs = int(self.num_motors / 2)
    self._pybullet_client = pybullet_client
    self._action_repeat = action_repeat
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._pd_control_enabled = pd_control_enabled
    self._motor_direction = [-1, -1, -1, -1, 1, 1, 1, 1]
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._max_force = 3.5
    self._pd_latency = pd_latency
    self._control_latency = control_latency
    self._observation_noise_stdev = observation_noise_stdev
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._remove_default_joint_damping = remove_default_joint_damping
    self._observation_history = collections.deque(maxlen=100)
    self._control_observation = []
    self._chassis_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._foot_link_ids = []
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    if self._accurate_motor_model_enabled:
      self._kp = motor_kp
      self._kd = motor_kd
      self._motor_model = motor.MotorModel(
          torque_control_enabled=self._torque_control_enabled,
          kp=self._kp,
          kd=self._kd)
    elif self._pd_control_enabled:
      self._kp = 8
      self._kd = 0.3
    else:
      self._kp = 1
      self._kd = 1
    self.time_step = time_step
    self._step_counter = 0
    # reset_time=-1.0 means skipping the reset motion.
    # See Reset for more details.
    self.Reset(reset_time=-1.0)

  def GetTimeSinceReset(self):
    return self._step_counter * self.time_step

  def Step(self, action):
    for _ in range(self._action_repeat):
      self.ApplyAction(action)
      self._pybullet_client.stepSimulation()
      self.ReceiveObservation()
      self._step_counter += 1

  def Terminate(self):
    pass

  def _RecordMassInfoFromURDF(self):
    self._base_mass_urdf = []
    for chassis_id in self._chassis_link_ids:
      self._base_mass_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
    self._leg_masses_urdf = []
    for leg_id in self._leg_link_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])
    for motor_id in self._motor_link_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, motor_id)[0])

  def _RecordInertiaInfoFromURDF(self):
    """Record the inertia of each body from URDF file."""
    self._link_urdf = []
    num_bodies = self._pybullet_client.getNumJoints(self.quadruped)
    for body_id in range(-1, num_bodies):  # -1 is for the base link.
      inertia = self._pybullet_client.getDynamicsInfo(self.quadruped,
                                                      body_id)[2]
      self._link_urdf.append(inertia)
    # We need to use id+1 to index self._link_urdf because it has the base
    # (index = -1) at the first element.
    self._base_inertia_urdf = [
        self._link_urdf[chassis_id + 1] for chassis_id in self._chassis_link_ids
    ]
    self._leg_inertia_urdf = [
        self._link_urdf[leg_id + 1] for leg_id in self._leg_link_ids
    ]
    self._leg_inertia_urdf.extend(
        [self._link_urdf[motor_id + 1] for motor_id in self._motor_link_ids])

  def _BuildJointNameToIdDict(self):
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _BuildUrdfIds(self):
    """Build the link Ids from its name in the URDF file."""
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._chassis_link_ids = [-1]
    # the self._leg_link_ids include both the upper and lower links of the leg.
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._foot_link_ids = []
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if _CHASSIS_NAME_PATTERN.match(joint_name):
        self._chassis_link_ids.append(joint_id)
      elif _MOTOR_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      elif _KNEE_NAME_PATTERN.match(joint_name):
        self._foot_link_ids.append(joint_id)
      else:
        self._leg_link_ids.append(joint_id)
    self._leg_link_ids.extend(self._foot_link_ids)
    self._chassis_link_ids.sort()
    self._motor_link_ids.sort()
    self._foot_link_ids.sort()
    self._leg_link_ids.sort()

  def _RemoveDefaultJointDamping(self):
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._pybullet_client.changeDynamics(
          joint_info[0], -1, linearDamping=0, angularDamping=0)

  def _BuildMotorIdList(self):
    self._motor_id_list = [
        self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES
    ]

  def IsObservationValid(self):
    """Whether the observation is valid for the current time step.

    In simulation, observations are always valid. In real hardware, it may not
    be valid from time to time when communication error happens between the
    Nvidia TX2 and the microcontroller.

    Returns:
      Whether the observation is valid for the current time step.
    """
    return True

  def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
    """Reset the minitaur to its initial states.

    Args:
      reload_urdf: Whether to reload the urdf file. If not, Reset() just place
        the minitaur back to its starting position.
      default_motor_angles: The default motor angles. If it is None, minitaur
        will hold a default pose (motor angle math.pi / 2) for 100 steps. In
        torque control mode, the phase of holding the default pose is skipped.
      reset_time: The duration (in seconds) to hold the default motor angles. If
        reset_time <= 0 or in torque control mode, the phase of holding the
        default pose is skipped.
    """
    if self._on_rack:
      init_position = INIT_RACK_POSITION
    else:
      init_position = INIT_POSITION
    if reload_urdf:
      if self._self_collision_enabled:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/quadruped/minitaur.urdf" % self._urdf_root,
            init_position,
            useFixedBase=self._on_rack,
            flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
      else:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/quadruped/minitaur.urdf" % self._urdf_root,
            init_position,
            useFixedBase=self._on_rack)
      self._BuildJointNameToIdDict()
      self._BuildUrdfIds()
      if self._remove_default_joint_damping:
        self._RemoveDefaultJointDamping()
      self._BuildMotorIdList()
      self._RecordMassInfoFromURDF()
      self._RecordInertiaInfoFromURDF()
      self.ResetPose(add_constraint=True)
    else:
      self._pybullet_client.resetBasePositionAndOrientation(
          self.quadruped, init_position, INIT_ORIENTATION)
      self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0],
                                              [0, 0, 0])
      self.ResetPose(add_constraint=False)
    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors
    self._step_counter = 0

    # Perform reset motion within reset_duration if in position control mode.
    # Nothing is performed if in torque control mode for now.
    # TODO(jietan): Add reset motion when the torque control is fully supported.
    self._observation_history.clear()
    if not self._torque_control_enabled and reset_time > 0.0:
      self.ReceiveObservation()
      for _ in range(100):
        self.ApplyAction([math.pi / 2] * self.num_motors)
        self._pybullet_client.stepSimulation()
        self.ReceiveObservation()
      if default_motor_angles is not None:
        num_steps_to_reset = int(reset_time / self.time_step)
        for _ in range(num_steps_to_reset):
          self.ApplyAction(default_motor_angles)
          self._pybullet_client.stepSimulation()
          self.ReceiveObservation()
    self.ReceiveObservation()

  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        force=torque)

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.POSITION_CONTROL,
        targetPosition=desired_angle,
        positionGain=self._kp,
        velocityGain=self._kd,
        force=self._max_force)

  def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
    self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name],
                                   desired_angle)

  def ResetPose(self, add_constraint):
    """Reset the pose of the minitaur.

    Args:
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    for i in range(self.num_legs):
      self._ResetPoseForLeg(i, add_constraint)

  def _ResetPoseForLeg(self, leg_id, add_constraint):
    """Reset the initial pose for the leg.

    Args:
      leg_id: It should be 0, 1, 2, or 3, which represents the leg at
        front_left, back_left, front_right and back_right.
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    knee_friction_force = 0
    half_pi = math.pi / 2.0
    knee_angle = -2.1834

    leg_position = LEG_POSITION[leg_id]
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "L_joint"],
        self._motor_direction[2 * leg_id] * half_pi,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["knee_" + leg_position + "L_link"],
        self._motor_direction[2 * leg_id] * knee_angle,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "R_joint"],
        self._motor_direction[2 * leg_id + 1] * half_pi,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["knee_" + leg_position + "R_link"],
        self._motor_direction[2 * leg_id + 1] * knee_angle,
        targetVelocity=0)
    if add_constraint:
      self._pybullet_client.createConstraint(
          self.quadruped,
          self._joint_name_to_id["knee_" + leg_position + "R_link"],
          self.quadruped,
          self._joint_name_to_id["knee_" + leg_position + "L_link"],
          self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
          KNEE_CONSTRAINT_POINT_RIGHT, KNEE_CONSTRAINT_POINT_LEFT)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      # Disable the default motor in pybullet.
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(
              self._joint_name_to_id["motor_" + leg_position + "L_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(
              self._joint_name_to_id["motor_" + leg_position + "R_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)

    else:
      self._SetDesiredMotorAngleByName(
          "motor_" + leg_position + "L_joint",
          self._motor_direction[2 * leg_id] * half_pi)
      self._SetDesiredMotorAngleByName(
          "motor_" + leg_position + "R_joint",
          self._motor_direction[2 * leg_id + 1] * half_pi)

    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "L_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "R_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)

  def GetBasePosition(self):
    """Get the position of minitaur's base.

    Returns:
      The position of minitaur's base.
    """
    position, _ = (
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return position

  def GetTrueBaseRollPitchYaw(self):
    """Get minitaur's base orientation in euler angle in the world frame.

    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame.
    """
    orientation = self.GetTrueBaseOrientation()
    roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
    return np.asarray(roll_pitch_yaw)

  def GetBaseRollPitchYaw(self):
    """Get minitaur's base orientation in euler angle in the world frame.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame polluted by noise
      and latency.
    """
    delayed_orientation = np.array(
        self._control_observation[3 * self.num_motors:3 * self.num_motors + 4])
    delayed_roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(
        delayed_orientation)
    roll_pitch_yaw = self._AddSensorNoise(
        np.array(delayed_roll_pitch_yaw), self._observation_noise_stdev[3])
    return roll_pitch_yaw

  def GetTrueMotorAngles(self):
    """Gets the eight motor angles at the current moment, mapped to [-pi, pi].

    Returns:
      Motor angles, mapped to [-pi, pi].
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
        for motor_id in self._motor_id_list
    ]
    motor_angles = np.multiply(motor_angles, self._motor_direction)
    return motor_angles

  def GetMotorAngles(self):
    """Gets the eight motor angles.

    This function mimicks the noisy sensor reading and adds latency. The motor
    angles that are delayed, noise polluted, and mapped to [-pi, pi].

    Returns:
      Motor angles polluted by noise and latency, mapped to [-pi, pi].
    """
    motor_angles = self._AddSensorNoise(
        np.array(self._control_observation[0:self.num_motors]),
        self._observation_noise_stdev[0])
    return MapToMinusPiToPi(motor_angles)

  def GetTrueMotorVelocities(self):
    """Get the velocity of all eight motors.

    Returns:
      Velocities of all eight motors.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
        for motor_id in self._motor_id_list
    ]
    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetMotorVelocities(self):
    """Get the velocity of all eight motors.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Velocities of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise(
        np.array(
            self._control_observation[self.num_motors:2 * self.num_motors]),
        self._observation_noise_stdev[1])

  def GetTrueMotorTorques(self):
    """Get the amount of torque the motors are exerting.

    Returns:
      Motor torques of all eight motors.
    """
    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [
          self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
          for motor_id in self._motor_id_list
      ]
      motor_torques = np.multiply(motor_torques, self._motor_direction)
    return motor_torques

  def GetMotorTorques(self):
    """Get the amount of torque the motors are exerting.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Motor torques of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise(
        np.array(
            self._control_observation[2 * self.num_motors:3 * self.num_motors]),
        self._observation_noise_stdev[2])

  def GetTrueBaseOrientation(self):
    """Get the orientation of minitaur's base, represented as quaternion.

    Returns:
      The orientation of minitaur's base.
    """
    _, orientation = (
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return orientation

  def GetBaseOrientation(self):
    """Get the orientation of minitaur's base, represented as quaternion.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      The orientation of minitaur's base polluted by noise and latency.
    """
    return self._pybullet_client.getQuaternionFromEuler(
        self.GetBaseRollPitchYaw())

  def GetTrueBaseRollPitchYawRate(self):
    """Get the rate of orientation change of the minitaur's base in euler angle.

    Returns:
      rate of (roll, pitch, yaw) change of the minitaur's base.
    """
    vel = self._pybullet_client.getBaseVelocity(self.quadruped)
    return np.asarray([vel[1][0], vel[1][1], vel[1][2]])

  def GetBaseRollPitchYawRate(self):
    """Get the rate of orientation change of the minitaur's base in euler angle.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      rate of (roll, pitch, yaw) change of the minitaur's base polluted by noise
      and latency.
    """
    return self._AddSensorNoise(
        np.array(self._control_observation[3 * self.num_motors + 4:
                                           3 * self.num_motors + 7]),
        self._observation_noise_stdev[4])

  def GetActionDimension(self):
    """Get the length of the action list.

    Returns:
      The length of the action list.
    """
    return self.num_motors

  def ApplyAction(self, motor_commands, motor_kps=None, motor_kds=None):
    """Set the desired motor angles to the motors of the minitaur.

    The desired motor angles are clipped based on the maximum allowed velocity.
    If the pd_control_enabled is True, a torque is calculated according to
    the difference between current and desired joint angle, as well as the joint
    velocity. This torque is exerted to the motor. For more information about
    PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.

    Args:
      motor_commands: The eight desired motor angles.
      motor_kps: Proportional gains for the motor model. If not provided, it
        uses the default kp of the minitaur for all the motors.
      motor_kds: Derivative gains for the motor model. If not provided, it
        uses the default kd of the minitaur for all the motors.
    """
    if self._motor_velocity_limit < np.inf:
      current_motor_angle = self.GetTrueMotorAngles()
      motor_commands_max = (
          current_motor_angle + self.time_step * self._motor_velocity_limit)
      motor_commands_min = (
          current_motor_angle - self.time_step * self._motor_velocity_limit)
      motor_commands = np.clip(motor_commands, motor_commands_min,
                               motor_commands_max)
    # Set the kp and kd for all the motors if not provided as an argument.
    if motor_kps is None:
      motor_kps = np.full(8, self._kp)
    if motor_kds is None:
      motor_kds = np.full(8, self._kd)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      q, qdot = self._GetPDObservation()
      qdot_true = self.GetTrueMotorVelocities()
      if self._accurate_motor_model_enabled:
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
            motor_commands, q, qdot, qdot_true, motor_kps, motor_kds)
        if self._motor_overheat_protection:
          for i in range(self.num_motors):
            if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
              self._overheat_counter[i] += 1
            else:
              self._overheat_counter[i] = 0
            if (self._overheat_counter[i] >
                OVERHEAT_SHUTDOWN_TIME / self.time_step):
              self._motor_enabled_list[i] = False

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque,
                                                 self._motor_direction)

        for motor_id, motor_torque, motor_enabled in zip(
            self._motor_id_list, self._applied_motor_torque,
            self._motor_enabled_list):
          if motor_enabled:
            self._SetMotorTorqueById(motor_id, motor_torque)
          else:
            self._SetMotorTorqueById(motor_id, 0)
      else:
        torque_commands = -1 * motor_kps * (
            q - motor_commands) - motor_kds * qdot

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = torque_commands

        # Transform into the motor space when applying the torque.
        self._applied_motor_torques = np.multiply(self._observed_motor_torques,
                                                  self._motor_direction)

        for motor_id, motor_torque in zip(self._motor_id_list,
                                          self._applied_motor_torques):
          self._SetMotorTorqueById(motor_id, motor_torque)
    else:
      motor_commands_with_direction = np.multiply(motor_commands,
                                                  self._motor_direction)
      for motor_id, motor_command_with_direction in zip(
          self._motor_id_list, motor_commands_with_direction):
        self._SetDesiredMotorAngleById(motor_id, motor_command_with_direction)

  def ConvertFromLegModel(self, actions):
    """Convert the actions that use leg model to the real motor actions.

    Args:
      actions: The theta, phi of the leg model.
    Returns:
      The eight desired motor angles that can be used in ApplyActions().
    """
    motor_angle = copy.deepcopy(actions)
    scale_for_singularity = 1
    offset_for_singularity = 1.5
    half_num_motors = int(self.num_motors / 2)
    quater_pi = math.pi / 4
    for i in range(self.num_motors):
      action_idx = int(i // 2)
      forward_backward_component = (-scale_for_singularity * quater_pi * (
          actions[action_idx + half_num_motors] + offset_for_singularity))
      extension_component = (-1)**i * quater_pi * actions[action_idx]
      if i >= half_num_motors:
        extension_component = -extension_component
      motor_angle[i] = (
          math.pi + forward_backward_component + extension_component)
    return motor_angle

  def GetBaseMassesFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetBaseInertiasFromURDF(self):
    """Get the inertia of the base from the URDF file."""
    return self._base_inertia_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def GetLegInertiasFromURDF(self):
    """Get the inertia of the legs from the URDF file."""
    return self._leg_inertia_urdf

  def SetBaseMasses(self, base_mass):
    """Set the mass of minitaur's base.

    Args:
      base_mass: A list of masses of each body link in CHASIS_LINK_IDS. The
        length of this list should be the same as the length of CHASIS_LINK_IDS.
    Raises:
      ValueError: It is raised when the length of base_mass is not the same as
        the length of self._chassis_link_ids.
    """
    if len(base_mass) != len(self._chassis_link_ids):
      raise ValueError(
          "The length of base_mass {} and self._chassis_link_ids {} are not "
          "the same.".format(len(base_mass), len(self._chassis_link_ids)))
    for chassis_id, chassis_mass in zip(self._chassis_link_ids, base_mass):
      self._pybullet_client.changeDynamics(
          self.quadruped, chassis_id, mass=chassis_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link masses, last 8 correspond
    to motor masses (24 total).

    Args:
      leg_masses: The leg and motor masses for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of masses is not equal to number
        of links + motors.
    """
    if len(leg_masses) != len(self._leg_link_ids) + len(self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_mass in zip(self._leg_link_ids, leg_masses):
      self._pybullet_client.changeDynamics(
          self.quadruped, leg_id, mass=leg_mass)
    motor_masses = leg_masses[len(self._leg_link_ids):]
    for link_id, motor_mass in zip(self._motor_link_ids, motor_masses):
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, mass=motor_mass)

  def SetBaseInertias(self, base_inertias):
    """Set the inertias of minitaur's base.

    Args:
      base_inertias: A list of inertias of each body link in CHASIS_LINK_IDS.
        The length of this list should be the same as the length of
        CHASIS_LINK_IDS.
    Raises:
      ValueError: It is raised when the length of base_inertias is not the same
        as the length of self._chassis_link_ids and base_inertias contains
        negative values.
    """
    if len(base_inertias) != len(self._chassis_link_ids):
      raise ValueError(
          "The length of base_inertias {} and self._chassis_link_ids {} are "
          "not the same.".format(
              len(base_inertias), len(self._chassis_link_ids)))
    for chassis_id, chassis_inertia in zip(self._chassis_link_ids,
                                           base_inertias):
      for inertia_value in chassis_inertia:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(
          self.quadruped, chassis_id, localInertiaDiagonal=chassis_inertia)

  def SetLegInertias(self, leg_inertias):
    """Set the inertias of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link inertia, last 8 correspond
    to motor inertia (24 total).

    Args:
      leg_inertias: The leg and motor inertias for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of inertias is not equal to
      the number of links + motors or leg_inertias contains negative values.
    """

    if len(leg_inertias) != len(self._leg_link_ids) + len(self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_inertia in zip(self._leg_link_ids, leg_inertias):
      for inertia_value in leg_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(
          self.quadruped, leg_id, localInertiaDiagonal=leg_inertia)

    motor_inertias = leg_inertias[len(self._leg_link_ids):]
    for link_id, motor_inertia in zip(self._motor_link_ids, motor_inertias):
      for inertia_value in motor_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, localInertiaDiagonal=motor_inertia)

  def SetFootFriction(self, foot_friction):
    """Set the lateral friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
    for link_id in self._foot_link_ids:
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, lateralFriction=foot_friction)

  # TODO(b/73748980): Add more API's to set other contact parameters.
  def SetFootRestitution(self, foot_restitution):
    """Set the coefficient of restitution at the feet.

    Args:
      foot_restitution: The coefficient of restitution (bounciness) of the feet.
        This value is shared by all four feet.
    """
    for link_id in self._foot_link_ids:
      self._pybullet_client.changeDynamics(
          self.quadruped, link_id, restitution=foot_restitution)

  def SetJointFriction(self, joint_frictions):
    for knee_joint_id, friction in zip(self._foot_link_ids, joint_frictions):
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=knee_joint_id,
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=friction)

  def GetNumKneeJoints(self):
    return len(self._foot_link_ids)

  def SetBatteryVoltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)

  def GetTrueObservation(self):
    observation = []
    observation.extend(self.GetTrueMotorAngles())
    observation.extend(self.GetTrueMotorVelocities())
    observation.extend(self.GetTrueMotorTorques())
    observation.extend(self.GetTrueBaseOrientation())
    observation.extend(self.GetTrueBaseRollPitchYawRate())
    return observation

  def ReceiveObservation(self):
    """Receive the observation from sensors.

    This function is called once per step. The observations are only updated
    when this function is called.
    """
    self._observation_history.appendleft(self.GetTrueObservation())
    self._control_observation = self._GetControlObservation()

  def _GetDelayedObservation(self, latency):
    """Get observation that is delayed by the amount specified in latency.

    Args:
      latency: The latency (in seconds) of the delayed observation.
    Returns:
      observation: The observation which was actually latency seconds ago.
    """
    if latency <= 0 or len(self._observation_history) == 1:
      observation = self._observation_history[0]
    else:
      n_steps_ago = int(latency / self.time_step)
      if n_steps_ago + 1 >= len(self._observation_history):
        return self._observation_history[-1]
      remaining_latency = latency - n_steps_ago * self.time_step
      blend_alpha = remaining_latency / self.time_step
      observation = (
          (1.0 - blend_alpha) * np.array(self._observation_history[n_steps_ago])
          + blend_alpha * np.array(self._observation_history[n_steps_ago + 1]))
    return observation

  def _GetPDObservation(self):
    pd_delayed_observation = self._GetDelayedObservation(self._pd_latency)
    q = pd_delayed_observation[0:self.num_motors]
    qdot = pd_delayed_observation[self.num_motors:2 * self.num_motors]
    return (np.array(q), np.array(qdot))

  def _GetControlObservation(self):
    control_delayed_observation = self._GetDelayedObservation(
        self._control_latency)
    return control_delayed_observation

  def _AddSensorNoise(self, sensor_values, noise_stdev):
    if noise_stdev <= 0:
      return sensor_values
    observation = sensor_values + np.random.normal(
        scale=noise_stdev, size=sensor_values.shape)
    return observation

  def SetControlLatency(self, latency):
    """Set the latency of the control loop.

    It measures the duration between sending an action from Nvidia TX2 and
    receiving the observation from microcontroller.

    Args:
      latency: The latency (in seconds) of the control loop.
    """
    self._control_latency = latency

  def GetControlLatency(self):
    """Get the control latency.

    Returns:
      The latency (in seconds) between when the motor command is sent and when
        the sensor measurements are reported back to the controller.
    """
    return self._control_latency

  def SetMotorGains(self, kp, kd):
    """Set the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: proportional gain of the motors.
      kd: derivative gain of the motors.
    """
    self._kp = kp
    self._kd = kd
    if self._accurate_motor_model_enabled:
      self._motor_model.set_motor_gains(kp, kd)

  def GetMotorGains(self):
    """Get the gains of the motor.

    Returns:
      The proportional gain.
      The derivative gain.
    """
    return self._kp, self._kd

  def SetMotorStrengthRatio(self, ratio):
    """Set the strength of all motors relative to the default value.

    Args:
      ratio: The relative strength. A scalar range from 0.0 to 1.0.
    """
    if self._accurate_motor_model_enabled:
      self._motor_model.set_strength_ratios([ratio] * self.num_motors)

  def SetMotorStrengthRatios(self, ratios):
    """Set the strength of each motor relative to the default value.

    Args:
      ratios: The relative strength. A numpy array ranging from 0.0 to 1.0.
    """
    if self._accurate_motor_model_enabled:
      self._motor_model.set_strength_ratios(ratios)

  def SetTimeSteps(self, action_repeat, simulation_step):
    """Set the time steps of the control and simulation.

    Args:
      action_repeat: The number of simulation steps that the same action is
        repeated.
      simulation_step: The simulation time step.
    """
    self.time_step = simulation_step
    self._action_repeat = action_repeat

  @property
  def chassis_link_ids(self):
    return self._chassis_link_ids
