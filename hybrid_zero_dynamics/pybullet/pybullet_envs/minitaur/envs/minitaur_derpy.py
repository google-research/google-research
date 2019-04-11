"""This file implements the functionalities of a minitaur derpy using pybullet.

It is the result of first pass system identification for the derpy robot. The


"""
import math

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import numpy as np
from pybullet_envs.minitaur.envs  import minitaur

KNEE_CONSTRAINT_POINT_LONG = [0, 0.0055, 0.088]
KNEE_CONSTRAINT_POINT_SHORT = [0, 0.0055, 0.100]


class MinitaurDerpy(minitaur.Minitaur):
  """The minitaur class that simulates a quadruped robot from Ghost Robotics.

  """

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
      init_position = minitaur.INIT_RACK_POSITION
    else:
      init_position = minitaur.INIT_POSITION
    if reload_urdf:
      if self._self_collision_enabled:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/quadruped/minitaur_derpy.urdf" % self._urdf_root,
            init_position,
            useFixedBase=self._on_rack,
            flags=(
                self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT))
      else:
        self.quadruped = self._pybullet_client.loadURDF(
            "%s/quadruped/minitaur_derpy.urdf" % self._urdf_root,
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
          self.quadruped, init_position, minitaur.INIT_ORIENTATION)
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

    leg_position = minitaur.LEG_POSITION[leg_id]
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "L_joint"],
        self._motor_direction[2 * leg_id] * half_pi,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["knee_" + leg_position + "L_joint"],
        self._motor_direction[2 * leg_id] * knee_angle,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "R_joint"],
        self._motor_direction[2 * leg_id + 1] * half_pi,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["knee_" + leg_position + "R_joint"],
        self._motor_direction[2 * leg_id + 1] * knee_angle,
        targetVelocity=0)
    if add_constraint:
      if leg_id < 2:
        self._pybullet_client.createConstraint(
            self.quadruped,
            self._joint_name_to_id["knee_" + leg_position + "R_joint"],
            self.quadruped,
            self._joint_name_to_id["knee_" + leg_position + "L_joint"],
            self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
            KNEE_CONSTRAINT_POINT_SHORT, KNEE_CONSTRAINT_POINT_LONG)
      else:
        self._pybullet_client.createConstraint(
            self.quadruped,
            self._joint_name_to_id["knee_" + leg_position + "R_joint"],
            self.quadruped,
            self._joint_name_to_id["knee_" + leg_position + "L_joint"],
            self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
            KNEE_CONSTRAINT_POINT_LONG, KNEE_CONSTRAINT_POINT_SHORT)

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
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "L_joint"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "R_joint"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)
