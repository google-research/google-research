# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

# fork of //third_party/bullet/examples/pybullet/gym/envs/bullet/kuka.py with
# fast reset capability.
# pylint: skip-file

import pybullet
from absl import logging
import numpy as np
import os
import copy
import math

class Kuka:

  def __init__(self, urdfRootPath='', timeStep=0.01, clientId=0, ikFix=False,
               returnPos=True):
    """Creates a Kuka robot.

    Args:
      urdfRootPath: The path to the root URDF directory.
      timeStep: The Pybullet timestep to use for simulation.
      clientId: The Pybullet client's ID.
      ikFix: A boolean for whether to apply the ikFix for control. This includes
        increase IK solver iterations, using intertial frame position instead
        of center of mass, and better tracking actual EEF pose. The old
        experiment results did not have these fixes.
      returnPos: A boolean for whether to return commanded EEF position.
    """
    self.cid = clientId
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.ikFix = ikFix
    self.returnPos = returnPos

    self.maxForce = 200.
    self.fingerAForce = 6
    self.fingerBForce = 5.5
    self.fingerTipForce = 6
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 0
    self.useOrientation = 1
    self.kukaEndEffectorIndex = 6
    #lower limits for null space
    self.ll=[-.967,-2 ,-2.96,0.19,-2.96,-2.09,-3.05]
    #upper limits for null space
    self.ul=[.967,2 ,2.96,2.29,2.96,2.09,3.05]
    #joint ranges for null space
    self.jr=[5.8,4,5.8,4,5.8,4,6]
    #restposes for null space
    self.rp=[0,0,0,0.5*math.pi,0,-math.pi*0.5*0.66,0]
    #joint damping coefficents
    self.jd = [.1] * 12
    kuka_path = os.path.join(urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf")
    self.kukaUid = pybullet.loadSDF(kuka_path, physicsClientId=self.cid)[0]
    tray_path = os.path.join(urdfRootPath, "tray/tray.urdf")
    self.trayUid = pybullet.loadURDF(tray_path,
                              [0.64, 0.075, -0.19],
                              [0.0, 0.0, 1.0, 0.0],
                              physicsClientId=self.cid)
    self.reset()


  def reset(self,
            base_pos=None,
            endeffector_pos=None):
    """Resets the kuka base and joint positions.

    Args:
      base_pos:  The [x, y, z] position of Kuka base.
      endeffector_pos: The [x, y, z] position of the initial endeffector
        position.
    """
    # Default values for the base position and initial endeffector position.
    if base_pos is None:
      base_pos = [-0.1, 0.0, 0.07]
    if endeffector_pos is None:
      endeffector_pos = [0.537, 0.0, 0.5]

    pybullet.resetBasePositionAndOrientation(self.kukaUid,
                                      base_pos,
                                      [0.000000, 0.000000, 0.000000, 1.000000],
                                      physicsClientId=self.cid)
    self.jointPositions=[0.006418, 0.413184, -0.011401, -1.589317, 0.005379,
                         1.137684, -0.006539, 0.000048, -0.299912, 0.000000,
                         -0.000043, 0.299960, 0.000000, -0.000200 ]
    self.numJoints = pybullet.getNumJoints(self.kukaUid,physicsClientId=self.cid)
    for jointIndex in range (self.numJoints):
      pybullet.resetJointState(self.kukaUid,
                        jointIndex,
                        self.jointPositions[jointIndex],
                        physicsClientId=self.cid)
      if self.useSimulation:
        pybullet.setJointMotorControl2(self.kukaUid,
                                jointIndex,
                                pybullet.POSITION_CONTROL,
                                targetPosition=self.jointPositions[jointIndex],
                                force=self.maxForce,
                                physicsClientId=self.cid)

    # Set the endeffector height to endEffectorPos.
    self.endEffectorPos = endeffector_pos

    self.endEffectorAngle = 0

    self.motorNames = []
    self.motorIndices = []

    for i in range (self.numJoints):
      jointInfo = pybullet.getJointInfo(self.kukaUid,i,physicsClientId=self.cid)
      qIndex = jointInfo[3]
      if qIndex > -1:
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6 # Position x,y,z and roll/pitch/yaw euler angles of end effector.

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    observation = []
    state = pybullet.getLinkState(
        self.kukaUid,self.kukaEndEffectorIndex,physicsClientId=self.cid)
    if self.ikFix:
      # state[0] is the linkWorldPosition, the center of mass of the link.
      # However, the IK solver uses localInertialFrameOrientation, the inertial
      # center of the link. So, we should use state[4] and not state[0].
      pos = state[4]
    else:
      pos = state[0]
    orn = state[1]

    observation.extend(list(pos))
    observation.extend(list(orn))

    return observation

  def applyFingerAngle(self, fingerAngle):
    # TODO(ejang) - replace with pybullet.setJointMotorControlArray (more
    # efficient).
    pybullet.setJointMotorControl2(
        self.kukaUid, 7, pybullet.POSITION_CONTROL,
        targetPosition=self.endEffectorAngle, force=self.maxForce,
        physicsClientId=self.cid)
    pybullet.setJointMotorControl2(
        self.kukaUid, 8, pybullet.POSITION_CONTROL,
        targetPosition=-fingerAngle, force=self.fingerAForce,
        physicsClientId=self.cid)
    pybullet.setJointMotorControl2(
        self.kukaUid, 11, pybullet.POSITION_CONTROL,
        targetPosition=fingerAngle, force=self.fingerBForce,
        physicsClientId=self.cid)
    pybullet.setJointMotorControl2(
        self.kukaUid, 10, pybullet.POSITION_CONTROL, targetPosition=0,
        force=self.fingerTipForce, physicsClientId=self.cid)
    pybullet.setJointMotorControl2(
        self.kukaUid, 13, pybullet.POSITION_CONTROL, targetPosition=0,
        force=self.fingerTipForce,physicsClientId=self.cid)

  def applyAction(self, motorCommands):
    pos = None

    if (self.useInverseKinematics):

      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      da = motorCommands[3]
      fingerAngle = motorCommands[4]

      state = pybullet.getLinkState(
          self.kukaUid,self.kukaEndEffectorIndex,physicsClientId=self.cid)

      if self.ikFix:
        actualEndEffectorPos = state[4]
        self.endEffectorPos = list(actualEndEffectorPos)
      else:
        actualEndEffectorPos = state[0]
      self.endEffectorPos[0] = self.endEffectorPos[0]+dx
      if (self.endEffectorPos[0]>0.75):
        self.endEffectorPos[0]=0.75
      if (self.endEffectorPos[0]<0.45):
        self.endEffectorPos[0]=0.45
      self.endEffectorPos[1] = self.endEffectorPos[1]+dy
      if (self.endEffectorPos[1]<-0.22):
        self.endEffectorPos[1]=-0.22
      if (self.endEffectorPos[1]>0.22):
        self.endEffectorPos[1]=0.22

      if (dz>0 or actualEndEffectorPos[2]>0.10):
        self.endEffectorPos[2] = self.endEffectorPos[2]+dz
      if (actualEndEffectorPos[2]<0.10):
        self.endEffectorPos[2] = self.endEffectorPos[2]+0.0001

      self.endEffectorAngle = self.endEffectorAngle + da
      pos = self.endEffectorPos
      orn = pybullet.getQuaternionFromEuler([0,-math.pi,0]) # -math.pi,yaw])
      if (self.useNullSpace==1):
        if (self.useOrientation==1):
          jointPoses = pybullet.calculateInverseKinematics(
              self.kukaUid, self.kukaEndEffectorIndex, pos,
              orn, self.ll, self.ul, self.jr, self.rp,
              maxNumIterations=1, physicsClientId=self.cid)
        else:
          jointPoses = pybullet.calculateInverseKinematics(
              self.kukaUid, self.kukaEndEffectorIndex, pos, lowerLimits=self.ll,
              upperLimits=self.ul, jointRanges=self.jr,
              restPoses=self.rp, maxNumIterations=1,
              physicsClientId=self.cid)
      else:
        if (self.useOrientation==1):
          if self.ikFix:
            jointPoses = pybullet.calculateInverseKinematics(
                self.kukaUid,self.kukaEndEffectorIndex,
                pos,orn,jointDamping=self.jd,maxNumIterations=50,
                residualThreshold=.001,
                physicsClientId=self.cid)
          else:
            jointPoses = pybullet.calculateInverseKinematics(
                self.kukaUid,self.kukaEndEffectorIndex,
                pos,orn,jointDamping=self.jd,maxNumIterations=1,
                physicsClientId=self.cid)
        else:
          jointPoses = pybullet.calculateInverseKinematics(
              self.kukaUid,self.kukaEndEffectorIndex, pos,
              maxNumIterations=1, physicsClientId=self.cid)
      if (self.useSimulation):
        for i in range (self.kukaEndEffectorIndex+1):
          #print(i)
          pybullet.setJointMotorControl2(
              bodyIndex=self.kukaUid,jointIndex=i,
              controlMode=pybullet.POSITION_CONTROL,
              targetPosition=jointPoses[i],
              targetVelocity=0, force=self.maxForce, positionGain=0.03,
              velocityGain=1, physicsClientId=self.cid)
      else:
        # Reset the joint state (ignoring all dynamics, not recommended to use
        # during simulation).

        # TODO(b/72742371) Figure out why if useSimulation = 0,
        # len(jointPoses) = 12 and self.numJoints = 14.
        for i in range(len(jointPoses)):
          pybullet.resetJointState(self.kukaUid, i, jointPoses[i],
                            physicsClientId=self.cid)
      # Move fingers.
      self.applyFingerAngle(fingerAngle)


    else:
      for action in range (len(motorCommands)):
        motor = self.motorIndices[action]
        pybullet.setJointMotorControl2(
            self.kukaUid, motor, pybullet.POSITION_CONTROL,
            targetPosition=motorCommands[action], force=self.maxForce,
            physicsClientId=self.cid)
    if self.returnPos:
      # Return the target position for metrics later.
      return pos
