import numpy as np



class PDControllerStableMultiDof(object):
    def __init__(self, pb):
      self._pb = pb

    def computeAngVel(self,ornStart, ornEnd, deltaTime, bullet_client):
      dorn = bullet_client.getDifferenceQuaternion(ornStart,ornEnd)
      axis,angle = bullet_client.getAxisAngleFromQuaternion(dorn)
      angVel = [(axis[0]*angle)/deltaTime,(axis[1]*angle)/deltaTime,(axis[2]*angle)/deltaTime]
      return angVel
    
    def quatMul(self, q1, q2):
      return [ q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
		           q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2],
           		q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0],
		          q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]]
		
    def computeAngVelRel(self,ornStart, ornEnd, deltaTime, bullet_client):
      ornStartConjugate = [-ornStart[0],-ornStart[1],-ornStart[2],ornStart[3]]
      q_diff = self.quatMul(ornStartConjugate, ornEnd)#bullet_client.multiplyTransforms([0,0,0], ornStartConjugate, [0,0,0], ornEnd)
      
      axis,angle = bullet_client.getAxisAngleFromQuaternion(q_diff)
      angVel = [(axis[0]*angle)/deltaTime,(axis[1]*angle)/deltaTime,(axis[2]*angle)/deltaTime]
      return angVel
    
    
    def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds, maxForces, timeStep):
      
      numJoints = len(jointIndices)#self._pb.getNumJoints(bodyUniqueId)
      curPos,curOrn = self._pb.getBasePositionAndOrientation(bodyUniqueId)
      q1 = [curPos[0],curPos[1],curPos[2],curOrn[0],curOrn[1],curOrn[2],curOrn[3]]
      #print("q1=",q1)
      
      
      #qdot1 = [0,0,0, 0,0,0,0]
      baseLinVel, baseAngVel = self._pb.getBaseVelocity(bodyUniqueId)
      #print("baseLinVel=",baseLinVel)
      qdot1 = [baseLinVel[0],baseLinVel[1],baseLinVel[2],baseAngVel[0],baseAngVel[1],baseAngVel[2],0]
      #qError = [0,0,0, 0,0,0,0]
      desiredOrn = [desiredPositions[3],desiredPositions[4],desiredPositions[5],desiredPositions[6]]
      axis1 = self._pb.getAxisDifferenceQuaternion(desiredOrn,curOrn)
      angDiff = [0,0,0]#self.computeAngVel(curOrn, desiredOrn, 1, self._pb)
      qError=[ desiredPositions[0]-curPos[0], desiredPositions[1]-curPos[1], desiredPositions[2]-curPos[2],angDiff[0],angDiff[1],angDiff[2],0]
      target_pos = np.array(desiredPositions)
      #np.savetxt("pb_target_pos.csv", target_pos, delimiter=",")
      
      
      qIndex=7
      qdotIndex=7
      zeroAccelerations=[0,0,0,  0,0,0,0]
      for i in range (numJoints):
        js = self._pb.getJointStateMultiDof(bodyUniqueId, jointIndices[i])
        
        jointPos=js[0]
        jointVel = js[1]
        q1+=jointPos
        
        if len(js[0])==1:
          desiredPos=desiredPositions[qIndex]
          
          qdiff=desiredPos - jointPos[0]
          qError.append(qdiff)
          zeroAccelerations.append(0.)
          qdot1+=jointVel
          qIndex+=1
          qdotIndex+=1
        if len(js[0])==4:
          desiredPos=[desiredPositions[qIndex],desiredPositions[qIndex+1],desiredPositions[qIndex+2],desiredPositions[qIndex+3]]
          #axis = self._pb.getAxisDifferenceQuaternion(desiredPos,jointPos)
          angDiff = self.computeAngVelRel(jointPos, desiredPos, 1, self._pb)
          #angDiff = self._pb.computeAngVelRel(jointPos, desiredPos, 1)
          
          jointVelNew = [jointVel[0],jointVel[1],jointVel[2],0]
          qdot1+=jointVelNew
          qError.append(angDiff[0])
          qError.append(angDiff[1])
          qError.append(angDiff[2])
          qError.append(0)
          desiredVel=[desiredVelocities[qdotIndex],desiredVelocities[qdotIndex+1],desiredVelocities[qdotIndex+2]]
          zeroAccelerations+=[0.,0.,0.,0.]
          qIndex+=4
          qdotIndex+=4
      
      q = np.array(q1)
      
      qerr = np.array(qError)
      
      #np.savetxt("pb_qerro.csv",qerr,delimiter=",")
      
      #np.savetxt("pb_q.csv", q, delimiter=",")
      
      qdot=np.array(qdot1)
      #np.savetxt("qdot.csv", qdot, delimiter=",")
      
      qdotdesired = np.array(desiredVelocities)
      qdoterr = qdotdesired-qdot
      
      
      Kp = np.diagflat(kps)
      Kd = np.diagflat(kds)
      
      p =  Kp.dot(qError)
      
      #np.savetxt("pb_qError.csv", qError, delimiter=",")
      #np.savetxt("pb_p.csv", p, delimiter=",")
      
      d = Kd.dot(qdoterr)

      #np.savetxt("pb_d.csv", d, delimiter=",")
      #np.savetxt("pbqdoterr.csv", qdoterr, delimiter=",")
      
      
      M1 = self._pb.calculateMassMatrix(bodyUniqueId,q1, flags=1)
      
      
      M2 = np.array(M1)
      #np.savetxt("M2.csv", M2, delimiter=",")
      
      M = (M2 + Kd * timeStep)
      
      #np.savetxt("pbM_tKd.csv",M, delimiter=",")
      
    
      
      c1 = self._pb.calculateInverseDynamics(bodyUniqueId, q1, qdot1, zeroAccelerations, flags=1)
      
      
      c = np.array(c1)
      #np.savetxt("pb_C.csv",c, delimiter=",")
      A = M
      #p = [0]*43
      #np.savetxt("pb_kp_dot_qError.csv", p)
      #np.savetxt("pb_kd_dot_vError.csv", d)
      
      b =  p + d -c
      #np.savetxt("pb_b_acc.csv",b, delimiter=",")
      
      
      useNumpySolver = True
      if useNumpySolver:
        qddot = np.linalg.solve(A, b)
      else:
        dofCount = len(b)
        print(dofCount)
        qddot = self._pb.ldltSolve(bodyUniqueId, jointPositions=q1, b=b.tolist(), kd=kds, t=timeStep)
      
      tau = p + d - Kd.dot(qddot) * timeStep
      #print("len(tau)=",len(tau))
      #np.savetxt("pb_tau_not_clamped.csv", tau, delimiter=",")
      
      maxF = np.array(maxForces)
      #print("maxF",maxF)
      forces = np.clip(tau, -maxF , maxF )
      
      #np.savetxt("pb_tau_clamped.csv", tau, delimiter=",")
      return forces
      
    
      
class PDControllerStable(object):
    def __init__(self, pb):
      self._pb = pb

    def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds, maxForces, timeStep):
      numJoints = self._pb.getNumJoints(bodyUniqueId)
      jointStates = self._pb.getJointStates(bodyUniqueId, jointIndices)
      q1 = []
      qdot1 = []
      zeroAccelerations = []
      for i in range (numJoints):
        q1.append(jointStates[i][0])
        qdot1.append(jointStates[i][1])
        zeroAccelerations.append(0)
      q = np.array(q1)
      qdot=np.array(qdot1)
      qdes = np.array(desiredPositions)
      qdotdes = np.array(desiredVelocities)
      qError = qdes - q
      qdotError = qdotdes - qdot
      Kp = np.diagflat(kps)
      Kd = np.diagflat(kds)
      p =  Kp.dot(qError)
      d = Kd.dot(qdotError)
      forces = p + d
      
      M1 = self._pb.calculateMassMatrix(bodyUniqueId,q1)
      M2 = np.array(M1)
      M = (M2 + Kd * timeStep)
      c1 = self._pb.calculateInverseDynamics(bodyUniqueId, q1, qdot1, zeroAccelerations)
      c = np.array(c1)
      A = M
      b = -c + p + d
      qddot = np.linalg.solve(A, b)
      tau = p + d - Kd.dot(qddot) * timeStep
      maxF = np.array(maxForces)
      forces = np.clip(tau, -maxF , maxF )
      #print("c=",c)
      return tau
