import pybullet as p
p.connect(p.GUI)
p.loadURDF("combined.urdf", useFixedBase=True)



#for j in range (p.getNumJoints(0)):
#	p.setJointMotorControl2(0,j,p.VELOCITY_CONTROL,targetVelocity=0.1)
p.setRealTimeSimulation(1)
while (p.isConnected()):
	p.getCameraImage(320,200)
	import time
	time.sleep(1./240.)
