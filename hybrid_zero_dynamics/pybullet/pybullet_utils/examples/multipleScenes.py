import pybullet_utils.bullet_client as bc
import pybullet
import pybullet_data

p0 = bc.BulletClient(connection_mode=pybullet.DIRECT)
p0.setAdditionalSearchPath(pybullet_data.getDataPath())

p1 = bc.BulletClient(connection_mode=pybullet.DIRECT)
p1.setAdditionalSearchPath(pybullet_data.getDataPath())

#can also connect using different modes, GUI, SHARED_MEMORY, TCP, UDP, SHARED_MEMORY_SERVER, GUI_SERVER
#pgui = bc.BulletClient(connection_mode=pybullet.GUI)

p0.loadURDF("r2d2.urdf")
p1.loadSDF("stadium.sdf")
print(p0._client)
print(p1._client)
print("p0.getNumBodies()=",p0.getNumBodies())
print("p1.getNumBodies()=",p1.getNumBodies())

