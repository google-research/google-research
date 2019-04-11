#rudimentary MuJoCo mjcf to ROS URDF converter using the UrdfEditor

import pybullet_utils.bullet_client as bc
import pybullet_data as pd

import pybullet_utils.urdfEditor as ed
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mjcf', help='MuJoCo xml file to be converted to URDF', default='mjcf/humanoid.xml')
args = parser.parse_args()

p = bc.BulletClient()
p.setAdditionalSearchPath(pd.getDataPath())
objs = p.loadMJCF(args.mjcf, flags=p.URDF_USE_IMPLICIT_CYLINDER)

for o in objs:
	#print("o=",o, p.getBodyInfo(o), p.getNumJoints(o))
	humanoid = objs[o]
	ed0 = ed.UrdfEditor()
	ed0.initializeFromBulletBody(humanoid, p._client)
	robotName = str(p.getBodyInfo(o)[1],'utf-8')
	partName = str(p.getBodyInfo(o)[0], 'utf-8')
	
	print("robotName=",robotName)
	print("partName=",partName)
	
	saveVisuals=False
	ed0.saveUrdf(robotName+"_"+partName+".urdf", saveVisuals)
