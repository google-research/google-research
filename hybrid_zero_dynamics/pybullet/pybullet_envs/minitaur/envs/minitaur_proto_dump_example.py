import argparse
import numpy

from pybullet_envs.minitaur.envs import minitaur_logging

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log_file', help='path to protobuf file', default='')
args = parser.parse_args()
logging = minitaur_logging.MinitaurLogging()
episode = logging.restore_episode(args.log_file)
#print(dir (episode))
#print("episode=",episode)
fields = episode.ListFields()

recs = []

for rec in fields[0][1]:
	#print(rec.time)
	for motorState in rec.motor_states:
		#print("motorState.angle=",motorState.angle)
		#print("motorState.velocity=",motorState.velocity)
		#print("motorState.action=",motorState.action)
		#print("motorState.torque=",motorState.torque)
		recs.append([motorState.angle,motorState.velocity,motorState.action,motorState.torque])
		
a = numpy.array(recs)
numpy.savetxt("motorq_qdot_action_torque.csv", a, delimiter=",")