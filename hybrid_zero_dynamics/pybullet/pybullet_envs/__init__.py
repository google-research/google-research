import gym
from gym.envs.registration import registry, make, spec
def register(id,*args,**kvargs):
	if id in registry.env_specs:
		return
	else:
		return gym.envs.registration.register(id,*args,**kvargs)

# ------------bullet-------------

register(
        id='HumanoidDeepMimicBulletEnv-v1',
        entry_point='pybullet_envs.deep_mimic:HumanoidDeepMimicGymEnv',
        max_episode_steps=1000,
        reward_threshold=20000.0,
)

register(
	id='CartPoleBulletEnv-v1',
	entry_point='pybullet_envs.bullet:CartPoleBulletEnv',
        max_episode_steps=200,
	reward_threshold=190.0,
)

register(
    id='MinitaurBulletEnv-v0',
    entry_point='pybullet_envs.bullet:MinitaurBulletEnv',
    timestep_limit=1000,
    reward_threshold=15.0,
)

register(
    id='MinitaurBulletDuckEnv-v0',
    entry_point='pybullet_envs.bullet:MinitaurBulletDuckEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)


register(
    id='MinitaurReactiveEnv-v0',
    entry_point='pybullet_envs.minitaur.envs:MinitaurReactiveEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)


register(
    id='MinitaurBallGymEnv-v0',
    entry_point='pybullet_envs.minitaur.envs:MinitaurBallGymEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)


register(
    id='MinitaurTrottingEnv-v0',
    entry_point='pybullet_envs.minitaur.envs:MinitaurTrottingEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)

register(
    id='MinitaurStandGymEnv-v0',
    entry_point='pybullet_envs.minitaur.envs:MinitaurStandGymEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)

register(
    id='MinitaurAlternatingLegsEnv-v0',
    entry_point='pybullet_envs.minitaur.envs:MinitaurAlternatingLegsEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)

register(
    id='MinitaurFourLegStandEnv-v0',
    entry_point='pybullet_envs.minitaur.envs:MinitaurFourLegStandEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)



register(
    id='RacecarBulletEnv-v0',
    entry_point='pybullet_envs.bullet:RacecarGymEnv',
    timestep_limit=1000,
    reward_threshold=5.0,
)

register(
	id='RacecarZedBulletEnv-v0',
	entry_point='pybullet_envs.bullet:RacecarZEDGymEnv',
	timestep_limit=1000,
	reward_threshold=5.0,
)


register(
	id='KukaBulletEnv-v0',
	entry_point='pybullet_envs.bullet:KukaGymEnv',
	timestep_limit=1000,
	reward_threshold=5.0,
)

register(
	id='KukaCamBulletEnv-v0',
	entry_point='pybullet_envs.bullet:KukaCamGymEnv',
	timestep_limit=1000,
	reward_threshold=5.0,
)

register(
	id='KukaDiverseObjectGrasping-v0',
	entry_point='pybullet_envs.bullet:KukaDiverseObjectEnv',
	timestep_limit=1000,
	reward_threshold=5.0,
)

register(
	id='InvertedPendulumBulletEnv-v0',
	entry_point='pybullet_envs.gym_pendulum_envs:InvertedPendulumBulletEnv',
	max_episode_steps=1000,
	reward_threshold=950.0,
	)

register(
	id='InvertedDoublePendulumBulletEnv-v0',
	entry_point='pybullet_envs.gym_pendulum_envs:InvertedDoublePendulumBulletEnv',
	max_episode_steps=1000,
	reward_threshold=9100.0,
	)

register(
	id='InvertedPendulumSwingupBulletEnv-v0',
	entry_point='pybullet_envs.gym_pendulum_envs:InvertedPendulumSwingupBulletEnv',
	max_episode_steps=1000,
	reward_threshold=800.0,
	)

register(
	id='ReacherBulletEnv-v0',
	entry_point='pybullet_envs.gym_manipulator_envs:ReacherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
	)

register(
	id='PusherBulletEnv-v0',
	entry_point='pybullet_envs.gym_manipulator_envs:PusherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
)

register(
	id='ThrowerBulletEnv-v0',
	entry_point='pybullet_envs.gym_manipulator_envs:ThrowerBulletEnv',
	max_episode_steps=100,
	reward_threshold=18.0,
)

register(
	id='StrikerBulletEnv-v0',
	entry_point='pybullet_envs.gym_manipulator_envs:StrikerBulletEnv',
	max_episode_steps=100,
	reward_threshold=18.0,
)

register(
	id='Walker2DBulletEnv-v0',
	entry_point='pybullet_envs.gym_locomotion_envs:Walker2DBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)
register(
	id='HalfCheetahBulletEnv-v0',
	entry_point='pybullet_envs.gym_locomotion_envs:HalfCheetahBulletEnv',
	max_episode_steps=1000,
	reward_threshold=3000.0
	)

register(
	id='AntBulletEnv-v0',
	entry_point='pybullet_envs.gym_locomotion_envs:AntBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HopperBulletEnv-v0',
	entry_point='pybullet_envs.gym_locomotion_envs:HopperBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HumanoidBulletEnv-v0',
	entry_point='pybullet_envs.gym_locomotion_envs:HumanoidBulletEnv',
	max_episode_steps=1000
	)

register(
	id='HumanoidFlagrunBulletEnv-v0',
	entry_point='pybullet_envs.gym_locomotion_envs:HumanoidFlagrunBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2000.0
	)

register(
	id='HumanoidFlagrunHarderBulletEnv-v0',
	entry_point='pybullet_envs.gym_locomotion_envs:HumanoidFlagrunHarderBulletEnv',
	max_episode_steps=1000
	)

#register(
#	id='AtlasBulletEnv-v0',
#	entry_point='pybullet_envs.gym_locomotion_envs:AtlasBulletEnv',
#	max_episode_steps=1000
#	)

def getList():
	btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet')>=0]
	return btenvs
