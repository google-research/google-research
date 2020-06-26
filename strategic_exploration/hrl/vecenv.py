# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Adapted from OpenAI baselines
import abc
import numpy as np
import pickle
from multiprocessing import Process, Pipe


class VecEnv(object):
  __metaclass__ = abc.ABCMeta
  """
    An abstract asynchronous, vectorized environment.
    """

  def __init__(self, num_envs, observation_space, action_space):
    self.num_envs = num_envs
    self.observation_space = observation_space
    self.action_space = action_space

  @abc.abstractmethod
  def reset(self):
    """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
    pass

  @abc.abstractmethod
  def step_async(self, actions):
    """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
    pass

  @abc.abstractmethod
  def step_wait(self):
    """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
    pass

  @abc.abstractmethod
  def close(self):
    """
        Clean up the environments' resources.
        """
    pass

  def step(self, actions):
    self.step_async(actions)
    return self.step_wait()

  def render(self, mode='human'):
    pass
    #logger.warn('Render not defined for %s'%self)

  @property
  def unwrapped(self):
    if isinstance(self, VecEnvWrapper):
      return self.venv.unwrapped
    else:
      return self


class CloudpickleWrapper(object):
  """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to
    use pickle)
    """

  def __init__(self, x):
    self.x = x

  def __getstate__(self):
    import cloudpickle
    return cloudpickle.dumps(self.x)

  def __setstate__(self, ob):
    import pickle
    self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
  parent_remote.close()
  env = env_fn_wrapper.x()
  while True:
    cmd, data = remote.recv()
    if cmd == 'step':
      ob, reward, done, info = env.step(data)
      if done:
        ob = env.reset()
      remote.send((ob, reward, done, info))
    elif cmd == 'reset':
      ob = env.reset()
      remote.send(ob)
    elif cmd == 'render':
      remote.send(env.render(mode='rgb_array'))
    elif cmd == 'close':
      remote.close()
      break
    elif cmd == 'get_spaces':
      remote.send((env.observation_space, env.action_space))
    else:
      raise NotImplementedError


class SubprocVecEnv(VecEnv):

  def __init__(self, env_fns, spaces=None):
    """
        envs: list of gym environments to run in subprocesses
        """
    self.waiting = False
    self.closed = False
    nenvs = len(env_fns)
    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
    self.ps = [
        Process(
            target=worker,
            args=(work_remote, remote, CloudpickleWrapper(env_fn)))
        for (work_remote, remote,
             env_fn) in zip(self.work_remotes, self.remotes, env_fns)
    ]
    for p in self.ps:
      p.daemon = True  # if the main process crashes, we should not cause things to hang
      p.start()
    for remote in self.work_remotes:
      remote.close()

    self.remotes[0].send(('get_spaces', None))
    observation_space, action_space = self.remotes[0].recv()
    VecEnv.__init__(self, len(env_fns), observation_space, action_space)

  def step_async(self, actions):
    for remote, action in zip(self.remotes, actions):
      remote.send(('step', action))
    self.waiting = True

  def step_wait(self):
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False
    obs, rews, dones, infos = zip(*results)
    return obs, rews, dones, infos

  def reset(self):
    for remote in self.remotes:
      remote.send(('reset', None))
    return [remote.recv() for remote in self.remotes]

  def close(self):
    if self.closed:
      return
    if self.waiting:
      for remote in self.remotes:
        remote.recv()
    for remote in self.remotes:
      remote.send(('close', None))
    for p in self.ps:
      p.join()
    self.closed = True

  def render(self, mode='human'):
    raise NotImplementedError()
