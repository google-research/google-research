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

"""Dataset handling."""

import argparse
import collections
import os
import pickle

import cv2
import numpy as np

from ravens import cameras
from ravens import tasks
from ravens import utils

# See transporter.py, regression.py, dummy.py, task.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


class Dataset:
  """Base dataset class."""

  def __init__(self, path):
    """A simple RGB-D image dataset."""
    self.path = path
    self.episode_id = []
    self.episode_set = []

    # Track existing dataset if it exists.
    color_path = os.path.join(self.path, 'color')
    if os.path.exists(color_path):
      for fname in sorted(os.listdir(color_path)):
        if '.pkl' in fname:
          num_samples = int(fname[(fname.find('-') + 1):-4])
          self.episode_id += [self.num_episodes] * num_samples

    self._cache = dict()

    # Goal-conditioned training, IF sampling goals other than final images.
    self.subsample_goals = False

  @property
  def num_episodes(self):
    return np.max(np.int32(self.episode_id)) + 1 if self.episode_id else 0

  def add(self, episode, last_stuff=None):
    """Add an episode to the dataset.

    The dataset is sampled from during training of Transporters. We also
    save a separate set of data into directories prefixed with `last_`
    which are used for goal-conditioned training, or debugging of the
    dataset to check that the final configuration of objects is as
    expected.

    Args:
      episode: List of tuples representing one demonstrator episode. These
        contain all time steps except for the very last time, e.g., for
        the insertion task (which is normally one action) the episode has
        only the starting image and the sole action.
      last_stuff: tuple of (last_obs, last_info), representing the stuff
        that happened right after the last action. This information is
        necessary for goal-conditioned training, but is not used for
        training vanilla Transporters, since there's no action associated
        with these observations. The first tuple item contains images, the
        second tuple item contains pose-based information for evaluation.
    """
    color, depth, action, info = [], [], [], []
    for obs, act, i in episode:
      color.append(obs['color'])
      depth.append(obs['depth'])
      action.append(act)
      info.append(i)

    color = np.uint8(color)
    depth = np.float32(depth)

    def dump(data, field):
      field_path = os.path.join(self.path, field)
      if not os.path.exists(field_path):
        os.makedirs(field_path)
      fname = f'{self.num_episodes:06d}-{len(episode)}.pkl'
      pickle.dump(data, open(os.path.join(field_path, fname), 'wb'))

    dump(color, 'color')
    dump(depth, 'depth')
    dump(action, 'action')
    dump(info, 'info')

    # Save final images (e.g., for goal conditioning) in separate files.
    if last_stuff is not None:
      last_color = np.uint8(last_stuff[0]['color'])
      last_depth = np.float32(last_stuff[0]['depth'])
      dump(last_color, 'last_color')
      dump(last_depth, 'last_depth')
      dump(last_stuff[1], 'last_info')

    self.episode_id += [self.num_episodes] * len(episode)

  def set(self, episodes):
    """Limit random samples to specific fixed set."""
    self.episode_set = episodes

  def random_sample(self, goal_images=False):
    """Randomly sample from the dataset uniformly.

    The 'cached_load' will to load the list, then extract the time step
    `i` within it as the data point. We also support a `goal_images`
    feature to load the last information. The last information isn't in a
    list, so there's no need to extract an index. That is, if loading an
    episode of length 1m we should see this in color vs last_color:

    In:  data = pickle.load( open('color/000099-1.pkl', 'rb') )
    In:  data.shape
    Out: (1, 3, 480, 640, 3)

    In:  data = pickle.load( open('last_color/000099-1.pkl', 'rb') )
    In:  data.shape
    Out: (3, 480, 640, 3)

    Args:
      goal_images: if True, load the last information.

    Returns:
      (obs, act, info): Tuple containing random sample.
    """
    if self.episode_set:
      iepisode = np.random.choice(self.episode_set)
    else:
      iepisode = np.random.choice(range(self.num_episodes))

    is_episode_sample = np.int32(self.episode_id) == iepisode
    episode_samples = np.argwhere(is_episode_sample).squeeze().reshape(-1)
    i = np.random.choice(range(len(episode_samples)))

    def load(iepisode, field):
      field_path = os.path.join(self.path, field)
      fname = f'{iepisode:06d}-{len(episode_samples)}.pkl'
      return pickle.load(open(os.path.join(field_path, fname), 'rb'))

    def cached_load(iepisode, field, index):

      # to do without cache:
      # return load(iepisode, field)[index]

      if iepisode not in self._cache:
        self._cache[iepisode] = dict()
      if field not in self._cache[iepisode]:
        self._cache[iepisode][field] = load(iepisode, field)
      return self._cache[iepisode][field][index]

    obs = {}
    obs['color'] = cached_load(iepisode, 'color', i)
    obs['depth'] = cached_load(iepisode, 'depth', i)
    act = cached_load(iepisode, 'action', i)
    info = cached_load(iepisode, 'info', i)

    # Load goal images if training goal-conditioned policies.
    if goal_images:
      ep_len = len(episode_samples)
      assert i < ep_len, f'{i} vs {ep_len}'
      goal = {}

      # Subsample the goal, from i+1 up to (and INCLUDING) the episode length.
      if self.subsample_goals:
        low = i + 1
        high = ep_len
        new_i = np.random.choice(range(low, high + 1))  # NOTE: high+1
      else:
        new_i = ep_len

      if new_i < ep_len:
        # Load a list and index in it by `new_i`.
        goal['color'] = cached_load(iepisode, 'color', new_i)
        goal['depth'] = cached_load(iepisode, 'depth', new_i)
        goal['info'] = cached_load(iepisode, 'info', new_i)
      else:
        # Stand-alone information about the final set of images.
        goal['color'] = load(iepisode, 'last_color')
        goal['depth'] = load(iepisode, 'last_depth')
        goal['info'] = load(iepisode, 'last_info')

      assert goal['color'].shape == (3, 480, 640, 3), goal['color'].shape
      assert goal['depth'].shape == (3, 480, 640), goal['depth'].shape
      return obs, act, info, goal

    return obs, act, info

  def inspect(self, save_imgs=True):
    """Inspect the data, purely for debugging and data analysis.

    Saves to `data_out` or `goals_out`, depending on if this is training
    data or data for held-out goal images (for goal-based Transporters).

    Args:
      save_imgs: True if saving images, False if otherwise. If saving
        images, we save one image for each time step, _including_ the
        final image taken after the last action (not normally part of
        training data as there's no action associated with it). Each
        image consists of three stacked together: (a) image from the
        frontal camera view, (b) fused color image, and (c) fused
        height map. The neural networks see (b) and (c), though for
        saving purposes here, (c) has values scaled to be in [0,255].
    """
    if 'data' in self.path:
      outdir = (self.path).replace('data/', 'data_out/')
    elif 'goals' in self.path:
      outdir = (self.path).replace('goals/', 'goals_out/')
    if os.path.exists(outdir):
      import shutil  # pylint: disable=g-import-not-at-top
      print(f'Removing: {outdir}')
      shutil.rmtree(outdir)
    os.makedirs(outdir)
    print(f'Saving to: {outdir}')
    print(f'episode_set: {self.episode_set}')
    print(f'num_episodes: {self.num_episodes}')

    def _load(i_ep, episode_len, field):
      field_path = os.path.join(self.path, field)
      fname = f'{i_ep:06d}-{episode_len}.pkl'
      return pickle.load(open(os.path.join(field_path, fname), 'rb'))

    # For data analysis later to evalute demonstrator quality. Anything for
    # 'maxlen_stats' should only be applied on episodes with max length.
    ep_lengths = []
    all_stats = collections.defaultdict(list)
    maxlen_stats = collections.defaultdict(list)

    for i_ep in range(self.num_episodes):
      if i_ep % 20 == 0:
        print(f'\ton episode {i_ep}')

      # is_episode_sample: list of [F,F,T,T,F,F,..], T for this epis only.
      is_episode_sample = np.int32(self.episode_id) == i_ep
      episode_samples = np.argwhere(is_episode_sample).squeeze().reshape(-1)
      episode_len = len(episode_samples)
      ep_lengths.append(episode_len)
      color_l = _load(i_ep, episode_len, 'color')
      depth_l = _load(i_ep, episode_len, 'depth')
      info_l = _load(i_ep, episode_len, 'info')

      # Save images if desired, but can take a lot of time.
      if save_imgs:
        self._save_images(episode_len, color_l, depth_l, info_l, outdir, i_ep)

      # If we have 'last' info, inspect it.
      if not os.path.exists(os.path.join(self.path, 'last_info')):
        continue
      last_color = _load(i_ep, episode_len, 'last_color')
      last_depth = _load(i_ep, episode_len, 'last_depth')
      last_info = _load(i_ep, episode_len, 'last_info')

      # Add stuff to `ep_` lists.
      self._track_data_statistics(info_l, last_info, episode_len, all_stats,
                                  maxlen_stats)

      if save_imgs:
        # See transporter.py, mirroring Transporter.get_heightmap().
        obs_input = {'color': last_color, 'depth': last_depth}
        colormap, heightmap = get_heightmap(obs_input)
        heightmap_proc = process_depth(img=heightmap)

        # Same as earlier, fusing images together.
        c_img_front = last_color[0]  # Shape (480, 640, 3)
        c_img_front = cv2.resize(c_img_front,
                                 (426, 320))  # numpy shape: (320,426)
        barrier = np.zeros((320, 4, 3))  # Black barrier of 4 pixels
        combo = np.concatenate(
            (cv2.cvtColor(c_img_front, cv2.COLOR_BGR2RGB), barrier,
             cv2.cvtColor(colormap,
                          cv2.COLOR_RGB2BGR), barrier, heightmap_proc),
            axis=1)

        # Optionally include title with more details, but env dependent.
        suffix_all = f'{i_ep:06d}-{episode_len:02d}-FINAL.png'
        suffix_all = self._change_name(suffix_all, last_info['extras'])
        cv2.imwrite(os.path.join(outdir, suffix_all), combo)

    # Debugging / inspection. First, debug all episodes.
    max_l = get_max_episode_len(self.path)
    ep_lengths = np.array(ep_lengths)
    print(f'\nStats over {self.num_episodes} demos:')
    print(
        f'ep len avg:     {np.mean(ep_lengths):0.3f} +/- {np.std(ep_lengths):0.1f}'
    )
    print(f'ep len median:  {np.median(ep_lengths):0.3f}')
    print(f'ep len min/max: {np.min(ep_lengths)}, {np.max(ep_lengths)}')
    num_max = np.sum(ep_lengths == max_l)
    print(f'ep equal to max len ({max_l}): {num_max} / {self.num_episodes}')
    print('You can approximate this as failures, but might overcount).\n')

    # Consider env-specific properties, prefacing with [maxlen] as needed.
    # Though we'll get NaNs if no episodes got to the max -- but thats OK. :)
    print('Now environment-specific statistics:')

    if 'cable-shape' in self.path or 'cable-line-notarget' in self.path:
      # For these tasks, I want to see performance as a function of nb_sides.
      low, high = 1, 1  # for cable-line-notarget (one side only)
      if 'cable-shape' in self.path:
        low, high = 2, 4
      for nb_sides in range(low, high + 1):
        md = maxlen_stats[f'done_{nb_sides}']
        mf = maxlen_stats[f'frac_{nb_sides}']
        ad = all_stats[f'done_{nb_sides}']
        af = all_stats[f'frac_{nb_sides}']
        al = all_stats[f'len_{nb_sides}']
        print(f'[maxlen] {nb_sides}_done: {np.sum(md)} / {len(md)}')
        print(
            f'[maxlen] {nb_sides}_frac: {np.mean(mf):0.3f} +/- {np.std(mf):0.1f}'
        )
        print(f'[alleps] {nb_sides}_done: {np.sum(ad)} / {len(ad)}')
        print(
            f'[alleps] {nb_sides}_frac: {np.mean(af):0.3f} +/- {np.std(af):0.1f}'
        )
        print(
            f'[alleps] {nb_sides}_len:  {np.mean(al):0.2f} +/- {np.std(al):0.1f}\n'
        )

    elif 'cable-ring' in self.path:
      # A bit tricky to interpret area, so I am using percentage improvement.
      md = maxlen_stats['done']
      mf = maxlen_stats['fraction']
      mf_c = maxlen_stats['fraction_delta']
      ma_pi = maxlen_stats['percent_improve']
      ad = all_stats['done']
      af = all_stats['fraction']
      af_c = all_stats['fraction_delta']
      aa_pi = all_stats['percent_improve']
      print(f'[maxlen] done:     {np.sum(md)} / {len(md)}')
      print(f'[maxlen] fraction: {np.mean(mf):0.3f} +/- {np.std(mf):0.1f}')
      print(f'[maxlen] f-delta:  {np.mean(mf_c):0.3f} +/- {np.std(mf_c):0.1f}')
      print(
          f'[maxlen] % area:   {np.mean(ma_pi):0.3f} +/- {np.std(ma_pi):0.1f}')
      print(f'[alleps] done:     {np.sum(ad)} / {len(ad)}')
      print(f'[alleps] fraction: {np.mean(af):0.3f} +/- {np.std(af):0.1f}')
      print(f'[alleps] f-delta:  {np.mean(af_c):0.3f} +/- {np.std(af_c):0.1f}')
      print(
          f'[alleps] % area:   {np.mean(aa_pi):0.3f} +/- {np.std(aa_pi):0.1f}')

    elif 'cloth-flat-' in self.path:
      # Report coverage deltas.
      md = maxlen_stats['done']
      mc = maxlen_stats['cloth_coverage']
      mc_d = maxlen_stats['coverage_delta']
      ad = all_stats['done']
      ac = all_stats['cloth_coverage']
      ac_d = all_stats['coverage_delta']
      print(f'[maxlen] done:      {np.sum(md)} / {len(md)}')
      print(f'[maxlen] cov-final: {np.mean(mc):0.3f} +/- {np.std(mc):0.1f}')
      print(f'[maxlen] cov-delta: {np.mean(mc_d):0.3f} +/- {np.std(mc_d):0.1f}')
      print(f'[alleps] done:      {np.sum(ad)} / {len(ad)}')
      print(f'[alleps] cov-final: {np.mean(ac):0.3f} +/- {np.std(ac):0.1f}')
      print(f'[alleps] cov-delta: {np.mean(ac_d):0.3f} +/- {np.std(ac_d):0.1f}')

    elif 'cloth-cover' in self.path:
      # As of Aug 12, this env should have all rollouts with 2 actions.
      md = maxlen_stats['done']
      print(f'[maxlen] done: {np.sum(md)} / {len(md)}')

    elif 'bag-alone-open' in self.path:
      # Similar to cable-ring.
      md = maxlen_stats['done']
      mf = maxlen_stats['fraction']
      mf_c = maxlen_stats['fraction_delta']
      ma_pi = maxlen_stats['percent_improve']
      ad = all_stats['done']
      af = all_stats['fraction']
      af_c = all_stats['fraction_delta']
      aa_pi = all_stats['percent_improve']
      print(f'[maxlen] done:     {np.sum(md)} / {len(md)}')
      print(f'[maxlen] fraction: {np.mean(mf):0.3f} +/- {np.std(mf):0.1f}')
      print(f'[maxlen] f-delta:  {np.mean(mf_c):0.3f} +/- {np.std(mf_c):0.1f}')
      print(
          f'[maxlen] % area:   {np.mean(ma_pi):0.3f} +/- {np.std(ma_pi):0.1f}')
      print(f'[alleps] done:     {np.sum(ad)} / {len(ad)}')
      print(f'[alleps] fraction: {np.mean(af):0.3f} +/- {np.std(af):0.1f}')
      print(f'[alleps] f-delta:  {np.mean(af_c):0.3f} +/- {np.std(af_c):0.1f}')
      print(
          f'[alleps] % area:   {np.mean(aa_pi):0.3f} +/- {np.std(aa_pi):0.1f}')

    elif 'bag-items-easy' in self.path or 'bag-items-hard' in self.path:
      # Unlike other tasks, here we already did heavy data filtering beforehand.
      md = maxlen_stats['done']
      m_ts = maxlen_stats['task_stage']
      m_zi = maxlen_stats['zone_items_rew']
      m_zb = maxlen_stats['zone_beads_rew']
      ad = all_stats['done']
      a_ts = all_stats['fraction']
      a_zi = all_stats['zone_items_rew']
      a_zb = all_stats['zone_beads_rew']
      print(f'[maxlen] done:       {np.sum(md)} / {len(md)}')
      print(
          f'[maxlen] task_stage: {np.mean(m_ts):0.3f} +/- {np.std(m_ts):0.1f}')
      print(
          f'[maxlen] zone_items: {np.mean(m_zi):0.3f} +/- {np.std(m_zi):0.1f}')
      print(
          f'[maxlen] zone_beads: {np.mean(m_zb):0.3f} +/- {np.std(m_zb):0.1f}')
      print(f'[alleps] done:       {np.sum(ad)} / {len(ad)}')
      print(
          f'[alleps] task_stage: {np.mean(a_ts):0.3f} +/- {np.std(a_ts):0.1f}')
      print(
          f'[alleps] zone_items: {np.mean(a_zi):0.3f} +/- {np.std(a_zi):0.1f}')
      print(
          f'[alleps] zone_beads: {np.mean(a_zb):0.3f} +/- {np.std(a_zb):0.1f}')

  def _change_name(self, suff, info_extra):
    """Depending on the env, make changes to image suffix `suff`."""
    if 'cable-ring' in self.path:
      i1 = info_extra['convex_hull_area']
      i2 = info_extra['best_possible_area']
      f = i1 / i2
      suff = suff.replace('.png',
                          f'-area-{i1:0.3f}-best-{i2:0.3f}-FRAC-{f:0.3f}.png')
    elif 'cloth-flat' in self.path:
      i1 = info_extra['cloth_coverage']
      suff = suff.replace('.png', f'-coverage-{i1:0.3f}.png')
    elif 'bag-alone' in self.path:
      i1 = info_extra['convex_hull_area']
      i2 = info_extra['best_possible_area']
      suff = suff.replace('.png', f'-area-{i1:0.3f}-best-{i2:0.3f}.png')
    else:
      pass
    return suff

  def _track_data_statistics(self, info_l, last_info, episode_len, all_stats,
                             maxlen_stats):  # pylint: disable=g-doc-args
    """To get more fine-grained analysis of environment performance.

    Many of these require the last_info, which is not saved in info_l,
    which has all the time steps BEFORE that. For cable-ring and
    bag-alone-open, we terminate based on a fraction.
    """
    maxlen = get_max_episode_len(self.path)
    start = info_l[0]['extras']
    last_ex = last_info['extras']

    if 'cable-shape' in self.path or 'cable-line-notarget' in self.path:
      nb_sides = start['nb_sides']
      frac_beads = last_ex['nb_zone'] / last_ex['nb_beads']
      if episode_len == maxlen:
        maxlen_stats[f'done_{nb_sides}'].append(last_ex['task.done'])
        maxlen_stats[f'frac_{nb_sides}'].append(frac_beads)
      all_stats[f'done_{nb_sides}'].append(last_ex['task.done'])
      all_stats[f'frac_{nb_sides}'].append(frac_beads)
      all_stats[f'len_{nb_sides}'].append(episode_len)

    elif 'cable-ring' in self.path:
      delta = last_ex['fraction'] - start['fraction']
      percent = last_ex['convex_hull_area'] - start['convex_hull_area']
      percent = 100 * percent / start['convex_hull_area']
      if episode_len == maxlen:
        maxlen_stats['done'].append(last_ex['task.done'])
        maxlen_stats['fraction'].append(last_ex['fraction'])
        maxlen_stats['fraction_delta'].append(delta)
        maxlen_stats['percent_improve'].append(percent)
      all_stats['done'].append(last_ex['task.done'])
      all_stats['fraction'].append(last_ex['fraction'])
      all_stats['fraction_delta'].append(delta)
      all_stats['percent_improve'].append(percent)

    elif 'cloth-flat' in self.path:
      delta = last_ex['cloth_coverage'] - start['cloth_coverage']
      if episode_len == maxlen:
        maxlen_stats['done'].append(last_ex['task.done'])
        maxlen_stats['coverage_delta'].append(delta)
        maxlen_stats['cloth_coverage'].append(last_ex['cloth_coverage'])
      all_stats['done'].append(last_ex['task.done'])
      all_stats['coverage_delta'].append(delta)
      all_stats['cloth_coverage'].append(last_ex['cloth_coverage'])

    elif 'cloth-cover' in self.path:
      if episode_len == maxlen:
        maxlen_stats['done'].append(last_ex['task.done'])

    elif 'bag-alone-open' in self.path:
      delta = last_ex['fraction'] - start['fraction']
      percent = last_ex['convex_hull_area'] - start['convex_hull_area']
      percent = 100 * percent / start['convex_hull_area']
      if episode_len == maxlen:
        maxlen_stats['done'].append(last_ex['task.done'])
        maxlen_stats['fraction'].append(last_ex['fraction'])
        maxlen_stats['fraction_delta'].append(delta)
        maxlen_stats['percent_improve'].append(percent)
      all_stats['done'].append(last_ex['task.done'])
      all_stats['fraction'].append(last_ex['fraction'])
      all_stats['fraction_delta'].append(delta)
      all_stats['percent_improve'].append(percent)

    elif 'bag-items-easy' in self.path or 'bag-items-hard' in self.path:
      # For this it'd be interesting to see what task stage we're at.
      if episode_len == maxlen:
        maxlen_stats['done'].append(last_ex['task.done'])
        maxlen_stats['task_stage'].append(last_ex['task_stage'])
        maxlen_stats['zone_items_rew'].append(last_ex['zone_items_rew'])
        maxlen_stats['zone_beads_rew'].append(last_ex['zone_beads_rew'])
      all_stats['done'].append(last_ex['task.done'])
      all_stats['task_stage'].append(last_ex['task_stage'])
      all_stats['zone_items_rew'].append(last_ex['zone_items_rew'])
      all_stats['zone_beads_rew'].append(last_ex['zone_beads_rew'])

  def _save_images(self, episode_len, color_l, depth_l, info_l, outdir, i_ep):
    """For each item (timestep) in this episode, save relevant images."""

    for t in range(episode_len):
      assert color_l[t].shape == (3, 480, 640, 3), color_l[t].shape
      assert depth_l[t].shape == (3, 480, 640), depth_l[t].shape

      # Recall that I added 'extras' to the info dict at each time.
      info = info_l[t]
      info_r = info['extras']

      # We saved three color/depth images per time step.
      for k in range(3):
        c_img = color_l[t][k]
        d_img = depth_l[t][k]
        assert c_img.dtype == 'uint8', c_img.dtype
        assert d_img.dtype == 'float32', d_img.dtype
        d_img = process_depth(img=d_img)

      # Andy uses U.reconstruct_heightmap(color, depth, configs, ...)
      obs_input = {'color': color_l[t], 'depth': depth_l[t]}
      colormap, heightmap = get_heightmap(obs_input)
      heightmap_proc = process_depth(img=heightmap)

      # Save image that combines the interesting ones above, makes it
      # easier to copy and paste. Horizontally concatenate images and
      # save. Also convert to BGR because OpenCV assumes that format
      # but PyBullet uses RGB (to be consistent). Network should be
      # seeing RGB images I believe (but just be consistent).
      c_img_front = color_l[t][0]  # Shape (480, 640, 3)
      c_img_front = cv2.resize(c_img_front,
                               (426, 320))  # numpy shape: (320,426)
      barrier = np.zeros((320, 4, 3))  # Black barrier of 4 pixels
      combo = np.concatenate(
          (cv2.cvtColor(c_img_front, cv2.COLOR_BGR2RGB), barrier,
           cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR), barrier, heightmap_proc),
          axis=1)

      # Optionally include title with more details, but env dependent.
      suffix_all = f'{i_ep:06d}-{t:02d}-OVERALL.png'
      suffix_all = self._change_name(suffix_all, info_r)
      cv2.imwrite(os.path.join(outdir, suffix_all), combo)


# ---------------------------------------------------------------------------- #
# Helper methods for inspecting the data
# ---------------------------------------------------------------------------- #


def get_max_episode_len(path):
  """A somewhat more scalable way to get the max episode lengths."""
  path = path.replace('data/', '')
  path = path.replace('goals/', '')
  task = tasks.names[path]()
  max_steps = task.max_steps - 1  # Remember, subtract one!
  return max_steps


def process_depth(img, cutoff=10):
  """Make depth images human-readable."""

  # Turn to three channels and zero-out values beyond cutoff.
  w, h = img.shape
  d_img = np.zeros([w, h, 3])
  img = img.flatten()
  img[img > cutoff] = 0.0
  img = img.reshape([w, h])
  for i in range(3):
    d_img[:, :, i] = img

  # Scale values into [0,255) and make type uint8.
  assert np.max(d_img) > 0.0
  d_img = 255.0 / np.max(d_img) * d_img
  d_img = np.array(d_img, dtype=np.uint8)
  for i in range(3):
    d_img[:, :, i] = cv2.equalizeHist(d_img[:, :, i])
  return d_img


def get_heightmap(obs):
  """Following same implementation as in transporter.py."""
  heightmaps, colormaps = utils.reconstruct_heightmaps(obs['color'],
                                                       obs['depth'],
                                                       CAMERA_CONFIG, BOUNDS,
                                                       PIXEL_SIZE)
  colormaps = np.float32(colormaps)
  heightmaps = np.float32(heightmaps)

  # Fuse maps from different views.
  valid = np.sum(colormaps, axis=3) > 0
  repeat = np.sum(valid, axis=0)
  repeat[repeat == 0] = 1
  colormap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
  colormap = np.uint8(np.round(colormap))
  heightmap = np.max(heightmaps, axis=0)
  return colormap, heightmap


def main():
  # Call the code like this: `python ravens/dataset.py --path data/[...]`.
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', default='data/cable-shape')
  args = parser.parse_args()
  args.path = (args.path).rstrip('/')
  assert os.path.exists(args.path), args.path
  assert 'data' in args.path or 'goals' in args.path, \
          'Data should be stored in `data/` or `goals/`.'
  dataset = Dataset(args.path)
  dataset.inspect(save_imgs=True)

  # TODO(daniel) this is perhaps an older way of analyzing the data?
  #     # Compute color and depth statistics.
  #     parser = argparse.ArgumentParser()
  #     parser.add_argument('--path',   default='data/stacking')
  #     args = parser.parse_args()
  #     dataset = Dataset(args.path)
  #     color_mean = np.mean(dataset.data['color'] / 255)
  #     depth_mean = np.mean(dataset.data['depth'])
  #     color_std = np.std(dataset.data['color'] / 255)
  #     depth_std = np.std(dataset.data['depth'])
  #     print(f'Dataset path: {args.path}')
  #     print(f'  Color mean: {color_mean:.8f}')
  #     print(f'  Depth mean: {depth_mean:.8f}')
  #     print(f'  Color std:  {color_std:.8f}')
  #     print(f'  Depth std:  {depth_std:.8f}')


if __name__ == '__main__':
  main()
