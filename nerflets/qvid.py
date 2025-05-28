# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess as sp
import os
from joblib import Parallel, delayed
import tqdm

import sys
import argparse

parser = argparse.ArgumentParser(
    description='Create a video of rasterized blob weights')
parser.add_argument('--every_k', type=int, default=10)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=-1)
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--fine', action='store_true')

args = parser.parse_args()
#if len(sys.argv) >= 2:
#  every_k = int(sys.argv[1])
#else:
#  every_k = 10

if args.fine:
  sifstr = 'sif_fine'
else:
  sifstr = 'sif'

expname = args.exp_name


def write_im(idx, i):
  print(f'Processing {idx}, {i}')
  outdir = f'{args.log_dir}/{expname}/qvid_{sifstr}'
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
  out_path = f'{outdir}/{str(i).zfill(6)}.png'
  #cmd = f'./qview {args.log_dir}/{expname}/{sifstr}/sif_{str(idx).zfill(6)}.txt -camera 2.8018 3.33692 4.7267  -0.443111 -0.459386 -0.769816  -0.209809 0.888016 -0.409154 -show_axes -image {out_path}'
  #cmd = f'./qview {args.log_dir}/{expname}/{sifstr}/sif_{str(idx).zfill(6)}.txt -camera -1.965012 -2.70461 2.252512 0.487459 0.670931 -0.5587794 -0.328441  -0.4520623 -0.829316  -image {out_path}'
  cmd = f'./qview {args.log_dir}/{expname}/{sifstr}/sif_{str(idx).zfill(6)}.txt -camera -0.7261900305747986 -8.996326446533203 5.8916144371032715 0.067375 0.834668 -0.546617 -0.04398 -0.544845 -0.83738  -image {out_path}'
  cmd = f'./qview {args.log_dir}/{expname}/{sifstr}/sif_{str(idx).zfill(6)}.txt -camera -0.7261900305747986 -8.996326446533203 5.8916144371032715 0.067375 0.834668 -0.546617 0.04398 0.544845 0.83738  -image {out_path}'
  cmd = f'./qview {args.log_dir}/{expname}/{sifstr}/sif_{str(idx).zfill(6)}.txt -camera -0.7261900305747986 -8.996326446533203 5.8916144371032715 0.067375 0.834668 -0.546617 -0.9968 0.08046 0.0  -image {out_path}'
  sp.check_output(cmd, shell=True)


idx = args.start_idx
i = 0

idxs = []
inds = []
while True:
  print(f'Idx: {idx}')
  path = f'{args.log_dir}/{expname}/{sifstr}/sif_{str(idx).zfill(6)}.txt'
  if not os.path.isfile(path):
    print(f'No sif {path}')
    break
  #write_im(idx, i)
  idxs.append(idx)
  inds.append(i)
  if args.end_idx > 0 and idx >= args.end_idx:
    break
  idx += args.every_k
  i += 1
to_proc = list(zip(idxs, inds))
print(to_proc)
for p in tqdm.tqdm(to_proc):
  write_im(*p)
#Parallel(n_jobs=1, backend='threading')(delayed(write_im(*a))(a) for a in tqdm.tqdm(to_proc))

outvid = f'{args.log_dir}/{expname}/{sifstr}.mp4'
cmd = f'ffmpeg -y -i {args.log_dir}/{expname}/qvid_{sifstr}/%06d.png {outvid}'
#cmd = f'convert -delay 10 -loop 1 /home/kgenova/nerflet/logs/{expname}/qvid/*.png /home/kgenova/nerflet/logs/{expname}/sifs.gif'
sp.check_output(cmd, shell=True)
sp.check_output(f'mplayer {outvid} -loop 0', shell=True)
