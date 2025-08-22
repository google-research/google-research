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

"""Launch multiprocess training job.

Launch multiple training processes and supplies them with the needed flags per
process. All flags passed to this script will be passed on verbatim
to study_recommend/study_recommend.py.
See study_recommend/study_recommend.py for all flags and descriptions.
If using this script then the study_recommend/study_recommend.py --num_processes
flags should be used with a value equal to the number of GPUs on the current
machine. The flag --coordinator_address should also be passed.
The flag --process_id should not be passed as this script will assign
a unique process ID to each subprocess and populate the --process_id
automatically in accordance to that.

If the environment you are running in has a managed way for launching
multiprocess jobs that is compatible with Jax we suggest you use that instead.
Managed launchers include SLURM or Open MPI. See
https://jax.readthedocs.io/en/latest/multi_process.html
for more details.

Do not use this script for launching jobs across multiple hosts/VMs as
it is not compatible with such usage.
"""
import argparse
import logging
import subprocess
import sys
import threading

TRAINING_SCRIPT = 'study_recommend.study_recommend'
PYTHON_INTERPRETER = 'python'


def launch_jax_process(process_id, process_args):
  """Launch a training process with process index=process_id."""
  # Build the command.
  args = [PYTHON_INTERPRETER, '-m', TRAINING_SCRIPT]
  args.extend(process_args)
  args.append(f'--process_id={process_id}')

  # Launch in a subprocess and stream stdout and stderr of subprocess
  # to stdout and stderr of this binary.
  logging.info('Launching subprocess with commmand %s', ' '.join(args))
  subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr).communicate()


def main():
  logging.basicConfig(level=logging.INFO)
  # We will incrementally build the subprocess commands for each process.
  # We start with the shared args.

  # We start with all args passed to this script (except the first which is
  # the path to this script). We will pass these args to TRAINING_SCRIPT
  all_args = sys.argv[1:]

  # Next we parse some of the passed args locally.
  parser = argparse.ArgumentParser(
      description=(
          'Launch multiprocess training job with 1 job per GPU. All args passed'
          f' transparentally to {TRAINING_SCRIPT} and the'
          ' --process_id flag is populated for each process by this script'
          ' automatically.'
      )
  )
  parser.add_argument('--num_processes', type=int, required=True)
  parser.add_argument(
      '--in_managed_parallel_env', action='store_true', default=False
  )
  args = parser.parse_known_args()[0]
  if args.in_managed_parallel_env:
    raise ValueError(
        'This script is not required for nor compatible with managed parallel'
        ' env mode.'
    )

  threads = []
  for i in range(args.num_processes):
    threads.append(
        threading.Thread(
            target=launch_jax_process,
            kwargs={'process_id': i, 'process_args': all_args},
        )
    )
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()


if __name__ == '__main__':
  main()
