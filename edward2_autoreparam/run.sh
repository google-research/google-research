#!/bin/bash

set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r edward2_autoreparam/requirements.txt
python -m edward2_autoreparam.run_experiments --method=baseline \
          --model=8schools --num_leapfrog_steps=1 \
          --num_mc_samples=1 --num_optimization_steps=5 --num_samples=5 \
          --burnin=2 --num_adaptation_steps=2 --results_dir=/tmp/results
python -m edward2_autoreparam.run_experiments --method=vip \
          --model=8schools --num_leapfrog_steps=1 \
          --num_mc_samples=1 --num_optimization_steps=5 --num_samples=5 \
          --burnin=2 --num_adaptation_steps=2 --results_dir=/tmp/results
python -m edward2_autoreparam.analyze_results --results_dir=/tmp/results \
          --model_and_dataset=8schools_na
