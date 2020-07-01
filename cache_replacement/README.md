# Introduction

*Authors*: Evan Zheran Liu, Milad Hashemi, Kevin Swersky, Parthasarathy
Ranganathan, Junwhan Ahn

Source code accompanying our ICML 2020 paper: [An Imitation Learning Approach for
Cache Replacement](https://arxiv.org/abs/2006.16239).

# Setup

Install the necessary Python3 packages, e.g., in a `virtualenv`:

```
# Current working directory is cache_replacement
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
# Install baselines separately, since it depends on installing Tensorflow first.
pip install -e git+https://github.com/openai/baselines.git@ea25b9e8b234e6ee1bca43083f8f3cf974143998#egg=baselines
```

# Collecting Traces
Traces should be placed in the `cache_replacement/cache/traces`,
which currently contains a toy sample trace
`cache_replacement/cache/traces/sample_trace.csv`.
Unfortunately, we are unable to release the exact traces used in our experiments, but provide two ways to obtain similar traces.

## Using Traces from the 2nd Cache Replacement Championship (Recommended)
The recommended and easier method to collect traces is to use the traces
released in the [2nd Cache Replacement
Championship (CRC2)](https://crc2.ece.tamu.edu/?page_id=41).

First, clone the ChampSim GitHub repo and apply our patch for collecting LLC
access traces.

```
# Current working directory is cache_replacement
git clone https://github.com/ChampSim/ChampSim
cd ChampSim
git checkout 8798bed8117b2873e34c47733d2ba4f79b6014d4
git apply ../environment/traces/champsim.patch
./build_champsim.sh bimodal no no no no create_llc_trace 1
```

Next, download the desired CRC2 traces from the [Dropbox
link](https://www.dropbox.com/sh/pgmnzfr3hurlutq/AACciuebRwSAOzhJkmj5SEXBa/CRC2_trace?dl=0&subfolder_nav_tracking=1)
found at the [CRC2 website](https://crc2.ece.tamu.edu/?page_id=41),
e.g., `astar_313B.trace.xz`.
Place this file in the current working directory under `ChampSim`.
The following command will create a LLC access trace named `llc_access_trace.csv`:

```
./run_champsim.sh bimodal-no-no-no-no-create_llc_trace-1core 0 313000 astar_313B.trace.xz
```

Finally, filter this trace to use the 64 cache sets used in our paper, and split
into train / valid / test (80 / 10 / 10):

```
cd ..
python3 policy_learning/cache/traces/train_test_split.py ChampSim/llc_access_trace.csv
```

Place these traces into the trace directory, e.g.:

```
mv train.csv policy_learning/cache/traces/astar_train.csv
mv valid.csv policy_learning/cache/traces/astar_valid.csv
mv test.csv policy_learning/cache/traces/astar_test.csv
```

## Collecting Custom Traces with DynamoRIO
We recommend using the already collected traces from the 2nd Cache Replacement
Championship following the instructions above.
However, custom [SPEC2006](https://www.spec.org/cpu2006/) traces can be collected with
[DynamoRIO](https://github.com/DynamoRIO/dynamorio) and should be placed into
the same trace directory `cache_replacement/cache/traces`.
Concretely, the following commands collect these traces:

```
$ git clone https://github.com/DynamoRIO/dynamorio.git
$ cd dynamorio
$ git reset --hard a49184ad1c7a6b9f0f078c70248ddc4f9959f065
$ git apply < miss_tracer.diff
$ cmake . && make -j
$ ./bin64/drrun -t drcachesim -trace_after_instrs "${NUM_INSTS_TO_SKIP}"
-exit_after_tracing "${TRACE_LENGTH}" -simulator_type miss_tracer -cores 1
-L1D_size 32768 -L1D_assoc 4 -LL_size 262144 -LL_assoc 8 -LL_miss_file trace.csv
-- "${PROG}" "${ARGS}"
```

# Cache Replacement Environment
We release a cache replacement OpenAI gym environment in
`cache_replacement/environment/`, along with various baselines, including S4LRU
(Huang, et. al., 2013), LRU, Belady's, and the Nearest Neighbors baseline from
our paper.
See `cache_replacement/environment/main.py` for an example usage script.

# Cache Simulation Usage

Simulate policies through the cache by passing the appropriate eviction policy
configs, the appropriate cache configs, and a selected memory trace.

Example usage with Belady's as the policy, and default SPEC cache configs:

```
# Current working directory is google_research
python3 -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_belady_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --cache_configs=cache_replacement/policy_learning/cache/configs/eviction_policy/belady.json \
  --memtrace_file=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Cache hit rate statistics will be logged to tensorboard files in
`/tmp/sample_belady_llc`.

# Cache Replacement Policy Learning

Train our model (Parrot) to learn access patterns from a particular trace by passing the
appropriate configurations.

Example usage with our full model with all additions, trained and validated on
the sample trace:

```
# Current working directory is google_research
python3 -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

The number of learned embeddings can be set with the `--model_bindings` flag
(e.g., set to 5000 above).
In our experiments, we set the number of learned embeddings to be the number of
unique memory addresses in the training split.
Hit rate statistics and accuracies will be logged to tensorboard files in
`/tmp/sample_model_llc`.

We also provide commands to run the various ablations reported in the paper.
Training with the byte embedder model:

```
# Current working directory is google_research
python3 -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_configs=cache_replacement/policy_learning/cache_model/configs/default.json \
  --model_configs=cache_replacement/policy_learning/cache_model/configs/byte_embedder.json \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Changing the history length to, e.g., 100:

```
# Current working directory is google_research
python3 -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --model_bindings="sequence_length=100" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Ablating the reuse distance auxiliary loss:

```
# Current working directory is google_research
python3 -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Ablating the ranking loss:

```
# Current working directory is google_research
python3 -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"log_likelihood\", \"reuse_dist\"]" \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

Ablating DAgger:

```
# Current working directory is google_research
python3 -m cache_replacement.policy_learning.cache_model.main \
  --experiment_base_dir=/tmp \
  --experiment_name=sample_model_llc \
  --cache_configs=cache_replacement/policy_learning/cache/configs/default.json \
  --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
  --dagger_schedule_bindings=["initial=0", "update_freq=1000000000000", "final=0", "num_steps=1"] \
  --model_bindings="address_embedder.max_vocab_size=5000" \
  --train_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv \
  --valid_memtrace=cache_replacement/policy_learning/cache/traces/sample_trace.csv
```

# Evaluating the Learned Policy

The commands for training the Parrot policy in the previous section also
periodically save model checkpoints.
Below, we provide commands for evaluating the saved checkpoints on a test set.
In the paper, we choose the checkpoint with the highest validation cache hit
rate, which can be done by inspecting the tensorboard files in the training
directory `/tmp/sample_model_llc`.
The following command evaluates the model checkpoint saved after 20000 steps on
the trace `cache_replacement/policy_learning/cache/traces/sample_trace.csv`:

```
# Current working directory is google_research
python3 -m cache_replacement.policy_learning.cache.main \
  --experiment_base_dir=/tmp \
  --experiment_name=evaluate_checkpoint \
  --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
  --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" \
  --memtrace_file="cache_replacement/policy_learning/cache/traces/sample_trace.csv" \
  --config_bindings="associativity=16" \
  --config_bindings="capacity=2097152" \
  --config_bindings="eviction_policy.scorer.checkpoint=\"/tmp/sample_model_llc/checkpoints/20000.ckpt\"" \
  --config_bindings="eviction_policy.scorer.config_path=\"/tmp/sample_model_llc/model_config.json\"" \
  --warmup_period=0
```

This logs the final cache hit rate to tensorboard files in the directory
`/tmp/evaluate_checkpoint`.

# Citation

```
@article{liu2020imitation,
  title={An imitation learning approach for cache replacement},
  author={Liu, Evan Zheran and Hashemi, Milad and Swersky, Kevin and Ranganathan, Parthasarathy and Ahn, Junwhan},
  journal={arXiv preprint arXiv:2006.16239},
  year={2020}
}
```
