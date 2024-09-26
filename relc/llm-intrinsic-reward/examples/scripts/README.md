# Maintained scripts

This folder contains the actively maintained examples from TRL that showcase how
to use the library's trainers in different scenarios:

-   `sft_trainer.py`: This script shows how to use the `SFTTrainer` to fine tune
    a model or adapters into a target dataset.
-   `reward_trainer.py`: This script shows how to use the `RewardTrainer` to
    train a reward model on your own dataset.
-   `sentiment_tuning.py`: This script shows how to use the `PPOTrainer` to
    fine-tune a sentiment analysis model using IMDB dataset
-   `multi_adapter_rl.py`: This script shows how to use the `PPOTrainer` to
    train a single base model with multiple adapters. This scripts requires you
    to run the example script with the reward model training beforehand.
-   `stable_diffusion_tuning_example.py`: This script shows to use DDPOTrainer
    to fine-tune a stable diffusion model using reinforcement learning.

## Distributed training

All of the scripts can be run on multiple GPUs by providing the path of an 🤗
Accelerate config file when calling `accelerate launch`. To launch one of them
on $N$ GPUs, use:

```shell
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

You can also adjust the parameters of the 🤗 Accelerate config file to suit your
needs (e.g. training in mixed precision).

### Distributed training with DeepSpeed

Most of the scripts can be run on multiple GPUs together with DeepSpeed
ZeRO-{1,2,3} for efficient sharding of the optimizer states, gradients, and
model weights. To do so, run:

```shell
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```
