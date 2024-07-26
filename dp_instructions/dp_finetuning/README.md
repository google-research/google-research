# Instruction Tuning with Differential Privacy

## Intro

This code implements DP fine-tuning of LLaMA 7B/13B models. To reduce memory cost, we use LoRA fine-tuning that freezes pre-trained weights and only update a tiny fraction of adapter parameters.

Currently, this code supports the following types of experiments.

**Type A**: Fine-tune LLaMA 7B/13B to generate synthetic instructions.

**Type B**: Sampling from models fine-tuned from Type A to generate synthetic instructions.

**Type C**: Fine-tune LLaMA 7B/13B with supervised (instruction, response) pairs.

**Type D**: Sampling from models fine-tuned from Type C to generate responses for given instructions.

Please see example commands below.

This implementation is built on the following packages: [transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets), [accelerate](https://github.com/huggingface/accelerate), and [PEFT](https://github.com/huggingface/peft).

Functions that support DP training are in ```utils/dp_utils.py```. Main functions are:
1. Per-example gradient computation with Pytorch forward/backward hooks for Linear layers.
2. Per-example gradient clipping and accumulation.
3. Adding noise to the aggregated gradient.

## Setup
```
pip install -r requirements.txt
```

## Example commands

### For Type A experiments

Fine-tune LLaMA 13B to generate synthetic instructions without DP. These commands use the ```chatbot_arena_instructions_train180k``` dataset in the data folder. The checkpoints will be saved to ```outputs/[job_sess]``` folder.
```
python generate_train_command.py --dataset_name chatbot_arena_instructions_train180k --model_name yahma/llama-13b-hf --job_sess debug --eps -1 --perdevice_bs 4 --gpus 1 --max_seq_len 1024 --total_bs 32 --num_epochs 3 --prompt_style uncond_generation --no_eval_at_start
```

Fine-tune LLaMA 13B to generate synthetic instructions with (2.86, 5e-7)-DP. Training with a single A100 GPU takes around 5 days.
```
python generate_train_command.py --dataset_name chatbot_arena_instructions_train180k --model_name yahma/llama-13b-hf --job_sess debug --eps 2.86 --delta 5e-7 --perdevice_bs 4 --gpus 1 --max_seq_len 1024 --total_bs 4096 --num_epochs 10 --prompt_style uncond_generation --lr 1e-3 --clip 0.5 --no_eval_at_start
```

Fine-tune LLaMA 13B to generate synthetic instructions with (5.94, 5e-7)-DP.
```
python generate_train_command.py --dataset_name chatbot_arena_instructions_train180k --model_name yahma/llama-13b-hf --job_sess debug --eps 5.94 --delta 5e-7 --perdevice_bs 4 --gpus 1 --max_seq_len 1024 --total_bs 4096 --num_epochs 10 --prompt_style uncond_generation --lr 1e-3 --clip 0.5 --no_eval_at_start
```

### For Type B experiments

Sampling from a fine-tuned checkpoint for unconditional generation. The outputs will be saved to ```inference_outputs/[job_sess]``` folder.
```
python generate_inference_command.py --instruction_file utils/syn_instruct.txt --model_name yahma/llama-13b-hf --adapter_path [path to the fine-tuned apater] --job_sess debug --max_seq_len 1024 --top_p 0.95 --prompt_style uncond_generation
```

### For Type C experiments

Fine-tune LLaMA 7B on pairs of real/syn instructions and responses. You will first need to label the real instructions in ```data/chatbot_arena_instructions_train180k``` or the synthetic instructions from the previous step. See the ```query_gpt35_for_annotation``` folder for code for labelling with GPT-3.5-Turbo. Note that you need to first extract the instructions into a python list (each entry is simply a string) and dump it into a .pkl file.

After labelling, the dataset should be prepared as a huggingface ```DatasetDict``` object with train, val, and test splits. Each split is a huggingface ```Dataset``` object with two columns: ```instruction``` and ```answer```. **The file name of the dataset should start with ```labelled```**.

Fine-tune LLaMA 7B on pairs of real instructions and responses with eps=5.94.
```
python generate_train_command.py --dataset_name labelled_real_arena180k --model_name yahma/llama-7b-hf --job_sess debug --eps 5.94 --delta 5e-7 --perdevice_bs 2 --gpus 1 --max_seq_len 2048 --total_bs 4096 --num_epochs 10 --prompt_style vicuna --lr 1e-3 --clip 1.0
```

Fine-tune LLaMA 7B on pairs of synthetic instructions and responses without DP.
```
python generate_train_command.py --dataset_name labelled_syn300k_13b_eps594_uncond_syn_clip05_13b_selected_sigma10_1000bins --model_name yahma/llama-7b-hf --job_sess debug --eps -1 --perdevice_bs 4 --gpus 1 --max_seq_len 1024 --total_bs 32 --num_epochs 3 --prompt_style vicuna
```


### For Type D experiments

Use a fine-tuned checkpoint to annotate responses for Chatbot Arena or AlpacaEval instructions. The outputs will be saved to ```inference_outputs/[job_sess]``` folder.
```
python generate_inference_command.py --instruction_file utils/alpaca_eval_instructions.txt --model_name yahma/llama-13b-hf --adapter_path [path to the fine-tuned apater] --job_sess debug --max_seq_len 1024 --top_p 0.95 --prompt_style vicuna
```

