# Reinforcement Learning with LLM Generated Intrinsic Rewards

## Summary

In this library, we implemnt the PPO algorithm with LLM generated token-level
intrinsic rewards for text generation tasks. We introduce a novel framework
leveraging the critiqueability of LLM to produce dense intrinsic rewards
throughout the learning process. Our approach incorporates a secondary critic
model alongside the policy model. This critic, takes the task description,
policy model's output, and environment's reward signal as input, provides token
or span-level intrinsic rewards that reflect the quality of each segment of the
output. We assess our approach on three text generation tasks: sentiment
control, language model detoxification, and summarization. Part of the
implementation is based on the [`TRL`](https://huggingface.co/docs/trl/)
library.

**Tasks:**

-   `Sentiment Control`: Steer the LM toward generating positive movie reviews.
-   `Detoxification`: Prevent the lanugage model from generating offensive,
    harmful, or biased content.
-   `Summarization`: Generating a concise and coherent summary of a longer text.
    <!-- - `Question Answer`: Answering a given question based on the information in the passages. -->

## Installation

```bash
pip install -e .
```

## Sentiment Control

The code for the sentiment control experiments is under:

```bash
cd examples/research_projects/sentiment/
```

Run PPO baseline:

```bash
source activate PPOIntrinsic
CKPT_DIR=$HOME/ppo-intrinsic-reward/ckpts

accelerate launch \
    --main_process_port 9876 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --mixed_precision bf16 \
    --multi_gpu \
    --dynamo_backend no ppo-intrinsic-rewards-sentiment.py \
        --epochs 2 \
        --query_dataset "imdb" \
        --min_new_tokens 15 \
        --max_new_tokens 20 \
        --num_shared_layers 20 \
        --save_freq 50 \
        --model_save_path $CKPT_DIR/sentiment/ppo_baseline_epoch2_bs16_mbs16 \
        --ppo_config.model_name "gpt2-large" \
        --ppo_config.learning_rate 1.41e-5 \
        --ppo_config.batch_size 16 \
        --ppo_config.mini_batch_size 16 \
        --ppo_config.gradient_accumulation_steps 1 \
        --ppo_config.target_kl 6.0 \
        --ppo_config.kl_penalty "kl" \
        --ppo_config.ppo_epochs 4;
```

Run PPO with intrinsic rewards:

```bash
source activate PPOIntrinsic
CKPT_DIR=$HOME/ppo-intrinsic-reward/ckpts

accelerate launch \
    --main_process_port 9876 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --mixed_precision bf16 \
    --multi_gpu \
    --dynamo_backend no ppo-intrinsic-rewards-sentiment.py \
        --use_instric_reward \
        --positive_reward_value 0.5 \
        --negative_reward_value -0.5 \
        --intrinsic_reward_threshold 10 \
        --prompt_file $HOME/ppo-intrinsic-reward/prompts/prompt_sentiment_3shot_v2.txt \
        --epochs 2 \
        --query_dataset "imdb" \
        --min_new_tokens 15 \
        --max_new_tokens 20 \
        --num_shared_layers 20 \
        --save_freq 50 \
        --model_save_path $CKPT_DIR/sentiment/ppo_intrinsic_epoch2_bs16_mbs16  \
        --ppo_config.model_name "gpt2-large" \
        --ppo_config.learning_rate 1.41e-5 \
        --ppo_config.batch_size 16 \
        --ppo_config.mini_batch_size 16 \
        --ppo_config.gradient_accumulation_steps 1 \
        --ppo_config.target_kl 6.0 \
        --ppo_config.kl_penalty "kl" \
        --ppo_config.ppo_epochs 4;
```

To evalute on the test set, run:

```bash
source activate PPOIntrinsic

MODEL_NAME_OR_PATH=$CKPT_DIR/sentiment/ppo_intrinsic_epoch2_bs16_mbs16
PROMPT_PATH=./eval_scripts/neutral_prompts.jsonl
OUTPUT_FILE=./outputs/ppo_intrinsic_neu_prompts.jsonl

python eval_sentiment_sst.py \
    $MODEL_NAME_OR_PATH \
    $PROMPT_PATH \
    --output_file $OUTPUT_FILE \
    --num_return 25 \
    --classifier_path distilbert-base-uncased-finetuned-sst-2-english;
```

For perplexity and diversity evaluation:

```bash
source activate PPOIntrinsic

GENERATIONS_FILE=./outputs/ppo_intrinsic_neu_prompts.jsonl
OUTPUT_FILE=./outputs/ppo_intrinsic_neu_prompts_eval.txt
echo $OUTPUT_FILE

CUDA_VISIBLE_DEVICES=0 python eval_perplexity_dist.py \
    --generations_file $GENERATIONS_FILE \
    --output_file $OUTPUT_FILE;
```

## Detoxification

<!-- Introduce the task here -->
The code for the detoxification experiments is under:

```bash
cd examples/research_projects/toxicity/
```

To run the detoxification experiment, add a
[PERSPECTIVE_API_KEY](https://developers.perspectiveapi.com/s/docs-get-started?language=en_US)
in `.bashrc`.

```bash
export PERSPECTIVE_API_KEY="YOUR_API_KEY"
```

Run PPO baseline:

```bash
source activate PPOIntrinsic
CKPT_DIR=$HOME/ppo-intrinsic-reward/ckpts

accelerate launch \
    --main_process_port 9987 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no ppo-intrinsic-rewards-toxicity.py \
        --perspective_api $PERSPECTIVE_API_KEY \
        --model_name gpt2-large \
        --model_save_path $CKPT_DIR/toxicity/ppo_baseline_epoch5_bs32_mbs32 \
        --epochs 5 \
        --prompt_toxicity_level 0.6 \
        --output_min_length 10 \
        --output_max_length 14 \
        --learning_rate 1.41e-5 \
        --batch_size 32 \
        --mini_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --ppo_epochs 4;
```

Run PPO with intrinsic rewards:

```bash
source activate PPOIntrinsic
CKPT_DIR=$HOME/ppo-intrinsic-reward/ckpts

accelerate launch \
    --main_process_port 9987 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no ppo-intrinsic-rewards-toxicity.py \
        --use_intrinsic_reward \
        --prompt_file $HOME/ppo-intrinsic-reward/prompts/prompt_toxicity_3shot_v2.txt \
        --positive_reward_value 0.0 \
        --negative_reward_value -0.5 \
        --intrinsic_reward_threshold 1.0 \
        --perspective_api $PERSPECTIVE_API_KEY \
        --model_name gpt2-large \
        --model_save_path $CKPT_DIR/toxicity/ppo_intrinsic_epoch5_threshold1.0_bs32_mbs32 \
        --epochs 5 \
        --prompt_toxicity_level 0.6 \
        --output_min_length 10 \
        --output_max_length 14 \
        --learning_rate 1.41e-5 \
        --batch_size 32 \
        --mini_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --ppo_epochs 4;
```

For the self-critique experiment, run:

```bash
source activate PPOIntrinsic
CKPT_DIR=$HOME/ppo-intrinsic-reward/ckpts

accelerate launch \
    --main_process_port 9999 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no ppo-intrinsic-rewards-toxicity-peft.py \
        --use_intrinsic_reward \
        --prompt_file $HOME/ppo-intrinsic-reward/prompts/prompt_toxicity_3shot_v2.txt \
        --positive_reward_value 0.0 \
        --negative_reward_value -0.5 \
        --intrinsic_reward_threshold 0.9 \
        --perspective_api $PERSPECTIVE_API_KEY \
        --model_name meta-llama/Llama-2-7b-chat-hf \
        --model_save_path $CKPT_DIR/toxicity/ppo_intrinsic_llama2_epoch3_bs16_mbs8 \
        --epochs 3 \
        --prompt_toxicity_level 0.6 \
        --output_min_length 10 \
        --output_max_length 14 \
        --learning_rate 1.41e-5 \
        --batch_size 16 \
        --mini_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --ppo_epochs 4;
```

For evaluation, first generate N=25 continuatins for each prompt in `nontoxic_prompts-10k_test.jsonl`:

```bash
source activate PPOIntrinsic

LM_DIR=$CKPT_DIR/toxicity/ppo_intrinsic_epoch5_threshold1.0_bs32_mbs32
PROMPTS_FILE=./eval_scripts/nontoxic_prompts-10k_test.jsonl
OUTPUT_FILE=./toxicity/outputs/ppo_intrinsic_gens.jsonl
echo $OUTPUT_FILE

CUDA_VISIBLE_DEVICES=0 python decoding.py \
    --lm_dir $LM_DIR \
    --prompts_file $PROMPTS_FILE \
    --output_file $OUTPUT_FILE \
    --top_p 0.9 \
    --max_length 20 \
    --num_returns 25 \
    --do_sample \
    --batch_size 8 \
    --sample_size 2000;
```

Then, run Perspective API to evalute model's generations:

```bash
source activate PPOIntrinsic

DATA_FILE=./outputs/ppo_intrinsic_gens.jsonl
OUTPUT_FILE=./outputs/ppo_intrinsic_gens.jsonl

python eval_toxicity_perspective.py \
    --data_file $DATA_FILE \
    --output_file $OUTPUT_FILE \
    --sample_size 5000 \
    --num_thread 8 \
    --save_scores;
```

To evaluate the perplexity and diversity of the model's output, run:

```bash
source activate PPOIntrinsic

GENS_FILE=./outputs/ppo_intrinsic_gens.jsonl
OUTPUT_FILE=./outputs/ppo_intrinsic_gens_eval_results.txt
echo $OUTPUT_FILE

CUDA_VISIBLE_DEVICES=0 python eval_perplexity_dist.py \
    --generations_file $GENS_FILE \
    --output_file $OUTPUT_FILE;
```

## Summarization

The code for the summarization task is under:

```bash
cd examples/research_projects/summarization/
```

For the summarization experiment, first need to install the `nltk` and
`evaluate` package:

```bash
pip install evaluate
pip intall nltk
```

Run PPO exeriment:

```bash
source activate PPOIntrinsic
CKPT_DIR=$HOME/ppo-intrinsic-reward/ckpts

accelerate launch \
    --main_process_port 8765 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no ppo_summ_rouge.py \
        --num_shared_layers 12 \
        --model_name $CKPT_DIR/summarization/sft_gpt2-medium/checkpoint-1000 \
        --model_save_path $CKPT_DIR/summarization/ppo-baseline-rouge \
        --epochs 5 \
        --output_min_length 30 \
        --output_max_length 50 \
        --learning_rate 1.41e-5 \
        --batch_size 8 \
        --mini_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --ppo_epochs 4 \
        --prompt_file $HOME/ppo-intrinsic-reward/prompts/prompt_summ_3shot_rouge_v2.txt;
```

Run PPO with intrinsic rewards:

```bash
source activate PPOIntrinsic
CKPT_DIR=$HOME/ppo-intrinsic-reward/ckpts

accelerate launch \
    --main_process_port 8765 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no ppo_summ_rouge.py \
        --use_instric_reward \
        --positive_reward_value 0.5 \
        --negative_reward_value -0.5 \
        --intrinsic_reward_threshold 100 \
        --use_score_scaling \
        --num_shared_layers 12 \
        --model_name $CKPT_DIR/summarization/sft_gpt2-medium/checkpoint-1000 \
        --model_save_path $CKPT_DIR/summarization/ppo-intrinsic-rouge \
        --epochs 5 \
        --output_min_length 30 \
        --output_max_length 50 \
        --learning_rate 1.41e-5 \
        --batch_size 8 \
        --mini_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --ppo_epochs 4 \
        --prompt_file $HOME/ppo-intrinsic-reward/prompts/prompt_summ_3shot_rouge_v2.txt;
```

For ROUGE and preference evaluatin, first download the 6B reward model:

```bash
mkdir rm_checkpoint
wget https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin -O rm_checkpoint/pytorch_model.bin
```

Then run:

```bash
source activate PPOIntrinsic
MODEL=$HOME/ppo-intrinsic-reward/ckpts/summarization/ppo-intrinsic-rouge

accelerate launch \
    --main_process_port 8765 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no summ_eval.py \
        --model_name $MODEL \
        --save_path eval_results/sft_gpt2-medium_bs64_step1000.csv \
        --num_samples_to_eval 6000 \
        --reward_model_ckpt_path ./rm_checkpoint/pytorch_model.bin \
        --output_max_length 50 \
        --batch_size 20 \
        --rw_batch_size 20 \
        --reward_model_device 0;
```
