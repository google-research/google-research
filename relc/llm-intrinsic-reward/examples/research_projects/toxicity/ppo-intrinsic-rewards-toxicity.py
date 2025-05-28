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

r"""PPP with intrinsic rewards for detoxification task."""

import argparse
import re
import torch

from os.path import join
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead
from trl import PPOConfig
from trl import PPOTrainer
from trl import create_reference_model
from trl import set_seed
from trl.core import LengthSampler

from trl.extras.toxicity_evaluator import (
    RobertaHateSpeechDetector,
    PerspectiveAPI
)
from trl.extras.openai_scores import query_batch_span
tqdm.pandas()


# Below is an example function to build the dataset. In our case, we use the
# IMDB dataset from the `datasets` library. One should customize this function
# to train the model on its own dataset.
def build_dataset(
    config, dataset_name="allenai/real-toxicity-prompts", toxicity_level=0.3
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="train")

    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > toxicity_level

    ds = ds.filter(filter_fn, batched=False)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"]["text"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def parse_intrinsic_rewards(
    spans,
    tokens,
    negative_reward=-1.0,
    positive_reward=1.0
):
    """Parse intrinsic rewards.

    Args:
        spans (`str`):
            The spans to parse.
        tokens (`str`):
            The tokens to parse.
        negative_reward (`float`):
            The negative reward value.
        positive_reward (`float`):
            The positive reward value.
    """
    results = []

    # For each item in spans and tokens
    for span, token_list in zip(spans, tokens):
        if span is None:
            results.append([0.0 for _ in token_list])
        else:
            if "None identified" in span:
                span = span.replace("None identified", "")

            actual_spans = re.findall(r"Toxic Span \d+: (.+)", span)

            # If actual_spans is empty, use the entire span if it's not
            # just whitespace
            if not actual_spans and span.strip():
                combined_span = span.strip()
            else:
                combined_span = ' '.join(actual_spans)

            # For each token in token_list, check if it's in the combined_span
            res = [negative_reward if token in combined_span else positive_reward for token in token_list]

            results.append(res)

    return results


def main(args):
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        tracker_project_name=args.tracker_project_name,
    )

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(config, toxicity_level=args.prompt_toxicity_level)

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    if args.bf16:
        # Now let's build the model, the reference model, and the tokenizer.
        # We first load the model in bfloat16 to save
        # memory using `transformers`.
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16)
        # And then pass the loaded model to `AutoModelForCausalLMWithValueHead`.
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name)

    # We create a reference model by sharing 20 layers
    ref_model = create_reference_model(
        model, num_shared_layers=args.num_shared_layers)

    # We make sure to use `Adam` optimizer on the model parameters
    # that require gradients.
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=config.learning_rate)

    # GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token
    # by default. We need to set it to eos_token.
    # only for this model.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # We then build the PPOTrainer, passing the model, the reference model,
    # the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then build the reward pipeline, we will use the toxicity model to
    # compute the reward.
    # We first load the toxicity model and tokenizer.
    if args.perspective_api is not None and len(args.perspective_api) > 0:
        toxicity_model = PerspectiveAPI(
            api_key=args.perspective_api, num_thread=5)
    else:
        toxicity_model = RobertaHateSpeechDetector(
            model_id="facebook/roberta-hate-speech-dynabench-r4-target",
            device=ppo_trainer.accelerator.device
        )

    # We then define the arguments to pass to the `generate` function.
    # These arguments are passed to the `generate` function of the PPOTrainer,
    # which is a wrapper around the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    output_length_sampler = LengthSampler(
        args.output_min_length, args.output_max_length)

    if args.use_intrinsic_reward:
        with open(args.prompt_file, "r") as rf:
            prompt = rf.read()

    step = 0
    for epoch in range(args.epochs):
        for _, batch in tqdm(enumerate(ppo_trainer.dataloader),
                             total=len(ppo_trainer.dataloader),
                             desc=f"Epoch {epoch+1}"):
            query_tensors = batch["input_ids"]

            # Get response from the policy model
            response_tensors = []
            for query in query_tensors:
                gen_len = output_length_sampler()
                generation_kwargs["max_new_tokens"] = gen_len
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze()[-gen_len:])
            batch["response"] = [
                tokenizer.decode(r.squeeze()) for r in response_tensors]

            # Compute sentiment score # noqa
            assert len(batch["response"]) == config.batch_size
            rewards = toxicity_model.evaluate(batch["response"])

            intrisic_rewards = []
            if args.use_intrinsic_reward:
                selected_query_indices, selected_queries = [], []
                assert len(rewards) == config.batch_size
                for idx, r in enumerate(rewards):
                    if r.item() <= args.intrinsic_reward_threshold:
                        selected_query_indices.append(idx)

                        temp_q = batch["query"][idx] + batch["response"][idx]
                        llm_query = prompt.format(
                            temp_q.replace("\n", " "))
                        selected_queries.append(llm_query)

                # Calcualte intrinsic rewards
                selected_query_llm_responses = []
                if len(selected_queries) > 0:
                    selected_query_llm_responses = query_batch_span(
                        selected_queries, max_workers=2)

                llm_responses = ["" for _ in range(config.batch_size)]
                idx_ = 0
                for s_idx in selected_query_indices:
                    llm_responses[s_idx] = selected_query_llm_responses[idx_]
                    idx_ += 1

                # Parse LLM responses to get numerical rewards
                batch_tokens =[]
                for i in range(config.batch_size):
                    temp = []
                    for t in torch.cat((query_tensors[i], response_tensors[i])):
                        temp.append(tokenizer.decode(t))
                    batch_tokens.append(temp)

                intrisic_rewards = parse_intrinsic_rewards(
                    llm_responses,
                    batch_tokens,
                    negative_reward=args.negative_reward_value,
                    positive_reward=args.positive_reward_value,
                )

            token_rewards = []
            for i in range(config.batch_size):
                tok_r = torch.zeros(
                    len(query_tensors[i]) + len(response_tensors[i]) - 1)
                tok_r[-1] = rewards[i]

                if args.use_intrinsic_reward:
                    if intrisic_rewards[i] is not None and \
                            len(intrisic_rewards[i]) == len(tok_r) + 1:
                        tok_r = torch.tensor(intrisic_rewards[i][1:])
                        tok_r[-1] += rewards[i]

                token_rewards.append(tok_r)

            # Run PPO step
            stats = ppo_trainer.step(
                query_tensors, response_tensors, token_rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            # Save model every 100 epochs
            if step % 100 == 0:
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(
                        join(args.model_save_path, f"step_{step}/"))
            step += 1

    # save final model after training
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.save_pretrained(args.model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="ybelkada/gpt-j-6b-sharded-bf16",
        type=str,
        help="the model name")
    parser.add_argument(
        "--log_with",
        default=None,
        type=str,
        help="use 'wandb' to log with wandb")
    parser.add_argument(
        "--learning_rate",
        default=(1.47e-5) * 2,
        type=float,
        help="the learning rate")
    parser.add_argument(
        "--mini_batch_size",
        default=4,
        type=int,
        help="the PPO minibatch size")
    parser.add_argument(
        "--batch_size", default=16, type=int, help="the batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="the number of gradient accumulation steps"
    )
    parser.add_argument(
        "--model_save_path",
        default="./gpt-j-6B-detoxified-long-context-26-shl-1e4-final",
        type=str,
        help="the path to save the model"
    )
    parser.add_argument(
        "--ppo_epochs",
        default=100,
        type=int,
        help="Number of optimisation epochs per batch of samples"
    )
    parser.add_argument("--seed", default=0, type=int, help="the random seed")
    parser.add_argument(
        "--tracker_project_name",
        default="trl",
        type=str,
        help="wandb project name")
    parser.add_argument("--num_shared_layers",
        default=None,
        type=int,
        help="Number of shared layers between reference model and model"
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument(
        "--output_min_length",
        default=10,
        type=int,
        help="Minimum length of generated responses"
    )
    parser.add_argument(
        "--output_max_length",
        default=20,
        type=int,
        help="Maximum length of generated responses"
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument(
        "--prompt_toxicity_level",
        default=0.3,
        type=float,
        help="Train the model using prompts above given toxicity level"
    )
    parser.add_argument(
        "--perspective_api",
        default=None,
        type=str,
        help="Perspective API key for toxicity evaluation"
    )
    parser.add_argument(
        "--prompt_file",
        default="./prompts/prompt_toxicity_3shot_v2.txt",
        type=str,
        help="Prompt file path"
    )
    parser.add_argument(
        "--positive_reward_value",
        default=1.0,
        type=float,
        help="Positive intrinsic reward value"
    )
    parser.add_argument(
        "--negative_reward_value",
        default=-1.0,
        type=float,
        help="Negative intrinsic reward value"
    )
    parser.add_argument(
        "--use_intrinsic_reward",
        action="store_true",
        help="whether to instrinsic rewards"
    )
    parser.add_argument(
        "--intrinsic_reward_threshold",
        default=1.0,
        type=float,
        help="Threshold to use intrinsic rewards."
    )

    args = parser.parse_args()
    main(args)