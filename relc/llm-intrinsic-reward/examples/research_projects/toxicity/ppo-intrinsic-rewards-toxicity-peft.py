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

r"""PPP with intrinsic rewards for detoxification task with LORA."""

from accelerate import Accelerator
import argparse
import re
import torch

from peft import LoraConfig
from os.path import join
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import pipeline

from trl import AutoModelForCausalLMWithValueHead
from trl import PPOConfig
from trl import PPOTrainer
from trl import set_seed
from trl.core import LengthSampler

from trl.extras.toxicity_evaluator import RobertaHateSpeechDetector
from trl.extras.toxicity_evaluator import PerspectiveAPI

# from trl.extras.openai_scores import query_batch_span
tqdm.pandas()



# Below is an example function to build the dataset. In our case, we use the
# IMDB dataset from the `datasets` library. One should customize this function
# to train the model on its own dataset.
def build_dataset(
    config, dataset_name="allenai/real-toxicity-prompts", toxicity_level=0.3
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`,
    one should customize this function to train the model on its own dataset.

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
    """
    Parse intrinsic rewards.

    Args:
        spans (`str`):
            The spans to parse.
        tokens (`str`):
            The tokens to parse.
        negative_reward (`float`):
            The negative reward to use for negative rewards.
        positive_reward (`float`):
            The positive reward to use for positive rewards.
    """
    results = []

    # For each item in spans and tokens
    for span, token_list in zip(spans, tokens):
        if "None identified" in span:
            span = span.replace("None identified", "")

        actual_spans = re.findall(r"Toxic Span \d+: (.+)", span)

        # If actual_spans is empty, use the entire span if it's not
        # just whitespace
        if not actual_spans and span.strip():
            combined_span = span.strip()
        else:
            combined_span = " ".join(actual_spans)

        # For each token in token_list, check if it's in the combined_span
        res = []
        for token in token_list:
            r = negative_reward if token in combined_span else positive_reward
            res.append(r)
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
        optimize_cuda_cache=True,
    )

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(config, toxicity_level=args.prompt_toxicity_level)

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        load_in_8bit=False,
        device_map={"": current_device},
        peft_config=lora_config,
    )
    ref_model = None
    optimizer = None

    # GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by
    # default. We need to set it to eos_token. only for this model.
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
    # compute the reward. We first load the toxicity model and tokenizer.
    if args.perspective_api is not None and len(args.perspective_api) > 0:
        toxicity_model = PerspectiveAPI(
            api_key=args.perspective_api, num_thread=5)
    else:
        toxicity_model = RobertaHateSpeechDetector(
            model_id="facebook/roberta-hate-speech-dynabench-r4-target",
            device=ppo_trainer.accelerator.device
        )

    # ========================== BUILD LLAMA-2 PIPELINE =======================
    model = "meta-llama/Llama-2-7b-chat-hf"
    llama2_tokenizer = AutoTokenizer.from_pretrained(model)
    llama2_pipeline = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        return_full_text=False,
    )
    if llama2_pipeline.tokenizer.pad_token_id is None:
        llama2_pipeline.tokenizer.pad_token_id = llama2_tokenizer.pad_token_id

    if llama2_pipeline.model.config.pad_token_id is None:
        llama2_pipeline.model.config.pad_token_id = llama2_tokenizer.pad_token_id

    def process_llm_output(llm_output):
        if "\n\n" in llm_output:
            return llm_output.split("\n\n")[0].strip()
        else:
            return llm_output.strip()

    def query_batch_span(selected_queries, max_workers=2, max_tokens=40):
        sequences = llama2_pipeline(
            selected_queries,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=llama2_tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )
        return [process_llm_output(seq[0]["generated_text"]) for seq in sequences]
    # =========================================================================

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

    prompt = None
    if args.use_intrinsic_reward:
        with open(args.prompt_file, "r") as rf:
            prompt = rf.read()

    step = 0
    for _ in range(args.epochs):
        for _, batch in tqdm(enumerate(ppo_trainer.dataloader)):
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

            if args.use_intrinsic_reward:
                # Select samples with an extrinsic reward below the threshold
                # for intrinsic reward calculation
                selected_query_indices, selected_queries = [], []
                assert len(rewards) == config.batch_size
                for idx, r in enumerate(rewards):
                    if r.item() <= args.intrinsic_reward_threshold:
                        selected_query_indices.append(idx)

                        temp_q = batch["query"][idx] + batch["response"][idx]
                        llm_query = prompt.format(temp_q.replace("\n", " "))
                        selected_queries.append(llm_query)

                # Calcualte intrinsic rewards
                selected_query_llm_responses = []
                if len(selected_queries) > 0:
                    selected_query_llm_responses = query_batch_span(
                        selected_queries, max_workers=2)
                    print("LLM OUTPUT: ", selected_query_llm_responses[0])

                llm_responses = ["" for _ in range(config.batch_size)]
                idx_ = 0
                for s_idx in selected_query_indices:
                    llm_responses[s_idx] = selected_query_llm_responses[idx_]
                    idx_ += 1

                # Parse LLM responses to get numerical rewards
                batch_tokens =[]
                for i in range(config.batch_size):
                    batch_tokens.append([tokenizer.decode(t) for t in torch.cat((query_tensors[i], response_tensors[i]))])

                intrisic_rewards = parse_intrinsic_rewards(
                    llm_responses,
                    batch_tokens,
                    negative_reward=args.negative_reward_value,
                    positive_reward=args.positive_reward_value,
                )
            else:
                intrisic_rewards = []

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
        type=str, help="the model name")
    parser.add_argument(
        "--log_with",
        default=None,
        type=str, help="use 'wandb' to log with wandb")
    parser.add_argument(
        "--learning_rate",
        default=(1.47e-5) * 2, type=float, help="the learning rate")
    parser.add_argument(
        "--mini_batch_size",
        default=4, type=int, help="the PPO minibatch size")
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
    parser.add_argument(
        "--seed", default=0, type=int, help="the random seed")
    parser.add_argument(
        "--tracker_project_name",
        default="trl",
        type=str,
        help="wandb project name")
    parser.add_argument(
        "--num_shared_layers",
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
