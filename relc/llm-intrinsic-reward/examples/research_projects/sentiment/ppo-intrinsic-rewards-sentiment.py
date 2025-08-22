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

r"""This file is for PPO on sentiment control task with intrinsic rewards."""

from dataclasses import dataclass, field
import re
from typing import Optional

import torch
import tyro

from os.path import join
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2Seq
    PPOTrainer,
    set_seed,
    create_reference_model,
)
from trl import PPOConfig
from trl.core import LengthSampler
from trl.extras.openai_scores import query_batch_span

tqdm.pandas()


@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="lvwerra/gpt2-imdb",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate=1.41e-5,
            log_with=None,
            mini_batch_size=128,
            batch_size=128,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    model_save_path: Optional[str] = field(
        default="./gpt-2-sentiment",
        metadata={"help": "the path to save the model"},
    )
    query_dataset: str = field(
        default="imdb",
        metadata={"help": "the dataset to query"})
    use_seq2seq: bool = field(
        default=False,
        metadata={"help": "whether to use seq2seq models"})
    use_peft: bool = field(
        default=False, metadata={"help": "whether to use peft"})
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    min_new_tokens: int = field(
        default=10, metadata={"help": "Min number of new tokens to sample"})
    max_new_tokens: int = field(
        default=20, metadata={"help": "Max number of new tokens to sample"})
    prompt_file: str = field(
        default="./prompts/prompt_3shot_v3.txt",
        metadata={"help": "Prompt file path"}
    )
    use_instric_reward: bool = field(
        default=False, metadata={"help": "whether to use instrinsic rewards"})
    num_shared_layers: int = field(
        default=None, metadata={"help": "Number of layers to freeze"})
    epochs: int = field(default=1, metadata={"help": "Number of epoches"})
    positive_reward_value: float = field(
        default=0.0, metadata={"help": "Positive intrinsic reward value"})
    negative_reward_value: float = field(
        default=-1.0, metadata={"help": "Negative intrinsic reward value"})
    intrinsic_reward_threshold: float = field(
        default=10.0, metadata={"help": "Threshold for using intrinsic reward"})
    save_freq: Optional[int] = field(
        default=None, metadata={"help": "n steps to save the model"})


# Below is an example function to build the dataset. In our case, we use the
# IMDB dataset from the `datasets` library. One should customize this function
# to train the model on its own dataset.
def build_dataset(
    config,
    query_dataset,
    input_min_text_length=4,
    input_max_text_length=10
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`,
    one should customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(
            sample["review"],
            truncation=True,
            max_length=1024,
        )[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def get_score(output, label="POSITIVE"):
    for class_output in output:
        if class_output["label"] == label:
            return torch.tensor(class_output["score"])
    return None


def create_pseudo_token_rewards(seq_lens, rewards):
    token_rewards = []
    for l, r in zip(seq_lens, rewards):
        assert l > 0
        t_reward = torch.zeros(l)
        t_reward[-1] = r
        token_rewards.append(t_reward)
    return token_rewards


def parse_intrinsic_rewards(
    spans,
    tokens,
    reward_matching=-1.0,
    reward_no_matching=1.0
):
    """Parse intrinsic rewards."""
    results = []
    for span, token_list in zip(spans, tokens):
        if "None identified" in span:
            span = span.replace("None identified", "").strip()

        res = []
        if len(span) == 0:
            for _ in token_list:
                res.append(reward_no_matching)
        else:
            combined_span = re.sub(r"\[Span \d+\]:\s*", "", span)
            for token in token_list:
                if token.strip() != "" and token.strip() in combined_span:
                    res.append(reward_matching)
                else:
                    res.append(reward_no_matching)

        results.append(res)

    return results


def get_intrinsic_rewards(
    args,
    batch,
    query_tensors,
    response_tensors,
    prompt,
    tokenizer,
    rewards
):
    """Get intrinsic rewards."""
    # Select samples with an extrinsic reward below the threshold 
    # for intrinsic reward calculation
    selected_query_indices, selected_queries = [], []
    for idx, r in enumerate(rewards):
        if r.item() <= args.intrinsic_reward_threshold:
            selected_query_indices.append(idx)
            llm_query = prompt.format(
                (batch["query"][idx] + batch["response"][idx]).replace("\n", "")
            )
            selected_queries.append(llm_query)

    # Calcualte intrinsic rewards
    selected_query_llm_responses = []
    if len(selected_queries) > 0:
        selected_query_llm_responses = query_batch_span(
            selected_queries,
            max_workers=4,
            max_tokens=60,
        )

    llm_responses = ["" for _ in range(args.ppo_config.batch_size)]
    idx_ = 0
    for s_idx in selected_query_indices:
        llm_responses[s_idx] = selected_query_llm_responses[idx_]
        idx_ += 1

    # Parse LLM responses to get numerical rewards
    batch_tokens = []
    for i in range(args.ppo_config.batch_size):
        temp = []
        for t in torch.cat((query_tensors[i], response_tensors[i])):
            temp.append(tokenizer.decode(t))
        batch_tokens.append(temp)

    llm_pos_responses, llm_neg_responses = [], []
    for response in llm_responses:
        try:
            llm_pos_responses.append(
                response.split("Identified Negative Text Span:")[0].strip())
            llm_neg_responses.append(
                response.split("Identified Negative Text Span:")[1].strip())
        except Exception as err:
            print()
            print("Response:", response)
            llm_pos_responses.append(response)
            llm_neg_responses.append("None identified")

    positive_intrisic_rewards = parse_intrinsic_rewards(
        llm_pos_responses,
        batch_tokens,
        reward_matching=args.positive_reward_value,
        reward_no_matching=0.0,
    )

    negative_intrisic_rewards = parse_intrinsic_rewards(
        llm_neg_responses,
        batch_tokens,
        reward_matching=args.negative_reward_value,
        reward_no_matching=0.0,
    )

    def add_lists(list1, list2):
        summed = []
        for sublist1, sublist2 in zip(list1, list2):
            summed.append([a + b for a, b in zip(sublist1, sublist2)])
        return summed

    intrisic_rewards = add_lists(
        positive_intrisic_rewards, negative_intrisic_rewards)

    return intrisic_rewards


def main():
    args = tyro.cli(ScriptArguments)

    trl_model_class = AutoModelForCausalLMWithValueHead
    if args.use_seq2seq:
        trl_model_class = AutoModelForSeq2Seq

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(
        args.ppo_config,
        args.query_dataset,
        input_min_text_length=4,
        input_max_text_length=10
    )

    # set seed before initializing value head for deterministic eval
    set_seed(args.ppo_config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    model = trl_model_class.from_pretrained(
        args.ppo_config.model_name,
        trust_remote_code=True,
    )
    ref_model = create_reference_model(
        model, num_shared_layers=args.num_shared_layers)

    tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # We then build the PPOTrainer, passing the model, the reference model, 
    # the tokenizer
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    ppo_trainer = PPOTrainer(
        args.ppo_config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collator
    )

    # We then build the sentiment analysis pipeline, passing the model name and
    # the sentiment analysis pipeline arguments. Let's also make sure to set
    # the device to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    task, model_name = args.ppo_config.reward_model.split(":")
    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            sentiment_pipe = pipeline(task, model=model_name, device=device)
    else:
        sentiment_pipe = pipeline(task, model=model_name, device=device)

    # Some tokenizers like GPT-2's don't have a padding token by default
    if sentiment_pipe.tokenizer.pad_token_id is None:
        sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

    if sentiment_pipe.model.config.pad_token_id is None:
        sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

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
        args.min_new_tokens, args.max_new_tokens)

    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each
    # token.
    sent_kwargs = {"top_k": 2, "function_to_apply": "none", "batch_size": 16}

    prompt = None
    if args.use_instric_reward:
        with open(args.prompt_file, "r") as rf:
            prompt = rf.read()

    step = 0
    for epoch in range(args.epochs):
        for _, batch in tqdm(
            enumerate(ppo_trainer.dataloader),
            total=len(ppo_trainer.dataloader),
            desc=f"Epoch {epoch+1}"
        ):
            query_tensors = batch["input_ids"]

            # Get response from gpt2
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs
            )
            batch["response"] = tokenizer.batch_decode(response_tensors)

            # Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            rewards = [get_score(output) for output in pipe_outputs]

            # convert to pseudo token-level rewards
            seq_lens = []
            for qt, rt in zip(query_tensors, response_tensors):
                seq_lens.append(qt.shape[0] + rt.shape[0] - 1)
            token_rewards = create_pseudo_token_rewards(seq_lens, rewards)

            # calculate intrinsic rewards
            if args.use_instric_reward and prompt is not None:
                intrisic_rewards = get_intrinsic_rewards(
                    args,
                    batch,
                    query_tensors,
                    response_tensors,
                    prompt,
                    tokenizer,
                    rewards
                )

                for tok_rewards, i_rewards in zip(
                    token_rewards, intrisic_rewards):
                    i_rewards = i_rewards[1:]
                    assert len(tok_rewards) == len(i_rewards)
                    for tok_i in range(len(tok_rewards)):
                        tok_rewards[tok_i] += i_rewards[tok_i]

            # Run PPO step
            stats = ppo_trainer.step(
                query_tensors, response_tensors, token_rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            if args.save_freq and step % args.save_freq == 0:
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(
                        join(args.model_save_path, f"step_{step}/"))

            step += 1

    # save final model after training
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.save_pretrained(args.model_save_path)


if __name__ == "__main__":
    main()
