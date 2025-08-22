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

r"""This file is for PPO training on sentiment control task."""

import argparse
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import pipeline

from trl import AutoModelForCausalLMWithValueHead
from trl import PPOConfig
from trl import PPOTrainer
from trl import set_seed
from trl.core import LengthSampler


tqdm.pandas()


# Below is an example function to build the dataset. In our case, we use the
# IMDB dataset from the `datasets` library. One should customize this function
# to train the model on its own dataset.
def build_dataset(
    tokenizer,
    dataset_name="imdb",
    input_min_text_length=2,
    input_max_text_length=8,
    filter_negative=False
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
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    if filter_negative:
        ds = ds.filter(
            lambda x: len(x["review"]) > 200 and x["label"] == 1,
            batched=False
        )
    else:
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main(args):
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping=args.early_stopping,
        target_kl=args.target_kl,
        kl_penalty=args.kl_penalty,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        tracker_project_name=args.tracker_project_name,
    )

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # GPT-2 tokenizer has a pad token, but it is not eos_token by default.
    # We need to set it to eos_token. only for this model.
    tokenizer.pad_token = tokenizer.eos_token

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(tokenizer, filter_negative=args.filter_negative)

    # We then build the PPOTrainer, passing the model, the reference model,
    # the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collator
    )

    # We then build the sentiment analysis pipeline, passing the model name and
    # the sentiment analysis pipeline arguments. Let's also make sure to set the
    # device to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="lvwerra/distilbert-imdb",
        device=device,
        function_to_apply="none",
        batch_size=args.batch_size,
        return_all_scores=True,
    )

    # We then define the arguments to pass to the `generate` function. These
    # arguments are passed to the `generate` function of the PPOTrainer, which
    # is a wrapper around the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
    }

    for _ in range(args.epochs):
        for _, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]

            # Get response from gpt2
            response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, **generation_kwargs)
            batch["response"] = tokenizer.batch_decode(response_tensors)

            # Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(texts)
            rewards = [
                torch.tensor(output[1]["score"]) for output in pipe_outputs
            ]

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="lvwerra/gpt2-imdb",
        type=str,
        help="the model name")
    parser.add_argument(
        "--log_with",
        default=None, type=str, help="use 'wandb' to log with wandb")
    parser.add_argument(
        "--batch_size", default=128, type=int, help="the batch size")
    parser.add_argument(
        "--learning_rate",
        default=1.41e-5,
        type=float,
        help="the learning rate")
    parser.add_argument(
        "--mini_batch_size",
        default=128,
        type=int,
        help="the PPO minibatch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="the number of gradient accumulation steps")
    parser.add_argument(
        "--early_stopping",
        default=False,
        type=bool,
        help="whether to early stop")
    parser.add_argument(
        "--target_kl",
        default=0.1,
        type=float,
        help="kl target for early stopping")
    parser.add_argument(
        "--kl_penalty",
        default="kl",
        type=str,
        help="kl penalty options: 'kl', 'abs' and 'mse'")
    parser.add_argument(
        "--seed", default=0, type=int, help="the random seed")
    parser.add_argument(
        "--ppo_epochs",
        default=4,
        type=int,
        help="Number of optimisation epochs per batch of samples")
    parser.add_argument(
        "--tracker_project_name",
        default="trl", type=str, help="wandb project name")
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument(
        "--filter_negative",
        action="store_true",
        help="Only use positive samples for training in IMDB")

    args = parser.parse_args()
    main(args)
