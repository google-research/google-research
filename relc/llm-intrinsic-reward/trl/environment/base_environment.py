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



import re
import warnings

import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList

from ..import_utils import is_rich_available


if is_rich_available():
    from rich import print
    from rich.text import Text


class StringStoppingCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generations in the batch are completed."""

    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.first_call = True

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the stop strings."""
        if self.first_call:
            self.generated_tokens = [1 for _ in range(input_ids.shape[0])]
            self.start_length = input_ids.shape[-1] - 1
            self.first_call = False
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []

        for i, decoded_generation in enumerate(decoded_generations):
            sequence_complete = any([stop_string in decoded_generation for stop_string in self.stop_strings])
            done.append(sequence_complete)
            if not sequence_complete:
                self.generated_tokens[i] += 1

        if all(done):
            self.first_call = True

        return all(done)


class TextHistory:
    """The TextHistory class keeps track of the history of an interaction between the language model and the environment."""

    def __init__(self, text, tokens, system=True):
        """
        Initialize TextHistory.

        args:
            text (`str`): The text of the first segment.
            tokens (`torch.LongTensor`): The tokens of the first segment.
            system (`bool`, *optional*): Whether the first segment is a system or user segment.
        """
        self.system_spans = []
        self.text_spans = []
        self.token_spans = []
        self.token_masks = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.text = ""
        self.tokens = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.completed = False
        self.truncated = False
        self.reward = 0.0

        self.prompt_color = "black on grey85"
        self.system_color = "black on cyan3"
        self.model_color = "black on deep_sky_blue1"
        self.reward_color = "black on plum1"

        self.append_segment(text, tokens, system=system)

    def append_segment(self, text, tokens, system=True):
        """
        Append a new segment to the history.

        args:
            text (`str`): The text of the new segment.
            tokens (`torch.LongTensor`): The tokens of the new segment.
            system (`bool`, *optional*): Whether the new segment is a system or user segment.
        """

        if len(text) == 0 or len(tokens) == 0:
            raise ValueError("Can't append empty text or token list to history.")

        original_text_length = len(self.text)

        self.text += text
        self.text_spans.append((original_text_length, len(self.text)))
        self.system_spans.append(system)

        original_token_length = len(self.tokens)

        self.tokens = torch.cat((self.tokens, tokens))
        if system:
            self.token_masks = torch.cat((self.token_masks, torch.zeros_like(tokens)))
        else:
            self.token_masks = torch.cat((self.token_masks, torch.ones_like(tokens)))
        self.token_spans.append((original_token_length, len(self.tokens)))

    def complete(self, truncated=False):
        """
        Mark the history as completed.
        """
        self.completed = True
        self.truncated = truncated

    @property
    def last_text_segment(self):
        """
        Get the last text segment.
        """
        start, end = self.text_spans[-1]
        return self.text[start:end]

    def split_query_response_tokens(self):
        """
        Split the tokens into query and response tokens.
        """
        split_index = self.token_spans[0][1]
        query = self.tokens[:split_index]
        response = self.tokens[split_index:]
        mask = self.token_masks[split_index:]

        return query, response, mask

    def show_text(self, show_legend=False):
        """
        Print the text history.
        """
        if not is_rich_available():
            warnings.warn("install rich to display text")
            return

        text = Text(self.text)
        text.stylize(self.prompt_color, self.text_spans[0][0], self.text_spans[1][0])
        for i, (start, end) in enumerate(self.text_spans[1:]):
            if self.system_spans[i + 1]:
                text.stylize(self.system_color, start, end)
            else:
                text.stylize(self.model_color, start, end)

        text.append(f"\n\nReward: {self.reward}", style=self.reward_color)
        print(text)

        if show_legend:
            self.show_colour_legend()

    def show_tokens(self, tokenizer, show_legend=False):
        """
        Print the history tokens.
        """
        if not is_rich_available():
            warnings.warn("install rich to display tokens")
            return

        text = Text()
        prompt_end = self.token_spans[0][1]
        for i, (token, mask) in enumerate(zip(self.tokens, self.token_masks)):
            if i < prompt_end:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.prompt_color)
                text.append(" ")
            elif mask == 0:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.system_color)
                text.append(" ")
            else:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.model_color)
                text.append(" ")
        text.append(f"\n\nReward: {self.reward}", style=self.reward_color)
        print(text)
        if show_legend:
            self.show_colour_legend()

    def show_colour_legend(self):
        """
        Print the colour legend.
        """
        if not is_rich_available():
            warnings.warn("install rich to display colour legend")
            return
        text = Text("\n\n(Colour Legend: ")
        text.append("Prompt", style=self.prompt_color)
        text.append("|")
        text.append("System", style=self.system_color)
        text.append("|")
        text.append("Model", style=self.model_color)
        text.append("|")
        text.append("Reward", style=self.reward_color)
        text.append(")")
        print(text)


class TextEnvironment:
    """
    The TextEnvironment enables interaction of a LLM with an environment using tools.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        tools=None,
        reward_fn=None,
        prompt=None,
        max_turns=4,
        max_tool_reponse=100,
        max_length=None,
        generation_kwargs=None,
    ):
        """
        Initialize TextEnvironment.

        Args:
            model (`PreTrainedModelWrapper`): The model to use for generation.
            tokenizer (`transformers.PreTrainedTokenizer`): The tokenizer to use for generation.
            tools (list): A list of tools to use for interaction.
            reward_fn (function): A function that takes a string and returns a reward.
            prompt (str): The base prompt to use for generation. Is prepended to the tasks.
            max_turns (Optional[int]): The maximum number of turns to allow.
            max_tool_response (Optional[int]): The maximum number of characters to allow in a tool response.
            max_length (Optional[int]): The maximum number of tokens to allow in an episode.
            generation_kwargs (Optional[dict]): A dictionary of keyword arguments to pass to the model's generate method.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        if isinstance(tools, dict):
            self.tools = tools
        else:
            self.tools = dict([(tool.__class__.__name__, tool) for tool in tools])
        self.reward_fn = reward_fn
        self.max_length = max_length
        self.request_token = "<request>"
        self.call_token = "<call>"
        self.response_token = "<response>"
        self.submit_token = "<submit>"
        self.max_turns = max_turns
        self.max_tool_response = max_tool_reponse

        if generation_kwargs is None:
            self.generation_kwargs = dict()
        else:
            self.generation_kwargs = generation_kwargs

        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.current_device = extract_model_from_parallel(self.model).pretrained_model.device

    def run(self, queries, **rewards_kwargs):
        """
        Run the environment on a list of queries.

        Args:
            queries (list[str]): A list of queries to run the model in the environment on.
        """
        turns = 0

        queries = [self.prompt + task for task in queries]
        queries_tokens = [
            self.tokenizer(query, return_tensors="pt").input_ids[0].to(self.model.pretrained_model.device)
            for query in queries
        ]

        histories = [TextHistory(q, qt, system=True) for q, qt in zip(queries, queries_tokens)]

        while any([not history.completed for history in histories]) and turns < self.max_turns:
            histories = self.generate(histories)
            histories = self.tasks_end_check(histories)
            # TODO: make this parallel rather than for-loop
            for i in range(len(histories)):
                histories[i] = self.step(histories[i])
            histories = self.tasks_end_check(histories, model_turn=False)
            turns += 1
        self.compute_reward(histories, **rewards_kwargs)

        # convert a list of (q, r, m) tuples to lists of all qs, rs, and ms respectively
        queries, responses, masks = map(list, zip(*[history.split_query_response_tokens() for history in histories]))

        rewards = [history.reward for history in histories]
        return queries, responses, masks, rewards, histories

    def step(self, history):
        """
        Step the environment forward one turn.

        Args:
            history (`TextHistory`): The history to step forward.
        """
        truncated, ended = self.task_end_check(history)
        if ended:
            history.complete(truncated=truncated)
        if history.completed:
            return history

        tool, query = self.parse_tool_call(history.last_text_segment)
        if tool is None or query is None:
            response = f"Unknown tool call: {history.last_text_segment}"
        else:
            if tool not in self.tools:
                response = f"Unknown tool {tool}."
            try:
                response = self.tools[tool](query)
            except Exception as error:
                response = f"Tool error: {str(error)}"

        if len(response) > self.max_tool_response:
            response = response[: (self.max_tool_response - 3)] + "..."

        history.append_segment(
            response + self.response_token,
            self.tokenizer(response + self.response_token, return_tensors="pt")
            .input_ids[0]
            .to(self.model.pretrained_model.device),
            system=True,
        )

        return history

    def parse_tool_call(self, text):
        """
        Parse request string. Expected format: <request><tool_name>query<call>
        """
        result = re.search(f"(?<={self.request_token}).*?(?={self.call_token})", text, re.DOTALL)

        # if we can't find a <request>/<call> span we return none
        if result is None:
            return None, None
        else:
            extracted_text = result.group()

        result = re.search(r"<(.*?)>", extracted_text)

        # if we can't find a tool name we return none
        if result is None:
            return None, None
        else:
            tool = result.group(1)

        # split off the tool name
        query = ">".join(extracted_text.split(">")[1:])

        return tool, query

    def compute_reward(self, histories, **reward_kwargs):
        """
        Compute the reward for a list of histories.
        """
        rewards = self.reward_fn([history.last_text_segment for history in histories], **reward_kwargs)
        for history, reward in zip(histories, rewards):
            history.reward = reward
        return histories

    def generate(self, histories):
        """
        Generate responses for a list of histories.
        """
        active_histories = [i for i, history in enumerate(histories) if not history.completed]

        query_tensors = [histories[i].tokens for i in active_histories]
        response_tensors = self._generate_batched(query_tensors)
        response_texts = self.tokenizer.batch_decode(response_tensors)

        for i, response_text, response_tensor in zip(active_histories, response_texts, response_tensors):
            histories[i].append_segment(response_text, response_tensor, system=False)

        return histories

    def tasks_end_check(self, histories, model_turn=True):
        """
        Check if the current generation sequences have finished.
        """
        for history in histories:
            if not history.completed:
                truncated, ended = self.task_end_check(history, model_turn=model_turn)
                if ended:
                    history.complete(truncated=truncated)
        return histories

    def task_end_check(self, history, model_turn=True):
        """
        Check if the current generation sequence has finished.
        """
        truncated = False
        ended = False
        if history.completed:
            return truncated, ended
        if self.max_length is not None and len(self.tokenizer(history.text).input_ids[0]) > self.max_length:
            truncated = True
            ended = True
        elif self.tokenizer.eos_token in history.text:
            ended = True
        elif model_turn and not (
            (self.request_token in history.last_text_segment and self.call_token in history.last_text_segment)
            or self.submit_token in history.last_text_segment
        ):
            ended = True
        elif self.submit_token in history.last_text_segment:
            ended = True
        return truncated, ended

    def _generate_batched(
        self,
        query_tensors,
        batch_size: int = 16,
        pad_to_multiple_of: int = None,
    ):
        """
        Generate responses for a list of query tensors.

        args:
            query_tensors (list[torch.Tensor]): A list of query tensors to generate responses for.
            batch_size (int): The batch size to use for generation.
            pad_to_multiple_of (int): The padding length to use for generation.
        """
        outputs = []
        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            stopping_criteria = StringStoppingCriteria([self.call_token, self.submit_token], self.tokenizer)

            self.generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping_criteria])

            generations = extract_model_from_parallel(self.model).generate(**padded_inputs, **self.generation_kwargs)

            for generation, mask, generated_tokens in zip(
                generations, padded_inputs["attention_mask"], stopping_criteria.generated_tokens
            ):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt

                # remove chunk generated after stopping criteria in batch mode
                outputs.append(output[:generated_tokens])
        self.tokenizer.padding_side = padding_side_default
        return outputs
