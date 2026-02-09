# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

import dataclasses

from browsergym.experiments import Agent, AbstractAgentArgs
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.action.python import PythonActionSet
from browsergym.utils.obs import flatten_axtree_to_str


class DemoAgent(Agent):
    """A basic agent using OpenAI API, to demonstrate BrowserGym's functionalities."""

    action_set = HighLevelActionSet(
        subsets=["chat", "bid"],  # define a subset of the action space
        # subsets=["chat", "bid", "coord"] # allow the agent to also use x,y coordinates
        strict=False,  # less strict on the parsing of the actions
        multiaction=True,  # enable to agent to take multiple actions at once
        demo_mode="default",  # add visual effects
    )
    # use this instead to allow the agent to directly use Python code
    # action_set = PythonActionSet())

    def obs_preprocessor(self, obs):
        return {
            "goal": obs["goal"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
        }

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        from openai import OpenAI

        self.openai_client = OpenAI()

    def get_action(self, obs):
        system_msg = f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

# Goal:
{obs["goal"]}"""

        prompt = f"""\
# Current Accessibility Tree:
{obs["axtree_txt"]}

# Action Space
{self.action_set.describe(with_long_description=False, with_examples=True)}

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12
```click("12")```
"
"""

        # query OpenAI model
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
        action = response.choices[0].message.content

        return action, {}


@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
    """
    This class is meant to store the arguments that define the agent.

    By isolating them in a dataclass, this ensures serialization without storing
    internal states of the agent.
    """

    model_name: str = "gpt-3.5-turbo"

    def make_agent(self):
        return DemoAgent(model_name=self.model_name)


def main():
    from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
    from pathlib import Path

    exp_root = Path().home() / "agent_experiments"
    exp_root.mkdir(exist_ok=True)

    exp_args = ExpArgs(
        agent_args=DemoAgentArgs(model_name="gpt-3.5-turbo"),
        env_args=EnvArgs(
            task_name="miniwob.click-test",
            task_seed=42,
            headless=False,  # shows the browser
        ),
    )

    exp_args.prepare(exp_root=exp_root)
    exp_args.run()

    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")
