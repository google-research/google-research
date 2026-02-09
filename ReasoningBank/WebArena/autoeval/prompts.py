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

def build_obs_simplifier_prompt(cap, intent, response):
    prompt = f"""Given the following user question and context, extract part of the context that is unbiased, so that using that text alone would be good context for providing an unbiased answer to the user query.

**User Query**: The bot responded with "{response}", does it execute this task "{intent}" successfully?

**Full Context**:
```md
{cap}
```

Start your answer with “Unbiased text context (includes all relevant content):"
"""
    return prompt


def build_naive_last_frame_eval_prompt(cap, intent, response):
    prompt = f"""**User Intent**: {intent}

**Bot's Final Observation**:

```md
{cap}
```

**Bot's response to the user**: {response if response else "None"}.

---

Based on the provided user intent, the caption of bot's final observation and its response, did the bot successfully execute the task? Please reason step by step.

Note:
- The trajectory descriptions are essentially noisy captions of the screenshots captured during bot's execution. And you should infer what actions the bot took yourself.
- You should categorize the execution into one of the three status:
    - task-possible-bot-success: The bot successfully executed the task.
    - task-possible-bot-fail: The bot failed to execute the task.
    - task-impossible: The task is impossible to execute in nature given the user intent and the environment. For example, if the user wants to buy a product that does not exist in the environment. You should carefully distinguish this from bot-fail.

Format your response as a valid json:
{{
    "thoughts": "{{Your thoughts here, discuss if and how the trajectory progress towards the task and then reason about the final status. You should provide an explicit reason when determining the final status.}}",
    "status": "task-possible-bot-success" or "task-possible-bot-fail" or "task-impossible"
}}"""
    return prompt


def build_naive_multi_frame_eval_prompt(caps, intent, response):
    captions_str = "\n".join(
        [f"{idx+1}:\n```md\n{caption}\n```\n" for idx, caption in enumerate(caps[-3:])]
    )
    prompt = f"""**User Intent**: {intent}

**Bot's observation through execution**:

{captions_str}

**Bot's response to the user**: {response if response else "None"}.

---

Based on the provided user intent, bot's observation in captions and its response, did the bot successfully execute the task? Please reason step by step.

Note:
- You should categorize the execution into one of the three status:
    - task-possible-bot-success: The bot successfully executed the task.
    - task-possible-bot-fail: The bot failed to execute the task.
    - task-impossible: The task is impossible to execute in nature given the user intent and the environment. For example, if the user wants to buy a product that does not exist in the environment. You should carefully distinguish this from bot-fail.

Format your response as a valid json:
{{
    "thoughts": "{{Your thoughts here, discuss if and how the trajectory progress towards the task and then reason about the final status. You should provide an explicit reason when determining the final status.}}",
    "status": "task-possible-bot-success" or "task-possible-bot-fail" or "task-impossible"
}}"""
    return prompt


def extract_content(text, start_tag):
    """
    Extract the content that follows 'Info:' in a given string.

    :param text: A string that may contain lines starting with 'Info:'
    :return: The content that follows 'Info:' or None if not found
    """
    # Split the text into lines
    lines = text.split("\n")

    # Loop through each line to find a line that starts with 'Info:'
    for line in lines:
        if line.startswith(start_tag):
            # Extract and return the content after 'Info:'
            return line[len(start_tag) :].strip()

    # Return None if 'Info:' is not found in any line
    return ""


def build_text_eval_prompt(
    cap, intent, response, last_actions
):
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, map info, comparison of map routes, etc. The bot's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the bot encounters an exception and respond with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked to list the top-searched items in a shop, the agent should order the items by the number of searches, and then return the top items. If the ordering action is missing, the task is likely to fail.
2. Site navigation: The user wants to navigate to a specific page. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>"
Status: "success" or "failure"
"""
    prompt = f"""User Intent: {intent}

Action History:
{last_actions}

The detailed final state of the webpage:

```md
{cap}
```

Bot response to the user: {response if response else "N/A"}."""
    return prompt, system_msg


def build_vision_eval_prompt(
    intent, response, last_actions
):
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, map info, comparison of map routes, etc. The bot's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the bot encounters an exception and respond with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked to list the top-searched items in a shop, the agent should order the items by the number of searches, and then return the top items. If the ordering action is missing, the task is likely to fail.
2. Site navigation: The user wants to navigate to a specific page. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
"""
    prompt = f"""User Intent: {intent}

Action History:
{last_actions}

The last snapshot of the web page is shown in the image."""
    return prompt, system_msg
