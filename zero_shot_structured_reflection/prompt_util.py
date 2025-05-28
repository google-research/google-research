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

"""Definition of prompts."""

import util


STATUS = util.STATUS


def ordinal(n):
  """Converts n to n-th like string. The n is 1-indexed."""
  if 11 <= (n % 100) <= 13:
    suffix = 'th'
  else:
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
  return f'{n}' + suffix


def prompt_for_long_summary(html):
  """Prompt for detailed screen summary."""
  return (
      'The screen is represented in HTML pseudo code. You are capable of'
      ' summarizing the content and functionalities that are shown on the'
      f' screen.\n<screen>\n{html}\n</screen>\n\nNow, summarize the screen in'
      ' details. You should mention every possible functionality on the screen.'
      '\nSummary:'
  )


def prompt_for_short_summary(html):
  """Prompt for brief screen summary."""
  return (
      'The screen is represented in HTML pseudo code. You are capable of'
      ' summarizing the content and functionalities that are shown on the'
      f' screen.\n<screen>\n{html}\n</screen>\n\nNow, in a few sentences,'
      ' briefly summarize the screen.\nSummary:'
  )


def prompt_for_action_summary(html, actions):
  """Summarize actions taken on the screen (in html)."""
  action_str = '\n'.join(actions).strip()
  return (
      'You are capable of describing actions taken on a computer. The computer'
      ' screen is represented by the following HTML pseudo'
      f' code:\n<screen>\n{html}\n</screen>\n\nAnd the actions taken'
      f' are:\n{action_str}\n\nNow, in plain language, please summarize what'
      ' has been done. You should describe the specific purpose for the'
      ' action, instead of simply referring to the element id or position of'
      ' the element.\nSummary:'
  )


def construct_trace_summary(goal, screen_summaries, actions):
  """Summarize action trace with prior screen summaries and actions."""
  del goal
  assert len(screen_summaries) == len(actions)
  prompt = ''
  if not screen_summaries:
    prompt += 'You have not taken any actions yet.\n'
  else:
    prompt += 'Your past actions are:\n'
    for k, summary in enumerate(screen_summaries):
      prompt += f'The {ordinal(k+1)} screen describes: {summary.strip()}\n'
      prompt += f'Your {ordinal(k+1)} action: {actions[k].strip()}\n\n'
  return prompt


def construct_reflection_summary(reflections):
  """Summarize reflection history."""
  if reflections:
    return '\n'.join(['- ' + p for p in reflections])
  return ''


def paraphrase_exception_action(action):
  """Summarize exception history."""
  return (
      f'On this screen, you tried to {action} but this action is blocked.'
      ' You should try a different action.'
  )


def construct_trace_summary_for_actions(actions):
  """Heuristically summarizes action trace."""
  prompt = ''
  if not actions:
    prompt += 'You have not taken any actions yet.\n'
  else:
    for k, act in enumerate(actions):
      prompt += f'{ordinal(k+1)} action: {act.strip()}\n'
  return prompt


def prompt_for_reflective_planning(task_name, goal, trace_summary, html):
  """Prompt for reflectively planning with html of current screen."""
  return (
      f'You are an action planner that uses a computer for a task: {task_name}.'
      ' Your past actions have brought you to a the following screen which is'
      f' represented in HTML pseudo code:\n<screen>\n{html}\n</screen>\n\nThe'
      f' goal you want to achieve is: {goal}\n\nThe actions you have executed'
      f' are:\n{trace_summary}\nHowever, the actions did not work out well.'
      ' Now, please identify any potential issues with the plan and revise it.'
      ' You should generate a plan with corrected actions.\nYour revised plan:'
  )


def status_prompt(status):
  """Construct status string from status."""
  status_str = ''
  if status == STATUS.FAILED:
    status_str = (
        'However, your actions did not complete the goal. Now, you need to'
        ' identify the earliest critical step where you made a mistake, and'
        ' suggest a correction.'
    )
  elif status == STATUS.CYCLE:
    status_str = (
        'However, your actions led you to a loop that did not progress the'
        ' task. Now, you need to identify the earliest critical'
        ' step where you made a mistake, and suggest a correction.'
    )
  elif status == STATUS.NO_CHANGE:
    status_str = (
        'However, your last action did not cause anything to change on the last'
        ' screen. You probably used the wrong action type. Now, you need to'
        ' identify the earliest critical step where you made a mistake, and'
        ' suggest a correction.'
    )
  elif status == STATUS.INCOMPLETE:
    status_str = (
        'However, your actions did not finish the task, likely more steps are'
        ' needed. Now, you need to identify the earliest critical'
        ' step where you made a mistake, and suggest a correction.'
    )
  elif status == STATUS.IN_PROGRESS:
    # TODO: this might not be a good point for reflect, since it's prob
    # halted by the max_step limit.
    status_str = (
        'However, you took too many steps and yet still did not finish the'
        ' task. Now, you need to identify the earliest critical'
        ' step where you made a mistake, and suggest a correction.'
    )
  elif status == STATUS.EXCEPTION:
    status_str = (
        'However, your last action is invalid. You should avoid doing that'
        ' again and try a different action.'
    )
  else:
    raise ValueError('seems no need for reflection.')
  return status_str


def prompt_for_reflective_replay_vanilla(
    task_name,
    goal,
    prior_htmls,
    prior_actions,
    status,
    raw_reflections,
    use_static_screen=False,
):
  """Prompt for reflectively planning with html of current screen."""
  assert len(prior_htmls) == len(prior_actions)
  replay_str = ''
  if not use_static_screen:
    for i, (html, act) in enumerate(zip(prior_htmls, prior_actions)):
      replay_str += (
          f'The index={i+1} screen:\n'
          f'<screen>\n{html}\n</screen>\n'
          f'Your index={i+1} action: {act}\n\n'
      )
  else:
    replay_str = f'<screen>\n{prior_htmls[0]}\n</screen>\n'
    for i, act in enumerate(prior_actions):
      replay_str += f'Your index={i+1} action: {act}\n'

  reflect_str = ''
  if raw_reflections:
    reflect_summary = '\n'.join(raw_reflections)
    reflect_str = (
        'There are some failed attempts you learnt in your previous'
        f' trials:\n{reflect_summary}\n'
    )
    reflect_str += '\n\n'

  status_str = status_prompt(status)

  return (
      f'You are operating a computer for a task: {task_name}. You went over a'
      ' series of screens and executed actions to fulfill a top-level'
      f' goal.\nYour action trajectory is as follows:\n{replay_str}\nYou'
      ' conducted the above actions for the top-level goal:'
      f' {goal}\n\n{reflect_str}{status_str} Your'
      ' suggestion should be in this format: "For action index=A, you should'
      ' B.", where A is the action index, and B is the suggested'
      ' action you should have taken.\nYour suggestion:'
  )


def prompt_for_reflective_replay(
    task_name, goal, prior_htmls, prior_actions, status, use_static_screen=False
):
  """Prompt for reflectively planning with html of current screen."""
  assert len(prior_htmls) == len(prior_actions)
  replay_str = ''
  if not use_static_screen:
    for i, (html, act) in enumerate(zip(prior_htmls, prior_actions)):
      replay_str += (
          f'The index={i+1} screen:\n'
          f'<screen>\n{html}\n</screen>\n'
          f'Your index={i+1} action: {act}\n\n'
      )
  else:
    replay_str = f'<screen>\n{prior_htmls[0]}\n</screen>\n'
    for i, act in enumerate(prior_actions):
      replay_str += f'Your index={i+1} action: {act}\n'

  status_str = status_prompt(status)

  return (
      f'You are operating a computer for a task: {task_name}. You went over a'
      ' series of screens and executed actions to fulfill a top-level'
      f' goal.\nYour action trajectory is as follows:\n{replay_str}\nYou'
      ' conducted the above actions for the top-level goal:'
      f' {goal}\n\n{status_str} Your'
      ' suggestion should be in this format: "For action index=A, you should'
      ' B.", where A is the action index, and B is the suggested'
      ' action you should have taken.\nYour suggestion:'
  )


def prompt_for_type_action_decomposor(action):
  """Prompt for separating type action into a click and a type action."""
  return (
      'You are capable of decomposing a typing action to control a computer'
      ' into atomic steps.\nWhenever you want to type in something, you should'
      ' decompose it into two steps. Firstly click on the element such as'
      ' "click id=..."; and then type in the text, such as "enter "...""'
      f' \nNow, decompose this typing action: {action}\nPlease separate'
      ' the steps with semicolon.\nYour decomposition:'
  )


def prompt_for_action_consistency(goal, actions):
  """Prompt for summarizing action beam."""
  action_str = '\n'.join(actions)
  return (
      'You are capable of deciding what action to take on a computer.\nYou are'
      f' given a top-level goal: {goal}\nNow, you have {len(actions)} candidate'
      ' actions to take. Decide on one. You should refer to the information in'
      ' all candidates and generate a well-specified'
      f' action.\nCandidates:\n{action_str}\nYour choice:'
  )


def prompt_for_grounding(html, action):
  """Grounding action for click."""
  return (
      'You are given a screen which is represented in HTML pseudo code.\n'
      'Now, predict the id of the UI element to click on the current screen:\n'
      f'<screen >\n{html}\n</screen>\n\n'
      f'Action: {action}\n'
      'Prediction: id='
  )


def prompt_for_oneshot_plan(html, goal):
  """Prompt for 1-shot planning."""
  return (
      'You are operating a computer. You can generate a series of atomic'
      ' actions to fulfill a top-level goal. There are three type of atomic'
      ' actions you can perform. Firstly, you can click an object by referring'
      ' to its id, such as "click id=...". Secondly, you can enter text to an'
      ' input field, such as "enter "..." to id=...". Specifically, you'
      ' should always wrap the text you want to type in with double quotes.'
      ' Lastly, you can press'
      ' and hold special keys on the keyboard, such as "hold CTRL" and'
      ' "release CTRL" before and after multiple selections.\n\nThe screen you'
      ' see is represented by the following HTML'
      f' code:\n<screen>\n{html}\n</screen>\n\nThe top-level goal you want to'
      f' achieve is: {goal}\nNow, you need to plan actions that are executable'
      ' on and only on the current screen. Your plan should consist of a list'
      ' of atomic actions on the screen.\nYour plan:'
  )


def prompt_for_staged_plan(
    task_name, goal, html, trace_summary, reflection_summary, refl_follow=False
):
  """Prompt for staged planning based on prior execution history."""
  reflect_str = ''
  if reflection_summary:
    reflect_str = (
        'Here are some lessons you learnt in your previous'
        f' trials:\n{reflection_summary}\n'
    )
    reflect_str += '\n'

  # Prompt carried over from early-stage dev that works best on a task.
  if task_name in ('click-scroll-list'):
    result = (
        'You are operating a computer. You can generate a series of atomic'
        ' actions to fulfill a top-level goal. There are three type of atomic'
        ' actions you can perform. Firstly, you can click an object by'
        ' referring to its id, such as "click id=...". Secondly, you can enter'
        ' text to an input field, such as "enter ... to id=...". However,'
        ' every time you want to enter something, you should always click the'
        ' desired input field before typing. Lastly, you can press and hold'
        ' special keys on the keyboard, such as "hold CTRL" and "release CTRL"'
        ' before and after multiple selections.\n\nThe screen you see is'
        ' represented by the following HTML'
        f' code:\n<screen>\n{html}\n</screen>\n\nThe top-level goal you want to'
        f' achieve is: {goal}\nNow, you need to generate a detailed plan for'
        ' the goal. Your plan should consist of a list of atomic'
        ' actions.\nYour plan:'
    )
  else:
    result = (
        f'You are operating a computer for a task: {task_name}. You can'
        ' generate a series of atomic actions to fulfill a top-level goal.'
        ' There are three types of atomic actions you can perform. Firstly,'
        ' you can click an object by referring to its id, such as "click'
        ' id=...". Secondly, you can enter text to an input field, such as'
        ' "enter "..." to id=...". Specifically, you should always wrap the'
        ' text you want to type in with double quotes. Lastly, you can operate'
        ' special keys on the keyboard, such as "hold CTRL" and "release CTRL"'
        ' before and after multiple selections. If dropdownlist is available,'
        ' you can "press ARROWUP x N" or "press ARROWDOWN x N" to press the'
        ' arrow key N times to iterate over list items, and then "press ENTER"'
        ' to select the current item.\n\nThe screen you see is represented by'
        f' the following HTML code:\n<screen>\n{html}\n</screen>\n\nThe'
        f' top-level goal you want to achieve is: {goal}\nYour past actions'
        f' are:\n{trace_summary}\n{reflect_str}'
    )
    if not refl_follow:
      result += (
          'Now, you need to plan actions that are executable on and only on'
          ' this screen. For actions that are not executable on this screen,'
          ' you should leave them to future planning. Your plan should consist'
          ' of a list of atomic actions on the screen. Please separate them by'
          ' newline.\nYour plan:'
      )
    else:
      # have a short ending to promote the next action pred to be consistent
      # with the reflection mem.
      result += (
          'Now, you need to follow the lessons to plan your next action. Your'
          ' action:'
      )
  return result


# Some task specific
# There might be ad-hoc prompts or pipeline simplification for tasks.
# For instance, for this 1-screen-n-steps, we might not need to summarize the
# screen at all since it would be only useful for multi-screen tasks.
# Also, below has a specialized prompt for action summary since we want the
# prior actions to be paraphrased in a very specific way for the task.


def prompt_for_action_summary_social_media(html, actions):
  """Summarize actions taken on a screen, only 1-screen-n-step task type."""
  action_str = '\n'.join(actions).strip()
  return (
      'You are capable of describing actions taken on a computer screen. The'
      ' computer screen is represented by the following HTML pseudo'
      f' code:\n<screen>\n{html}\n</screen>\n\nAnd the action(s) already taken'
      f' are:\n{action_str}\n\nYou should describe the specific purpose for the'
      ' action and also referring to the element id. For instance, if you'
      ' clicked the like/reply button of a tweet, you should mention whose'
      ' tweet as well as the clicked element id. The output format should be'
      ' like: "I clicked ... (id=...)."\nSummary:'
  )


CUSTOM_ACTION_SUMMARY = {
    'social-media-all': prompt_for_action_summary_social_media,
    'social-media-some': prompt_for_action_summary_social_media,
    'social-media': prompt_for_action_summary_social_media,
}

CUSTOM_REFLECTION_MEMORY = {
    'use-autocomplete': (
        'You should always finish your task by clicking the submit button, if'
        ' it is presented.'
    ),
    'login-user-popup': (
        'If you see a popup window, you should click cancel immediately.'
    ),
}
