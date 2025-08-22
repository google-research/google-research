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

"""Runs a specific miniwob task."""

import dataclasses
from absl import logging
from miniwob import instance as miniwob_instance
import prompt_util
from saxml.client.python import sax
from selenium import webdriver
from selenium.common import exceptions
import util

LOGGING = logging.info
# LOGGING = print
STATUS = util.STATUS

BASE_URL = 'YOUR_MINIWOB/html/miniwob'


@dataclasses.dataclass()
class CursorState:
  ptr_down: bool = False
  ptr_x: int = 10
  ptr_y: int = 10
  cursor: str = 'pointer'


def create_driver(chrome_options):
  """Create a driver."""
  chrome_options.add_argument('disable-gpu')
  chrome_options.add_argument('headless')
  print('chrome_options: %s' % chrome_options.arguments)
  return webdriver.Chrome(chrome_options=chrome_options)


def get_instance(subdomain, seed):
  """Return MiniWoB instance."""

  base_url = BASE_URL

  instance = miniwob_instance.MiniWoBInstance(
      index=0,
      subdomain=subdomain,
      seed=seed,
      web_driver_creator=create_driver,
      base_url=base_url,
      headless=True,
  )
  return instance


def parse_response(response):
  if '[eod]' in response:
    return response.split('[eod]')[0].strip()
  return response.strip()


def get_status(metadata):
  if metadata['done']:
    if metadata['env_reward'] > 0:
      return STATUS.CORRECT
    else:
      return STATUS.FAILED
  return STATUS.IN_PROGRESS


class ReflectionMemory:
  """Tracks reflection and environment feedback on actions."""

  def __init__(self, size, sticky):
    self._mem = ['' for _ in range(size)]
    self._mem_act = ['' for _ in range(size)]
    self._mem_enforce = [False for _ in range(size)]
    self._raw_reflections = []  # simply concat, no management
    self._disabled_clicks = [set() for _ in range(size)]
    self._sticky: list[str] = sticky

  def get_mem(self, idx):
    if self._mem[idx]:
      return self._sticky + [self._mem[idx]]
    return self._sticky

  def has_mem(self, idx):
    return bool(self._mem[idx])

  def is_enforced(self, idx):
    return self._mem_enforce[idx]

  def get_disabled_clicks(self, idx):
    return self._disabled_clicks[idx]

  def get_raw_reflections(self, with_sticky):
    if with_sticky:
      return self._sticky + self._raw_reflections
    return self._raw_reflections

  def update_mem(
      self, action_taken, refl_statement, enforce = False
  ):
    """Update reflection memory and disabled set for the screen."""
    # parse reflection statement
    idx, new_action = util.parse_reflection_action(refl_statement)
    if idx == -1:
      return False
    new_action_type = util.categorize_action(new_action)
    # parse action taken
    taken_action_type = util.categorize_action(action_taken)
    # Update disable set
    # Existing element in the memory can be considered outdated and the attempt
    # must have failed. So in the next round of reflection, we should skip it.
    # TODO(tlinlp): only work on click action types for now.
    if taken_action_type == 'click' and new_action_type == 'click':
      new_click_id = util.parse_int(new_action)
      taken_click_id = util.parse_int(action_taken)
      if new_click_id != taken_click_id:
        self._disabled_clicks[idx].add(taken_click_id)
    # update the memory
    self._mem[idx] = refl_statement
    self._mem_act[idx] = new_action
    # clear up future memory
    self._mem_enforce[idx] = enforce
    # record raw
    self._raw_reflections.append(refl_statement)

    for k in range(idx + 1, len(self._mem)):
      self._mem[k] = ''
      self._mem_act[k] = ''
      self._mem_enforce[k] = False
      self._disabled_clicks[k].clear()
    return True


class MiniWoBRunner:
  """Run a Miniwob task."""

  def __init__(
      self,
      sax_model,
      max_step,
      max_step_per_view=10,
      llm_temp=0.5,
      llm_max_step=128,
  ):
    sax_opts = sax.Options()
    sax_opts.num_conn = 1
    self.model = sax.Model(sax_model, sax_opts).LM()
    self.max_step = max_step
    self.max_step_per_view = min(max_step, max_step_per_view)
    self.llm_temp = llm_temp
    self.llm_max_step = llm_max_step

  def model_inf(self, prompt, temperature=0.5, max_decode_steps=128):
    options = sax.ModelOptions()
    options.SetExtraInput('temperature', temperature)
    options.SetExtraInput('per_example_max_decode_steps', max_decode_steps)
    sax_results = self.model.Generate(prompt, options)
    return [parse_response(response) for response, _ in sax_results]

  def begin_task(
      self,
      task_name,
      seed,
      task_type,
      wait_duration=0.5,
      given_plan='',
      cycle_detect=True,
      no_change_detect=True,
  ):
    """Begin the task with env variables."""
    self.task_name = task_name
    self.task_type = task_type
    self.wait_duration = wait_duration
    self.given_plan = given_plan
    self.instance = get_instance(task_name, seed)
    util.initialize_instance(self.instance)
    self.instance.begin_task()
    util.pause(self.instance, self.wait_duration)
    self.cycle_detect = cycle_detect
    self.no_change_detect = no_change_detect
    self.pack = None

  def end_task(self):
    if self.instance:
      self.instance.close()
    self.given_plan = ''

  def infer_id_to_click(self, html, action):
    """Infer the object id from a click action."""
    try:
      return int(util.extract_id_to_click(action))
    except ValueError:
      pass
    prompt = prompt_util.prompt_for_grounding(html, action)
    obj_ids = self.model_inf(prompt, self.llm_temp, self.llm_max_step)
    extracted_ids = []
    for idx in obj_ids:
      try:
        extracted_ids.append(util.extract_id_to_click(idx))
      except ValueError:
        pass
    if not extracted_ids:
      raise ValueError(prompt, obj_ids)
    return int(util.maj_vote_text(extracted_ids))

  def plan_follow(self, step_idx, plan):
    """Iteratively following the exact steps in the plan."""
    steps = plan.split('\n')
    next_act = ''
    if step_idx < len(steps) and step_idx < self.max_step_per_view:
      next_act = steps[step_idx]
    return next_act

  def plan_and_execute(
      self,
      start_step_idx,
      prior_actions,
      refl_mem,
      given_plan='',
      use_structured_mem=True,
      debug_print=False,
  ):
    """Do a stage of plan and execution."""
    assert self.instance
    goal = self.instance.get_state().utterance
    cursor_state = CursorState()
    html, _, _ = util.flat_html(self.instance.get_state().dom_elements)
    if debug_print:
      LOGGING('goal %s', goal)
    cur_plan = given_plan
    # start planning on the screen.
    if not cur_plan:
      refl_texts = refl_mem.get_mem(start_step_idx)
      cur_plan = ''
      # if there is a memory tagged to be enforced,
      # we enforce it without calling planner.
      if refl_mem.has_mem(start_step_idx) and refl_mem.is_enforced(
          start_step_idx
      ):
        _, enforced_act = util.parse_reflection_action(refl_texts[-1])
        cur_plan = enforced_act
        if debug_print:
          LOGGING('enforcing act from reflection %s', enforced_act)

      # if still no active plan at this point, call planner.
      if not cur_plan:
        trace_summary = prompt_util.construct_trace_summary_for_actions(
            prior_actions
        )
        if not use_structured_mem:
          refl_summary = '\n'.join(
              refl_mem.get_raw_reflections(with_sticky=True)
          )
        else:
          refl_summary = prompt_util.construct_reflection_summary(refl_texts)
        refl_follow = refl_mem.has_mem(start_step_idx)
        prompt = prompt_util.prompt_for_staged_plan(
            self.task_name,
            goal,
            html,
            trace_summary,
            refl_summary,
            refl_follow=refl_follow,
        )
        cur_plan = self.model_inf(prompt, self.llm_temp, self.llm_max_step)[0]
        if debug_print:
          LOGGING('cur_plan_prompt %s', prompt)

    if debug_print:
      LOGGING('cur_plan %s', cur_plan)

    inner_step_idx = 0
    # bookkeeping prior
    prior_htmls = []
    prior_actions = []
    prior_exact_actions = []
    pack = {
        'htmls': prior_htmls,
        'actions': prior_actions,
        'exact_actions': prior_exact_actions,
    }
    halt_status = None
    if not cur_plan:
      return STATUS.INCOMPLETE, pack

    while True:
      if start_step_idx + inner_step_idx + 1 >= self.max_step:
        halt_status = STATUS.FAILED
        break
      if inner_step_idx + 1 >= self.max_step_per_view:
        halt_status = STATUS.FAILED
        break

      html, _, _ = util.flat_html(self.instance.get_state().dom_elements)
      action = self.plan_follow(inner_step_idx, cur_plan)
      if not action:
        halt_status = STATUS.IN_PROGRESS
        break
      # if the current step idx matches a reflection entry, we need to start
      # another round of planning to incorporate the refl mem.
      match_refl = refl_mem.has_mem(start_step_idx + inner_step_idx)
      if inner_step_idx != 0 and match_refl:
        halt_status = STATUS.IN_PROGRESS
        break

      if debug_print:
        LOGGING('next action %s', action)
      action_type = util.categorize_action(action)
      if action_type == 'click':
        obj_idx = util.parse_int(action)
        # fallback to another round of inference
        if obj_idx == 'N/A':
          prompt = prompt_util.prompt_for_grounding(html, action)
          obj_ids = self.model_inf(prompt, self.llm_temp, self.llm_max_step)
          obj_ids = [util.parse_int(p) for p in obj_ids]
          obj_idx = int(util.maj_vote_text(obj_ids))
          action = f'click id={obj_idx}'
      # Actual execution
      try:
        # NOTE, skip scroll action since doing it would change obj ids.
        if action_type == 'scroll':
          continue
        LOGGING('action to execute %s', action)
        util.execute_action(self.instance, cursor_state, action)
        util.pause(self.instance, self.wait_duration)
      except (exceptions.WebDriverException, ValueError):
        # raise exceptions.WebDriverException(e)
        # Instead of raising exception, continue the execution
        # This error can often happen in interruptive tasks (like popup)
        util.pause(self.instance, 1)
        html, _, _ = util.flat_html(self.instance.get_state().dom_elements)
        halt_status = STATUS.EXCEPTION
        last_action_paraphrase = self.paraphrase_action(html, action)
        prior_htmls.append(html)
        prior_actions.append(last_action_paraphrase)
        prior_exact_actions.append(action)
        if debug_print:
          LOGGING('exception detected %s', action)
        break

      last_action_paraphrase = self.paraphrase_action(html, action)

      # Cycle detection
      # Continue to the next planning iteration if we got in cycles.
      # TODO(tlinlp), moving mouse might not change anything, need improvement.
      html_after, _, _ = util.flat_html(self.instance.get_state().dom_elements)
      if html_after in prior_htmls and self.cycle_detect:
        LOGGING('cycle detected, move on to the next iteration.')
        halt_status = STATUS.CYCLE
        prior_htmls.append(html)
        prior_actions.append(last_action_paraphrase)
        prior_exact_actions.append(action)
        break
      if html_after == html and self.no_change_detect:
        LOGGING('no change, moving to the next iteration.')
        halt_status = STATUS.NO_CHANGE
        prior_htmls.append(html)
        prior_actions.append(last_action_paraphrase)
        prior_exact_actions.append(action)
        break

      # bookkeeping
      prior_htmls.append(html)
      prior_actions.append(last_action_paraphrase)
      prior_exact_actions.append(action)
      inner_step_idx += 1

      pack = {
          'htmls': prior_htmls,
          'actions': prior_actions,
          'exact_actions': prior_exact_actions,
      }
      metadata = self.instance.get_metadata()
      status = get_status(metadata)
      if status != STATUS.IN_PROGRESS:
        return status, pack

    pack = {
        'htmls': prior_htmls,
        'actions': prior_actions,
        'exact_actions': prior_exact_actions,
    }
    # final check
    if halt_status is None:
      metadata = self.instance.get_metadata()
      status = get_status(metadata)
    else:
      status = halt_status
    return status, pack

  def do_trial(
      self,
      task_name,
      seed,
      task_type,
      given_plans=None,
      refl_mem=None,
      cycle_detect=True,
      no_change_detect=True,
      use_structured_mem=True,
      debug_print=False,
  ):
    """Do a trial for the task."""
    self.begin_task(
        task_name,
        seed,
        task_type,
        cycle_detect=cycle_detect,
        no_change_detect=no_change_detect,
    )
    goal = self.instance.get_state().utterance
    step_idx = 0
    screen_idx = 0
    status = STATUS.IN_PROGRESS
    prior_htmls = []
    prior_actions = []
    prior_exact_actions = []
    plan_cnt = 0
    while True:
      given_plan = (
          given_plans[screen_idx]
          if given_plans and screen_idx < len(given_plans)
          else ''
      )
      status, pack = self.plan_and_execute(
          start_step_idx=step_idx,
          prior_actions=prior_actions,
          refl_mem=refl_mem,
          given_plan=given_plan,
          use_structured_mem=use_structured_mem,
          debug_print=debug_print,
      )
      plan_cnt += 1
      step_idx += len(pack['actions'])
      screen_idx += 1
      prior_htmls.extend(pack['htmls'])
      prior_actions.extend(pack['actions'])
      prior_exact_actions.extend(pack['exact_actions'])

      if status != STATUS.IN_PROGRESS:
        break
      if step_idx + 1 >= self.max_step:
        break
    self.end_task()
    # return the status and the trace on the very last screen.
    return status, {
        'goal': goal,
        'htmls': prior_htmls,
        'actions': prior_actions,
        'exact_actions': prior_exact_actions,
        'plan_cnt': plan_cnt,
        'status': status.name,
    }

  def do_trial_and_reflect(
      self,
      task_name,
      seed,
      task_type,
      num_trial,
      cycle_detect=True,
      no_change_detect=True,
      use_static_screen=False,
      use_structured_mem=True,
      debug_print=False,
      trace_pool=None,
  ):
    """Do the task in multiple trials with reflection."""
    assert num_trial > 0
    sticky_mem = []
    if task_name in prompt_util.CUSTOM_REFLECTION_MEMORY:
      sticky_mem.append(prompt_util.CUSTOM_REFLECTION_MEMORY[task_name])

    refl_mem = ReflectionMemory(self.max_step, sticky_mem)

    status = STATUS.IN_PROGRESS
    for trial_idx in range(num_trial):
      if debug_print:
        LOGGING('############################ trial %s', trial_idx + 1)
      status, pack = self.do_trial(
          task_name,
          seed,
          task_type,
          given_plans=None,
          refl_mem=refl_mem,
          cycle_detect=cycle_detect,
          no_change_detect=no_change_detect,
          use_structured_mem=use_structured_mem,
          debug_print=debug_print,
      )
      goal = pack['goal']
      prior_htmls = pack['htmls']
      prior_actions = pack['actions']
      prior_exact_actions = pack['exact_actions']

      if trace_pool is not None and isinstance(trace_pool, dict):
        trace_pool[trial_idx] = {}
        trace_pool[trial_idx]['goal'] = goal
        trace_pool[trial_idx]['htmls'] = prior_htmls
        trace_pool[trial_idx]['actions'] = prior_actions
        trace_pool[trial_idx]['exact_actions'] = prior_exact_actions
        trace_pool[trial_idx]['status'] = status.name
        trace_pool[trial_idx]['plan_cnt'] = pack['plan_cnt']
      # do reflection
      if status != STATUS.CORRECT:
        if use_structured_mem:
          updated_prior_htmls = []
          for t, html in enumerate(prior_htmls):
            disabled_set = refl_mem.get_disabled_clicks(t)
            if disabled_set:
              if debug_print:
                LOGGING(
                    'disabling clicks %s on html at %s', str(disabled_set), t
                )
              updated_prior_htmls.append(
                  util.disable_clicks_in_html(html, disabled_set)
              )
            else:
              updated_prior_htmls.append(html)
            # updated_prior_htmls.append(html)
          prior_htmls = updated_prior_htmls

        if use_structured_mem:
          reflect_prompt = prompt_util.prompt_for_reflective_replay(
              self.task_name,
              goal,
              prior_htmls,
              prior_exact_actions,
              status,
              use_static_screen=use_static_screen,
          )
        else:
          reflect_prompt = prompt_util.prompt_for_reflective_replay_vanilla(
              self.task_name,
              goal,
              prior_htmls,
              prior_exact_actions,
              status,
              raw_reflections=refl_mem.get_raw_reflections(with_sticky=True),
              use_static_screen=use_static_screen,
          )

        reflections = self.model_inf(
            reflect_prompt, self.llm_temp, self.llm_max_step
        )
        reflection = util.take_longest(reflections)

        # Update reflection memory
        enforce = use_structured_mem
        refl_act_idx, _ = util.parse_reflection_action(reflection)
        if refl_act_idx >= 0 and refl_act_idx < len(prior_exact_actions):
          action_taken = prior_exact_actions[refl_act_idx]
          refl_mem.update_mem(action_taken, reflection, enforce=enforce)

        if debug_print:
          LOGGING('************** reflect_prompt')
          LOGGING('reflect_prompt %s', reflect_prompt)
          LOGGING('reflection %s', reflection)
        if trace_pool is not None and isinstance(trace_pool, dict):
          trace_pool[trial_idx]['reflection'] = reflection
      else:
        return status
    return status

  def paraphrase_action(self, html, action):
    """Summarizes the actual executed action."""
    # For press action, it is what it is.
    action_type = util.categorize_action(action)
    if action_type == 'press':
      return action

    prompt_for_action_summary = prompt_util.prompt_for_action_summary
    if self.task_name in prompt_util.CUSTOM_ACTION_SUMMARY:
      prompt_for_action_summary = prompt_util.CUSTOM_ACTION_SUMMARY[
          self.task_name
      ]
    prompt = prompt_for_action_summary(html, [action])
    paraphrase = self.model_inf(prompt, self.llm_temp, self.llm_max_step)[0]
    return paraphrase
