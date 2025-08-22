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

"""Example execution of a rule-based optimal policy on gminiwob shopping."""

import time

from absl import app
from absl import flags
from absl import logging
from CoDE import test_websites
from CoDE import utils
from CoDE import vocabulary_node
from CoDE import web_environment


flags.DEFINE_string("data_dep_path", None,
                    "Data dep path for local miniwob files.")
flags.DEFINE_boolean(
    "run_headless_mode", False,
    "Run in headless mode. On borg, this should always be true.")
flags.DEFINE_boolean(
    "use_conceptual", False,
    "If true, use abstract web navigation where it is assumed to known which profile field corresponds to which element."
)
FLAGS = flags.FLAGS


def run_policy_on_shopping_website():
  """Run an optimal policy on the shopping website and visualize in browser."""
  # Create a generic web environment to which we will add primitives and
  # transitions to create a shopping website. These parameters will work to
  # observe a simple policy running but they might be insufficient in a training
  # setting as observations will be converted into arrays and these parameters
  # are used to shape them. In this example, they don't have that effect.
  env = web_environment.GMiniWoBWebEnvironment(
      base_url="file://{}/".format(FLAGS.data_dep_path),
      subdomain="gminiwob.generic_website",
      profile_length=5,
      number_of_fields=5,
      use_only_profile_key=False,
      number_of_dom_elements=150,
      dom_attribute_sequence_length=5,
      keyboard_action_size=5,
      kwargs_dict={
          "headless": FLAGS.run_headless_mode,
          "threading": False
      },
      step_limit=25,
      global_vocabulary=vocabulary_node.LockedVocabulary(),
      use_conceptual=FLAGS.use_conceptual)

  # Create a shopping website design with difficulty = 3.
  website = test_websites.create_shopping_website(3)
  design = test_websites.generate_website_design_from_created_website(
      website)

  # Design the actual environment.
  env.design_environment(
      design, auto_num_pages=True)
  # Make sure raw_state=True as this will return raw observations not numpy
  # arrays.
  state = env.reset(raw_state=True)

  # Optimal sequences of elements to visit. Some might be redundant and will be
  # skipped.
  optimal_actions = [
      "group_next_p0",
      "group_username",
      "group_password",
      "group_rememberme",
      "group_captcha",
      "group_stayloggedin",
      "group_next_p1",
      "group_next_p2",
      "group_name_first",
      "group_name_last",
      "group_address_line1",
      "group_address_line2",
      "group_city",
      "group_postal_code",
      "group_state",
      "group_submit_p2",
  ]
  # Corresponding pages of these elements:
  # [0, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3]

  reward = 0.0
  logging.info("Utterance: %s", str(state.utterance))
  logging.info("\n\n")
  logging.info("All available primitives: %s",
               str(env.get_all_actionable_primitives()))
  logging.info("\n\n")

  # Iterate over all optimal actions. For each action, iterate over all elements
  # in the current observation. If an element matches, execute the optimal
  # action and continue.

  # Iterate over optimal actions.
  for action in optimal_actions:
    logging.info("Element at focus: %s", str(action))
    # Iterate over all elements in the current observation.
    # order_dom_elements returns an ordered list of DOM elements to make the
    # order and elements consistent.
    for i, element in enumerate(
        utils.order_dom_elements(state.dom_elements, html_id_prefix=None)):
      # If HTML if of the element matches the action, execute the action.
      if element.id == action.replace("group", "actionable"):
        logging.info("Acting on (%s)", str(element))
        logging.info("\tAttributes of the element: %s",
                     str(utils.dom_attributes(element, 5)))
        # Get the corresponding profile fields.
        profile_keys = env.raw_profile.keys
        # Execute the (element index, profile field index) action on the
        # website. Environment step function accepts a single scalar action.
        # We flatten the action from a tuple to a scalar which is deflattened
        # back to a tuple in the step function.
        if action[len("group") +
                  1:] in profile_keys and not FLAGS.use_conceptual:
          logging.info("Profile: %s, Element ID: %s",
                       str(profile_keys.index(action[len("group") + 1:])),
                       str(action[len("group") + 1:]))
          # action=element_index + profile_field_index * number_of_elements
          # This is converted back into a tuple using a simple modulo
          # arithmetic.
          state, r, _, _ = env.step(
              i + profile_keys.index(action[len("group") + 1:]) *
              env.number_of_dom_elements, True)
        else:  # This is the case where we have abstract navigation problem.
          logging.info("Element ID: %s", str(action[len("group") + 1:]))
          # We don't need to convert a tuple into a scalar because in this case
          # the environment expects the index of the element.
          state, r, _, _ = env.step(i, True)
        logging.info("Current reward: %f", r)
        reward += r
        if not FLAGS.run_headless_mode:
          # wait 1 sec so that the action can be observed on the browser.
          time.sleep(1)
        break
  logging.info("Final reward: %f", reward)
  if not FLAGS.run_headless_mode:
    # wait 30 secs so that the users can inspect the html in the browser.
    time.sleep(30)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  run_policy_on_shopping_website()


if __name__ == "__main__":
  app.run(main)
