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

"""Predefined test websites in gMiniWoB using design primitives."""

import json

from CoDE import web_primitives
import gin


# Below is a list of predefined websites for testing
@gin.configurable
def create_login_pages(difficulty, is_final_page=True):
  """Create a login website with a single login page.

  Args:
    difficulty: Difficulty level of the website. 0 indexed.
    is_final_page: If true, the login page is the final page in the website and
      a submit button will be added. Otherwise, a next button will be added. By
      default, we assume this is the final page.

  Returns:
    A dictionary of primitives for the login page in the website.
  """
  if difficulty == 0:
    site = {0: ['username', 'password']}
  elif difficulty == 1:
    site = {0: ['navbar', 'username', 'password', 'rememberme']}
  elif difficulty == 2:
    site = {
        0: [
            'navbar', 'username', 'password', 'rememberme', 'stayloggedin',
            'forgotusername'
        ]
    }
  else:
    site = {
        0: [
            'navbar', 'username', 'password', 'rememberme', 'stayloggedin',
            'captcha', 'forgotusername', 'forgotpassowrd'
        ]
    }
  if is_final_page:
    site[0].append('submit')
  else:
    site[0].append('next_login')
  return site


@gin.configurable
def create_address_pages(difficulty, is_final_page=True):
  """Create an address website with a single address form page.

  Args:
    difficulty: Difficulty level of the website. 0 indexed.
    is_final_page: If true, the login page is the final page in the website and
      a submit button will be added. Otherwise, a next button will be added. By
      default, we assume this is the final page.

  Returns:
    A dictionary of primitives for the address form page in the website.
  """
  if difficulty == 0:
    site = {0: ['firstname', 'lastname']}
  elif difficulty == 1:
    site = {0: ['navbar', 'firstname', 'lastname', 'addressline1']}
  elif difficulty == 2:
    site = {
        0: [
            'navbar', 'firstname', 'lastname', 'addressline1', 'addressline2',
            'city'
        ]
    }
  else:
    site = {
        0: [
            'navbar', 'firstname', 'lastname', 'addressline1', 'addressline2',
            'city', 'zipcode', 'state'
        ]
    }
  if is_final_page:
    site[0].append('submit')
  else:
    site[0].append('next_checkout')
  return site


@gin.configurable
def create_home_page_and_login(difficulty):
  """Create a website with a home page and login page.

  Args:
    difficulty: Difficulty level of the website. 0 indexed.

  Returns:
    A dictionary of primitives for each page in the website.
  """
  if difficulty == 0:
    home = {0: ['navbar', 'next_login_page']}
  elif difficulty == 1:
    home = {0: ['navbar', 'carousel', 'next_login_page']}
  elif difficulty == 2:
    home = {0: ['navbar', 'carousel', 'dealmedia', 'next_login_page']}
  else:
    home = {
        0: [
            'navbar', 'inpgroup1', 'carousel', 'dealmedia', 'deck', 'footer1',
            'next_login_page'
        ]
    }
  login = create_login_pages(difficulty, is_final_page=False)
  home[1] = login[0]
  return home


@gin.configurable
def create_home_page_and_address(difficulty):
  """Create a website with a home page and address page.

  Args:
    difficulty: Difficulty level of the website. 0 indexed.

  Returns:
    A dictionary of primitives for each page in the website.
  """
  if difficulty == 0:
    home = {0: ['navbar', 'next_login_page']}
  elif difficulty == 1:
    home = {0: ['navbar', 'carousel', 'next_login_page']}
  elif difficulty == 2:
    home = {0: ['navbar', 'carousel', 'dealmedia', 'next_login_page']}
  else:
    home = {
        0: [
            'navbar', 'inpgroup1', 'carousel', 'dealmedia', 'deck', 'footer1',
            'next_login_page'
        ]
    }
  address = create_address_pages(difficulty, is_final_page=False)
  home[1] = address[0]
  return home


@gin.configurable
def create_shopping_website(difficulty):
  """Create a shopping website with a home page and login and address pages.

  Args:
    difficulty: Difficulty level of the website. 0 indexed.

  Returns:
    A dictionary of primitives for each page in the website.
  """
  if difficulty == 0:
    home = {0: ['navbar', 'next_login_page']}
  elif difficulty == 1:
    home = {0: ['navbar', 'carousel', 'next_login_page']}
  elif difficulty == 2:
    home = {0: ['navbar', 'carousel', 'dealmedia', 'next_login_page']}
  else:
    home = {
        0: [
            'navbar', 'inpgroup1', 'carousel', 'dealmedia', 'deck', 'footer1',
            'next_login_page'
        ]
    }
  login = create_login_pages(difficulty, is_final_page=False)
  home[1] = login[0]
  address = create_address_pages(difficulty, is_final_page=True)
  home[2] = address[0]
  return home


@gin.configurable
def create_flight_website(difficulty):
  """Create a flight website with a single flight page.

  Args:
    difficulty: Difficulty level of the website. 0 indexed.

  Returns:
    A dictionary of primitives for each page in the website.
  """
  if difficulty == 0:
    return {
        0: [
            'navbar', 'departureairport', 'destinationairport', 'footer1',
            'submit'
        ]
    }
  elif difficulty == 1:
    return {
        0: [
            'navbar', 'departureairport', 'destinationairport', 'departuredate',
            'destinationdate', 'footer1', 'submit'
        ]
    }
  elif difficulty == 2:
    return {
        0: [
            'navbar', 'departureairport', 'destinationairport', 'departuredate',
            'destinationdate', 'numberofpeople', 'cabin', 'footer1', 'submit'
        ]
    }
  else:
    return {
        0: [
            'navbar', 'departureairport', 'destinationairport', 'departuredate',
            'destinationdate', 'numberofpeople', 'cabin', 'flighttype',
            'footer1', 'submit'
        ]
    }


@gin.configurable
def create_payment_website(difficulty):
  """Create a payment website with a single payment page.

  Args:
    difficulty: Difficulty level of the website. 0 indexed.

  Returns:
    A dictionary of primitives for each page in the website.
  """
  if difficulty == 0:
    return {0: ['navbar', 'cc', 'fullname', 'footer1', 'submit']}
  elif difficulty == 1:
    return {0: ['navbar', 'cc', 'fullname', 'ccnumber', 'footer1', 'submit']}
  elif difficulty == 2:
    return {
        0: [
            'navbar', 'cc', 'fullname', 'ccnumber', 'ccexpdate', 'footer1',
            'submit'
        ]
    }
  else:
    return {
        0: [
            'navbar', 'cc', 'fullname', 'ccnumber', 'ccexpdate', 'cccvv',
            'footer1', 'submit'
        ]
    }


@gin.configurable
def create_global_website():
  """Create a global website with the largest possible profile.

  Returns:
    A dictionary of primitives for each page in the website.
  """
  return {0: web_primitives.CONCEPTS}


def generate_website_design_from_created_website(website,
                                                 predefined_transitions=False):
  """Create a website from given design primitive names.

  Args:
    website: A list of primitive names that can be generated from the test
      website functions in this file.
    predefined_transitions: A list of predefined transitions. If given, these
      will be added to the website. These should be actual transition
      descriptions, not transition names. See gminiwob_we_primitives for a list
      of transition descriptions.

  Returns:
    A design object that can be used in gminiwob web environment to create
    a real website.
  """
  design = {'number_of_pages': len(website), 'action': [], 'action_page': []}
  for page in website:
    for key in website[page]:
      design['action'].append(web_primitives.CONCEPTS.index(key))
      design['action_page'].append(page)
  if predefined_transitions:
    for transition in predefined_transitions:
      design['action'].append(transition)
      design['action_page'].append(-1)  # Transitions already assigned pages
  return design


def generate_msgi_all_core_precondition_transitions(website):
  """Generate all precondition transitions for MSGI.

  Args:
    website: A dictionary of website definition in similar format as test
      websites from this file.
  Returns:
    A list of transitions that can be used with gMiniWoB to create new websites
    with preconditions.
  """
  transitions = []
  for page_index in website:
    page = website[page_index]
    preconditions = []
    for primitive_name in page:
      if '"is_core_concept": true' in web_primitives.CONCEPTS2DESIGN[
          primitive_name]:
        preconditions.append('\"actionable_{}\"'.format(primitive_name))
    for primitive_name in page:
      # This means, there is a next button that only works after all core
      # primitives are interacted with.
      if primitive_name.startswith(
          'next'
      ):
        transition = json.loads(
            web_primitives.TRANSITIONS2DESIGN['addOpenPageTransition']
            .replace(web_primitives.SOURCE_PH,
                     'group_next_p{}'.format(page_index)).replace(
                         web_primitives.TARGET_PH,
                         'page{}'.format(page_index + 1)).replace(
                             web_primitives.PRECONDITION_PH,
                             ','.join(preconditions)))
        transitions.append(transition)
      # This means, there is a submit button that only works after all core
      # primitives are interacted with.
      elif primitive_name.startswith('submit'):
        transition = json.loads(
            web_primitives.TRANSITIONS2DESIGN['addSubmitTransition']
            .replace(web_primitives.SOURCE_PH,
                     'group_submit_p{}'.format(page_index)).replace(
                         web_primitives.PRECONDITION_PH,
                         ','.join(preconditions)))
        transitions.append(transition)
  return transitions
