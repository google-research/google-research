# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Extracting XPaths of the values of all fields for SWDE dataset."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import json
import os
import pickle
import random
import re
import sys
import unicodedata

from absl import app
from absl import flags
from bert import tokenization
import lxml
from lxml import etree
from lxml.html.clean import Cleaner
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from simpdom import constants

FLAGS = flags.FLAGS
random.seed(42)

flags.DEFINE_boolean(
    "build_circle_features", False,
    "If true, the circle features (friends and partner) will be extracted.")
flags.DEFINE_integer("n_pages", 2000, "The maximum number of pages to read.")
flags.DEFINE_integer("max_depth_search", 4,
                     "The maximum depth of DOM Tree to search.")
flags.DEFINE_integer("max_friends_num", 15,
                     "The maximum number of friends to extract.")
flags.DEFINE_integer("max_length_text", 10,
                     "The maximum lengths of the node texts.")
flags.DEFINE_integer(
    "max_xpath_dist", 2, "The maximum tolerable distance between two xpaths,"
    "used for removing irrelevant friends.")
flags.DEFINE_string(
    "input_groundtruth_path", "",
    "The root path to parent folder of all ground truth files.")
flags.DEFINE_string("input_pickle_path", "",
                    "The root path to pickle file of swde html content.")
flags.DEFINE_string(
    "output_data_path", "",
    "The path of the output file containing both the input sequences and "
    "output sequences of the sequence tagging version of swde dataset.")
flags.DEFINE_string("vertical", "", "The vertical to run.")
flags.DEFINE_string("website", "", "The website to run.")


def clean_spaces(text):
  r"""Clean extra spaces in a string.

  Example:
    input: " asd  qwe   " --> output: "asd qwe"
    input: " asd\t qwe   " --> output: "asd qwe"
  Args:
    text: the input string with potentially extra spaces.

  Returns:
    a string containing only the necessary spaces.
  """
  return " ".join(re.split(r"\s+", text.strip()))


def clean_format_str(text):
  """Cleans unicode control symbols, non-ascii chars, and extra blanks."""
  text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
  text = "".join([c if ord(c) < 128 else "" for c in text])
  text = clean_spaces(text)
  return text


def non_ascii_equal(website, field, value, node_text):
  """Compares value and node_text by their non-ascii texts.

  Website/field are used for handling special cases.

  Args:
    website: the website that the value belongs to, used for dealing with
      special cases.
    field: the field that the value belongs to, used for dealing with special
      cases.
    value: the value string that we want to compare.
    node_text: the clean text of the node that we want to compare.

  Returns:
    a boolean variable indicating if the value and node_text are equal.
  """
  value = clean_format_str(value)
  node_text = clean_format_str(node_text)

  # A special case in the ALLMOVIE website's MPAA_RATING,
  # the truth values are not complete but only the first character.
  # For example, truth value in the file:"P", which should be "PG13" in htmls.
  # Note that the length of the truth should be less than 5.
  if website == "allmovie" and field == "mpaa_rating" and len(node_text) <= 5:
    return node_text.strip().startswith(value.strip())

  # A special case in the AMCTV website, DIRECTOR field.
  # The name are not complete in the truth values.
  # E.g. truth value in files, "Roy Hill" and real value: "Geogre Roy Hill".
  if website == "amctv" and field == "director":
    return node_text.strip().endswith(value.strip())
  return value.strip() == node_text.strip()


def match_value_node(node, node_text, current_xpath_data, overall_xpath_dict,
                     text_part_flag, groundtruth_value, matched_xpaths, website,
                     field, dom_tree):
  """Matches the ground truth value with a specific node in the domtree.

  In the function, the current_xpath_data, overall_xpath_dict, matched_xpaths
  will be updated.

  Args:
    node: the node on the domtree that we are going to match.
    node_text: the text inside this node.
    current_xpath_data: the dictionary of the xpaths of the current domtree.
    overall_xpath_dict: the dictionary of the xpaths of the current website.
    text_part_flag: to match the "text" or the "tail" part of the node.
    groundtruth_value: the value of our interest to match.
    matched_xpaths: the existing matched xpaths list for this value on domtree.
    website: the website where the value is from.
    field: the field where the value is from.
    dom_tree: the current domtree object, used for getting paths.
  """
  assert text_part_flag in ["node_text", "node_tail_text"]
  # Dealing with the cases with multiple <br>s in the node text,
  # where we need to split and create new tags of matched_xpaths.
  # For example, "<div><span>asd<br/>qwe</span></div>"
  len_brs = len(node_text.split("--BRRB--"))  # The number of the <br>s.
  for index, etext in enumerate(node_text.split("--BRRB--")):
    if text_part_flag == "node_text":
      xpath = dom_tree.getpath(node)
    elif text_part_flag == "node_tail_text":
      xpath = dom_tree.getpath(node) + "/tail"
    if len_brs >= 2:
      xpath += "/br[%d]" % (index + 1)  # E.g. /div/span/br[1]
    clean_etext = clean_spaces(etext)
    # Exactly match the text.
    if non_ascii_equal(website, field, groundtruth_value, clean_etext):
      matched_xpaths.append(xpath)
    # Update the dictionary.
    current_xpath_data[xpath] = clean_etext
    overall_xpath_dict[xpath].add(clean_etext)


def get_value_xpaths(dom_tree,
                     truth_value,
                     overall_xpath_dict,
                     website="",
                     field=""):
  """Gets a list of xpaths that contain a text truth_value in DOMTree objects.

  Args:
    dom_tree: the DOMTree object of a specific HTML page.
    truth_value: a certain groundtruth value.
    overall_xpath_dict: a dict maintaining all xpaths data of a website.
    website: the website name.
    field: the field name.

  Returns:
    xpaths: a list of xpaths containing the truth_value exactly as inner texts.
    current_xpath_data: the xpaths and corresponding values in this DOMTree.
  """
  if not truth_value:
    #  Some values are empty strings, that are not in the DOMTree.
    return []

  xpaths = []  # The resulting list of xpaths to be returned.
  current_xpath_data = dict()  # The resulting dictionary to save all page data.

  # Some values contains HTML tags and special strings like "&nbsp;"
  # So we need to escape the HTML by parsing and then extract the inner text.
  value_dom = lxml.html.fromstring(truth_value)
  value = " ".join(etree.XPath("//text()")(value_dom))
  value = clean_spaces(value)

  # Iterate all the nodes in the given DOMTree.
  for e in dom_tree.iter():
    # The value can only be matched in the text of the node or the tail.
    if e.text:
      match_value_node(
          e,
          e.text,
          current_xpath_data,
          overall_xpath_dict,
          text_part_flag="node_text",
          groundtruth_value=value,
          matched_xpaths=xpaths,
          website=website,
          field=field,
          dom_tree=dom_tree)
    if e.tail:
      match_value_node(
          e,
          e.tail,
          current_xpath_data,
          overall_xpath_dict,
          text_part_flag="node_tail_text",
          groundtruth_value=value,
          matched_xpaths=xpaths,
          website=website,
          field=field,
          dom_tree=dom_tree)
  return xpaths, current_xpath_data


def get_dom_tree(html, website):
  """Parses a HTML string to a DOMTree.

  We preprocess the html string and use lxml lib to get a tree structure object.

  Args:
    html: the string of the HTML document.
    website: the website name for dealing with special cases.

  Returns:
    A parsed DOMTree object using lxml libraray.
  """
  cleaner = Cleaner()
  cleaner.javascript = True
  cleaner.style = True
  html = html.replace("\0", "")  # Delete NULL bytes.
  # Replace the <br> tags with a special token for post-processing the xpaths.
  html = html.replace("<br>", "--BRRB--")
  html = html.replace("<br/>", "--BRRB--")
  html = html.replace("<br />", "--BRRB--")
  html = html.replace("<BR>", "--BRRB--")
  html = html.replace("<BR/>", "--BRRB--")
  html = html.replace("<BR />", "--BRRB--")

  # A special case in this website, where the values are inside the comments.
  if website == "careerbuilder":
    html = html.replace("<!--<tr>", "<tr>")
    html = html.replace("<!-- <tr>", "<tr>")
    html = html.replace("<!--  <tr>", "<tr>")
    html = html.replace("<!--   <tr>", "<tr>")
    html = html.replace("</tr>-->", "</tr>")

  html = clean_format_str(html)
  etree_root = cleaner.clean_html(lxml.html.fromstring(html))
  dom_tree = etree.ElementTree(etree_root)
  return dom_tree


def clean_footer_nav(dom_tree, website):
  """Clears the navigation bars and footers in DOM trees."""

  # Remove nodes in navigation bars and footer to deliminate noise.
  # Translate was used to convert upper case to lower case in Xpath 1.0.
  nav1 = dom_tree.xpath("//*[contains(translate(@class, 'ANV', 'anv'),'nav')]")
  nav2 = dom_tree.xpath("//*[contains(translate(@id, 'ANV', 'anv'),'nav')]")
  footer1 = dom_tree.xpath(
      "//*[contains(translate(@class, 'EFORT', 'efort'),'footer')]")
  footer2 = dom_tree.xpath(
      "//*[contains(translate(@id, 'EFORT', 'efort'),'footer')]")

  # Special cases in websites "collegenavigator, careerbuilder, yahoo,
  # rottentomatoes, matchcollege, studentaid".
  if website == "collegenavigator":
    nav3 = dom_tree.xpath("//*[contains(@id,'CollegeNavBody')]")
    nav2 = list(set(nav2) - set(nav3))
  if website == "careerbuilder":
    nav3 = dom_tree.xpath("//*[contains(@class,'overview_nav')]")
    nav1 = list(set(nav1) - set(nav3))
    nav2 = list(set(nav2) - set(nav3))
  if website == "yahoo":
    nav3 = dom_tree.xpath("//*[contains(@class,'nba-player-nav')]")
    nav1 = list(set(nav1) - set(nav3))
    nav2 = list(set(nav2) - set(nav3))
  if website == "rottentomatoes":
    nav3 = dom_tree.xpath("//*[contains(@id,'nav_shadow_bg')]")
    nav2 = list(set(nav2) - set(nav3))
  if website == "matchcollege":
    nav3 = dom_tree.xpath("//*[contains(@id, 'navarro_college')]")
    nav2 = list(set(nav2) - set(nav3))
  if website == "studentaid":
    nav3 = dom_tree.xpath("//*[contains(@id, 'tournavbar')]")
    nav2 = list(set(nav2) - set(nav3))

  for node in nav1 + nav2 + footer1 + footer2:
    try:
      # A special case in borders website where "nav" appears in the body xpath.
      if node.tag == "body":
        continue
      node.getparent().remove(node)
    except AttributeError:
      print("'NoneType' object has no attribute 'remove'.")

  return dom_tree


def load_html_and_groundtruth(vertical_to_load, website_to_load):
  """Loads and returns the html sting and ground turth data as a dictionary."""
  all_data_dict = collections.defaultdict(dict)
  vertical_to_websites_map = constants.VERTICAL_WEBSITES
  gt_path = FLAGS.input_groundtruth_path
  for v in vertical_to_websites_map:
    for truthfile in tf.gfile.ListDirectory(os.path.join(gt_path, v)):
      # For example, a groundtruth file name can be "auto-yahoo-price.txt".
      vertical, website, field = truthfile.replace(".txt", "").split("-")

      if vertical != vertical_to_load or website != website_to_load:
        continue
      with tf.gfile.Open(os.path.join(gt_path, v, truthfile), "r") as gfo:
        lines = gfo.readlines()
        for line in lines[2:]:
          # Each line should contains more than 3 elements splitted by \t
          # which are: index, number of values, value1, value2, etc.
          item = line.strip().split("\t")
          index = item[0]
          num_values = int(item[1])  # Can be 0 (when item[2] is "<NULL>").
          all_data_dict[index]["field-" +
                               field] = dict(values=item[2:2 + num_values])

  print("Reading the pickle of SWDE original dataset.....", file=sys.stderr)
  with tf.gfile.Open(FLAGS.input_pickle_path, "rb") as gfo:
    swde_html_data = pickle.load(gfo)
  for page in tqdm(swde_html_data, desc="Loading HTML data"):
    path = page["path"]  # For example, auto/auto-aol(2000)/0000.htm
    html_str = page["html_str"]
    vertical, website, index = path.split("/")
    website = website[website.find("-") + 1:website.find("(")]
    if vertical != vertical_to_load or website != website_to_load:
      continue
    index = index.replace(".htm", "")
    all_data_dict[index]["html_str"] = html_str
    all_data_dict[index]["path"] = path
  return all_data_dict


def get_field_xpaths(all_data_dict, vertical_to_process, website_to_process,
                     n_pages, max_variable_nodes_per_website):
  """Gets xpaths data for each page in the data dictionary.

  Args:
    all_data_dict: the dictionary saving both the html content and the truth.
    vertical_to_process: the vertical that we are working on;
    website_to_process: the website that we are working on.
    n_pages: we will work on the first n_pages number of the all pages.
    max_variable_nodes_per_website: top N frequent variable nodes as the final
      set.
  """
  # Saving the xpath info of the whole website,
  #  - Key is a xpath.
  #  - Value is a set of text appeared before inside the node.
  overall_xpath_dict = collections.defaultdict(set)
  #  Update page data with groundtruth xpaths and the overall xpath-value dict.
  for index in tqdm(
      all_data_dict, desc="Processing %s" % website_to_process, total=n_pages):
    if int(index) >= n_pages:
      continue
    page_data = all_data_dict[index]
    html = page_data["html_str"]
    dom_tree = get_dom_tree(html, website=website_to_process)
    page_data["dom_tree"] = dom_tree
    # Match values of each field for the current page.
    for field in page_data:
      if not field.startswith("field-"):
        continue
      # Saving the xpaths of the values for each field.
      page_data[field]["groundtruth_xpaths"] = set()
      for value in page_data[field]["values"]:
        xpaths, current_xpath_data = get_value_xpaths(dom_tree, value,
                                                      overall_xpath_dict,
                                                      website_to_process,
                                                      field[6:])
        # Assert each truth value can be founded in >=1 nodes.
        assert len(xpaths) >= 1, "%s;\t%s;\t%s;\t%s; is not found" % (
            website_to_process, field, index, value)
        # Update the page-level xpath information.
        page_data[field]["groundtruth_xpaths"].update(xpaths)
    page_data["xpath_data"] = current_xpath_data

  # Define the fixed-text nodes and variable nodes.
  fixed_nodes = set()
  variable_nodes = set()
  node_variability = sorted([
      (xpath, len(text_set)) for xpath, text_set in overall_xpath_dict.items()
  ],
                            key=lambda x: x[1],
                            reverse=True)

  for xpath, variability in node_variability:
    if variability > 5 and len(variable_nodes) < max_variable_nodes_per_website:
      variable_nodes.add(xpath)
    else:
      fixed_nodes.add(xpath)

  print("Vertical: %s; Website: %s; fixed_nodes: %d; variable_nodes: %d" %
        (vertical_to_process, website_to_process, len(fixed_nodes),
         len(variable_nodes)))
  assure_value_variable(all_data_dict, variable_nodes, fixed_nodes, n_pages)
  all_data_dict["fixed_nodes"] = list(fixed_nodes)
  all_data_dict["variable_nodes"] = list(variable_nodes)
  return


def assure_value_variable(all_data_dict, variable_nodes, fixed_nodes, n_pages):
  """Makes sure all values are in the variable nodes by updating sets.

  Args:
    all_data_dict: the dictionary saving all data with groundtruth.
    variable_nodes: the current set of variable nodes.
    fixed_nodes: the current set of fixed nodes.
    n_pages: to assume we only process first n_pages pages from each website.
  """
  for index in all_data_dict:
    if not index.isdigit() or int(index) >= n_pages:
      # the key should be an integer, to exclude "fixed/variable nodes" entries.
      # n_pages to stop for only process part of the website.
      continue
    for field in all_data_dict[index]:
      if not field.startswith("field-"):
        continue
      xpaths = all_data_dict[index][field]["groundtruth_xpaths"]
      if not xpaths:  # There are zero value for this field in this page.
        continue
      flag = False
      for xpath in xpaths:
        if flag:  # The value's xpath is in the variable_nodes.
          break
        flag = xpath in variable_nodes
      variable_nodes.update(xpaths)  # Add new xpaths if they are not in.
      fixed_nodes.difference_update(xpaths)


def get_previous_texts(dom_tree, variable_value_set, n_prev_nodes=3):
  """Extracts the texts from the previous three nodes as features for each node.

  Args:
    dom_tree:  the DOMTree that we will to the search.
    variable_value_set: the set of variable values in this page.
    n_prev_nodes: the number of the previous nodes that we need to look at.

  Returns:
    prev_texts: a dictionary mapping the xpath and its list of previous texts.
  """
  prev_texts = dict()
  prev_text_buffer = []
  for e in dom_tree.iter():
    cur_path = dom_tree.getpath(e)
    prev_texts[cur_path] = prev_text_buffer[-n_prev_nodes:]
    if e.text:
      text = clean_spaces(e.text)
      if text and text not in variable_value_set:
        prev_text_buffer.append(text)
    if e.tail:
      text = clean_spaces(e.tail)
      if text and text not in variable_value_set:
        prev_text_buffer.append(text)
  return prev_texts


def reorder_candidate(candidates, text):
  """Reorder the candidate list by prioritizing the closest friends."""

  # For example, candidates: [a, b, c, target, d, e, f] -> [d, c, e, b, f, a].
  # Every candidate is a two-element tuple (xpath, text).
  reverse, result = [], []
  text_met = False
  # Make sure the current node is in the candidates,
  # otherwise cannot locate its closest friends.
  if text not in [x[1] for x in candidates]:
    return []
  for candidate in candidates:
    # Locate the current node and record the next/previous nodes one by one.
    if not text_met:
      # candidate[0]: xpath, candidate[1]: text.
      if candidate[1] != text:
        reverse.append(candidate)
      else:
        text_met = True
    else:
      if reverse:
        result.append(reverse.pop())
      result.append(candidate)
  # Keep the remaining candidates in "reverse" list.
  while reverse:
    result.append(reverse.pop())

  return result


def clean_noise(text):
  """Cleans noisy nodes."""

  noise = {",", "|", ".", "-", "*", "", "/", " ", ":", ";", ">", "<"}
  if text in noise:
    return ""
  else:
    return text


def xpath_distance(path1, path2):
  """Quantify the difference between two xpaths."""

  # For example:
  # xpath_distance('/div/body/div[2]/table[1]/tr[2]/td[3]/div/div[4]/span',
  #                '/div/body/div[2]/table[1]/tr[2]/td[2]/h2') = 3
  formatting_tags = {
      "strong", "a", "b", "font", "br", "span", "tail", "em", "img", "u", "i"
  }

  def clean_tag(path):
    path = [x.split("[")[0] for x in path.split("/")]
    return list(filter(lambda x: x not in formatting_tags, path))

  path_list1 = clean_tag(path1)
  path_list2 = clean_tag(path2)
  distance = abs(len(path_list1) - len(path_list2))
  for i in range(min(len(path_list1), len(path_list2))):
    if path_list1[i] != path_list2[i]:
      distance += 1
  return distance


def get_xpath_to_descendants_dict(page, depth=4):
  """Extracts the mappings from xpath to its descendants for each page.

  Args:
    page: dict, A mapping from xpath to text in each web page.
    depth: int, A threshold to limit the searching range in the DOM tree.

  Returns:
    xpath_to_descendants: dict, A mapping from xpath to its descendants.

  Example:
  ('/div/body/div/table[4]/tr[2]/td[1]/p[1]/font[1]/b', 'Location') ->
  {
   '/div/body/div/table[4]/tr[2]/td[1]/p[1]/font[1]/b':[('', 'Location')],
   '/div/body/div/table[4]/tr[2]/td[1]/p[1]/font[1]': [('b', 'Location')],
   '/div/body/div/table[4]/tr[2]/td[1]/p[1]': [('font[1]/b', 'Location')],
   '/div/body/div/table[4]/tr[2]/td[1]': [('p[1]/font[1]/b', 'Location')],
   }
  """

  xpath_to_descendants = collections.defaultdict(list)
  for path, text in page.items():
    text = clean_noise(clean_spaces(text))
    path_list = path.split("/")
    if len(path_list) < depth or not text:
      continue
    xpath_to_descendants[path].append(("", text))
    level = depth
    # Regard itself as its first descendant. Intend to form potential node
    # pairs where parent and child both contain texts.
    # Example:
    # '/div/body/div/table[4]/tr[2]/td[1]/p[1]/font[1]/b':[('', 'Location'),
    #                                                  ('tail', 'Sunnyvale')]
    while level > 0:
      xpath_to_descendants["/".join(path_list[:-level])].append(
          ("/".join(path_list[-level:]), text))
      level -= 1
  return xpath_to_descendants


def get_circle(page, variable_value_set, depth=4):
  """Extracts the friend circle features for each node.

  Args:
    page: The DOMTree that we will do the search.
    variable_value_set: The set of text in variable nodes of this page.
    depth: number of DOMTree depths to move upper when extracting
      partner/friends.

  Returns:
    circle: mapping from each node's xpath to its partner and friends.
  """

  def combine_paths(prefix, suffix):
    return prefix + "/" + suffix if suffix else prefix

  circle = dict()
  xpath_to_descendants = get_xpath_to_descendants_dict(page, depth)

  for path, text in page.items():
    if text not in variable_value_set:
      continue
    text = clean_noise(clean_spaces(text))
    if not text:
      continue

    level = 1
    path_list = path.split("/")
    # The friends_var_list denotes a list of variable friends.
    # while friends_fix_list denotes a list of fixed friends.
    partner_set, friends_fix_list, friends_var_list = set(), list(), list()

    # Search from the bottom to the top of the tree for partner and friends.
    while level <= depth:
      # Use prefix as the key to retrieve all the descendants.
      prefix = "/".join(path_list[:-level])
      candidates = xpath_to_descendants[prefix]
      candidates = [(c_xpath, " ".join(c_text.split(" ")))
                    for (c_xpath, c_text) in candidates]

      # Reorder the candidates to prioritize the closer friends.
      candidates = reorder_candidate(candidates, text)
      fix_candidates = list(
          filter(lambda x: x[1] not in variable_value_set, candidates))
      var_candidates = list(
          filter(lambda x: x[1] in variable_value_set, candidates))

      # Partner should be unique.
      if len(fix_candidates) == 1:
        (c_xpath, c_text) = fix_candidates[0]
        # Store the full xpath with text in a set to remove duplicates.
        partner_set.add((combine_paths(prefix, c_xpath), c_text))

      # Friends can be multiple.
      if len(var_candidates) > 1:
        for (c_xpath, c_text) in var_candidates:
          full_path = combine_paths(prefix, c_xpath)
          # Remove duplicates got from the previous loops.
          if (full_path, c_text) in friends_var_list:
            continue
          # Drop it if its xpath is very different from the current one.
          if xpath_distance(full_path, path) > FLAGS.max_xpath_dist:
            continue
          friends_var_list.append((full_path, c_text))

      if len(fix_candidates) > 1:
        for (c_xpath, c_text) in fix_candidates:
          full_path = combine_paths(prefix, c_xpath)
          if (full_path, c_text) in friends_fix_list + list(partner_set):
            continue
          if xpath_distance(full_path, path) > FLAGS.max_xpath_dist:
            continue
          friends_fix_list.append((full_path, c_text))

      level += 1

    circle[path] = {
        "partner": list(partner_set),
        "friends": friends_fix_list[:FLAGS.max_friends_num],
        "friends_var": friends_var_list[:FLAGS.max_friends_num]
    }
  return circle


def generate_nodes_seq_and_write_to_file(vertical, website):
  """Extracts all the xpaths and labels the nodes for all the pages."""
  tokenizer = tokenization.BasicTokenizer()
  all_data_dict = load_html_and_groundtruth(vertical, website)
  get_field_xpaths(
      all_data_dict,
      vertical_to_process=vertical,
      website_to_process=website,
      n_pages=FLAGS.n_pages,
      max_variable_nodes_per_website=300)
  # Label all variable nodes with their types of truth.
  nodes_seq_data = dict()
  all_label_dict = collections.defaultdict(lambda: 0)
  nodes_seq_data["fixed_nodes"] = all_data_dict["fixed_nodes"]
  nodes_seq_data["variable_nodes"] = all_data_dict["variable_nodes"]
  nodes_seq_data["features"] = list()  # Features for each page.
  for index in all_data_dict:
    page_features = []
    if not index.isdigit():
      # Skip the cases when index is actually the "fixed/variable_nodes" keys.
      continue
    if int(index) >= FLAGS.n_pages:
      break
    page_data = all_data_dict[index]
    if "xpath_data" not in page_data:
      continue
    dom_tree = page_data["dom_tree"]

    # Annotate every nodes.
    variable_value_set = set()
    for xpath, text in page_data["xpath_data"].items():
      if xpath in nodes_seq_data["variable_nodes"]:
        variable_value_set.add(text)

    prev_texts = get_previous_texts(dom_tree, variable_value_set)
    if FLAGS.build_circle_features:
      # Construct friend circle features.
      circle = get_circle(
          page_data["xpath_data"],
          variable_value_set,
          depth=FLAGS.max_depth_search)

    for xpath, text in page_data["xpath_data"].items():
      if xpath not in nodes_seq_data["variable_nodes"]:
        continue
      # Confirm the field.
      label = "none"
      for field in page_data:
        if not field.startswith("field-"):
          continue
        if xpath in page_data[field]["groundtruth_xpaths"]:
          label = field.replace("field-", "")
          break
      # Mapping the BR xpaths back to the original xpath.
      xpath_no_tail = xpath.replace("/tail", "")
      xpath_units = xpath_no_tail.split("/")
      if xpath_units[-1].startswith("br"):
        mapped_xpath = "/".join(xpath_units[:-1])
      else:
        mapped_xpath = xpath_no_tail
      prev_text_mapped = prev_texts[mapped_xpath]
      if mapped_xpath != xpath and mapped_xpath in page_data["xpath_data"]:
        prev_text_mapped.append(page_data["xpath_data"][mapped_xpath])
      # Add one entry in the data results.
      node_features = dict(
          html_path=page_data["path"],
          xpath=xpath,
          label=label,
          text=tokenizer.tokenize(clean_format_str(text).replace(
              "$", "$ "))[:FLAGS.max_length_text],
          prev_text=[
              tokenizer.tokenize(clean_format_str(t))[-FLAGS.max_length_text:]
              for t in prev_text_mapped
          ])
      if FLAGS.build_circle_features:
        # Safely retrieve text features from circle.
        node_circle = circle.get(xpath, {})
        friends = [x[1] for x in node_circle.get("friends", [])]
        friends_var = [x[1] for x in node_circle.get("friends_var", [])]
        partner = [x[1] for x in node_circle.get("partner", [])]
        # Get the graph neighbors for each node (xpaths).
        neighbors = [x[0] for x in node_circle.get("friends_var", [])]
        # Add four features to each data piece.
        node_features["partner_text"] = [
            tokenizer.tokenize(clean_format_str(t))[:FLAGS.max_length_text]
            for t in partner
        ]
        node_features["friends_text"] = [
            tokenizer.tokenize(clean_format_str(t))[:FLAGS.max_length_text]
            for t in friends
        ]
        # Follow the processing step for node text to add a space after "$".
        node_features["friends_var_text"] = [
            tokenizer.tokenize(clean_format_str(t).replace(
                "$", "$ "))[:FLAGS.max_length_text] for t in friends_var
        ]
        node_features["neighbors"] = neighbors
      page_features.append(node_features)
      # Count the number of entries of each label.
      all_label_dict[label] += 1
    nodes_seq_data["features"].append(page_features)

  print(vertical, website, "all_labels:", all_label_dict, file=sys.stderr)
  with tf.gfile.Open(
      os.path.join(FLAGS.output_data_path,
                   "%s-%s" % (vertical, website) + ".json"), "w") as gfo:
    json.dump(nodes_seq_data, gfo, indent=2)


def main(_):
  vertical_to_websites_map = constants.VERTICAL_WEBSITES
  verticals = [FLAGS.vertical
              ] if FLAGS.vertical else vertical_to_websites_map.keys()
  for vertical in verticals:
    websites = [FLAGS.website
               ] if FLAGS.website else vertical_to_websites_map[vertical]
    for website in websites:
      generate_nodes_seq_and_write_to_file(vertical, website)


if __name__ == "__main__":
  app.run(main)
