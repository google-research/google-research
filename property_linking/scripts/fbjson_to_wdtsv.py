# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Convert freebase json files (dev500) to a wikidata tsv, making the appropriate adjustments.

Usage:
python fbjson_to_wdtsv.py dev500.json mid_to_wd.tsv rels_fb_to_wd.tsv
                          yago_s_cats.tsv yago_s_cats_dev.tsv
"""
from __future__ import print_function

from collections import Counter  # pylint: disable=g-importing-member
import json
import sys


dev = open(sys.argv[1], "r")
read_file = json.load(dev)
mid_file = open(sys.argv[2], "r")
mid_dict = {f.split("\t")[2].strip()[2:]:
            f.split("\t")[1].strip()
            for f in mid_file}
manual_file = open(sys.argv[3], "r")
manual_dict = {}
for f in manual_file:
  if len(f.split("\t")) != 2:
    print (f)
  if f.split("\t")[1].strip():
    manual_dict[f.split("\t")[0].strip()] = f.split("\t")[1].strip()

# manual rules
mid_dict["?x"] = "?x"
mid_dict["?X"] = "?X"
mid_dict["?Y"] = "?Y"
mid_dict["?y"] = "?y"
mid_dict["rdf:type"] = "i/P31"
mid_dict.update(manual_dict)
missed = []


def find(word):
  """Find fbid and attempt to translate."""
  if word.startswith("ns"):
    if "/"+word[3:] in mid_dict:
      return mid_dict["/"+word[3:]]
    if word[3:] in mid_dict:
      return mid_dict[word[3:]]
  if word in mid_dict:
    return mid_dict[word]
  if word.startswith("/") and word[1:] in mid_dict:
    return mid_dict[word[1:]]
  if (not word.startswith("/m/")
      and not word.startswith("ns")
      and len(word) == 4):
    # years
    return "c/"+word
  missed.append(word)
  return word


def remap(cmps):
  cmps = [c.replace(".", "/") for c in cmps]
  cmps = [find(c) for c in cmps]
  return tuple(cmps)


def tripleize(exno, query):
  """Convert semantic parse logical form to relation triples."""
  components = query.split(" ")
  components = [c for c in components if c != "date"]
  if len(components) % 3 != 0:
    print (exno, len(components), query)
    return []
  properties = []
  i = 0
  while i < len(components):
    # convert these types to occupation (P106)
    if (components[i+1] == "rdf:type"
        and components[i+2] == "ns:soccer.football_player"):
      properties.append(("?x", "i/P106", "i/Q937857"))
    elif (components[i+1] == "rdf:type"
          and components[i+2] == "ns:basketball.basketball_player"):
      properties.append(("?x", "i/P106", "i/Q3665646"))
    elif (components[i+1] == "rdf:type"
          and components[i+2] == "ns:american_football.football_player"):
      properties.append(("?x", "i/P106", "i/Q19204627"))
    elif (components[i+1] == "rdf:type"
          and components[i+2] == "ns:military.military_person"):
      properties.append(("?x", "i/P106", "i/Q47064"))
    elif (components[i+1] == "rdf:type"
          and (components[i+2] == "ns:sports/pro_athlete" or
               components[i+2] == "ns:sports/pro_athelete")):
      properties.append(("?x", "i/P106", "i/Q2066131"))
    else:
      properties.append(remap((components[i],
                               components[i+1],
                               components[i+2])))

    i += 3
  return properties


def good(triple):
  for term in triple:
    if "ns:" in term or "/m/" in term:
      return False
  return True


def process_example(exno, ex):
  name = ex["noun_phrase"]
  property_set = set()
  for query in ex["query"]:
    property_set.update(tripleize(exno, query))
  return (name, property_set)

all_examples = {}
examples = []
bad_examples = []
bad_count = 0
for j in range(1, 501):
  if j == 136 or j == 165 or j == 363 or j == 381:
    # sad but will have to admit it manually
    # continue
    pass
  this_blob = read_file[str(j)]
  example = process_example(j, this_blob)
  example_count = (sum([good(x) for x in example[1]]), len(example[1]))
  all_examples[this_blob["noun_phrase"]] = example[1]
  examples.append("{}/{}|| {}".format(example_count[0],
                                      example_count[1],
                                      example))
  if example_count[0] == 0:
    bad_examples.append("{}/{}|| {}".format(example_count[0],
                                            example_count[1],
                                            example))
    bad_count += 1

# print (all_examples.items())
print ("\n".join(examples))
print (bad_count)

print (len(missed), len(set(missed)))
rels = [prop for prop in missed if prop.startswith("ns")]
ents = [prop for prop in missed if prop.startswith("/m/")]
print (len(rels), len(set(rels)))
print (len(ents), len(set(ents)))
# print (rels)
# print (ents)

# print [prop for prop in missed
#        if (not prop.startswith("/m/") and not prop.startswith("ns"))]
print ("\n".join(["{}".format(p) for p in Counter(rels).most_common()]))


def clean_triples(properties):
  """Convert triples to just relation + tail entity."""
  cleaned = []
  for triple in properties:
    if triple[1].startswith("i/P") and triple[0].startswith("i/Q"):
      if "?" not in triple[1] and "?" not in triple[0]:
        cleaned.append((triple[1], triple[0]))
        continue
    if triple[1].startswith("i/P") and triple[2].startswith("i/Q"):
      if "?" not in triple[1] and "?" not in triple[2]:
        cleaned.append((triple[1], triple[2]))
        continue
    # print (triple)
  return cleaned

nulls = 0
cleaned_examples = {}
for noun_phrase, all_property_sets in all_examples.items():
  # print (noun_phrase)
  matched_properties = clean_triples(all_property_sets)
  if matched_properties:
    # print ("nothing")
    nulls += 1
  else:
    # print ("props: {}".format(matched_properties))
    pass
  cleaned_examples[noun_phrase] = "|".join(["{},{}".format(p[0], p[1])
                                            for p in matched_properties])

# print (cleaned_examples)

cats_file = open(sys.argv[4], "r")
cats_dict = {line.split("\t")[1]: line for line in cats_file}
# print (cats_dict.keys()[:10])
output = []
lost = []
print (len(cleaned_examples))
for noun_phrase, property_string in cleaned_examples.items():
  if noun_phrase in cats_dict:
    existing_stuff = cats_dict[noun_phrase].strip().split("\t")
    while len(existing_stuff) < 3:
      existing_stuff.append("")
    existing_stuff.append(property_string)
    output.append("\t".join(existing_stuff))
  else:
    lost.append(noun_phrase)
# print len(output)

out_file = open(sys.argv[5], "w+")
for line in output:
  out_file.write(line + "\n")

# lost_file = open(sys.argv[6], "w+")
# for line in lost:
#   lost_file.write(line + "\n")
