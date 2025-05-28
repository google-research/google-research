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

"""Util to Read TIDAL from https://github.com/google-research-datasets/TIDAL."""

from typing import Union

import xml.etree.ElementTree as ET
import pandas as pd
import requests


def read_tidal():
  """Read TIDAL dataset."""

  tidal_github_url = 'https://raw.githubusercontent.com/google-research-datasets/TIDAL/main/dataset/TIDAL.xml'

  response = requests.get(tidal_github_url)

  try:
    response.raise_for_status()
  except requests.exceptions.HTTPError as e:
    raise ValueError('Error downloading TIDAL file: ' + str(e)) from e

  # Parse the XML file
  root = ET.fromstring(response.text)

  # Extract the data
  data = []
  for lexical_entry in root.findall('Lexicon/LexicalEntry'):
    row = {}

    for child in lexical_entry:
      if child.tag == 'Lemma' or child.tag == 'WordForm':
        row['Term'] = child.find('FormRepresentation').attrib['writtenForm']
        if child.tag == 'Lemma':
          row['IsRootTerm'] = True
        else:
          row['IsRootTerm'] = False
      if child.tag == 'Sense':
        for context in child:
          for elem in context:
            if elem.tag == 'Connotation':
              row[elem.tag] = elem.attrib['value']
            elif elem.tag == 'HasNonIdentityMeaning':
              row[elem.tag] = bool(elem.attrib['value'])
            elif elem.tag == 'Group' or elem.tag == 'Subgroup':
              row[f'Identity{elem.tag}'] = elem.attrib['value']
              if elem.find('Provenance').attrib['id'] == 'HCOMP':
                row['IdentityGroup_Connotation_ConvergenceScore'] = int(
                    float(elem.find('Provenance').attrib['ConvergenceScore'])
                )
      if (
          child.tag == 'RelatedForm'
          and child.attrib['relType'] == 'PersonNounCombinationOf'
      ):
        row['IsPTCTerm'] = True
      else:
        row['IsPTCTerm'] = False
        row['IsPTCTerm'] = False

    data.append(row)

  # Convert the data to a dataframe
  df = pd.DataFrame(data).drop_duplicates().reset_index(drop=True)
  # Explicitly convert column to bool dtype
  df['HasNonIdentityMeaning'] = df['HasNonIdentityMeaning'].astype(bool)

  return df
