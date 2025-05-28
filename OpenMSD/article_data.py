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

r"""Define the class used for storing the papers' information."""

import json


class ArticleData:

  """Data structure used to store papers' information.

  Attributes:
    title: title of the paper
    venue: publication venue
    doi: DOI string
    urls: a list of strings, each being a URL for the paper
    abstract: abstract text
    content: content text
    language: language of the paper, either extracted from metadata or guessed
      from the papers title+abstract+content.
    category_labels: category labels extracted from CrossRef
    sources: openalex or crossref
  """

  def __init__(self):
    self.title = None
    self.venue = None
    self.doi = None
    self.urls = []
    self.abstract = None
    self.content = None
    self.language = None
    self.category_labels = []
    self.sources = []

  def deduplicate_urls(self):
    unique_urls = list(set(self.urls))
    self.urls = unique_urls

  def get_text_output(self):
    """Get a text output for ArticleData."""
    output_dict = {
        'title': self.title,
        'venue': self.venue,
        'doi': self.doi,
        'urls': self.urls,
        'abstract': self.abstract,
        'content': self.content,
        'language': self.language,
        'category_labels': self.category_labels,
        'sources': self.sources,
    }
    return json.dumps(output_dict)
