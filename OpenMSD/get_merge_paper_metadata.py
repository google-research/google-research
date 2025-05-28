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

r"""Merge the metadata of papers from CrossRef and OpenAlex."""

import argparse
import gzip
import json
import os
import re
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.io.fileio import MatchFiles
from apache_beam.options.pipeline_options import PipelineOptions
import article_data
import cld3


class MergeArticleData(beam.DoFn):
  """Merge the article metadata from Crossref and OpenAlex."""

  def _get_title(self, papers_list):
    for paper in papers_list:
      if paper.title:
        return paper.title
    return None

  def _get_language(self, papers_list):
    for paper in papers_list:
      if paper.language:
        return paper.language
    # If the abstract or the content of the paper is available, guess the
    # language from the text.
    abstract = self._get_abstract(papers_list)
    content = self._get_content(papers_list)
    if abstract or content:
      abstract = abstract if abstract else ''
      content = content if content else ''
      text = abstract + ' ' + content
      lang_pred = cld3.get_language(text)
      return lang_pred.language

  def _get_venue(self, papers_list):
    for paper in papers_list:
      if paper.venue:
        return paper.venue
    return None

  def _get_abstract(self, papers_list):
    for paper in papers_list:
      if paper.abstract:
        return paper.abstract
    return None

  def _get_content(self, papers_list):
    for paper in papers_list:
      if paper.content:
        return paper.content
    return None

  def _merge_all_urls(self, papers_list):
    merged_urls = []
    for paper in papers_list:
      merged_urls.extend(paper.urls)
    return list(set(merged_urls))

  def _merge_all_category_labels(self, papers_list):
    merged_cats = []
    for paper in papers_list:
      merged_cats.extend(paper.category_labels)
    return list(set(merged_cats))

  def process(self, papers_by_doi):
    doi = papers_by_doi[0]
    openalex_papers_list = papers_by_doi[1]['openalex']
    crossref_papers_list = papers_by_doi[1]['crossref']
    papers_list = openalex_papers_list + crossref_papers_list
    if not papers_list:
      return
    else:
      merged_article_data = article_data.ArticleData()
      merged_article_data.doi = doi
      merged_article_data.title = self._get_title(papers_list)
      merged_article_data.venue = self._get_venue(papers_list)
      merged_article_data.language = self._get_language(papers_list)
      merged_article_data.abstract = self._get_abstract(papers_list)
      merged_article_data.content = self._get_content(papers_list)
      merged_article_data.urls = self._merge_all_urls(papers_list)
      merged_article_data.category_labels = self._merge_all_category_labels(
          papers_list
      )
      if openalex_papers_list:
        merged_article_data.sources.append('openalex')
      if crossref_papers_list:
        merged_article_data.sources.append('crossref')
      yield merged_article_data


class ParseOpenalexJson(beam.DoFn):
  """Parse the json files from OpenAlex."""

  def process(self, input_row):
    paper_entry = json.loads(input_row)
    paper_info = article_data.ArticleData()
    if 'doi' in paper_entry and paper_entry['doi']:
      raw_doi = paper_entry['doi']
      if 'doi.org/' in raw_doi:
        extracted_doi = raw_doi.split('doi.org/')[1]
        paper_info.doi = extracted_doi
        paper_info.urls.append(raw_doi)
      else:
        paper_info.doi = raw_doi
        paper_info.urls.append('https://doi.org/' + raw_doi)
    if 'title' in paper_entry:
      paper_info.title = paper_entry['title']
    if 'primary_location' in paper_entry and paper_entry['primary_location']:
      if 'landing_page_url' in paper_entry['primary_location']:
        paper_info.urls.append(
            paper_entry['primary_location']['landing_page_url']
        )
    if 'language' in paper_entry:
      paper_info.language = paper_entry['language']
    if 'ids' in paper_entry:
      if 'pmid' in paper_entry['ids']:
        paper_info.urls.append(paper_entry['ids']['pmid'])
      if 'pmcid' in paper_entry['ids']:
        paper_info.urls.append(paper_entry['ids']['pmcid'])

    if paper_info.doi:
      paper_info.deduplicate_urls()
      yield (paper_info.doi, paper_info)


class ParseCrossrefJson(beam.DoFn):
  """Parse the json files from Crossref."""

  def _clean_abstract(self, abstract):
    abstract = re.sub(r'</?jats:.+>', '', abstract)
    abstract = re.sub(r'\\n', ' ', abstract)
    abstract = re.sub(r'\s+', ' ', abstract)
    return abstract.strip()

  def _clean_title(self, title):
    title = re.sub(r'</?mml:.+>', '', title)
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'</?sup>', ' ', title)
    return title.strip()

  def process(self, file_metadata):
    jsonfilename = file_metadata.path
    with gzip.open(jsonfilename, 'r') as fin:
      data = json.loads(fin.read().decode('utf-8'))
      for paper_entry in data['items']:
        paper_info = article_data.ArticleData()
        if 'DOI' in paper_entry:
          paper_info.doi = paper_entry['DOI']
        if 'URL' in paper_entry:
          paper_info.urls.append(paper_entry['URL'])
        if 'title' in paper_entry:
          cleaned_title = self._clean_title(paper_entry['title'][0])
          if cleaned_title:
            paper_info.title = cleaned_title
        if 'abstract' in paper_entry:
          cleaned_abstract = self._clean_abstract(paper_entry['abstract'])
          if cleaned_abstract:
            paper_info.abstract = cleaned_abstract
        if 'container-title' in paper_entry:
          paper_info.venue = paper_entry['container-title'][0]
        if 'subject' in paper_entry:
          paper_info.category_labels.extend(paper_entry['subject'])
        # Get URLs from other fields.
        if 'link' in paper_entry:
          for link_dict in paper_entry['link']:
            if 'URL' in link_dict:
              paper_info.urls.append(link_dict['URL'])
        if 'resource' in paper_entry and 'primary' in paper_entry['resource']:
          if 'URL' in paper_entry['resource']:
            paper_info.urls.append(paper_entry['resource']['URL'])
        if paper_info.doi:
          paper_info.deduplicate_urls()
          yield (paper_info.doi, paper_info)


def run(argv=None):
  """Main entry point; defines and runs the pipeline."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--crossref_input_pattern',
      dest='crossref_input_pattern',
      default='./raw_data/crossref_metadata/example.json.gz',
      help='Input files pattern for Crossref metadata.',
  )
  parser.add_argument(
      '--openalex_input_pattern',
      dest='openalex_input_pattern',
      default='./raw_data/open_alex/openalex-snapshot/data/works/updated_date=2023-04-07/example.gz',
      help='Input files pattern for OpenAlex metadata.',
  )
  parser.add_argument(
      '--only_output_papers_with_text',
      dest='only_output_papers_with_text',
      default=False,
      action=argparse.BooleanOptionalAction,
      help='Whether only to output papers with abstract or content.',
  )
  parser.add_argument(
      '--output_path',
      dest='output_path',
      default='./processed_data/example_papers/merged_paper_metadata.json',
      help='Output file the results will be written to.',
  )
  known_args, pipeline_args = parser.parse_known_args(argv)

  print('arguments')
  print(known_args)

  # If no pipeline specifications are provided, the DirectRunner
  # (https://beam.apache.org/documentation/runners/direct/) is used by default.
  # However, when processing large amount of data, DirectRunner easily runs out
  # of memory and you should use more powerful backends, e.g., the Spark runner
  # (https://beam.apache.org/documentation/runners/spark/).
  if not pipeline_args:
    print('pipeline args:', pipeline_args)
  pipeline_options = PipelineOptions(pipeline_args)

  def maybe_filter_by_text(input_article_data):
    if (
        known_args.only_output_papers_with_text
        and not input_article_data.abstract
        and not input_article_data.content
    ):
      return False
    else:
      return True

  with beam.Pipeline(options=pipeline_options) as p:
    # Create the output dir if not existent.
    if not os.path.isdir(os.path.dirname(known_args.output_path)):
      os.mkdir(os.path.dirname(known_args.output_path))

    # Read papers from OpenAlex, keyed by their DOIs.
    papers_from_openalex = p | 'CreateEmptyOpenalexPapers' >> beam.Create([])
    if known_args.openalex_input_pattern:
      papers_from_openalex = (
          p
          | 'ReadOpenAlexData'
          >> ReadFromText(known_args.openalex_input_pattern)
          | 'ParseOpenAlexJson' >> beam.ParDo(ParseOpenalexJson())
      )

    # Read papers from Crossref, keyed by their DOIs.
    papers_from_crossref = p | 'CreateEmptyCrossrefPapers' >> beam.Create([])
    if known_args.crossref_input_pattern:
      papers_from_crossref = (
          p
          | 'GetCrossrefFiles' >> MatchFiles(known_args.crossref_input_pattern)
          | 'ParseCrossrefJson' >> beam.ParDo(ParseCrossrefJson())
      )

    # Merge papers from two sources.
    merged_article_data = (
        ({'crossref': papers_from_crossref, 'openalex': papers_from_openalex})
        | 'CoGroupByKey' >> beam.CoGroupByKey()
        | 'MergeArticleData' >> beam.ParDo(MergeArticleData())
    )

    _ = (
        merged_article_data
        | 'MaybeTextFilter' >> beam.Filter(maybe_filter_by_text)
        | 'GetOutput'
        >> beam.Map(lambda article_data: article_data.get_text_output())
        | 'Write' >> WriteToText(known_args.output_path)
    )


if __name__ == '__main__':
  run()
