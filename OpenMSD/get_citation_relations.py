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

r"""Generate citation, co-citation, and bib-coupling pairs."""

import argparse
import json
import os

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions


class GetCitationPairs(beam.DoFn):

  def process(self, data_row):
    if data_row.strip() and 'oci' not in data_row:
      items = data_row.split(',')
      yield (items[1].strip(), items[2].strip())


class GetDois(beam.DoFn):

  def process(self, data_row):
    metadata = json.loads(data_row)
    if 'doi' in metadata:
      yield metadata['doi'].strip()


def select_pairs(citations, dois):
  """Select citation pairs that have both the citing and cited papers DOIs included in the metadata input files."""
  if dois is None:
    return citations

  doi_pairs = dois | 'BuildPairs' >> beam.Map(lambda doi: (doi, doi))

  def get_pair_from_group_results(result):
    for pair in result[1]['citations']:
      yield pair

  pairs_keyed_by_citing_papers = citations | 'KeyByCitingPaper' >> beam.Map(
      lambda pair: (pair[0], pair)
  )

  results_by_citing_papers = (
      {'dois': doi_pairs, 'citations': pairs_keyed_by_citing_papers}
      | 'GroupByCitingPapers' >> beam.CoGroupByKey()
      | 'FilterByCitingPapers'
      >> beam.Filter(
          lambda result: result[1]['dois'] and result[1]['citations']
      )
      | 'GetPair1' >> beam.ParDo(get_pair_from_group_results)
  )

  pairs_keyed_by_cited_papers = (
      results_by_citing_papers
      | 'KeyByCitedPaper' >> beam.Map(lambda pair: (pair[1], pair))
  )

  results_by_cited_papers = (
      {'dois': doi_pairs, 'citations': pairs_keyed_by_cited_papers}
      | 'GroupByCitedPapers' >> beam.CoGroupByKey()
      | 'FilterByCitedPapers'
      >> beam.Filter(
          lambda result: result[1]['dois'] and result[1]['citations']
      )
      | 'GetPair2' >> beam.ParDo(get_pair_from_group_results)
  )

  return results_by_cited_papers


def remove_repeated_pairs(pairs_pcoll, pair_type):
  def remove_duplicate(joined_result):
    candidates = set(joined_result[1])
    for cand in candidates:
      yield (joined_result[0], cand)

  deduplicated_pairs = (
      pairs_pcoll
      | f'RRP-GroupByKey-{pair_type}' >> beam.GroupByKey()
      | f'RRP-RemoveDeplicates-{pair_type}' >> beam.ParDo(remove_duplicate)
  )
  return deduplicated_pairs


def get_cocite_bibcouple_pairs(all_pairs, pair_type, bidir_pairs=False):
  """Generate co-citation and bib-coupling pairs from the citation pairs."""

  def get_all_pairs(joined_result, bidir_pairs):
    for i in range(len(joined_result[1]) - 1):
      for j in range(i + 1, len(joined_result[1])):
        yield (joined_result[1][i], joined_result[1][j])
        if bidir_pairs:
          yield (joined_result[1][i + 1], joined_result[1][i])

  cocite_or_bibcouple_pairs = (
      all_pairs
      | f'{pair_type}-GroupByKey' >> beam.GroupByKey()
      | f'{pair_type}-GetPairs' >> beam.ParDo(get_all_pairs, bidir_pairs)
  )
  deduplicated_pairs = remove_repeated_pairs(
      cocite_or_bibcouple_pairs, pair_type
  )

  return deduplicated_pairs


def run(argv=None):
  """Main entry point; defines and runs the pipeline."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--metadata_input_pattern',
      dest='metadata_input_pattern',
      default='./processed_data/example_papers/*.json*',
      help='Input files pattern for paper metadata.',
  )
  parser.add_argument(
      '--opencitations_input_pattern',
      dest='opencitations_input_pattern',
      default='./raw_data/open_citations/example.csv',
      help='Input files pattern for open citations csv files.',
  )
  parser.add_argument(
      '--want_cocitations',
      dest='want_cocitations',
      default=True,
      action=argparse.BooleanOptionalAction,
      help='Whether to get and output co-citation pairs.',
  )
  parser.add_argument(
      '--want_bibcouples',
      dest='want_bibcouples',
      default=True,
      action=argparse.BooleanOptionalAction,
      help='Whether to get and output bib-coupling pairs.',
  )
  parser.add_argument(
      '--output_dir',
      dest='output_dir',
      default='./processed_data/example_citation_relations/',
      help='Dir for the output files.',
  )
  known_args, pipeline_args = parser.parse_known_args(argv)

  print('arguments:')
  print(known_args)

  # If no pipeline specifications are provided, the DirectRunner
  # (https://beam.apache.org/documentation/runners/direct/) is used by default.
  # However, when processing large amount of data, DirectRunner easily runs out
  # of memory and you should use more powerful backends, e.g., the Spark runner
  # (https://beam.apache.org/documentation/runners/spark/).
  if not pipeline_args:
    print('pipeline args:', pipeline_args)
  pipeline_options = PipelineOptions(pipeline_args)

  with beam.Pipeline(options=pipeline_options) as p:
    # Create the output dir if not existent.
    if not os.path.isdir(os.path.dirname(known_args.output_dir)):
      os.mkdir(known_args.output_dir)

    # Read papers metadata, keyed by their DOIs.
    if known_args.metadata_input_pattern:
      wanted_dois = (
          p
          | 'ReadMetadata' >> ReadFromText(known_args.metadata_input_pattern)
          | 'GetWantedDOIs' >> beam.ParDo(GetDois())
      )
      _ = wanted_dois | 'WriteWantedDois' >> WriteToText(
          os.path.join(known_args.output_dir, 'wanted_dois.txt')
      )
    else:
      wanted_dois = None

    # Read citations from open citations.
    citation_pairs = (
        p
        | 'ReadOpenCitations'
        >> ReadFromText(known_args.opencitations_input_pattern)
        | 'GetCitationPairs' >> beam.ParDo(GetCitationPairs())
    )

    # Keep the pairs that have both papers' metadata available.
    selected_pairs = select_pairs(citation_pairs, wanted_dois)

    pcolls_to_write = {
        'citation': selected_pairs,
    }

    # Get cocitation pairs.
    if known_args.want_cocitations:
      cocite_pairs = get_cocite_bibcouple_pairs(selected_pairs, 'cocite')
      pcolls_to_write['cocitation'] = cocite_pairs

    # Get bib-couple pairs.
    if known_args.want_bibcouples:
      backward_pairs = selected_pairs | 'GetBackwardPairs' >> beam.Map(
          lambda pair: (pair[1], pair[0])
      )
      bibcouple_pairs = get_cocite_bibcouple_pairs(backward_pairs, 'bibcouple')
      pcolls_to_write['bibcouple'] = bibcouple_pairs

    for pair_type in pcolls_to_write:
      _ = (
          pcolls_to_write[pair_type]
          | f'GetOutput-{pair_type}'
          >> beam.Map(lambda pair: '{},{}'.format(pair[0], pair[1]))
          | f'Write-{pair_type}'
          >> WriteToText(
              os.path.join(known_args.output_dir, f'{pair_type}_pairs.csv')
          )
      )


if __name__ == '__main__':
  run()
