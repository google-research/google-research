# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Licensed under the Apache License, Version 2.0."""

import collections
import gzip
import io
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import six

FOUNDATION_SUBSTR = "Fd"
END_SUBSEQ = "ATCTCGTATGCCGTCTTCTGCTTG"  # Subseq to truncate DNA read on.
CYCLE_SEP = "-"  # Delimiter to identify cycle number (end of partner string).
CYCLE_NUM_INDEX = -1  # Default = -1, the last item in the CYCLE_SEP split str.
EMPTY_FOUNDATION = "Empty_Foundation"
BCS_SEP = ";"


class Read(
    collections.namedtuple("Read",
                           ["title", "title_aux", "sequence", "quality"])):
  """NamedTuple to hold the information in one FASTQ sequence record.

  A Read is one record from a FASTQ file and represents one read off a
  sequencer. All the values are strings. The record contains:

  (1) the title of the read (unique identifier),
  (2) any auxiliary information from the title line (e.g., barcode info)
  (3) the DNA sequence as recorded by the sequencer, and
  (4) the per-base quality of the sequencing.

  The per-base quality is a string of the same length as the DNA sequence,
  which can be converted to an integer based on the ascii + an offset.
  The numbers are generally between 0 and 40 and the exact encoding and
  offset depends on the version of the sequencer.
  See https://en.wikipedia.org/wiki/FASTQ_format
  """
  pass


def handle_title_string(title_line):
  """Find the title and (if it exists) the auxiliary title information.

  Args:
    title_line: line of FASTQ file containing the title.  Starts with '@'.

  Returns:
    title (string).
    title_aux (string, which is "" if there is no auxiliary information).
  """
  # Strip off leading "@"; only go to first space character.
  #
  # Although the standard specifies that there should be no spaces, in practice
  # the title string looks like:
  #
  # <true sequence record title>[space character]<extra info>
  #
  # where "extra info" is things like barcode index read.
  title_pieces = title_line[1:].split()
  title = title_pieces[0]
  title_aux = ""
  if len(title_pieces) > 1:
    title_aux = " ".join(title_pieces[1:])
  return title, title_aux


def fastq_iterator(file_handle):
  """Reads a FASTQ file and yields Read named tuples.

  Conforms to the FASTQ spec (see https://en.wikipedia.org/wiki/FASTQ_format),
  specifically we expect that the whole sequence record is on a single line,
  even if the sequence is very long. (This is generally true for FASTQ, which
  are short reads that come off the sequencer but generally not true for
  FASTA records where one sequence record can be a whole chromosome.)

  Args:
    file_handle: open file handle to a fastq file

  Raises:
    Error: If the file is malformated.
  Yields:
    Read named tuple for each sequence record in the file.
  """
  while True:
    next_title = six.ensure_str(file_handle.readline()).strip()
    # between records (or at beginning) there can be comment lines
    if next_title.startswith("#"):
      continue
    if not next_title:
      break
    if not next_title.startswith("@"):
      raise ValueError("Expected the next sequence record to start with @. "
                       "Unexpected title line '%s'" % (next_title))

    title, title_aux = handle_title_string(next_title)

    # Grab sequence and quality info.
    sequence = six.ensure_str(file_handle.readline()).strip()
    quality_title = six.ensure_str(file_handle.readline()).strip()
    quality_str = six.ensure_str(file_handle.readline()).strip()

    # checks
    if not quality_title.startswith("+"):
      raise ValueError("Expected the quality title to start with + for record "
                       "'%s'" % title)
    if len(quality_title) > 1 and quality_title[1:] != next_title[1:]:
      raise ValueError("If the quality title exists, it should be the same as "
                       "the title. Quality title: '%s', Title: '%s'" %
                       (quality_title[1:], next_title[1:]))
    if len(sequence) != len(quality_str):
      raise ValueError(
          "Expected the sequence length to be the same as the quality "
          "length for record '%s'. Sequence is %d long but quality "
          "is %d long." % (title, len(sequence), len(quality_str)))

    yield Read(title, title_aux, sequence, quality_str)


def get_unique_seqs_r1(file_handle, num_reads):
  """Counts occurrences of unique strings.

  Args:
    file_handle: File handle.
    num_reads: (int) Number of reads to sample from file.

  Returns:
    (Counter) Where the key is a sequence and the value is the number of
    occurrences.
  """
  r1_handle = io.TextIOWrapper(gzip.GzipFile(fileobj=file_handle))
  r1_reader = fastq_iterator(r1_handle)
  reads = collections.Counter()

  # Sometimes files are empty, just return empty Counter.
  try:
    r1 = next(r1_reader)
  except StopIteration:
    return reads

  counter = 0
  while r1 and counter < num_reads:
    seq = r1.sequence
    if END_SUBSEQ:
      index = seq.find(END_SUBSEQ)
      if index >= 0:
        seq = seq[:(index + len(END_SUBSEQ))]
    reads[seq] += 1
    r1 = next(r1_reader, False)
    counter += 1
  return reads


def update_dataframe_with_tags_r1(df, mers):
  """Creates a dataframe with locations of mers within DNA sequences.

  Args:
    df: (pd.DataFrame) Dataframe with sequences and counts.
    mers: (list) Motifs to search for in sequences.  Returns; (pd.DataFrame)
      where original dataframe has new columns added for each mer and its
      reverse complement.  The value at each row corresponds to the location of
      that mer in the DNA sequence.
  Returns:
    The updated dataframe.
  """
  df["r1rc"] = list(map(lambda x: reverse_complement(x), df["r1"]))
  num_initial_columns = len(df.columns)

  for mer in mers:
    df[mer] = list(map(lambda x: x.find(mer), df["r1"]))
    df["rc_" + mer] = list(map(lambda x: x.find(mer), df["r1rc"]))
  cols = df.columns

  # Resort the columns so that the sequences are easier to look at visually.
  new_cols = list(cols[num_initial_columns:])
  new_cols.sort()
  reorder_cols = list(cols[:num_initial_columns]) + new_cols
  df = df[reorder_cols]

  df = df.sort_values(by="cluster_count", ascending=False)
  return df


def get_consensus_sequence(reads):
  """Gets simple consensus by taking the max character at each position.

  Args:
    reads: (list) DNA sequences.

  Returns:
    (str) A consensus DNA sequence.
  """
  n = len(reads[0])
  consensus = []
  for i in range(n):
    nuc_count = collections.Counter()
    for j in range(len(reads)):
      nuc_count[reads[j][i]] += 1
    consensus.append(nuc_count.most_common(1)[0][0])
  return "".join(consensus)


def simple_cluster_R1(reads, n_most_common=100, allowed_dist=1):
  """Clusters DNA sequences based on hamming distance.

  Args:
    reads: (Counter) DNA sequences (key) with their counts (value).
    n_most_common: (int) Number of sequences from counter for clustering.
    allowed_dist: (int) Sequences less than or equal to this hamming distance
      will be clustered.

  Returns:
    (pd.Dataframe) A pandas dataframe of the clustered sequences with a single
    representative for the cluster and counts being the sum of values in the
    cluster.
  """
  top_seqs = reads.most_common(n=n_most_common)
  seqs1 = [seq for seq, seq_count in top_seqs]
  if allowed_dist > 0:
    dists1 = hamming_dist_matrix(seqs1)

  # Build graph with edges between sequences within allowed hamming distance.
  g = nx.Graph()
  n = len(seqs1)
  current_graph_seqs = set()
  for i in range(n):
    if seqs1[i] in current_graph_seqs:
      continue
    cluster_seqs = set()
    g.add_node(seqs1[i])
    current_graph_seqs.add(seqs1[i])
    if allowed_dist > 0:  # Give number of exact sequence matches from counter.
      for j in range(i + 1, n):
        if seqs1[j] in current_graph_seqs:
          continue
        if dists1[i, j] <= allowed_dist:
          g.add_edge(seqs1[i], seqs1[j])
          current_graph_seqs.add(seqs1[j])
          cluster_seqs.add(seqs1[j])

  df_input = []
  for c in nx.connected_components(g):
    c_seqs = list(c)
    # Build tuples of sequence and their counts.
    c_seqs_count = [(r, reads[r]) for r in c_seqs]
    c_seqs_count.sort(key=lambda x: x[1])
    c_sum = sum([x[1] for x in c_seqs_count])
    top_count = c_seqs_count[0]
    consensus_seq = get_consensus_sequence(c_seqs)
    # Add a tuple of:
    #  - size of connected component
    #  - most frequent seq,
    #  - count of most frequent seq
    #  - sum of counts for cluster
    #  - consensus sequence (may not actually ever occur)
    df_input.append((len(c), top_count[0], top_count[1], c_sum, consensus_seq))

  if len(df_input) == 0:
    return None

  df = pd.DataFrame(df_input)
  df.columns = ["cluster_size", "r1", "count", "cluster_count", "consensus_seq"]
  df["freq"] = df["cluster_count"] / np.sum(df["cluster_count"])
  return df


_RCMAP = {
    "a": "t",
    "c": "g",
    "g": "c",
    "t": "a",
    "n": "n",
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "N": "N",
    "-": "-"
}


def complement(sequence, output_type="str"):
  """Gets the complement of the input DNA sequence.

  Args:
    sequence: (str) DNA sequence.
    output_type: (str) output data type, should be either 'str' or 'array'

  Returns:
    A str/array of str representing the reverse complement of seq.

  Raises:
    ValueError: if output_type not in ['str', 'array']
  """
  nuc_list = [_RCMAP[nuc] for nuc in sequence]
  if output_type == "str":
    return "".join(nuc_list)
  elif output_type == "array":
    return np.array(nuc_list)
  else:
    raise ValueError('output_type must be one of ["array", "str"]')


def reverse_complement(sequence, output_type="str"):
  """Gets the reverse complement of the input DNA sequence.

  Args:
    sequence: (str) DNA sequence.
    output_type: (str) output data type, should be either 'str' or 'array'

  Returns:
    A str/array of str representing the reverse complement of seq.

  Raises:
    ValueError: if output_type not in ['str', 'array']
  """
  return complement(sequence, output_type)[::-1]


def read_fastqs_for_experiment(expt_dict, num_reads, subseqs, n_common,
                               dist_for_clustering):
  row = 0
  expts = []
  freqs = []
  merged_dfs = []
  for expt, expt_fastq_seqs in expt_dict.items():
    # Much of the work here is for handling related experiments.
    # This may not be necessary as we are only looking at one lane for now.
    # E.g.: Maybe eliminate these two variables.
    sub_plot_index = 0
    expt_freq = 0

    for fastq_seq in expt_fastq_seqs:
      with open(fastq_seq, "rb") as f:
        r1_counter = get_unique_seqs_r1(f, num_reads)
      # Build df summarizing clusters from the most common sequences.
      df = simple_cluster_R1(
          r1_counter, n_most_common=n_common, allowed_dist=dist_for_clustering)
      if df is None:  # Ignore if empty.
        continue
      df = update_dataframe_with_tags_r1(df, subseqs)
      df["lane"] = "L001"  # Hard code lane as it is not in fastq filename.
      df["expt"] = expt
      df["sub_plot_index"] = sub_plot_index
      top_freq = list(df["freq"])[0]
      expt_freq += top_freq
      sub_plot_index += 1

    row += 1
    if sub_plot_index == 0:
      expt_freq = 0
    else:
      expt_freq /= sub_plot_index  # Average over all related fastqs (lanes).
      freqs.append(expt_freq)
      expts.append(expt)
      merged_dfs.append(df)
  return pd.concat(merged_dfs)


def get_subseqs_from_component_names(component_names_str):
  component_names = component_names_str.split("\n")
  component_dict = {}
  rc_dict = {}
  for component_name in component_names:
    if not component_name.strip():
      # Ignore empty lines.
      continue
    component_elements = component_name.split()
    if len(component_elements) < 2 or len(component_elements) > 3:
      print("Warning: Number of fields not 2 or 3 in line: %s" % component_name)
    elif len(component_elements) == 3:
      component_id, component_mer, component_rcmer = component_elements
      if component_mer in component_dict or component_rcmer in component_dict:
        print("Warning: Ignoring repeated occurrences of sequence in: %s" %
              component_name)
      elif component_id in component_dict.values():
        print("Warning: Ignoring repeated occurrences of ID in: %s" %
              component_name)
      else:
        component_mer = component_mer.upper()
        component_rcmer = component_rcmer.upper()
        component_dict[component_mer] = component_id
        rc_dict[component_mer] = component_rcmer
    else:
      component_id, component_mer = component_elements
      if component_mer in component_dict:
        print("Warning: Ignoring repeated occurrences of sequence in: %s" %
              component_name)
      elif component_id in component_dict.values():
        print("Warning: Ignoring repeated occurrences of ID in: %s" %
              component_name)
      else:
        component_mer = component_mer.upper()
        component_dict[component_mer] = component_id

  subseqs = component_dict.keys()
  subseqs = list(map(lambda x: x.upper(), subseqs))
  seq_rev_comp_dict = rc_dict
  return subseqs, component_dict, seq_rev_comp_dict


def get_foundation_dict(component_dict):
  # Auto-generation of foundation_dict may drop round-encoding of binders.
  # We will likely need to enforce strict naming and extract this from names.
  foundation_dict = {}
  non_fd_components = [
      component for component in component_dict.values()
      if FOUNDATION_SUBSTR not in component
  ]

  for component in component_dict.values():
    if FOUNDATION_SUBSTR in component:
      foundation_dict[component] = non_fd_components

  return foundation_dict


def generate_ordered_match_str_from_subseqs(r1,
                                            subseqs_to_track,
                                            rc_component_dict,
                                            allow_overlaps=False):
  """Generates an ordered subsequences match string for the input sequence.

  Args:
    r1: (str) R1 sequence to scan for subsequence matches.
    subseqs_to_track: (list) Subsequences to look for in R1.
    rc_component_dict: (dict) Dict mapping DNA sequence to label.
    allow_overlaps: (boolean) Whether to allow matches that overlap on R1.  If
      False, then it will identify a maximal non-overlapping set of matches.

  Returns:
    (str) labeled components for r1 in the form: 'label_1;label_2;...;label_n'
  """

  # Generate ordered set of subseq matches to r1 sequence.
  match_tups = []
  for mer_label in subseqs_to_track:
    mer = rc_component_dict[mer_label]
    for match in re.finditer(mer, r1):
      xstart = match.start()
      xend = xstart + len(mer)
      match_tups.append((xstart, xend, mer_label))
  match_tups.sort(reverse=True)

  # Create a maximal independent set that does not allow overlapping subseqs.
  if not allow_overlaps and len(match_tups) > 0:
    mer_graph = nx.Graph()
    mer_graph.add_nodes_from(match_tups)
    for i in range(len(match_tups)):
      for j in range(i + 1, len(match_tups)):
        # Check if the end of match_tups[j] overlaps the start of match_tups[i].
        if match_tups[i][0] < match_tups[j][1]:
          mer_graph.add_edge(match_tups[i], match_tups[j])
    # Generate a non-overlapping list of subseqs.
    match_tups = nx.maximal_independent_set(mer_graph)
    match_tups.sort(reverse=True)

  match_str = BCS_SEP.join([match_tup[-1] for match_tup in match_tups])
  return match_str


def foundation_label_locations(df,
                               foundation_to_targets,
                               component_dict,
                               allow_overlaps=False):
  """Generates a string of ordered foundations and foundation partners.

  Foundations and foundation partners are ordered in reverse order of occurrence
  of the sequenced DNA string.  The diagram below indicates a possible ordering
  of partners and foundation sequence.

  DNA Sequence         [------------------------------------------------->
                          {   } . . .       {   }      {   }      {----}
  expected structure:    (partnerm)       (partner2) (partner1) (foundation)
  returned labels:       (labeln)         (label3)   (label2)   (label1)


  For each DNA sequence we return a new string consisting of the labels from the
  foundation(s) and/or partner(s), joined by BCS_SEP. While these could be
  maintained as an array, we use strings to facilitate manual ordering
  exaimination by experimentalists.

  Args:
    df: (pd.DataFrame) Summary clustered sequence dataframe from pipeline.
    foundation_to_targets: (dict) Dict mapping BCS foundation to aptamer
      targets.
    component_dict: (dict) Dict mapping the labels to the DNA sequence.
    allow_overlaps: (boolean) Whether to allow matches that overlap on R1.  If
      False, then it will identify a maximal non-overlapping set of matches.

  Returns:
    (list) Ordered list of labeled components for each clustered sequence in df.
    Of the form: 'label_1;label_2;...;label_n'.

  """
  # Create set of all foundations and partner labels.
  subseqs_to_track = set(foundation_to_targets.keys())
  for values in foundation_to_targets.values():
    for val in values:
      subseqs_to_track.add(val)

  # Create dict of label -> sequence.
  rc_component_dict = {}
  for key, value in component_dict.items():
    rc_component_dict[value] = key

  # Generate ordered list of labels based on reverse location in cluster r1 seq.
  match_strs = []
  for _, row in df.iterrows():
    r1 = row["r1"]
    match_str = generate_ordered_match_str_from_subseqs(r1, subseqs_to_track,
                                                        rc_component_dict,
                                                        allow_overlaps)
    match_strs.append(match_str)
  return match_strs


def foundation_partner_cycle_table(df,
                                   foundation_to_targets,
                                   component_dict,
                                   cycles_to_check,
                                   col_to_sum="count",
                                   heatmap=False,
                                   allow_overlaps=False):
  """Constructs tables with the number of foundation partners at each cycle.

  The resulting table has foundation labels as columns and partner labels as
  rows with the count corresponding to the occurrence of the given pair at a
  a given cycle, where a cycle here is simplified to mean the relative ordering
  of the foundation (partners most adjacent to the foundating are considered
  cycle 1, partners that have 1 intervening partner between the foundation are
  considered cycle 2, and so on).  Note, that the distance (in base-pairs) does
  not yet impact cycle number.

  Args:
    df: (pd.DataFrame) Summary clustered sequence dataframe from pipeline.
    foundation_to_targets: (dict) BCS foundation to aptamer targets.
    component_dict: (dict) Dict mapping the labels to the DNA sequence.
    cycles_to_check: (int) Number of cycles to check.
    col_to_sum: (str) Column from the dataframe to produce heatmap on.  Options
      are: 'count', 'cluster_count', 'freq'.
    heatmap: (boolean) Whether to render a heatmap of the dataframe for viz.
    allow_overlaps: (boolean) Whether to allow matches that overlap on R1.  If
      False, then it will identify a maximal non-overlapping set of matches.

  Returns:
    (list of pd.DataFrames)  A list of tables (ordered by cycle number) in which
    each index is the aptamer label and columns are the BCS foundation labels.

  """
  df["ordered_foundation_labels"] = foundation_label_locations(
      df, foundation_to_targets, component_dict, allow_overlaps=allow_overlaps)
  foundation_pair_df = df.groupby(["ordered_foundation_labels"]).sum()

  # Make a list of the aptamer partners to all foundations.
  foundation_partners = set()
  for values in foundation_to_targets.values():
    for val in values:
      foundation_partners.add(val)
  foundation_partners = list(foundation_partners)

  # Initialize cycle -> foundation -> partner dict.
  foundations = list(foundation_to_targets.keys())
  foundations.append(EMPTY_FOUNDATION)
  foundation_partner_pair_counts = {}
  for cycle in range(cycles_to_check + 1):
    foundation_partner_pair_counts[cycle] = {}
    for foundation in foundations:
      foundation_partner_pair_counts[cycle][foundation] = collections.Counter()

  # Add counts to cycle -> foundation -> partner dict.
  foundations_at_cycle_0 = 0
  for foundation_str, row in foundation_pair_df.iterrows():
    count = row[col_to_sum]
    foundation_and_partners = foundation_str.split(BCS_SEP)
    foundation = foundation_and_partners[0]
    if foundation in foundation_to_targets:
      foundations_at_cycle_0 += count
      partner_start_index = 1
    else:
      foundation = EMPTY_FOUNDATION
      partner_start_index = 0
    # Only include partners up to the CYCLES_TO_CHECK.
    for cycle in range(partner_start_index,
                       min(len(foundation_and_partners), cycles_to_check + 1)):
      foundation_partner = foundation_and_partners[cycle]
      foundation_partner_pair_counts[cycle][foundation][
          foundation_partner] += count

  # sort the foundations and partners so that heatmaps will be easier to read.
  foundations.sort()
  foundation_partners.sort()

  # Convert dict of Counters to list of dataframes.
  cycle_dfs = []
  cycle_correspondence = np.zeros((cycles_to_check + 1, cycles_to_check + 1))
  cycle_correspondence[0, 0] = foundations_at_cycle_0
  for cycle in range(cycles_to_check + 1):
    current_cycle_dict = {}
    for foundation in foundations:
      counts = []
      for partner in foundation_partners:
        count = foundation_partner_pair_counts[cycle][foundation][partner]
        counts.append(count)
        # NOTE: Until we standardize this protocol a bit more we use a hack to
        # extract the expected cycle number from label string.
        expected_cycle = partner.split(CYCLE_SEP)[CYCLE_NUM_INDEX]
        if expected_cycle.isdigit():
          expected_cycle = int(expected_cycle)
          cycle_correspondence[cycle][expected_cycle] += count
      current_cycle_dict[foundation] = counts
    cycle_df = pd.DataFrame.from_dict(current_cycle_dict)
    cycle_df.index = foundation_partners
    cycle_dfs.append(cycle_df)

  # Render heatmap.
  if heatmap:
    plt.figure(figsize=(15, 5))
    sns.heatmap(cycle_correspondence, annot=True, fmt=".0f")
    plt.title("Correspodence Between Observed And Expected Cycle Number")
    plt.xlabel("Expected Cycle")
    plt.ylabel("Observed Cycle")
    plt.show()

    for cycle, cycle_df in enumerate(cycle_dfs):
      plt.figure(figsize=(15, 5))
      sns.heatmap(cycle_df, annot=True, fmt=".0f")
      plt.title("Cycle %i" % (cycle))
      plt.xlabel("Foundation")
      plt.ylabel("Foundation Partner")
      plt.show()

  return cycle_dfs, cycle_correspondence


def make_plot_df(cycle_df):
  targets = []
  base_targets = []
  partners = []
  fracs = []
  counts = []
  cols = list(cycle_df.columns)[:-1]
  for _, row in cycle_df.iterrows():
    row_sum = row[cols].sum()
    for col in cols:
      partners.append(row["binder"])
      targets.append(col)
      base_targets.append(col.split(".")[0].split("_")[0])
      counts.append(row[col])
      fracs.append(row[col] / row_sum)
  plot_df = pd.DataFrame.from_dict({
      "target": targets,
      "Binder for": partners,
      "Binder Target": base_targets,
      "Count": counts,
      "Count Fraction": fracs
  })
  return plot_df
