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

"""Annotation tool for a single text example."""

import sys

import bs4
import pandas as pd
import requests
import spacy

import tide_nlp as tide_nlp
from tide_nlp import identity_annotator as ia
from tide_nlp import tidal_util
from tide_nlp.entity_annotator import non_ptc_annotator as non_ptc_a
from tide_nlp.entity_annotator import ptc_annotator as ptc_a
from tide_nlp.entity_annotator import ptc_helper as ptc
from tide_nlp.entity_annotator import spacy_annotator as spacy_a
from tide_nlp.lexicon import tidal_lexicon as lex
from tide_nlp.tokenizer import spacy_tokenizer as tok

PERSON_NOUN_LEXICON_URLS = ['https://en.wiktionary.org/w/index.php?title=Category:English_terms_of_address',
                            'https://en.wiktionary.org/w/index.php?title=Category:English_terms_of_address&pagefrom=SNOOKUMS%0Asnookums#mw-pages']


def main(argv):
  if len(argv) != 1 or not isinstance(argv[0], str):
    raise AssertionError('expecting a single string argument for annotation.')

  text_ = argv[0]

  # Initialize TIDAL lexicon
  tidal_lexicon_df = tidal_util.read_tidal()

  # Download person noun lexicon
  person_noun_terms = []
  for url in PERSON_NOUN_LEXICON_URLS:
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content, 'html.parser')
    mw_category_divs = soup.find_all('div', {'class': 'mw-category-group'})

    for div in mw_category_divs:
      for a in div.find_all('a'):
        noun = a.text.lower()

        # Remove terms that are less than 3 characters (eg Mt)
        if len(noun) < 3:
          continue
        # Remove terms that have a period (eg Mr. President)
        if '.' in noun:
          continue
        person_noun_terms.append(noun)

  person_lexicon_df = pd.DataFrame(person_noun_terms, columns=['noun'])

  # Configure annotation options
  model_path = 'en_core_web_sm'
  nlp = spacy.load(model_path)

  lexicon = lex.TidalLexicon(tidal_lexicon_df)
  tokenizer = tok.SpacyTokenizer(nlp)

  person_helper_lexicon = ptc.PersonMentionHelper(nlp, person_lexicon_df)
  ptc_lexicon_annotator = ptc_a.PtcAnnotator(person_helper_lexicon)
  non_ptc_lexicon_annotator = non_ptc_a.NonPtcAnnotator(person_helper_lexicon)

  spacy_annotator = spacy_a.SpacyAnnotator(nlp)

  # This uses a simple token-based annotation logic to determine whether an
  # identity term is modifying a known person noun based on the lexicon.
  entity_annotators = [ptc_lexicon_annotator, spacy_annotator]

  non_entity_annotators = [non_ptc_lexicon_annotator]

  annotator = ia.IdentityAnnotator(lexicon=lexicon,
                                   tokenizer=tokenizer,
                                   entity_annotators=entity_annotators,
                                   non_entity_annotators=non_entity_annotators)

  groups, terms, group_term_dict, df = annotator.annotate(text_.lower())

  print('identity groups: ', groups)
  print('identity terms: ', terms)
  print('identity group-term dictionary:\n', group_term_dict)
  print('annotation candidates:\n', df.to_csv())

if __name__ == '__main__':
  main(sys.argv[1:])
