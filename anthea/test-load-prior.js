// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * To run this test, visit the URL:
 *     http://.../anthea.html?test=test-load-prior.js
 * and then click around through the rating process.
 */
const raw_prior_results = [
  {
    'prior_result': {
      'errors': [{
        'location': 'source',
        'prefix': 'This is the ',
        'selected': 'first-sentence',
        'type': 'source_error',
        'subtype': '',
        'display': 'Source issue',
        'start': 6,
        'end': 8,
        'severity': 'minor',
        'override_all_errors': false,
        'needs_note': false,
        'metadata': {
          'sentence_index': 0,
          'side': 0,
          'timestamp': 1691699358223,
          'timing': {},
        }
      }],
      'doc': 0,
      'visited': true,
      'timestamp': 1691699363918,
      'feedback': {}
    },
    'prior_rater': 'rater1'
  },
  {
    'prior_result': {
      'errors': [],
      'doc': 0,
      'visited': true,
      'timestamp': 1691699368942,
    },
    'prior_rater': 'rater1'
  },
  {
    'prior_result': {
      'errors': [{
        'location': 'translation',
        'prefix': 'A translation of the ',
        'selected': '3rd sentence',
        'type': 'accuracy',
        'subtype': 'reinterpretation',
        'display': 'Accuracy',
        'start': 8,
        'end': 10,
        'severity': 'major',
        'override_all_errors': false,
        'needs_note': false,
        'metadata': {
          'sentence_index': 0,
          'side': 1,
          'timestamp': 1691699380360,
          'timing': {},
        }
      }],
      'doc': 0,
      'visited': true,
      'timestamp': 1691699382695,
    },
    'prior_rater': 'rater1'
  },
  {
    'prior_result': {
      'errors': [],
      'doc': 0,
      'visited': true,
      'timestamp': 1691699385184,
    },
    'prior_rater': 'rater1'
  },
  {
    'prior_result': {
      'errors': [
        {
          'location': 'translation',
          'prefix': '',
          'selected': 'Translater had',
          'type': 'fluency',
          'subtype': 'inconsistency',
          'display': 'Fluency',
          'start': 0,
          'end': 2,
          'severity': 'minor',
          'override_all_errors': false,
          'needs_note': false,
          'metadata': {
            'sentence_index': 0,
            'side': 1,
            'timestamp': 1691699397911,
            'timing': {},
          }
        },
        {
          'location': 'translation',
          'prefix': 'Translater ',
          'selected': 'had no clue on',
          'type': 'style',
          'subtype': 'awkward',
          'display': 'Style',
          'start': 2,
          'end': 8,
          'severity': 'major',
          'override_all_errors': false,
          'needs_note': false,
          'metadata': {
            'sentence_index': 0,
            'side': 1,
            'timestamp': 1691699407997,
            'timing': {},
          }
        }
      ],
      'doc': 0,
      'visited': true,
      'timestamp': 1691699407997,
      'timing': {},
    },
    'prior_rater': 'rater1'
  },
  {
    'prior_result': {
      'errors': [],
      'doc': 1,
      'visited': false,
      'timestamp': 1691699316043,
      'timing': {},
      'hotw_list': [],
      'feedback': {}
    },
    // prior_rater intentionally left blank
  },
  {
    'prior_result': {
      'errors': [],
      'doc': 1,
      'visited': false,
      'timestamp': 1691699316043,
      'timing': {},
    },
    // prior_rater intentionally left blank
  },
  {
    'prior_result': {
      'errors': [],
      'doc': 2,
      'visited': false,
      'timestamp': 1691699316043,
      'timing': {},
      'hotw_list': [],
      'feedback': {}
    },
    'prior_rater': 'rater2'
  },
  {
    'prior_result': {
      'errors': [],
      'doc': 2,
      'visited': false,
      'timestamp': 1691699316043,
      'timing': {},
    },
    'prior_rater': 'rater2'
  }
];

const prior_results = raw_prior_results.map(e => JSON.stringify(e));

const testProjectTSVData = `
This is the first​-​sentence​. 	This is its translation​.	doc-42	system-GL	${prior_results[0]}
This is the second sentence. It includes this long string that tests text-wrapping: http://01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789​.	This is the translation (​of the second sentence​)​.	doc-42	system-GL	${prior_results[1]}
The third sentence​. 	A translation of the 3rd sentence​. 	doc-42	system-GL	${prior_results[2]}
# A sentence beginning with # 4​.	Translated sentence 4​.	doc-42	system-GL	${prior_results[3]}

A second paragraph​. This is a long sentence with meaninglessness embedded as an essential artifact that requires the reader to comtemplate their exact place in the vast expanse of existence​.	Translater had no clue on this one​.	doc-42	system-GL	${prior_results[4]}
The first sentence in the second document​.	The translation of the first sentence in the second document​.	doc-99	system-DL	${prior_results[5]}
The 2nd sentence in the second document​.	The translation of the doosra sentence in the second document​.	doc-99	system-DL	${prior_results[6]}
The third and final document​.	The translation​, of the opening sentence in the third document​.	doc-1531	system-DL	${prior_results[7]}
The last line​. The last word​. Waiting for whom​?	Given the existence as uttered forth in the public works of Puncher and Wattmann of a personal God quaquaquaqua with white beard quaquaquaqua outside time without extension who from the heights of divine apathia divine athambia divine aphasia loves us dearly with some exceptions for reasons unknown but time will tell and suffers like the divine Miranda with those who for reasons unknown .​.​.	doc-1531	system-DL	${prior_results[8]}
      `;
const testProjectName = 'Google-MQM-Test-Load-Prior-42';
const testTemplateName = 'MQM';
const activeName = antheaManager.activeName(testProjectName, testTemplateName);
try {
  const activeDataKey = antheaManager.ACTIVE_KEY_PREFIX_ + activeName;
  window.localStorage.removeItem(activeDataKey);
  const activeResultsKey =
      antheaManager.ACTIVE_RESULTS_KEY_PREFIX_ + activeName;
  window.localStorage.removeItem(activeResultsKey);
} catch (err) {
  console.log('Caught exception (harmless if for "no such item": ' + err);
}

// Instead of providing prior results through annotations (the 5th column in
// testProjectTSVData), they could be provided via the "prior_results" field in
// the parameters below. Note that this prevents specifying the prior rater on a
// per-segment basis.
const parameters = {
  "source_language": "en",
  "target_language": "en",
  "prior_rater": "default_rater_from_json_parameters",
  // "prior_results": raw_prior_results.map(e => e.prior_result),
};
antheaManager.createActive(
    testTemplateName, testProjectName,
    JSON.stringify(parameters) + '\n' + testProjectTSVData,
    100  /** test that segments with prior annos do not get any hotw */);
