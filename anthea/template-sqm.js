// Copyright 2022 The Google Research Authors.
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
 * This template is an illustrative example of a "scalar quality metric"
 * template.
 */
antheaTemplates['SQM'] = {
  sqm: [
    {
      value: 0,
      shortcut: '0',
      color: 'orange',
      display: 'Nonsense/No meaning preserved',
      description: 'Nearly all information is lost between the translation and source. Grammar is irrelevant.',
    },
    {
      value: 1,
      shortcut: '1',
      color: '#ffd000',
      display: 'Better than nonsense, but close to no meaning preserved',
      description: 'The translation is not completely nonsense. Grammar is poor.',
    },
    {
      value: 2,
      shortcut: '2',
      color: 'yellow',
      display: 'Some Meaning Preserved',
      description: 'The translation preserves some of the meaning of the source but misses significant parts. The narrative is hard to follow due to fundamental errors. Grammar may be poor.',
    },
    {
      value: 3,
      shortcut: '3',
      color: '#c8f648',
      display: 'A lot of meaning preserved but not most, perhaps with grammatical errors',
      description: 'The translation retains a lot but not most of the meaning of the source. It may have some grammar mistakes or minor contextual inconsistencies.',
    },
    {
      value: 4,
      shortcut: '4',
      color: 'lightgreen',
      display: 'Most Meaning Preserved and Few Grammar Mistakes',
      description: 'The translation retains most of the meaning of the source. It may have some grammar mistakes or minor contextual inconsistencies.',
    },
    {
      value: 5,
      shortcut: '5',
      color: '#48b748',
      display: 'Close to perfect in preserving meaning, and with no grammatical errors',
      description: 'The meaning of the translation is almost completely consistent with the source and the surrounding context (if applicable). The grammar is also correct.',
    },
    {
      value: 6,
      shortcut: '6',
      color: 'green',
      display: 'Perfect Meaning and Grammar',
      description: 'The meaning of the translation is completely consistent with the source and the surrounding context (if applicable). The grammar is also correct.',
    },
  ],
  /**
   * Instructions to show at the top, in HTML format.
   */
  instructions: `
    <h3>Instructions Overview</h3>
    <p>
      In this task, you will be presented with several sentences from a single
      document. Each sentence from the document is shown along with its
      translation.
    </p>
    <p>
      Your job is to evaluate the quality of the translation, while taking into
      account the entire document for context. Please use this
      surrounding context to resolve any ambiguity in the source texts, and pay
      close attention to any word choice inconsistencies across sentences within
      a source text.
    </p>

    <h3>General Guidelines</h3>
      Please start by reading all of the source texts from top to bottom in
      order to understand the full context. Then, please provide a rating
      between 0 (worst) and 6 (best) for each translation. The full scale is
      summarized in <b>Table of ratings</b>.
    </p>
  `,
};
