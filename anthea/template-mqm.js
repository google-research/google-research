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
 * This is the main MQM template. Other MQM template variants are derived
 * from this by overriding severitities/errors and changing configuration
 * constants.
 */
antheaTemplates['MQM'] = {
  severities: {
    major: {
      display: 'Major severity',
      shortcut: 'M',
      color: 'pink',
      description: 'Major severity errors significantly alter the meaning ' +
                   'of the source text, or significantly degrade the quality ' +
                   'of the text.',
    },
    minor: {
      display: 'Minor severity',
      shortcut: 'm',
      color: '#fbec5d',
      description: 'Minor severity errors are noticeable but minor flaws in the ' +
                   'translated text. They do not significantly alter the ' +
                   'meaning of the source text, and they do not ' +
                   'significantly degrade the quality of the text.',
    },
  },

  /**
   * @const {string} Template version identifier.
   */
  VERSION: 'v1.00-Mar-24-2025',

  /**
   * @const {boolean} Show two translations when set to true.
   */
  SIDE_BY_SIDE: false,

  /**
   * @const {boolean} Collect per-segment quality scores when set to true. Also
   *    disables splitting segments into sub-paragraphs.
   */
  COLLECT_QUALITY_SCORE: false,

  /**
   * @const {boolean} Only rate the target side, i.e., the translated text.
   */
  TARGET_SIDE_ONLY: false,

  /**
   * @const {number} Allow marking at most these many errors per sentence. If
   *     set to 0, then no limit is imposed.
   */
  MAX_ERRORS: 0,

  /**
   * @const {boolean} Set this to true if the template instructions already
   *     include listings of errors and severities, and you do not want to
   *     auto-append the lists of errors/severities to the instructions.
   */
  SKIP_RATINGS_TABLES: true,

  /**
   * @const {boolean} Set this to true if you want to allow error spans to
   *     start on whitespace.
   */
  ALLOW_SPANS_STARTING_ON_SPACE: true,

  /**
   * @const {boolean} Set this to true if you want to present the error
   *    types/subtypes in a short, flat list (avoid using this if you have
   *    more than 10 error type/subtype combinatons).
   */
  FLATTEN_SUBTYPES: false,

  /**
   * @const {boolean} Set this to true if your TSV data for each docsys
   *     includes a web page rendering and bounding box provided as an
   *     annotation on each first segment. The annotation should be a JSON
   *     string that looks like this (exccept that it should not have
   *     newlines):
   * {
   *   "source": {
   *     "url": "http://screenshot-image-url-for-source",
   *     "w": 100, "h": 120, "box": {"x": 5, "y": 7, "w": 90, "h": 30}
   *   },"
   *   "target": {
   *     "url": "http://screenshot-image-url-for-target",
   *     "w": 100, "h": 120, " "box": {"x": 5, "y": 7, "w": 90, "h": 30}
   *   },
   * }
   */
  USE_PAGE_CONTEXT: false,

  errors: {
    accuracy: {
      display: 'Accuracy',
      description: 'The target text does not accurately reflect the source text.',
      subtypes: {
        reinterpretation: {
          display: 'Creative Reinterpretation',
          description: 'The target text reinterprets the source, but preserves its intent within its broader context (the document and its purpose, and the target locale). This can be because the translation adds, removes, or modifies text in accordance with the target locale, or simply makes creative changes that fully preserve the intent of the source text.',
        },
        mistranslation: {
          display: 'Mistranslation',
          description: 'The target text does not accurately represent the source text.',
        },
        gender_mismatch: {
          display: 'Gender Mismatch',
          description: 'The gender is incorrect (incorrect pronouns, noun/adjective endings, etc).',
        },
        untranslated: {
          display: 'Source language fragment',
          description: 'Content that should have been translated has been left untranslated.',
        },
        addition: {
          display: 'Addition',
          description: 'The target text includes information not present in the source.',
        },
        omission: {
          display: 'Omission',
          description: 'Content is missing from the translation that is present in the source.',
          source_side_only: true,
        },
      },
    },
    fluency: {
      display: 'Fluency',
      description: 'Issues related to the form or content of translated text, independent of its relation to the source text; errors in the translated text that prevent it from being understood clearly.',
      subtypes: {
        inconsistency: {
          display: 'Inconsistency',
          description: 'The text shows internal inconsistency (not related to terminology).',
        },
        grammar: {
          display: 'Grammar',
          description: 'Issues related to the grammar or syntax of the text, other than spelling and orthography.',
        },
        register: {
          display: 'Register',
          description: 'The content uses the wrong grammatical register, such as using informal pronouns or verb forms when their formal counterparts are required.',
        },
        spelling: {
          display: 'Spelling',
          description: 'Issues related to spelling or capitalization of words, and incorrect omission/addition of whitespace.',
        },
        breaking: {
          display: 'Text-Breaking',
          description: 'Issues related to missing or unwarranted paragraph breaks or line breaks.',
        },
        punctuation: {
          display: 'Punctuation',
          description: 'Punctuation is used incorrectly (for the locale or style).',
        },
        display: {
          display: 'Character encoding',
          description: 'Characters are garbled due to incorrect application of an encoding.',
        },
      },
    },
    style: {
      display: 'Style',
      description: 'The text has stylistic problems.',
      subtypes: {
        awkward: {
          display: 'Unnatural or awkward',
          description: 'The text is literal, written in an awkward style, unidiomatic or inappropriate in the context.',
        },
        sentence_structure: {
          display: 'Bad sentence structure',
          description: 'The marked span of text is an unnecessary repetition, or makes the sentence unnecessarily long, or would have been better as a clause in the previous sentence.'
        },
        archaic_word: {
          display: 'Archaic or obscure word choice',
          description: 'An archaic or lesser-known word is used where a more colloquial term would be a better fit.',
        },
      },
    },
    terminology: {
      display: 'Terminology',
      description: 'A term (domain-specific word) is translated with a term other than the one expected for the domain implied by the context.',
      subtypes: {
        inappropriate: {
          display: 'Inappropriate for context',
          description: 'Translation does not adhere to appropriate industry standard terminology or contains terminology that does not fit the context.',
        },
        inconsistent: {
          display: 'Inconsistent',
          description: 'Terminology is used in an inconsistent manner within the text.',
        },
      },
    },
    locale_convention: {
      display: 'Locale convention',
      description: 'The text does not adhere to locale-specific mechanical conventions and violates requirements for the presentation of content in the target locale.',
      subtypes: {
        address: {
          display: 'Address format',
          description: 'Content uses the wrong format for addresses.',
        },
        date: {
          display: 'Date format',
          description: 'A text uses a date format inappropriate for its locale.',
        },
        currency: {
          display: 'Currency format',
          description: 'Content uses the wrong format for currency.',
        },
        telephone: {
          display: 'Telephone format',
          description: 'Content uses the wrong form for telephone numbers.',
        },
        time: {
          display: 'Time format',
          description: 'Content uses the wrong form for time.',
        },
        name: {
          display: 'Name format',
          description: 'Content uses the wrong form for name.',
        },
      },
    },
    other: {
      display: 'Other',
      description: 'Any other issues (please provide a short description when prompted).',
      needs_note: true,
      source_side_ok: true,
    },
    non_translation: {
      display: 'Non-translation!',
      description: 'The whole sentence is completely not a translation of the source. This rare category, when used, overrides any other marked errors for that sentence, and labels the full translated sentence as the error span. Only available after choosing a major error.',
      forced_severity: 'major',
      override_all_errors: true,
    },
    source_error: {
      display: 'Source issue',
      description: 'Any issue in the source.',
      source_side_only: true,
    },
  },

  /**
   * Instructions are built using default section order and contents in
   * template-base.js.
   */
};
