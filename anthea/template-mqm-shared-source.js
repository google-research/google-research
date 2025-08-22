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
 * This is very similar to the MQM template, but with the restriction that all
 * documents in the evaluation represent different translations of the same
 * source. The instructions have been updated to reflect this. In the future,
 * additional features will be added that leverage the fact that the source is
 * shared across all translations.
 */
antheaTemplates['MQM-Shared-Source'] = {
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
      description:
          'Minor severity errors are noticeable but minor flaws in the ' +
          'translated text. They do not significantly alter the ' +
          'meaning of the source text, and they do not ' +
          'significantly degrade the quality of the text.',
    },
  },

  /**
   * @const {string} Template version identifier.
   */
  VERSION: 'v1.00-Jun-12-2024',

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

  /**
   * @const {boolean} Set this to true if the evaluation is guaranteed to
   *     consist of multiple translations of the same single source document.
   */
  SHARED_SOURCE: true,

  errors: {
    accuracy: {
      display: 'Accuracy',
      description:
          'The target text does not accurately reflect the source text.',
      subtypes: {
        reinterpretation: {
          display: 'Creative Reinterpretation',
          description:
              'The target text reinterprets the source, but preserves its ' +
              'intent within its broader context (the document and its ' +
              'purpose, and the target locale). This can be because the ' +
              'translation adds, removes, or modifies text in accordance ' +
              'with the target locale, or simply makes creative changes that ' +
              'fully preserve the intent of the source text.',
        },
        mistranslation: {
          display: 'Mistranslation',
          description:
              'The target text does not accurately represent the source text.',
        },
        untranslated: {
          display: 'Source language fragment',
          description:
              'Content that should have been translated has been left ' +
              'untranslated.',
        },
        addition: {
          display: 'Addition',
          description:
              'The target text includes information not present in the source.',
        },
        omission: {
          display: 'Omission',
          description:
              'Content is missing from the translation that is present in ' +
              'the source.',
          source_side_only: true,
        },
      },
    },
    fluency: {
      display: 'Fluency',
      description:
          'Issues related to the form or content of translated text, ' +
          'independent of its relation to the source text; errors in the ' +
          'translated text that prevent it from being understood clearly.',
      subtypes: {
        inconsistency: {
          display: 'Inconsistency',
          description:
              'The text shows internal inconsistency (not related to ' +
              'terminology).',
        },
        grammar: {
          display: 'Grammar',
          description:
              'Issues related to the grammar or syntax of the text, other ' +
              'than spelling and orthography.',
        },
        register: {
          display: 'Register',
          description:
              'The content uses the wrong grammatical register, such as ' +
              'using informal pronouns or verb forms when their formal ' +
              'counterparts are required.',
        },
        spelling: {
          display: 'Spelling',
          description:
              'Issues related to spelling or capitalization of words, and ' +
              'incorrect omission/addition of whitespace.',
        },
        breaking: {
          display: 'Text-Breaking',
          description:
              'Issues related to missing or unwarranted paragraph breaks or ' +
              'line breaks.',
        },
        punctuation: {
          display: 'Punctuation',
          description:
              'Punctuation is used incorrectly (for the locale or style).',
        },
        display: {
          display: 'Character encoding',
          description:
              'Characters are garbled due to incorrect application of an ' +
              'encoding.',
        },
      },
    },
    style: {
      display: 'Style',
      description: 'The text has stylistic problems.',
      subtypes: {
        awkward: {
          display: 'Unnatural or awkward',
          description:
              'The text is literal, written in an awkward style, unidiomatic ' +
              'or inappropriate in the context.',
        },
        sentence_structure: {
          display: 'Bad sentence structure',
          description:
              'The marked span of text is an unnecessary repetition, or ' +
              'makes the sentence unnecessarily long, or would have been ' +
              'better as a clause in the previous sentence.'
        },
      },
    },
    terminology: {
      display: 'Terminology',
      description:
          'A term (domain-specific word) is translated with a term other ' +
          'than the one expected for the domain implied by the context.',
      subtypes: {
        inappropriate: {
          display: 'Inappropriate for context',
          description:
              'Translation does not adhere to appropriate industry standard ' +
              'terminology or contains terminology that does not fit the ' +
              'context.',
        },
        inconsistent: {
          display: 'Inconsistent',
          description:
              'Terminology is used in an inconsistent manner within the text.',
        },
      },
    },
    locale_convention: {
      display: 'Locale convention',
      description:
          'The text does not adhere to locale-specific mechanical ' +
          'conventions and violates requirements for the presentation of ' +
          'content in the target locale.',
      subtypes: {
        address: {
          display: 'Address format',
          description: 'Content uses the wrong format for addresses.',
        },
        date: {
          display: 'Date format',
          description:
              'A text uses a date format inappropriate for its locale.',
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
      description:
          'Any other issues (please provide a short description when ' +
          'prompted).',
      needs_note: true,
      source_side_ok: true,
    },
    non_translation: {
      display: 'Non-translation!',
      description:
          'The whole sentence is completely not a translation of the source. ' +
          'This rare category, when used, overrides any other marked errors ' +
          'for that sentence, and labels the full translated sentence as ' +
          'the error span. Only available after choosing a major error.',
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
   * Only the 'Navigation' section of the instructions differs from the content
   * and order defined in template-base.js.
   */
  instructions_section_contents: {
    'Navigation': `
      <h2>Navigation</h2>
      <p>
        Each task consists of multiple translations of the same single source
        document. Sometimes it can be a very short document (even a single
        sentence), but typically a document will have 5-10 sentences. Each
        translation of the document is presented separately.
      </p>
      <ul>
        <li>
        You will go through a document in convenient steps of small
        "sub-paragraphs" (groups of consecutive sentences from the same
        paragraph). You can move from one sub-paragraph to the next (and back)
        using the arrow keys or using the buttons labeled with left and right
        arrows. Note that these sub-paragraphs are automatically created on the
        source side and the translation side, <i>independently</i> of each
        other, purely for convenient navigation through potentially long
        documents. In particular, they are <i>not</i> expected to be
        aligned—i.e., the translation of the third (say) sub-paragraph on the
        source side need not be exactly just the third sub-paragraph on the
        translation side.
        </li>
        <li>
        If the document was translated in steps of segments that were smaller
        than the whole document, then at the end of the last sub-paragraph on
        the source side of each segment, the right arrow key/button will take
        you to the start (i.e., the first sub-paragraph) of the translation of
        that segment. And, at the end of the translation side of that segment,
        it will take you to the start of the source side of the next segment.
        </li>
        <li>
        If the entire document was translated as one segment, then the right
        arrow key/button will take you to the start of the translation only at
        the end of the source side of the whole document.
        </li>
        <li>
        If a source segment appears to be too long, you can (at any time) choose
        to jump to the translation side after reading some part of the source,
        using the Tab key. The Tab key will similarly allow you to jump back to
        the source side from the translation side. We leave the judgment to you,
        as to how to divide your annotation process for long segments into
        smaller units.
        </li>
        <li>
        You can also use the left arrow key/button to go <i>back</i> through the
        sub-paragraphs and segments. You can also directly click on any
        previously read part of the text to jump back to it.
        </li>
        <li>
        When you are done annotating one translation, click the
        "Next Translation" button to move to the next translation. You can use
        "Prev Translation" to go back. If "Next Translation" is not visible,
        then you have reached the final translation.
        </li>
        <li>
        Note that the source is the same for all translations, but source-side
        ratings are not currently shared between translations. Please mark any
        source-side errors in each translation.
        </li>
      </ul>
      `,
  }
};
