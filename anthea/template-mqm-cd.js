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
 * This simplified template aims at finding examples of text that is
 * challenging to translate. It only has two error types: accuracy and fluency.
 */
antheaTemplates['MQM-CD'] = {
  severities: {
    major: {
      display: 'Major error',
      shortcut: 'M',
      color: 'pink',
      description: `
        A mistake that may mislead or confuse the reader. Examples: choosing the
        wrong pronoun or the wrong sense for a word, using the wrong term or
        name, omitting significant non-redundant content, introducing
        significant spurious content, making grammatical errors, using a very
        inappropriate style or register, or formatting text in a very odd way
        (for example in all caps when the source is not).`,
    },
    minor: {
      display: 'Minor error',
      shortcut: 'm',
      color: '#fbec5d',
      description: `
        Any imperfection that is not a major error. Examples: choosing a
        slightly different sense for a word, omitting or introducing content
        that doesn’t significantly affect the meaning, stilted (but
        grammatically correct) language, subtle problems with style, or typos
        that are easy to correct from context, such as missing or inconsistent
        punctuation (note that some punctuation errors can be major, however).`,
    },
  },

  /**
   * @const {number} Allow marking at most these many errors per segment. If
   *     set to 0, then no limit is imposed.
   */
  MAX_ERRORS: 0,

  /**
   * @const {number} If there are preceding sentence groups at the beginning of
   *     of the first document, we show these many of them, making the others
   *     accessible with a click on an expansion widget.
   */
  NUM_PRECEDING_VISIBLE: 0,

  /**
   * @const {boolean} Set this to true if the template instructions already
   *     include listings of errors and severities, and you do not want to
   *     auto-append the lists of errors/severities to the instructions.
   */
  SKIP_RATINGS_TABLES: false,

  /**
   * @const {boolean} Set this to true if you want the error span to be maked
   *     before the severity level is picked.
   */
  MARK_SPAN_FIRST: true,

  /**
   * @const {boolean} Set this to true if you want to allow error spans to start
   *     on whitespace.
   */
  ALLOW_SPANS_STARTING_ON_SPACE: true,

  errors: {
    accuracy: {
      display: 'Accuracy',
      description: 'Any error or imperfection that makes the target text not an accurate translation of the source.',
    },
    fluency: {
      display: 'Fluency',
      description: 'Any error or imperfection in the target text on its own - one that you would notice if you didn’t have access to the source text.',
    },
  },

  /**
   * Instructions to show at the top, in HTML format.
   */
  instructions: `
    <h3>Task Description</h3>
    <p>
      In this task, you will be shown a translation of a document that you will
      review and annotate for errors.
    </p>
    <p>
      An error-free translation should precisely reflect the source, and it
      should also read like text originally written in the target language.
      Report every occurrence where the translation falls short of that
      standard.
    </p>
    <p>
      You will be annotating short groups of one or more sentences that form
      part of a larger document. <b>It is important to take document context
      into account.</b> Words or phrases should be marked as errors if they are
      inconsistent in any way with previously-translated text. On the other
      hand, translations at the sentence level don’t have to strictly match the
      source if they are part of a coherent and complete translation of the
      whole document.
    </p>
    <p>
      Try to mark all errors, and be as fine-grained as possible. If two
      contiguous words are both wrong and the errors are independent, indicate
      two errors. It is fine to mark different errors that have overlapping
      spans. Use whitespace when available to indicate the location of
      omissions, otherwise mark the next word.
    <p>
      You need to indicate a severity level and a category for each error. The
      available choices are described below.
      <b>Please read these carefully before starting.</b>
    </p>
    <h3>Annotation Procedure</h3>
    <p>
      To mark an error:
      <ol>
        <li>Select the span of words affected by the error by clicking on the
          word/particle where the error begins, then clicking on the
          word/particle where the error ends. If it is only one word, then you
          have to click on it twice.</li>
        <li>Choose a severity using the buttons in the rightmost column.</li>
        <li>Select a category for the error.</li>
      </ol>
    </p>
    <p>
      If you change your mind, you can delete an error using the x button on the
      left side.
    </p>
    <p>
      After annotating all errors in a sentence group, use the right arrow key
      (or the button) to go to the next sentence. You can also navigate to
      previously-annotated sentences, to remove errors or add new ones.
    </p>
    <p>
      When you finish annotating the entire document, use the Select menu to
      download a json file, optionally deleting the evaluation in the process.
      If not deleted, your evaluation will remain available for revision next
      time you open Anthea.
    </p>
  `,

};
