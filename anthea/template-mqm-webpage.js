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
 * This is an MQM template for evaluating the translation of a piece of text
 * from a web page, while showing an image rendering of the web page for
 * additional context.
 */
antheaTemplates['MQM-WebPage'] = {
  severities: {
    major: {
      display: 'Major error',
      shortcut: 'M',
      color: 'pink',
      description: `
        Major errors are those that may mislead or confuse the reader.
        For example:
        (1) Translations that misrepresent the source text in
        some significant way.
        (2) Serious grammatical or syntactical or stylistic flaws.`,
    },
    minor: {
      display: 'Minor error',
      shortcut: 'm',
      color: '#fbec5d',
      description: `
        Minor errors are those that don't lead to significant loss of meaning
        and wouldn't confuse or mislead the reader. These errors would still
        be noticed and might decrease stylistic quality, fluency, or clarity.`,
    },
  },

  /**
   * @const {string} Template version identifier.
   */
  VERSION: 'v1.00-Feb-13-2023',

  /**
   * @const {number} Allow marking at most these many errors per segment. If
   *     set to 0, then no limit is imposed.
   */
  MAX_ERRORS: 5,

  /**
   * @const {boolean} Set this to true if the template instructions already
   *     include listings of errors and severities, and you do not want to
   *     auto-append the lists of errors/severities to the instructions.
   */
  SKIP_RATINGS_TABLES: false,

  /**
   * @const {boolean} Set this to true if you want to allow error spans to
   *     start on whitespace.
   */
  ALLOW_SPANS_STARTING_ON_SPACE: true,

  /**
   * @const {boolean} Set this to true if you want to present the error
   *     types/subtypes in a short, flat list (avoid using this if you have
   *     more than 10 error type/subtype combinatons).
   */
  FLATTEN_SUBTYPES: true,

  /**
   * @const {boolean} Set this to true if your TSV data for each docsys includes
   *     a web oage rendering and bounding box provided as an annotation on each
   *     first segment. The annotation should be a JSON string that looks like
   *     this (exccept that it should not have newlines):
   * {
   *   "source_context": {
   *     "url": "http://screenshot-image-url-for-source",
   *     "w": 100, "h": 120, "box": {"x": 5, "y": 7, "w": 90, "h": 30}
   *   },"
   *   "target_context": {
   *     "url": "http://screenshot-image-url-for-target",
   *     "w": 100, "h": 120, " "box": {"x": 5, "y": 7, "w": 90, "h": 30}
   *   },
   * }
   */
  USE_PAGE_CONTEXT: true,

  errors: {
    accuracy: {
      display: 'Accuracy',
      description:
          'The target text does not accurately reflect the source text.',
      subtypes: {
        mistranslation: {
          display: 'Mistranslation',
          description:
              'The target text does not accurately represent the source text. Examples: The source text talks about a person A knowing another person B, but the English translation says "A was intimate with B." The source text states that something never happens, whereas the translation says it happens "often" or "rarely." If the translation might have been OK for this sentence by itself but is wrong in context, use the Contextual Mistranslation subcategory instead of this one.',
        },
        contextual_mistranslation: {
          display: 'Contextual Mistranslation',
          description:
              'The target text does not accurately represent the source text, given the context (surronding sentences or other document elements). Examples: Consider the translation: "The woman was upset. In anger, he slammed the door as he left." The gender of the pronouns in the second sentence is wrong, because of the context established by the preceding sentence. Another example would be a document about insurance, containing the word "quote" (referring to an insurance quote), translated to "कहावत" in Hindi, which means "proverb."',
        },
        addition: {
          display: 'Addition',
          description:
              'The target text includes information not present in the source. Examples: A translated sentence that includes adverbs or adjectives without equivalents in by the original text.',
        },
        omission: {
          display: 'Omission',
          description:
              'Content is missing from the translation that is present in the source. Examples: A phrase has been dropped from the translation.',
          source_side_only: true,
        },
      },
    },
    fluency: {
      display: 'Fluency',
      description:
          'Issues related to the form or content of translated text, independent of its relation to the source text; errors in the translated text that prevent it from being understood.',
      subtypes: {
        grammar: {
          display: 'Grammar',
          description:
              'Issues related to the grammar or syntax of the text. Examples: An English text reads "They goes together," or "He could of fixed it."',
        },
        punctuation: {
          display: 'Punctuation',
          description:
              'Punctuation is used incorrectly (for the locale or style). Examples: An English compound adjective appearing before a noun is not hyphenated, as in "dog friendly hotel."',
        },
      },
    },
    style: {
      display: 'Style',
      description: 'The text has stylistic problems.',
      subtypes: {
        awkward: {
          display: 'Awkward',
          description:
              'The text is literal, written in an awkward style, unidiomatic or inappropriate in the context. Examples: The English metaphor, "putting the cart before the horse" is literally translated into Japanese. A sentence is unnecessarily convoluted or long, such as "The people transporting mechanism went skywards."',
        },
      },
    },
    other: {
      display: 'Other',
      description:
          'Any other issues (please describe very briefly). This could include inappropriate terminology (e.g., "acide sulfurique" is translated to "acid of sulfur" instead of "sulfuric acid."), errors in locale convention (name/address/date/time formats), typos or omissions in the source text, etc.',
      needs_note: true,
      source_side_ok: true,
    },
    non_translation: {
      display: 'Non-translation!',
      description:
          'The whole sentence is completely not a translation of the source. This rare category, when used, overrides any other marked errors for that sentence, and labels the full translated sentence as the error span. Only available after choosing a major error. Examples: the translated sentence is completely unrelated to the source sentence or is gibberish or is such a bad translation that there is virtually no part of the meaning of the source that has persisted.',
      forced_severity: 'major',
      override_all_errors: true,
    },
    unrateable: {
      display: 'Not rateable',
      description: `
            Pick this category if there are problems with the visibility
            of the text bounding boxes in the page images, or if a bounding box
            does not contain the text shown, or if the translation text clearly
            corresponds to a different page element than the source text.`,
      forced_severity: 'major',
      override_all_errors: true,
    },
  },

  /**
   * Instructions to show at the top, in HTML format.
   */
  instructions: `
    <h3>Instructions Overview</h3>
    <p>
      In this task, you will be shown translations of one or more sections of
      documents that you will review and annotate for errors. The goal of this
      task is to evaluate the quality of various human and machine translation
      outputs and compare them.
    </p>
    <p>
      You will rate the quality of the translations by marking errors, one
      segment at a time.
    </p>
    <p>
      In most cases, a single task will correspond to a single document taken
      from a single source (a web page, article). In some cases, though, there
      may be multiple documents in a task. The first sentence shown may be from
      the middle of the source document, not necessarily from the beginning.
      In that case, you can expand the preceding sentences to make sure the
      translation is contextually correct.
    </p>
    <p>
      In this task, you will also see web page images of the source page and
      the translated page, with bounding boxes drawn around the text sections
      being evaluated.
    </p>
    <h3>General Guidelines</h3>
    <p>
      The standard you should be reviewing the translations against is human
      translation quality. Report every occurrence where the translation falls
      short of that standard. Remember that the content you are reviewing may be
      a human translation or a machine translation. Apply the same standard
      regardless. The translation should be:
    </p>
    <ul>
      <li>Linguistically correct</li>
      <li>Accurate</li>
      <li>Readable (fluent and natural-sounding)</li>
      <li>With correct terminology appropriate in the context</li>
      <li>Consistent</li>
      <li>Faithful in tone and register to the source</li>
      <li>Cultural references or humor should be substituted with equivalents in
          the target language where appropriate (e.g. it's raining cats and
          dogs \u2192 it's pouring)</li>
    </ul>
    <h3>Annotation Process</h3>
    <ol>
      <li>Review the translation of each sentence against the source, following
          the general guidelines above.</li>
      <li>When you find an error, select the severity of the error using the
          buttons in the rightmost column (<b>Evaluations</b>):
        <ul>
          <li>Major error</li>
          <li>Minor error</li>
        </ul>
      </li>
      <li>Select the span of words affected by the error by clicking on the
          word/particle where the error "begins", then clicking on the
          word/particle where the error "ends". If it is only one
          word, then you have to click on it twice. (Errors can appear either on
          the translation side, or rarely, for the "Source error" type, on the
          source side).</li>
      <li>Select the category and subcategory of the error found. For example:
          Accuracy &gt; Mistranslation.</li>
      <li>Note the special error category named "<b>Non-Translation!</b>
          This should be used when the translated sentence is completely
          unrelated to the source sentence or is gibberish or is such a bad
          translation that there is virtually no part of the meaning of the
          source that has persisted.</li>
      <li>After annotating all errors in a sentence, use the <b>right arrow
          key</b> (or the <b>button</b>) to go to the next sentence.</li>
    </ol>
    <p>
      Sometimes, the bounding boxes of the selected text sections may not be
      visible. Or the translation text shown might have been misidentified
      (with the real translation clearly visible elsewhere on the translated
      page). In such cases, use the "Not rateable!" error category and enter a
      short reason as to what seems to be the problem (such as,
      "box not visible" or "translation is seen elsewhere"). Just like
      "Non-Translation!" this error overrides all other errors in the sentence
      and has Major severity.
    </p>
    <p>
      Occasionally, the translated sentence would have been altered to include
      an artificially injected error. Evaluating translation quality is a
      difficult and demanding task, and such "test sentences" are used to help
      you maintain the high level of attention the task needs. Once you have
      marked any error in a test sentence, its unaltered version will be shown.
      If you miss marking any error in a test sentence, you will be shown a
      cautionary reminder about that. In either case, once the unaltered
      version is revealed, you have to proceed to rate its quality just like all
      the other sentences.
    </p>
    <p><details>
      <summary>
        <b>Linguistic Guidelines</b>
      </summary>
      Please follow the general linguistic guidelines for your target language
      and mark an error for any part of any sentence that does not comply with
      them. Below is a reminder of what you should be looking for. Please note
      that we expect you to perform basic online research to verify the
      translation of company names, brand names, acronyms, titles of documents
      and cultural works whenever you are unsure.
      <ul>
        <li><b>Capitalization</b>: Product names, titles, and proper nouns
            should be capitalized even if they are not capitalized in the
            source. Titles should use sentence case or title case, based on what
            is used in your target locale. Title case is used mainly in English.
        </li>
        <li><b>Company names, brands, and institutions</b>: Company names and
            brand names should not be localized/translated. Names of
            institutions and organizations can be translated if appropriate in
            the target language.</li>
        <li><b>Acronyms</b>: Acronyms should be translated (with an equivalent
            acronym in the target language or a spelled out target version)
            where the translated version is more common than the source language
            version. Conversely, if the untranslated version is more common, a
            target version has not been established or the abbreviation or
            acronym constitutes a registered name or trademark, the abbreviation
            or acronym should be kept in the source language.</li>
        <li><b>Titles</b>: When evaluating translations of titles of books,
            movies, songs, games, research papers, academic journal titles,
            legal or official documents (foreign acts and regulations,
            international agreements, conventions, resolutions) use the
            following guidelines. Where an official translation exists, it
            should be used. Where no official translation is available, the
            title should be kept in the source language or an appropriate
            translation should be provided, based on what is accepted/most
            common in the particular case. Where no official translation exists,
            it is a common translation practice to keep the title in the source
            language and follow it with a translation in brackets. However,
            <i>do not expect this and do not log an error</i> if an explanation
            in brackets is not provided.</li>
        <li><b>Numbers</b>: Make sure the correct decimal and thousands
            separators are used</li>
        <li><b>Measurements</b>: Do not expect measurements to be converted
            between imperial and metric. They will most likely be just
            translated. Do not log errors as long as they are correctly
            translated.</li>
        <li><b>Currency</b>: Make sure currency acronyms and symbols are
            correctly localized and incorporated into the sentence.</li>
        <li><b>Time</b>: Make sure time is correctly indicated in the format
            that is acceptable in your target locale in the context of the
            sentence (24-hour clock vs. 12-hour clock)</li>
        <li><b>Dates</b>: Make sure the correct date format is used as per your
            target locale.</li>
      </ul>
    </details></p>
    <p><details>
      <summary>
        <b>Help &amp; Tips</b>
      </summary>
      Please be mindful of the following:
      <ol>
        <li>Before you start annotating, please read carefully the definitions
            for severities and error types listed below. Be sure to follow these
            definitions so we can have consistency between annotators and
            reliable results.\n</li>
        <li>Please take document context into account when annotating:
          <ol>
            <li>If a translation might be questionable on its own but is
                acceptable in the context of the document, then it should not be
                considered as an error.</li>
            <li>Similarly, if a translation might be acceptable in some
                contexts, but not within the current document, then it should be
                considered as an error.</li>
          </ol>
        </li>
        <li>When identifying errors, please be as fine-grained as possible.
            For example, if a sentence contains two words that are
            mistranslated, two separate errors should be recorded.</li>
        <li>If the same error occurs more than once in a particular document,
            all occurrences should be reported.</li>
        <li>Do not log more than 5 errors per segment.
          <ul>
            <li>If a sentence contains more than 5 errors, annotate only the 5
                most severe errors then stop, even if there are more errors
                left.</li>
            <li>If the 5 most severe errors cannot be reliably identified
                because the whole translation is too bad (e.g. word salad,
                completely nonsensical output, severely broken syntax), then
                apply the "Major" severity and mark the whole sentence
                as "Non-translation!".</li>
          </ul>
        </li>
        <li>To change the rating for a previous sentence in the current
            document, you can click on it. You can delete any individual error
            you might have mistakenly added.</li>
        <li>The estimated time is 1 minute per sentence, so if there are 10
            sentences then you will see 10 minutes at the top of the page.
            Some sentences will take more or less time.&nbsp;</li>
        <li>The task will expire after 30 minutes and you will lose your work,
            so be mindful of the time.</li>
      </ol>
    </details></p>
  `,
};
