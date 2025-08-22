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
 * This is the MQM "Target-First" template. For each segment (typically
 * a paragraph), the target side is shown first, with the source invisible.
 * Once the target has been read and annotated, then the source is shown
 * for a lightweight review where perhaps some major accuracy errors can then
 * be marked.
 */
antheaTemplates['MQM-Target-First'] = {
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
  subheadings: {
    source: 'Review after reading translation, to mark uncaught major errors',
    target: 'Read first and mark all issues that you can find',
  },

  /**
   * @const {string} Template version identifier.
   */
  VERSION: 'v1.00-Feb-13-2023',

  /**
   * @const {boolean} Only rate the target side, i.e., the translated text.
   */
  TARGET_SIDE_ONLY: false,

  /**
   * Make raters go through the translation text of each segment first, before
   * the source text of the segment is revealed.
   */
  TARGET_SIDE_FIRST: true,

  /**
   * @const {number} Allow marking at most these many errors per segment. If
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
   *     includes a web oage rendering and bounding box provided as an
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
      needs_source: true,
      subtypes: {
        reinterpretation: {
          display: 'Creative Reinterpretation',
          description: 'The target text reinterprets the source, but preserves its intent within its broader context (the document and its purpose, and the target locale). This can be because the translation adds, removes, or modifies text in accordance with the target locale, or simply makes creative changes that fully preserve the intent of the source text.',
        },
        mistranslation: {
          display: 'Mistranslation',
          description: 'The target text does not accurately represent the source text.',
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
      },
    },
    coherence: {
      display: 'Coherence',
      description: 'The marked text does not seem logical or does not make ' +
                   'sense in some way other than a Fluency or Style error ' +
                   'described earlier.',
      subtypes: {
        misfit: {
          display: 'Misfit',
          description: 'A span of text that is out of place, illogical, or is badly phrased, or does not make sense in the context of the text around it.',
        },
        gibberish: {
          display: 'Gibberish',
          description: 'The marked span of text is gibberish.'
        },
      },
    },
    terminology: {
      display: 'Terminology',
      description: 'A term (domain-specific word) is not the one expected for the domain implied by the context.',
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
   * Instructions are built using default section order and some default section
   * contents in template-base.js, with some sections' content modified here.
   */
  instructions_section_contents: {
    'Overview': `
      <h2>Overview</h2>
      <p>
        In this project, you will be shown translations of different documents
        that you will review and annotate for errors and issues. You will
        annotate by selecting spans of words affected by errors, then labeling
        them with severities and issue types. The goal of this project is to
        evaluate the quality of various human and machine translation outputs.
      </p>
      <p>
        The content for annotation consists of documents that are broken down
        into parallel segments. A segment may be one sentence, multiple
        sentences, or an entire paragraph. You will be able to read and annotate
        each segment in steps of sentences (on both the translation side and the
        source side).  Navigation through the document is explained in detail
        in the "Navigation" subsection.
      </p>
      `,
    'Navigation': `
      <h2>Navigation</h2>
      <p>
      Please note how the document is broken up into parallel segments and how
      segments are to be evaluated sentence-by-sentence, target-side first.
      </p>
      <ul>
        <li>
        Each task consists of text from a single document. Sometimes it can be
        a very short document, even a single sentence, but typically a document
        will have 10-20 sentences.
        </li>
        <li>
        Within each segment, you will be shown the target side first. Our goal
        is to favor natural translations that are fluent in the target language,
        and we believe that a significant part of this quality evaluation can be
        done by simply reading the translated text. Once you finish reading and
        annotating the translation side of the segment, the source segment will
        be made visible for a lightweight review to catch any major accuracy
        errors that may not have been noticeable earlier.
        </li>
        <li>
        When reading the target segment without having the source side
        available, you can still judge and annotate most quality issues, such as
        fluency, grammar, and style. In addition, you can use the “Coherence”
        error type to mark spans of text that seem to not belong, or appear
        illogical or incoherent in some manner.
        </li>
        <li>
        You should read each translated segment in its entirety, going
        sentence-by-sentence. The right arrow keyboard key and the right
        arrow button in the Evaluation column will let you move to the next
        sentence when you've finished reading and annotating a sentence.
        Once you finish reading and annotating a target segment, the right arrow
        key/button will take the focus to the first sentence of the source
        segment, which you will then proceed to read sentence-by-sentence
        (possibly marking some additional errors).
        </li>
        <li>
        When reading the source side, you can choose to jump to the target side
        by using the Tab key. The Tab key will similarly allow you to jump back
        to the source segment from the target segment.
        </li>
        <li>
        You can use the left arrow key or button to go back through the
        sentences and segments. You can also directly click on any previously
        read sentence to jump to it.
        </li>
        <li>
        We expect the evaluation phase when the source has been revealed to be a
        lightweight step, meant mainly to catch major accuracy errors. It is
        fine for a translation to have omitted minor details or have added minor
        explanations, as long as they are natural in the target context.
        </li>
      </ul>
      `,
    'Error Types and Subtypes defined': `
      <h2>Error Types and Subtypes defined</h2>
      <ul>
        <li>
          <b>Accuracy</b>.
          The target text does not accurately reflect the source text.
          <details>
            <summary>Subtypes of Accuracy:</summary>
            <ul>
              <li>
                <b>Creative Reinterpretation</b>.
                The target text <i>reinterprets the source, but preserves its
                intent</i>. Note that if the translation reinterprets the source
                text to such a great degree that it <i>changes</i> the intent,
                then it should be marked as using Mistranslation, Addition, or
                Omission subtypes, as appropriate.
                <ul>
                  <li>
                    Mark Creative Reinterpretation if the translation includes
                    additional text or omits some text that provides
                    explanations or context that may be obvious in the source
                    (target) locale, but not in the target (source) locale. For
                    example, an added short introduction of an entity not well
                    known in the target locale, or an omitted introduction of an
                    entity well known in the target locale.
                  </li>
                  <li>
                    The translation edits the source creatively, perhaps to make
                    the target text more fluent or expressive or localized. For
                    example, a reordering of sentences or an addition of text
                    that is supported by the overall passage being translated
                    and does not change the intent of the source text. Another
                    example would be that of the name of an exemplifying entity
                    being changed to a better local equivalent.
                  </li>
                </ul>
              </li>
              <li>
                <b>Mistranslation</b>.
                The target text does not accurately represent the source text.
                Examples: (1) The source text talks about a person A knowing
                another person B, but the English translation says "A was
                intimate with B." (2) The source text states that something
                never happens, whereas the translation says it happens "often"
                or "rarely." (3) Incorrectly gendered pronouns not warranted by
                the context, such as "Mary slammed the door as he left."
                Misgendering errors typically have a major severity as they
                significantly alter the meaning of the source text.
              </li>
              <li>
                <b>Source language fragment</b>.
                Content that should have been translated has been left
                untranslated. Example: A word, phrase or sentence in a German
                document has been copied verbatim into the English translation,
                but a natural translation is possible. Note that if only a short
                phrase or term is left untranslated, and that phrase or term
                is commonly used in the target language (for example, the French
                phrase "c'est la vie" used in English), then it's not an
                error (or is, at best, a minor error).
              </li>
              <li>
                <b>Addition</b>.
                The target text includes information not present in the source.
                Example: A translated sentence that includes adverbs or
                adjectives without equivalents in the original text, even after
                taking context into account.
              </li>
              <li>
                <b>Omission</b>.
                Content is missing from the translation that is present in the
                source. Example: A phrase has been dropped from the translation
                and it is not already implied by context. This error type needs
                to be annotated on the source side.
              </li>
            </ul>
          </details>
        </li>
        <li>
          <b>Fluency</b>.
          Issues related to the form or content of translated text, independent
          of its relation to the source text; errors in the translated text that
          make it harder to understand.
          <details>
            <summary>Subtypes of Fluency:</summary>
            <ul>
              <li>
                <b>Inconsistency</b>.
                The text shows internal inconsistency (not related to
                terminology). Examples: (1) A person is referred to with a
                masculine pronoun in one sentence and a feminine pronoun in the
                next sentence. This would be a major error. (2) An entity is
                referred to as "Secretary of State" in one paragraph but as
                "Minister of State" in the next. This would be a minor error,
                unless the context is one where both "Minister of State" and
                "Secretary of State" are valid technical terms with different
                meanings, in which case it would be a major error.
              </li>
              <li>
                <b>Grammar</b>.
                Issues related to the grammar or syntax of the text, other than
                spelling and orthography. Example: An English text reads "They
                goes together," or "He could of fixed it." Both of these
                examples have jarring flaws that significantly degrade the
                fluency of the text and would justify a major severity. However,
                it's possible that these sentence constructs are present in a
                context where such colloquial usage would not be out of place,
                and in such contexts they may not be errors.
              </li>
              <li>
                <b>Register</b>.
                The content uses the wrong grammatical register, such as using
                informal pronouns or verb forms when their formal counterparts
                are required. Example: A formal invitation uses the German
                informal pronoun "du" instead of "Sie." The annotator has to
                judge how egregious such a mistake is, in the context of the
                document, to decide the severity level of such an error. The use
                of an informal pronoun instead of a formal pronoun, in the
                context of a formal invitation may merit a major severity level,
                for example.
              </li>
              <li>
                <b>Spelling</b>.
                Issues related to spelling or capitalization of words, and
                whitespace. Example: The French word "mer" (sea) is used instead
                of the identically pronounced "maire" (mayor). This example
                would merit a major severity, as the meaning is substantially
                altered. Example: "Nobody in arizona expected snow." The word
                "Arizona" should start with an upper case letter, and this would
                be a minor error. Example: "Theyreached safely." This is a minor
                severity Spelling error because of the missing whitespace.
              </li>
              <li>
                <b>Punctuation</b>.
                Punctuation is used incorrectly (for the locale or style).
                Example: An English compound adjective appearing before a noun
                is not hyphenated, as in "dog friendly hotel." The reader can
                still grasp the intent quite easily in this case, so this
                example would have a minor severity.
              </li>
              <li>
                <b>Character encoding</b>.
                Characters are garbled due to incorrect application of an
                encoding. Examples: "ﾊｸｻ�ｽ､ｱ" and "瓣в眏." See
                <a href="https://en.wikipedia.org/wiki/Mojibake">en.wikipedia.org/wiki/Mojibake</a>
                for more. If such garbling is limited in scope and the overall
                text can still be understood, the severity level would be minor.
              </li>
            </ul>
          </details>
        </li>
        <li>
          <b>Style</b>.
          The text has stylistic problems.
          <details>
            <summary>Subtypes of Style:</summary>
            <ul>
              <li>
                <b>Unnatural or awkward</b>.
                The text is literal, written in an awkward style, unidiomatic or
                inappropriate in the context. Examples: (1) A sentence is
                translated literally, which copies a style that is not used in
                the target language. For example, “कौन कौन आया था?” in Hindi is
                translated into English as “Who who had come?” This would be a
                minor severity error. (2) A sentence is unnecessarily convoluted
                or too wordy, such as, "The lift traveled away from the ground
                floor." This would be a minor error. (3) Grammatically correct
                but slightly unnatural sounding sentences such as “From where
                did he come?” This would also be a minor error.
              </li>
              <li>
                <b>Bad sentence structure</b>.
                This error type is related to the arrangement of the sentence
                structure. The marked span of text is an unnecessary repetition,
                or it makes the sentence unnecessarily long, or it would have
                been better expressed as a clause in the previous sentence.
                Example (repetition): "Alexander had an idea. Alexander had a
                thought." This example would be a minor error, unless the
                context dictates otherwise. Example (long sentence): "The party,
                after blaming its losses on poor leadership, that the spokesman
                said could have paid more attention to the people's needs, split
                into two factions." This sentence could have been split into
                multiple sentences. Example (mergeable): "He gave him the money.
                He accepted the reward." These two sentences can be phrased
                better as a single sentence that makes it clearer who accepted
                the reward. This example is a minor error, without additional
                contextual information.
              </li>
            </ul>
          </details>
        </li>
        <li>
          <b>Coherence</b>.
          The marked text does not seem logical or does not make sense in some
          way other than a Fluency or Style error described earlier.
          <details>
            <summary>Subtypes of Coherence:</summary>
            <ul>
              <li>
                <b>Misfit</b>.
                A span of text (short phrase or even a whole sentence) that is
                out of place, illogical, or is badly phrased, or does not make
                sense in the context of the text around it. Examples: (1) A long
                and uncommon French phrase in English (major severity). (2) An
                idiom that makes no sense, such as "made milk of milk and water
                of water" (major severity). (3) An enumerated list with missing
                or additional items, such as “Step 1. Lather. Step 1. Lather.
                Step 4: Repeat." (without additional context, this would be two
                minor severity Coherence/Misfit errors).
              </li>
              <li>
                <b>Gibberish</b>.
                The marked span of text is gibberish. That is, it is nonsensical
                (words that do not form a coherent, logical phrase).
                Example, "The building drank chairs." This example is a major
                error.
              </li>
            </ul>
          </details>
        </li>
        <li>
          <b>Terminology</b>.
          A term (domain-specific word) is translated with a term other than the
          one expected for the domain or otherwise specified.
          <details>
            <summary>Subtypes of Terminology:</summary>
            <ul>
              <li>
                <b>Inappropriate for context</b>.
                Translation does not adhere to appropriate or contains
                terminology that does not fit the context. Example: "acide
                sulfurique" is translated to "acid of sulfur" instead of
                "sulfuric acid." This example would have a minor severity level.
              </li>
              <li>
                <b>Inconsistent</b>.
                Terminology is used in an inconsistent manner within the text.
                Example: The translation of a phone manual alternates between
                the terms "front camera" and "front lens." This example would
                have a minor severity level.
              </li>
            </ul>
          </details>
        </li>
        <li>
          <b>Locale convention</b>.
          The text does not adhere to locale-specific mechanical conventions and
          violates requirements for the presentation of content in the target
          locale.
          <details>
            <summary>Subtypes of Locale convention:</summary>
            <ul>
              <li>
                <b>Address format</b>.
                Content uses the wrong format for addresses. A part of the
                address was translated that should be kept in English. Examples:
                "1600 Pennsylvania Ave" is translated to Russian as
                "1600 Пенсильвания авеню" instead of "Пенсильвания авеню 1600."
                This example would have a minor severity level.
              </li>
              <li>
                <b>Date format</b>.
                A text uses a date format inappropriate for its locale. Example:
                The date "1969年1月6日" is shown as "6/1/1969" (instead of
                "1/6/1969") and the target locale can be clearly inferred to be
                U.S. English. For this example, the severity level would be
                major as the meaning of the date has been significantly altered.
              </li>
              <li>
                <b>Currency format</b>.
                Content uses the wrong format for currency. Example: The dollar
                symbol is used as a suffix, as in "100$." This example would
                have a minor severity level.
              </li>
              <li>
                <b>Telephone format</b>.
                Content uses the wrong form for telephone numbers. Example: An
                Indian phone number such as "xxxx-nnnnnn" is formatted as
                "(xxx) xnn-nnnn". This example would have a minor severity
                level.
              </li>
              <li>
                <b>Time format</b>.
                Content uses the wrong form for time. Example: Time is shown as
                "11.0" instead of "11:00" in a language where the former is a
                mistake. This example would have a minor severity level.
              </li>
              <li>
                <b>Name format</b>.
                Content uses the wrong form for name. Example: The Chinese name
                (which lists surname first) "马琳" is translated as "Lin Ma"
                instead of "Ma Lin". This example would also have a minor
                severity level as it the reader can make out the true intent
                quite easily.
              </li>
            </ul>
          </details>
        </li>
        <li>
          <b>Other</b>.
          Any other issues (please provide a short description when prompted).
        </li>
        <li>
          <b>Non-translation!</b>
          The sentence as a whole is completely not a translation of the source.
          This rare category, when used, overrides any other marked errors for
          that sentence and labels the full translated sentence as the error
          span. Only available after choosing a major severity error. Example:
          the translated sentence is completely unrelated to the source sentence
          or is gibberish or is such a bad translation that there is virtually
          no part of the meaning of the source that has been retained.
        </li>
        <li>
          <b>Source issue</b>.
          Any issue in the source. Examples: The source has meaning-altering
          typos or omissions ("The was blue.") or is nonsensical
          ("Th,jad jakh ;ih"). Note that even in the presence of source issues,
          translation errors should be annotated when possible. If some part of
          the source is completely garbled, then the corresponding translation
          need only be checked for Fluency/Style errors.
        </li>
      </ul>
      `
  },
};
