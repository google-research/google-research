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
 * This is an MQM template in which the rater only evaluates the
 * target side.
 */
antheaTemplates['MQM-Monolingual'] = {
  severities: {
    major: {
      display: 'Major severity',
      shortcut: 'M',
      color: 'pink',
      description: 'Major severity errors significantly hinder comprehension ' +
                   'or significantly degrade the quality of the text.',
    },
    minor: {
      display: 'Minor severity',
      shortcut: 'm',
      color: '#fbec5d',
      description: 'Minor severity errors are noticeable but minor flaws in ' +
                   'the text. They do not significantly hinder comprehension ' +
                   'and they do not significantly degrade the quality of the text.',
    },
  },

  /**
   * @const {string} Template version identifier.
   */
  VERSION: 'v1.00-Feb-13-2023',

  /**
   * @const {boolean} Only rate the target side, i.e., the translated text.
   */
  TARGET_SIDE_ONLY: true,

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
    fluency: {
      display: 'Fluency',
      description: 'Issues related to the form or content of text, errors that prevent it from being understood clearly.',
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
          description: 'Text does not adhere to appropriate industry standard terminology or contains terminology that does not fit the context.',
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
    },
  },

  /**
   * Instructions to show at the top, in HTML format.
   */
  instructions: `
    <style>
      .anthea-mqm-instructions .summary-heading {
        font-weight: bold;
        font-size: 125%;
      }
      .anthea-mqm-instructions th,
      .anthea-mqm-instructions td {
        border: 1px solid gray;
        vertical-align: top;
        padding: 4px;
      }
      .anthea-mqm-instructions td:first-child {
        font-weight: bold;
      }
      .anthea-mqm-instructions table {
        border-collapse: collapse;
      }
      .span-major {
        background-color: pink;
      }
      .span-minor {
        background-color: #fbec5d;
      }
    </style>

    <h2>Overview</h2>
    <p>
      In this project, you will be shown text from different documents that
      you will review and annotate for errors. You will annotate by
      selecting spans of words affected by errors, then labeling them with the
      severities and error types.
    </p>
    <p>
      The content for annotation comes in tasks, that are broken down into
      segments (typically consisting of one sentence per segment, but can also
      consist of multiple sentences). In most cases, a single task will
      correspond to a single document taken from a single source (a web page,
      article). In some project types, though, there may be multiple documents
      in a task. The first segment shown may be from the middle of the source
      document, not necessarily from the beginning.
    </p>

    <h2>General Guidelines</h2>
    <p>
      The standard you should be reviewing the document text against is human
      authorship by a native speaker of the language. Report every occurrence
      where the text falls short of that standard. The text should be:
    </p>
    <ul>
      <li>Linguistically correct</li>
      <li>Coherent</li>
      <li>Readable (fluent and natural-sounding)</li>
      <li>With correct terminology appropriate in the context</li>
      <li>Consistent in its use of terms, pronouns, register</li>
    </ul>
    <p>
      Please be mindful of the following:
    </p>
    <ol>
      <li>
        Before you start annotating, please read carefully the definitions
        for severities and error/issue types. Be sure to follow these
        definitions so we can have consistency between annotators and
        reliable results. In particular, it is important to understand the
        difference between “Major” and “Minor” severities and how to
        label text spans that cover the issue identified.
      </li>
      <li>
        When identifying errors, please be as fine-grained as possible, limiting
        the marked error spans to minimal pieces of text that capture the error.
      </li>
      <li>
        If the same error occurs more than once in a particular document or
        even in a sentence, all occurrences should be reported. For consistency
        problems, assume that the <b>first</b> instance of a set of inconsistent
        entities is correct, and label any subsequent entities that are
        inconsistent with it as errors. For example, if a text contains
        “Doctor Smith … Dr. Smith … Doctor Smith … Dr. Smith”, the 2nd and 4th
        occurrences should be marked.
      </li>
    </ol>

    <h2>Annotation Process</h2>
    <ol>
      <li>
        Review the text of each segment, following the general guidelines
        above.
      </li>
      <li>
        Select the <b>span</b> of words affected by the issue by clicking on
        the word/particle where the identified issue “begins”, then clicking on
        the word/particle where the issue “ends”. If it is only one word, then
        you have to click on it twice.
        <ul>
          <li>The marked span should be the minimal contiguous sequence such
              that modifying the word(s) within the span, deleting the span, or
              moving the word(s) somewhere else in the segment will remove the
              identified issue. The span should not include adjoining words that
              are not directly affected by the identified issue and do
              not need to be modified in order for the issue to be fixed.</li>
          <li>The shorter the span, the more useful it is.</li>
          <li>When it comes to "Style/Unnatural or awkward" errors, please
              pinpoint the error rather than extend the span to an entire
              clause.</li>
          <li>If a single issue affects words that do not directly follow each
              other, as is the case with split verbs in German
              (“teilte die Feuerwehr auf”) or phrasal verbs in English
              (“called Mary and her brother up”), log the issue only for the
              first part (“teilte”, “called”) and do not log anything for the
              latter part (“auf”, “up”). The text between “teilte” and “auf”,
              or between “called” and “up”, should not be included in the span
              if the issue is with the verb only (“aufteilen”, “call up”).
          </li>
        </ul>
      </li>
      <li>
        Select the <b>severity</b> of the issue using the buttons in the
        rightmost column ("Evaluations") or their keyboard shortcuts:
        <ul>
          <li>Major severity (M)</li>
          <li>Minor severity (m)</li>
        </ul>
      </li>
      <li>Select the <b>category</b> (also called <b>type</b>) and
          <b>subcategory</b> (also called <b>subtype</b>) of the error found.
          For example: Coherence &gt; Misfit.</li>
      <li>After annotating all identified issues in a segment, use the <b>right
          arrow key</b> (or the <b>button</b>) to go to the next segment.</li>
    </ol>

    <details>
      <summary>
        <span class="summary-heading">Annotation Tips</span>
      </summary>
      <ol>
        <li>To change the rating for a previous segment in the current
            document, you can click on it. You can delete any individual issue
            that you might have mistakenly added.</li>
        <li>
          Occasionally, the text segment would have been altered to include an
          artificially injected error. Evaluating text quality is a difficult
          and demanding task, and such "test segments" are used to help you
          maintain the high level of attention the task needs. Once you have
          marked any error in a test segment, its unaltered version
          will be shown. If you miss marking any error in a test segment, you
          will be shown a cautionary reminder about that. In either case, once
          the unaltered version is revealed, you have to proceed to rate its
          quality just like all the other segment.
        </li>
      </ol>
    </details>

    <h2>Severities defined</h2>
    <p>We define error/issue severity levels in this section.</p>
    <ul>
      <li>
        <b>Major severity</b>:
        Major severity errors significantly hinder comprehension or
        significantly degrade the quality of the text.
        <ul>
          <li>
            Typically, coherence and terminology errors fall here, as well as
            egregious style or grammar errors.
          </li>
          <li>
            <details>
              <summary><b>Examples</b></summary>
              <table>
                <tr>
                  <th>Language</th>
                  <th>Text</th>
                  <th>Comments</th>
                </tr>
                <tr>
                  <td>DE</td>
                  <td><span class="span-major">mache die Schrecken
                      leicht</span></td>
                  <td>
                      The phrase makes no sense.</td>
                </tr>
                <tr>
                  <td>EN</td>
                  <td>But this system <span class="span-major">is not
                      anxious</span></td>
                  <td>"Systems" are not "anxious".</td>
                </tr>
                <tr>
                  <td>EN</td>
                  <td>Tsinghua celebrates its 110th birthday, President Qiu
                      Yong: The greatness of a university lies in cultivating
                      <span class="span-major">capitalized people</span></td>
                  <td>"Capitalized people" makes no sense.</td>
                </tr>
              </table>
            </details>
          </li>
        </ul>
      </li>
      <li>
        <b>Minor severity</b>:
        Minor severity errors are noticeable but minor flaws in the text.
        They do not significantly hinder comprehension and they do not
        significantly degrade the quality of the text. They might make the
        text seem slightly odd, or they may slightly decrease the stylistic
        quality of the text.
        <ul>
          <li>
            Typically, the kinds of errors that fall under this severity level
            are grammar, spelling (including capitalization and whitespace),
            style, punctuation, and locale convention.
          </li>
          <li>
            <details>
              <summary><b>Examples</b></summary>
              <table>
                <tr>
                  <th>Language</th>
                  <th>Text</th>
                  <th>Comments</th>
                </tr>
                <tr>
                  <td>DE</td>
                  <td>In einem exklusiven Interview mit Fast Company scherzte
                      Berners-Lee, dass die Absicht hinter Inrupt die
                      <span class="span-minor">"</span>Weltherrschaft<span
                      class="span-minor">"</span> sei.</td>
                  <td>German uses curly quotes, opening down, closing up.</td>
                </tr>
                <tr>
                  <td>DE</td>
                  <td>
                    Die zeitlich begrenzte Veranstaltung
                    <span class="span-minor">Taste of Knotts Essen, Bier und
                    Wein</span> läuft bis zum 13. September ohne Fahrgeschäfte,
                    Achterbahnen oder andere Attraktionen des Themenparks.
                  </td>
                  <td>Awkward syntax.</td>
                </tr>
                <tr>
                  <td>EN</td>
                  <td>
                    Because most of them are released, the hope is that there is
                    a <span class="span-minor">reproduction</span> cell
                    contained in the pollen.
                  </td>
                  <td>A minor terminological error. It should read
                      "reproductive cell."</td>
                </tr>
              </table>
            </details>
          </li>
        </ul>
      </li>
    </ul>

    <h2>Error Types and Subtypes defined</h2>
    <ul>
      <li>
        <b>Fluency</b>.
        Issues related to the form or content of the text.
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
              goes together," or "He could of fixed it." Both of these examples
              have jarring flaws that significantly degrade the fluency of
              the text and would justify a major severity. However, it's
              possible that these sentence constructs are present in a context
              where such colloquial usage would not be out of place, and in
              such contexts they may not be errors.
            </li>
            <li>
              <b>Register</b>.
              The content uses the wrong grammatical register, such as using
              informal pronouns or verb forms when their formal counterparts are
              required. Example: A formal invitation uses the German informal
              pronoun "du" instead of "Sie." The annotator has to judge how
              egregious such a mistake is, in the context of the document,
              to decide the severity level of such an error. The use of an
              informal pronoun instead of a formal pronoun, in the context of
              a formal invitation may merit a major severity level, for example.
            </li>
            <li>
              <b>Spelling</b>.
              Issues related to spelling or capitalization of words, and
              whitespace. Example: The French word "mer" (sea) is used instead
              of the identically pronounced "maire" (mayor). This example would
              merit a major severity, as the text would be substantially hard
              to understand.
              Example: "Nobody in arizona expected snow." The word "Arizona"
              should start with an upper case letter, and this would be a minor
              error. Example: "Theyreached safely." This is a minor severity
              Spelling error because of the missing whitespace.
            </li>
            <li>
              <b>Punctuation</b>.
              Punctuation is used incorrectly (for the locale or style).
              Example: An English compound adjective appearing before a noun is
              not hyphenated, as in "dog friendly hotel." The reader can
              still grasp the intent quite easily in this case, so this example
              would have a minor severity.
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
              The text is written in an awkward style, unidiomatic or
              inappropriate in the context. Examples:
              (1) A sentence is unnecessarily convoluted or too wordy, such as
              "The lift traveled away from the ground floor." This would be a
              minor error. (2) Grammatically correct but slightly unnatural
              sounding sentences such as “From where did he come?” This would
              also be a minor error.
            </li>
              <b>Bad sentence structure</b>.
              This error type is related to the arrangement of the sentence
              structure. The marked span of text is an unnecessary repetition,
              or it makes the sentence unnecessarily long, or it would have been
              better expressed as a clause in the previous sentence. Example
              (repetition): "Alexander had an idea. Alexander had a thought."
              This example would be a minor error, unless the context dictates
              otherwise. Example (long sentence): "The party, after blaming its
              losses on poor leadership, that the spokesman said could have
              paid more attention to the people's needs, split into two
              factions." This sentence could have been split into multiple
              sentences. Example (mergeable): "He gave him the money. He
              accepted the reward." These two sentences can be phrased better
              as a single sentence that makes it clearer who accepted the
              reward. This example is a minor error, without additional
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
        A term (domain-specific word) is not what one would expect for the
        domain.
        <details>
          <summary>Subtypes of Terminology:</summary>
          <ul>
            <li>
              <b>Inappropriate for context</b>.
              The text does not adhere to appropriate or contains terminology
              that does not fit the context. Example: "acid of sulfur" instead
              of "sulfuric acid."
              This example would have a minor severity level.
            </li>
            <li>
              <b>Inconsistent</b>.
              Terminology is used in an inconsistent manner within the text.
              Example: The text in a phone manual alternates between the
              terms "front camera" and "front lens." This example would
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
              Content uses the wrong format for addresses. Example:
              an address written in Russian as
              "1600 Пенсильвания авеню" instead of "Пенсильвания авеню 1600."
              This example would have a minor severity level.
            </li>
            <li>
              <b>Date format</b>.
              A text uses a date format inappropriate for its locale. Example:
              The date is shown as "13-1-1969" and the document locale can be
              clearly inferred to be U.S. English. For this example, the
              severity level would be major as it might significantly confuse
              the reader.
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
              "(xxx) xnn-nnnn". This example would have a minor severity level.
            </li>
            <li>
              <b>Time format</b>.
              Content uses the wrong form for time. Example: Time is shown as
              "11.0" instead of "11:00" in a language where the former is a
              mistake. This example would have a minor severity level.
            </li>
            <li>
              <b>Name format</b>.
              Content uses the wrong form for name. Example: "Biden Joe" instead
              of "Joe Biden." This example would have a minor severity level.
            </li>
          </ul>
        </details>
      </li>
      <li>
        <b>Other</b>.
        Any other issues (please provide a short description when prompted).
      </li>
    </ul>

    <details>
      <summary>
        <span class="summary-heading">Annotations exemplified in detail</span>
      </summary>
      <table>
        <tr>
          <th>Language</th>
          <th>Text</th>
          <th>Correct annotations</th>
          <th>Comments</th>
        </tr>
        <tr>
          <td>DE</td>
          <td>Dies ist das einzigartige <span class="span-minor">Pent
              House</span> <span class="span-major">1BR</span> Apartment mit
              einer umlaufenden Terrasse.</td>
          <td>
            1. Fluency - Spelling - Minor ("Pent House")
            <br>
            2. Coherence - Misphrased - Major ("1BR")
          </td>
          <td>
            1. "Pent House" should be spelled as "Penthouse" in German; this is
               a minor spelling issue.
            <br>
            2. The abbreviation "1BR" means "one bedroom" in English and is not
              used in German. This is a major error because the text is
              not understandable.
          </td>
        </tr>
        <tr>
          <td>EN</td>
          <td>Pyrite with a small amount of impurities and
              <span class="span-major">star points</span> is very valuable in
              comparison.</td>
          <td>Coherence - Misphrased - Major</td>
          <td>The expression "star points" does not make sense here, so it
              should be rated as a major error.</td>
        </tr>
      </table>
    </details>

    <h2>Feedback</h2>
    <p>
    At the bottom right in the Evaluation column, there is a Feedback section.
    Please feel free to provide any feedback or notes. You can express things
    like:
    </p>
    <ul>
      <li>The document was too complex, or the topic was unfamiliar.</li>
      <li>Some parts of the instructions were unclear.</li>
      <li>Something in the user interface worked well or did not work well.</li>
      <li>You can also provide an overall thumbs up/down rating for your
          experience in evaluating that document.</li>
      <li>Any other comments or notes that you would like to provide about your
          experience.</li>
    </ul>
    <br>
  `,
};
