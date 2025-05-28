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
 * This is an MQM template for evaluating the translation of a piece of media
 * (image or video). The source is either an image of text to be translated, or
 * a video containing audio to be translated, and the target is text.
 */
antheaTemplates['MQM-Source-Media'] = {
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
          'meaning of the source media, and they do not ' +
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
  TARGET_SIDE_ONLY: true,

  /**
   * @const {boolean} Use media (image/video) for the source instead of text,
   * provided via JSON annotations on the first segment of each docsys that look
   * like this (for a video):
   * {"source_media":
   *   {"url": "https://youtube-or-drive-embed-link", "type": "video"}}
   * or this (for an image):
   * {"source_media":
   *   {"url": "https://image-link", "type": "image"}}
   */
  USE_SOURCE_MEDIA: true,

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
      description:
          'The target text does not accurately reflect the source media.',
      subtypes: {
        reinterpretation: {
          display: 'Creative Reinterpretation',
          description:
              'The target text reinterprets the source, but preserves its intent within its broader context (the document and its purpose, and the target locale). This can be because the translation adds, removes, or modifies text in accordance with the target locale, or simply makes creative changes that fully preserve the intent of the source media.',
        },
        mistranslation: {
          display: 'Mistranslation',
          description:
              'The target text does not accurately represent the source media.',
        },
        gender_mismatch: {
          display: 'Gender Mismatch',
          description:
              'The gender is incorrect (incorrect pronouns, noun/adjective endings, etc).',
        },
        untranslated: {
          display: 'Source language fragment',
          description:
              'Content that should have been translated has been left untranslated.',
        },
        addition: {
          display: 'Addition',
          description:
              'The target text includes information not present in the source.',
        },
        omission: {
          display: 'Omission',
          description:
              'Content is missing from the translation that is present in the source.',
          source_side_only: true,
          needs_note: true,
        },
      },
    },
    fluency: {
      display: 'Fluency',
      description:
          'Issues related to the form or content of translated text, independent of its relation to the source media; errors in the translated text that prevent it from being understood clearly.',
      subtypes: {
        inconsistency: {
          display: 'Inconsistency',
          description:
              'The text shows internal inconsistency (not related to terminology).',
        },
        grammar: {
          display: 'Grammar',
          description:
              'Issues related to the grammar or syntax of the text, other than spelling and orthography.',
        },
        register: {
          display: 'Register',
          description:
              'The content uses the wrong grammatical register, such as using informal pronouns or verb forms when their formal counterparts are required.',
        },
        spelling: {
          display: 'Spelling',
          description:
              'Issues related to spelling or capitalization of words, and incorrect omission/addition of whitespace.',
        },
        breaking: {
          display: 'Text-Breaking',
          description:
              'Issues related to missing or unwarranted paragraph breaks or line breaks.',
        },
        punctuation: {
          display: 'Punctuation',
          description:
              'Punctuation is used incorrectly (for the locale or style).',
        },
        display: {
          display: 'Character encoding',
          description:
              'Characters are garbled due to incorrect application of an encoding.',
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
              'The text is literal, written in an awkward style, unidiomatic or inappropriate in the context.',
        },
        sentence_structure: {
          display: 'Bad sentence structure',
          description:
              'The marked span of text is an unnecessary repetition, or makes the sentence unnecessarily long, or would have been better as a clause in the previous sentence.'
        },
        archaic_word: {
          display: 'Archaic or obscure word choice',
          description:
              'An archaic or lesser-known word is used where a more colloquial term would be a better fit.',
        },
      },
    },
    terminology: {
      display: 'Terminology',
      description:
          'A term (domain-specific word) is translated with a term other than the one expected for the domain implied by the context.',
      subtypes: {
        inappropriate: {
          display: 'Inappropriate for context',
          description:
              'Translation does not adhere to appropriate industry standard terminology or contains terminology that does not fit the context.',
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
          'The text does not adhere to locale-specific mechanical conventions and violates requirements for the presentation of content in the target locale.',
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
          'Any other issues (please provide a short description when prompted).',
      needs_note: true,
    },
    non_translation: {
      display: 'Non-translation!',
      description:
          'The whole sentence is completely not a translation of the source. This rare category, when used, overrides any other marked errors for that sentence, and labels the full translated sentence as the error span. Only available after choosing a major error.',
      forced_severity: 'major',
      override_all_errors: true,
    },
  },

  /**
   * Overwrite some section contents from template-base.js, and add a
   * "Source Media Guidelines" section.
   */
  instructions_section_contents: {
    'Overview': `
      <h2>Overview</h2>
      <p>
        In this project, you will be shown translations of different media
        that you will review and annotate for errors and issues. You will
        annotate by selecting spans of words affected by errors, then labeling
        them with severities and issue types. The goal of this project is to
        evaluate the quality of various human and machine translation outputs.
      </p>
      <p>
        The content for annotation consists of source media (either a video or
        an image) and a translation of that media, broken down into segments. A
        segment may be one sentence, multiple sentences, an entire paragraph, or
        an entire document. After reading/watching the source media (with audio for videos), you will be able
        to read and annotate each target segment in steps of groups of
        sentences. Navigation through the document is explained in detail in the
        "Navigation" subsection.
      </p>
      `,
    'General Guidelines': `
      <h2>General Guidelines</h2>
      <p>
        The standard you should be reviewing the translations against is
        <b>human translation quality</b>. Report every occurrence where the
        translation falls short of that standard. Remember that the content you
        are reviewing may be a human translation or a machine translation.
        <b><i>Apply the same standard regardless</i></b>. The translation should
        be:
      </p>
      <ul>
        <li>Linguistically correct</li>
        <li>Accurate</li>
        <li>Readable (fluent, grammatically correct, and natural-sounding)</li>
        <li>With terminology appropriate in the context</li>
        <li>Consistent</li>
        <li>Faithful in tone and register to the source</li>
        <li>Appropriately transformed for the target context: cultural
            references or humor should be substituted with equivalents in
            the target language where appropriate (e.g. it's raining cats and
            dogs \u2192 it's pouring).</li>
      </ul>
      <p>
        Please be mindful of the following:
      </p>
      <ol>
        <li>
          Before you start annotating, please read carefully the definitions
          for severities and error/issue types. Be sure to follow these
          definitions so that we can have consistency between annotators, and
          reliable results. In particular, it is important to understand the
          difference between “Major” and “Minor” severities and how to
          label text spans that cover the issue identified.
        </li>
        <li>
          We deliberately use the terms "error" and "issue" interchangeably in
          these instructions. Most issues with translations are also errors,
          but occasionally, an issue may or may not be perceived as an error,
          depending upon the application, domain, and/or subjective preferences.
        </li>
        <li>Please take document context into account when annotating:
          <ul>
            <li>If a translation might be questionable on its own but is
                acceptable in the context of the document, then it should not be
                considered as an error. For example, a noun may get replaced
                by a pronoun in the translation even though a noun was used in
                the source, and unless this change makes the translation awkward
                to read, it should not be marked as an error.</li>
            <li>Similarly, it is OK for the translation to make use of the
                context to omit some part of the source that is obvious
                from the context (even if it is not omitted in the source).
                For example, an adjective may not have to be repeated in the
                translated text, if it is naturally obvious from the context.
            </li>
          </ul>
        </li>
        <li>
          When identifying issues, please be as fine-grained as possible. If a
          sentence contains multiple words that are independently mistranslated,
          separate errors should be recorded. For example, if “boathouses” is
          translated to “hangar de bateaux” in French, mark separate errors for
          “hangar” (should not be singular) and “de” (wrong preposition).
        </li>
        <li>
          If the same error occurs more than once in a particular document or
          even in a sentence, all occurrences should be reported. For
          consistency problems, assume that the <b>first</b> instance of a set
          of inconsistent entities is correct, and label any subsequent entities
          that are inconsistent with it as errors. For example, if a text
          contains “Doctor Smith … Dr. Smith … Doctor Smith … Dr. Smith”, the
          2nd and 4th occurrences should be marked.
        </li>
        <li>
          If the whole translation of a sentence is so bad that all or nearly
          all of it is completely wrong (e.g., word salad, completely
          nonsensical output, severely broken syntax), then apply the “Major”
          severity and pick the error type “Non-translation!”. Note that picking
          Non-translation! will automatically select the whole sentence as the
          error span, even if you had selected a subspan to start with.
        </li>
      </ol>
      `,
    'Navigation': `
      <h2>Navigation</h2>
      <p>
        Each task consists of a piece of source media (image or video) alongside
        its translation.
      </p>
      <ul>
        <li>
        When the source media is an image, it will be displayed immediately.
        When it is a video, you will need to click the play button to start
        watching the video. Use the video player's controls to pause/resume
        playback and seek forward or backward.
        </li>
        <li>
        You will go through the target text in convenient steps of small
        "sub-paragraphs" (groups of consecutive sentences from the same
        paragraph). You can move from one sub-paragraph to the next (and back)
        using the arrow keys or using the buttons labeled with left and right
        arrows. Note that these sub-paragraphs are automatically created
        purely for convenient navigation through potentially long
        documents.
        </li>
        <li>
        You can also use the left arrow key/button to go <i>back</i> through the
        sub-paragraphs and segments. You can also directly click on any
        previously read part of the text to jump back to it.
        </li>
      </ul>
      `,
    'Annotation Process': `
      <h2>Annotation Process</h2>
      <ol>
        <li>Review the translation of each segment against the source, following
            the general guidelines above.</li>
        <li>
          Select the <b>span</b> of words affected by the issue by clicking on
          the word/particle where the identified issue “begins”, then clicking
          on the word/particle where the issue “ends”. If it is only one word,
          then you have to click on it twice.
          <ul>
            <li>The marked span should be the minimal contiguous sequence such
                that modifying the word(s) within the span, deleting the span,
                or moving the word(s) somewhere else in the sentence will remove
                the identified issue. The span should not include adjoining
                words that are not directly affected by the identified issue and
                do not need to be modified in order for the issue to be
                fixed.</li>
            <li>You can only mark spans within sentences. In the rare case that
                an error straddles multiple sentences (e.g., when there is an
                incorrect sentence break), just mark the first part of the span
                that lies within a sentence.</li>
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
            <li>Note: all error spans must be marked on the target side. For
                Accuracy/Omission errors, select the space where the omitted
                content should have gone, and when prompted for a note,
                provide a brief description of the omitted content.</li>
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
            <b>subcategory</b> (also called <b>subtype</b>) of the error/issue
            found. For example: Accuracy &gt; Mistranslation.</li>
        <li>After annotating all identified issues in a sub-paragraph, use the
            <b>right arrow key</b> (or the <b>button</b>) to go to the next
            sub-paragraph.</li>
      </ol>`,
    'Severities defined': `
      <h2>Severities defined</h2>
      <p>We define error/issue severity levels in this section.</p>
      <ul>
        <li>
          <b>Major severity</b>:
          Major severity errors significantly alter the meaning of the source
          media, or significantly degrade the quality of the target text.
          <ul>
            <li>
              The translated text says something different from the intent of
              the source media, or is substantially difficult to understand, or
              has some very jarring linguistic flaw.
            </li>
            <li>
              Typically, accuracy and terminology errors fall here, as well as
              egregious style or grammar errors.
            </li>
            <li>
              The context of the document is sometimes the key in determining
              whether an error is major or minor. For example, changing the
              tense of a standalone sentence may be a minor error, but doing so
              in the middle of a narrative would be a major error.
            </li>
            <li>
              <details>
                <summary><b>Examples</b></summary>
                <table>
                  <tr>
                    <th>Language pair</th>
                    <th>Source</th>
                    <th>Translation</th>
                    <th>Comments</th>
                  </tr>
                  <tr>
                    <td>EN_DE</td>
                    <td>makes light of the horrors</td>
                    <td><span class="span-major">mache die Schrecken
                        leicht</span></td>
                    <td>"Make light of something" means "to treat something as
                        unimportant". The phrase was translated literally and
                        the translation makes no sense.</td>
                  </tr>
                  <tr>
                    <td>EN_DE</td>
                    <td>pro-Remain Tories</td>
                    <td><span class="span-major">Brexit-freundlichen</span>
                        Tories</td>
                    <td>Translated to the opposite meaning (pro-Remain vs.
                        pro-Brexit)</td>
                  </tr>
                  <tr>
                    <td>EN_DE</td>
                    <td>ACP</td>
                    <td><span class="span-major">ACP</span></td>
                    <td>The acronym "ACP" means nothing in German. It should
                        have been expanded to "Assistant Commissioner of Police"
                        and translated accordingly ("Stellvertretender
                        Polizeikommissar")</td>
                  </tr>
                  <tr>
                    <td>ZH_EN</td>
                    <td>但是这个过程急不来，所以</td>
                    <td>But this process <span class="span-major">is not
                        anxious</span></td>
                    <td>The meaning of the source is "This process cannot be
                        rushed". The translation makes no sense. </td>
                  </tr>
                  <tr>
                    <td>ZH_EN</td>
                    <td>清华迎来110岁生日，校长邱勇：大学之大在于培养大写的人</td>
                    <td>Tsinghua celebrates its 110th birthday, President Qiu
                        Yong: The greatness of a university lies in cultivating
                        <span class="span-major">capitalized people</span></td>
                    <td>"Capitalized people" makes no sense. It should be
                        "upstanding people". </td>
                  </tr>
                  <tr>
                    <td>EN_ZH</td>
                    <td>unmute yourself please</td>
                    <td>请你<span class="span-major">闭麦</span></td>
                    <td>Mistranslated as "mute yourself". </td>
                  </tr>
                  <tr>
                    <td>ZH_EN</td>
                    <td>
                      帕梅拉的意识，待在贝塔的灵魂世界，透过他的双眼，看到了眼前的景像。
                    </td>
                    <td>
                      Pamela's consciousness stayed in Beta's spiritual world.
                      She saw the scene in front of
                      <span class="span-major">her</span> through his eyes.
                    </td>
                    <td>
                      This  should read "him". The meaning is substantially
                      altered, since the source text means that "she sees the
                      scene that he is seeing".
                    </td>
                  </tr>
                  <tr>
                    <td>DE_EN</td>
                    <td>
                      Koch-Ottes Haare sind kurz, streng gescheitelt und
                      gekämmt, der Blick energisch.
                    </td>
                    <td>
                      Koch-Otte's hair is short, severely parted and combed;
                      <span class="span-major">his</span> look spirited.
                    </td>
                    <td>
                      The choice of pronoun is incorrect for Benita Koch-Ottes,
                      a German female textile designer.
                    </td>
                  </tr>
                </table>
              </details>
            </li>
          </ul>
        </li>
        <li>
          <b>Minor severity</b>:
          Minor severity errors are noticeable but minor flaws in the translated
          text. They do not significantly alter the meaning of the source media,
          and they do not significantly degrade the quality of the target text.
          <ul>
            <li>
              Minor severity errors might add, drop, or modify minor details, or
              they may slightly decrease the stylistic quality of the text.
            </li>
            <li>
              Typically, the kinds of errors that fall under this severity level
              are grammar, spelling (including capitalization and whitespace),
              style, punctuation, locale convention, and creative
              reinterpretation.
            </li>
            <li>
              As mentioned earlier, the context of the document is sometimes the
              key in determining whether an error is major or minor.
            </li>
            <li>
              <details>
                <summary><b>Examples</b></summary>
                <table>
                  <tr>
                    <th>Language pair</th>
                    <th>Source</th>
                    <th>Translation</th>
                    <th>Comments</th>
                  </tr>
                  <tr>
                    <td>EN_DE</td>
                    <td>
                      When cooking for a crowd, Eunsook Pai sears the dumplings
                      a couple of hours in advance and then steams them just
                      before serving.
                    </td>
                    <td>
                      Beim Kochen für eine
                      <span class="span-minor">Menschenmenge</span> braten
                      Eunsook Pai die Teigtaschen ein paar Stunden im Voraus an
                      und dampft sie dann kurz vor dem Servieren.
                    </td>
                    <td>
                      A minor word choice error (a contextually incorrect
                      expression but still understandable).
                    </td>
                  </tr>
                  <tr>
                    <td>EN_DE</td>
                    <td>Heat another tablespoon vegetable oil, and saute onion,
                        stirring occasionally, until just softened, 2 to 3
                        minutes.</td>
                    <td>
                      Erhitzen Sie einen weiteren Esslöffel Pflanzenöl und
                      braten Sie die Zwiebel unter gelegentlichem Rühren 2 bis 3
                      Minuten an, bis sie
                      <span class="span-minor">gerade weich</span> ist.
                    </td>
                    <td>
                      Literal, unidiomatic, but still understandable. The
                      expression "just softened" means "at the point of becoming
                      soft."
                    </td>
                  </tr>
                  <tr>
                    <td>EN_DE</td>
                    <td>In an exclusive interview with Fast Company, Berners-Lee
                        joked that the intent behind Inrupt is
                        "world domination."</td>
                    <td>In einem exklusiven Interview mit Fast Company scherzte
                        Berners-Lee, dass die Absicht hinter Inrupt die
                        <span class="span-minor">"</span>Weltherrschaft<span
                        class="span-minor">"</span> sei.</td>
                    <td>German uses curly quotes, opening down, closing up. The
                        translation uses the same quote style as the source
                        English, so both of these wrong quotes are minor
                        punctuation errors.</td>
                  </tr>
                  <tr>
                    <td>EN_DE</td>
                    <td>
                      The limited-time Taste of Knott's food, beer and wine
                      event runs through Sept. 13 without rides, coasters or
                      other theme park attractions.
                    </td>
                    <td>
                      Die zeitlich begrenzte Veranstaltung
                      <span class="span-minor">Taste of Knotts Essen, Bier und
                      Wein</span> läuft bis zum 13. September ohne
                      Fahrgeschäfte, Achterbahnen oder andere Attraktionen des
                      Themenparks.
                    </td>
                    <td>Awkward syntax.</td>
                  </tr>
                  <tr>
                    <td>ZH_EN</td>
                    <td>
                      因为其中大多数都被抛弃了，希望包含在花粉中的生殖细胞，雄性生殖细胞
                    </td>
                    <td>
                      Because most of them are released, the hope is that there
                      is a <span class="span-minor">reproduction</span> cell
                      contained in the pollen.
                    </td>
                    <td>A minor terminological error. It should read
                        "reproductive cell."</td>
                  </tr>
                  <tr>
                    <td>EN_ZH</td>
                    <td>
                      This is also explained on the flip side of the widget.
                    </td>
                    <td>
                      这在该<span class="span-minor">插件</span>的<!--
                      --><span class="span-minor">另一方面</span>也有解释。
                    </td>
                    <td>Inaccurate translation of "widget" and "flip side".
                        Widget usually refers to a small gadget or mechanical
                        device, especially one whose name is unknown or
                        unspecified. "flip side" means the opposite side.</td>
                  </tr>
                </table>
              </details>
            </li>
          </ul>
        </li>
      </ul>`,
    'Error Types and Subtypes defined': `
      <h2>Error Types and Subtypes defined</h2>
      <ul>
        <li>
          <b>Accuracy</b>.
          The translated text does not accurately reflect the source media.
          <details>
            <summary>Subtypes of Accuracy:</summary>
            <ul>
              <li>
                <b>Creative Reinterpretation</b>.
                The translated text <i>reinterprets the source, but preserves
                its intent</i>. Note that if the translation reinterprets the
                source text to such a great degree that it <i>changes</i> the
                intent, then it should be marked as using Mistranslation,
                Addition, or Omission subtypes, as appropriate.
                <ul>
                  <li>
                    Mark Creative Reinterpretation if the translation includes
                    additional text or omits some content that provides
                    explanations or context that may be obvious in the source
                    (target) locale, but not in the target (source) locale. For
                    example, an added short introduction of an entity not well
                    known in the target locale, or an omitted introduction of an
                    entity well known in the target locale.
                  </li>
                  <li>
                    The translation edits the source creatively, perhaps to make
                    the translated text more fluent or expressive or localized.
                    For example, a reordering of sentences or an addition of
                    text that is supported by the overall passage being
                    translated and does not change the intent of the source
                    text. Another example would be that of the name of an
                    exemplifying entity being changed to a better local
                    equivalent.
                  </li>
                </ul>
              </li>
              <li>
                <b>Mistranslation</b>.
                The translated text does not accurately represent the source
                media. Examples: (1) The source text talks about a person A
                knowing another person B, but the English translation says
                "A was intimate with B." (2) The source text states that
                something never happens, whereas the translation says it happens
                "often" or "rarely."
              </li>
              <li>
                <b>Gender Mismatch</b>.
                Given the context, the gender is incorrect (incorrect pronouns,
                noun/adjective endings, etc). Misgendering errors typically have
                a major severity as they significantly alter the meaning of the
                source text. If the correct gender is not clear from the source
                context, assume the first instance of gender is correct and mark
                subsequent gender inconsistencies as errors. Examples: (1) "Mary
                slammed the door as he left." (2) Given the source text "My
                friend is an engineer. She is great at coding.", if its
                translation used a male form for the term "engineer", it would
                be incorrect, as we know the subject is female from the source
                context.
              </li>
              <li>
                <b>Source language fragment</b>.
                Content that should have been translated has been left
                untranslated. Example: A word, phrase or sentence in German
                has been copied verbatim into the English translation,
                but a natural translation is possible. Note that if only a short
                phrase or term is left untranslated, and that phrase or term
                is commonly used in the target language (for example, the French
                phrase "c'est la vie" used in English), then it's not an
                error (or is, at most, a minor error).
              </li>
              <li>
                <b>Addition</b>.
                The translated text includes information not present in the
                source. Example: A translated sentence that includes adverbs or
                adjectives without equivalents in the source media, even after
                taking context into account.
              </li>
              <li>
                <b>Omission</b>.
                Content is missing from the translation that is present in the
                source. Example: A phrase has been dropped from the translation
                and it is not already implied by context. This error type needs
                to be annotated where the missing content should be. Prefer
                marking the error on a space if possible, otheriwise mark the
                token that the missing content should appear before. Please
                provide a short description of what content was omitted when
                prompted. <b>Important:</b> Do not mark omissions from the
                end/bottom of the source media. See "Source Media Guidelines"
                for more details.
              </li>
            </ul>
          </details>
        </li>
        <li>
          <b>Fluency</b>.
          Issues related to the form or content of translated text, independent
          of its relation to the source media; errors in the translated text
          that make it harder to understand.
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
                <b>Text-Breaking</b>.
                Issues related to paragraph breaks and line breaks.
                If a sentence ends with an incorrect or missing paragraph break
                or line break, then mark the last part of it (word or
                punctuation) with this error type. Examples: (1) There should be
                a paragraph break but there is only a line break or there is not
                even a line break. (2) There should not be any break but a
                paragraph break is present in the middle of a sentence.
                <br><br>
                Certain paragraph breaks are very
                important for establishing the proper flow of the text: for
                example, before and after a section heading or a block-quote.
                If an important paragraph break is completely missing (there is
                not even a line break), then that is a major error, as it
                severely degrades the quality of the text. If an unwarranted
                paragraph break is seen in the middle of a sentence, that is
                also a major error. Most other errors of type
                "Fluency / Text-Breaking" are usually minor errors.
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
              <li>
                <b>Archaic or obscure word choice</b>.
                The text contains archaic or obscure words that an average
                speaker of the target language may find hard to understand. This
                includes cases where obscure translations of terms are used,
                where instead transliterating or copying the source language
                would be more natural. (Archaic/formal terms may be okay in
                certain contexts such as government or historical documents,
                academic papers, etc.) Examples: (1) "He was transparently evil"
                gets translated to "वह पारदर्शी रूप से दुष्ट था।" in Hindi which uses an
                obscure term for "transparently" instead of the more common
                "स्पष्ट". (2) The term "computer" is translated into Hindi as
                "संगणक" (saṅgaṇaka) but the transliterated version "कंप्यूटर"
                (kampyootar) is much more common and easily understood. (3) In
                the sentence "Last night we went to a bar and drank a homerkin
                of beer", the term 'homerkin' (meaning 'several gallons') is
                unnecessarily obscure. It would be much more colloquial to say
                'we drank a ton of beer', or similar.
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
      </ul>`,
    'Annotations exemplified in detail': `
      <details>
        <summary>
          <span class="summary-heading">Annotations exemplified in detail</span>
        </summary>
        <table>
          <tr>
            <th>Language pair</th>
            <th>Source</th>
            <th>Translation</th>
            <th>Correct annotations</th>
            <th>Comments</th>
          </tr>
          <tr>
            <td>DE_EN</td>
            <td>... das in Hannover oder Cuxhaven auf den Teller kommt.</td>
            <td>... that ends up on plates in Hannover or
                <span class="span-minor">Hamburg</span>.</td>
            <td>Accuracy - Creative Reinterpretation - Minor</td>
            <td>This changes the name of a city that is purely used as an
                example, from Cuxhaven (not a widely known name outside
                Germany) to Hamburg (a widely known metropolitan area that
                Cuxhaven belongs to). Note that if, instead of using
                Cuxhaven as an example to make a larger point, the
                source text were talking about some specific issues
                related to that city, then that would have been an Accuracy
                error.</td>
          </tr>
          <tr>
            <td>DE_EN</td>
            <td>... das in Hannover oder Cuxhaven auf den Teller kommt.</td>
            <td>... that ends up on plates in Hannover or Cuxhaven
                <span class="span-minor">(a part of Hamburg)</span>.</td>
            <td>Accuracy - Creative Reinterpretation - Minor</td>
            <td>Here, the translator added some explanatory text to make
                the text more comprehensible in a non-German setting.</td>
          </tr>
          <tr>
            <td>EN_DE</td>
            <td>[Preceding text about COVID issues in restaurants.]
                Parties are another situation when ...</td>
            <td><span class="span-minor">Ähnlich wie im Restaurant</span> ist es
                auch auf Partys wahrscheinlicher, dass ...</td>
            <td>Accuracy - Creative Reinterpretation - Minor</td>
            <td>Here, the translator took the liberty to add some introductory
                text to a sentence, that is not present in the source, but is
                supported by the whole passage getting translated, in order to
                make the translation more natural.</td>
          </tr>
          <tr>
            <td>EN_DE</td>
            <td>Still Bombay starts with frames
                of Bandra's yesteryear villas, but swiftly moves on to consider
                the rest of the city, the vignettes accompanied by short essays
                and musings written by Tekchandaney.</td>
            <td><span class="span-major">Bombay</span> beginnt immer noch mit Rahmen von Bandras alten Villen,
                geht aber schnell weiter, um den Rest der Stadt zu betrachten,
                die Vignetten, begleitet von kurzen Essays und Überlegungen von
                Tekchandaney.</td>
            <td>Accuracy - Omission - Major</td>
            <td>"Still Bombay" is the name of the book; the word "Still" was
                omitted in the translation. The severity in this case must be
                major. As the error is an omission, but there was no whitespace
                to select where the omitted content should have gone, the first
                token after the omission is selected.
                </td>
          </tr>
          <tr>
            <td>EN_DE</td>
            <td>This is the one of a kind Pent House 1BR apartment with a
                wrap-around Terrace.</td>
            <td>Dies ist das einzigartige <span class="span-minor">Pent
                House</span> <span class="span-major">1BR</span> Apartment mit
                einer umlaufenden Terrasse.</td>
            <td>
              1. Fluency - Spelling - Minor ("Pent House")
              <br>
              2. Accuracy - Source language fragment - Major ("1BR")
            </td>
            <td>
              1. "Pent House" should be spelled as "Penthouse" in German; this
                is a minor spelling issue
              <br>
              2. The abbreviation "1BR" means "one bedroom" in English and is
                not used in German. This is a major error because the
                translation is not understandable.
            </td>
          </tr>
          <tr>
            <td>ZH_EN</td>
            <td>
              本次拍卖由衢州市柯城区人民法院作为处置单位，在阿里拍卖平台进行公开拍卖。
            </td>
            <td>In this auction, the People's Court of Kecheng District, Quzhou
                City was the <span class="span-major">disposal unit</span>, and
                the public auction was conducted on the Ali auction
                platform.</td>
            <td>Accuracy - Mistranslation - Major</td>
            <td>The meaning of the translated text is completely altered since
                it should read "the department responsible for this auction". In
                this case, the error should be rated as major.</td>
          </tr>
          <tr>
            <td>ZH_EN</td>
            <td>含少量杂质和星点的黄铁矿，相比之下就非常有收藏价值了。</td>
            <td>Pyrite with a small amount of impurities and
                <span class="span-major">star points</span> is very valuable in
                comparison.</td>
            <td>Accuracy - Mistranslation - Major</td>
            <td>The expression "star points" does not make sense here, so it
                should be rated as a major error.</td>
          </tr>
          <tr>
            <td>EN_DE</td>
            <td>COAI had cautioned that any decision on delicensing or
                administrative allocation of high commercial value E and V bands
                spectrum would be against prevailing policy framework.</td>
            <td>COAI hatte davor gewarnt, dass jede Entscheidung über die
                Delizenzierung oder administrative
                <span class="span-minor">Zuweisung</span> von Frequenzen der
                E- und V-Bänder mit hohem kommerziellen Wert gegen den
                geltenden politischen Rahmen verstoßen würde.</td>
            <td>Accuracy - Mistranslation - Minor</td>
            <td>The term "Zuweisung" is used when the subject is a person, so it
                should be replaced by "Zuteilung" or "Verteilung." However, this
                error should be marked as minor since the meaning is not
                severely affected.</td>
          </tr>
          <tr>
            <td>EN_FR</td>
            <td>Opinion
                <p>I believe the honourable Prime Minister made a serious error.
                We should never have negotiated under the present circumstances.
                </p>
            <td><span class="span-major">Avis</span>
                Je crois que l'honorable premier ministre a commis une grave
                erreur.<br><span class="span-minor">Nous</span><br>n’aurions
                jamais dû négocier dans les
                circonstances actuelles.</td>
            <td>
              1. Fluency - Text-Breaking - Major <br>
              2. Fluency - Text-Breaking - Minor
            </td>
            <td>The translation is missing the important paragraph break needed
                between the section heading and the section body. The
                second error is an extraneous line-break in the middle of a
                sentence, which is a minor error.</td>
          </tr>
          <tr>
            <td>ZH_EN</td>
            <td>
              4月25日，北京春风和煦，清华园内彩旗招展，庆祝清华大学建校110<!--
              -->周年大会在新清华学堂举行。
            </td>
            <td>On April 25, the spring breeze in Beijing was warm, and the
                colorful flags in the Tsinghua
                <span class="span-minor">Park</span> were on display to
                celebrate the 110th anniversary of the founding of Tsinghua
                University in the New Tsinghua Academy.</td>
            <td>Accuracy - Mistranslation - Minor</td>
            <td>
              In this example, the correct expression is "Tsinghua Garden." This
              error should be rated as a minor accuracy mistake because it does
              not represent a severe alteration of the meaning.
            </td>
          </tr>
          <tr>
            <td>ZH_EN</td>
            <td>
              在盛唐诗人中，王维、孟浩然长于五绝，王昌龄等七绝写得很好，<!--
              -->兼长五绝与七绝而且同臻极境的，只有李白一人。
            </td>
            <td>Among the poets in the prosperous Tang Dynasty, Wang Wei and
                Meng Haoran
                <span class="span-major">were better than Wujue</span>,
                and Wang Changling and <span class="span-minor">other</span>
                <span class="span-major">Qijue</span> wrote very well. Only Li
                Bai is the <span class="span-minor">only</span> person who has
                both the Wujue and the <span class="span-major">Qijue</span>
                and <span class="span-minor">is at the same level</span>.</td>
            <td>
              1. Accuracy - Mistranslation - Major.
              <br>
              2. Fluency - Grammar - Minor
              <br>
              3. Accuracy - Mistranslation - Major
              <br>
              4. Fluency - Grammar - Minor
              <br>
              5. Accuracy - Mistranslation - Major
              <br>
              6. Accuracy - Omission - Minor
            </td>
            <td>
              1. Wujue, or Jueju, is a genre of poetry. The source means that
                Wang Wei and Meng Haoran were better at Wujue, and Wang
                Changling and others wrote Qijue very well.
              <br>
              2. The highlighted "other" should be "others".
              <br>
              3. Same as 1.
              <br>
              4. The text "is the only person who" contains a repetition of the
                word "only" which is already present at the beginning of the
                sentence.
              <br>
              5. Li Bai is the only one who was good at both. The translation
                considers Wujue and Qijue as persons instead of poetry genres.
              <br>
              6. The original text means the poet wrote both genres incredibly
                well. The translated text "is at the same level" missed the
                information that the poet is very good at the genres.</td>
          </tr>
        </table>
      </details>
      <br>`,
    'Source Media Guidelines': `
      <h2>Source Media Guidelines</h2>
      <p>
        Most of the rest of the instructions assume that it is clear what is
        meant to be translated. However, that may not always be the case. This
        section provides guidance specific to the fact that the source is not
        plain text.
      </p>
      <p>
        When the source is a video, the translation should be of the speech in
        the video. When it is an image, the translation should be of the
        "main" visible text: generally the image will be of a social media
        thread/conversation, and only the text content of the posts should be
        translated (i.e. not the text of buttons, advertisements, et cetera).
      </p>
      <p>
        For videos, if you are unsure of the accuracy of any portion of the
        translation because the audio was not understandable, do not mark an
        accuracy error for that portion of the translation.
      </p>
      <p>
        You should apply a very permissive standard for the formatting of the
        translation: for example, there is no standard convention for where to
        introduce paragraph breaks, so do not mark errors related to this unless
        the translation is obviously wrong or difficult to understand.
      </p>
      <p>
        Occasionally, the source media may include more content than was
        intended for translation. In these cases, some speech from the end of
        the video or text from the bottom of the image may appear to have been
        omitted in the translation. Assume that this was intentinoal, and do not
        mark these cases as Accuracy/Omission errors. Omissions should only be
        marked in cases where the omitted content has translated content both
        before and after it.
      </p>
      `,
  },
  instructions_section_order: [
    '_style', 'Overview', 'Source Media Guidelines', 'General Guidelines',
    'Navigation', 'Annotation Process', 'Annotation Tips', 'Severities defined',
    'Error Types and Subtypes defined', 'Annotations exemplified in detail',
    'Style &amp; Convention Guidelines', 'Feedback'
  ],
};
