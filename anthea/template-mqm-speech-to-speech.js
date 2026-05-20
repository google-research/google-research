// Copyright 2026 The Google Research Authors.
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
 * This is an MQM template for evaluating Speech-to-Speech translation.
 * It provides play buttons aligned to text segments.
 *
 * To function, audio URLs and alignment info must be provided via annotations
 * (the last column in the input TSV) for each segment. An example annotation
 * is shown below:
 * {"audio":
 *   {
 *     "source": {
 *       "url": "https://url/to/source/audio.wav",
 *       "alignment": [{"substring": "Word1", "time_start": 0.01234567,
 *                      "time_end": 0.12345678}, {"substring": "Word2",
 *                      "time_start": 0.15, "time_end": 0.20}],
 *     },
 *     "target": {
 *       "url": "https://url/to/target/audio.wav",
 *       "alignment": [{"substring": "Palabra1", "time_start": 0.01234567,
 *                      "time_end": 0.22222222}, {"substring": "Palabra2",
 *                      "time_start": 0.3, "time_end": 0.6}],
 *     }
 *   }
 * }
 */
antheaTemplates['MQM-Speech-to-Speech'] = {
  severities: {
    major: {
      display: 'Major severity',
      shortcut: 'M',
      color: '#fca5a5',
      description: 'Major severity errors significantly alter the meaning ' +
                   'of the source speech, or significantly degrade the quality ' +
                   'of the speech.',
    },
    minor: {
      display: 'Minor severity',
      shortcut: 'm',
      color: '#fef08a',
      description: 'Minor severity errors are noticeable but minor flaws in the ' +
                   'translated speech. They do not significantly alter the ' +
                   'meaning of the source speech, and they do not ' +
                   'significantly degrade the quality of the speech.',
    },
  },

  /**
   * @const {string} Template version identifier.
   */
  VERSION: 'v1.00-Apr-10-2026',

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
   *    more than 10 error type/subtype combinations).
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
      description: 'The translation does not accurately reflect the source speech.',
      subtypes: {
        reinterpretation: {
          display: 'Creative Reinterpretation',
          description: 'The translation reinterprets the source, but preserves its intent within its broader context (the document and its purpose, and the target locale). This can be because the translation adds, removes, or modifies content in accordance with the target locale, or simply makes creative changes that fully preserve the intent of the source.',
        },
        mistranslation: {
          display: 'Mistranslation',
          description: 'The translation does not accurately represent the source speech.',
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
          description: 'The translation includes information not present in the source.',
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
      description: 'Issues related to the form or content of the translation, independent of its relation to the source; errors in the translation that prevent it from being understood clearly.',
      subtypes: {
        inconsistency: {
          display: 'Inconsistency',
          description: 'The translation shows internal inconsistency (not related to terminology).',
        },
        grammar: {
          display: 'Grammar',
          description: 'Issues related to the grammar or syntax of the translation, other than spelling and orthography.',
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
        character_encoding: {
          display: 'Character encoding',
          description: 'Characters are garbled due to incorrect application of an encoding.',
        },
      },
    },
    style: {
      display: 'Style',
      description: 'The translation has stylistic problems.',
      subtypes: {
        awkward: {
          display: 'Unnatural or awkward',
          description: 'The translation is literal, awkward, unidiomatic, or inappropriate in the context.',
        },
        sentence_structure: {
          display: 'Bad sentence structure',
          description: 'The marked span is an unnecessary repetition, or makes the sentence unnecessarily long, or would have been better as a clause in the previous sentence.'
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
          description: 'Terminology is used in an inconsistent manner within the translation.',
        },
      },
    },
    locale_convention: {
      display: 'Locale convention',
      description: 'The translation does not adhere to locale-specific conventions and violates requirements for the presentation of content in the target locale.',
      subtypes: {
        address: {
          display: 'Address format',
          description: 'Content uses the wrong format for addresses.',
        },
        date: {
          display: 'Date format',
          description: 'A date format inappropriate for the locale is used.',
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
    voice_inconsistency: {
      display: 'Voice Inconsistency',
      description: 'The target speaker voice is inconsistent with itself or with the source speaker voice.',
      subtypes: {
        voice_drifting: {
          display: 'Voice Drifting',
          description: 'The target speaker voice appears to change (e.g. in pitch or accent), in a way that the source speaker voice does not.',
        },
        voice_mismatch: {
          display: 'Source-Target Voice Mismatch',
          description: 'The target speaker voice is different from the source speaker voice.',
          auto_expand_span: true,
        },
      }
    },

    prosody: {
      display: 'Prosody',
      description: 'Issues related to the rhythm, intonation, volume, and emotional tone of the target audio.',
      subtypes: {
        affective_mismatch: {
          display: 'Affective Mismatch',
          description: 'Overall emotional tone (happy, sad, excited, etc.) does not match the source.',
        },
        awkward_pauses: {
          display: 'Awkward Pauses',
          description: 'Unnatural or awkward silences between words or sentences.',
        },
        intonation: {
          display: 'Intonation',
          description: 'Pitch change is inappropriate for the sentence type in the target locale (e.g. in English, a non-rising pitch for a question).',
        },
        volume_mismatch: {
          display: 'Volume Mismatch',
          description: 'The volume of the target voice does not match the volume of the source voice.',
        },
        robotic_cadence: {
          display: 'Robotic Cadence',
          description: 'Evenly spaced, monotonous timing across the segment.',
          auto_expand_span: true,
        },
      },
    },
    pronunciation: {
      display: 'Pronunciation',
      description: 'Issues related to the pronunciation of words in the target audio.',
      subtypes: {
        emphasis_error: {
          display: 'Emphasis',
          description: 'Stress or emphasis is placed on the wrong syllable.',
        },
        mispronunciation: {
          display: 'Mispronunciation',
          description: 'Words are pronounced incorrectly (prefer more specific sub-category if applicable).',
        },
      },
    },
    audio_naturalness: {
      display: 'Audio Naturalness',
      description: 'Issues related to how (un)natural the target audio sounds.',
      subtypes: {
        choppiness: {
          display: 'Choppiness',
          description: 'Audio sounds stitched together; words or sounds are cut off abruptly.',
        },
        echo_repetition: {
          display: 'Echo or Repetition',
          description: 'A word, phrase, or sound is unnaturally repeated.',
        },
        audio_artifacts: {
          display: 'Audio Artifacts',
          description: 'Static, high-pitched noises, or non-speech hums.',
        },
        breath_irregularity: {
          display: 'Breath Irregularity',
          description: 'Unnatural breathing sounds (e.g. gasping).',
        },
        metallic_thin: {
          display: 'Metallic or Thin',
          description: 'The overall voice sounds "tinny" or artificial.',
        },
      },
    },
    background_staining: {
      display: 'Background Staining',
      description: 'Background noise from the source is present in the target audio.',
      auto_expand_span: true,
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
   * Add instructions about audio playback and new error categories.
   */
  instructions_section_contents: {
    '_style': `
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
          background-color: #fca5a5;
        }
        .span-minor {
          background-color: #fef08a;
        }
      </style>
      `,
    'Overview': `
      <h2>Overview</h2>
      <p>
        In this project, you will evaluate speech-to-speech translations.
        For each segment, you can listen to the source or target audio by
        clicking the Play button next to the text (or pressing <code>Spacebar</code>).
      </p>
      <p>
        Please listen to the audio while reading the transcript to evaluate
        the translation quality. You will mark both standard translation errors
        and speech-specific quality issues.
      </p>
      <p>
        <b>Important:</b> The primary object of evaluation is the <b>audio</b>. The text transcript is provided to facilitate annotation. The transcript may contain automatic transcription errors (e.g., a word in the text that sounds similar to what was actually spoken). In these cases, you should only mark an error if the <b>audio</b> itself contains an error. Text-only error types (e.g. character encoding) should be marked based on the text as normal.
      </p>
      `,
    'General Guidelines': `
      <h2>General Guidelines</h2>
      <p>
        The standard you should be reviewing the translations against is
        <b>human speech-to-speech translation quality</b>. Report every occurrence where the
        translation falls short of that standard. Remember that the content you
        are reviewing may be machine-generated.
        <b><i>Apply the same standard regardless</i></b>. The translation should
        be:
      </p>
      <ul>
        <li>Linguistically correct</li>
        <li>Accurate</li>
        <li>Intelligible and natural-sounding (correct pronunciation, appropriate prosody, and good voice quality)</li>
        <li>With terminology appropriate in the context</li>
        <li>Consistent</li>
        <li>Faithful in tone and register to the source speaker</li>
        <li>Appropriately transformed for the target context.</li>
      </ul>
      <p>
        Please be mindful of the following:
      </p>
      <ol>
        <li>
          Before you start annotating, please read carefully the definitions
          for severities and error/issue types.
        </li>
        <li>
          Take both the audio and the transcript into account when annotating.
        </li>
        <li>
          If the whole translation of a segment is so bad that all or nearly
          all of it is completely wrong (e.g., completely nonsensical output, severely broken syntax, or unintelligible audio), then apply the “Major”
          severity and pick the error type “Non-translation!”.
        </li>
      </ol>
      `,
    'Navigation': `
      <h2>Navigation</h2>
      <p>
        Each task consists of audio and transcript from a single document alongside its translation.
      </p>
      <ul>
        <li>
          You will go through a document in steps of segments. You can move from one segment to the next (and back) using the arrow keys or using the buttons labeled with left and right arrows.
        </li>
        <li>
          <b>Audio Playback:</b> You must listen to the source and target audio completely before moving to the next segment. The play button will turn <span style="color: green; font-weight: bold;">green</span> when fully played.
        </li>
        <li>
          Press <code>Spacebar</code> or click the play button to <b>pause</b> and <b>resume</b> audio. Paused audio remembers its position.
        </li>
        <li>
          You can use the <code>Tab</code> key to jump between the source side and the translation side. Switching sides with <code>Tab</code> will not affect the other side's playback position or highlighting.
        </li>
        <li>
          <b>Synchronous Highlighting:</b> As the audio plays, the corresponding text tokens will be highlighted.
        </li>
        <li>
          You can also directly click on any previously read/listened word in the text to resume playback from that point.
        </li>
      </ul>
      `,
    'Annotation Process': `
      <h2>Annotation Process</h2>
      <ol>
        <li>Review the translation of each segment against the source by listening to the audio and reading the transcript, following the general guidelines above.</li>
        <li>
          Select the <b>span</b> of words affected by the issue.
          <ul>
            <li>For most errors, click on the word where the issue begins, then on the word where it ends. If it's only one word, click on it twice.</li>
            <li>The marked span should be the minimal contiguous sequence affected by the issue.</li>
            <li><b>Full-Segment Errors:</b> For error categories labeled <b>(Full-Segment)</b> (e.g., Source-Target Voice Mismatch, Background Staining, Robotic Cadence), the span will automatically expand to cover the full text block regardless of which word you select. Visually, these are shown as a colored border around the text block. Mark each full-segment error only <b>once per segment</b>.</li>
            <li>When the error is an omission, the error span must be selected on the source side.</li>
          </ul>
        </li>
        <li>
          Select the <b>severity</b> of the issue (Major or Minor).
        </li>
        <li>Select the <b>category</b> and <b>subcategory</b> of the error/issue found.</li>
        <li><b>Play from Here:</b> When you click on a word to start marking an error span, a small <b>▶</b> button labeled "Play from here" will appear above the clicked word. Clicking it will begin audio playback from that word's position.</li>
        <li><b>Error Audio:</b> When you mark an error, a mini play button will appear next to the error description on the right, allowing you to replay just the audio corresponding to the marked span.</li>
        <li>After annotating all identified issues in a segment and ensuring audio has been fully played, use the <b>right arrow key</b> (or the <b>button</b>) to go to the next segment.</li>
      </ol>
      `,
    'Speech Error Categories': `
      <h2>Speech Error Categories</h2>
      <p>
        In addition to the standard translation errors (Accuracy, Fluency,
        Style, etc.), this template introduces several categories specific to
        speech evaluation:
      </p>
      <ul>
        <li>
          <b>Voice Inconsistency.</b> The target speaker voice is inconsistent with itself or with the source speaker voice.
          <details>
            <summary>Subtypes of Voice Inconsistency:</summary>
            <ul>
              <li><b>Voice Drifting.</b> The target speaker voice appears to change (e.g. in pitch or accent), in a way that the source speaker voice does not. Please mark the location where the voice change begins.</li>
              <li><b>Source-Target Voice Mismatch</b> <i>(Full-Segment)</i>. The target speaker voice is consistent throughout the target audio but is different from the source speaker voice. If the target speaker voice <b>changes</b> during the segment, select Voice Drifting instead.</li>
            </ul>
          </details>
        </li>
        <li>
          <b>Prosody.</b> Issues related to the rhythm, intonation, volume, and emotional tone of the target audio.
          <details>
            <summary>Subtypes of Prosody:</summary>
            <ul>
              <li><b>Affective Mismatch.</b> Overall emotional tone (happy, sad, excited, etc.) does not match the source.</li>
              <li><b>Awkward Pauses.</b> Unnatural or awkward silences between words or sentences.</li>
              <li><b>Intonation.</b> Pitch change is inappropriate for the sentence type in the target locale (e.g. in English, a non-rising pitch for a question).</li>
              <li><b>Volume Mismatch.</b> The volume of the target voice does not match the volume of the source voice.</li>
              <li><b>Robotic Cadence</b> <i>(Full-Segment)</i>. Evenly spaced, monotonous timing across the segment.</li>
            </ul>
          </details>
        </li>
        <li>
          <b>Pronunciation.</b> Issues related to the pronunciation of words in the target audio.
          <details>
            <summary>Subtypes of Pronunciation:</summary>
            <ul>
              <li><b>Emphasis.</b> Stress or emphasis is placed on the wrong syllable.</li>
              <li><b>Mispronunciation.</b> Words are pronounced incorrectly (prefer more specific sub-category if applicable).</li>
            </ul>
          </details>
        </li>
        <li>
          <b>Audio Naturalness.</b> Issues related to how (un)natural the target audio sounds.
          <details>
            <summary>Subtypes of Audio Naturalness:</summary>
            <ul>
              <li><b>Choppiness.</b> Audio sounds stitched together; words or sounds are cut off abruptly.</li>
              <li><b>Echo or Repetition.</b> A word, phrase, or sound is unnaturally repeated.</li>
              <li><b>Audio Artifacts.</b> Static, high-pitched noises, or non-speech hums.</li>
              <li><b>Breath Irregularity.</b> Unnatural breathing sounds (e.g. gasping).</li>
              <li><b>Metallic or Thin.</b> The overall voice sounds "tinny" or artificial.</li>
            </ul>
          </details>
        </li>
        <li>
          <b>Background Staining</b> <i>(Full-Segment)</i>. Background noise from the source is present in the target audio.
        </li>
      </ul>
      `,
    'Severities defined': `
      <h2>Severities defined</h2>
      <p>We define error/issue severity levels in this section.</p>
      <ul>
        <li>
          <b>Major severity</b>:
          Major severity errors significantly alter the meaning of the source
          speech, or significantly degrade the quality of the speech.
          <ul>
            <li>
              The translation says something different from the intent of
              the source, or is substantially difficult to understand, or
              has some very jarring linguistic or audio quality flaw.
            </li>
            <li>
              Typically, accuracy and terminology errors fall here, as well as
              egregious style, grammar, or audio quality errors.
            </li>
            <li>
              The context of the document is sometimes the key in determining
              whether an error is major or minor. For example, changing the
              tense of a standalone sentence may be a minor error, but doing so
              in the middle of a narrative would be a major error.
            </li>
          </ul>
        </li>
        <li>
          <b>Minor severity</b>:
          Minor severity errors are noticeable but minor flaws in the
          translation. They do not significantly alter the meaning of the
          source speech, and they do not significantly degrade the quality of
          the speech.
          <ul>
            <li>
              Minor severity errors might add, drop, or modify minor details, or
              they may slightly decrease the stylistic quality of the
              translation.
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
          </ul>
        </li>
      </ul>`,
    'Error Types and Subtypes defined': `
      <h2>Error Types and Subtypes defined</h2>
      <ul>
        <li>
          <b>Accuracy</b>.
          The translation does not accurately reflect the source speech.
          <details>
            <summary>Subtypes of Accuracy:</summary>
            <ul>
              <li>
                <b>Creative Reinterpretation</b>.
                The translation <i>reinterprets the source, but preserves
                its intent</i>. Note that if the translation reinterprets the
                source to such a great degree that it <i>changes</i> the
                intent, then it should be marked as using Mistranslation,
                Addition, or Omission subtypes, as appropriate.
                <ul>
                  <li>
                    Mark Creative Reinterpretation if the translation includes
                    additional content or omits some content that provides
                    explanations or context that may be obvious in the source
                    (target) locale, but not in the target (source) locale. For
                    example, an added short introduction of an entity not well
                    known in the target locale, or an omitted introduction of an
                    entity well known in the target locale.
                  </li>
                  <li>
                    The translation edits the source creatively, perhaps to make
                    the translation more fluent or expressive or localized.
                    For example, a reordering of sentences or an addition of
                    content that is supported by the overall passage being
                    translated and does not change the intent of the source.
                    Another example would be that of the name of an
                    exemplifying entity being changed to a better local
                    equivalent.
                  </li>
                </ul>
              </li>
              <li>
                <b>Mistranslation</b>.
                The translation does not accurately represent the source
                speech. Examples: (1) The source talks about a person A
                knowing another person B, but the translation says
                "A was intimate with B." (2) The source states that
                something never happens, whereas the translation says it happens
                "often" or "rarely."
              </li>
              <li>
                <b>Gender Mismatch</b>.
                Given the context, the gender is incorrect (incorrect pronouns,
                noun/adjective endings, etc). Misgendering errors typically have
                a major severity as they significantly alter the meaning of the
                source. If the correct gender is not clear from the source
                context, assume the first instance of gender is correct and mark
                subsequent gender inconsistencies as errors. Examples: (1) "Mary
                slammed the door as he left." (2) Given the source "My
                friend is an engineer. She is great at coding.", if the
                translation used a male form for the term "engineer", it would
                be incorrect, as we know the subject is female from the source
                context.
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
                The translation includes information not present in the
                source. Example: A translated sentence that includes adverbs or
                adjectives without equivalents in the source, even after
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
          Issues related to the form or content of the translation, independent
          of its relation to the source; errors in the translation that
          make it harder to understand.
          <details>
            <summary>Subtypes of Fluency:</summary>
            <ul>
              <li>
                <b>Inconsistency</b>.
                The translation shows internal inconsistency (not related to
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
                Issues related to the grammar or syntax of the translation,
                other than spelling and orthography. Example: An English
                translation reads "They goes together," or "He could of fixed
                it." Both of these examples have jarring flaws that
                significantly degrade the fluency of the translation and would
                justify a major severity. However,
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
                <br><br>
                Note that if sentences in the translation are structured
                differently compared to the source (eg., a source sentence has
                been translated into two sentences, or two source sentences
                have been combined into a single translated sentence), that
                by itself is not a text-breaking error.
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
          The translation has stylistic problems.
          <details>
            <summary>Subtypes of Style:</summary>
            <ul>
              <li>
                <b>Unnatural or awkward</b>.
                The translation is literal, awkward, unidiomatic, or
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
                structure. The marked span is an unnecessary repetition,
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
                The translation contains archaic or obscure words that an average
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
                Translation does not adhere to appropriate industry standard
                terminology or contains terminology that does not fit the
                context. Example: "acide sulfurique" is translated to
                "acid of sulfur" instead of "sulfuric acid." This example would
                have a minor severity level.
              </li>
              <li>
                <b>Inconsistent</b>.
                Terminology is used in an inconsistent manner within the translation.
                Example: The translation of a phone manual alternates between
                the terms "front camera" and "front lens." This example would
                have a minor severity level.
              </li>
            </ul>
          </details>
        </li>
        <li>
          <b>Locale convention</b>.
          The translation does not adhere to locale-specific conventions and
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
                A date format inappropriate for the locale is used. Example:
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
      </ul>`,
  },

  // Default instructions section order.
  instructions_section_order: [
    '_style', 'Overview', 'General Guidelines', 'Navigation',
    'Annotation Process', 'Annotation Tips', 'Severities defined',
    'Error Types and Subtypes defined', 'Speech Error Categories', 'Annotations exemplified in detail',
    'Style &amp; Convention Guidelines', 'Feedback'
  ],
};
