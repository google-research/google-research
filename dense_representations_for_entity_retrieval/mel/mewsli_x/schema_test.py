# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for schema."""

import copy
import json

from absl.testing import absltest
from absl.testing import parameterized

from dense_representations_for_entity_retrieval.mel.mewsli_x import schema


class SchemaTest(parameterized.TestCase):

  _TEST_SPANS = (schema.Span(0, 12), schema.Span(13, 17))

  def test_text_span_from(self):
    got = schema.TextSpan.from_elements(
        start=13, end=35, context="Large place. Inhabited by everyone.")
    exp = schema.TextSpan(start=13, end=35, text="Inhabited by everyone.")
    self.assertEqual(exp, got)

  def test_text_span_from_raises(self):
    with self.assertRaises(ValueError):
      schema.TextSpan.from_elements(
          context="Large place. Inhabited by everyone.",
          start=13,
          # Span end is outside text.
          end=50)

  @parameterized.named_parameters([
      # For easy reference: _TEST_SPANS is [(0,12), (13,17)].
      dict(
          testcase_name="strict_subspan",
          target=schema.Span(6, 11),
          expected_span_idx=0),
      dict(
          testcase_name="target_at_span_start",
          target=schema.Span(13, 16),
          expected_span_idx=1),
      dict(
          testcase_name="target_at_span_end",
          target=schema.Span(14, 17),
          expected_span_idx=1),
      dict(
          testcase_name="target_covers_whole_span",
          target=schema.Span(13, 17),
          expected_span_idx=1),
  ])
  def test_locate_in(self, target, expected_span_idx):
    self.assertEqual(target.locate_in(self._TEST_SPANS), expected_span_idx)

  @parameterized.named_parameters([
      # For easy reference: _TEST_SPANS is [(0,12), (13,17)].
      dict(
          testcase_name="crossing",
          target=schema.Span(5, 15),
          spans=_TEST_SPANS),
      dict(
          testcase_name="out_of_range_right",
          target=schema.Span(20, 25),
          spans=_TEST_SPANS),
      dict(
          testcase_name="out_of_range_left",
          target=schema.Span(15, 20),
          spans=[schema.Span(25, 30)]),
      dict(testcase_name="empty_spans", target=schema.Span(15, 20), spans=[]),
  ])
  def test_locate_in_target_not_found(self, target, spans):
    self.assertIsNone(target.locate_in(spans))


class EntityTest(absltest.TestCase):
  TEST_ENTITY = schema.Entity(
      entity_id="Q1",
      title="World",
      description="Large place. Inhabited by everyone.",
      description_url="http://www.x.com",
      description_language="en",
      sentence_spans=(schema.Span(0, 12), schema.Span(13, 35)))

  def test_sentences(self):
    exp = [
        schema.TextSpan(start=0, end=12, text="Large place."),
        schema.TextSpan(start=13, end=35, text="Inhabited by everyone.")
    ]
    got = list(self.TEST_ENTITY.sentences)
    self.assertEqual(exp, got)

  def test_add_sentence_spans(self):
    got = schema.add_sentence_spans(
        schema.Entity(
            entity_id="Q1",
            title="World",
            description="Large place. Inhabited by everyone.",
            description_url="http://www.x.com",
            description_language="en",
            sentence_spans=()),
        # OK to pass list instead of tuple to add_sentence_spans.
        sentence_spans=[schema.Span(0, 12),
                        schema.Span(13, 35)])
    self.assertEqual(got, self.TEST_ENTITY)

  def test_json_roundtrip(self):
    json_string = json.dumps(self.TEST_ENTITY.to_json())
    got = schema.Entity.from_json(json.loads(json_string))
    self.assertEqual(self.TEST_ENTITY, got)

  def test_invalid_spans_raises(self):
    with self.assertRaises(ValueError):
      _ = schema.Entity(
          entity_id="Q1",
          title="World",
          # Truncated text to invalidate the sentence spans.
          description="Large place.",
          description_url="http://www.x.com",
          description_language="en",
          sentence_spans=(schema.Span(0, 12), schema.Span(13, 35)))

  def test_bad_span_raises(self):
    with self.assertRaises(ValueError):
      schema.Entity(
          entity_id="Q1",
          title="World",
          description="Large place. Inhabited by everyone.",
          description_url="http://www.x.com",
          description_language="en",
          sentence_spans=(
              # Zero-length span is invalid.
              schema.Span(12, 12),
              schema.Span(13, 35),
          ))


class ContextualMentionsTest(parameterized.TestCase):
  _CONTEXT_A = context = schema.Context(
      document_title="Planet Earth",
      document_url="www.xyz.com",
      document_id="xyz-123",
      language="en",
      text="We all live here. Here in the World.",
      sentence_spans=(schema.Span(0, 17), schema.Span(18, 36)))
  _CONTEXT_B = schema.Context(
      document_title="Planet Earth",
      document_url="www.xyz.com",
      document_id="xyz-123",
      section_title="Intro",
      language="en",
      text="We all live 🌵. Here in the World.",
      sentence_spans=(schema.Span(0, 14), schema.Span(15, 33)))
  _CONTEXT_C = schema.Context(
      document_title="Planet Earth",
      document_url="www.xyz.com",
      document_id="xyz-123",
      section_title="Intro",
      language="en",
      text="We all live 🌵. Here in the World.",
      # For brevity, mimick sentences using phrases.
      sentence_spans=(
          schema.Span(0, 2),  # We
          schema.Span(3, 6),  # all
          schema.Span(7, 14),  # live 🌵.
          schema.Span(15, 19),  # Here
          schema.Span(20, 26),  # in the
          schema.Span(27, 33),  # World.
      ))
  # Two test mentions consistent with CONTEXT_C.
  _ALL_MENTION = schema.Mention(
      mention_span=schema.TextSpan(start=3, end=6, text="all"),
      entity_id="Qx",
      example_id="82e3")
  _WORLD_MENTION = schema.Mention(
      mention_span=schema.TextSpan(start=27, end=32, text="World"),
      entity_id="Q1",
      example_id="9024f")

  def test_round_trip(self):
    context_mentions = schema.ContextualMentions(
        context=self._CONTEXT_A,
        mentions=[
            schema.Mention(
                example_id="12fe",
                mention_span=schema.TextSpan(start=12, end=16, text="here"),
                entity_id="Q1"),
            schema.Mention(
                example_id="30ba",
                mention_span=schema.TextSpan(start=30, end=35, text="World"),
                entity_id="Q1",
                metadata={"bin_name": "bin03_100-1000"},
            ),
        ])
    context_mentions.validate()

    self.assertEqual(
        list(context_mentions.context.sentences), [
            schema.TextSpan(0, 17, "We all live here."),
            schema.TextSpan(18, 36, "Here in the World.")
        ])

    json_string = json.dumps(context_mentions.to_json())
    got = schema.ContextualMentions.from_json(json.loads(json_string))
    self.assertEqual(context_mentions, got)

    # Perturb text to invalidate the sentence spans.
    perturbed = copy.deepcopy(context_mentions)
    perturbed.context.text = perturbed.context.text[::2]
    with self.assertRaises(ValueError):
      perturbed.validate()

  def test_context_add_sentences(self):
    test_context = schema.add_sentence_spans(
        schema.Context(
            document_title="Planet Earth",
            document_url="www.xyz.com",
            document_id="xyz-123",
            language="en",
            text="We all live here. Here in the World.",
            sentence_spans=()),
        # OK to pass list instead of tuple to add_sentence_spans.
        sentence_spans=[schema.Span(0, 17),
                        schema.Span(18, 36)])
    self.assertEqual(test_context, self._CONTEXT_A)

  def test_round_trip_including_section(self):
    context_mentions = schema.ContextualMentions(
        context=self._CONTEXT_A,
        mentions=[
            schema.Mention(
                example_id="12fe",
                mention_span=schema.TextSpan(start=12, end=16, text="here"),
                entity_id="Q1"),
            schema.Mention(
                example_id="30ba",
                mention_span=schema.TextSpan(start=30, end=35, text="World"),
                entity_id="Q1",
                metadata={"bin_name": "bin03_100-1000"},
            ),
        ])
    context_mentions.validate()
    json_string = json.dumps(context_mentions.to_json())
    got = schema.ContextualMentions.from_json(json.loads(json_string))
    self.assertEqual(context_mentions, got)

  def test_simple_multibyte(self):
    context_mentions = schema.ContextualMentions(
        context=self._CONTEXT_B,
        mentions=[
            schema.Mention(
                example_id="12fe",
                mention_span=schema.TextSpan(start=12, end=13, text="🌵"),
                entity_id="Q1"),
        ])
    context_mentions.validate()

    self.assertEqual(
        list(context_mentions.context.sentences), [
            schema.TextSpan(0, 14, "We all live 🌵."),
            schema.TextSpan(15, 33, "Here in the World.")
        ])

    json_string = json.dumps(context_mentions.to_json())
    got = schema.ContextualMentions.from_json(json.loads(json_string))
    self.assertEqual(context_mentions, got)

  def test_simple_multibyte_single(self):
    context_mention = schema.ContextualMention(
        context=self._CONTEXT_B,
        mention=schema.Mention(
            example_id="12fe",
            mention_span=schema.TextSpan(start=12, end=13, text="🌵"),
            entity_id="Q1"),
    )
    context_mention.validate()

    self.assertEqual(
        list(context_mention.context.sentences), [
            schema.TextSpan(0, 14, "We all live 🌵."),
            schema.TextSpan(15, 33, "Here in the World.")
        ])

    json_string = json.dumps(context_mention.to_json())
    got = schema.ContextualMention.from_json(json.loads(json_string))
    self.assertEqual(context_mention, got)

    # Perturb text to invalidate the sentence spans.
    perturbed = copy.deepcopy(context_mention)
    perturbed.context.text = perturbed.context.text[::2]
    with self.assertRaises(ValueError):
      perturbed.validate()

  def test_invalid_raises(self):
    with self.assertRaises(ValueError):
      _ = schema.ContextualMentions(
          context=self._CONTEXT_A,
          mentions=[
              schema.Mention(
                  example_id="12fe",
                  mention_span=schema.TextSpan(
                      # Start and end does not refer to substring 'here'.
                      start=0,
                      end=5,
                      text="here"),
                  entity_id="Q1"),
          ])

  def test_unnest(self):
    context_mentions = schema.ContextualMentions(
        context=self._CONTEXT_A,
        mentions=[
            schema.Mention(
                example_id="12fe",
                mention_span=schema.TextSpan(start=12, end=16, text="here"),
                entity_id="Q1"),
            schema.Mention(
                example_id="30ba",
                mention_span=schema.TextSpan(start=30, end=35, text="World"),
                entity_id="Q1",
            ),
        ])
    context_mentions.validate()
    expected = [
        schema.ContextualMention(
            context=self._CONTEXT_A,
            mention=schema.Mention(
                example_id="12fe",
                mention_span=schema.TextSpan(start=12, end=16, text="here"),
                entity_id="Q1"),
        ),
        schema.ContextualMention(
            context=self._CONTEXT_A,
            mention=schema.Mention(
                example_id="30ba",
                mention_span=schema.TextSpan(start=30, end=35, text="World"),
                entity_id="Q1",
            ),
        )
    ]
    self.assertEqual(
        list(context_mentions.unnest_to_single_mention_per_context()), expected)

  def test_unnest_discards_input_without_mentions(self):
    context_mentions = schema.ContextualMentions(
        context=self._CONTEXT_A, mentions=[])
    self.assertEmpty(
        list(context_mentions.unnest_to_single_mention_per_context()))

  @parameterized.named_parameters([
      dict(
          testcase_name="only_focus_sentence",
          contextual_mention=schema.ContextualMention(
              context=_CONTEXT_C, mention=_ALL_MENTION),
          window_size=0,
          expected_sentences_text="all",
          expected_mention=schema.Mention(
              # Shifted mention span because first "sentence" gets dropped.
              mention_span=schema.TextSpan(start=0, end=3, text="all"),
              entity_id="Qx",
              example_id="82e3")),
      dict(
          testcase_name="window_1",
          contextual_mention=schema.ContextualMention(
              context=_CONTEXT_C, mention=_ALL_MENTION),
          window_size=1,
          expected_sentences_text="We/all/live 🌵.",
          expected_mention=_ALL_MENTION),
      dict(
          testcase_name="window_exceeds_context",
          contextual_mention=schema.ContextualMention(
              context=_CONTEXT_C, mention=_ALL_MENTION),
          window_size=10,
          expected_sentences_text="We/all/live 🌵./Here/in the/World.",
          expected_mention=_ALL_MENTION),
      dict(
          testcase_name="window_2_carryover_to_right",
          contextual_mention=schema.ContextualMention(
              context=_CONTEXT_C, mention=_ALL_MENTION),
          window_size=2,
          expected_sentences_text="We/all/live 🌵./Here/in the",
          expected_mention=_ALL_MENTION),
      dict(
          testcase_name="window_2_carryover_to_left",
          contextual_mention=schema.ContextualMention(
              context=_CONTEXT_C, mention=_WORLD_MENTION),
          window_size=2,
          expected_sentences_text="all/live 🌵./Here/in the/World.",
          expected_mention=schema.Mention(
              # Shifted mention span because first "sentence" gets dropped.
              mention_span=schema.TextSpan(start=24, end=29, text="World"),
              entity_id="Q1",
              example_id="9024f",
          )),
  ])
  def test_truncate(self, contextual_mention, window_size,
                    expected_sentences_text, expected_mention):
    truncated = contextual_mention.truncate(window_size)

    # For brevity, the truncated ContextualMention is validated only in terms of
    # its concatenated sentences (delimited with "/" for readability).
    self.assertEqual(
        "/".join(s.text for s in truncated.context.sentences),
        expected_sentences_text,
        msg=f"In {truncated}")

    self.assertEqual(truncated.mention, expected_mention)

  def test_truncate_skips_mention_that_crosses_boundary(self):
    contextual_mention = schema.ContextualMention(
        context=self._CONTEXT_C,
        mention=schema.Mention(
            # (0,6) covers the first two sentences (0,2), (3,6).
            mention_span=schema.TextSpan(start=0, end=6, text="We all"),
            entity_id="Qx",
            example_id="82e3"))
    self.assertIsNone(contextual_mention.truncate(window_size=1))

  def test_truncate_skips_mention_if_context_has_no_sentences(self):
    contextual_mention = schema.ContextualMention(
        context=schema.Context(
            document_title="Planet Earth",
            document_url="www.xyz.com",
            document_id="xyz-123",
            section_title="Intro",
            language="en",
            text="We all live 🌵. Here in the World.",
            sentence_spans=()),
        mention=self._ALL_MENTION)
    self.assertIsNone(contextual_mention.truncate(window_size=1))

  def test_truncate_raises(self):
    with self.assertRaises(ValueError):
      self._CONTEXT_A.truncate(focus=0, window_size=-1)
    with self.assertRaises(IndexError):
      self._CONTEXT_B.truncate(focus=-1, window_size=0)
    with self.assertRaises(IndexError):
      self._CONTEXT_C.truncate(focus=50, window_size=0)


class MentionEntityPairTest(absltest.TestCase):

  def test_round_trip(self):
    example = schema.MentionEntityPair(
        contextual_mention=schema.ContextualMention(
            context=ContextualMentionsTest._CONTEXT_B,
            mention=schema.Mention(
                example_id="12fe",
                mention_span=schema.TextSpan(start=12, end=13, text="🌵"),
                entity_id="Q1")),
        entity=EntityTest.TEST_ENTITY)
    example.validate()

    json_string = json.dumps(example.to_json())
    got = schema.MentionEntityPair.from_json(json.loads(json_string))
    self.assertEqual(example, got)


if __name__ == "__main__":
  absltest.main()
