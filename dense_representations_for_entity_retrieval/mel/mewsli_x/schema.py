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

"""Data representations."""

from __future__ import annotations

import collections
import copy
import dataclasses
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type, TypeVar, Union

from dense_representations_for_entity_retrieval.mel.mewsli_x import io_util

JsonValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JsonDict = Dict[str, JsonValue]
JsonList = List[JsonValue]
StrOrPurePath = io_util.StrOrPurePath


def to_jsonl(obj: JsonDict) -> str:
  return json.dumps(obj, ensure_ascii=False)


@dataclasses.dataclass(frozen=True)
class Span:
  """A [start:end]-span in some external string."""
  start: int
  end: int

  def __post_init__(self):
    if self.start < 0:
      raise ValueError(f"start offset is out of bounds {self}")
    if self.end < 0:
      raise ValueError(f"end offset is out of bounds {self}")
    if self.start >= self.end:
      raise ValueError(f"start and end offsets are non-monotonic {self}")

  @staticmethod
  def from_json(json_dict: JsonDict) -> Span:
    """Creates a new Span instance from the given JSON-dictionary."""
    return Span(start=json_dict["start"], end=json_dict["end"])

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    return dict(start=self.start, end=self.end)

  def validate_offsets_relative_to_context(self, context: str) -> None:
    """Validates the span's offsets relative to a context string."""
    if self.start >= len(context):
      raise ValueError(
          f"start offset in {self} is out of bounds w.r.t. '{context}'")
    if self.end > len(context):
      raise ValueError(
          f"end offset in {self} is out of bounds w.r.t. '{context}'")

  def locate_in(self, spans: Iterable[Span]) -> Optional[int]:
    """Returns the index of the first span that fully contains `self`.

    Args:
      spans: The spans to search.

    Returns:
      First i such that spans[i].{start,end} covers `self.{start,end}`, or None
      if there is no such span, indicating that `self` either is out of range
      relative to spans or crosses span boundaries.
    """
    for i, span in enumerate(spans):
      # The starts may coincide and the ends may coincide.
      if (span.start <= self.start and self.start < span.end and
          span.start < self.end and self.end <= span.end):
        return i
    return None


@dataclasses.dataclass(frozen=True)
class TextSpan(Span):
  """A text span relative to an external string T, with text=T[start:end]."""
  text: str

  def validate_relative_to_context(self, context: str) -> None:
    """Validates that `self.text` matches the designated span in `context`."""
    self.validate_offsets_relative_to_context(context)
    ref_text = context[self.start:self.end]
    if self.text != ref_text:
      raise ValueError(f"{self} does not match against context '{context}': "
                       f"'{self.text}' != '{ref_text}'")

  @staticmethod
  def from_context(span: Span, context: str) -> TextSpan:
    """Creates a new TextSpan by extracting the given `span` from `context`."""
    span.validate_offsets_relative_to_context(context)
    return TextSpan(span.start, span.end, text=context[span.start:span.end])

  @staticmethod
  def from_elements(start: int, end: int, context: str) -> TextSpan:
    """Creates a new TextSpan by extracting [start:end] from `context`."""
    return TextSpan.from_context(span=Span(start, end), context=context)

  @staticmethod
  def from_json(json_dict: JsonDict) -> TextSpan:
    """Creates a new TextSpan from the given JSON-dictionary."""
    return TextSpan(
        start=json_dict["start"], end=json_dict["end"], text=json_dict["text"])

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    return dict(start=self.start, end=self.end, text=self.text)


@dataclasses.dataclass(frozen=True)
class Entity:
  """An entity and its textual representation.

  Attributes:
    entity_id: Unique identifier of the entity, e.g. WikiData QID.
    title: A title phrase that names the entity.
    description: A definitional description of the entity that serves as its
      unique textual representation, e.g. taken from the beginning of the
      entity's Wikipedia page.
    sentence_spans: Sentence break annotations for the description, as
      character-level Span objects that index into `description`
    sentences: Sentences extracted from `description` according to
      `sentence_spans`. These TextSpan objects include the actual sentence text
      for added convenience. E.g., the string of the description's first
      sentence is `sentences[0].text`.
    description_language: Primary language code of the description and title,
      matching the Wikipedia language edition from which they were extracted.
    description_url: URL of the page where the description was extracted from.
  """
  entity_id: str
  title: str
  description: str
  sentence_spans: Tuple[Span, ...]
  description_language: str
  description_url: str

  def __post_init__(self):
    self.validate()

  @property
  def sentences(self) -> Iterator[TextSpan]:
    for span in self.sentence_spans:
      yield TextSpan.from_context(span, self.description)

  def validate(self):
    for sentence_span in self.sentence_spans:
      sentence_span.validate_offsets_relative_to_context(self.description)

  @staticmethod
  def from_json(json_dict: JsonDict) -> Entity:
    """Creates a new Entity from the given JSON-dictionary."""
    return Entity(
        entity_id=json_dict["entity_id"],
        title=json_dict["title"],
        description=json_dict["description"],
        description_language=json_dict["description_language"],
        description_url=json_dict["description_url"],
        sentence_spans=tuple(
            Span.from_json(t) for t in json_dict["sentence_spans"]),
    )

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    return dict(
        entity_id=self.entity_id,
        title=self.title,
        description=self.description,
        description_language=self.description_language,
        description_url=self.description_url,
        sentence_spans=[t.to_json() for t in self.sentence_spans],
    )


@dataclasses.dataclass(frozen=True)
class Mention:
  """A single mention of an entity, referring to some external context.

  Attributes:
    example_id: Unique identifier for the mention instance.
    mention_span: A TextSpan denoting one mention, relative to external context.
    entity_id: ID of the mentioned entity.
    metadata: Optional dictionary of additional information about the instance.
  """
  example_id: str
  mention_span: TextSpan
  entity_id: str
  metadata: Optional[Dict[str, str]] = None

  @staticmethod
  def from_json(json_dict: JsonDict) -> Mention:
    """Creates a new Mention from the given JSON-dictionary."""
    return Mention(
        example_id=json_dict["example_id"],
        mention_span=TextSpan.from_json(json_dict["mention_span"]),
        entity_id=json_dict["entity_id"],
        metadata=json_dict.get("metadata"),
    )

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        example_id=self.example_id,
        mention_span=self.mention_span.to_json(),
        entity_id=self.entity_id,
    )
    if self.metadata is not None:
      json_dict["metadata"] = self.metadata
    return json_dict


@dataclasses.dataclass()
class Context:
  """A document text fragment and metadata.

  Attributes:
    document_title: Title of the document.
    document_url: URL of the document.
    document_id: An identifier for the document. For a Wikipedia page, this may
      be the associated WikiData QID.
    language: Primary language code of the document.
    text: Original text from the document.
    sentence_spans: Sentence break annotations for the text, as character-level
      Span objects that index into `text`.
    sentences: Sentences extracted from `text` according to `sentence_spans`.
      These TextSpan objects include the actual sentence text for added
      convenience. E.g., the first sentence's string is `sentences[0].text`.
    section_title: Optional title of the section under which `text` appeared.
  """
  document_title: str
  document_url: str
  document_id: str
  language: str
  text: str
  sentence_spans: Tuple[Span, ...]
  section_title: Optional[str] = None

  def __post_init__(self):
    self.validate()

  @property
  def sentences(self) -> Iterator[TextSpan]:
    for span in self.sentence_spans:
      yield TextSpan.from_context(span, self.text)

  def validate(self):
    for sentence_span in self.sentence_spans:
      sentence_span.validate_offsets_relative_to_context(self.text)

  @staticmethod
  def from_json(json_dict: JsonDict) -> Context:
    """Creates a new Context from the given JSON-dictionary."""
    return Context(
        document_title=json_dict["document_title"],
        section_title=json_dict.get("section_title"),
        document_url=json_dict["document_url"],
        document_id=json_dict["document_id"],
        language=json_dict["language"],
        text=json_dict["text"],
        sentence_spans=tuple(
            Span.from_json(t) for t in json_dict["sentence_spans"]),
    )

  def to_json(self, keep_text: bool = True) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        document_title=self.document_title,
        document_url=self.document_url,
        document_id=self.document_id,
        language=self.language,
        text=self.text if keep_text else "",
        sentence_spans=[t.to_json() for t in self.sentence_spans],
    )
    if self.section_title is not None:
      json_dict["section_title"] = self.section_title
    return json_dict

  def truncate(self, focus: int, window_size: int) -> Tuple[int, Context]:
    """Truncates the Context to window_size sentences each side of focus.

    This seeks to truncate the text and sentence_spans of `self` to
      self.sentence_spans[focus - window_size:focus + window_size + 1].

    When there are fewer than window_size sentences available before (after) the
    focus, this attempts to retain additional context sentences after (before)
    the focus.

    Args:
      focus: The index of the focus sentence in self.sentence_spans.
      window_size: Number of sentences to retain on each side of the focus.

    Returns:
      - c, the number of characters removed from the start of the text, which is
        useful for updating any Mention defined in relation to this Context.
      - new_context, a copy of the Context that is updated to contain the
        truncated text and sentence_spans.

    Raises:
      IndexError: if focus is not within the range of self.sentence_spans.
      ValueError: if window_size is negative.
    """
    if focus < 0 or focus >= len(self.sentence_spans):
      raise IndexError(f"Index {focus} invalid for {self.sentence_spans}")
    if window_size < 0:
      raise ValueError(f"Expected a positive window, but got {window_size}")

    snt_window = self._get_sentence_window(focus, window_size)
    relevant_sentences = self.sentence_spans[snt_window.start:snt_window.end]

    char_offset = relevant_sentences[0].start
    char_end = relevant_sentences[-1].end
    new_text = self.text[char_offset:char_end]

    new_sentences = [
        Span(old_sentence.start - char_offset, old_sentence.end - char_offset)
        for old_sentence in relevant_sentences
    ]
    new_context = dataclasses.replace(
        self, text=new_text, sentence_spans=tuple(new_sentences))
    return char_offset, new_context

  def _get_sentence_window(self, focus: int, window_size: int) -> Span:
    """Gets Span of sentence indices to cover window around the focus index."""
    # Add window to the left of focus. If there are fewer sentences before the
    # focus sentence, carry over the remainder.
    left_index = max(focus - window_size, 0)
    remainder_left = window_size - (focus - left_index)
    assert remainder_left >= 0, remainder_left

    # Add window to the right of focus, including carryover. (Note, right_index
    # is an inclusive index.) If there are fewer sentences after the focus
    # sentence, carry back the remainder.
    right_index = min(focus + window_size + remainder_left,
                      len(self.sentence_spans) - 1)
    remainder_right = window_size - (right_index - focus)

    if remainder_right > 0:
      # Extend further leftward.
      left_index = max(left_index - remainder_right, 0)

    return Span(left_index, right_index + 1)


@dataclasses.dataclass()
class ContextualMentions:
  """Multiple entity mentions in a shared context."""
  context: Context
  mentions: List[Mention]

  def __post_init__(self):
    self.validate()

  def validate(self):
    self.context.validate()
    for mention in self.mentions:
      mention.mention_span.validate_relative_to_context(self.context.text)

  @staticmethod
  def from_json(json_dict: JsonDict) -> ContextualMentions:
    """Creates a new ContextualMentions from the given JSON-dictionary."""
    return ContextualMentions(
        context=Context.from_json(json_dict["context"]),
        mentions=[Mention.from_json(m) for m in json_dict["mentions"]],
    )

  def to_json(self, keep_text: bool = True) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        context=self.context.to_json(keep_text=keep_text),
        mentions=[m.to_json() for m in self.mentions],
    )
    return json_dict

  def unnest_to_single_mention_per_context(self) -> Iterator[ContextualMention]:
    for mention in self.mentions:
      yield ContextualMention(
          context=copy.deepcopy(self.context), mention=copy.deepcopy(mention))

  @staticmethod
  def nest_mentions_by_shared_context(
      contextual_mentions: Iterable[ContextualMention]
  ) -> Iterator[ContextualMentions]:
    """Inverse of unnest_to_single_mention_per_context."""
    contexts = {}
    groups = collections.defaultdict(list)
    for cm in contextual_mentions:
      context = cm.context
      key = (context.document_id, context.section_title, context.text)
      if key in contexts:
        assert contexts[key] == context, key
      else:
        contexts[key] = context
      groups[key].append(cm.mention)

    for key, mentions in groups.items():
      yield ContextualMentions(contexts[key], mentions)


@dataclasses.dataclass()
class ContextualMention:
  """A single entity mention in context."""
  context: Context
  mention: Mention

  def __post_init__(self):
    self.validate()

  def validate(self):
    self.context.validate()
    self.mention.mention_span.validate_relative_to_context(self.context.text)

  @staticmethod
  def from_json(json_dict: JsonDict) -> ContextualMention:
    """Creates a new ContextualMention from the given JSON-dictionary."""
    return ContextualMention(
        context=Context.from_json(json_dict["context"]),
        mention=Mention.from_json(json_dict["mention"]),
    )

  def to_json(self, keep_text: bool = True) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        context=self.context.to_json(keep_text=keep_text),
        mention=self.mention.to_json(),
    )
    return json_dict

  def truncate(self, window_size: int) -> Optional[ContextualMention]:
    """Truncates the context to window_size sentences each side of the mention.

    Args:
      window_size: Number of sentences to retain on each side of the sentence
        containing the mention. See Context.truncate for more detail.

    Returns:
      Returns None if no sentence spans were present or if the mention crosses
      sentence boundaries. Otherwise, returns an update copy of the
      ContextualMention where `.context` contains the truncated text and
      sentences, and the character offsets in `.mention` updated accordingly.
    """
    focus_snt = self.mention.mention_span.locate_in(self.context.sentence_spans)
    if focus_snt is None:
      # The context has no sentences or the mention crosses sentence boundaries.
      return None

    offset, new_context = self.context.truncate(
        focus=focus_snt, window_size=window_size)

    # Internal consistency check.
    max_valid = window_size * 2 + 1
    assert len(new_context.sentence_spans) <= max_valid, (
        f"Got {len(new_context.sentence_spans)}>{max_valid} sentences for "
        f"window_size={window_size} in truncated Context: {new_context}")

    new_mention = dataclasses.replace(
        self.mention,
        mention_span=TextSpan(
            start=self.mention.mention_span.start - offset,
            end=self.mention.mention_span.end - offset,
            text=self.mention.mention_span.text))
    return ContextualMention(context=new_context, mention=new_mention)


@dataclasses.dataclass()
class MentionEntityPair:
  """A ContextualMention paired with the Entity it refers to."""
  contextual_mention: ContextualMention
  entity: Entity

  def __post_init__(self):
    self.validate()

  def validate(self):
    self.contextual_mention.validate()
    self.entity.validate()

  @staticmethod
  def from_json(json_dict: JsonDict) -> MentionEntityPair:
    """Creates a new MentionEntityPair from the given JSON-dictionary."""
    return MentionEntityPair(
        contextual_mention=ContextualMention.from_json(
            json_dict["contextual_mention"]),
        entity=Entity.from_json(json_dict["entity"]),
    )

  def to_json(self) -> JsonDict:
    """Returns instance as JSON-compatible nested dictionary."""
    json_dict = dict(
        contextual_mention=self.contextual_mention.to_json(),
        entity=self.entity.to_json(),
    )
    return json_dict


SchemaAnyT = TypeVar("SchemaAnyT", ContextualMention, ContextualMentions,
                     Entity, MentionEntityPair)
SchemaAny = Union[ContextualMention, ContextualMentions, Entity,
                  MentionEntityPair]

EntityOrContext = TypeVar("EntityOrContext", Entity, Context)


def add_sentence_spans(item: EntityOrContext,
                       sentence_spans: Iterable[Span]) -> EntityOrContext:
  """Returns a copy of item, adding the given sentence_spans."""
  if item.sentence_spans:
    raise ValueError(f"sentence_spans already populated: {item}")
  return dataclasses.replace(item, sentence_spans=tuple(sentence_spans))


def load_text(path: StrOrPurePath) -> str:
  """Returns the contents of a text file."""
  with io_util.open_file(path, "rt") as input_file:
    return input_file.read()


def load_jsonl_as_dicts(path: StrOrPurePath) -> List[JsonDict]:
  """Returns dict-records from JSONL file (without parsing into dataclasses)."""
  with io_util.open_file(path) as input_file:
    return [json.loads(line) for line in input_file]


def load_jsonl(path: StrOrPurePath,
               schema_cls: Type[SchemaAnyT]) -> List[SchemaAnyT]:
  """Loads the designated type of schema dataclass items from a JSONL file.

  Args:
    path: File path to load. Each line in the file is a JSON-serialized object.
    schema_cls: The dataclass to parse into, e.g. `ContextualMention`, `Entity`,
      etc.

  Returns:
    A list of validated instances of `schema_cls`, one per input line.
  """
  result = []
  for json_dict in load_jsonl_as_dicts(path):
    result.append(schema_cls.from_json(json_dict))
  return result


def write_jsonl(path: StrOrPurePath, items: Iterable[SchemaAny]) -> None:
  """Writes a list of any of the schema dataclass items to JSONL file.

  Args:
    path: Output file path that will store each item as a JSON-serialized line.
    items: Items to output. Instances of a schema dataclass, e.g.
      `ContextualMention`, `Entity`, etc.
  """
  with io_util.open_file(path, "wt") as output_file:
    for item in items:
      print(to_jsonl(item.to_json()), file=output_file)
