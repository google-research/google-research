# MathWriting Dataset

## TL;DR

This dataset is a collection of online handwritten mathematical expressions.

Inks are stored in InkML format. The ground truth is in the `normalizedLabel`
field. Use data from `train/`, `synthetic/` and optionally `symbols/` as
training set. Use `valid/` for hyperparameter tuning and `test/` for the final
evaluation.

## Overview

All directories (`train`, `test`, `valid`, `symbols`, `synthetic`) contain a
set of inks. Each ink is stored as a separate InkML file, whose file name
serves as an identifier. See the next sections for more details.

- `valid` and `test` contain inks intended for *testing* a model's
  performance. *Every other directory contains inks intended for training*.
- `train`: inks written by human contributors. People were shown an image of
  a LaTeX-rendered single mathematical expression and asked to handwrite it
  using either a touchscreen or a digital pen.
- `symbols`: inks of individual handwritten glyphs, extracted from inks from
  the `train` directory. These inks have been used to generate the synthetic
  inks.
- `synthetic`: inks obtained by stitching together individual handwritten
  glyphs from `symbols/` using bounding boxes obtained through the LaTeX
  compiler.

The two files at the top-level (apart from this readme) contain the necessary
information to enable the regeneration of the synthetic samples. Their content
is described below.

## InkML files

Each ink is stored as a separate file, using the InkML format:
https://www.w3.org/TR/InkML/. Filenames follow the pattern
`[a-f0-9]{16}.inkml`. The 16-character hexadecimal part is a
random number used as the ink identifier.

Inks contain only stroke information: spatial coordinates and relative time
information. No segmentation information is included. Brush size is not
specified, but a round shape of radius 1.5 is a reasonable value for most
inks. Note that *ink coordinates are not normalized* neither in space nor
time: human-written inks have been obtained on a large variety of devices with
varying screen resolutions. The vertical extent of a given letter can vary
greatly depending on the ink and on the writer. Similarly, the sampling rate
is not identical across inks.

Some metadata is provided through `annotation` elements:

- `label`: the original LaTeX mathematical expression that was used to render
  the image that was copied by contributors or used to generate bounding boxes
  in the case of synthetic inks.
- `normalizedLabel`: a normalized version of 'label', in which synonyms have
  been canonicalized (e.g. \over -> \frac), extra spaces have been removed,
  usage of curly braces made consistent, constructs like "italics" that can't
  be rendered in handwriting have been eliminated, etc.
- `splitTagOriginal`: a string containing the name of the split (i.e. "train",
  "valid", "test", "synthetic", "symbols") this ink belongs to. It is the same
  as the name of the directory containing the file.
- `sampleId`: the ink identifier, as an 16-character hexadecimal string. It is
  the same as in the filename.
- `inkCreationMethod`: a brief indication of how the ink was obtained.
  Possible values are:
  - 'human': obtained from human contributors.
  - 'boundingBoxes': obtained by stitching together individual handwritten
    glyphs.
- `labelCreationMethod`: in case it is provided, indicates how the value in
  the `label` field was obtained. Possible values are:
  - 'wikipedia': extracted from Wikipedia.
  - 'synthetic': generated programmatically. There are ~25k such cases,
    covering a range of complex fractions that were not represented in the
    Wikipedia expressions.

Note about the `label` and `normalizedLabel` fields.
- `normalizedLabel` is meant to be used as the ground truth to train a
  recognizer. It is missing for inks under `symbols/`: `label` is to be used
  instead.
- `label` is provided to enable experimentation with alternative normalization
  methods.

No demographic data is included, to preserve contributors' privacy. In
particular, no writer id is present. Note that writer ids were used prior to
publication to split the dataset to ensure that some writers only contributed
samples to the 'valid' and 'test' splits. Some writers contributed samples to
all of the 'train', 'valid' and 'test' splits because we also did a split per
expression, ensuring that some expressions only appeared in exactly one of
the 'valid', 'test' and 'train' splits.

## Auxiliary files

### Bounding boxes

`synthetic-bboxes.jsonl` contains bounding boxes for mathematical expressions
obtained through the LaTeX compiler. These bounding boxes have been used
to generate inks in the `synthetic/` folder. This file is provided to enable
experimentation with bounding-box based generation. In particular, one could
try to use other individual glyphs than the ones provided in this dataset,
alter the bounding boxes, etc.

The file contains one JSON object per line.

Example content (this would be a single line in the file, reformatted here for
readability):

```
{
  "label": "\\frac{0}{u}",
  "normalizedLabel": "\\frac{0}{u}",
  "bboxes": [
    {
      "token": "0",
      "xMin": 102.39,
      "yMin": -865.7,
      "xMax": 430.07,
      "yMax": -443.36
    },
    {
      "token": "\\frac",
      "xMin": 78.64,
      "yMin": -176.95,
      "xMax": 453.81,
      "yMax": -150.73
    },
    {
      "token": "u",
      "xMin": 78.64,
      "yMin": 167.38,
      "xMax": 453.81,
      "yMax": 449.54
    }
  ]
}
```
Fields `label` and `normalizedLabel` have the same meaning as in InkML files.

For each object in `bboxes`:
- `xMin`, `xMax`, `yMin`, `yMax` are the coordinates of the bounding box. x
  coordinates are horizontal increasing to the right, y coordinates are
  vertical increasing *downward*. This is the same as in InkML files.
- `token` is a LaTeX string representing the glyph that should be rendered in
  the bounding box.

Note: `synthetic-bboxes.jsonl` can be joined with InkML files from the
`synthetic/` folder using values from their `label` fields.

### Reference to individual glyphs

`symbols.jsonl` contains references to inks and strokes therein that have
been used to generate inks in the `symbols/` folder, from inks in the `train/`
folder.

The file contains one JSON object per line.

Example content (this is a single line in the file, reformatted here for
readability):

```
{
  "sourceSampleId": "dcd6e0f2f06af904",
  "strokeIndices": [3, 4],
  "label": "f"
}
```

- `sourceSampleId` is an ink's identifier. All inks referenced in
  `symbols.jsonl` are from the `train/` folder.
- `strokeIndices` is a list of stroke indices that correspond to a single
  glyph.
- `label` is the LaTeX string representing that single glyph.

Since a mathematical expression contains several glyphs, the same ink can (and
is) referenced multiple times.

## Counts of inks

- `train`: 229864
- `valid`: 15674
- `test`: 7644
- `synthetic`: 396014 (of which 25699 have programmatically-generated labels,
   the rest have been obtained from Wikipedia)
- `symbols`: 6423

## Licensing Information
Unless otherwise stated, the content of this archive has been placed under the
CC BY-NC-SA licence: https://creativecommons.org/licenses/by-nc-sa/4.0/.

Unless otherwise stated, mathematical expressions in LaTeX format used in this
archive have been extracted from a Wikipedia dump either in 2015 or 2023, with
or without further processing. This content is covered by the Wikipedia
licence: CC BY-SA https://creativecommons.org/licenses/by-sa/4.0/
Processing on the LaTeX expressions obtained from Wikipedia include the
automatic normalization mentioned above, and some manual modifications in
cases the ink written by contributors did not match the original LaTeX
expression.
