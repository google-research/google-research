# Marot

This repository contains a web app  that can be used to analyze
[Multidimensional Quality Metrics (MQM)](http://www.qt21.eu/mqm-definition/definition-2015-06-16.html)
data from a human evaluation of translation quality. The web app can also
display metrics computed by automated evaluations, such as BLEURT.

To use the web app, download the files `marot.html`, `marot.js`,
`marot-histogram.js`, `marot-sigtests.js`, and `marot.css` to your computer:

```
wget https://raw.githubusercontent.com/google-research/google-research/master/marot/marot{-sigtests.js,-histogram.js,.html,.js,.css}
```

Then, simply open the `marot.html` file in a web browser, and use
the "Choose files" button to pick one or more Marot data files. Marot data spans
several columns, so it's best to use a desktop or laptop computer with a wide
screen.

A simpler option may be to just download the `marot-lite.html` file and
open it in a web browser (it loads the needed JavaScript and CSS files from
a Google-hosted server).

This is not an officially supported Google product.

## Data file format

The data file should have tab-separated UTF-8-encoded data with the following
ten columns, one line per marked error:

- **system**: Name of the translation system.
- **doc**: Name of the document. It's useful to suffix this with language-pair,
  (eg., "doc42:English-German"), especially as you may want to view the data
  from several evaluations together.
- **docSegId**: Id of segment (sentence or group of sentences) within the
  document.
- **globalSegId**: Id of segment across all documents. If you do not have
  such numbering available, set this to a constant value, say 0.
- **rater**: Rater who evaluated segment. If this row only carries metadata
  such as automated metrics and/or references, then `rater` should be the empty
  string (as should be `category` and `severity`).
- **source**: Source text for segment.
- **target**: Translated text for segment.
- **category**: MQM error category (or "no-error").
- **severity**: MQM error severity (or "no-error").
- **metadata**: JSON-formatted object that may contain the following fields,
  among others:
  - **timestamp**: Time at which this annotation was obtained (milliseconds
    since Unix epoch)
  - **note**: Free-form text note provided by the rater with some annotations
    (notably, with the "Other" error category)
  - **corrected_translation**: If the rater provided a corrected translation,
    for the segment, it will be included here.
  - **source_not_seen**: This will be set to true if this annotation was marked
    without the source text of the segment being visible.
  - **source_spans**: Array of pairs of 0-based indices (usually just one)
    identifying the indices of the first and last source tokens in the marked
    span. These indices refer to the source_tokens array in the segment
    object.
  - **target_spans**: Array of pairs of 0-based indices (usually just one)
    identifying the indices of the first and last target tokens in the marked
    span. These indices refer to the target_tokens array in the segment
    object.
  - **marked_text**: The text that has been marked by the rater (or the
    empty string if this metadata is not associated with an marked span). This
    field is computed from source_spans/target_spans. It can be useful
    when filtering.
  - **segment**: An object that has information about the segment (from the
    current doc+docSegId+system) that is not specific to any particular
    annotation/rater. This object may not necessarily be repeated across
    multiple ratings for the same segment. The segment object may contain the
    following fields:
      - **references**: A mapping from names of references to the references
        themselves (e.g., {"ref_A": "The reference", "ref_B": "..."}). This
        field need not be repeated across different systems.
      - **primary_reference**: The name of the primary reference, which is
        a key in the "references" mapping (e.g., "ref_A"). This field is
        required if "references" is present. This field too need not be repeated
        across different systems.
      - **metrics**: A dictionary in which the keys are the names of metrics
        (such as "Bleurt-X") and values are the numbers for those metrics. The
        metric name "MQM" is used for the MQM score. Note that this MQM score
        for the segment is computed *without any filtering*.
      - **source_tokens**: An array of source text tokens.
      - **target_tokens**: An array of target text tokens.
      - **source_sentence_tokens**: An array specifying sentence segmentation
        in the source segment. Each entry is the number of tokens in one
        sentence.
      - **target_sentence_tokens**: An array specifying sentence segmentation
        in the target segment. Each entry is the number of tokens in one
        sentence.
      - **starts_paragraph**: A boolean that is true if this segment is the
        start of a new paragraph.
      - In addition, any text annotation fields present in the input data are
        copied here. In [Anthea's data format](https://github.com/google-research/google-research/blob/master/anthea/anthea-help.html),
        this would be all the fields present in the optional last column.
  - **feedback**: An object optionally present in the metadata of the first
    segment of a doc. This captures any feedback the rater may have provided.
    It can include a free-form text field (keyed by **notes**) and a string
    keyed by **thumbs** that is set to either "up" or "down".
  - **evaluation**: An object that has information about the evaluation used.
    This field is typically only present in the very first data row, and is
    not repeated, in order to save space. This object may contain the following
    fields:
      - **template**: The name of the template used ("MQM", "MQM-WebPage",
        etc.).
      - **config**: The configuration parameters that define the template. This
        includes "errors" and "severities". Some bulky fields, notably
        "instructions" and "description" may have been stripped out from this
        object.
      - **source_language**, **target_language**: Language codes.
    In Marot, each metadata.evaluation object found is logged in the
    JavaScript debug console.

The "metadata" column used to be an optional "note" column, and Marot
continues to support that legacy format. Going forward, the metadata object
may be augmented to contain additional information about the rating/segment.

An optional header line in the data file will be ignored (identified by the
presence of the text "system\tdoc").

Example data files and details on score computations can be found in this
[GitHub repository](https://github.com/google/wmt-mqm-human-evaluation).

## Data format conversion

You can easily add format conversion code that can convert arbitrarily
formatted data (for example, JSON lines from a BLEURT decoder), by adding a
JavaScript function with the following name and behavior:

```
/**
 * Transform data (that may be in some custom format) into the Marot data format.
 * Pass through the data if no conversion was appropriate or necessary.
 * @param {string} sourceName The file name or URL source for the data.
 * @param {string} data The original data.
 * @return {string} The Marot-data-formatted data.
 */
function marotDataConverter(sourceName, data) {
  ...
  return data;
}
```

## Data from URLs

You can pass a `?dataurls=<url1>,...` parameter to Marot, to load data
from the URLs listed. Note that any URLs have to be hosted on the same site
as the viewer itself, or need to have a CORS exception.

If your domain uses some custom way of storing data (Google uses the CNS file
system, for example) that uses a way to convert data names to URLs, and you wish
to directly pass such data names as URLs (to `?dataurls=`), then you can add a
JavaScript function with the following name and behavior:
```
/**
 * Transform a data name (that may be in some custom format) to a URL.
 * @param {string} dataName The name or identifier for the data.
 * @return {string} The URL from which the data can be loaded.
 */
function marotURLMaker(dataName) {
  /** Code to convert dataName into url */
  let url = ...;
  return url;
}
```

## Filtering

This web app facilitates interactive slicing and dicing of the data to identify
interesting subsets, to compare translation systems along various dimensions,
etc. The scores shown are always updated to reflect the currently active
filters.

- You can click on any System/Doc/ID/Rater/Category/Severity (or pick
  from the drop-down list under the column name) to set its **column
  filter** to that specific value.
- You can provide **column filter** regular expressions for filtering
  one or more columns, in the input fields provided under the column names.
- You can create sophisticated filters (involving multiple columns, for
  example) using a **JavaScript filter expression**.
  - This allows you to filter using any expression
    involving the columns. It can use the following
    variables: **system**, **doc**, **docSegId**,
    **globalSegId**, **rater**, **category**, **severity**,
    **source**, **target**, **metadata**.
  - Filter expressions also have access to three aggregated objects in
    variables named **aggrDoc**, **aggrDocSeg**, and **aggrDocSegSys**.
    The aggrDocSegSys dict also contains aggrDocSeg (with the key
    "aggrDocSeg"), which in turn similarly contains aggrDoc.
    - **aggrDoc** has the following properties:
      **doc**, **thumbsUpCount**, **thumbsDownCount**.
    - **aggrDocSeg** is an object with the following properties:
      - **aggrDocSeg.catsBySystem**,
      - **aggrDocSeg.catsByRater**,
      - **aggrDocSeg.sevsBySystem**,
      - **aggrDocSeg.sevsByRater**,
      - **aggrDocSeg.sevcatsBySystem**,
      - **aggrDocSeg.sevcatsByRater**,
      - **aggrDocSeg.source_tokens**,
      - **aggrDocSeg.source_sentence_tokens**,
      - **aggrDocSeg.starts_paragraph**,
      - **aggrDocSeg.references** (if available),
      - **aggrDocSeg.primary_reference** (if available),
      Each of these properties is an object keyed by system or rater, with the
      values being arrays of strings. The "sevcats\*" values look like
      "Minor/Fluency/Punctuation" or are just the same as severities if
      categories are empty. This segment-level aggregation allows you
      to select specific segments rather than just specific error ratings.
    - **aggrDocSeg.metrics** is an object keyed by the metric name and then by
      system name. It provides the segment's metric scores (including MQM) for
      all systems for which a metric is available for that segment.
    - **aggrDocSegSys** is just an alias for metadata.segment.
  - **Example**: docSegId > 10 || severity == 'Major'
  - **Example**: target.indexOf('thethe') >= 0
  - **Example**: metadata.marked_text.length >= 10
  - **Example**: aggrDocSeg.sevsBySystem['System-42'].includes('Major')
  - **Example**: aggrDocSegSys.metrics['MQM'] > 4 &&
    (aggrDocSegSys.metrics['BLEURT-X'] ?? 1) < 0.1.
  - **Example**: JSON.stringify(aggrDocSeg.sevcatsBySystem).includes('Major/Fl')
  - You can examine the metadata associated with any using the **Log metadata**
    interface shown in the **Filters** section. This can be useful for crafting
    filter expressions.

## Significance tests
When there are multiple systems that have been evaluated on common document
segments, significance tests are run for each pair of systems and the resulting
p-values are displayed in a table. The testing is done via paired one-sided
approximate randomization (PAR), which corresponds to 'alternative="greater"'
in [scipy's API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

The significance tests are recomputed with any filtering that is applied. The
computations are run in a background Worker thread. The tests include any
available automated metrics in addition to MQM.

## Data Notes
There are some nuances to the data format which are useful to be aware of:

  - Marked spans are noted in the source/target text using `<v>...</v>` tags
    to enclose them. For example: `The error is <v>here</v>.`
  - Except in some legacy data, error spans are also identified at precise
    token-level using the `metadata.source_spans` and `metadata.target_spans`
    fields.
  - Severity and category names come directly from annotation tools and may
    have subtle variations (such as lowercase/uppercase differences or
    space-underscore changes).
  - Error spans may include leading/trailing whitespace if the annotation tool
    allows for this, which may or may not be part of the actual errors.
    For example, `The error is<v> here</v>.`
    The error spans themselves can also be entirely whitespace.

