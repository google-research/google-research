# MQM Viewer

This repository contains a web app  that can be used to analyze
[Multidimensional Quality Metrics (MQM)](http://www.qt21.eu/mqm-definition/definition-2015-06-16.html) data from a translation quality
evaluation.

To use it, download the files `mqm-viewer.html`, `mqm-viewer.js`, and
`mqm-viewer.css` to your computer:

```
wget https://raw.githubusercontent.com/google-research/google-research/master/mqm_viewer/mqm-viewer.{html,js,css}
```

Then, simply open the `mqm-viewer.html` file in a web browser, and use
the "Choose file" button to pick an MQM data file. MQM data spans several
columns, so it's best to use a desktop or laptop computer with a wide screen.

This is not an officially supported Google product.

## Data file format

The data file should have tab-separated UTF-8-encoded data with the following
nine (or ten) columns, one line per marked error:

- **system**: Name of the translation system.
- **doc**: Name of the document.
- **doc_id**: Id of segment (sentence or group of sentences) within the
  document.
- **seg_id**: Id of segment across all documents.
- **rater**: Rater who evaluated segment.
- **source**: Source text for segment.
- **target**: Translated text for segment.
- **category**: MQM error category (or "no-error").
- **severity**: MQM error severity (or "no-error").
- **note**: Optional note.

An optional header line in the data file will be ignored (identified by the
presence of the text "system\tdoc").

Example data files and details on score computations can be found in this
[GitHub repository](https://github.com/google/wmt-mqm-human-evaluation).

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
    variables: **system**, **doc**, **seg_id**,
    **doc_id**, **rater**, **category**, **severity**,
    **source**, **target**.
  - Filter expressions also have access to an aggregated **segment**
    variable that is an object with the following properties:
    **segment.cats_by_system**,
    **segment.cats_by_rater**,
    **segment.sevs_by_system**,
    **segment.sevs_by_rater**,
    **segment.sevcats_by_system**,
    **segment.sevcats_by_rater**.
    Each of these properties is an object keyed by system or rater, with the
    values being arrays of strings. The "sevcats_\*" values look like
    "Minor/Fluency/Punctuation" or are just the same as severities if
    categories are empty. This segment-level aggregation allows you
    to select specific segments rather than just specific error ratings.
  - **Example**: seg_id > 10 || severity == 'Major'
  - **Example**: target.indexOf('thethe') >= 0
  - **Example**: segment.sevs_by_system['System-42'].includes('Major')
  - **Example**: JSON.stringify(segment.sevcats_by_system).includes('Major/Fl')
