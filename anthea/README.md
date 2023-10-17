# Anthea
Anthea is a translation quality evaluation tool. It presents source text
and translated text side-by-side in a natural, flowing fashion (as opposed
to in an artificial, tabulated format). Raters evaluate the quality of the
text in terms of segments whose sizes can be configured.

Anthea is designed to be used for evaluating using
Multidimensional Quality Metrics (MQM), for which a few exemplary evaluation templates are included.

## Setup
Anthea can be set up very easily by simply serving the following files from a
web server:

- [`anthea-eval.css`](anthea-eval.css)
- [`anthea-eval.js`](anthea-eval.js)
- [`anthea-help.html`](anthea-help.html)
- [`anthea-manager.css`](anthea-manager.css)
- [`anthea-manager.js`](anthea-manager.js)
- [`anthea.html`](anthea.html): This is the main web page for Anthea.
- Template files:
  - [`template-mqm.js`](template-mqm.js)
  - [`template-mqm-cd.js`](template-mqm-cd.js)
  - [`template-mqm-paragraph.js`](template-mqm-paragraph.js)
  - [`template-mqm-webpage.js`](template-mqm-webpage.js)
  - [`template-mqm-monolingual.js`](template-mqm-monolingual.js)
- Marot files (from a sibling project):
  - [`marot.css`](https://github.com/google-research/google-research/blob/master/mqm_viewer/marot.css)
  - [`marot.js`](https://github.com/google-research/google-research/blob/master/mqm_viewer/marot.js)
  - [`marot-histogram.js`](https://github.com/google-research/google-research/blob/master/mqm_viewer/marot-histogram.js)
  - [`marot-sigtests.js`](https://github.com/google-research/google-research/blob/master/mqm_viewer/marot-sigtests.js)

## User guide
The user guide is in [anthea-help.html](anthea-help.html) and is available as a
menu item in the tool itself.
