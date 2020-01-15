# Learning Dense Representations for Entity Retrieval

Code supporting the publication _Learning Dense Representations for Entity
Retrieval_ by Daniel Gillick, Sayali Kulkarni, Larry Lansing, Alessandro Presta,
Jason Baldridge, Eugene Ie, and Diego Garcia-Olano.

Paper available at: https://arxiv.org/abs/1909.10506

If you use this code in your research, please cite the paper as follows:

```
@misc{gillick2019learning,
    title={Learning Dense Representations for Entity Retrieval},
    author={Daniel Gillick and Sayali Kulkarni and Larry Lansing and Alessandro Presta and Jason Baldridge and Eugene Ie and Diego Garcia-Olano},
    year={2019},
    eprint={1909.10506},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Wikinews

The ```parse_wikinews.py``` script generates the 2018 Wikinews dataset described
in _Learning Dense Representations for Entity Retrieval_ by parsing the wikitext
from the Jan 1, 2019 dump of Wikinews found on archive.org.

To generate the dataset yourself, download the Wikinews dump from
https://archive.org/download/enwikinews-20190101/enwikinews-20190101-pages-articles.xml.bz2
and run ```parse_wikinews.py``` with the ```wikinews_archive``` and
```output_dir``` flags set appropriately. See ```parse_wikinews.sh``` for an
example of correct usage.

## Disclaimer

This is not an official Google product.
