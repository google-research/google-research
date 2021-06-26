The multi-part gzip files in this directory where split using

```shell
split --numeric-suffixes -b 1M --suffix-length 1 ${FILE}.tsv.gz ${FILE}.tsv.gz.
```

They need to be concatenated before attempting decompression.
