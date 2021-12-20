Our raw dataset can be retrieved by running:

```
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
```

Alternatively, to copy the entire directory you may also use:

```
gsutil cp -r gs://gresearch/goemotions/data/full_dataset/ .
```
