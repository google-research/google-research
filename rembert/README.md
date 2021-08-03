This is a repository for the ICLR 2021 paper: [Rethinking Embedding Coupling in
Pre-trained Language Models](https://openreview.net/forum?id=xpFFI_NtgpW)

RemBERT is a multilingual encoder pre-trained on the combination of mC4 and Wikipedia corpus over 110 languages. Its main design features are

  * Decoupled input and output embedding
  * Small input embedding and larger output embedding (discarded after
    pre-training)
  * Reinvested input embedding parameters in the form or deeper and wider
    Transformer layers.

With these design features, RemBERT outperforms similarly sized XLM-R and mT5 on
XTREME benchmarks. It performs slightly better than mT5-XL, which is about 6 times
larger.

In this repo, we only provide the pre-trained checkpoints of the RemBERT model.

GCS link: [gs://gresearch/rembert](https://console.cloud.google.com/storage/browser/gresearch/rembert)


The list of languages RemBERT is trained on is: ['af', 'am', 'ar', 'az', 'be', 'bg', 'bg-Latn', 'bn', 'bs', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'el-Latn', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'hi', 'hi-Latn', 'hmn', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'ja-Latn', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'ru-Latn', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tr', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zh-Hans', 'zh-Hant', 'zh-Latn', 'zu']
