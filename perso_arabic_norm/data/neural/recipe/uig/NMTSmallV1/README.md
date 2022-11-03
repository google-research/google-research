```shell
  onmt-build-vocab --tokenizer_config tokenizer.yml --size 50000 --save_vocab src-vocab.txt src-train.txt
  onmt-build-vocab --tokenizer_config tokenizer.yml --size 50000 --save_vocab tgt-vocab.txt tgt-train.txt
  (nohup ./train.sh > train.log 2>&1) > /dev/null 2>&1 &
```
