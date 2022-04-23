To regenerate the Python protocol buffers in this directory (files with extension
`_pb2.py`) install Google protocol buffer compiler (`protoc`) and run

```shell
protoc --python_out=${PWD} feature_neighborhood_tensor_opts.proto
```
