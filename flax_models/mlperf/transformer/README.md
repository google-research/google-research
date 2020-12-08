# MLPerf Transformer in Flax

This code is intended to run on the new JAX Cloud TPU setup (in alpha as of
Q4 2020), _not_ on the "Cloud TPU Preview" that began in 2019. The difference
is that the alpha allows direct SSH access to TPU hosts as individual Cloud VMs.

Request early access to this alpha at goo.gle/jax-tpu-signup. Please direct any
questions related to using this codebase to the "JAX on Cloud TPU" chatroom
associated with the alpha.

Since the codebase is meant for large TPU pod slices, we wrote a set of Bash
scripts for orchestrating the many hosts in the pod slice. Run
```bash pssh.sh [tpu_slice_name] [script_name.sh]```
to run a script on every host in a particular slice that you've created.

In particular, run `setup.sh` in this way, then `train.sh` in order to launch
training, then watch the logs that come out. (Note that only the log for the
host with JAX host_id=0, which is unfortunately not the same host as TPU worker
0, will include all the MLPerf logs; you can find which host this is using the
`host_id.sh` script). Use `less` and `shift-F` or `tail -f` to watch logs live.

`pssh.sh`:
```
#!/bin/bash

POD_IPS=`gcloud alpha compute tpus tpu-vm describe $1 --zone europe-west4-a --project jax-research | grep externalIp | awk '{print $2}' | xargs`

mkdir -p "logs/$1"
for i in $POD_IPS
do
  cat $2 | ssh $i -o StrictHostKeyChecking=no -P "bash -s" &> "logs/$1/$i.log" &
done
wait
```

The setup script needs to be customized to the particular TPU slice you're
using. `TPU_HOST_BOUNDS` should be the slice bounds divided by 2 in X and Y
(so this script is for a v3-1024, with slice bounds 16x32).
`TPU_MESH_CONTROLLER_ADDRESS` should include the private IP of TPU worker 0
(_not_ JAX host 0).
`setup.sh`:
```
#!/bin/bash
pip install --upgrade --user https://storage.googleapis.com/jax-releases/tpu/jaxlib-0.1.55+tpu-cp36-none-manylinux2010_x86_64.whl
pip install --upgrade --user jax

# for tf.data
pip install --upgrade --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.3.0-cp36-cp36m-manylinux2010_x86_64.whl

rm -rf flax/
git clone https://github.com/google/flax
pip install --user -e ./flax

rm -rf google-research/
git clone https://github.com/google-research/google-research --depth 1

echo "
export CLOUD_TPU_TASK_ID=`echo $HOSTNAME | grep -o -- '-[0-9]*$' | cut -c2-`
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=8,16,1
export TPU_MESH_CONTROLLER_ADDRESS=10.164.0.96:8476
export TPU_MESH_CONTROLLER_PORT=8476" > tpu_env_vars.sh
```

`host_id.sh`:
```
#!/bin/bash
source ~/tpu_env_vars.sh
python3 -c 'import jax; print("host_id", jax.host_id())'
```

`train.sh`:
```
#!/bin/bash
source ~/tpu_env_vars.sh
cd google-research
python3 -m flax_models.mlperf.transformer.train --model_dir=/tmp/wmt --mlperf_logs=0
```

